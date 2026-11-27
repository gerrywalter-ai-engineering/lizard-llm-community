import os from "node:os";
import { execFile } from "node:child_process";
import { readLibreHardwareMonitor, libreEndpoint, bundledLibreExecutable } from "./libre-hardware-monitor.js";
import { getAppRoot } from "./app-root.js";

// Live CPU / RAM / GPU / VRAM readings with temperatures, for the analyzed
// view's thermal cards.
//
// Everything here is async and every child process is bounded. The dashboard
// serves this from its request path, and a sync child process blocks the whole
// event loop for its duration: a WMI query or nvidia-smi call that wedges - a
// known Windows failure mode while a display driver resets, which is exactly
// the state a struggling model load puts it in - would freeze the entire app.
// A missing temperature is a dashed-out card; a hung probe is a dead dashboard.
// The Windows GPU counters alone take ~3.8s on this class of machine, so the
// budget has to clear that with room to spare - but nothing waits on it; see
// cachedProbe.
const PROBE_TIMEOUT_MS = 9_000;

// Successes are cheap to refresh and the cards poll every few seconds.
// Failures are not: a machine with no thermal sensor re-pays the full probe
// timeout on every poll, so a failed probe is remembered far longer than a
// successful one.
const SUCCESS_TTL_MS = 2_000;
const FAILURE_TTL_MS = 90_000;

const probeCache = new Map();

function run(command, args) {
  return new Promise((resolve) => {
    let child = null;
    try {
      child = execFile(command, args, {
        encoding: "utf8",
        windowsHide: true,
        timeout: PROBE_TIMEOUT_MS,
        killSignal: "SIGKILL",
        maxBuffer: 1024 * 1024,
      }, (error, stdout) => resolve(error ? null : String(stdout ?? "").trim()));
    } catch {
      resolve(null);
      return;
    }
    // execFile's timeout only fires once the process has started. A spawn that
    // never gets that far would leave this promise pending forever, and the
    // dashboard request with it.
    const guard = setTimeout(() => {
      try { child?.kill("SIGKILL"); } catch { /* already gone */ }
      resolve(null);
    }, PROBE_TIMEOUT_MS + 500);
    child?.on("close", () => clearTimeout(guard));
  });
}

// Serve what is known now, refresh behind the response. The GPU counters take
// nearly four seconds; awaiting them would put that on every telemetry request
// and, with the cards polling, keep a request in flight almost permanently. The
// first call therefore returns null and the next one has real values - which is
// the right trade for a live gauge, where being one poll late is invisible and
// a four-second stall is not.
function cachedProbe(key, probe) {
  const cached = probeCache.get(key);
  const now = Date.now();
  if (cached && now < cached.expires) return cached.value;
  if (cached?.refreshing) return cached.value;

  const entry = cached ?? { value: null, expires: 0 };
  entry.refreshing = true;
  probeCache.set(key, entry);

  Promise.resolve()
    .then(probe)
    .catch(() => null)
    .then((value) => {
      probeCache.set(key, {
        value,
        refreshing: false,
        expires: Date.now() + (value === null ? FAILURE_TTL_MS : SUCCESS_TTL_MS),
      });
    });

  return entry.value;
}

function powershell(script) {
  return run("powershell.exe", [
    "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script,
  ]);
}

// os.cpus() reports cumulative tick counters since boot, so a single reading is
// an average over the machine's whole uptime and never moves. Utilisation has
// to come from the delta between two samples.
let previousCpuTicks = null;

function sampleCpuTicks() {
  let idle = 0;
  let total = 0;
  for (const cpu of os.cpus() ?? []) {
    for (const [mode, ticks] of Object.entries(cpu.times ?? {})) {
      total += ticks;
      if (mode === "idle") idle += ticks;
    }
  }
  return { idle, total };
}

export function readCpuLoadPercent() {
  const current = sampleCpuTicks();
  const previous = previousCpuTicks;
  previousCpuTicks = current;
  if (!previous) return null;
  const totalDelta = current.total - previous.total;
  const idleDelta = current.idle - previous.idle;
  if (totalDelta <= 0) return null;
  return clampPercent(((totalDelta - idleDelta) / totalDelta) * 100);
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return null;
  return Math.min(100, Math.max(0, Math.round(value * 10) / 10));
}

function firstNumber(text) {
  const match = String(text ?? "").match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : null;
}

// MSAcpi_ThermalZoneTemperature reports deci-kelvin. It is the only CPU-side
// temperature available without a kernel driver, and plenty of consumer
// laptops simply do not expose it - hence the long failure cache rather than a
// retry loop.
async function readWindowsCpuTemperature() {
  const raw = await powershell(
    "(Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature"
    + " -ErrorAction Stop | Select-Object -First 1).CurrentTemperature",
  );
  const deciKelvin = firstNumber(raw);
  if (deciKelvin === null || deciKelvin <= 0) return null;
  const celsius = deciKelvin / 10 - 273.15;
  return celsius > -50 && celsius < 150 ? Math.round(celsius * 10) / 10 : null;
}

async function readLinuxCpuTemperature() {
  const raw = await run("sh", [
    "-c",
    "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -n 1",
  ]);
  const milliCelsius = firstNumber(raw);
  if (milliCelsius === null) return null;
  const celsius = milliCelsius > 1000 ? milliCelsius / 1000 : milliCelsius;
  return celsius > -50 && celsius < 150 ? Math.round(celsius * 10) / 10 : null;
}

function readCpuTemperature() {
  return cachedProbe("cpu-temp", () => (process.platform === "win32"
    ? readWindowsCpuTemperature()
    : readLinuxCpuTemperature()));
}

// NVIDIA reports utilisation, VRAM and temperature in one call, so when it is
// present it is strictly better than the generic counters below.
async function readNvidiaGpus() {
  const raw = await run("nvidia-smi", [
    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
  ]);
  if (!raw) return null;
  const gpus = [];
  for (const line of raw.split("\n")) {
    const parts = line.split(",").map((value) => value.trim());
    if (parts.length < 5 || !parts[0]) continue;
    gpus.push({
      name: parts[0],
      vendor: "nvidia",
      utilizationPercent: clampPercent(Number(parts[1])),
      memoryUsedBytes: Number(parts[2]) * 1024 * 1024,
      memoryTotalBytes: Number(parts[3]) * 1024 * 1024,
      temperatureC: Number.isFinite(Number(parts[4])) ? Number(parts[4]) : null,
      temperatureSource: "nvidia-smi",
    });
  }
  return gpus.length > 0 ? gpus : null;
}

// Vendor-neutral fallback: Windows publishes per-adapter GPU counters for every
// vendor on Win10+, which is the only way to get Intel and AMD utilisation
// without a vendor SDK. There is no temperature counter, so those cards show
// utilisation and memory only.
async function readWindowsGpu() {
  const script = [
    "$ErrorActionPreference='Stop';",
    // One Get-Counter, not three: each wildcard enumeration costs over a
    // second on its own, and the three together were the whole probe budget.
    "$p=(Get-Counter '\\GPU Engine(*)\\Utilization Percentage',",
    "'\\GPU Adapter Memory(*)\\Dedicated Usage',",
    "'\\GPU Adapter Memory(*)\\Shared Usage'",
    " -ErrorAction SilentlyContinue).CounterSamples;",
    "$u=$p | Where-Object { $_.Path -like '*utilization percentage*' }",
    " | Measure-Object -Property CookedValue -Sum;",
    "$d=$p | Where-Object { $_.Path -like '*dedicated usage*' }",
    " | Measure-Object -Property CookedValue -Sum;",
    "$s=$p | Where-Object { $_.Path -like '*shared usage*' }",
    " | Measure-Object -Property CookedValue -Sum;",
    "$a=Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM;",
    "[pscustomobject]@{util=$u.Sum;dedicated=$d.Sum;shared=$s.Sum;name=$a.Name;adapterRam=$a.AdapterRAM}",
    " | ConvertTo-Json -Compress",
  ].join("");
  const raw = await powershell(script);
  if (!raw) return null;
  let parsed = null;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const dedicated = Number(parsed.dedicated) || 0;
  const shared = Number(parsed.shared) || 0;
  // Win32_VideoController.AdapterRAM is a 32-bit field, so it wraps at 4 GB and
  // reports garbage for larger cards. On integrated graphics it is a share of
  // system RAM anyway, which is why shared usage is reported alongside it.
  const adapterRam = Number(parsed.adapterRam);
  // On integrated graphics the adapter has no dedicated pool: usage is shared
  // system RAM and routinely exceeds AdapterRAM, which would render as a
  // 112%-full VRAM bar. A total is only reported when there is a real
  // dedicated pool being used; otherwise the card shows usage without a bar.
  const total = dedicated > 0
    && Number.isFinite(adapterRam) && adapterRam > 0 && adapterRam < 4 * 1024 ** 3
    ? adapterRam
    : null;
  return [{
    name: String(parsed.name || "GPU"),
    vendor: "generic",
    utilizationPercent: clampPercent(Number(parsed.util)),
    memoryUsedBytes: dedicated + shared,
    memoryDedicatedBytes: dedicated,
    memorySharedBytes: shared,
    memoryTotalBytes: total,
    temperatureC: null,
    temperatureSource: null,
  }];
}

function readGpus() {
  return cachedProbe("gpu", async () => {
    const nvidia = await readNvidiaGpus();
    if (nvidia) return nvidia;
    if (process.platform === "win32") return readWindowsGpu();
    return null;
  });
}

/**
 * One snapshot of the machine's live load and temperatures. Never throws and
 * never blocks: any probe that fails contributes null, and the card renders as
 * unavailable rather than the whole panel disappearing.
 */
function safeAppRoot() {
  try {
    return getAppRoot();
  } catch {
    return null;
  }
}

function readLibre() {
  return cachedProbe("libre", () => readLibreHardwareMonitor());
}

export async function readSystemTelemetry() {
  const [wmiCpuTemperatureC, gpus] = await Promise.all([
    readCpuTemperature(),
    readGpus(),
  ]);
  const libre = readLibre();

  // LibreHardwareMonitor wins when it is running. It reads the CPU's own
  // digital thermal sensor through a kernel driver, which is the number the
  // vendor quotes and the one throttling is keyed to; the WMI thermal zone is
  // a board sensor and usually is not even readable unelevated.
  const cpuTemperatureC = libre?.summary?.cpu?.temperatureC ?? wmiCpuTemperatureC;
  const cpuTemperatureSource = libre?.summary?.cpu?.temperatureC != null
    ? "LibreHardwareMonitor - " + (libre.summary.cpu.temperatureSensor || "CPU")
    : (wmiCpuTemperatureC === null
      ? null
      : (process.platform === "win32" ? "MSAcpi_ThermalZoneTemperature" : "thermal_zone"));

  // Same for the GPU: the vendor-neutral Windows counters carry utilisation and
  // memory but no temperature at all.
  const libreGpu = libre?.summary?.gpu ?? null;
  if (libreGpu?.temperatureC != null && gpus?.[0] && gpus[0].temperatureC === null) {
    gpus[0].temperatureC = libreGpu.temperatureC;
    gpus[0].temperatureSource = "LibreHardwareMonitor - " + (libreGpu.temperatureSensor || "GPU Core");
    gpus[0].hotSpotC = libreGpu.hotSpotC;
    gpus[0].powerW = libreGpu.powerW;
    gpus[0].coreClockMhz = libreGpu.coreClockMhz;
  }

  const totalMemoryBytes = os.totalmem();
  const freeMemoryBytes = os.freemem();
  const usedMemoryBytes = totalMemoryBytes - freeMemoryBytes;

  return {
    sampledAt: new Date().toISOString(),
    platform: process.platform,
    cpu: {
      model: os.cpus()?.[0]?.model ?? null,
      logicalCores: os.cpus()?.length ?? null,
      loadPercent: readCpuLoadPercent(),
      temperatureC: cpuTemperatureC,
      // Named so the card can say where a temperature came from, or why there
      // is not one, instead of showing a bare dash.
      temperatureSource: cpuTemperatureSource,
      packagePowerW: libre?.summary?.cpu?.packagePowerW ?? null,
      clockMhz: libre?.summary?.cpu?.clockMhz ?? null,
      temperatureUnavailableReason: cpuTemperatureC !== null
        ? null
        : (process.platform === "win32"
          ? "Windows exposes CPU temperature only to an elevated process."
            + " Run LibreHardwareMonitor as administrator with its web server on."
          : "no readable thermal zone"),
    },
    memory: {
      totalBytes: totalMemoryBytes,
      usedBytes: usedMemoryBytes,
      freeBytes: freeMemoryBytes,
      usedPercent: clampPercent((usedMemoryBytes / totalMemoryBytes) * 100),
    },
    gpus: Array.isArray(gpus) ? gpus : [],
    libre: {
      available: Boolean(libre),
      endpoint: libreEndpoint(),
      // Whether the installer shipped a copy we can offer to start. In a
      // source checkout there is none, and the panel says to install it.
      bundled: Boolean(bundledLibreExecutable(safeAppRoot())),
      hardware: libre?.hardware ?? [],
      fans: libre?.summary?.fans ?? [],
      storageTemps: libre?.summary?.storageTemps ?? [],
    },
  };
}

// Test seam: the CPU delta is stateful, and the probe caches would otherwise
// leak between cases.
export function resetSystemTelemetryState() {
  probeCache.clear();
  previousCpuTicks = null;
}
