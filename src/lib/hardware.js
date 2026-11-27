import os from "node:os";
import path from "node:path";
import { execFileSync, execSync } from "node:child_process";

function toGiB(bytes) {
  return Number((bytes / 1024 / 1024 / 1024).toFixed(2));
}

// These probes are SYNCHRONOUS and run on the request path (scanHardware is the
// first thing /api/warm-model does). A sync child process blocks the event loop
// for its whole duration, so a WMI query or nvidia-smi call that wedges - a
// known Windows failure mode when a display driver is resetting, which is
// exactly the state a struggling model load puts it in - freezes the entire app:
// no page loads, no API answers, nothing. A missing hardware fact is a degraded
// reading; a hung probe is a dead app, so the probe is the one that gives way.
const HARDWARE_PROBE_TIMEOUT_MS = 5_000;

function safeExec(command, args = []) {
  try {
    // windowsHide: execFileSync defaults this to false, which briefly flashes
    // a console window for each call - and detectStorage/detectNvidiaGpus/
    // detectWindowsDisplayAdapters all run synchronously and unconditionally
    // on every dashboard boot (scanHardware -> createDashboardCommandCenter),
    // so without this every app launch flashed 2-3 console windows.
    return execFileSync(command, args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      windowsHide: true,
      timeout: HARDWARE_PROBE_TIMEOUT_MS,
      killSignal: "SIGKILL",
    }).trim();
  } catch {
    return null;
  }
}

function safeShell(command) {
  try {
    return execSync(command, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      windowsHide: true,
      timeout: HARDWARE_PROBE_TIMEOUT_MS,
      killSignal: "SIGKILL",
    }).trim();
  } catch {
    return null;
  }
}

function detectStorage() {
  if (process.platform === "win32") {
    const output = safeExec("powershell.exe", [
      "-NoProfile",
      "-Command",
      "Get-PSDrive -PSProvider FileSystem | Select-Object Name,Free,Used | ConvertTo-Json -Compress",
    ]);
    if (!output) {
      return null;
    }
    const drives = JSON.parse(output);
    const driveList = Array.isArray(drives) ? drives : [drives];
    const currentDrive = path.parse(process.cwd()).root[0].toUpperCase();
    const match = driveList.find((drive) => String(drive.Name).toUpperCase() === currentDrive);
    if (!match) {
      return null;
    }
    return {
      drive: `${match.Name}:`,
      freeGB: toGiB(Number(match.Free)),
      usedGB: toGiB(Number(match.Used)),
    };
  }

  if (process.platform === "linux") {
    const output = safeShell("df -k . | tail -1");
    if (!output) {
      return null;
    }
    const parts = output.split(/\s+/);
    return {
      drive: parts[5],
      freeGB: Number((Number(parts[3]) / 1024 / 1024).toFixed(2)),
      usedGB: Number((Number(parts[2]) / 1024 / 1024).toFixed(2)),
    };
  }

  return null;
}

// Physical cores, not logical ones.
//
// This used to report os.cpus().length for both, which on any SMT machine is
// simply the logical count twice - an i7-1165G7 (4 cores, 8 threads) claimed 8
// physical cores. Everything downstream that wanted "one worker per core" was
// therefore sizing thread pools against hyperthread siblings, and on the GPU
// hybrid lane that measurably costs throughput (8 workers decoded 6.25 tok/s
// against 7.16 at 4 on that exact chip).
//
// Falls back to the logical count when the query is unavailable: over-reporting
// cores is what the code already assumed, so a failed probe changes nothing.
let physicalCoreCache = null;

function detectPhysicalCores(logicalThreads) {
  // Core topology cannot change while the process runs, and the probe costs a
  // PowerShell start-up (~1.5 s on Windows). scanHardware is called from many
  // paths, so memoise it rather than paying that on each one.
  if (physicalCoreCache != null) {
    return physicalCoreCache;
  }
  physicalCoreCache = probePhysicalCores(logicalThreads);
  return physicalCoreCache;
}

function probePhysicalCores(logicalThreads) {
  const fallback = Math.max(1, Number(logicalThreads) || 1);
  const parse = (value) => {
    const cores = Math.trunc(Number(String(value ?? "").trim()));
    return Number.isFinite(cores) && cores > 0 ? Math.min(cores, fallback) : null;
  };

  if (process.platform === "win32") {
    const output = safeExec("powershell.exe", [
      "-NoProfile",
      "-Command",
      "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum",
    ]);
    return parse(output) ?? fallback;
  }

  if (process.platform === "linux") {
    // "cpu cores" is per-socket, so count distinct (physical id, core id) pairs
    // to stay correct on multi-socket machines.
    const output = safeShell("lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l");
    return parse(output) ?? fallback;
  }

  if (process.platform === "darwin") {
    return parse(safeShell("sysctl -n hw.physicalcpu")) ?? fallback;
  }

  return fallback;
}

function detectNvidiaGpus() {
  const output = safeExec("nvidia-smi", [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
  ]);
  if (!output) {
    return [];
  }
  return output
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const [name, memoryMb, driverVersion] = line.split(",").map((part) => part.trim());
      return {
        vendor: "nvidia",
        name,
        vramGB: Number((Number(memoryMb) / 1024).toFixed(2)),
        driverVersion,
        backends: {
          cuda: true,
          metal: false,
          vulkan: false,
          rocm: false,
        },
      };
    });
}

function detectWindowsDisplayAdapters() {
  const output = safeExec("powershell.exe", [
    "-NoProfile",
    "-Command",
    "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
  ]);
  if (!output) {
    return [];
  }

  const parsed = JSON.parse(output);
  const adapters = Array.isArray(parsed) ? parsed : [parsed];
  return adapters
    .filter((adapter) => adapter?.Name)
    .map((adapter) => {
      const name = String(adapter.Name).trim();
      const lower = name.toLowerCase();
      const vendor =
        lower.includes("nvidia") ? "nvidia" :
        lower.includes("amd") || lower.includes("radeon") ? "amd" :
        lower.includes("intel") ? "intel" :
        "unknown";
      return {
        vendor,
        name,
        vramGB: adapter.AdapterRAM ? toGiB(Number(adapter.AdapterRAM)) : null,
        backends: {
          cuda: vendor === "nvidia",
          metal: false,
          vulkan: vendor !== "unknown",
          rocm: vendor === "amd",
        },
      };
    });
}

function detectLinuxGpus() {
  const output = safeShell("lspci | grep -Ei \"vga|3d|display\"");
  if (!output) {
    return [];
  }
  return output.split(/\r?\n/).filter(Boolean).map((line) => ({
    vendor: /nvidia/i.test(line) ? "nvidia" : /amd|radeon/i.test(line) ? "amd" : /intel/i.test(line) ? "intel" : "unknown",
    name: line.replace(/^[^:]+:\s*/, "").trim(),
    vramGB: null,
    backends: {
      cuda: /nvidia/i.test(line),
      metal: false,
      vulkan: true,
      rocm: /amd|radeon/i.test(line),
    },
  }));
}

function detectMacGpu() {
  const output = safeExec("system_profiler", ["SPDisplaysDataType", "-json"]);
  if (!output) {
    return [];
  }
  const parsed = JSON.parse(output);
  const cards = parsed.SPDisplaysDataType ?? [];
  return cards.map((card) => ({
    vendor: "apple",
    name: card.sppci_model ?? "Apple GPU",
    vramGB: card.spdisplays_vram_shared ? null : null,
    backends: {
      cuda: false,
      metal: true,
      vulkan: false,
      rocm: false,
    },
  }));
}

function dedupeGpus(gpus) {
  const seen = new Set();
  return gpus.filter((gpu) => {
    const key = `${gpu.vendor}:${gpu.name}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

export function scanHardware({ config, logger }) {
  const cpus = os.cpus();
  const totalMemoryGB = toGiB(os.totalmem());
  const freeMemoryGB = toGiB(os.freemem());
  const plannerBudgetGB = Number((totalMemoryGB * (config.maxRamUsagePercent / 100)).toFixed(2));

  const gpuCandidates = [
    ...detectNvidiaGpus(),
    ...(process.platform === "win32" ? detectWindowsDisplayAdapters() : []),
    ...(process.platform === "linux" ? detectLinuxGpus() : []),
    ...(process.platform === "darwin" ? detectMacGpu() : []),
  ];

  const gpus = dedupeGpus(gpuCandidates);
  const primaryGpu = gpus[0] ?? null;
  const storage = detectStorage();
  logger?.info("hardware-scan", {
    platform: process.platform,
    cpu: cpus[0]?.model ?? "Unknown CPU",
    totalMemoryGB,
    gpuCount: gpus.length,
  });

  return {
    platform: process.platform,
    arch: process.arch,
    cpu: {
      model: cpus[0]?.model ?? "Unknown CPU",
      physicalCores: detectPhysicalCores(cpus.length),
      logicalThreads: cpus.length,
    },
    memory: {
      totalGB: totalMemoryGB,
      freeGB: freeMemoryGB,
      availableGB: plannerBudgetGB,
      plannerBudgetGB,
    },
    storage,
    gpus,
    primaryGpu,
    backends: {
      cuda: Boolean(primaryGpu?.backends?.cuda),
      metal: Boolean(primaryGpu?.backends?.metal),
      vulkan: Boolean(primaryGpu?.backends?.vulkan),
      rocm: Boolean(primaryGpu?.backends?.rocm),
      cpu: true,
    },
  };
}
