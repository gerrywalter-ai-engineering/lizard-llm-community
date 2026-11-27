// LibreHardwareMonitor bridge.
//
// Windows does not let an ordinary process read CPU die temperatures. The
// counters are behind model-specific registers, which need ring 0, which is
// why the plain WMI probe in system-telemetry.js gets "access denied" on most
// consumer laptops. LibreHardwareMonitor (MPL-2.0) solves that with a signed
// kernel driver, and publishes everything it reads over a small HTTP server.
//
// So the split is: LHM runs elevated and owns the driver, and Lizard stays an
// ordinary unelevated process that only reads JSON over loopback. Nothing here
// installs, elevates, or loads a driver.
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const DEFAULT_ENDPOINT = "http://127.0.0.1:8085/data.json";
const REQUEST_TIMEOUT_MS = 2_000;

// The copy bundled by the Windows installer (see stageSensorService in
// scripts/build-windows-installer.mjs). Absent in a source checkout.
export function bundledLibreExecutable(appRoot) {
  if (process.platform !== "win32" || !appRoot) return null;
  const candidate = path.join(appRoot, "vendor", "LibreHardwareMonitor", "LibreHardwareMonitor.exe");
  return fs.existsSync(candidate) ? candidate : null;
}

/**
 * Starts the bundled sensor service elevated. It reads model-specific
 * registers through a kernel driver, so it cannot work unelevated - and the
 * Lizard installer deliberately requires no admin rights, so the elevation
 * prompt belongs here, at the moment the user asks for temperatures, rather
 * than at install time for everyone.
 */
export function sensorSnapshotPath() {
  const base = process.env.LOCALAPPDATA || process.env.APPDATA || process.env.TEMP || ".";
  return path.join(base, "lizard-llm", "sensors.json");
}

export function startBundledLibreHardwareMonitor(appRoot) {
  const library = bundledLibreLibrary(appRoot);
  const agent = bundledSensorAgent(appRoot);
  if (!library || !agent) {
    return { started: false, reason: "The sensor service is not bundled with this build." };
  }
  const quote = (value) => "'" + String(value).replace(/'/g, "''") + "'";
  try {
    // The agent, not LibreHardwareMonitor.exe. The GUI's remote web server can
    // only be switched on by clicking its menu item - the stored setting is
    // loaded into a UserOption whose Changed handler is attached after
    // construction, so a pre-seeded config is silently ignored (MainForm.cs
    // 316-323, v0.9.6). Driving the library from our own agent also drops the
    // GUI, the HTTP port and the URL ACL.
    const child = spawn("powershell.exe", [
      "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
      "Start-Process -FilePath 'powershell.exe' -Verb RunAs -WindowStyle Hidden -ArgumentList "
      + [
        "'-NoProfile'", "'-ExecutionPolicy'", "'Bypass'", "'-File'", quote(agent),
        "'-LibraryPath'", quote(library), "'-OutputPath'", quote(sensorSnapshotPath()),
      ].join(","),
    ], { windowsHide: true, detached: true, stdio: "ignore" });
    child.unref();
    return { started: true, agent };
  } catch (error) {
    return { started: false, reason: error?.message ?? "could not start the sensor service" };
  }
}

export function bundledLibreLibrary(appRoot) {
  if (process.platform !== "win32" || !appRoot) return null;
  const candidate = path.join(appRoot, "vendor", "LibreHardwareMonitor", "LibreHardwareMonitorLib.dll");
  return fs.existsSync(candidate) ? candidate : null;
}

export function bundledSensorAgent(appRoot) {
  if (process.platform !== "win32" || !appRoot) return null;
  const candidate = path.join(appRoot, "scripts", "assets", "lizard-sensor-agent.ps1");
  return fs.existsSync(candidate) ? candidate : null;
}

// The agent writes a snapshot already in this module's normalized shape, so it
// needs no locale parsing at all - PowerShell serialises numbers invariantly.
const SNAPSHOT_MAX_AGE_MS = 15_000;

function readSensorSnapshot() {
  const file = sensorSnapshotPath();
  try {
    if (!fs.existsSync(file)) return null;
    // A snapshot older than a few poll intervals means the agent stopped;
    // showing its last readings as live would be worse than showing none.
    if (Date.now() - fs.statSync(file).mtimeMs > SNAPSHOT_MAX_AGE_MS) return null;
    // Windows PowerShell 5.1 writes UTF-8 *with* a BOM, and JSON.parse rejects
    // a leading U+FEFF outright - the snapshot read as absent with the file
    // sitting there, fresh, on disk.
    const parsed = JSON.parse(fs.readFileSync(file, "utf8").replace(/^﻿/, ""));
    const hardware = Array.isArray(parsed?.hardware) ? parsed.hardware : [];
    return hardware.length > 0 ? hardware : null;
  } catch {
    return null;
  }
}

export function libreEndpoint(env = process.env) {
  return env?.LIZARD_LHM_ENDPOINT || DEFAULT_ENDPOINT;
}

// LHM formats values for display in the operator's locale, so a temperature
// arrives as "45,0 °C" on a German machine and "45.0 °C" on an English one,
// and a clock as "2.803,4 MHz" or "2,803.4 MHz". Strip the unit, then work out
// which separator is the decimal one from whichever comes last.
// Units whose values are large enough to carry a thousands separator. Anything
// else (V, °C, %, W) is read as a decimal.
const GROUPED_UNITS = /^(MHz|GHz|RPM|KB|MB|GB|TB|KB\/s|MB\/s|GB\/s)$/i;

export function parseLocalizedNumber(text) {
  const raw = String(text ?? "").trim();
  if (!raw) return null;
  const numeric = raw.replace(/[^\d,.\-]/g, "");
  if (!numeric || numeric === "-") return null;

  const lastComma = numeric.lastIndexOf(",");
  const lastDot = numeric.lastIndexOf(".");
  let normalized = numeric;
  if (lastComma >= 0 && lastDot >= 0) {
    // Both present: the one further right is the decimal point.
    normalized = lastComma > lastDot
      ? numeric.replace(/\./g, "").replace(",", ".")
      : numeric.replace(/,/g, "");
  } else if (lastComma >= 0 || lastDot >= 0) {
    const separator = lastComma >= 0 ? "," : ".";
    // A lone separator with exactly three digits after it is genuinely
    // ambiguous: "3.204 RPM" is three thousand on a German machine, and
    // "1.096 V" is one point one on an English one. The unit settles it -
    // clocks, fan speeds and byte counts get grouped, voltages have three
    // decimal places and never reach a thousand.
    const grouped = new RegExp("\\" + separator + "\\d{3}$").test(numeric)
      && GROUPED_UNITS.test(parseUnit(raw) ?? "");
    normalized = grouped
      ? numeric.split(separator).join("")
      : numeric.replace(separator, ".");
  }
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

export function parseUnit(text) {
  const match = String(text ?? "").match(/[°]?[A-Za-z%/]+$/);
  if (!match) return null;
  return match[0].trim() || null;
}

const HARDWARE_KINDS = [
  ["cpu", /cpu/i],
  ["gpu", /gpu|ati|nvidia/i],
  ["memory", /ram|memory/i],
  ["storage", /hdd|ssd|nvme/i],
  ["motherboard", /mainboard|motherboard/i],
  ["network", /nic|network/i],
  ["battery", /battery/i],
  ["cooler", /fan|cooler/i],
];

function hardwareKind(node) {
  const haystack = String(node?.ImageURL || "") + " " + String(node?.Text || "");
  for (const [kind, pattern] of HARDWARE_KINDS) {
    if (pattern.test(haystack)) return kind;
  }
  return "other";
}

// LHM groups a device's sensors under a category node ("Temperatures",
// "Clocks", "Load", "Powers", "Fans", "Voltages", "Data", "Throughput"), so
// the category is simply the parent of every leaf.
const CATEGORY_BY_UNIT = {
  "°C": "Temperatures",
  "%": "Load",
  W: "Powers",
  V: "Voltages",
  MHz: "Clocks",
  GHz: "Clocks",
  RPM: "Fans",
  GB: "Data",
  MB: "Data",
  "MB/s": "Throughput",
  "KB/s": "Throughput",
};

/**
 * Flattens LHM's nested Sensor tree into one hardware list. Tolerant by
 * design: the shape has changed across LHM versions and a missing branch must
 * degrade to fewer sensors, never to a throw.
 */
export function parseLibreSensorTree(root) {
  const hardware = [];

  const walk = (node, device, category) => {
    if (!node || typeof node !== "object") return;
    const text = String(node.Text ?? "").trim();
    const children = Array.isArray(node.Children) ? node.Children : [];
    const hasValue = typeof node.Value === "string" && node.Value.trim() !== "";

    if (hasValue && children.length === 0) {
      if (!device) return;
      const value = parseLocalizedNumber(node.Value);
      if (value === null) return;
      const unit = parseUnit(node.Value);
      device.sensors.push({
        name: text || "sensor",
        category: category || CATEGORY_BY_UNIT[unit] || "Other",
        value,
        unit,
        min: parseLocalizedNumber(node.Min),
        max: parseLocalizedNumber(node.Max),
        sensorId: node.SensorId ? String(node.SensorId) : null,
      });
      return;
    }

    // A node carrying an ImageURL is a device; the machine root and the
    // category nodes never do.
    const isDevice = Boolean(node.ImageURL) && children.length > 0 && !device;
    if (isDevice) {
      const next = { name: text || "device", kind: hardwareKind(node), sensors: [] };
      hardware.push(next);
      for (const child of children) walk(child, next, null);
      return;
    }

    for (const child of children) {
      walk(child, device, device ? (text || category) : null);
    }
  };

  walk(root, null, null);
  return hardware.filter((device) => device.sensors.length > 0);
}

function findSensor(device, category, patterns) {
  if (!device) return null;
  for (const pattern of patterns) {
    const hit = device.sensors.find((sensor) => sensor.category === category
      && pattern.test(sensor.name));
    if (hit) return hit;
  }
  return null;
}

function maxSensor(device, category, pattern) {
  if (!device) return null;
  let best = null;
  for (const sensor of device.sensors) {
    if (sensor.category !== category || !pattern.test(sensor.name)) continue;
    if (!best || sensor.value > best.value) best = sensor;
  }
  return best;
}

/**
 * The few readings the thermal cards need, lifted out of the full sensor list.
 * Package temperature is preferred over per-core: it is the number a vendor
 * quotes and the one thermal throttling is keyed to. Hottest core is the
 * fallback, because an average across cores hides exactly the spike that
 * matters.
 */
export function summarizeLibreSensors(hardware) {
  const cpu = hardware.find((device) => device.kind === "cpu") ?? null;
  const gpu = hardware.find((device) => device.kind === "gpu") ?? null;

  const cpuTemp = findSensor(cpu, "Temperatures", [/package/i, /^cpu total$/i])
    ?? maxSensor(cpu, "Temperatures", /core/i)
    ?? (cpu?.sensors.find((sensor) => sensor.category === "Temperatures") ?? null);
  const gpuTemp = findSensor(gpu, "Temperatures", [/^gpu core$/i, /core/i, /hot ?spot/i])
    ?? (gpu?.sensors.find((sensor) => sensor.category === "Temperatures") ?? null);

  return {
    cpu: {
      name: cpu?.name ?? null,
      temperatureC: cpuTemp?.value ?? null,
      temperatureSensor: cpuTemp?.name ?? null,
      packagePowerW: findSensor(cpu, "Powers", [/package/i, /total/i])?.value ?? null,
      clockMhz: maxSensor(cpu, "Clocks", /core/i)?.value ?? null,
    },
    gpu: {
      name: gpu?.name ?? null,
      temperatureC: gpuTemp?.value ?? null,
      temperatureSensor: gpuTemp?.name ?? null,
      hotSpotC: findSensor(gpu, "Temperatures", [/hot ?spot/i])?.value ?? null,
      powerW: findSensor(gpu, "Powers", [/^gpu package$/i, /package/i, /total/i])?.value ?? null,
      coreClockMhz: findSensor(gpu, "Clocks", [/^gpu core$/i, /core/i])?.value ?? null,
      memoryUsedMb: findSensor(gpu, "Data", [/memory used/i])?.value ?? null,
      memoryTotalMb: findSensor(gpu, "Data", [/memory total/i])?.value ?? null,
    },
    fans: hardware.flatMap((device) => device.sensors
      .filter((sensor) => sensor.category === "Fans")
      .map((sensor) => ({ device: device.name, name: sensor.name, rpm: sensor.value }))),
    storageTemps: hardware
      .filter((device) => device.kind === "storage")
      .map((device) => ({
        device: device.name,
        temperatureC: device.sensors.find((s) => s.category === "Temperatures")?.value ?? null,
      }))
      .filter((entry) => entry.temperatureC !== null),
  };
}

/**
 * Reads LHM's remote web server. Returns null - never throws - whenever LHM is
 * not running, the web server is off, or the reply is not what we expect; the
 * caller falls back to the WMI probe and the UI explains what is missing.
 */
export async function readLibreHardwareMonitor({ endpoint = libreEndpoint() } = {}) {
  // Our own elevated agent first; the LHM GUI's web server second, so someone
  // already running it with the remote server switched on still works.
  const fromAgent = readSensorSnapshot();
  if (fromAgent) {
    return { endpoint: "lizard-sensor-agent", hardware: fromAgent, summary: summarizeLibreSensors(fromAgent) };
  }
  let payload = null;
  try {
    const response = await fetch(endpoint, {
      cache: "no-store",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (!response.ok) return null;
    payload = await response.json();
  } catch {
    return null;
  }
  let hardware = [];
  try {
    hardware = parseLibreSensorTree(payload);
  } catch {
    return null;
  }
  if (hardware.length === 0) return null;
  return {
    endpoint,
    hardware,
    summary: summarizeLibreSensors(hardware),
  };
}
