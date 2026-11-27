import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  parseLocalizedNumber,
  parseUnit,
  parseLibreSensorTree,
  summarizeLibreSensors,
  readLibreHardwareMonitor,
} from "../src/lib/libre-hardware-monitor.js";

// LibreHardwareMonitor formats every value for display in the operator's
// locale, so the same machine reports "61,0 °C" in Germany and "61.0 °C" in
// the US. Reading these as raw numbers turns 3204 RPM into 3.204 RPM.
test("localized values parse identically in comma and dot locales", () => {
  for (const [de, en, expected] of [
    ["2.803,4 MHz", "2,803.4 MHz", 2803.4],
    ["61,0 °C", "61.0 °C", 61],
    ["3.204 RPM", "3,204 RPM", 3204],
    ["1,096 V", "1.096 V", 1.096],
    ["1.148,0 MB", "1,148.0 MB", 1148],
  ]) {
    assert.equal(parseLocalizedNumber(de), expected, `de: ${de}`);
    assert.equal(parseLocalizedNumber(en), expected, `en: ${en}`);
  }
});

test("a lone separator before three digits is read from the unit", () => {
  // Genuinely ambiguous in isolation: three thousand two hundred RPM, or one
  // point zero nine six volts. Only the unit distinguishes them.
  assert.equal(parseLocalizedNumber("3.204 RPM"), 3204);
  assert.equal(parseLocalizedNumber("1.096 V"), 1.096);
  assert.equal(parseLocalizedNumber("3,204 RPM"), 3204);
  assert.equal(parseLocalizedNumber("1,096 V"), 1.096);
});

test("unreadable values yield null rather than NaN", () => {
  for (const input of [null, undefined, "", "   ", "-", "n/a", "°C"]) {
    assert.equal(parseLocalizedNumber(input), null, JSON.stringify(input));
  }
});

test("parseUnit keeps the degree sign and per-second units", () => {
  assert.equal(parseUnit("61,0 °C"), "°C");
  assert.equal(parseUnit("18,4 W"), "W");
  assert.equal(parseUnit("120,5 MB/s"), "MB/s");
  assert.equal(parseUnit("45 %"), "%");
});

function sampleTree() {
  return {
    id: 0,
    Text: "Sensor",
    Children: [{
      id: 1,
      Text: "DESKTOP-TEST",
      Children: [
        {
          id: 2,
          Text: "Intel Core i7-1165G7",
          ImageURL: "images_icon/cpu.png",
          Children: [
            { Text: "Temperatures", Children: [
              { Text: "CPU Core #1", Value: "61,0 °C", Max: "92,0 °C" },
              { Text: "CPU Core #2", Value: "78,0 °C", Max: "95,0 °C" },
              { Text: "CPU Package", Value: "64,0 °C", Max: "97,0 °C" },
            ] },
            { Text: "Powers", Children: [{ Text: "CPU Package", Value: "18,4 W" }] },
            { Text: "Clocks", Children: [{ Text: "CPU Core #1", Value: "2.803,4 MHz" }] },
          ],
        },
        {
          id: 3,
          Text: "Intel Iris Xe Graphics",
          ImageURL: "images_icon/gpu.png",
          Children: [
            { Text: "Temperatures", Children: [{ Text: "GPU Core", Value: "58,0 °C" }] },
          ],
        },
        {
          id: 4,
          Text: "Samsung SSD 980 PRO",
          ImageURL: "images_icon/hdd.png",
          Children: [{ Text: "Temperatures", Children: [{ Text: "Temperature", Value: "44,0 °C" }] }],
        },
        {
          id: 5,
          Text: "MSI Mainboard",
          ImageURL: "images_icon/mainboard.png",
          Children: [{ Text: "Fans", Children: [{ Text: "CPU Fan", Value: "3.204 RPM" }] }],
        },
      ],
    }],
  };
}

test("the sensor tree flattens into devices with categorised sensors", () => {
  const hardware = parseLibreSensorTree(sampleTree());
  assert.deepEqual(hardware.map((device) => device.kind), ["cpu", "gpu", "storage", "motherboard"]);
  const cpu = hardware[0];
  assert.equal(cpu.sensors.length, 5);
  assert.equal(cpu.sensors.filter((s) => s.category === "Temperatures").length, 3);
  assert.equal(cpu.sensors.find((s) => s.name === "CPU Package" && s.category === "Powers").value, 18.4);
});

test("CPU temperature prefers the package sensor over the hottest core", () => {
  const summary = summarizeLibreSensors(parseLibreSensorTree(sampleTree()));
  // Package is 64 and the hottest core is 78: package is the number the vendor
  // quotes and the one throttling is keyed to, so it must win.
  assert.equal(summary.cpu.temperatureC, 64);
  assert.equal(summary.cpu.temperatureSensor, "CPU Package");
  assert.equal(summary.cpu.packagePowerW, 18.4);
  assert.equal(summary.gpu.temperatureC, 58);
  assert.deepEqual(summary.fans, [{ device: "MSI Mainboard", name: "CPU Fan", rpm: 3204 }]);
  assert.deepEqual(summary.storageTemps, [{ device: "Samsung SSD 980 PRO", temperatureC: 44 }]);
});

test("the hottest core is the fallback when no package sensor exists", () => {
  const tree = sampleTree();
  const cpu = tree.Children[0].Children[0];
  const temps = cpu.Children.find((group) => group.Text === "Temperatures");
  temps.Children = temps.Children.filter((sensor) => sensor.Text !== "CPU Package");
  const summary = summarizeLibreSensors(parseLibreSensorTree(tree));
  // An average across cores would hide exactly the spike that matters.
  assert.equal(summary.cpu.temperatureC, 78);
  assert.equal(summary.cpu.temperatureSensor, "CPU Core #2");
});

test("a malformed tree degrades to fewer sensors instead of throwing", () => {
  assert.deepEqual(parseLibreSensorTree(null), []);
  assert.deepEqual(parseLibreSensorTree({}), []);
  assert.deepEqual(parseLibreSensorTree({ Children: [{ Text: "no image url" }] }), []);
});

test("an unreachable LibreHardwareMonitor reads as null, not an error", async () => {
  // Point the snapshot lookup at an empty directory first: the elevated agent
  // takes precedence over the HTTP endpoint, and on a machine where it is
  // running this would otherwise read its live values instead of exercising
  // the unreachable path.
  const previous = process.env.LOCALAPPDATA;
  process.env.LOCALAPPDATA = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-lhm-"));
  try {
    // Port 1 never has an HTTP server on it. The whole point is that the caller
    // falls back to the WMI probe and the UI explains what is missing, rather
    // than a rejected promise reaching the dashboard request path.
    const result = await readLibreHardwareMonitor({ endpoint: "http://127.0.0.1:1/data.json" });
    assert.equal(result, null);
  } finally {
    if (previous === undefined) delete process.env.LOCALAPPDATA;
    else process.env.LOCALAPPDATA = previous;
  }
});

test("a stale snapshot is ignored rather than shown as live", async () => {
  const previous = process.env.LOCALAPPDATA;
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lizard-lhm-"));
  process.env.LOCALAPPDATA = root;
  try {
    const file = path.join(root, "lizard-llm", "sensors.json");
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const snapshot = {
      hardware: [{
        name: "Test CPU",
        kind: "cpu",
        sensors: [{ name: "CPU Package", category: "Temperatures", value: 64, unit: "°C" }],
      }],
    };
    // Windows PowerShell writes UTF-8 with a BOM, and JSON.parse rejects a
    // leading U+FEFF - the snapshot read as absent with the file on disk.
    fs.writeFileSync(file, "﻿" + JSON.stringify(snapshot), "utf8");
    const fresh = await readLibreHardwareMonitor({ endpoint: "http://127.0.0.1:1/data.json" });
    assert.equal(fresh?.summary.cpu.temperatureC, 64);

    // The agent stopping must not leave its last readings on screen as live.
    const old = Date.now() - 120_000;
    fs.utimesSync(file, old / 1000, old / 1000);
    const stale = await readLibreHardwareMonitor({ endpoint: "http://127.0.0.1:1/data.json" });
    assert.equal(stale, null);
  } finally {
    if (previous === undefined) delete process.env.LOCALAPPDATA;
    else process.env.LOCALAPPDATA = previous;
  }
});
