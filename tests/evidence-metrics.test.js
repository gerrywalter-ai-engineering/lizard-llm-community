import test from "node:test";
import assert from "node:assert/strict";

import {
  compactEvidenceMetricsLine,
  evidenceMetricFields,
  extractEvidenceMetrics,
  formatEvidenceMetrics,
} from "../src/lib/evidence-metrics.js";

test("evidence metrics normalize the native worker decode contract", () => {
  const source = {
    tokensPerSecond: 9.5,
    nativeExecution: {
      sessionMetrics: {
        evidenceMetrics: {
          schemaVersion: 1,
          measurementScope: "decode",
          sampleSource: "native-worker-process",
          tokensPerSecond: 9.5,
          normalizedCpuUtilizationPercent: 57.5,
          peakWorkingSetBytes: 2_500_000_000,
          peakPrivateBytes: 504_000_000,
        },
      },
    },
  };
  const evidence = extractEvidenceMetrics(source);
  assert.equal(evidence.tokensPerSecond, 9.5);
  assert.equal(evidence.normalizedCpuUtilizationPercent, 57.5);
  assert.equal(Number(evidence.tokensPerSecondPerFullCpu.toFixed(4)), 16.5217);
  assert.equal(evidence.peakWorkingSetBytes, 2_500_000_000);
  assert.match(compactEvidenceMetricsLine(evidence), /cpu=57\.50%/);
  assert.match(formatEvidenceMetrics(evidence).join("\n"), /CPU efficiency/);
  assert.equal(evidenceMetricFields(source).workerPrivateBytes, 504_000_000);
});

test("older results without worker evidence remain compatible", () => {
  assert.equal(extractEvidenceMetrics({ tokensPerSecond: 8 }), null);
  assert.deepEqual(evidenceMetricFields({ tokensPerSecond: 8 }), {});
  assert.deepEqual(formatEvidenceMetrics(null), []);
});
