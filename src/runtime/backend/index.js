function normalizeVector(values, label) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error(`${label} must be a non-empty numeric array.`);
  }
  return values.map((value, index) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      throw new Error(`${label}[${index}] must be a finite number.`);
    }
    return numeric;
  });
}

function normalizeMatrix(rows, label) {
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error(`${label} must be a non-empty matrix.`);
  }
  const matrix = rows.map((row, index) => normalizeVector(row, `${label}[${index}]`));
  const width = matrix[0].length;
  if (!matrix.every((row) => row.length === width)) {
    throw new Error(`${label} rows must all have the same width.`);
  }
  return matrix;
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

const NATIVE_BACKEND_MATRIX = {
  cpu: {
    supported: true,
    planned: false,
    featureDetection: "always-on",
    quantizations: ["f32"],
    architectures: ["tiny-linear"],
    reason: "Experimental native CPU backend is the only executable backend today.",
  },
  cuda: {
    supported: false,
    planned: true,
    featureDetection: "capability-registry",
    quantizations: [],
    architectures: [],
    reason: "CUDA-native graph execution is planned but not implemented yet.",
  },
  metal: {
    supported: false,
    planned: true,
    featureDetection: "capability-registry",
    quantizations: [],
    architectures: [],
    reason: "Metal-native graph execution is planned but not implemented yet.",
  },
  vulkan: {
    supported: false,
    planned: true,
    featureDetection: "capability-registry",
    quantizations: [],
    architectures: [],
    reason: "Vulkan-native graph execution is planned but not implemented yet.",
  },
};

const NATIVE_BACKEND_TEST_MATRIX = [
  {
    backend: "cpu",
    status: "active",
    requiredCoverage: [
      "tiny-model correctness",
      "golden output",
      "memory guardrails",
      "long-context shell stability",
    ],
  },
  {
    backend: "cuda",
    status: "planned",
    requiredCoverage: [
      "kernel correctness parity",
      "quantized matmul coverage",
      "fallback safety",
    ],
  },
  {
    backend: "metal",
    status: "planned",
    requiredCoverage: [
      "graph parity on Apple Silicon",
      "memory residency checks",
      "fallback safety",
    ],
  },
  {
    backend: "vulkan",
    status: "planned",
    requiredCoverage: [
      "dispatch parity",
      "driver capability validation",
      "fallback safety",
    ],
  },
];

function normalizeBackendKey(value, fallback = "cpu") {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized || fallback;
}

export function buildNativeBackendFeatureRegistry({
  capabilityRegistry = null,
  requestedBackend = null,
} = {}) {
  const requested = normalizeBackendKey(requestedBackend, "");
  const capabilityBackends = capabilityRegistry?.backends ?? {};
  const backends = Object.fromEntries(Object.entries(NATIVE_BACKEND_MATRIX).map(([backend, spec]) => {
    const capability = capabilityBackends[backend] ?? null;
    return [backend, {
      ...clone(spec),
      detected: backend === "cpu"
        ? true
        : Boolean(capability?.supported),
      detectionReason: capability?.reason ?? spec.reason,
      warnings: capability?.warnings ?? [],
    }];
  }));

  const selectedBackend = requested && backends[requested]
    ? requested
    : "cpu";
  const selected = backends[selectedBackend] ?? backends.cpu;

  return {
    engine: "lizard-native",
    selectedBackend,
    backends,
    fallbackPolicy: {
      explicitOnly: true,
      defaultRoute: selected.supported ? selectedBackend : "cpu",
    },
    reason: selected.reason,
  };
}

export function buildNativeBackendTestMatrix() {
  return clone(NATIVE_BACKEND_TEST_MATRIX);
}

export function resolveNativeBackendRoute({
  capabilityRegistry = null,
  requestedBackend = null,
  allowBackendFallback = false,
} = {}) {
  const registry = buildNativeBackendFeatureRegistry({
    capabilityRegistry,
    requestedBackend,
  });
  const backend = registry.backends[registry.selectedBackend] ?? registry.backends.cpu;
  if (backend.supported) {
    return {
      backend: registry.selectedBackend,
      registry,
      fallbackUsed: false,
      fallbackReason: null,
    };
  }
  if (allowBackendFallback) {
    return {
      backend: "cpu",
      registry,
      fallbackUsed: true,
      fallbackReason: `Experimental native backend '${registry.selectedBackend}' is not implemented yet. Falling back to CPU because explicit backend fallback was allowed.`,
    };
  }
  throw new Error(`Experimental native backend '${registry.selectedBackend}' is not implemented yet. ${backend.reason} Use backend 'cpu' or explicitly allow backend fallback.`);
}

export function createCpuBackend() {
  return {
    kind: "cpu",
    supportedQuantizations: ["f32"],
    matmulVectorMatrix(vector, matrix) {
      const normalizedVector = normalizeVector(vector, "vector");
      const normalizedMatrix = normalizeMatrix(matrix, "matrix");
      if (normalizedVector.length !== normalizedMatrix.length) {
        throw new Error("CPU backend matmul requires vector length to match matrix height.");
      }
      const columns = normalizedMatrix[0].length;
      const output = new Array(columns).fill(0);
      for (let row = 0; row < normalizedVector.length; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          output[column] += normalizedVector[row] * normalizedMatrix[row][column];
        }
      }
      return output;
    },
    addBias(vector, bias = null) {
      const normalizedVector = normalizeVector(vector, "vector");
      if (bias == null) {
        return [...normalizedVector];
      }
      const normalizedBias = normalizeVector(bias, "bias");
      if (normalizedVector.length !== normalizedBias.length) {
        throw new Error("CPU backend bias length must match vector width.");
      }
      return normalizedVector.map((value, index) => value + normalizedBias[index]);
    },
    argmax(values) {
      const normalizedValues = normalizeVector(values, "values");
      let bestIndex = 0;
      for (let index = 1; index < normalizedValues.length; index += 1) {
        if (normalizedValues[index] > normalizedValues[bestIndex]) {
          bestIndex = index;
        }
      }
      return bestIndex;
    },
  };
}
