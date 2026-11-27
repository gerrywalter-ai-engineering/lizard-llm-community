// Runtime adapter seam (model wrapper Phase 6, spec s5).
//
// The spec asks for a uniform interface -
//   canLoad / prepare / load / generate / stream / unload / health
// - so that adding a backend becomes "implement an adapter" rather than
// "rewrite the app". This phase installs the seam WITHOUT changing execution
// behavior: canLoad and health are real (they answer from the UniversalModel
// descriptor and the provider registry, the same facts the app already uses);
// the execution verbs delegate to the existing per-engine code paths through a
// declared boundary rather than reimplementing them.
//
// The point is the shape and canLoad. A future real backend (or the
// Safetensors slot below) plugs in here instead of touching call sites.

import { RUNTIME_STATUS } from "../../lib/universal-model.js";
import { getRuntimeProvider, runtimeProviderSupportsCapability } from "../../lib/runtime-providers.js";

// Which UniversalModel.runtimes[].id maps to which registered provider.
const ADAPTER_TO_PROVIDER = {
  native: "lizard-native",
  caterpillar: "caterpillar",
  ostrich: "ostrich",
  "llama.cpp": "llama.cpp",
  ollama: "Ollama",
};

function runtimeVerdict(model, adapterId) {
  const rows = Array.isArray(model?.runtimes) ? model.runtimes : [];
  // native adapter also covers the caterpillar engine (both are the native
  // GGUF path); the descriptor only carries a single "native" row.
  const key = (adapterId === "caterpillar" || adapterId === "ostrich") ? "native" : adapterId;
  return rows.find((r) => r.id === key) ?? null;
}

class RuntimeAdapter {
  constructor(adapterId) {
    this.id = adapterId;
    this.providerId = ADAPTER_TO_PROVIDER[adapterId] ?? adapterId;
  }

  // Pure decision from the descriptor - no I/O, no side effects.
  canLoad(model) {
    const verdict = runtimeVerdict(model, this.id);
    if (!verdict) {
      return { ok: false, reason: `No ${this.id} verdict for this model.` };
    }
    if (verdict.status === RUNTIME_STATUS.SUPPORTED) {
      return { ok: true, reason: null };
    }
    if (verdict.status === RUNTIME_STATUS.IMPORTABLE) {
      return { ok: false, importable: true, reason: verdict.reason };
    }
    if (verdict.status === RUNTIME_STATUS.PULLABLE) {
      return { ok: false, pullable: true, reason: verdict.reason };
    }
    return { ok: false, reason: verdict.reason ?? `${this.id} cannot load this model.` };
  }

  // Health = is this backend registered and does the app consider it able to
  // run in this build. Real answer, no execution.
  health() {
    const provider = getRuntimeProvider(this.providerId);
    if (!provider) return { ok: false, reason: `Unknown provider "${this.providerId}".` };
    return {
      ok: provider.implementationStatus === "implemented"
        || provider.implementationStatus === "experimental"
        || provider.implementationStatus === "partial",
      status: provider.implementationStatus,
      canExecuteInDashboard: runtimeProviderSupportsCapability(this.providerId, "dashboardExecution"),
      reason: null,
    };
  }

  // Execution verbs delegate to the existing engine paths. This phase keeps
  // behavior identical, so these route through the caller-provided delegate
  // rather than a new pipeline. A backend that has no delegate (Safetensors)
  // refuses here as clearly as it does at canLoad.
  async prepare(model, ctx = {}) { return this.#delegate("prepare", model, ctx); }
  async load(model, ctx = {}) { return this.#delegate("load", model, ctx); }
  async generate(request, ctx = {}) { return this.#delegate("generate", request, ctx); }
  async stream(request, ctx = {}) { return this.#delegate("stream", request, ctx); }
  async unload(model, ctx = {}) { return this.#delegate("unload", model, ctx); }

  #delegate(verb, arg, ctx) {
    const delegate = ctx?.delegates?.[verb];
    if (typeof delegate === "function") return delegate(arg, ctx);
    throw new Error(
      `Runtime adapter "${this.id}" has no ${verb} delegate wired. `
      + `Execution still runs through the existing engine path; the adapter seam `
      + `only owns canLoad/health in this phase.`,
    );
  }
}

// The honest Safetensors slot the spec asks for (s5): the interface exists, and
// canLoad tells the truth - there is no local Safetensors runtime to execute
// it. Phase 7 makes such models route to Ollama import where possible.
class SafetensorsAdapter extends RuntimeAdapter {
  constructor() { super("safetensors"); }
  canLoad() {
    return {
      ok: false,
      reason: "No local Safetensors runtime is installed. Safetensors models are routed to a GGUF conversion or an Ollama import where the architecture allows it.",
    };
  }
  health() { return { ok: false, status: "not-installed", canExecuteInDashboard: false, reason: "No Safetensors runtime." }; }
}

const REGISTRY = new Map();

export function createRuntimeAdapter(adapterId) {
  if (adapterId === "safetensors") {
    if (!REGISTRY.has(adapterId)) REGISTRY.set(adapterId, new SafetensorsAdapter());
    return REGISTRY.get(adapterId);
  }
  if (!REGISTRY.has(adapterId)) REGISTRY.set(adapterId, new RuntimeAdapter(adapterId));
  return REGISTRY.get(adapterId);
}

// For a UniversalModel, the adapters that can actually load it right now, best
// first (native > caterpillar > llama.cpp > ollama). This is the "compatibility
// resolver -> runtime adapter" arrow from the spec's final architecture, as a
// pure function over the descriptor.
export function resolveLoadableAdapters(model) {
  const order = ["native", "caterpillar", "ostrich", "llama.cpp", "ollama"];
  return order
    .map((id) => ({ id, adapter: createRuntimeAdapter(id), verdict: createRuntimeAdapter(id).canLoad(model) }))
    .filter((entry) => entry.verdict.ok)
    .map((entry) => entry.id);
}

export const ADAPTER_IDS = Object.freeze([...Object.keys(ADAPTER_TO_PROVIDER), "safetensors"]);
