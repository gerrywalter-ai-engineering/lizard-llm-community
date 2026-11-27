#pragma once

namespace lizard_native {

// `lizard-native-runner --probe-hardware`: enumerate the CPU (CPUID +
// OS-enabled extended state), system memory, and every DXGI adapter with
// stable identity (LUID), D3D12 feature/architecture facts, a minimal
// dispatch/readback self-test, and live video-memory budgets. Prints one
// JSON object to stdout and returns 0 even when individual adapters fail -
// a failed probe is a reportable fact, not a process error. Phase 1 of
// .Agent/firstBgRefactor/01_firstPhaseBigRefactor.md (Detection stages A-C).
int run_hardware_probe_main();

}  // namespace lizard_native
