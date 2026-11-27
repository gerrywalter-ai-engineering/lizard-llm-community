// EnhancementRun_04 Phase 2 -- the step-loop engine.
//
// The runner has always been a batch-of-one machine by construction: the stdio
// loop reads a line, computes it to completion, and writes a line. Sessions
// exist but strictly take turns, and there is no point in the control flow at
// which a second sequence could be advanced. That is the structural reason
// continuous batching is impossible today, and it is what this file changes.
//
// This phase changes STRUCTURE ONLY. `max_batch` is 1, so exactly one sequence
// is ever running and the work performed is identical to the synchronous path.
// The gate is byte-identical output and an unmodified existing test suite --
// the proof that the refactor moved the walls without moving the furniture.
//
// The split is deliberate:
//
//   reader thread  parses lines and enqueues commands. Never computes, never
//                  touches the model, so a slow model cannot stall the pipe.
//   engine thread  the ONLY thread that touches weights. Runs step(), which in
//                  this phase dispatches exactly one command per iteration.
//   writer         any thread may write, under a mutex, one whole line at a
//                  time. Interleaved half-lines would corrupt the protocol.
//
// Nothing in here knows what a model is. The dispatch hook is supplied by
// main.cpp, which keeps this translation unit free of runner internals and
// keeps the rebase surface on main.cpp down to a call site.

#ifndef LIZARD_NATIVE_ENGINE_LOOP_H
#define LIZARD_NATIVE_ENGINE_LOOP_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace lizard_native {

// LIZARD_NATIVE_ENGINE_LOOP=1 opts in. Default off: until the gate passes, the
// synchronous loop remains the shipping path, and reverting is one env var.
bool engine_loop_enabled();

// Bounded so a fast producer cannot grow memory without limit. When the queue
// is full the reader blocks, which applies backpressure up the pipe rather
// than buffering an unbounded amount of unprocessed work.
std::size_t engine_loop_queue_capacity();

struct EngineLoopHooks {
  // Handles one request line, including writing its own output. Called on the
  // engine thread only.
  //
  // The handler writes rather than returning a line because one request can
  // produce many: a streaming generate emits a `generate.token` notification
  // per token and then a response. Returning "the response" would either
  // serialise nothing useful or force the streaming path to buffer a whole
  // completion before any of it reached the client.
  //
  // Must not throw. The loop has no channel to report an exception on.
  std::function<void(const std::string &request_line)> handle_line;

  // Polled after every handled line. True ends the loop.
  std::function<bool()> shutdown_requested;
};

// Runs until stdin closes or `shutdown_requested` returns true. Returns the
// process exit code.
int run_engine_loop(const EngineLoopHooks &hooks);

// Telemetry for the `metrics` RPC. Zeroed when the loop is not running, so a
// reader can tell "not enabled" from "enabled and idle".
struct EngineLoopTelemetry {
  bool running = false;
  std::uint64_t steps = 0;
  std::uint64_t commands_handled = 0;
  std::uint64_t queue_depth_max = 0;
  std::uint64_t reader_blocked_count = 0;
};

EngineLoopTelemetry engine_loop_telemetry();

}  // namespace lizard_native

#endif  // LIZARD_NATIVE_ENGINE_LOOP_H
