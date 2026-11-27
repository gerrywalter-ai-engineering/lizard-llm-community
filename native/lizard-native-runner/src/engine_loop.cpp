#include "engine_loop.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace lizard_native {
namespace {

struct EngineLoopState {
  std::mutex mutex;
  std::condition_variable not_empty;
  std::condition_variable not_full;
  std::deque<std::string> queue;
  bool input_closed = false;

  std::atomic<bool> running {false};
  std::atomic<std::uint64_t> steps {0};
  std::atomic<std::uint64_t> commands_handled {0};
  std::atomic<std::uint64_t> queue_depth_max {0};
  std::atomic<std::uint64_t> reader_blocked_count {0};
};

EngineLoopState &state() {
  static EngineLoopState instance;
  return instance;
}

bool is_blank(const std::string &line) {
  for (const char character : line) {
    if (character != ' ' && character != '\t' && character != '\r' && character != '\n') {
      return false;
    }
  }
  return true;
}

}  // namespace

bool engine_loop_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_ENGINE_LOOP");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

std::size_t engine_loop_queue_capacity() {
  static const std::size_t capacity = [] {
    const char *value = std::getenv("LIZARD_NATIVE_ENGINE_QUEUE");
    if (value == nullptr) {
      return static_cast<std::size_t>(64);
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return static_cast<std::size_t>(parsed > 0 ? parsed : 64);
  }();
  return capacity;
}

EngineLoopTelemetry engine_loop_telemetry() {
  EngineLoopState &engine = state();
  EngineLoopTelemetry telemetry;
  telemetry.running = engine.running.load(std::memory_order_relaxed);
  telemetry.steps = engine.steps.load(std::memory_order_relaxed);
  telemetry.commands_handled = engine.commands_handled.load(std::memory_order_relaxed);
  telemetry.queue_depth_max = engine.queue_depth_max.load(std::memory_order_relaxed);
  telemetry.reader_blocked_count = engine.reader_blocked_count.load(std::memory_order_relaxed);
  return telemetry;
}

int run_engine_loop(const EngineLoopHooks &hooks) {
  if (!hooks.handle_line) {
    return 1;
  }

  EngineLoopState &engine = state();
  engine.running.store(true, std::memory_order_relaxed);

  // Reader: parse-and-enqueue only. Keeping compute off this thread is what
  // lets a request be accepted while an earlier one is still decoding -- the
  // property the whole run is built on, even though at max_batch 1 the engine
  // still finishes them one at a time.
  std::thread reader([&engine] {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (is_blank(line)) {
        continue;
      }
      std::unique_lock<std::mutex> lock(engine.mutex);
      if (engine.queue.size() >= engine_loop_queue_capacity()) {
        engine.reader_blocked_count.fetch_add(1, std::memory_order_relaxed);
        engine.not_full.wait(lock, [&engine] {
          return engine.queue.size() < engine_loop_queue_capacity();
        });
      }
      engine.queue.push_back(std::move(line));
      const std::uint64_t depth = static_cast<std::uint64_t>(engine.queue.size());
      std::uint64_t observed = engine.queue_depth_max.load(std::memory_order_relaxed);
      while (depth > observed
        && !engine.queue_depth_max.compare_exchange_weak(observed, depth, std::memory_order_relaxed)) {
        // observed is refreshed by compare_exchange_weak
      }
      lock.unlock();
      engine.not_empty.notify_one();
      line.clear();
    }

    {
      std::lock_guard<std::mutex> lock(engine.mutex);
      engine.input_closed = true;
    }
    engine.not_empty.notify_all();
  });

  // Engine: the only thread that touches the model. One step handles one
  // command at max_batch 1; the loop shape is what later phases fill in with
  // an actual batch.
  int exit_code = 0;
  for (;;) {
    std::string request;
    {
      std::unique_lock<std::mutex> lock(engine.mutex);
      engine.not_empty.wait(lock, [&engine] {
        return !engine.queue.empty() || engine.input_closed;
      });
      if (engine.queue.empty()) {
        break;  // input closed and drained
      }
      request = std::move(engine.queue.front());
      engine.queue.pop_front();
    }
    engine.not_full.notify_one();

    engine.steps.fetch_add(1, std::memory_order_relaxed);
    hooks.handle_line(request);
    engine.commands_handled.fetch_add(1, std::memory_order_relaxed);

    if (hooks.shutdown_requested && hooks.shutdown_requested()) {
      break;
    }
  }

  engine.running.store(false, std::memory_order_relaxed);

  // A shutdown request leaves the reader parked on getline. Detaching rather
  // than joining is deliberate: the process is exiting, and joining would hang
  // until the peer happens to close the pipe.
  reader.detach();
  return exit_code;
}

}  // namespace lizard_native
