// pie_driver_cuda — standalone executable shim.
//
// All logic lives in entry.cpp / pie_driver_cuda_lib so the same
// translation units back both this executable and the static lib that
// `server/standalone` links into the all-Rust binary when built with
// `--features driver-cuda`.

#include <iostream>

#include "entry.hpp"

namespace {

// Default capability handshake: emit `READY <json>` on stdout so the
// Python wrapper (`pie/src/pie_driver_cuda_native/worker.py`) can pick
// it up via line-buffered stdout. Library callers in server/standalone
// supply their own callback to receive the JSON in-process.
void default_ready_to_stdout(const char* caps_json, void* /*ctx*/) {
    std::cout << "READY " << caps_json << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    // Standalone mode already surfaces the fatal reason on stderr; no fatal_cb.
    return pie_driver_cuda_run(argc, argv,
                               /*install_signal_handlers=*/1,
                               default_ready_to_stdout,
                               /*ready_ctx=*/nullptr,
                               /*fatal_cb=*/nullptr,
                               /*fatal_ctx=*/nullptr);
}
