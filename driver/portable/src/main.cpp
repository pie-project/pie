// pie_driver_portable — standalone executable shim.
//
// All logic lives in entry.cpp / pie_driver_portable_lib so the same
// translation units back both this executable and the static lib that
// `server/standalone` links into the all-Rust binary.

#include <iostream>

#include "entry.hpp"

namespace {

// Default capability handshake: emit `READY <json>` on stdout so the
// Python wrapper (`pie/src/pie_driver_portable/worker.py`) can pick it
// up via line-buffered stdin. The library callers in server/standalone
// supply their own callback to receive the JSON in-process.
void default_ready_to_stdout(const char* caps_json, void* /*ctx*/) {
    std::cout << "READY " << caps_json << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    return pie_driver_portable_run(argc, argv,
                                   /*install_signal_handlers=*/1,
                                   default_ready_to_stdout,
                                   /*ready_ctx=*/nullptr);
}
