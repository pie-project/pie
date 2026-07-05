// pie_driver_metal — standalone executable shim.
//
// All logic lives in entry.cpp / pie_driver_metal_lib so the same
// translation units back both this executable and the static lib that
// `pie-worker` links into the all-Rust binary when built with
// `--features driver-metal`.
//
// The standalone path is a build/config self-check: it loads the config
// and emits the capability handshake, then exits (serving happens only via
// the in-process `pie_driver_metal_run_inproc` entry the host drives).

#include <iostream>

#include "entry.hpp"

namespace {

void default_ready_to_stdout(const char* caps_json, void* /*ctx*/) {
    std::cout << "READY " << caps_json << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    return pie_driver_metal_run(argc, argv,
                                /*install_signal_handlers=*/1,
                                default_ready_to_stdout,
                                /*ready_ctx=*/nullptr);
}
