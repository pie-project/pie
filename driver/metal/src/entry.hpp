// pie_driver_metal — library entry point.
//
// Same shape as driver/cuda/src/entry.hpp and driver/portable/src/entry.hpp:
// the translation units here back both the standalone executable
// (`pie_driver_metal`) and the in-process static library
// (`pie_driver_metal_lib`), which `pie-worker` links when built with
// `--features driver-metal`.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Capability handshake callback. Fires once after the driver loads and is
// ready, before it enters its serve loop. `caps_json` is owned by the lib
// and only valid for the duration of the callback.
typedef void (*pie_driver_metal_ready_cb)(const char* caps_json, void* ctx);

// Standalone entry: parses argv, loads the config, emits the capability
// handshake, and returns. This path does NOT serve requests (there is no
// host in-process channel) — it exists for the `pie_driver_metal`
// executable as a build/config self-check. Returns a process-style exit
// code. `ready_cb` must be non-NULL.
int pie_driver_metal_run(int argc,
                         char** argv,
                         int install_signal_handlers,
                         pie_driver_metal_ready_cb ready_cb,
                         void* ready_ctx);

// An in-process variant (`pie_driver_metal_run_inproc`) is also exported
// from entry.cpp; its signature uses a C++-namespaced vtable type
// (`pie_driver::PieInProcVTable`) so we keep its declaration out of this
// C-style header. See `pie_ipc/inproc_server.hpp`.

// Signal the running driver's serve loop to exit. Idempotent; safe to call
// from any thread; no-op until the serve loop is reached.
void pie_driver_metal_request_stop(void);

#ifdef __cplusplus
}
#endif
