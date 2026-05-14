// pie_driver_portable — library entry point.
//
// The portable backend is embedded-only: linked into `pie-server` as the
// `pie_driver_portable_lib` static archive and invoked via the FFI
// vtable handed in by the Rust runtime's `InProcChannel`. There is no
// standalone executable.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Capability handshake callback. Fired exactly once per
// `pie_driver_portable_run_inproc` invocation, after the model is loaded
// but before the driver begins serving requests. `caps_json` points at a
// NUL-terminated UTF-8 string; the buffer is owned by the lib and only
// valid for the duration of the callback. `ctx` is whatever the caller
// passed alongside the callback.
typedef void (*pie_driver_portable_ready_cb)(const char* caps_json, void* ctx);

// In-process entry: the runtime hands us a vtable of FFI callbacks for
// receiving requests / sending responses. The vtable type
// (`pie_portable_driver::PieInProcVTable`) lives in `inproc_server.hpp`
// — declared in entry.cpp since it carries C++-namespaced types.

// Signal the running driver's serve loop to exit. Idempotent; safe to
// call from any thread; safe to call before the serve loop has been
// reached (no-op until then). After this returns, the run thread
// completes its current request (if any), exits the serve loop, and
// `pie_driver_portable_run_inproc` returns 0.
void pie_driver_portable_request_stop(void);

#ifdef __cplusplus
}
#endif
