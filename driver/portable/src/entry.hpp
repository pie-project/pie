// pie_driver_portable — library entry point.
//
// Exposes the driver's main loop as a C-callable function so the same
// translation units back both the standalone executable
// (`pie_driver_portable`, used by the Python subprocess path) and the
// in-process static library (`pie_driver_portable_lib`, linked into
// `server/standalone`'s Rust binary).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Capability handshake callback. Fired exactly once per `pie_driver_portable_run`
// invocation, after the model is loaded and the shmem server is open but
// before the driver begins serving requests. `caps_json` points at a
// NUL-terminated UTF-8 string with the same JSON shape the executable
// emits as its `READY <json>` line; the buffer is owned by the lib and
// only valid for the duration of the callback. `ctx` is whatever the
// caller passed alongside the callback in `pie_driver_portable_run`.
typedef void (*pie_driver_portable_ready_cb)(const char* caps_json, void* ctx);

// Run the driver with the given argv. Returns a process-style exit code.
//
// `install_signal_handlers != 0` installs SIGINT/SIGTERM handlers that
// stop the shmem server. The standalone executable passes 1; library
// callers (e.g. server/standalone) pass 0 because the host owns signal
// handling.
//
// `ready_cb` must be non-NULL: it is the only mechanism the driver uses
// to publish capabilities. The standalone executable provides a default
// callback that writes `READY <json>` to stdout (preserving the Python
// wrapper's protocol); library callers route the JSON wherever they want.
int pie_driver_portable_run(int argc,
                            char** argv,
                            int install_signal_handlers,
                            pie_driver_portable_ready_cb ready_cb,
                            void* ready_ctx);

// Signal the running driver's shmem-serve loop to exit. Idempotent;
// safe to call from any thread; safe to call before `pie_driver_portable_run`
// has reached the serve loop (no-op until then). After this returns,
// the run thread completes its current request (if any), exits the
// serve loop, and `pie_driver_portable_run` returns 0.
//
// Single-instance only — v0 supports one in-process driver at a time.
// Multi-driver will need a handle-based API.
void pie_driver_portable_request_stop(void);

#ifdef __cplusplus
}
#endif
