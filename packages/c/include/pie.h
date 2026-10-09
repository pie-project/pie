#ifndef PIE_H
#define PIE_H

/* pie in the caller's process: the runtime, this build's engine (Metal on
   Apple, Vulkan on Android, else what the library was built with) and the
   inferlet sandbox. Clients reach it through sessions that carry the
   MessagePack frames of `pie serve`'s WebSocket (crates/client-api), or,
   with `listen`, through the gateway itself.

   Every call blocks and is safe from any thread, including during and after
   pie_server_shutdown (calls then fail with PIE_ERR_SHUT_DOWN); only
   pie_server_free invalidates the handle. A call that fails returns its
   status and, when `error` is non-NULL, stores a message there for
   pie_string_free; a call that succeeds leaves `*error` as it was. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PIE_VERSION_MAJOR 0
#define PIE_VERSION_MINOR 5
#define PIE_VERSION_PATCH 4
#define PIE_VERSION_STRING "0.5.4"

typedef enum pie_status {
    PIE_OK = 0,
    /* A NULL or non-UTF-8 argument, or a config or `listen` that does not parse. */
    PIE_ERR_INVALID_ARGUMENT = 1,
    /* The server was shut down. */
    PIE_ERR_SHUT_DOWN = 2,
    /* The call failed; `*error` says why. */
    PIE_ERR_FAILED = 3,
    /* A bug in pie, caught at the boundary; `*error` carries its message. */
    PIE_ERR_PANIC = 4,
} pie_status;

typedef struct pie_server pie_server;

/* Called once per server frame. It must return normally: no C++ exception
   and no longjmp may cross it. */
typedef void (*pie_frame_fn)(void *ctx, const uint8_t *frame, size_t len);

/* The linked library's version, "MAJOR.MINOR.PATCH"; compare it with
   PIE_VERSION_STRING to catch a header and a library that disagree. */
const char *pie_version(void);

/* Boots `artifact` (a `.zt` for this build's engine) and stores the server in
   `*out`. `config` is JSON or TOML with any of these keys, or NULL for the
   defaults:

     {"max_total_pages": 512, "max_forward_tokens": 512, "max_forward_requests": 8,
      "max_state_slots": 64, "max_model_len": 4096, "gpu_mem_utilization": 0.9,
      "sandbox_memory_mb": 512, "max_concurrent_processes": null, "sku": null,
      "engine": true, "frame_size": 8, "frame_dispatch_depth": 2, "verbose": false}

   `home` is a writable directory the runtime keeps its files under;
   `listen` (`"host:port"`, or NULL) also serves pie's gateway there. One
   server per process. */
pie_status pie_server_start(const char *artifact, const char *config, const char *home,
                            const char *listen, pie_server **out, char **error);

/* The boot summary as JSON; valid until pie_server_free. */
const char *pie_server_summary(const pie_server *server);

/* `"host:port"` the gateway listens on (the OS's port for `listen` port 0),
   or NULL without `listen`; valid until pie_server_free. */
const char *pie_server_listen_addr(const pie_server *server);

/* Installs a program (`file` names it: `x.wasm`, `x.py`, `x.js`); `version`
   may be NULL. Stores its `name@version` in `*name`, for pie_string_free. */
pie_status pie_server_install(pie_server *server, const uint8_t *bytes, size_t len,
                              const char *file, const char *version, char **name,
                              char **error);

/* Installs a language component (`python`, `javascript`). */
pie_status pie_server_install_language(pie_server *server, const char *language,
                                       const uint8_t *bytes, size_t len, char **error);

pie_status pie_server_open_session(pie_server *server, uint32_t *session, char **error);
void pie_server_close_session(pie_server *server, uint32_t session);

/* Hands the session one client frame. */
pie_status pie_server_send_frame(pie_server *server, uint32_t session, const uint8_t *frame,
                                 size_t len, char **error);

/* Waits up to `max_wait_ms` (less once the session closes or the server
   shuts down), calls `on_frame` once per server frame, at most `max_frames`,
   and stores how many in `*received`. */
pie_status pie_server_recv_frames(pie_server *server, uint32_t session, uint32_t max_wait_ms,
                                  size_t max_frames, pie_frame_fn on_frame, void *ctx,
                                  size_t *received, char **error);

/* Stops the runtime and releases the engine. Idempotent. */
void pie_server_shutdown(pie_server *server);

/* Shuts down if needed and frees the handle. */
void pie_server_free(pie_server *server);

void pie_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif
