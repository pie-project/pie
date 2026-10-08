#ifndef PIE_SERVER_H
#define PIE_SERVER_H

/* pie in-process: the runtime and the Metal engine, reached through sessions
   that carry the MessagePack frames of `pie serve`'s WebSocket
   (crates/client-api). Calls block; a failed call returns NULL or nonzero and
   stores a message in `*error` (when non-NULL) to free with pie_string_free.
   Every call is safe during and after pie_server_shutdown (it then fails);
   only pie_server_free invalidates the handle. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PieServer PieServer;

/* Boots `artifact` (a `.metal.zt`). `config` is a runtime::embed::BootConfig
   as TOML or JSON, or NULL; `home` is a writable directory. One per process. */
PieServer *pie_server_start(const char *artifact, const char *config,
                            const char *home, char **error);

/* The boot summary as JSON; valid until pie_server_free. */
const char *pie_server_summary(const PieServer *server);

/* Installs a program (`file` names it: `x.wasm`, `x.py`, ...); `version` may
   be NULL. Returns its `name@version`. */
char *pie_server_install(const PieServer *server, const uint8_t *bytes, size_t len,
                         const char *file, const char *version, char **error);

/* Installs a language component (`python`, `javascript`). */
int32_t pie_server_install_language(const PieServer *server, const char *language,
                                    const uint8_t *bytes, size_t len, char **error);

int32_t pie_server_open_session(const PieServer *server, uint32_t *session, char **error);
void pie_server_close_session(const PieServer *server, uint32_t session);

/* Hands the session one client frame. */
int32_t pie_server_send_frame(const PieServer *server, uint32_t session,
                              const uint8_t *frame, size_t len, char **error);

/* Waits up to `max_wait_ms` (less if the session closes or the server shuts
   down) and calls `sink` once per server frame, at most `max`. Returns the
   count, or -1. */
typedef void (*pie_frame_sink)(void *ctx, const uint8_t *frame, size_t len);
int32_t pie_server_recv_frames(const PieServer *server, uint32_t session, uint32_t max_wait_ms,
                               uint32_t max, pie_frame_sink sink, void *ctx, char **error);

/* Stops the runtime and releases the engine. Idempotent. */
void pie_server_shutdown(const PieServer *server);

/* Shuts down if needed and frees the handle. */
void pie_server_free(PieServer *server);

void pie_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif
