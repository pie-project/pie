# pie for C

`libpie` runs pie inside the caller's process: the runtime, this build's
engine and the inferlet sandbox, booted by the same worker `pie serve` runs
(`worker::Server`). Anything with a C FFI can link it (Dart, C#, Go, Python's
ctypes, C++); the Swift package does.

```c
#include "pie.h"

pie_server *server = NULL;
char *error = NULL;
if (pie_server_start("qwen.vulkan.zt", NULL, "/var/lib/myapp/pie", NULL, &server, &error) != PIE_OK) {
    fprintf(stderr, "%s\n", error);
    pie_string_free(error);
    return 1;
}
uint32_t session;
size_t received;
pie_server_open_session(server, &session, &error);
pie_server_send_frame(server, session, frame, len, &error);   /* a MessagePack ClientMessage */
pie_server_recv_frames(server, session, 5000, 64, on_frame, ctx, &received, &error);
pie_server_free(server);
```

Every fallible call returns a `pie_status` (`PIE_OK`, or why it failed) and
writes its results through out-parameters; `pie_version()` against
`PIE_VERSION_STRING` catches a header and a library that disagree.

Sessions carry the MessagePack frames of `pie serve`'s WebSocket
(crates/client-api); `listen` (`"127.0.0.1:8080"`) also serves the gateway
itself, HTTP routes included. `config` is `worker::embedded::Settings` as JSON
or TOML (`{"max_total_pages": 512, "gpu_mem_utilization": 0.6}`). Every call's
contract is in `include/pie.h`.

## Build

```bash
cargo build --release -p pie-c                        # Apple: Metal; Android: Vulkan
cargo build --release -p pie-c --features vulkan      # Linux, Windows (or cuda, wgpu)
```

That writes `target/release/libpie.{a,so,dylib}` (`pie.dll` and `pie.lib` on
Windows). `test/smoke.c` exercises the ABI (CI runs it):

```bash
cc test/smoke.c -Iinclude -L../../target/release -lpie -o smoke && ./smoke [model.zt]
```
