# pie for C

`libpie` runs pie inside the caller's process: the runtime, this build's
engine and the inferlet sandbox, booted by the same worker `pie serve` runs
(`worker::Server`). Anything with a C FFI can link it (Dart, C#, Go, Python's
ctypes, C++); the Swift package does.

```c
#include "pie.h"

char *error = NULL;
PieServer *server = pie_server_start("qwen.vulkan.zt", NULL, "/var/lib/myapp/pie", NULL, &error);
uint32_t session;
pie_server_open_session(server, &session, &error);
pie_server_send_frame(server, session, frame, len, &error);   /* a MessagePack ClientMessage */
pie_server_recv_frames(server, session, 5000, 64, on_frame, ctx, &error);
pie_server_free(server);
```

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
Windows). `test/smoke.c` exercises the ABI:

```bash
cc test/smoke.c -Iinclude -L../../target/release -lpie -o smoke && ./smoke [model.zt]
```
