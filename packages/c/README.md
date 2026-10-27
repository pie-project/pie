# libpie (C API)

In-process C ABI (`include/pie.h`) wrapping `worker::Server`.

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
pie_server_send_frame(server, session, frame, len, &error);
pie_server_recv_frames(server, session, 5000, 64, on_frame, ctx, &received, &error);
pie_server_free(server);
```

## Build

```bash
cargo build --release -p pie-c                    # macOS/iOS (Metal), Android (Vulkan)
cargo build --release -p pie-c --features vulkan  # Linux, Windows (or cuda, wgpu)
```
