# pie-rpc

Python bindings for `pie::device::RpcServer` — the IPC primitive that
lets a Python driver receive cold-path requests from the pie runtime
and return responses.

Pairs with the shmem fast path that each driver mounts at
`/pie_shmem_g{group_id}` for `fire_batch`. The two channels are
independent: shmem carries the hot path (one batch step), `pie_rpc`
carries everything else (capability negotiation, KV swap RPCs,
control plane).

## Install

This wheel is a build-time dep of the driver wheels; you usually don't
install it directly. If you do:

```sh
uv pip install pie-rpc
```

## Use

```python
from pie_rpc import RpcServer

server = RpcServer.create()
print(server.server_name())  # publish to the parent via the handshake

while True:
    req = server.poll_blocking(timeout_ms=1000)
    if req is None:
        continue
    request_id, method, payload, _ts = req
    result = handle(method, payload)
    server.respond(request_id, result)
```

`server_name()` is the IPC handle the Rust runtime dials in via
`RpcClient::connect`. The handshake protocol — how the Python launcher
ships this name back to the parent process so it can be wired into
`bootstrap::DeviceConfig.hostname` — is each driver's responsibility
(see each `driver/<name>/src/pie_driver_<name>/__main__.py`).
