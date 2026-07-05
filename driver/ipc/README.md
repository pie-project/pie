# pie-ipc

In-proc IPC mechanism for the pie runtime ↔ driver interface: the POSIX
shared-memory ring, the in-process C-ABI vtable, and the rkyv wire
helpers. The wire-schema *vocabulary* itself lives in the
[`pie-driver-abi`](../../interface/driver) crate (the dependency floor); this
crate is the transport machinery layered over it.

## Layout

- `src/wire.rs` — thin encode/access/deserialize helpers around
  `rkyv::to_bytes` / `rkyv::access` over `pie_driver_abi`'s frame types.
- `src/ipc/` — shmem ring transport (`ShmemServer` / `ShmemClient` /
  `Lease`), with per-platform park/wake (`linux`/`macos`/`windows`/
  `fallback`, POSIX setup shared via `posix`).
- `src/ffi.rs` — in-process FFI vtable (`InProcVTable`,
  `FfiRequestSink` / `FfiResponseSource`) for in-tree Rust/C++ drivers.
- `src/python.rs` — PyO3 wrappers (`ShmemServer` / `Lease`) for Python
  downstream drivers. Schema `Py<T>` classes live in `pie-driver-abi`.
- `include/pie_ipc.h` — C header for the in-proc vtable
  (`PieInProcVTable`) plus the header-only `pie_ipc/inproc_server.hpp`
  loop; both include `pie_driver_abi.h` for the descriptor types.
- `tests/` — shmem round-trip + wire/cabi tests.

## Features

- `ipc` (default) — shmem ring transport.
- `cabi` (default) — forwards to `pie-driver-abi/cabi` so the C-ABI
  descriptor accessors are available to downstream drivers.
- `python` — PyO3 shmem wrappers (forwards to `pie-driver-abi/python`).

## Cross-language story

| Consumer                   | Path                                                        |
|----------------------------|-------------------------------------------------------------|
| Rust (`pie`, dummy driver) | `pie_ipc::{ipc, ffi, wire}` + schema types from `pie_driver_abi`|
| C++ (`driver/cuda`, etc.)  | Include `pie_ipc.h` + `pie_driver_abi.h`; fill the vtable        |
| Python (`driver/vllm`, …)  | `pie_ipc` maturin wheel → `from pie_ipc import ShmemServer`  |

## Handshake

The `SCHEMA_HASH` (built in `pie-driver-abi` from the schema sources,
re-exported as `pie_ipc::SCHEMA_HASH`) is written into the shmem ring
header and compared on connect; mismatched schema versions fail loudly
at handshake time. Bump `ipc::MAGIC` to force older binaries to refuse a
wire-incompatible connection.
