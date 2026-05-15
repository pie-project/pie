# pie-bridge

Canonical wire schema (rkyv) + shmem IPC for the pie runtime ↔ driver interface.

## How the schema works

The Rust types in `src/schema.rs` ARE the wire format. Each carries `#[schema]`, which is a proc-macro (`driver/bridge-macros`) that derives `rkyv::{Archive, Serialize, Deserialize}` *and* emits the entire C-ABI + PyO3 surface:

- **Reader**: `pie_<type>_<field>(*const ArchivedT) -> ...` per field
- **Parse**: `pie_parse_<type>(bytes, len) -> *const ArchivedT`
- **Descriptor**: `#[repr(C)] Pie<T>Desc` — C-friendly mirror used by writers
- **Builder**: `pie_build_<type>(*const Pie<T>Desc, out, cap) -> usize`
- **PyO3**: `Py<T>` class with all of the above as methods/getters
- **Enum dispatch**: `pie_<enum>_kind`, `pie_<enum>_as_<variant>` (data enums); `pie_<enum>_value` (unit enums)
- **Discriminant consts**: `PIE_<ENUM>_<VARIANT>: u8`

Symbol names derive mechanically from the type name in snake_case — no per-field attributes, no hand-written accessors. `#[schema]` itself takes no arguments; users add `#[derive(Default, Copy, PartialEq, Eq)]` as normal Rust derives where needed.

## Layout

- `src/schema.rs` — the schema types. Single source of truth.
- `src/wire.rs` — thin helpers around `rkyv::to_bytes` / `rkyv::access`.
- `src/ipc/` — shmem ring transport (protocol-agnostic).
- `src/ffi/` — in-process FFI for Rust drivers.
- `src/python.rs` — one-line `schema_module!{...}` listing every type for PyO3.
- `include/pie_bridge.h` — C header for non-Rust drivers.
- `python/pie_bridge/` — ctypes wrapper used by Python downstreams.
- `tests/` — Rust round-trip tests against both rkyv directly and the C ABI.

## Features

- `cabi` (default) — emits the `extern "C"` reader + builder surface.
- `ipc` (default) — shmem ring transport.
- `python` — PyO3 wrappers.

## Cross-language story

| Consumer                   | Path                                              |
|----------------------------|---------------------------------------------------|
| Rust (`pie`, dummy driver) | Import schema types directly                      |
| C++ (`driver/cuda`, etc.)  | Link `libpie_bridge`, include `pie_bridge.h`     |
| Python (`driver/dev`, …)   | `pip`-install the maturin wheel; use ctypes or PyO3 |

All three read the same rkyv bytes; the C ABI and PyO3 surfaces are just navigation helpers over the archived form.

## Evolution rules

- Append fields only — removing or reordering changes the archived layout.
- Append enum variants only — the discriminant byte is on the wire.
- Type changes (e.g. `u32` → `u64`) are protocol breaks. Bump `ipc::shmem::MAGIC` to force older binaries to refuse the connection.
- The `SCHEMA_HASH` (build-script hash of `src/schema.rs`) goes into the shmem handshake; mismatched versions fail loudly at connect time.
