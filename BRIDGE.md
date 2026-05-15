# `pie-bridge` — Driver/Runtime Interface (Top-Level Overview)

> **Status (2026-05-14):** rkyv-based bridge crate is feature-complete. C++ drivers compile via a compatibility shim, Python drivers are migrated, runtime InProcChannel is wired. 89 tests pass across the bridge surface. Real-GPU e2e is a user-side step (needs Python env + a model).
>
> **Owner:** in.gim@yale.edu

This document is a short pointer to the authoritative docs. The bridge has gone through several design pivots; if you find this file stale, the actual source of truth is in `driver/bridge/`.

## Where to read

| Doc | Contents |
|---|---|
| `driver/bridge/README.md` | What the bridge is, how `#[schema]` works, the macro-emitted surface, evolution rules. **Start here.** |
| `driver/bridge/MIGRATION.md` | Old→new API mapping. Per-consumer migration checklist for the C++ drivers (cuda/portable) and Python drivers (dev/sglang/vllm). |
| `driver/bridge/BENCHMARKS.md` | Wire-format + IPC + direct-FFI latency baselines with interpretation. |

## Architecture (one paragraph)

The Rust structs in `driver/bridge/src/schema.rs` carry `#[schema]`, a proc-macro that derives rkyv `Archive/Serialize/Deserialize` (the wire format) and emits the complete C ABI + PyO3 wrappers for every type:

- **Readers** (`pie_<type>_<field>(...)`) over `*const Archived<T>` — zero-copy.
- **Parse entries** (`pie_parse_<type>(bytes, len) -> *const Archived<T>`).
- **Descriptors** (`#[repr(C)] Pie<T>Desc`) — C-friendly mirrors used by writers and the direct-FFI handoff.
- **Builders** (`pie_build_<type>(*const Pie<T>Desc, out, cap) -> usize`).
- **Views** (`pie_<t>_view(&native) -> Pie<T>View<'_>`) for in-process zero-rkyv handoff.
- **PyO3 classes** (`Py<T>`) gated by the `python` feature.

Adding a field to `schema.rs` produces accessors automatically; the C header (`include/pie_bridge.h`) is the only piece that's hand-maintained, and `tests/desc_layout.rs` catches accidental drift.

## Two transports

| Transport | Path | Latency at 16K tokens | Used by |
|---|---|---:|---|
| **In-process FFI** | runtime ↔ `InProcChannel` ↔ vtable ↔ C++ driver, exchanges `*const PieFrameDesc` | 66 ns (view emission) | cuda, portable |
| **Shmem IPC** | runtime ↔ `ShmemClient` ↔ ring ↔ `ShmemServer` ↔ Python driver, exchanges rkyv bytes | ~74 µs (5.78 µs encode + 68 µs ring) | dev, sglang, vllm |

The in-process path is >1000× faster — encoding cost is irrelevant when no transport hop happens.

## C++ compatibility shim (F1)

`driver/{cuda,portable}/src/_bridge/legacy_view.{hpp,cpp}` redefines the OLD `PieSlice<T>` / `PieInProcRequestView` / `PieForwardRequestView` types locally, populated from the new `PieFrameDesc` on each `recv`. Bulk of the C++ handler code (`forward.cpp`, `request_handler.cpp`, `entry.cpp`) needed no changes.

## Test coverage (89 tests)

- 76 bridge-crate tests across rkyv layout, C-ABI round trip, direct-FFI views, edge cases (alignment, empty fields, truncated input), shmem in-proc, schema round trip, Desc layout regression.
- 7 `InProcChannel` tests in `runtime/src/driver/inproc.rs` covering the vtable handshake + concurrency stress.
- 6 Python ctypes round-trip tests.

## Performance (release profile, single thread)

| Operation | 16 tokens | 16K tokens |
|---|---:|---:|
| rkyv `access` (zero-copy parse) | 42 ns | 42 ns |
| `pie_forward_request_view` (direct-FFI) | 66 ns | 66 ns |
| rkyv `encode` (request side) | 360 ns | 5.78 µs |
| `pie_build_response_frame` (Tier-1 direct-write) | 212 ns | 2.9 µs |
| shmem ring roundtrip (any payload up to a few KB) | ~68 µs | ~68 µs |

See `driver/bridge/BENCHMARKS.md` for analysis. The headline: zero-copy reads are constant-time; direct-FFI is constant-time; IPC is dominated by the ring's wake latency, not by serialization.

## What's _not_ in this bridge crate

- Scheduler/inference logic (sampler dispatch, batch builder, KV cache management) — pie's runtime concerns.
- Model loading, tokenization, adapter persistence — driver internals.
- Authentication, workflows, telemetry — server-level concerns.

The bridge owns the wire schema and the transport primitives; everything else is a consumer.

## Status by consumer

| Consumer | Status |
|---|---|
| pie-bridge crate | ✅ feature-complete |
| pie-driver-dummy (Rust shmem reference) | ✅ |
| pie (runtime) | ✅ `InProcChannel` + `ShmemChannel` wired; 7 unit tests |
| pie-server | ✅ compiles with `driver-portable` + `driver-cuda` |
| dev / sglang / vllm Python | ✅ `_bridge/` adapters migrated; awaits real-model run |
| cuda / portable C++ | ✅ F1 shim compiles both |
| Real GPU e2e on Qwen3-0.6B / inferlet broad suite | ⏭ user-side (Python env setup) |
