# pie-bridge migration plan

Status as of 2026-05-14 — bridge crate side is complete. Downstream consumers still reference the pre-rkyv API. This document maps exactly what each consumer needs to do.

## What changed in the bridge

| Old API | New API |
|---|---|
| `pie_bridge::Method` (Rust enum) | `pie_bridge::{ArchivedRequestPayload, ArchivedResponsePayload}` (tagged unions); `PIE_REQUEST_PAYLOAD_*` / `PIE_RESPONSE_PAYLOAD_*` u8 constants |
| `pie_bridge::DriverCapabilities` | _removed; restore as a separate `#[schema]` type if still needed_ |
| `pie_bridge::wire::encode_*` (custom codec) | `pie_bridge::wire::{encode_request, encode_response}` (rkyv) |
| Hand-written `cabi.rs` accessors | Macro-emitted `pie_<type>_<field>` per field |
| `PieInProcRequestView` (typed FFI view, cbindgen) | `PieFrameDesc` (POD with embedded `PieRequestPayloadDesc` tagged union) |
| `PieInProcResponseView` | `PieResponseFrameDesc` |
| `PieForwardRequestView.{token_ids,sampler_types,...}` (SoA, one field per attribute) | `PieForwardRequestDesc.{token_ids_ptr,token_ids_len,…}` + `PieSamplerDesc` (tagged union per sampler) |
| `PieMethod` (enum class via cbindgen) | _gone; dispatch via `PieRequestPayloadDesc.kind`_ |
| `PieSlice<T>` template (`.as<T>()`) | direct `(*const T, size_t)` pairs in Desc fields |
| `InProcVTable.recv(ctx, *PieInProcRequestView, *u32)` | `InProcVTable.recv(ctx, **PieFrameDesc, *u32)` |
| `InProcVTable.send_response(ctx, u32, *u8, usize)` | `InProcVTable.send_response(ctx, u32, *PieResponseFrameDesc)` |
| `pie_bridge.ShmemServer` (PyO3) | `pie_bridge.Frame.parse(bytes)` + ctypes for build helpers |
| `pie_bridge.ResponseBuilder` (PyO3) | `pie_bridge.build_forward_response(...)` (constructs PieResponseFrameDesc + calls pie_build_response_frame) |
| `pie_bridge.parse_request(bytes) -> dict` | `pie_bridge.Frame.parse(bytes)` returns a Frame wrapper; navigate via `.payload_kind`, `.as_forward()`, etc. |
| `pie_bridge.sampler_type.*` constants | `pie_bridge.SAMPLER_*` constants |

The sampler model changed shape too: old was SoA arrays (`sampler_types[i]`, `sampler_temperatures[i]`, …); new is `Vec<Sampler>` as an AoS of tagged-union `PieSamplerDesc`. Per-sampler dispatch reads `sampler_descs[i].kind` and the relevant attribute fields.

## Migration checklist

### A. Bridge crate
- [x] rkyv `#[schema]` types + macro emission (readers, writers, views, vtable, PyO3)
- [x] Tier 1 direct-write builder
- [x] Direct-FFI `pie_<t>_view()` for in-process callers
- [x] 44 Rust tests + 6 Python tests pass
- [x] `DriverCapabilities` re-added (non-rkyv JSON struct, server-side use)
- [x] `ShmemServer` + `Lease` PyO3 wrappers re-added (needed by Python downstream drivers)
- [x] `SCHEMA_HASH` exposed to Python so downstream can pass to `ShmemClient::open` (handshake check)
- [x] `schema_module!` macro refactored to emit a helper fn so `python.rs` can add extra classes

### B. Runtime (`runtime/`)
- [ ] `runtime/src/driver/inproc.rs`: replace the stub `InProcChannel` with a real implementation
  - [ ] `InProcChannel::new()` (no args) returns an empty channel; `ffi_vtable()` produces an `InProcVTable` whose `recv`/`send_response` callbacks dispatch into the channel's per-req-id queue
  - [ ] `InProcChannel::release(group_id)` static method (used by server on shutdown — current callers expect it)
  - [ ] `submit(req)` enqueues a `DriverRequest`, parks an awaitable response slot, returns when `send_response` callback fires for that req_id
  - [ ] Build a `PieFrameView<'static>` from `DriverRequest` and park it (heap allocation owned by the channel)
  - [ ] In `send_response` callback: `__pie_response_frame_from_desc(&*desc)` → native `ResponseFrame` → signal the awaiting slot

### C. Server (`server/`)
- [ ] `server/src/embedded_driver.rs`: rebuild `InProcChannel::new()` call sites once runtime exposes the new API
- [ ] Resolve `pie_bridge::DriverCapabilities` import — either re-add as a `#[schema]` type or move ownership into the runtime/server crate
- [ ] Verify `pie_server` compiles with `--no-default-features` and with `--features driver-portable`/`driver-cuda`

### D. Python vllm driver (`driver/vllm/src/pie_driver_vllm/_bridge/`)
- [ ] `methods.py`: rewrite using `pie_bridge.SAMPLER_*` / `pie_bridge.REQUEST_*` constants
- [ ] `shmem_ipc.py`: `ShmemServer` was a PyO3 class — re-expose via a small PyO3 module or move shmem ring logic into pure Python here (the C ABI emits `pie_parse_frame` / `pie_build_*` already)
- [ ] `shmem_schema.py`: replace `pie_bridge.parse_request(view)` with `pie_bridge.Frame.parse(bytes(view))` and convert the returned wrapper to the dict the worker expects; replace `pie_bridge.ResponseBuilder` with a thin Python adapter that constructs `PieForwardResponseDesc` field-by-field and calls `pie_build_response_frame`
- [ ] `_bridge/__init__.py`: re-export constants under the names the rest of pie_driver_vllm expects

### E. Python sglang / tensorrt_llm drivers
- [ ] Mirror D for `driver/sglang/` and `driver/tensorrt_llm/` — same `_bridge/` subpackage pattern, same migration steps

### F. C++ portable (`driver/portable/`)
- [ ] `src/_bridge/inproc_server.{hpp,cpp}`: rewrite to dispatch via `PieRequestPayloadDesc.kind` instead of `req.method`. Two layout options:
  - **(F1) Adapter shim**: hide the new API inside `_bridge/`. Provide an "old-style" `PieInProcRequestView`/`PieForwardRequestView` (locally defined) that's populated from `PieFrameDesc` on each `recv`. The bulk of `request_handler.cpp` / `entry.cpp` doesn't change. ~300 LOC adapter, isolated.
  - **(F2) Native migration**: rewrite `request_handler.cpp` etc. to read `PieFrameDesc` fields directly. Larger touch (~1500 LOC across cuda+portable) but cleaner long-term.
- [ ] Adjust `entry.cpp`'s method dispatch (`req.method == PIE_METHOD_FORWARD` → `req->payload.kind == PIE_REQUEST_PAYLOAD_FORWARD`)
- [ ] Adjust sampler loop in `request_handler.cpp` (was SoA via `view.sampler_types.as<u32>()` etc.; now AoS via `view.payload.forward.samplers_ptr[i].kind`)
- [ ] Update `CMakeLists.txt` to consume the new `pie_bridge.h` location (already at `driver/bridge/include/pie_bridge.h`, but `bridge_cxx/` references may be stale)

### G. C++ cuda (`driver/cuda/`)
- [ ] Mirror F. Same shape of changes, larger codebase (~1057 + 1614 + 280 LOC touching the bridge surface).
- [ ] Test on real GPU (L40, CUDA 12.4) via the inferlet broad suite

### H. Dummy driver (`driver/dummy/`)
- [ ] Likely needs no changes — it's pure Rust and probably uses `pie_bridge::*` types directly. Verify it still compiles after server compiles.

## Recommended order

1. **B** (runtime InProcChannel) — unblocks everything else.
2. **C** (server) — depends on B; small.
3. **D** (dev Python) — independent track; needed for Python e2e.
4. **F1** (portable C++, shim approach) — fastest path to a working C++ build.
5. **G1** (cuda C++, same shim pattern as F1) — gets GPU e2e working.
6. **E** (sglang/vllm Python) — same pattern as D.
7. Later: convert F1/G1 from shim → native field access if profiling shows the shim conversion is hot.

After 1–5: Python e2e via `dev` driver works (shmem path). After 5: cuda e2e works (in-process direct-FFI path).

## Approach notes for the C++ shim (F1 / G1)

`_bridge/inproc_server.hpp` re-exports + defines compatibility types:

```cpp
namespace pie_driver {
    template<typename T> struct PieSlice {
        const T* ptr; size_t len;
        std::span<const T> as() const { return {ptr, len}; }
    };
    struct PieForwardRequestView {
        PieSlice<uint32_t> token_ids;
        PieSlice<uint32_t> position_ids;
        // ... mirror of old SoA shape
        PieSlice<uint32_t> sampler_types;
        PieSlice<float>    sampler_temperatures;
        // ... etc.
    };
    enum PieMethod : uint32_t { FORWARD = 0, COPY_D2H = 1, /* ... */ };
    struct PieInProcRequestView { uint32_t method; PieForwardRequestView forward; /* ... */ };
}
```

`_bridge/inproc_server.cpp::serve_forever` builds the old-style view from the new `PieFrameDesc`:

```cpp
void InProcServer::serve_forever(const RequestHandler& handler) {
    while (!stop_) {
        const PieFrameDesc* desc = nullptr;
        uint32_t req_id = 0;
        if (vt_.recv(vt_.ctx, &desc, &req_id) != 0) break;

        PieInProcRequestView view{};
        switch (desc->payload.kind) {
            case PIE_REQUEST_PAYLOAD_FORWARD: {
                view.method = PIE_METHOD_FORWARD;
                build_forward_view(view.forward, desc->payload.forward, arena_);
                break;
            }
            // ... other variants
        }

        PieInProcResponseView out{};
        handler(req_id, view, out);

        PieResponseFrameDesc response{};
        build_response_desc(response, out);
        vt_.send_response(vt_.ctx, req_id, &response);
    }
}
```

The sampler conversion (AoS → SoA) lives in `build_forward_view`: iterate `desc->payload.forward.samplers_ptr[0..len]`, fill the appropriate `sampler_temperatures`/`sampler_top_k`/etc. arrays from a per-batch `arena_`. The arena clears between requests.

This keeps the bulk of `request_handler.cpp` and `entry.cpp` byte-for-byte unchanged.
