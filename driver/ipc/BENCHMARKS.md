# pie-ipc benchmarks

Measured on Linux x86_64, single thread, release profile. Run:

```
cargo bench -p pie-ipc --features cabi --bench wire
cargo bench -p pie-ipc --features cabi --bench direct_ffi
cargo bench -p pie-ipc --features "cabi ipc" --bench shmem
cargo bench -p pie --bench driver_channel
```

Workload: a `Frame::Forward(ForwardRequest)` with N tokens, 1 sampler, 1 adapter binding. The interesting axis is the number of tokens — the slice-bearing Vec fields dominate the encode/decode cost.

## Read side — rkyv encode + access

| Tokens | encode_forward | access_forward | read_all_fields (parse + touch each Vec) |
|---:|---:|---:|---:|
| 16 | 360 ns | 42 ns | 42 ns |
| 256 | 414 ns | 42 ns | 42 ns |
| 4K | 1.58 µs | 42 ns | 42 ns |
| 16K | 5.78 µs | 42 ns | 42 ns |

- **`access_forward` is constant time** (~42 ns) regardless of payload size. rkyv's zero-copy access is just a header validate + pointer cast.
- **`encode_forward` scales linearly** with payload bytes (~120 GB/s at 16K tokens). Matches a single memcpy of the 60 KB token+position+indptr blob plus rkyv's bookkeeping.
- `read_all_fields` matches `access_forward` — touching the slice header (ptr+len) is free because rkyv lazy-resolves; no per-token cost unless you actually iterate.

## Write side — `pie_build_response_frame`

(After Tier 1 — direct-write into the caller's `out_buf` via `rkyv::api::high::to_bytes_in`.)

| Tokens | build_response |
|---:|---:|
| 16 | 212 ns |
| 256 | 233 ns |
| 4K | 795 ns |
| 16K | 2.9 µs |

- Smaller payload than request (no samplers/masks/spec arrays); roughly half of encode_forward at same N.
- Linear scaling matches encode_forward shape; Tier 1 dropped one memcpy + one allocation vs the original `to_bytes` + copy_to(out_buf).

## Direct-FFI view path (in-process)

| Tokens | forward_view | frame_view |
|---:|---:|---:|
| 16 | 66 ns | 132 ns |
| 256 | 66 ns | 132 ns |
| 4K | 66 ns | 132 ns |
| 16K | 66 ns | 132 ns |

- **Constant time regardless of size** — `pie_forward_request_view` just plumbs slice ptrs from native Vec headers into the `Pie<T>Desc`. Vec data is *not* touched.
- The only non-constant work is the Vec<Sampler> AoS→Pie<T>Desc demux into a holder, but with 1 sampler in the workload that's a single allocation.
- `frame_view` is 2× `forward_view` because it wraps a forward-view internally + adds the discriminant byte for the tagged union.

## Runtime channel roundtrip — embedded FFI vs subprocess IPC

The production-path comparison now lives in `runtime/benches/driver_channel.rs`:

```
cargo bench -p pie --bench driver_channel
```

This benchmark replaces the old bridge-only `roundtrip_paths` bench. The old
bench was useful for transport intuition, but it was not the actual runtime
flow: it used custom Rust queues, built `pie_frame_view` on the producer side,
returned only a status code for FFI, and skipped the real `InProcChannel`
pending map, vtable callbacks, response completion, C++ legacy-view demux, and
Rust-side response-desc conversion.

`driver_channel` measures the real runtime channel entry points:

- `inproc_channel/*` calls `pie::driver::InProcChannel::submit`, uses the real
  `ffi_vtable()` callbacks, real pending `HashMap`, real inbox `Mutex` +
  `Condvar`, real `pie_frame_view` construction inside `vt_recv`, and real
  response-slot completion with `PieResponseFrameDesc` -> owned
  `DriverResponse` conversion inside `vt_send_response`.
- The fake embedded driver thread mirrors the C++ `InProcServer` loop closely
  enough for channel benchmarking: it calls `recv`, demuxes the forward
  descriptor into legacy SoA scratch, builds a small `ForwardResponse` desc,
  then calls `send_response`.
- `shmem_channel/*` calls `pie::driver::ShmemChannel::submit`, which performs
  the runtime's real request encode, `ShmemClient::roundtrip`, response parse,
  and owned response deserialize.
- The fake subprocess driver uses a real `pie_ipc::ipc::ShmemServer`, copies
  `lease.payload()` into bytes to mirror `PyLease.payload`, parses/touches the
  archived request, builds a small `ForwardResponse`, and commits it.

The subprocess side is still not a full Python benchmark: it does not execute
PyO3 wrapper code, NumPy view creation, or `shmem_schema.py`'s full legacy dict
conversion. Use `python/tests/bench_zero_copy.py` for the Python read-side
microbench. The runtime channel bench is intended to answer: "what does the
Rust runtime pay to submit through the real channel shape?"

The `inproc_channel` and `shmem_channel` benches use the `balanced` profile
default (`spin_budget_us = 1000`). The `inproc_polling_channel` bench uses the
`latency` profile default (unbounded spin). Set these constants in
`runtime/benches/driver_channel.rs` if you want a park-only comparison.

Current focused reruns after replacing in-proc Tokio oneshot completion with a
synchronous response slot, Criterion mean:

| Tokens | inproc_channel | shmem_channel |
|---:|---:|---:|
| 16 | 4.46 µs | 3.21 µs |
| 256 | 3.57 µs | 3.00 µs |
| 4K | 3.15 µs | 13.1 µs |
| 16K | 3.74 µs | 18.5 µs |

The in-process path is no longer represented as the raw 100 ns descriptor-view
microbench; the runtime path includes `submit`, the pending map, wakeup,
response-slot wait, and owned response conversion. Before the response-slot
fix, the same in-proc benchmark sat around 8-12 µs because `vt_send_response`
completed a `tokio::sync::oneshot` from the driver thread. The shmem path grows
with payload size because it still pays request encode/copy plus response
materialization.

### In-proc latency breakdown

Run:

```
cargo bench -p pie --bench driver_channel -- inproc_
```

Current run after the response-slot fix, Criterion mean:

| Probe | Time |
|---|---:|
| `status_ack_only/tokens=16` | 3.06 µs |
| `forward_ack_only/tokens=16` | 3.31 µs |
| `forward_with_legacy_demux/tokens=16` | 3.14 µs |
| `tokio_block_on_ready` | 49.6 ns |
| `tokio_oneshot_ready_same_thread` | 85.1 ns |
| `tokio_oneshot_cross_thread` | 9.74 µs |
| `box_frame_from_request/tokens=16` | 244 ns |
| `mutex_vecdeque_push_pop_notify` | 540 ns |
| `response_desc_to_owned_forward` | 59.0 ns |
| `pie_frame_view/tokens=16` | 140 ns |
| `legacy_demux_only/tokens=16` | 32.7 ns |
| `pie_frame_view_plus_legacy_demux/tokens=16` | 193 ns |

The original culprit was the cross-thread rendezvous plus waking the Tokio
awaiter, not descriptor materialization. The `tokio_oneshot_cross_thread` probe
is still ~10 µs, while the fixed in-proc channel's smallest status-only path is
~3 µs. The local FFI-side work remains sub-microsecond: frame view + legacy
demux is ~0.2 µs, and response-desc conversion is ~0.06 µs.

`InProcChannel::submit` now waits on a per-request response slot under
`tokio::task::block_in_place` on multithread Tokio runtimes, with a
`spawn_blocking` fallback for current-thread tests. The remaining ~3-4 µs floor
is the mutex/queue handoff, response-slot spin/park loop, and thread scheduling
noise.

## Python parse + slice access — zero-copy via `numpy.frombuffer`

Measured on the PyO3 wrappers used by `driver/{dev,sglang,vllm}` on the shmem hot path. Workload: parse a 16K-token forward-request frame (131 KB), then `np.asarray(fr.<field>)` over the 8 main vector fields (`token_ids`, `position_ids`, `kv_page_indices`, `kv_page_indptr`, `kv_last_page_lens`, `qo_indptr`, `spec_token_ids`, `spec_position_ids`). Run:

```
.venv/bin/python python/tests/bench_zero_copy.py /tmp/frame_16k.bin
```

| Stage | Before (copy-on-access) | **After (zero-copy view)** |
|---|---:|---:|
| `Frame.parse(bytes)` | ~7 µs (full 131 KB memcpy + Arc alloc) | **0.22 µs** (refcount bump on `Py<PyBytes>`) |
| Parse + 8× `np.asarray(slice)` | ~100 µs (8 × 64 KB memcpy + NumPy alloc) | **8.4 µs** (numpy.frombuffer over the parent bytes) |

**~12× speedup on the Python path at 16K tokens.** The 8.4 µs floor is the cost of 8 `np.frombuffer` calls (each ~1 µs of NumPy object machinery), not memcpy. To go below this you'd need to amortize the NumPy calls (return a single struct-of-arrays view) — diminishing returns.

How: the PyO3 wrappers now store `Py<PyBytes>` instead of `Arc<Vec<u8>>`; `Frame.parse(bytes)` keeps a refcount on the input rather than copying its contents. Slice getters compute `offset = field_ptr - bytes_start` and call `numpy.frombuffer(self.bytes, dtype=<elem>, count=n, offset=offset)`, which NumPy returns as a 1-D view that aliases the same memory. `tok.base is not None` confirms it's a view; the underlying archive stays alive as long as any view does.

Side win: the C-ABI `pie_parse_<type>` now uses `rkyv::access_unchecked` (the same-process producer is trusted; cross-process bytes still go through the checked `wire::parse_request`). Saves ~100 ns per parse and removes the bytecheck dependency from the hot path.

## What's still not benchmarked

- **Actual C++ instruction cost** — `driver_channel` mirrors `build_request_view`
  in Rust scratch so the runtime channel shape is realistic, but it does not
  compile or time the C++ implementation. Add a C++ microbench if that exact
  shim shows up in profiles.
- **Full Python worker path** — `driver_channel` models the Rust runtime side
  plus a Rust shmem server. It does not time PyO3 wrapper calls, NumPy object
  creation, or `shmem_schema.py`'s full dict conversion. Use
  `python/tests/bench_zero_copy.py` and a Python worker microbench for that.
- **Sampler-heavy batches** — the default benchmark workload uses one sampler.
  Add a second workload with many sampler variants if sampler demux is a
  suspected hotspot.
- **Shmem ring scaling under contention** — multiple concurrent clients hitting the same server.

## Optimization candidates — what's left

With the hybrid spin path landed on both FFI and IPC, the old synthetic
per-call floor was **~1-2 µs for small payloads**. Treat that as primitive
transport context only; use `cargo bench -p pie --bench driver_channel` for
the production-shaped channel cost. Further wins are mostly payload-driven
(encode/copy/response materialization) once the wait path stays in the spin
window:

1. **Tier 2 encode** — custom `Serialize` over borrowed slices. Would drop the 6 µs encode at 16K tokens to ~3 µs. ~150 LOC of macro work. Now genuinely meaningful (it's ~60% of the IPC roundtrip at large payloads), but only worth doing if profiling on a real workload shows encode in the top-N hotspots.
2. **Drop the FFI `Mutex` for a lock-free SPSC ring** — would put FFI on par with IPC (~1 µs floor). ~200 LOC. Marginal because nothing in pie's workload is currently bottlenecked on the FFI path.
3. **Coalesce calls per driver fire** — both paths' per-call cost amortizes by batching multiple operations into one Frame. The runtime already does this for forward passes; cold-path adapter/copy ops still pay the full ~1 µs each.
4. **Pre-allocated rkyv arena** — saves ~100 ns per encode. < 5% even at small payloads now that the spin floor is so low.

Items (1) and (2) are real options gated on a profiling result; (3) is a runtime design knob, not a bridge change.

## Goal 2 status

Baseline established and pushed past the original cliffs:

- Reads are constant-time (~56 ns) and zero-copy through to NumPy on the Python side.
- Writes are linear-in-payload (~125 ns/KB encoded). At the new spin-mode floor, encode is the dominant term for large IPC payloads (~60% at 16K tokens) — would be the natural next target if profiling demands it.
- Direct-FFI views are constant-time (~147 ns for `pie_frame_view` in the current run). The old synthetic FFI roundtrip numbers have been replaced by the runtime `driver_channel` benchmark because they did not use the real `InProcChannel` path.
- IPC primitive roundtrip remains useful as a transport-only bench (`pie-ipc --bench shmem`), but production-shaped runtime IPC should be measured with `cargo bench -p pie --bench driver_channel`.
- Python parse + slice access is **~12× faster** after the zero-copy switch (100 µs → 8.4 µs at 16K tokens); see [Python parse + slice access](#python-parse--slice-access--zero-copy-via-numpyfrombuffer) above.
