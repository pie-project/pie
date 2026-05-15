# `pie-bridge` redesign — execution plan (FlatBuffers)

> **Status:** ready to execute.
> **Working location:** `driver/bridge_new/` (renamed to `driver/bridge/` at cutover).
> **Working crate name:** `pie-bridge-new` (renamed to `pie-bridge` at cutover).
> **Predecessor:** `driver/bridge/` — read `BRIDGE.md` for the prior architecture; it will be deleted at M7.
> **Supersedes:** an earlier TOML-DSL-with-custom-codec plan that occupied this file. That plan reinvented ~80% of FlatBuffers; this revision adopts FlatBuffers directly.
> **Owner:** in.gim@yale.edu.

This document is a self-contained briefing for an agent that will implement the new bridge. Read top-to-bottom. Each milestone (M0–M7) is independent enough to be a separate PR; ordering and dependencies are explicit.

---

## 1. Mission

Replace the current `pie-bridge` crate with a redesign whose **single source of truth** is a small set of FlatBuffers schemas under `driver/bridge_new/schema/`. The FlatBuffers compiler (`flatc`) emits Rust, C++, and Python types in parallel — so adding a wire field, a method, or a sampler variant is one schema edit, not a coordinated multi-file change.

**Concretely, this redesign delivers:**

1. `schema/bridge.fbs` declares the entire wire format: methods as a `union RequestPayload`, samplers as a `union Sampler`, opaque blobs (logits, BRLE masks) as `[ubyte]` / `[uint32]`.
2. `flatc` codegen for Rust (request/response views, builders), C++ (accessor headers for cuda/portable backends), Python (table classes + numpy views).
3. Pure-FlatBuffers shmem payload — no parallel slot header, no hand-written ABI hash. Routing fields live on the `Frame` table; the shmem ring carries only state-machine bytes.
4. Lease-based shmem transport API (`Lease` value with RAII commit).
5. PyO3 module exposing the IPC surface (`ShmemServer`, `Lease`, `InProc*`); flatc-generated Python sits alongside as `pie_bridge.fbs.*`. Consumers compose: PyO3 hands out `bytes`, flatc-Python parses.
6. `Sampler` becomes the FlatBuffers union; `Brle` stays as a tiny Rust helper that produces `[uint32]` BRLE blobs.
7. `ResponseBuilder` (the per-request accumulator + indptr-builder) is **not** shipped — each consumer builds responses with its language's native `FlatBufferBuilder`. The previously-vendored C++ copies are deleted.

**Drift this eliminates** (extends the table in `BRIDGE.md` §1):

| Concept | Today | After redesign |
|---|---|---|
| 28-field wire layout | Constants + named struct + slot_mut match + encoder walk + decoder walk + C++ static_assert mirror + Python parse dict | One `table ForwardRequest { ... }` in `bridge.fbs` |
| `Method` enum, `CopyDir`, `AdapterOp` | Hand-written Rust enums + hand-written Python wrappers + C++ static asserts | `union RequestPayload`, `enum CopyDir`, `enum AdapterOp` in the schema |
| Sampler type-id tags + Sampler enum | `methods::sampler_type::*` u32 constants in bridge, hand-written `Sampler` enum in pie | One `union Sampler { ... }` in the schema; `type_id` is the union discriminant |
| ABI hash on every frame | Hand-rolled xxh3-32 in slot header | FlatBuffers `file_identifier "PIE4"` (4 ASCII bytes built into every buffer) |
| ResponseBuilder (Rust) + vendored C++ copies in `cuda/src/_bridge/` + `portable/src/_bridge/` | Three independent implementations | None — consumers use `flatbuffers::FlatBufferBuilder` / `flatbuffers::Builder` directly |
| BRLE helpers | `pie::inference::brle` + C++ `cuda/src/brle.cpp` | bridge (Rust); cuda links against bridge |
| Shmem transport API | `poll_slot` + `request_payload_view` + `respond_view` + `commit_respond` + `stop` + `close` | `poll()` returns `Lease`; `Lease::commit(buf)` or `Lease::abort()`; drop = abort |
| Python bulk types | `PyList<int>`, `PyList<float>`, dict-of-list | flatc-generated Python types with `*AsNumpy()` accessors |
| Sentinel-encoded `Vec<Option<u64>>` (D10 from prior plan) | u64::MAX / i64::MIN sentinels in u64/i64 vectors | Same convention for adapter_ids/adapter_seeds (FlatBuffers vectors can't be element-optional); standalone scalars use FlatBuffers `= null` |

---

## 2. Ground rules

1. **No backward compatibility.** The old bridge will be deleted at M7. Consumers (`pie`, `pie-server`, `pie-driver-dummy`, the C++ backends, the Python downstream drivers) **will be modified at M7** to consume the new API. Do not invent compat shims.
2. **The wire bytes can change.** No requirement to match the old byte layout — FlatBuffers picks the layout.
3. **Coexistence during dev.** Until M7, both `pie-bridge` and `pie-bridge-new` are workspace members. Nothing imports `pie-bridge-new` except its own tests and `pie-driver-dummy-new` (a smoke-test driver that appears in M2).
4. **Stop at the milestone boundaries.** Each milestone ends in a verifiable state — green tests + green builds. Do not partially complete M3 work as part of M2.
5. **Never edit `driver/bridge/` or `BRIDGE.md`** during M0–M6. M7 is the only milestone that touches them.
6. **Verification first.** Each milestone has a verification command. The work isn't done until that command produces the documented output.
7. **No new runtime dependencies without justification.** Approved runtime deps: `flatbuffers` (the Rust runtime), `anyhow`, `bytemuck`, `thiserror`, `libc` (optional, `ipc` feature), `windows-sys` (optional, `ipc` feature), `pyo3` (optional, `python` feature). Approved build-deps: `flatc-rust` (a thin wrapper that spawns the `flatc` binary). The `flatc` binary itself must be on PATH; see §7.4.

---

## 3. Decisions already made — do not re-litigate

| # | Decision | Rationale |
|---|---|---|
| D1 | Schema format: **FlatBuffers `.fbs`** (not TOML, not a custom IDL, not protobuf). | Mature codegen in Rust + C++ + Python; native zero-copy reads of `[uint32]` / `[float]` / `[ubyte]`; unions express method dispatch and sampler variants natively. |
| D2 | Shmem payload is a **size-prefixed FlatBuffers `Frame` buffer**. No separate slot-header struct. The shmem ring still has slot state bytes (Idle/Producing/Ready/...); those are transport metadata, not message content. | One source of truth. Routing fields (`driver_id`, method tag, `aborted`) live on the `Frame` / `ResponseFrame` tables. |
| D3 | **`Sampler` is a FlatBuffers union** declared in `bridge.fbs`. Bridge ships **no hand-written Rust mirror** (no `OwnedSampler` enum). Callers (pie's sampling-kernel dispatch) keep their own typed enum where they actually use it, and at the wire boundary they call the flatc-generated `fbs::SMultinomial::create` etc. directly. Bumped from the prior plan: §16.2 A1 + §16.3 R2(b) anticipated this, and live code reinforced it — see the "no schema mirrors" memory. | Schema is the single source of truth; adding a sampler variant = one `.fbs` edit. No mirror enum to drift. |
| D4 | **`Brle` stays in pie.** The redesign was originally going to migrate it; reversed because bridge is a pure FlatBuffers + IPC layer (see [[no-schema-mirrors]] memory). BRLE is an opaque application encoding for a `[uint32]` field; how consumers fill those bytes is their concern. Pie keeps its `Brle`; cuda keeps `brle.cpp`. | Bridge ships nothing that isn't expressible as `.fbs` schema or IPC transport. BRLE is an encoder for opaque bytes, not a wire-format primitive. |
| D5 | `Method` does not exist as a separate enum. The active method is `frame.payload_type()` (the FlatBuffers union discriminant). Sub-ops (`CopyDir`, `AdapterOp`) are FlatBuffers enums and live as fields on the respective request tables, not as a separate header. | Method-dispatch becomes the same primitive FlatBuffers uses internally; no parallel enum to maintain. |
| D6 | FlatBuffers `file_identifier = "PIE4"` is the ABI tag. Bumping to "PIE5" is the protocol-break signal. Within `PIE4`, schema evolution rules apply (append-only fields; never reorder; never repurpose). | Built-in, validated by the FlatBuffers verifier on every read. |
| D7 | `ResponseBuilder` is **not shipped by bridge**. Each consumer (cuda C++, pie Rust, dev Python) builds responses via the language-native `FlatBufferBuilder`. The vendored C++ copies under `cuda/src/_bridge/` and `portable/src/_bridge/` are deleted at M7. | Replaces the prior plan's Rust-with-C-ABI builder. Native FlatBuffers builders are already idiomatic in each language; cross-language consistency is provided by `flatc`, not by linking the same C library. |
| D8 | Shmem transport: **lease-based, owned API** (`Lease` is `Arc<ShmemServerInner>` + slot id + `committed: AtomicBool`, `Send`). | RAII commit/abort; PyO3-safe. |
| D9 | PyO3 surface exposes **only the IPC layer**: `ShmemServer`, `ShmemClient`, `Lease.payload() -> bytes`, `Lease.commit(bytes)`, `Lease.commit_status(int)`, `InProcVTable` plumbing. Flatc-generated Python sits in a sibling subpackage `pie_bridge.fbs`. The `pie_bridge/__init__.py` re-exports both so consumers `import pie_bridge` and see one namespace. | "PyO3 for IPC, flatc for schema" composes cleanly via `bytes`. No hand-written Python accessors. |
| D10 | **No sentinel conventions in bridge.** Per-element optional in a vector is expressed natively via a wrapper table: `table AdapterBinding { adapter_id: uint64 = null; seed: int64 = null; }`, then `adapter_bindings: [AdapterBinding]`. Consumers see `Option<u64>` / `Option<i64>` directly via flatc accessors — no sentinel widening, no `u64::MAX = none` constant. Reverses an earlier version of this row that had bridge ship sentinel helpers; see [[no-schema-mirrors]] memory. | Schema expresses optionality; bridge ships nothing the schema can't say. |
| D11 | `flag` bits previously in the slot header (`single_token_mode`, `has_user_mask`) become `bool` fields on `ForwardRequest`. `handler_aborted` becomes `bool aborted` on `ResponseFrame`. | Same data, schema-driven, no parallel layout. |
| D12 | `driver_id` is a `uint32` field on `Frame` / `ResponseFrame`. C++ callers read `frame.driver_id()` instead of `view.driver_id`. | Same data, schema-driven. |
| D13 | **No `ForwardPassRequestData` DTO ships from bridge. No `src/data.rs` either.** Consumers build a `ForwardRequest` by calling `fbs::ForwardRequest::create(&mut builder, &ForwardRequestArgs { ... })` directly. There are NO bridge-side wire-convention helpers (sentinels are gone — see D10). `add_request` stays in pie. Bumped twice from the prior version of this row; the second bump dropped the sentinel helpers along with the DTO. | Bridge ships only what's expressible in `.fbs` schema or as IPC transport. See [[no-schema-mirrors]] memory. |
| D14 | `flatc` is invoked from `build.rs` for Rust output (writes to `OUT_DIR`), from CMake for C++ output (writes to `${CMAKE_BINARY_DIR}/gen`), from `build.rs` for Python output (writes to a maturin-staged location). **No generated files committed to git.** | Standard FlatBuffers integration pattern. Build-step ownership matches consumer language. |
| D15 | `flatc` binary version is pinned to **24.3.25** (or newer; verified at build time with `flatc --version` check). Provided either via a release tarball pinned in `tools/flatc/` (Linux x86_64 + arm64 + macOS arm64) or by user installation; build script checks PATH first, falls back to vendored. | flatc is mature; ABI-incompatible changes between minor versions are rare but possible. Pinning avoids surprise. |
| D16 | FlatBuffers root parsing uses the **verifier** by default (`root_with_opts`) on shmem reads (untrusted producer) and the **unsafe trusted variant** (`root_unchecked`) on inproc reads (producer is in the same process and has the same schema linked). | The verifier costs ~1µs on a typical forward batch; that's fine for shmem. Inproc is hot enough to skip it. |
| D17 | Lease abort writes a minimal `ResponseFrame { driver_id, aborted = true, payload = StatusResponse { status: -1 } }` buffer (~64 bytes) into the response slot, then transitions to `Done`. The client checks `frame.aborted()` after standard FlatBuffers verification passes. | Replaces D25 from the prior plan with a far simpler mechanism — same `ResponseFrame` schema, just with a flag set. |
| D18 | `dtypes.json` is **deleted**. Python types come from `flatc --python`; numpy views from `<Table>.<Field>AsNumpy()`. | FlatBuffers Python codegen does what `dtypes.json` was reaching for. |
| D19 | The hand-written `[PieSlice; N]` FFI views (D6/D7 from the prior plan) are **deleted**. C++ callers use the flatc-generated C++ accessor classes: `auto fr = frame->payload_as_ForwardRequest(); auto tokens = fr->token_ids()->Data();` returns `const uint32_t*` zero-copy. | flatc provides what we were going to hand-roll. |

---

## 4. Target directory structure

```
driver/bridge_new/
├── Cargo.toml                     # crate name: pie-bridge-new
├── README.md                      # short pointer to this plan
├── build.rs                       # invokes flatc for Rust + Python, emits cargo:include for C++
├── pyproject.toml                 # maturin entrypoint for wheel
│
├── schema/                        # ── ONE SOURCE OF TRUTH ──
│   ├── bridge.fbs                 # Frame + RequestPayload union + all request tables + Sampler union
│   └── bridge_responses.fbs       # ResponseFrame + ResponsePayload union + all response tables
│
├── src/
│   ├── lib.rs                     # pub use only
│   ├── ids.rs                     # AdapterId, ContextId, DriverId (Rust newtypes; not in schema)
│   ├── capabilities.rs            # DriverCapabilities + Transport enum
│   ├── brle.rs                    # Brle (migrated from pie)
│   ├── data.rs                    # ForwardPassRequestData DTO + encode() impl (hand-written, D13)
│   ├── generated.rs               # `include!(concat!(env!("OUT_DIR"), "/bridge_generated.rs"));`
│   │                              # and `include!(concat!(env!("OUT_DIR"), "/bridge_responses_generated.rs"));`
│   ├── wire.rs                    # thin helpers: build_forward_request_frame, parse_frame, write_abort
│   │
│   ├── ffi/
│   │   ├── mod.rs
│   │   ├── vtable.rs              # InProcVTable (passes opaque FlatBuffers bytes)
│   │   └── inproc.rs              # In-process sink/source over FlatBuffers buffers
│   │
│   ├── ipc/
│   │   ├── mod.rs
│   │   └── shmem/
│   │       ├── mod.rs
│   │       ├── ring.rs            # PIE4 magic header, slot state machine, slot table
│   │       ├── server.rs          # ShmemServer + Lease API
│   │       ├── client.rs          # ShmemClient
│   │       └── platform/          # posix.rs, windows.rs
│   │
│   └── python.rs                  # PyO3 bindings (IPC surface only), feature = "python"
│
├── include/                       # (no hand-written headers; the build emits them under OUT_DIR
│                                  #  and CMake re-runs flatc for the C++ consumers)
│
├── python/pie_bridge_new/
│   ├── __init__.py                # re-exports compiled .so + .fbs subpackage
│   └── fbs/                       # written by build.rs at wheel build time (flatc --python output)
│       ├── __init__.py
│       ├── Frame.py
│       ├── ForwardRequest.py
│       └── ... (one .py per table)
│
├── tests/
│   ├── round_trip_forward.rs
│   ├── round_trip_response.rs
│   ├── round_trip_cold_methods.rs
│   ├── round_trip_sampler_union.rs
│   ├── golden_buffer_bytes.rs     # frozen wire bytes regression
│   └── ipc_cross_process.rs       # spawns helper binary
│
└── tests/helpers/
    └── echo_server.rs             # for ipc_cross_process.rs
```

**Files marked generated** are written by `flatc` into `OUT_DIR/` (Rust), into a maturin-staged path (Python), or under `${CMAKE_BINARY_DIR}/gen/` (C++, by the consuming target's CMakeLists). None are committed to git. See §7.

---

## 5. Wire format

### 5.1 Frame structure (every transport)

Every wire payload is a **size-prefixed FlatBuffers buffer** with file identifier `"PIE4"`:

```
offset  size      content
------  --------  -------
0       4         size_prefix: uint32 little-endian, == total buffer length minus 4
4       4         file_identifier: ASCII "PIE4" (FlatBuffers convention; at offset 4 inside the prefixed buffer)
8       N-8       FlatBuffers `Frame` (request) or `ResponseFrame` (response) root table
```

The size prefix lets the shmem ring (and any future stream transport) read the payload length without parsing FlatBuffers. The file identifier is validated by the FlatBuffers verifier.

**Routing fields, all on the root table:**

- `driver_id: uint32` — read by transports and C++ logging.
- `payload_type` (FlatBuffers-generated union discriminant) — selects the active method.
- `aborted: bool = false` (responses only) — set when the server-side `Lease` is dropped without commit.

There is **no parallel slot header**. Everything previously in the 16-byte slot header collapses into the FlatBuffers root.

### 5.2 Shmem slot mechanics

The shmem ring uses a fixed-size mmap with header + N request slots + N response slots. Each slot is a fixed-size byte region; payload is the size-prefixed FlatBuffers buffer from §5.1.

**Slot state machine** (unchanged from current bridge):

```
Idle → Producing → Ready → Consuming → Responding → Done → Idle
```

State bytes live in the ring's metadata region, not inside the payload. The state byte is one `AtomicU8` per slot; transitions use release/acquire semantics.

**Ring header:**

```
offset  size  field
------  ----  -----
0       4     magic           ASCII "PIE4"
4       4     schema_version  bump to break incompatibly
8       4     slot_count      N
12      4     slot_capacity   fixed bytes per slot (e.g., 1 MiB)
16      ...   slot_state[]    N × AtomicU8, padded to cache line
...     ...   request_slots   N × slot_capacity bytes
...     ...   response_slots  N × slot_capacity bytes
```

The `"PIE4"` magic in both ring header AND FlatBuffers file_identifier is intentional — they're checked at different layers (ring init handshake vs per-frame verification).

### 5.3 Abort protocol (D17)

When a server-side `Lease` is dropped without `commit`, its `Drop` impl writes a minimal abort response:

1. Build a FlatBuffers `ResponseFrame { driver_id, aborted: true, payload_type: StatusResponse, payload: StatusResponse { status: -1 } }` into the slot bytes (~64 bytes total).
2. Transition slot state to `Done`.

The client side runs the normal `flatbuffers::root_with_opts::<ResponseFrame>(bytes)` — verification passes because the buffer is well-formed. The caller then checks `frame.aborted()` and returns `WireError::HandlerAborted` if true.

**No precedence rule between verification and abort.** The abort buffer is a valid `ResponseFrame`; the flag is just set.

### 5.4 ABI versioning

- **`PIE4` → `PIE5` bump:** required when changing field types, removing fields, reordering union variants, or any other backward-incompatible change. Old binaries reject the buffer at verifier time (file_identifier mismatch).
- **Within `PIE4`:** append fields only (FlatBuffers schema evolution rules). Adding a new optional field is forward/backward compatible. Adding a new union variant is forward-compatible (old readers see `_NONE`); adding a new method ID requires bumping `PIE5` only if old servers cannot ignore unknown methods (which they can't, so a new method = `PIE5`).
- **Build-time check:** `build.rs` hashes the `.fbs` files and exposes `pub const SCHEMA_HASH: [u8; 8]` as a Rust constant. A handshake in `ShmemServer::create` writes this into the ring header; clients compare on connect. This catches the case "both sides compiled with `PIE4` but against different schema versions" before any traffic flows.

---

## 6. Schemas

### 6.1 `schema/bridge.fbs` (requests)

Full text (the agent should write this literally; field order matters for FlatBuffers vtable layout but field IDs are auto-assigned — order is documentation, not contract):

```fbs
// pie-bridge request schema.
// One source of truth for the request wire format across Rust, C++, and Python.
//
// EVOLUTION RULES (per D6, §5.4):
//   - Append fields only. Never reorder, never repurpose.
//   - Adding a new method = bump file_identifier "PIE4" → "PIE5".
//   - Adding a new sampler variant = safe (verifier accepts unknown union tags as _NONE for old readers).

namespace pie.bridge;

file_identifier "PIE4";
file_extension "pieframe";

// --------- enums ---------

enum CopyDir : ubyte {
    D2H = 0,
    H2D = 1,
    D2D = 2,
    H2H = 3,
}

enum AdapterOp : ubyte {
    Load     = 0,
    Save     = 1,
    ZoInit   = 2,
    ZoUpdate = 3,
}

// --------- sampler union (D3) ---------

table SMultinomial { temperature: float;  seed: uint32 = null; }
table STopK        { temperature: float;  k: uint32; }
table STopP        { temperature: float;  p: float; }
table SMinP        { temperature: float;  p: float; }
table STopKTopP    { temperature: float;  k: uint32; p: float; }
table SDist        { temperature: float;  num_tokens: uint32; }
table SLogprob     { token_id: uint32; }
table SLogprobs    { token_ids: [uint32]; }
table SEmbedding   {}
table SRawLogits   {}
table SEntropy     {}

union Sampler {
    SMultinomial,
    STopK,
    STopP,
    SMinP,
    STopKTopP,
    SDist,
    SLogprob,
    SLogprobs,
    SEmbedding,
    SRawLogits,
    SEntropy,
}

// --------- forward request ---------

table ForwardRequest {
    // SoA token arrays.
    token_ids:           [uint32];
    position_ids:        [uint32];

    // KV cache layout.
    kv_page_indices:     [uint32];
    kv_page_indptr:      [uint32];
    kv_last_page_lens:   [uint32];
    qo_indptr:           [uint32];

    // Attention masks (BRLE-encoded; see Brle helper).
    flattened_masks:     [uint32];
    mask_indptr:         [uint32];

    // Logit masks (BRLE-encoded).
    logit_masks:         [uint32];
    logit_mask_indptr:   [uint32];

    // Sampling.
    sampling_indices:    [uint32];
    sampling_indptr:     [uint32];
    samplers:            [Sampler];   // typed union vector — discriminant per element
    sampler_indptr:      [uint32];

    // Adapter binding (per-request optional, encoded as u64::MAX / i64::MIN sentinel).
    // Per-slot adapter binding. Wrapper table makes optionality native;
    // no sentinel widening (D10).
    adapter_bindings:    [AdapterBinding];

    // Speculative decoding.
    spec_token_ids:      [uint32];
    spec_position_ids:   [uint32];
    spec_indptr:         [uint32];
    output_spec_flags:   [bool];

    // Context routing.
    context_ids:         [uint64];

    // Mode flags (D11).
    single_token_mode:   bool = false;
    has_user_mask:       bool = false;
}

// --------- cold methods ---------

table CopyRequest {
    dir:  CopyDir;
    srcs: [uint32];
    dsts: [uint32];
}

table AdapterRequest {
    op:         AdapterOp;
    adapter_id: uint64;
    path:       string;   // only used when op == Load
}

table HealthRequest {}

// --------- method dispatch (D5) ---------

union RequestPayload {
    ForwardRequest,
    CopyRequest,
    AdapterRequest,
    HealthRequest,
}

table Frame {
    driver_id: uint32;
    payload:   RequestPayload;
}

root_type Frame;
```

### 6.2 `schema/bridge_responses.fbs` (responses)

```fbs
namespace pie.bridge;

file_identifier "PIE4";
file_extension "pieframe";

include "bridge.fbs";   // re-uses no tables, but ties evolution together

table ForwardResponse {
    num_requests:         uint32;

    tokens_indptr:        [uint32];   // length R+1
    tokens:               [uint32];

    dists_req_indptr:     [uint32];
    dists_kv_indptr:      [uint32];
    dists_ids:            [uint32];
    dists_probs:          [float];

    logits_req_indptr:    [uint32];
    logits_byte_indptr:   [uint32];
    logits_bytes:         [ubyte];    // opaque vocab*f32 buffer

    logprobs_req_indptr:  [uint32];
    logprobs_val_indptr:  [uint32];
    logprobs_values:      [float];

    entropies_indptr:     [uint32];
    entropies:            [float];
}

table StatusResponse {
    // 0 = success; negative = error; positive = method-specific.
    // -1 is reserved for abort (D17).
    status: int32;
}

union ResponsePayload {
    ForwardResponse,
    StatusResponse,
}

table ResponseFrame {
    driver_id: uint32;
    aborted:   bool = false;   // set by Lease::Drop without commit (D17)
    payload:   ResponsePayload;
}

root_type ResponseFrame;
```

### 6.3 Conventions

- **Optional adapter binding (D10):** `adapter_bindings: [AdapterBinding]` carries one entry per request slot. Each `AdapterBinding` has `adapter_id: uint64 = null;` and `seed: int64 = null;` — natively optional. Rust accessors return `Option<u64>` / `Option<i64>` directly; C++ uses the matching nullable accessor; Python returns `None` when unset. No sentinel constants anywhere.
- **BRLE blobs:** `flattened_masks` and `logit_masks` are `[uint32]`. The encoding is whatever `pie_bridge::brle` produces. Decoders treat these as opaque to the codec; only the GPU kernels interpret BRLE structure.
- **Opaque logits:** `logits_bytes` is `[ubyte]`. Consumers `bytemuck::cast_slice` it to `&[f32]` (Rust) or `reinterpret_cast<const float*>` (C++) or `np.frombuffer(..., dtype=np.float32)` (Python). The vocab size is communicated via `logits_byte_indptr`.
- **Method response routing:** Forward → `ForwardResponse`; Copy / Adapter / Health → `StatusResponse`. The mapping is documented here and encoded in `wire.rs`'s `expected_response_for(payload_type)` helper.

---

## 7. Build pipeline

### 7.1 Rust (build.rs)

```rust
// driver/bridge_new/build.rs
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let schema_dir = PathBuf::from("schema");

    // Tell cargo to rerun if any schema changes.
    println!("cargo:rerun-if-changed=schema/bridge.fbs");
    println!("cargo:rerun-if-changed=schema/bridge_responses.fbs");

    // Verify flatc is on PATH and version is acceptable (D15).
    check_flatc_version("24.3.25");

    // Emit Rust bindings.
    flatc_rust::run(flatc_rust::Args {
        lang: "rust",
        files: &[&schema_dir.join("bridge.fbs"), &schema_dir.join("bridge_responses.fbs")],
        out_dir: &out_dir,
        ..Default::default()
    }).expect("flatc Rust codegen failed");

    // Emit Python bindings (D18) into the maturin-staged tree.
    // CARGO_MANIFEST_DIR points at the crate root; we write into python/pie_bridge_new/fbs/.
    let py_out = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("python/pie_bridge_new/fbs");
    std::fs::create_dir_all(&py_out).expect("mkdir fbs/");
    flatc_rust::run(flatc_rust::Args {
        lang: "python",
        files: &[&schema_dir.join("bridge.fbs"), &schema_dir.join("bridge_responses.fbs")],
        out_dir: &py_out,
        ..Default::default()
    }).expect("flatc Python codegen failed");

    // Compute schema hash for the handshake check (§5.4).
    let hash = hash_schema_files(&[&schema_dir.join("bridge.fbs"), &schema_dir.join("bridge_responses.fbs")]);
    std::fs::write(out_dir.join("schema_hash.rs"),
        format!("pub const SCHEMA_HASH: [u8; 8] = {hash:?};")).unwrap();

    // Advertise the include path so cuda/portable CMake can find C++ headers
    // (which they regenerate themselves via flatc; see §7.2). We only emit the
    // schema dir, not the Rust OUT_DIR.
    println!("cargo:schema-include={}", schema_dir.canonicalize().unwrap().display());
}
```

The Python output going into `CARGO_MANIFEST_DIR/python/...` is intentional — maturin packages everything under `python/pie_bridge_new/` into the wheel. The `fbs/` subdirectory is `.gitignore`d.

### 7.2 C++ (CMake, in consumer crates)

Each C++ backend (`driver/cuda`, `driver/portable`) gets a small CMake snippet:

```cmake
# driver/cuda/CMakeLists.txt — add near the top
find_program(FLATC flatc REQUIRED)

set(BRIDGE_SCHEMA_DIR "${CMAKE_SOURCE_DIR}/../bridge_new/schema")
set(BRIDGE_GEN_DIR    "${CMAKE_BINARY_DIR}/gen/pie_bridge")
file(MAKE_DIRECTORY ${BRIDGE_GEN_DIR})

add_custom_command(
    OUTPUT  ${BRIDGE_GEN_DIR}/bridge_generated.h
            ${BRIDGE_GEN_DIR}/bridge_responses_generated.h
    COMMAND ${FLATC} --cpp --scoped-enums --gen-object-api
            -o ${BRIDGE_GEN_DIR}
            ${BRIDGE_SCHEMA_DIR}/bridge.fbs
            ${BRIDGE_SCHEMA_DIR}/bridge_responses.fbs
    DEPENDS ${BRIDGE_SCHEMA_DIR}/bridge.fbs
            ${BRIDGE_SCHEMA_DIR}/bridge_responses.fbs
    COMMENT "flatc → C++ headers"
)
add_custom_target(pie_bridge_cpp_gen DEPENDS
    ${BRIDGE_GEN_DIR}/bridge_generated.h
    ${BRIDGE_GEN_DIR}/bridge_responses_generated.h)

target_include_directories(driver_cuda PRIVATE ${BRIDGE_GEN_DIR})
target_include_directories(driver_cuda PRIVATE ${BRIDGE_SCHEMA_DIR}/../include)
add_dependencies(driver_cuda pie_bridge_cpp_gen)
```

`flatbuffers` headers themselves come from a vendored copy at `driver/bridge_new/include/flatbuffers/` (header-only library — drop in the FlatBuffers release tarball's `include/` once at M0).

### 7.3 Python (maturin)

`pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.4"]
build-backend = "maturin"

[project]
name = "pie-bridge-new"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["flatbuffers>=24.3.25"]

[tool.maturin]
features = ["python"]
python-source = "python"
module-name = "pie_bridge_new._native"
```

The `python-source = "python"` plus `module-name = "pie_bridge_new._native"` layout means maturin:
1. Compiles the Rust `cdylib` into `python/pie_bridge_new/_native.so`.
2. Bundles everything in `python/pie_bridge_new/` (including `fbs/` populated by build.rs) into the wheel.

`python/pie_bridge_new/__init__.py`:

```python
"""pie-bridge — IPC bindings + FlatBuffers schema."""

# IPC surface (Rust-backed via PyO3).
from ._native import (
    ShmemServer,
    ShmemClient,
    Lease,
    InProcVTable,
    SCHEMA_HASH,
    PIE_FILE_IDENTIFIER,
)

# Schema accessors (flatc-generated; D18).
from . import fbs  # noqa: F401

__all__ = [
    "ShmemServer", "ShmemClient", "Lease", "InProcVTable",
    "SCHEMA_HASH", "PIE_FILE_IDENTIFIER",
    "fbs",
]
```

Consumer usage:

```python
import pie_bridge
import flatbuffers
import numpy as np

# --- build a request ---
b = flatbuffers.Builder(1024)
tokens_off = b.CreateNumpyVector(np.array([1, 2, 3], dtype=np.uint32))

pie_bridge.fbs.ForwardRequest.Start(b)
pie_bridge.fbs.ForwardRequest.AddTokenIds(b, tokens_off)
fr_off = pie_bridge.fbs.ForwardRequest.End(b)

pie_bridge.fbs.Frame.Start(b)
pie_bridge.fbs.Frame.AddDriverId(b, 0)
pie_bridge.fbs.Frame.AddPayloadType(b, pie_bridge.fbs.RequestPayload.RequestPayload().ForwardRequest)
pie_bridge.fbs.Frame.AddPayload(b, fr_off)
b.FinishSizePrefixed(pie_bridge.fbs.Frame.End(b), b"PIE4")
buf = bytes(b.Output())

# --- send via IPC ---
server = pie_bridge.ShmemServer.create(...)
lease = server.poll_blocking()
frame = pie_bridge.fbs.Frame.Frame.GetRootAs(lease.payload(), 0)
# ... handle, build response, lease.commit(response_buf) ...
```

### 7.4 `flatc` binary management (D15)

Three options, in order of preference:

1. **System PATH:** `apt install flatbuffers-compiler` or download from upstream releases. `build.rs` checks `which flatc` and version-gates.
2. **Vendored:** `tools/flatc/<arch>/flatc` committed for Linux x86_64, Linux aarch64, macOS arm64. ~3 MB each. `build.rs` falls back to this if PATH doesn't have an acceptable version.
3. **`flatc-rust` crate:** spawns flatc; doesn't bundle it. Same dependency on PATH.

Document the install requirement in `README.md`. CI installs flatc as a prerequisite step. Local dev: `nix shell` profile includes it; for non-Nix users, `make tools` downloads to `tools/flatc/`.

---

## 8. `Sampler` migration (M4)

Old: hand-written enum at `pie/runtime/src/inference/request.rs:155` plus `pie_bridge::methods::sampler_type::*` constants.

New: the `Sampler` union in `schema/bridge.fbs`. `flatc -r` generates:

```rust
pub enum Sampler<'a> {
    NONE,
    SMultinomial(SMultinomial<'a>),
    STopK(STopK<'a>),
    STopP(STopP<'a>),
    // ... 8 more variants ...
}
```

Plus `pub enum SamplerType { SMultinomial = 1, STopK = 2, ... }` — the union discriminant, which replaces the old `sampler_type::MULTINOMIAL` constants. `type_id()` becomes `sampler.variant_type()`.

**Migration at M4:**

1. **Bridge** ships the generated `Sampler` union and a tiny owned variant `pie_bridge::sampler::OwnedSampler` (an enum mirror that owns its values, for use as a Rust DTO before encoding). The owned version is hand-written in `src/data.rs` and has an `encode_into(builder)` method.
2. **Pie** is *not* modified yet. Bridge merely makes the type available.
3. At M7, pie's hand-written `Sampler` enum is deleted; pie consumes `pie_bridge::OwnedSampler`. The variants are identical; this is a rename + re-export.

---

## 9. `Brle` migration (NO-OP — reversed by D4)

**Brle stays in pie.** The prior plan migrated it into bridge; under the stricter principle from [[no-schema-mirrors]] memory, BRLE is an application encoder for opaque `[uint32]` bytes — not a wire-format primitive — and bridge ships only what's expressible in `.fbs` schema or as IPC transport. Pie keeps `pie/runtime/src/inference/brle.rs`; cuda keeps `driver/cuda/src/brle.cpp`. No M4 brle move; bridge has no `src/brle.rs`.

---

## 10. `ResponseBuilder` is not shipped (D7)

The prior plan shipped a `ResponseBuilder` Rust type with a C-ABI export, intended to be linked from cuda/portable as a replacement for their vendored C++ copies. **In this redesign, no such type exists.**

Instead:

- **Cuda C++** uses `flatbuffers::FlatBufferBuilder` directly. The accumulator is hand-rolled in `driver/cuda/src/response_accumulator.cpp` (~150 LOC, replaces `cuda/src/_bridge/response_builder.cpp`).
- **Portable C++** similarly uses `flatbuffers::FlatBufferBuilder` in `driver/portable/src/response_accumulator.cpp`.
- **Pie Rust** uses `flatbuffers::FlatBufferBuilder` in `pie/runtime/src/inference/response.rs`.
- **Dev Python** uses `flatbuffers.Builder` in `driver/dev/src/pie_driver_dev/response.py`.

Each one is ~150 LOC. Cross-language consistency is guaranteed by `flatc` (all four use the same schema-derived `Add*` calls); the duplication is shallow boilerplate.

**Why not factor:** the duplication is genuinely shallow (push-to-vec, build indptr, call Add* in order). The previous "one C library" approach paid a coordination cost (vendored copies, C ABI, manual sync) that exceeded the duplication cost.

---

## 11. PyO3 + flatc-Python composition (D9)

The Python module has two layers:

1. **`pie_bridge._native`** (PyO3, Rust-backed, ~400 LOC):
   - `ShmemServer.create(path: str, slot_count: int, slot_capacity: int) -> ShmemServer`
   - `ShmemServer.poll_blocking(timeout_ms: int | None = None) -> Lease | None`
   - `ShmemServer.poll_nonblocking() -> Lease | None`
   - `ShmemClient.connect(path: str) -> ShmemClient`
   - `ShmemClient.send_blocking(payload: bytes) -> bytes`
   - `Lease.payload() -> bytes` (copied — D24 from prior plan stands; no writable shmem memoryview)
   - `Lease.commit(payload: bytes) -> None`
   - `Lease.commit_status(status: int) -> None` (convenience: builds + commits a StatusResponse)
   - `Lease.abort() -> None` (explicit abort; same as dropping without commit)
   - `InProcVTable` for embedding inside the same process
   - Module constants: `SCHEMA_HASH: bytes`, `PIE_FILE_IDENTIFIER: bytes`

2. **`pie_bridge.fbs`** (flatc-generated, ~30 .py files):
   - Auto-generated by `flatc --python` at build time.
   - One file per FlatBuffers table/union/enum.
   - Pure data accessors; no IPC code.
   - Numpy view methods (`*AsNumpy()`) on every scalar vector field.

The split is clean: `bytes` is the protocol. PyO3 hands out `bytes`; flatc-Python parses `bytes`. No type leaks across the boundary.

---

## 12. Milestones

### M0 — Skeleton + flatc wired (1 day)

**Goal:** crate exists, flatc runs from build.rs, empty .fbs schemas compile.

**Tasks:**

1. Create `driver/bridge_new/` directory tree per §4.
2. `Cargo.toml`:
   ```toml
   [package]
   name = "pie-bridge-new"
   version = "0.1.0"
   edition = "2021"

   [lib]
   crate-type = ["cdylib", "rlib"]

   [features]
   default = []
   ipc = ["dep:libc", "dep:windows-sys"]
   python = ["dep:pyo3"]

   [dependencies]
   anyhow = "1"
   bytemuck = "1"
   thiserror = "1"
   flatbuffers = "24.3.25"
   libc = { version = "0.2", optional = true }
   windows-sys = { version = "0.59", optional = true, features = ["Win32_Foundation", "Win32_System_Memory"] }
   pyo3 = { version = "0.22", optional = true, features = ["extension-module", "abi3-py310"] }

   [build-dependencies]
   flatc-rust = "0.2"
   xxhash-rust = { version = "0.8", features = ["xxh3"] }
   ```
3. `schema/bridge.fbs` — minimum viable: only `table Frame { driver_id: uint32; }` and `root_type Frame; file_identifier "PIE4";`.
4. `schema/bridge_responses.fbs` — only `table ResponseFrame { driver_id: uint32; aborted: bool = false; }`.
5. `build.rs` per §7.1 (full version; the Python output path just creates the dir even if no Python files are emitted yet — flatc emits placeholders).
6. `src/lib.rs`:
   ```rust
   //! pie-bridge-new — replacement bridge crate. See BRIDGE_REDESIGN.md.
   pub mod generated;
   include!(concat!(env!("OUT_DIR"), "/schema_hash.rs"));
   pub const PIE_FILE_IDENTIFIER: &[u8; 4] = b"PIE4";
   ```
7. `src/generated.rs`:
   ```rust
   include!(concat!(env!("OUT_DIR"), "/bridge_generated.rs"));
   include!(concat!(env!("OUT_DIR"), "/bridge_responses_generated.rs"));
   ```
8. Vendor `flatbuffers` C++ headers into `driver/bridge_new/include/flatbuffers/` from the 24.3.25 release tarball.
9. Add `driver/bridge_new` to root `Cargo.toml` workspace members.

**Verify:**
```bash
flatc --version | grep -q "24.3"        # prerequisite
cargo build -p pie-bridge-new            # green; generated.rs compiles
cargo build -p pie-bridge                # still green (untouched)
cargo build --workspace                  # still green
ls target/debug/build/pie-bridge-new-*/out/bridge_generated.rs   # exists
```

---

### M1 — Forward request + response schemas, Rust round-trip (1–2 days)

**Goal:** full `bridge.fbs` and `bridge_responses.fbs` per §6.1/§6.2. Rust round-trip tests pass. No transport yet.

**Tasks:**

1. Write the full `schema/bridge.fbs` per §6.1.
2. Write the full `schema/bridge_responses.fbs` per §6.2.
3. `src/data.rs`: hand-write `ForwardPassRequestData` (the Rust input DTO) with all 22 fields, plus `OwnedSampler` enum (mirrors the FlatBuffers union for building).
   ```rust
   pub struct ForwardPassRequestData {
       pub token_ids: Vec<u32>,
       // ... all fields from ForwardRequest table, owned ...
       pub samplers: Vec<OwnedSampler>,
       pub adapter_ids: Vec<Option<u64>>,    // encoded with sentinel by encode_into
       pub adapter_seeds: Vec<Option<i64>>,
       pub output_spec_flags: Vec<bool>,
       pub single_token_mode: bool,
       pub has_user_mask: bool,
   }
   impl ForwardPassRequestData {
       pub fn encode_into(&self, builder: &mut flatbuffers::FlatBufferBuilder, driver_id: u32);
   }
   ```
4. `src/wire.rs`: thin parse helper.
   ```rust
   pub fn parse_frame(buf: &[u8]) -> Result<generated::pie::bridge::Frame, WireError>;
   pub fn parse_response_frame(buf: &[u8]) -> Result<generated::pie::bridge::ResponseFrame, WireError>;

   #[derive(thiserror::Error, Debug)]
   pub enum WireError {
       #[error("flatbuffers verification failed: {0}")]
       Verify(String),
       #[error("file identifier mismatch")]
       FileIdentifier,
       #[error("handler aborted")]
       HandlerAborted,
       // ...
   }
   ```
5. Tests (`tests/round_trip_forward.rs`):
   - `empty_request_round_trip` — build a `ForwardRequest` with all empty vectors, encode + parse, verify all accessors return empty slices.
   - `populated_request_round_trip` — realistic batch, encode + parse, verify every field matches.
   - `adapter_ids_sentinel` — mix of Some/None in `adapter_ids`, verify sentinel encoding then decode via `Option::from_sentinel` helper.
   - `samplers_union_vector` — vector of mixed sampler variants, verify each element's discriminant + payload.
   - `output_spec_flags_bool_vec` — verify `[bool]` round-trips as a `Vec<bool>`.
   - `verification_rejects_corrupt_buffer` — random bytes, parser returns `WireError::Verify`.
   - `wrong_file_identifier_rejected` — buffer with `"PIE3"` instead of `"PIE4"` returns `WireError::FileIdentifier`.

6. Tests (`tests/round_trip_response.rs`):
   - `forward_response_round_trip` — populated response, all fields read back.
   - `status_response_round_trip` — `status: 0` round-trips.
   - `aborted_flag_round_trip` — `aborted: true` propagates and `WireError::HandlerAborted` surfaces.

**Verify:**
```bash
cargo test -p pie-bridge-new --test round_trip_forward
cargo test -p pie-bridge-new --test round_trip_response
# All green.
```

**Risk gate:** open the generated `OUT_DIR/bridge_generated.rs`. Confirm:
- The `Sampler` union exposes a Rust `enum` with `_None` + 11 variants.
- `ForwardRequest::token_ids()` returns `Option<flatbuffers::Vector<u32>>`.
- The verifier code path is present (`run_verifier` calls).

If any are missing, debug flatc invocation before proceeding.

---

### M2 — Cold methods, Sampler/Brle migration, dummy driver (1 day)

**Goal:** all four methods round-trip, Sampler + Brle live in bridge, a tiny dummy driver exercises the full bridge surface in-process.

**Tasks:**

1. The cold-method tables (`CopyRequest`, `AdapterRequest`, `HealthRequest`) and the `RequestPayload` union are already in M1's `bridge.fbs`. Verify they're working.
2. **Sampler migration:** copy `pie/runtime/src/inference/request.rs:155`'s enum verbatim into `src/data.rs` as `OwnedSampler`. Implement `encode_into(builder)` that fans out to the appropriate FlatBuffers `Sampler*` table.
3. **Brle migration:** copy `pie/runtime/src/inference/brle.rs` into `src/brle.rs`. Re-export tests.
4. `tests/round_trip_cold_methods.rs`:
   - `copy_d2h_round_trip` — encode `CopyRequest { dir: D2H, srcs, dsts }`, parse, verify.
   - `adapter_load_round_trip` — encode `AdapterRequest { op: Load, adapter_id, path }`, parse.
   - `adapter_zo_init_no_path` — `op: ZoInit` with empty path, parse OK.
   - `health_request_round_trip` — empty `HealthRequest`, parse OK.
   - `status_response_round_trip` — covered by M1.
   - `status_response_handler_aborted` — `aborted: true` + `status: -1`, parse returns `WireError::HandlerAborted`.
5. `tests/round_trip_sampler_union.rs`:
   - One test per sampler variant.
6. **Dummy driver** (`driver/dummy_new/`): a tiny binary that loads a `bridge_new` `InProcVTable`, sends one `HealthRequest`, receives a `StatusResponse { status: 0 }`. Smoke-tests the codec end-to-end without any transport machinery.

**Verify:**
```bash
cargo test -p pie-bridge-new --test round_trip_cold_methods
cargo test -p pie-bridge-new --test round_trip_sampler_union
cargo test -p pie-bridge-new --lib -- brle::
cargo run -p pie-driver-dummy-new
# All green.
```

---

### M3 — Shmem transport, Lease API, cross-process test (2 days)

**Goal:** server + client over shmem, full RAII Lease, cross-process integration test.

**Tasks:**

1. `src/ipc/shmem/ring.rs` — ring header struct (§5.2), slot state machine, atomic transitions. Hand-written; no FlatBuffers involvement here (it's pure ring-buffer logic).
2. `src/ipc/shmem/server.rs`:
   ```rust
   pub struct ShmemServer { inner: Arc<ShmemServerInner> }
   pub struct Lease {
       inner: Arc<ShmemServerInner>,
       slot_id: u32,
       committed: AtomicBool,
       // Capture method type at poll time for the abort response.
       request_payload_type: PayloadType,
       driver_id: u32,
   }
   impl Lease {
       pub fn payload(&self) -> &[u8];                 // immutable borrow of the slot bytes
       pub fn commit(self, response_buf: &[u8]);       // copies into response slot, transitions to Done
       pub fn commit_status(self, status: i32);        // builds a StatusResponse, commits
       pub fn abort(self);                             // explicit abort
   }
   impl Drop for Lease {
       fn drop(&mut self) {
           if !self.committed.load(Ordering::Acquire) {
               self.write_abort_response();   // §5.3
           }
       }
   }
   ```
3. `src/ipc/shmem/client.rs` — mirror server: open existing ring, post a request, await response.
4. `src/ipc/shmem/platform/{posix,windows}.rs` — mmap + futex shims.
5. **Handshake**: server writes `SCHEMA_HASH` into the ring header at create; clients compare on connect and refuse mismatch.
6. `tests/ipc_cross_process.rs` + `tests/helpers/echo_server.rs`:
   - Helper binary mmaps the ring, polls, builds a tiny `ForwardResponse { num_requests: 0 }`, commits.
   - Test spawns the helper, sends a `Frame { driver_id: 42, payload: HealthRequest }`, receives the response, verifies.
   - Test 2: client sends, helper drops the lease without commit, client receives `aborted: true`.

**Verify:**
```bash
cargo test -p pie-bridge-new --test ipc_cross_process --features ipc
# Both subtests green.
```

---

### M4 — FFI (InProc) path for C++ embedding (1 day)

**Goal:** cuda's C++ can in principle link against the Rust crate and consume FlatBuffers buffers via the InProc vtable. No actual cuda integration here — that's M7.

**Tasks:**

1. `src/ffi/vtable.rs`:
   ```rust
   #[repr(C)]
   pub struct InProcVTable {
       /// Block until a request is available; return its bytes + length.
       /// Returned pointer remains valid until send_response is called for this req_id.
       pub recv: unsafe extern "C" fn(
           ctx: *mut c_void,
           out_payload: *mut *const u8,
           out_payload_len: *mut usize,
           out_req_id: *mut u32,
       ) -> i32,
       /// Send a response for req_id.
       pub send_response: unsafe extern "C" fn(
           ctx: *mut c_void,
           req_id: u32,
           payload: *const u8,
           payload_len: usize,
       ),
       pub ctx: *mut c_void,
   }
   ```

   Note: `recv` returns a `*const u8` to the request payload bytes, not a typed view. The C++ side does `auto frame = pie::bridge::GetSizePrefixedFrame(payload);` and proceeds with the flatc-generated accessors.

2. `src/ffi/inproc.rs` — `FfiRequestSink` / `FfiResponseSource` Rust types that drive the vtable from the Rust side. Same opaque-bytes interface.

3. Tests (`tests/ffi_inproc.rs`):
   - `fake_vtable_round_trip` — Rust test harness that implements `InProcVTable` via Rust closures. Push a request through, pull a response.

4. Hand-written `include/pie_bridge.h` (the only hand-written C header in the project):
   ```c
   #ifndef PIE_BRIDGE_H
   #define PIE_BRIDGE_H

   #include <stdint.h>

   #ifdef __cplusplus
   extern "C" {
   #endif

   typedef struct PieInProcVTable {
       int32_t (*recv)(void* ctx,
                       const uint8_t** out_payload,
                       size_t*         out_payload_len,
                       uint32_t*       out_req_id);
       void    (*send_response)(void* ctx,
                                uint32_t req_id,
                                const uint8_t* payload,
                                size_t         payload_len);
       void* ctx;
   } PieInProcVTable;

   extern const uint8_t  PIE_BRIDGE_SCHEMA_HASH[8];
   extern const uint8_t  PIE_BRIDGE_FILE_IDENTIFIER[4];

   #ifdef __cplusplus
   }
   #endif
   #endif
   ```

   Tiny and stable. All schema-shaped types come from `flatc`-generated headers, not from here.

**Verify:**
```bash
cargo test -p pie-bridge-new --test ffi_inproc
# Green.
```

---

### M5 — PyO3 bindings (1 day)

**Goal:** Python wheel exposes IPC surface; flatc-generated `.py` files ship alongside; `import pie_bridge_new` works end-to-end.

**Tasks:**

1. `src/python.rs` — PyO3 wrappers per §11 (D9):
   - `#[pyclass] ShmemServer` with `create`, `poll_blocking`, `poll_nonblocking`.
   - `#[pyclass] ShmemClient` with `connect`, `send_blocking`.
   - `#[pyclass(unsendable)] Lease` — wait, no, **D8/D16: Lease is Send.** Drop `unsendable`. `payload()` returns `&PyBytes` (copied from shmem). `commit(payload: &PyBytes)` consumes self.
   - Module constants: `SCHEMA_HASH`, `PIE_FILE_IDENTIFIER`.
2. `python/pie_bridge_new/__init__.py` per §7.3.
3. `python/pie_bridge_new/fbs/` — populated automatically by build.rs (§7.1).
4. `pyproject.toml` per §7.3.
5. `python/tests/test_python_round_trip.py`:
   - Build a `Frame { driver_id: 42, payload: HealthRequest }` using `flatbuffers.Builder` + `pie_bridge_new.fbs.*`.
   - Connect to a ShmemServer (in a helper process), send, receive, parse the response.
   - Verify abort path: drop the lease on the server side, client sees `aborted: true`.

**Verify:**
```bash
maturin develop -p pie-bridge-new --features python,ipc
pytest driver/bridge_new/python/tests/ -v
# All green.
```

---

### M6 — Golden tests, frozen-bytes fixtures, integration polish (1 day)

**Goal:** prevent silent schema drift; check in regression fixtures.

**Tasks:**

1. `tests/golden_buffer_bytes.rs`:
   - Build a canonical `ForwardRequest` with hardcoded inputs.
   - Encode → bytes → write `tests/fixtures/forward_request_canonical.bin` (committed to git).
   - On test run: rebuild the buffer; assert byte-equal to the fixture (modulo FlatBuffers builder nondeterminism — use `force_defaults(true)` and a fresh `Builder` to control layout).
2. `tests/fixtures/` — canonical buffers for ForwardRequest, ForwardResponse, AdapterRequest, abort response, each sampler variant.
3. **Integration test pass** (A7 from the prior plan's review): run the existing `tests/inferlets/test_helloworld.py`, `test_text_completion.py`, `test_sampler_suite.py`, `test_empty_forward.py` through `pie-driver-dummy-new` (the M2 dummy driver). They should all pass — confirms the codec is good enough to drive real model workflows once the consumers are migrated.
4. Document schema-evolution rules in `schema/EVOLUTION.md`:
   - Append fields only.
   - Bump `"PIE4"` → `"PIE5"` for incompatible changes.
   - Every change requires updating the golden fixture in the same PR.

**Verify:**
```bash
cargo test -p pie-bridge-new
pytest tests/inferlets/test_{helloworld,text_completion,sampler_suite,empty_forward}.py --driver=dummy_new
# All green.
```

---

### M7 — Cutover (1–2 days)

**Goal:** delete the old bridge. All consumers move to `pie-bridge`. (The new crate is renamed from `pie-bridge-new` to `pie-bridge` as part of this milestone.)

**Tasks (in order):**

1. **Rename:** `driver/bridge_new/` → `driver/bridge/`; crate name `pie-bridge-new` → `pie-bridge`; module name `pie_bridge_new` → `pie_bridge`. The old `driver/bridge/` is deleted in step 9.
2. **Pie runtime:**
   - Delete `pie/runtime/src/inference/request.rs:155` `Sampler` enum; replace with `pub use pie_bridge::OwnedSampler as Sampler;`.
   - Delete `pie/runtime/src/inference/brle.rs`; replace with `pub use pie_bridge::Brle;`.
   - Update all imports of `pie::inference::brle::Brle`, `pie::inference::request::Sampler`.
   - `add_request` ext trait in `pie/runtime/src/inference/builder.rs` (or wherever it lands) operates on `pie_bridge::ForwardPassRequestData` as per D13. The trait body unchanged in spirit; just imports update.
3. **Pie-server:** update `pie/server/src/embedded_driver.rs` to construct the new shmem ring + Lease-based polling. The `DriverChannel` async wrapper is unchanged in shape (D11 from the prior plan stands — async wrapper stays in pie).
4. **Driver flavors:**
   - `pie-driver-dummy`: replace internals with M2's `pie-driver-dummy-new`.
   - `pie-driver-dev`: update to consume the FlatBuffers payload via `pie_bridge.fbs.*`. The torch-free constraint (per the user-memory note) still holds; torch lives in downstream wheels.
5. **Cuda backend** (`driver/cuda`):
   - Update CMakeLists per §7.2 (flatc invocation, include path).
   - Replace `cuda/src/_bridge/*` (vendored response builder + headers) with includes from flatc-generated headers + the hand-written `include/pie_bridge.h`.
   - Update `cuda/src/entry.cpp` to read `frame->driver_id()` instead of `view.driver_id`; replace all `view.token_ids` accesses with `frame->payload_as_ForwardRequest()->token_ids()->data()`.
   - Replace the vendored `response_builder.cpp` with a fresh `response_accumulator.cpp` per §10.
6. **Portable backend** (`driver/portable`): same as cuda, in parallel.
7. **Delete old artifacts:**
   - `driver/bridge/` (the original; rename above replaces it).
   - `cuda/src/_bridge/`, `portable/src/_bridge/`.
   - `pie/runtime/src/inference/request.rs:155..` (the `Sampler` enum portion).
   - `pie/runtime/src/inference/brle.rs`.
8. **Update `BRIDGE.md`** to reflect the new architecture (or replace it with a pointer to `BRIDGE_REDESIGN.md` if we want one source of post-cutover docs).
9. **Final verification suite:** all driver flavors + all inferlet tests + dev driver subprocess test.

**Verify:**
```bash
cargo build --workspace
cargo test --workspace
pytest tests/inferlets/ -v --driver=dummy
pytest tests/inferlets/ -v --driver=dev
pytest tests/inferlets/ -v --driver=cuda            # if a cuda host is available
# All green.
```

**Risk:** this is the only large-blast-radius PR. See §14 R1 for the dual-stack alternative if soak risk is high.

---

## 13. Test plan summary

| Test | Owner | Scope |
|---|---|---|
| `round_trip_forward.rs` | M1 | Forward request encode/decode |
| `round_trip_response.rs` | M1 | Forward + status response encode/decode |
| `round_trip_cold_methods.rs` | M2 | Copy, Adapter, Health round-trip |
| `round_trip_sampler_union.rs` | M2 | All 11 sampler variants |
| `ffi_inproc.rs` | M4 | InProc vtable round-trip |
| `ipc_cross_process.rs` | M3 | Shmem + Lease + abort across processes |
| `golden_buffer_bytes.rs` | M6 | Frozen wire bytes — catches schema drift |
| `python/tests/test_python_round_trip.py` | M5 | PyO3 + flatc-Python end-to-end |
| `tests/inferlets/test_*.py` (existing) | M6 + M7 | Real workloads through dummy/dev drivers |

---

## 14. Risks and judgment notes

**R1: Flag-gated dual-stack alternative to single-PR M7 cutover.** For a system currently verified working at every layer (10+ verification runs across 4 driver flavors, 2 architectures, real models, concurrent load), a big-bang merge is the highest-risk way to land this.

Alternative: route through a `--driver-bridge=v2` config flag for one release cycle. Both `pie-bridge` (old) and `pie-bridge-new` coexist; the entire `tests/inferlets/test_*.py` suite + dev driver subprocess test runs against both back-to-back, results compared, then `pie-bridge` (old) deleted in a follow-up PR after soak.

Cost: a few weeks of dual maintenance + a feature flag's worth of plumbing in `pie-server::embedded_driver`. Benefit: subtle bugs are reversible by flag flip, not by revert PR.

Decide before M0.

**R2: Sentinel-in-vector for `adapter_ids` / `adapter_seeds` is not type-safe.** A future hand-rolled consumer could forget the sentinel check and treat `u64::MAX` as a real adapter id. Mitigations:

- Document the convention in `bridge.fbs` (already done in §6.1).
- Provide a Rust helper `pie_bridge::adapter::iter_optional(ids: &[u64]) -> impl Iterator<Item = Option<u64>>` to make the right access pattern the easy one.
- Optional: in `tests/golden_buffer_bytes.rs`, include a fixture with a sentinel value and assert the helper decodes it as `None`.

**R3: `flatc` is an external binary build dependency.** Mitigations in §7.4 (system PATH preferred, vendored fallback, version-pinned). Reproducibility risk is bounded — `flatc` output for a given `.fbs` is deterministic across versions for stable features. The 24.3.25 pin gives a 1-year stability horizon.

**R4: FlatBuffers verifier overhead on hot inproc reads.** The FlatBuffers verifier costs ~1µs per buffer. For shmem (untrusted) we eat it. For inproc (trusted, same process, same schema-linked), use `flatbuffers::root_unchecked` — bypasses verification. The decision is per-call-site, controlled by `FrameReader::trusted()` vs `FrameReader::verified()`.

**R5: Generated Rust code under `OUT_DIR` is invisible to IDEs by default.** Most modern setups (rust-analyzer) handle `include!` correctly, but stack traces point at files developers don't have open. Mitigations:

- A `cargo xtask dump-generated` xtask command that copies `OUT_DIR/*.rs` into `target/dump/` for manual inspection.
- The `tests/golden_buffer_bytes.rs` fixture-based test makes the wire format self-documenting independent of the codegen source.

---

## 15. Open decisions before M0

These are the calls that haven't been made yet in this doc and should be locked in before any code lands:

1. **R1 dual-stack vs single-PR M7.** Default: single-PR per current §2. Flip if the team prefers slower, safer rollout.
2. **C++ flatbuffers headers: vendored or fetched via CMake?** Default: vendor at `driver/bridge_new/include/flatbuffers/`. Alternative: `find_package(FlatBuffers REQUIRED)` and let the system provide.
3. **`pie-driver-dummy-new` vs replacing `pie-driver-dummy` in-place during M2.** Default: new crate during dev, rename at M7. Alternative: feature-flag the existing dummy.
4. **Does this doc replace `BRIDGE.md` at M7, or stay alongside?** Default: rewrite `BRIDGE.md` to describe the new architecture; this redesign doc moves to `docs/history/BRIDGE_REDESIGN.md` for posterity.

---

## 16. How to use this document

1. Read `BRIDGE.md` first for context on the current bridge.
2. Read this document § 1–11 (mission through PyO3 composition).
3. Read § 12 only when starting the relevant milestone.
4. Skim § 14 once before starting; revisit when judgment calls arise.

For each milestone, the order is: read § 12.M(n), write the code, run the verification command, open a PR with the milestone tag in the title.

---

**End of plan.**
