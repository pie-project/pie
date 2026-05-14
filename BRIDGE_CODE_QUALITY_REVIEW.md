# Code Quality Re-Review: `driver/bridge` and `driver/bridge-macros`

Date: 2026-05-14

Scope reviewed:

- `driver/bridge-macros/src/lib.rs`
- `driver/bridge/src/*.rs`
- `driver/bridge/src/ipc/*.rs`
- `driver/bridge/include/pie_bridge.h`
- `driver/bridge/python/pie_bridge/__init__.py`
- `driver/bridge/tests`, `driver/bridge/benches`, and `driver/bridge/examples`

This is a re-review after the latest workspace updates. The code is moving in a good direction: the Python read path now caches NumPy objects, direct C builders use caller-provided output buffers, and `Brle` has been pulled into the Rust schema. The serialization/parsing path is intentionally internal-only, so unchecked rkyv access is treated here as a performance contract rather than an untrusted-input security boundary. The remaining issues are concentrated around incomplete API call-site updates, shared-memory slot correctness, pointer ownership, and making the trusted-parse contract explicit.

## Verification Run

- PASS: `cargo check -p pie-bridge --features cabi,ipc`
- PASS: `cargo check -p pie-bridge --features python,cabi,ipc`
- PASS: `cargo check -p pie-bridge-macros`
- PASS: `cargo test -p pie-bridge --features cabi,ipc --lib` (`42` tests)
- PASS: `cargo test -p pie-bridge-macros` (`0` tests)
- PASS: `cargo test -p pie-bridge --features cabi,ipc`
- PASS: `cargo check -p pie-bridge --all-targets --features cabi,ipc`
- FAIL: `cargo clippy -p pie-bridge --all-targets --features cabi,ipc -- -D warnings`
  - Stops in `driver/bridge-macros/src/lib.rs` at `216` and `1254`.
- FAIL: `cargo fmt --check -p pie-bridge -p pie-bridge-macros`
  - Many files need rustfmt, including new/updated bridge files.

## Highest Priority Findings

### Resolved: tests and benches now compile after the IPC API change

Locations:

- `driver/bridge/src/ipc.rs:309-316`
- `driver/bridge/src/ipc.rs:589-593`
- `driver/bridge/tests/ipc_shmem_inproc.rs:39`
- `driver/bridge/tests/ipc_shmem_inproc.rs:41`
- `driver/bridge/tests/ipc_shmem_inproc.rs:49`
- `driver/bridge/tests/ipc_shmem_inproc.rs:70`
- `driver/bridge/tests/ipc_shmem_inproc.rs:87`
- `driver/bridge/tests/ipc_shmem_inproc.rs:105`
- `driver/bridge/tests/ipc_shmem_inproc.rs:119`
- `driver/bridge/tests/ipc_shmem_inproc.rs:141`
- `driver/bridge/tests/ipc_shmem_inproc.rs:171`
- `driver/bridge/tests/ipc_shmem_inproc.rs:214`
- `driver/bridge/benches/shmem.rs:36`
- `driver/bridge/benches/shmem.rs:60`
- `driver/bridge/benches/shmem.rs:84`

`ShmemServer::create` now takes `spin_budget_us`, and `ShmemClient::open` now takes `spin_budget_us`. The stale test and `shmem` benchmark call sites were updated to pass a shared `SPIN_BUDGET_US` constant. The full bridge test target and all-target check now compile and pass this issue.

### 1. Unchecked C/Python parsing is an internal contract, not a validation boundary

Locations:

- `driver/bridge-macros/src/lib.rs:407-435`
- `driver/bridge-macros/src/lib.rs:1546-1556`
- `driver/bridge/include/pie_bridge.h:16-18`
- `driver/bridge/include/pie_bridge.h:97-98`

The generated `pie_parse_<type>` functions call `rkyv::access_unchecked` and only reject null or zero-length input. Given the stated invariant that all serialized bytes are produced and consumed internally, removing verification is a reasonable hot-path optimization and should not be treated as an external-input security risk.

The remaining quality issue is contract clarity: the C header presents these as ordinary parse functions with error handling via null, and Python raises `"invalid rkyv buffer"` only for null/empty. That wording implies validation that the code deliberately does not perform. There is also a separate correctness precondition: unchecked rkyv access still requires the byte buffer to satisfy rkyv's validity and alignment requirements.

Suggested improvements:

- Rename or document the C/Python entry points as trusted/internal parse APIs.
- Keep checked helpers in `wire.rs` for debug tests, diagnostics, and any future less-trusted boundary.
- Add debug-mode tests or assertions that the internal buffers passed to unchecked parse satisfy the expected archive alignment.

### 2. Shared-memory slot geometry can misalign atomics and overflow

Locations:

- `driver/bridge/src/ipc.rs:318-330`
- `driver/bridge/src/ipc.rs:638-644`

`slot_stride` is computed as `SLOT_HEADER_SIZE + req_buf + resp_buf`. If `req_buf + resp_buf` is not a multiple of 8, later slots can start at an address that misaligns the `AtomicU64` fields at the slot head. The same arithmetic also lacks checked addition/multiplication and casts values to `u32` for the header without validating bounds.

Suggested improvements:

- Round `slot_stride` up to at least `align_of::<AtomicU64>()`, or 64 bytes.
- Use checked arithmetic for `slot_stride` and `total_size`.
- Reject `num_slots == 0` server-side and reject geometry values that do not fit in the header.
- Add an odd-buffer-size regression test with at least two slots.

### 3. IPC trusts shared-memory lengths and can publish stale responses after timeout/reuse

Locations:

- `driver/bridge/src/ipc.rs:249-257`
- `driver/bridge/src/ipc.rs:262-268`
- `driver/bridge/src/ipc.rs:491-497`
- `driver/bridge/src/ipc.rs:779-786`
- `driver/bridge/src/ipc.rs:842-846`

The server reads `req_payload_len` from shared memory and later creates a slice of that length without bounding it by `req_buf_size`. The client likewise reads `resp_payload_len` and slices before checking it against `resp_buf_size`.

The timeout/reuse bug also remains: `Lease` does not capture the sequence it polled. `commit` reloads the current `req_seq` at publish time, so if a client times out and reuses the slot, a late server response for the old request can be published under the new request sequence.

Suggested improvements:

- Validate request and response lengths before slicing.
- Store the observed `req_seq` in `PolledSlot`/`Lease`.
- Publish the captured sequence, not the current slot sequence.
- If the slot sequence changed before commit, discard or abort the stale lease instead of waking the newer request.

### 4. Python ctypes builder pointers can dangle

Locations:

- `driver/bridge/python/pie_bridge/__init__.py:300-324`
- `driver/bridge/python/pie_bridge/__init__.py:385-408`

For Python lists, `_u32_ptr`, `_f32_ptr`, and `_u8_ptr` allocate temporary ctypes arrays and return only the raw pointer. The array object can be freed immediately after the helper returns, leaving the descriptor with a dangling pointer before `_pie_build_response_frame` reads it.

NumPy arrays are also accepted without dtype or contiguity checks. Passing a wrong dtype or a strided view will serialize incorrect bytes.

Suggested improvements:

- Return `(ptr, owner)` from pointer helpers and keep all owners alive until `_try_build` returns.
- Require exact dtype and C-contiguity for NumPy inputs, or coerce to a contiguous array with explicit ownership.
- Add tests that build from Python lists and from non-contiguous/wrong-dtype NumPy arrays.

## Medium Priority Findings

### Schema hash omits `src/brle.rs`

Locations:

- `driver/bridge/build.rs:9-15`
- `driver/bridge/src/schema.rs:34-37`
- `driver/bridge/src/brle.rs:28-35`

`Brle` is now a `#[schema]` type and is re-exported through `schema.rs`, but `SCHEMA_HASH` still hashes only `src/schema.rs`. Changing `Brle` fields or enum shape can leave old clients thinking they are compatible.

Suggested improvement: hash every schema-affecting file, or generate/hash an ABI manifest from macro output.

### Invalid descriptor enum kinds are not handled consistently

Locations:

- `driver/bridge-macros/src/lib.rs:927-934`
- `driver/bridge-macros/src/lib.rs:1150-1154`

Unit enum descriptors default unknown discriminants to the first variant. Data enum descriptors either default to the first unit variant or panic for non-unit first variants. For a C ABI builder, invalid descriptors should be a clean build failure, not silent coercion or panic.

Suggested improvement: generate checked `from_desc` paths that return `Result`, then make `pie_build_*` return `0` or a structured error on invalid discriminants.

### `AdapterBinding::Default` contradicts the sentinel docs

Locations:

- `driver/bridge/src/schema.rs:149-158`

The docs say `-1` means "unbound" and "no caller-provided seed", but `#[derive(Default)]` creates `{ adapter_id: 0, seed: 0 }`.

Suggested improvement: implement `Default` manually as `{ -1, -1 }`, or remove `Default` if callers must always be explicit.

### Python `poll_blocking` holds the GIL

Locations:

- `driver/bridge/src/python.rs:219-224`

The blocking wait calls into `ShmemServer::poll_blocking` while still holding the GIL. A Python worker waiting for IPC can block unrelated Python execution in the same process.

Suggested improvement: accept `py: Python<'_>` in the PyO3 method and wrap the blocking call with `py.allow_threads`.

### NumPy slice views need bounds checks

Locations:

- `driver/bridge/src/python.rs:121-135`
- `driver/bridge-macros/src/lib.rs:1751-1763`

`slice_to_numpy` uses `wrapping_sub` to compute the offset into the parent `PyBytes`. That relies on every generated pointer being correct. With the current unchecked parser, malformed input can produce pointers outside the parent buffer.

Suggested improvement: use checked pointer range validation and return `PyValueError` if the slice is outside the parent bytes or if `count * element_size` overflows the buffer.

### Macro/header docs still drift from implementation

Locations:

- `driver/bridge-macros/src/lib.rs:17-28`
- `driver/bridge-macros/src/lib.rs:45`
- `driver/bridge-macros/src/lib.rs:182-189`
- `driver/bridge/include/pie_bridge.h:203-209`
- `driver/bridge/include/pie_bridge.h:217-225`

The macro docs describe container args, but `schema(_args, input)` ignores the args. The docs/header still describe `Option<T>` support, but `classify` rejects `Option<T>`. `PieBrleDesc` exists in the header, but the header does not declare `pie_build_brle` or `pie_size_brle`.

Suggested improvement: either implement the documented API or narrow the docs/header to the supported surface, then add a header-vs-generated-ABI check.

### Formatting and lint cleanup is still needed

Locations:

- `driver/bridge-macros/src/lib.rs:216-220`
- `driver/bridge-macros/src/lib.rs:1254-1258`

`cargo fmt --check` fails across many touched files. Clippy with `-D warnings` fails on a collapsible `if` and identical branches in the macro crate.

Suggested improvement: run rustfmt after the compile regression is fixed, then clean up or justify Clippy warnings with targeted `#[allow]` attributes.

## Performance Notes

- Positive change: `driver/bridge/src/python.rs:39-85` now caches NumPy callables and dtype objects. That removes repeated `py.import("numpy")` and dtype lookups from generated vector getters.
- `wire::encode_request` and `encode_response` serialize to rkyv bytes and then copy into `Vec<u8>` at `driver/bridge/src/wire.rs:27-37`. IPC then copies the request again into shared memory at `driver/bridge/src/ipc.rs:745-748`. For large forward requests, direct serialization into the selected shmem slot would remove one allocation and one copy.
- The generated C builder path is better: it writes into caller-provided memory with `to_bytes_in` at `driver/bridge-macros/src/lib.rs:620-635`. Python `_try_build` still retries by allocating larger buffers and rerunning serialization at `driver/bridge/python/pie_bridge/__init__.py:327-336`; wiring `pie_size_response_frame` first would avoid repeated failed serializations for large responses.
- Direct-FFI views for nested slices allocate holder vectors at `driver/bridge-macros/src/lib.rs:728-747`. That is fine for small sampler/mask counts, but it should be measured on real large batches.
- Server polling scans slots linearly at `driver/bridge/src/ipc.rs:382-390`. With small slot counts this is reasonable; if slot counts grow, a ready queue or per-slot notification ownership would reduce scan work.
- `poll_blocking` spin loops intentionally sample time every 256 iterations at `driver/bridge/src/ipc.rs:409-430`. That is reasonable for a hot wait path, but the default spin budget should be benchmarked under expected Python worker concurrency because it trades latency for CPU.

## Suggested Remediation Order

1. Run rustfmt and resolve the two macro Clippy errors.
2. Make the trusted unchecked-parse contract explicit in C/Python docs and add alignment-focused tests or debug assertions.
3. Fix IPC slot alignment, arithmetic validation, request/response length validation, and timeout/reuse sequencing.
4. Fix Python ctypes pointer ownership and NumPy validation.
5. Expand schema hashing to include `Brle` and any future schema-affecting files.
6. Add regression tests for malformed parse input, odd IPC buffer sizes, corrupted lengths, stale lease commits, invalid enum discriminants, and Python builder ownership.
