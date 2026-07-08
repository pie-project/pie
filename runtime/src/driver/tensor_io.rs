//! #27 cut #1 — programmable-sampling output→tensor fast-path (host side).
//!
//! Normally a forward pass's program outputs ride the `ForwardResponse` SoA
//! channels back over IPC, and [`build_output_tensors`](crate::api::inference) marshals
//! them into host `tensor`s. The fast-path skips that round-trip: the host
//! pre-allocates one **pinned** host buffer per declared output value
//! ([`pie_pinned_alloc`]), threads the raw pointer into the request's
//! `sampling_output_*` carrier, and the driver eager-D2H's each sampled output
//! VALUE straight into it (off a copy stream that overlaps the next forward).
//! `output()` then copies the pinned bytes into the returned `tensor` and frees
//! the buffer (the buffer's [`Drop`]). The driver's forward-done response is
//! deferred until the D2H lands (the `(a2)` seam), so the buffers are filled by
//! the time `output().await` reads them.
//!
//! ## Feature gate
//!
//! The whole path is gated on `driver-cuda`. OFF (host-only builds /
//! `cargo test -p pie`): [`populate_output_fastpath`] is a no-op returning an
//! empty `Vec`, the `extern "C"` symbols are never referenced (no undefined-symbol
//! link error), and every pass takes the legacy `ForwardResponse` path. ON
//! (propagated from `pie-worker`'s `driver-cuda`, which links
//! `pie_driver_cuda_lib`): the fast-path is active and the symbols resolve.

use crate::api::pie::core::tensor::Dtype;
use pie_driver_abi::ForwardRequest;
use pie_sampling_ir::OutputKind;

#[cfg(feature = "driver-cuda")]
unsafe extern "C" {
    /// Allocate `n_bytes` of page-locked host memory from the driver's pinned
    /// tensor-I/O arena (`pie_driver_cuda_lib`, `tensor_io.cpp`). Returns a raw
    /// host pointer valid in the (in-proc) driver's address space, or null on
    /// failure. Freed with [`pie_pinned_free`].
    fn pie_pinned_alloc(n_bytes: usize) -> *mut core::ffi::c_void;
    /// Return a [`pie_pinned_alloc`] buffer to the arena's free-list.
    fn pie_pinned_free(host_ptr: *mut core::ffi::c_void);

    // ── #27 cut #2 late-input device-alias channel (input H2D) ──────────
    /// Co-allocate a device buffer (`*out_device_dst`) + its R12 self-arm `u32`
    /// flag (`*out_device_flag`, cleared) from the driver's device tensor-I/O
    /// arena. Freed with [`pie_device_free`].
    fn pie_device_alloc(
        n_bytes: usize,
        out_device_dst: *mut *mut core::ffi::c_void,
        out_device_flag: *mut *mut u32,
    );
    /// Return a [`pie_device_alloc`] buffer + flag to the arena.
    fn pie_device_free(device_dst: *mut core::ffi::c_void, device_flag: *mut u32);
    /// Async H2D copy `host_src[..n_bytes]` → `device_dst`, setting `device_flag`
    /// stream-ordered AFTER the copy (the Model-A self-arm). Returns an opaque
    /// completion event (`cudaEvent_t`) for [`pie_event_sync`].
    fn pie_tensor_write_async(
        device_dst: *mut core::ffi::c_void,
        host_src: *const core::ffi::c_void,
        n_bytes: usize,
        device_flag: *mut u32,
    ) -> *mut core::ffi::c_void;
    /// Synchronize on (and reclaim) a [`pie_tensor_write_async`] event.
    fn pie_event_sync(ev: *mut core::ffi::c_void);
}

/// One device-resident late-input value the host uploads directly (H2D) for a
/// `host{key, late-bound}` program input (the #27 cut #2 direct mask path). Owns
/// its [`pie_device_alloc`] buffer + R12 self-arm flag (freed on [`Drop`]); the
/// raw device pointer + flag ride `sampling_late_device_ptrs`/`_flags` to the
/// driver's `HostLate` resolution.
///
/// `Send` because it's a device handle whose H2D is ordered before the consuming
/// kernel by the R12 flag (driver-side) + the pre-fire `pie_event_sync`.
#[cfg(feature = "driver-cuda")]
pub struct DeviceLateInput {
    device_dst: *mut core::ffi::c_void,
    device_flag: *mut u32,
    byte_len: usize,
}

#[cfg(feature = "driver-cuda")]
unsafe impl Send for DeviceLateInput {}

#[cfg(feature = "driver-cuda")]
impl core::fmt::Debug for DeviceLateInput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceLateInput")
            .field("device_dst", &self.device_dst)
            .field("device_flag", &self.device_flag)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

#[cfg(feature = "driver-cuda")]
impl DeviceLateInput {
    /// Device value pointer (rides `sampling_late_device_ptrs`).
    pub fn device_ptr(&self) -> u64 {
        self.device_dst as u64
    }
    /// R12 self-arm flag pointer (rides `sampling_late_device_flags`).
    pub fn flag_ptr(&self) -> u64 {
        self.device_flag as u64
    }
    /// Device value byte length (rides `sampling_late_device_lens`) — the size of
    /// the uploaded host slice, for the merged per-row device gather (`[N, len]`).
    pub fn byte_len(&self) -> u32 {
        self.byte_len as u32
    }
}

#[cfg(feature = "driver-cuda")]
impl Drop for DeviceLateInput {
    fn drop(&mut self) {
        if !self.device_dst.is_null() {
            // SAFETY: `device_dst`/`device_flag` are a live `pie_device_alloc`
            // pair, freed exactly once here.
            unsafe { pie_device_free(self.device_dst, self.device_flag) };
        }
    }
}

/// Upload `data` directly to a fresh device buffer (H2D from the host slice, no
/// IPC staging) for a late-bound input, ordered-complete before return (the
/// sequential-mask MVP: `pie_event_sync` before the fire). Returns the owning
/// [`DeviceLateInput`] whose `device_ptr`/`flag_ptr` ride the carrier. `None` on
/// alloc failure (caller falls back to the staged path). No-op without
/// `driver-cuda`.
#[cfg(feature = "driver-cuda")]
pub fn upload_late_input(data: &[u8]) -> Option<DeviceLateInput> {
    // Serialize the host-side Late-upload across concurrent inferlet `execute()`
    // threads. `pie_device_alloc`/`pie_tensor_write_async`/`pie_event_sync` all
    // drive the driver's single process-wide `TensorIoEngine` — one copy stream +
    // device-arena + the R12-flag set/clear staging. Two WASM proc threads
    // uploading masks simultaneously race that shared mutable state: this is the
    // item-1 concurrent-Late crash, and it is NOT the run-ahead fire-overlap (it
    // reproduces with `MAX_IN_FLIGHT=1`, fires serialized) — it's the two host
    // upload threads before `submit_async`. One global lock makes each upload
    // (alloc → H2D → self-arm → sync) atomic w.r.t. the others. The sequential-mask
    // MVP already blocks on `pie_event_sync`, and this is a plain sync fn (no
    // `.await` while held), so the lock adds no async-runtime hazard — it only
    // serializes the device-arena/stream/flag access the engine isn't internally
    // atomic over. (The true-async overlap follow-up that drops the pre-sync will
    // move this to per-stream / per-proc arenas; for the MVP, serialize.)
    static UPLOAD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _upload_guard = UPLOAD_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    let mut device_dst: *mut core::ffi::c_void = core::ptr::null_mut();
    let mut device_flag: *mut u32 = core::ptr::null_mut();
    // SAFETY: FFI to the driver's device arena; out-params populated on success.
    unsafe { pie_device_alloc(data.len(), &mut device_dst, &mut device_flag) };
    if device_dst.is_null() {
        return None;
    }
    let handle = DeviceLateInput {
        device_dst,
        device_flag,
        byte_len: data.len(),
    };
    // SAFETY: `device_dst` is a live buffer of `data.len()` bytes; `host_src` is
    // a valid host slice; the event is synced (ordered-complete) before use.
    let ev = unsafe {
        pie_tensor_write_async(
            device_dst,
            data.as_ptr() as *const core::ffi::c_void,
            data.len(),
            device_flag,
        )
    };
    // Sequential-mask MVP: order the 19KB H2D complete before the fire reads it
    // ("already on device this fire"); the R12 flag also gates it driver-side
    // (load-bearing for the true-async overlap follow-up where this sync drops).
    if !ev.is_null() {
        // SAFETY: `ev` is a live event from `pie_tensor_write_async`.
        unsafe { pie_event_sync(ev) };
    }
    Some(handle)
}

/// Upload a working-set `slot_to_block` dictionary
/// ([`crate::working_set::kv::KvWorkingSet::slot_to_block_table`]) to a device
/// buffer for the C1-FINAL device-geometry resolve (beam / §6.1 Design B). Rides
/// the SAME direct-H2D path as a late-bound input: the returned handle's
/// [`DeviceLateInput::device_ptr`] is the `slot_to_block_dev` base the driver's
/// `launch_resolve_slot_to_block` indexes, [`DeviceLateInput::flag_ptr`] the R12
/// self-arm that orders the H2D before the resolve kernel. The host uploads its
/// OWN authoritative slot→page map — NOT the per-beam geometry (which stays
/// device-produced and host-unread), so Design B's "geometry never leaves the
/// device" invariant holds. `None` on alloc failure. No-op without `driver-cuda`.
#[cfg(feature = "driver-cuda")]
pub fn upload_slot_to_block_dict(dict: &[u32]) -> Option<DeviceLateInput> {
    upload_late_input(bytemuck::cast_slice::<u32, u8>(dict))
}

/// Host-only build: late device-alias upload is compiled out (legacy staged
/// path). Never references the `pie_driver_cuda_lib` device symbols.
#[cfg(not(feature = "driver-cuda"))]
#[derive(Debug)]
pub struct DeviceLateInput;

#[cfg(not(feature = "driver-cuda"))]
impl DeviceLateInput {
    pub fn device_ptr(&self) -> u64 {
        0
    }
    pub fn flag_ptr(&self) -> u64 {
        0
    }
    pub fn byte_len(&self) -> u32 {
        0
    }
}

#[cfg(not(feature = "driver-cuda"))]
pub fn upload_late_input(_data: &[u8]) -> Option<DeviceLateInput> {
    None
}

/// One host pinned-memory destination the driver eager-D2H's a single program
/// output VALUE into. Owns its [`pie_pinned_alloc`] buffer (freed on [`Drop`]).
///
/// `Send` because the buffer is plain pinned host memory whose access is ordered
/// across threads by the `(a2)` forward-done signal: the driver's copy-stream
/// host-func writes it (on a CUDA thread) and signals forward-done only after the
/// D2H lands; the host reads it (on the runtime thread) only after `rx` resolves.
pub struct PinnedOutput {
    ptr: *mut u8,
    len: usize,
    /// Declared tensor shape for this output value (e.g. `[1]` for a token).
    pub shape: Vec<u32>,
    /// Declared tensor dtype (`i32` token / `f32` scalar).
    pub dtype: Dtype,
}

// SAFETY: see the type doc — pinned host memory, single logical owner, access
// ordered across threads by the forward-done signal.
unsafe impl Send for PinnedOutput {}

impl Drop for PinnedOutput {
    fn drop(&mut self) {
        #[cfg(feature = "driver-cuda")]
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a live `pie_pinned_alloc` buffer (or null, guarded),
            // freed exactly once here.
            unsafe { pie_pinned_free(self.ptr as *mut core::ffi::c_void) };
        }
    }
}

/// Copy the driver-filled pinned bytes out into an owned `Vec` (the returned
/// `tensor`'s backing). The buffer itself is freed when `out` is dropped. Call
/// only AFTER the forward-done signal (`(a2)`) guarantees the D2H has landed.
pub fn read(out: &PinnedOutput) -> Vec<u8> {
    #[cfg(feature = "driver-cuda")]
    {
        if out.ptr.is_null() {
            return Vec::new();
        }
        // SAFETY: `ptr`/`len` describe a live pinned buffer the driver has
        // finished writing (ordered by the forward-done signal we already awaited).
        let bytes = unsafe { core::slice::from_raw_parts(out.ptr, out.len) }.to_vec();
        if std::env::var_os("PIE_CUT1_DEBUG").is_some() {
            tracing::warn!(
                "cut1-fastpath READ: ptr={:p} len={} bytes={:?}",
                out.ptr,
                out.len,
                &bytes[..bytes.len().min(8)]
            );
        }
        bytes
    }
    #[cfg(not(feature = "driver-cuda"))]
    {
        let _ = out;
        // Unreachable: `populate_output_fastpath` returns empty without the
        // feature, so no `PinnedOutput` is ever produced to read.
        Vec::new()
    }
}

/// True iff `kind` is on the cut #1 first slice (fixed 4-byte, submit-time-known
/// size): `token` → `[1] i32`, `scalar`/`entropy` → `[1] f32`. The remaining
/// kinds (`logits`/`logprobs` → vocab·4 / k·4; `distribution`/`embedding`) ride
/// the legacy `ForwardResponse` path until later cut #1 slices.
fn fastpath_kind(kind: &OutputKind) -> Option<(Vec<u32>, Dtype, usize)> {
    match kind {
        OutputKind::Token => Some((vec![1], Dtype::I32, 4)),
        OutputKind::Scalar | OutputKind::Entropy => Some((vec![1], Dtype::F32, 4)),
        _ => None,
    }
}

/// Populate `req`'s `sampling_output_*` carrier with one pinned destination per
/// declared output value (flattened across programs in declared output order,
/// per-program CSR), and return the owning [`PinnedOutput`]s to stash on the
/// in-flight pass. The pass is all-fast-path-or-all-legacy: anything not on the
/// driver's eager-D2H MVP slice is a no-op returning empty (legacy
/// `ForwardResponse` path), keeping `output()` from mixing pinned +
/// channel-marshaled tensors. No-op without `driver-cuda`.
///
/// MVP eligibility = a **single `Token`** output. The driver eager-D2H
/// (`executor.cpp`, "one Token per program") only fills `sampling_output_dst_ptrs`
/// when it holds exactly one slot and copies `pi.sampled[N-1]` (the sampled
/// token) into it; multi-output programs (e.g. mirostat's `[Token, Scalar]`)
/// AND single non-`Token` outputs (a lone `Scalar`/`Entropy`, which would get
/// the token's int bits as f32) MUST fall through to the legacy rich path, which
/// marshals every declared output from the kernel's `out_ptrs`. Over-enabling
/// here (any-arity, any fast-kind) desyncs from the driver: the runtime
/// short-circuits to the never-filled pinned buffers while the driver's correct
/// rich `out_resp` is discarded → token-0 / S-0 (#19).
#[cfg(feature = "driver-cuda")]
pub fn populate_output_fastpath(
    req: &mut ForwardRequest,
    programs_output_kinds: &[Vec<OutputKind>],
    programs_output_elem_counts: &[Vec<u32>],
    has_custom_program: bool,
) -> Vec<PinnedOutput> {
    // Eligible iff exactly one declared output, a `Token`, with elem_count == 1 —
    // the only shape the driver eager-D2H MVP fills (`pi.sampled[N-1]`, a single
    // i32). A `[k]`-Token (elem_count > 1), multi-output, or non-`Token` output
    // must take the rich path, where the `r×o×k` marshal handles the shape.
    //
    // #36 AND no attached custom IR program: a custom program marshals its output
    // to the rich `per_req` (`marshal_ir_program_output`) and returns BEFORE the
    // eager-D2H, so the pinned dst is NEVER filled → `output()` mis-reads the stale
    // pinned (the #19 carrier class, exposed by single-`[Token]` custom programs on
    // the merged/M-batch path). Only a recognized-STANDARD sampler writes
    // `pi.sampled` → eager-D2H fills the pinned. The runtime has no host-side
    // standard-recognition (that's the driver's #8 recognizer), so conservatively
    // ALL attached programs → rich; restoring the recognized-STANDARD fast-path
    // (which IS pinned-correct) is a perf follow-on (#37).
    let total: usize = programs_output_kinds.iter().map(|p| p.len()).sum();
    let single_token = total == 1
        && !has_custom_program
        && programs_output_kinds
            .iter()
            .flatten()
            .all(|k| matches!(k, OutputKind::Token))
        && programs_output_elem_counts.iter().flatten().all(|&n| n == 1);
    if !single_token {
        return Vec::new();
    }

    // Build into locals; only commit to `req` if every alloc succeeds (a null
    // alloc ⇒ drop the partial buffers via RAII + fall back to the legacy path).
    let mut outs: Vec<PinnedOutput> = Vec::with_capacity(total);
    let mut dst_ptrs: Vec<u64> = Vec::with_capacity(total);
    let mut dst_lens: Vec<u32> = Vec::with_capacity(total);
    let mut indptr: Vec<u32> = Vec::with_capacity(programs_output_kinds.len() + 1);
    indptr.push(0);

    for kinds in programs_output_kinds {
        for kind in kinds {
            let (shape, dtype, len) = fastpath_kind(kind).expect("eligibility checked above");
            // SAFETY: FFI to the driver's pinned arena; null is handled below.
            let ptr = unsafe { pie_pinned_alloc(len) } as *mut u8;
            if ptr.is_null() {
                // `outs` drops here → frees the buffers allocated so far; `req`
                // is untouched ⇒ the pass cleanly takes the legacy path.
                return Vec::new();
            }
            dst_ptrs.push(ptr as u64);
            dst_lens.push(len as u32);
            outs.push(PinnedOutput {
                ptr,
                len,
                shape,
                dtype,
            });
        }
        indptr.push(outs.len() as u32);
    }

    req.sampling_output_dst_ptrs = dst_ptrs;
    req.sampling_output_dst_lens = dst_lens;
    req.sampling_output_indptr = indptr;
    outs
}

/// Host-only build: the fast-path is compiled out — every pass takes the legacy
/// `ForwardResponse` path. Never references the `pie_driver_cuda_lib` symbols.
#[cfg(not(feature = "driver-cuda"))]
pub fn populate_output_fastpath(
    _req: &mut ForwardRequest,
    _programs_output_kinds: &[Vec<OutputKind>],
    _programs_output_elem_counts: &[Vec<u32>],
    _has_custom_program: bool,
) -> Vec<PinnedOutput> {
    Vec::new()
}
