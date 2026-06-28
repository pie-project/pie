//! #27 cut #1 — programmable-sampling output→tensor fast-path (host side).
//!
//! Normally a forward pass's program outputs ride the `ForwardResponse` SoA
//! channels back over IPC, and [`build_output_tensors`](super::inference) marshals
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
/// in-flight pass. The pass is all-fast-path-or-all-legacy: if ANY declared
/// output is not on the first slice (or there are no outputs), this is a no-op
/// returning empty (legacy `ForwardResponse` path), keeping `output()` from
/// mixing pinned + channel-marshaled tensors. No-op without `driver-cuda`.
#[cfg(feature = "driver-cuda")]
pub fn populate_output_fastpath(
    req: &mut ForwardRequest,
    programs_output_kinds: &[Vec<OutputKind>],
) -> Vec<PinnedOutput> {
    // Eligible iff there is ≥1 output and every output is on the first slice.
    let total: usize = programs_output_kinds.iter().map(|p| p.len()).sum();
    if total == 0
        || !programs_output_kinds
            .iter()
            .flatten()
            .all(|k| fastpath_kind(k).is_some())
    {
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
) -> Vec<PinnedOutput> {
    Vec::new()
}
