//! #11 prefetch seam — the in-proc registration entry + the `&[Binding]` →
//! `(kind[], key[])` marshal shared by the embedded driver channels.
//!
//! The C++ backend registers a [`PrefetchFn`] trampoline once at ready-time (via
//! `InProcVTable::register_prefetch`); the channel stores it and invokes it on
//! `driver::prefetch_compile`. The marshal mirrors the **submit carrier** exactly
//! (`SamplingBinding::kind()` / `key()`, dropping readiness) so the C++ side can
//! reconstruct `ready = SubmitBound` identically to the submit path — making the
//! warmed `program_identity_hash` match the real fire (cache hit = the TTFT win).

use std::ffi::c_void;

use pie_ipc::ffi::PrefetchFn;

/// A driver's registered JIT prefetch entry: the C++ trampoline plus the opaque
/// backend pointer to thread back into it. Registered once at backend-ready.
pub(crate) struct PrefetchEntry {
    prefetch: PrefetchFn,
    backend_ctx: *mut c_void,
}

// SAFETY: the registered trampoline and `backend_ctx` are contractually safe to
// invoke from any thread — charlie's `prefetch_compile` submits to the
// off-context PTX pool (no per-context / `programs_` state touched), so it
// tolerates being called from the host execute thread off the driver loop.
unsafe impl Send for PrefetchEntry {}
unsafe impl Sync for PrefetchEntry {}

impl PrefetchEntry {
    pub(crate) fn new(prefetch: PrefetchFn, backend_ctx: *mut c_void) -> Self {
        Self {
            prefetch,
            backend_ctx,
        }
    }

    /// Marshal `manifest` to the wire `(kind, key)` arrays and fire the trampoline
    /// (fire-and-forget). Mirrors the submit carrier; readiness is dropped (the
    /// C++ side reconstructs `SubmitBound`), so the warmed hash matches submit.
    pub(crate) fn invoke(&self, bytecode: &[u8], manifest: &[pie_sampling_ir::Binding]) {
        let mut kinds: Vec<u8> = Vec::with_capacity(manifest.len());
        let mut keys: Vec<u32> = Vec::with_capacity(manifest.len());
        for b in manifest {
            let (kind, key) = binding_kind_key(b);
            kinds.push(kind);
            keys.push(key);
        }
        // SAFETY: `self.prefetch` + `backend_ctx` were registered by the in-proc
        // C++ backend; the trampoline is thread-safe and copies what it retains
        // before returning, so the borrowed slices (valid for this call) suffice.
        unsafe {
            (self.prefetch)(
                self.backend_ctx,
                bytecode.as_ptr(),
                bytecode.len(),
                kinds.as_ptr(),
                keys.as_ptr(),
                kinds.len(),
            );
        }
    }
}

/// Map an IR [`Binding`](pie_sampling_ir::Binding) to the wire `(kind, key)` the
/// submit carrier conveys — via the canonical [`SamplingBinding::kind`] /
/// [`key`](pie_driver_abi::SamplingBinding::key), so it cannot drift from submit.
/// **Readiness is intentionally dropped**: the carrier conveys only `(kind,key)`
/// and the driver reconstructs `SubmitBound` (executor.cpp), so the prefetch must
/// too or the warmed `program_identity_hash` would miss the real fire.
fn binding_kind_key(b: &pie_sampling_ir::Binding) -> (u8, u32) {
    let sb = match b {
        pie_sampling_ir::Binding::Logits => pie_driver_abi::SamplingBinding::Logits,
        pie_sampling_ir::Binding::MtpLogits => pie_driver_abi::SamplingBinding::MtpLogits,
        pie_sampling_ir::Binding::MtpDrafts => pie_driver_abi::SamplingBinding::MtpDrafts,
        pie_sampling_ir::Binding::Tensor { key, .. } => {
            pie_driver_abi::SamplingBinding::Tensor { key: *key }
        }
    };
    (sb.kind(), sb.key())
}

#[cfg(test)]
mod tests {
    use super::binding_kind_key;
    use pie_sampling_ir::{Binding, Readiness};

    #[test]
    fn marshal_mirrors_submit_carrier_and_drops_readiness() {
        // The prefetch marshal MUST equal the submit carrier's (kind, key)
        // (SamplingBinding::kind()/key()) so the warmed program_identity_hash
        // matches the real fire (the cache-HIT invariant @delta asserts e2e).
        assert_eq!(binding_kind_key(&Binding::Logits), (0, 0));
        assert_eq!(binding_kind_key(&Binding::MtpLogits), (2, 0));
        assert_eq!(
            binding_kind_key(&Binding::Tensor { key: 7, ready: Readiness::Submit }),
            (1, 7),
        );
        // Readiness is dropped: Late vs Submit marshal identically (the driver
        // reconstructs SubmitBound for both).
        assert_eq!(
            binding_kind_key(&Binding::Tensor { key: 7, ready: Readiness::Late }),
            binding_kind_key(&Binding::Tensor { key: 7, ready: Readiness::Submit }),
        );
    }
}
