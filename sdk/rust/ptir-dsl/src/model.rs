//! Model constants (overview §4) — trace-known, learned from the model at trace
//! time (as today's `Graph::new(vocab)`). The inferlet configures them once; the
//! trace is specialized to the concrete values (a different model = a different
//! trace, batch-by-program). Also builds echo's bind-time
//! [`ModelProfile`](pie_ptir::registry::ModelProfile).

use core::cell::Cell;

use pie_ptir::registry::ModelProfile;
use pie_ptir::types::DType;

#[derive(Clone, Copy, Debug)]
struct ModelConfig {
    vocab: u32,
    page_size: u32,
    num_layers: u32,
    has_mtp_logits: bool,
    has_value_head: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig { vocab: 32_000, page_size: 16, num_layers: 32, has_mtp_logits: true, has_value_head: true }
    }
}

thread_local! {
    static MODEL: Cell<ModelConfig> = Cell::new(ModelConfig::default());
}

/// Configure the model constants for subsequent traces on this thread.
pub fn configure(vocab: u32, page_size: u32, num_layers: u32) {
    MODEL.with(|m| {
        let mut c = m.get();
        c.vocab = vocab;
        c.page_size = page_size;
        c.num_layers = num_layers;
        m.set(c);
    });
}

/// Declare model-gated intrinsic availability (`mtp_logits`, `value_head`).
pub fn configure_gates(has_mtp_logits: bool, has_value_head: bool) {
    MODEL.with(|m| {
        let mut c = m.get();
        c.has_mtp_logits = has_mtp_logits;
        c.has_value_head = has_value_head;
        m.set(c);
    });
}

pub(crate) fn vocab() -> u32 {
    MODEL.with(|m| m.get().vocab)
}
pub(crate) fn page_size() -> u32 {
    MODEL.with(|m| m.get().page_size)
}
pub(crate) fn num_layers() -> u32 {
    MODEL.with(|m| m.get().num_layers)
}

/// The bind-time [`ModelProfile`] echo's validator uses (activation = F32 for the
/// reference interpreter; the SDK carries no second-party kernels until P7).
/// Native/test-only: the guest does not bind (D6); parity tests bind explicitly
/// with this profile.
pub fn profile() -> ModelProfile {
    let c = MODEL.with(|m| m.get());
    ModelProfile {
        vocab: c.vocab,
        page_size: c.page_size,
        num_layers: c.num_layers,
        activation: DType::F32,
        has_mtp_logits: c.has_mtp_logits,
        has_mtp_drafts: c.has_mtp_logits,
        has_value_head: c.has_value_head,
        kernels: alloc::vec::Vec::new(),
    }
}
