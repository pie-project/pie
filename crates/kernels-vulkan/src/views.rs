//! The plane's carriers for the tier-1 runtime vocabulary.
//!
//! Identity in `kernels::runtime`, carrier HERE, answer in `driver-vulkan` —
//! the split `kernels::raises` documents, applied to the resident objects.
//! A routine takes one as `In<Struct<KvCache>>`: positional, counted by
//! `arity_problem`, visible in the derived column. The fields below replace
//! the `ctx.ask` keys named beside each; the driver builds one view per
//! (fire, layer) instead of answering the keys one at a time.
//!
//! CUDA's `PagedKvView` holds pointers; a shader plane's holds what its asks
//! carried — `Tensor<E>` binding handles and the strides beside them. Same
//! contract (`.wiki/designs/design-no-ask.md` §2), this plane's carriers.

use kernels::shader::{Tensor, Usize, bf16};

/// `In<Struct<KvCache>>` — the paged KV cache, one view per (fire, layer).
///
/// Field per retired key. On this plane the cache planes are binding
/// handles, so there is no null write half: a fire that appends nothing
/// never fires a routine that names `write_page`.
#[derive(Debug, Clone, Copy)]
pub struct PagedKvView {
    /// `keys::KvKeys` — the key plane.
    pub keys: Tensor<bf16>,
    /// `keys::KvValues`.
    pub values: Tensor<bf16>,
    /// `keys::KvPageIndices`.
    pub page_indices: Tensor<u32>,
    /// `keys::KvPageIndptr`.
    pub page_indptr: Tensor<u32>,
    /// `keys::KvWritePage`.
    pub write_page: Tensor<u32>,
    /// `keys::KvWriteOffset`.
    pub write_offset: Tensor<u32>,
    /// `keys::KvPageSize`.
    pub page_size: i32,
    /// `keys::KvSeqStride`.
    pub seq_stride: Usize,
    /// `keys::KvHeadStride`.
    pub head_stride: Usize,
}

/// `In<Struct<RecurrentState>>` — the GDN/mamba state, one view per
/// (fire, layer): the recurrent slab and the conv-window half.
#[derive(Debug, Clone, Copy)]
pub struct RecurrentView {
    /// `keys::RecurrentState`.
    pub state: Tensor<f32>,
    /// `keys::RecurrentSlots`.
    pub slots: Tensor<u32>,
    /// `keys::ConvState`.
    pub conv_state: Tensor<f32>,
    /// `keys::NewConvState`.
    pub new_conv_state: Tensor<f32>,
}

/// `In<Struct<AttnMask>>` — the custom-mask triple.
///
/// `enabled` is a per-request byte plane on this plane, not a bool: the
/// shader reads it per row, which is what `keys::AttentionMaskEnabled`
/// carried here.
#[derive(Debug, Clone, Copy)]
pub struct MaskView {
    /// `keys::AttentionMask`.
    pub mask: Tensor<u8>,
    /// `keys::AttentionMaskEnabled`.
    pub enabled: Tensor<u8>,
    /// `keys::AttentionMaskStride`.
    pub stride: u32,
}

kernels::resident!(
    /// The paged KV cache. Tier-1: `kernels::runtime::TIER1` names it.
    KvCache = "kv_cache" => PagedKvView
);
kernels::resident!(
    /// The recurrent-state slabs. Tier-1.
    RecurrentState = "recurrent_state" => RecurrentView
);
kernels::resident!(
    /// The custom-mask triple. Per-fire, staged by the driver.
    AttnMask = "attention_mask" => MaskView
);
