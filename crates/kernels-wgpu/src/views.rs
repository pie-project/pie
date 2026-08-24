//! The plane's carriers for the tier-1 runtime vocabulary.
//!
//! Identity in `kernels::runtime`, carrier HERE, answer in `driver-wgpu` —
//! the split `kernels::raises` documents, applied to the resident objects.
//! The fields below replace the `ctx.ask` keys named beside each; the driver
//! builds one view per (fire, layer) instead of answering the keys one at a
//! time.
//!
//! TWO IDENTITIES, NOT FIVE. `KvCache`, `AttnMask` and `AttnSplit` stood here
//! too, one `In<Struct<..>>` per raise, because a launcher stated its whole
//! input on the operand column and took the three separately. A point states
//! operands and scalars only, so a body reaches exactly one object the driver
//! built for this fire — [`AttnFire`] for attention, [`RecurrentState`] for
//! the mixers — and the three view STRUCTS survive as fields of the first.
//!
//! CUDA's `PagedKvView` holds pointers; a shader plane's holds what its asks
//! carried — `Tensor<E>` binding handles and the strides beside them. Same
//! contract (`.wiki/designs/design-no-ask.md` §2), this plane's carriers.

use kernels::shader::{Tensor, Usize, bf16};

/// The paged KV cache, one view per (fire, layer); reached as
/// [`AttnFireView::kv`].
///
/// Field per retired key. On this plane the cache planes are binding
/// handles, so there is no null write half: a fire that appends nothing
/// never fires an entrypoint that names `write_page`.
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

/// The custom-mask triple; reached as [`AttnFireView::mask`].
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
    /// The recurrent-state slabs. Tier-1.
    RecurrentState = "recurrent_state" => RecurrentView
);

/// The driver's decode split policy; reached as [`AttnFireView::split`]. How
/// many KV splits this fire's decode runs, and the partials plane the split
/// form folds. `splits <= 1` is the unsplit reading and the partials handle
/// is then never read. What `keys::AttnSplits`/`keys::AttnPartials` (vulkan)
/// and the optional `keys::AttnScratch` (wgpu) asked; metal fires unsplit and
/// its driver answers `splits: 1`.
#[derive(Debug, Clone, Copy)]
pub struct SplitView {
    /// The partials/scratch plane, `[splits, rows, heads, head_dim + 1]`-ish
    /// in the plane's own layout. Handle 0 when `splits <= 1`.
    pub partials: Tensor<f32>,
    /// The split count; `<= 1` means unsplit.
    pub splits: i32,
}

/// `Cache<Self::Pages>` — the paged KV row AND the per-fire staging every
/// sdpa arm on this plane reads.
///
/// # Why the points layer needs a wider view than the launchers did
///
/// A launcher stated its whole input on the operand column, so the paged
/// decode took FIVE raises and tensors the statement does not carry — the
/// pool row, `positions`, `request_of_token`, the mask triple, the split
/// policy — and the driver answered each one separately. A POINT states
/// operands and scalars only (`.wiki/baker.md`: "Plane staging never appears
/// in a declaration — the body pulls it from `self`"), and
/// `attention.decode` declares exactly `q`, the pool row, `window`,
/// `head_dim`, `sm_scale`, `o`.
///
/// On cuda the body pulls the rest off `self`, because `Ctx` is a struct with
/// an env behind it. On this plane `Ctx` is `dyn Encode` and has no env, so
/// the only object a body holds that the driver built for THIS fire is the
/// pool row. Which is exactly the move W7 made on cuda when `mla.kv_append`
/// became a claimed body: "the pool view grew the fire's qo_indptr/row_valid/
/// requests it was always built out of". This is that, for wgpu's five.
///
/// # SEAM — what P5 owes
///
/// * **A builder.** `driver-wgpu` builds `PagedKvView`, `MaskView` and
///   `SplitView` today and hands each as its own `In<Struct<..>>`. It must
///   build ONE of these per (fire, layer) instead. Every field below is a
///   value it already has at that point; nothing new is measured.
/// * **`kv_heads`.** `attention.decode` declares no KV head count — cuda
///   reads it off the pool's strides (`head_split`), which needs a layout
///   flag this plane's view does not carry. Rather than invent one, the count
///   is stated here, by the POOL, which is the party that chose it when the
///   slab was allocated. `attention.prefill` declares its own `kv_heads` and
///   the two must agree; a body that finds them disagreeing refuses.
/// * **The capability tier is NOT here.** `attn/sdpa_paged_mma.wgsl` needs
///   `Capability::Matrix` and `sdpa_paged.wgsl`'s tiled arm does not, and
///   that is a DEVICE fact, not a fire's. It belongs on `Encode` — a
///   `fn capability(&self) -> Capability` — so a body can branch on it the
///   way cuda's bodies branch on `Ctx::device()`. Until then every prefill
///   claim fires the tiled arm and says so; [`crate::attn::mma`] is the other
///   arm, written and waiting for the branch.
#[derive(Debug, Clone, Copy)]
pub struct AttnFireView {
    /// The pool row itself, unchanged.
    pub kv: PagedKvView,

    /// `keys::Positions` — one per token of this fire.
    pub positions: Tensor<i32>,

    /// `keys::RequestOfToken`.
    pub request_of_token: Tensor<i32>,

    /// The custom-mask triple, folded in: a point declares no mask slot and
    /// every sdpa entrypoint binds all three words.
    pub mask: MaskView,

    /// The decode split policy, folded in for the same reason.
    pub split: SplitView,

    /// The KV head count the pool was laid out with. See the seam above.
    pub kv_heads: i32,
}

kernels::resident!(
    /// The paged KV row as a POINT's body reads it. Tier-1: the same pool
    /// `KvCache` names, one view wider.
    AttnFire = "attn.fire" => AttnFireView
);
