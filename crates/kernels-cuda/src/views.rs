//! The plane's carriers for the tier-1 runtime vocabulary.
//!
//! Identity in `kernels::runtime`, carrier HERE, answer in `driver-cuda` —
//! the split `kernels::raises` documents, applied to the resident objects.
//! A routine takes one as `In<Struct<KvCache>>`: positional, counted by
//! `arity_problem`, visible in the derived column. The fields below replace
//! the `ctx.ask` keys named beside each; the driver builds one view per
//! (fire, layer) instead of answering the keys one at a time.

use core::ffi::c_void;

/// `In<Struct<KvCache>>` — the paged KV cache, one view per (fire, layer).
///
/// Field per retired key, `.wiki/designs/design-no-ask.md` §2. The write
/// half is nullable: a fire that appends nothing carries nulls, which is
/// what `KvWritePageOrNull` spelled as a second key.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PagedKvView {
    /// `keys::KvKeys` — the key plane's base.
    pub keys: *mut u8,
    /// `keys::KvValues`.
    pub values: *mut u8,
    /// `keys::KvBf16Keys` — the bf16 shadow plane under a quantised scheme.
    pub bf16_keys: *mut u8,
    /// `keys::KvBf16Values`.
    pub bf16_values: *mut u8,
    /// `keys::KvPageIndices`.
    pub page_indices: *const i32,
    /// `keys::KvPageIndptr`.
    pub page_indptr: *const i32,
    /// `keys::KvLastPageLens`.
    pub last_page_lens: *const i32,
    /// `keys::KvKeyScales`.
    pub key_scales: *const c_void,
    /// `keys::KvValueScales`.
    pub value_scales: *const c_void,
    /// `keys::KvWritePage` / `keys::KvWritePageOrNull` — null when the fire
    /// writes nothing.
    pub write_page: *const i32,
    /// `keys::KvWriteOffset` / `keys::KvWriteOffsetOrNull`.
    pub write_offset: *const i32,
    /// `keys::KvPageSize`.
    pub page_size: i32,
    /// `keys::KvSeqStride`.
    pub seq_stride: i64,
    /// `keys::KvHeadStride`.
    pub head_stride: i64,
    /// `keys::KvHndLayout` — nonzero for HND.
    pub layout: i32,
    /// `keys::KvStorageDtype`.
    pub storage_dtype: i32,
    /// `keys::KvSchemeByte`.
    pub scheme_byte: i32,
    /// `keys::KvNativeBf16`.
    pub native_bf16: bool,
    /// `keys::KvHasEnvelopes`.
    pub has_envelopes: bool,
    /// `keys::KvEnvMin` — the dequant envelope floor PLANE, at the cache's
    /// width; null when the cache carries no envelopes, which is what
    /// `has_envelopes` says.
    pub env_min: *const u16,
    /// `keys::KvEnvMax` — the ceiling plane; see `env_min`.
    pub env_max: *const u16,
    /// `keys::KvBlockSize`.
    pub block_size: i32,
    /// `keys::KvMaxPagesPerRequest`.
    pub max_pages_per_request: i32,
    /// `keys::KvPagesInBatch`.
    pub pages_in_batch: i32,
    /// THE FIRE'S QUERY CSR, device-resident `[requests + 1]` — where each
    /// request's token rows begin in this fire's rectangle.
    ///
    /// A POOL VIEW ALREADY CARRIES THE FIRE, which is what makes this a
    /// field and not an operand: `write_page`/`write_offset` are per-ROW of
    /// this fire, `pages_in_batch` and `max_pages_per_request` are per-FIRE,
    /// and `driver-cuda/src/bind/views.rs::kv_view` says in its own header
    /// that it builds one view "from the layer's pool descriptor and the
    /// fire-wide CSRs/write descriptors on `AttnCtx`". An append resolves
    /// its destination out of exactly this CSR plus the page table beside
    /// it, so a point whose whole statement is "leave these rows in that
    /// pool" can be claimed by a body that reads the pool row — and nothing
    /// else. Was answered as `keys::QoIndptr` / the `qo_indptr` runtime
    /// stream, which is still how a ROUTINE with its own operand column
    /// takes it.
    pub qo_indptr: *const i32,
    /// THE FIRE'S ROW VALIDITY, one BYTE per token row, or null when every
    /// row is valid. See [`Self::qo_indptr`] for why it lives here; the
    /// appending kernels all test for the null.
    pub row_valid: *const u8,
    /// How many requests [`Self::qo_indptr`] bounds — `indptr.len() - 1`,
    /// which a device pointer does not spell.
    pub requests: i32,
}

/// `In<Struct<RecurrentState>>` — the GDN/mamba state, one view per
/// (fire, layer): the recurrent slab and the conv-window half.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RecurrentView {
    /// `keys::GdnRecurrentSlab`.
    pub slab: *mut c_void,
    /// `keys::GdnSlotIds`.
    pub slot_ids: *const i32,
    /// `keys::GdnStateStride`.
    pub slot_stride_elems: i64,
    /// `keys::RecurrentSlots`.
    pub slots: *const i32,
    /// `keys::RecurrentState`.
    pub state: *mut c_void,
    /// `keys::ConvState`.
    pub conv_state: *mut c_void,
    /// `keys::NewConvState`.
    pub new_conv_state: *mut c_void,
    /// `keys::GdnConvSlab`.
    pub conv_slab: *mut c_void,
    /// `keys::GdnConvStride`.
    pub conv_stride: i64,
}

/// `In<Struct<AttnMask>>` — the custom-mask triple, null/false when absent.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MaskView {
    /// `keys::AttentionMask` / `keys::AttnMask`.
    pub mask: *const u8,
    /// `keys::AttnMaskIndptr`.
    pub indptr: *const i32,
    /// `keys::AttentionMaskEnabled`.
    pub enabled: bool,
    /// `keys::AttentionMaskStride`.
    pub stride: i64,
}

/// `In<Struct<ExpertWeights>>` — a layer's quantised expert-weight banks:
/// device arrays of per-expert pointers, built at load and resident.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ExpertWeightsView {
    /// `keys::WeightExpertPtrs` — per-expert packed weight planes.
    pub ptrs: *const u8,
    /// `keys::WeightExpertScalePtrs` — per-expert scale planes.
    pub scale_ptrs: *const u8,
    /// `keys::WeightBias` — per-expert bias planes; null when the checkpoint
    /// carries none.
    pub bias_ptrs: *const u8,
}

/// `In<Struct<MoeBanks>>` — the nemotron MoE grouped-GEMM banks, carved per
/// fire: the per-expert weight pointer arrays, the routed scratch planes and
/// the A/B/C pointer arrays the build kernels fill.
///
/// One view serves both build forms; the half a fire does not route through
/// (`expert_*` for the aligned form, `aligned_*` for the decode form) is
/// null.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MoeBanksView {
    /// `keys::MoeUpWeightPtrs`.
    pub up_weight_ptrs: *const *const c_void,
    /// `keys::MoeDownWeightPtrs`.
    pub down_weight_ptrs: *const *const c_void,
    /// `keys::MoeExpertUp`.
    pub expert_up: *mut c_void,
    /// `keys::MoeExpertAct`.
    pub expert_act: *mut c_void,
    /// `keys::MoeExpertOut`.
    pub expert_out: *mut c_void,
    /// `keys::MoeAlignedUp`.
    pub aligned_up: *mut c_void,
    /// `keys::MoeAlignedAct`.
    pub aligned_act: *mut c_void,
    /// `keys::MoeAlignedOut`.
    pub aligned_out: *mut c_void,
    /// `keys::MoeAUpPtrs`.
    pub a_up_ptrs: *mut *const c_void,
    /// `keys::MoeBUpPtrs`.
    pub b_up_ptrs: *mut *const c_void,
    /// `keys::MoeCUpPtrs`.
    pub c_up_ptrs: *mut *mut c_void,
    /// `keys::MoeADownPtrs`.
    pub a_down_ptrs: *mut *const c_void,
    /// `keys::MoeBDownPtrs`.
    pub b_down_ptrs: *mut *const c_void,
    /// `keys::MoeCDownPtrs`.
    pub c_down_ptrs: *mut *mut c_void,
    /// `keys::MoeRouteWeights`.
    pub route_weights: *mut f32,
}

/// `In<Struct<GemmGroups>>` — the grouped cuBLAS GEMM's pointer arrays,
/// staged per fire: device arrays of per-group operand pointers and the
/// host-side M array the grouped call reads.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GemmGroupsView {
    /// `keys::GemmActPtrs`.
    pub act_ptrs: *const *const c_void,
    /// `keys::GemmWeightPtrs`.
    pub weight_ptrs: *const *const c_void,
    /// `keys::GemmOutPtrs`.
    pub out_ptrs: *const *mut c_void,
    /// `keys::GemmMArrayHost` — HOST memory, read by the grouped call, not
    /// by a kernel.
    pub m_array_host: *const i32,
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
kernels::resident!(
    /// A layer's quantised expert-weight banks. Was `keys::WeightExpertPtrs`
    /// / `WeightExpertScalePtrs` / `WeightBias`.
    ExpertWeights = "moe.expert_weights" => ExpertWeightsView
);
kernels::raise!(
    /// The MoE grouped-GEMM banks, carved per fire.
    MoeBanks = "moe.banks" => MoeBanksView
);
kernels::raise!(
    /// The grouped GEMM's pointer arrays, staged per fire.
    GemmGroups = "gemm.groups" => GemmGroupsView
);
kernels::resident!(
    /// The DSV4 compression state's KV half. Was `keys::Dsv4StateKv`.
    Dsv4StateKv = "dsv4.state_kv" => crate::jit::abi::bf16
);
kernels::resident!(
    /// The DSV4 compression state's score half. Was `keys::Dsv4StateScore`.
    Dsv4StateScore = "dsv4.state_score" => crate::jit::abi::bf16
);
kernels::resident!(
    /// The DSV4 absolute-position-encoding table. Was `keys::Dsv4Ape`.
    Dsv4Ape = "dsv4.ape" => f32
);
kernels::resident!(
    /// The DSV4 compressed-KV page pool. Was `keys::Dsv4CompKvPages`.
    Dsv4CompKvPages = "dsv4.comp_kv_pages" => crate::jit::abi::bf16
);
kernels::resident!(
    /// The MTP pending-hidden slab; element type is the model's. Was
    /// `keys::MtpPendingHidden`.
    MtpPendingHidden = "mtp.pending_hidden" => c_void
);
kernels::raise!(
    /// The fire's qo indptr, staged in HOST memory for planning. Was
    /// `keys::QoIndptrHost`.
    QoIndptrHost = "qo_indptr.host" => u32
);
kernels::raise!(
    /// The fire's KV page indptr, staged in HOST memory for planning. Was
    /// `keys::KvPageIndptrHost`.
    KvPageIndptrHost = "kv_page_indptr.host" => u32
);

/// `In<Struct<AttnScore>>` — the attention-score observation the driver
/// keeps: the per-request CSR of observed rows and the window each keeps.
/// Both are the FIRE's (the CSR is staged per fire; the window is
/// boot-configured policy), so neither is a statement's to state — the
/// capture forms read this view where one carried a `Const` zero and the
/// other minted a stream no driver answered.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ScoreView {
    /// The observed-rows CSR, `[Requests + 1]`.
    pub indptr: *const i32,
    /// Rows each request keeps; `0` means the fire observes nothing.
    pub window: u32,
}

kernels::resident!(
    /// The score observation. Per-fire, driver policy.
    AttnScore = "attn.score" => ScoreView
);
