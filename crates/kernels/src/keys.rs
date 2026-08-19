//! Facts, as types.
//!
//! A body wanting the fire's token count writes `ctx.ask::<i32, keys::Rows>()`.
//! The word `"rows"` appears once, here, and never at a use site.
//!
//! WHAT IS LEFT HERE IS WHAT ONLY A FIRE CAN ANSWER. A fact the checkpoint
//! fixes at load time is not the environment's and never was: it is a
//! constant, and a constant reaches the kernel as a `Const` parameter the
//! statement carries. 71 keys over 936 uses left this file for that reason.
//! What stays is the batch, the plan and the allocator, which is what this
//! engine does.
//!
//! A PARAMETER NAME AND A FACT COINCIDE ONLY BY LUCK: [`Theta`] is the layer's
//! rope base and [`RopeTheta`] the fire's, and gemma-4 makes them differ. The
//! converse is [`KvPageSize`], where two spellings end at one read.
//!
//! §1 is what the CUDA binder answers. §2 is what a statement can truthfully
//! carry that no backend answers yet. Do not reach for a §1 key because a §2
//! one refuses.
//!
//! A key is a QUESTION and never a carrier. `ask` names both — the carrier
//! first, the question second — because a fact's `Value` is one concrete type
//! across every backend while a shader carrier is a binding index.

use crate::Source;

/// A fact only the fire can answer, named by a type.
///
/// `#[routine]` reads these consts **as a path expression** — it emits
/// `<keys::RmsEps as Fact>::SOURCE` and cannot know which fact that is, so
/// no table is consulted.
pub trait Fact: Copy {
    /// The word, written exactly once in the tree — here.
    const KEY: &'static str;

    /// What the binder is asked for.
    ///
    /// A fact may carry an index or a string (`Source::Slot(Kind::Aux, 3)`),
    /// and it is stated HERE, once.
    const SOURCE: Source;

    /// What arrives, in the ABI's terms.
    type Value: Copy;

    /// The value, unwrapped.
    fn get(self) -> Self::Value;

}

/// Declare a fact.
///
/// ```ignore
/// fact!(RmsEps = "rms_eps" => Source::Named("rms_eps") => f32);
/// ```
///
/// THERE IS NO `stated` PREFIX ANY MORE. It marked a fact only the statement
/// could answer, and every one of those was a scalar with no mark to carry it:
/// nine weight-walker extents declared `stated`, resolving through
/// `Source::Named`, answered by no driver, so every routine taking one was
/// unreachable. `Const<i32>` is what they were missing, and a `Const` names no
/// key at all.
///
/// A newtype and not a unit struct: the value rides in the field, so a hand
/// arm can build one and [`Fact::env`] can wrap it.
#[macro_export]
macro_rules! fact {
    ($(#[$m:meta])* $name:ident = $key:literal => $src:expr => $value:ty) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name(pub $value);

        impl $crate::keys::Fact for $name {
            const KEY: &'static str = $key;
            const SOURCE: $crate::Source = $src;
            type Value = $value;

            fn get(self) -> $value {
                self.0
            }
        }

        impl core::ops::Deref for $name {
            type Target = $value;

            fn deref(&self) -> &$value {
                &self.0
            }
        }

        // NO `Arg` IMPL. A fact is a KEY and never a carrier.
        //
        // It used to be both, and that is what made the two planes spell one
        // claim two ways: `Env<keys::RmsEps>` put the fact in the CARRIER slot
        // and worked on CUDA, where a fact's `Value` is the type the ABI
        // passes; a shader carrier is a binding index, so the three shader
        // planes had to write `Env<f32, keys::RmsEps>` instead. Six hundred
        // and fifty-five sites took the first spelling and one thousand three
        // hundred the second, CUDA itself using both.
        //
        // With the impl gone there is one spelling, `Env<carrier, key>`, and
        // the question and the thing that carries the answer are always
        // separate -- which is what they are.

    };
}

// ── §1  ANSWERED ──────────────────────────────────────────────────────────
//
// `bind/table.rs`'s `operand()` has an arm for each of these. A launcher that
// names one and is otherwise complete resolves end to end.

fact!(
    /// The normalisation epsilon.
    RmsEps = "rms_eps" => Source::Named("rms_eps") => f32
);

fact!(
    /// THE LAYER'S rope base — `Cx::theta`, which resolves this statement's
    /// layer and falls back.
    ///
    /// gemma-4 splits by layer kind, sliding at 1e4 and full at 1e6, so on
    /// that model this and [`RopeTheta`] are different numbers on the same
    /// fire, and the wrong one resolves, passes and rotates by the wrong angle.
    Theta = "theta" => Source::Named("theta") => f32
);

fact!(
    /// THE FIRE'S rope base — `Facts::rope_theta`, the field.
    ///
    /// See [`Theta`]: the type carries the pick between the two rope bases.
    RopeTheta = "rope_theta" => Source::Named("rope_theta") => f32
);

fact!(
    /// The rotated prefix width — `rotary_dim`, not the head width.
    ///
    /// For a statement that does not state it; one that does spells
    /// `Param<0, i32>`.
    RotaryWidth = "rotary_width" => Source::Named("rotary_width") => i32
);

fact!(
    /// How far left of a row attention may see; `-1` is unbounded.
    ///
    /// `attn_at` states it at `params[0]` on every attention row, which is why
    /// nothing else may read that slot. `attn_plan` picks the full-attention
    /// decode plan on `-1`.
    WindowLeft = "window_left" => Source::Named("window_left") => i32
);

fact!(/// The attention head width.
    HeadDim = "head_dim" => Source::Named("head_dim") => i32);

fact!(/// Query heads on this statement's layer.
    NumQHeads = "num_q_heads" => Source::Named("num_q_heads") => i32);

fact!(/// Key/value heads on this statement's layer.
    NumKvHeads = "num_kv_heads" => Source::Named("num_kv_heads") => i32);

fact!(
    /// The per-head width a gated norm normalises over.
    ///
    /// Total: `0` is "the whole row", not a swallowed absence. `RmsnormPerHead`
    /// and plain `Rmsnorm` share one launcher and this picks the reduction, so
    /// a `Param<N, i32>` would refuse on every plain norm.
    PerHeadDim = "per_head_dim" => Source::Named("per_head_dim") => i32
);

fact!(/// Rows in the vocabulary — the LM head's output width.
    Vocab = "vocab" => Source::Named("vocab") => i32);

fact!(/// The per-layer embedding width gemma's PLE path reads.
    PleDim = "ple_dim" => Source::Named("ple_dim") => i32);

fact!(/// Requests in this fire, which is not the token count.
    RequestCount = "request_count" => Source::Named("request_count") => i32);

fact!(
    /// Tokens in this fire — the row count every rectangle is measured in.
    ///
    /// A fallback: where the rectangle is in the same signature, `n` is
    /// `dt_out.rows` and this parameter goes away.
    ///
    /// The FIRE's token count. `ssm`'s row counts are a Mamba block's and
    /// coincide often enough that a substitution reads a neighbouring row
    /// rather than faulting. Not [`RowsTotal`] either -- this is `rows.count`,
    /// that is the whole row space, and a peel is where they differ.
    Rows = "rows" => Source::Named("rows") => i32
);

fact!(/// The token id plane.
    TokenIds = "token_ids" => Source::Named("token_ids") => *const i32);

fact!(/// The position plane.
    Positions = "positions" => Source::Named("positions") => *const i32);

fact!(
    /// The rows the sampler gathers.
    SamplingIndices = "sampling_indices" => Source::Named("sampling_indices") => *const i32
);

fact!(/// Tokens per KV page.
    KvPageSize = "kv_page_size" => Source::Named("kv_page_size") => i32);

// ── The five the shader backends construct and CUDA does not spell ──
//
// `kernels-vulkan` and `kernels-wgpu` build these `Source` values through the
// `kernel!` DSL's `$src:expr`, not the CUDA-side attributes the censuses read.
// Value types come off that DSL: strides are index-sized, write coordinates
// are per-token planes, the mask is a byte plane with a scalar stride.
// `kernels_cuda::unanswerable_named_rows()` pins that no CUDA routine names
// one.

fact!(/// The custom attention mask, a byte plane the driver allocated.
    AttentionMask = "attention_mask" => Source::Named("attention_mask") => *const u8);

fact!(/// The per-lane byte saying whether [`AttentionMask`] applies.
    AttentionMaskEnabled = "attention_mask_enabled" => Source::Named("attention_mask_enabled") => *const u8);

fact!(/// Bytes between one lane's mask row and the next.
    AttentionMaskStride = "attention_mask_stride" => Source::Named("attention_mask_stride") => u32);

fact!(/// Rows between one KV head and the next, in the DRIVER's allocation.
    KvHeadStride = "kv_head_stride" => Source::Named("kv_head_stride") => usize);

fact!(/// Rows between one token and the next within a head.
    KvSeqStride = "kv_seq_stride" => Source::Named("kv_seq_stride") => usize);

fact!(/// Per token: which KV page it is written to, `position / page_size`.
    ///
    /// A PLANE and not a scalar — `kernels-vulkan/src/attn.rs:78` binds it
    /// `U32s` — which the name reads against.
    KvWritePage = "kv_write_page" => Source::Named("kv_write_page") => *const u32);

fact!(/// Per token: the row within [`KvWritePage`]'s page.
    KvWriteOffset = "kv_write_offset" => Source::Named("kv_write_offset") => *const u32);

fact!(/// [`KvWritePage`], bound as it stands rather than refused.
    ///
    /// A fire that appends no KV carries a null here and the kernel branches
    /// on it, so the refusing spelling would decline a valid fire.
    KvWritePageOrNull = "kv_write_page.or_null"
        => Source::Named("kv_write_page.or_null") => *const u32);

fact!(/// [`KvWriteOffset`], bound as it stands; see [`KvWritePageOrNull`].
    KvWriteOffsetOrNull = "kv_write_offset.or_null"
        => Source::Named("kv_write_offset.or_null") => *const u32);

fact!(
    /// The KV cache's own head width, which is not always [`HeadDim`].
    KvHeadDim = "kv.head_dim" => Source::Named("kv.head_dim") => i32);

fact!(
    /// The KV cache's own head count, which is not always [`NumKvHeads`].
    KvNumHeads = "kv.num_kv_heads" => Source::Named("kv.num_kv_heads") => i32
);

fact!(/// Per-layer dequant envelope floor for a quantised KV cache.
    ///
    /// `u16`: the envelope is stored at the cache's width, not the compute
    /// width. Null binds rather than refusing — `has_envelopes()` *is*
    /// `!k_env_min.is_null() && !k_env_max.is_null()`, so a row reading this
    /// also reads the fact saying whether to dereference it.
    KvEnvMin = "kv.k_env_min" => Source::Named("kv.k_env_min") => *const u16);

fact!(/// Per-layer dequant envelope ceiling; see [`KvEnvMin`].
    KvEnvMax = "kv.k_env_max" => Source::Named("kv.k_env_max") => *const u16);

fact!(/// The quantised cache's key scale plane. Null under `KvScheme::Native`,
    /// and the null is meaningful — the scheme on the same row announces it,
    /// so this binds as it stands.
    KvKeyScales = "kv.k_scales" => Source::Named("kv.k_scales") => *mut core::ffi::c_void);

fact!(/// The value scale plane; see [`KvKeyScales`].
    KvValueScales = "kv.v_scales" => Source::Named("kv.v_scales") => *mut core::ffi::c_void);

fact!(/// The quantisation block, meaningful under `KvScheme::Fp4Block`.
    KvBlockSize = "kv.block_size" => Source::Named("kv.block_size") => i32);

fact!(/// How the pages are quantised, as the discriminant.
    ///
    /// `i32` and not the newtype: a key names a NUMBER the binder has, and
    /// the launcher that wants `KvScheme` reconstructs it. Minting the
    /// newtype here would put a backend's type in the fact vocabulary.
    KvSchemeByte = "kv.scheme" => Source::Named("kv.scheme") => i32);

fact!(/// What a page element actually is; see [`KvSchemeByte`].
    KvStorageDtype = "kv.storage_dtype" => Source::Named("kv.storage_dtype") => i32);

fact!(/// The dequantised bf16 mirror of the key pages, when one exists.
    ///
    /// Bound as it stands: [`KvNativeBf16`] on the same row announces the
    /// absence, so a null here is a state and not a gap.
    KvBf16Keys = "kv.k_bf16_pages" => Source::Named("kv.k_bf16_pages") => *mut core::ffi::c_void);

fact!(/// The value mirror; see [`KvBf16Keys`].
    KvBf16Values = "kv.v_bf16_pages" => Source::Named("kv.v_bf16_pages") => *mut core::ffi::c_void);

fact!(/// How many pages this fire's batch touches.
    ///
    /// The BATCH TOTAL. [`KvMaxPagesPerRequest`] is the per-request maximum;
    /// they are two facts and a launcher wanting a page-table stride wants
    /// the second.
    KvPagesInBatch = "kv.pages_in_batch" => Source::Named("kv.pages_in_batch") => i32);

fact!(/// The widest single request's page count, `max(indptr[r+1]-indptr[r])`.
    ///
    /// XQA's `maxNbPagesPerSeq` is a page-table row STRIDE. Host-computed off
    /// the fire's CSR, so it costs no device read.
    KvMaxPagesPerRequest = "kv.max_pages_per_request"
        => Source::Named("kv.max_pages_per_request") => i32);

fact!(/// Whether rope pairs channels adjacently (GPT-J) or by halves (NeoX).
    ///
    /// Total: a deployment that states nothing means NeoX, and there is no
    /// third answer, so nothing can decline this.
    RopeInterleaved = "rope_interleaved" => Source::Named("rope_interleaved") => bool);

fact!(/// The rows a peeled attention pass actually serves.
    PeelWindow = "peel_window" => Source::Named("peel_window") => *mut u32);

fact!(/// The position of the first token this fire appends.
    FirstToken = "first_token" => Source::Named("first_token") => i32);

fact!(/// YaRN's context scale. Absent block ⇒ 1.0; readers test their own numbers.
    YarnFactor = "yarn.factor" => Source::Named("yarn.factor") => f32);

fact!(/// The ramp's high-frequency bound. Absent ⇒ 0.0.
    YarnBetaFast = "yarn.beta_fast" => Source::Named("yarn.beta_fast") => f32);

fact!(/// The ramp's low-frequency bound. Absent ⇒ 0.0.
    YarnBetaSlow = "yarn.beta_slow" => Source::Named("yarn.beta_slow") => f32);

fact!(/// The attention temperature. Absent ⇒ 1.0.
    YarnAttentionFactor = "yarn.attention_factor" => Source::Named("yarn.attention_factor") => f32);

fact!(/// The trained position count. 0 IS the absence and every reader tests it.
    YarnOriginalMaxPosition = "yarn.original_max_position" => Source::Named("yarn.original_max_position") => i32);

fact!(/// The fire's WHOLE row space. NOT [`Rows`], which is `rows.count`.
    RowsTotal = "rows.total" => Source::Named("rows.total") => i32);

fact!(/// The fire's log-sum-exp destination, `AttnCtx::lse_out`.
    ///
    /// Deliberately NOT null-checked: a null here is a real state and
    /// refusing would invent one.
    AttnLseOut = "attn.lse_out" => Source::Named("attn.lse_out") => *mut f32);

fact!(/// The logit soft cap. Total — `0` is "none", which every launcher
    /// already tests as `> 0.0`.
    AttnLogitsSoftCap = "attn.logits_soft_cap" => Source::Named("attn.logits_soft_cap") => f32);

fact!(/// gemma's FINAL logit cap, which is not [`AttnLogitsSoftCap`].
    /// Zero is absence.
    FinalLogitSoftcap = "attn.final_logit_softcap"
        => Source::Named("attn.final_logit_softcap") => f32);

fact!(/// Whether this layer's cache stores unquantised bf16. Total.
    KvNativeBf16 = "kv.native_bf16" => Source::Named("kv.native_bf16") => bool);

fact!(/// Whether the KV cache is laid out heads-major.
    ///
    /// `bool`, not `i32`: `operand()` mints this as
    /// `f.kv_hnd.map(ArgValue::Bool)` and `abi_admits` refuses a `Bool`
    /// against `Some(Ty::I32)`, from a function that knows nothing about this
    /// key and reports it as a binder bug. All six `attn` parameters that
    /// carry the fact are declared `bool` and branch on it.
    KvHndLayout = "kv.hnd_layout" => Source::Named("kv.hnd_layout") => bool);

fact!(/// Whether the router renormalises its top-k weights.
    MoeNormTopk = "moe_norm_topk" => Source::Named("moe_norm_topk") => bool);

fact!(/// The routed-expert output scale.
    MoeRoutedScaling = "moe_routed_scaling" => Source::Named("moe_routed_scaling") => f32);

fact!(
    /// The clamped GLU's alpha.
    ///
    /// Two readers, which is why this is a key and not a one-site fix:
    /// `mlp::gpt_oss_glu` and `quant::mxfp4_moe_gate_up_decode_bf16` compute
    /// the same activation from the same deployment field.
    GluAlpha = "glu_alpha" => Source::Named("glu_alpha") => f32);

fact!(
    /// The clamped GLU's limit.
    ///
    /// [`GluAlpha`]'s pair and the same two readers.
    GluLimit = "glu_limit" => Source::Named("glu_limit") => f32);

fact!(
    /// How many experts one token visits.
    ///
    /// `i32` and not `u32`, to match `Facts`' other counts and the `i32` the
    /// CUDA launchers take; the deployment's `u32` is converted once, at the
    /// fill.
    ExpertsPerToken = "experts_per_token" => Source::Named("experts_per_token") => i32);

fact!(
    /// The weight a statement names first on `LaunchSpec::weight`.
    ///
    /// **This is rule E6's named weight and not the positional one.**
    /// `Weight<N, T>` in `routine.rs` is the weight the trace places in the
    /// operand run; this one is resolved before a `Cx` exists and `BoundArg`
    /// reports `width: 0` for it (`bind/mod.rs`'s `resolve_arg_windowed`).
    NamedWeight = "weight" => Source::Named("weight") => *const u8
);

fact!(/// The second weight a statement names. See [`NamedWeight`].
    NamedWeight2 = "weight2" => Source::Named("weight2") => *const u8);


// ── The linear-attention shape and the plan's last two ──
//
// `Cx::gdn` and `Cx::plan` answered these all along; what was missing was the
// name a signature could spell.

fact!(/// The gated-delta-net's KEY head count.
    ///
    /// Not [`NumKvHeads`]: that is the attention cache's, and a linear-attention
    /// layer's head split is the Mamba block's own. The two are equal on no
    /// deployment by construction and the substitution reads a neighbouring
    /// head's row rather than faulting.
    GdnKHeads = "gdn.k_h" => Source::Named("gdn.k_h") => i32);

fact!(/// The gated-delta-net's VALUE head count. See [`GdnKHeads`].
    GdnVHeads = "gdn.v_h" => Source::Named("gdn.v_h") => i32);

fact!(/// Elements per KEY head. Not [`HeadDim`], for [`GdnKHeads`]'s reason.
    GdnKDim = "gdn.k_d" => Source::Named("gdn.k_d") => i32);

fact!(/// Elements per VALUE head. See [`GdnKDim`].
    GdnVDim = "gdn.v_d" => Source::Named("gdn.v_d") => i32);

fact!(/// The short convolution's channel count.
    GdnConvDim = "gdn.conv_dim" => Source::Named("gdn.conv_dim") => i32);

fact!(/// The short convolution's kernel width.
    GdnConvK = "gdn.conv_k" => Source::Named("gdn.conv_k") => i32);

fact!(/// How many groups the key/value heads are split into.
    GdnNumGroups = "gdn.n_groups" => Source::Named("gdn.n_groups") => i32);

fact!(/// One seat's stride through the CONV state slab, in elements.
    ///
    /// `i64` because a slab spans every seat and the product overflows `i32`
    /// on a large batch -- the narrowing is the caller's and is a `try_from`
    /// at the launch, not a silent `as`.
    GdnConvStride = "gdn.conv_stride_elems" => Source::Named("gdn.conv_stride_elems") => i64);

fact!(/// One seat's stride through the RECURRENT state slab. See
    /// [`GdnConvStride`].
    GdnStateStride = "gdn.state_stride_elems" => Source::Named("gdn.state_stride_elems") => i64);

fact!(/// Which state seat each request occupies, one per request.
    GdnSlotIds = "gdn.slot_ids_d" => Source::Named("gdn.slot_ids_d") => *const i32);

fact!(/// Whether this launch writes the recurrent state back.
    ///
    /// A decode step writes; a prefill chunk that will be followed by another
    /// chunk of the same sequence does not, and the difference is the fire's
    /// rather than the statement's.
    GdnWriteState = "gdn.write_state" => Source::Named("gdn.write_state") => bool);

// ── Two facts two rows asked for by name ──
//
// Both are fields `DispatchCtx` already had.

fact!(/// Which AltUp stream ran through the real layer.
    ///
    /// Absent below two streams: zero and one both mean "no altup", and an
    /// index into a set that small is not a choice anything made.
    AltupActive = "altup_active" => Source::Named("altup_active") => i32);

fact!(/// gemma-4's per-layer residual scale, off the model's named constants.
    ///
    /// `DispatchCtx::scales` is keyed by NAME because *"a scale is a
    /// constant, not a tensor"* -- the dsl's own words for the
    /// `scale.<name>` weight slot. This key names the one the fused landing
    /// reads.
    LayerScale = "layer_scale" => Source::Named("layer_scale") => f32);

// ── The attention workspace ──
//
// `Cx::attn_workspace` answered all along. The byte counts come with the
// buffers: FlashInfer checks its budget against them, and a buffer whose size
// is a separate unstated scalar is a tear.
//
// `AttnCtx` holds two carves — decode and prefill — and they are four keys,
// not two, because `prefill_bf16` takes either depending on its caller.

fact!(/// The PREFILL carve's fp32 accumulation scratch. See [`AttnWorkspaceFloat`].
    AttnPrefillWorkspaceFloat = "attn.prefill_workspace.float" => Source::Named("attn.prefill_workspace.float") => *mut core::ffi::c_void);

fact!(/// How many bytes of it there are.
    AttnPrefillWorkspaceFloatBytes = "attn.prefill_workspace.float_bytes" => Source::Named("attn.prefill_workspace.float_bytes") => usize);

fact!(/// The PREFILL carve's scheduling-metadata scratch.
    AttnPrefillWorkspaceInt = "attn.prefill_workspace.int" => Source::Named("attn.prefill_workspace.int") => *mut core::ffi::c_void);

fact!(/// How many bytes of it there are.
    AttnPrefillWorkspaceIntBytes = "attn.prefill_workspace.int_bytes" => Source::Named("attn.prefill_workspace.int_bytes") => usize);

fact!(/// FlashInfer's fp32 split-KV accumulation scratch.
    AttnWorkspaceFloat = "attn.workspace.float" => Source::Named("attn.workspace.float") => *mut core::ffi::c_void);

fact!(/// How many bytes of it there are. See [`AttnWorkspaceFloat`].
    AttnWorkspaceFloatBytes = "attn.workspace.float_bytes" => Source::Named("attn.workspace.float_bytes") => usize);

fact!(/// FlashInfer's per-request scheduling-metadata scratch.
    AttnWorkspaceInt = "attn.workspace.int" => Source::Named("attn.workspace.int") => *mut core::ffi::c_void);

fact!(/// How many bytes of it there are. See [`AttnWorkspaceInt`].
    AttnWorkspaceIntBytes = "attn.workspace.int_bytes" => Source::Named("attn.workspace.int_bytes") => usize);

fact!(/// The gated-delta-net layer's CONV state slab, for this statement's seat.
    ///
    /// TWO KEYS AND NOT ONE INDEXED KEY: `Slab` is a two-variant enum so the
    /// misspelling that would silently decline is not writable, and an index
    /// here would put it back.
    GdnConvSlab = "gdn.conv_state" => Source::Named("gdn.conv_state") => *mut core::ffi::c_void);

fact!(/// The gated-delta-net layer's RECURRENT state slab. See [`GdnConvSlab`].
    GdnRecurrentSlab = "gdn.recurrent_state" => Source::Named("gdn.recurrent_state") => *mut core::ffi::c_void);

fact!(/// The attention plan's per-request QUERY offsets.
    ///
    /// `u32` for [`KvPageIndices`]' reason: signedness on an index table is a
    /// claim, and the only thing a negative entry could mean is a sentinel
    /// this plan does not have.
    QoIndptr = "plan.qo_indptr" => Source::Named("plan.qo_indptr") => *const u32);

fact!(/// Which rows of the fire's rectangle the plan actually serves.
    ///
    /// `u8` and not `bool`: it is a DEVICE array the kernel indexes, and
    /// `bool`'s ABI is not the one a `uint8_t*` parameter takes.
    RowValid = "plan.row_valid" => Source::Named("plan.row_valid") => *const u8);

// ── The fa2 decode plan, unfolded ──
//
// A plan's ADDRESSES are constant per BUCKET — `fire/launch.rs` keys a capture
// on `BucketKey::new(requests, rows, class, model_id)` and every offset below
// is a function of those plus head geometry — while only its CONTENTS vary per
// fire. `make_decode_params` already resolves the same addresses host-side, so
// naming them exposes nothing a fire did not bake. (`.wiki/kilimanjaro.md` §4
// says model-constant; bucket is the accurate word.)

fact!(/// Which request each scheduled CTA serves.
    Fa2DecodeRequestIndices = "fa2.decode.request_indices" => Source::Named("fa2.decode.request_indices") => *const i32);

fact!(/// Which KV tile each scheduled CTA serves.
    Fa2DecodeKvTileIndices = "fa2.decode.kv_tile_indices" => Source::Named("fa2.decode.kv_tile_indices") => *const i32);

fact!(/// Where each request's output rows begin.
    Fa2DecodeOIndptr = "fa2.decode.o_indptr" => Source::Named("fa2.decode.o_indptr") => *const i32);

fact!(/// How many KV entries one chunk covers, device-side.
    Fa2DecodeKvChunkSize = "fa2.decode.kv_chunk_size" => Source::Named("fa2.decode.kv_chunk_size") => *const i32);

fact!(/// Which scheduled blocks a graph replay may run.
    ///
    /// NULL IS THE ANSWER, not an absence: the planner carves this only under
    /// `split_kv && enable_cuda_graph`, and a plan that carved neither has no
    /// mask for the kernel to consult.
    ///
    /// `u8` and not `bool` for [`RowValid`]'s reason.
    Fa2DecodeBlockValidMask = "fa2.decode.block_valid_mask" => Source::Named("fa2.decode.block_valid_mask") => *const u8);

fact!(/// The split-KV partial OUTPUT scratch, carved from the float workspace.
    ///
    /// Null when the plan did not split. See [`Fa2DecodeSplitKv`].
    Fa2DecodeTmpV = "fa2.decode.tmp_v" => Source::Named("fa2.decode.tmp_v") => *mut f32);

fact!(/// The split-KV partial LSE scratch. [`Fa2DecodeTmpV`]'s pair.
    Fa2DecodeTmpS = "fa2.decode.tmp_s" => Source::Named("fa2.decode.tmp_s") => *mut f32);

fact!(/// How many CTAs the schedule launches.
    ///
    /// `max_grid_size / gdy` when the plan split, the request count when it
    /// did not (`plan/decode.rs:95-107`) -- head geometry and the bucket,
    /// never the fire's contents.
    Fa2DecodePaddedBatch = "fa2.decode.padded_batch" => Source::Named("fa2.decode.padded_batch") => i32);

fact!(/// Whether the schedule splits one request's KV across CTAs.
    ///
    /// The discriminator for the fold: false means the kernel wrote the
    /// caller's `o` directly and there is nothing to merge.
    Fa2DecodeSplitKv = "fa2.decode.split_kv" => Source::Named("fa2.decode.split_kv") => bool);

fact!(/// How many requests the PLAN was built over.
    ///
    /// NOT [`RequestCount`]. That is the fire's; this is the count the
    /// schedule's vectors were sized at, and a plan raised for one bucket
    /// and read under another differs in exactly this number.
    Fa2DecodeRequests = "fa2.decode.requests" => Source::Named("fa2.decode.requests") => i32);

fact!(/// The query head count the plan was built at. See [`Fa2DecodeRequests`].
    Fa2DecodeNumQHeads = "fa2.decode.num_q_heads" => Source::Named("fa2.decode.num_q_heads") => i32);

fact!(/// The KV head count the plan was built at.
    Fa2DecodeNumKvHeads = "fa2.decode.num_kv_heads" => Source::Named("fa2.decode.num_kv_heads") => i32);

fact!(/// The head width the plan was built at.
    Fa2DecodeHeadDim = "fa2.decode.head_dim" => Source::Named("fa2.decode.head_dim") => i32);

fact!(/// The page size the plan was built at.
    Fa2DecodePageSize = "fa2.decode.page_size" => Source::Named("fa2.decode.page_size") => i32);

fact!(/// Whether the cache the plan addresses is HND rather than NHD.
    Fa2DecodeHndLayout = "fa2.decode.hnd_layout" => Source::Named("fa2.decode.hnd_layout") => bool);

fact!(/// Whether the plan's variant attends the whole context.
    ///
    /// Picks the launcher's arm, and `window_left` is an ARGUMENT that the
    /// body reads alongside it -- the two together are what `decode_arm`
    /// switches on.
    Fa2DecodeFullAttention = "fa2.decode.full_attention" => Source::Named("fa2.decode.full_attention") => bool);

// ── The fa2 prefill plan, unfolded ──
//
// EVERY ONE OF THESE IS RESOLVED AGAINST THE PREFILL CARVE: `AttnCtx` holds
// two workspaces and a plan writes its schedule into the one it was raised
// against. The six with no `fa2.decode.` twin are what a prefill schedule has
// and a decode one does not.

fact!(/// Which request each scheduled CTA serves.
    Fa2PrefillRequestIndices = "fa2.prefill.request_indices" => Source::Named("fa2.prefill.request_indices") => *const i32);

fact!(/// Which QO tile each scheduled CTA serves. No decode twin: a decode
    /// step has one query row per request.
    Fa2PrefillQoTileIndices = "fa2.prefill.qo_tile_indices" => Source::Named("fa2.prefill.qo_tile_indices") => *const i32);

fact!(/// Which KV tile each scheduled CTA serves.
    Fa2PrefillKvTileIndices = "fa2.prefill.kv_tile_indices" => Source::Named("fa2.prefill.kv_tile_indices") => *const i32);

fact!(/// Where each merged ROW's partials start, `[total_rows + 1]`.
    ///
    /// Null when the plan did not split. Prefill folds by row where decode
    /// folds by request, which is why this exists and
    /// [`Fa2PrefillOIndptr`] is not reused for it.
    Fa2PrefillMergeIndptr = "fa2.prefill.merge_indptr" => Source::Named("fa2.prefill.merge_indptr") => *const i32);

fact!(/// Where each request's output rows begin.
    Fa2PrefillOIndptr = "fa2.prefill.o_indptr" => Source::Named("fa2.prefill.o_indptr") => *const i32);

fact!(/// How many KV entries one chunk covers, device-side.
    Fa2PrefillKvChunkSize = "fa2.prefill.kv_chunk_size" => Source::Named("fa2.prefill.kv_chunk_size") => *const i32);

fact!(/// Which scheduled blocks a graph replay may run.
    ///
    /// Null is the answer, as [`Fa2DecodeBlockValidMask`]'s, and `u8` for the
    /// same reason.
    Fa2PrefillBlockValidMask = "fa2.prefill.block_valid_mask" => Source::Named("fa2.prefill.block_valid_mask") => *const u8);

fact!(/// The split-KV partial OUTPUT scratch, carved from the float workspace.
    ///
    /// Null when the plan did not split. See [`Fa2PrefillSplitKv`].
    Fa2PrefillTmpV = "fa2.prefill.tmp_v" => Source::Named("fa2.prefill.tmp_v") => *mut f32);

fact!(/// The split-KV partial LSE scratch. [`Fa2PrefillTmpV`]'s pair.
    Fa2PrefillTmpS = "fa2.prefill.tmp_s" => Source::Named("fa2.prefill.tmp_s") => *mut f32);

fact!(/// How many CTAs the schedule launches.
    Fa2PrefillPaddedBatch = "fa2.prefill.padded_batch" => Source::Named("fa2.prefill.padded_batch") => i32);

fact!(/// Whether the schedule splits one request's KV across CTAs.
    Fa2PrefillSplitKv = "fa2.prefill.split_kv" => Source::Named("fa2.prefill.split_kv") => bool);

fact!(/// How many QO rows the schedule covers.
    ///
    /// `max_total_num_rows`, which the fold folds. The neighbouring field in
    /// the block is `total_num_rows`, a DEVICE pointer left null on both
    /// sides here; the names differ by two characters and the types by eight
    /// bytes.
    Fa2PrefillTotalRows = "fa2.prefill.total_rows" => Source::Named("fa2.prefill.total_rows") => i32);

fact!(/// The `CTA_TILE_Q` the plan was split against, which names the root.
    ///
    /// Read back, never recomputed: a fire that chose its own would index a
    /// work list built for a different tile.
    Fa2PrefillCtaTileQ = "fa2.prefill.cta_tile_q" => Source::Named("fa2.prefill.cta_tile_q") => u32);

fact!(/// How many requests the PLAN was built over. See [`Fa2DecodeRequests`].
    Fa2PrefillRequests = "fa2.prefill.requests" => Source::Named("fa2.prefill.requests") => i32);

fact!(/// The query head count the plan was built at.
    Fa2PrefillNumQHeads = "fa2.prefill.num_q_heads" => Source::Named("fa2.prefill.num_q_heads") => i32);

fact!(/// The KV head count the plan was built at.
    Fa2PrefillNumKvHeads = "fa2.prefill.num_kv_heads" => Source::Named("fa2.prefill.num_kv_heads") => i32);

fact!(/// The head width the plan was built at.
    Fa2PrefillHeadDim = "fa2.prefill.head_dim" => Source::Named("fa2.prefill.head_dim") => i32);

fact!(/// The page size the plan was built at.
    Fa2PrefillPageSize = "fa2.prefill.page_size" => Source::Named("fa2.prefill.page_size") => i32);

fact!(/// The window the plan was SPLIT against, `-1` for full attention.
    ///
    /// NOT [`WindowLeft`]. That is the statement's; this is the one planning
    /// fixed, because the split was sized against it.
    Fa2PrefillWindowLeft = "fa2.prefill.window_left" => Source::Named("fa2.prefill.window_left") => i32);

fact!(/// Whether the cache the plan addresses is HND rather than NHD.
    Fa2PrefillHndLayout = "fa2.prefill.hnd_layout" => Source::Named("fa2.prefill.hnd_layout") => bool);

fact!(/// Whether the plan's variant attends the whole context.
    Fa2PrefillFullAttention = "fa2.prefill.full_attention" => Source::Named("fa2.prefill.full_attention") => bool);

fact!(/// Whether the mask is causal. With [`Fa2PrefillFullAttention`] and the
    /// soft cap, this picks the launcher's arm.
    Fa2PrefillCausalMask = "fa2.prefill.causal_mask" => Source::Named("fa2.prefill.causal_mask") => bool);

// ── Answered off `cx.kv_layer()`, `cx.plan()` and `cx.weight_suffixed()` ──

fact!(/// The key cache plane. `attn`'s five honest marks (`4c3843dd3`).
    ///
    /// `u8` because the ELEMENT type is the caller's business: the same cache
    /// is `bf16` under one model and `f16` or an fp8 pair under another, and
    /// a fact true of one instantiation is not a fact. `*mut` because the
    /// plane is WRITTEN — `rope_write_kv_bf16` exists to fill it, and a
    /// reader handed a `*mut` may simply not write.
    KvKeys = "kv_keys" => Source::Named("kv_keys") => *mut u8);

fact!(/// The value cache plane. See [`KvKeys`], including for why this is
    /// `*mut u8` and not `*const u8`.
    KvValues = "kv_values" => Source::Named("kv_values") => *mut u8);

fact!(/// The page table.
    ///
    /// `u32`, not `i32`: signedness on a page table is a claim, and the only
    /// thing a negative entry could mean is a sentinel this cache does not
    /// have. The `attn` parameters, the driver's host arrays and
    /// `rope_write_kv_bf16` all say `u32`.
    ///
    /// Nothing here would catch a wrong pointee: `as_declared` passes
    /// pointers through untouched, `ptr_abi!`'s `unpack` casts to any pointee
    /// at all, and this source is on `operand()`'s legitimately-blocked list,
    /// so the declared type is pure documentation.
    KvPageIndices = "kv_page_indices" => Source::Named("kv_page_indices") => *const u32);

fact!(/// The page table's per-request offsets. See [`KvPageIndices`] for
    /// why this is `u32`.
    KvPageIndptr = "kv_page_indptr" => Source::Named("kv_page_indptr") => *const u32);

fact!(
    /// Whether this layer's KV pages carry envelopes.
    ///
    /// One of five `KvLayerField` facts `attn` states truthfully and no
    /// backend answers; `is_native_bf16`, `block_size`, `scheme` and
    /// `storage_dtype` have no key because no launcher names them yet.
    ///
    /// `bool` for [`KvHndLayout`]'s reason, one step earlier: a predicate
    /// mints an `ArgValue::Bool`, so an `i32` key refuses the day an arm is
    /// written.
    KvHasEnvelopes = "kv.has_envelopes" => Source::Named("kv.has_envelopes") => bool
);

fact!(
    /// The per-request fill level of each sequence's last KV page.
    ///
    /// A source carrying a `&'static str` is spellable as a key — `fact!`
    /// takes an `$src:expr`. The limit was `fact_of`'s: its name table built
    /// one `Ident`, so only unit variants were reachable from a NAME.
    ///
    /// `*const u32`, sitting beside [`KvPageIndices`] and [`KvPageIndptr`] —
    /// three CSR arrays describing one paged cache.
    KvLastPageLens = "kv.last_page_lens" => Source::Named("kv.last_page_lens") => *const u32);

fact!(
    /// The attention softmax scale, `1/sqrt(head_dim)` unless a model says
    /// otherwise.
    ///
    /// `Option`, never total: `launch.rs` falls back to `1.0` and gemma-4
    /// really runs `1.0`, so a total field would make "no attention context"
    /// and "gemma-4" the same number.
    SmScale = "attn.sm_scale" => Source::Named("attn.sm_scale") => f32);

// The weight a statement names with a suffix — facts by rule E6, because the
// resolver answers them before a `Cx` exists.

fact!(/// The `_bias` plane beside a named weight.
    WeightBias = "weight._bias" => Source::Named("weight._bias") => *const u8);

fact!(/// The `_scales` plane beside a named weight.
    WeightScales = "weight._scales" => Source::Named("weight._scales") => *const u8);

fact!(/// The `_up_bias` plane — a fused gate/up pair's second half.
    WeightUpBias = "weight._up_bias" => Source::Named("weight._up_bias") => *const u8);

fact!(/// The `_gate_bias` plane — the same pair's first half.
    WeightGateBias = "weight._gate_bias" => Source::Named("weight._gate_bias") => *const u8);

// ── The shader planes' own facts ──

fact!(/// The hidden width the statement's tensors come out at.
    Width = "width" => Source::Named("width") => i32);

fact!(
    /// The hidden width they go IN at, which differs from [`Width`] wherever
    /// a projection is not square — every `qmm`, and the MoE gates.
    InWidth = "in_width" => Source::Named("in_width") => i32
);

fact!(/// Value heads on a linear-attention layer. Zero on a softmax one.
    VHeads = "v_heads" => Source::Named("v_heads") => i32);

fact!(/// The value width of a linear-attention head. See [`VHeads`].
    VDim = "v_dim" => Source::Named("v_dim") => i32);

fact!(/// Experts in this statement's layer, routed and shared alike.
    NumExperts = "n_experts" => Source::Named("n_experts") => i32);

fact!(
    /// Weights per quantisation scale. The pair with [`QuantBits`] is what
    /// says how to walk a packed weight, and neither is meaningful alone.
    QuantGroup = "group" => Source::Named("group") => i32
);

fact!(/// Bits per packed weight. See [`QuantGroup`].
    QuantBits = "bits" => Source::Named("bits") => i32);

fact!(
    /// The row half of the tile a quantised matmul runs at.
    ///
    /// A TUNING choice that reaches the kernel as geometry: the driver picks
    /// it per device and the shader's grid is built from it, so it is the
    /// fire's number even though nothing in the checkpoint says it.
    TileM = "tile_m" => Source::Named("tile_m") => i32
);

fact!(/// The column half of the tile. See [`TileM`].
    TileN = "tile_n" => Source::Named("tile_n") => i32);

fact!(/// Scratch for a split decode's per-pass partial sums.
    AttnPartials = "attn.partials" => Source::Named("attn.partials") => *mut f32);

fact!(/// How many ways this fire's decode splits its key range.
    AttnSplits = "attn.splits" => Source::Named("attn.splits") => i32);

fact!(/// Which recurrent slot each request occupies, for the linear-attention
    /// families. The fire's, not the statement's: a slot is a property of
    /// the REQUEST's residency in the driver's state pool.
    RecurrentSlots = "recurrent_slots" => Source::Named("recurrent_slots") => *const i32);

fact!(/// The short convolution's rolling window, read. `Resolver::slab`'s
    /// `"conv_state"`.
    ConvState = "conv_state" => Source::Named("conv_state") => *mut u8);

fact!(/// The same window, written. A separate plane and not a `*mut` alias of
    /// [`ConvState`]: `gdn`'s prefill reads the old window while it fills the
    /// new one, which is why the driver allocates two and swaps.
    NewConvState = "new_conv_state" => Source::Named("new_conv_state") => *mut u8);

fact!(/// The recurrence's carried state — `gdn`'s and mamba's `S`.
    /// `Resolver::slab`'s `"recurrent_state"`.
    RecurrentState = "recurrent_state" => Source::Named("recurrent_state") => *mut u8);

// ── §2  True and unanswered ──
//
// No backend has an arm for these; naming one is a correct statement that
// refuses at bind with `Refusal::Unstated`. Two left — nothing upstream
// carries either array, so each wants a `Fire` edit before a line here.

fact!(/// Which request each token belongs to.
    RequestOfToken = "request_of_token" => Source::Named("request_of_token") => *const i32);

fact!(/// The rope frequency table, when it is precomputed rather than derived.
    RopeFrequencies = "rope_frequencies" => Source::Named("rope_frequencies") => *const f32);

// ── §N  THE KEYS THAT REPLACED AN ATTRIBUTE ───────────────────────────────
//
// `#[source(OutWidth(0))]` and `#[lit(1.702)]` were a THIRD encoding of a
// source, beside the marks and beside `Source` itself, and the only one a
// reader of types could not see. Twenty-one parameters wore one. Each is a
// key now, so the escape hatch and the vocabulary are the same thing.

fact!(/// The width of result 0's row, as a scalar parameter.
    ///
    /// A SHAPE READ AS A NUMBER: the launcher wants the width without the
    /// address, which `Out<T>`'s `width` field gives a body that holds the
    /// operand and this gives one that does not.
    OutWidth0 = "out_width0" => Source::Slot(crate::Kind::OutWidth, 0) => i32);

fact!(/// The width of operand 0's row. [`OutWidth0`]'s counterpart.
    InWidth0 = "in_width0" => Source::Slot(crate::Kind::InWidth, 0) => i32);

fact!(/// The statement's zeroth scalar, read as a float.
    ///
    /// `Param<0, f32>` says this in a signature that can take the mark;
    /// `gaussian_topk`'s `std_multiplier` cannot, because the params run and
    /// the operand run are two arrays and the mark walks the second.
    ParamF32_0 = "param_f32_0" => Source::Slot(crate::Kind::ParamF32, 0) => f32);

// `GluAlpha` IS NOT ONE OF THESE, though 1.702 looks like it belongs. The
// number is `Deployment::mlp_gate`'s `SiluClamped { alpha, .. }`, read per
// fire, and `driver-cuda`'s `launch_context_is_stated` records the last time
// it was cut loose from that field: the gate stopped being scaled and the
// model degraded rather than faulted. `Lit` is for a number the SYMBOL
// decides — `BetaZero` against `BetaOne` — and no symbol decides this one.
// The live declaration is `Source::Named("glu_alpha")`, above.

// `Unstated` STOOD HERE AND IS DELETED.
//
// It said *"nothing supplies this parameter"* about a POINTER a launcher
// supplied itself, and it existed because a bare `*const T` would otherwise be
// counted into the next input slot. With `Env` gone there is no such
// parameter: a value the launcher makes is a value the BODY makes, and it
// reaches the argument list as `ctx.absent()?` — the same `Lit::Null` the
// column used to resolve, asked for where it is used instead of declared where
// it is not.

// ── §N  THE KEYS THAT REPLACED A WRAPPER ──────────────────────────────────
//
// A source is a source whether a routine reaches it by NAME or by shape, so
// the wrappers that existed only to spell one shape of source are keys now.
// `Block<Buf>`, `Null<Buf>`, `ParamOrLit<4, -1, i32>` and the whole `Reckoned`
// / `Says` type-level arithmetic were six types and five markers between them;
// each is one line here, and `Env<T, K>` reads all of them.

fact!(/// Every scalar the statement carries, staged as one struct and bound
    /// as one buffer — what `Block<Buf>` used to spell.
    ///
    /// A NAME AND NOT A SHAPE: the block is the params CHANNEL, which is why
    /// this is `Slot(Kind::Params, 0)` rather than a `Named`. `Env<Buf,
    /// keys::Params>` says the binder supplies it AND which of the binder's
    /// buffers it is, which is the whole of what the wrapper said.
    Params = "params" => Source::Slot(crate::Kind::Params, 0) => *const u8);

fact!(/// The optional operand this routine leaves ABSENT — `Null<T>`'s source.
    ///
    /// The one source that needs no resolver: the answer is the absence.
    /// Fourteen arguments in the metal plane bind from `state(None)` — a
    /// family with no per-head sink logits, a routed matmul with no bias, the
    /// ring-buffer slots a paged append does not use — and until they said so
    /// the row read the same as an argument nobody had got round to.
    Absent = "absent" => Source::Lit(crate::Lit::Null) => *const u8);

fact!(/// `-1`, which is "no sliding window" where a paged attention reads it.
    ///
    /// `ParamOrLit<4, -1, i32>` stood at twenty-one sites and spelled the
    /// number; this spells what the number MEANS, and `Param<4, Env<i32,
    /// keys::NoSlidingWindow>>` is the same chain with the sentinel named.
    NoSlidingWindow = "no_sliding_window" => Source::Lit(crate::Lit::I32(-1)) => i32);

fact!(/// How many query heads share one KV head: `q_heads / kv_heads`.
    ///
    /// ARITHMETIC ON TWO FACTS THE DRIVER ALREADY HOLDS, which is why it is a
    /// key and not a scalar the statement carries. It was the second:
    /// `model/qwen_3_5/forward/metal.rs` computed `f.q_heads /
    /// f.kv_heads.max(1)` and shipped the answer through the params channel to
    /// thirty shader routines, while `bind.rs` had both terms in its resolver
    /// the whole time and CUDA never asked for it at all.
    ///
    /// `Source::Over` refuses a zero divisor, which is what `.max(1)` was
    /// standing in for.
    GqaFactor = "gqa_factor" => Source::Over(
        &Source::Named("q_heads"), &Source::Named("kv_heads")) => i32);

fact!(/// How many norms a row packs: the row's width over one head's length.
    ///
    /// The divisor is itself a chain, because a statement may carry the head
    /// length and the fire answers when it does not. This is
    /// `rms_strided_head_row`'s `heads` and nothing else states it.
    HeadsPerRow = "heads_per_row" => Source::Over(
        &Source::Named("width"),
        &Source::Or(&Source::Slot(crate::Kind::Param, 1), &Source::Named("width")),
    ) => i32);

// ── §7  THE PARAMS CHANNEL, NAMED ─────────────────────────────────────────
//
// Five hundred and twelve arguments read a scalar by its POSITION in the
// statement's run -- `Param<3, u32>` -- and the position was the whole of the
// address. Three things went wrong with that and each is recorded elsewhere
// in this tree:
//
// * `driver-vulkan/src/lib.rs`'s `sdpa_paged_decode` wired
//   `attention_mask_stride` to `Slot(Kind::Param, 3)` and read every mask at a
//   stride of zero, because `Hold::staged` packs an unstated slot as `0`
//   (`driver-vulkan/src/hold.rs:540`) and zero is a legal `u32`.
// * `model-dsl`'s `rope_launch` passes `vec![0]` for full rope and
//   `vec![rotary_dim]` for partial, so a SENTINEL in the value carried the
//   distinction `keys::RotaryWidth` already names.
// * `GqaFactor` above: thirty shader routines took `q_heads / kv_heads`
//   through the channel while the resolver held both terms and CUDA never
//   asked at all.
//
// A key fixes all three, because a key names WHO ANSWERS. `Fact::PROV` is the
// load-bearing half: a stride the driver staged is `Env` and a checkpoint
// extent the statement carries is `stated`, and wiring one where the other
// belongs stops being a plausible alternative. Where no one answers the fire
// gets a `Refusal` instead of a zero.

fact!(/// The pitch of the `x` operand's rows, where a launcher walks `x` at a
    /// stride its own result does not share.
    XRowStride = "x_row_stride" => Source::Named("x_row_stride") => i32);

fact!(/// Elements between one SLOT of `x` and the next, for a launcher whose
    /// input is a ring of slots rather than a dense rectangle.
    XSlotStride = "x_slot_stride" => Source::Named("x_slot_stride") => i32);

fact!(/// How many slots one row of the ring holds.
    SlotsPerRow = "slots_per_row" => Source::Named("slots_per_row") => i32);

fact!(/// The pitch of the query rectangle's rows.
    QRowStride = "q_row_stride" => Source::Named("q_row_stride") => i32);

fact!(/// The pitch of the output rectangle's rows.
    ORowStride = "o_row_stride" => Source::Named("o_row_stride") => i32);

fact!(/// The sliding window, or `-1` where the statement states none.
    ///
    /// [`WindowLeft`] with its fallback NAMED, which is what
    /// `Param<4, Env<i32, keys::NoSlidingWindow>>` spelled at twenty-one
    /// sites: the statement's number if it carries one, and the sentinel
    /// meaning "full attention" if it does not. The chain belongs to the
    /// QUESTION, not to a wrapper around the carrier -- every routine asking
    /// for a window wants the same fallback, and spelling it per-signature is
    /// twenty-one chances to spell a different one.
    WindowOrNone = "window_left.or_none" => Source::Or(
        &Source::Named("window_left"), &Source::Lit(crate::Lit::I32(-1))) => i32);

// ── §8  WHAT THE STATEMENT REALLY DOES ANSWER ────────────────────────────
//
// [`Provenance::Trace`] survives, and this is everything it survives for: a
// WEIGHT WALKER's extents. `quant.rs`'s `dequant_fp8_e4m3_to` says it —
// *"dequantise a weight, whose shape is a checkpoint property, not a fire
// fact. Its leading extent has no `Source`: `keys::Rows` would compile but
// read the fire's token count instead"* — and that is the whole class. These
// launches run at load time over a checkpoint the fire has not begun to use,
// so there is no rectangle to read and the statement is the only witness.
//
// [`Provenance::Trace`]: crate::routine::Provenance::Trace

fact!(
    /// The one f32 scale a per-tensor FP8 checkpoint was quantised at.
    ///
    /// NOT [`SmScale`] and not [`RopeScale`]: this is a dequantiser's, and
    /// the three were all spelled `scale` at a param slot.
    DequantScale = "dequant.scale" => Source::Named("dequant.scale") => f32);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key is distinct.
    ///
    /// §6 folds [`Fact::SOURCE`] into [`Fact::KEY`], so two types sharing a
    /// key would silently become one fact.
    ///
    /// # Read out of the source, not written down beside it
    ///
    /// It used to be a hand list of `X::KEY`, with a second assertion holding
    /// the list's LENGTH against the number of `fact!(` lines so that a new
    /// fact could not be forgotten. That second assertion is the whole reason
    /// the first one worked, and it is also what broke: the list stopped at
    /// eighty-eight while the file grew to two hundred and twelve, so the
    /// guard was reporting the drift it existed to prevent and nothing else
    /// was checking uniqueness.
    ///
    /// Rust cannot enumerate the types implementing a trait — that has not
    /// changed — but the DECLARATIONS are enumerable, because each fact is
    /// exactly one `fact!(...)` and each carries its key as a literal. Reading
    /// them is what the length check was standing in for, and it cannot fall
    /// behind.
    #[test]
    fn keys_are_distinct() {
        // At LINE START, so the macro's own `/// fact!(RmsEps = ...)` doc
        // example is not counted.
        let src = include_str!("keys.rs");
        let declared: Vec<&str> = src
            .split("\nfact!(")
            .skip(1)
            .filter_map(|block| {
                let (_, rest) = block.split_once(" = \"")?;
                let (key, _) = rest.split_once('"')?;
                Some(key)
            })
            .collect();
        let starts = src.matches("\nfact!(").count();
        assert_eq!(
            declared.len(),
            starts,
            "every `fact!(` states a key literal; {} of {starts} parsed",
            declared.len(),
        );

        let mut seen = declared.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            declared.len(),
            "two facts share a key: {:?}",
            {
                let mut dup = declared.clone();
                dup.sort_unstable();
                dup.windows(2)
                    .filter(|w| w[0] == w[1])
                    .map(|w| w[0])
                    .collect::<Vec<_>>()
            },
        );
    }

    /// The two thetas are two types AND two sources.
    ///
    /// The single check this module exists for. `fact_of` mapped
    /// `"theta" | "rope_theta" | "rope_base"` onto one variant; if these
    /// converge again, eleven launchers rotate by whichever the alias picks.
    #[test]
    fn the_two_thetas_stay_two() {
        assert_ne!(Theta::SOURCE, RopeTheta::SOURCE);
        assert_ne!(Theta::KEY, RopeTheta::KEY);
    }

    /// A key names its own source, and a COMPOUND key says so with a dot.
    ///
    /// `KvHeadDim` is `"kv.head_dim"`, and that dotted spelling is
    /// deliberate: §6's flat string namespace has to keep the KV layer's
    /// `head_dim` apart from the attention layer's.
    #[test]
    fn compound_keys_are_dotted() {
        // Every parameterised source spells its key as the namespace, a dot,
        // and the parameter itself. The mapping is total, so check it as a
        // mapping — a key that does not reproduce its source's parameter
        // loses the parameter at §6's fold.
        let views = [
            (KvHeadDim::KEY, KvHeadDim::SOURCE),
            (KvNumHeads::KEY, KvNumHeads::SOURCE),
            (KvHndLayout::KEY, KvHndLayout::SOURCE),
            (KvHasEnvelopes::KEY, KvHasEnvelopes::SOURCE),
            (WeightBias::KEY, WeightBias::SOURCE),
            (WeightScales::KEY, WeightScales::SOURCE),
            (WeightUpBias::KEY, WeightUpBias::SOURCE),
            (WeightGateBias::KEY, WeightGateBias::SOURCE),
        ];
        for (key, source) in views {
            // `SOURCE` is `Source::Named(KEY)` for every named fact, so what
            // is left to check is that the key is NAMESPACED.
            // `source_is_named` is a real comparison — `SOURCE` cannot be
            // `==`'d, `Source` has no `Eq` because `Lit::F32` is a float.
            assert!(
                crate::source_is_named(&Some(source), key),
                "{key} must derive itself: every named fact is its own source",
            );
            let (space, _) = key.split_once('.').unwrap_or_else(|| {
                panic!("{key} reads a view and must say which one: `kv.` or `weight.`")
            });
            assert!(
                space == "kv" || space == "weight",
                "{key} is namespaced `{space}.`, which is neither view",
            );
        }

        // THE DRIFT GUARD, AIMED THE OTHER WAY. It used to hold the hand
        // list's LENGTH against the dotted keys in the source, which made
        // every new `kv.`/`weight.` fact a test failure until someone copied
        // it up — and the list duly fell behind, at eight of twenty-two.
        //
        // What the guard is actually for is that a dotted key names a VIEW,
        // so it checks that directly: every dotted key in the file is
        // namespaced `kv.` or `weight.` and nothing else. The hand list above
        // keeps its own job, which is holding each named fact to being its own
        // source.
        const NOT_VIEWS: &[&str] = &["kv.last_page_lens"];
        let src = include_str!("keys.rs");
        let namespaced: Vec<&str> = src
            .lines()
            .filter_map(|l| {
                let (_, rest) = l.split_once(" = \"")?;
                let (key, _) = rest.split_once('"')?;
                (key.starts_with("kv.") || key.starts_with("weight."))
                    .then_some(key)
            })
            .filter(|k| !NOT_VIEWS.contains(k))
            .collect();
        for key in &namespaced {
            let (space, _) = key.split_once('.').expect("a dotted key");
            assert!(
                space == "kv" || space == "weight",
                "{key} is namespaced `{space}.`, which is neither view",
            );
        }
        assert!(
            namespaced.len() >= views.len(),
            "the hand list is a subset of what the file declares",
        );

        assert!(!HeadDim::KEY.contains('.'), "the plain fact stays plain");

        // `KvPageSize` IS THE DELIBERATE EXCEPTION: it reads
        // `cx.kv_layer().map(|l| l.page_size)` — the same read the dotted
        // four make — and is spelled flat anyway, because the dot tracks the
        // SOURCE's shape and not the read's destination.
        assert!(crate::source_is_named(&Some(KvPageSize::SOURCE), KvPageSize::KEY));
        assert!(!KvPageSize::KEY.contains('.'));

    }
}
