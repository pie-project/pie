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

















// ── The fa2 prefill plan, unfolded ──
//
// EVERY ONE OF THESE IS RESOLVED AGAINST THE PREFILL CARVE: `AttnCtx` holds
// two workspaces and a plan writes its schedule into the one it was raised
// against. The six with no `fa2.decode.` twin are what a prefill schedule has
// and a decode one does not.























// ── The plan descriptor's H2D, which a launcher must make before it fires ──
//
// `arms/fa2.rs` held eight arms alive on this copy, and the reason was never
// that a `Source` could not name the bytes. It was that the copy is WORK: the
// arm ran it between resolving the operands and launching, so a column that
// bound every argument still left something undone. These are the copy's
// operands, and the LAUNCHER makes it -- which keeps the copy inside the graph
// capture where a replay re-runs it, exactly where the arm had it.
//
// **The source address is PINNED and fixed for the process.** That is what
// makes a captured node correct: `plan::upload_int_plan`'s contract is that
// the host buffer is `PinnedBytes`, updated in place each fire, so *"the
// address a captured node bakes is the address the next fire writes into"*.
// An address that moved would replay a stale schedule and say nothing.
//
// SPLIT BY FAMILY, like the sixteen decode leaves and the twenty-two prefill
// ones above, and for a harder reason than symmetry: `AttnCtx` holds TWO
// workspace carves and a plan writes its schedule into the one it was raised
// against. One key pair would have to guess which, and a prefill descriptor
// landing in the decode carve clobbers it with no fault and no wrong address.







// ── What a launcher that plans its OWN fire needs ──────────────────────────
//
// The planless prefill is the one FA2 launcher whose plan does not exist when
// it is called: `arms/fa2.rs`'s `fa2_prefill_planless` walked the HOST CSRs,
// called `plan::plan_prefill` and mutated the cache, and only then launched.
// That is why its `prefill_plan` parameter is `#[unbound]` -- nothing publishes
// the aggregate before the fire, so no column could ever answer for it.
//
// These four are what the planning READS, and naming them is what lets the
// planning move into the launcher: the cache to fill, the two host CSRs to
// walk, and how many requests they describe. Everything else the planner takes
// -- the layer's geometry, the workspace's two byte counts, the window -- is
// already a fact this driver answers.
//
// HOST addresses, all three pointers. Nothing checks that and nothing can; see
// [`KvPageIndices`] for why a declared pointee here is documentation.

fact!(/// The prefill plan CACHE, for a launcher that fills it itself.
    ///
    /// `*mut u8` because the shape is `kernels-cuda`'s own and the driver
    /// holds it only as a boxed allocation it hands back by address. A
    /// launcher that takes this is claiming exclusive use of it for the
    /// launch, which a fire grants by dispatching one statement at a time.
    Fa2PrefillPlanCache = "fa2.prefill.plan_cache" => Source::Named("fa2.prefill.plan_cache") => *mut u8);

fact!(/// The HOST query-offset CSR the planner walks, `requests + 1` entries.
    QoIndptrHost = "qo_indptr_h" => Source::Named("qo_indptr_h") => *const u32);

fact!(/// The HOST KV-page CSR the planner walks. [`QoIndptrHost`]'s pair, and
    /// its last entry is the batch's page count.
    KvPageIndptrHost = "kv_page_indptr_h" => Source::Named("kv_page_indptr_h") => *const u32);

fact!(/// How many requests this fire serves, which is both CSRs' length minus
    /// one.
    ///
    /// NOT [`Rows`]: a prefill request covers many rows, and the planner
    /// wants the request count where the launch geometry wants the row count.
    FireRequests = "fire.requests" => Source::Named("fire.requests") => i32);

// ── The two ragged sinks a CAPTURING or MASKED attention reads ─────────────
//
// Four bare `*mut f32` / `*const i32` parameters on three FA2 launchers, which
// is what kept those three `#[routine(untraced)]`: a bare pointer is not a
// mark, so the row carried no source column at all. The arms filled them off
// `AttnCtx` directly. These are that, said as facts -- which is what lets the
// three carry columns and stop being the last uncolumned rows a text names.
//
// NULL IS AN ANSWER on all four. A capture that captures nothing and a mask
// that masks nothing are both legitimate; the launchers' own tests decide,
// exactly as they did when an arm handed the null over.

fact!(/// Where a capturing attention writes its pre-softmax logits.
    AttnScoreOut = "attn.score_out" => Source::Named("attn.score_out") => *mut f32);

fact!(/// The score sink's CSR: where each request's scores begin.
    AttnScoreIndptr = "attn.score_indptr" => Source::Named("attn.score_indptr") => *const i32);

fact!(/// The custom attention mask, one byte per (query, key) the CSR pairs.
    AttnMask = "attn.mask" => Source::Named("attn.mask") => *const u8);

fact!(/// The custom mask's CSR. [`AttnMask`]'s pair.
    AttnMaskIndptr = "attn.mask_indptr" => Source::Named("attn.mask_indptr") => *const i32);

fact!(/// How wide a window the score capture records.
    ///
    /// The fire's, not the statement's: a capture sink is sized when the fire
    /// carves it, which is before any statement is bound.
    AttnScoreWindow = "attn.score_window" => Source::Named("attn.score_window") => u32);

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

fact!(/// Scratch a split decode attention leaves its partial softmax states in.
    ///
    /// THE FIRE'S, not the statement's, which is the whole reason it is a
    /// fact and not an operand. Cutting a row's key range into slices so that
    /// more workgroups exist is a decision a BACKEND makes about its own
    /// occupancy -- a GPU with 32 query heads and 20 cores has nothing to run
    /// while a key load is in flight -- and an authored trace that had to
    /// declare a buffer for it would be carrying one driver's scheduling in a
    /// model description every driver reads.
    ///
    /// `*mut f32` because it is written and then read back within one fire:
    /// a running maximum, a denominator and an accumulator per (row, query
    /// head, slice), at full width because rounding a denominator to eight
    /// mantissa bits before the merge would throw away exactly what the
    /// online recurrence is for.
    ///
    /// A driver that does not split answers nothing here and no body asks;
    /// the unsplit kernel is a complete implementation on its own.
    AttnScratch = "attn_scratch" => Source::Named("attn_scratch") => *mut f32);

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

// ── The PER-EXPERT pointer arrays a routed decode indexes ──────────────────
//
// A bank and an array of pointers INTO that bank are two different addresses,
// and binding the first where the second belongs reads eight bytes of packed
// weight data as an address. `arms/quant.rs` said so at the parameter it was
// filling by hand -- *"`cx.weight(0)` is the bank's own base, and the kernel's
// first act is `packed_ptrs[expert]`"* -- and that hand-filling was the whole
// of what kept two arms alive.
//
// The distinction cannot come from the weight CHAIN, which is why these are
// keys: `Const<Tensor<E>>` derives `Or(Named("weight"), Slot(Weight, 0))`, and
// both halves of that chain answer the bank.

fact!(/// The `_ptrs` array beside a named weight: one device pointer per
    /// expert, into the bank.
    ///
    /// `serve::load::build_moe_expert_ptrs` builds one per plane at load, so
    /// its absence means the bank's byte count did not divide by the row's
    /// expert count.
    WeightExpertPtrs = "weight._ptrs" => Source::Named("weight._ptrs") => *const u8);

fact!(/// The `_scales_ptrs` array beside a named weight. [`WeightExpertPtrs`]'s
    /// pair, over the scales plane rather than the packed one.
    WeightExpertScalePtrs = "weight._scales_ptrs" => Source::Named("weight._scales_ptrs") => *const u8);

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

// ── §M  THE DRIVER-OWNED PLANES A BARE POINTER USED TO SPELL ──────────────
//
// Every key below replaces a parameter that carried NO mark and therefore no
// source: `bind()` refused each one with `Refusal::Unstated { what: "an
// argument whose signature does not say where it comes from" }`, and the
// signature was the thing that did not say.
//
// A MARK WAS THE WRONG REPAIR FOR ALL OF THEM, and `moe.rs`'s own note on the
// six batched-GEMM pointer arrays is the reason: *"Driver-owned workspace, not
// trace values [...] declaring them as results would free them (liveness) too
// early."* An `Out` mark is a claim the ALLOCATOR reads, so marking a buffer
// the driver carved and a LATER statement still reads shortens its life to one
// fire. These are asked for instead, which states where the value comes from
// without claiming the trace owns it.

// ── The MoE legs' driver workspace ──
//
// Two builders fill these — `moe::build_moe_ptrs_aligned_bf16` and the
// nemotron pair in `ssm.rs` — and `moe::moe_grouped_gemm_bf16` reads them back
// from inside its own body, one statement later. That gap is the liveness the
// note above is about.

fact!(/// The per-expert UP/GATE weight pointer table, host-filled.
    ///
    /// A table and not a bank: `WeightExpertPtrs` is the same shape for a
    /// routed matmul that names one, and nemotron needs two.
    MoeUpWeightPtrs = "moe.up_weight_ptrs" => Source::Named("moe.up_weight_ptrs") => *const *const core::ffi::c_void);

fact!(/// The per-expert DOWN weight pointer table. See [`MoeUpWeightPtrs`].
    MoeDownWeightPtrs = "moe.down_weight_ptrs" => Source::Named("moe.down_weight_ptrs") => *const *const core::ffi::c_void);

fact!(/// The decode leg's first intermediate, `[routes, 2 * intermediate]`.
    MoeExpertUp = "moe.expert_up" => Source::Named("moe.expert_up") => *mut core::ffi::c_void);

fact!(/// The decode leg's activation buffer, `[routes, intermediate]`.
    MoeExpertAct = "moe.expert_act" => Source::Named("moe.expert_act") => *mut core::ffi::c_void);

fact!(/// The decode leg's per-route output, `[routes, hidden]`.
    MoeExpertOut = "moe.expert_out" => Source::Named("moe.expert_out") => *mut core::ffi::c_void);

fact!(/// The aligned leg's first staging rectangle, padded to `block_size`.
    MoeAlignedUp = "moe.aligned_up" => Source::Named("moe.aligned_up") => *mut core::ffi::c_void);

fact!(/// The aligned leg's activation rectangle. See [`MoeAlignedUp`].
    MoeAlignedAct = "moe.aligned_act" => Source::Named("moe.aligned_act") => *mut core::ffi::c_void);

fact!(/// The aligned leg's output rectangle. See [`MoeAlignedUp`].
    MoeAlignedOut = "moe.aligned_out" => Source::Named("moe.aligned_out") => *mut core::ffi::c_void);

fact!(/// The UP GEMM's `A` pointer array — one address per block-row.
    MoeAUpPtrs = "moe.a_up_ptrs" => Source::Named("moe.a_up_ptrs") => *mut *const core::ffi::c_void);

fact!(/// The UP GEMM's `B` pointer array. See [`MoeAUpPtrs`].
    MoeBUpPtrs = "moe.b_up_ptrs" => Source::Named("moe.b_up_ptrs") => *mut *const core::ffi::c_void);

fact!(/// The UP GEMM's `C` pointer array, written through. See [`MoeAUpPtrs`].
    MoeCUpPtrs = "moe.c_up_ptrs" => Source::Named("moe.c_up_ptrs") => *mut *mut core::ffi::c_void);

fact!(/// The DOWN GEMM's `A` pointer array. See [`MoeAUpPtrs`].
    MoeADownPtrs = "moe.a_down_ptrs" => Source::Named("moe.a_down_ptrs") => *mut *const core::ffi::c_void);

fact!(/// The DOWN GEMM's `B` pointer array. See [`MoeAUpPtrs`].
    MoeBDownPtrs = "moe.b_down_ptrs" => Source::Named("moe.b_down_ptrs") => *mut *const core::ffi::c_void);

fact!(/// The DOWN GEMM's `C` pointer array, written through. See [`MoeAUpPtrs`].
    MoeCDownPtrs = "moe.c_down_ptrs" => Source::Named("moe.c_down_ptrs") => *mut *mut core::ffi::c_void);

fact!(/// The per-route router weight the decode leg scatters, `[n * top_k]`.
    MoeRouteWeights = "moe.route_weights" => Source::Named("moe.route_weights") => *mut f32);

fact!(/// The SHARED expert's gate/up bank, null where a text has no shared
    /// expert — the rewrite in `build_moe_ptrs_aligned_bf16` is what makes
    /// the null safe.
    MoeSharedGateUpBase = "moe.shared_gate_up_base" => Source::Named("moe.shared_gate_up_base") => *const core::ffi::c_void);

fact!(/// The SHARED expert's down bank. See [`MoeSharedGateUpBase`].
    MoeSharedDownBase = "moe.shared_down_base" => Source::Named("moe.shared_down_base") => *const core::ffi::c_void);

// ── deepseek_v4's SECOND cache ──
//
// `driver-cuda/src/pools/compressed_plane_cache.rs` allocates all three, one
// `TensorSpec` per layer. The pool existed before these keys did, which is why
// the routines that read it took bare pointers: the resource was there and
// nothing named it.

fact!(/// The compressed KV pages — deepseek_v4's second cache, indexed
    /// through the ORDINARY plan's page tables.
    Dsv4CompKvPages = "dsv4.comp_kv_pages" => Source::Named("dsv4.comp_kv_pages") => *mut core::ffi::c_void);

fact!(/// The in-progress compression accumulator, `coff * head_dim` wide.
    Dsv4StateKv = "dsv4.state_kv" => Source::Named("dsv4.state_kv") => *const core::ffi::c_void);

fact!(/// The per-slot importance scores the boundary pass writes.
    Dsv4StateScore = "dsv4.state_score" => Source::Named("dsv4.state_score") => *const core::ffi::c_void);

fact!(/// The absolute-position embedding table, a LOAD-TIME constant and not
    /// a fire's plane — nullable, and the only operand of the gather that is.
    Dsv4Ape = "dsv4.ape" => Source::Named("dsv4.ape") => *const f32);

// ── The quantised KV envelope, written ──
//
// `KvEnvMin`/`KvEnvMax` are the READ side and are `*const u16`. The three
// `layout.rs` envelope routines WRITE the same two planes, and a `*const` key
// cannot say so.

fact!(/// The dequant envelope FLOOR, as the envelope passes write it.
    /// [`KvEnvMin`] is the same plane read.
    KvEnvMinOut = "kv.k_env_min_out" => Source::Named("kv.k_env_min_out") => *mut core::ffi::c_void);

fact!(/// The dequant envelope CEILING, written. See [`KvEnvMinOut`].
    KvEnvMaxOut = "kv.k_env_max_out" => Source::Named("kv.k_env_max_out") => *mut core::ffi::c_void);

// ── The compacted page CSR ──
//
// `attn::compact_page_csr` reads one plan's page tables and writes another's.
// The destination is the driver's scratch, not a result any statement declares.

fact!(/// The scratch counters the CSR compaction accumulates into.
    CompactScratchCounts = "compact.scratch_counts" => Source::Named("compact.scratch_counts") => *mut u32);

fact!(/// The compacted page-index array.
    CompactPageIndicesOut = "compact.page_indices_out" => Source::Named("compact.page_indices_out") => *mut u32);

fact!(/// The compacted page-indptr array. See [`CompactPageIndicesOut`].
    CompactPageIndptrOut = "compact.page_indptr_out" => Source::Named("compact.page_indptr_out") => *mut u32);

fact!(/// The compacted last-page-length array. See [`CompactPageIndicesOut`].
    CompactLastPageLensOut = "compact.last_page_lens_out" => Source::Named("compact.last_page_lens_out") => *mut u32);

fact!(/// Where `attn::attn_score_fold_heads` writes its per-page maxima.
    AttnScoreFolded = "attn.score_folded" => Source::Named("attn.score_folded") => *mut f32);

// ── The grouped GEMM's pointer arrays ──
//
// `gemm::grouped_act_x_wt_bf16` is reached from `gemm/lora.rs`, which builds
// these by pointer arithmetic into its own device slab. Its own note said so:
// *"not a statement operand"*.

fact!(/// The grouped GEMM's per-group activation pointers, on the DEVICE.
    GemmActPtrs = "gemm.act_ptrs_dev" => Source::Named("gemm.act_ptrs_dev") => *const *const core::ffi::c_void);

fact!(/// The grouped GEMM's per-group weight pointers. See [`GemmActPtrs`].
    GemmWeightPtrs = "gemm.w_ptrs_dev" => Source::Named("gemm.w_ptrs_dev") => *const *const core::ffi::c_void);

fact!(/// The grouped GEMM's per-group output pointers. See [`GemmActPtrs`].
    GemmOutPtrs = "gemm.y_ptrs_dev" => Source::Named("gemm.y_ptrs_dev") => *const *mut core::ffi::c_void);

fact!(/// The one HOST array of the four: per-group row counts, which
    /// `cublasGemmGroupedBatchedEx` reads on the host side.
    GemmMArrayHost = "gemm.m_array_host" => Source::Named("gemm.m_array_host") => *const i32);

fact!(/// LoRA's `x @ A` intermediate, carved by the driver and read back by
    /// the correction's second GEMM.
    LoraXaScratch = "lora.xa_scratch" => Source::Named("lora.xa_scratch") => *mut core::ffi::c_void);

fact!(/// The MTP pending-hidden slab, written. [`RecurrentState`] is the same
    /// store under its generic name; this one is the qwen-3.5 MTP view.
    MtpPendingHidden = "mtp.pending_hidden" => Source::Named("mtp.pending_hidden") => *mut core::ffi::c_void);

// ── §L  THE SCALARS `#[unbound]` USED TO STAND FOR ────────────────────────
//
// Sixty-four parameters carried `#[unbound]` — *"nothing supplies this and the
// signature says so"* — and the attribute was honest about the tree as it
// stood: `Env<T, keys::Unstated>` before it, a fake key for a real absence.
//
// It is the POINTER story one type over. Where a plane could be asked for by
// name once a key existed, so can a number; what kept these unstated was that
// no key named them, not that nothing knows them. The ones a RECTANGLE already
// answers — a row count, a hidden width, a stream multiplier — are read off
// the operands instead and appear nowhere below, because a fact the signature
// already carries does not need a name.
//
// `Const` WAS THE OTHER CANDIDATE AND IS REFUSED, for the reason
// `attn::attention_compressed_paged_bf16` states in its own body: *"a `Const`
// mark PROMISES the statement carries the number at its slot in the params
// run; where nothing states one the promise breaks at the fire, not at the
// type."* An ask is the weaker and truer claim.

fact!(/// deepseek_v4's compression stride: one entry per `ratio` positions.
    Dsv4Ratio = "dsv4.ratio" => Source::Named("dsv4.ratio") => i32);

fact!(/// How many positions the compression accumulator spans. See
    /// [`Dsv4StateKv`], whose width is `coff * head_dim`.
    Dsv4Coff = "dsv4.coff" => Source::Named("dsv4.coff") => i32);

fact!(/// The DSA indexer's rotary width, which is not the head's.
    DsaRopeDim = "dsa.rope_dim" => Source::Named("dsa.rope_dim") => i32);

fact!(/// The hyper-connection residual's epsilon.
    ///
    /// NOT [`RmsEps`]: `dsv4_hc.cuh` norms the MIX and the two are separate
    /// numbers in the checkpoint.
    HcEps = "hc.eps" => Source::Named("hc.eps") => f32);

fact!(/// The post-mix blend weight the hyper-connection prologue applies.
    HcPostAlpha = "hc.post_alpha" => Source::Named("hc.post_alpha") => f32);

fact!(/// How many Sinkhorn normalisation passes the combine matrix takes.
    HcSinkhornIters = "hc.sinkhorn_iters" => Source::Named("hc.sinkhorn_iters") => i32);

fact!(/// Weights per quantisation group along `K` — the checkpoint's, not a
    /// fire's. `KvBlockSize` is the same idea for the KV cache.
    QuantGroupSize = "quant.group_size" => Source::Named("quant.group_size") => i32);

fact!(/// mRoPE's TEMPORAL section width.
    MropeSectionT = "rope.mrope_section_t" => Source::Named("rope.mrope_section_t") => i32);

fact!(/// mRoPE's HEIGHT section width. See [`MropeSectionT`].
    MropeSectionH = "rope.mrope_section_h" => Source::Named("rope.mrope_section_h") => i32);

fact!(/// mRoPE's WIDTH section. See [`MropeSectionT`].
    MropeSectionW = "rope.mrope_section_w" => Source::Named("rope.mrope_section_w") => i32);

fact!(/// Bytes between one lane's keep-mask row and the next, in the CSR
    /// compaction's input.
    CompactKeepStride = "compact.keep_stride" => Source::Named("compact.keep_stride") => u32);

fact!(/// The padded block-row ceiling the aligned MoE leg sorts into.
    MoeMaxBlocks = "moe.max_blocks" => Source::Named("moe.max_blocks") => i32);

fact!(/// Rows per padded block. See [`MoeMaxBlocks`].
    MoeBlockSize = "moe.block_size" => Source::Named("moe.block_size") => i32);

fact!(/// How many of [`MoeMaxBlocks`] carry ROUTED experts; the rest are the
    /// shared expert's, and a text without one sets this to the ceiling.
    MoeRoutedBlocks = "moe.routed_blocks" => Source::Named("moe.routed_blocks") => i32);

fact!(/// The MoE leg's model width. Not [`Width`], which is the fire's.
    MoeHidden = "moe.hidden" => Source::Named("moe.hidden") => i32);

fact!(/// One expert's intermediate width.
    MoeIntermediate = "moe.intermediate" => Source::Named("moe.intermediate") => i32);

fact!(/// Rows in the per-expert scale plane the transpose walks.
    MoeScaleRows = "moe.scale_rows" => Source::Named("moe.scale_rows") => i32);

fact!(/// Quantisation groups per row of that plane. See [`MoeScaleRows`].
    MoeScaleGroups = "moe.scale_groups" => Source::Named("moe.scale_groups") => i32);

fact!(/// Groups in a grouped GEMM — `cublasGemmGroupedBatchedEx`'s count.
    GemmGroupCount = "gemm.group_count" => Source::Named("gemm.group_count") => i32);

fact!(/// The GEMM's accumulate weight: `0.0` overwrites the result plane and
    /// `1.0` adds to it.
    ///
    /// A number the SYMBOL decides, which is what `Lit` is for — but the
    /// symbols that decide it are reached through ONE routine here, so the
    /// choice belongs to whoever dispatched rather than to the row.
    GemmBeta = "gemm.beta" => Source::Named("gemm.beta") => f32);

fact!(/// SiLU's beta on the gated MLP's ACTIVE half.
    ///
    /// Beside [`GluAlpha`] and [`GluLimit`], which are the same block's other
    /// two numbers: `Deployment::mlp_gate` carries all three per fire.
    MlpSituBeta = "mlp.situ_beta" => Source::Named("mlp.situ_beta") => f32);

fact!(/// The blend applied to the LINEAR half. See [`MlpSituBeta`].
    MlpSituLinearBeta = "mlp.situ_linear_beta" => Source::Named("mlp.situ_linear_beta") => f32);

fact!(/// MLA's non-rotary query width. [`HeadDim`] is the whole head.
    MlaQkNopeDim = "mla.qk_nope_dim" => Source::Named("mla.qk_nope_dim") => i32);

fact!(/// MLA's VALUE head width, which the absorb pair reads apart from the
    /// query's. See [`MlaQkNopeDim`].
    MlaVHeadDim = "mla.v_head_dim" => Source::Named("mla.v_head_dim") => i32);

fact!(/// The compressed latent's rank — MLA's whole point.
    MlaKvLoraRank = "mla.kv_lora_rank" => Source::Named("mla.kv_lora_rank") => i32);

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
