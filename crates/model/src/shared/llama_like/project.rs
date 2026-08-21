//! The shared projection every llama-lineage generation's rows go
//! through.
//!
//! Twelve `model_type` strings dispatched to one derivation
//! (`llama_like_facts_from_hf`), and the reason was never that twelve
//! families are alike — it is that they are ONE family whose rows differ
//! in their numbers. Qwen3, Llama 3, Mistral, OLMo, Phi-3, Gemma 3 and
//! the Qwen mixtures are all `LlamaLikeFacts`; what a generation module
//! adds is which chat template speaks for it and which author writes its
//! contract, and those are the two things a shape cannot state.
//!
//! So the projections live here, once, taking a `&LlamaLikeFacts`, and a
//! generation's `impl Variant` calls them. That is the same N:1 the old
//! `HF_ROWS` column expressed — spelled as a call rather than as a table
//! nothing held to the other two tables.

// Only the texts name a backend, and only they are gated.
use crate::catalog::{Backend, Deployed, MetalBinding};
use crate::deployment::{
    Advertised, AttnOutput, Deployment, Geometry, KvStyle, LayerAttention, NormPlacement,
    PrefillStyle, round_up_attn_head_dim,
};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::LlamaLikeFacts;

use model_ir::facts::{NormPlacement as SpecNorm, QkNorm};

/// This row's tensors.
///
/// Every extent is the row's own arithmetic, which is what makes the
/// manifest a check rather than a second statement: `q_proj` is
/// `[q_heads * head_dim, hidden]` because that is what `q_heads` and
/// `head_dim` MEAN.
#[must_use]
pub fn manifest(f: &LlamaLikeFacts) -> Manifest {
    let (hidden, vocab) = (u64::from(f.hidden), u64::from(f.vocab));
    let (q, kv) = (u64::from(f.q_width()), u64::from(f.kv_width()));
    let head_dim = u64::from(f.head_dim);
    let dense = f.n_experts == 0;

    Manifest::new(f.layers)
        .with(TensorSpec::required("embed_tokens", [vocab, hidden]))
        .with(TensorSpec::required("norm", [hidden]))
        // TIED vs UNTIED as presence, which is the only way a manifest
        // can tell them apart: every extent agrees.
        .tie(f.tied_embeddings, "lm_head", [vocab, hidden])
        .with(TensorSpec::required(
            "layer.{}.self_attn.q_proj",
            [q, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.k_proj",
            [kv, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.v_proj",
            [kv, hidden],
        ))
        .with(TensorSpec::required(
            "layer.{}.self_attn.o_proj",
            [hidden, q],
        ))
        // The q/k-norm question the old derivation answered by dividing
        // a byte count: per-head ships `[head_dim]`, global ships the
        // whole projection width, and off ships nothing.
        .with(match f.qk_norm {
            QkNorm::Off => TensorSpec::absent("layer.{}.self_attn.q_norm"),
            QkNorm::PerHead => TensorSpec::required("layer.{}.self_attn.q_norm", [head_dim]),
            QkNorm::Global => TensorSpec::required("layer.{}.self_attn.q_norm", [q]),
        })
        // Pre-norm ships one input norm per sub-layer; post-norm
        // (olmo2/olmo3) ships the pair that follows them instead. This
        // is the `ends_with("input_layernorm.weight")` alias probe, as
        // an expectation.
        //
        // The DISCRIMINATING pair is `input_layernorm` and
        // `post_feedforward_layernorm`, and not the name that reads as
        // though it were: HuggingFace calls the norm in front of the
        // MLP `post_attention_layernorm` because of where it sits in
        // the block, not because of which placement it belongs to, and
        // EVERY row here ships it. A manifest that made it the marker
        // for `Post` refused llama-3.2 on the one tensor llama-3.2
        // certainly has.
        .either(
            f.norm_placement == SpecNorm::Pre,
            "layer.{}.input_layernorm",
            [hidden],
        )
        .with(TensorSpec::required(
            "layer.{}.post_attention_layernorm",
            [hidden],
        ))
        .either(
            f.norm_placement == SpecNorm::Post,
            "layer.{}.post_feedforward_layernorm",
            [hidden],
        )
        .with_if(
            f.qkv_bias,
            TensorSpec::required("layer.{}.self_attn.q_proj.bias", [q]),
        )
        .with_if(
            dense,
            TensorSpec::required(
                "layer.{}.mlp.gate_proj",
                [u64::from(f.intermediate), hidden],
            ),
        )
        .with_if(
            dense,
            TensorSpec::required(
                "layer.{}.mlp.down_proj",
                [hidden, u64::from(f.intermediate)],
            ),
        )
        .with_if(
            !dense,
            TensorSpec::required("layer.{}.mlp.gate", [u64::from(f.n_experts), hidden]),
        )
        .with_if(
            !dense,
            TensorSpec::present("layer.{}.mlp.experts.0.gate_proj"),
        )
}

/// This row's deployment.
///
/// A projection, and a short one: every value below was already in the
/// row. The eleven-function derivation it replaces read the same numbers
/// out of a parsed `config.json`, one family at a time.
///
/// It takes the same [`RowScalars`] [`trace`] does, rather than the three
/// loose scalars it used to, so a generation cannot page at one theta and
/// fire at another. See that struct for why.
#[must_use]
pub fn deployment(f: &LlamaLikeFacts, row: RowScalars) -> Deployment {
    let RowScalars {
        rope_theta,
        norm_eps,
        window: sliding_window,
        norm_topk_prob,
        ..
    } = row;
    let head_dim = round_up_attn_head_dim(f.head_dim);
    let attention = (0..f.layers)
        .map(|l| LayerAttention {
            // One shape for every layer, which is what this row was
            // already saying by having no per-layer count.
            kv_heads: f.kv_heads,
            head_dim,
            window: sliding_window,
            // Every layer owns its pages. KV sharing is gemma-4's, and
            // it is a fact about a LAYER there rather than a family
            // here.
            kv_source: l,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            rope_theta,
            // Full rotation at the head dim.
            rotary_dim: 0,
            q_gate: false,
        })
        .collect();
    Deployment {
        layers: f.layers,
        norm_eps,
        // The row's own numbers, so a LAUNCH and a TRACE cannot
        // disagree about how many heads there are. `driver-cuda` read
        // thirty of these off a resident `HfConfig`, which is a second
        // reading of the same document.
        shape: Geometry {
            hidden: f.hidden,
            q_heads: f.q_heads,
            kv_heads: f.kv_heads,
            head_dim: f.head_dim,
            head_dim_kernel: round_up_attn_head_dim(f.head_dim),
            intermediate: f.intermediate,
            moe_intermediate: f.moe_intermediate,
            experts_per_token: f.experts_per_token,
            shared_intermediate: f.shared_intermediate,
            vocab: f.vocab,
        },
        attention,
        kv: KvStyle::Paged,
        recurrent: None,
        prefill: PrefillStyle::Planned,
        // The guard region records no SSA output for this text, so the
        // driver owns the landing buffer. `pins_attention_values()`
        // defaulted to true with the doc "Only gemma-4 does" — the
        // exception is gemma-4's row, and it says so there.
        attn_output: AttnOutput::DriverPinned,
        logit_softcap: 0.0,
        // No ATTENTION cap: gemma-2's `attn_logit_softcapping` is
        // gemma-2's alone, and a zero here is "no cap" rather than a
        // cap at zero — which would flatten every score to `tanh(inf)`.
        attn_logit_softcap: 0.0,
        ple_dim: 0,
        norm: match f.norm_placement {
            SpecNorm::Post => NormPlacement::Post,
            SpecNorm::Pre | SpecNorm::Sandwich => NormPlacement::Pre,
        },
        // `false` even for the `Sandwich` arm above, and that is the whole
        // point of the field. The sandwich is an ARCHITECTURE and the
        // offset is a WEIGHT CONVENTION; every stack this shared
        // projection serves stores the multiplier directly, and the
        // gemmas that do not have projections of their own that say so.
        norm_unit_offset: false,
        v_norm: false,
        norm_topk_prob,
        // No router of this family states a scaling factor.
        routed_scaling: 1.0,
        mlp_gate: crate::deployment::MlpGate::Silu,
        scales: std::collections::BTreeMap::new(),
        // Filled by the ROW, not by the shape: a family label and a
        // published context ceiling are facts about a checkpoint, and a
        // projection only sees geometry.
        advertised: Advertised::default(),
        rope_scaling: None,
        towers: Default::default(),
    }
}

/// The CUDA binding facts for this row.
///
/// Every field the old derivation hardcoded is hardcoded here, in one
/// place, with the two that are not constants — the padded head dim and
/// the TP width — coming from the row and the load respectively.
#[must_use]
pub fn cuda_facts(
    f: &LlamaLikeFacts,
    load: Deployed<'_>,
) -> super::forward::facts::LlamaLikeCudaFacts {
    let kernel = round_up_attn_head_dim(f.head_dim);
    super::forward::facts::LlamaLikeCudaFacts {
        xqa_decode: false,
        decode_fused_post: false,
        rope_table: true,
        force_prefill_path: false,
        head_dim_padded: kernel != f.head_dim,
        head_dim_kernel: if kernel == f.head_dim { 0 } else { kernel },
        gate_up_fused: true,
        proj_repr: model_dsl::WeightRepr::Bf16,
        tp_size: load.tp_size.max(1),
        window_left: Vec::new(),
        all_reduce_p2p_max_rows: 0,
    }
}

/// The GEMM tile this build's Metal shaders were STAMPED at.
///
/// A property of the BINARY and not of any checkpoint, exactly as
/// [`ATTN_HEAD_DIMS`] is: `affine_qmm_t` is instantiated over
/// `(group × bits × bm × bn)`, so a batched projection's SYMBOL carries a
/// tile. Neither number is a row's to state nor a load's to observe, which is
/// why this is a `const` here and not a field of [`MetalBinding`].
///
/// A symbol whose tile is WRONG reads the wrong bytes; a symbol with no tile
/// at all does not resolve, which is the better failure -- so `qmm_tile` is
/// stated rather than left at the serde default of `(0, 0)`.
///
/// 32 rather than 16 because 16 has no cooperative-matrix build to compile
/// into. Measured on an RTX 4090 with 4-bit qwen3-0.6B, a 1024-token prefill
/// takes 2563 ms at `(16, 32)` and 565 ms at `(32, 32)`; 64 is a tenth behind
/// on short prompts and refuses more prompts at the `TokensMultipleOf` guard.
/// `kernels-metal`, `kernels-wgpu` and `kernels-vulkan` all declare `TILE_M`
/// as `_bm_16 | _bm_32 | _bm_64` and all hand-list a `bm_32` build, so 32
/// resolves in every tree; the CUDA backend does not read this field. A
/// backend wanting its own tile has `qmm_tile` on [`MetalBinding`].
///
/// `driver-wgpu` now DOES, and the escape hatch above is how: it states
/// `(32, 64)` at its engine seam. The reason is the one this doc gives for
/// 32 -- there is no cooperative-matrix build on that backend, so the
/// argument for the shared value does not reach it -- and the number is a
/// 1.36x pp512 measured on an M4 Pro. It is a WIDER tile and not a
/// narrower one, so it does not move `TokensMultipleOf`, which is `bm`'s.
pub const QMM_TILE: (u32, u32) = (32, 32);

/// Whether this build's dense GEMM stages its activation to `half`.
///
/// See [`LlamaLikeMetalFacts::qmm_fp16_precast`] for what the staging buys.
/// Spelled `(group, bits)` to read like `Tuning::fp16_gemm_format`, which is
/// the driver's half of the same question.
///
/// The condition is the CODEC's, not the family's: `qmm_t.metal` stamps
/// `affine_qmm_t_fp16_precast` at `gs = 64, b = 4` and at nothing else, so
/// any other quantization has no symbol to name and has to take the
/// emulated `bfloat` tiles.
///
/// Not gated on the device here. A trace is authored once and read on
/// whatever machine loads it, and the family-9 parts where the `bfloat`
/// matrix unit is real are also the parts where the staging pass is pure
/// cost — that half of the choice is `Tuning::fp16_qmm`'s, on the driver
/// side, where the device is a thing that can be asked.
///
/// [`LlamaLikeMetalFacts::qmm_fp16_precast`]:
///     crate::shared::llama_like::forward::LlamaLikeMetalFacts::qmm_fp16_precast
#[must_use]
pub const fn qmm_fp16_precast(group: u32, bits: u32) -> bool {
    bits == 4 && group == 64
}

/// The ROUTED GEMM's tile this build stamps, which is deliberately narrower
/// than [`QMM_TILE`].
///
/// See [`LlamaLikeMetalFacts::moe_tile`] for why the two differ: a routed
/// tile's rows are ONE EXPERT'S share of the fire, so the row tile is also
/// the sort's padding block and a wide one pads more than it amortises. A
/// prefill of 128 tokens at top-4 over 128 experts stacks 512 rows across
/// 128 runs — four rows a run on average, which `bm = 16` rounds to sixteen
/// and `bm = 64` would round to sixty-four.
///
/// `bm = 32` is measured, not assumed. On gemma-4-26b-a4b (128 experts,
/// top-8) a 128-token prefill runs 397 tok/s at `bm = 16`, **421** at 32
/// and 371 at 64: sixteen pays too many tiles for the weight traffic and
/// sixty-four pads too many rows into the arithmetic. gpt-oss (32 experts,
/// top-4) is flat across all three, so nothing else in the tree argues
/// for a different point.
///
/// `bn` is measured too, and it does NOT follow the row tile's argument.
/// The column tile is not the sort's padding block — it tiles the expert's
/// own output width, which is whole — so widening it only trades tiles for
/// register reuse. A 128-token prefill, tok/s:
///
/// | `(bm, bn)` | gemma-4-26b-a4b | gpt-oss-20b | Qwen3.6-35B-A3B |
/// |---|---|---|---|
/// | (16, 32) | 408.5 | | |
/// | (32, 16) | 395.5 | | |
/// | (32, 32) | 433.4 | 477.3 | 297.0 |
/// | (32, 64) | **436.7** | **479.1** | 296.9 |
/// | (64, 32) | 382.1 | | |
///
/// The gemma-4 pair is three repeats each and does not overlap
/// (433.2/433.3/433.6 against 436.4/436.5/437.1). It is under one percent,
/// and it is free: the wider symbol is already instantiated. A3B is flat
/// because its expert width (768) leaves the wider tile with the same tile
/// count it had at 32.
///
/// The DENSE tile does not want the same point. Swept on the same model
/// with the routed tile held at (32, 64): (32, 32) 436.9, (32, 64) 431.7,
/// (16, 32) 417.4, (64, 32) 420.0, (64, 64) 414.9. A dense GEMM's rows are
/// the whole batch rather than one expert's share of it, so it is already
/// wide enough that a wider tile only costs occupancy.
///
/// # The padding argument above is the right answer for the wrong reason
///
/// Everything before this paragraph reasons about `bm` as a padding block,
/// and the reasoning does not survive being measured. Cutting the fire at
/// the MoE block's boundaries (`tier_one_prefill_then_decode` documents the
/// method) prices one layer's three routed GEMMs at 5.7 ms with `bm = 32`
/// and **6.2 ms with `bm = 16`** -- half the padded arithmetic, nine percent
/// SLOWER. If padding were what `bm` bought, that number could not exist.
///
/// What the same cut says when the token count moves instead: 2.6 ms at
/// n=32, 5.7 at 128, 15.8 at 512. That fits a per-touched-expert term of
/// 2.33 ms -- 428 MB of expert weights a layer at 184 GB/s, which is this
/// machine's real bandwidth -- plus a per-REAL-row term of 3.37 ms. No
/// padding term is needed to fit it, because the padded rows ride along with
/// a weight fetch that was going to happen anyway.
///
/// So `bm = 32` wins on arithmetic intensity and not on padding: at
/// `(16, 32)` a lane holds two accumulator fragments where `(32, 32)` gives
/// it four, and the tile becomes loader-bound. `(16, 64)` restores the four
/// -- 8 rows x 32 columns a simdgroup, one A fragment against four B -- and
/// it recovers most but not all of the loss (430.4 against 436.7), which is
/// the cleanest statement of the axis there is: the row tile is bought for
/// what a lane accumulates, and the sort's padding is nearly free.
pub const ROUTED_QMM_TILE: (u32, u32) = (32, 64);

/// Why a SHARDED load has no Metal text here.
///
/// A `const` so the test that asserts the refusal names the missing
/// thing compares against the same sentence the operator is shown,
/// rather than against a paraphrase that can drift away from it — the
/// discipline `csm::project::NO_TRACE` sets.
pub const NO_METAL_SHARD: &str = "this Metal load states a tensor-parallel width above one and \
     `LlamaLikeMetalFacts` has no shard vocabulary: the CUDA facts carry \
     a `tp_size` that narrows every projection width in the text, and the \
     Metal ones carry nothing, so the text would state the WHOLE model's \
     widths against one rank's slice of the weights and read past the end \
     of every projection. Refused rather than traced, because a shard \
     read at full width is arithmetic that runs";

/// The head widths `sdpa_paged.metal` instantiates.
///
/// A row's `head_dim` names the symbol directly — the `_d_256` literal
/// that strode 128-wide heads was fixed by reading it — so a width the
/// shader never compiled names a symbol nothing exports. That used to be
/// left to fail at pipeline construction, and phi-3's 96 is why it
/// cannot be: `model-compiler` validates a declaration against the
/// kernel rows and PANICS on an undeclared launch, so the row never
/// reaches the driver to be refused by name.
///
/// The set is `sdpa_paged_decode`'s own axis, minus the page-shape tails
/// (`_p32`, `_p32_sg8`), which are points of a different axis.
pub const METAL_SDPA_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// [`METAL_SDPA_HEAD_DIMS`] as a refusal.
pub const NO_METAL_HEAD_DIM: &str = "this row's heads are a width `sdpa_paged.metal` does not \
     instantiate: the shader compiles the paged decode at 64, 128, 256 \
     and 512, and the text names `sdpa_paged_decode_bfloat16_d_<width>` \
     from the row. CUDA pads such a row to `head_dim_kernel` and strips \
     the pad back off; Metal has no pad kernel in the text, so the \
     choices here are a symbol no shader exports or an attention that \
     reads 32 columns of whatever the loader staged next. Refused";

/// The one affine point `quant/qmv.metal` instantiates the ROUTED
/// matvec at.
///
/// `AffineQ::group_size` is a template constant, so `affine_qmv_routed`
/// exists at exactly `(64, 4)` and a second point would name an
/// instantiation that dequantises at 64 whatever its name claimed —
/// reading every scale from the wrong offset, which is the 909,207-NaN
/// defect the MXFP4 arm was added to fix. MXFP4 is a different codec
/// rather than another point and has its own symbol at group 32.
pub const METAL_ROUTED_AFFINE: (u32, u32) = (64, 4);

/// [`METAL_ROUTED_AFFINE`] as a refusal.
pub const NO_METAL_ROUTED_ENCODING: &str = "this row's expert bank reached the device at an affine \
     point `quant/qmv.metal` does not instantiate the routed matvec at: \
     the shader compiles `affine_qmv_routed` only at group 64 / 4 bits, \
     because `AffineQ::group_size` is a template constant. A bank at \
     another group dequantised by that kernel reads every scale from the \
     wrong offset and answers bf16 garbage, which is NaN more often than \
     not. Refused";

/// A landing bias under a norm placement that lands through the shared
/// closure.
///
/// The Metal text adds `o_bias` at two places: after the FUSED landing
/// (`gemm_add`, the `Pre` arm), and it used to have a second copy inside
/// the `land` closure the `Post` and `Sandwich` arms share. That second
/// copy could not run -- gpt-oss is the only row in the catalog that
/// publishes an `o_proj` bias and it norms `Pre` -- so it was a branch
/// standing in for a decision nobody had made.
///
/// Deleting it alone would have made the pairing a SILENT DROP: the
/// contract declares `o_bias`, the loader reads it off disk and stages
/// it, and the text would never sum it. No shape error and no unbound
/// symbol, just a per-channel offset missing from every layer. So the
/// pairing is refused by name instead, and the day a checkpoint arrives
/// with both, the refusal says which two facts it is holding rather
/// than the model answering slightly wrong forever.
pub const NO_METAL_NORMED_LANDING_BIAS: &str = "this row publishes a bias on its attention landing AND norms that \
     landing's output, and the Metal text adds the landing bias only on \
     the arm that fuses the residual into the projection. The normed \
     arms land through a shared statement that has no bias in it, so \
     this row would load the tensor, stage it, and never sum it. \
     Refused rather than dropped";

/// What this build's Metal kernels cannot run, asked of the FACTS.
///
/// Three refusals about the KERNEL SET rather than about a row: a width
/// `sdpa_paged.metal` never instantiated, an affine point `quant/qmv.metal`
/// never stamped the routed matvec at, and a shard count no Metal deployment
/// has. One function rather than three lines in [`trace`] because `gemma_3`
/// and `gemma_4` reach [`llama_like_metal`] directly; a row that reaches an
/// unstamped width unrefused meets `model-compiler`'s abort on an undeclared
/// launch instead of a sentence.
///
/// BOTH head dims are checked. gemma-4 has two -- 256 sliding against 512
/// full -- and the text names `sdpa_paged_decode_bfloat16_d_<width>` for each.
/// `global_head_dim` is zero when the stack has one shape, which is why zero
/// is skipped rather than refused.
///
/// # Errors
///
/// [`NO_METAL_SHARD`], [`NO_METAL_HEAD_DIM`] or
/// [`NO_METAL_ROUTED_ENCODING`], in the order a load meets them.
pub fn metal_kernel_refusal(
    f: &LlamaLikeFacts,
    m: &super::forward::facts::LlamaLikeMetalFacts,
    load: Deployed<'_>,
    bind: &MetalBinding,
) -> Result<(), crate::deployment::Refusal> {
    if load.tp_size > 1 {
        return Err(crate::deployment::Refusal::Unsupported(NO_METAL_SHARD));
    }
    if !METAL_SDPA_HEAD_DIMS.contains(&f.head_dim) {
        return Err(crate::deployment::Refusal::Unsupported(NO_METAL_HEAD_DIM));
    }
    if m.global_head_dim > 0 && !METAL_SDPA_HEAD_DIMS.contains(&m.global_head_dim) {
        return Err(crate::deployment::Refusal::Unsupported(NO_METAL_HEAD_DIM));
    }
    // MXFP4 banks take their own symbol at group 32, so the affine
    // point is not asked of them.
    if f.o_bias && f.norm_placement != SpecNorm::Pre {
        return Err(crate::deployment::Refusal::Unsupported(
            NO_METAL_NORMED_LANDING_BIAS,
        ));
    }
    if f.n_experts > 0
        && !bind.moe_mxfp4
        && (bind.quant_group, bind.quant_bits) != METAL_ROUTED_AFFINE
    {
        return Err(crate::deployment::Refusal::Unsupported(
            NO_METAL_ROUTED_ENCODING,
        ));
    }
    Ok(())
}

/// The row's own numbers, which [`LlamaLikeFacts`] does not hold.
///
/// Gathered here rather than added to the shape because twelve generations
/// share that struct and a field on it is a field every one of them restates.
/// [`deployment`] and [`trace`] both take this struct, so the row is read
/// once and spent twice: a row that FIRED at one theta and PAGED at another
/// would otherwise compile.
///
/// `llama_like_cuda` names no epsilon and no rotary base -- the CUDA driver
/// carries both on its own `fwd_cfg` -- while `llama_like_metal` names all
/// three, because a Metal statement carries every scalar its kernel reads.
/// Not gated on `forward`: [`deployment`] and the `fn row` that feeds it are
/// not gated either.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowScalars {
    /// `rope_theta`, the rotary base every layer of this row rotates at.
    ///
    /// Stated rather than defaulted for the reason `ModelFacts::rope_theta`
    /// gives about its own: a reader that only knows the flat key finds
    /// nothing on a config that nests it, silently keeps its default, and
    /// the rotated channels come out wrong in a way that compounds layer
    /// over layer until the activations saturate.
    pub rope_theta: f32,
    /// `rms_norm_eps`, the epsilon every norm of this row carries.
    ///
    /// A norm handed zero divides by the root of the mean square alone,
    /// which for a near-zero row is an infinity the next kernel spreads
    /// everywhere. The generation states it — most of them once, as a
    /// constant, because every published config of the generation agrees.
    pub norm_eps: f32,
    /// The window every layer attends over, `-1` for the whole context.
    ///
    /// ONE number because this family's rows state one: the schedules
    /// that ALTERNATE are gemma's and gpt-oss's, and each of those has a
    /// projection of its own that expands its own rule. A row that
    /// alternated and stated a single width here would attend the whole
    /// context on every layer or a window on every layer, and both read
    /// as a working model.
    pub window: i32,
    /// The row states a rope RESCALING, so no base expresses its ladder.
    ///
    /// llama-3's piecewise `rope_scaling` and OLMo-3's YaRN. The table itself
    /// is the DRIVER's -- derived at load and answered as
    /// `Source::Named(<keys::RopeFrequencies as keys::Fact>::KEY)` -- and this
    /// only says which form the statement takes. A factor of zero reads as "no
    /// rescaling", so getting it wrong degrades rather than fails.
    pub rope_rescaled: bool,
    /// Whether the router renormalizes over the SELECTED experts.
    ///
    /// HF's `norm_topk_prob`, and the third word of `moe/route.metal`'s
    /// `RouterParams`. True softmaxes the k chosen logits so the weights sum
    /// to one; false softmaxes over ALL the experts and then selects, so they
    /// sum to less than one and scale the routed FFN's whole contribution
    /// down with them. Both produce weights and neither faults.
    ///
    /// Stated by the ROW and not defaulted anywhere, because the default is
    /// the trap. `Qwen3MoeConfig.norm_topk_prob` is `False` in transformers,
    /// and `Qwen/Qwen3-30B-A3B`'s published `config.json` says `true` — the
    /// row this catalog serves disagrees with the class it is loaded by. Of
    /// the routed checkpoints on this machine (Qwen3.6-35B-A3B,
    /// gemma-4-26b-a4b, gpt-oss-20b) not one states the key at all, so a
    /// reader that fell through to a class default would be answering a
    /// question the checkpoint never asked.
    ///
    /// A DENSE row states it too and it is never staged: no `router_topk` is
    /// in a dense text. `RowScalars` has no `Default` for the reason the facts
    /// have none — "this one has no router" is part of the measurement, and a
    /// row added later should have to answer.
    pub norm_topk_prob: bool,
}

/// The METAL binding facts for this load.
///
/// The twin of [`cuda_facts`]: six fields come from `bind` because a LOAD
/// observed them, one is this build's stamp ([`QMM_TILE`]), and the rest are
/// the [`RowScalars`] a Metal text names and `LlamaLikeFacts` does not hold.
///
/// It takes no [`LlamaLikeFacts`] because `LlamaLikeMetalFacts` has no shape
/// vocabulary at all -- the Metal text is handed the row's facts directly
/// beside these (`llama_like_metal(f, &metal_facts(..), class)`) and reads
/// every extent off them. `cuda_facts` needs its own because its `tp_size`
/// NARROWS those extents, and the narrowed widths have to be stated where the
/// text will read them.
///
/// The zeros are STATED one at a time with their reason, and
/// [`LlamaLikeMetalFacts`] deliberately has no `Default` to fall through to:
/// a row is a MEASUREMENT of a real checkpoint, and "this one has no
/// per-layer embeddings" is part of the measurement.
///
/// [`LlamaLikeMetalFacts`]: super::forward::facts::LlamaLikeMetalFacts
#[must_use]
pub fn metal_facts(
    row: RowScalars,
    load: Deployed<'_>,
    bind: &MetalBinding,
) -> super::forward::facts::LlamaLikeMetalFacts {
    // The LOAD's, and unread here: `LlamaLikeMetalFacts` has no shard
    // field, so a Metal deployment is one rank by construction and
    // `trace` refuses a load that says otherwise ([`NO_METAL_SHARD`])
    // rather than letting this projection quietly drop the width. The
    // parameter stays for the reason `csm::project::deployment` keeps
    // its own: the day a Metal collective lands, this grows a body and
    // its callers do not move.
    let _ = load;
    super::forward::facts::LlamaLikeMetalFacts {
        // ── what the LOAD observed ──
        //
        // Six fields, and the shortness of this list is the same point
        // `Deployed`'s is. Three are about the kernels this build
        // compiled and three about the bytes this checkpoint shipped;
        // no row can state either kind.
        fuse_residual_gemv: bind.fuse_residual_gemv,
        paged_multi_batch: bind.paged_multi_batch,
        qmm_multi_batch: bind.qmm_multi_batch,
        add_bias: bind.add_bias,
        fused_qk_rope: bind.fused_qk_rope,
        // The checkpoint's own affine format. MLX stores the pair beside
        // the packed weight as `.scales` and `.biases`, which is a
        // zero-point layout, and the GROUP is asked of the load rather
        // than inferred from a tensor's shape because g64/b8 and g128/b4
        // pack to identical extents. A pipeline built for the wrong
        // point returns fluent nonsense rather than failing.
        proj_repr: model_dsl::WeightRepr::Scaled {
            layout: model_dsl::ScaleLayout::PerGroup,
            group: bind.quant_group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bind.quant_bits,
        // The ROUTER GATE's own format, when the checkpoint published it
        // wider than the stack it routes. `None` is "the same as the dense
        // projections", which is every checkpoint but gpt-oss's -- and
        // getting it wrong is the QUIET failure: a bank read at the wrong
        // format is 909,207 NaNs, and a gate read at the wrong width is a
        // fluent model routing every token to almost the right experts.
        router_repr: (bind.router_quant_group != 0).then_some(model_dsl::WeightRepr::Scaled {
            layout: model_dsl::ScaleLayout::PerGroup,
            group: bind.router_quant_group,
            axis: 0,
            zero_point: true,
        }),
        router_bits: bind.router_quant_bits,
        moe_repr: bind.moe_mxfp4.then_some(model_dsl::WeightRepr::Mxfp4Marlin),
        // FOUR by the format's own definition — MXFP4 is a four-bit
        // mantissa under a shared block exponent — so it is a constant
        // and not a second thing for the load to observe. It is read
        // only when `moe_repr` is `Some`, and stating it beside the
        // repr rather than folding it in is what `affine_bits` does one
        // field up, for the reason stated there: `WeightRepr::Scaled`
        // has nowhere to put a bit width.
        moe_bits: 4,
        // ── what this BUILD stamped ──
        // [`QMM_TILE`] unless this BUILD measured its own. See the
        // constant: the shared value was chosen against a
        // cooperative-matrix build, which is not every backend's situation,
        // and the escape hatch its doc names is this field.
        qmm_tile: bind.qmm_tile.unwrap_or(QMM_TILE),
        qmm_partial_rows: bind.qmm_partial_rows,
        // THE BUILD'S PERMISSION AND THE CODEC'S, CONJOINED. The codec test
        // alone is not enough: the staged symbol exists at one codec, but a
        // backend whose GEMM does not read back what the cast wrote is wrong
        // at that codec too, and `driver-wgpu` is one. See
        // `MetalBinding::qmm_fp16_precast`.
        qmm_fp16_precast: bind.qmm_fp16_precast
            && qmm_fp16_precast(bind.quant_group, bind.quant_bits),
        // False at the codec that would allow it. See the fact: this family's
        // routed checkpoint reordered a next-layer top-k under half rounding
        // in `llama_numerics_test`, and a reordered top-k is a different
        // model and not a tolerance. gemma-4 asks for true from its own
        // projector because its mixture did not move.
        routed_qmm_fp16: false,
        moe_tile: Some(ROUTED_QMM_TILE),
        // ── what the ROW states ──
        //
        // FALSE, and it is the row's answer rather than a policy: this
        // family's manifest states `mlp.gate_proj` and `mlp.up_proj` as
        // two tensors, so a checkpoint that matched it published two.
        // `silu_mul` takes gate and up as TWO buffers, so a text that
        // stated one packed value would bind the OUTPUT where `up`
        // belongs and leave the output unbound — a fire that runs. The
        // Metal text refuses the packed arm at trace time for exactly
        // that reason.
        gate_up_fused: false,
        // The row's epsilon and the row's rotary base, carried on
        // [`RowScalars`] because `LlamaLikeFacts` states neither — see
        // that struct for why a shape shared by twelve generations
        // holds no scalar a `Deployment` already carries, and why the
        // Metal text needs both anyway. They are the two numbers the
        // deleted `facts_from_with` read off a `DecodeGeometry` that had
        // read them off a `config.json`, which is a checkpoint
        // describing itself a second time.
        rms_eps: row.norm_eps,
        rope_theta: row.rope_theta,
        // ZERO, which is "one base for every layer". Two bases are
        // gemma's — `rope_parameters` gives `full_attention` 1e6 and
        // `sliding_attention` 1e4 — and a gemma states them through a
        // projection of its own. Two orders of magnitude apart is not a
        // near miss: the rotation is wrong from the second channel on
        // and compounds layer over layer.
        rope_theta_sliding: 0.0,
        // ZERO for all three, which is "one attention shape for the
        // whole stack". gemma-4's full layers are twice as wide per head
        // as its sliding ones, carry a quarter the KV heads and rotate
        // only 128 of their 512 channels; every row this projection
        // serves states ONE `head_dim` and ONE `kv_heads`, and its
        // manifest is that claim — `q_proj` is `[q_heads * head_dim,
        // hidden]` at every layer or the checkpoint is not this row.
        global_head_dim: 0,
        global_kv_heads: 0,
        full_partial_rotary: 0.0,
        // FALSE: every row here projects its own V. The manifest
        // requires `layer.{}.self_attn.v_proj` at every layer, so a
        // checkpoint that took V from K — gemma-4's full layers ship no
        // `v_proj` at all — could not have matched this row.
        v_from_k: false,
        // FALSE, and STATED rather than left off. `v_norm` is a norm
        // with NO WEIGHT, so there is no tensor to ask about it and
        // nothing in a checkpoint could ever have contradicted a wrong
        // answer here — every `has_tensor` question about it answers no,
        // correctly and uselessly. It is gemma-4's alone, and gemma-4
        // does not trace Metal at all; the deleted `facts_from_with`
        // carried it from a `DecodeGeometry` that had read it off a
        // `config.json`, which is the second reading this crate exists
        // to remove. `Deployment::v_norm` is where the answer lives now.
        v_norm: false,
        // FALSE: this family's mixture REPLACES the dense MLP rather
        // than sitting beside it. The manifest is that claim too, and it
        // is exclusive — a dense row states `mlp.gate_proj`, a routed
        // row states `mlp.gate` and the expert banks, and no row states
        // both. gemma-4 runs both branches off the post-attention
        // residual and adds them, which is five norms round one block.
        dense_beside_moe: false,
        // Every routed family but gemma-4 projects its router off the same
        // value its experts read and publishes no per-expert gain.
        router_input_norm: false,
        router_expert_scale: false,
        // The ROW's, and the one field of the routed arm that no bank, no
        // manifest and no load can be asked for: the weights are identical
        // either way and only their denominator differs.
        norm_topk_prob: row.norm_topk_prob,
        // FALSE: no row here ships a `layer_scalar`. It is gemma's, for
        // a deployment with no per-layer embeddings, and the driver
        // asked the TENSORS for it because gemma-4-31b states
        // `hidden_size_per_layer_input: 0` and has the scalar anyway —
        // "the gemma-shaped fields are populated" and "has a PLE" are
        // not the same question.
        per_layer_scalar: false,
        // ZERO, which is "scaled not at all". gemma multiplies its
        // gathered embeddings by `sqrt(hidden)`; nothing in this lineage
        // does, and the difference is not subtle — a gemma that got no
        // scale had a widest gathered value of 0.058 where MLX's
        // reference for the same snapshot is about seventy times that.
        embed_scale: 0.0,
        // ZERO, which is "derive `1/sqrt(head_dim)`" — and for this
        // family the derived answer IS the row's answer, so stating a
        // number here would only be a second place for it to be wrong.
        //
        // The head dim the Metal text derives it from is the
        // CHECKPOINT's, which is the right one and differs from the CUDA
        // side's on exactly one row: phi-3-mini's 96-wide heads run on
        // the 128-wide CUDA kernel and `deployment` scales by
        // `1/sqrt(128)` to match it, while Metal takes `head_dim` as an
        // operand and never pads. The two backends divide by different
        // numbers because they attend over different widths.
        attn_scale: 0.0,
        // ZERO: no side network. gemma's PLE is a second embedding table
        // gathered once per step, projected, normed and joined into
        // `[n_layers, ple_dim]` that each layer reads its own slice of,
        // and nothing llama-like has a counterpart.
        per_layer_emb_dim: 0,
        // ZERO: every layer of this family owns its pages. KV sharing is
        // gemma-4's tail, where a shared layer rotates its own Q and
        // reads the pages its source wrote — no k/v projection, no k/v
        // norm, no append — and suppressing those dispatches is not an
        // optimisation but which tensors the checkpoint ships. This
        // family's manifest requires `k_proj` and `v_proj` at every
        // layer, and `LoadShape::dense` states the same zero.
        kv_shared_layers: 0,
        // ZERO, which is "no softcap" and names nothing. gemma-2 caps
        // its readout at 30 and gemma-3 dropped it — `final_logit_
        // softcapping` is null in every published gemma-3 config — and
        // passing a cap so large it does nothing would be a kernel run
        // per fire to compute the identity.
        logit_softcap: 0.0,
        // FALSE: no row here ships `self_attn.sinks`. A sink is a
        // per-head learned logit that joins the softmax without a value
        // behind it, so a sinked attention normalizes over one more term
        // than it sums. It is gpt-oss's, and gpt-oss has a text of its
        // own.
        attn_sinks: false,
        // `silu_mul` — `silu(gate) * up` — every llama-lineage
        // deployment's. Three symbols exist rather than one with flags
        // because the difference is not cosmetic: gpt-oss clamps the
        // gate ABOVE only, clamps the linear branch both ways and adds
        // one to it, and gemma's gelu is the TANH approximation rather
        // than the erf one. Dropping any of that produces a model that
        // runs and is wrong — the two agree to about 2% at the origin
        // and diverge from there.
        activation: super::forward::facts::Activation::SiluMul,
        // Whether the driver hands over a frequency TABLE instead of a
        // base. True for llama-3's piecewise rescaling and OLMo-3's YaRN,
        // which no `rope_theta` expresses.
        rope_freq_table: row.rope_rescaled,
        // FALSE for every family this projection serves. The proportional
        // reading of a partial rotary is gemma-4's alone, and it is stated
        // where gemma-4's other rotations are -- `gemma_4::metal_facts`.
        rope_proportional: false,
        // The window, as the per-layer list the text reads through
        // `window_left_at`. EMPTY when the row attends the whole
        // context, because empty is what the accessor reads as `-1`
        // everywhere and a list of repeated `-1`s is a table walked for
        // nothing; ONE entry otherwise, because the accessor's last
        // entry covers the tail and this row's layers all agree.
        //
        // Stated, and not left empty the way `cuda_facts` leaves its
        // own: the CUDA driver carries a window per layer on its
        // `fwd_cfg` and the Metal text has only this. Mistral-7B-v0.3
        // and Qwen-2 are the rows it is not `-1` for, and a window
        // dropped is a model that attends its whole prefix — fluent, and
        // about a context the checkpoint was never trained to see.
        window_left: if row.window < 0 {
            Vec::new()
        } else {
            vec![row.window]
        },
    }
}

/// Trace this row's text for one fire class, on the backend that asked.
///
/// The two arms are two TEXTS over one row, which is the shape the
/// catalog was rewritten for: a model is a row, a backend is a
/// parameter of the question, and neither one is a string a driver
/// looks up. What stood here before was `driver-metal`'s eleven-entry
/// table of architecture NAMES — `llama`, `llama3`, `qwen3_moe`,
/// `gemma4` — reduced by a punctuation-stripping `canonical()` and
/// consulted before any text was traced. It was the third dispatch key,
/// and it disagreed with the other two exactly the way the first three
/// did: `gemma4` was listed as served by a driver that refused it on a
/// different ground, and `gemma3` was absent from a table whose text
/// models it.
///
/// # Errors
///
/// [`Refusal::Unsupported`](crate::deployment::Refusal) carrying
/// [`NO_METAL_SHARD`] when a Metal load says its weights were sharded,
/// [`NO_METAL_HEAD_DIM`] when the row's heads are a width the paged
/// decode shader has no instantiation for, or
/// [`NO_METAL_ROUTED_ENCODING`] when a routed bank arrived affine at
/// some point other than [`METAL_ROUTED_AFFINE`].
///
/// The last two are refusals about the KERNEL SET and not about the row,
/// which is why they are here rather than at a generation: twelve
/// families share this text, so a width or an encoding it cannot name is
/// a gap in the same one sentence for all of them.
pub fn trace(
    f: &LlamaLikeFacts,
    row: RowScalars,
    class: model_ir::trace::FireClass,
    load: Deployed<'_>,
) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
    match load.backend {
        // THE SHIPPED POINT, read off the shared text's one axis
        // spelling (`forward::Shipped*`, beside its CATALOG). One
        // catalogued pair of axes; the row's own `proj_repr` stays data
        // (see `forward::CATALOG`) — which is also why these rows'
        // manifests make no repr claim — and the coverage test is what
        // keeps the point loadable.
        Backend::Cuda => Ok(super::forward::llama_like_cuda::<
            super::forward::ShippedA,
            super::forward::ShippedKv,
        >(
            f,
            &cuda_facts(f, load),
            class,
            row.norm_eps,
            row.rope_theta,
        )),
        Backend::Metal(bind) => {
            let m = metal_facts(row, load, bind);
            metal_kernel_refusal(f, &m, load, bind)?;
            Ok(super::forward::llama_like_metal(f, &m, class))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The epsilon these projections are exercised at.
    ///
    /// One number for every test below, because none of them reads it:
    /// it is a fact about a CHECKPOINT that rides through `deployment`
    /// untouched, and a test that varied it would only be asserting that
    /// a field is copied. The rows' own values are held against their
    /// published configs by `tests/catalog_differential.rs`, which is
    /// where a transcribed number belongs.
    const NORM_EPS: f32 = 1e-6;

    /// A landing bias the normed arms would have dropped is refused.
    ///
    /// Two facts that are individually fine and jointly unserviceable:
    /// `o_bias` says the checkpoint publishes a bias on `o_proj`, and a
    /// non-`Pre` placement says the landing's output is normed, which
    /// routes it through the shared closure that has no bias in it. Only
    /// the fused arm adds one.
    ///
    /// The whole point of the refusal is that the alternative is
    /// SILENT: the tensor is declared, read, staged and never summed, so
    /// the model answers with a per-channel offset missing from every
    /// layer and nothing faults. No row in the catalog pairs them today
    /// -- gpt-oss publishes the bias and norms `Pre` -- which is exactly
    /// why the pairing needs a name now rather than a bug report later.
    #[test]
    fn a_landing_bias_under_a_normed_landing_is_refused_by_name() {
        use crate::deployment::Refusal;
        let bind = binding(64, 4);
        let metal = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        let mut f = LlamaLikeFacts::qwen3_0_6b();
        f.o_bias = true;
        for placement in [SpecNorm::Post, SpecNorm::Sandwich] {
            f.norm_placement = placement;
            assert_eq!(
                metal_kernel_refusal(&f, &metal, Deployed::metal(&bind), &bind),
                Err(Refusal::Unsupported(NO_METAL_NORMED_LANDING_BIAS)),
                "{placement:?} lands through the closure with no bias in it"
            );
        }
        // Either fact alone is serviceable, which is what makes the
        // pairing worth stating.
        f.norm_placement = SpecNorm::Pre;
        metal_kernel_refusal(&f, &metal, Deployed::metal(&bind), &bind)
            .expect("the fused arm adds the bias it publishes");
        f.o_bias = false;
        f.norm_placement = SpecNorm::Sandwich;
        metal_kernel_refusal(&f, &metal, Deployed::metal(&bind), &bind)
            .expect("a normed landing with no bias to drop is fine");
    }

    /// The build's instantiation set, as a rounding rule rather than a
    /// table each caller re-derives.
    #[test]
    fn a_head_dim_rounds_up_to_one_this_build_instantiates() {
        assert_eq!(round_up_attn_head_dim(64), 64);
        assert_eq!(round_up_attn_head_dim(96), 128, "phi-3's 96 pads to 128");
        assert_eq!(round_up_attn_head_dim(128), 128);
        // Nothing holds it, so the caller surfaces the dispatch error
        // rather than getting a silently mis-sized answer.
        assert_eq!(round_up_attn_head_dim(768), 768);
    }

    /// The four probes the old derivation made of the LOAD are four
    /// manifest rows, so two variants that differ only in them cannot
    /// both match one checkpoint.
    #[test]
    fn the_load_probes_became_expectations() {
        let per_head = manifest(&LlamaLikeFacts::qwen3_0_6b());
        let global = manifest(&LlamaLikeFacts::olmo2_1b());
        let names = |m: &Manifest| -> Vec<String> {
            m.tensors
                .iter()
                .map(|t| format!("{}:{:?}:{:?}", t.name, t.extents, t.presence))
                .collect()
        };
        assert_ne!(names(&per_head), names(&global));
    }

    /// olmo2's post-norm placement reaches the deployment, which is
    /// what `model_type.starts_with("olmo")` used to decide — a string
    /// test on the identity, for a fact the row states.
    #[test]
    fn norm_placement_is_stated_rather_than_matched_on_a_name() {
        let olmo = deployment(&LlamaLikeFacts::olmo2_1b(), row(500_000.0, -1));
        assert_eq!(olmo.norm, NormPlacement::Post);
        let qwen = deployment(&LlamaLikeFacts::qwen3_0_6b(), row(1e6, -1));
        assert_eq!(qwen.norm, NormPlacement::Pre);
    }

    /// The launch geometry is the row's own numbers, not a second
    /// reading of anything.
    #[test]
    fn the_launch_geometry_is_the_rows_own_numbers() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        let d = deployment(&f, row(1e6, -1));
        assert_eq!(d.shape.hidden, f.hidden);
        assert_eq!(d.shape.q_heads, f.q_heads);
        assert_eq!(d.shape.kv_heads, f.kv_heads);
        assert_eq!(d.shape.head_dim, f.head_dim);
        assert_eq!(d.shape.intermediate, f.intermediate);
        assert_eq!(d.shape.vocab, f.vocab);
        assert_eq!(d.shape.gqa_group(), 2, "16 q over 8 kv");
        assert_eq!(
            d.shape.head_dim_alloc(),
            128,
            "128 is instantiated; nothing pads"
        );
    }

    /// Phi-3's 96-wide heads run on the 128-wide kernel, and the
    /// difference is a WIDTH rather than a boolean: a buffer sized
    /// `heads * head_dim` is short by a third.
    #[test]
    fn a_padded_head_reaches_the_geometry_as_a_width() {
        let f = LlamaLikeFacts::phi3_mini();
        let d = deployment(&f, row(10_000.0, -1));
        assert_eq!(d.shape.head_dim, 96, "the checkpoint's own");
        assert_eq!(d.shape.head_dim_kernel, 128, "the one instantiated");
        assert_eq!(d.shape.head_dim_alloc(), 128, "the one to allocate");
        for a in &d.attention {
            assert_eq!(a.head_dim, 128, "attention is sized for the kernel");
        }
    }

    /// The GQA ratios a build's decode instantiates. Outside the set
    /// FlashInfer THROWS, and a throw crossing a C ABI is undefined
    /// behaviour — which is why the question is asked at the door.
    ///
    /// The SET is not stated here any more: it is the driver's
    /// (`driver_cuda::serve::DECODE_GQA_GROUPS`), because it describes what
    /// that build instantiated and not what this lineage is. What this crate
    /// owes the question is the RATIO, which is what is asserted.
    #[test]
    fn the_gqa_set_is_the_builds_and_not_the_familys() {
        let g = deployment(&LlamaLikeFacts::qwen3_0_6b(), row(1e6, -1))
            .shape
            .gqa_group();
        assert_eq!(g, 2, "qwen3-0.6b's 16 query heads over 8 kv heads");
        assert_eq!(Geometry::EMPTY.gqa_group(), 0, "no division by zero");
    }

    /// A sliding window reaches every layer, and full attention is -1.
    #[test]
    fn the_window_is_stated_per_layer() {
        let f = LlamaLikeFacts::mistral_7b_v03();
        let full = deployment(&f, row(1e6, -1));
        assert!(full.attention.iter().all(|a| a.window == -1));
        let windowed = deployment(&f, row(1e6, 4096));
        assert!(windowed.attention.iter().all(|a| a.window == 4096));
    }

    /// Every layer owns its pages here. KV sharing is gemma-4's, and it
    /// is a fact about a LAYER there rather than a family here.
    #[test]
    fn every_layer_owns_its_own_pages() {
        let d = deployment(&LlamaLikeFacts::qwen3_0_6b(), row(1e6, -1));
        for (l, a) in d.attention.iter().enumerate() {
            assert_eq!(a.kv_source, l as u32);
            assert_eq!(a.rope_theta, 1e6);
            assert_eq!(a.rotary_dim, 0, "full rotation at the head dim");
        }
        assert_eq!(d.kv, KvStyle::Paged);
        assert_eq!(d.prefill, PrefillStyle::Planned);
        assert_eq!(d.attn_output, AttnOutput::DriverPinned);
        assert_eq!(d.logit_softcap, 0.0);
        assert_eq!(d.ple_dim, 0);
        assert!(d.recurrent.is_none());
        assert!(d.scales.is_empty());
    }

    /// A tie is an ABSENCE, and the manifest says so — which is the only
    /// way tied and untied can be told apart when every extent agrees.
    #[test]
    fn a_tie_is_an_absence_the_manifest_expects() {
        use crate::manifest::Presence;
        let tied = manifest(&LlamaLikeFacts::qwen3_0_6b());
        let head = tied
            .tensors
            .iter()
            .find(|t| t.name == "lm_head")
            .expect("stated");
        assert_eq!(head.presence, Presence::Absent);

        let untied = manifest(&LlamaLikeFacts::phi3_mini());
        let head = untied
            .tensors
            .iter()
            .find(|t| t.name == "lm_head")
            .expect("stated");
        assert_eq!(head.presence, Presence::Required);
    }

    /// A mixture ships a router and a dense block does not, so the
    /// manifest tells them apart without reading a `model_type`.
    #[test]
    fn a_mixture_ships_a_router() {
        let dense = manifest(&LlamaLikeFacts::qwen3_0_6b());
        assert!(
            dense
                .tensors
                .iter()
                .any(|t| t.name.contains("mlp.gate_proj"))
        );
        assert!(!dense.tensors.iter().any(|t| t.name.ends_with("mlp.gate")));

        let moe = manifest(&LlamaLikeFacts::qwen3_30b_a3b());
        assert!(moe.tensors.iter().any(|t| t.name.ends_with("mlp.gate")));
        assert!(!moe.tensors.iter().any(|t| t.name.contains("mlp.gate_proj")));
    }

    /// Attention biases are a Qwen-2 fact, and the manifest expects the
    /// tensor rather than inferring it from a name.
    #[test]
    fn attention_biases_are_expected_when_the_row_says_so() {
        let with = manifest(&LlamaLikeFacts::qwen2_5_1_5b());
        assert!(with.tensors.iter().any(|t| t.name.ends_with("q_proj.bias")));
        let without = manifest(&LlamaLikeFacts::qwen3_0_6b());
        assert!(
            !without
                .tensors
                .iter()
                .any(|t| t.name.ends_with("q_proj.bias"))
        );
    }

    /// The q/k-norm question the old derivation answered by dividing a
    /// byte count is three distinct expectations.
    #[test]
    fn qk_norm_is_three_expectations_and_not_a_division() {
        use crate::manifest::Presence;
        let spec = |f: &LlamaLikeFacts| {
            manifest(f)
                .tensors
                .into_iter()
                .find(|t| t.name.ends_with("q_norm"))
                .expect("every row states it")
        };
        let per_head = spec(&LlamaLikeFacts::qwen3_0_6b());
        assert_eq!(per_head.presence, Presence::Required);
        assert_eq!(per_head.extents, vec![128]);

        let global = spec(&LlamaLikeFacts::olmo2_1b());
        assert_eq!(global.presence, Presence::Required);
        assert_ne!(global.extents, vec![128], "global is the projection width");

        let off = spec(&LlamaLikeFacts::mistral_7b_v03());
        assert_eq!(off.presence, Presence::Absent);
    }

    /// Two bindings of one checkpoint, which is what a `MetalBinding`
    /// is for: the g64/b4 publication and the g128/b8 one.
    fn binding(group: u32, bits: u32) -> MetalBinding {
        MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: group,
            quant_bits: bits,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
            fused_qk_rope: false,
        }
    }

    /// A row for the deployment tests, so they state one the way a
    /// generation does rather than four loose scalars.
    ///
    /// Ungated, because [`deployment`] is: these tests are the ones
    /// that hold a `contract`-only build honest.
    fn row(rope_theta: f32, window: i32) -> RowScalars {
        RowScalars {
            rope_theta,
            norm_eps: NORM_EPS,
            window,
            rope_rescaled: false,
            norm_topk_prob: true,
        }
    }

    /// qwen3-0.6b's row, as the generation states it — a full-attention
    /// dense stack on one rope base.
    fn qwen3_row() -> RowScalars {
        RowScalars {
            rope_theta: 1e6,
            norm_eps: 1e-6,
            window: -1,
            rope_rescaled: false,
            norm_topk_prob: true,
        }
    }

    /// THE claim this projection makes: the binding is the LOAD's
    /// answer and everything else is the ROW's.
    ///
    /// It is the whole of what `driver-metal`'s `facts_from_with`
    /// stopped having to do. That function filled twenty-six fields
    /// from a `DecodeGeometry` and nine tensor probes, and eighteen of
    /// them were facts the row already stated — the epsilon, the rotary
    /// base, the head counts, the activation, the softcap. Asking a
    /// checkpoint what it is, twice, is how the answers came to differ:
    /// a gemma read as a llama folded `(1 + w)` as `w` and dropped two
    /// norms per layer, and nothing faulted.
    #[test]
    fn the_metal_binding_is_the_loads_answer_and_the_rest_is_the_rows() {
        let bind = binding(64, 4);
        let m = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);

        // The LOAD's six, field for field.
        assert!(m.fuse_residual_gemv);
        assert!(m.paged_multi_batch);
        assert!(m.qmm_multi_batch);
        assert_eq!(m.affine_bits, 4, "the checkpoint's own bit width");
        assert_eq!(
            m.proj_repr,
            model_dsl::WeightRepr::Scaled {
                layout: model_dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            },
            "MLX stores `.scales` and `.biases`, which is a zero-point layout"
        );
        assert_eq!(m.moe_repr, None, "this load left no bank in MXFP4");

        // The ROW's, which no load observed.
        assert_eq!(m.rms_eps, 1e-6);
        assert_eq!(m.rope_theta, 1e6);
        assert!(!m.rope_freq_table, "a plain geometric ladder");
        assert!(m.window_left.is_empty(), "qwen3 attends its whole prefix");
        assert_eq!(
            m.activation,
            super::super::forward::facts::Activation::SiluMul
        );
        assert!(
            !m.gate_up_fused,
            "the manifest states gate and up as two tensors"
        );
    }

    /// The same row under a DIFFERENT publication of the same weights.
    ///
    /// `mlx-community` ships the 4-bit and the 8-bit build of one
    /// checkpoint and they differ in nothing else, which is why an
    /// encoding is a policy and not an identity — the module doc's rule,
    /// asked of the one projection that has to hold both.
    #[test]
    fn a_second_publication_of_one_row_moves_only_the_binding() {
        let four = binding(64, 4);
        let eight = binding(128, 8);
        let a = metal_facts(qwen3_row(), Deployed::metal(&four), &four);
        let b = metal_facts(qwen3_row(), Deployed::metal(&eight), &eight);

        assert_ne!(a.affine_bits, b.affine_bits);
        assert_ne!(
            a.proj_repr, b.proj_repr,
            "g64/b8 and g128/b4 pack identically"
        );
        // And nothing the ROW states moved, because nothing about the
        // model did.
        assert_eq!(a.rms_eps, b.rms_eps);
        assert_eq!(a.rope_theta, b.rope_theta);
        assert_eq!(a.window_left, b.window_left);
        assert_eq!(a.activation, b.activation);
        assert_eq!(a.qmm_tile, b.qmm_tile);
    }

    /// A second row, so the semantic half is read off THAT row and not
    /// off a constant that happens to match qwen3's.
    ///
    /// Mistral-7B-v0.3 is the sharpest one available: it states a
    /// different rotary base, a different epsilon and — unlike every
    /// other row this projection serves — a WINDOW.
    #[test]
    fn a_second_row_states_its_own_window_and_its_own_base() {
        let f = LlamaLikeFacts::mistral_7b_v03();
        let bind = binding(64, 4);
        let row = RowScalars {
            rope_theta: 1e6,
            norm_eps: 1e-5,
            window: 4096,
            rope_rescaled: false,
            norm_topk_prob: true,
        };
        let m = metal_facts(row, Deployed::metal(&bind), &bind);

        assert_eq!(m.rms_eps, 1e-5, "mistral's, not qwen3's");
        assert_eq!(m.window_left, vec![4096]);
        // One entry covers the tail, which is what the accessor means by
        // a list shorter than the layer count — every mistral layer
        // slides the same 4096.
        for l in 0..f.layers {
            assert_eq!(m.window_left_at(l), 4096, "layer {l}");
        }
        // And the row with no window states none, which the accessor
        // reads as the whole context rather than as a zero-width one.
        let full = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        assert_eq!(full.window_left_at(0), -1);
    }

    /// The SECOND head width — the one only a two-geometry row has.
    ///
    /// `metal_kernel_refusal` is shared, and the sibling above only ever
    /// reaches its first head-dim gate because every row this projection
    /// serves states one attention shape. gemma-4 is the only caller
    /// that arrives with a `global_head_dim` at all, and every gemma-4
    /// row states 512, which the shader does instantiate — so the second
    /// gate has never once answered.
    ///
    /// It is the gate that matters more, not less: a row whose SLIDING
    /// layers are a compiled width and whose FULL layers are not would
    /// pass the first check and be admitted, then die inside the trace
    /// with `model-compiler`'s panic on an undeclared launch — a
    /// backtrace two thirds of the way through a load rather than a
    /// refusal naming the row.
    ///
    /// Asserted through `metal_kernel_refusal` directly because no
    /// projection in this module can produce a nonzero `global_head_dim`
    /// to feed it, which is the very reason the gate went untried.
    #[test]
    fn a_full_layer_width_no_metal_shader_compiled_is_refused_too() {
        use super::super::forward::facts::LlamaLikeMetalFacts;
        use crate::deployment::Refusal;
        let bind = binding(64, 4);
        let f = LlamaLikeFacts::qwen3_0_6b();
        assert!(
            METAL_SDPA_HEAD_DIMS.contains(&f.head_dim),
            "the sliding width must PASS, or this would be the sibling's test"
        );
        let base = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        assert_eq!(
            metal_kernel_refusal(&f, &base, Deployed::metal(&bind), &bind),
            Ok(()),
            "one attention shape is admitted"
        );
        for w in [96, 80, 192] {
            let m = LlamaLikeMetalFacts {
                global_head_dim: w,
                ..base.clone()
            };
            assert_eq!(
                metal_kernel_refusal(&f, &m, Deployed::metal(&bind), &bind),
                Err(Refusal::Unsupported(NO_METAL_HEAD_DIM)),
                "d_{w} full layers are not instantiated and must be refused"
            );
        }
        for w in METAL_SDPA_HEAD_DIMS {
            let m = LlamaLikeMetalFacts {
                global_head_dim: *w,
                ..base.clone()
            };
            assert_eq!(
                metal_kernel_refusal(&f, &m, Deployed::metal(&bind), &bind),
                Ok(()),
                "d_{w} full layers are instantiated and must not be refused"
            );
        }
    }

    /// The gemma-shaped fields are STATED zero, one at a time.
    ///
    /// `LlamaLikeMetalFacts` has no `Default` impl and this projection
    /// spells no `..Default::default()`, for the reason the fixtures
    /// give: a row is a MEASUREMENT of a real checkpoint, and "this one
    /// has no per-layer embeddings" is part of the measurement. This
    /// test is that discipline held rather than described — it fails if
    /// a future field is defaulted into existence.
    #[test]
    fn the_gemma_shaped_facts_are_stated_rather_than_defaulted() {
        let bind = binding(64, 4);
        let m = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        assert_eq!(m.rope_theta_sliding, 0.0, "one base for every layer");
        assert_eq!(m.global_head_dim, 0, "one attention shape");
        assert_eq!(m.global_kv_heads, 0);
        assert_eq!(m.full_partial_rotary, 0.0);
        assert!(!m.v_from_k, "every layer here projects its own V");
        assert!(!m.dense_beside_moe, "the mixture replaces the dense MLP");
        assert!(!m.per_layer_scalar);
        assert_eq!(m.embed_scale, 0.0, "gemma's `sqrt(hidden)` is gemma's");
        assert_eq!(m.attn_scale, 0.0, "derive `1/sqrt(head_dim)`");
        assert_eq!(m.per_layer_emb_dim, 0, "no side network");
        assert_eq!(m.kv_shared_layers, 0, "every layer owns its pages");
        assert_eq!(m.logit_softcap, 0.0);
        assert!(!m.attn_sinks);
        // The accessors read those zeros as "one shape everywhere",
        // which is the reading the text depends on.
        assert_eq!(m.head_dim_at(0, 128), 128);
        assert_eq!(m.kv_heads_at(0, 8), 8);
        assert_eq!(m.rotary_dim_at(0, 128), 128, "the whole head rotates");
        assert_eq!(m.rope_theta_at(0), 1e6);
    }

    /// The tile is the BUILD's stamp and the two bits that are not.
    ///
    /// A symbol whose tile is wrong reads the wrong bytes; a symbol with
    /// no tile does not resolve at all, which is the better failure —
    /// so a `(0, 0)` here would be a fixture written before the field
    /// existed and not a deployment that wants no tile.
    #[test]
    fn the_gemm_tile_is_the_builds_stamp_and_not_the_rows() {
        let bind = binding(64, 4);
        let m = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        assert_eq!(QMM_TILE, (32, 32));
        assert_eq!(m.qmm_tile, QMM_TILE);
        // A rung `TILE_M` states in all three kernel trees, and the only
        // column tile the residual variant carries. Not the narrowest rung:
        // that is 16, and it is 4.5x slower on a 1024-token prefill for the
        // reason the constant's own comment measures.
        assert_ne!(m.qmm_tile, (0, 0), "a stem does not resolve");
    }

    /// The partial-row tolerance is the BUILD's stamp too, and both arms
    /// reach the facts.
    ///
    /// This is the statement a fixture pair would have made, and it is here
    /// rather than there for the same reason `qmm_tile` above is: the answer
    /// is a fact about which kernels were compiled, so it does not vary by
    /// model family and three family fixtures would state it identically.
    /// `facts::tests::every_metal_predicate_is_stated_more_than_one_way_or_
    /// excused` excuses it by name and points here.
    ///
    /// FALSE is the default and the one that matters: a build that says
    /// nothing gets the contract `qmm_t` is written to, where the caller
    /// allocates a whole number of `BM` rows. Only `driver-wgpu` says true,
    /// and only because its `write_out` returns on `row >= params.m`.
    #[test]
    fn the_partial_row_tolerance_is_the_builds_stamp_and_not_the_familys() {
        let mut bind = binding(64, 4);
        assert!(
            !bind.qmm_partial_rows,
            "a binding that says nothing does not tolerate a partial tile"
        );
        let off = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        assert!(!off.qmm_partial_rows);
        bind.qmm_partial_rows = true;
        let on = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        assert!(
            on.qmm_partial_rows,
            "the binding's answer reaches the facts, or the guard cannot read it"
        );
    }

    /// The expert bank states its own format only when the load left it
    /// in MXFP4.
    ///
    /// Reading a bank with the dense format is not a near miss: every
    /// scale comes from the wrong offset, and the fire that did it
    /// produced 909,207 NaNs beginning at the first routed projection of
    /// layer 0. `None` is "the same as the dense projections", which is
    /// every checkpoint this family serves.
    #[test]
    fn the_expert_bank_names_its_own_format_only_when_the_load_left_one() {
        let plain = binding(64, 4);
        let mixed = MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            moe_mxfp4: true,
            ..binding(64, 4)
        };

        let uniform = metal_facts(qwen3_row(), Deployed::metal(&plain), &plain);
        assert_eq!(uniform.moe_repr, None);

        let split = metal_facts(qwen3_row(), Deployed::metal(&mixed), &mixed);
        assert_eq!(split.moe_repr, Some(model_dsl::WeightRepr::Mxfp4Marlin));
        assert_eq!(
            split.moe_bits, 4,
            "MXFP4 is four bits by the format's own name"
        );
    }

    /// A rescaled ladder reaches the text as a TABLE, not as a base.
    ///
    /// llama-3's piecewise rescaling and OLMo-3's YaRN. No `rope_theta`
    /// expresses either, so a text that stated one would rotate by the
    /// wrong frequencies from the second channel on, at every position
    /// but zero — degrading rather than failing, which is the shape of
    /// defect this catalog exists to make impossible.
    #[test]
    fn a_rescaled_ladder_is_a_table_and_a_plain_one_is_a_base() {
        let bind = binding(64, 4);
        let plain = metal_facts(qwen3_row(), Deployed::metal(&bind), &bind);
        assert!(!plain.rope_freq_table);

        let rescaled = RowScalars {
            rope_rescaled: true,
            ..qwen3_row()
        };
        let m = metal_facts(rescaled, Deployed::metal(&bind), &bind);
        assert!(m.rope_freq_table, "the driver hands over a table instead");
    }

    /// One row, two backends, two texts — and neither is reached by a
    /// name in a table.
    ///
    /// This is the guard that replaces `driver-metal`'s `LLAMA_LIKE`
    /// list of eleven architecture strings and its `canonical()`
    /// reduction. The row is the dispatch; the backend is a parameter of
    /// the question.
    #[test]
    fn one_row_traces_on_either_backend_and_says_which() {
        use model_ir::trace::FireClass;
        let f = LlamaLikeFacts::qwen3_0_6b();
        let bind = binding(64, 4);
        for class in [FireClass::Prefill, FireClass::Decode] {
            let cuda = trace(&f, qwen3_row(), class, Deployed::single())
                .expect("the CUDA text is written");
            let metal = trace(&f, qwen3_row(), class, Deployed::metal(&bind))
                .expect("and so is the Metal one");
            assert!(
                cuda.family.starts_with("llama_like-kv-bf16.cuda."),
                "{}",
                cuda.family
            );
            assert!(
                metal.family.starts_with("llama_like.metal."),
                "{}",
                metal.family
            );
            assert_ne!(cuda.ops.len(), 0);
            assert_ne!(metal.ops.len(), 0);
        }
    }

    /// A SHARDED Metal load is refused rather than traced at full width.
    ///
    /// The CUDA facts carry a `tp_size` and the text divides every
    /// projection width by it; the Metal facts carry nothing, so the
    /// same trace would state the whole model's widths against one
    /// rank's slice of the weights. That is not a crash — it is a
    /// projection reading past the end of its own tensor — so the door
    /// says no.
    #[test]
    fn a_sharded_metal_load_is_refused_rather_than_traced_at_full_width() {
        use crate::deployment::Refusal;
        use model_ir::trace::FireClass;
        let f = LlamaLikeFacts::qwen3_0_6b();
        let bind = binding(64, 4);
        let sharded = Deployed {
            backend: Backend::Metal(&bind),
            tp_size: 4,
            layer_scalars: &[],
        };
        let err = trace(&f, qwen3_row(), FireClass::Decode, sharded)
            .expect_err("four ranks and no shard vocabulary");
        assert_eq!(err, Refusal::Unsupported(NO_METAL_SHARD));
        // One rank is the shape of every Metal deployment there is, and
        // it traces.
        assert!(trace(&f, qwen3_row(), FireClass::Decode, Deployed::metal(&bind)).is_ok());
        // The CUDA side shards and keeps tracing: `tp_size` is a fact
        // its own facts carry.
        let cuda = Deployed {
            backend: Backend::Cuda,
            tp_size: 4,
            layer_scalars: &[],
        };
        assert!(trace(&f, qwen3_row(), FireClass::Decode, cuda).is_ok());
    }

    /// A head width no shader compiled is refused, not named.
    ///
    /// phi-3's heads are 96 and `sdpa_paged.metal` instantiates the paged
    /// decode at 64, 128, 256 and 512. The text spells the width into the
    /// symbol -- deliberately, since the `_d_256` literal was a real
    /// defect -- so 96 produced `sdpa_paged_decode_bfloat16_d_96`.
    ///
    /// Nothing caught that at the call, because nothing got there:
    /// `model-compiler` validates a declaration against the kernel rows
    /// and PANICS on an undeclared launch, so `driver-metal`'s
    /// catalog-wide binding gate died with a backtrace instead of
    /// reporting a row. The refusal is what the doc always promised --
    /// "a width no kernel instantiates simply does not resolve, and the
    /// driver's row check reports it by name" -- said early enough to be
    /// a sentence.
    #[test]
    fn a_head_width_no_metal_shader_compiled_is_refused_by_name() {
        use crate::deployment::Refusal;
        use model_ir::trace::FireClass;
        let bind = binding(64, 4);
        let phi3 = LlamaLikeFacts::phi3_mini();
        assert_eq!(phi3.head_dim, 96, "the fixture this test reads");
        for class in [FireClass::Prefill, FireClass::Decode] {
            let err = trace(&phi3, qwen3_row(), class, Deployed::metal(&bind))
                .expect_err("96 is not a point sdpa_paged.metal has");
            assert_eq!(err, Refusal::Unsupported(NO_METAL_HEAD_DIM));
        }
        // CUDA pads to `head_dim_kernel` and strips the pad back off, so
        // the same row keeps tracing there. The refusal is about this
        // build's SHADERS and nothing about phi-3.
        assert!(trace(&phi3, qwen3_row(), FireClass::Decode, Deployed::single()).is_ok());
        // And every width the shader does have still traces.
        for w in METAL_SDPA_HEAD_DIMS {
            let mut f = LlamaLikeFacts::qwen3_0_6b();
            f.head_dim = *w;
            assert!(
                trace(&f, qwen3_row(), FireClass::Decode, Deployed::metal(&bind)).is_ok(),
                "d_{w} is instantiated and must not be refused"
            );
        }
    }

    /// A routed bank at an affine point the routed matvec has no
    /// instantiation for is refused.
    ///
    /// `AffineQ::group_size` is a template constant, so
    /// `affine_qmv_routed` exists at group 64 / 4 bits and nowhere else.
    /// The symbol takes the repr the caller states -- which is right, and
    /// is what fixed gpt-oss's MXFP4 bank being read as affine-64 -- so a
    /// g128/b8 bank named `affine_qmv_routed_bfloat16_gs_128_b_8`, which
    /// no shader exports.
    ///
    /// This is the one refusal in this file that moves with the LOAD
    /// rather than the row, which is why `driver-metal`'s
    /// `a_row_is_served_the_same_way_at_every_encoding` had to become
    /// one-directional: a pre-staging probe may be permissive, it may not
    /// be wrong.
    #[test]
    fn a_routed_bank_at_an_uninstantiated_affine_point_is_refused() {
        use crate::deployment::Refusal;
        use model_ir::trace::FireClass;
        let moe = LlamaLikeFacts::qwen3_30b_a3b();
        assert!(moe.n_experts > 0, "the fixture this test reads");

        let (g, b) = METAL_ROUTED_AFFINE;
        let good = binding(g, b);
        assert!(trace(&moe, qwen3_row(), FireClass::Decode, Deployed::metal(&good)).is_ok());

        let bad = binding(128, 8);
        let err = trace(&moe, qwen3_row(), FireClass::Decode, Deployed::metal(&bad))
            .expect_err("group 128 has no routed instantiation");
        assert_eq!(err, Refusal::Unsupported(NO_METAL_ROUTED_ENCODING));

        // An MXFP4 bank is a different codec with its own symbol at group
        // 32, so the affine point is not asked of it.
        let mut mxfp4 = binding(128, 8);
        mxfp4.moe_mxfp4 = true;
        assert!(
            trace(
                &moe,
                qwen3_row(),
                FireClass::Decode,
                Deployed::metal(&mxfp4)
            )
            .is_ok()
        );

        // A DENSE row is not asked either: the point governs the expert
        // bank, and a row with no bank has no stake in it.
        let dense = LlamaLikeFacts::qwen3_0_6b();
        assert_eq!(dense.n_experts, 0);
        assert!(
            trace(
                &dense,
                qwen3_row(),
                FireClass::Decode,
                Deployed::metal(&bad)
            )
            .is_ok()
        );
    }
}
#[cfg(test)]
mod fp16_precast_tests {
    /// Both arms of the staged-GEMM predicate, which no fixture pair states.
    ///
    /// See `LlamaLikeMetalFacts::qmm_fp16_precast`'s entry in `EXCUSED`: every
    /// Metal row this catalog ships is g64/b4, so the three fact fixtures
    /// agree and the census cannot see the false arm. It is reachable —
    /// `affine_qmm_t_fp16_precast` is stamped at ONE codec — and this is
    /// where that is written down.
    #[test]
    fn the_staged_gemm_is_the_g64_b4_codecs_alone() {
        assert!(super::qmm_fp16_precast(64, 4), "the stamped codec");
        // The 8-bit affine snapshots `mlx-community` publishes of these same
        // rows. `instantiate_qmm_t_fp16_precast` takes a tile and nothing
        // else, so there is no `_b_8` precast symbol to name.
        assert!(
            !super::qmm_fp16_precast(64, 8),
            "no 8-bit precast is stamped"
        );
        // g128/b4 and g32/b4 pack to the same extents as g64 and would
        // resolve to a symbol that reads every scale from the wrong offset.
        assert!(!super::qmm_fp16_precast(128, 4));
        assert!(!super::qmm_fp16_precast(32, 4));
    }
}
