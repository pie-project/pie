use checkpoint::contract::ModelContract;
use model_dsl::{Dtype, Weight};

use crate::contract::{Claim, ModelError, claim, elaborate};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// **THE ONE READING EVERY ATTENTION SITE OF THIS FAMILY IS CARVED FOR**
    /// (build log 21). An attention schedule is carved for a reading — query
    /// heads, kv heads, per-head width — so the reading is stated ONCE, here,
    /// and not once per site: the trunk's full-attention layers and the draft
    /// head's own block are built by one `gated_attn` from these very numbers
    /// (`mtp.layers.0.self_attn.*` has a trunk layer's shapes tensor for
    /// tensor), and this family has no second reading for a layer to carry.
    /// `forward` states them on the plan ops it builds, and every launch that
    /// reads one of those plans restates its share of them.
    ///
    /// PER RANK, both head counts: `Model::new` cuts them by `tp`, which is
    /// what the banks are cut by and therefore what a rank's own schedule
    /// reads. `head_dim` is not cut — a head is whole wherever it lands.
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    /// The adapter banks this family seats (palo design §8). Per layer, and
    /// the same two numbers at every one of them: the correction is a
    /// per-lane axis, not a per-layer one.
    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Head,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,

    /// **THE VISION TOWER, WHEN THE CHECKPOINT SHIPS ONE** (multimodal §0,
    /// campaign M-1/M-2).
    ///
    /// `Option` for [`mtp`](Model::mtp)'s reason and one more. It is a fact
    /// about the ARTIFACT — Qwen3.5-0.8B publishes a twelve-block
    /// `model.visual.*` and Qwen3.6-27B a twenty-seven-block one, and the
    /// 4-bit conversions of both publish neither — and it is the whole of what
    /// makes this plan a TWO-UNIT one: the tower's rectangles are
    /// `Dim::Patches`, `model_compiler::unit` reads the axis off the shapes,
    /// and a `None` here is a plan with one capture unit and not one node of a
    /// second.
    ///
    /// **AND IT IS WHY THE TRUNK'S ROTATION IS A PROPERTY OF THIS FIELD.** A
    /// lane carrying an image needs the three-section mrope both qwen SKUs'
    /// `text_config` states; a lane that cannot carry one never has a position
    /// that is not `(p, p, p)`, and `elemwise.rope_mrope` over three copies of
    /// a scalar is `rope_partial` to the last bit. So a text-only row keeps
    /// the scalar rotation and its artifact is the artifact it was — which is
    /// what G4 and `the_new_axes_cost_the_old_words_nothing` are pinned on —
    /// and a tower row states the triple once, here, for every layer.
    pub tower: Option<Tower>,

    /// **THE DRAFT HEAD, WHEN THE ARTIFACT CARRIES ONE** (palo C3, design §8;
    /// campaign M-4).
    ///
    /// `Option` because it is a fact about the ARTIFACT and not about the
    /// family: Qwen3.6-27B publishes `mtp.*` — one `fc`, two pre-norms, one
    /// full transformer block and a final norm, fifteen tensors, verified
    /// name-for-name against the cached checkpoint index — and no earlier
    /// qwen35 SKU publishes anything of the kind. A `None` here is not a
    /// disabled feature; it is a model that has no second head, and its trace
    /// carries not one node of the arm.
    ///
    /// **AND "ONE" IS NOW TWO RECIPES OF ONE DECLARATION** ([`Recipe`]). An
    /// EAGLE head arrives in a SECOND checkpoint rather than in the base one,
    /// and what it is once it arrives is this same shape with two pieces
    /// missing. So the axis did not grow a second field, a second fact bit, a
    /// second seam or a second kv row — it grew two `Option`s.
    pub mtp: Option<Mtp>,
}

/// One MTP (multi-token-prediction / NEXTN) head: the fusion of a hidden state
/// with the next token's embedding, one transformer block over the fused
/// stream, and a readout through the base model's own `lm_head`.
///
/// **THE ALGEBRA, AS THE CHECKPOINT AND THE DEV LINEAGE BOTH STATE IT.** Read
/// off `mtp.*` in `Qwen3.6-27B` and off `qwen3_5_mtp_forward`
/// (`origin/dev:driver/cuda/src/model/qwen3_5/qwen3_5_forward.cpp`):
///
/// ```text
/// h = fc · [ rms(embed(tok)) · Wₑ | rms(hidden) · W_h ]
/// h += attn(rms(h))            one gated full-attention block, own kv row
/// h += mlp(rms(h))             the family's own dense SwiGLU
/// draft = lm_head(rms(h))      the BASE head — `mtp_use_dedicated_embeddings`
///                              is false and no `mtp.lm_head` is published
/// ```
///
/// **`fc` IS DECLARED AS TWO BANKS AND THE CHECKPOINT SHIPS ONE.** The stored
/// `mtp.fc.weight` is `[hidden, 2·hidden]`, multiplying a row-concatenation of
/// the two normed streams; `[a|b]·[Wₑ|W_h]ᵀ = a·Wₑᵀ + b·W_hᵀ` exactly, so the
/// text states two `[hidden, hidden]` banks and one `residual_add` instead of
/// a concatenation. That is not a convenience: **the IR has no concat op**,
/// and adding one would be a variant every shell's `Dispatch` would have to
/// grow an arm for. The import contract slices the stored bank at column
/// `hidden` and the halves are named in the order dev concatenates them —
/// embedding first, hidden second (`launch_concat_bf16_rows(ws.q /* normed
/// embedding */, ws.y /* normed hidden */, ...)`). The one thing this costs is
/// a rounding: two fp32 accumulations summed in the output dtype instead of
/// one accumulation over 2·hidden. On a DRAFT whose every token is verified by
/// the target model, that is a proposal that may differ, never an answer that
/// may be wrong.
///
/// # And the same eight sentences are EAGLE's ([`Recipe::Eagle`], M-4)
///
/// An EAGLE head is this head with two pieces absent. Its own recipe fuses the
/// RAW embedding with the RAW hidden — no pre-fusion norms — and reads out
/// through the base `lm_head` with no final norm of its own, because the
/// hidden it was trained against is the one the trunk's `final_norm` already
/// produced. Everything between is identical: one `fc` over a concatenation,
/// one prenorm block with its own kv row, one readout through the base head.
///
/// So the declaration is one declaration with two `Option`s, and NOT a second
/// struct. What a second struct would have bought is a second name for the
/// same fifteen-minus-three planes; what it would have cost is a second arm in
/// `forward`, a second seam, a second fact bit and a second kv row — four
/// engine-side spellings of `mtp` that are the contract, not the vocabulary
/// (`seam::MTP`, `kv.mtp`, `Facts::drafts`, `IntrinsicId::MtpLogits`). This
/// type keeps the MTP name for exactly that reason: it is the name the ENGINE
/// knows the axis by, and a text that renamed it would be renaming a contract
/// it does not own.
pub struct Mtp {
    /// Which recipe this head was trained under — the one thing that is not
    /// derivable from the planes, because it is what says which planes exist.
    pub recipe: Recipe,
    /// The two pre-fusion norms, or `None` for a recipe that fuses the raw
    /// streams.
    pub pre_fc: Option<PreFc>,
    /// The embedding half of the stored `[hidden, 2·hidden]` fusion bank.
    pub fc_embed: Weight,
    /// The hidden half of it.
    pub fc_hidden: Weight,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    /// The head's own block. Full attention with the family's q-gate, its own
    /// kv row in the model's one page-id space — `mtp.layers.0.self_attn`,
    /// shapes identical to a trunk attention layer's.
    pub attn: Attn,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    /// Dense SwiGLU at the trunk's own intermediate width.
    pub mlp: Mlp,
    /// The final norm before the readout, or `None` for a recipe that reads
    /// out of the block directly.
    pub norm: Option<Weight>,
    pub norm_eps: f32,
}

/// The two pre-fusion norms, when the recipe has them. One epsilon for the
/// pair, as every norm in this family shares one: `rms_norm_eps` is a single
/// number in the config, and stating it per site is what keeps a site from
/// silently inheriting another's.
pub struct PreFc {
    /// Scales the embedding of the row's token before the fusion.
    pub embedding: Weight,
    /// Scales the trunk's hidden state before the fusion.
    pub hidden: Weight,
    pub eps: f32,
}

/// **WHICH DRAFT-HEAD RECIPE AN ARTIFACT CARRIES** (campaign M-4).
///
/// Not a flag on a shape — the two recipes have DIFFERENT PLANES, under
/// different names, and this is what says which. It is read by
/// `Model::new` (which pieces to declare, and under which prefix) and by
/// `import` (which stored tensors to bind them to), and by nothing at fire
/// time: once the trace is written, a head is a head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recipe {
    /// **THE CHECKPOINT'S OWN.** Fifteen `mtp.*` tensors published inside the
    /// base artifact, pre-fusion norms and a final norm included. Qwen3.6-27B
    /// is the one shipping SKU that carries them.
    Mtp,
    /// **AN OVERLAY.** A separately-obtained head, baked into the artifact
    /// beside the base by `pie model import --aux` under an `aux.` prefix, and
    /// therefore under a prefix that cannot collide with anything a base
    /// checkpoint publishes. No pre-fusion norms, no final norm.
    ///
    /// **THE PREFIX IS THE WHOLE OF THE OVERLAY CONTRACT.** `--aux` is
    /// family-blind — it copies a second checkpoint's tensors into the same
    /// `.zt` with every name prefixed — so what makes those bytes a draft head
    /// rather than a second model is this text naming them, exactly as the
    /// base's own names are the text's to name.
    Eagle,
}

impl Recipe {
    /// The plane prefix this recipe's head is named under.
    #[must_use]
    pub fn prefix(self) -> &'static str {
        match self {
            Recipe::Mtp => "mtp",
            Recipe::Eagle => "aux",
        }
    }
}

/// **THE SECOND ROW AXIS'S MODEL** (multimodal §0–§2): a windowed region of the
/// SAME fire whose rows are PATCHES and whose lanes are IMAGES.
///
/// # The algebra, transcribed
///
/// Off `Qwen3_5VisionModel.forward` and `Qwen3_5VisionBlock`
/// (`transformers/models/qwen3_5/modeling_qwen3_5.py`, v5.15.1), which the two
/// checkpoints' own tensors agree with plane for plane:
///
/// ```text
/// y  = patch_embed(x) + pos_embed[interp(grid)]     one GEMM, one gather
/// per block:
///   y += proj(dense_attn(rope(qkv(LN(y)))))         bidirectional, per image
///   y += fc2(gelu(fc1(LN(y))))                      UNGATED gelu_pytorch_tanh
/// out = mfc2(gelu(mfc1(merge(LN(y)))))              4 rows in, 1 row out
/// ```
///
/// # Four sentences that are not obvious, and what settles each
///
/// **THE NORMS ARE `nn.LayerNorm`, AND THE TEXT SAYS THE WHOLE OF ONE**
/// (multimodal §6.1, §9.1; next.md B5). The checkpoints publish `norm1.bias`
/// beside `.weight`, which an RMSNorm has none of. The fold into the
/// following GEMM that §6.1 proposed is HALF expressible (`Expr::Scale` can
/// scale a bank, `Expr::Bias` cannot add `b·Mᵀ`) and the two halves do not
/// compose, so the text says the whole norm in ops and the import contract
/// stays a copy. `LN` above is `elemwise.layernorm` — one node, centred and
/// scaled and biased. It was three (`layernorm_no_scale`, then an
/// `rmsnorm` that normalized nothing and only read the weight, then
/// `add_bias`) until B5 fused them: 25 norms a tower fire, 75 launches down
/// to 25.
///
/// **THE MLP IS UNGATED** (§6.2): `linear_fc2(act(linear_fc1(x)))` with
/// nothing to multiply, which is `linear.mlp_gelu_tanh` and not a geglu.
///
/// **THE ROTATION IS TWO-AXIS AND BLOCK-LAID** (§6.3, §7.2).
/// `Qwen3_5VisionRotaryEmbedding(head_dim / 2)` builds `head_dim/4`
/// frequencies and `freqs[pos_ids].flatten(1)` indexes that ONE ladder once
/// per axis, so each section restarts it — [`MropeForm::Blocked`] — and the
/// tower has no time axis, which it states as `sections[0] == 0` rather than
/// as a two-wide position stream. `theta` is 10 000: the class default, and
/// neither SKU's `vision_config` overrides it.
///
/// **AND IT IS REPLICATED UNDER `tp`, EVERY PLANE.** A tower is twelve or
/// twenty-seven blocks of 768 or 1152 — under a gibibyte against the trunk's
/// tens — and cutting it would put a collective inside the patch unit for a
/// rectangle every rank can hold whole. What the ranks then duplicate is the
/// tower's arithmetic; what they save is a reduce per block per image.
///
/// [`MropeForm::Blocked`]: model_dsl::MropeForm::Blocked
pub struct Tower {
    /// The tower's own residual width — NOT the trunk's.
    pub hidden: u32,
    pub heads: u32,
    /// `hidden / heads`. Stated rather than derived at each site for the
    /// reason `Model::head_dim` is: an attention schedule is carved for a
    /// reading, and the reading is said once.
    pub head_dim: u32,
    /// `spatial_merge_size`. The merger folds `merge²` consecutive patch rows
    /// into one, which is `layout.merge_rows` and is why the submission's
    /// merge-block-major ordering is a statute (§2).
    pub merge: u32,
    /// `C · T · P²` — the width of one PRE-UNFOLDED patch row, and therefore
    /// the carve size `Input::patches` states. A submission that unfolded to a
    /// different width would be describing a different rectangle.
    pub patch_width: u32,
    /// How many rows of [`pos_embed`](Tower::pos_embed) one patch gathers:
    /// 4 for the bilinear resample every non-native grid needs (§11.2).
    ///
    /// **FOUR AND NOT ONE, AND THE DEGENERATE CASE IS FREE ANYWAY.** A
    /// deployment whose resize policy locks every image to the stored 48×48
    /// grid could state 1 and reserve no weight stream at all; one that
    /// resizes — which is every policy that admits a second aspect ratio —
    /// needs the hat weights, and §11.4's gate proves the native grid
    /// degenerates to weight 1 on the patch's own row under exactly this
    /// call. So four is the reading that is always right, and one is an
    /// optimization a grid-locked deployment may take by changing this line.
    pub taps: u32,
    /// `num_position_embeddings` — the learned table's row count, and the
    /// square of the grid side the host resamples from.
    pub positions: u32,
    pub theta: f32,
    pub norm_eps: f32,
    pub sm_scale: f32,
    /// `[hidden, patch_width]` — the patch "convolution", which is a matmul
    /// because the submission ships patch VECTORS (§2's contract decision).
    pub patch_embed: Weight,
    pub patch_embed_bias: Weight,
    /// `[positions, hidden]`, gathered per patch row (§11.3).
    pub pos_embed: Weight,
    pub blocks: Vec<TowerBlock>,
    pub merger: Merger,
}

/// One vision block: a prenorm bidirectional attention and a prenorm ungated
/// MLP, both with biases, which is what makes every projection here a
/// `matmul` plus an `add_bias` rather than a bare `matmul`.
pub struct TowerBlock {
    pub norm1: Weight,
    pub norm1_bias: Weight,
    /// `[3·hidden, hidden]` — q, k and v fused, as the checkpoint stores them.
    pub qkv: Weight,
    pub qkv_bias: Weight,
    pub proj: Weight,
    pub proj_bias: Weight,
    pub norm2: Weight,
    pub norm2_bias: Weight,
    pub fc1: Weight,
    pub fc1_bias: Weight,
    pub fc2: Weight,
    pub fc2_bias: Weight,
}

/// The patch merger: the one place the patch rectangle's row COUNT changes.
///
/// **THE NORM IS ON THE UNMERGED ROWS, AND THE CHECKPOINT SAYS SO.**
/// `Qwen3_5VisionPatchMerger` norms before the shuffle by default
/// (`use_postshuffle_norm = False`) and its `norm` is `[hidden]` and not
/// `[merge²·hidden]`, which is the same fact read off the shapes. So the text
/// writes the norm at `hidden` and `layout.merge_rows` after it.
pub struct Merger {
    pub norm: Weight,
    pub norm_bias: Weight,
    /// `[merge²·hidden, merge²·hidden]`.
    pub fc1: Weight,
    pub fc1_bias: Weight,
    /// `[out_hidden, merge²·hidden]` — `out_hidden` is the TRUNK's width, which
    /// is what makes the tower's answer a token row.
    pub fc2: Weight,
    pub fc2_bias: Weight,
}

pub enum Head {
    Tied,
    Bank(Weight),
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub mixer: Mixer,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    /// This layer's adapter bank, `[slots, rank, hidden]` and
    /// `[slots, hidden, rank]` — the down and up planes of one correction
    /// site (design §8).
    ///
    /// **THE SITE IS THE MIXER SUBLAYER, AND IT IS THE SITE BECAUSE OF THE
    /// COLLECTIVE.** Both ends are REPLICATED values: the input is this
    /// layer's normed residual, and the output is the mixer's result AFTER
    /// `all_reduce`. A correction stated one statement earlier — on
    /// `o_proj`'s own output, which is what a checkpoint's `o_proj` LoRA
    /// names — reads a rows-cut partial product and lands before the reduce,
    /// so every rank would contribute the whole `ΔW·x` and the sum would
    /// carry it `tp` times. `MoeBiasSum` states the identical argument about
    /// the identical hazard, and takes the identical way out: say the
    /// additive term once, after the reduce, where it lands exactly once.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub enum Mixer {
    Attn(Attn),
    Gdn(Gdn),
}

pub struct Attn {
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub qg_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub k_norm: Weight,
    pub k_norm_eps: f32,
    pub kv: String,
}

pub struct Gdn {
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub conv_kernel: u32,
    pub in_qkvz: Weight,
    pub in_ba: Weight,
    pub conv: Weight,
    pub dt_bias: Weight,
    pub a_log: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
    pub out_proj: Weight,
    pub conv_state: String,
    pub delta_state: String,
}

impl Gdn {
    #[must_use]
    pub fn qkv_width(k_heads: u32, v_heads: u32, k_dim: u32, v_dim: u32) -> u32 {
        2 * k_heads * k_dim + v_heads * v_dim
    }
}

#[allow(clippy::large_enum_variant)]
pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
    },
    Routed {
        router: Weight,
        gate_up: Weight,
        down: Weight,
        shared_gate_up: Weight,
        shared_down: Weight,
        shared_gate: Weight,
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
    },
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
}

enum MlpDims {
    Dense { inter: u32 },
    Routed(MoeDims),
}

/// A tower's own numbers, read off `config.json`'s `vision_config`.
///
/// Separate from [`Dims`] because they size a DIFFERENT ROW SPACE: nothing
/// here divides by `tp` (the tower is replicated, see [`Tower`]) and nothing
/// here is a trunk fact. `out_hidden` is the one number that crosses — it is
/// the trunk's `hidden`, which is what makes the tower's answer a token row —
/// and it is asserted against it rather than restated.
#[derive(Clone, Copy)]
struct TowerDims {
    depth: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    /// `in_channels · temporal_patch_size · patch_size²`.
    patch_width: u32,
    merge: u32,
    positions: u32,
    out_hidden: u32,
    /// `Qwen3_5VisionRotaryEmbedding`'s own default; neither SKU's
    /// `vision_config` states a `rope_parameters` block at all.
    theta: f32,
    norm_eps: f32,
    /// How many table rows a patch's position gathers ([`Tower::taps`]).
    taps: u32,
}

impl TowerDims {
    /// **THE TWO SHIPPED TOWERS, AND EVERY NUMBER IS `vision_config`'s.**
    ///
    /// qwen35-0.8B (snapshot `2fc06364`) and qwen36-27B (`6a9e13bd`) publish
    /// the same tower with two sizes: `patch_size: 16`, `temporal_patch_size:
    /// 2`, `in_channels: 3` (so `patch_width` is 1536 in both),
    /// `spatial_merge_size: 2`, `num_position_embeddings: 2304`,
    /// `hidden_act: "gelu_pytorch_tanh"`. What differs is depth, width, head
    /// count, the MLP's waist and `out_hidden_size`.
    const fn qwen35() -> TowerDims {
        TowerDims {
            depth: 12,
            hidden: 768,
            heads: 12,
            inter: 3072,
            patch_width: 1536,
            merge: 2,
            positions: 2304,
            out_hidden: 1024,
            theta: 10_000.0,
            norm_eps: 1e-6,
            taps: 4,
        }
    }

    const fn qwen36() -> TowerDims {
        TowerDims {
            depth: 27,
            hidden: 1152,
            heads: 16,
            inter: 4304,
            patch_width: 1536,
            merge: 2,
            positions: 2304,
            out_hidden: 5120,
            theta: 10_000.0,
            norm_eps: 1e-6,
            taps: 4,
        }
    }
}

struct Dims {
    hidden: u32,
    layers: u32,
    attn_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_kernel: u32,
    mlp: MlpDims,
    vocab: u32,
    tied: bool,
    norm_eps: f32,
    /// The vision tower this SKU's checkpoint publishes, or `None`.
    tower: Option<TowerDims>,
    /// Which draft-head recipe this SKU's artifact carries, or `None` for one
    /// that carries none. One layer is all any shipped qwen states
    /// (`mtp_num_hidden_layers: 1`) and EAGLE's own recipe is one decoder
    /// layer too, so there is no count here — a second layer would be a
    /// second block in the text, and no artifact asks for one.
    draft: Option<Recipe>,
}

impl Model {
    pub fn a3b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 40,
                attn_every: 4,
                q_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 32,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Routed(MoeDims {
                    experts: 256,
                    top_k: 8,
                    inter: 512,
                    shared_inter: 512,
                }),
                vocab: 248_320,
                tied: false,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
            },
        )
    }

    /// **A ROUTED MoE THIS FAMILY'S SHAPE, SMALL ENOUGH TO HOLD TWICE** — the
    /// text the weight-residency gate loads (alto design §7, wave D2).
    ///
    /// Not a catalog row and deliberately not one: no checkpoint ships it, no
    /// deployment selects it, and adding it to `CATALOG` would oblige an
    /// import contract and a chat template for a model nobody serves. What it
    /// is for is the one claim `a3b` cannot be used to make on a single card —
    /// that a load whose `device_weight_budget` holds HALF the experts
    /// produces the logits full residency produces — because that claim needs
    /// BOTH loads on one device and `a3b` is sixty-four gibibytes.
    ///
    /// Every number below is `a3b`'s, divided until two copies fit in a
    /// gate's patience, with two deliberate exceptions:
    ///
    /// * `attn_every: 1` — every layer is a gated-attention layer rather than
    ///   one in four. The residency tier is about the MLP's expert banks and
    ///   the GDN mixer is orthogonal to it; holding the mixer fixed is what
    ///   makes a difference in the logits a difference in the weights.
    /// * `tied: true` — no `lm_head` plane, because a second
    ///   `[vocab, hidden]` rectangle is the largest dense thing here and the
    ///   dense floor is not what is under test.
    ///
    /// It is built through the same `Model::new` every shipped size is, so
    /// nothing about its banks, cuts or names can drift from `a3b`'s.
    pub fn a3b_micro(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 512,
                layers: 4,
                attn_every: 1,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 64,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 8,
                v_heads: 8,
                k_dim: 64,
                v_dim: 64,
                conv_kernel: 4,
                mlp: MlpDims::Routed(MoeDims {
                    experts: 32,
                    top_k: 4,
                    inter: 128,
                    shared_inter: 128,
                }),
                vocab: 2048,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
            },
        )
    }

    pub fn d0_8b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d0_8b_dims(None, None))
    }

    /// **THE SAME TWENTY-FOUR LAYERS, WITH AN EAGLE HEAD OVERLAID** (campaign
    /// M-4).
    ///
    /// A SECOND ROW AND NOT A FLAG, for [`d27b_undrafted`]'s reason read the
    /// other way round: whether a head is there is a fact about the ARTIFACT,
    /// and this artifact is a different one — `pie model import <base> --aux
    /// <head>` writes the base's tensors and a second checkpoint's under an
    /// `aux.` prefix into one `.zt`. A row that declared the head optionally
    /// would be a row that loads a plain qwen35 artifact and zeroes eleven
    /// planes, which is a draft head that proposes noise and a gate that
    /// cannot tell that from a bug.
    ///
    /// The trunk is `d0_8b`'s, sentence for sentence — one `d0_8b_dims` builds
    /// both — so nothing about this row can drift from the row it drafts for.
    ///
    /// [`d27b_undrafted`]: Model::d27b_undrafted
    pub fn d0_8b_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d0_8b_dims(None, Some(Recipe::Eagle)))
    }

    /// **THE PILOT** (campaign M-1): the same twenty-four layers, reading the
    /// twelve-block 768 → 1024 tower its own checkpoint has always shipped.
    ///
    /// A SECOND ROW AND NOT A WIDENED ONE, for `d0_8b_eagle`'s reason and one
    /// that is sharper here: a tower is the whole of what makes this plan a
    /// TWO-UNIT one, and a text that declared it optionally would bake a patch
    /// axis into every qwen35 artifact — a `PatchLadder` every deployment
    /// would have to state, a second exec every fire would have to skip, and
    /// G4's "every pre-campaign SKU is exactly one capture unit" gone. The
    /// trunk is `d0_8b`'s, sentence for sentence, out of one `d0_8b_dims`.
    pub fn d0_8b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d0_8b_dims(Some(TowerDims::qwen35()), None),
        )
    }

    /// **THE M-4 RIG'S OWN ROW** (campaign M-4, and the shadowing the M-1 flip
    /// would otherwise cause).
    ///
    /// An overlay artifact is a base checkpoint plus `aux.*`, and a base
    /// checkpoint SHIPS `model.visual.*` — so once a vision row precedes the
    /// text-only one, an eagle artifact lands on the vision row and its draft
    /// head is never bound. No row read both, and the gate that exercises the
    /// draft mechanism went quietly unloadable. This is that row: the same
    /// trunk, the tower its checkpoint ships, and the head `--aux` wrote
    /// beside them, so the strictly most demanding artifact has a row that
    /// asks for everything it holds.
    pub fn d0_8b_vision_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d0_8b_dims(Some(TowerDims::qwen35()), Some(Recipe::Eagle)),
        )
    }

    fn d0_8b_dims(tower: Option<TowerDims>, draft: Option<Recipe>) -> Dims {
        Dims {
            hidden: 1024,
            layers: 24,
            attn_every: 4,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            theta: 10_000_000.0,
            k_heads: 16,
            v_heads: 16,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            mlp: MlpDims::Dense { inter: 3584 },
            vocab: 248_320,
            tied: true,
            norm_eps: 1e-6,
            tower,
            draft,
        }
    }

    pub fn d3b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 24,
                attn_every: 4,
                q_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 32,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 8192 },
                vocab: 151_936,
                tied: true,
                norm_eps: 1e-6,
                tower: None,
                draft: None,
            },
        )
    }

    /// **Qwen3.6-27B, and it is a SKU of this family and not a family of its
    /// own** (palo C3).
    ///
    /// The ruling, and the evidence for it, read off
    /// `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B` at snapshot
    /// `6a9e13bd`. `config.json` says `model_type: "qwen3_5"` and
    /// `architectures: ["Qwen3_5ForConditionalGeneration"]` — the checkpoint
    /// names ITSELF a Qwen3.5 — and every structural number the trunk of this
    /// text reads is the same KIND of number one layer up: `layer_types`
    /// alternates three `linear_attention` to one `full_attention`, so
    /// `attn_every = 4` and the full layers land at `l % 4 == 3` (verified:
    /// layer 0 publishes `linear_attn.*`, layer 3 publishes `self_attn.*`);
    /// `attn_output_gate: true`, so `q_proj` is `[2·q·d, hidden]`;
    /// `partial_rotary_factor: 0.25` of `head_dim: 256`, so `rotary_dim = 64`.
    /// Not one op of the trunk changes. What is NEW is the `mtp.*` head, and a
    /// head is a declaration, not an architecture.
    ///
    /// **THE TWO THINGS THIS SKU DOES NOT SERVE, SAID OUT LOUD.** The
    /// checkpoint also ships a 27-block SigLIP-shaped `model.visual.*` tower
    /// and an interleaved-mrope section (`rope_parameters.mrope_interleaved`,
    /// `mrope_section: [11, 11, 10]`). This row is the TEXT-ONLY reading of the
    /// artifact — `config.json`'s own `language_model_only` switch names that
    /// reading — and it declares neither. A text lane's positions are scalar,
    /// which is what `elemwise.rope_partial` takes; an image lane's are a
    /// three-section triple, and that is a fourth axis with a fourth fact, not
    /// a flag on this one.
    pub fn d27b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, Some(Recipe::Mtp)))
    }

    /// The same sixty-four layers and the same one reading, read WITHOUT the
    /// draft head.
    ///
    /// Not a smaller model and not a disabled feature — [`Mtp`]'s own doc says
    /// a `None` there is a model that has no second head, and this is how a
    /// SKU says so. It exists because the fifteen `mtp.*` planes are a fact
    /// about the ARTIFACT: the 4-bit conversions of this model are produced by
    /// `mlx_lm`, which implements no multi-token-prediction arm for the family
    /// and therefore carries none of them, and a text that demanded them would
    /// refuse every 4-bit artifact of the model it is named after. What it
    /// costs is the draft path and nothing else — the trunk is the same
    /// sentences, and `mtp.*` planes an artifact happens to ship are simply
    /// not read.
    pub fn d27b_undrafted(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, None))
    }

    /// **THE COMBINED SKU** (campaign M-2): the twenty-seven-block SigLIP-shaped
    /// tower AND the `mtp.*` draft head, which is the row multimodal §0 said
    /// this checkpoint gives for free.
    ///
    /// Free is not the same as absent. What it means is that the two axes are
    /// declared by two independent fields and share nothing — the tower is a
    /// patch-axis unit and the head is a token-axis window — so this row is
    /// `d27b`'s sentences plus `d0_8b_vision`'s, and neither had to learn
    /// about the other.
    pub fn d27b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d27b_dims(Some(TowerDims::qwen36()), Some(Recipe::Mtp)),
        )
    }

    fn d27b_dims(tower: Option<TowerDims>, draft: Option<Recipe>) -> Dims {
        Dims {
            hidden: 5120,
            layers: 64,
            attn_every: 4,
            q_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            theta: 10_000_000.0,
            k_heads: 16,
            v_heads: 48,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            mlp: MlpDims::Dense { inter: 17_408 },
            vocab: 248_320,
            tied: false,
            norm_eps: 1e-6,
            tower,
            draft,
        }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        // Everything this text declares that is NOT a matmul bank: the norms,
        // the depthwise convolution, the deltanet's per-head biases, and the
        // adapter planes the host writes. See `crate::dense` for why they are
        // asked for rather than assumed to be `w`.
        let dense = crate::dense(w);
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let k_heads = d.k_heads / tp;
        let v_heads = d.v_heads / tp;
        let hidden = d.hidden as u64;
        let attn_at = |l: u32| l % d.attn_every == d.attn_every - 1;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                let mixer = if attn_at(l) {
                    Mixer::Attn(gated_attn(
                        w,
                        &d,
                        q_heads,
                        kv_heads,
                        &format!("layer.{l}"),
                        format!("kv.{l}"),
                    ))
                } else {
                    let k_w = k_heads as u64 * d.k_dim as u64;
                    let v_w = v_heads as u64 * d.v_dim as u64;
                    let qkv = u64::from(Gdn::qkv_width(k_heads, v_heads, d.k_dim, d.v_dim));
                    let qkvz = qkv + v_w;
                    Mixer::Gdn(Gdn {
                        k_heads,
                        v_heads,
                        k_dim: d.k_dim,
                        v_dim: d.v_dim,
                        conv_kernel: d.conv_kernel,
                        in_qkvz: Weight::sym(n("in_qkvz"), [qkvz, hidden], w)
                            .packed([k_w, k_w, v_w, v_w]),
                        in_ba: Weight::sym(n("in_ba"), [2 * v_heads as u64, hidden], w)
                            .packed([v_heads as u64, v_heads as u64]),
                        // NOT BANKS, EITHER OF THEM. The convolution contracts
                        // over four taps and the bias over nothing, and a
                        // sixty-four-code group has neither to group; MLX ships
                        // both unquantized for exactly that reason.
                        conv: Weight::sym(n("conv"), [qkv, d.conv_kernel as u64], dense)
                            .packed([k_w, k_w, v_w]),
                        dt_bias: Weight::sym(n("dt_bias"), [v_heads as u64], dense).columns(),
                        a_log: Weight::sym(n("a_log"), [v_heads as u64], Dtype::F32).columns(),
                        norm: Weight::sym(n("gdn_norm"), [d.v_dim as u64], Dtype::F32),
                        norm_eps: d.norm_eps,
                        out_proj: Weight::sym(n("out_proj"), [hidden, v_w], w).rows(),
                        conv_state: format!("conv.{l}"),
                        delta_state: format!("delta.{l}"),
                    })
                };
                let mlp = match &d.mlp {
                    MlpDims::Dense { inter } => {
                        dense_mlp(w, hidden, inter / tp, &format!("layer.{l}"))
                    }
                    MlpDims::Routed(m) => {
                        let inter = m.inter / tp;
                        let shared_inter = m.shared_inter / tp;
                        Mlp::Routed {
                            router: Weight::sym(n("router"), [m.experts as u64, hidden], w),

                            gate_up: Weight::sym(
                                n("experts_gate_up"),
                                [m.experts as u64, 2 * inter as u64, hidden],
                                w,
                            )
                            .bank([inter as u64, inter as u64]),
                            down: Weight::sym(
                                n("experts_down"),
                                [m.experts as u64, hidden, inter as u64],
                                w,
                            )
                            .rows(),
                            shared_gate_up: Weight::sym(
                                n("shared_gate_up"),
                                [2 * shared_inter as u64, hidden],
                                w,
                            )
                            .packed([shared_inter as u64, shared_inter as u64]),
                            shared_down: Weight::sym(
                                n("shared_down"),
                                [hidden, shared_inter as u64],
                                w,
                            )
                            .rows(),
                            shared_gate: Weight::sym(n("shared_gate"), [1, hidden], w),
                            experts: m.experts,
                            top_k: m.top_k,
                            inter,
                            shared_inter,
                        }
                    }
                };
                Layer {
                    mixer,
                    mixer_norm: norm("mixer_norm", hidden),
                    mixer_norm_eps: d.norm_eps,
                    mlp_norm: norm("mlp_norm", hidden),
                    mlp_norm_eps: d.norm_eps,
                    mlp,
                    // REPLICATED under tp, because both ends of this site are:
                    // the input is the replicated normed residual and the output
                    // is the reduced mixer result. Nothing here is cut, so
                    // nothing here is summed twice. `crate::adapter::banks` is
                    // where the orientations and the registration live, for
                    // every family at once.
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        // The draft head, when the artifact publishes one. Its mlp is stated at
        // the trunk's own dense width whatever the trunk's own mlp is: the
        // checkpoint's `mtp.layers.0.mlp` is dense even where the trunk routes,
        // because a draft block is one block and has no experts to route to.
        // **THE TOWER, WHEN THE CHECKPOINT PUBLISHES ONE** (multimodal §2).
        // Every plane replicated — `Tower`'s own note argues why — and every
        // name under `visual.`, which is the checkpoint's own namespace with
        // its `model.` stripped, so a reader can hold the two lists side by
        // side.
        let tower = d.tower.map(|t| {
            assert_eq!(
                t.out_hidden, d.hidden,
                "a tower's `out_hidden_size` is the TRUNK's width — the merger's \
                 answer is a token row, and a mismatch would scatter a rectangle \
                 of the wrong width into the embedding"
            );
            assert_eq!(
                t.hidden % t.heads,
                0,
                "a {}-wide tower does not divide into {} heads",
                t.hidden,
                t.heads
            );
            let th = t.hidden as u64;
            let ti = t.inter as u64;
            let merged = u64::from(t.merge) * u64::from(t.merge) * th;
            let head_dim = t.hidden / t.heads;
            let n = |s: String| format!("visual.{s}");
            let plane = |s: String, dims: [u64; 2]| Weight::sym(n(s), dims, w);
            let vec1 = |s: String, len: u64| Weight::sym(n(s), [len], dense);
            Tower {
                hidden: t.hidden,
                heads: t.heads,
                head_dim,
                merge: t.merge,
                patch_width: t.patch_width,
                taps: t.taps,
                positions: t.positions,
                theta: t.theta,
                norm_eps: t.norm_eps,
                sm_scale: (head_dim as f32).sqrt().recip(),
                patch_embed: plane("patch_embed".into(), [th, u64::from(t.patch_width)]),
                patch_embed_bias: vec1("patch_embed_bias".into(), th),
                pos_embed: plane("pos_embed".into(), [u64::from(t.positions), th]),
                blocks: (0..t.depth)
                    .map(|l| {
                        let b = |s: &str| format!("block.{l}.{s}");
                        TowerBlock {
                            norm1: vec1(b("norm1"), th),
                            norm1_bias: vec1(b("norm1_bias"), th),
                            qkv: plane(b("qkv"), [3 * th, th]),
                            qkv_bias: vec1(b("qkv_bias"), 3 * th),
                            proj: plane(b("proj"), [th, th]),
                            proj_bias: vec1(b("proj_bias"), th),
                            norm2: vec1(b("norm2"), th),
                            norm2_bias: vec1(b("norm2_bias"), th),
                            fc1: plane(b("fc1"), [ti, th]),
                            fc1_bias: vec1(b("fc1_bias"), ti),
                            fc2: plane(b("fc2"), [th, ti]),
                            fc2_bias: vec1(b("fc2_bias"), th),
                        }
                    })
                    .collect(),
                merger: Merger {
                    norm: vec1("merger_norm".into(), th),
                    norm_bias: vec1("merger_norm_bias".into(), th),
                    fc1: plane("merger_fc1".into(), [merged, merged]),
                    fc1_bias: vec1("merger_fc1_bias".into(), merged),
                    fc2: plane("merger_fc2".into(), [hidden, merged]),
                    fc2_bias: vec1("merger_fc2_bias".into(), hidden),
                },
            }
        });

        //
        // **THE KV ROW STAYS `kv.mtp` UNDER EITHER RECIPE, AND THE PREFIX DOES
        // NOT REACH IT.** A cache row's name is what `caches()` seats and what
        // `Input::kv` asks by; it is a fact about THIS PLAN's page-id space and
        // not about which checkpoint the bytes came out of. Two spellings of
        // one row would be two page tables for one sequence.
        let mtp = d.draft.map(|recipe| {
            let inter = match &d.mlp {
                MlpDims::Dense { inter } => *inter,
                MlpDims::Routed(m) => m.inter,
            } / tp;
            let p = recipe.prefix();
            let n = |s: &str| format!("{p}.{s}");
            Mtp {
                recipe,
                pre_fc: matches!(recipe, Recipe::Mtp).then(|| PreFc {
                    embedding: Weight::sym(n("pre_fc_norm_embedding"), [hidden], dense),
                    hidden: Weight::sym(n("pre_fc_norm_hidden"), [hidden], dense),
                    eps: d.norm_eps,
                }),
                // REPLICATED, both halves. A fusion bank contracts over `hidden`
                // and produces `hidden`, and both ends of it are replicated
                // values — the embedding of a token every rank holds, and the
                // trunk's residual stream after its reduce. Cutting either way
                // would put a partial sum where a whole one is read.
                fc_embed: Weight::sym(n("fc_embed"), [hidden, hidden], w),
                fc_hidden: Weight::sym(n("fc_hidden"), [hidden, hidden], w),
                mixer_norm: Weight::sym(n("mixer_norm"), [hidden], dense),
                mixer_norm_eps: d.norm_eps,
                attn: gated_attn(w, &d, q_heads, kv_heads, p, "kv.mtp".to_string()),
                mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                mlp_norm_eps: d.norm_eps,
                mlp: dense_mlp(w, hidden, inter, p),
                norm: matches!(recipe, Recipe::Mtp)
                    .then(|| Weight::sym(n("norm"), [hidden], dense)),
                norm_eps: d.norm_eps,
            }
        });

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            kv_heads,
            head_dim: d.head_dim,
            adapters: ADAPTERS,
            kv,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], w),
            head: if d.tied {
                Head::Tied
            } else {
                Head::Bank(Weight::sym("lm_head", [d.vocab as u64, hidden], w))
            },
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            tower,
            mtp,
        }
    }
}

/// What every SKU of this family seats.
///
/// One pair of numbers rather than a `Dims` field, because they are not a
/// fact about the checkpoint the way `hidden` and `layers` are — no
/// pretrained artifact states them. They are the DEPLOYMENT's ceiling written
/// where a shape has to be written, and a deployment that wants a different
/// one changes this line and re-traces, which is exactly the "load-time
/// recompile, never a runtime extension" design §9 asks for.
///
/// Eight slots of rank sixteen is what a bank costs at qwen35-d0.8b: two
/// planes of `8 x 16 x 1024` bf16 per layer, 512 KiB a layer, 12 MiB over
/// twenty-four — against 1.40 GiB of weights.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

/// One gated full-attention site's banks, named under `prefix`.
///
/// SHARED BY THE TRUNK AND THE DRAFT HEAD because the checkpoint shares them:
/// `mtp.layers.0.self_attn.*` has the shapes of
/// `model.language_model.layers.3.self_attn.*` tensor for tensor —
/// `q_proj [2·q·d, hidden]`, `k_proj`/`v_proj [kv·d, hidden]`,
/// `o_proj [hidden, q·d]`, per-head `q_norm`/`k_norm [d]`. Two spellings of
/// one site would be two places for a head count to be wrong in.
///
/// The head counts and the per-head width are not written into the returned
/// site at all: they are the model's ONE reading (`Model::q_heads`,
/// `kv_heads`, `head_dim`), and what this function takes them for is shaping
/// the banks — the per-rank counts, because the banks are cut by `tp`.
fn gated_attn(w: Dtype, d: &Dims, q_heads: u32, kv_heads: u32, prefix: &str, kv: String) -> Attn {
    let n = |s: &str| format!("{prefix}.{s}");
    let dense = crate::dense(w);
    let hidden = d.hidden as u64;
    let hd = d.head_dim as u64;
    Attn {
        rotary_dim: d.rotary_dim,
        theta: d.theta,
        sm_scale: (d.head_dim as f32).sqrt().recip(),
        qg_proj: Weight::sym(n("qg_proj"), [2 * q_heads as u64 * hd, hidden], w).columns(),
        k_proj: Weight::sym(n("k_proj"), [kv_heads as u64 * hd, hidden], w).columns(),
        v_proj: Weight::sym(n("v_proj"), [kv_heads as u64 * hd, hidden], w).columns(),
        o_proj: Weight::sym(n("o_proj"), [hidden, q_heads as u64 * hd], w).rows(),
        q_norm: Weight::sym(n("q_norm"), [hd], dense),
        q_norm_eps: d.norm_eps,
        k_norm: Weight::sym(n("k_norm"), [hd], dense),
        k_norm_eps: d.norm_eps,
        kv,
    }
}

/// One dense SwiGLU sublayer's banks, named under `prefix`. The draft head's
/// mlp is the trunk's at the same intermediate width — `mtp.layers.0.mlp.*` is
/// `[17408, 5120]` twice and `[5120, 17408]` once, exactly a trunk layer's.
fn dense_mlp(w: Dtype, hidden: u64, inter: u32, prefix: &str) -> Mlp {
    let n = |s: &str| format!("{prefix}.{s}");
    Mlp::Dense {
        gate_up: Weight::sym(n("gate_up"), [2 * inter as u64, hidden], w)
            .packed([inter as u64, inter as u64]),
        down: Weight::sym(n("down"), [hidden, inter as u64], w).rows(),
        inter,
    }
}

impl Model {
    pub fn load(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let mut claims: Vec<Claim> = vec![
            claim(&self.embed, self.tp),
            claim(&self.final_norm, self.tp),
        ];

        match &self.head {
            Head::Tied => {}
            Head::Bank(head) => claims.push(claim(head, self.tp)),
        }

        for layer in &self.layers {
            claims.push(claim(&layer.mixer_norm, self.tp));
            claims.push(claim(&layer.mlp_norm, self.tp));

            match &layer.mixer {
                Mixer::Attn(a) => {
                    claims.push(claim(&a.qg_proj, self.tp));
                    claims.push(claim(&a.k_proj, self.tp));
                    claims.push(claim(&a.v_proj, self.tp));
                    claims.push(claim(&a.o_proj, self.tp));
                    claims.push(claim(&a.q_norm, self.tp));
                    claims.push(claim(&a.k_norm, self.tp));
                }
                Mixer::Gdn(g) => {
                    claims.push(claim(&g.in_qkvz, self.tp));
                    claims.push(claim(&g.in_ba, self.tp));
                    claims.push(claim(&g.conv, self.tp));
                    claims.push(claim(&g.dt_bias, self.tp));
                    claims.push(claim(&g.a_log, self.tp));
                    claims.push(claim(&g.norm, self.tp));
                    claims.push(claim(&g.out_proj, self.tp));
                }
            }

            match &layer.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    claims.push(claim(gate_up, self.tp));
                    claims.push(claim(down, self.tp));
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    ..
                } => {
                    claims.push(claim(router, self.tp));
                    claims.push(claim(gate_up, self.tp));
                    claims.push(claim(down, self.tp));
                    claims.push(claim(shared_gate_up, self.tp));
                    claims.push(claim(shared_down, self.tp));
                    claims.push(claim(shared_gate, self.tp));
                }
            }
        }

        // The tower's planes. Stated in one walk rather than folded into the
        // layer loop above because they are a different row space's weights:
        // nothing here is cut by `tp`, and nothing here has an adapter bank.
        if let Some(t) = &self.tower {
            claims.push(claim(&t.patch_embed, self.tp));
            claims.push(claim(&t.patch_embed_bias, self.tp));
            claims.push(claim(&t.pos_embed, self.tp));
            for b in &t.blocks {
                for w in [
                    &b.norm1,
                    &b.norm1_bias,
                    &b.qkv,
                    &b.qkv_bias,
                    &b.proj,
                    &b.proj_bias,
                    &b.norm2,
                    &b.norm2_bias,
                    &b.fc1,
                    &b.fc1_bias,
                    &b.fc2,
                    &b.fc2_bias,
                ] {
                    claims.push(claim(w, self.tp));
                }
            }
            let m = &t.merger;
            for w in [
                &m.norm,
                &m.norm_bias,
                &m.fc1,
                &m.fc1_bias,
                &m.fc2,
                &m.fc2_bias,
            ] {
                claims.push(claim(w, self.tp));
            }
        }

        // The draft head's own planes. Stated here and not folded into the
        // layer walk because a head is not a layer: it has no adapter bank
        // (nothing routes a correction into a draft), and its `fc` halves are
        // two claims over one stored bank — which `import.rs` is where the
        // slicing is said, and this is where the shapes are demanded.
        if let Some(mtp) = &self.mtp {
            if let Some(pre) = &mtp.pre_fc {
                claims.push(claim(&pre.embedding, self.tp));
                claims.push(claim(&pre.hidden, self.tp));
            }
            claims.push(claim(&mtp.fc_embed, self.tp));
            claims.push(claim(&mtp.fc_hidden, self.tp));
            claims.push(claim(&mtp.mixer_norm, self.tp));
            claims.push(claim(&mtp.attn.qg_proj, self.tp));
            claims.push(claim(&mtp.attn.k_proj, self.tp));
            claims.push(claim(&mtp.attn.v_proj, self.tp));
            claims.push(claim(&mtp.attn.o_proj, self.tp));
            claims.push(claim(&mtp.attn.q_norm, self.tp));
            claims.push(claim(&mtp.attn.k_norm, self.tp));
            claims.push(claim(&mtp.mlp_norm, self.tp));
            match &mtp.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    claims.push(claim(gate_up, self.tp));
                    claims.push(claim(down, self.tp));
                }
                Mlp::Routed { .. } => panic!(
                    "`{}`: a draft head is one block and routes to no experts",
                    mtp.mixer_norm.name,
                ),
            }
            if let Some(norm) = &mtp.norm {
                claims.push(claim(norm, self.tp));
            }
        }

        elaborate(src, claims)
    }
}
