use checkpoint::contract::ModelContract;
use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// The query heads are the same count under both readings, so they are a
    /// fact about the text and not about a layer.
    pub q_heads: u32,
    /// The two readings this text carves attention schedules for. A layer
    /// names one of them and states nothing about it itself.
    pub sliding: Sliding,
    pub global: Global,

    /// The adapter banks this family seats (palo design §8). Per layer, and
    /// the same two numbers at every one of them: the correction is a
    /// per-lane axis, not a per-layer one.
    pub adapters: Adapters,

    /// **THE VISION TOWER, WHEN THE CHECKPOINT SHIPS ONE** (multimodal §0,
    /// §12; campaign M-2). `Option` for the qwen rows' reason: it is a fact
    /// about the ARTIFACT — E4B publishes a sixteen-block
    /// `model.vision_tower.*` and B31 publishes none — and it is the whole of
    /// what makes this plan a TWO-UNIT one.
    pub tower: Option<Tower>,

    pub kv: Dtype,
    pub softcap: Option<f32>,
    pub embed: Weight,
    pub ple: Option<Ple>,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,

    /// **THE AUX DRAFT HEAD, WHEN AN OVERLAY CARRIES ONE** (campaign M-4,
    /// design §M5).
    ///
    /// `None` for every stock gemma4 artifact, and that is not a disabled
    /// feature: **no gemma checkpoint publishes a draft head** — the cached
    /// E4B and E2B hold not one tensor matching `mtp`/`nextn`/`draft`/`eagle`,
    /// which `the_checkpoints_state_what_the_texts_read` asserts as a verdict
    /// rather than a note. A head reaches this family the way EAGLE heads
    /// reach any family: separately obtained, and baked in beside the base by
    /// `pie model import --aux`.
    ///
    /// **AND GEMMA IS WHERE THE IDENTITY GATE BELONGS** (multimodal §17). A
    /// greedy speculative run must answer the greedy sequential run token for
    /// token, and on qwen35 it cannot: that text is a HYBRID, and its
    /// gated-delta layers carry a recurrent state that a rejected draft row
    /// folds into and no mask can cut. Gemma attends and does not recur, so a
    /// rejected row leaves nothing behind but a kv cell the next fire
    /// overwrites — which is the whole contract speculation rests on.
    pub draft: Option<Draft>,
}

/// One EAGLE-style aux head: the fusion of a hidden state with the next
/// token's embedding, one decoder block over the fused stream, and a readout
/// through the base model's own head.
///
/// **THE SAME SHAPE `qwen_3::Mtp` DECLARES, MINUS THE TWO PIECES EAGLE HAS
/// NOT.** No pre-fusion norms — the recipe fuses the raw pair — and no final
/// norm of its own, because the hidden it was trained against is the one this
/// trunk's `final_norm` already produced. What is gemma's rather than qwen's
/// is the BLOCK: four norms around two sublayers, the family's own sentence,
/// so a head trained for gemma is written the way gemma writes a layer.
///
/// **`fc` IS TWO BANKS AND THE OVERLAY SHIPS ONE.** `[a|b]·[Wₑ|W_h]ᵀ =
/// a·Wₑᵀ + b·W_hᵀ` exactly, and this IR states no concatenation; `import`
/// slices the stored `[hidden, 2·hidden]` at column `hidden`, embedding half
/// first, which is the order the fusion concatenates in.
pub struct Draft {
    pub fc_embed: Weight,
    pub fc_hidden: Weight,
    pub attn_norm: Weight,
    pub post_attn_norm: Weight,
    pub pre_ffw_norm: Weight,
    pub post_ffw_norm: Weight,
    /// The head's own block reads GLOBALLY — full attention, its own kv row.
    /// One arm and not two, for `qwen_3::mtp_attn`'s reason: the head is the
    /// small speculative forward, and a batched-prefill read of a one-row lane
    /// is the same numbers as a decode read.
    pub attn: Attn,
    pub o_proj: Weight,
    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,
    pub norm_eps: f32,
}

pub struct Ple {
    pub dim: u32,
    pub model_proj: Weight,
    pub model_norm: Weight,
    pub model_norm_eps: f32,
    pub per_layer: Vec<PleLayer>,
}

pub struct PleLayer {
    pub table: Weight,
    pub gate: Weight,
    pub proj: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
    pub scalar: Weight,
}

/// **GEMMA'S SECOND ROW AXIS** (multimodal §12, campaign M-2).
///
/// The same patch rectangle qwen's tower reads, and a different model on it.
/// Four sentences are gemma's own, each settled against
/// `transformers/models/gemma4/modeling_gemma4.py` and the E4B checkpoint
/// rather than assumed:
///
/// **THE BLOCK IS THIS FAMILY'S OWN FOUR-NORM SENTENCE.**
/// `Gemma4VisionEncoderLayer` norms before AND after each sublayer and adds
/// the residual last — `input → attn → post_attention → +residual`, then
/// `pre_ffw → mlp → post_ffw → +residual` — which is exactly what the trunk
/// above writes. RMSNorm throughout, weight only, so no `layernorm_no_scale`
/// here: gemma's tower is the one that never needed §6.1.
///
/// **AND `sm_scale` IS 1.0, NOT `head_dim^-0.5`.** `Gemma4VisionAttention`
/// states `self.scaling = 1.0`, the same number the trunk carries, and a
/// tower that took the usual reciprocal square root would be a plausible
/// wrong answer. `v_norm` is a norm with NO SCALE (`with_scale=False`), which
/// is `elemwise.rmsnorm_no_scale` per head — the trunk's `qkv_unfused` writes
/// that line already.
///
/// **THE POSITION TABLE IS TWO SEPARABLE LOOKUPS, NOT AN INTERPOLATION.**
/// `_position_embeddings` gathers `table[0][x] + table[1][y]` — no bilinear
/// resample anywhere, which is what qwen's tower needs four taps for. The
/// stored `[2, positions, hidden]` is TRANSMUTED to `[2 · positions, hidden]`
/// and read with `layout.embed_weighted` at two taps and weights of one: the
/// `y` stream indexes the second half, so one node answers the sum and the
/// weight stream carries ones rather than a hat. Exact, and one gather where
/// two plus an add would have been.
///
/// **AND THE CLIPPED LINEARS FORBID A FUSED MLP** (§12). Every projection
/// clamps its input and its output to bounds the CHECKPOINT states — 448
/// scalars over the E4B tower — so `gate_proj` and `up_proj`, which read the
/// same `x`, clamp it to DIFFERENT bounds and cannot share a packed bank the
/// way the trunk's `gate_up` does. The MLP is therefore two matmuls and the
/// two-value `mlp_geglu_tanh`, not the packed one. **The wide tower turns the
/// clamps OFF** (`use_clipped_linears: false`) and keeps the unpacked MLP
/// anyway — `Gemma4VisionMLP` holds three separate `Gemma4ClippableLinear`s
/// whatever the flag says, and the checkpoint publishes `gate_proj` and
/// `up_proj` as two banks, so there is no packed bank to read.
///
/// **AND THE WIDE TOWER STANDARDIZES ITS ANSWER** (§21).
/// `vision_config.standardize: true` publishes `std_bias`/`std_scale` as two
/// `[hidden]` buffers and `Gemma4VisionModel.forward` ends
/// `(h − std_bias) · std_scale`, after the pooler's `√hidden` and before the
/// projection. E4B states `standardize: false` and ships neither plane, which
/// is why this is an `Option` and not a pair of ones.
pub struct Tower {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    /// `pooling_kernel_size`: `layout.pool_rows` folds `pool²` consecutive
    /// patch rows into one soft token, which is the same merge-block-major
    /// statute qwen's fold reads at a different `k`.
    pub pool: u32,
    /// `C · P²` — no temporal axis, which is the one shape difference from
    /// qwen's patch row.
    pub patch_width: u32,
    /// `2 · position_embedding_size`: the two axis tables, end to end.
    pub positions: u32,
    pub theta: f32,
    pub norm_eps: f32,
    pub sm_scale: f32,
    pub patch_embed: Weight,
    pub pos_embed: Weight,
    pub blocks: Vec<TowerBlock>,
    /// `[trunk hidden, hidden]` — `model.embed_vision.embedding_projection`,
    /// which is what makes the pooled soft token a token row.
    ///
    /// **AND IT IS THE ONE TOWER-SIDE PLANE A 4-BIT ARTIFACT QUANTIZES.**
    /// Every other bank below is `dense(w)`; this one is `w`. See
    /// [`Model::new`]'s tower block for the census that settles it.
    pub projection: Weight,
    /// `vision_tower.std_{bias,scale}`, when the tower states
    /// `standardize: true`.
    pub std: Option<Standardization>,
}

/// The two `[hidden]` planes `vision_config.standardize` publishes:
/// `y = (x − bias) · scale`, the last thing the tower does to a soft token.
pub struct Standardization {
    pub bias: Weight,
    pub scale: Weight,
}

/// One `Gemma4ClippableLinear`: the bank, and the four bounds the checkpoint
/// states for it when `use_clipped_linears` is on (§12). `lo`/`hi` are `[1]`
/// weights and not plan constants, because a trace is built with no
/// checkpoint in the room.
///
/// **CLIPPABLE, NOT CLIPPED**, and the name is the difference this row exists
/// to carry: the upstream class always wraps the linear and the CONFIG says
/// whether it clamps. E4B's tower says yes and ships 448 finite scalars; the
/// 27-block tower says `use_clipped_linears: false` and ships none, so its
/// projections are a bank and nothing else — a bare `matmul`, and eight
/// fewer elementwise launches a block.
pub struct Clippable {
    pub bank: Weight,
    pub clip: Option<Bounds>,
}

/// The four saturating bounds of one clipped linear, in the order
/// `Gemma4ClippableLinear.forward` reads them.
pub struct Bounds {
    pub in_lo: Weight,
    pub in_hi: Weight,
    pub out_lo: Weight,
    pub out_hi: Weight,
}

/// One vision block: four norms, a bidirectional attention over seven clipped
/// linears' worth of banks, and a gated ungated-by-fusion MLP.
pub struct TowerBlock {
    pub attn_norm: Weight,
    pub post_attn_norm: Weight,
    pub pre_ffw_norm: Weight,
    pub post_ffw_norm: Weight,
    pub q: Clippable,
    pub k: Clippable,
    pub v: Clippable,
    pub o: Clippable,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub gate: Clippable,
    pub up: Clippable,
    pub down: Clippable,
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub attn: Attn,
    pub o_proj: Weight,
    pub attn_norm: Weight,
    pub attn_norm_eps: f32,
    pub post_attn_norm: Weight,
    pub post_attn_norm_eps: f32,
    pub pre_ffw_norm: Weight,
    pub pre_ffw_norm_eps: f32,
    pub post_ffw_norm: Weight,
    pub post_ffw_norm_eps: f32,

    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,

    /// **THE PER-LAYER OUTPUT SCALAR, AND IT IS NOT A PLE FACT.**
    ///
    /// `mlx_lm/models/gemma4_text.py`'s decoder layer ends with two
    /// statements, in this order and both unconditional on the second:
    ///
    /// ```python
    /// if self.post_per_layer_input_norm is not None:   # the PLE relay
    ///     h = residual + gate
    /// if self.layer_scalar is not None:
    ///     h = h * self.layer_scalar
    /// ```
    ///
    /// The scalar multiplies whatever the layer produced, PLE or no PLE.
    /// This text had it only under [`PleLayer::scalar`], where it is the last
    /// term of the relay — which is right for `e4b`, and left `b31` with
    /// nothing: sixty `layers.{l}.layer_scalar` planes in
    /// `mlx-community/gemma-4-31b-it-4bit`, every one of them read by nobody.
    ///
    /// **THEY ARE NOT ONES.** Measured over all sixty: 0.0894 at layer 0,
    /// 0.0654 at layer 1, 0.0364 at layer 59, and between 0.75 and 0.99
    /// through the middle of the stack — a factor of twenty-seven between the
    /// smallest and the largest. Dropping them is not a rounding difference,
    /// it is a different model.
    ///
    /// `Some` exactly when this text declares no PLE, so the scalar is
    /// claimed, imported and applied ONCE whichever stack it is in. A PLE
    /// stack's stays where `e4b` already had it, and neither the `e4b`
    /// contract nor its tensor names move.
    pub scalar: Option<Weight>,

    /// This layer's adapter bank, `[slots, rank, hidden]` and
    /// `[slots, hidden, rank]` — the down and up planes of one correction site
    /// (palo design §8, campaign A-6).
    ///
    /// **THE SITE IS THE ATTENTION SUBLAYER, AND IT IS THE SITE BECAUSE OF THE
    /// COLLECTIVE.** Both ends are REPLICATED values: the input is this
    /// layer's `attn_norm`ed residual and the output is `o_proj`'s result
    /// AFTER `all_reduce`. A correction stated one statement earlier — on
    /// `o_proj`'s own output per rank, which is what a checkpoint's `o_proj`
    /// LoRA names — reads a rows-cut partial product and lands before the
    /// reduce, so every rank would contribute the whole `ΔW·x` and the sum
    /// would carry it `tp` times.
    ///
    /// AND BEFORE `post_attn_norm`, which is where a `o_proj` LoRA belongs:
    /// this family normalizes the sublayer's OUTPUT before the residual add,
    /// so a correction stated after that norm would be corrected-then-not
    /// normalized — a different function of the same weights, and not the one
    /// the adapter was trained as.
    ///
    /// **A SHARED-KV LAYER CARRIES ITS OWN BANK ANYWAY.** The tail layers of
    /// e4b borrow another layer's kv row and publish only a `q_proj`, but the
    /// correction site is not an attention bank — it is the sublayer's two
    /// replicated ends, and those exist at every layer. A skipped bank there
    /// would make a bound adapter mean something different in the tail than in
    /// the trunk.
    pub lora_a: Weight,
    pub lora_b: Weight,

    /// **THE SECOND FEEDFORWARD BRANCH, WHEN THE CHECKPOINT SHIPS ONE**
    /// (`text_config.enable_moe_block`).
    ///
    /// `None` on every dense member of this family, and the `None` is what
    /// keeps those rows byte-for-byte what they were: the branch declares its
    /// own weights, states its own norms and adds nothing to a stack that has
    /// no mixture.
    pub moe: Option<Moe>,
}

/// **THE ROUTED BRANCH OF gemma-4-26B-A4B, AND IT SITS BESIDE THE DENSE MLP
/// RATHER THAN REPLACING IT.**
///
/// `mlx_lm/models/gemma4_text.py`'s decoder layer, verbatim, at
/// `enable_moe = True`:
///
/// ```python
/// h1 = self.pre_feedforward_layernorm(h)
/// h1 = self.mlp(h1)
/// h1 = self.post_feedforward_layernorm_1(h1)
///
/// top_k_indices, top_k_weights = self.router(h)
/// h2 = self.pre_feedforward_layernorm_2(h)
/// h2 = self.experts(h2, top_k_indices, top_k_weights)
/// h2 = self.post_feedforward_layernorm_2(h2)
///
/// h = h1 + h2
/// ```
///
/// followed by the sandwich's own `post_feedforward_layernorm` and the
/// residual add, exactly as on a dense layer. So the mixture adds THREE norm
/// planes per layer to a family that already carries four around two
/// sublayers — `mlx-community/gemma-4-26b-a4b-it-4bit` publishes thirty of
/// each of `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2` and
/// `pre_feedforward_layernorm_2` — and both branches read the SAME `h`, the
/// post-attention residual, so they are siblings rather than a chain.
///
/// **AND THE ROUTER IS THIS FAMILY'S OWN.** `Router.__call__`:
///
/// ```python
/// x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)
/// expert_scores = self.proj(x)
/// top_k_indices = argpartition(expert_scores, -top_k)[..., -top_k:]
/// top_k_weights = mx.softmax(take_along_axis(expert_scores, top_k_indices))
/// top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
/// ```
///
/// Two learned planes no other router here has: `scale`, a hidden-wide RMS
/// gain, and `per_expert_scale`, a gain per EXPERT gathered by the ids the
/// top-k just chose. The softmax is over the SELECTED k — `take_along_axis`
/// first, `softmax` after — which is `linear.moe_topk_softmax`'s own
/// denominator, so the only new arithmetic is the last line and
/// `linear.moe_topk_softmax_scaled` is that line.
pub struct Moe {
    /// `router.scale`, TIMES `hidden**-0.5`.
    ///
    /// The constant is folded into the plane at import rather than said in the
    /// forward, because mlx folds it into the gain too — `self.scale *
    /// self._root_size` is one array handed to one `rms_norm` — and because a
    /// scalar on a norm's OUTPUT is not free here: it feeds a softmax, so it
    /// does not cancel and a separate elementwise pass over `[tokens, hidden]`
    /// would be thirty more of them a step.
    pub router_norm: Weight,
    pub router_norm_eps: f32,
    /// `router.proj`, `[experts, hidden]`, no bias.
    pub router: Weight,
    /// `router.per_expert_scale`, `[experts]`, indexed by expert.
    pub per_expert_scale: Weight,
    /// `pre_feedforward_layernorm_2` — the routed branch's entry norm.
    pub pre_ffw_norm_2: Weight,
    pub pre_ffw_norm_2_eps: f32,
    /// `post_feedforward_layernorm_1` — the DENSE branch's exit norm, which
    /// exists only where there is a second branch to be added to.
    pub post_ffw_norm_1: Weight,
    pub post_ffw_norm_1_eps: f32,
    /// `post_feedforward_layernorm_2` — the routed branch's exit norm.
    pub post_ffw_norm_2: Weight,
    pub post_ffw_norm_2_eps: f32,
    /// `[experts, 2 · inter, hidden]`, gate first, cut at axis 1.
    pub gate_up: Weight,
    /// `[experts, hidden, inter]`.
    pub down: Weight,
    pub experts: u32,
    pub top_k: u32,
    /// `moe_intermediate_size` — a SECOND width, much narrower than the dense
    /// `Layer::inter`: 704 against 2112 on the 26B.
    pub inter: u32,
}

pub struct Attn {
    pub reading: Reading,
    pub sm_scale: f32,
    pub q_norm: Weight,
    pub q_norm_eps: f32,

    pub kv: String,
    pub banks: AttnBanks,
}

/// Which of the text's two readings of the one sequence a layer takes. The
/// discriminant is the index: anything the forward pass carves per reading is
/// a two-element array `[sliding, global]` indexed by `reading as usize`.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Reading {
    Sliding = 0,
    Global = 1,
}

/// The local reading: narrow heads over a window of recent keys.
pub struct Sliding {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub window: u32,
    pub theta: f32,
}

/// The global reading: wide heads over the whole sequence, rotated over only
/// the leading `rotary_dim` of each head.
pub struct Global {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub rotary_dim: u32,
    pub theta: f32,
}

#[allow(clippy::large_enum_variant)]
pub enum AttnBanks {
    Owned {
        qkv: Weight,
        k_norm: Weight,
        k_norm_eps: f32,
    },
    Shared {
        q_proj: Weight,
    },
}

/// The tower's own numbers, off `config.json`'s `vision_config`. Nothing here
/// divides by `tp` — a sixteen-block 768-wide tower is replicated for the
/// reason qwen's is, and cutting it would put a collective inside the patch
/// unit.
#[derive(Clone, Copy)]
struct TowerDims {
    depth: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    /// `in_channels · patch_size²`; gemma's patch has no temporal extent.
    patch_width: u32,
    pool: u32,
    /// `position_embedding_size`, per AXIS. The declared table is twice it.
    positions: u32,
    out_hidden: u32,
    theta: f32,
    norm_eps: f32,
    sm_scale: f32,
    /// `use_clipped_linears`: whether every projection reads four learned
    /// bounds beside its bank.
    clipped: bool,
    /// `standardize`: whether the pooled answer is centred and scaled by two
    /// `[hidden]` planes before the projection.
    standardize: bool,
}

impl TowerDims {
    /// **E4B's TOWER**, every number `vision_config`'s own:
    /// `num_hidden_layers: 16`, `hidden_size: 768`, `num_attention_heads: 12`
    /// (so `head_dim: 64`, which the config also states),
    /// `intermediate_size: 3072`, `patch_size: 16` over three channels,
    /// `pooling_kernel_size: 3`, `position_embedding_size: 10240`,
    /// `rope_parameters.rope_theta: 100.0`, `rms_norm_eps: 1e-6`, and
    /// `Gemma4VisionAttention`'s own `scaling = 1.0`.
    const fn e4b() -> TowerDims {
        TowerDims {
            depth: 16,
            hidden: 768,
            heads: 12,
            inter: 3072,
            patch_width: 3 * 16 * 16,
            pool: 3,
            positions: 10_240,
            out_hidden: 2560,
            theta: 100.0,
            norm_eps: 1e-6,
            sm_scale: 1.0,
            clipped: true,
            standardize: false,
        }
    }

    /// **THE WIDE TOWER**, and it is ONE tower serving two trunks:
    /// `mlx-community/gemma-4-31b-it-4bit` and
    /// `mlx-community/gemma-4-26b-a4b-it-4bit` publish IDENTICAL
    /// `vision_config`s and 358 vision tensors each of the same names and
    /// shapes. Only `out_hidden` — the trunk width the projection lands in —
    /// tells the two rows apart, which is why it is this function's argument
    /// and not a second constant.
    ///
    /// Every number is that `vision_config`'s own: `num_hidden_layers: 27`,
    /// `hidden_size: 1152`, `num_attention_heads: 16` with `head_dim: 72`
    /// (and `1152 / 16 = 72`, so the derived width agrees with the stated
    /// one), `intermediate_size: 4304`, `patch_size: 16` over three channels,
    /// `pooling_kernel_size: 3`, `position_embedding_size: 10240`,
    /// `rope_parameters.rope_theta: 100.0`, `rms_norm_eps: 1e-6`, and
    /// `Gemma4VisionAttention`'s own `scaling = 1.0`.
    ///
    /// **THE TWO FLAGS ARE WHERE IT PARTS FROM
    /// [`e4b`](TowerDims::e4b).** `use_clipped_linears: false` and
    /// `standardize: true` — the small tower says the opposite of both — and
    /// the checkpoints agree tensor for tensor: no
    /// `encoder.layers.*.{input,output}_{min,max}` anywhere in the index, and
    /// `vision_tower.std_bias` / `vision_tower.std_scale` at `[1152]` each.
    /// Everything else — the four-norm block, the separable position table,
    /// the `v_norm` with no scale, the `3 × 3` pool — is the same sentence at
    /// a different width.
    const fn wide(out_hidden: u32) -> TowerDims {
        TowerDims {
            depth: 27,
            hidden: 1152,
            heads: 16,
            inter: 4304,
            patch_width: 3 * 16 * 16,
            pool: 3,
            positions: 10_240,
            out_hidden,
            theta: 100.0,
            norm_eps: 1e-6,
            sm_scale: 1.0,
            clipped: false,
            standardize: true,
        }
    }
}

struct Dims {
    tower: Option<TowerDims>,
    /// Whether this SKU's artifact carries an `aux.*` overlay head.
    draft: bool,
    hidden: u32,
    layers: u32,
    full_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    global_head_dim: u32,
    global_kv_heads: u32,
    global_rotary_dim: u32,
    theta_local: f32,
    theta_global: f32,
    sm_scale: f32,
    intermediate: u32,
    vocab: u32,
    shared_tail: Option<u32>,
    ple_dim: Option<u32>,
    softcap: Option<f32>,
    window: u32,
    norm_eps: f32,
    moe: Option<MoeDims>,
}

/// `text_config`'s mixture: `num_experts`, `top_k_experts` and
/// `moe_intermediate_size`.
#[derive(Clone, Copy)]
struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
}

impl Model {
    pub fn e4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::e4b_dims())
    }

    fn e4b_dims() -> Dims {
        Dims {
                tower: None,
                draft: false,
                hidden: 2560,
                layers: 42,
                full_every: 6,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 256,
                global_head_dim: 512,
                global_kv_heads: 2,
                global_rotary_dim: 128,
                theta_local: 10_000.0,
                theta_global: 1_000_000.0,
                sm_scale: 1.0,
                intermediate: 10_240,
                vocab: 262_144,
                shared_tail: Some(18),
                ple_dim: Some(256),
                softcap: Some(30.0),
                window: 512,
                norm_eps: 1e-6,
                moe: None,
        }
    }

    /// **THE THIRD SKU** (campaign M-2): the same forty-two layers reading the
    /// sixteen-block tower E4B's own checkpoint ships.
    ///
    /// A second row for `qwen35-d0.8b-vision`'s reason: a tower is what makes
    /// the plan two capture units, and a row that declared it optionally would
    /// put a patch axis in every gemma4 artifact.
    pub fn e4b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        d.tower = Some(TowerDims::e4b());
        Model::new(w, kv, tp, d)
    }

    /// **THE M-4 IDENTITY RIG** (campaign M-4, multimodal §17): the same
    /// forty-two layers with an EAGLE head overlaid.
    ///
    /// A row of its own for `qwen35-d0.8b-eagle`'s reason — whether a head is
    /// there is a fact about the ARTIFACT, and this artifact is a different
    /// one — and it exists on GEMMA because gemma attends and does not recur.
    /// The qwen rig's identity gate cannot close while a rejected draft row
    /// folds into a gated-delta state; this one has no state to fold into.
    pub fn e4b_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        d.draft = true;
        Model::new(w, kv, tp, d)
    }

    pub fn b31(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::b31_dims())
    }

    /// **THE 31B READING THE TOWER ITS OWN CHECKPOINT SHIPS** — the wide
    /// tower of [`TowerDims::wide`], landing in this trunk's 5376.
    ///
    /// A row of its own for `e4b_vision`'s reason, and one more that only
    /// this row has: it is the FIRST gemma vision SKU whose weights are on
    /// this tree's own disk. `mlx-community/gemma-4-31b-it-4bit` publishes
    /// all 358 vision tensors beside a 4-bit trunk, so the pairing this row
    /// states — a bf16 tower over a U4 trunk, with only the projection
    /// quantized — is the artifact's own and not a configuration choice.
    pub fn b31_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::b31_dims();
        d.tower = Some(TowerDims::wide(d.hidden));
        Model::new(w, kv, tp, d)
    }

    fn b31_dims() -> Dims {
            Dims {
                tower: None,
                draft: false,
                hidden: 5376,
                layers: 60,
                full_every: 6,
                q_heads: 32,
                kv_heads: 16,
                head_dim: 256,
                global_head_dim: 512,
                global_kv_heads: 4,
                global_rotary_dim: 128,
                theta_local: 10_000.0,
                theta_global: 1_000_000.0,
                sm_scale: 1.0,
                intermediate: 21_504,
                vocab: 262_144,
                shared_tail: None,
                ple_dim: None,
                softcap: Some(30.0),
                // **`text_config.sliding_window`, READ OFF THE CHECKPOINT
                // THAT SHIPS THE WEIGHTS.** `mlx-community/gemma-4-31b-it-4bit`
                // says 1024 and this had said 512 — half of what the model was
                // trained to look back over, which is a difference no prompt
                // short enough to fit inside either number can notice, and
                // every longer one does. `e4b` above keeps its own 512;
                // the two stacks state their windows separately because they
                // are separate models.
                window: 1024,
                norm_eps: 1e-6,
                moe: None,
            }
    }

    /// **THE MIXTURE OF THE FAMILY** (`mlx-community/gemma-4-26b-a4b-it-4bit`),
    /// every number `config.json`'s `text_config` and every one of them
    /// different from `b31`'s.
    ///
    /// Thirty layers of hidden 2816; sixteen query heads reading 256-wide
    /// sliding heads with eight kv heads, and 512-wide global heads with two —
    /// the same two-shape attention `b31` carries, at its own widths. The
    /// global layers are `layer_types`' `full_attention` entries, which are
    /// indices 5, 11, 17, 23 and 29: `l % 6 == 5`. `attention_k_eq_v: true`,
    /// so exactly those five publish no `v_proj` — twenty-five of the thirty
    /// hold one, which is what the checkpoint's index says.
    ///
    /// `num_kv_shared_layers: 0` and `hidden_size_per_layer_input: 0`: no
    /// shared tail and NO per-layer embeddings, so the thirty `layer_scalar`
    /// planes are claimed by [`Layer::scalar`] the way `b31`'s sixty are.
    /// `use_double_wide_mlp: false`, so `intermediate_size: 2112` is every
    /// layer's dense width, and `moe_intermediate_size: 704` is the routed
    /// one — a third of it, over 128 experts at a fan-out of 8.
    pub fn a4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a4b_dims())
    }

    /// **THE MIXTURE READING THE SAME TOWER**, landing in 2816 instead of
    /// 5376. `mlx-community/gemma-4-26b-a4b-it-4bit`'s `vision_config` is
    /// `gemma-4-31b-it-4bit`'s field for field and its 358 vision tensors are
    /// the same names at the same shapes, so [`TowerDims::wide`] serves both
    /// and the trunk width is the whole of the difference.
    pub fn a4b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a4b_dims();
        d.tower = Some(TowerDims::wide(d.hidden));
        Model::new(w, kv, tp, d)
    }

    fn a4b_dims() -> Dims {
            Dims {
                tower: None,
                draft: false,
                hidden: 2816,
                layers: 30,
                full_every: 6,
                q_heads: 16,
                kv_heads: 8,
                head_dim: 256,
                global_head_dim: 512,
                global_kv_heads: 2,
                global_rotary_dim: 128,
                theta_local: 10_000.0,
                theta_global: 1_000_000.0,
                sm_scale: 1.0,
                intermediate: 2112,
                vocab: 262_144,
                shared_tail: None,
                ple_dim: None,
                softcap: Some(30.0),
                window: 1024,
                norm_eps: 1e-6,
                moe: Some(MoeDims {
                    experts: 128,
                    top_k: 8,
                    inter: 704,
                }),
            }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        // Everything this text declares that is NOT a matmul bank: the norms
        // — of which this family has more per layer than any other here — and
        // the per-layer-embedding scalar. See `crate::dense`.
        let dense = crate::dense(w);
        // **THE ROUTER'S PROJECTION IS EIGHT BITS WHERE THE STACK IS FOUR**,
        // and the checkpoint says so before its shapes do: `quantization`
        // names every one of the thirty `router.proj` entries at
        // `{group_size: 64, bits: 8}` and nothing else. The shapes agree — a
        // `[128, 2816]` router stored four bits to a code would ship a
        // `[128, 352]` `u32` plane and the file holds `[128, 704]`, which is
        // four codes to a word rather than eight. `qwen_3::Model::new` reads
        // its own gates the same way and for the same reason.
        let gate = match w {
            Dtype::U4g64 => Dtype::U8g64,
            other => other,
        };
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let global_kv_heads = d.global_kv_heads / tp;
        let intermediate = d.intermediate / tp;

        let hidden = d.hidden as u64;
        let full_at = |l: u32| l % d.full_every == d.full_every - 1;
        let shared_at = |l: u32| d.shared_tail.is_some_and(|tail| l >= d.layers - tail);
        let source = |l: u32| {
            (0..l)
                .rev()
                .find(|&s| !shared_at(s) && full_at(s) == full_at(l))
        };
        let owner = |l: u32| match d.shared_tail {
            None => l,
            Some(tail) if l < d.layers - tail => l,
            Some(tail) => source(l).unwrap_or_else(|| {
                panic!(
                    "layer {l} borrows its kv cache and none of the {} layers \
                     before the shared tail is of its kind (full_every {}, \
                     shared_tail {tail})",
                    d.layers - tail,
                    d.full_every,
                )
            }),
        };
        let sliding = Sliding {
            head_dim: d.head_dim,
            kv_heads,
            window: d.window,
            theta: d.theta_local,
        };
        let global = Global {
            head_dim: d.global_head_dim,
            kv_heads: global_kv_heads,
            rotary_dim: d.global_rotary_dim,
            theta: d.theta_global,
        };

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                let reading = if full_at(l) {
                    Reading::Global
                } else {
                    Reading::Sliding
                };
                let (head_dim, row_heads) = match reading {
                    Reading::Sliding => (sliding.head_dim, sliding.kv_heads),
                    Reading::Global => (global.head_dim, global.kv_heads),
                };
                let hd = head_dim as u64;
                let q_w = q_heads as u64 * hd;
                let kv_w = row_heads as u64 * hd;
                let iw = intermediate as u64;
                Layer {
                    attn: Attn {
                        sm_scale: d.sm_scale,
                        q_norm: norm("q_norm", hd),
                        q_norm_eps: d.norm_eps,
                        kv: format!("kv.{}", owner(l)),
                        banks: if shared_at(l) {
                            AttnBanks::Shared {
                                q_proj: Weight::sym(n("q_proj"), [q_w, hidden], w).columns(),
                            }
                        } else {
                            AttnBanks::Owned {
                                qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], w)
                                    .packed([q_w, kv_w, kv_w]),
                                k_norm: norm("k_norm", hd),
                                k_norm_eps: d.norm_eps,
                            }
                        },
                        reading,
                    },
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], w).rows(),
                    attn_norm: norm("attn_norm", hidden),
                    attn_norm_eps: d.norm_eps,
                    post_attn_norm: norm("post_attn_norm", hidden),
                    post_attn_norm_eps: d.norm_eps,
                    pre_ffw_norm: norm("pre_ffw_norm", hidden),
                    pre_ffw_norm_eps: d.norm_eps,
                    post_ffw_norm: norm("post_ffw_norm", hidden),
                    post_ffw_norm_eps: d.norm_eps,
                    gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
                    inter: intermediate,
                    down: Weight::sym(n("down"), [hidden, iw], w).rows(),
                    scalar: d
                        .ple_dim
                        .is_none()
                        .then(|| Weight::sym(n("scalar"), [1], dense)),
                    lora_a,
                    lora_b,
                    moe: d.moe.map(|m| {
                        let mi = (m.inter / tp) as u64;
                        Moe {
                            router_norm: norm("router_norm", hidden),
                            router_norm_eps: d.norm_eps,
                            router: Weight::sym(
                                n("router"),
                                [m.experts as u64, hidden],
                                gate,
                            ),
                            per_expert_scale: Weight::sym(
                                n("per_expert_scale"),
                                [m.experts as u64],
                                dense,
                            )
                            .columns(),
                            pre_ffw_norm_2: norm("pre_ffw_norm_2", hidden),
                            pre_ffw_norm_2_eps: d.norm_eps,
                            post_ffw_norm_1: norm("post_ffw_norm_1", hidden),
                            post_ffw_norm_1_eps: d.norm_eps,
                            post_ffw_norm_2: norm("post_ffw_norm_2", hidden),
                            post_ffw_norm_2_eps: d.norm_eps,
                            gate_up: Weight::sym(
                                n("experts_gate_up"),
                                [m.experts as u64, 2 * mi, hidden],
                                w,
                            )
                            .bank([mi, mi]),
                            down: Weight::sym(
                                n("experts_down"),
                                [m.experts as u64, hidden, mi],
                                w,
                            )
                            .rows(),
                            experts: m.experts,
                            top_k: m.top_k,
                            inter: m.inter / tp,
                        }
                    }),
                }
            })
            .collect();

        // **THE TOWER, WHEN THE CHECKPOINT PUBLISHES ONE.** Every plane
        // replicated and every name under `vision.`, which is the
        // checkpoint's own namespace with `model.vision_tower.` stripped.
        let tower = d.tower.map(|t| {
            assert_eq!(
                t.out_hidden, d.hidden,
                "a tower's projection lands a TRUNK row; a mismatch would scatter a \
                 rectangle of the wrong width into the embedding"
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
            let head_dim = t.hidden / t.heads;
            let n = |s: String| format!("vision.{s}");
            // **THE TOWER'S BANKS ARE `dense(w)` AND THE PROJECTION IS `w`**,
            // which is a fact about the ARTIFACTS and was read off them
            // rather than assumed. In every 4-bit checkpoint this catalog can
            // reach — `mlx-community/gemma-4-31b-it-4bit`,
            // `mlx-community/gemma-4-26b-a4b-it-4bit`, and
            // `mlx-community/Qwen3.6-27B-4bit` one family over — the vision
            // namespace holds NO `.scales` / `.biases` plane at all: 358
            // gemma vision tensors and 333 qwen ones, every projection stored
            // whole. The single quantized multimodal tensor anywhere is
            // gemma's `embed_vision.embedding_projection`, and it is
            // quantized because it is a TRUNK-width bank — `[5376, 1152]` on
            // the 31B — living on the language model's side of the seam.
            //
            // A row that declared the tower at `w` would ask a bf16 file for
            // triplets it does not hold and refuse at the door; a row that
            // declared the projection dense would read `[5376, 144]` `u32`
            // codes as though they were numbers.
            let bank = |s: String, dims: [u64; 2]| Weight::sym(n(s), dims, dense);
            let vec1 = |s: String, len: u64| Weight::sym(n(s), [len], dense);
            // One clippable linear: the bank, and the four bounds beside it
            // when `use_clipped_linears` says the checkpoint ships them.
            let clip = |s: &str, dims: [u64; 2]| Clippable {
                bank: bank(s.to_string(), dims),
                clip: t.clipped.then(|| Bounds {
                    in_lo: vec1(format!("{s}_in_lo"), 1),
                    in_hi: vec1(format!("{s}_in_hi"), 1),
                    out_lo: vec1(format!("{s}_out_lo"), 1),
                    out_hi: vec1(format!("{s}_out_hi"), 1),
                }),
            };
            Tower {
                hidden: t.hidden,
                heads: t.heads,
                head_dim,
                pool: t.pool,
                patch_width: t.patch_width,
                positions: 2 * t.positions,
                theta: t.theta,
                norm_eps: t.norm_eps,
                sm_scale: t.sm_scale,
                patch_embed: bank("patch_embed".into(), [th, u64::from(t.patch_width)]),
                pos_embed: bank("pos_embed".into(), [2 * u64::from(t.positions), th]),
                blocks: (0..t.depth)
                    .map(|l| {
                        let b = |s: &str| format!("block.{l}.{s}");
                        TowerBlock {
                            attn_norm: vec1(b("attn_norm"), th),
                            post_attn_norm: vec1(b("post_attn_norm"), th),
                            pre_ffw_norm: vec1(b("pre_ffw_norm"), th),
                            post_ffw_norm: vec1(b("post_ffw_norm"), th),
                            q: clip(&b("q"), [th, th]),
                            k: clip(&b("k"), [th, th]),
                            v: clip(&b("v"), [th, th]),
                            o: clip(&b("o"), [th, th]),
                            q_norm: vec1(b("q_norm"), u64::from(head_dim)),
                            k_norm: vec1(b("k_norm"), u64::from(head_dim)),
                            gate: clip(&b("gate"), [ti, th]),
                            up: clip(&b("up"), [ti, th]),
                            down: clip(&b("down"), [th, ti]),
                        }
                    })
                    .collect(),
                projection: Weight::sym(n("projection".into()), [hidden, th], w),
                std: t.standardize.then(|| Standardization {
                    bias: vec1("std_bias".into(), th),
                    scale: vec1("std_scale".into(), th),
                }),
            }
        });

        // **THE AUX HEAD, WHEN AN OVERLAY CARRIES ONE.** Named under `aux.`,
        // which is the namespace `pie model import --aux` prefixes a second
        // checkpoint's tensors with; the block is this family's own, read
        // GLOBALLY, and its kv row is `kv.mtp` in the trunk's one page-id
        // space — one page table for one sequence, which is the ruling the
        // qwen head's row rests on too.
        let draft = d.draft.then(|| {
            let hd = global.head_dim as u64;
            let q_w = q_heads as u64 * hd;
            let kv_w = global.kv_heads as u64 * hd;
            let iw = intermediate as u64;
            let n = |s: &str| format!("aux.{s}");
            let norm = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
            Draft {
                // REPLICATED, both halves: a fusion bank contracts over
                // `hidden` and produces `hidden`, and both ends are replicated
                // values — the embedding of a token every rank holds, and the
                // trunk's residual after its reduce.
                fc_embed: Weight::sym(n("fc_embed"), [hidden, hidden], w),
                fc_hidden: Weight::sym(n("fc_hidden"), [hidden, hidden], w),
                attn_norm: norm("attn_norm", hidden),
                post_attn_norm: norm("post_attn_norm", hidden),
                pre_ffw_norm: norm("pre_ffw_norm", hidden),
                post_ffw_norm: norm("post_ffw_norm", hidden),
                attn: Attn {
                    sm_scale: d.sm_scale,
                    q_norm: norm("q_norm", hd),
                    q_norm_eps: d.norm_eps,
                    kv: "kv.mtp".to_string(),
                    banks: AttnBanks::Owned {
                        qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], w)
                            .packed([q_w, kv_w, kv_w]),
                        k_norm: norm("k_norm", hd),
                        k_norm_eps: d.norm_eps,
                    },
                    reading: Reading::Global,
                },
                o_proj: Weight::sym(n("o_proj"), [hidden, q_w], w).rows(),
                gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
                inter: intermediate,
                down: Weight::sym(n("down"), [hidden, iw], w).rows(),
                norm_eps: d.norm_eps,
            }
        });

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            sliding,
            global,
            adapters: ADAPTERS,
            tower,
            kv,
            softcap: d.softcap,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], w),
            ple: d.ple_dim.map(|dim| {
                let ple = dim as u64;
                Ple {
                    dim,
                    model_proj: Weight::sym("ple.model_proj", [d.layers as u64 * ple, hidden], w),
                    model_norm: Weight::sym("ple.model_norm", [ple], dense),
                    model_norm_eps: d.norm_eps,
                    per_layer: (0..d.layers)
                        .map(|l| PleLayer {
                            table: Weight::sym(
                                format!("layer.{l}.ple_table"),
                                [d.vocab as u64, ple],
                                w,
                            ),
                            gate: Weight::sym(format!("layer.{l}.ple_gate"), [ple, hidden], w),
                            proj: Weight::sym(format!("layer.{l}.ple_proj"), [hidden, ple], w),
                            norm: Weight::sym(format!("layer.{l}.ple_norm"), [hidden], dense),
                            norm_eps: d.norm_eps,
                            scalar: Weight::sym(format!("layer.{l}.ple_scalar"), [1], dense),
                        })
                        .collect(),
                }
            }),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            draft,
        }
    }
}

/// What every SKU of this family seats.
///
/// Not a `Dims` field, because it is not a fact about the checkpoint the way
/// `hidden` and `layers` are — no pretrained artifact states it. It is the
/// DEPLOYMENT's ceiling written where a shape has to be written, and a
/// deployment that wants a different one changes this line and re-traces,
/// which is exactly the "load-time recompile, never a runtime extension"
/// design §9 asks for.
///
/// Eight slots of rank sixteen costs e4b 1.25 MiB a layer — two planes of
/// `8 x 16 x 2560` in the compute element — and 52.5 MiB over forty-two;
/// b31 pays 2.63 MiB a layer and 157.5 MiB over sixty.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {
    pub fn load(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        let mut b = checkpoint_dsl::Builder::new(src, self.tp);
        let mut claim = |w: &Weight| b.read_own(w);

        claim(&self.embed)?;
        claim(&self.final_norm)?;

        for layer in &self.layers {
            claim(&layer.attn_norm)?;
            claim(&layer.post_attn_norm)?;
            claim(&layer.pre_ffw_norm)?;
            claim(&layer.post_ffw_norm)?;
            claim(&layer.attn.q_norm)?;
            match &layer.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    claim(k_norm)?;
                    claim(qkv)?;
                }

                AttnBanks::Shared { q_proj } => {
                    claim(q_proj)?;
                }
            }
            claim(&layer.o_proj)?;
            claim(&layer.gate_up)?;
            claim(&layer.down)?;
            if let Some(scalar) = &layer.scalar {
                claim(scalar)?;
            }
        }

        if let Some(t) = &self.tower {
            claim(&t.patch_embed)?;
            claim(&t.pos_embed)?;
            claim(&t.projection)?;
            if let Some(std) = &t.std {
                claim(&std.bias)?;
                claim(&std.scale)?;
            }
            for b in &t.blocks {
                for w in [
                    &b.attn_norm,
                    &b.post_attn_norm,
                    &b.pre_ffw_norm,
                    &b.post_ffw_norm,
                    &b.q_norm,
                    &b.k_norm,
                ] {
                    claim(w)?;
                }
                for c in [&b.q, &b.k, &b.v, &b.o, &b.gate, &b.up, &b.down] {
                    claim(&c.bank)?;
                    if let Some(k) = &c.clip {
                        for w in [&k.in_lo, &k.in_hi, &k.out_lo, &k.out_hi] {
                            claim(w)?;
                        }
                    }
                }
            }
        }

        if let Some(a) = &self.draft {
            for w in [
                &a.fc_embed,
                &a.fc_hidden,
                &a.attn_norm,
                &a.post_attn_norm,
                &a.pre_ffw_norm,
                &a.post_ffw_norm,
                &a.attn.q_norm,
                &a.o_proj,
                &a.gate_up,
                &a.down,
            ] {
                claim(w)?;
            }
            if let AttnBanks::Owned { qkv, k_norm, .. } = &a.attn.banks {
                claim(qkv)?;
                claim(k_norm)?;
            }
        }

        if let Some(ple) = &self.ple {
            claim(&ple.model_proj)?;
            claim(&ple.model_norm)?;
            for per in &ple.per_layer {
                claim(&per.table)?;
                claim(&per.gate)?;
                claim(&per.proj)?;
                claim(&per.norm)?;
                claim(&per.scalar)?;
            }
        }

        Ok(b.build())
    }
}
