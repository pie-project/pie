use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// Same query head count under both readings (a fact about the text,
    /// not a layer).
    pub q_heads: u32,
    /// The two readings this text carves attention schedules for; a layer
    /// only names which one it uses.
    pub sliding: Sliding,
    pub global: Global,

    /// Adapter banks, one set of numbers per layer — the correction is a
    /// per-lane axis, not a per-layer one.
    pub adapters: Adapters,

    /// The vision tower, when the checkpoint ships one. `Option` because
    /// it's a fact about the artifact (not every SKU has a tower), and its
    /// presence is what makes the plan a two-unit one.
    pub tower: Option<Tower>,

    pub kv: Dtype,
    pub softcap: Option<f32>,
    pub embed: Weight,
    pub ple: Option<Ple>,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,

    /// The aux draft head, when an overlay carries one. `None` for every
    /// stock checkpoint — no gemma checkpoint publishes one; it is obtained
    /// separately and baked in via `pie model import --aux`.
    ///
    /// Speculative decoding is exact here because gemma attends and does
    /// not recur: a rejected draft row leaves only a kv cell to be
    /// overwritten, unlike a hybrid text with recurrent state.
    pub draft: Option<Draft>,

    /// Google's own Gemma 4 assistant — the trained MTP drafter
    /// (`google/gemma-4-*-it-assistant`), when an overlay carries one. See
    /// [`Assistant`].
    pub assistant: Option<Assistant>,
}

/// **GEMMA 4'S OWN DRAFTER**: a four-layer text stack that reads the trunk's
/// kv instead of keeping any. Per draft step it is fed the concatenation of
/// the trunk's token embedding (of the token the trunk chose) and a 2816-wide
/// hidden state — the trunk's post-norm readout at step 0, its own
/// `post_projection` after — projected by `pre_projection` to its hidden
/// width. Its layers are ordinary Gemma 4 layers whose attention has only a
/// `q_proj`: the sliding ones attend over the trunk's LAST sliding layer's kv
/// row, the global one over the last global's, every query rotated at the
/// row's own position and held there across the chain (transformers'
/// `SinglePositionMultiTokenCandidateGenerator`, mlx-vlm's `draft_block`).
/// It reads out through its own tied `embed_tokens`, no softcap. Its banks
/// take the row's weight dtype — the import quantizes the bf16 source on the
/// way in — because a chain step reads the whole head, and at bf16 that is
/// 0.8 GB a step, more than the trunk's own active bytes. A bank encoded on
/// the way in is named `*.weight`, where its scales are named beside it.
pub struct Assistant {
    /// Draft tokens chained per row; what `mtp_depth` advertises.
    pub depth: u32,
    /// `pre_projection`'s two column halves: `[hidden, trunk_hidden]` each,
    /// the embedding half first.
    pub pre_embed: Weight,
    pub pre_hidden: Weight,
    /// `post_projection`: `[trunk_hidden, hidden]`.
    pub post: Weight,
    /// Its own `[vocab, hidden]` table, read out through as the lm head.
    pub embed: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
    pub layers: Vec<AssistantLayer>,
}

pub struct AssistantLayer {
    /// `banks` is always [`AttnBanks::Shared`]; `kv` names the trunk row it
    /// borrows.
    pub attn: Attn,
    pub o_proj: Weight,
    pub attn_norm: Weight,
    pub post_attn_norm: Weight,
    pub pre_ffw_norm: Weight,
    pub post_ffw_norm: Weight,
    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,
    pub scalar: Weight,
}

/// The assistant's shape, the same for every published size: the trunk's
/// width differs, the drafter's does not.
const ASSISTANT_HIDDEN: u32 = 1024;
const ASSISTANT_INTER: u32 = 8192;
const ASSISTANT_READINGS: [Reading; 4] =
    [Reading::Sliding, Reading::Sliding, Reading::Sliding, Reading::Global];
/// Chained drafts per verify. Every chain step is a head pass paid whether
/// or not the window uses it, and on the M1 Max a verify row costs 0.5–0.6
/// of a first row, so the only window that pays for itself is `k = 1`:
/// measured on the 26B-A4B, depth 2 at `k = 1` is 18.0 ms a token against
/// 16.3 plain, depth 1 is 16.6 — parity. mlx-vlm's best block (three, two
/// drafts) is a batch-of-four number on an M3 Max. A device where rows are
/// cheap wants this at 2 or 3; that is a re-import, not a runtime knob.
pub const ASSISTANT_DEPTH: u32 = 1;

/// One EAGLE-style aux head: fuses a hidden state with the next token's
/// embedding, runs one decoder block, and reads out through the base
/// model's own head. `fc` is two banks (`[a|b]·[We|Wh]^T = a·We^T + b·Wh^T`);
/// import slices the stored `[hidden, 2*hidden]` bank at column `hidden`.
pub struct Draft {
    pub fc_embed: Weight,
    pub fc_hidden: Weight,
    pub attn_norm: Weight,
    pub post_attn_norm: Weight,
    pub pre_ffw_norm: Weight,
    pub post_ffw_norm: Weight,
    /// Reads globally (full attention), its own kv row. One prefill plan
    /// covers both prefill and decode shapes.
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

/// Gemma's vision tower: four-norm block, `sm_scale` fixed at 1.0 (not
/// `head_dim^-0.5`), `v_norm` unscaled, and a separable two-tap position
/// table (`[2, positions, hidden]`) rather than bilinear interpolation.
/// Every projection's input/output clamps to per-checkpoint learned bounds,
/// so `gate`/`up` can't share a packed bank.
pub struct Tower {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    /// `pooling_kernel_size`: folds `pool^2` consecutive patch rows into one
    /// soft token.
    pub pool: u32,
    /// `C * P^2`; no temporal axis.
    pub patch_width: u32,
    /// `2 * position_embedding_size`: the two axis tables, end to end.
    pub positions: u32,
    pub theta: f32,
    pub norm_eps: f32,
    pub sm_scale: f32,
    pub patch_embed: Weight,
    pub pos_embed: Weight,
    pub blocks: Vec<TowerBlock>,
    /// `[trunk hidden, hidden]`: makes a pooled soft token a token row. The
    /// one tower-side plane a 4-bit artifact quantizes (every other tower
    /// bank is dense).
    pub projection: Weight,
    /// `vision_tower.std_{bias,scale}`, when the tower states
    /// `standardize: true`.
    pub std: Option<Standardization>,
}

/// The two `[hidden]` planes `vision_config.standardize` publishes:
/// `y = (x - bias) * scale`, the last thing the tower does to a soft token.
pub struct Standardization {
    pub bias: Weight,
    pub scale: Weight,
}

/// One `Gemma4ClippableLinear`: the bank, and the four bounds (as `[1]`
/// weights, not plan constants) when `use_clipped_linears` is on.
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

    /// Per-layer output scalar; applied whether or not there's a PLE relay.
    /// Learned values vary widely across layers (not close to 1), so
    /// dropping them changes results materially.
    ///
    /// `Some` exactly when this text declares no PLE — a PLE stack's scalar
    /// lives in [`PleLayer::scalar`] instead.
    pub scalar: Option<Weight>,

    /// This layer's adapter bank (`[slots, rank, hidden]` /
    /// `[slots, hidden, rank]`) for the attention sublayer's correction
    /// site: input is the `attn_norm`ed residual, output is `o_proj`'s
    /// result after `all_reduce` — both replicated values, so the
    /// correction can't be folded in earlier (a per-rank partial) or later
    /// (past `post_attn_norm`).
    ///
    /// Applied even on shared-kv tail layers: the correction site is not
    /// the attention bank, so every layer needs its own.
    pub lora_a: Weight,
    pub lora_b: Weight,

    /// The routed feedforward branch, when the checkpoint ships one
    /// (`text_config.enable_moe_block`). `None` on dense layers, adding
    /// nothing to the stack.
    pub moe: Option<Moe>,
}

/// The routed branch of gemma-4-26B-A4B: runs beside the dense MLP (both
/// read the same post-attention residual `h`), not in place of it, then
/// both outputs sum before the sandwich's own closing norm.
///
/// The router's norm gain is `router.scale * hidden**-0.5`; scores get
/// top-k, softmax over the selected k, then a per-expert gain
/// (`per_expert_scale`).
pub struct Moe {
    /// `router.scale`, times `hidden**-0.5`, folded into the plane at import.
    pub router_norm: Weight,
    pub router_norm_eps: f32,
    /// `router.proj`, `[experts, hidden]`, no bias.
    pub router: Weight,
    /// `router.per_expert_scale`, `[experts]`, indexed by expert.
    pub per_expert_scale: Weight,
    /// `pre_feedforward_layernorm_2` — the routed branch's entry norm.
    pub pre_ffw_norm_2: Weight,
    pub pre_ffw_norm_2_eps: f32,
    /// `post_feedforward_layernorm_1` — the dense branch's exit norm (only
    /// present because there's a sibling to add to).
    pub post_ffw_norm_1: Weight,
    pub post_ffw_norm_1_eps: f32,
    /// `post_feedforward_layernorm_2` — the routed branch's exit norm.
    pub post_ffw_norm_2: Weight,
    pub post_ffw_norm_2_eps: f32,
    /// `[experts, 2 * inter, hidden]`, gate first, cut at axis 1.
    pub gate_up: Weight,
    /// `[experts, hidden, inter]`.
    pub down: Weight,
    pub experts: u32,
    pub top_k: u32,
    /// `moe_intermediate_size`, narrower than the dense `Layer::inter`
    /// (704 vs 2112 on the 26B).
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

/// The tower's own numbers. Nothing here divides by `tp`: the tower is
/// replicated, not sharded.
#[derive(Clone, Copy)]
struct TowerDims {
    depth: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    /// `in_channels * patch_size^2`; gemma's patch has no temporal extent.
    patch_width: u32,
    pool: u32,
    /// `position_embedding_size`, per axis. The declared table is twice it.
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
    /// E4B's tower dims, read off `vision_config`.
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

    /// The wide tower, shared by both 31B and A4B (identical `vision_config`,
    /// only `out_hidden` differs, so it is this function's argument).
    /// Unlike [`e4b`](TowerDims::e4b): `use_clipped_linears: false`,
    /// `standardize: true`.
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
    /// Whether it carries Google's assistant instead (see [`Assistant`]).
    assistant: bool,
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

    /// E4B cut to its first `layers` layers — the miniature a parity gate
    /// reads against an external reference truncated the same way, when the
    /// full stack cannot be tapped layer by layer. The shared tail keeps its
    /// place: layers from 24 on borrow their kv, however many of them the cut
    /// keeps. Everything else is the E4B's own.
    pub fn e4b_mini(layers: u32, w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        let owned = d.layers - d.shared_tail.unwrap_or(0);
        d.layers = layers;
        d.shared_tail = (layers > owned).then(|| layers - owned);
        Model::new(w, kv, tp, d)
    }

    fn e4b_dims() -> Dims {
        Dims {
                tower: None,
                draft: false,
                assistant: false,
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

    /// The same 42 layers reading the 16-block tower E4B's checkpoint ships.
    /// A separate row (not an optional tower field) because whether a tower
    /// exists changes the plan's unit count.
    pub fn e4b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        d.tower = Some(TowerDims::e4b());
        Model::new(w, kv, tp, d)
    }

    /// The same 42 layers with an EAGLE draft head overlaid. A separate row
    /// because whether a head exists is a fact about the artifact.
    pub fn e4b_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::e4b_dims();
        d.draft = true;
        Model::new(w, kv, tp, d)
    }

    /// The 31B with Google's own drafter overlaid
    /// (`gemma-4-31B-it-assistant`): the same four-layer head over a
    /// 5376-wide trunk. See [`Model::a4b_mtp`].
    pub fn b31_mtp(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::b31_dims();
        d.assistant = true;
        Model::new(w, kv, tp, d)
    }

    pub fn b31(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::b31_dims())
    }

    /// The 31B reading its own checkpoint's wide tower ([`TowerDims::wide`]),
    /// landing in this trunk's 5376-wide embedding. Weights (bf16 tower over
    /// a U4 trunk) come from the checkpoint's own pairing, not a
    /// configuration choice.
    pub fn b31_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::b31_dims();
        d.tower = Some(TowerDims::wide(d.hidden));
        Model::new(w, kv, tp, d)
    }

    fn b31_dims() -> Dims {
            Dims {
                tower: None,
                draft: false,
                assistant: false,
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
                // `text_config.sliding_window`, from the checkpoint. Distinct
                // from e4b's 512; the two stacks are separate models.
                window: 1024,
                norm_eps: 1e-6,
                moe: None,
            }
    }

    /// The mixture SKU: 30 layers, hidden 2816. Global layers are
    /// `layer_types`'s `full_attention` entries (`l % 6 == 5`); those five
    /// publish no `v_proj` (`attention_k_eq_v: true`). No shared kv tail, no
    /// per-layer embeddings.
    pub fn a4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a4b_dims())
    }

    /// The mixture with Google's own drafter overlaid
    /// (`gemma-4-26B-A4B-it-assistant`). A separate row: whether a head
    /// exists is a fact about the artifact.
    pub fn a4b_mtp(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a4b_dims();
        d.assistant = true;
        Model::new(w, kv, tp, d)
    }

    /// The mixture reading the same wide tower as [`Model::b31_vision`],
    /// landing in 2816 instead of 5376.
    pub fn a4b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a4b_dims();
        d.tower = Some(TowerDims::wide(d.hidden));
        Model::new(w, kv, tp, d)
    }

    fn a4b_dims() -> Dims {
            Dims {
                tower: None,
                draft: false,
                assistant: false,
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
        // Everything declared here that is not a matmul bank: norms and the
        // per-layer scalar. See `crate::dense`.
        let dense = crate::dense(w);
        // The router's projection is quantized at 8 bits (group 64) even
        // when the rest of the stack is 4-bit.
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

        // Every tower plane is replicated (not sharded); names live under
        // the `vision.` namespace.
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
            // Tower banks are always dense; only `projection` (a trunk-width
            // bank) is quantized to `w` — the checkpoints ship no quantized
            // tower plane besides it.
            let bank = |s: String, dims: [u64; 2]| Weight::sym(n(s), dims, dense);
            let vec1 = |s: String, len: u64| Weight::sym(n(s), [len], dense);
            // The bank, plus four bounds when `use_clipped_linears` says the
            // checkpoint ships them.
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

        // Named under `aux.` (the namespace `pie model import --aux`
        // prefixes a second checkpoint's tensors with). Reads globally; its
        // kv row is `kv.mtp` in the trunk's page-id space.
        let draft = d.draft.then(|| {
            let hd = global.head_dim as u64;
            let q_w = q_heads as u64 * hd;
            let kv_w = global.kv_heads as u64 * hd;
            let iw = intermediate as u64;
            let n = |s: &str| format!("aux.{s}");
            let norm = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
            Draft {
                // Both fusion banks are replicated: a token embedding every
                // rank holds, and the trunk's residual after its reduce.
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

        // The assistant borrows the trunk's LAST row of each reading —
        // the kv the trunk itself publishes as `shared_kv_states`.
        let assistant = d.assistant.then(|| {
            assert_eq!(tp, 1, "the assistant head is written for one rank");
            let last = |want_full: bool| {
                (0..d.layers)
                    .rev()
                    .find(|&l| !shared_at(l) && full_at(l) == want_full)
                    .map(owner)
                    .expect("the trunk has a layer of each reading")
            };
            let ah = ASSISTANT_HIDDEN as u64;
            let iw = ASSISTANT_INTER as u64;
            let n = |s: &str| format!("aux.{s}");
            let layers = ASSISTANT_READINGS
                .iter()
                .enumerate()
                .map(|(l, &reading)| {
                    let n = |s: &str| format!("aux.layer.{l}.{s}");
                    let norm = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
                    let hd = match reading {
                        Reading::Sliding => sliding.head_dim,
                        Reading::Global => global.head_dim,
                    } as u64;
                    let q_w = q_heads as u64 * hd;
                    AssistantLayer {
                        attn: Attn {
                            sm_scale: d.sm_scale,
                            q_norm: norm("q_norm", hd),
                            q_norm_eps: d.norm_eps,
                            kv: format!("kv.{}", last(reading == Reading::Global)),
                            banks: AttnBanks::Shared {
                                q_proj: Weight::sym(n("q_proj.weight"), [q_w, ah], w),
                            },
                            reading,
                        },
                        o_proj: Weight::sym(n("o_proj.weight"), [ah, q_w], w),
                        attn_norm: norm("attn_norm", ah),
                        post_attn_norm: norm("post_attn_norm", ah),
                        pre_ffw_norm: norm("pre_ffw_norm", ah),
                        post_ffw_norm: norm("post_ffw_norm", ah),
                        gate_up: Weight::sym(n("gate_up.weight"), [2 * iw, ah], w).packed([iw, iw]),
                        inter: ASSISTANT_INTER,
                        down: Weight::sym(n("down.weight"), [ah, iw], w),
                        scalar: norm("scalar", 1),
                    }
                })
                .collect();
            Assistant {
                depth: ASSISTANT_DEPTH,
                pre_embed: Weight::sym(n("pre_embed.weight"), [ah, hidden], w),
                pre_hidden: Weight::sym(n("pre_hidden.weight"), [ah, hidden], w),
                post: Weight::sym(n("post.weight"), [hidden, ah], w),
                embed: Weight::sym(n("embed.weight"), [d.vocab as u64, ah], w),
                norm: Weight::sym(n("final_norm"), [ah], dense),
                norm_eps: d.norm_eps,
                layers,
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
            assistant,
        }
    }
}

/// What every SKU seats. Not a checkpoint fact — a deployment ceiling,
/// changed by editing this line and re-tracing.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {
 }
