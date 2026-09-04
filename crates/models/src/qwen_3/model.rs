use model_dsl::{Dtype, Weight};


pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// The one attention reading this whole family (trunk and draft head)
    /// shares. Per-rank: `q_heads`/`kv_heads` are already divided by `tp`;
    /// `head_dim` is not.
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    /// Adapter banks, one pair of numbers per layer — the correction is a
    /// per-lane axis, not a per-layer one.
    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Head,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,

    /// `None` for a text-only checkpoint (4-bit conversions ship no tower
    /// either) — a single-unit plan rather than the two-unit plan a tower
    /// adds. Also decides the trunk's rotation: `Some` gives every layer the
    /// three-section mrope; `None` keeps the plain scalar rotation.
    pub tower: Option<Tower>,

    /// `None` if the artifact carries no draft head. [`Recipe`] distinguishes
    /// the checkpoint's own `mtp.*` head from an EAGLE overlay imported from
    /// a second checkpoint; both share this same declaration.
    pub mtp: Option<Mtp>,

    /// The block drafter, when this SKU's recipe is [`Recipe::DFlash`].
    /// Exclusive with [`mtp`](Model::mtp): a plan carries one draft head or
    /// none, and the two shapes share no declaration.
    pub dflash: Option<DFlash>,
}

/// Passes of the draft head a fire chains — the `mtp_drafts` seam's width and
/// the most a guest may ask for as `k`. The checkpoint ships one prediction
/// layer trained for one step; a second pass is that layer fed its own argmax
/// and residual (as the qwen4 head chains its two).
///
/// Measured at 2 on Qwen3.8-27B-4bit (M4 Pro, 2026-09-04, decode-only, with
/// a three-row fire at 1.17x a one-row fire): the second step lands ~0.60 of
/// its drafts on math and ~0.23 on prose against a break-even near 0.30, so
/// k=2 is +18% on math (24.9 -> 29.2 tok/s) and -4% on prose (18.9 -> 18.2);
/// and every pass costs the fire ~4 ms on the device (mostly the 389 MB
/// `lm_head` readout), which a k=1 round pays for a draft it never reads —
/// about -5%. A static depth is the wrong knob for a workload-shaped gain;
/// one, until the window is chosen per request off realized acceptance.
pub const DRAFT_DEPTH: u32 = 1;

/// One MTP (multi-token-prediction / NEXTN) head: fuses a hidden state with
/// the next token's embedding, runs one transformer block over the fused
/// stream, and reads out through the base model's own `lm_head`.
///
/// ```text
/// h = fc · [ rms(embed(tok)) · Wₑ | rms(hidden) · W_h ]
/// h += attn(rms(h))
/// h += mlp(rms(h))
/// draft = lm_head(rms(h))
/// ```
///
/// The stored `mtp.fc.weight` is one `[hidden, 2·hidden]` bank, split into
/// `fc_embed`/`fc_hidden` and summed via `residual_add` (the IR has no
/// concat op). An EAGLE head ([`Recipe::Eagle`]) reuses this struct with
/// `pre_fc` and `norm` both `None`.
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
    /// The head's own block: full attention with the family's q-gate and its
    /// own kv row (`mtp.layers.0.self_attn`), same shape as a trunk layer.
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

/// **A DFlash block drafter** — the shape `z-lab/Qwen3.6-27B-DFlash` ships.
///
/// Where [`Mtp`] fuses one hidden state with one token embedding and runs a
/// single block chained a token at a time, this fuses [`taps`](DFlash::taps)
/// TRUNK HIDDEN STATES and runs [`blocks`](DFlash::blocks) layers ONCE over a
/// whole block of rows whose tail is the mask token, so a single pass
/// proposes [`block`](DFlash::block) tokens:
///
/// ```text
/// h  = rms(Σᵢ tapᵢ · fcᵢ)                       the five taps, fused
/// for each block:  h += attn(rms(h));  h += mlp(rms(h))
/// draft = lm_head(rms(h))                       through the TARGET's head
/// ```
///
/// Three things differ from every other head this family carries, and each
/// is why this is its own declaration rather than a flag on [`Mtp`]:
///
/// * **The fusion is five-wide.** The stored `fc.weight` is one
///   `[hidden, 5·hidden]` bank; the IR has no concat, so it is sliced into
///   five `[hidden, hidden]` banks summed with `residual_add`, the same way
///   [`Mtp`] splits its two-wide bank.
/// * **The attention is NOT the family's gated site.** Its `q_proj` is
///   `[q_heads·head_dim, hidden]`, not the trunk's `2·q_heads·head_dim` — no
///   gate to split off — and its geometry is its own (32 q heads, 8 kv, head
///   dim 128 against the trunk's 24/4/256). Only the plane width coincides
///   (8·128 = 4·256 = 1024), which is what lets its kv rows share the
///   trunk's page-id space.
/// * **The pass is bidirectional over the block.** Four of five layers are
///   sliding-window; the mask that makes the block see itself is the
///   guest's (`inputs.mask()`), not the model's.
pub struct DFlash {
    /// Which trunk layers feed the fusion, in the order their banks are
    /// sliced out of `fc` — `[1, 16, 31, 46, 61]` for the shipped head.
    pub taps: Vec<u32>,
    /// One `[hidden, hidden]` column slice of the stored `fc.weight` per tap.
    pub fc: Vec<Weight>,
    /// Scales the fused stream before the first block.
    pub hidden_norm: Weight,
    pub hidden_norm_eps: f32,
    pub blocks: Vec<DFlashBlock>,
    /// The final norm before the readout through the target's `lm_head`.
    pub norm: Weight,
    pub norm_eps: f32,
    /// Rows one draft pass proposes — the width of the `mtp.drafts` seam.
    pub block: u32,
    /// The token every block row but the first carries on the way in.
    pub mask_token: u32,
}

/// One layer of a [`DFlash`] drafter: the standard pre-norm decoder block,
/// with its own ungated attention and a sliding window on all but the last.
pub struct DFlashBlock {
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub attn: DraftAttn,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    /// Keys older than this many positions are not attended; `None` on the
    /// one full-attention layer the shipped head ends with.
    pub window: Option<u32>,
}

/// A [`DFlash`] layer's attention site. [`Attn`]'s twin without the q-gate
/// and with its own head geometry; the two cannot share a struct because
/// `Attn::qg_proj` is declared at twice the q width and every reader of it
/// splits a gate off.
pub struct DraftAttn {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub k_norm: Weight,
    pub k_norm_eps: f32,
    /// This layer's kv row, in the trunk's page-id space.
    pub kv: String,
}

/// The two pre-fusion norms, when the recipe has them. One shared epsilon,
/// since `rms_norm_eps` is a single config value for this family.
pub struct PreFc {
    /// Scales the embedding of the row's token before the fusion.
    pub embedding: Weight,
    /// Scales the trunk's hidden state before the fusion.
    pub hidden: Weight,
    pub eps: f32,
}

/// Which draft-head recipe an artifact carries. Read by `Model::new` (which
/// pieces to declare) and `import` (which tensors to bind); irrelevant once
/// the trace is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recipe {
    /// The checkpoint's own head: 15 `mtp.*` tensors in the base artifact,
    /// with pre-fusion norms and a final norm.
    Mtp,
    /// A separately-obtained head baked into the artifact by
    /// `pie model import --aux` under an `aux.` prefix (can't collide with
    /// base checkpoint names). No pre-fusion norms, no final norm.
    Eagle,
    /// A DFlash block drafter, also `--aux`-imported under `aux.`: five
    /// decoder layers of its OWN geometry fed by a fusion of five tapped
    /// trunk hidden states, run bidirectionally over a masked block so one
    /// pass proposes [`DFlash::block`] tokens. Declared as [`DFlash`], not
    /// [`Mtp`] — it shares neither the fusion shape nor the attention shape.
    DFlash,
}

impl Recipe {
    /// The plane prefix this recipe's head is named under.
    #[must_use]
    pub fn prefix(self) -> &'static str {
        match self {
            Recipe::Mtp => "mtp",
            Recipe::Eagle | Recipe::DFlash => "aux",
        }
    }
}

/// The vision tower: a windowed region of the same fire whose rows are
/// patches and whose lanes are images.
///
/// ```text
/// y  = patch_embed(x) + pos_embed[interp(grid)]
/// per block:
///   y += proj(dense_attn(rope(qkv(LN(y)))))
///   y += fc2(gelu(fc1(LN(y))))
/// out = mfc2(gelu(mfc1(merge(LN(y)))))
/// ```
///
/// Norms are real `nn.LayerNorm`, not RMSNorm. MLP is ungated. Rotation is
/// two-axis, block-laid ([`MropeForm::Blocked`]), no time axis. Replicated
/// whole under `tp` rather than sharded — too small to be worth a collective.
///
/// [`MropeForm::Blocked`]: model_dsl::MropeForm::Blocked
pub struct Tower {
    /// The tower's own residual width — NOT the trunk's.
    pub hidden: u32,
    pub heads: u32,
    /// `hidden / heads`, stated rather than derived (same reason as
    /// `Model::head_dim`).
    pub head_dim: u32,
    /// `spatial_merge_size`: the merger folds `merge²` consecutive patch rows
    /// into one (`layout.merge_rows`).
    pub merge: u32,
    /// `C · T · P²`: width of one pre-unfolded patch row, the carve size
    /// `Input::patches` uses.
    pub patch_width: u32,
    /// Rows of [`pos_embed`](Tower::pos_embed) gathered per patch: 4 for
    /// bilinear resample of a non-native grid. Could be 1 only for a
    /// deployment that never resizes.
    pub taps: u32,
    /// `num_position_embeddings`: learned table row count, the square of the
    /// grid side the host resamples from.
    pub positions: u32,
    pub theta: f32,
    pub norm_eps: f32,
    pub sm_scale: f32,
    /// `[hidden, patch_width]`: patch embedding as a matmul (patches arrive
    /// as vectors, not images).
    pub patch_embed: Weight,
    pub patch_embed_bias: Weight,
    /// `[positions, hidden]`, gathered per patch row.
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

/// The patch merger: the one place the patch rectangle's row count changes.
/// Norm runs before the merge shuffle, on `[hidden]`, not `[merge²·hidden]`.
pub struct Merger {
    pub norm: Weight,
    pub norm_bias: Weight,
    /// `[merge²·hidden, merge²·hidden]`.
    pub fc1: Weight,
    pub fc1_bias: Weight,
    /// `[out_hidden, merge²·hidden]`; `out_hidden` is the trunk's width.
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
    /// site.
    ///
    /// Sits on the mixer sublayer's replicated input/output, after the
    /// `all_reduce` — not on `o_proj`'s own (rank-cut) output, which would
    /// have every rank add the full `ΔW·x` and sum it `tp` times.
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

/// A tower's own numbers, read off `config.json`'s `vision_config`. Separate
/// from [`Dims`] since nothing here divides by `tp` (the tower is
/// replicated). `out_hidden` is the trunk's `hidden`, asserted against it
/// rather than restated.
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
    /// Rotary embedding class default; not set by either SKU's `vision_config`.
    theta: f32,
    norm_eps: f32,
    /// How many table rows a patch's position gathers ([`Tower::taps`]).
    taps: u32,
}

impl TowerDims {
    /// The two shipped towers share `patch_width`, `merge`, `positions` and
    /// `theta`/`norm_eps`; they differ in depth, hidden, heads, inter and
    /// `out_hidden`.
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
    /// Which draft-head recipe this SKU's artifact carries, or `None`. No
    /// layer count: every shipped draft head (either recipe) is exactly one
    /// decoder layer.
    draft: Option<Recipe>,
}

impl Model {
    pub fn a3b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a3b_dims())
    }

    /// The A3B with its published MTP head (`mlx-community/Qwen3.6-35B-A3B-MTP-4bit`,
    /// the head alone, overlaid by `--aux`): one dense block over the mixture.
    pub fn a3b_mtp(w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::a3b_dims();
        d.draft = Some(Recipe::Mtp);
        Model::new(w, kv, tp, d)
    }

    /// The shipped A3B geometry, factored out so the miniature below can move
    /// the two numbers it moves and nothing else.
    fn a3b_dims() -> Dims {
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
        }
    }

    /// Width-invariance fixture (`mini-l5-e16-k8`), carved from the shipped
    /// A3B checkpoint. Only depth (5) and expert count (16) change from
    /// `a3b_dims`; everything else stays production width so accumulation
    /// order over K stays comparable. `top_k` stays 8 to keep a contested
    /// tail among the 16 experts.
    pub fn a3b_mini(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a3b_mini_dims(16))
    }

    /// Same fixture with a crowded tail (`mini-l5-e64-k8`): 64 routed
    /// experts instead of 16, top-k still 8, so the router has more rejected
    /// experts to break ties among. Only the expert count differs from
    /// `a3b_mini`.
    pub fn a3b_mini64(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::a3b_mini_dims(64))
    }

    /// The miniature's geometry; only `experts` varies between the two
    /// public rows above.
    fn a3b_mini_dims(experts: u32) -> Dims {
        let mut d = Model::a3b_dims();
        d.layers = 5;
        let MlpDims::Routed(moe) = &mut d.mlp else {
            unreachable!("the a3b dims carry a routed MLP")
        };
        moe.experts = experts;
        d
    }

    /// Test-only routed MoE, `a3b`'s shape scaled down to fit two copies on
    /// one device. Not a catalog row — no checkpoint ships it. Used to check
    /// that a load holding half the experts produces the same logits as
    /// full residency. `attn_every: 1` isolates the MLP's expert banks from
    /// the (orthogonal) GDN mixer; `tied: true` drops the `lm_head` plane
    /// since the dense floor isn't under test.
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

    /// Same 24-layer trunk as `d0_8b`, with an EAGLE head from a second
    /// checkpoint overlaid via `pie model import <base> --aux <head>`. A
    /// separate row rather than an optional field, since whether the head
    /// exists is a fact about which artifact was imported.
    pub fn d0_8b_eagle(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d0_8b_dims(None, Some(Recipe::Eagle)))
    }

    /// Same 24-layer trunk as `d0_8b`, reading the 12-block tower its own
    /// checkpoint ships. A separate row rather than an optional tower field,
    /// since a tower makes this a two-unit plan.
    pub fn d0_8b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d0_8b_dims(Some(TowerDims::qwen35()), None),
        )
    }

    /// Trunk, tower and EAGLE head together. Needed because vision rows are
    /// ordered ahead of the plain eagle row, so a checkpoint that has both
    /// the tower and the overlaid head would otherwise match the vision row
    /// and never bind its draft head.
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

    /// Qwen3.6-27B: a SKU of this family, not a separate one —
    /// `config.json` names itself `qwen3_5`. `attn_every = 4` (3 linear : 1
    /// full attention), `q_proj` is gated (`attn_output_gate`), `rotary_dim
    /// = 64` (`partial_rotary_factor: 0.25` of `head_dim: 256`). Adds the
    /// `mtp.*` draft head. Text-only reading: the checkpoint also ships a
    /// 27-block tower and interleaved mrope, neither declared here.
    pub fn d27b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, Some(Recipe::Mtp)))
    }

    /// The 27B with a DFlash block drafter overlaid by `--aux`
    /// (`z-lab/Qwen3.6-27B-DFlash`, the drafter alone: `fc.*`, `layers.0..4.*`,
    /// `hidden_norm`, `norm` at its root). The trunk is `d27b_undrafted`'s —
    /// the drafter brings its own layers rather than sharing one.
    pub fn d27b_dflash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, Some(Recipe::DFlash)))
    }

    /// Same 64 layers without the draft head. Used for 4-bit conversions:
    /// `mlx_lm` implements no MTP arm for this family, so those artifacts
    /// carry none of the `mtp.*` planes.
    pub fn d27b_undrafted(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::d27b_dims(None, None))
    }

    /// Both the 27-block tower and the `mtp.*` head — two independent
    /// fields, so this row is `d27b`'s dims plus a tower.
    pub fn d27b_vision(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d27b_dims(Some(TowerDims::qwen36()), Some(Recipe::Mtp)),
        )
    }

    /// Tower without the draft head — the pairing the 4-bit artifact
    /// actually ships (tower present, no `mtp.*` planes).
    pub fn d27b_vision_undrafted(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Model::d27b_dims(Some(TowerDims::qwen36()), None),
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
        // Non-matmul values (norms, depthwise conv, deltanet per-head
        // biases, adapter planes) use `dense`, not `w`. See `crate::dense`.
        let dense = crate::dense(w);
        // Router gates (`mlp.gate`, `mlp.shared_expert_gate`) are stored at
        // 8 bits even when the rest of a 4-bit stack is 4-bit; a bf16 stack
        // keeps its gates at bf16.
        let gate = match w {
            Dtype::U4g64 => Dtype::U8g64,
            other => other,
        };
        // `U4g64tiled` reorders `U4g64` codes into m16n8k16 fragment order
        // for `linear::tiled`. Applies only to 2D dense projections read by
        // `ops::linear::matmul`: `embed`/tied head (gather), routed expert
        // banks (3D, grouped select) and the MTP `fc` slices stay row-major
        // since none have a tiled reader. `model_dsl::place` resolves the
        // actual per-platform layout: CUDA gets tiled, Metal gets canonical
        // `U4g64` (Metal's quant kernels have no fragment-order reader).
        let proj = match w {
            Dtype::U4g64 => Dtype::U4g64tiled,
            other => other,
        };
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
                        proj,
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
                        in_qkvz: Weight::sym(n("in_qkvz"), [qkvz, hidden], proj)
                            .packed([k_w, k_w, v_w, v_w]),
                        in_ba: Weight::sym(n("in_ba"), [2 * v_heads as u64, hidden], proj)
                            .packed([v_heads as u64, v_heads as u64]),
                        // conv and dt_bias stay unquantized: neither has 64
                        // values to group.
                        conv: Weight::sym(n("conv"), [qkv, d.conv_kernel as u64], dense)
                            .packed([k_w, k_w, v_w]),
                        dt_bias: Weight::sym(n("dt_bias"), [v_heads as u64], dense).columns(),
                        a_log: Weight::sym(n("a_log"), [v_heads as u64], Dtype::F32).columns(),
                        norm: Weight::sym(n("gdn_norm"), [d.v_dim as u64], Dtype::F32),
                        norm_eps: d.norm_eps,
                        out_proj: Weight::sym(n("out_proj"), [hidden, v_w], proj).rows(),
                        conv_state: format!("conv.{l}"),
                        delta_state: format!("delta.{l}"),
                    })
                };
                let mlp = match &d.mlp {
                    MlpDims::Dense { inter } => {
                        dense_mlp(proj, hidden, inter / tp, &format!("layer.{l}"))
                    }
                    MlpDims::Routed(m) => {
                        let inter = m.inter / tp;
                        let shared_inter = m.shared_inter / tp;
                        Mlp::Routed {
                            router: Weight::sym(n("router"), [m.experts as u64, hidden], gate),

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
                                proj,
                            )
                            .packed([shared_inter as u64, shared_inter as u64]),
                            shared_down: Weight::sym(
                                n("shared_down"),
                                [hidden, shared_inter as u64],
                                proj,
                            )
                            .rows(),
                            shared_gate: Weight::sym(n("shared_gate"), [1, hidden], gate),
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
                    // Replicated under tp: both ends (normed residual in,
                    // reduced mixer result out) are replicated, so nothing
                    // here is summed twice.
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        // Draft head's mlp is always dense at the trunk's width, even when
        // the trunk routes — one block, no experts to route to.
        //
        // Tower, when published: every plane replicated, names under
        // `visual.` (checkpoint's own namespace, `model.` stripped).
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
            // Every tower plane is `dense(w)`, merger included: 4-bit
            // checkpoints still ship the tower unquantized, so a
            // `-vision-u4g64` row is a bf16 tower over a U4 trunk.
            let plane = |s: String, dims: [u64; 2]| Weight::sym(n(s), dims, dense);
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

        // kv row stays `kv.mtp` under either recipe — a fact about this
        // plan's page-id space, not about which checkpoint the bytes came
        // from.
        let mtp = d.draft.filter(|r| !matches!(r, Recipe::DFlash)).map(|recipe| {
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
                // Replicated, both halves: token embedding and the trunk's
                // reduced residual stream are both replicated values.
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

        // The block drafter's geometry is its OWN (see `DFlash`), so nothing
        // here is read off `Dims` but the trunk's hidden width and the
        // element types.
        let dflash = d.draft.filter(|r| matches!(r, Recipe::DFlash)).map(|recipe| {
            let p = recipe.prefix();
            let n = |s: &str| format!("{p}.{s}");
            let (dq, dkv, dhd) = (DFLASH_Q_HEADS / tp, DFLASH_KV_HEADS / tp, DFLASH_HEAD_DIM);
            let hd = dhd as u64;
            let inter = DFLASH_INTER / tp;
            DFlash {
                taps: DFLASH_TAPS.to_vec(),
                // One column slice of the stored `[hidden, 5·hidden]` bank
                // per tap; replicated, since a trunk hidden state is.
                fc: (0..DFLASH_TAPS.len())
                    .map(|i| Weight::sym(n(&format!("fc_tap{i}")), [hidden, hidden], w))
                    .collect(),
                hidden_norm: Weight::sym(n("hidden_norm"), [hidden], dense),
                hidden_norm_eps: d.norm_eps,
                blocks: (0..DFLASH_LAYERS)
                    .map(|l| {
                        let b = |s: &str| format!("{p}.layers.{l}.{s}");
                        DFlashBlock {
                            mixer_norm: Weight::sym(b("mixer_norm"), [hidden], dense),
                            mixer_norm_eps: d.norm_eps,
                            attn: DraftAttn {
                                q_heads: dq,
                                kv_heads: dkv,
                                head_dim: dhd,
                                rotary_dim: dhd,
                                theta: DFLASH_THETA,
                                sm_scale: (dhd as f32).sqrt().recip(),
                                q_proj: Weight::sym(b("q_proj"), [dq as u64 * hd, hidden], w).columns(),
                                k_proj: Weight::sym(b("k_proj"), [dkv as u64 * hd, hidden], w).columns(),
                                v_proj: Weight::sym(b("v_proj"), [dkv as u64 * hd, hidden], w).columns(),
                                o_proj: Weight::sym(b("o_proj"), [hidden, dq as u64 * hd], w).rows(),
                                q_norm: Weight::sym(b("q_norm"), [hd], dense),
                                q_norm_eps: d.norm_eps,
                                k_norm: Weight::sym(b("k_norm"), [hd], dense),
                                k_norm_eps: d.norm_eps,
                                kv: format!("kv.dflash.{l}"),
                            },
                            mlp_norm: Weight::sym(b("mlp_norm"), [hidden], dense),
                            mlp_norm_eps: d.norm_eps,
                            mlp: dense_mlp(w, hidden, inter, &format!("{p}.layers.{l}")),
                            // The shipped head is four sliding layers then
                            // one full; `layer_types` in its config.
                            window: (l + 1 < DFLASH_LAYERS).then_some(DFLASH_WINDOW),
                        }
                    })
                    .collect(),
                norm: Weight::sym(n("norm"), [hidden], dense),
                norm_eps: d.norm_eps,
                block: DFLASH_BLOCK,
                mask_token: DFLASH_MASK_TOKEN,
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
            dflash,
        }
    }
}

/// **The shipped DFlash drafter's geometry** (`z-lab/Qwen3.6-27B-DFlash`'s
/// `config.json`), which is the head's own and not the trunk's — see
/// [`DFlash`]. Constants rather than [`Dims`] fields because exactly one SKU
/// carries this head and every number below is a fact about that published
/// checkpoint, not a knob.
const DFLASH_TAPS: [u32; 5] = [1, 16, 31, 46, 61];
const DFLASH_LAYERS: u32 = 5;
const DFLASH_Q_HEADS: u32 = 32;
const DFLASH_KV_HEADS: u32 = 8;
const DFLASH_HEAD_DIM: u32 = 128;
const DFLASH_INTER: u32 = 17_408;
const DFLASH_THETA: f32 = 10_000_000.0;
const DFLASH_WINDOW: u32 = 2_048;
pub const DFLASH_BLOCK: u32 = 16;
/// `dflash_config.mask_token_id` — the id every block row but the first
/// carries into the draft pass. Public with [`DFLASH_BLOCK`] because a guest
/// seeding a draft block needs both; they belong on the load's own
/// advertisement (beside `mtp_depth`) once there is a guest asking.
pub const DFLASH_MASK_TOKEN: u32 = 248_070;

/// Adapter ceiling for every SKU of this family. Not a checkpoint fact (no
/// pretrained artifact states it) — a deployment setting baked in at trace
/// time; changing it means re-tracing.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

/// One gated full-attention site's banks, named under `prefix`. Shared by
/// trunk and draft head — `mtp.layers.0.self_attn.*` matches a trunk
/// layer's shapes tensor for tensor. `q_heads`/`kv_heads` passed in here are
/// already per-rank (cut by `tp`).
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
/// mlp is the trunk's at the same intermediate width.
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
 }
