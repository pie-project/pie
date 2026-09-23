use model_dsl::ops::elemwise::Yarn;
use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub act: Dtype,

    pub heads: u32,
    pub head_dim: u32,
    pub window: u32,

    pub adapters: Adapters,

    pub kv: Dtype,
    pub hyper: Hyper,

    pub embed: Weight,
    pub head: Option<Weight>,
    pub hc_head: Option<HcHead>,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
    pub mtp: Option<Mtp>,
    /// V4.1: the tokenizer-compressed id every token hashes as in an Engram
    /// n-gram (`engram.token_map`, one i32 per vocabulary row).
    pub token_map: Option<Weight>,
}

pub struct Mtp {
    pub enorm: Weight,
    pub hnorm: Weight,
    pub e_proj: Weight,
    pub h_proj: Weight,
    pub block: Layer,
    pub hc_head: HcHead,
    pub norm: Weight,
    pub norm_eps: f32,
    pub depth: u32,
}

struct Site {
    prefix: String,
    ratio: Option<u32>,
    hash: bool,
    experts: u32,
    split: bool,
    gate: Dtype,
    up: Dtype,
    down: Dtype,
    weights: Dtype,
    dense: Dtype,
    kv: String,
    pool: String,
    index: String,
}

pub struct Hyper {
    pub streams: u32,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,
    /// V4.1 Single-Pass mHC: a sub-block mixes its input with the coefficients
    /// the *previous* sub-block predicted, so the residual is read once.
    pub single_pass: bool,
}

pub struct HcHead {
    pub base: Weight,
    pub dynamic: Weight,
    pub scale: Weight,
}

pub struct Mix {
    pub scale: Weight,
    pub base: Weight,
    pub dynamic: Option<Weight>,
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub attn_mix: Mix,
    pub attn_norm: Option<Weight>,
    pub attn: Attn,
    pub mlp_mix: Mix,
    pub mlp_norm: Option<Weight>,
    pub mlp: Mlp,
    pub lora_a: Weight,
    pub lora_b: Weight,
    /// V4.1: the conditional-memory module written into the residual streams
    /// before this layer runs.
    pub engram: Option<Engram>,
}

/// Where a layer's sparse selection over the compressed entries comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// No compressed branch (sliding window only).
    None,
    /// This layer runs an indexer and ranks the entries itself.
    Own,
    /// CSA2 Reuse Mode: the latest ranking a preceding layer produced.
    Shared,
}

pub struct Attn {
    pub rope_dim: u32,
    pub theta: f32,
    pub yarn: Option<Yarn>,
    pub sm_scale: f32,
    pub q_down: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub q_up: Weight,
    pub kv_down: Weight,
    pub kv_norm: Weight,
    pub kv_norm_eps: f32,
    pub o_down: Weight,
    pub o_up: Weight,
    pub o_groups: u32,
    pub sink: Weight,
    pub kv: String,
    pub pool: Option<Pool>,
    pub indexer: Option<Indexer>,
    pub selection: Selection,
}

pub struct Pool {
    pub ratio: u32,
    pub entries: String,
    pub compressor: Option<Compressor>,
    /// Whether this layer writes the entries (`true`) or reads a preceding
    /// layer's (CSA2 cross-layer KV reuse, `false`).
    pub owner: bool,
}

pub struct Compressor {
    pub wkv: Weight,
    /// The softmax gate over the pooled positions; absent at ratio 1 (a plain
    /// projection) in V4.1.
    pub wgate: Option<Weight>,
    /// V4's absolute positional embedding over the pooled window; V4.1 has none.
    pub ape: Option<Weight>,
    pub norm: Weight,
    pub norm_eps: f32,
}

pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub rope_dim: u32,
    pub theta: f32,
    pub yarn: Option<Yarn>,
    pub window: u32,
    pub wq_b: Weight,
    pub weights_proj: Weight,
    /// V4: the index keys come from their own compressor over the hidden state.
    pub compressor: Option<Compressor>,
    /// V4.1: the index keys are projected from the compressed KV latent.
    pub wk: Option<Weight>,
    pub k_norm: Option<Weight>,
    pub keys: String,
    /// Whether this layer writes the index keys or reads a preceding layer's.
    pub owns_keys: bool,
}

/// Engram (Cheng et al., 2026): an n-gram hash lookup gated into every
/// residual stream. The position's 2..N-grams over tokenizer-compressed ids
/// are hashed per head into disjoint prime-sized bucket ranges of one table.
pub struct Engram {
    pub table: Weight,
    pub wkv: Weight,
    /// Stored as `q_weight - 1` / `k_weight - 1`: the gate's per-stream
    /// normalisations are `rmsnorm_grouped_plus_one`s.
    pub q_weight: Weight,
    pub k_weight: Weight,
    pub mults: Vec<u64>,
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub ngram: u32,
    pub heads_per_ngram: u32,
    pub head_dim: u32,
    /// The raw id that fills n-gram slots before the sequence starts (mapped
    /// through the token map like every other id).
    pub pad: u32,
    pub eps: f32,
    pub ids_state: String,
}

pub enum Gate {
    Hash { tid2eid: Weight },
    Bias { bias: Weight },
}

pub enum GateUp {
    Fused(Weight),
    Split { gate: Weight, up: Weight },
}

#[allow(clippy::large_enum_variant)]
pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
        limit: f32,
    },
    Routed {
        router: Weight,
        bias: Weight,
        gate_up: Weight,
        down: Weight,
        experts: u32,
        top_k: u32,
        inter: u32,
        limit: f32,
        renorm: bool,
        scaling: f32,
    },
    MoeFlash {
        router: Weight,
        gate: Gate,
        gate_up: GateUp,
        down: Weight,
        shared_gate_up: Weight,
        shared_down: Weight,
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
        limit: f32,
        renorm: bool,
        scaling: f32,
    },
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    pool: &'static [Option<u32>],
    heads: u32,
    head_dim: u32,
    q_lora: u32,
    o_lora: u32,
    rope_dim: u32,
    theta: f32,
    window: u32,
    streams: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    dense_inter: u32,
    experts: u32,
    top_k: u32,
    moe_inter: u32,
    renorm: bool,
    scaling: f32,
    swiglu_limit: f32,
    vocab: u32,
    norm_eps: f32,
}

struct FlashDims {
    hidden: u32,
    layers: u32,
    pool: &'static [Option<u32>],
    num_hash_layers: u32,
    heads: u32,
    head_dim: u32,
    q_lora: u32,
    kv_latent: u32,
    o_groups: u32,
    o_lora: u32,
    rope_dim: u32,
    theta: f32,
    compress_theta: f32,
    yarn: Yarn,
    draft: bool,
    window: u32,
    index_heads: u32,
    index_head_dim: u32,
    index_top_k: u32,
    index_window: u32,
    streams: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    experts: u32,
    top_k: u32,
    moe_inter: u32,
    shared_inter: u32,
    renorm: bool,
    scaling: f32,
    swiglu_limit: f32,
    vocab: u32,
    norm_eps: f32,
}

const FLASH_RATIOS: [Option<u32>; 43] = flash_ratios();

const fn flash_ratios() -> [Option<u32>; 43] {
    let mut out = [None; 43];
    let mut layer = 2;
    while layer < 43 {
        out[layer] = if layer % 2 == 0 { Some(4) } else { Some(128) };
        layer += 1;
    }
    out
}

const FLASH_MICRO_RATIOS: [Option<u32>; 5] = [None, None, Some(4), Some(128), Some(4)];

#[derive(Clone, Copy, Debug)]
pub struct Routed {
    pub gate: Dtype,
    pub gate_at: &'static [(u32, Dtype)],
    pub up: Dtype,
    pub down: Dtype,
    pub split: bool,
}

impl Routed {
    #[must_use]
    pub const fn uniform(w: Dtype) -> Routed {
        Routed {
            gate: w,
            gate_at: &[],
            up: w,
            down: w,
            split: false,
        }
    }

    /// One dtype for every routed plane, stored as separate gate / up planes
    /// (the V4.1 checkpoints keep `w1` and `w3` apart).
    #[must_use]
    pub const fn split(w: Dtype) -> Routed {
        Routed {
            gate: w,
            gate_at: &[],
            up: w,
            down: w,
            split: true,
        }
    }

    pub const DQ_2BIT: Routed = Routed {
        gate: Dtype::U2g32,
        gate_at: &[(4, Dtype::U2g64)],
        up: Dtype::U2g64,
        down: Dtype::U2g64,
        split: true,
    };

    pub const DQ_2BIT_FULL: Routed = Routed {
        gate: Dtype::U2g32,
        gate_at: &[(42, Dtype::U2g64)],
        up: Dtype::U2g64,
        down: Dtype::U2g64,
        split: true,
    };

    #[must_use]
    pub fn gate_of(&self, layer: u32) -> Dtype {
        self.gate_at
            .iter()
            .find_map(|(at, dtype)| (*at == layer).then_some(*dtype))
            .unwrap_or(self.gate)
    }
}

// ---------------------------------------------------------------------------
// DeepSeek-V4.1-Flash: Causal Encoder-Decoder over CSA2 with cross-layer KV
// and index reuse, Single-Pass mHC, and Engram conditional memory.
// ---------------------------------------------------------------------------

/// How a V4.1 layer takes part in Compressed Sparse Attention 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Csa {
    /// Sliding window only (the first two layers).
    Swa,
    /// Computes the compressed KV, projects the index keys from it, and ranks.
    Full,
    /// Reuses a preceding layer's compressed KV and index keys; ranks afresh.
    Reindex,
    /// Reuses the compressed KV and the latest ranking.
    Reuse,
}

/// The published layer plan: `compress_ratios`, `kv_source_layer_ids`,
/// `index_source_layer_ids` and `engram_layer_ids` of the checkpoint's config.
#[derive(Clone, Copy, Debug)]
pub struct Plan41 {
    pub layers: u32,
    pub ratios: &'static [u32],
    pub kv_sources: &'static [u32],
    pub index_sources: &'static [u32],
    pub engram_layers: &'static [u32],
}

const V41_RATIOS: [u32; 40] = v41_ratios();

const fn v41_ratios() -> [u32; 40] {
    let mut out = [1; 40];
    out[0] = 0;
    out[1] = 0;
    let mut layer = 2;
    while layer < 20 {
        out[layer] = 2;
        layer += 1;
    }
    out
}

pub const V41_PLAN: Plan41 = Plan41 {
    layers: 40,
    ratios: &V41_RATIOS,
    kv_sources: &[2, 8, 14, 20],
    index_sources: &[2, 8, 14, 20, 24, 28, 32, 36],
    engram_layers: &[1, 14],
};

/// The miniature: source layers `0,1,2,3,20,21,24,25` of the release, so every
/// layer kind is present once — SWA (0), SWA with Engram (1), Full at ratio 2
/// (2), Reuse at ratio 2 (3), the decoder's Full at ratio 1 (20), Reuse (21),
/// Reindex (24) and a Reuse after it (25) — with 16 of the 384 experts and an
/// Engram table hashed at base 20 000 instead of 16 000 000.
pub const V41_MINI_PLAN: Plan41 = Plan41 {
    layers: 8,
    ratios: &[0, 0, 2, 2, 1, 1, 1, 1],
    kv_sources: &[2, 4],
    index_sources: &[2, 4, 6],
    engram_layers: &[1],
};

pub const V41_MINI_LAYERS: [u32; 8] = [0, 1, 2, 3, 20, 21, 24, 25];
pub const V41_MINI_EXPERTS: u32 = 16;
pub const V41_MINI_ENGRAM_BASE: u64 = 20_000;

#[derive(Clone, Copy)]
struct EngramDims {
    ngram: u32,
    heads: u32,
    head_dim: u32,
    base_vocab: u64,
    compressed_vocab: u64,
    pad: u32,
}

struct Dims41 {
    hidden: u32,
    plan: Plan41,
    heads: u32,
    head_dim: u32,
    q_lora: u32,
    kv_latent: u32,
    o_groups: u32,
    o_lora: u32,
    rope_dim: u32,
    theta: f32,
    compress_theta: f32,
    yarn: Yarn,
    window: u32,
    index_heads: u32,
    index_head_dim: u32,
    index_top_k: u32,
    streams: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    experts: u32,
    top_k: u32,
    moe_inter: u32,
    shared_inter: u32,
    renorm: bool,
    scaling: f32,
    swiglu_limit: f32,
    vocab: u32,
    norm_eps: f32,
    engram: EngramDims,
}

impl Model {
    pub fn base(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            act,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 6,
                dense_layers: 1,
                pool: &[Some(1), Some(2), Some(4), None, None, None],
                heads: 16,
                head_dim: 128,
                q_lora: 768,
                o_lora: 512,
                rope_dim: 64,
                theta: 10_000.0,
                window: 2048,
                streams: 4,
                gate_eps: 1e-6,
                alpha: 2.0,
                sinkhorn: 20,
                dense_inter: 5632,
                experts: 64,
                top_k: 6,
                moe_inter: 1024,
                renorm: false,
                scaling: 2.5,
                swiglu_limit: 7.0,
                vocab: 129_280,
                norm_eps: 1e-5,
            },
        )
    }

    pub fn flash(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::flash_mixed(w, Routed::uniform(w), act, kv, tp)
    }

    pub fn flash_mixed(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new_flash(
            w,
            routed,
            act,
            kv,
            tp,
            Model::flash_dims(43, &FLASH_RATIOS, 3),
        )
    }

    pub fn flash_mixed_mtp(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(43, &FLASH_RATIOS, 3);
        d.draft = true;
        Model::new_flash(w, routed, act, kv, tp, d)
    }

    pub fn flash_mini_mtp(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(5, &FLASH_MICRO_RATIOS, 3);
        d.experts = 16;
        d.draft = true;
        Model::new_flash(w, routed, act, kv, tp, d)
    }

    pub fn flash_mini(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(5, &FLASH_MICRO_RATIOS, 3);
        d.experts = 16;
        Model::new_flash(w, routed, act, kv, tp, d)
    }

    pub fn flash_micro(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(5, &FLASH_MICRO_RATIOS, 3);
        d.hidden = 256;
        d.heads = 8;
        d.head_dim = 64;
        d.rope_dim = 16;
        d.q_lora = 128;
        d.kv_latent = 64;
        d.o_groups = 2;
        d.o_lora = 128;
        d.index_heads = 8;
        d.index_head_dim = 32;
        d.index_top_k = 16;
        d.moe_inter = 64;
        d.shared_inter = 64;
        d.experts = 16;
        d.vocab = 512;
        Model::new_flash(w, Routed::uniform(w), act, kv, tp, d)
    }

    /// DeepSeek-V4.1-Flash as released: 40 layers, 384 routed experts.
    pub fn flash41(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new_flash41(w, routed, act, kv, tp, Model::dims41(V41_PLAN))
    }

    /// The eight-layer, sixteen-expert miniature of DeepSeek-V4.1-Flash.
    pub fn flash41_mini(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::dims41(V41_MINI_PLAN);
        d.experts = V41_MINI_EXPERTS;
        d.engram.base_vocab = V41_MINI_ENGRAM_BASE;
        Model::new_flash41(w, routed, act, kv, tp, d)
    }

    fn dims41(plan: Plan41) -> Dims41 {
        Dims41 {
            hidden: 5120,
            plan,
            heads: 64,
            head_dim: 512,
            q_lora: 1280,
            kv_latent: 512,
            o_groups: 8,
            o_lora: 1024,
            rope_dim: 64,
            theta: 10_000.0,
            compress_theta: 160_000.0,
            yarn: Yarn {
                factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 65_536,
            },
            window: 128,
            index_heads: 32,
            index_head_dim: 128,
            index_top_k: 512,
            streams: 4,
            gate_eps: 1e-6,
            alpha: 2.0,
            sinkhorn: 20,
            experts: 384,
            top_k: 6,
            moe_inter: 2304,
            shared_inter: 2304,
            renorm: true,
            scaling: 1.5,
            swiglu_limit: 10.0,
            vocab: 129_280,
            norm_eps: 1e-20,
            engram: EngramDims {
                ngram: 4,
                heads: 8,
                head_dim: 256,
                base_vocab: 16_000_000,
                compressed_vocab: 99_092,
                pad: 2,
            },
        }
    }

    fn flash_dims(layers: u32, pool: &'static [Option<u32>], hash: u32) -> FlashDims {
        FlashDims {
            hidden: 4096,
            layers,
            pool,
            num_hash_layers: hash,
            heads: 64,
            head_dim: 512,
            q_lora: 1024,
            kv_latent: 512,
            o_groups: 8,
            o_lora: 1024,
            rope_dim: 64,
            theta: 10_000.0,
            compress_theta: 160_000.0,
            yarn: Yarn {
                factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 65_536,
            },
            draft: false,
            window: 128,
            index_heads: 64,
            index_head_dim: 128,
            index_top_k: 512,
            index_window: 128,
            streams: 4,
            gate_eps: 1e-6,
            alpha: 2.0,
            sinkhorn: 20,
            experts: 256,
            top_k: 6,
            moe_inter: 2048,
            shared_inter: 2048,
            renorm: true,
            scaling: 1.5,
            swiglu_limit: 10.0,
            vocab: 129_280,
            norm_eps: 1e-6,
        }
    }

    fn new(weights: Dtype, act: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );

        let heads = d.heads / tp;
        let dense_inter = d.dense_inter / tp;
        let moe_inter = d.moe_inter / tp;

        let hidden = d.hidden as u64;
        let streams = d.streams as u64;
        let q_w = heads as u64 * d.head_dim as u64;
        let q_lora = d.q_lora as u64;
        let o_lora = d.o_lora as u64;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], weights);
                let mix = |s: &str| Mix {
                    scale: Weight::sym(n(&format!("{s}_scale")), [3], Dtype::F32),
                    base: Weight::sym(
                        n(&format!("{s}_base")),
                        [2 * streams + streams * streams],
                        Dtype::F32,
                    ),
                    dynamic: None,
                };
                let (lora_a, lora_b) = crate::adapter::banks(
                    &format!("layer.{l}"),
                    ADAPTERS,
                    hidden,
                    crate::dense(weights),
                );
                Layer {
                    attn_mix: mix("attn_mix"),
                    attn_norm: None,
                    mlp_norm: None,
                    attn: Attn {
                        rope_dim: d.rope_dim,
                        theta: d.theta,
                        yarn: None,
                        sm_scale: (d.head_dim as f32).sqrt().recip(),
                        q_down: Weight::sym(n("q_down"), [q_lora, hidden], weights),
                        q_norm: norm("q_norm", q_lora),
                        q_norm_eps: d.norm_eps,
                        q_up: Weight::sym(n("q_up"), [q_w, q_lora], weights).columns(),
                        kv_down: Weight::sym(n("kv_down"), [q_w, hidden], weights).columns(),
                        kv_norm: Weight::sym(n("kv_norm"), [q_w], weights).columns(),
                        kv_norm_eps: d.norm_eps,
                        o_down: Weight::sym(n("o_down"), [o_lora, q_w], weights).rows(),
                        o_up: Weight::sym(n("o_up"), [hidden, o_lora], weights),
                        o_groups: 1,
                        sink: Weight::sym(n("attn_sink"), [heads as u64], weights).columns(),
                        kv: format!("kv.{l}"),
                        pool: d.pool[l as usize].map(|ratio| Pool {
                            ratio,
                            entries: format!("pool.{l}"),
                            compressor: None,
                            owner: true,
                        }),
                        indexer: None,
                        selection: Selection::None,
                    },
                    mlp_mix: mix("mlp_mix"),
                    mlp: if l < d.dense_layers {
                        Mlp::Dense {
                            gate_up: Weight::sym(
                                n("gate_up"),
                                [2 * dense_inter as u64, hidden],
                                weights,
                            )
                            .packed([dense_inter as u64, dense_inter as u64]),
                            down: Weight::sym(n("down"), [hidden, dense_inter as u64], weights)
                                .rows(),
                            inter: dense_inter,
                            limit: d.swiglu_limit,
                        }
                    } else {
                        Mlp::Routed {
                            router: Weight::sym(n("router"), [d.experts as u64, hidden], weights),
                            bias: Weight::sym(n("router_bias"), [d.experts as u64], weights),
                            gate_up: Weight::sym(
                                n("experts_gate_up"),
                                [d.experts as u64, 2 * moe_inter as u64, hidden],
                                weights,
                            )
                            .bank([moe_inter as u64, moe_inter as u64]),
                            down: Weight::sym(
                                n("experts_down"),
                                [d.experts as u64, hidden, moe_inter as u64],
                                weights,
                            )
                            .rows(),
                            experts: d.experts,
                            top_k: d.top_k,
                            inter: moe_inter,
                            limit: d.swiglu_limit,
                            renorm: d.renorm,
                            scaling: d.scaling,
                        }
                    },
                    lora_a,
                    lora_b,
                    engram: None,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            act,
            heads,
            head_dim: d.head_dim,
            window: d.window,
            adapters: ADAPTERS,
            kv,
            hyper: Hyper {
                streams: d.streams,
                norm_eps: d.norm_eps,
                gate_eps: d.gate_eps,
                alpha: d.alpha,
                sinkhorn: d.sinkhorn,
                single_pass: false,
            },
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: None,
            hc_head: None,
            layers,
            final_norm: Weight::sym("final_norm", [hidden], weights),
            final_norm_eps: d.norm_eps,
            mtp: None,
            token_map: None,
        }
    }

    fn new_flash(
        weights: Dtype,
        routed: Routed,
        act: Dtype,
        kv: Dtype,
        tp: u32,
        d: FlashDims,
    ) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );

        let dense = crate::dense(weights);

        let heads = d.heads / tp;
        let moe_inter = d.moe_inter / tp;
        let shared_inter = d.shared_inter / tp;

        let hidden = d.hidden as u64;
        let streams = d.streams as u64;
        let hc_base = 2 * streams + streams * streams;
        let hc_fan = streams * hidden;
        let q_w = heads as u64 * d.head_dim as u64;
        let q_lora = d.q_lora as u64;
        let kv_latent = d.kv_latent as u64;
        let o_lora = d.o_lora as u64;
        let o_out = d.o_groups as u64 * o_lora;
        let idx_w = d.index_heads as u64 * d.index_head_dim as u64;
        let idx_norm_eps = d.norm_eps;

        let compressor = |prefix: String, ratio: u32, entries: u64, norm_w: u64| Compressor {
            wkv: Weight::sym(format!("{prefix}.wkv"), [entries, hidden], weights),
            wgate: Some(Weight::sym(
                format!("{prefix}.wgate"),
                [entries, hidden],
                weights,
            )),
            ape: Some(Weight::sym(
                format!("{prefix}.ape"),
                [ratio as u64, entries],
                Dtype::F32,
            )),
            norm: Weight::sym(format!("{prefix}.norm"), [norm_w], dense),
            norm_eps: d.norm_eps,
        };

        let layer_at = |site: Site| -> Layer {
            let prefix = site.prefix;
            let n = |s: &str| format!("{prefix}.{s}");
            let weights = site.weights;
            let dense = site.dense;
            let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], dense);
            let mix = |s: &str| Mix {
                scale: Weight::sym(n(&format!("{s}_scale")), [3], Dtype::F32),
                base: Weight::sym(n(&format!("{s}_base")), [hc_base], Dtype::F32),
                dynamic: Some(Weight::sym(
                    n(&format!("{s}_fn")),
                    [hc_base, hc_fan],
                    Dtype::F32,
                )),
            };
            let (lora_a, lora_b) = crate::adapter::banks(&prefix, ADAPTERS, hidden, dense);

            let ratio = site.ratio;
            let has_indexer = ratio == Some(4);
            let pool = ratio.map(|ratio| {
                let entries = if has_indexer {
                    2 * kv_latent
                } else {
                    kv_latent
                };
                Pool {
                    ratio,
                    entries: site.pool.clone(),
                    compressor: Some(compressor(n("compressor"), ratio, entries, kv_latent)),
                    owner: true,
                }
            });
            let indexer = has_indexer.then(|| Indexer {
                heads: d.index_heads,
                head_dim: d.index_head_dim,
                top_k: d.index_top_k,
                rope_dim: d.rope_dim,
                theta: d.compress_theta,
                yarn: Some(d.yarn),
                window: d.index_window,
                wq_b: Weight::sym(n("indexer.wq_b"), [idx_w, q_lora], weights),
                weights_proj: Weight::sym(
                    n("indexer.weights_proj"),
                    [d.index_heads as u64, hidden],
                    weights,
                ),
                compressor: Some(compressor(
                    n("indexer.compressor"),
                    ratio.unwrap_or(4),
                    2 * d.index_head_dim as u64,
                    d.index_head_dim as u64,
                )),
                wk: None,
                k_norm: None,
                keys: site.index.clone(),
                owns_keys: true,
            });

            let gate = if site.hash {
                Gate::Hash {
                    tid2eid: Weight::sym(
                        n("gate.tid2eid"),
                        [d.vocab as u64, d.top_k as u64],
                        Dtype::I64,
                    ),
                }
            } else {
                Gate::Bias {
                    bias: Weight::sym(n("gate.bias"), [site.experts as u64], Dtype::F32),
                }
            };

            Layer {
                attn_mix: mix("attn_mix"),
                attn_norm: Some(norm("attn_norm", hidden)),
                mlp_norm: Some(norm("ffn_norm", hidden)),
                attn: Attn {
                    rope_dim: d.rope_dim,
                    theta: if ratio.is_some() {
                        d.compress_theta
                    } else {
                        d.theta
                    },
                    yarn: ratio.map(|_| d.yarn),
                    sm_scale: (d.head_dim as f32).sqrt().recip(),
                    q_down: Weight::sym(n("q_down"), [q_lora, hidden], weights),
                    q_norm: norm("q_norm", q_lora),
                    q_norm_eps: d.norm_eps,
                    q_up: Weight::sym(n("q_up"), [q_w, q_lora], weights).columns(),
                    kv_down: Weight::sym(n("kv_down"), [kv_latent, hidden], weights),
                    kv_norm: norm("kv_norm", kv_latent),
                    kv_norm_eps: d.norm_eps,
                    o_down: Weight::sym(n("o_down"), [o_out, hidden], weights),
                    o_up: Weight::sym(n("o_up"), [hidden, o_out], weights).rows(),
                    o_groups: d.o_groups,
                    sink: Weight::sym(n("attn_sink"), [heads as u64], dense).columns(),
                    kv: site.kv.clone(),
                    pool,
                    indexer,
                    selection: if has_indexer {
                        Selection::Own
                    } else {
                        Selection::None
                    },
                },
                mlp_mix: mix("mlp_mix"),
                mlp: Mlp::MoeFlash {
                    router: Weight::sym(n("gate"), [site.experts as u64, hidden], dense),
                    gate,
                    gate_up: if site.split {
                        let half = |what: &str, dtype: Dtype| {
                            Weight::sym(
                                n(what),
                                [site.experts as u64, moe_inter as u64, hidden],
                                dtype,
                            )
                            .bank([moe_inter as u64])
                        };
                        GateUp::Split {
                            gate: half("experts_gate", site.gate),
                            up: half("experts_up", site.up),
                        }
                    } else {
                        GateUp::Fused(
                            Weight::sym(
                                n("experts_gate_up"),
                                [site.experts as u64, 2 * moe_inter as u64, hidden],
                                site.gate,
                            )
                            .bank([moe_inter as u64, moe_inter as u64]),
                        )
                    },
                    down: Weight::sym(
                        n("experts_down"),
                        [site.experts as u64, hidden, moe_inter as u64],
                        site.down,
                    )
                    .rows(),
                    shared_gate_up: Weight::sym(
                        n("shared_gate_up"),
                        [2 * shared_inter as u64, hidden],
                        weights,
                    )
                    .packed([shared_inter as u64, shared_inter as u64]),
                    shared_down: Weight::sym(
                        n("shared_down"),
                        [hidden, shared_inter as u64],
                        weights,
                    )
                    .rows(),
                    experts: site.experts,
                    top_k: d.top_k,
                    inter: moe_inter,
                    shared_inter,
                    limit: d.swiglu_limit,
                    renorm: d.renorm,
                    scaling: d.scaling,
                },
                lora_a,
                lora_b,
                engram: None,
            }
        };

        let layers = (0..d.layers)
            .map(|l| {
                layer_at(Site {
                    prefix: format!("layer.{l}"),
                    ratio: d.pool[l as usize],
                    hash: l < d.num_hash_layers,
                    experts: d.experts,
                    split: routed.split,
                    gate: routed.gate_of(l),
                    up: routed.up,
                    down: routed.down,
                    weights,
                    dense,
                    kv: format!("kv.{l}"),
                    pool: format!("pool.{l}"),
                    index: format!("index.{l}"),
                })
            })
            .collect();

        let mtp = d.draft.then(|| {
            let streams_n = d.streams as u64;
            Mtp {
                enorm: Weight::sym("mtp.enorm", [hidden], Dtype::Bf16),
                hnorm: Weight::sym("mtp.hnorm", [hidden], Dtype::Bf16),
                e_proj: Weight::sym("mtp.e_proj", [hidden, hidden], Dtype::Bf16),
                h_proj: Weight::sym("mtp.h_proj", [streams_n * hidden, hidden], Dtype::Bf16),
                block: layer_at(Site {
                    prefix: "mtp.decoder".to_string(),
                    ratio: None,
                    hash: false,
                    experts: DRAFT_EXPERTS,
                    split: true,
                    gate: Dtype::Mxfp4,
                    up: Dtype::Mxfp4,
                    down: Dtype::Mxfp4,
                    weights: Dtype::Bf16,
                    dense: Dtype::Bf16,
                    kv: "kv.mtp".to_string(),
                    pool: "pool.mtp".to_string(),
                    index: "index.mtp".to_string(),
                }),
                hc_head: HcHead {
                    base: Weight::sym("mtp.hc_head.base", [streams], Dtype::F32),
                    dynamic: Weight::sym("mtp.hc_head.fn", [streams, hc_fan], Dtype::F32),
                    scale: Weight::sym("mtp.hc_head.scale", [1], Dtype::F32),
                },
                norm: Weight::sym("mtp.norm", [hidden], Dtype::Bf16),
                norm_eps: d.norm_eps,
                depth: DRAFT_DEPTH,
            }
        });

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            act,
            heads,
            head_dim: d.head_dim,
            window: d.window,
            adapters: ADAPTERS,
            kv,
            hyper: Hyper {
                streams: d.streams,
                norm_eps: d.norm_eps,
                gate_eps: d.gate_eps,
                alpha: d.alpha,
                sinkhorn: d.sinkhorn,
                single_pass: false,
            },
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: Some(Weight::sym("lm_head", [d.vocab as u64, hidden], weights)),
            hc_head: Some(HcHead {
                base: Weight::sym("hc_head.base", [streams], Dtype::F32),
                dynamic: Weight::sym("hc_head.fn", [streams, hc_fan], Dtype::F32),
                scale: Weight::sym("hc_head.scale", [1], Dtype::F32),
            }),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: idx_norm_eps,
            mtp,
            token_map: None,
        }
    }

    fn new_flash41(
        weights: Dtype,
        routed: Routed,
        act: Dtype,
        kv: Dtype,
        tp: u32,
        d: Dims41,
    ) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let plan = d.plan;
        assert_eq!(
            plan.ratios.len(),
            plan.layers as usize,
            "the plan states one compression ratio per layer"
        );

        let dense = crate::dense(weights);

        let heads = d.heads / tp;
        let moe_inter = d.moe_inter / tp;
        let shared_inter = d.shared_inter / tp;

        let hidden = d.hidden as u64;
        let streams = d.streams as u64;
        let hc_base = 2 * streams + streams * streams;
        let hc_fan = streams * hidden;
        let q_w = heads as u64 * d.head_dim as u64;
        let q_lora = d.q_lora as u64;
        let kv_latent = d.kv_latent as u64;
        let o_lora = d.o_lora as u64;
        let o_out = d.o_groups as u64 * o_lora;
        // wo_a is block-diagonal over the output groups: each group projects
        // its own heads' slice of the attention output.
        let o_in = q_w / d.o_groups as u64;
        let idx_w = d.index_heads as u64 * d.index_head_dim as u64;

        let engram_rows = |layer_index: usize| -> u64 {
            let (_, primes, _) = engram_hash_constants(&d.engram, plan.engram_layers, layer_index);
            primes.iter().sum()
        };

        let layers = (0..plan.layers)
            .map(|l| {
                let prefix = format!("layer.{l}");
                let n = |s: &str| format!("{prefix}.{s}");
                let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], dense);
                let mix = |s: &str| Mix {
                    scale: Weight::sym(n(&format!("{s}_scale")), [3], Dtype::F32),
                    base: Weight::sym(n(&format!("{s}_base")), [hc_base], Dtype::F32),
                    dynamic: Some(Weight::sym(
                        n(&format!("{s}_fn")),
                        [hc_base, hc_fan],
                        Dtype::F32,
                    )),
                };
                let (lora_a, lora_b) = crate::adapter::banks(&prefix, ADAPTERS, hidden, dense);

                let ratio = plan.ratios[l as usize];
                let csa = csa_of(&plan, l);
                // The most recent KV source at or before this layer.
                let kv_of = plan.kv_sources.iter().copied().filter(|s| *s <= l).max();
                let owner = kv_of == Some(l);
                let pool = (ratio > 0).then(|| {
                    let kv_of = kv_of.unwrap_or_else(|| {
                        panic!("layer {l} compresses at ratio {ratio} but no KV source precedes it")
                    });
                    Pool {
                        ratio,
                        entries: format!("pool.{kv_of}"),
                        compressor: owner.then(|| Compressor {
                            wkv: Weight::sym(n("compressor.wkv"), [kv_latent, hidden], dense),
                            wgate: (ratio > 1).then(|| {
                                Weight::sym(n("compressor.wgate"), [kv_latent, hidden], dense)
                            }),
                            ape: None,
                            norm: norm("compressor.norm", kv_latent),
                            norm_eps: d.norm_eps,
                        }),
                        owner,
                    }
                });
                let indexer = matches!(csa, Csa::Full | Csa::Reindex).then(|| Indexer {
                    heads: d.index_heads,
                    head_dim: d.index_head_dim,
                    top_k: d.index_top_k,
                    rope_dim: d.rope_dim,
                    theta: d.compress_theta,
                    yarn: Some(d.yarn),
                    window: d.window,
                    wq_b: Weight::sym(n("indexer.wq_b"), [idx_w, q_lora], weights),
                    weights_proj: Weight::sym(
                        n("indexer.weights_proj"),
                        [d.index_heads as u64, hidden],
                        dense,
                    ),
                    compressor: None,
                    wk: owner.then(|| {
                        Weight::sym(n("indexer.wk"), [d.index_head_dim as u64, kv_latent], dense)
                    }),
                    k_norm: owner.then(|| norm("indexer.k_norm", d.index_head_dim as u64)),
                    keys: format!("index.{}", kv_of.unwrap_or(l)),
                    owns_keys: owner,
                });
                let selection = match csa {
                    Csa::Swa => Selection::None,
                    Csa::Full | Csa::Reindex => Selection::Own,
                    Csa::Reuse => Selection::Shared,
                };

                let engram = plan
                    .engram_layers
                    .iter()
                    .position(|e| *e == l)
                    .map(|which| {
                        let e = &d.engram;
                        let (mults, primes, offsets) =
                            engram_hash_constants(e, plan.engram_layers, which);
                        let rows = engram_rows(which);
                        let cols = u64::from(e.ngram - 1) * u64::from(e.heads);
                        Engram {
                            table: Weight::sym(
                                n("engram.embed"),
                                [rows, u64::from(e.head_dim)],
                                dense,
                            ),
                            wkv: Weight::sym(
                                n("engram.wkv"),
                                [(streams + 1) * hidden, cols * u64::from(e.head_dim)],
                                weights,
                            )
                            .packed([streams * hidden, hidden]),
                            q_weight: Weight::sym(n("engram.q_weight"), [streams, hidden], dense),
                            k_weight: Weight::sym(n("engram.k_weight"), [streams, hidden], dense),
                            mults,
                            primes,
                            offsets,
                            ngram: e.ngram,
                            heads_per_ngram: e.heads,
                            head_dim: e.head_dim,
                            pad: e.pad,
                            eps: d.norm_eps,
                            ids_state: format!("engram.{l}"),
                        }
                    });

                Layer {
                    attn_mix: mix("attn_mix"),
                    attn_norm: Some(norm("attn_norm", hidden)),
                    mlp_norm: Some(norm("ffn_norm", hidden)),
                    attn: Attn {
                        rope_dim: d.rope_dim,
                        theta: if ratio > 0 { d.compress_theta } else { d.theta },
                        yarn: (ratio > 0).then_some(d.yarn),
                        sm_scale: (d.head_dim as f32).sqrt().recip(),
                        q_down: Weight::sym(n("q_down"), [q_lora, hidden], weights),
                        q_norm: norm("q_norm", q_lora),
                        q_norm_eps: d.norm_eps,
                        q_up: Weight::sym(n("q_up"), [q_w, q_lora], weights).columns(),
                        kv_down: Weight::sym(n("kv_down"), [kv_latent, hidden], weights),
                        kv_norm: norm("kv_norm", kv_latent),
                        kv_norm_eps: d.norm_eps,
                        o_down: Weight::sym(n("o_down"), [o_out, o_in], weights),
                        o_up: Weight::sym(n("o_up"), [hidden, o_out], weights).rows(),
                        o_groups: d.o_groups,
                        sink: Weight::sym(n("attn_sink"), [heads as u64], dense).columns(),
                        kv: format!("kv.{l}"),
                        pool,
                        indexer,
                        selection,
                    },
                    mlp_mix: mix("mlp_mix"),
                    mlp: Mlp::MoeFlash {
                        router: Weight::sym(n("gate"), [d.experts as u64, hidden], dense),
                        gate: Gate::Bias {
                            bias: Weight::sym(n("gate.bias"), [d.experts as u64], Dtype::F32),
                        },
                        gate_up: {
                            let half = |what: &str, dtype: Dtype| {
                                Weight::sym(
                                    n(what),
                                    [d.experts as u64, moe_inter as u64, hidden],
                                    dtype,
                                )
                                .bank([moe_inter as u64])
                            };
                            GateUp::Split {
                                gate: half("experts_gate", routed.gate_of(l)),
                                up: half("experts_up", routed.up),
                            }
                        },
                        down: Weight::sym(
                            n("experts_down"),
                            [d.experts as u64, hidden, moe_inter as u64],
                            routed.down,
                        )
                        .rows(),
                        shared_gate_up: Weight::sym(
                            n("shared_gate_up"),
                            [2 * shared_inter as u64, hidden],
                            weights,
                        )
                        .packed([shared_inter as u64, shared_inter as u64]),
                        shared_down: Weight::sym(
                            n("shared_down"),
                            [hidden, shared_inter as u64],
                            weights,
                        )
                        .rows(),
                        experts: d.experts,
                        top_k: d.top_k,
                        inter: moe_inter,
                        shared_inter,
                        limit: d.swiglu_limit,
                        renorm: d.renorm,
                        scaling: d.scaling,
                    },
                    lora_a,
                    lora_b,
                    engram,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            act,
            heads,
            head_dim: d.head_dim,
            window: d.window,
            adapters: ADAPTERS,
            kv,
            hyper: Hyper {
                streams: d.streams,
                norm_eps: d.norm_eps,
                gate_eps: d.gate_eps,
                alpha: d.alpha,
                sinkhorn: d.sinkhorn,
                single_pass: true,
            },
            embed: Weight::sym("embed", [d.vocab as u64, hidden], dense),
            head: Some(Weight::sym("lm_head", [d.vocab as u64, hidden], dense)),
            hc_head: None,
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            mtp: None,
            token_map: (!plan.engram_layers.is_empty())
                .then(|| Weight::sym("engram.token_map", [d.vocab as u64], Dtype::I32)),
        }
    }
}

/// The CSA2 mode the plan assigns a layer.
#[must_use]
pub fn csa_of(plan: &Plan41, layer: u32) -> Csa {
    if plan.ratios[layer as usize] == 0 {
        Csa::Swa
    } else if plan.kv_sources.contains(&layer) {
        Csa::Full
    } else if plan.index_sources.contains(&layer) {
        Csa::Reindex
    } else {
        Csa::Reuse
    }
}

/// Engram's hash geometry for the `which`-th Engram module of a plan, as the
/// reference derives it: every (module, n-gram order, head) owns a distinct
/// prime-sized bucket range — the primes are drawn in module order, each order
/// restarting the search at `base_vocab - 1` and never reusing a prime — and
/// the multipliers are the module's own `numpy.random.default_rng(10007 * layer)`
/// draws, made odd.
#[must_use]
fn engram_hash_constants(
    e: &EngramDims,
    engram_layers: &[u32],
    which: usize,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let mut seen: Vec<u64> = Vec::new();
    let mut primes = Vec::new();
    let mut offsets = Vec::new();
    for module in 0..engram_layers.len() {
        let mut total = 0u64;
        for _ in 0..(e.ngram - 1) {
            let mut current = e.base_vocab - 1;
            for _ in 0..e.heads {
                current = next_prime_after(current, &seen);
                seen.push(current);
                if module == which {
                    primes.push(current);
                    offsets.push(total);
                    total += current;
                }
            }
        }
    }
    let layer = u64::from(engram_layers[which]);
    let bound = ((i64::MAX as u64) / e.compressed_vocab.max(1) / 2).max(1);
    let mut rng = crate::numpy::Generator::seeded(u128::from(10_007 * layer));
    let mults = rng
        .integers(bound, e.ngram as usize)
        .into_iter()
        .map(|v| v * 2 + 1)
        .collect();
    (mults, primes, offsets)
}

fn next_prime_after(start: u64, seen: &[u64]) -> u64 {
    let mut candidate = start + 1;
    while !is_prime(candidate) || seen.contains(&candidate) {
        candidate += 1;
    }
    candidate
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

pub const DRAFT_DEPTH: u32 = 1;
const DRAFT_EXPERTS: u32 = 256;

impl Model {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_release_plan_assigns_the_published_modes() {
        let modes: Vec<Csa> = (0..40).map(|l| csa_of(&V41_PLAN, l)).collect();
        assert_eq!(&modes[..2], &[Csa::Swa, Csa::Swa]);
        for group in 0..3 {
            let first = 2 + 6 * group;
            assert_eq!(modes[first], Csa::Full);
            assert!(modes[first + 1..first + 6].iter().all(|m| *m == Csa::Reuse));
        }
        assert_eq!(modes[20], Csa::Full);
        assert!(modes[21..24].iter().all(|m| *m == Csa::Reuse));
        for group in 0..4 {
            let first = 24 + 4 * group;
            assert_eq!(modes[first], Csa::Reindex);
            assert!(modes[first + 1..first + 4].iter().all(|m| *m == Csa::Reuse));
        }
    }

    #[test]
    fn the_engram_table_is_as_tall_as_the_checkpoint_states() {
        let d = Model::dims41(V41_PLAN);
        let rows = |which| -> u64 {
            let (_, primes, _) = engram_hash_constants(&d.engram, V41_PLAN.engram_layers, which);
            primes.iter().sum()
        };
        // `engram_num_embeddings: [384006168, 384016682]`
        assert_eq!(rows(0), 384_006_168);
        assert_eq!(rows(1), 384_016_682);
    }

    #[test]
    fn a_reuse_layer_reads_its_sources_caches() {
        let m = Model::flash41(
            Dtype::Bf16,
            Routed::split(Dtype::Mxfp4),
            Dtype::Bf16,
            Dtype::Bf16,
            1,
        );
        let at = |l: usize| &m.layers[l].attn;
        assert_eq!(at(3).pool.as_ref().unwrap().entries, "pool.2");
        assert!(!at(3).pool.as_ref().unwrap().owner);
        assert_eq!(at(3).selection, Selection::Shared);
        assert_eq!(at(24).pool.as_ref().unwrap().entries, "pool.20");
        let ix = at(24).indexer.as_ref().unwrap();
        assert_eq!(ix.keys, "index.20");
        assert!(!ix.owns_keys && ix.wk.is_none());
        let owner = at(20).indexer.as_ref().unwrap();
        assert!(owner.owns_keys && owner.wk.is_some());
        assert!(
            at(20)
                .pool
                .as_ref()
                .unwrap()
                .compressor
                .as_ref()
                .unwrap()
                .wgate
                .is_none()
        );
        assert!(
            at(2)
                .pool
                .as_ref()
                .unwrap()
                .compressor
                .as_ref()
                .unwrap()
                .wgate
                .is_some()
        );
        assert!(m.layers[1].engram.is_some() && m.layers[14].engram.is_some());
        assert!(m.layers[0].engram.is_none());
    }
}
