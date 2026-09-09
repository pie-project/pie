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
}

pub struct Pool {
    pub ratio: u32,
    pub entries: String,
    pub compressor: Option<Compressor>,
}

pub struct Compressor {
    pub wkv: Weight,
    pub wgate: Weight,
    pub ape: Weight,
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
    pub compressor: Compressor,
    pub keys: String,
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
                        }),
                        indexer: None,
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
            },
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: None,
            hc_head: None,
            layers,
            final_norm: Weight::sym("final_norm", [hidden], weights),
            final_norm_eps: d.norm_eps,
            mtp: None,
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
            wgate: Weight::sym(format!("{prefix}.wgate"), [entries, hidden], weights),
            ape: Weight::sym(format!("{prefix}.ape"), [ratio as u64, entries], Dtype::F32),
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
                compressor: compressor(
                    n("indexer.compressor"),
                    ratio.unwrap_or(4),
                    2 * d.index_head_dim as u64,
                    d.index_head_dim as u64,
                ),
                keys: site.index.clone(),
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
        }
    }
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

pub const DRAFT_DEPTH: u32 = 1;
const DRAFT_EXPERTS: u32 = 256;

impl Model {}
