use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub act: Dtype,

    pub heads: u32,
    pub kv_lora_rank: u32,

    pub adapters: Adapters,

    pub kv: Dtype,
    pub hyper: Hyper,

    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
    pub mtp: Option<Mtp>,
    pub tower: Option<Tower>,
}

pub struct Tower {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub merge: u32,
    pub patch_width: u32,
    pub inter: u32,
    pub merger_inter: u32,
    pub limit: f32,
    pub theta: f32,
    pub norm_eps: f32,
    pub sm_scale: f32,
    pub patch_embed: Weight,
    pub patch_embed_bias: Weight,
    pub blocks: Vec<TowerBlock>,
    pub post_norm: Weight,
    pub downsample: Weight,
    pub downsample_bias: Weight,
    pub merger: Merger,
}

pub struct TowerBlock {
    pub norm1: Weight,
    pub qkv: Weight,
    pub qkv_bias: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub proj: Weight,
    pub proj_bias: Weight,
    pub norm2: Weight,
    pub gate_up: Weight,
    pub gate_up_bias: Weight,
    pub down: Weight,
    pub down_bias: Weight,
}

pub struct Merger {
    pub proj: Weight,
    pub norm: Weight,
    pub norm_bias: Weight,
    pub gate_up: Weight,
    pub down: Weight,
}

pub use crate::adapter::Adapters;

pub struct Mtp {
    pub enorm: Weight,
    pub hnorm: Weight,
    pub e_proj: Weight,
    pub h_proj: Weight,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub attn: Mla,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    pub norm: Weight,
    pub norm_eps: f32,
    pub depth: u32,
}

pub struct Hyper {
    pub streams: u32,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,
}

pub struct Mix {
    pub scale: Weight,
    pub base: Weight,
    pub dynamic: Weight,
}

pub struct Layer {
    pub attn_mix: Mix,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub mixer: Mixer,
    pub mlp_mix: Mix,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    pub lora_a: Weight,
    pub lora_b: Weight,
}

#[allow(clippy::large_enum_variant)]
pub enum Mixer {
    Mla(Mla),
    Kda(Kda),
}

pub struct Mla {
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub sm_scale: f32,
    pub q_a_proj: Weight,
    pub q_a_norm: Weight,
    pub q_a_norm_eps: f32,
    pub q_b_proj: Weight,
    pub kv_a_proj: Weight,
    pub kv_a_norm: Weight,
    pub kv_a_norm_eps: f32,
    pub kv_b_proj: Weight,
    pub o_proj: Weight,
    pub indexer: Indexer,
    pub kv: String,
}

pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub kpool: u32,
    pub rope_dim: u32,
    pub theta: f32,
    pub wq_b: Weight,
    pub wk: Weight,
    pub weights_proj: Weight,
    pub k_norm: Weight,
    pub k_norm_bias: Weight,
    pub k_norm_eps: f32,
    pub kpool_ape: Weight,
    pub kpool_gate: Weight,
    pub keys: String,
}

pub struct Kda {
    pub gate_floor: f32,
    pub heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    pub norm_eps: f32,
    pub qkv: Weight,
    pub conv: Weight,
    pub f_a: Weight,
    pub f_b: Weight,
    pub g_a: Weight,
    pub g_b: Weight,
    pub b: Weight,
    pub dt_bias: Weight,
    pub a_log: Weight,
    pub o_norm: Weight,
    pub o_norm_eps: f32,
    pub o_proj: Weight,
    pub conv_state: String,
    pub delta_state: String,
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
        shared: Option<Shared>,
        experts: u32,
        top_k: u32,
        inter: u32,
        limit: f32,
        renorm: bool,
        scaling: f32,
    },
}

pub struct Shared {
    pub gate_up: Weight,
    pub down: Weight,
    pub inter: u32,
}

struct MlaDims {
    heads: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
}

struct KdaDims {
    heads: u32,
    head_dim: u32,
    f_rank: u32,
    conv_kernel: u32,
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
    renorm: bool,
    scaling: f32,
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    full_attn_every: u32,
    mla: MlaDims,
    kda: KdaDims,
    index_heads: u32,
    index_head_dim: u32,
    index_top_k: u32,
    index_kpool: u32,
    streams: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    dense_inter: u32,
    moe: MoeDims,
    swiglu_limit: f32,
    theta: f32,
    vocab: u32,
    norm_eps: f32,
}

impl Model {
    pub fn flash(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, experts, None, false, kv, tp, Model::flash_dims())
    }

    pub fn flash_vision(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, experts, None, true, kv, tp, Model::flash_dims())
    }

    pub fn flash_mtp_vision(
        w: Dtype,
        experts: Dtype,
        head_experts: Dtype,
        kv: Dtype,
        tp: u32,
    ) -> Model {
        Model::new(
            w,
            experts,
            Some(head_experts),
            true,
            kv,
            tp,
            Model::flash_dims(),
        )
    }

    /// The first `layers` layers of GLM-5.3-Flash at full width with the first
    /// `experts` routed experts (top-k unchanged): a performance delegate cut
    /// by `scripts/bench/shrink_checkpoint.py` (`glm5_next_mlx`). Eight layers
    /// hold every layer kind: three dense KDA, DSA + MoE, three KDA + MoE, DSA.
    pub fn flash_mini(
        layers: u32,
        experts: u32,
        w: Dtype,
        bank: Dtype,
        kv: Dtype,
        tp: u32,
    ) -> Model {
        let mut d = Model::flash_dims();
        d.layers = layers;
        d.moe.experts = experts;
        d.moe.top_k = d.moe.top_k.min(experts);
        Model::new(w, bank, None, false, kv, tp, d)
    }

    pub fn flash_mtp(w: Dtype, experts: Dtype, head_experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            experts,
            Some(head_experts),
            false,
            kv,
            tp,
            Model::flash_dims(),
        )
    }

    fn flash_dims() -> Dims {
        Dims {
            hidden: 4096,
            layers: 45,
            dense_layers: 3,
            full_attn_every: 4,
            mla: MlaDims {
                heads: 64,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 256,
                qk_rope_head_dim: 0,
                v_head_dim: 256,
            },
            kda: KdaDims {
                heads: 64,
                head_dim: 128,
                f_rank: 128,
                conv_kernel: 4,
            },
            index_heads: 32,
            index_head_dim: 128,
            index_top_k: 2048,
            index_kpool: 4,
            streams: 4,
            gate_eps: 1e-6,
            alpha: 2.0,
            sinkhorn: 20,
            dense_inter: 12_288,
            moe: MoeDims {
                experts: 288,
                top_k: 8,
                inter: 2048,
                shared_inter: 2048,
                renorm: true,
                scaling: 2.5,
            },
            swiglu_limit: 10.0,
            theta: 10_000.0,
            vocab: 154_880,
            norm_eps: 1e-5,
        }
    }

    fn new(
        weights: Dtype,
        experts: Dtype,
        draft: Option<Dtype>,
        vision: bool,
        kv: Dtype,
        tp: u32,
        d: Dims,
    ) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let dense = crate::dense(weights);

        let mla_heads = d.mla.heads / tp;
        let kda_heads = d.kda.heads / tp;
        let dense_inter = d.dense_inter / tp;
        let moe_inter = d.moe.inter / tp;
        let shared_inter = d.moe.shared_inter / tp;

        let hidden = d.hidden as u64;
        let streams = d.streams as u64;
        let hc_base = 2 * streams + streams * streams;
        let hc_fan = streams * hidden;

        let a = &d.mla;
        let k = &d.kda;
        let q_lora = a.q_lora_rank as u64;
        let kv_lora = a.kv_lora_rank as u64;
        let qk_head_dim = (a.qk_nope_head_dim + a.qk_rope_head_dim) as u64;
        let q_b_width = mla_heads as u64 * qk_head_dim;
        let kv_a_width = kv_lora + a.qk_rope_head_dim as u64;
        let kv_b_width = mla_heads as u64 * (a.qk_nope_head_dim + a.v_head_dim) as u64;
        let v_width = mla_heads as u64 * a.v_head_dim as u64;
        let kda_width = kda_heads as u64 * k.head_dim as u64;
        let index_width = d.index_heads as u64 * d.index_head_dim as u64;

        let dsa_at = |l: u32| d.full_attn_every > 0 && (l + 1).is_multiple_of(d.full_attn_every);

        let mla_at = |prefix: String, kv_row: String, index_row: String| -> Mla {
            let n = |s: &str| format!("{prefix}.{s}");
            let norm = |s: &str, width: u64| Weight::sym(n(s), [width], dense);
            Mla {
                qk_nope_head_dim: a.qk_nope_head_dim,
                qk_rope_head_dim: a.qk_rope_head_dim,
                v_head_dim: a.v_head_dim,
                sm_scale: (qk_head_dim as f32).sqrt().recip(),
                q_a_proj: Weight::sym(n("q_a_proj"), [q_lora, hidden], weights),
                q_a_norm: norm("q_a_norm", q_lora),
                q_a_norm_eps: d.norm_eps,
                q_b_proj: Weight::sym(n("q_b_proj"), [q_b_width, q_lora], weights).columns(),
                kv_a_proj: Weight::sym(n("kv_a_proj"), [kv_a_width, hidden], weights),
                kv_a_norm: norm("kv_a_norm", kv_lora),
                kv_a_norm_eps: d.norm_eps,
                kv_b_proj: Weight::sym(n("kv_b_proj"), [kv_b_width, kv_lora], weights).columns(),
                o_proj: Weight::sym(n("o_proj"), [hidden, v_width], weights).rows(),
                indexer: Indexer {
                    heads: d.index_heads,
                    head_dim: d.index_head_dim,
                    top_k: d.index_top_k,
                    kpool: d.index_kpool,
                    rope_dim: a.qk_rope_head_dim,
                    theta: d.theta,
                    wq_b: Weight::sym(n("index_q_proj"), [index_width, q_lora], weights),
                    wk: Weight::sym(
                        n("index_k_proj"),
                        [d.index_head_dim as u64, hidden],
                        weights,
                    ),
                    weights_proj: Weight::sym(
                        n("index_weights"),
                        [d.index_heads as u64, hidden],
                        weights,
                    ),
                    k_norm: norm("index_k_norm", d.index_head_dim as u64),
                    k_norm_bias: Weight::sym(
                        n("index_k_norm_bias"),
                        [d.index_head_dim as u64],
                        dense,
                    ),
                    k_norm_eps: d.norm_eps,
                    kpool_ape: Weight::sym(
                        n("index_kpool_ape"),
                        [d.index_kpool as u64, d.index_head_dim as u64],
                        Dtype::F32,
                    ),
                    kpool_gate: Weight::sym(
                        n("index_kpool_gate"),
                        [d.index_head_dim as u64, hidden],
                        dense,
                    ),
                    keys: index_row,
                },
                kv: kv_row,
            }
        };
        let routed_at = |prefix: String, experts: Dtype| -> Mlp {
            let n = |s: &str| format!("{prefix}.{s}");
            let m = &d.moe;
            let iw = moe_inter as u64;
            let sw = shared_inter as u64;
            Mlp::Routed {
                router: Weight::sym(n("router"), [m.experts as u64, hidden], dense),
                bias: Weight::sym(n("router_bias"), [m.experts as u64], Dtype::F32),
                gate_up: Weight::sym(
                    n("experts_gate_up"),
                    [m.experts as u64, 2 * iw, hidden],
                    experts,
                )
                .bank([iw, iw]),
                down: Weight::sym(n("experts_down"), [m.experts as u64, hidden, iw], experts)
                    .rows(),
                shared: (shared_inter > 0).then(|| Shared {
                    gate_up: Weight::sym(n("shared_gate_up"), [2 * sw, hidden], weights)
                        .packed([sw, sw]),
                    down: Weight::sym(n("shared_down"), [hidden, sw], weights).rows(),
                    inter: shared_inter,
                }),
                experts: m.experts,
                top_k: m.top_k,
                inter: moe_inter,
                limit: d.swiglu_limit,
                renorm: m.renorm,
                scaling: m.scaling,
            }
        };
        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, width: u64| Weight::sym(n(s), [width], dense);
                let mix = |s: &str| Mix {
                    scale: Weight::sym(n(&format!("{s}_scale")), [3], Dtype::F32),
                    base: Weight::sym(n(&format!("{s}_base")), [hc_base], Dtype::F32),
                    dynamic: Weight::sym(n(&format!("{s}_fn")), [hc_base, hc_fan], Dtype::F32),
                };
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);

                let mixer = if dsa_at(l) {
                    Mixer::Mla(mla_at(
                        format!("layer.{l}"),
                        format!("kv.{l}"),
                        format!("index.{l}"),
                    ))
                } else {
                    Mixer::Kda(Kda {
                        heads: kda_heads,
                        head_dim: k.head_dim,
                        conv_kernel: k.conv_kernel,
                        norm_eps: d.norm_eps,
                        gate_floor: -5.0,
                        qkv: Weight::sym(n("kda_qkv"), [3 * kda_width, hidden], weights)
                            .packed([kda_width, kda_width, kda_width]),
                        conv: Weight::sym(
                            n("kda_conv"),
                            [3 * kda_width, k.conv_kernel as u64],
                            dense,
                        )
                        .packed([kda_width, kda_width, kda_width]),
                        f_a: Weight::sym(n("kda_f_a"), [k.f_rank as u64, hidden], weights),
                        f_b: Weight::sym(n("kda_f_b"), [kda_width, k.f_rank as u64], weights)
                            .columns(),
                        g_a: Weight::sym(n("kda_g_a"), [k.f_rank as u64, hidden], weights),
                        g_b: Weight::sym(n("kda_g_b"), [kda_width, k.f_rank as u64], weights)
                            .columns(),
                        b: Weight::sym(n("kda_b"), [kda_heads as u64, hidden], weights).columns(),
                        dt_bias: Weight::sym(
                            n("kda_dt_bias"),
                            [kda_heads as u64, k.head_dim as u64],
                            Dtype::F32,
                        )
                        .columns(),
                        a_log: Weight::sym(n("kda_a_log"), [kda_heads as u64], Dtype::F32)
                            .columns(),
                        o_norm: Weight::sym(n("kda_o_norm"), [k.head_dim as u64], Dtype::F32),
                        o_norm_eps: d.norm_eps,
                        o_proj: Weight::sym(n("kda_o_proj"), [hidden, kda_width], weights).rows(),
                        conv_state: format!("conv.{l}"),
                        delta_state: format!("delta.{l}"),
                    })
                };

                let mlp = if l < d.dense_layers {
                    let iw = dense_inter as u64;
                    Mlp::Dense {
                        gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], weights)
                            .packed([iw, iw]),
                        down: Weight::sym(n("down"), [hidden, iw], weights).rows(),
                        inter: dense_inter,
                        limit: d.swiglu_limit,
                    }
                } else {
                    routed_at(format!("layer.{l}"), experts)
                };

                Layer {
                    attn_mix: mix("attn_hc"),
                    mixer_norm: norm("mixer_norm", hidden),
                    mixer_norm_eps: d.norm_eps,
                    mixer,
                    mlp_mix: mix("ffn_hc"),
                    mlp_norm: norm("mlp_norm", hidden),
                    mlp_norm_eps: d.norm_eps,
                    mlp,
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        let mtp = draft.map(|head_experts| Mtp {
            enorm: Weight::sym("mtp.enorm", [hidden], dense),
            hnorm: Weight::sym("mtp.hnorm", [hidden], dense),
            e_proj: Weight::sym("mtp.e_proj", [hidden, hidden], Dtype::Bf16),
            h_proj: Weight::sym("mtp.h_proj", [hidden, hidden], Dtype::Bf16),
            mixer_norm: Weight::sym("mtp.mixer_norm", [hidden], dense),
            mixer_norm_eps: d.norm_eps,
            attn: mla_at(
                "mtp".to_string(),
                "kv.mtp".to_string(),
                "index.mtp".to_string(),
            ),
            mlp_norm: Weight::sym("mtp.mlp_norm", [hidden], dense),
            mlp_norm_eps: d.norm_eps,
            mlp: routed_at("mtp".to_string(), head_experts),
            norm: Weight::sym("mtp.norm", [hidden], dense),
            norm_eps: d.norm_eps,
            depth: DRAFT_DEPTH,
        });

        let tower = vision.then(|| Model::tower(&d));

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            act: dense,
            heads: mla_heads,
            kv_lora_rank: a.kv_lora_rank,
            adapters: ADAPTERS,
            kv,
            hyper: Hyper {
                streams: d.streams,
                norm_eps: d.norm_eps,
                gate_eps: d.gate_eps,
                alpha: d.alpha,
                sinkhorn: d.sinkhorn,
            },
            embed: Weight::sym("embed", [d.vocab as u64, hidden], Dtype::U4g64),
            head: Weight::sym("lm_head", [d.vocab as u64, hidden], Dtype::U4g64),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            mtp,
            tower,
        }
    }

    fn tower(d: &Dims) -> Tower {
        let (hidden, heads, depth, inter, merger_inter) =
            (1024u64, 16u32, 24u32, 4096u64, 10240u64);
        let (patch, temporal, merge) = (14u64, 2u64, 2u64);
        let out = d.hidden as u64;
        let head_dim = hidden / u64::from(heads);
        let bf = Dtype::Bf16;
        let v = |s: &str| format!("vision.{s}");
        let blocks = (0..depth)
            .map(|l| {
                let n = |s: &str| v(&format!("blocks.{l}.{s}"));
                TowerBlock {
                    norm1: Weight::sym(n("norm1"), [hidden], bf),
                    qkv: Weight::sym(n("qkv"), [3 * hidden, hidden], bf)
                        .packed([hidden, hidden, hidden]),
                    qkv_bias: Weight::sym(n("qkv_bias"), [3 * hidden], bf)
                        .packed([hidden, hidden, hidden]),
                    q_norm: Weight::sym(n("q_norm"), [head_dim], bf),
                    k_norm: Weight::sym(n("k_norm"), [head_dim], bf),
                    proj: Weight::sym(n("proj"), [hidden, hidden], bf),
                    proj_bias: Weight::sym(n("proj_bias"), [hidden], bf),
                    norm2: Weight::sym(n("norm2"), [hidden], bf),
                    gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], bf)
                        .packed([inter, inter]),
                    gate_up_bias: Weight::sym(n("gate_up_bias"), [2 * inter], bf)
                        .packed([inter, inter]),
                    down: Weight::sym(n("down"), [hidden, inter], bf),
                    down_bias: Weight::sym(n("down_bias"), [hidden], bf),
                }
            })
            .collect();
        Tower {
            hidden: hidden as u32,
            heads,
            head_dim: head_dim as u32,
            merge: merge as u32,
            patch_width: (3 * temporal * patch * patch) as u32,
            inter: inter as u32,
            merger_inter: merger_inter as u32,
            limit: d.swiglu_limit,
            theta: 10_000.0,
            norm_eps: d.norm_eps,
            sm_scale: (head_dim as f32).sqrt().recip(),
            patch_embed: Weight::sym(v("patch_embed"), [hidden, 3 * temporal * patch * patch], bf),
            patch_embed_bias: Weight::sym(v("patch_embed_bias"), [hidden], bf),
            blocks,
            post_norm: Weight::sym(v("post_norm"), [hidden], bf),
            downsample: Weight::sym(v("downsample"), [out, merge * merge * hidden], bf),
            downsample_bias: Weight::sym(v("downsample_bias"), [out], bf),
            merger: Merger {
                proj: Weight::sym(v("merger_proj"), [out, out], bf),
                norm: Weight::sym(v("merger_norm"), [out], bf),
                norm_bias: Weight::sym(v("merger_norm_bias"), [out], bf),
                gate_up: Weight::sym(v("merger_gate_up"), [2 * merger_inter, out], bf)
                    .packed([merger_inter, merger_inter]),
                down: Weight::sym(v("merger_down"), [out, merger_inter], bf),
            },
        }
    }
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

const DRAFT_DEPTH: u32 = 1;
