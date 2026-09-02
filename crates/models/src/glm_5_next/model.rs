use model_dsl::{Dtype, Weight};

/// GLM-5.3-Flash (`glm5_next`): a hyper-connected tower whose mixers alternate
/// Kimi-Delta linear attention with nope-only MLA behind a sparse indexer.
pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// The element the pooled index entries are gathered as.
    pub act: Dtype,

    /// The MLA reading every `deepseek_sparse_attention` layer shares.
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
    /// The draft head, or `None` for a row without one.
    pub mtp: Option<Mtp>,
}

pub use crate::adapter::Adapters;

/// **THE DRAFT HEAD** — GLM-5.3-Flash's one `nextn` layer (`layers.45`, the
/// DeepSeek-V3 `MTP` shape): the next token's embedding and the trunk's
/// collapsed residual each normed, fused by `eh_proj` (stored as one
/// `[hidden, 2·hidden]` plane, read as its two column halves), one DSA+MoE
/// block over the fused row, the head's own norm (`shared_head.norm`) and
/// the base `lm_head`. No hyper connections: the head reads the collapsed
/// residual, not the streams.
///
/// ```text
/// x   = e_proj(enorm(embed(tok))) + h_proj(hnorm(y))
/// x   = block(x)                                   (DSA + MoE, pre-norm)
/// out = lm_head(norm(x))
/// ```
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
    /// How many tokens past a readout row the head drafts (the checkpoint
    /// ships one prediction layer, run at depth 1). The `mtp.drafts` seam is
    /// `[rows, depth]` and the shell advertises `depth` as `mtp_depth`.
    pub depth: u32,
}

/// The manifold hyper-connection tower's own constants (`hc_mult`, `hc_eps`,
/// `hc_sinkhorn_iters`).
pub struct Hyper {
    pub streams: u32,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,
}

/// One sublayer's hyper mix: `scale [3]`, `base [2M + M²]`, `fn [2M + M², M·hidden]`.
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
    /// The mixer sublayer's adapter bank, applied to the replicated output
    /// after `all_reduce` and before the hyper fold.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

#[allow(clippy::large_enum_variant)]
pub enum Mixer {
    Mla(Mla),
    Kda(Kda),
}

/// `mla_use_nope`: `qk_rope_head_dim` is zero, so no plane of this mixer ropes.
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

/// The sparse indexer, keyed by `index_kpool`-pooled compressed rows
/// (`index_kpool_compress`) rather than by token.
pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    /// `index_kpool`: how many tokens one cached key entry pools.
    pub kpool: u32,
    pub rope_dim: u32,
    pub theta: f32,
    pub wq_b: Weight,
    pub wk: Weight,
    pub weights_proj: Weight,
    pub k_norm: Weight,
    pub k_norm_bias: Weight,
    pub k_norm_eps: f32,
    /// `index_kpool_compress_ape`: the intra-block position plane the gather adds.
    pub kpool_ape: Weight,
    /// `index_kpool_compress_gate`: the gate whose logits weight the pooled rows.
    pub kpool_gate: Weight,
    pub keys: String,
}

pub struct Kda {
    /// `gate_lower_bound`: the decay is `floor * sigmoid(exp(A_log) * g)` when negative.
    pub gate_floor: f32,
    pub heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    pub norm_eps: f32,
    pub qkv: Weight,
    pub conv: Weight,
    pub f_a: Weight,
    pub f_b: Weight,
    /// The output gate's low rank pair (`g_a_proj`, `g_b_proj`).
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
        /// `e_score_correction_bias`, the `noaux_tc` ranking correction.
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
    /// A `deepseek_sparse_attention` layer is one whose index is this many
    /// short of a whole block (`layer_types`: every fourth, from layer 3).
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
    /// `Vontra/GLM-5.3-Flash-MLX-2bit-MTP`, text only: 45 layers, hidden 4096,
    /// 288 routed experts top-8, the KDA/DSA cadence, mHC over four streams.
    pub fn flash(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, experts, None, kv, tp, Model::flash_dims())
    }

    /// [`flash`](Model::flash) with the draft head over it: the checkpoint's
    /// `layers.45` block, its routed experts in `head_experts` (Q4 in
    /// Vontra's conversion, where the trunk's are Q2).
    pub fn flash_mtp(w: Dtype, experts: Dtype, head_experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, experts, Some(head_experts), kv, tp, Model::flash_dims())
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
        kv: Dtype,
        tp: u32,
        d: Dims,
    ) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        // Norms, the router, the conv bank, the pooled position plane and the
        // hyper planes ship unquantized; they are stated in the compute dtype.
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

        // **ONE MLA, STATED FOR A SITE**: the trunk's eleven DSA mixers and the
        // draft head's one are the same block; what differs per site is its
        // name prefix and its cache rows.
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
                q_b_proj: Weight::sym(n("q_b_proj"), [q_b_width, q_lora], weights)
                    .columns(),
                kv_a_proj: Weight::sym(n("kv_a_proj"), [kv_a_width, hidden], weights),
                kv_a_norm: norm("kv_a_norm", kv_lora),
                kv_a_norm_eps: d.norm_eps,
                kv_b_proj: Weight::sym(n("kv_b_proj"), [kv_b_width, kv_lora], weights)
                    .columns(),
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
        // **ONE ROUTED MLP, STATED FOR A SITE**: the trunk's and the head's
        // differ in prefix and in the experts' dtype.
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
                down: Weight::sym(
                    n("experts_down"),
                    [m.experts as u64, hidden, iw],
                    experts,
                )
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
                        conv: Weight::sym(n("kda_conv"), [3 * kda_width, k.conv_kernel as u64], dense)
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
                        // `rmsnorm_gated_by` scales by an f32 weight.
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

        // **THE DRAFT HEAD** (`layers.45` of the checkpoint): its planes come
        // in under `mtp.`; `eh_proj` is read as two column halves.
        let mtp = draft.map(|head_experts| Mtp {
            enorm: Weight::sym("mtp.enorm", [hidden], dense),
            hnorm: Weight::sym("mtp.hnorm", [hidden], dense),
            e_proj: Weight::sym("mtp.e_proj", [hidden, hidden], Dtype::Bf16),
            h_proj: Weight::sym("mtp.h_proj", [hidden, hidden], Dtype::Bf16),
            mixer_norm: Weight::sym("mtp.mixer_norm", [hidden], dense),
            mixer_norm_eps: d.norm_eps,
            attn: mla_at("mtp".to_string(), "kv.mtp".to_string(), "index.mtp".to_string()),
            mlp_norm: Weight::sym("mtp.mlp_norm", [hidden], dense),
            mlp_norm_eps: d.norm_eps,
            mlp: routed_at("mtp".to_string(), head_experts),
            norm: Weight::sym("mtp.norm", [hidden], dense),
            norm_eps: d.norm_eps,
            depth: DRAFT_DEPTH,
        });

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
            embed: Weight::sym("embed", [d.vocab as u64, hidden], dense),
            head: Weight::sym("lm_head", [d.vocab as u64, hidden], dense),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            mtp,
        }
    }
}

/// Deployment ceiling for adapter slots/rank; change and re-trace to grow it.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

/// Tokens the draft head drafts past a readout row: the checkpoint's one
/// prediction layer, run as trained.
const DRAFT_DEPTH: u32 = 1;
