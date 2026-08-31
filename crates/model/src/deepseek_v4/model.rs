use checkpoint::contract::ModelContract;
use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub act: Dtype,

    /// The one attention reading this text is carved for. Every layer reads
    /// the same schedule, so the numbers that schedule states — query heads,
    /// kv heads, head width, window — are facts about the model, not about a
    /// layer. The latent plane is shared across heads, so the kv heads a
    /// reader restates are these query `heads`.
    pub heads: u32,
    pub head_dim: u32,
    pub window: u32,

    /// The adapter banks this family seats (palo design §8). Per layer, and
    /// the same two numbers at every one of them: the correction is a
    /// per-lane axis, not a per-layer one.
    pub adapters: Adapters,

    pub kv: Dtype,
    pub hyper: Hyper,

    pub embed: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
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
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub attn_mix: Mix,
    pub attn: Attn,
    pub mlp_mix: Mix,
    pub mlp: Mlp,
    /// This layer's adapter bank, `[slots, rank, hidden]` and
    /// `[slots, hidden, rank]` — the down and up planes of one correction site
    /// (palo design §8, campaign A-6).
    ///
    /// **THE SITE IS THE ATTENTION SUBLAYER, AND IT IS THE SITE BECAUSE OF THE
    /// COLLECTIVE.** Both ends are REPLICATED values: the input is the gated
    /// stream `hc_gates` hands the mixer, and the output is what `o_up`
    /// answers — which is already past `all_reduce`, because this text reduces
    /// between its two output projections (`o_down` is rows-cut, `o_up` is
    /// replicated). A correction stated before the reduce would read a rows-cut
    /// partial product, and every rank would contribute the whole `ΔW·x` for
    /// the sum to carry `tp` times.
    ///
    /// AND BEFORE `hc_fold`, not after: the fold writes the correction into
    /// the hyper-connection streams the way it writes the mixer's own output,
    /// so a corrected site stays one site instead of becoming a second
    /// contribution with its own mixing weights.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub struct Attn {
    pub rope_dim: u32,
    pub theta: f32,
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
    pub sink: Weight,
    pub kv: String,
    pub pool: Option<Pool>,
}

pub struct Pool {
    pub ratio: u32,
    pub entries: String,
}

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
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    pool: &'static [Option<u32>; 6],
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
                };
                let (lora_a, lora_b) = crate::adapter::banks(
                    &format!("layer.{l}"),
                    ADAPTERS,
                    hidden,
                    crate::dense(weights),
                );
                Layer {
                    attn_mix: mix("attn_mix"),
                    attn: Attn {
                        rope_dim: d.rope_dim,
                        theta: d.theta,
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
                        sink: Weight::sym(n("attn_sink"), [heads as u64], weights).columns(),
                        kv: format!("kv.{l}"),
                        pool: d.pool[l as usize].map(|ratio| Pool {
                            ratio,
                            entries: format!("pool.{l}"),
                        }),
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
            layers,
            final_norm: Weight::sym("final_norm", [hidden], weights),
            final_norm_eps: d.norm_eps,
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
/// Eight slots of rank sixteen costs the `base` row 1 MiB a layer — two planes
/// of `8 x 16 x 2048` in the compute element — and 6 MiB over its six.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {
    pub fn load(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        let mut b = checkpoint_dsl::Builder::new(src, self.tp);
        let mut stated = |w: &Weight| b.read_own(w);
        stated(&self.embed)?;
        stated(&self.final_norm)?;

        for layer in &self.layers {
            stated(&layer.attn_mix.scale)?;
            stated(&layer.attn_mix.base)?;
            stated(&layer.mlp_mix.scale)?;
            stated(&layer.mlp_mix.base)?;

            let at = &layer.attn;
            stated(&at.q_down)?;
            stated(&at.q_norm)?;
            stated(&at.q_up)?;
            stated(&at.kv_down)?;
            stated(&at.kv_norm)?;
            stated(&at.o_down)?;
            stated(&at.o_up)?;
            stated(&at.sink)?;

            match &layer.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    stated(gate_up)?;
                    stated(down)?;
                }
                Mlp::Routed {
                    router,
                    bias,
                    gate_up,
                    down,
                    ..
                } => {
                    stated(router)?;
                    stated(bias)?;
                    stated(gate_up)?;
                    stated(down)?;
                }
            }
        }

        Ok(b.build())
    }
}
