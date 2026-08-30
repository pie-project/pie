use checkpoint::contract::ModelContract;
use model_dsl::{Dtype, Weight};

use crate::contract::{ModelError, claim, elaborate};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// The MLA reading, stated once for the whole text: every MLA layer reads
    /// the same page-id space the same way, so one schedule per class is
    /// carved for these numbers at the top of `forward` and every layer reads
    /// it. The latent kernels size their output at `mla_heads × kv_lora_rank`,
    /// which is what `MlaDecode`/`MlaPrefill` restate. `Kda` keeps its own
    /// `heads`: a linear mixer carves no schedule.
    pub mla_heads: u32,
    pub kv_lora_rank: u32,

    /// The adapter banks this family seats (palo design §8). Per layer, and
    /// the same two numbers at every one of them: the correction is a
    /// per-lane axis, not a per-layer one.
    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub res_blend: Option<ResBlend>,
    pub mixer: Mixer,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    /// This layer's adapter bank, `[slots, rank, hidden]` and
    /// `[slots, hidden, rank]` — the down and up planes of one correction site
    /// (palo design §8, campaign A-6).
    ///
    /// **THE SITE IS THE MIXER SUBLAYER, AND IT IS THE SITE BECAUSE OF THE
    /// COLLECTIVE.** Both ends are REPLICATED values: the input is this
    /// layer's normed residual and the output is the mixer's result AFTER
    /// `all_reduce`. A correction stated one statement earlier — on
    /// `o_proj`'s own output, which is what a checkpoint's `o_proj` LoRA names
    /// — reads a rows-cut partial product and lands before the reduce, so
    /// every rank would contribute the whole `ΔW·x` and the sum would carry it
    /// `tp` times.
    ///
    /// **ONE SITE FOR BOTH MIXERS, AND THAT IS THE HONEST READING.** This
    /// family alternates MLA with a gated linear recurrence, and the two have
    /// no bank in common for a per-projection correction to name; what they DO
    /// share is this pair of replicated ends. So a bound adapter corrects the
    /// mixer sublayer of every layer, whichever mixer it is, which is the same
    /// sentence `Mixer` itself makes about the reading.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub struct ResBlend {
    pub norm: Weight,
    pub norm_eps: f32,
    pub proj: Weight,
}

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
    pub gate: Option<Weight>,
    pub o_proj: Weight,
    pub kv: String,
}

pub struct Kda {
    pub heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    pub norm_eps: f32,
    pub qkv: Weight,
    pub conv: Weight,
    pub f_a: Weight,
    pub f_b: Weight,
    pub b: Weight,
    pub dt_bias: Weight,
    pub a_log: Weight,
    pub gate: Weight,
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
        beta: f32,
        up_cap: Option<f32>,
    },
    Routed {
        router: Weight,
        gate_up: Weight,
        down: Weight,
        shared: Option<Shared>,
        experts: u32,
        top_k: u32,
        routed_scaling: f32,
        inter: u32,
        beta: f32,
        up_cap: Option<f32>,
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
    output_gate: bool,
}

struct KdaDims {
    heads: u32,
    head_dim: u32,
    f_rank: u32,
    conv_kernel: u32,
    norm_eps: f32,
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    routed_scaling: f32,
    inter: u32,
    shared_inter: u32,
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    full_attn_every: u32,
    res_block: u32,
    mla: MlaDims,
    kda: KdaDims,
    moe: MoeDims,
    dense_inter: u32,
    situ_beta: f32,
    situ_cap: Option<f32>,
    vocab: u32,
    norm_eps: f32,
}

fn closes_a_block(l: u32, every: u32) -> bool {
    every > 0 && (l + 1).is_multiple_of(every)
}

impl Model {
    pub fn k3(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            experts,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 8,
                dense_layers: 1,
                full_attn_every: 4,
                res_block: 4,
                mla: MlaDims {
                    heads: 16,
                    q_lora_rank: 768,
                    kv_lora_rank: 256,
                    qk_nope_head_dim: 128,
                    qk_rope_head_dim: 64,
                    v_head_dim: 128,
                    output_gate: true,
                },
                kda: KdaDims {
                    heads: 16,
                    head_dim: 128,
                    f_rank: 128,
                    conv_kernel: 4,
                    norm_eps: 1e-5,
                },
                moe: MoeDims {
                    experts: 64,
                    top_k: 6,
                    routed_scaling: 2.0,
                    inter: 1024,
                    shared_inter: 1024,
                },
                dense_inter: 5632,
                situ_beta: 1.0,
                situ_cap: None,
                vocab: 163_840,
                norm_eps: 1e-5,
            },
        )
    }

    fn new(weights: Dtype, experts: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let mla_heads = d.mla.heads / tp;
        let kda_heads = d.kda.heads / tp;
        let moe_inter = d.moe.inter / tp;
        let shared_inter = d.moe.shared_inter / tp;
        let dense_inter = d.dense_inter / tp;

        let hidden = d.hidden as u64;
        let full_at = |l: u32| closes_a_block(l, d.full_attn_every);
        let moe_at = |l: u32| l >= d.dense_layers;
        let blend_at = |l: u32| l > 0 && closes_a_block(l - 1, d.res_block);

        let a = &d.mla;
        let k = &d.kda;
        let qk_head_dim = (a.qk_nope_head_dim + a.qk_rope_head_dim) as u64;
        let q_b_width = mla_heads as u64 * qk_head_dim;
        let kv_a_width = (a.kv_lora_rank + a.qk_rope_head_dim) as u64;
        let kv_b_width = mla_heads as u64 * (a.qk_nope_head_dim + a.v_head_dim) as u64;
        let v_width = mla_heads as u64 * a.v_head_dim as u64;
        let kda_width = kda_heads as u64 * k.head_dim as u64;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, width: u64| Weight::sym(n(s), [width], weights);
                let mixer = if full_at(l) {
                    Mixer::Mla(Mla {
                        qk_nope_head_dim: a.qk_nope_head_dim,
                        qk_rope_head_dim: a.qk_rope_head_dim,
                        v_head_dim: a.v_head_dim,
                        sm_scale: (qk_head_dim as f32).sqrt().recip(),
                        q_a_proj: Weight::sym(
                            n("q_a_proj"),
                            [a.q_lora_rank as u64, hidden],
                            weights,
                        ),
                        q_a_norm: norm("q_a_norm", a.q_lora_rank as u64),
                        q_a_norm_eps: d.norm_eps,
                        q_b_proj: Weight::sym(
                            n("q_b_proj"),
                            [q_b_width, a.q_lora_rank as u64],
                            weights,
                        )
                        .columns(),
                        kv_a_proj: Weight::sym(n("kv_a_proj"), [kv_a_width, hidden], weights),
                        kv_a_norm: norm("kv_a_norm", a.kv_lora_rank as u64),
                        kv_a_norm_eps: d.norm_eps,
                        kv_b_proj: Weight::sym(
                            n("kv_b_proj"),
                            [kv_b_width, a.kv_lora_rank as u64],
                            weights,
                        )
                        .columns(),
                        gate: a.output_gate.then(|| {
                            Weight::sym(n("o_gate"), [v_width, hidden], weights).columns()
                        }),
                        o_proj: Weight::sym(n("o_proj"), [hidden, v_width], weights).rows(),
                        kv: format!("kv.{l}"),
                    })
                } else {
                    Mixer::Kda(Kda {
                        heads: kda_heads,
                        head_dim: k.head_dim,
                        conv_kernel: k.conv_kernel,
                        norm_eps: k.norm_eps,
                        qkv: Weight::sym(n("kda_qkv"), [3 * kda_width, hidden], weights)
                            .packed([kda_width, kda_width, kda_width]),
                        conv: Weight::sym(
                            n("kda_conv"),
                            [3 * kda_width, k.conv_kernel as u64],
                            weights,
                        )
                        .packed([kda_width, kda_width, kda_width]),
                        f_a: Weight::sym(n("kda_f_a"), [k.f_rank as u64, hidden], weights),
                        f_b: Weight::sym(n("kda_f_b"), [kda_width, k.f_rank as u64], weights)
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
                        gate: Weight::sym(n("kda_gate"), [kda_width, hidden], weights).columns(),
                        o_norm: Weight::sym(n("kda_o_norm"), [k.head_dim as u64], weights),
                        o_norm_eps: k.norm_eps,
                        o_proj: Weight::sym(n("kda_o_proj"), [hidden, kda_width], weights).rows(),
                        conv_state: format!("conv.{l}"),
                        delta_state: format!("delta.{l}"),
                    })
                };
                let mlp = if moe_at(l) {
                    let m = &d.moe;
                    let inter = moe_inter as u64;
                    let shared_width = shared_inter as u64;
                    Mlp::Routed {
                        router: Weight::sym(n("router"), [m.experts as u64, hidden], weights),
                        gate_up: Weight::sym(
                            n("experts_gate_up"),
                            [m.experts as u64, 2 * inter, hidden],
                            experts,
                        )
                        .bank([inter, inter]),
                        down: Weight::sym(
                            n("experts_down"),
                            [m.experts as u64, hidden, inter],
                            experts,
                        )
                        .rows(),
                        shared: (shared_inter > 0).then(|| Shared {
                            gate_up: Weight::sym(
                                n("shared_gate_up"),
                                [2 * shared_width, hidden],
                                weights,
                            )
                            .packed([shared_width, shared_width]),
                            down: Weight::sym(n("shared_down"), [hidden, shared_width], weights)
                                .rows(),
                            inter: shared_inter,
                        }),
                        experts: m.experts,
                        top_k: m.top_k,
                        routed_scaling: m.routed_scaling,
                        inter: moe_inter,
                        beta: d.situ_beta,
                        up_cap: d.situ_cap,
                    }
                } else {
                    let inter = dense_inter as u64;
                    Mlp::Dense {
                        gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], weights)
                            .packed([inter, inter]),
                        down: Weight::sym(n("down"), [hidden, inter], weights).rows(),
                        inter: dense_inter,
                        beta: d.situ_beta,
                        up_cap: d.situ_cap,
                    }
                };
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, crate::dense(weights));
                Layer {
                    res_blend: blend_at(l).then(|| ResBlend {
                        norm: norm("res_norm", hidden),
                        norm_eps: d.norm_eps,
                        proj: Weight::sym(n("res_proj"), [1, hidden], weights),
                    }),
                    mixer,
                    mixer_norm: norm("mixer_norm", hidden),
                    mixer_norm_eps: d.norm_eps,
                    mlp_norm: norm("mlp_norm", hidden),
                    mlp_norm_eps: d.norm_eps,
                    mlp,
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            mla_heads,
            kv_lora_rank: a.kv_lora_rank,
            adapters: ADAPTERS,
            kv,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: Weight::sym("lm_head", [d.vocab as u64, hidden], weights),
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
/// Eight slots of rank sixteen costs the k3 row 1 MiB a layer — two planes of
/// `8 x 16 x 2048` in the compute element — and 8 MiB over its eight, against
/// a table this family measures in hundreds of gibibytes.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {
    pub fn load(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        let mut claims = Vec::new();
        let mut state = |w: &Weight| claims.push(claim(w, self.tp));
        state(&self.embed);
        state(&self.final_norm);
        state(&self.head);
        for layer in &self.layers {
            state(&layer.mixer_norm);
            state(&layer.mlp_norm);
            if let Some(res) = &layer.res_blend {
                state(&res.norm);
                state(&res.proj);
            }
            match &layer.mixer {
                Mixer::Mla(a) => {
                    state(&a.q_a_proj);
                    state(&a.q_a_norm);
                    state(&a.q_b_proj);
                    state(&a.kv_a_proj);
                    state(&a.kv_a_norm);
                    state(&a.kv_b_proj);
                    if let Some(gate) = &a.gate {
                        state(gate);
                    }
                    state(&a.o_proj);
                }
                Mixer::Kda(k) => {
                    state(&k.qkv);
                    state(&k.conv);
                    state(&k.f_a);
                    state(&k.f_b);
                    state(&k.b);
                    state(&k.dt_bias);
                    state(&k.a_log);
                    state(&k.gate);
                    state(&k.o_norm);
                    state(&k.o_proj);
                }
            }
            match &layer.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    state(gate_up);
                    state(down);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    ..
                } => {
                    state(router);
                    state(gate_up);
                    state(down);
                    if let Some(s) = shared {
                        state(&s.gate_up);
                        state(&s.down);
                    }
                }
            }
        }
        elaborate(src, claims)
    }
}
