use model_dsl::{Dtype, Weight};
use model_loader::contract::ModelContract;

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

    /// **THE DRAFT HEAD, WHEN THE CHECKPOINT SHIPS ONE** (palo C3, design §8).
    ///
    /// `Option` because it is a fact about the ARTIFACT and not about the
    /// family: Qwen3.6-27B publishes `mtp.*` — one `fc`, two pre-norms, one
    /// full transformer block and a final norm, fifteen tensors, verified
    /// name-for-name against the cached checkpoint index — and no earlier
    /// qwen35 SKU publishes anything of the kind. A `None` here is not a
    /// disabled feature; it is a model that has no second head, and its trace
    /// carries not one node of the arm.
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
pub struct Mtp {
    /// Scales the embedding of the row's token before the fusion.
    pub pre_fc_norm_embedding: Weight,
    /// Scales the trunk's hidden state before the fusion.
    pub pre_fc_norm_hidden: Weight,
    /// The embedding half of the stored `[hidden, 2·hidden]` fusion bank.
    pub fc_embed: Weight,
    /// The hidden half of it.
    pub fc_hidden: Weight,
    /// The two pre-fusion norms share one epsilon, as every norm in this
    /// family does: `rms_norm_eps` is one number in the config and stating it
    /// per site is what keeps a site from silently inheriting another's.
    pub pre_fc_norm_eps: f32,
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
    /// The final norm before the readout.
    pub norm: Weight,
    pub norm_eps: f32,
}

pub enum Head {
    Tied,
    Bank(Weight),
}

/// **THE BUDGET IS THE SHAPE** (design §8, decision 17).
///
/// How many adapters a load can hold and how wide each one's waist is are
/// declared HERE, in the model text, because they are the leading axes of two
/// weights and shapes are the text's. `Budgets::max_adapters` is what a
/// deployment asks to be able to register, and `model_compiler::compile`
/// refuses a load whose ask is bigger than what these numbers seat — one
/// refusal, at the door, instead of a capacity discovered at a registration.
///
/// **RANK DIVERSITY IS BUCKETED BY BANK, NOT BY A BRANCH.** An adapter trained
/// at a lower rank is registered zero-padded into `rank` — which is exact, a
/// zero row of `A` contributing a zero to the waist — and a deployment whose
/// adapters spread widely across ranks declares a second family SKU with a
/// second `rank` rather than a runtime rank table nothing but the padding
/// would read. That is design §8's "rank-bucketed grouped GEMM" read
/// literally: the buckets are banks.
#[derive(Clone, Copy, Debug)]
pub struct Adapters {
    /// How many adapters the bank's first axis seats.
    pub slots: u32,
    /// The waist every adapter of this bank is padded to.
    pub rank: u32,
}

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
    /// Whether this SKU's checkpoint publishes an `mtp.*` draft head. One
    /// layer is all any shipped qwen states (`mtp_num_hidden_layers: 1`), so
    /// this is a boolean and not a count — a second layer would be a second
    /// block in the text, and no artifact asks for one.
    mtp: bool,
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
                mtp: false,
            },
        )
    }

    pub fn d0_8b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
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
                mtp: false,
            },
        )
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
                mtp: false,
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
        Model::new(
            w,
            kv,
            tp,
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
                mtp: true,
            },
        )
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let k_heads = d.k_heads / tp;
        let v_heads = d.v_heads / tp;
        let hidden = d.hidden as u64;
        let attn_at = |l: u32| l % d.attn_every == d.attn_every - 1;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], w);
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
                        conv: Weight::sym(n("conv"), [qkv, d.conv_kernel as u64], w)
                            .packed([k_w, k_w, v_w]),
                        dt_bias: Weight::sym(n("dt_bias"), [v_heads as u64], w).columns(),
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
                    // Registered, not landed: the checkpoint publishes neither of
                    // these and the loader must not demand them. Reserved at load
                    // and zeroed, and a zeroed `A` is the identity — so a fire
                    // through an unwritten row of the bank says exactly what the
                    // base model says.
                    //
                    // REPLICATED under tp, because both ends of this site are:
                    // the input is the replicated normed residual and the output
                    // is the reduced mixer result. Nothing here is cut, so
                    // nothing here is summed twice.
                    lora_a: Weight::sym(
                        n("lora_a"),
                        [ADAPTERS.slots as u64, ADAPTERS.rank as u64, hidden],
                        w,
                    )
                    .registered(),
                    lora_b: Weight::sym(
                        n("lora_b"),
                        [ADAPTERS.slots as u64, hidden, ADAPTERS.rank as u64],
                        w,
                    )
                    .registered(),
                }
            })
            .collect();

        // The draft head, when the artifact publishes one. Its mlp is stated at
        // the trunk's own dense width whatever the trunk's own mlp is: the
        // checkpoint's `mtp.layers.0.mlp` is dense even where the trunk routes,
        // because a draft block is one block and has no experts to route to.
        let mtp = d.mtp.then(|| {
            let inter = match &d.mlp {
                MlpDims::Dense { inter } => *inter,
                MlpDims::Routed(m) => m.inter,
            } / tp;
            Mtp {
                pre_fc_norm_embedding: Weight::sym("mtp.pre_fc_norm_embedding", [hidden], w),
                pre_fc_norm_hidden: Weight::sym("mtp.pre_fc_norm_hidden", [hidden], w),
                // REPLICATED, both halves. A fusion bank contracts over `hidden`
                // and produces `hidden`, and both ends of it are replicated
                // values — the embedding of a token every rank holds, and the
                // trunk's residual stream after its reduce. Cutting either way
                // would put a partial sum where a whole one is read.
                fc_embed: Weight::sym("mtp.fc_embed", [hidden, hidden], w),
                fc_hidden: Weight::sym("mtp.fc_hidden", [hidden, hidden], w),
                pre_fc_norm_eps: d.norm_eps,
                mixer_norm: Weight::sym("mtp.mixer_norm", [hidden], w),
                mixer_norm_eps: d.norm_eps,
                attn: gated_attn(w, &d, q_heads, kv_heads, "mtp", "kv.mtp".to_string()),
                mlp_norm: Weight::sym("mtp.mlp_norm", [hidden], w),
                mlp_norm_eps: d.norm_eps,
                mlp: dense_mlp(w, hidden, inter, "mtp"),
                norm: Weight::sym("mtp.norm", [hidden], w),
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
            final_norm: Weight::sym("final_norm", [hidden], w),
            final_norm_eps: d.norm_eps,
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
        q_norm: Weight::sym(n("q_norm"), [hd], w),
        q_norm_eps: d.norm_eps,
        k_norm: Weight::sym(n("k_norm"), [hd], w),
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

        // The draft head's own planes. Stated here and not folded into the
        // layer walk because a head is not a layer: it has no adapter bank
        // (nothing routes a correction into a draft), and its `fc` halves are
        // two claims over one stored bank — which `import.rs` is where the
        // slicing is said, and this is where the shapes are demanded.
        if let Some(mtp) = &self.mtp {
            claims.push(claim(&mtp.pre_fc_norm_embedding, self.tp));
            claims.push(claim(&mtp.pre_fc_norm_hidden, self.tp));
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
                    mtp.norm.name,
                ),
            }
            claims.push(claim(&mtp.norm, self.tp));
        }

        elaborate(src, claims)
    }
}
