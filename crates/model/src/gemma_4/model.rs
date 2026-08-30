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

    pub kv: Dtype,
    pub softcap: Option<f32>,
    pub embed: Weight,
    pub ple: Option<Ple>,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
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

struct Dims {
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
}

impl Model {
    pub fn e4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
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
            },
        )
    }

    pub fn b31(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            kv,
            tp,
            Dims {
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
            },
        )
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
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            sliding,
            global,
            adapters: ADAPTERS,
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
    ) -> Result<ModelContract, crate::contract::ModelError> {
        let mut claims = Vec::new();
        let mut claim = |w: &Weight| claims.push(crate::contract::claim(w, self.tp));

        claim(&self.embed);
        claim(&self.final_norm);

        for layer in &self.layers {
            claim(&layer.attn_norm);
            claim(&layer.post_attn_norm);
            claim(&layer.pre_ffw_norm);
            claim(&layer.post_ffw_norm);
            claim(&layer.attn.q_norm);
            match &layer.attn.banks {
                AttnBanks::Owned { qkv, k_norm, .. } => {
                    claim(k_norm);
                    claim(qkv);
                }

                AttnBanks::Shared { q_proj } => {
                    claim(q_proj);
                }
            }
            claim(&layer.o_proj);
            claim(&layer.gate_up);
            claim(&layer.down);
            if let Some(scalar) = &layer.scalar {
                claim(scalar);
            }
        }

        if let Some(ple) = &self.ple {
            claim(&ple.model_proj);
            claim(&ple.model_norm);
            for per in &ple.per_layer {
                claim(&per.table);
                claim(&per.gate);
                claim(&per.proj);
                claim(&per.norm);
                claim(&per.scalar);
            }
        }

        crate::contract::elaborate(src, claims)
    }
}
