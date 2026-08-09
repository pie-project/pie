//! Name-binding the llama-like weight schema out of a loaded checkpoint.
//!
//! Port of `bind_llama_like`, `bind_phi3` and `bind_olmo3` from
//! `driver-cuda/csrc/src/model/llama_like/qwen3.cpp`, plus the
//! `Qwen3Weights` / `Qwen3LayerWeights` structs they fill. The generated
//! forward bodies reach through those structs 7,836 times.
//!
//! # A null slot here is a decision, not an absence
//!
//! Most of these pointers are optional, and the forward path reads a missing
//! one as "this architecture does not have that" — it skips a bias add, or an
//! RMSNorm, or takes the unfused GEMM path. Nothing checks afterwards. So a
//! binder that filled a slot it should have left empty, or left one empty it
//! should have filled, produces a model that loads, runs, and is quietly
//! wrong.
//!
//! Three flags drive it, and they are independent:
//!
//! - `attention_bias` — Qwen-2 / OLMo-3 / GPT-OSS carry q/k/v biases.
//! - `use_qk_norm` — Qwen3 / Gemma-3 / OLMo-3 normalise per head.
//! - `tie_word_embeddings` — the lm_head may be the embedding table itself.
//!
//! `tests/weight_bind_parity.rs` runs the same grid the C++ oracle does and
//! compares every slot of every layer, nulls included.
//!
//! # Optional in two different senses
//!
//! `q_norm` being absent means the architecture has no per-head norm.
//! `qkv_proj_fused` being absent means something else entirely: the loader's
//! contract *declined to fuse* this group — it refuses quantized and
//! non-bf16 groups, because per-weight scales do not compose across a concat —
//! and the forward path must stay on its three-narrow-GEMM fallback. Both are
//! `None` here, and the type cannot distinguish them, which is why
//! [`Qwen3LayerWeights::fusion`] exists to name the second one at the point
//! where the forward path branches on it.
//!
//! # `use_qk_norm` set with the tensors missing is an error, deliberately
//!
//! The binder calls `must` for the norms rather than probing, so a config that
//! claims per-head norms without shipping them fails at load with the missing
//! name. The alternative — silently leaving the slot null — would turn a
//! broken checkpoint into a model that runs with the norm skipped.

use std::collections::BTreeMap;

use crate::model::weight_view::QuantMeta;

/// The four `HfConfig` fields the binders read.
///
/// The real `HfConfig` has dozens; carrying only what is read keeps the
/// binder's inputs visible, and makes it obvious when a new architecture
/// starts depending on a fifth.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BindConfig {
    /// Number of transformer blocks to bind.
    pub num_hidden_layers: i32,
    /// Whether the lm_head may fall back to the embedding table.
    pub tie_word_embeddings: bool,
    /// Whether q/k/v projections carry bias terms.
    pub attention_bias: bool,
    /// Whether attention normalises per head.
    pub use_qk_norm: bool,
}

/// What a binder needs from a loaded checkpoint.
///
/// Five methods, which is the whole of `LoadedModel` the binders touch. Being
/// a trait rather than a concrete type is what lets the parity test drive the
/// real binding logic without a GPU or a checkpoint on disk.
pub trait WeightSource {
    /// How the caller refers to a bound tensor.
    ///
    /// `Eq` because `tie_word_embeddings` is implemented by binding the same
    /// handle twice, and callers ask whether the lm_head aliases the embed
    /// table.
    type Handle: Copy + Eq;

    /// The config fields that decide which slots get filled.
    fn config(&self) -> BindConfig;

    /// Look a tensor up by its checkpoint name.
    fn get(&self, name: &str) -> Option<Self::Handle>;

    /// The quantization side-map entry for a weight, if it has one.
    fn quant_meta(&self, name: &str) -> Option<QuantMeta>;
}

/// A weight the binder required and the checkpoint did not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindError {
    /// A required tensor is absent.
    MissingWeight {
        /// The name that was looked up.
        name: String,
    },
    /// No `lm_head` and no permission to reuse the embedding table.
    UntiedLmHeadMissing,
    /// Phi-3 reached the binder with its fused checkpoint tensors unsplit.
    Phi3NotMaterialized {
        /// Which group the loader failed to split.
        group: &'static str,
    },
}

impl BindError {
    /// The C++ `what()` string, so a parity transcript can compare messages.
    #[must_use]
    pub fn cpp_message(&self) -> String {
        
        match self {
            Self::MissingWeight { name } => {
                format!("llama-like: missing weight '{name}'")
            }
            Self::UntiedLmHeadMissing => {
                "llama-like: lm_head missing and tie_word_embeddings=false".into()
            }
            Self::Phi3NotMaterialized { group } => {
                format!("bind_phi3: storage loader did not materialize {group}")
            }
        }
    }
}

type Result<T> = core::result::Result<T, BindError>;

fn must<S: WeightSource>(src: &S, name: &str) -> Result<S::Handle> {
    src.get(name).ok_or_else(|| BindError::MissingWeight {
        name: name.into(),
    })
}

/// Which projection groups the loader fused for a layer.
///
/// Names the second sense of "optional" described in the module docs, so the
/// forward path's branch reads as a choice between two code paths rather than
/// as a null check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fusion {
    /// A single wide q/k/v GEMM is available.
    pub qkv: bool,
    /// A single wide gate/up GEMM is available.
    pub gate_up: bool,
}

/// One transformer block's weights.
///
/// Field-for-field with the C++ `Qwen3LayerWeights`, including which slots are
/// optional. It does not cross the ABI — only the [`WeightView`]s built from
/// it do — so this is an ordinary Rust struct.
///
/// [`WeightView`]: crate::model::weight_view::WeightView
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3LayerWeights<H: Copy + Eq> {
    /// Pre-attention RMSNorm. On post-norm architectures this is the
    /// *post*-attention norm; see [`bind_olmo3`].
    pub attn_norm: H,
    /// The MLP-side RMSNorm.
    pub mlp_norm: H,
    /// Query projection.
    pub q_proj: H,
    /// Key projection.
    pub k_proj: H,
    /// Value projection.
    pub v_proj: H,
    /// Output projection.
    pub o_proj: H,
    /// Query bias, when `attention_bias` is set.
    pub q_bias: Option<H>,
    /// Key bias, when `attention_bias` is set.
    pub k_bias: Option<H>,
    /// Value bias, when `attention_bias` is set.
    pub v_bias: Option<H>,
    /// Per-head query norm, when `use_qk_norm` is set.
    pub q_norm: Option<H>,
    /// Per-head key norm, when `use_qk_norm` is set.
    pub k_norm: Option<H>,
    /// Gate projection.
    pub gate_proj: H,
    /// Up projection.
    pub up_proj: H,
    /// Down projection.
    pub down_proj: H,
    /// Packed q/k/v weights, when the loader's contract fused the group.
    pub qkv_proj_fused: Option<H>,
    /// Packed gate/up weights, when the loader's contract fused the group.
    pub gate_up_proj_fused: Option<H>,
    /// Quantization metadata, keyed by slot name (`"q_proj"`, `"down_proj"`…).
    ///
    /// The C++ has seven separate `optional<QuantMeta>` members, which means
    /// the binder has seven near-identical lines and a swap between two of
    /// them — handing the gate GEMM the up projection's scales — is a
    /// one-token edit that reads correctly. Keying the map by the same name
    /// the weight is fetched under makes the pairing structural instead.
    pub quant: BTreeMap<&'static str, QuantMeta>,
}

impl<H: Copy + Eq> Qwen3LayerWeights<H> {
    /// Which projection groups can take the wide-GEMM path.
    #[must_use]
    pub fn fusion(&self) -> Fusion {
        Fusion {
            qkv: self.qkv_proj_fused.is_some(),
            gate_up: self.gate_up_proj_fused.is_some(),
        }
    }

    /// The quant metadata for a projection slot, if it has any.
    #[must_use]
    pub fn quant_for(&self, slot: &str) -> Option<&QuantMeta> {
        self.quant.get(slot)
    }
}

/// The llama-like weight schema: shared by Qwen3, Llama 3, Qwen 2, Mistral,
/// Phi-3 and OLMo-3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3Weights<H: Copy + Eq> {
    /// Token embedding table.
    pub embed: H,
    /// Final RMSNorm before the head.
    pub final_norm: H,
    /// Output projection. May be the same handle as [`Self::embed`].
    pub lm_head: H,
    /// Per-block weights, in layer order.
    pub layers: Vec<Qwen3LayerWeights<H>>,
}

impl<H: Copy + Eq> Qwen3Weights<H> {
    /// Whether the head reuses the embedding table.
    ///
    /// The observable consequence of `tie_word_embeddings`. Worth a method
    /// because the lm_head GEMM transposes when tied, and the call site that
    /// decides should not have to know how the tie was arranged.
    #[must_use]
    pub fn lm_head_is_tied(&self) -> bool {
        self.lm_head == self.embed
    }
}

fn layer_prefix(i: i32) -> String {
    format!("model.layers.{i}.")
}

/// The seven projections that can carry quantization metadata, as
/// `(slot name, checkpoint suffix)`.
///
/// One table instead of fourteen lines, so the slot and the name it is keyed
/// on cannot drift apart.
const QUANTIZABLE: [(&str, &str); 7] = [
    ("q_proj", "self_attn.q_proj.weight"),
    ("k_proj", "self_attn.k_proj.weight"),
    ("v_proj", "self_attn.v_proj.weight"),
    ("o_proj", "self_attn.o_proj.weight"),
    ("gate_proj", "mlp.gate_proj.weight"),
    ("up_proj", "mlp.up_proj.weight"),
    ("down_proj", "mlp.down_proj.weight"),
];

fn bind_head<S: WeightSource>(src: &S, cfg: BindConfig) -> Result<(S::Handle, S::Handle, S::Handle)> {
    let embed = must(src, "model.embed_tokens.weight")?;
    let final_norm = must(src, "model.norm.weight")?;
    let lm_head = match src.get("lm_head.weight") {
        Some(h) => h,
        None if cfg.tie_word_embeddings => embed,
        None => return Err(BindError::UntiedLmHeadMissing),
    };
    Ok((embed, final_norm, lm_head))
}

fn bind_quant<S: WeightSource>(src: &S, p: &str) -> BTreeMap<&'static str, QuantMeta> {
    let mut out = BTreeMap::new();
    for (slot, suffix) in QUANTIZABLE {
        if let Some(m) = src.quant_meta(&format!("{p}{suffix}")) {
            out.insert(slot, m);
        }
    }
    out
}

/// Bind the canonical llama-like schema.
///
/// Reads `use_qk_norm` to decide whether the per-head norms are required, and
/// `attention_bias` to decide whether the q/k/v biases are. The fused
/// projections are probed, never required: their absence is the loader's
/// contract declining to fuse.
///
/// # Errors
///
/// Returns [`BindError::MissingWeight`] for any required tensor the checkpoint
/// lacks, or [`BindError::UntiedLmHeadMissing`] when there is no `lm_head` and
/// `tie_word_embeddings` is unset.
pub fn bind_llama_like<S: WeightSource>(src: &S) -> Result<Qwen3Weights<S::Handle>> {
    let cfg = src.config();
    let (embed, final_norm, lm_head) = bind_head(src, cfg)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers.max(0) as usize);
    for i in 0..cfg.num_hidden_layers {
        let p = layer_prefix(i);
        // The norms are resolved before the fused projections are probed, and
        // the fused probes before the q/k/v lookups. Lookup order is
        // observable and the parity transcript records it: when a name is
        // present under two spellings, the order is what decides which one
        // binds. Hoisting these two out of the struct literal is what keeps
        // that order, since Rust evaluates literal fields as written.
        let attn_norm = must(src, &format!("{p}input_layernorm.weight"))?;
        let mlp_norm = must(src, &format!("{p}post_attention_layernorm.weight"))?;

        let qkv_proj_fused = src.get(&format!("{p}self_attn.qkv_proj.fused.weight"));
        let gate_up_proj_fused = src.get(&format!("{p}mlp.gate_up_proj.fused.weight"));

        layers.push(Qwen3LayerWeights {
            attn_norm,
            mlp_norm,
            q_proj: must(src, &format!("{p}self_attn.q_proj.weight"))?,
            k_proj: must(src, &format!("{p}self_attn.k_proj.weight"))?,
            v_proj: must(src, &format!("{p}self_attn.v_proj.weight"))?,
            o_proj: must(src, &format!("{p}self_attn.o_proj.weight"))?,
            q_bias: bias(src, &p, cfg.attention_bias, "q")?,
            k_bias: bias(src, &p, cfg.attention_bias, "k")?,
            v_bias: bias(src, &p, cfg.attention_bias, "v")?,
            q_norm: qk_norm(src, &p, cfg.use_qk_norm, "q")?,
            k_norm: qk_norm(src, &p, cfg.use_qk_norm, "k")?,
            gate_proj: must(src, &format!("{p}mlp.gate_proj.weight"))?,
            up_proj: must(src, &format!("{p}mlp.up_proj.weight"))?,
            down_proj: must(src, &format!("{p}mlp.down_proj.weight"))?,
            qkv_proj_fused,
            gate_up_proj_fused,
            quant: bind_quant(src, &p),
        });
    }

    Ok(Qwen3Weights {
        embed,
        final_norm,
        lm_head,
        layers,
    })
}

fn bias<S: WeightSource>(
    src: &S,
    p: &str,
    enabled: bool,
    which: &str,
) -> Result<Option<S::Handle>> {
    if !enabled {
        return Ok(None);
    }
    must(src, &format!("{p}self_attn.{which}_proj.bias")).map(Some)
}

fn qk_norm<S: WeightSource>(
    src: &S,
    p: &str,
    enabled: bool,
    which: &str,
) -> Result<Option<S::Handle>> {
    if !enabled {
        return Ok(None);
    }
    must(src, &format!("{p}self_attn.{which}_norm.weight")).map(Some)
}

/// Bind Phi-3, after checking the loader already split its fused checkpoint
/// tensors into the canonical names.
///
/// The check is the whole of what this adds. Without it a Phi-3 checkpoint
/// whose split step was skipped would fail inside [`bind_llama_like`] naming
/// `q_proj` — true but unhelpful, because the missing step is upstream.
///
/// # Errors
///
/// [`BindError::Phi3NotMaterialized`] when the split did not happen, plus
/// anything [`bind_llama_like`] returns.
pub fn bind_phi3<S: WeightSource>(src: &S) -> Result<Qwen3Weights<S::Handle>> {
    let cfg = src.config();
    for i in 0..cfg.num_hidden_layers {
        let p = layer_prefix(i);
        let have_qkv = ["q", "k", "v"]
            .iter()
            .all(|w| src.get(&format!("{p}self_attn.{w}_proj.weight")).is_some());
        if !have_qkv {
            return Err(BindError::Phi3NotMaterialized {
                group: "q/k/v projections",
            });
        }
        let have_gate_up = ["gate", "up"]
            .iter()
            .all(|w| src.get(&format!("{p}mlp.{w}_proj.weight")).is_some());
        if !have_gate_up {
            return Err(BindError::Phi3NotMaterialized {
                group: "gate/up projections",
            });
        }
    }
    bind_llama_like(src)
}

/// Bind OLMo-3, which is post-norm and stores its norms at HF positions that
/// do not match llama.
///
/// The mapping is the point:
///
/// | slot | llama-like | OLMo-3 |
/// |---|---|---|
/// | `attn_norm` | `input_layernorm` | `post_attention_layernorm` |
/// | `mlp_norm` | `post_attention_layernorm` | `post_feedforward_layernorm` |
///
/// `post_attention_layernorm` exists in both and means different things, so a
/// checkpoint carrying both spellings binds differently depending on which
/// binder runs — which is why the parity oracle populates both and checks
/// which one wins rather than checking that the only option was taken.
///
/// The per-head norms are unconditional here: they are OLMo-3's defining
/// feature alongside post-norm, and it does not consult `use_qk_norm`.
/// OLMo-3 also never fuses, and reads no quantization metadata.
///
/// # Errors
///
/// As [`bind_llama_like`].
pub fn bind_olmo3<S: WeightSource>(src: &S) -> Result<Qwen3Weights<S::Handle>> {
    let cfg = src.config();
    let (embed, final_norm, lm_head) = bind_head(src, cfg)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers.max(0) as usize);
    for i in 0..cfg.num_hidden_layers {
        let p = layer_prefix(i);
        layers.push(Qwen3LayerWeights {
            attn_norm: must(src, &format!("{p}post_attention_layernorm.weight"))?,
            mlp_norm: must(src, &format!("{p}post_feedforward_layernorm.weight"))?,
            q_proj: must(src, &format!("{p}self_attn.q_proj.weight"))?,
            k_proj: must(src, &format!("{p}self_attn.k_proj.weight"))?,
            v_proj: must(src, &format!("{p}self_attn.v_proj.weight"))?,
            o_proj: must(src, &format!("{p}self_attn.o_proj.weight"))?,
            q_bias: bias(src, &p, cfg.attention_bias, "q")?,
            k_bias: bias(src, &p, cfg.attention_bias, "k")?,
            v_bias: bias(src, &p, cfg.attention_bias, "v")?,
            q_norm: Some(must(src, &format!("{p}self_attn.q_norm.weight"))?),
            k_norm: Some(must(src, &format!("{p}self_attn.k_norm.weight"))?),
            gate_proj: must(src, &format!("{p}mlp.gate_proj.weight"))?,
            up_proj: must(src, &format!("{p}mlp.up_proj.weight"))?,
            down_proj: must(src, &format!("{p}mlp.down_proj.weight"))?,
            qkv_proj_fused: None,
            gate_up_proj_fused: None,
            quant: BTreeMap::new(),
        });
    }

    Ok(Qwen3Weights {
        embed,
        final_norm,
        lm_head,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::weight_view::QuantKind;

    /// A checkpoint that is just a set of names, plus a probe log.
    #[derive(Default)]
    struct Fake {
        cfg: BindConfig,
        names: Vec<String>,
        quant: Vec<(String, QuantMeta)>,
    }

    impl Fake {
        fn with(mut self, names: &[&str]) -> Self {
            self.names.extend(names.iter().map(|s| (*s).to_string()));
            self
        }

        fn full(layers: i32) -> Self {
            let mut f = Fake::default().with(&["model.embed_tokens.weight", "model.norm.weight"]);
            f.cfg.num_hidden_layers = layers;
            for i in 0..layers {
                let p = layer_prefix(i);
                for s in [
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ] {
                    f.names.push(format!("{p}{s}"));
                }
            }
            f
        }
    }

    impl WeightSource for Fake {
        type Handle = usize;

        fn config(&self) -> BindConfig {
            self.cfg
        }

        fn get(&self, name: &str) -> Option<usize> {
            self.names.iter().position(|n| n == name)
        }

        fn quant_meta(&self, name: &str) -> Option<QuantMeta> {
            self.quant.iter().find(|(n, _)| n == name).map(|(_, m)| *m)
        }
    }

    #[test]
    fn a_missing_lm_head_is_only_an_error_when_untied() {
        let mut f = Fake::full(0);
        assert_eq!(
            bind_llama_like(&f).unwrap_err(),
            BindError::UntiedLmHeadMissing
        );
        f.cfg.tie_word_embeddings = true;
        let w = bind_llama_like(&f).unwrap();
        assert!(w.lm_head_is_tied());
    }

    #[test]
    fn a_present_lm_head_wins_over_the_tie_flag() {
        // Both available: the explicit head is not shadowed by the flag.
        let mut f = Fake::full(0).with(&["lm_head.weight"]);
        f.cfg.tie_word_embeddings = true;
        let w = bind_llama_like(&f).unwrap();
        assert!(!w.lm_head_is_tied());
    }

    #[test]
    fn the_qk_norm_flag_is_required_not_probed() {
        // The distinction that keeps a broken checkpoint from running with
        // the norm silently skipped.
        let mut f = Fake::full(1).with(&["lm_head.weight"]);
        assert!(bind_llama_like(&f).unwrap().layers[0].q_norm.is_none());
        f.cfg.use_qk_norm = true;
        assert_eq!(
            bind_llama_like(&f).unwrap_err(),
            BindError::MissingWeight {
                name: "model.layers.0.self_attn.q_norm.weight".into()
            }
        );
    }

    #[test]
    fn bias_and_qk_norm_are_independent() {
        let mut f = Fake::full(1).with(&[
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ]);
        f.cfg.attention_bias = true;
        let l = bind_llama_like(&f).unwrap().layers.remove(0);
        assert!(l.q_bias.is_some() && l.k_bias.is_some() && l.v_bias.is_some());
        assert!(
            l.q_norm.is_none(),
            "the bias flag must not pull in the norms"
        );

        f.cfg.attention_bias = false;
        f.cfg.use_qk_norm = true;
        let l = bind_llama_like(&f).unwrap().layers.remove(0);
        assert!(l.q_bias.is_none(), "and the norm flag must not pull in bias");
        assert!(l.q_norm.is_some() && l.k_norm.is_some());
    }

    #[test]
    fn fused_projections_are_probed_never_required() {
        let f = Fake::full(1)
            .with(&["lm_head.weight", "model.layers.0.mlp.gate_up_proj.fused.weight"]);
        let l = &bind_llama_like(&f).unwrap().layers[0];
        assert_eq!(
            l.fusion(),
            Fusion {
                qkv: false,
                gate_up: true
            },
            "each group fuses independently"
        );
    }

    #[test]
    fn each_quant_entry_is_keyed_on_the_weight_it_belongs_to() {
        let mut f = Fake::full(1).with(&["lm_head.weight"]);
        for (slot, suffix) in QUANTIZABLE {
            f.quant.push((
                format!("model.layers.0.{suffix}"),
                QuantMeta {
                    kind: QuantKind::PerGroup,
                    group_size: slot.len() as i32,
                    ..QuantMeta::default()
                },
            ));
        }
        let l = &bind_llama_like(&f).unwrap().layers[0];
        for (slot, _) in QUANTIZABLE {
            assert_eq!(
                l.quant_for(slot).unwrap().group_size,
                slot.len() as i32,
                "{slot} got another projection's metadata"
            );
        }
    }

    #[test]
    fn olmo3_reads_post_attention_layernorm_into_the_attn_norm_slot() {
        // Both spellings present, so this tests a choice rather than the only
        // option. `input_layernorm` must be ignored.
        let mut f = Fake::default().with(&["model.embed_tokens.weight", "model.norm.weight"]);
        f.cfg.num_hidden_layers = 1;
        f.cfg.tie_word_embeddings = true;
        for s in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            f.names.push(format!("model.layers.0.{s}"));
        }
        let l = &bind_olmo3(&f).unwrap().layers[0];
        assert_eq!(
            l.attn_norm,
            f.get("model.layers.0.post_attention_layernorm.weight").unwrap()
        );
        assert_ne!(
            l.attn_norm,
            f.get("model.layers.0.input_layernorm.weight").unwrap()
        );
        assert_eq!(
            l.mlp_norm,
            f.get("model.layers.0.post_feedforward_layernorm.weight")
                .unwrap()
        );
        assert!(
            l.q_norm.is_some(),
            "OLMo-3 norms are unconditional, not use_qk_norm-gated"
        );
    }

    #[test]
    fn phi3_names_the_split_step_rather_than_the_missing_weight() {
        let mut f = Fake::default().with(&[
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ]);
        f.cfg.num_hidden_layers = 1;
        assert_eq!(
            bind_phi3(&f).unwrap_err(),
            BindError::Phi3NotMaterialized {
                group: "q/k/v projections"
            },
            "pointing upstream, not at q_proj"
        );
    }

    #[test]
    fn zero_layers_still_binds_the_head() {
        let f = Fake::full(0).with(&["lm_head.weight"]);
        let w = bind_llama_like(&f).unwrap();
        assert!(w.layers.is_empty());
        assert!(!w.lm_head_is_tied());
    }
}
