//! Resolving the names a trace states against a loaded checkpoint.
//!
//! [`executor::Resolver`] is a trait with two questions — where does this
//! weight live, where does this named value live — and the crate docs call its
//! implementation *"the one thing that stays per-family: a **map** rather than
//! a switch"*. This is that map.
//!
//! # Why a map is not a violation
//!
//! "Nothing in the driver may choose a kernel" is about *behaviour*. A
//! resolver chooses nothing: it answers `layer.3.qkv` with an address. What
//! makes it per-family is only that a checkpoint and a text spell the same
//! tensor differently, and translating a spelling is not a decision about what
//! runs. The test is whether removing the map changes which kernels fire — and
//! it does not; it changes whether they find their operands.
//!
//! # The two spellings
//!
//! A text states `layer.3.qkv`, layer-unrolled and concrete, because that is
//! what makes a trace readable. A checkpoint states
//! `model.layers.3.self_attn.qkv_proj.weight`, because that is what the
//! exporter wrote. [`Names`] is the translation, and it is data — a prefix, a
//! suffix per role, and the affine sidecar's two names.
//!
//! **The sidecar spelling is the one that bites.** `MatW::scale_names` emits
//! `.scales` and `.zeros`; MLX checkpoints write `.scales` and `.biases`. Both
//! are right for their own side, and this is exactly the kind of disagreement
//! a map exists to absorb rather than one either side should have bent for.

use std::collections::HashMap;

use super::executor::{Resolver, Slice};
use model_compiler::trace::ValueId;

/// How a checkpoint spells what a text names.
///
/// Data rather than code: a family that spells its tensors differently is a
/// different `Names`, not a different resolver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
    /// What a layer-scoped name gets in front of its index —
    /// `model.layers.` for a HuggingFace export.
    pub layer_prefix: String,
    /// The text's role name → the checkpoint's path within a layer.
    /// `qkv` → `self_attn.qkv_proj`.
    pub roles: HashMap<String, String>,
    /// Names with no layer at all — `embed`, `lm_head`, `final_norm`.
    pub globals: HashMap<String, String>,
    /// What the checkpoint calls a packed weight's own tensor.
    pub weight_suffix: String,
    /// What it calls the zero point the text spells `.zeros`.
    pub zero_point_suffix: String,
}

impl Names {
    /// The convention `model::llama_3::contract` publishes, which is what
    /// `stage_plan_weights` keys its map by.
    ///
    /// **Read off the contract, not guessed.** The lowering maps
    /// `model.layers.{l}.{member}` to `layers.{l}.{member}`,
    /// `model.embed_tokens.*` to `shared_embedding.*` when the deployment ties
    /// its embeddings, and `model.norm.weight` to `final_norm.weight`. An
    /// earlier draft of this map assumed the HuggingFace spelling and was
    /// wrong on all three — it was self-consistent, and the test that held the
    /// text against it passed, because both sides were this file.
    ///
    /// # The two names with no tensor
    ///
    /// `qkv` and `gate_up` are FUSED handles, and **no Metal deployment has
    /// them**: `compile_load_plan` authors with `Projections::InPlace`, and
    /// `dense_fused_projection_joins` returns before doing anything under that
    /// policy. So the MLX path publishes the three and two projections
    /// separately, which is also what `weight_binds` binds.
    ///
    /// They are still mapped, to the spelling a JOINING plan would publish —
    /// `…qkv_proj.fused.weight`, with the loader's own `.fused` infix — so that
    /// a deployment which does join resolves. The text asks the tensors which
    /// it is (`LlamaLikeFacts::fused_qkv`, `LlamaLikeMetalFacts::gate_up_fused`)
    /// rather than either side assuming.
    #[must_use]
    pub fn mlx() -> Self {
        let roles = [
            // The fused pair — see the note above. The `.fused` infix is the
            // loader's own: `dense_fused_projection_joins` publishes
            // `…qkv_proj.fused.weight`, not `…qkv_proj.weight`.
            ("qkv", "self_attn.qkv_proj.fused"),
            ("gate_up", "mlp.gate_up_proj.fused"),
            // The projections as the checkpoint ships them.
            ("q_proj", "self_attn.q_proj"),
            ("k_proj", "self_attn.k_proj"),
            ("v_proj", "self_attn.v_proj"),
            ("o_proj", "self_attn.o_proj"),
            // The DSL's own handle names (`Layer::gate_proj` / `up_proj`),
            // which is what the text spells.
            ("gate_proj", "mlp.gate_proj"),
            ("up_proj", "mlp.up_proj"),
            ("down", "mlp.down_proj"),
            // The norms.
            ("q_norm", "self_attn.q_norm"),
            ("k_norm", "self_attn.k_norm"),
            ("attn_norm", "input_layernorm"),
            ("mlp_norm", "post_attention_layernorm"),
        ]
        .into_iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        let globals = [
            // Tied: one table serves both ends, which is why the readout and
            // the embedding answer to the same name.
            ("embed", "shared_embedding"),
            ("lm_head", "shared_embedding"),
            ("final_norm", "final_norm"),
        ]
        .into_iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        Self {
            layer_prefix: "layers.".to_string(),
            roles,
            globals,
            weight_suffix: ".weight".to_string(),
            // The text says `.zeros`, the checkpoint says `.biases`. Both are
            // right on their own side; the map is where they meet.
            zero_point_suffix: ".biases".to_string(),
        }
    }
}

/// What a traced name decomposes into.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Traced<'a> {
    layer: Option<u32>,
    role: &'a str,
    /// `.scales` / `.zeros`, or empty for the packed tensor itself.
    sidecar: &'a str,
}

/// Split `layer.3.qkv.scales` into its three parts.
///
/// Returns `None` for a name that is not in the text's shape at all, which is
/// drift rather than a spelling this map has not learned.
fn decompose(name: &str) -> Option<Traced<'_>> {
    let (rest, sidecar) = match name.rfind('.') {
        Some(at) if matches!(&name[at..], ".scales" | ".zeros") => (&name[..at], &name[at..]),
        _ => (name, ""),
    };
    if let Some(tail) = rest.strip_prefix("layer.") {
        let (index, role) = tail.split_once('.')?;
        Some(Traced {
            layer: Some(index.parse().ok()?),
            role,
            sidecar,
        })
    } else {
        Some(Traced {
            layer: None,
            role: rest,
            sidecar,
        })
    }
}

/// Answers a trace's weight and value names out of a loaded checkpoint.
pub struct Store<'a> {
    names: Names,
    /// Checkpoint tensor name → where it was staged.
    tensors: &'a HashMap<String, Slice>,
    /// The values a seam binds, by id.
    named: &'a HashMap<ValueId, Slice>,
    /// Every traced name this store could not answer, in ask order.
    ///
    /// Collected rather than logged: a fire that cannot bind is diagnosed by
    /// the WHOLE list, and stopping at the first turns one debugging session
    /// into as many as there are missing tensors.
    missed: Vec<String>,
}

impl<'a> Store<'a> {
    /// A store over `tensors`, spelled by `names`.
    #[must_use]
    pub fn new(
        names: Names,
        tensors: &'a HashMap<String, Slice>,
        named: &'a HashMap<ValueId, Slice>,
    ) -> Self {
        Self {
            names,
            tensors,
            named,
            missed: Vec::new(),
        }
    }

    /// The checkpoint tensor a traced name means, spelled the checkpoint's way.
    ///
    /// `None` when the name is not in the text's shape — which is drift, not a
    /// gap in this map.
    #[must_use]
    pub fn checkpoint_name(&self, traced: &str) -> Option<String> {
        let t = decompose(traced)?;
        let base = match t.layer {
            Some(l) => {
                let role = self.names.roles.get(t.role)?;
                format!("{}{l}.{role}", self.names.layer_prefix)
            }
            None => self.names.globals.get(t.role)?.clone(),
        };
        Some(match t.sidecar {
            "" => format!("{base}{}", self.names.weight_suffix),
            ".scales" => format!("{base}.scales"),
            _ => format!("{base}{}", self.names.zero_point_suffix),
        })
    }

    /// Every traced name this store was asked for and could not answer.
    #[must_use]
    pub fn missed(&self) -> &[String] {
        &self.missed
    }
}

impl Resolver for Store<'_> {
    fn weight(&mut self, name: &str) -> Option<Slice> {
        let found = self
            .checkpoint_name(name)
            .and_then(|spelled| self.tensors.get(&spelled).copied());
        if found.is_none() {
            self.missed.push(name.to_string());
        }
        found
    }

    fn named(&mut self, value: ValueId) -> Option<Slice> {
        self.named.get(&value).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store<'a>(
        tensors: &'a HashMap<String, Slice>,
        named: &'a HashMap<ValueId, Slice>,
    ) -> Store<'a> {
        Store::new(Names::mlx(), tensors, named)
    }

    #[test]
    fn a_layer_scoped_name_becomes_the_checkpoints_path() {
        let (t, n) = (HashMap::new(), HashMap::new());
        let s = store(&t, &n);
        assert_eq!(
            s.checkpoint_name("layer.3.qkv").as_deref(),
            Some("layers.3.self_attn.qkv_proj.fused.weight")
        );
        assert_eq!(
            s.checkpoint_name("layer.11.attn_norm").as_deref(),
            Some("layers.11.input_layernorm.weight")
        );
    }

    #[test]
    fn the_zero_point_the_text_calls_zeros_is_the_checkpoints_biases() {
        // The disagreement this map exists for: `MatW::scale_names` emits
        // `.zeros`, MLX writes `.biases`, and both are right on their own side.
        let (t, n) = (HashMap::new(), HashMap::new());
        let s = store(&t, &n);
        assert_eq!(
            s.checkpoint_name("layer.0.qkv.scales").as_deref(),
            Some("layers.0.self_attn.qkv_proj.fused.scales")
        );
        assert_eq!(
            s.checkpoint_name("layer.0.qkv.zeros").as_deref(),
            Some("layers.0.self_attn.qkv_proj.fused.biases")
        );
    }

    #[test]
    fn a_global_name_carries_no_layer_and_no_prefix() {
        let (t, n) = (HashMap::new(), HashMap::new());
        let s = store(&t, &n);
        assert_eq!(
            s.checkpoint_name("embed").as_deref(),
            Some("shared_embedding.weight")
        );
        assert_eq!(
            s.checkpoint_name("embed.scales").as_deref(),
            Some("shared_embedding.scales")
        );
    }

    #[test]
    fn a_name_outside_the_texts_shape_is_drift_and_not_a_missing_spelling() {
        let (t, n) = (HashMap::new(), HashMap::new());
        let s = store(&t, &n);
        assert_eq!(s.checkpoint_name("layer.3.nonesuch"), None);
        assert_eq!(s.checkpoint_name("nonesuch"), None);
    }

    #[test]
    fn a_store_collects_every_name_it_missed_rather_than_stopping_at_one() {
        // A fire that cannot bind is diagnosed by the whole list; stopping at
        // the first turns one debugging session into as many as are missing.
        let mut tensors = HashMap::new();
        tensors.insert("layers.0.self_attn.qkv_proj.fused.weight".to_string(), Slice {
            address: 0x100,
            bytes: 64,
        });
        let n = HashMap::new();
        let mut s = store(&tensors, &n);
        assert_eq!(s.weight("layer.0.qkv").map(|x| x.address), Some(0x100));
        assert_eq!(s.weight("layer.0.qkv.scales"), None);
        assert_eq!(s.weight("layer.0.o_proj"), None);
        assert_eq!(s.missed(), ["layer.0.qkv.scales", "layer.0.o_proj"]);
    }
}
