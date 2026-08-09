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

/// The scales sidecar, which no checkpoint spells two ways.
static SCALES: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| ".scales".to_string());
use model_compiler::trace::ValueId;

/// How a checkpoint spells what a text names.
///
/// Data rather than code: a family that spells its tensors differently is a
/// different spelling in this map, not a different resolver.
///
/// # Why a role has SEVERAL spellings
///
/// One role, one name was the earlier shape, and it forced a second map
/// (`Names::mlx_gemma4`) the moment a second convention appeared — and then
/// the driver had to CHOOSE which map, which is the driver choosing, which is
/// the one thing this crate may not do.
///
/// So a role names every spelling it has ever been seen under, and the
/// CHECKPOINT decides: [`Store::checkpoint_name`] takes the first candidate
/// the loaded tensor map actually publishes. Adding a convention is adding a
/// string, and no caller learns anything.
///
/// Three real disagreements this covers, all measured against checkpoints on
/// disk rather than guessed:
///
///   - `embed`/`lm_head`: `shared_embedding` when the deployment TIES them,
///     `embed_tokens` and `lm_head` when it does not (gpt-oss).
///   - the expert bank: `mlp.switch_mlp.*` (qwen3-moe),
///     `experts.switch_glu.*` (gemma4), `mlp.experts.*` (gpt-oss).
///   - the router: `mlp.gate` (qwen3-moe), `mlp.router` (gpt-oss).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
    /// What a layer-scoped name gets in front of its index —
    /// `model.layers.` for a HuggingFace export.
    pub layer_prefix: String,
    /// The text's role name → the checkpoint's paths within a layer, in the
    /// order they are tried. `qkv` → `self_attn.qkv_proj`.
    pub roles: HashMap<String, Vec<String>>,
    /// Names with no layer at all — `embed`, `lm_head`, `final_norm`.
    pub globals: HashMap<String, Vec<String>>,
    /// What the checkpoint calls a packed weight's own tensor.
    ///
    /// `.weight` for a tensor that hangs under a module, and **the empty
    /// string** for one that IS the value: gpt-oss ships `self_attn.sinks`,
    /// not `self_attn.sinks.weight`, because a sink is a vector and not a
    /// linear layer.
    pub weight_suffix: Vec<String>,
    /// What it calls the zero point the text spells `.zeros`.
    ///
    /// `.biases` for MLX's affine quantisation and `.bias` for the MXFP4
    /// expert banks, which spell the same role one character apart.
    pub zero_point_suffix: Vec<String>,
}

impl Names {
    /// gemma4's expert bank, which is now [`Names::mlx`] and nothing else.
    ///
    /// Kept as a name so callers need not all change at once. It USED to be a
    /// second map, because gemma4 ships `layers.N.experts.switch_glu.*` where
    /// qwen3-moe ships `layers.N.mlp.switch_mlp.*` -- and that made the driver
    /// pick a map per checkpoint, which is the driver choosing.
    ///
    /// Both spellings are candidates of the one map now, so there is nothing
    /// left to pick.
    #[deprecated(note = "`Names::mlx` carries gemma4's spellings as candidates")]
    #[must_use]
    pub fn mlx_gemma4() -> Self {
        Self::mlx()
    }

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
            // The mixture. `mlp.gate` is MLX's name for the ROUTER -- an
            // unfortunate collision with `mlp.gate_proj`, which is an
            // expert's gate half, and worth spelling out here because the two
            // are one character apart and mean entirely different tensors.
            //
            // The expert banks carry no expert index: `switch_mlp` stores all
            // of them in one `[experts, out, in]` tensor and the routed kernel
            // indexes it by the slot it read.
            // Three conventions for one bank, and the checkpoint picks:
            // qwen3-moe's `switch_mlp`, gemma4's `switch_glu`, gpt-oss's
            // plain `experts`.
            ("router", "mlp.gate|mlp.router"),
            (
                "expert_gate",
                "mlp.switch_mlp.gate_proj|experts.switch_glu.gate_proj|mlp.experts.gate_proj",
            ),
            (
                "expert_up",
                "mlp.switch_mlp.up_proj|experts.switch_glu.up_proj|mlp.experts.up_proj",
            ),
            (
                "expert_down",
                "mlp.switch_mlp.down_proj|experts.switch_glu.down_proj|mlp.experts.down_proj",
            ),
            ("shared_gate", "mlp.shared_expert.gate_proj"),
            ("shared_up", "mlp.shared_expert.up_proj"),
            ("shared_down", "mlp.shared_expert.down_proj"),
            ("shared_gate_proj", "mlp.shared_expert_gate"),
            // The norms.
            // gemma's per-layer embedding network: a second table, its
            // projection and norm, and the per-layer gate and output.
            ("ple_gate", "per_layer_gate"),
            ("ple_out", "per_layer_projection"),
            // `layer_scalar`, measured against
            // `mlx-community/gemma-4-26b-a4b-it-4bit`'s index and NOT the
            // `per_layer_scalar` the role is called. A role name and a
            // checkpoint name are two different things, which is what this map
            // is for.
            ("scalar", "layer_scalar"),
            // The attention sink, one learned logit per head.
            ("attn_sinks", "self_attn.sinks"),
            ("q_norm", "self_attn.q_norm"),
            ("k_norm", "self_attn.k_norm"),
            ("attn_norm", "input_layernorm"),
            // TWO spellings, and which one a checkpoint means is decided by
            // which one it ships. Under `NormPlacement::Pre` the pre-FFN norm
            // IS `post_attention_layernorm` — it sits after the attention and
            // before the MLP, and llama publishes nothing else. gemma splits
            // that position in two (`post_attention_layernorm` norms the
            // attention's OUTPUT, `pre_feedforward_layernorm` norms the MLP's
            // input), so it must take the second.
            //
            // Ordered gemma-first because the alternative resolves to the
            // first spelling the checkpoint HAS: a llama checkpoint ships no
            // `pre_feedforward_layernorm` and falls through, and a gemma one
            // would otherwise bind its attention-output norm as the MLP's
            // input norm and drop two norms entirely.
            (
                "mlp_norm",
                "pre_feedforward_layernorm|post_attention_layernorm",
            ),
            // The SANDWICH's output norms, which only a gemma text names.
            ("post_attn_norm", "post_attention_layernorm"),
            ("post_mlp_norm", "post_feedforward_layernorm"),
        ]
        .into_iter()
        .map(|(a, b): (&str, &str)| {
            (a.to_string(), b.split('|').map(str::to_string).collect())
        })
        .collect();
        let globals = [
            // Tied: one table serves both ends, which is why the readout and
            // the embedding answer to the same name.
            // Tied deployments publish ONE table under `shared_embedding`;
            // untied ones (gpt-oss) publish `embed_tokens` and `lm_head`
            // separately. Which is which is the checkpoint's to say.
            ("embed", "shared_embedding|embed_tokens"),
            // gemma's SECOND embedding table and its projection: layer-less,
            // gathered once per step, so they are globals rather than a
            // layer's.
            ("ple_embed", "per_layer_embedding"),
            ("ple_proj", "per_layer_input_projection"),
            ("ple_proj_norm", "per_layer_input_norm"),
            ("lm_head", "shared_embedding|lm_head"),
            ("final_norm", "final_norm"),
        ]
        .into_iter()
        .map(|(a, b): (&str, &str)| {
            (a.to_string(), b.split('|').map(str::to_string).collect())
        })
        .collect();
        Self {
            layer_prefix: "layers.".to_string(),
            roles,
            globals,
            // `.weight` first, then the bare name: a role whose tensor IS
            // the value (`self_attn.sinks`) hangs under no module.
            weight_suffix: vec![".weight".to_string(), String::new()],
            // The text says `.zeros`, the checkpoint says `.biases` -- or
            // `.bias`, for the MXFP4 expert banks. Both are right on their own
            // side; the map is where they meet.
            zero_point_suffix: vec![".biases".to_string(), ".bias".to_string()],
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
    /// The layer KV pages a statement's state reference resolves through.
    ///
    /// `None` for a store with no pool — the host checks, the name-map tests —
    /// which is why [`Resolver::kv`] defaults to `None` rather than being
    /// required.
    kv: Option<&'a dyn Fn(u16, bool) -> Option<Slice>>,
    /// The fire's own tables, when this store has a fire behind it.
    fire: Option<&'a dyn Fn(super::executor::FireTable) -> Option<Slice>>,
    /// The pool's geometry, when this store has a pool behind it.
    pool: Option<super::kv::Shape>,
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
            kv: None,
            fire: None,
            pool: None,
            missed: Vec::new(),
        }
    }

    /// The same store, answering a statement's KV state through `pages`.
    ///
    /// A closure rather than a borrowed pool, so this module stays portable:
    /// the pool is Apple-only and the map is not, and a resolver that named
    /// the pool's type would drag one into the other.
    #[must_use]
    pub fn with_kv(mut self, pages: &'a dyn Fn(u16, bool) -> Option<Slice>) -> Self {
        self.kv = Some(pages);
        self
    }

    /// The same store, answering the FIRE's tables through `tables`.
    #[must_use]
    pub fn with_fire(
        mut self,
        tables: &'a dyn Fn(super::executor::FireTable) -> Option<Slice>,
    ) -> Self {
        self.fire = Some(tables);
        self
    }

    /// The same store, answering the POOL's geometry from `shape`.
    ///
    /// Separate from [`Self::with_kv`], which hands out the pages themselves,
    /// because the two answer different questions: where a layer's keys are,
    /// and how far apart the rows in them sit.
    #[must_use]
    pub fn with_pool(mut self, shape: super::kv::Shape) -> Self {
        self.pool = Some(shape);
        self
    }

    /// The checkpoint tensor a traced name means, spelled the checkpoint's way.
    ///
    /// `None` when the name is not in the text's shape — which is drift, not a
    /// gap in this map.
    #[must_use]
    pub fn checkpoint_name(&self, traced: &str) -> Option<String> {
        let mut candidates = self.checkpoint_names(traced).into_iter().peekable();
        let first = candidates.peek()?.clone();
        // The one the checkpoint actually has. A store with no tensors -- the
        // name-map tests, and the load-time gate that asks a plan rather than
        // a staged map -- has nothing to choose with, so it gets the first
        // candidate and a caller that wants them all asks for them all.
        Some(
            candidates
                .find(|c| self.tensors.contains_key(c))
                .unwrap_or(first),
        )
    }

    /// Every spelling `traced` could have, in the order they are tried.
    ///
    /// The cross product of the role's candidate paths and the sidecar's
    /// candidate suffixes. Published because the LOAD-TIME gate needs it: it
    /// holds the text's names against a load plan rather than a staged tensor
    /// map, so it has no `tensors` to choose with and must ask whether ANY
    /// spelling is one the plan publishes.
    ///
    /// Empty for a name that is not in the text's shape at all -- which is
    /// drift, not a gap in this map.
    #[must_use]
    pub fn checkpoint_names(&self, traced: &str) -> Vec<String> {
        let Some(t) = decompose(traced) else {
            return Vec::new();
        };
        let bases: Vec<String> = match t.layer {
            Some(l) => {
                let Some(roles) = self.names.roles.get(t.role) else {
                    return Vec::new();
                };
                roles
                    .iter()
                    .map(|role| format!("{}{l}.{role}", self.names.layer_prefix))
                    .collect()
            }
            None => match self.names.globals.get(t.role) {
                Some(g) => g.clone(),
                None => return Vec::new(),
            },
        };
        let suffixes: &[String] = match t.sidecar {
            "" => &self.names.weight_suffix,
            ".scales" => std::slice::from_ref(&*SCALES),
            _ => &self.names.zero_point_suffix,
        };
        bases
            .iter()
            .flat_map(|b| suffixes.iter().map(move |s| format!("{b}{s}")))
            .collect()
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
            .checkpoint_names(name)
            .into_iter()
            .find_map(|spelled| self.tensors.get(&spelled).copied());
        if found.is_none() {
            self.missed.push(name.to_string());
        }
        found
    }

    fn named(&mut self, value: ValueId) -> Option<Slice> {
        self.named.get(&value).copied()
    }

    fn kv(&mut self, layer: u16, values: bool) -> Option<Slice> {
        self.kv.and_then(|pages| pages(layer, values))
    }

    fn fire(&mut self, table: super::executor::FireTable) -> Option<Slice> {
        self.fire.and_then(|tables| tables(table))
    }

    fn pool(&mut self, which: super::executor::FireTable) -> Option<u32> {
        use super::executor::FireTable;
        let shape = self.pool?;
        // The pool is `[page, token, head, dim]` -- `row_bytes` is "every
        // head's channels, contiguously", so a token row INTERLEAVES the heads
        // rather than a head owning a contiguous span of tokens. The strides
        // therefore come out the other way around from the head-major layout
        // the names suggest: one head is `head_dim` away, one token is a whole
        // row away.
        //
        // Worth stating rather than deriving at the call site. Swapping these
        // two is a fire that reads real memory at every step and attends to
        // the wrong tokens, which no bounds check catches.
        Some(match which {
            FireTable::KvHeadStride => shape.head_dim,
            FireTable::KvSeqStride => shape.kv_heads * shape.head_dim,
            FireTable::KvPageSize => shape.page_size,
            _ => return None,
        })
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
