//! Resolving the names a trace states against a loaded checkpoint.
//!
//! [`crate::lowering::executor::Resolver`] is a trait with two questions — where does this
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

use crate::lowering::executor::{Resolver, Slice};

/// The scales sidecar, which no checkpoint spells two ways.
static SCALES: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| ".scales".to_string());
use model_ir::trace::ValueId;

/// How a checkpoint spells what a text names — [`model`]'s map, re-exported.
///
/// **The map itself is no longer here.** It is
/// [`model::shared::weight_names::Names`], beside the HuggingFace map that
/// module already owned, because both are translations between two spellings
/// `model` authors: the DSL invents `layer.3.qkv`, a contract author writes
/// `layers.3.self_attn.qkv_proj.fused.weight`, and neither is this crate's to
/// choose. While the table lived here, this backend knew what a gemma-4
/// per-layer projection, a gpt-oss attention sink and a qwen3-moe expert bank
/// are each called.
///
/// The alias stays because [`Store`] is built from one and the resolver is
/// this crate's, so the two names meet here whichever side owns the strings.
pub use model::shared::weight_names::Names;

/// What a traced name decomposes into.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Traced<'a> {
    layer: Option<u32>,
    role: &'a str,
    /// `.scales`, `.zeros`, `.bias`, or empty for the packed tensor itself.
    sidecar: &'a str,
}

/// Split `layer.3.qkv.scales` into its three parts.
///
/// Returns `None` for a name that is not in the text's shape at all, which is
/// drift rather than a spelling this map has not learned.
fn decompose(name: &str) -> Option<Traced<'_>> {
    // `.bias` is a THIRD sidecar and not a spelling of `.zeros`: a routed
    // expert bank publishes one additive value per output row beside the
    // codec's per-group plane, and `qmv_routed_bias` reads them at two
    // different buffers. The text spells the roles that carry a bias into
    // their own name (`q_bias`, `router_bias`) with an underscore, so no
    // existing name ends in this suffix and nothing is reclassified by
    // adding it.
    let (rest, sidecar) = match name.rfind('.') {
        Some(at) if matches!(&name[at..], ".scales" | ".zeros" | ".bias") => {
            (&name[..at], &name[at..])
        }
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
    fire: Option<&'a dyn Fn(crate::lowering::executor::FireTable) -> Option<Slice>>,
    /// The pool's geometry, when this store has a pool behind it.
    pool: Option<crate::layout::kv::Shape>,
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
        tables: &'a dyn Fn(crate::lowering::executor::FireTable) -> Option<Slice>,
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
    pub fn with_pool(mut self, shape: crate::layout::kv::Shape) -> Self {
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
            // A role whose name ends in `_bias` asks its module for the bias
            // tensor rather than the weight. The role table holds MODULES --
            // `q_bias` and `q_proj` both name `self_attn.q_proj` -- so the
            // role is the only thing that distinguishes the two tensors, and
            // the underscore is the only thing that distinguishes this role
            // from the `.bias` SIDECAR two arms down.
            "" if t.role.ends_with("_bias") => &self.names.bias_suffix,
            "" => &self.names.weight_suffix,
            ".scales" => std::slice::from_ref(&*SCALES),
            ".bias" => &self.names.bias_suffix,
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

    fn fire(&mut self, table: crate::lowering::executor::FireTable) -> Option<Slice> {
        self.fire.and_then(|tables| tables(table))
    }

    fn pool(&mut self, which: crate::lowering::executor::FireTable) -> Option<u32> {
        use crate::lowering::executor::FireTable;
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

    /// The router has TWO module spellings and this map offers both.
    ///
    /// MEASURED on `mlx-community/gemma-4-26b-a4b-it-4bit`, where the whole
    /// gap between the text and the load plan was this one role: ninety
    /// unpublished names, which is thirty layers times the packed weight,
    /// its scales and its zero point, and nothing else. `experts.switch_glu.*`
    /// already resolved -- the refusal that stood in front of this row said
    /// the contract published none of them, and the contract published all
    /// of them.
    ///
    /// The two are not nested. gpt-oss replaces its dense MLP with the routed
    /// block, so its router hangs under `mlp.`; gemma-4 keeps both blocks and
    /// its router is a sibling of the pair with a `proj` inside it. A rule
    /// that stripped or appended `mlp.` would get one of them wrong.
    #[test]
    fn the_router_answers_to_both_spellings_a_checkpoint_gives_it() {
        let (t, n) = (HashMap::new(), HashMap::new());
        let s = store(&t, &n);
        for (traced, want) in [
            ("layer.0.router", "layers.0.router.proj.weight"),
            ("layer.0.router.scales", "layers.0.router.proj.scales"),
            ("layer.0.router.zeros", "layers.0.router.proj.biases"),
        ] {
            assert!(
                s.checkpoint_names(traced).iter().any(|c| c == want),
                "`{traced}` offers {:?}, which does not include gemma-4's \
                 `{want}`",
                s.checkpoint_names(traced)
            );
        }
        assert!(
            s.checkpoint_names("layer.0.router")
                .iter()
                .any(|c| c == "layers.0.mlp.router.weight"),
            "gpt-oss's spelling must survive gemma-4's being added"
        );
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

    /// The additive bias and the zero point are ONE CHARACTER apart and are
    /// not the same tensor.
    ///
    /// This map used to answer both from `zero_point_suffix`, which listed
    /// `.biases` and `.bias` as "the same role one character apart" for the
    /// MXFP4 expert banks. Measured on `mlx-community/gpt-oss-20b-MXFP4-Q4`:
    /// an expert bank publishes `weight`, `scales` and `bias` and NO
    /// `biases`, and the bias is `[32, 2880]` — one value per output row,
    /// where the zero point beside `scales` would be `[32, 2880, 90]`, one
    /// per group. `qmv_routed_bias` reads them at two different buffers.
    ///
    /// It went uncontradicted because it was unreachable: `.zeros` is asked
    /// only of an affine weight, and `.biases` answers every one of those
    /// first. The `.bias` entry could only ever have fired for an affine
    /// checkpoint that spelled its zero point the other way, and none does.
    #[test]
    fn the_expert_banks_bias_is_not_its_zero_point() {
        let (t, n) = (HashMap::new(), HashMap::new());
        let s = store(&t, &n);
        // Three conventions answer `expert_gate` — qwen3-moe's `switch_mlp`,
        // gemma4's `switch_glu`, gpt-oss's plain `experts` — and the store
        // tries them in order, so the question is what the LIST holds.
        for sidecar in ["", ".scales", ".bias"] {
            assert!(
                s.checkpoint_names(&format!("layer.0.expert_gate{sidecar}"))
                    .contains(&format!(
                        "layers.0.mlp.experts.gate_proj{}",
                        match sidecar {
                            "" => ".weight",
                            other => other,
                        }
                    )),
                "`{sidecar}` does not reach gpt-oss's spelling"
            );
        }
        // And an affine weight's zero point still gets the plural, which is
        // the half that would break if `.bias` were simply prepended to the
        // list instead of being given its own sidecar.
        assert_eq!(
            s.checkpoint_name("layer.0.o_proj.zeros").as_deref(),
            Some("layers.0.self_attn.o_proj.biases")
        );
    }

    /// A role whose NAME ends in `_bias` is not a sidecar.
    ///
    /// `q_bias`, `k_bias`, `v_bias` and `router_bias` are whole ROLES — the
    /// checkpoint hangs them off a different module, not off the weight they
    /// bias — and they are spelled with an underscore. Adding `.bias` to the
    /// sidecar list must not reclassify them, and the separator is the only
    /// thing that tells the two apart.
    ///
    /// This asks `decompose` rather than the map, because the claim is about
    /// the SPLIT alone. What the three roles then RESOLVE to is
    /// `the_qwen_2_projection_biases_resolve_to_the_tensors_the_checkpoint_
    /// publishes` below.
    #[test]
    fn a_role_that_ends_in_bias_is_a_role_and_not_a_sidecar() {
        let underscored = decompose("layer.7.q_bias").expect("a layer-scoped name");
        assert_eq!(underscored.role, "q_bias");
        assert_eq!(underscored.sidecar, "");

        let dotted = decompose("layer.7.expert_gate.bias").expect("a layer-scoped name");
        assert_eq!(dotted.role, "expert_gate");
        assert_eq!(dotted.sidecar, ".bias");
    }

    /// The Qwen-2 family's three projection biases reach the tensors an MLX
    /// checkpoint actually publishes.
    ///
    /// The strings were in `weight_names.rs` all along — in the CUDA
    /// `Wiring`, which builds its aliases eagerly and never consulted
    /// `Names::mlx()`. The role MAP, which is what this driver reads, did not
    /// have them, and nothing noticed for as long as no Metal text could
    /// state a bias: `LlamaLikeMetalFacts::add_bias` was defaulted off
    /// because `lowering::dispatch` had no `Source::OutWidth` arm, so the
    /// text never asked and the missing role never answered.
    ///
    /// Three absences in a row, each of which made the next invisible. This
    /// is the last of them, and it is the one that would have surfaced as a
    /// loud `UnknownWeight("layer.0.q_bias")` rather than as seven models
    /// quietly computing their projections without a bias — which is what
    /// they did. `driver-vulkan`'s numpy oracle put numbers on the
    /// difference: `[88204, 6100, 41777, 2930]` against `[5937, 1560, 16925,
    /// 43715]`, entirely different tokens.
    #[test]
    fn the_qwen_2_projection_biases_resolve_to_the_tensors_the_checkpoint_publishes() {
        let (t, n) = (HashMap::new(), HashMap::new());
        let s = store(&t, &n);
        for (role, tensor) in [
            ("q_bias", "layers.3.self_attn.q_proj.bias"),
            ("k_bias", "layers.3.self_attn.k_proj.bias"),
            ("v_bias", "layers.3.self_attn.v_proj.bias"),
        ] {
            assert_eq!(
                s.checkpoint_name(&format!("layer.3.{role}")).as_deref(),
                Some(tensor),
                "`{role}` does not reach the tensor Qwen-2.5 ships"
            );
        }
        // The bias is a whole tensor and not a sidecar of the projection, so
        // asking the projection for one must still miss — otherwise the two
        // spellings would collide and either could answer.
        assert_ne!(
            s.checkpoint_name("layer.3.q_proj").as_deref(),
            Some("layers.3.self_attn.q_proj.bias")
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
        tensors.insert(
            "layers.0.self_attn.qkv_proj.fused.weight".to_string(),
            Slice {
                address: 0x100,
                bytes: 64,
            },
        );
        let n = HashMap::new();
        let mut s = store(&tensors, &n);
        assert_eq!(s.weight("layer.0.qkv").map(|x| x.address), Some(0x100));
        assert_eq!(s.weight("layer.0.qkv.scales"), None);
        assert_eq!(s.weight("layer.0.o_proj"), None);
        assert_eq!(s.missed(), ["layer.0.qkv.scales", "layer.0.o_proj"]);
    }
}
