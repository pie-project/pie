//! `Deployment` — everything a driver needs to serve a checkpoint, with
//! no family name in it.
//!
//! # Why this is a type and not a trait
//!
//! The drivers used to ask a `Box<dyn PlannedFamily>` thirteen
//! questions, and its own doc comments named the exception in the
//! method: `pins_attention_values()` said *"Only gemma-4 does"*, and
//! `decode_plan_head_dims()` existed because gemma-4's two layer kinds
//! disagree. Then the callers undid the abstraction to get the name
//! back — `let is_gemma4 = family.planless_prefill();` appears twice in
//! the CUDA shell and once in its transfer path.
//!
//! Wrapping a family name in a virtual predicate and then recovering the
//! name at the call site means **the axis was the family all along**.
//!
//! It also cost at run time. `facts_from_hf` was called from the
//! admission of EVERY fire, allocating a box and cloning per-layer
//! `Vec`s — while the lowering it feeds is cached precisely because it
//! costs 3.3 ms. The expensive answer was cached; its input was
//! rederived.
//!
//! # What changes about gemma-4
//!
//! Its two head dims stop being an exception and become
//! `attention[l].head_dim` differing between layers, which is what they
//! are. A `Vec` of per-layer facts has no opinion about which family
//! produced it.
//!
//! # Why it lives HERE
//!
//! `crates/model/tests/one_normalizer.rs` states the rule: *"what a
//! driver reads is the answer, never the question."* The drivers obeyed
//! the letter — they read `pie.model/1`, not `config.json` — while the
//! answer was still SHAPED like the question and they still switched on
//! it: 33 `FACTS_ROWS` rows and 11 derivations in the CUDA shell alone,
//! against the 25 `model_type` conditionals of the C++ normalizer that
//! test was written to hunt.
//!
//! A `Deployment` with no family name in it is a type a driver
//! **cannot** branch on. That is the difference between a guard that
//! must be remembered and one that cannot be routed around.

use std::collections::BTreeMap;

/// How a layer attends.
///
/// PER LAYER, unconditionally — not "per layer for the families that
/// need it". A stack whose layers agree fills this with equal entries,
/// which costs a `Vec` and buys the absence of a special case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerAttention {
    /// The kernel-facing head dim for this layer.
    pub head_dim: u32,
    /// Sliding window, or `-1` for a layer that attends the whole
    /// context.
    pub window: i32,
    /// Which layer's KV pages this layer reads.
    ///
    /// Its own index for the ordinary case. Gemma-4's trailing layers
    /// name an earlier one and own no pages themselves — which is a
    /// fact about a LAYER, and was a fact about a family.
    pub kv_source: u32,
    /// The attention scale. Usually `1/sqrt(head_dim)`; gemma-4 runs
    /// `1.0` because its q/k norms carry the scaling.
    pub sm_scale: f32,
    /// Rope base for this layer. The stacks that use one theta
    /// throughout repeat it.
    pub rope_theta: f32,
    /// Rotary width, or `0` for full rotation at the head dim.
    pub rotary_dim: u32,
}

/// What kind of KV this deployment needs.
///
/// AN ENUM, so a shape the driver has no pool for is an
/// `unimplemented!` ARM rather than a row in a registry that loads
/// successfully and dies at its first fire. That was a real defect:
/// the MLA lineage registered in `FACTS_ROWS`, answered `facts_from_hf`
/// happily, and had no forward path at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvStyle {
    /// Ordinary paged K and V.
    Paged,
    /// Multi-head latent attention: a compressed KV plane and a
    /// positional one, which do not fit the standard k/v pair.
    Mla {
        /// The compressed KV rank.
        kv_lora_rank: u32,
        /// The rope head dim carried beside it.
        qk_rope_head_dim: u32,
    },
    /// DeepSeek-V4's per-layer compression ratios.
    Dsv4 {
        /// One ratio per layer; `None` for an uncompressed layer.
        ratios: Vec<i32>,
    },
}

/// A recurrent stack's slab geometry — what a driver must allocate and
/// stride before it can run one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentShape {
    /// Which layers are linear-attention layers.
    pub linear_layers: Vec<u32>,
    /// Bytes per conv slot.
    pub conv_stride: usize,
    /// Bytes per recurrent-state slot.
    pub state_stride: usize,
    /// Element width of the recurrent state.
    pub state_elem: usize,
    /// Key heads.
    pub k_h: i32,
    /// Value heads.
    pub v_h: i32,
    /// Key head dim.
    pub k_d: i32,
    /// Value head dim.
    pub v_d: i32,
    /// Conv channel count.
    pub conv_dim: i32,
    /// Conv kernel width.
    pub conv_k: i32,
}

/// Whether the prefill path can be planned ahead of the fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillStyle {
    /// The ordinary case: a plan is raised before the fire and bound.
    Planned,
    /// The plan is built inside the fire from the host CSR mirrors, so
    /// there is nothing to raise. Gemma-4's 512-wide layers take the
    /// naive kernel and plan internally.
    Planless,
}

/// Where a layer's norm sits relative to its projections.
///
/// It is on the deployment rather than inside a family's facts because
/// a DRIVER needs it: an adapter's staging reads the projection input,
/// and which buffer that is depends on this. `Pre` ships one input
/// norm; `Post` (olmo2) ships `post_attention` and `post_feedforward`
/// instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormPlacement {
    /// The norm precedes the projections; the input is the normed value.
    Pre,
    /// The norm follows them; the input is the residual stream.
    Post,
}

/// Where the attention output lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnOutput {
    /// The guard region records no SSA output, so the driver owns the
    /// landing buffer and pins the query.
    DriverPinned,
    /// The forward states `[q, o]` as SSA args, so there is nothing to
    /// pin.
    StatedArgs,
}

/// Everything a driver needs to serve a checkpoint.
///
/// `Clone`, `Debug`, comparable and derivable once at load — not per
/// fire. Nothing in it is a string naming a family.
#[derive(Debug, Clone, PartialEq)]
pub struct Deployment {
    /// How many layers.
    pub layers: u32,
    /// Per-layer attention facts.
    pub attention: Vec<LayerAttention>,
    /// What kind of KV to provision.
    pub kv: KvStyle,
    /// The recurrent slabs, for a hybrid stack.
    pub recurrent: Option<RecurrentShape>,
    /// Whether prefill can be planned ahead.
    pub prefill: PrefillStyle,
    /// Where attention output lands.
    pub attn_output: AttnOutput,
    /// Final-logit softcap, `0.0` for none.
    pub logit_softcap: f32,
    /// Per-layer-embedding width, `0` for a stack without one.
    pub ple_dim: i32,
    /// Where the norm sits — read by anything that needs to name the
    /// projection input, which today is the adapter staging.
    pub norm: NormPlacement,
    /// Named scalar constants the forward refers to by name.
    pub scales: BTreeMap<String, f32>,
}

impl Deployment {
    /// A placeholder for a driver that must build its model value before
    /// it can derive one.
    ///
    /// Zero layers, which is a shape no fire can take — so a driver that
    /// forgot to fill it in refuses at its first admission rather than
    /// serving a stack it never derived.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            layers: 0,
            attention: Vec::new(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::StatedArgs,
            logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            scales: BTreeMap::new(),
        }
    }

    /// The distinct decode head dims this stack needs plans for.
    ///
    /// `None` when every layer agrees, which is the ordinary case.
    /// `Some((a, b))` when two kinds disagree — which used to be
    /// `decode_plan_head_dims()`, a method that existed because
    /// gemma-4 has two, and is now a question about the `Vec`.
    #[must_use]
    pub fn decode_head_dims(&self) -> Option<(u32, u32)> {
        let first = self.attention.first()?.head_dim;
        let other = self.attention.iter().find(|a| a.head_dim != first)?.head_dim;
        Some((first, other))
    }

    /// Does any layer read another's KV pages?
    #[must_use]
    pub fn shares_kv(&self) -> bool {
        self.attention.iter().enumerate().any(|(l, a)| a.kv_source as usize != l)
    }

    /// The sliding window per layer, as the fire path binds it.
    #[must_use]
    pub fn windows(&self) -> Vec<i32> {
        self.attention.iter().map(|a| a.window).collect()
    }

    /// Rope base per layer, or empty when one theta serves the stack.
    ///
    /// EMPTY RATHER THAN REPEATED, because the binder's fast path
    /// checks emptiness — a table of identical values is a table it
    /// would walk for nothing.
    #[must_use]
    pub fn theta_by_layer(&self) -> Vec<f32> {
        let first = self.attention.first().map_or(0.0, |a| a.rope_theta);
        if self.attention.iter().all(|a| a.rope_theta == first) {
            return Vec::new();
        }
        self.attention.iter().map(|a| a.rope_theta).collect()
    }

    /// Rotary width per layer, or empty when every layer rotates fully.
    #[must_use]
    pub fn rotary_by_layer(&self) -> Vec<u32> {
        if self.attention.iter().all(|a| a.rotary_dim == 0) {
            return Vec::new();
        }
        self.attention.iter().map(|a| a.rotary_dim).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(head_dim: u32) -> LayerAttention {
        LayerAttention {
            head_dim,
            window: -1,
            kv_source: 0,
            sm_scale: 1.0,
            rope_theta: 10_000.0,
            rotary_dim: 0,
        }
    }

    fn stack(dims: &[u32]) -> Deployment {
        Deployment {
            layers: dims.len() as u32,
            attention: dims
                .iter()
                .enumerate()
                .map(|(l, &d)| LayerAttention { kv_source: l as u32, ..layer(d) })
                .collect(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::DriverPinned,
            logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            scales: BTreeMap::new(),
        }
    }

    /// The exception that stopped being one. `decode_plan_head_dims()`
    /// existed as a vtable method because gemma-4 has two layer kinds;
    /// it is a question about a `Vec` now, and a uniform stack answers
    /// `None` without anyone having to know which family it is.
    #[test]
    fn two_head_dims_is_a_property_of_the_layers_not_of_a_family() {
        assert_eq!(stack(&[128, 128, 128]).decode_head_dims(), None);
        assert_eq!(stack(&[128, 128, 256]).decode_head_dims(), Some((128, 256)));
    }

    /// Likewise KV sharing: gemma-4's trailing layers read an earlier
    /// layer's pages, and that is a fact about a LAYER.
    #[test]
    fn kv_sharing_is_a_property_of_the_layers() {
        assert!(!stack(&[128, 128]).shares_kv());
        let mut shared = stack(&[128, 128]);
        shared.attention[1].kv_source = 0;
        assert!(shared.shares_kv());
    }

    /// A uniform table is EMPTY rather than repeated, because the
    /// binder's fast path checks emptiness and a table of identical
    /// values is one it would walk for nothing.
    #[test]
    fn a_uniform_table_is_empty_rather_than_repeated() {
        assert!(stack(&[128, 128]).theta_by_layer().is_empty());
        let mut mixed = stack(&[128, 128]);
        mixed.attention[1].rope_theta = 1_000_000.0;
        assert_eq!(mixed.theta_by_layer().len(), 2);
    }

    /// The MLA orphan, as a type. It used to be a row in `FACTS_ROWS`
    /// that loaded successfully and died at its first fire; a driver
    /// matching on `KvStyle` has to write the arm or refuse.
    #[test]
    fn an_unservable_kv_shape_is_a_variant_rather_than_a_registry_row() {
        let mut d = stack(&[128]);
        d.kv = KvStyle::Mla { kv_lora_rank: 512, qk_rope_head_dim: 64 };
        assert!(matches!(d.kv, KvStyle::Mla { .. }));
    }
}

/// Why a checkpoint cannot be served.
///
/// An ENUM rather than an ABI status, because this crate has no ABI. A
/// driver maps it to whatever its own boundary speaks — which is the
/// point of §4: the derivation used to return `PIE_STATUS_UNSUPPORTED`,
/// the engine's vocabulary, from a crate that has no engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// No row for this checkpoint's `model_type`.
    Unsupported,
    /// A row exists and the checkpoint contradicts it.
    Malformed(&'static str),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "no deployment derivation for this model type"),
            Self::Malformed(why) => write!(f, "the checkpoint contradicts its own type: {why}"),
        }
    }
}

impl std::error::Error for Refusal {}
