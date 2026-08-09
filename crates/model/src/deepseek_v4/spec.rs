//! DeepSeek-V4's SHAPE: the numbers a checkpoint of this generation has.
//!
//! Ungated, for the reason `kimi_k2/spec.rs` states: a catalog row is
//! written in these words, and a row has to exist under every aspect —
//! `chat` asks which template speaks for it, `contract` asks how to
//! author it, `forward` asks what to trace. One struct, three readers,
//! and that cannot hold if the struct only exists when the tracer is
//! compiled in.
//!
//! Neither MLA nor a plain attention. Two things are this generation's
//! own and no other row in the catalog has either:
//!
//! * **Hyper-connections** — a rank-K residual. The stream is `hc_mult`
//!   copies wide, and each layer reads a MIX of them and writes a mix
//!   back, which is why `hc_expand` opens the body and `hc_head` closes
//!   it. gemma3n's AltUp is the other scheme of this kind; they are not
//!   the same and share no statement.
//!
//! * **Compressed attention** — the KV of distant tokens is COMPRESSED
//!   into per-block entries, and a fire attends both the sliding window
//!   (uncompressed) and the compressed history, then combines the two
//!   outputs by their LSEs. That is why `combine_attn_outputs` and
//!   `lse_log2_to_ln` are statements in this generation's text and
//!   nowhere else.
//!
//! The compression SCHEDULE ([`Dsv4Facts::ratios`]) is new here and is
//! not a transcription: the vtable this replaces answered the question
//! with `KvStyle::Dsv4 { ratios: Vec::new() }` — an empty list, on every
//! load, whatever the checkpoint said — and an empty list is what
//! `dsv4_geometry::compress_bytes_per_token` reads as "this model needs
//! no compressor cache". The planner then sized a V4's KV pool without
//! the three tensors per compressing layer that have to survive across
//! fires.

use serde::{Deserialize, Serialize};

/// The attention block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dsv4AttnFacts {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    /// The query's latent (`q_lora_rank`); the KV has none — this family
    /// projects KV straight and compresses it instead.
    pub q_lora_rank: u32,
    pub qk_rope_head_dim: u32,
    /// `dsv4_sliding_window`: how far back the UNCOMPRESSED attention
    /// reaches. Everything older is served by the compressed pass.
    pub sliding_window: u32,
    /// `dsv4_o_lora_rank` / `dsv4_o_groups`: the output projection is
    /// itself low-rank and grouped.
    pub o_lora_rank: u32,
    pub o_groups: u32,
}

impl Dsv4AttnFacts {
    /// Every head's slice of the query, side by side.
    ///
    /// `const` so a row can be written in it; the callers are unchanged
    /// by that, since a `const fn` is still an ordinary call.
    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.heads * self.head_dim
    }
}

/// The hyper-connection residual.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dsv4HcFacts {
    /// `dsv4_hc_mult`: how many residual streams. 1 would be an ordinary
    /// residual and this family never sets it.
    pub mult: u32,
}

/// The MoE block. `topk_sqrtsoftplus` scoring and a CLAMPED swiglu.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dsv4MoeFacts {
    pub num_experts: u32,
    pub top_k: u32,
    /// Whether the router renormalizes over the SELECTED experts.
    ///
    /// `deepseek-ai/DeepSeek-V3` publishes `"norm_topk_prob": false`, and
    /// its `routed_scaling_factor` of 2.5 is what pays for weights that
    /// sum to less than one. No V4 config is out; this is the lineage's
    /// value, stated so the day one is, this is the line that changes.
    pub norm_topk_prob: bool,
    /// `routed_scaling_factor`, 2.5 on `deepseek-ai/DeepSeek-V3`. It is
    /// what pays for weights that sum to less than one; the pair is only
    /// meaningful together.
    pub routed_scaling: f32,
    pub moe_intermediate: u32,
    /// `cfg.swiglu_limit`, the clamp the activation applies.
    pub swiglu_limit_milli: u32,
    /// The router is a HASH TABLE lookup rather than a learned gate on
    /// some deployments (`kernels::moe::hash_route_lookup`).
    pub hash_routed: bool,
}

/// The whole family.
///
/// `Deserialize` is deliberately absent where the other three structs
/// here derive it, and [`Self::ratios`] is why: a `const` row cannot
/// hold a `Vec`, so the schedule is a `&'static [i32]`, and there is no
/// owner for a deserializer to hand one back. Nothing loses anything —
/// no facts struct in this crate is ever built from JSON. They are
/// written down, which is the whole premise of a catalog.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Dsv4Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub dense_intermediate: u32,
    pub dense_layers: u32,
    /// The compression ratio of each layer, in layer order.
    ///
    /// `compress_ratios` in a V4 config. One entry per layer, `0` (or
    /// anything below it) for a layer that does not compress, and a list
    /// SHORTER than the stack leaves the trailing layers uncompressed —
    /// that last is not an accident of iteration but the C++'s explicit
    /// `li < ratios.size() ? ... : 0`, which
    /// `dsv4_geometry::compress_cache_bytes` documents as a supported
    /// input.
    ///
    /// It is a fact about the MODEL and not about a fire: which layers
    /// carry a compressor cache decides what a memory planner has to set
    /// aside per token before a single request arrives.
    pub ratios: &'static [i32],
    pub attn: Dsv4AttnFacts,
    pub hc: Dsv4HcFacts,
    pub moe: Dsv4MoeFacts,
}

impl Dsv4Facts {
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_compiler::facts::after_dense_prefix(self.dense_layers, l)
    }

    /// How many tokens this layer pools into one compressed entry.
    ///
    /// `0` for a layer that keeps every token, which is both what a
    /// short list means past its end and what a checkpoint states
    /// in-line for a layer it does not compress.
    #[must_use]
    pub fn compress_ratio_at(&self, l: u32) -> i32 {
        self.ratios.get(l as usize).copied().unwrap_or(0)
    }

    /// Does this layer run the second, compressed attention at all?
    ///
    /// The one question a driver's allocator asks: a compressing layer
    /// owns three per-token tensors that outlive a fire (`state_kv`,
    /// `state_score`, `comp_kv`) and a plain one owns none.
    #[must_use]
    pub fn compresses(&self, l: u32) -> bool {
        self.compress_ratio_at(l) > 0
    }

    pub fn dsv4_synthetic() -> Self {
        Dsv4Facts {
            layers: 6,
            vocab: 129280,
            hidden: 2048,
            dense_intermediate: 5632,
            dense_layers: 1,
            // The only `compress_ratios` written down anywhere in this
            // tree — `synthetic--deepseek-v4.json` states exactly this
            // list — and the fixture is six layers deep where that file
            // is four, so the last three layers do not compress. That is
            // the supported reading of a short list and not a rounding
            // of one: a published V4 config replaces the list and
            // nothing else here.
            ratios: &[1, 2, 4],
            attn: Dsv4AttnFacts {
                hidden: 2048,
                heads: 16,
                head_dim: 128,
                q_lora_rank: 768,
                qk_rope_head_dim: 64,
                sliding_window: 2048,
                o_lora_rank: 512,
                o_groups: 4,
            },
            hc: Dsv4HcFacts { mult: 4 },
            moe: Dsv4MoeFacts {
                num_experts: 64,
                top_k: 6,
                norm_topk_prob: false,
                routed_scaling: 2.5,
                moe_intermediate: 1024,
                swiglu_limit_milli: 7000,
                hash_routed: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f() -> Dsv4Facts {
        Dsv4Facts::dsv4_synthetic()
    }

    /// The hyper-connection is a rank-K residual, and K > 1 is what makes
    /// it one. A fixture at 1 would lower the same as an ordinary
    /// residual and prove nothing about the scheme.
    #[test]
    fn the_residual_is_actually_rank_k() {
        assert!(f().hc.mult > 1);
    }

    /// Every number a stack needs to exist is positive, and the query
    /// splits into whole heads. A zero here does not fail loudly: it
    /// divides into a workspace size and the driver sizes a buffer at
    /// nothing.
    #[test]
    fn the_fixture_states_a_stack_that_could_exist() {
        let f = f();
        for (what, n) in [
            ("layers", f.layers),
            ("vocab", f.vocab),
            ("hidden", f.hidden),
            ("dense intermediate", f.dense_intermediate),
            ("heads", f.attn.heads),
            ("head dim", f.attn.head_dim),
        ] {
            assert!(n > 0, "a stack with no {what} is not a stack");
        }
        assert_eq!(
            f.attn.hidden, f.hidden,
            "the attention block reads the residual, so a disagreement here is \
             a projection reading a width nothing writes",
        );
        assert!(
            f.attn.qk_rope_head_dim < f.attn.head_dim,
            "the rope is PARTIAL — it turns the last channels of a head and \
             leaves the rest — so a rope half as wide as the head is a full \
             rotation stated as a partial one",
        );
    }

    /// The query's width is every head side by side, which is the number
    /// `wq_b`, `wkv` and the output projection's input all read.
    #[test]
    fn the_query_is_every_head_side_by_side() {
        let f = f();
        assert_eq!(f.attn.q_width(), f.attn.heads * f.attn.head_dim);
        assert_eq!(f.attn.q_width(), 2048);
    }

    /// A dense prefix is a PREFIX: the leading layers are dense and
    /// every layer after them routes. Three families wrote this
    /// predicate and disagreed at the boundary, which is why it is
    /// `model_compiler::facts`'s and is asserted at the edge here.
    #[test]
    fn the_dense_prefix_is_a_prefix_and_everything_after_it_routes() {
        let f = f();
        assert!(!f.is_moe_layer(0), "layer 0 is inside a prefix of 1");
        assert!(f.is_moe_layer(1), "the layer AFTER the prefix routes");
        assert!(f.is_moe_layer(f.layers - 1));
    }

    /// The compression schedule is per layer, and it is what makes this
    /// generation a V4 rather than a wide-headed V3.
    #[test]
    fn the_schedule_says_which_layers_compress() {
        let f = f();
        assert_eq!(f.compress_ratio_at(0), 1);
        assert_eq!(f.compress_ratio_at(1), 2);
        assert_eq!(f.compress_ratio_at(2), 4);
        for l in 0..3 {
            assert!(f.compresses(l), "layer {l} states a ratio above zero");
        }
    }

    /// A list shorter than the stack leaves the tail uncompressed, which
    /// is the C++'s `li < ratios.size() ? ... : 0` and not an accident.
    /// Reading past the end as "compress at ratio 0" instead would ask a
    /// driver to pool zero tokens into an entry.
    #[test]
    fn a_short_schedule_leaves_the_trailing_layers_uncompressed() {
        let f = f();
        assert!(
            f.ratios.len() < f.layers as usize,
            "the fixture is the short case"
        );
        for l in f.ratios.len() as u32..f.layers {
            assert_eq!(f.compress_ratio_at(l), 0);
            assert!(
                !f.compresses(l),
                "layer {l} is past the end of the schedule"
            );
        }
    }

    /// A ratio at or below zero is a layer that does not compress —
    /// stated in-line rather than by running off the end, which is how a
    /// checkpoint spells "these two layers keep every token".
    #[test]
    fn a_zero_or_negative_ratio_is_a_layer_that_keeps_every_token() {
        let f = Dsv4Facts {
            ratios: &[0, -1, 4],
            ..Dsv4Facts::dsv4_synthetic()
        };
        assert!(!f.compresses(0), "zero is not a ratio");
        assert!(
            !f.compresses(1),
            "a negative ratio is the C++'s `ratio <= 0`"
        );
        assert!(f.compresses(2));
    }

    /// A mixture states all of its numbers or none of them. A router
    /// over zero experts, or experts with no width, is a block that
    /// cannot run — and it fails inside a GEMM rather than here.
    #[test]
    fn a_mixture_states_all_of_its_numbers_or_none() {
        let m = f().moe;
        let stated = [m.num_experts, m.top_k, m.moe_intermediate];
        assert!(
            stated.iter().all(|n| *n > 0) || stated.iter().all(|n| *n == 0),
            "a half-stated mixture routes to experts that are not there",
        );
        assert!(
            m.top_k <= m.num_experts,
            "routing to more experts than exist is a top-k over a short list",
        );
    }

    /// The output projection is low-rank AND grouped, and both numbers
    /// have to be stated together: a rank with no groups is a projection
    /// the text's `wo_a`/`wo_b` pair cannot describe.
    #[test]
    fn the_output_projection_states_its_rank_and_its_groups_together() {
        let a = f().attn;
        assert!(a.o_lora_rank > 0 && a.o_groups > 0);
        assert!(
            a.o_lora_rank < a.q_width(),
            "a rank as wide as what it compresses is not a compression",
        );
    }
}
