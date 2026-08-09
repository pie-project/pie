//! Gemma 2's SHAPE: the numbers a gemma-2 checkpoint has.
//!
//! Ungated, for the reason `shared/llama_like/spec.rs` states: a
//! catalog row is written in these words, and a row is the crate's
//! identity under EVERY aspect — `chat` asks which template speaks for
//! it, `contract` asks who authors its load, `forward` asks what to
//! trace. One struct, three readers, which cannot hold if the struct
//! only exists when the tracer is compiled in.
//!
//! It is also what makes a row a `const`. A `const` cannot own a `Vec`,
//! and that is not a limitation to work around here — it is a question
//! worth answering, and the answer for gemma-2 is below.

use serde::{Deserialize, Serialize};

/// The attention block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma2AttnFacts {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// `cfg.use_qk_norm`: gemma-2 proper has none; the flag exists
    /// because the same loader serves later gemmas that do.
    pub qk_norm: bool,
    /// `cfg.query_pre_attn_scalar` — the query is scaled by a named
    /// constant BEFORE attention, which is a launch
    /// (`launch_scalar_mul_bf16`) and not a kernel parameter.
    pub query_pre_attn_scale: bool,
    /// `cfg.attn_logit_softcap` — a DISPATCH parameter, not a launch:
    /// the attention kernel takes it, so nothing states it separately.
    pub attn_logit_softcap: bool,
}

impl Gemma2AttnFacts {
    pub const fn q_width(&self) -> u32 {
        self.heads * self.head_dim
    }
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

/// The whole family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma2Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub tied_embeddings: bool,
    /// `cfg.final_logit_softcap` — the attention cap is a dispatch
    /// parameter, but this one IS a launch, at the end.
    pub final_logit_softcap: bool,
    /// `cfg.sliding_window`: how far back a LOCAL layer sees.
    ///
    /// This field and [`Self::full_attn_interval`] replace a
    /// `window_left: Vec<i32>` that held one entry per layer. On the
    /// 9B that vector was forty-two numbers with two distinct values in
    /// it, alternating — which is a RULE somebody wrote out longhand,
    /// and a rule stated as data has all the failure modes data has:
    /// it can be the wrong length, it can disagree with itself in the
    /// middle, and nothing about its type says it may not. The rule
    /// cannot. It also cannot be a `const`, and a row is a `const`.
    pub sliding_window: i32,
    /// Every `interval`-th layer attends the WHOLE context — gemma-2's
    /// alternation is `interval = 2`, and the shared predicate
    /// [`model_compiler::facts::full_attn_at`] spells it the way
    /// gemma-4 and qwen3_5 spell theirs.
    pub full_attn_interval: u32,
    pub attn: Gemma2AttnFacts,
}

impl Gemma2Facts {
    /// A layer attends the whole context when the schedule says so.
    ///
    /// Was `window_left[l] < 0`: a read of the table, which is the same
    /// answer one indirection later and one length check away from a
    /// panic.
    #[must_use]
    pub fn is_global(&self, l: u32) -> bool {
        model_compiler::facts::full_attn_at(self.full_attn_interval, l)
    }

    /// The window layer `l` attends over, `-1` for the whole context —
    /// the value the driver used to read out of the vector.
    #[must_use]
    pub fn window_left_at(&self, l: u32) -> i32 {
        if self.is_global(l) { -1 } else { self.sliding_window }
    }

    /// The whole schedule, materialised.
    ///
    /// For the one caller that genuinely wants the list — a deployment
    /// states a window per layer because `LayerAttention` is per layer —
    /// and for the test that holds the rule to the vector it replaced.
    #[must_use]
    pub fn window_by_layer(&self) -> Vec<i32> {
        (0..self.layers).map(|l| self.window_left_at(l)).collect()
    }

    /// `google/gemma-2-9b-it`.
    pub fn gemma_2_9b() -> Self {
        Gemma2Facts {
            layers: 42,
            vocab: 256_000,
            hidden: 3584,
            intermediate: 14336,
            tied_embeddings: true,
            final_logit_softcap: true,
            // Every other layer is global; the rest see 4096 back.
            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 16,
                kv_heads: 8,
                head_dim: 256,
                qk_norm: false,
                query_pre_attn_scale: true,
                attn_logit_softcap: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both window kinds, or the fixture is not exercising the thing that
    /// makes gemma-2 gemma-2.
    #[test]
    fn the_fixture_alternates_local_and_global() {
        let f = Gemma2Facts::gemma_2_9b();
        assert_eq!(f.window_by_layer().len() as u32, f.layers);
        assert!((0..f.layers).any(|l| f.is_global(l)));
        assert!((0..f.layers).any(|l| !f.is_global(l)));
    }

    /// The rule and the vector it replaced are the same forty-two
    /// numbers.
    ///
    /// This is the whole justification for the change, so it is a test
    /// and not a comment: `window_left` held
    /// `(0..42).map(|l| if l % 2 == 1 { -1 } else { 4096 })`, written
    /// out, and `full_attn_at(2, l)` is `(l + 1) % 2 == 0` — the same
    /// predicate with the parity spelled the way every other alternating
    /// family in the tree spells it.
    #[test]
    fn the_rule_is_the_vector_the_fixture_used_to_carry() {
        let f = Gemma2Facts::gemma_2_9b();
        let longhand: Vec<i32> = (0..42).map(|l| if l % 2 == 1 { -1 } else { 4096 }).collect();
        assert_eq!(f.window_by_layer(), longhand);
        assert_eq!(f.window_left_at(0), 4096, "layer 0 is local");
        assert_eq!(f.window_left_at(1), -1, "layer 1 is global");
    }

    /// A window read past the stack is still the rule's answer rather
    /// than a panic — which the vector could not promise, because a
    /// vector one entry short answered by indexing out of bounds.
    #[test]
    fn the_rule_answers_for_every_index_a_caller_can_ask() {
        let f = Gemma2Facts::gemma_2_9b();
        assert_eq!(f.window_left_at(41), -1);
        assert_eq!(f.window_left_at(9999), -1, "odd, so global; no bounds to run past");
    }

    /// The projection widths are the row's own arithmetic — 16 heads of
    /// 256 is a 4096-wide q, and 8 kv heads of the same width is half
    /// of it.
    #[test]
    fn the_projection_widths_are_heads_times_head_dim() {
        let a = Gemma2Facts::gemma_2_9b().attn;
        assert_eq!(a.q_width(), 4096);
        assert_eq!(a.kv_width(), 2048);
    }
}
