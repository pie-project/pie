//! `gpt-oss`'s SHAPE: the numbers a checkpoint of it has.
//!
//! Ungated, for the reason `llama_like`'s and `qwen_3_5`'s `spec.rs`
//! give: a row is written in these words and a row must answer under
//! every aspect, not only under `forward`. Two of the fields here earn
//! that especially -- the alternating window and the attention sinks
//! are what a DEPLOYMENT states per layer, and a driver that had to
//! link the tracer to learn its window schedule would be linking a
//! tracer to serve.
//!
//! What stayed behind in `forward/facts.rs` is [`GptOssCudaFacts`], the
//! per-backend BINDING facts: which MXFP4 policy the device resolved to,
//! the fused leg's admission threshold, whether the experts are streamed.
//! Those name kernels and env, so they belong to the aspect that has them.
//!
//! [`GptOssCudaFacts`]: super::forward::facts::GptOssCudaFacts

/// gpt-oss's shape. The family rides `mixtral.cpp`, so these are the
/// facts that text reads — not a checkpoint dump.
///
/// Two of them are here because the driver ANSWERS them per layer and
/// the declaration would otherwise have to re-derive them per fire: the
/// alternating window kind, and whether a layer carries attention sinks.
/// Both are load-time, so both erase at trace time.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// One expert's MLP width (`intermediate_size`). gpt-oss's is equal
    /// to `hidden`, which is a coincidence of this checkpoint and not a
    /// rule the text may lean on.
    pub intermediate: u32,
    pub experts: u32,
    pub top_k: u32,
    pub vocab: u32,
    pub tied_embeddings: bool,
    /// `swiglu_limit`; 0 means the plain SwiGLU. gpt-oss clamps at 7.0,
    /// and the clamp is a DIFFERENT KERNEL, so this decides which
    /// activation the text states rather than being a runtime scalar.
    pub swiglu_limit: f32,
    /// Whether the checkpoint biases q/k/v/o, the router, and the expert
    /// projections (`attention_bias`). gpt-oss biases all of them; the
    /// q/k/v biases FOLD INTO the projection's epilogue and the rest are
    /// their own launches.
    pub attention_bias: bool,
    /// Whether this deployment's rope is the YaRN-paper one. gpt-oss's
    /// config asks for it (factor 32 over an original 4096 context) and
    /// the driver resolves it at load, so it is a fact and not a fire's
    /// question — and a WRONG one here is not a crash but a silently
    /// unscaled rotation, which is how it went unnoticed.
    pub rope_yarn_original: bool,
    /// Every layer carries `attn_sinks` on gpt-oss. The driver asks
    /// `layer.attn_sinks != nullptr` per layer and only requests an LSE
    /// from attention where the answer is yes — so this is what decides
    /// whether the attention statement produces one value or two.
    pub attn_sinks: bool,
}

impl GptOssFacts {
    /// Whether layer `l` attends over the SLIDING window. gpt-oss
    /// alternates from layer 0 (`layer_types` reads
    /// sliding, full, sliding, full, …), which the driver reaches
    /// through `per_layer_window_left` — a scalar the text does not
    /// state, since the window is an argument and not a kernel.
    pub fn is_sliding(&self, l: u32) -> bool {
        l % 2 == 0
    }

    /// openai/gpt-oss-20b, read from the checkpoint's `config.json`
    /// (2026-08-06). `layer_types` alternates from sliding.
    ///
    /// The yarn `rope_scaling` is not a field HERE, and the reason
    /// changed: it used to be absent because the driver never applied it
    /// — `mixtral.cpp:320` passed the plain `rope_theta` — and stating
    /// it would have laundered a latent bug into a fact. That bug is
    /// fixed. The rescaling is now stated once, as this generation's
    /// private `ROPE_SCALING` const in `gpt_oss/mod.rs`, and reaches the
    /// driver through `Deployment::rope_scaling`. It stays out of the
    /// SHAPE because it is not a per-checkpoint number: every gpt-oss
    /// release rescales the same way, so it belongs to the generation
    /// and not the row.
    pub fn gpt_oss_20b() -> Self {
        Self {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 32,
            top_k: 4,
            vocab: 201088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
            attention_bias: true,
            rope_yarn_original: true,
            attn_sinks: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GptOssFacts;

    /// The alternation's PHASE, which is the whole risk in `is_sliding`.
    ///
    /// `layer_types` reads sliding, full, sliding, full — starting at
    /// sliding. An implementation off by one phase is not a crash and
    /// not a wrong number anywhere a test would normally look: every
    /// layer still attends, half of them just attend over the wrong
    /// span. Layer 0 is therefore asserted on its own, before the
    /// alternation is, because it is the fact the alternation hangs on.
    #[test]
    fn the_first_layer_slides_and_the_second_does_not() {
        let f = GptOssFacts::gpt_oss_20b();
        assert!(
            f.is_sliding(0),
            "gpt-oss's `layer_types` opens with sliding"
        );
        assert!(!f.is_sliding(1), "and the second layer is the full one");
    }

    #[test]
    fn the_window_alternates_over_every_layer_of_the_row() {
        let f = GptOssFacts::gpt_oss_20b();
        for l in 0..f.layers {
            assert_eq!(
                f.is_sliding(l),
                l % 2 == 0,
                "layer {l} breaks the alternation, so the driver's \
                 `per_layer_window_left` gets a span this shape did not state"
            );
        }
    }

    /// Half the layers slide, and the count is what a reader can check
    /// against the config without reimplementing the predicate.
    #[test]
    fn exactly_half_the_layers_slide() {
        let f = GptOssFacts::gpt_oss_20b();
        assert_eq!(f.layers % 2, 0, "an odd layer count would not split evenly");
        let sliding = (0..f.layers).filter(|l| f.is_sliding(*l)).count();
        assert_eq!(sliding as u32, f.layers / 2);
    }

    /// The predicate is TOTAL: it answers past the end rather than
    /// panicking, because a caller that walks `0..layers` is the only
    /// caller and a bounds check here would be dead code that reads
    /// like a guarantee.
    #[test]
    fn the_predicate_answers_beyond_the_stack_rather_than_panicking() {
        let f = GptOssFacts::gpt_oss_20b();
        assert!(f.is_sliding(f.layers + 2), "even, so sliding");
        assert!(!f.is_sliding(f.layers + 1), "odd, so full");
    }

    /// gpt-oss's attention is WIDER than its residual, and that is not
    /// a transcription slip.
    ///
    /// `q_heads * head_dim` is 4096 against a hidden of 2880, so the
    /// out-projection is `[hidden, q_heads * head_dim]` and not square.
    /// Every family this shape sits beside has them equal, which is
    /// exactly why the inequality is asserted rather than left to be
    /// noticed: a reader who "fixes" one of these numbers to make them
    /// match breaks the checkpoint.
    #[test]
    fn the_attention_width_is_not_the_hidden_width() {
        let f = GptOssFacts::gpt_oss_20b();
        assert_eq!(f.q_heads * f.head_dim, 4096);
        assert_eq!(f.hidden, 2880);
        assert_ne!(
            f.q_heads * f.head_dim,
            f.hidden,
            "gpt-oss's out-projection is rectangular; making these equal \
             is a shape error the loader would report as a size mismatch"
        );
    }

    #[test]
    fn the_head_counts_form_whole_gqa_groups() {
        let f = GptOssFacts::gpt_oss_20b();
        assert_eq!(
            f.q_heads % f.kv_heads,
            0,
            "a group size that does not divide leaves a kv head serving \
             a fractional fan-in"
        );
        assert_eq!(f.q_heads / f.kv_heads, 8);
    }

    /// The router takes 4 of 32, and `top_k` must be reachable.
    #[test]
    fn the_router_selects_fewer_experts_than_it_has() {
        let f = GptOssFacts::gpt_oss_20b();
        assert!(
            f.top_k > 0,
            "a router selecting nothing produces no MLP at all"
        );
        assert!(
            f.top_k < f.experts,
            "selecting every expert is a dense MLP wearing a router's cost"
        );
    }

    /// The clamp is a DIFFERENT KERNEL, so the number decides which
    /// activation the text states — 0 would silently pick the plain
    /// SwiGLU for a checkpoint trained against a clamped one.
    #[test]
    fn the_swiglu_clamp_is_set_and_selects_the_clamped_activation() {
        let f = GptOssFacts::gpt_oss_20b();
        assert!(
            f.swiglu_limit > 0.0,
            "0 means the plain SwiGLU, which is a different kernel from \
             the one gpt-oss was trained with"
        );
        assert!((f.swiglu_limit - 7.0).abs() < f32::EPSILON);
    }

    /// The three booleans the driver branches on, pinned together.
    ///
    /// Each one selects a launch: `attn_sinks` decides whether attention
    /// yields an LSE alongside its value, `attention_bias` decides
    /// whether the projections carry an epilogue, and
    /// `tied_embeddings` decides whether the unembedding is its own
    /// tensor. A flipped bool here is a plan that still lowers.
    #[test]
    fn the_branching_booleans_are_the_ones_the_checkpoint_states() {
        let f = GptOssFacts::gpt_oss_20b();
        assert!(f.attn_sinks, "every gpt-oss layer carries `attn_sinks`");
        assert!(
            f.attention_bias,
            "gpt-oss biases q/k/v/o, the router, and the experts"
        );
        assert!(
            !f.tied_embeddings,
            "gpt-oss ships a separate `lm_head`; tying it would read the \
             unembedding out of the token table"
        );
        assert!(
            f.rope_yarn_original,
            "the row rescales rope by the YaRN paper's rule"
        );
    }

    /// `intermediate == hidden` is a COINCIDENCE of this checkpoint, and
    /// the doc says so. The test records that the equality is observed
    /// rather than derived, so a later row that breaks it — and the
    /// 120b nearly does, holding 2880 while everything else grows — is
    /// read as a new measurement rather than as a bug here.
    #[test]
    fn the_expert_width_happens_to_equal_the_hidden_width() {
        let f = GptOssFacts::gpt_oss_20b();
        assert_eq!(f.intermediate, f.hidden);
        assert_eq!(f.intermediate, 2880, "the measurement, not the coincidence");
    }

    /// The fixture is a value, not a builder: two calls agree, and the
    /// derived `PartialEq` is what the row-equality test in `mod.rs`
    /// leans on.
    #[test]
    fn the_fixture_is_the_same_measurement_every_time() {
        assert_eq!(GptOssFacts::gpt_oss_20b(), GptOssFacts::gpt_oss_20b());
        let mut other = GptOssFacts::gpt_oss_20b();
        other.layers += 1;
        assert_ne!(
            GptOssFacts::gpt_oss_20b(),
            other,
            "an equality that ignores a field cannot hold the row to the fixture"
        );
    }
}
