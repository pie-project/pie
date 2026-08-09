//! Decode geometry for the llama-shaped families.
//!
//! This is the plain transformer decoder every one of `llama`, `llama3`,
//! `mistral`, `qwen2`, `qwen3` and `qwen3_moe` is: an RMSNorm sandwich,
//! GQA with NEOX rope, and a gated MLP. Nothing here is new to this
//! driver — gemma4 is this shape plus a norm sandwich, per-layer
//! embeddings and a sliding/full split; qwen3.5 is this shape plus
//! interleaved linear attention. A family that adds nothing still needs
//! its own geometry, because the SHAPE is the thing that varies.
//!
//! Two axes vary across the families this covers, and both are fields
//! rather than separate families:
//!
//! * **QK-norm.** Qwen3 RMS-normalises q and k per head before the
//!   rotation; llama, llama3 and mistral do not. The checkpoint says
//!   which by shipping `self_attn.q_norm` or not.
//! * **The FFN.** Dense SwiGLU, or a routed mixture. Qwen3-MoE is
//!   Qwen3's attention with the MLP replaced, so it is this geometry
//!   with `n_experts` set rather than a seventh family.
//!
//! The rope carries the one split worth stating twice:
//! [`LlamaGeometry::rope_scale`] divides POSITIONS (the linear schedule)
//! and [`LlamaGeometry::rope_scaling_factor`] divides FREQUENCIES inside
//! llama3's piecewise table. They are the same `factor` field in the
//! config, and letting one field mean both applies it twice — a rope
//! running at a plausible-looking wrong rate, not a crash.

use crate::facts::ModelFacts;

use super::abi::Kernel;
use super::consts::KN;
use super::geometry::{AffineFormat, DecodeGeometry};
use super::geometry_facts::{GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K};

/// The family's shape. Defaults are `meta-llama/Meta-Llama-3-8B`'s.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaGeometry {
    /// The model width.
    pub hidden: u32,
    /// Decoder layers.
    pub n_layers: u32,
    /// The head's vocabulary.
    pub vocab: u32,
    /// The RMS-norm epsilon.
    pub eps: f32,
    /// Whether `lm_head` IS `embed_tokens`. Llama 3 8B ships both;
    /// Qwen3's smaller sizes tie them. Picks which pair of dispatch
    /// kinds the ends of the DAG use.
    pub tied_embeddings: bool,
    /// Query heads.
    pub n_q_heads: u32,
    /// Key/value heads.
    pub n_kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// The affine width and group every quantized kernel name is spelled
    /// with — one fact, stated by the config; a pipeline built for the
    /// wrong pair answers instead of failing.
    pub quant: AffineFormat,
    /// Qwen3 RMS-normalises q and k per head before the rotation. Not a
    /// scale difference — the norm is over `head_dim` with its own
    /// learned weight, and omitting it on a checkpoint that has one is a
    /// wrong model that still produces fluent text.
    pub qk_norm: bool,
    /// The rope base.
    pub rope_theta: f32,
    /// Linear POSITION scale, 1.0 for none. Under `llama3` the config's
    /// `factor` belongs to the table instead and this resets to 1.
    pub rope_scale: f32,
    /// Llama 3.1's piecewise schedule: its `factor`, dividing
    /// FREQUENCIES inside the table. Kept apart from
    /// [`rope_scale`](Self::rope_scale) — see the module docs.
    pub rope_scaling_factor: f32,
    /// Llama 3.1: the low-frequency wavelength factor.
    pub rope_low_freq_factor: f32,
    /// Llama 3.1: the high-frequency wavelength factor.
    pub rope_high_freq_factor: f32,
    /// Llama 3.1: the pre-scaling context length.
    pub rope_original_max_position: u32,
    /// Whether the rotary frequencies are a TABLE rather than a
    /// geometric series in `rope_theta`. One predicate asked by the PSO
    /// choice, the constant binding and the kernel ABI alike — they must
    /// agree, and a disagreement is a rotation by the wrong angle rather
    /// than a crash.
    pub rope_freq_table: bool,
    /// Dense SwiGLU width. Unused when [`is_moe`](Self::is_moe).
    pub intermediate: u32,
    /// Routed experts; 0 is a dense FFN, which is the whole difference
    /// between qwen3 and qwen3_moe on this side.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
    /// One expert's MLP width.
    pub moe_intermediate: u32,
    /// The widest fire the pools are sized for.
    pub max_tokens: u32,
    /// The most requests one fire may carry.
    pub max_requests: u32,
    /// Recurrent-state slots (none for this family; carried for the
    /// shared plumbing).
    pub max_slots: u32,
    /// The KV page size.
    pub kv_page_size: u32,
    /// Physical pages in the paged pool.
    pub total_pages: u32,
    /// Whether the paged-KV regions exist.
    pub paged_kv_enabled: bool,
}

impl Default for LlamaGeometry {
    fn default() -> Self {
        LlamaGeometry {
            hidden: 4096,
            n_layers: 32,
            vocab: 128_256,
            eps: 1e-5,
            tied_embeddings: false,
            n_q_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            quant: AffineFormat { bits: 4, group: 64 },
            qk_norm: false,
            rope_theta: 500_000.0,
            rope_scale: 1.0,
            rope_scaling_factor: 1.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_original_max_position: 8192,
            rope_freq_table: false,
            intermediate: 14_336,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            max_tokens: 1,
            max_requests: 1,
            max_slots: 1,
            kv_page_size: 32,
            total_pages: 1,
            paged_kv_enabled: false,
        }
    }
}

impl LlamaGeometry {
    /// Whether the FFN is a routed mixture.
    #[must_use]
    pub const fn is_moe(&self) -> bool {
        self.n_experts > 0 && self.experts_per_token > 0
    }

    /// The width one expert's gate/up produce, or the dense width.
    #[must_use]
    pub const fn ffn_width(&self) -> u32 {
        if self.is_moe() {
            self.moe_intermediate
        } else {
            self.intermediate
        }
    }

    /// Full rotary: every family here rotates the whole head.
    #[must_use]
    pub const fn rotary_dims(&self) -> u32 {
        self.head_dim
    }

    /// The packed query width.
    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.n_q_heads * self.head_dim
    }

    /// The packed key/value width.
    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
}

/// The shared-machinery view of a llama shape: every layer full
/// attention, no GDN, no shared expert. What the shared weight walk,
/// the staging and the scratch sizing read.
#[must_use]
pub fn llama_decode_geometry(g: &LlamaGeometry) -> DecodeGeometry {
    DecodeGeometry {
        hidden: g.hidden,
        n_layers: g.n_layers,
        vocab: g.vocab,
        eps: g.eps,
        tied_embeddings: g.tied_embeddings,
        n_q_heads: g.n_q_heads,
        n_kv_heads: g.n_kv_heads,
        head_dim: g.head_dim,
        quant: g.quant,
        rotary_dims: g.rotary_dims(),
        rope_theta: g.rope_theta,
        gdn_k_heads: 0,
        gdn_v_heads: 0,
        gdn_k_dim: 0,
        gdn_v_dim: 0,
        gdn_conv_k: 0,
        gdn_conv_dim: 0,
        gdn_v_total: 0,
        intermediate: g.intermediate,
        n_experts: g.n_experts,
        experts_per_token: g.experts_per_token,
        moe_intermediate: g.moe_intermediate,
        mxfp4_experts: false,
        shared_intermediate: 0,
        max_tokens: g.max_tokens,
        max_requests: g.max_requests,
        max_slots: g.max_slots,
        kv_page_size: g.kv_page_size,
        total_pages: g.total_pages,
        paged_kv_enabled: g.paged_kv_enabled,
        full_attn_interval: 1,
        ..DecodeGeometry::default()
    }
}

/// This family's K and N per matvec kind — its OWN table, not the shared
/// one: `qmv_kn` answers for qwen3.5, whose `QmvQ` is the 2×-wide
/// `[query | gate]` projection. The same `Kernel::QmvQ` here is llama's
/// plain query, and reading the shared table would run every q matvec at
/// twice its width — off the end of the weight, not a crash. The routed
/// three carry the same K and N as their dense counterparts: the expert
/// axis is a stride into the weight stack, not a wider matrix.
#[must_use]
pub fn llama_qmv_kn(kind: Kernel, g: &LlamaGeometry) -> KN {
    let h = g.hidden;
    let kn = |k, n| KN { k, n };
    match kind {
        Kernel::QmvQ => kn(h, g.q_width()),
        Kernel::QmvK | Kernel::QmvV => kn(h, g.kv_width()),
        Kernel::QmvO => kn(g.q_width(), h),
        Kernel::QmvGate | Kernel::QmvUp => kn(h, g.intermediate),
        Kernel::QmvDown => kn(g.intermediate, h),
        Kernel::LlRouter => kn(h, g.n_experts),
        Kernel::LlExpertGate | Kernel::LlExpertUp => kn(h, g.moe_intermediate),
        Kernel::LlExpertDown => kn(g.moe_intermediate, h),
        Kernel::QmvLmHead | Kernel::LmHeadUntied => kn(h, g.vocab),
        _ => kn(0, 0),
    }
}

/// Build the geometry a config describes, or report why it cannot be
/// built. Refused rather than defaulted: a checkpoint whose shape this
/// driver silently guessed at would produce plausible-looking wrong
/// tokens.
///
/// # Errors
///
/// [`GeometryRefused`] naming the missing or inconsistent fact.
#[allow(clippy::too_many_lines)] // one ladder; splitting hides the order
pub fn llama_geometry_from_facts(f: &ModelFacts) -> Result<LlamaGeometry, GeometryRefused> {
    let refuse = |why: String| Err(GeometryRefused(format!("llama: {why}")));
    if f.ll_num_hidden_layers <= 0 {
        return refuse("config carried no decoder shape".to_string());
    }
    if f.ll_hidden_size <= 0 {
        return refuse("n_layers and hidden_size must both be positive".to_string());
    }
    if f.ll_num_attention_heads <= 0 || f.ll_num_key_value_heads <= 0 {
        return refuse("head counts must be positive".to_string());
    }
    if f.ll_num_attention_heads % f.ll_num_key_value_heads != 0 {
        return refuse(format!(
            "n_q_heads {} is not a multiple of n_kv_heads {}, which GQA requires",
            f.ll_num_attention_heads, f.ll_num_key_value_heads
        ));
    }
    // `llama3` is a table; linear and default are the plain geometric
    // series. Anything else — yarn, longrope, dynamic — is refused,
    // because approximating a per-channel schedule with the linear scale
    // RUNS and is wrong past the original context length.
    let kind = f.ll_rope_scaling_kind.as_str();
    if !kind.is_empty() && kind != "linear" && kind != "default" && kind != "llama3" {
        return refuse(format!(
            "rope_scaling type '{kind}' is not implemented; only an absent, linear or llama3 scaling is"
        ));
    }

    let mut g = LlamaGeometry {
        n_layers: f.ll_num_hidden_layers.unsigned_abs(),
        hidden: f.ll_hidden_size.unsigned_abs(),
        vocab: f.ll_vocab_size.unsigned_abs(),
        eps: f.ll_rms_norm_eps,
        n_q_heads: f.ll_num_attention_heads.unsigned_abs(),
        n_kv_heads: f.ll_num_key_value_heads.unsigned_abs(),
        head_dim: if f.ll_head_dim > 0 {
            f.ll_head_dim.unsigned_abs()
        } else {
            (f.ll_hidden_size / f.ll_num_attention_heads).unsigned_abs()
        },
        qk_norm: f.ll_qk_norm,
        tied_embeddings: f.ll_tied_embeddings,
        rope_theta: f.ll_rope_theta,
        rope_scale: if f.ll_rope_scale > 0.0 {
            f.ll_rope_scale
        } else {
            1.0
        },
        rope_freq_table: kind == "llama3",
        rope_low_freq_factor: f.ll_rope_low_freq_factor,
        rope_high_freq_factor: f.ll_rope_high_freq_factor,
        rope_original_max_position: f.ll_rope_original_max_position.unsigned_abs(),
        intermediate: f.ll_intermediate_size.unsigned_abs(),
        n_experts: f.ll_num_experts.unsigned_abs(),
        experts_per_token: f.ll_num_experts_per_tok.unsigned_abs(),
        moe_intermediate: f.ll_moe_intermediate_size.unsigned_abs(),
        ..LlamaGeometry::default()
    };
    if g.rope_freq_table {
        // The table divides frequencies; the kernel's own `scale`
        // divides positions. Leaving `factor` in both applies it twice —
        // a rope that runs at a plausible-looking wrong rate.
        g.rope_scaling_factor = g.rope_scale;
        if (g.rope_low_freq_factor - g.rope_high_freq_factor).abs() < f32::EPSILON {
            return refuse(format!(
                "llama3 rope_scaling needs low_freq_factor != high_freq_factor; both are {}",
                g.rope_low_freq_factor
            ));
        }
        if g.rope_original_max_position == 0 {
            return refuse(
                "llama3 rope_scaling needs a positive original_max_position_embeddings".to_string(),
            );
        }
        g.rope_scale = 1.0;
    }

    if g.head_dim == 0 {
        return refuse(
            "head_dim is absent and hidden/n_q_heads is not positive either".to_string(),
        );
    }
    if g.vocab == 0 {
        return refuse("vocab_size must be positive".to_string());
    }
    if g.n_experts > 0 || g.experts_per_token > 0 {
        if g.experts_per_token == 0 {
            return refuse("a routed FFN needs num_experts_per_tok".to_string());
        }
        if g.experts_per_token > g.n_experts {
            return refuse(format!(
                "experts_per_token {} exceeds n_experts {}",
                g.experts_per_token, g.n_experts
            ));
        }
        if g.moe_intermediate == 0 {
            return refuse("a routed FFN needs moe_intermediate_size".to_string());
        }
        // `router_topk` holds the chosen logits in a fixed threadgroup
        // array and clamps k to its size. Clamping silently would route
        // with fewer experts than the config asks for while every
        // consumer still strides by the configured k, so the refusal the
        // kernel documents lives here.
        if g.experts_per_token > ROUTER_MAX_TOP_K {
            return refuse(format!(
                "experts_per_token {} exceeds the router's top-k limit of {ROUTER_MAX_TOP_K}",
                g.experts_per_token
            ));
        }
        // One lane per expert, one threadgroup per row: the expert count
        // is bounded by the threadgroup size. The dispatch helper clamps,
        // which would route among the first 1024 experts and never
        // mention it.
        if g.n_experts > ROUTER_MAX_EXPERTS {
            return refuse(format!(
                "n_experts {} exceeds the {ROUTER_MAX_EXPERTS} a single threadgroup can rank",
                g.n_experts
            ));
        }
        // The router softmaxes the SELECTED logits, so its weights sum
        // to one over the chosen experts — exactly `norm_topk_prob:
        // true`. A config that says false wants weights from the softmax
        // over ALL experts, which sum to less than one and scale the
        // FFN's whole contribution down. Same tokens, quietly wrong
        // magnitudes; refuse instead.
        if !f.ll_norm_topk_prob {
            return refuse(
                "norm_topk_prob is false; the router normalizes over the selected experts only"
                    .to_string(),
            );
        }
    } else if g.intermediate == 0 {
        return refuse("a dense FFN needs intermediate_size".to_string());
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama3_8b_facts() -> ModelFacts {
        ModelFacts {
            ll_num_hidden_layers: 32,
            ll_hidden_size: 4096,
            ll_vocab_size: 128_256,
            ll_num_attention_heads: 32,
            ll_num_key_value_heads: 8,
            ll_intermediate_size: 14_336,
            ll_rms_norm_eps: 1e-5,
            // The facts default to HF's `tie_word_embeddings: true`;
            // Llama 3 8B's config states false and ships both matrices.
            ll_tied_embeddings: false,
            ..ModelFacts::default()
        }
    }

    #[test]
    fn the_8b_config_lands_on_the_defaults_and_head_dim_is_derived() {
        let g = llama_geometry_from_facts(&llama3_8b_facts()).expect("the 8B shape");
        assert_eq!(g, LlamaGeometry::default());
        // head_dim was never stated: 4096 / 32.
        assert_eq!(g.head_dim, 128);
        assert!(!g.is_moe());
        assert_eq!(g.ffn_width(), 14_336);
    }

    #[test]
    fn the_family_kn_table_is_not_the_shared_one() {
        let g = LlamaGeometry::default();
        // The shared table answers for qwen3.5, whose QmvQ is the
        // 2x-wide [query | gate]; llama's is plain. Same Kernel, half
        // the width — which is the whole reason this table exists.
        assert_eq!(llama_qmv_kn(Kernel::QmvQ, &g).n, g.q_width());
        assert_eq!(llama_qmv_kn(Kernel::QmvO, &g).k, g.q_width());
        assert_eq!(llama_qmv_kn(Kernel::LmHeadUntied, &g).n, g.vocab);
        let moe = LlamaGeometry {
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            ..LlamaGeometry::default()
        };
        assert_eq!(llama_qmv_kn(Kernel::LlExpertGate, &moe).n, 768);
        assert_eq!(llama_qmv_kn(Kernel::LlExpertDown, &moe).k, 768);
        assert_eq!(llama_qmv_kn(Kernel::Rms, &g).n, 0, "not a matvec");
        // And the shared view keeps every layer on the attention path.
        let view = llama_decode_geometry(&g);
        assert!((0..32).all(|l| view.is_full_attn(l)));
        assert_eq!(view.moe_intermediate, 0);
    }

    #[test]
    fn gqa_needs_the_head_counts_to_divide() {
        let f = ModelFacts {
            ll_num_key_value_heads: 7,
            ..llama3_8b_facts()
        };
        let err = llama_geometry_from_facts(&f).unwrap_err();
        assert!(err.0.contains("GQA requires"), "{}", err.0);
    }

    #[test]
    fn the_factor_moves_into_the_table_instead_of_applying_twice() {
        let f = ModelFacts {
            ll_rope_scaling_kind: "llama3".to_string(),
            ll_rope_scale: 8.0,
            ll_rope_low_freq_factor: 1.0,
            ll_rope_high_freq_factor: 4.0,
            ll_rope_original_max_position: 8192,
            ..llama3_8b_facts()
        };
        let g = llama_geometry_from_facts(&f).expect("the 3.1 schedule");
        assert!(g.rope_freq_table);
        assert!((g.rope_scaling_factor - 8.0).abs() < f32::EPSILON);
        assert!(
            (g.rope_scale - 1.0).abs() < f32::EPSILON,
            "the position scale must reset, or the factor divides twice"
        );
        // A degenerate ramp is refused, not divided by.
        let flat = ModelFacts {
            ll_rope_low_freq_factor: 4.0,
            ..f
        };
        assert!(llama_geometry_from_facts(&flat).is_err());
        // A per-channel schedule the driver does not implement is
        // refused, not approximated with the linear scale.
        let yarn = ModelFacts {
            ll_rope_scaling_kind: "yarn".to_string(),
            ..llama3_8b_facts()
        };
        let err = llama_geometry_from_facts(&yarn).unwrap_err();
        assert!(err.0.contains("not implemented"), "{}", err.0);
    }

    #[test]
    fn the_mixture_ladder_refuses_each_missing_or_oversized_fact() {
        let moe = |n: i32, k: i32, width: i32, norm: bool| ModelFacts {
            ll_num_experts: n,
            ll_num_experts_per_tok: k,
            ll_moe_intermediate_size: width,
            ll_norm_topk_prob: norm,
            ..llama3_8b_facts()
        };
        // qwen3_moe's real shape passes.
        let g = llama_geometry_from_facts(&moe(128, 8, 768, true)).expect("qwen3-moe");
        assert!(g.is_moe());
        assert_eq!(g.ffn_width(), 768);
        // Each rung of the ladder, in the order the C++ asks.
        assert!(llama_geometry_from_facts(&moe(128, 0, 768, true)).is_err());
        assert!(llama_geometry_from_facts(&moe(4, 8, 768, true)).is_err());
        assert!(llama_geometry_from_facts(&moe(128, 8, 0, true)).is_err());
        assert!(
            llama_geometry_from_facts(&moe(128, ROUTER_MAX_TOP_K as i32 + 1, 768, true)).is_err()
        );
        assert!(
            llama_geometry_from_facts(&moe(ROUTER_MAX_EXPERTS as i32 + 1, 8, 768, true)).is_err()
        );
        // Weights from the softmax over ALL experts sum below one and
        // scale the FFN down — same tokens, quietly wrong magnitudes.
        let err = llama_geometry_from_facts(&moe(128, 8, 768, false)).unwrap_err();
        assert!(err.0.contains("norm_topk_prob"), "{}", err.0);
        // Dense with no width is refused too.
        let dense = ModelFacts {
            ll_intermediate_size: 0,
            ..llama3_8b_facts()
        };
        assert!(llama_geometry_from_facts(&dense).is_err());
    }
}
