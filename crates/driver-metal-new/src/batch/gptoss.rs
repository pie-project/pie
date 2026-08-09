//! GPT-OSS decode geometry.
//!
//! Read off `openai/gpt-oss-20b`, whose shape the defaults carry.
//! Everything is runtime state: compiling one checkpoint's shape into the
//! type is what stopped a second family from loading at all, and this is
//! the third.
//!
//! Four things make this family's shape different, all in the SCHEDULE
//! rather than the arithmetic: every layer is a sparse MoE (the dispatch
//! shape depends on a value the GPU computes); attention has per-head
//! sinks (one learned scalar joining the softmax denominator, letting a
//! head attend to nothing); every projection is biased; and
//! sliding/full attention alternate with a 128 window — small enough to
//! bind at almost every realistic context, unlike gemma4's 512.
//!
//! Three quantization facts are SOLVED FROM THE STAGED TENSORS, never
//! assumed, because `config.json` cannot be trusted for any of them and
//! each wrong guess is fluent wrong text rather than a crash:
//! [`GptOssGeometry::router_bits`] (mlx_lm's predicate leaves a 32×2880
//! router at 8 bits inside a 4-bit checkpoint),
//! [`GptOssGeometry::proj_bits`] (the MXFP4-Q8 publish declares a global
//! mxfp4/4/g32 and then overrides every attention projection back to
//! affine/8/g64 — two hardcoded `4`s once read those at half the stride),
//! and [`GptOssGeometry::mxfp4_experts`] (the config says "mxfp4" for a
//! checkpoint the loader may still have converted; the only honest witness
//! is what is in the heap).

use crate::facts::ModelFacts;

use super::geometry::{AffineFormat, DecodeGeometry};
use super::geometry_facts::{GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K};

/// The family's shape. Defaults are `openai/gpt-oss-20b`'s.
#[derive(Clone, Debug, PartialEq)]
pub struct GptOssGeometry {
    /// The model width.
    pub hidden: u32,
    /// Decoder layers.
    pub n_layers: u32,
    /// The head's vocabulary.
    pub vocab: u32,
    /// The RMS-norm epsilon.
    pub eps: f32,
    /// gpt-oss ships a separate `lm_head`; it does not reuse the embedding.
    pub tied_embeddings: bool,
    /// Query heads.
    pub n_q_heads: u32,
    /// Key/value heads.
    pub n_kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// YaRN: the base. The driver needs the same four numbers mlx_lm's
    /// `YarnRoPE` takes.
    pub rope_theta: f32,
    /// YaRN: the context-scaling factor.
    pub rope_factor: f32,
    /// YaRN: the fast interpolation bound.
    pub rope_beta_fast: f32,
    /// YaRN: the slow interpolation bound.
    pub rope_beta_slow: f32,
    /// YaRN: the pre-scaling context length.
    pub rope_original_max_position: u32,
    /// Sliding layers attend the last this-many positions. Layer 0 is
    /// sliding and they alternate — `layer_types` is a strict alternation,
    /// stated by [`is_sliding`](Self::is_sliding) rather than derived from
    /// an interval the way gemma4's is.
    pub sliding_window: u32,
    /// Routed experts; every layer is a mixture.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
    /// One expert's MLP width — NOT a dense intermediate: the layer
    /// computes `experts_per_token` of these and sums them.
    pub intermediate: u32,
    /// The clamp gpt-oss applies inside its SwiGLU.
    pub swiglu_limit: f32,
    /// The SwiGLU sigmoid's gain.
    pub swiglu_alpha: f32,
    /// The router's affine width — solved from the staged tensors.
    pub router_bits: u32,
    /// The attention/embedding projections' affine width — solved from the
    /// staged tensors (group 64 whatever the width).
    pub proj_bits: u32,
    /// Whether the expert bank stays in the checkpoint's MXFP4 — solved
    /// from the staged tensors; chooses both what is bound and which
    /// matvec runs.
    pub mxfp4_experts: bool,
    /// The widest fire the pools are sized for.
    pub max_tokens: u32,
    /// The most requests one fire may carry.
    pub max_requests: u32,
    /// Recurrent-state slots (none for this family; carried for the shared
    /// plumbing).
    pub max_slots: u32,
    /// The KV page size.
    pub kv_page_size: u32,
    /// Physical pages in the paged pool.
    pub total_pages: u32,
    /// Whether the paged-KV regions exist.
    pub paged_kv_enabled: bool,
}

impl Default for GptOssGeometry {
    fn default() -> Self {
        GptOssGeometry {
            hidden: 2880,
            n_layers: 24,
            vocab: 201_088,
            eps: 1e-5,
            tied_embeddings: false,
            n_q_heads: 64,
            n_kv_heads: 8,
            head_dim: 64,
            rope_theta: 150_000.0,
            rope_factor: 32.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_original_max_position: 4096,
            sliding_window: 128,
            n_experts: 32,
            experts_per_token: 4,
            intermediate: 2880,
            swiglu_limit: 7.0,
            swiglu_alpha: 1.702,
            router_bits: 8,
            proj_bits: 4,
            mxfp4_experts: false,
            max_tokens: 1,
            max_requests: 1,
            max_slots: 1,
            kv_page_size: 32,
            total_pages: 1,
            paged_kv_enabled: false,
        }
    }
}

impl GptOssGeometry {
    /// Sliding and full alternate, starting sliding — verified against the
    /// checkpoint's own `layer_types`.
    #[must_use]
    pub fn is_sliding(&self, layer: u32) -> bool {
        layer.is_multiple_of(2)
    }

    /// The complement of [`is_sliding`](Self::is_sliding).
    #[must_use]
    pub fn is_full_attn(&self, layer: u32) -> bool {
        !self.is_sliding(layer)
    }

    /// The packed query width.
    #[must_use]
    pub const fn q_dim(&self) -> u32 {
        self.n_q_heads * self.head_dim
    }

    /// The packed key/value width.
    #[must_use]
    pub const fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    /// Query heads per key/value head.
    #[must_use]
    pub const fn gqa_factor(&self) -> u32 {
        match self.n_q_heads.checked_div(self.n_kv_heads) {
            Some(factor) => factor,
            None => 1,
        }
    }
}

/// The shared-geometry VIEW of this family, for the passes that are
/// genuinely shared: storage staging, the weight/state/IO bind walk, and
/// the scratch binds all read [`DecodeGeometry`], and every field they
/// read has a gpt-oss answer.
///
/// Every layer reads `full_attn_interval: 1` — KV pairs for all of them,
/// GDN state for none — because the sliding layers SHARE the full-attn
/// storage shape: a window is a property of the attention read (the
/// consts walk binds it per layer), not of what is stored. The GDN
/// widths stay zero and are never reached: no gpt-oss layer is
/// recurrent, so the staging loop takes the KV arm at every index.
#[must_use]
pub fn gptoss_decode_geometry(g: &GptOssGeometry) -> DecodeGeometry {
    DecodeGeometry {
        hidden: g.hidden,
        n_layers: g.n_layers,
        vocab: g.vocab,
        eps: g.eps,
        tied_embeddings: g.tied_embeddings,
        n_q_heads: g.n_q_heads,
        n_kv_heads: g.n_kv_heads,
        head_dim: g.head_dim,
        quant: AffineFormat {
            bits: g.proj_bits,
            group: 64,
        },
        alt_quant: AffineFormat {
            bits: g.router_bits,
            group: 64,
        },
        rotary_dims: g.head_dim,
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
        moe_intermediate: g.intermediate,
        mxfp4_experts: g.mxfp4_experts,
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

/// The widest value the M=1 DAG parks in one scratch slot, in ELEMENTS.
///
/// The candidates: the sorted expert stack (`experts_per_token` rows of
/// `intermediate` — top-4 at tile 1 keeps `sorted_rows` equal to the
/// top-k) and its `hidden`-wide gather twin, against the packed query.
/// The caller picks the byte width; four covers the f32 values.
#[must_use]
pub fn gptoss_scratch_elems(g: &GptOssGeometry) -> u64 {
    let sorted = g.experts_per_token.max(1);
    let stack = u64::from(sorted) * u64::from(g.intermediate.max(g.hidden));
    stack.max(u64::from(g.q_dim())).max(u64::from(g.n_experts))
}

fn positive(v: i32) -> Option<u32> {
    u32::try_from(v).ok().filter(|&v| v > 0)
}

/// Build the geometry a config describes, or report why it cannot be
/// built. Refused rather than defaulted: a shape this driver silently
/// guessed at would produce plausible-looking wrong tokens.
///
/// # Errors
///
/// [`GeometryRefused`] naming the missing or inconsistent fact.
pub fn gptoss_geometry_from_facts(f: &ModelFacts) -> Result<GptOssGeometry, GeometryRefused> {
    let refuse = |why: String| Err(GeometryRefused(why));
    if f.go_num_hidden_layers <= 0 {
        return refuse("gpt-oss: config carried no decoder shape".to_string());
    }
    let (Some(n_experts), Some(experts_per_token)) = (
        positive(f.go_num_local_experts),
        positive(f.go_num_experts_per_tok),
    ) else {
        return refuse(
            "gpt-oss: every layer is a sparse MoE, so both num_local_experts \
             and num_experts_per_tok must be positive"
                .to_string(),
        );
    };
    if experts_per_token > n_experts {
        return refuse("gpt-oss: num_experts_per_tok exceeds num_local_experts".to_string());
    }
    // The same two bounds the llama routed branch refuses, because it is
    // the same router_topk: it ranks one expert per lane in a fixed
    // threadgroup array and CLAMPS — silently routing with fewer experts
    // than the config asks for while every consumer keeps striding by the
    // configured k.
    if experts_per_token > ROUTER_MAX_TOP_K {
        return refuse(format!(
            "gpt-oss: num_experts_per_tok {experts_per_token} exceeds the \
             router's top-k limit of {ROUTER_MAX_TOP_K}"
        ));
    }
    if n_experts > ROUTER_MAX_EXPERTS {
        return refuse(format!(
            "gpt-oss: num_local_experts {n_experts} exceeds the \
             {ROUTER_MAX_EXPERTS} a single threadgroup can rank"
        ));
    }
    let (Some(n_q_heads), Some(n_kv_heads)) = (
        positive(f.go_num_attention_heads),
        positive(f.go_num_key_value_heads),
    ) else {
        return refuse(
            "gpt-oss: the query heads do not divide evenly into the key/value \
             heads, which grouped attention assumes"
                .to_string(),
        );
    };
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return refuse(
            "gpt-oss: the query heads do not divide evenly into the key/value \
             heads, which grouped attention assumes"
                .to_string(),
        );
    }
    let mut out = GptOssGeometry::default();
    out.n_layers = positive(f.go_num_hidden_layers).unwrap_or(out.n_layers);
    out.hidden = positive(f.go_hidden_size).unwrap_or(out.hidden);
    out.vocab = positive(f.go_vocab_size).unwrap_or(out.vocab);
    out.eps = f.go_rms_norm_eps;
    out.n_q_heads = n_q_heads;
    out.n_kv_heads = n_kv_heads;
    out.head_dim = positive(f.go_head_dim).unwrap_or(out.head_dim);
    out.sliding_window = positive(f.go_sliding_window).unwrap_or(out.sliding_window);
    out.n_experts = n_experts;
    out.experts_per_token = experts_per_token;
    out.intermediate = positive(f.go_intermediate_size).unwrap_or(out.intermediate);
    out.swiglu_limit = f.go_swiglu_limit;
    out.rope_theta = f.go_rope_theta;
    out.rope_factor = f.go_rope_factor;
    out.rope_beta_fast = f.go_rope_beta_fast;
    out.rope_beta_slow = f.go_rope_beta_slow;
    out.rope_original_max_position =
        positive(f.go_rope_original_max_position).unwrap_or(out.rope_original_max_position);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn go_facts() -> ModelFacts {
        ModelFacts {
            go_num_hidden_layers: 24,
            go_hidden_size: 2880,
            go_vocab_size: 201_088,
            go_num_attention_heads: 64,
            go_num_key_value_heads: 8,
            go_head_dim: 64,
            go_sliding_window: 128,
            go_num_local_experts: 32,
            go_num_experts_per_tok: 4,
            go_intermediate_size: 2880,
            go_rope_original_max_position: 4096,
            go_rms_norm_eps: 1e-5,
            go_swiglu_limit: 7.0,
            go_rope_theta: 150_000.0,
            go_rope_factor: 32.0,
            go_rope_beta_fast: 32.0,
            go_rope_beta_slow: 1.0,
            ..ModelFacts::default()
        }
    }

    #[test]
    fn the_20b_shape_reads_back_and_the_layers_alternate() {
        let g = gptoss_geometry_from_facts(&go_facts()).expect("the shipped shape");
        assert_eq!((g.hidden, g.n_experts, g.gqa_factor()), (2880, 32, 8));
        assert!(g.is_sliding(0), "layer 0 slides");
        assert!(g.is_full_attn(1));
        assert!(g.is_sliding(2));
        assert_eq!(g.q_dim(), 4096);
        assert_eq!(g.kv_dim(), 512);
    }

    #[test]
    fn a_mixture_less_config_is_refused_because_every_layer_routes() {
        let mut f = go_facts();
        f.go_num_local_experts = 0;
        assert!(gptoss_geometry_from_facts(&f).is_err());
        let mut f = go_facts();
        f.go_num_experts_per_tok = 17;
        assert!(
            gptoss_geometry_from_facts(&f).is_err(),
            "past the router's top-k"
        );
        let mut f = go_facts();
        f.go_num_key_value_heads = 7;
        assert!(
            gptoss_geometry_from_facts(&f).is_err(),
            "GQA needs a whole ratio"
        );
    }
}
