//! Turning a config's facts into this family's [`DecodeGeometry`].
//!
//! The whole point is that it can FAIL. This family's geometry used to be
//! a default-constructed struct whose defaults were one preview
//! checkpoint's dimensions, so every other checkpoint of the family ran at
//! the wrong shape and said nothing — the loader binds by name and a name
//! says nothing about a dimension. Anything not derivable is refused
//! rather than defaulted, and the linear-attention refusals carry the
//! highest stakes: the conv and recurrent strides are computed from those
//! numbers, and a wrong stride reads one head's state as another's — not
//! a crash, a fluent model with the wrong memory.

use crate::facts::ModelFacts;

use super::geometry::DecodeGeometry;

/// The router kernel's two hard bounds (`moe_route.metal`), mirrored so
/// the geometry that refuses an oversized config and the launch shape read
/// the same number. The kernel clamps; a host that also clamped would
/// route with fewer experts than the config asked for and say nothing, so
/// this refuses instead.
pub const ROUTER_MAX_TOP_K: u32 = 16;
/// One lane per expert; see [`ROUTER_MAX_TOP_K`].
pub const ROUTER_MAX_EXPERTS: u32 = 1024;

/// Why a config's facts did not make a geometry. The message is the whole
/// diagnosis; nothing branches on which refusal it was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryRefused(pub String);

impl std::fmt::Display for GeometryRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "qwen3.5 geometry: {}", self.0)
    }
}

impl std::error::Error for GeometryRefused {}

fn positive(v: i32) -> Option<u32> {
    u32::try_from(v).ok().filter(|&v| v > 0)
}

/// Build the geometry a config describes, or say why it cannot be built.
///
/// Fills the SHAPE; the caller sets the quantization
/// ([`DecodeGeometry::quant`]/`alt_quant`) and the capacity fields
/// (`max_tokens`, `max_slots`, paging) which are the operator's, not the
/// config's.
///
/// # Errors
///
/// [`GeometryRefused`] naming the missing or inconsistent fact.
#[allow(clippy::too_many_lines)] // one refusal ladder; splitting hides the order
pub fn geometry_from_facts(f: &ModelFacts) -> Result<DecodeGeometry, GeometryRefused> {
    let refuse = |why: &str| Err(GeometryRefused(why.to_string()));
    if f.q35_num_hidden_layers <= 0 {
        return refuse("config carried no decoder shape");
    }
    let Some(n_layers) = positive(f.q35_num_hidden_layers) else {
        return refuse("num_hidden_layers and hidden_size must both be positive");
    };
    let Some(hidden) = positive(f.q35_hidden_size) else {
        return refuse("num_hidden_layers and hidden_size must both be positive");
    };
    let (Some(n_q_heads), Some(n_kv_heads)) = (
        positive(f.q35_num_attention_heads),
        positive(f.q35_num_key_value_heads),
    ) else {
        return refuse("attention head counts must be positive");
    };
    if n_q_heads % n_kv_heads != 0 {
        return Err(GeometryRefused(format!(
            "num_attention_heads {n_q_heads} is not a multiple of \
             num_key_value_heads {n_kv_heads}, which GQA requires"
        )));
    }
    let Some(vocab) = positive(f.q35_vocab_size) else {
        return refuse("vocab_size must be positive");
    };
    let head_dim = match positive(f.q35_head_dim) {
        Some(head_dim) => head_dim,
        None => {
            let derived = hidden / n_q_heads;
            if derived == 0 {
                return refuse("head_dim is absent and hidden/num_attention_heads is not positive");
            }
            derived
        }
    };

    // ── the linear-attention block ──
    let (Some(gdn_k_heads), Some(gdn_v_heads), Some(gdn_k_dim), Some(gdn_v_dim)) = (
        positive(f.q35_linear_key_heads),
        positive(f.q35_linear_value_heads),
        positive(f.q35_linear_key_head_dim),
        positive(f.q35_linear_value_head_dim),
    ) else {
        return refuse(
            "the linear-attention block needs linear_num_key_heads, \
             linear_num_value_heads, linear_key_head_dim and linear_value_head_dim",
        );
    };
    let Some(gdn_conv_k) = positive(f.q35_linear_conv_kernel) else {
        return refuse("the linear-attention block needs linear_conv_kernel_dim");
    };
    if gdn_v_heads % gdn_k_heads != 0 {
        return Err(GeometryRefused(format!(
            "linear_num_value_heads {gdn_v_heads} is not a multiple of \
             linear_num_key_heads {gdn_k_heads}, which the GDN's head repeat requires"
        )));
    }
    if gdn_k_dim % 32 != 0 {
        return Err(GeometryRefused(format!(
            "linear_key_head_dim {gdn_k_dim} is not a multiple of 32; the GDN \
             core reduces a head across one simdgroup's lanes"
        )));
    }
    if gdn_k_dim / 32 > 8 {
        return Err(GeometryRefused(format!(
            "linear_key_head_dim {gdn_k_dim} exceeds the 256 the GDN core's \
             per-lane registers hold"
        )));
    }

    if f.q35_full_attn_interval < 0 {
        return refuse(
            "layer_types lists an irregular full-attention pattern; this driver \
             places one every fixed interval and will not round an irregular \
             stack to a regular one",
        );
    }
    if f.q35_full_attn_interval == 0 {
        return refuse(
            "the config states no full_attention_interval and no layer_types; \
             which layers are linear cannot be guessed",
        );
    }

    let mut out = DecodeGeometry {
        n_layers,
        hidden,
        vocab,
        eps: f.q35_rms_norm_eps,
        n_q_heads,
        n_kv_heads,
        head_dim,
        tied_embeddings: f.q35_tied_embeddings,
        intermediate: positive(f.q35_intermediate_size).unwrap_or(0),
        gdn_k_heads,
        gdn_v_heads,
        gdn_k_dim,
        gdn_v_dim,
        gdn_conv_k,
        // DERIVED, not read: the convolution runs over q, k and v
        // concatenated and the value total is what the out projection
        // consumes, so neither a config nor this driver can state them
        // inconsistently with the head counts.
        gdn_v_total: gdn_v_heads * gdn_v_dim,
        gdn_conv_dim: 2 * gdn_k_heads * gdn_k_dim + gdn_v_heads * gdn_v_dim,
        full_attn_interval: positive(f.q35_full_attn_interval).unwrap_or(1),
        n_experts: 0,
        experts_per_token: 0,
        moe_intermediate: 0,
        shared_intermediate: 0,
        ..DecodeGeometry::default()
    };

    // ── the FFN ──
    if f.q35_num_experts > 0 || f.q35_num_experts_per_tok > 0 {
        let Some(experts_per_token) = positive(f.q35_num_experts_per_tok) else {
            return refuse("a routed FFN needs num_experts_per_tok");
        };
        let n_experts = positive(f.q35_num_experts).unwrap_or(0);
        if experts_per_token > n_experts {
            return Err(GeometryRefused(format!(
                "num_experts_per_tok {experts_per_token} exceeds num_experts {n_experts}"
            )));
        }
        let Some(moe_intermediate) = positive(f.q35_moe_intermediate_size) else {
            return refuse("a routed FFN needs moe_intermediate_size");
        };
        if experts_per_token > ROUTER_MAX_TOP_K {
            return Err(GeometryRefused(format!(
                "num_experts_per_tok {experts_per_token} exceeds the router's \
                 top-k limit of {ROUTER_MAX_TOP_K}"
            )));
        }
        if n_experts > ROUTER_MAX_EXPERTS {
            return Err(GeometryRefused(format!(
                "num_experts {n_experts} exceeds the {ROUTER_MAX_EXPERTS} a \
                 single threadgroup can rank"
            )));
        }
        if f.q35_shared_expert_intermediate < 0 {
            return refuse("shared_expert_intermediate_size is negative");
        }
        if f.q35_decoder_sparse_step != 1 {
            return Err(GeometryRefused(format!(
                "decoder_sparse_step {} routes only some layers; this driver \
                 routes every layer or none",
                f.q35_decoder_sparse_step
            )));
        }
        if f.q35_mlp_only_layer_count != 0 {
            return Err(GeometryRefused(format!(
                "mlp_only_layers exempts {} layers from routing; this driver \
                 routes every layer or none",
                f.q35_mlp_only_layer_count
            )));
        }
        out.norm_topk_prob = f.q35_norm_topk_prob;
        out.n_experts = n_experts;
        out.experts_per_token = experts_per_token;
        out.moe_intermediate = moe_intermediate;
        out.shared_intermediate = positive(f.q35_shared_expert_intermediate).unwrap_or(0);
    } else if f.q35_intermediate_size <= 0 {
        return refuse("a dense FFN needs intermediate_size");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_facts() -> ModelFacts {
        ModelFacts {
            q35_num_hidden_layers: 24,
            q35_hidden_size: 1024,
            q35_vocab_size: 248_320,
            q35_num_attention_heads: 8,
            q35_num_key_value_heads: 2,
            q35_head_dim: 256,
            q35_intermediate_size: 3584,
            q35_linear_key_heads: 16,
            q35_linear_value_heads: 16,
            q35_linear_key_head_dim: 128,
            q35_linear_value_head_dim: 128,
            q35_linear_conv_kernel: 4,
            q35_full_attn_interval: 4,
            q35_rms_norm_eps: 1e-6,
            q35_tied_embeddings: true,
            ..ModelFacts::default()
        }
    }

    #[test]
    fn a_sound_config_builds_the_shape_and_derives_the_conv_width() {
        let g = geometry_from_facts(&dense_facts()).expect("a sound config");
        assert_eq!(g.gdn_conv_dim, 2 * 16 * 128 + 16 * 128);
        assert_eq!(g.gdn_v_total, 2048);
        assert_eq!(g.head_dim, 256);
        assert!(!g.is_moe());
    }

    #[test]
    fn anything_not_derivable_is_refused_rather_than_defaulted() {
        let mut f = dense_facts();
        f.q35_full_attn_interval = 0;
        assert!(geometry_from_facts(&f).is_err(), "no interval, no guess");
        let mut f = dense_facts();
        f.q35_linear_key_head_dim = 48;
        assert!(
            geometry_from_facts(&f).is_err(),
            "48 is not a simdgroup multiple"
        );
        let mut f = dense_facts();
        f.q35_num_key_value_heads = 3;
        assert!(geometry_from_facts(&f).is_err(), "GQA needs a whole ratio");
    }

    #[test]
    fn a_routed_config_is_checked_against_the_routers_own_bounds() {
        let mut f = dense_facts();
        f.q35_num_experts = 512;
        f.q35_num_experts_per_tok = 10;
        f.q35_moe_intermediate_size = 768;
        f.q35_shared_expert_intermediate = 512;
        f.q35_decoder_sparse_step = 1;
        let g = geometry_from_facts(&f).expect("a sound mixture");
        assert!(g.is_moe() && g.has_shared_expert());
        f.q35_num_experts_per_tok = 17;
        assert!(geometry_from_facts(&f).is_err(), "past the router's top-k");
    }
}
