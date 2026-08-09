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

use super::geometry::{AffineFormat, DecodeGeometry};

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
    // Whichever block the config filled. `from_descriptor` reads a llama-like
    // config into `ll_*` and a qwen3.5 one into `q35_*` -- the marker fields
    // its own doc describes -- and this read only the second, so a llama
    // snapshot was refused as "carrying no decoder shape" while carrying it in
    // the other block.
    //
    // Projecting rather than branching: every field below is the same question
    // asked of a different prefix, and a `match` per field is the per-family
    // ladder this crate is retiring. The GDN and interval fields have no `ll_`
    // twin because a llama-like config has no linear attention -- zero there is
    // the honest answer and not a default.
    // gemma4's block, projected the same way. Its own marker is
    // `g4_num_hidden_layers`, and it is read BEFORE the llama-like one because
    // a gemma4 config fills both.
    // gpt-oss's block, projected first because a gpt-oss config fills ONLY
    // `go_*` -- `from_descriptor` takes the `model_type == "gpt_oss"` branch
    // and never touches the llama-like or gemma4 readers. Without this
    // projection a real gpt-oss checkpoint was refused as "carrying no decoder
    // shape" while carrying it, which meant the GENERIC path could not reach
    // gpt-oss weights at all: only the retiring per-family layer could, and it
    // read `go_*` directly.
    let f = &if f.q35_num_hidden_layers > 0 || f.go_num_hidden_layers == 0 {
        f.clone()
    } else {
        ModelFacts {
            q35_num_hidden_layers: f.go_num_hidden_layers,
            q35_hidden_size: f.go_hidden_size,
            q35_vocab_size: f.go_vocab_size,
            q35_num_attention_heads: f.go_num_attention_heads,
            q35_num_key_value_heads: f.go_num_key_value_heads,
            q35_head_dim: f.go_head_dim,
            // gpt-oss is a mixture in EVERY layer and has no dense FFN beside
            // it, so the dense width and the expert width are the same number
            // -- `intermediate_size`. Stating it in both places is what lets
            // the shared reader below size a mixture without asking which
            // family it is.
            q35_intermediate_size: f.go_intermediate_size,
            q35_moe_intermediate_size: f.go_intermediate_size,
            q35_num_experts: f.go_num_local_experts,
            q35_num_experts_per_tok: f.go_num_experts_per_tok,
            q35_rms_norm_eps: f.go_rms_norm_eps,
            // gpt-oss unties its read-out; the config says so and the
            // checkpoint publishes a separate `lm_head`.
            q35_tied_embeddings: false,
            // Every layer attends and every layer routes: no GDN interleave,
            // no MLP-only tail, no sparse step.
            q35_full_attn_interval: 1,
            q35_mlp_only_layer_count: 0,
            q35_decoder_sparse_step: 1,
            ..f.clone()
        }
    };
    let f = &if f.q35_num_hidden_layers > 0 || f.g4_num_hidden_layers == 0 {
        f.clone()
    } else {
        ModelFacts {
            q35_num_hidden_layers: f.g4_num_hidden_layers,
            q35_hidden_size: f.g4_hidden_size,
            // The descriptor's TOP-LEVEL vocab. gemma4's own block states
            // none — the parser reads it once for every family — and
            // `ll_vocab_size` is only filled inside the llama-like branch,
            // which a gemma4 config does not take.
            q35_vocab_size: i32::try_from(f.vocab_size).unwrap_or(i32::MAX),
            q35_num_attention_heads: f.g4_num_attention_heads,
            q35_num_key_value_heads: f.g4_num_key_value_heads,
            q35_head_dim: f.g4_head_dim,
            q35_intermediate_size: f.g4_intermediate_size,
            q35_num_experts: f.g4_num_experts,
            q35_num_experts_per_tok: f.g4_experts_per_token,
            q35_moe_intermediate_size: f.g4_moe_intermediate,
            // The eps and the tying are read by the llama-like reader, which
            // a gemma4 config also fills.
            // gemma4's block states neither, and `ModelFacts` has no
            // family-free field for them. 1e-6 is gemma's published epsilon;
            // the tying comes from the llama-like reader, which a gemma4
            // config also fills at the top level.
            q35_rms_norm_eps: 1e-6,
            q35_tied_embeddings: f.ll_tied_embeddings,
            // A gemma4 stack alternates sliding and full attention rather
            // than interleaving linear layers, so it has no GDN block and
            // every layer attends.
            q35_full_attn_interval: 1,
            q35_mlp_only_layer_count: 0,
            q35_decoder_sparse_step: 1,
            ..f.clone()
        }
    };
    let f = &if f.q35_num_hidden_layers > 0 {
        f.clone()
    } else {
        ModelFacts {
            q35_num_hidden_layers: f.ll_num_hidden_layers,
            q35_hidden_size: f.ll_hidden_size,
            q35_vocab_size: f.ll_vocab_size,
            q35_num_attention_heads: f.ll_num_attention_heads,
            q35_num_key_value_heads: f.ll_num_key_value_heads,
            q35_head_dim: f.ll_head_dim,
            q35_intermediate_size: f.ll_intermediate_size,
            q35_num_experts: f.ll_num_experts,
            q35_num_experts_per_tok: f.ll_num_experts_per_tok,
            q35_moe_intermediate_size: f.ll_moe_intermediate_size,
            q35_rms_norm_eps: f.ll_rms_norm_eps,
            q35_tied_embeddings: f.ll_tied_embeddings,
            q35_norm_topk_prob: f.ll_norm_topk_prob,
            // Every layer attends: a llama-like stack has no GDN interleave,
            // so the interval is one and there are no MLP-only layers.
            q35_full_attn_interval: 1,
            q35_mlp_only_layer_count: 0,
            // Every layer that has a mixture has it: a llama-like config
            // states no sparse step, and 1 is "no layer is skipped".
            q35_decoder_sparse_step: 1,
            ..f.clone()
        }
    };
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
    //
    // A stack whose full-attention interval is ONE has no linear layers, so
    // there is no block to state and demanding one refuses every llama-like
    // config for lacking a thing it correctly does not have. The refusals
    // below still apply to a stack that DOES interleave -- which is the
    // distinction that matters: absent because there is none, versus absent
    // because the config is short.
    let interleaves = f.q35_full_attn_interval != 1;
    let (gdn_k_heads, gdn_v_heads, gdn_k_dim, gdn_v_dim, gdn_conv_k) = if interleaves {
        let (Some(k_heads), Some(v_heads), Some(k_dim), Some(v_dim)) = (
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
        let Some(conv_k) = positive(f.q35_linear_conv_kernel) else {
            return refuse("the linear-attention block needs linear_conv_kernel_dim");
        };
        (k_heads, v_heads, k_dim, v_dim, conv_k)
    } else {
        (1, 1, 32, 32, 1)
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

    // gemma4's attention geometry is PER LAYER TYPE, and it is carried now
    // rather than refused.
    //
    // Measured on `mlx-community/gemma-4-31b-it-4bit`'s own tensors, layer 0
    // (sliding) against layer 5 (full):
    //
    // | | sliding | full |
    // |---|---|---|
    // | `q_norm` | `[256]` | `[512]` |
    // | `q_proj` | `[8192, ...]` = 32x256 | `[16384, ...]` = 32x512 |
    // | `k_proj` | `[4096, ...]` = 16x256 | `[2048, ...]` = 4x512 |
    // | `v_proj` | ships | absent |
    //
    // `global_head_dim: 512` and `num_global_key_value_heads` state both
    // halves and both releases on this machine carry them. The descriptor has
    // parsed them since it was written and nothing read either, so the text
    // ran all sixty layers at the sliding shape -- half of each full layer's
    // Q, a quarter past the end of its K.
    //
    // A stack whose two shapes AGREE states zero, which is what every other
    // family here does and what `head_dim_at`/`kv_heads_at` read as "one
    // shape everywhere".
    let mut out = DecodeGeometry {
        n_layers,
        hidden,
        vocab,
        eps: f.q35_rms_norm_eps,
        // gemma4's attention shape: which layers slide, how often a full one
        // appears, and whether those full ones take V from K.
        attention_k_eq_v: f.g4_attention_k_eq_v,
        full_attn_every: u32::try_from(f.g4_full_attn_interval.max(0)).unwrap_or(0),
        sliding_window: u32::try_from(f.g4_sliding_window.max(0)).unwrap_or(0),
        // gemma's readout cap, when the config read as gemma.
        final_logit_softcap: if f.g4_num_hidden_layers > 0 {
            f.g4_final_softcap
        } else {
            0.0
        },
        // WHICH family this config read as, and the same marker the softcap
        // above keys on. Three facts hang off it that nothing else carries —
        // the `(1 + w)` norm, the four-norm sandwich, and GEGLU — and every
        // one of them was hardcoded to llama's answer, so a gemma checkpoint
        // passed `serves` and then ran as a llama. Finite numbers, fluent
        // text, a different model.
        gemma: f.g4_num_hidden_layers > 0,
        // The FULL-attention layers' own shape, zero when the stack states
        // one. `head_dim_at`/`kv_heads_at` read the zero as "same as the
        // sliding layers", so every family but gemma-4 is unaffected.
        //
        // Only carried when it DIFFERS: a config restating its own head_dim
        // under the global key is a stack with one shape, and answering it
        // with a second would make every layer take the `is_full_attention`
        // branch for no reason.
        global_head_dim: if f.g4_global_head_dim > 0 && f.g4_global_head_dim != f.g4_head_dim {
            u32::try_from(f.g4_global_head_dim).unwrap_or(0)
        } else {
            0
        },
        // gemma-4's partial rotary, carried rather than refused: the rope
        // rows name the statement's scalar with `grid_param`, so the extent
        // is per statement and no longer has to be one number for the fire.
        // Zero means "rotate the whole head", which is every other family.
        full_partial_rotary: if f.g4_num_hidden_layers > 0
            && f.g4_full_partial_rotary > 0.0
            && (f.g4_full_partial_rotary - 1.0).abs() > 1e-6
        {
            f.g4_full_partial_rotary
        } else {
            0.0
        },
        global_kv_heads: if f.g4_num_global_kv_heads > 0
            && f.g4_num_global_kv_heads != f.g4_num_key_value_heads
        {
            u32::try_from(f.g4_num_global_kv_heads).unwrap_or(0)
        } else {
            0
        },
        // The PLE's width and the KV sharing, both already parsed off the
        // descriptor and neither ever assigned. Zero for gemma-4-31b, which
        // is why "is gemma" cannot stand in for either.
        per_layer_emb_dim: u32::try_from(f.g4_per_layer_emb_dim.max(0)).unwrap_or(0),
        kv_shared_layers: u32::try_from(f.g4_num_kv_shared_layers.max(0)).unwrap_or(0),
        // gpt-oss's activation constants, when the config read as gpt-oss.
        // `go_num_hidden_layers` is the marker its own doc describes.
        swiglu_limit: if f.go_num_hidden_layers > 0 {
            f.go_swiglu_limit
        } else {
            0.0
        },
        // 1.702, the constant gpt-oss's SwiGLU is defined with. Not read from
        // the config because no config states it -- it is part of the
        // ACTIVATION, the way `silu`'s sigmoid is, and a deployment that
        // changed it would be a different activation.
        swiglu_alpha: 1.702,
        // The checkpoint's affine point, which nothing read either.
        //
        // `DecodeGeometry::default` is G64_B4, which is what every MLX 4-bit
        // llama ships -- so the default was right by coincidence for the one
        // checkpoint the reference gate runs, and wrong for anything else. An
        // 8-bit checkpoint would have been dequantised as 4-bit, which is not
        // a near miss: the symbol carries the point
        // (`affine_qmv_fast_bfloat16_gs_64_b_4`), so a wrong point either
        // names a pipeline that reads the wrong bytes or names nothing at all.
        //
        // Zero means the config declared no quantization, which is a dense
        // checkpoint; the default stands there rather than a `gs_0_b_0`
        // symbol no shader exports.
        quant: match (positive(f.quant_bits), positive(f.quant_group_size)) {
            (Some(bits), Some(group)) => AffineFormat { bits, group },
            _ => DecodeGeometry::default().quant,
        },
        // The rope BASE, which nothing read.
        //
        // Measured on `mlx-community/Llama-3.2-1B-Instruct-4bit`: its config
        // says 500000, `ModelFacts` reads 500000, and the geometry answered
        // **10000000** -- `DecodeGeometry::default`'s value, because no
        // assignment existed. Every deployment ran at one theta.
        //
        // Nothing fails on a wrong theta. The rotated channels come out wrong
        // by a factor that grows with position, so position ZERO is exactly
        // right and everything after it drifts -- which is why the reference
        // gate, whose first test is one token at position zero, agreed with
        // MLX while this was broken.
        //
        // The family block first, the top level second: `from_descriptor`
        // fills `ll_`/`go_`/`g4_` inside the branch it took and also reads the
        // flat key for every config, so preferring the block gets the value a
        // family-specific reader validated and falls back to the one every
        // config states.
        // Keyed on the MARKER field, not on "which value is non-zero": every
        // family block's theta has a non-zero DEFAULT (`ll_` is 500000), so
        // picking the first positive one gives llama's answer to a gpt-oss
        // config. The marker says which block `from_descriptor` actually
        // filled, and it is the same question `geometry_from_facts` already
        // asks three times above to project the shape.
        rope_theta: if f.go_num_hidden_layers > 0 {
            f.go_rope_theta
        } else if f.g4_num_hidden_layers > 0 {
            // gemma4 alternates a sliding base and a full one per layer. This
            // is the FULL one; the sliding layers take `rope_theta_sliding`,
            // and `LlamaLikeMetalFacts::rope_theta_at` picks between them off
            // the same window list that decides which layers slide.
            f.g4_rope_theta_full
        } else if f.ll_num_hidden_layers > 0 {
            f.ll_rope_theta
        } else {
            f.rope_theta
        },
        // The SLIDING base, where the single-base reading was wrong on fifty
        // of gemma-4-31b's sixty layers. Zero for every stack that states one
        // base, which `rope_theta_at` reads as "the same everywhere".
        rope_theta_sliding: if f.g4_num_hidden_layers > 0 {
            f.g4_rope_theta_sliding
        } else {
            0.0
        },
        // The rope RESCALING, when the config states one. `llama3` is the
        // only kind whose four numbers this reads; a config that states
        // another kind gets a factor of zero, which the derivation treats as
        // "no rescaling" rather than guessing at a shape it does not know.
        rope_freq_factor: if f.ll_rope_scaling_kind == "llama3" {
            f.ll_rope_scale
        } else {
            0.0
        },
        rope_low_freq_factor: f.ll_rope_low_freq_factor,
        rope_high_freq_factor: f.ll_rope_high_freq_factor,
        rope_original_max_position: f
            .ll_rope_original_max_position
            .try_into()
            .unwrap_or(0),
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

    // Can any Metal kernel READ this checkpoint's weights?
    //
    // The C++ shell asked this at load and refused by name
    // (`heap_bind.cpp:845-890`: *"no metal kernel here reads '<name>'"*).
    // Nothing asked after the port, so an unreadable scheme travelled all the
    // way to the first fire and surfaced as the runtime compiler declining a
    // mangled symbol -- loud, but after the weights are staged, and naming
    // `affine_qmv_fast_bfloat16_gs_128_b_8` rather than the two config keys
    // that chose it.
    //
    // Asked of the TABLE rather than a list here, so a point added or dropped
    // in `kernels-metal` moves this answer with it.
    //
    // Scope, measured: `affine_qmv_fast` is stamped over the whole
    // `(group x bits)` grid, so this does not catch a narrow kernel table --
    // it catches a config whose numbers are off the axes ENTIRELY, a group or
    // bit width nothing was ever stamped for. That is the case the C++
    // refusal existed for.
    if out.quant.is_set() && !out.quant.is_readable() {
        return Err(GeometryRefused(format!(
            "this checkpoint states group_size {} at {} bits and no metal \
             kernel here reads it -- `affine_qmv_fast` is instantiated at {}. \
             Refused at the geometry rather than at the first fire, where it \
             would surface as a missing symbol after the weights are staged",
            out.quant.group,
            out.quant.bits,
            AffineFormat::readable()
                .iter()
                .map(|f| format!("g{}/b{}", f.group, f.bits))
                .collect::<Vec<_>>()
                .join(", ")
        )));
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

    /// The rope base and the affine point come from the CONFIG.
    ///
    /// Both were unassigned, so every deployment ran at
    /// `DecodeGeometry::default`'s theta (1e7) and its point (g64/b4)
    /// whatever its config said. Measured before the fix:
    /// `Llama-3.2-1B-Instruct-4bit` states 500000 and the geometry answered
    /// 10000000.
    ///
    /// Neither FAILS when wrong, which is why they survived. A wrong theta
    /// rotates by a factor that grows with position, so position ZERO is
    /// exactly right -- and the reference gate's first test is one token at
    /// position zero, which agreed with MLX throughout. A wrong affine point
    /// names a symbol that reads the wrong bytes, and g64/b4 is what every
    /// MLX 4-bit llama ships, so the default was right by coincidence for the
    /// one checkpoint anything ran.
    ///
    /// Per FAMILY, because each block has its own non-zero default: reading
    /// "the first positive theta" gives llama's 500000 to a gpt-oss config
    /// that states 150000. The marker field says which block was filled.
    #[test]
    fn the_rope_base_and_the_affine_point_come_from_the_config() {
        let llama = ModelFacts {
            ll_num_hidden_layers: 24,
            ll_hidden_size: 1024,
            ll_num_attention_heads: 8,
            ll_num_key_value_heads: 2,
            ll_head_dim: 128,
            ll_vocab_size: 32_000,
            ll_intermediate_size: 3584,
            ll_rope_theta: 500_000.0,
            quant_bits: 4,
            quant_group_size: 64,
            ..ModelFacts::default()
        };
        let g = geometry_from_facts(&llama).expect("a llama config");
        assert_eq!(g.rope_theta, 500_000.0, "the llama block's theta");
        assert_eq!(g.quant, AffineFormat { bits: 4, group: 64 });

        // gpt-oss states its own of both, and `ll_rope_theta` still holds its
        // 500000 default -- so this is the case that "first positive wins"
        // gets wrong.
        let gptoss = ModelFacts {
            go_num_hidden_layers: 24,
            go_hidden_size: 2880,
            go_num_attention_heads: 64,
            go_num_key_value_heads: 8,
            go_head_dim: 64,
            go_vocab_size: 201_088,
            go_intermediate_size: 2880,
            go_num_local_experts: 32,
            go_num_experts_per_tok: 4,
            go_rope_theta: 150_000.0,
            quant_bits: 4,
            quant_group_size: 32,
            ..ModelFacts::default()
        };
        let g = geometry_from_facts(&gptoss).expect("a gpt-oss config");
        assert_eq!(
            g.rope_theta, 150_000.0,
            "gpt-oss got another family's theta, so the marker field is not \
             deciding"
        );
        assert_eq!(g.quant, AffineFormat { bits: 4, group: 32 });

        // A config that states no quantization is DENSE, and the default
        // stands rather than a `gs_0_b_0` symbol no shader exports.
        let dense = ModelFacts {
            quant_bits: 0,
            quant_group_size: 0,
            ..llama
        };
        assert_eq!(
            geometry_from_facts(&dense).expect("a dense config").quant,
            DecodeGeometry::default().quant
        );
    }

    /// gemma is a DIFFERENT model, and the geometry has to say so.
    ///
    /// Three facts hang off `gemma` that nothing else in the geometry
    /// carries — the `(1 + w)` norm scale, the four-norm sandwich, and the
    /// GEGLU activation — and all three were hardcoded to llama's answer in
    /// `model::text`. So a gemma checkpoint passed `serves`, loaded, fired,
    /// and produced finite plausible tokens from a model that was not the
    /// one on disk. This asserts the two families get DIFFERENT answers,
    /// because a test that only asks "does gemma resolve" passes either way.
    #[test]
    fn a_gemma_config_is_not_read_as_a_llama() {
        let llama = ModelFacts {
            ll_num_hidden_layers: 24,
            ll_hidden_size: 1024,
            ll_num_attention_heads: 8,
            ll_num_key_value_heads: 2,
            ll_head_dim: 128,
            ll_vocab_size: 32_000,
            ll_intermediate_size: 3584,
            ll_rope_theta: 500_000.0,
            ..ModelFacts::default()
        };
        let g = geometry_from_facts(&llama).expect("a llama config");
        assert!(!g.gemma, "a llama config is not gemma");

        // `mlx-community/gemma-4-31b-it-4bit`'s own `text_config`, and the
        // two zeros are the measurement: it states `num_kv_shared_layers: 0`
        // and `hidden_size_per_layer_input: 0`, so "is gemma" does NOT imply
        // "has a PLE" and neither can stand in for the other.
        let gemma = ModelFacts {
            g4_num_hidden_layers: 60,
            g4_hidden_size: 5376,
            g4_num_attention_heads: 32,
            g4_num_key_value_heads: 16,
            g4_head_dim: 256,
            vocab_size: 262_144,
            g4_intermediate_size: 21504,
            g4_final_softcap: 30.0,
            g4_attention_k_eq_v: true,
            g4_sliding_window: 1024,
            g4_full_attn_interval: 6,
            g4_rope_theta_full: 1_000_000.0,
            g4_rope_theta_sliding: 10_000.0,
            g4_per_layer_emb_dim: 0,
            g4_num_kv_shared_layers: 0,
            ..ModelFacts::default()
        };
        let g = geometry_from_facts(&gemma).expect("a gemma-4 config");
        assert!(g.gemma, "a gemma config that reads as a llama runs as one");
        assert_eq!(g.final_logit_softcap, 30.0);
        assert!(g.attention_k_eq_v);
        assert_eq!(g.per_layer_emb_dim, 0, "gemma-4-31b states no PLE width");
        assert_eq!(g.kv_shared_layers, 0, "gemma-4-31b shares no KV");
        // TWO bases, and the second is not a corner: gemma-4-31b's
        // `layer_types` slides fifty of its sixty layers, so reading one base
        // was wrong on 83% of the stack, by two orders of magnitude.
        assert_eq!(g.rope_theta, 1_000_000.0, "the FULL layers' base");
        assert_eq!(g.rope_theta_sliding, 10_000.0, "the SLIDING layers' base");

        // The gemma release that DOES state both, so the fields are read and
        // not merely defaulted to the same zero the hardcode produced.
        let ple = ModelFacts {
            g4_per_layer_emb_dim: 256,
            g4_num_kv_shared_layers: 4,
            ..gemma
        };
        let g = geometry_from_facts(&ple).expect("a gemma-4 config with a PLE");
        assert_eq!(g.per_layer_emb_dim, 256);
        assert_eq!(g.kv_shared_layers, 4);

        // A llama config states ONE base for every layer, which is what zero
        // means — and `rope_theta_at` reads it that way rather than as a base
        // of zero.
        let g = geometry_from_facts(&llama).expect("a llama config");
        assert_eq!(g.rope_theta_sliding, 0.0);
    }

    /// A quantization scheme no Metal kernel reads is refused at the
    /// GEOMETRY, by name.
    ///
    /// The C++ shell asked this at load — *"no metal kernel here reads
    /// '<name>'"* (`heap_bind.cpp:845-890`) — and nothing asked after the
    /// port. An unreadable scheme then travelled to the first fire and
    /// surfaced as the runtime compiler declining a mangled symbol: loud, but
    /// after the weights are staged, and naming the entrypoint rather than
    /// the two config keys that chose it.
    #[test]
    fn a_format_no_kernel_reads_is_refused_by_name() {
        let base = ModelFacts {
            ll_num_hidden_layers: 24,
            ll_hidden_size: 1024,
            ll_num_attention_heads: 8,
            ll_num_key_value_heads: 2,
            ll_head_dim: 128,
            ll_vocab_size: 32_000,
            ll_intermediate_size: 3584,
            ..ModelFacts::default()
        };

        // What the table actually instantiates `affine_qmv_fast` at.
        // Asserted rather than assumed, because the refusal above is only as
        // honest as this list — and a point dropped from `kernels-metal` has
        // to change this test rather than quietly narrow what the driver
        // serves.
        //
        // Measured: the dense GEMV covers the WHOLE grid, `_gs_{32,64,128}` ×
        // `_b_{4,8}`. So this check does not catch a narrow kernel table; it
        // catches a config whose numbers are off the axes entirely — a group
        // or bit width nothing was ever stamped for. That is the case the C++
        // refusal existed for, and it is the case that otherwise reaches the
        // first fire as a missing symbol.
        let readable = AffineFormat::readable();
        assert_eq!(
            readable.len(),
            6,
            "the dense GEMV is stamped over the whole (group x bits) grid; \
             if that changed, the refusal's scope changed with it: {readable:?}"
        );
        assert!(
            readable.contains(&AffineFormat { bits: 4, group: 64 }),
            "g64/b4 is what every MLX 4-bit checkpoint ships; if the table \
             stopped instantiating it this driver serves nothing"
        );

        for f in &readable {
            let g = geometry_from_facts(&ModelFacts {
                quant_bits: f.bits as i32,
                quant_group_size: f.group as i32,
                ..base.clone()
            });
            assert!(
                g.is_ok(),
                "g{}/b{} is instantiated and must not be refused",
                f.group,
                f.bits
            );
        }

        // A format nothing is compiled for. `group: 17` is not on any axis,
        // so no entrypoint can end with its suffix.
        let why = geometry_from_facts(&ModelFacts {
            quant_bits: 4,
            quant_group_size: 17,
            ..base
        })
        .expect_err("a format no kernel reads must be refused")
        .0;
        assert!(
            why.contains("group_size 17") && why.contains("no metal kernel here reads it"),
            "the refusal must name the checkpoint's own numbers: {why}"
        );
        assert!(
            why.contains("g64/b4"),
            "and say what IS instantiated, so the reader knows the shape of \
             the answer: {why}"
        );
    }
    /// gemma-4's PARTIAL rotary is carried, not flattened to a full one.
    ///
    /// Its full-attention layers state `partial_rotary_factor: 0.25`, so they
    /// rotate 128 of their 512 channels while its sliding layers rotate all
    /// 256 of theirs. `Geometry::rotary_dims` is one number for the fire, so
    /// this used to be refused; the rope rows now name the STATEMENT's own
    /// scalar with `grid_param`, and the extent travels per statement.
    #[test]
    fn a_partial_rotary_is_carried_because_the_row_names_the_statements_scalar() {
        let gemma = ModelFacts {
            g4_num_hidden_layers: 60,
            g4_hidden_size: 5376,
            g4_num_attention_heads: 32,
            g4_num_key_value_heads: 16,
            g4_head_dim: 256,
            vocab_size: 262_144,
            g4_intermediate_size: 21504,
            g4_sliding_window: 1024,
            g4_full_attn_interval: 6,
            ..ModelFacts::default()
        };
        // `ModelFacts::default` carries gemma's real 0.25, the way it carries
        // gemma's real rope bases — so this IS the shipped configuration and
        // not a constructed one.
        assert!((gemma.g4_full_partial_rotary - 0.25).abs() < 1e-6);
        let g = geometry_from_facts(&gemma).expect("a partial rotary is expressible now");
        assert!((g.full_partial_rotary - 0.25).abs() < 1e-6);

        // A stack that rotates the WHOLE head states zero, so nothing takes
        // the partial branch for a distinction it does not have.
        let g = geometry_from_facts(&ModelFacts {
            g4_full_partial_rotary: 1.0,
            ..gemma
        })
        .expect("full rotation is what every other family does");
        assert_eq!(g.full_partial_rotary, 0.0);
    }

    /// A gemma4 checkpoint states TWO attention geometries, and both are
    /// carried through to the text and the pool.
    ///
    /// Both releases on this machine have it: `global_head_dim: 512` against
    /// `head_dim: 256`, and `num_global_key_value_heads` of 4 (31b) or 2
    /// (26b) against `num_key_value_heads` of 16 or 8. Measured on the 31b's
    /// own tensors — layer 0's `q_norm` is `[256]` and layer 5's is `[512]`.
    ///
    /// The descriptor had parsed both since it was written and NOTHING read
    /// either, so all sixty layers ran at the sliding shape: half of each
    /// full layer's Q, a quarter past the end of its K.
    #[test]
    fn a_gemma_configs_second_head_shape_is_carried_and_not_flattened() {
        let base = ModelFacts {
            g4_num_hidden_layers: 60,
            g4_hidden_size: 5376,
            g4_num_attention_heads: 32,
            g4_num_key_value_heads: 16,
            g4_head_dim: 256,
            vocab_size: 262_144,
            g4_intermediate_size: 21504,
            g4_sliding_window: 1024,
            g4_full_attn_interval: 6,
            ..ModelFacts::default()
        };
        // `mlx-community/gemma-4-31b-it-4bit` as it actually reads.
        let g = geometry_from_facts(&ModelFacts {
            g4_global_head_dim: 512,
            g4_num_global_kv_heads: 4,
            ..base.clone()
        })
        .expect("a gemma-4 config is decodable now that both shapes are carried");
        assert_eq!(g.global_head_dim, 512);
        assert_eq!(g.global_kv_heads, 4);
        assert_eq!(g.head_dim, 256, "the SLIDING shape stays the stack's");
        assert_eq!(g.n_kv_heads, 16);

        // A config that RESTATES its own head_dim under the global key has
        // one shape, and must answer zero -- otherwise every layer takes the
        // `is_full_attention` branch for a distinction that does not exist.
        let g = geometry_from_facts(&ModelFacts {
            g4_global_head_dim: 256,
            g4_num_global_kv_heads: 16,
            ..base
        })
        .expect("one shape restated twice is still one shape");
        assert_eq!(g.global_head_dim, 0);
        assert_eq!(g.global_kv_heads, 0);
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
