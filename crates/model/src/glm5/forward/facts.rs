//! glm5's shape, as the declaration needs it.
//!
//! Read off `driver-cuda/csrc/src/model/glm5/` — `glm5_forward.cpp`'s
//! `cfg.` reads for the dims, `glm5.hpp`'s weight struct for which
//! tensors a layer has, and `Lw.is_moe` for the layer schedule.
//!
//! Two things here are glm5's own and neither llama_like nor qwen3_5 has
//! either: MLA (the query and the KV both go through a LATENT of their
//! own rank, so `hidden` never appears in the attention core's widths)
//! and DSA (a lightning indexer that scores pages and hands attention a
//! top-k mask, which is a SECOND, smaller attention beside the real one).

use serde::{Deserialize, Serialize};

/// The MLA attention block's dims. `qk_nope_head_dim + qk_rope_head_dim`
/// is the query width per head; the CACHE stores the latent plus the rope
/// half, which is why `kv_lora_rank + qk_rope_head_dim` is its own number
/// and not derivable from the query's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Glm5MlaFacts {
    pub hidden: u32,
    pub heads: u32,
    /// `q_lora_rank`: the query's own latent. glm5 always has one.
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
}

impl Glm5MlaFacts {
    /// The per-head query width the `q_b` projection produces.
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// The width `q_b_proj` writes: every head's nope+rope halves.
    pub fn q_b_width(&self) -> u32 {
        self.heads * self.qk_head_dim()
    }
    /// The width `kv_a_proj_with_mqa` writes: the latent plus ONE shared
    /// rope half (MQA — the rope key is not per-head).
    pub fn kv_a_width(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
    /// The width the attention core's output carries before `o_proj`.
    pub fn v_width(&self) -> u32 {
        self.heads * self.v_head_dim
    }
}

/// The DSA lightning indexer. A separate small attention whose only
/// output is a top-k page mask for the real one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Glm5DsaFacts {
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
}

/// The MoE block. `first_k_dense` is not a config field in the driver —
/// it reads `Lw.is_moe` per layer — but the schedule it encodes is a
/// prefix of dense layers, so the declaration states the prefix length
/// and a layer asks whether it is past it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Glm5MoeFacts {
    pub hidden: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub moe_intermediate: u32,
    /// `n_shared_experts * moe_intermediate`; zero means no shared expert.
    pub shared_intermediate: u32,
    /// `kernels::moe::moe_aligned_block(maxR, num_experts)`, resolved once at
    /// workspace setup. A deployment fact rather than a per-fire number,
    /// which is why it can shape a `Dim`.
    pub aligned_block: u32,
}

/// The whole family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Glm5Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    /// The DENSE MLP width, used by the first `dense_layers` layers.
    pub dense_intermediate: u32,
    /// How many leading layers take the dense MLP instead of the MoE.
    pub dense_layers: u32,
    pub attn: Glm5MlaFacts,
    pub dsa: Glm5DsaFacts,
    pub moe: Glm5MoeFacts,
}

impl Glm5Facts {
    pub fn is_moe_layer(&self, l: u32) -> bool {
        model_compiler::facts::after_dense_prefix(self.dense_layers, l)
    }
}

impl Glm5Facts {
    /// `zai-org/GLM-5-106B-A12B`, read off its `config.json` the way the
    /// driver reads it. A fixture rather than a guess: every dim here is
    /// a `cfg.` field `glm5_forward.cpp` actually consumes.
    pub fn glm5_106b_a12b() -> Self {
        Glm5Facts {
            layers: 46,
            vocab: 151552,
            hidden: 4096,
            dense_intermediate: 10944,
            // The first three layers are dense; the rest route.
            dense_layers: 3,
            attn: Glm5MlaFacts {
                hidden: 4096,
                heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
            },
            dsa: Glm5DsaFacts {
                index_n_heads: 64,
                index_head_dim: 128,
                index_topk: 2048,
            },
            moe: Glm5MoeFacts {
                hidden: 4096,
                num_experts: 128,
                top_k: 8,
                moe_intermediate: 1408,
                shared_intermediate: 1408,
                aligned_block: 16,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived widths, against the weight shapes `glm5.hpp` documents
    /// in its own comments — the only cross-check available without the
    /// checkpoint, and enough to catch a transposed pair.
    #[test]
    fn the_derived_widths_match_the_weight_comments() {
        let f = Glm5Facts::glm5_106b_a12b();
        // `q_b_proj` is `[local_heads*(nope+rope), q_lora_rank]`.
        assert_eq!(f.attn.qk_head_dim(), 192);
        assert_eq!(f.attn.q_b_width(), 96 * 192);
        // `kv_a_proj_with_mqa` is `[kv_lora_rank+rope, H]` — ONE rope
        // half for every head, which is the MQA in its name.
        assert_eq!(f.attn.kv_a_width(), 512 + 64);
        // `o_proj` is `[H, local_heads*v_dim]`.
        assert_eq!(f.attn.v_width(), 96 * 128);
    }

    #[test]
    fn the_dense_prefix_is_a_prefix() {
        let f = Glm5Facts::glm5_106b_a12b();
        assert!(!f.is_moe_layer(0));
        assert!(!f.is_moe_layer(2));
        assert!(f.is_moe_layer(3));
        assert!(f.is_moe_layer(f.layers - 1));
    }
}
