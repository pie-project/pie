//! deepseek_v4's shape.
//!
//! Neither MLA nor a plain attention. Two things are this family's own
//! and no other declared family has either:
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
//!   `lse_log2_to_ln` are statements here and nowhere else.

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
    pub fn q_width(&self) -> u32 {
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dsv4MoeFacts {
    pub num_experts: u32,
    pub top_k: u32,
    pub moe_intermediate: u32,
    /// `cfg.swiglu_limit`, the clamp the activation applies.
    pub swiglu_limit_milli: u32,
    /// The router is a HASH TABLE lookup rather than a learned gate on
    /// some deployments (`launch_hash_route_lookup`).
    pub hash_routed: bool,
}

/// The whole family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dsv4Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub dense_intermediate: u32,
    pub dense_layers: u32,
    pub attn: Dsv4AttnFacts,
    pub hc: Dsv4HcFacts,
    pub moe: Dsv4MoeFacts,
}

impl Dsv4Facts {
    pub fn is_moe_layer(&self, l: u32) -> bool {
        l >= self.dense_layers
    }

    pub fn dsv4_synthetic() -> Self {
        Dsv4Facts {
            layers: 6,
            vocab: 129280,
            hidden: 2048,
            dense_intermediate: 5632,
            dense_layers: 1,
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

    /// The hyper-connection is a rank-K residual, and K > 1 is what makes
    /// it one. A fixture at 1 would lower the same as an ordinary
    /// residual and prove nothing about the scheme.
    #[test]
    fn the_residual_is_actually_rank_k() {
        let f = Dsv4Facts::dsv4_synthetic();
        assert!(f.hc.mult > 1);
    }
}
