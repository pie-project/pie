//! gemma-2's shape.
//!
//! The simplest family in the tree, and the last one that existed only as
//! hand-written C++. Plain attention, a geglu MLP, and three things that
//! are gemma's rather than anyone else's: a PAIR of norms around each
//! block (pre and post, not just pre), a per-layer alternating sliding
//! window, and softcaps on both the attention logits and the final ones.

use serde::{Deserialize, Serialize};

/// The attention block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub fn q_width(&self) -> u32 {
        self.heads * self.head_dim
    }
    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

/// The whole family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gemma2Facts {
    pub layers: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub tied_embeddings: bool,
    /// `cfg.final_logit_softcap` — this one IS a launch, at the end.
    pub final_logit_softcap: bool,
    /// `cfg.per_layer_window_left`: gemma-2 ALTERNATES local and global
    /// attention. Carried per layer rather than as an interval, because
    /// what the driver reads is the list.
    pub window_left: Vec<i32>,
    pub attn: Gemma2AttnFacts,
}

impl Gemma2Facts {
    /// A layer attends the whole context when its window is negative.
    pub fn is_global(&self, l: u32) -> bool {
        self.window_left[l as usize] < 0
    }

    /// `google/gemma-2-9b-it`.
    pub fn gemma_2_9b() -> Self {
        // Every other layer is global; the rest see 4096 back.
        let window_left = (0..42).map(|l| if l % 2 == 1 { -1 } else { 4096 }).collect();
        Gemma2Facts {
            layers: 42,
            vocab: 256000,
            hidden: 3584,
            intermediate: 14336,
            tied_embeddings: true,
            final_logit_softcap: true,
            window_left,
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
        assert_eq!(f.window_left.len() as u32, f.layers);
        assert!((0..f.layers).any(|l| f.is_global(l)));
        assert!((0..f.layers).any(|l| !f.is_global(l)));
    }
}
