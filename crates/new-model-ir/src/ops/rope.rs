use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Rope {
    Full {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
        #[out(alias = q)]
        q_out: ValueId,
        #[out(alias = k)]
        k_out: ValueId,
    },
    Partial {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        #[out(alias = q)]
        q_out: ValueId,
        #[out(alias = k)]
        k_out: ValueId,
    },
    PartialQ {
        q: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        #[out(alias = q)]
        q_out: ValueId,
    },
    /// Partial rope over the last `rotary_dim` lanes of each head.
    PartialLast {
        q: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
        #[out(alias = q)]
        q_out: ValueId,
    },
    Yarn {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
        #[out(alias = q)]
        q_out: ValueId,
        #[out(alias = k)]
        k_out: ValueId,
    },
}
