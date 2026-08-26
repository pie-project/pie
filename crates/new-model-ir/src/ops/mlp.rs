use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Mlp {
    Swiglu {
        packed: ValueId,
        intermediate: u32,
        #[out]
        y: ValueId,
    },
    SwigluClamp {
        packed: ValueId,
        intermediate: u32,
        limit: f32,
        #[out]
        y: ValueId,
    },
    SwigluClampAlpha {
        packed: ValueId,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        #[out]
        y: ValueId,
    },
    GegluTanh {
        gate: ValueId,
        up: ValueId,
        #[out]
        y: ValueId,
    },
    GegluTanhPacked {
        packed: ValueId,
        intermediate: u32,
        #[out]
        y: ValueId,
    },
    Situ {
        packed: ValueId,
        intermediate: u32,
        beta: f32,
        up_cap: Option<f32>,
        #[out]
        y: ValueId,
    },
}
