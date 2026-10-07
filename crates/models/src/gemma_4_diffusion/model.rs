use poem_dsl::Dtype;

use crate::gemma_4;

pub const CANVAS: u32 = 256;

pub const HIDDEN: u32 = 2816;

pub struct Model {
    pub trunk: gemma_4::model::Model,
}

impl Model {
    pub fn a4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model {
            trunk: gemma_4::model::Model::a4b_diffusion(w, kv, tp),
        }
    }

    pub fn a4b_experts(w: Dtype, xw: Dtype, kv: Dtype, tp: u32) -> Model {
        Model {
            trunk: gemma_4::model::Model::a4b_diffusion_experts(w, xw, kv, tp),
        }
    }

    pub fn a4b_experts_self_cond(w: Dtype, xw: Dtype, sw: Dtype, kv: Dtype, tp: u32) -> Model {
        Model {
            trunk: gemma_4::model::Model::a4b_diffusion_experts_self_cond(w, xw, sw, kv, tp),
        }
    }
}
