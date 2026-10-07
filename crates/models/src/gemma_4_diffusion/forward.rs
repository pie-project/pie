use poem_dsl::{ForwardHybrid, HybridSpec, Input, Value};

pub use crate::gemma_4::forward::Facts;

use super::model::Model;

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        self.trunk.caches()
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        self.trunk.forward(inputs)
    }
}
