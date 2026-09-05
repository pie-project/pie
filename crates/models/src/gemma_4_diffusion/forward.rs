use model_dsl::{ForwardHybrid, HybridSpec, Input, Value};

pub use crate::gemma_4::forward::Facts;

use super::model::Model;

/// The encoder reading: Gemma 4's own forward, unchanged. The denoise class
/// joins here.
impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        self.trunk.caches()
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        self.trunk.forward(inputs)
    }
}
