use poem_dsl::{ForwardHybrid, HybridSpec, Input, Value};

use super::model::Model;

impl ForwardHybrid for Model {
    fn caches(&self) -> HybridSpec {
        self.trunk.caches()
    }

    fn forward(&self, inputs: Input) -> Value {
        self.trunk.forward(inputs)
    }
}
