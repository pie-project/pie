use model_exec::{DispatchCustomCuda, DispatchProbe, DispatchSpatial, KernelError};
use model_ir::{CustomCuda, Operands, Spatial};

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl DispatchSpatial for Run<'_> {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        Err(KernelError::Unsupported { op: op.name() })
    }
}

/// A probed value is noted right after the node that writes it ran; the
/// whole root it lives in comes back as an extra output of the fire.
impl DispatchProbe for Run<'_> {
    fn probe(&mut self, node: &model_ir::Node) {
        if self.probes().is_empty() {
            return;
        }
        let mut outputs = Vec::new();
        node.op.outputs(&mut outputs);
        for value in outputs {
            if !self.probes().contains(&value) {
                continue;
            }
            let t = self.tensor(value);
            self.record_probe(value, t);
        }
    }
}
