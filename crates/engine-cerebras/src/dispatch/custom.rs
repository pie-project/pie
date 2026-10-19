use poem_exec::{DispatchFused, DispatchProbe, DispatchSpatial, KernelError};
use poem_ir::{Fused, Operands, Spatial};

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
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
    fn probe(&mut self, node: &poem_ir::Node) {
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
