use poem_exec::{DispatchConv, DispatchFused, DispatchProbe, KernelError};
use poem_ir::{Conv, Fused, Operands};

use crate::run::Run;

impl DispatchFused for Run<'_> {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl DispatchConv for Run<'_> {
    fn dispatch(&mut self, op: &Conv) -> Result<(), KernelError> {
        use kernels_cerebras::conv;
        self.ctx().scope(op.name());
        match op {
            Conv::Cells {
                tgrid,
                positions,
                axis,
                y,
            } => conv::cells(
                self.ctx(),
                self.tensor(*tgrid),
                self.tensor(*positions),
                *axis,
                self.tensor(*y),
                self.grid_base(*tgrid, *y),
            ),
            Conv::Patchify {
                x,
                grid,
                p,
                tgrid,
                y,
            } => conv::patchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *p,
                self.tensor(*tgrid),
                self.tensor(*y),
                [self.grid_base(*grid, *x), self.grid_base(*tgrid, *y)],
            ),
            Conv::Unpatchify {
                x,
                tgrid,
                p,
                grid,
                y,
            } => conv::unpatchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*tgrid),
                *p,
                self.tensor(*grid),
                self.tensor(*y),
                [self.grid_base(*tgrid, *x), self.grid_base(*grid, *y)],
            ),
            _ => return Err(KernelError::Unsupported { op: op.name() }),
        }
        .map_err(crate::error::kernel)
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
