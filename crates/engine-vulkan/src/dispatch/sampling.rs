use kernels_vulkan::layout;
use poem_exec::{DispatchSampling, KernelError};
use poem_ir::{Operands, Sampling};

use crate::run::Run;

impl DispatchSampling for Run<'_> {
    fn dispatch(&mut self, op: &Sampling) -> Result<(), KernelError> {
        self.sampling(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn sampling(&mut self, op: &Sampling) -> Result<(), kernels_vulkan::Error> {
        match op {
            Sampling::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    layout::argmax(
                        self.ctx(),
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            Sampling::TopK { .. } | Sampling::MarkovWalk { .. } | Sampling::SelectorWalk { .. } => {
                Err(kernels_vulkan::Error::Unsupported { op: op.name() })
            }
        }
    }
}
