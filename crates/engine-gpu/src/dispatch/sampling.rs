use poem_exec::{DispatchSampling, KernelError};
use poem_ir::{Operands, Sampling};

use crate::backend::Backend;
use crate::run::Run;

impl<B: Backend> DispatchSampling for Run<'_, B> {
    fn dispatch(&mut self, op: &Sampling) -> Result<(), KernelError> {
        self.sampling(op).map_err(crate::error::kernel)
    }
}

impl<B: Backend> Run<'_, B> {
    fn sampling(&mut self, op: &Sampling) -> Result<(), kernels_slang::Error> {
        match op {
            Sampling::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    kernels_slang::sampling::argmax::argmax(
                        self.ctx,
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            Sampling::TopK { .. } | Sampling::MarkovWalk { .. } | Sampling::SelectorWalk { .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }
        }
    }
}
