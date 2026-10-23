use kernels_cerebras::sampling;
use poem_exec::{DispatchSampling, KernelError};
use poem_ir::{Operands, Sampling};

use crate::run::Run;

impl DispatchSampling for Run<'_> {
    fn dispatch(&mut self, op: &Sampling) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.sampling(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn sampling(&mut self, op: &Sampling) -> Result<(), kernels_cerebras::Error> {
        match op {
            Sampling::TopK {
                x,
                k,
                values,
                indices,
            } => sampling::topk(
                self.ctx(),
                self.tensor(*x),
                *k,
                self.tensor(*values),
                self.tensor(*indices),
            ),
            Sampling::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    sampling::argmax(
                        self.ctx(),
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            Sampling::SelectorWalk {
                cand,
                unary,
                hp,
                tokens,
                pred,
                succ,
                first,
                picks,
            } => sampling::selector_walk::selector_walk(
                self.ctx(),
                self.ragged(*cand),
                self.tensor(*unary),
                hp.map(|h| self.tensor(h)),
                self.tensor(*tokens),
                self.tensor(*pred),
                self.tensor(*succ),
                *first,
                self.tensor(*picks),
            ),
            Sampling::MarkovWalk { .. } => {
                Err(kernels_cerebras::Error::Unsupported { op: op.name() })
            }
        }
    }
}
