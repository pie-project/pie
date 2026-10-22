use kernels_xla::{attn, layout};
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
    fn sampling(&mut self, op: &Sampling) -> Result<(), kernels_xla::Error> {
        match op {
            Sampling::TopK {
                x,
                k,
                values,
                indices,
            } => layout::topk(
                self.ctx(),
                self.tensor(*x),
                *k,
                self.tensor(*values),
                self.tensor(*indices),
            ),
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
            Sampling::SelectorWalk {
                cand,
                unary,
                hp,
                tokens,
                pred,
                succ,
                first,
                picks,
            } => attn::ple::selector_walk(
                self.ctx(),
                self.ragged(*cand),
                self.tensor(*unary),
                // The candidates are readout-rowed (a block drafter reads
                // every row back), the tokens and the projected hidden
                // token-rowed, padded past them: the GPU kernel reads the
                // first rows.
                hp.map(|hp| self.first_rows(self.tensor(hp), *cand)),
                self.first_rows(self.tensor(*tokens), *cand),
                self.tensor(*pred),
                self.tensor(*succ),
                *first,
                self.tensor(*picks),
            ),
            Sampling::MarkovWalk { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
        }
    }
}
