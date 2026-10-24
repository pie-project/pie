use poem_exec::{DispatchSampling, KernelError};
use poem_ir::Sampling;

use crate::run::Run;

impl DispatchSampling for Run<'_> {
    fn dispatch(&mut self, op: &Sampling) -> Result<(), KernelError> {
        self.sampling(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn sampling(&mut self, op: &Sampling) -> Result<(), kernels_cuda::Error> {
        match op {
            Sampling::TopK {
                x,
                k,
                values,
                indices,
            } => kernels_cuda::sampling::top_k(
                self.ctx(),
                self.tensor(*x),
                *k,
                &mut self.tensor(*values),
                &mut self.tensor(*indices),
            ),
            Sampling::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    kernels_cuda::sampling::argmax(
                        self.ctx(),
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        &mut self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            Sampling::MarkovWalk {
                logits,
                tokens,
                positions,
                ends,
                embed,
                head,
                block,
                biased,
                picks,
            } => kernels_cuda::sampling::markov_walk(
                self.ctx(),
                self.tensor(*logits),
                self.tensor(*tokens),
                self.tensor(*positions),
                self.tensor(*ends),
                self.tensor(*embed),
                self.tensor(*head),
                *block,
                &mut self.tensor(*biased),
                &mut self.tensor(*picks),
            ),
            Sampling::SelectorWalk {
                cand,
                unary,
                hp,
                tokens,
                pred,
                succ,
                first,
                picks,
            } => kernels_cuda::sampling::selector_walk(
                self.ctx(),
                self.ragged(*cand),
                self.tensor(*unary),
                hp.map(|hp| self.tensor(hp)),
                self.tensor(*tokens),
                self.tensor(*pred),
                self.tensor(*succ),
                *first,
                &mut self.tensor(*picks),
            ),
        }
    }
}
