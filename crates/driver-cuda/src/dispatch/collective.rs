//! `Collective`: the cross-rank reductions and gathers.

use kernels::{DispatchCollective, KernelError};
use kernels_cuda::collective;
use model_ir::Collective;

use crate::run::Run;

impl DispatchCollective for Run<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        match op {
            Collective::AllReduce { buf, buf_out: _ } => {
                collective::all_reduce(self.ctx(), &mut self.tensor(*buf))
            }
            Collective::AllGather { x, y } => {
                collective::all_gather(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Collective::ReduceScatter { x, y } => {
                collective::reduce_scatter(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
        }
    }
}
