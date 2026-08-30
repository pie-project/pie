//! `Collective`: the cross-rank reductions and gathers.

use kernels_cuda::collective;
use model_exec::{DispatchCollective, KernelError};
use model_ir::Collective;

use crate::run::Run;

impl DispatchCollective for Run<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.collective(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-cuda`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn collective(&mut self, op: &Collective) -> Result<(), kernels_cuda::Error> {
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
