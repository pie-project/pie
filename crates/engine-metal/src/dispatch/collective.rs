//! The `collective` family: `impl DispatchCollective for Run<'_>`.

use kernels_metal::collective;
use model_exec::{DispatchCollective, KernelError};
use model_ir::Collective;

use crate::run::Run;

impl DispatchCollective for Run<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.collective(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-metal`'s error vocabulary, lifted by
    /// [`kernel`](crate::error::kernel) above.
    fn collective(&mut self, op: &Collective) -> Result<(), kernels_metal::Error> {
        match op {
            Collective::AllReduce { buf, buf_out: _ } => {
                collective::all_reduce(self.ctx(), self.tensor(*buf))
            }
            Collective::AllGather { x, y } => {
                collective::all_gather(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
            Collective::ReduceScatter { x, y } => {
                collective::reduce_scatter(self.ctx(), self.tensor(*x), self.tensor(*y))
            }
        }
    }
}
