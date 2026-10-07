use kernels_xla::collective;
use poem_exec::{DispatchCollective, KernelError};
use poem_ir::Collective;

use poem_ir::Operands;

use crate::run::Run;

impl DispatchCollective for Run<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.collective(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn collective(&mut self, op: &Collective) -> Result<(), kernels_xla::Error> {
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
