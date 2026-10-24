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
            Collective::AllReduce { buf, buf_out: _ } => collective::all_reduce(self.tensor(*buf)),
            Collective::AllGather { x, y } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = collective::all_gather(l.f(), x_v, o_y.landing())?;
                l.land(o_y, v)
            }),
            Collective::ReduceScatter { x, y } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = collective::reduce_scatter(l.f(), x_v, o_y.landing())?;
                l.land(o_y, v)
            }),
        }
    }
}
