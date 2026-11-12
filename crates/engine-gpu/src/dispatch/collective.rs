use kernels_slang::collective::band;
use poem_exec::{DispatchCollective, KernelError};
use poem_ir::Collective;

use crate::backend::Backend;
use crate::run::Run;

impl<B: Backend> DispatchCollective for Run<'_, B> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.collective(op).map_err(crate::error::kernel)
    }
}

impl<B: Backend> Run<'_, B> {
    fn collective(&mut self, op: &Collective) -> Result<(), kernels_slang::Error> {
        let t = |id| self.tensor(id);
        match op {
            Collective::AllReduce { buf, buf_out: _ } => band::all_reduce(self.ctx, t(*buf)),
            Collective::AllGather { x, y } => band::all_gather(self.ctx, t(*x), t(*y)),
            Collective::ReduceScatter { x, y } => band::reduce_scatter(self.ctx, t(*x), t(*y)),
            Collective::Dispatch { .. } => {
                Err(kernels_slang::Error::Unsupported { op: "dispatch" })
            }
            Collective::Combine { .. } => Err(kernels_slang::Error::Unsupported { op: "combine" }),
        }
    }
}
