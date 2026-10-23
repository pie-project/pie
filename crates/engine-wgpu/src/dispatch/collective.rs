use kernels_slang::collective::band;
use poem_exec::{DispatchCollective, KernelError};
use poem_ir::Collective;

use crate::encode::{from_slang, slang_tensor};
use crate::run::Run;

impl DispatchCollective for Run<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.collective(op)
            .map_err(from_slang)
            .map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn collective(&mut self, op: &Collective) -> Result<(), kernels_slang::Error> {
        let t = |id| slang_tensor(self.tensor(id));
        match op {
            Collective::AllReduce { buf, buf_out: _ } => band::all_reduce(&self.slang(), t(*buf)),
            Collective::AllGather { x, y } => band::all_gather(&self.slang(), t(*x), t(*y)),
            Collective::ReduceScatter { x, y } => band::reduce_scatter(&self.slang(), t(*x), t(*y)),
        }
    }
}
