use poem_exec::{DispatchCollective, KernelError};
use poem_ir::{Collective, Operands};

use crate::run::Run;

/// One replica: every collective is refused by name.
impl DispatchCollective for Run<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        Err(KernelError::Unsupported { op: op.name() })
    }
}
