use std::fmt::Write as _;

use crate::plan::LoadPlan;

pub fn describe(plan: &LoadPlan) -> String {
    let rewrites: usize = plan.passes.iter().map(|pass| pass.rewrites).sum();
    let mut out = String::new();
    let _ = write!(
        out,
        "load_plan(source_tensors={}, tensors={}, \
         buffers={}, instrs={}, schedule={}, passes={}, \
         rewrites={}, persistent_bytes={}, scratch_bytes={}, \
         read_bytes={}, write_bytes={})",
        plan.sources.len(),
        plan.tensors.len(),
        plan.buffers.len(),
        plan.instrs.len(),
        plan.schedule.len(),
        plan.passes.len(),
        rewrites,
        plan.memory.persistent_bytes,
        plan.memory.scratch_bytes,
        plan.memory.checkpoint_read_bytes,
        plan.memory.device_write_bytes,
    );
    out
}
