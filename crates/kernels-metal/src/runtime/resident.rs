//! A probe of how many full workgroups the GPU runs at once
//! (`resident_groups`): what the compiler prices a device's parallelism at
//! where the device does not say its cores.

use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "runtime/resident.metal";

/// The threads a probe group takes: a full workgroup.
pub const GROUP_THREADS: u32 = 1024;

/// Launches `groups` full workgroups, each working `iters` steps, counting
/// in `counters` (two u32: the groups in now, the most in at once; zeroed by
/// the caller). Read `counters[1]` once the launch completes.
pub fn resident_groups(
    ctx: &Ctx<'_>,
    counters: Tensor,
    sink: Tensor,
    groups: u32,
    iters: u32,
) -> Result<(), Error> {
    if groups == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, "resident_groups").apply(Grid::of(
            [GROUP_THREADS * groups, 1, 1],
            [GROUP_THREADS, 1, 1],
        )),
        &[counters.arg_mut(), iters.arg(), sink.arg_mut()],
    )
}
