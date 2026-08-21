use core::ffi::c_void;
use kernels::{Bind, Fire};

use crate::jit::{Ctx, Launch, Root};
use kernels::Refusal;

pub static ROOT: Root = Root::new("graph/supergraph.cuh");

const ARM: Launch = Launch::grid([1, 1, 1], [1, 1, 1]);

pub fn supergraph_set_cond(
    ctx: &Ctx<'_>,
    handle: usize,
    preds: *const c_void,
    slot: i32,
) -> Result<(), Refusal> {
    arm(
        ctx,
        "::pie::graph::supergraph_set_cond",
        handle,
        preds,
        slot,
    )
}

pub fn supergraph_set_switch(
    ctx: &Ctx<'_>,
    handle: usize,
    preds: *const c_void,
    slot: i32,
) -> Result<(), Refusal> {
    arm(
        ctx,
        "::pie::graph::supergraph_set_switch",
        handle,
        preds,
        slot,
    )
}

#[cfg(feature = "_cuda")]
pub fn warm() -> Result<(), Refusal> {
    for instantiation in [
        "::pie::graph::supergraph_set_cond",
        "::pie::graph::supergraph_set_switch",
    ] {
        if crate::jit::cache::resolve(&ROOT, instantiation).is_err() {
            return Err(Refusal::Device {
                why: "an arming kernel would not compile or load; see the log",
            });
        }
    }
    Ok(())
}

#[cfg(not(feature = "_cuda"))]
#[allow(clippy::unnecessary_wraps)]
pub fn warm() -> Result<(), Refusal> {
    Ok(())
}

fn arm(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    handle: usize,
    preds: *const c_void,
    slot: i32,
) -> Result<(), Refusal> {
    if slot < 0 {
        return Err(Refusal::Narrow {
            what: "the predicate slot",
            at: i64::from(slot),
        });
    }

    ctx.fire(
        Fire::at("graph/supergraph.cuh", instantiation).apply(ARM),
        &[handle.arg(), preds.cast::<u8>().arg(), slot.arg()],
    )
}
