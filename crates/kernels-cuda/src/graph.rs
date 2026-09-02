//! Launches that set a CUDA graph's own conditional control flow (device-side
//! stores into a `cudaGraphSetConditional` handle).

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, symbol};

const FILE: &str = "graph/conditional.cuh";

/// One thread, and it is the whole geometry: the kernel makes one store.
fn once() -> Launch {
    Launch::grid([1, 1, 1], [1, 1, 1])
}

/// Whether a launch of a setter arms it or merely warms it.
///
/// Warm loads the kernel module without storing; module load is host work,
/// which is disallowed during stream capture, so a `Warm` fire runs eagerly
/// once before the captured `Set` launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Compile and load, store nothing.
    Warm,
    /// Store the predicate into the handle.
    Set,
}

impl Arm {
    pub(crate) const fn armed(self) -> i32 {
        match self {
            Arm::Warm => 0,
            Arm::Set => 1,
        }
    }
}

/// Sets a conditional handle from a window's row count.
///
/// `indptr` is the device address of a window's rebased row CSR; `lanes` its
/// lane count. The handle is set to `indptr[lanes] != 0`. `absent` says what
/// a null `indptr` means: `true` runs the body anyway, `false` skips it.
///
/// # Errors
///
/// Whatever the launch refused, tagged with this op's name.
pub fn set_conditional(
    ctx: &Ctx,
    handle: u64,
    indptr: u64,
    lanes: u32,
    absent: bool,
    arm: Arm,
    win: u64,
) -> Result<(), Error> {
    const OP: &str = "graph.set_conditional";
    let lanes = i32::try_from(lanes).unwrap_or(i32::MAX);
    ctx.fire(
        OP,
        Fire::at(FILE, symbol("::pie::graph::set_conditional")).apply(once()),
        &[
            handle.arg(),
            crate::ArgValue::Ptr(indptr),
            lanes.arg(),
            u32::from(absent).arg(),
            arm.armed().arg(),
            // win[2] is this fire's live lane count; `lanes` is the count seen
            // at capture time, which a replay may not match.
            crate::ArgValue::Ptr(win),
        ],
    )
}

/// Same store, from a device byte the caller staged rather than a window table.
///
/// # Errors
///
/// Whatever the launch refused, tagged with this op's name.
pub fn set_conditional_byte(
    ctx: &Ctx,
    handle: u64,
    live: u64,
    absent: bool,
    arm: Arm,
) -> Result<(), Error> {
    const OP: &str = "graph.set_conditional_byte";
    ctx.fire(
        OP,
        Fire::at(FILE, symbol("::pie::graph::set_conditional_byte")).apply(once()),
        &[
            handle.arg(),
            crate::ArgValue::Ptr(live),
            u32::from(absent).arg(),
            arm.armed().arg(),
        ],
    )
}

/// Sets a switch handle to this arm's index, if this arm has rows.
///
/// The `SWITCH` twin of [`set_conditional`]: the handle holds an arm index in
/// `0..arms`, and any value at or past `arms` means no body runs (the
/// recorder mints that as the default, so an empty fire needs no store).
///
/// Called once per arm, each with its own `indptr`; a null `indptr` (zero)
/// means this arm stands down.
///
/// # Errors
///
/// Whatever the launch refused, tagged with this op's name.
pub fn set_switch(
    ctx: &Ctx,
    handle: u64,
    arm: u32,
    indptr: u64,
    lanes: u32,
    warm: Arm,
    win: u64,
) -> Result<(), Error> {
    const OP: &str = "graph.set_switch";
    let lanes = i32::try_from(lanes).unwrap_or(i32::MAX);
    ctx.fire(
        OP,
        Fire::at(FILE, symbol("::pie::graph::set_switch")).apply(once()),
        &[
            handle.arg(),
            arm.arg(),
            crate::ArgValue::Ptr(indptr),
            lanes.arg(),
            warm.armed().arg(),
            // Same live-lane-count seat as set_conditional's win arg.
            crate::ArgValue::Ptr(win),
        ],
    )
}
