//! `graph`: the launches a CUDA graph's own control flow is made of.
//!
//! **NOT A MODEL OP, AND THAT IS WHY IT IS ITS OWN FAMILY.** Every other
//! family here computes something a `Trace` node names; this one computes
//! nothing and appears in no plan. It exists because
//! `cudaGraphSetConditional` is DEVICE-side — the only way a recorded `IF`
//! node learns whether to take its body is a store a kernel makes into a
//! handle the driver minted during the capture — so the engine's recording
//! cursor needs a launch to put in the graph, and a launch is this plane's
//! currency.
//!
//! The engine side is `engine_cuda::device::conditional`, which mints the
//! handle, places the node and captures its body; this is the one piece of
//! that sequence that has to be device text.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, symbol};

const FILE: &str = "graph/conditional.cuh";

/// One thread, and it is the whole geometry: the kernel makes one store.
fn once() -> Launch {
    Launch::grid([1, 1, 1], [1, 1, 1])
}

/// Whether a launch of a setter ARMS it or merely warms it.
///
/// **THE WARM ARM IS NOT A COURTESY, IT IS THE ONLY PLACE THE MODULE CAN BE
/// LOADED.** A unit is compiled and its module loaded on first launch; that is
/// host work, and host work inside `cudaStreamBeginCapture` is what the
/// thread-local capture mode exists to refuse. So a shell that is about to
/// record a conditional fires the setter EAGERLY once with [`Arm::Warm`] —
/// which returns before it reaches the handle, because a
/// `cudaGraphSetConditional` outside a conditional graph's launch has nothing
/// to store into — and the captured launch that follows finds the module
/// resident.
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

/// **SET A CONDITIONAL HANDLE FROM A WINDOW'S ROW COUNT.**
///
/// `indptr` is the device address of a window's rebased row CSR and `lanes`
/// its lane count, so `indptr[lanes]` is the window's rows and the handle is
/// set to `rows != 0` — the zero-row rule of decision #3, read on the device
/// instead of taken on the host. `absent` is what a null `indptr` means: `1`
/// for "a window with no staged table runs anyway", `0` for "it does not".
///
/// `handle` is a `CUgraphConditionalHandle`, which is a `u64`.
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
        ],
    )
}

/// The same store, from a device byte the caller staged rather than from a
/// window table — the form a gate drives both arms of.
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

/// **SET A SWITCH HANDLE TO THIS ARM'S INDEX, IF THIS ARM HAS ROWS.**
///
/// The `SWITCH` twin of [`set_conditional`], and the difference is what a
/// handle holds: an `IF`'s is a bool and a `SWITCH`'s is an arm index in
/// `0..arms`, with "at or past `arms`" meaning no body runs at all. So a
/// group's empty fire needs no store — the handle's DEFAULT is what says
/// nothing runs, and the recorder mints it out of range on purpose.
///
/// **CALLED ONCE PER ARM.** There is no single vector holding every arm's row
/// count (each arm is its own region with its own window), so each arm gets
/// its own launch with its own `indptr`, and each stores only if it is live.
/// P3 proves at most one arm is demanded by any admissible composition, so at
/// most one of those stores happens and their order cannot matter.
///
/// `indptr` at zero stores nothing: this arm stands down. The recorder never
/// passes one — it refuses a SWITCH whose arm cannot state a row count — so
/// that spelling belongs to a gate.
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
        ],
    )
}
