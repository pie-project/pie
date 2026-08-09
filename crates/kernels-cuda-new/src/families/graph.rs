//! The supergraph's two arming kernels — the one unit whose argument is a
//! SHELL object rather than a tensor.
//!
//! # Why this family exists at all
//!
//! Every other family here is named after a kind of value: `attn`, `gemm`,
//! `norm`, `moe`. This one is named after the thing its kernels write, which
//! is a `cudaGraphConditionalHandle` — an object the CUDA graph API owns and
//! no tensor vocabulary can describe. That is the argument
//! `driver-cuda/src/device/graph.rs`'s header made when the device text lived
//! in `driver-cuda/csrc/supergraph.cu` (*"its argument is a conditional
//! handle — a shell object — rather than a tensor"*), and it survives the
//! move: the text had to come here because this is the tree NVRTC compiles,
//! and inside this tree it belongs to no value family, so it has its own
//! directory and its own module.
//!
//! # What moved, and what is gone with it
//!
//! `driver-cuda/csrc/supergraph.cu` held two `__global__`s and two
//! `extern "C"` launchers, and `driver-cuda/build.rs` gave it its own nvcc
//! archive — `cc::Build::new().cuda(true) … .compile("pie_supergraph")` —
//! because *"this needs nvcc"*. **It does not**, and the measurement that
//! says so is at the top of `csrc/src/graph/supergraph.cuh`: NVRTC compiles
//! the call, the PTX carries `.extern .func cudaGraphSetConditional` and a
//! `call.uni`, and the DRIVER resolves that symbol at `cuModuleLoadData` —
//! which it must, because the symbol has no definition in any toolkit header
//! and is not in `libcudadevrt.a`. The `.cu`, its archive and its two
//! `extern "C"` declarations in `device/graph.rs` are deleted; the host half
//! is `driver-cuda/src/fire/supergraph.rs`.
//!
//! # Neither row is in `table::TABLES`, and that is the whole shim story
//!
//! A row here is a `__global__`'s contract. A row in [`crate::table`] is a
//! `pie_k_*` entry point's contract, and `abi::emit_c_shim` writes a
//! forwarding body for every STATED row of the tables it is handed. These two
//! symbols have no table row of either kind, so no entry point is generated
//! for them, so there is nothing for a deleted host launcher to leave
//! dangling — which is the same end the three shim-dropping mechanisms reach
//! by three other routes, and none of those three is available or needed
//! here:
//!
//! * [`crate::device::JIT_DISPATCHED`] routes a DISPATCH ARM at a table row;
//!   `abi::emit_rust_dispatch` writes the arm from the row's operand list,
//!   and no lowering, statement or trace names these symbols — the driver
//!   composes them while it builds a graph, exactly as `layout/graph_pad.cuh`
//!   describes for its own kernel.
//! * [`crate::execution::RUST_SERVED`] SUBTRACTS from `stated(tables)` —
//!   `abi.rs:144` — so it can only silence a symbol that has a table row to
//!   be silenced. Naming one of these there would subtract nothing from a set
//!   it was never in. Its neighbour `execution::WALKED` is refused outright:
//!   `execution::tests::a_walk_is_only_a_walk` asserts a walked symbol is
//!   hosted by NO unit, and these are hosted here.
//! * Stating no `operands` is the third, and it is what a table row for these
//!   would have to do to stay silent — a row that says nothing, to prevent an
//!   entry point nobody wants. The row that says nothing is better spelled as
//!   the row that is not there.
//!
//! The `_dev` symbol split (`new-horizon.md` §60.6, `fire/moe_dispatch.rs`)
//! is the tool for the other shape: an ABI symbol a statement names, whose
//! device twin must be a different string so the walk and the unit do not
//! collide. There is no ABI symbol here to split from.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The supergraph's arming kernels: one block, one thread, one handle.
pub const SUPERGRAPH: Unit = Unit {
    name: "graph/supergraph",
    root: include_str!("../../csrc/src/graph/supergraph.cuh"),
    rows: SUPERGRAPH_ROWS,
    options: &[],
};

/// The units `graph` compiles.
pub static UNITS: &[Unit] = &[SUPERGRAPH];

/// [`SUPERGRAPH`]'s instantiations.
///
/// Both are [`DeviceKernel::PLAIN`]: neither `__global__` has a template
/// parameter list, because neither touches an element type. A handle is 64
/// bits whatever the model is in, and the predicate word is bytes.
static SUPERGRAPH_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &SUPERGRAPH_SIGS[0],
        template_path: "graph::device::supergraph_set_cond",
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &SUPERGRAPH_SIGS[1],
        template_path: "graph::device::supergraph_set_switch",
        elem: DeviceKernel::PLAIN,
    },
];

/// The contracts, in [`SUPERGRAPH_ROWS`]' order.
///
/// # `LaunchRule::Unstated`, and why not a rule
///
/// Both launches were `<<<1, 1, 0, stream>>>` — `csrc/supergraph.cu:61` and
/// `:74`, recorded verbatim at the top of `graph/supergraph.cuh` because the
/// file they stood in is gone. One block of one thread is not a rectangle
/// derived from a fire: it is a property of the kernel, which writes one word
/// and whose second thread would be the racing call CUDA calls undefined. No
/// [`LaunchRule`] states it, and §10.5's bar — a rule must serve more kernels
/// than the one that wants it — refuses to invent one for two rows.
/// `driver-cuda/src/fire/supergraph.rs` states the `Launch` and cites both
/// lines.
///
/// # Unsourced, on purpose
///
/// No operand carries a [`kernels::Source`], because no statement produces
/// one. The handle is created by `SupergraphBuilder::open_cond` at capture
/// time; `preds` is the driver-owned `PredicateWord`'s device address; and
/// `slot` is a `GuardPred` wire number the builder is passing through. A
/// `Source` on any of the three would claim a fire can name it, which is the
/// claim `table::driver_internal`'s header refuses on behalf of every
/// launcher the driver composes for itself.
#[rustfmt::skip]
static SUPERGRAPH_SIGS: [KernelSig; 2] = [
    // `handle` is `Ty::Usize` and the `__global__` takes the prelude's `u64`,
    // which IS `size_t` — see the header's last section for why the pair has
    // to be spelled that way rather than as `unsigned long long`.
    kernel!(supergraph_set_cond "graph::supergraph_set_cond",
        file = Some("graph/supergraph.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            handle: Usize, preds: U8s, slot: I32,
        ]),
    // The same three operands, and deliberately so: the switch form differs
    // from the IF form only in how the DRIVER reads the value written, so a
    // second shape here would be a difference the device text does not have.
    kernel!(supergraph_set_switch "graph::supergraph_set_switch",
        file = Some("graph/supergraph.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            handle: Usize, preds: U8s, slot: I32,
        ]),
];
