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
#[rustfmt::skip]
static SUPERGRAPH_SIGS: [KernelSig; 2] = [
    kernel!(supergraph_set_cond "graph::supergraph_set_cond",
        file = Some("graph/supergraph.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            handle: Usize, preds: U8s, slot: I32,
        ]),
    kernel!(supergraph_set_switch "graph::supergraph_set_switch",
        file = Some("graph/supergraph.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            handle: Usize, preds: U8s, slot: I32,
        ]),
];
