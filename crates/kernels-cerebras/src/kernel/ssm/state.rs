//! A recurrence on a unit's state column (P8, `ops/sst.csl`, `ops/cy_sst.csl`): each head's state
//! held over the column's PEs, its root running the head's conv, gates and step. Gated
//! DeltaNet, Mamba-2 and Kimi's delta attention are its variants.

use super::super::{Kernel, Source};

/// A recurrence on a state column.
pub struct State;

/// The registry's state column.
pub static STATE: State = State;

impl Kernel for State {
    fn name(&self) -> &'static str {
        "ssm.state"
    }

    fn ops(&self) -> &'static [&'static str] {
        &[
            "ssm.causal_conv1d",
            "ssm.gdn_gates",
            "ssm.gated_delta",
            "ssm.mamba2",
            "ssm.kda",
        ]
    }

    fn sources(&self) -> &'static [Source] {
        &[
            (
                "sst.csl",
                include_str!("../../../programs/wafer/ops/sst.csl"),
            ),
            (
                "cy_sst.csl",
                include_str!("../../../programs/wafer/ops/cy_sst.csl"),
            ),
        ]
    }
}
