//! Conditional graph nodes: HIP graphs have none, so these refuse.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Warm,
    Set,
}

crate::error::unsupported! {
    #[allow(clippy::too_many_arguments)]
    set_conditional = "graph.set_conditional" (handle: u64, indptr: u64, lanes: u32,
        absent: bool, arm: Arm, win: u64);
    #[allow(clippy::too_many_arguments)]
    set_switch = "graph.set_switch" (handle: u64, arm: u32, indptr: u64, lanes: u32,
        warm: Arm, win: u64);
}
