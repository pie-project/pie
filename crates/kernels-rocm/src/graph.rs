use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, symbol};

const FILE: &str = "graph/conditional.cuh";

fn once() -> Launch {
    Launch::grid([1, 1, 1], [1, 1, 1])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Warm,
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
            crate::ArgValue::Ptr(win),
        ],
    )
}

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
            crate::ArgValue::Ptr(win),
        ],
    )
}
