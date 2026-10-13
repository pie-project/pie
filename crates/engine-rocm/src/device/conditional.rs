use core::ffi::c_void;

use crate::error::{Fault, Result};

// HIP graphs have no conditional node. This engine registers conditionals
// as priced past any launch they could skip (`chip::registry`), so the
// compiler lowers every region to `AlwaysLaunch` and a launch that is empty
// for this fire reads its zero extent from the window and returns. A profile
// that prices them back in reaches these, which refuse.

#[derive(Debug, Clone, Copy)]
pub struct Conditional {
    pub handle: u64,
    pub node: *mut c_void,
    bodies: *mut *mut c_void,
    pub arms: u32,
}

impl Conditional {
    #[must_use]
    pub fn body(&self, arm: u32) -> Option<*mut c_void> {
        if self.bodies.is_null() || arm >= self.arms {
            return None;
        }
        // SAFETY: `bodies` is the driver's array of `arms` graphs from
        // `cuGraphAddNode_v2`, valid for the node's lifetime; `arm` is bounds-checked.
        Some(unsafe { *self.bodies.add(arm as usize) })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    If,
    Switch { arms: u32 },
}

impl Kind {
    #[must_use]
    pub const fn size(self) -> u32 {
        match self {
            Kind::If => 1,
            Kind::Switch { arms } => arms,
        }
    }

    #[must_use]
    pub const fn quiescent(self) -> u32 {
        match self {
            Kind::If => 0,
            Kind::Switch { arms } => arms,
        }
    }
}

fn absent(what: &str) -> Fault {
    Fault::Unbound {
        what: format!(
            "{what}: HIP graphs have no conditional node, so a region lowered to `If` or \
             `Switch` has nowhere to go on ROCm; leave `compiler.cond_fixed_us` at this \
             engine's default so every region lowers to `AlwaysLaunch`"
        ),
    }
}

pub fn handle(stream: *mut c_void, kind: Kind) -> Result<u64> {
    let _ = (stream, kind);
    Err(absent("a conditional handle"))
}

pub fn open(stream: *mut c_void, handle: u64, kind: Kind) -> Result<Conditional> {
    let _ = (stream, handle, kind);
    Err(absent("a conditional node"))
}

pub fn begin_body(stream: *mut c_void, body: *mut c_void) -> Result<()> {
    let _ = (stream, body);
    Err(absent("a conditional body"))
}

pub fn end_body(stream: *mut c_void) -> Result<()> {
    let _ = stream;
    Err(absent("a conditional body"))
}
