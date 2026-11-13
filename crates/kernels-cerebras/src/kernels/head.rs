//! The head's kernels: `head_pe.csl` (vocabulary rows of the lm_head, the head input's relays,
//! the embedding's return turns) and `head_ctl.csl` (a head row's reduction root, the token
//! loop's max-reduce).

use crate::abi::{Kernel, params};
use crate::block::head::ROLE;
use crate::block::token_loop::TURN;
use crate::floor::CtlLine;

use super::Pe;
use super::common::{self, HeadIo};

type R<T> = Result<T, String>;

params! {
    /// `head_pe.csl`'s params: vocabulary rows a PE, x's dims a head column (two CX rows'),
    /// columns a PE reads x from, the f16 accumulator.
    pub struct HeadPe { VR: u32, HDc: u32, NPX: u32, HACC16: u32 }
}

params! {
    /// [`HeadPe`]'s with head groups and the tied embedding (`GRP` 1).
    pub struct HeadPeGroups { VR: u32, HDc: u32, NPX: u32, HACC16: u32, GRP: u32 }
}

params! {
    /// [`HeadPe`]'s on the split layout's band (`BAND` 1: x one CX row's dims a dims-row, the
    /// logits reduced along a vocabulary column, the loop's row sent along the bus) with x scaled
    /// into f16 by a power of two (`XSC` 1: the 2^15 lm_head tile, the raw residual).
    pub struct HeadPeBand { VR: u32, HDc: u32, BAND: u32, XSC: u32 }
}

params! {
    /// `head_ctl.csl`'s params: vocabulary rows a PE, the FIFO depth, forced tokens, the head
    /// rows the forced token is multicast over.
    pub struct HeadCtl { VR: u32, FIFOW: u32, WFORCE: u32, WFN: u32 }
}

params! {
    /// [`HeadCtl`]'s with the token bus on (`TB` 1: a per-token side input's storage reads the
    /// winning token's rows).
    pub struct HeadCtlTb { VR: u32, FIFOW: u32, WFORCE: u32, WFN: u32, TB: u32 }
}

pub const HEAD_PE: Kernel = Kernel {
    source: "head_pe.csl",
    owns: &["hw", "hg", "rcfg", "hemb", "tdbg"],
    params: HeadPe::NAMES,
    route_slots: 12,
    rcfg: Some(head_pe),
};

pub const HEAD_CTL: Kernel = Kernel {
    source: "head_ctl.csl",
    owns: &["res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce"],
    params: HeadCtl::NAMES,
    route_slots: 12,
    rcfg: Some(head_ctl),
};

/// A head PE: [`common::head`] with its input role and turns from the blocks' notes.
fn head_pe(pe: &Pe) -> R<Vec<u32>> {
    common::head(
        pe,
        HeadIo {
            row: pe.at.y,
            role: pe.note(ROLE),
            relay_north: true,
            turns: pe.note(TURN),
            token_loop: pe.prog.plan.contract.token_loop,
        },
    )
}

/// A head control PE: [`common::hctl`] at its vocabulary offset (its place along the control
/// line times the vocabulary rows a PE).
fn head_ctl(pe: &Pe) -> R<Vec<u32>> {
    let along = match pe.prog.floor.head.ctl {
        CtlLine::Column(_) => pe.at.y,
        CtlLine::Row(_) => pe.at.x,
    };
    let vr = pe.prog.plan.partition.head.vocab_rows;
    common::hctl(pe, along * vr, pe.prog.plan.contract.token_loop)
}
