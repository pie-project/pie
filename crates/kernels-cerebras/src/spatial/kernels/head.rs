//! The head's kernels: `head_pe.csl` (vocabulary rows of the lm_head, the head input's relays,
//! the embedding's return turns) and `head_ctl.csl` (a head row's reduction root, the token
//! loop's max-reduce).

use crate::spatial::abi::{Kernel, params};
use crate::spatial::block::head::ROLE;
use crate::spatial::block::token_loop::TURN;
use crate::spatial::floor::CtlLine;

use super::Pe;
use super::common::{self, HeadIo};

type R<T> = Result<T, String>;

params! {
    /// `head_pe.csl`'s params: vocabulary rows a PE, x's dims a head column (two CX rows'),
    /// columns a PE reads x from, the f16 accumulator.
    pub struct HeadPe { VR: u32, HDc: u32, NPX: u32, HACC16: u32 }
}

params! {
    /// `head_ctl.csl`'s params: vocabulary rows a PE, the FIFO depth, forced tokens, the head
    /// rows the forced token is multicast over.
    pub struct HeadCtl { VR: u32, FIFOW: u32, WFORCE: u32, WFN: u32 }
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
