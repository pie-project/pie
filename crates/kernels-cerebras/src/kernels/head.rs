//! The head's kernels: `head_pe.csl` (vocabulary rows of the lm_head, the head input's relays,
//! the embedding's return turns) and `head_ctl.csl` (a head row's reduction root, the token
//! loop's max-reduce), by where the floorplan puts the head: east of the units ([`east`], the
//! token bus on where a side input's storage reads the winner), east fed by row groups
//! ([`groups`]); the band's are [`super::split::band`].

use crate::abi::{Kernel, KernelBuild, Kernels, params};
use crate::block::head::ROLE;
use crate::block::head_groups::group_dims;
use crate::block::ple::TOKEN;
use crate::block::token_loop::TURN;
use crate::floor::CtlLine;
use crate::program::Program;
use crate::role::{Kind, Place};

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

/// `head_ctl.csl` with the token bus to a side input's storage (`TB 1`).
pub const HEAD_CTL_TB: Kernel = Kernel {
    source: "head_ctl.csl",
    owns: &[
        "res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce", "tin",
    ],
    params: HeadCtlTb::NAMES,
    route_slots: 12,
    rcfg: Some(head_ctl_tb),
};

/// `head_pe.csl` fed by row groups (`GRP 1`).
pub const HEAD_GROUPS: Kernel = Kernel {
    source: "head_pe.csl",
    owns: &["hw", "hg", "rcfg", "hemb", "tdbg"],
    params: HeadPeGroups::NAMES,
    route_slots: 12,
    rcfg: Some(head_groups),
};

/// A head control PE: the shared words at its vocabulary offset, then whether its row carries
/// the side input's token word, and its colour.
fn head_ctl_tb(pe: &Pe) -> R<Vec<u32>> {
    let vr = pe.prog.plan.partition.head.vocab_rows;
    let mut w = common::hctl(pe, pe.at.y * vr, true)?;
    let tb = pe
        .prog
        .floor
        .stores
        .iter()
        .any(|s| pe.at.y >= s.rect.y && pe.at.y < s.rect.y + s.rect.h);
    w.extend_from_slice(&[u32::from(tb), pe.c(TOKEN)?]);
    Ok(w)
}

/// A grouped head PE: [`common::head`] with the blocks' notes (a group root takes no relay from
/// the north), then its group's dims.
fn head_groups(pe: &Pe) -> R<Vec<u32>> {
    let Place::Head { row, col } = pe.role().place else {
        return Err(format!("{:?}: a grouped head kernel off a head PE", pe.at));
    };
    let h = HeadIo {
        row,
        role: pe.note(ROLE),
        relay_north: false,
        turns: pe.note(TURN),
        token_loop: true,
    };
    let mut w = common::head(pe, h)?;
    w.push(group_dims(&pe.prog.plan, col as usize));
    Ok(w)
}

/// The head east of the units: a head column takes two CX rows' dims; its control column's token
/// bus is on where a side input's storage reads the winning token (`store`).
pub fn east(prog: &Program, store: bool) -> R<Kernels> {
    let p = &prog.plan.partition;
    let hcc = p.row_dims.first().ok_or("head: no row dims")?.dims;
    let mut out = Kernels::new();
    out.insert(
        Kind::HEAD,
        KernelBuild::of(
            &HEAD_PE,
            HeadPe {
                VR: p.head.vocab_rows,
                HDc: 2 * hcc,
                NPX: 1,
                HACC16: 0,
            },
        )?,
    );
    let ctl = if store {
        KernelBuild::of(
            &HEAD_CTL_TB,
            HeadCtlTb {
                VR: p.head.vocab_rows,
                FIFOW: 32,
                WFORCE: 0,
                WFN: p.head.rows,
                TB: 1,
            },
        )?
    } else {
        KernelBuild::of(
            &HEAD_CTL,
            HeadCtl {
                VR: p.head.vocab_rows,
                FIFOW: 32,
                WFORCE: 0,
                WFN: p.head.rows,
            },
        )?
    };
    out.insert(Kind::HEAD_CTL, ctl);
    Ok(out)
}

/// The head east of the units fed by row groups: a head column takes its group's dims (the
/// widest group's the kernel's buffer).
pub fn groups(prog: &Program) -> R<Kernels> {
    let plan = &prog.plan;
    let p = &plan.partition;
    let hdc = (0..p.head.cols as usize)
        .map(|c| group_dims(plan, c))
        .max()
        .unwrap_or(0);
    let mut out = Kernels::new();
    out.insert(
        Kind::HEAD,
        KernelBuild::of(
            &HEAD_GROUPS,
            HeadPeGroups {
                VR: p.head.vocab_rows,
                HDc: hdc,
                NPX: 1,
                HACC16: 0,
                GRP: 1,
            },
        )?,
    );
    out.insert(
        Kind::HEAD_CTL,
        KernelBuild::of(
            &HEAD_CTL,
            HeadCtl {
                VR: p.head.vocab_rows,
                FIFOW: 32,
                WFORCE: 0,
                WFN: p.head.rows,
            },
        )?,
    );
    Ok(out)
}
