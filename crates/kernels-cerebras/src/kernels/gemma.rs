//! Gemma 4 E2B / E4B: `weight.csl` (18 route slots), `cx.csl` (deferred norms, the PLE phase),
//! `cy.csl`, `head_ctl.csl` with the token bus, `ple_gemma.csl` (the PLE store), with the shared
//! head PE ([`super::head::HEAD_PE`]). Their params from the plan ([`kernels`]) and their rcfg
//! writers.

use crate::abi::{Kernel, KernelBuild, Kernels, params};
use crate::block::ple::{self, STORE, TOKEN, bus, column_of, store_of};
use crate::block::shared_kv::{self, LINE};
use crate::block::token_loop::EMB_IN;
use crate::block::units::{RELAY_POST, RELAY_PRE, X_IN, X_OUT};
use crate::fabric::Colour;
use crate::plan::LayerKind;
use crate::program::Program;
use crate::role::{Kind, Place};

use super::Pe;
use super::common::{self, CT5, CT6, CxIo};
use super::head::{HEAD_PE, HeadCtlTb, HeadPe};

type R<T> = Result<T, String>;

/// Every name the Gemma kernels export, in export order: the ABI between the kernels and the
/// host's memcpy.
pub const SYMBOLS: &[(&str, &str)] = &[
    ("w", "u16"),
    ("rcfg", "u32"),
    ("tdbg", "u32"),
    ("kvlog", "f32"),
    ("x", "f32"),
    ("xdbg", "f32"),
    ("xred", "f32"),
    ("gam", "f32"),
    ("qkw", "f32"),
    ("rot", "f32"),
    ("tab", "u32"),
    ("pl", "f32"),
    ("plog", "f32"),
    ("hw", "u16"),
    ("hg", "f32"),
    ("hemb", "u16"),
    ("res", "f32"),
    ("ring", "u32"),
    ("lgt", "f32"),
    ("force", "u32"),
    ("nforce", "u32"),
    ("ptab", "u16"),
    ("psend", "u32"),
    ("tin", "u32"),
];

params! {
    /// `weight.csl` (deferred norms, sliding / global caches, the PLE round; its 18 route
    /// slots, bind_iqs's bookkeeping in memory, the scores clamped at ln(65504), kvlog): layers a unit and the global layers' mask, x's dims a row, the MLP
    /// width a column, the KV sharing, the sliding / global heads and KV entries, the KV rows,
    /// the window (0: the sliding entries are an exact ring), the levers' defaults.
    pub struct WeightGemma {
        K: u32, Hc: u32, Ic: u32, MV: u32, KM: u32, SH: u32, Qh: u32, DcS: u32, DcG: u32,
        NES: u32, NEG: u32, PYW: u32, HDS: u32, HDG: u32, WIN: u32, WSPEC: u32, PLOG: u32,
        DIAG: u32, FIFOW: u32, ATTL: u32, TSTAMP: u32, NRM: u32, MIXK: u32, PLE: u32, NR: u32,
        DONEG: u32, XHI: u32, KVX: u32,
    }
}

params! {
    /// `cx.csl` (NRM 1, PLE 1): layers a unit, x's dims a row, the levers' defaults.
    pub struct CtlCxGemma { K: u32, Hc: u32, DIAG: u32, FIFOW: u32, XKEEP: u32, NRM: u32, PLE: u32 }
}

params! {
    /// `cy.csl` (NRM 1, MIXK 1, ACT 1, PLE 1, KVS 1): the layer mask and shapes as the weight
    /// kernel's, the hidden size (`HR`), the levers' defaults.
    pub struct CtlCyGemma {
        K: u32, KM: u32, SH: u32, Qh: u32, DcS: u32, DcG: u32, Ic: u32, HR: u32, DIAG: u32,
        FIFOW: u32, ATTL: u32, TSTAMP: u32, HDS: u32, HDG: u32, PLOG: u32, NRM: u32, MIXK: u32,
        ACT: u32, PLE: u32, KVS: u32,
    }
}

params! {
    /// `ple_gemma.csl`: the words a store PE holds and sends, the send delay.
    pub struct PleGemma { NTW: u32, DLY: u32 }
}

pub const WEIGHT_GEMMA: Kernel = Kernel {
    source: "weight.csl",
    owns: &["w", "rcfg", "tdbg", "kvlog"],
    params: WeightGemma::NAMES,
    route_slots: 18,
    rcfg: Some(weight),
};

pub const CTL_CX_GEMMA: Kernel = Kernel {
    source: "cx.csl",
    owns: &["x", "gam", "rcfg", "xdbg", "tdbg", "xred"],
    params: CtlCxGemma::NAMES,
    route_slots: 12,
    rcfg: Some(cx),
};

pub const CTL_CY_GEMMA: Kernel = Kernel {
    source: "cy.csl",
    owns: &["qkw", "rot", "tab", "pl", "plog", "rcfg", "tdbg"],
    params: CtlCyGemma::NAMES,
    route_slots: 12,
    rcfg: Some(cy),
};

/// `head_ctl.csl` with the token bus to the PLE storage.
pub const HEAD_CTL_GEMMA: Kernel = Kernel {
    source: "head_ctl.csl",
    owns: &[
        "res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce", "tin",
    ],
    params: HeadCtlTb::NAMES,
    route_slots: 12,
    rcfg: Some(head_ctl),
};

pub const PLE_GEMMA: Kernel = Kernel {
    source: "ple_gemma.csl",
    owns: &["ptab", "rcfg", "psend"],
    params: PleGemma::NAMES,
    route_slots: 8,
    rcfg: Some(store),
};

/// A weight PE's own word bits: column 0 sends the deferred scalar, a sharer takes the shared
/// KV, the last unit flushes.
pub const W_DEFER: u32 = 1;
pub const W_SHARER: u32 = 2;
pub const W_FLUSH: u32 = 4;
/// The last CX's flush bits.
pub const CX_FLUSH: u32 = 8 | 16;
/// A CY's PLE bits: on a bus, the shared-KV source, the last unit.
pub const CY_BUS: u32 = 1;
pub const CY_KV_SOURCE: u32 = 2;
pub const CY_FLUSH: u32 = 4;
/// A store PE's bits: it stores, takes the token word, is its column's top.
pub const S_STORES: u32 = 1;
pub const S_TOKEN: u32 = 2;
pub const S_TOP: u32 = 4;

fn last_unit(pe: &Pe) -> u32 {
    pe.prog.plan.partition.units - 1
}

/// A weight PE: the shared words, its own two its bits and the shared-KV colour, handoff turns
/// of x | the pending term | its row's |h|^2 (2 Hc + 1 words).
fn weight(pe: &Pe) -> R<Vec<u32>> {
    let Place::Weight { unit, row, col } = pe.role().place else {
        return Err(format!("{:?}: weight off a weight PE", pe.at));
    };
    let plan = &pe.prog.plan;
    let fl = if col == 0 { W_DEFER } else { 0 }
        | if shared_kv::units(plan).1.contains(&unit) {
            W_SHARER
        } else {
            0
        }
        | if unit == last_unit(pe) { W_FLUSH } else { 0 };
    let hn = 2 * plan.partition.row_dims[0].dims + 1;
    let relays = [pe.turn(RELAY_PRE), pe.turn(RELAY_POST)];
    common::weight(pe, 18, row, [fl, pe.c(LINE)?], hn, relays)
}

/// A CX PE: the shared words, the last unit's flush bits.
fn cx(pe: &Pe) -> R<Vec<u32>> {
    let Place::Cx { first, last, .. } = pe.role().place else {
        return Err(format!("{:?}: cx off a CX PE", pe.at));
    };
    let x = CxIo {
        first,
        inn: pe.colour(X_IN),
        out: pe.colour(X_OUT),
        emb: pe.colour(EMB_IN),
        flags: if last { CX_FLUSH } else { 0 },
        ..CxIo::default()
    };
    common::cx(pe, x, 3, &CT5)
}

/// A CY PE: the shared words (no segment), then its PLE bits, its column's bus colour, its place
/// in the column, the bus words, the shared-KV colour.
fn cy(pe: &Pe) -> R<Vec<u32>> {
    let Place::Cy { unit, .. } = pe.role().place else {
        return Err(format!("{:?}: cy off a CY PE", pe.at));
    };
    let plan = &pe.prog.plan;
    let mut w = common::cy(pe, None)?;
    let col = column_of(&pe.prog.floor, plan, unit);
    let fl = if col.is_some() { CY_BUS } else { 0 }
        | if shared_kv::units(plan).0 == Some(unit) {
            CY_KV_SOURCE
        } else {
            0
        }
        | if unit == last_unit(pe) { CY_FLUSH } else { 0 };
    let (pb, ui, lv) = match col {
        Some((i, ui, lv)) => (pe.c(bus(i))?, ui, lv),
        None => (0, 0, 0),
    };
    w.extend_from_slice(&[fl, pb, ui, lv, pe.c(LINE)?]);
    Ok(w)
}

/// A head control PE: the shared words at its vocabulary offset, then whether its row carries
/// the PLE token word, and its colour.
fn head_ctl(pe: &Pe) -> R<Vec<u32>> {
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

/// A store PE: its route table (8 slots), its bits, first token, tokens, bus words, bus colour,
/// the token word's queue, a spare colour. A route-only PE parks its queues.
fn store(pe: &Pe) -> R<Vec<u32>> {
    let Place::Store {
        first,
        count,
        level,
        top,
    } = pe.role().place
    else {
        return Err(format!("{:?}: ple_gemma off a store PE", pe.at));
    };
    let tb = pe.c(TOKEN)?;
    let mut w = pe.routes(8)?;
    let stored = count > 0;
    let tb_in = pe
        .prog
        .fabric
        .route(pe.at, Colour(tb as u8))
        .is_some_and(|r| r.to_ramp());
    let pk = pe.parks(2)?;
    let fl = if stored { S_STORES } else { 0 }
        | if tb_in { S_TOKEN } else { 0 }
        | if stored && top { S_TOP } else { 0 };
    let tbq = if tb_in { tb } else { pk[0] };
    let pb = if stored {
        let i = store_of(&pe.prog.floor, pe.at)
            .ok_or_else(|| format!("{:?}: a storage PE off the storage", pe.at))?;
        pe.c(bus(i))?
    } else {
        pk[1]
    };
    let (first, tokens, level) = if stored {
        (first, count, level)
    } else {
        (0, 0, 0)
    };
    w.extend_from_slice(&[fl, first, tokens, level, pb, tbq]);
    w.push(pe.spare(&[tbq], &CT6)?);
    Ok(w)
}

/// Every kind's kernel of a Gemma program, its params from the plan: the KV-owning units'
/// `SH` / `Ic` (the sharers' are [`shared_kv::SharedKv`]'s overrides), the store PEs' `NTW` the
/// largest column's (the route-only PEs': [`ple::Ple`]'s).
pub fn kernels(prog: &Program) -> R<Kernels> {
    let plan = &prog.plan;
    let (s, p) = (&plan.shape, &plan.partition);
    let k = p.layers_per_unit;
    let km: u32 = (0..k)
        .filter(|&l| matches!(s.layers[l as usize], LayerKind::Global))
        .map(|l| 1 << l)
        .sum();
    let hcc = p.row_dims[0].dims;
    let qh = s.attn.q_heads / 2;
    let hds = s.attn.head_dim;
    let hdg = s.attn.global_head_dim.unwrap_or(hds);
    let (dcs, dcg) = (hds / (p.cols / 2), hdg / (p.cols / 2));
    let ic = s.mlp.intermediate / p.cols;
    let window = s
        .layers
        .iter()
        .find_map(|l| match *l {
            LayerKind::Sliding { window } => Some(window),
            _ => None,
        })
        .unwrap_or(0);
    let nes = p.kv_entries.sliding.unwrap_or(0);
    let neg = p.kv_entries.global.unwrap_or(0);
    let mut out = Kernels::new();
    out.insert(
        Kind::WEIGHT,
        KernelBuild::of(
            &WEIGHT_GEMMA,
            WeightGemma {
                K: k,
                Hc: hcc,
                Ic: ic,
                MV: 1,
                KM: km,
                SH: 0,
                Qh: qh,
                DcS: dcs,
                DcG: dcg,
                NES: nes,
                NEG: neg,
                PYW: p.kv_rows,
                HDS: hds,
                HDG: hdg,
                WIN: if nes * p.kv_rows == window { 0 } else { window },
                WSPEC: 0,
                PLOG: 0,
                DIAG: 0,
                FIFOW: 32,
                ATTL: 2,
                TSTAMP: 0,
                NRM: 1,
                MIXK: 1,
                PLE: 1,
                NR: WEIGHT_GEMMA.route_slots as u32,
                DONEG: 1,
                XHI: 1,
                KVX: 1,
            },
        )?,
    );
    out.insert(
        Kind::CX,
        KernelBuild::of(
            &CTL_CX_GEMMA,
            CtlCxGemma {
                K: k,
                Hc: hcc,
                DIAG: 0,
                FIFOW: 32,
                XKEEP: 0,
                NRM: 1,
                PLE: 1,
            },
        )?,
    );
    out.insert(
        Kind::CY,
        KernelBuild::of(
            &CTL_CY_GEMMA,
            CtlCyGemma {
                K: k,
                KM: km,
                SH: 0,
                Qh: qh,
                DcS: dcs,
                DcG: dcg,
                Ic: ic,
                HR: s.hidden,
                DIAG: 0,
                FIFOW: 32,
                ATTL: 2,
                TSTAMP: 0,
                HDS: hds,
                HDG: hdg,
                PLOG: 0,
                NRM: 1,
                MIXK: 1,
                ACT: 1,
                PLE: 1,
                KVS: 1,
            },
        )?,
    );
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
    out.insert(
        Kind::HEAD_CTL,
        KernelBuild::of(
            &HEAD_CTL_GEMMA,
            HeadCtlTb {
                VR: p.head.vocab_rows,
                FIFOW: 32,
                WFORCE: 0,
                WFN: p.head.rows,
                TB: 1,
            },
        )?,
    );
    out.insert(
        STORE,
        KernelBuild::of(
            &PLE_GEMMA,
            PleGemma {
                NTW: ple::words(plan, &prog.floor),
                DLY: 1500,
            },
        )?,
    );
    Ok(out)
}
