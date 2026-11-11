//! Gemma 4 12B: `weight.csl` (deferred norms, sliding / global caches, plain rows), `cx.csl` (deferred norms, head groups), `ctl_cy_gemma12.csl`, `head_pe.csl` (GRP 1), with the
//! shared head control ([`super::head::HEAD_CTL`]). Their params from the plan ([`kernels`]) and
//! their rcfg writers.

use crate::abi::{Kernel, KernelBuild, Kernels, params};
use crate::block::head::ROLE;
use crate::block::head_groups::{CHAIN_IN, CHAIN_OUT, FORWARD, group_dims};
use crate::block::row_runs::{PLAIN_ROW, dims_of};
use crate::block::token_loop::{EMB_IN, TURN};
use crate::block::units::{RELAY_POST, RELAY_PRE, X_IN, X_OUT};
use crate::plan::{LayerKind, Plan};
use crate::program::Program;
use crate::role::{Kind, Place};

use super::Pe;
use super::common::{self, CT6, CX_EMB, CX_NO_OUTPUT, CxIo, HeadIo};
use super::head::{HEAD_CTL, HeadCtl, HeadPeGroups};

type R<T> = Result<T, String>;

/// Every name the kernels export, in export order: the ABI between the kernels and the host's memcpy.
pub const SYMBOLS: &[(&str, &str)] = &[
    ("w", "u16"),
    ("rcfg", "u32"),
    ("tdbg", "u32"),
    ("x", "f32"),
    ("xdbg", "f32"),
    ("xred", "f32"),
    ("gam", "f32"),
    ("qkw", "f32"),
    ("rot", "f32"),
    ("tab", "u32"),
    ("hw", "u16"),
    ("hg", "f32"),
    ("hemb", "u16"),
    ("res", "f32"),
    ("ring", "u32"),
    ("lgt", "f32"),
    ("force", "u32"),
    ("nforce", "u32"),
];

params! {
    /// `weight.csl`: layers a unit, x's dims a row (the plain rows'; the attention rows'
    /// by override), the MLP width a column, the global layers' mask, the row role (`AROW`), q
    /// heads a KV head, a head dim's columns a segment PE (sliding / global), the KV entries, the
    /// KV rows, the head dims, the window (0: the ring is exact), the levers' defaults.
    pub struct WeightGemma12 {
        K: u32, Hc: u32, Ic: u32, MV: u32, KM: u32, AROW: u32, Qh: u32, DcS: u32, DcG: u32,
        NES: u32, NEG: u32, PYW: u32, HDS: u32, HDG: u32, WIN: u32, WSPEC: u32, RING: u32,
        DIAG: u32, FIFOW: u32, TSTAMP: u32, NRM: u32, MIXK: u32, KEV: u32, NR: u32, DONEG: u32,
    }
}

params! {
    /// `cx.csl` (NRM 1, HG 1): layers a unit, x's dims a row (by override per row run), the levers'.
    pub struct CtlCxGemma12 { K: u32, Hc: u32, DIAG: u32, FIFOW: u32, XKEEP: u32, NRM: u32, HG: u32 }
}

params! {
    /// `ctl_cy_gemma12.csl`: the weight kernel's shapes, the hidden size (`HR`), the levers'.
    pub struct CtlCyGemma12 {
        K: u32, KM: u32, Qh: u32, DcS: u32, DcG: u32, Ic: u32, HR: u32, DIAG: u32, FIFOW: u32,
        ATTL: u32, TSTAMP: u32, HDS: u32, HDG: u32,
    }
}

pub const WEIGHT_GEMMA12: Kernel = Kernel {
    source: "weight.csl",
    owns: &["w", "rcfg", "tdbg"],
    params: WeightGemma12::NAMES,
    route_slots: 18,
    rcfg: Some(weight),
};

pub const CTL_CX_GEMMA12: Kernel = Kernel {
    source: "cx.csl",
    owns: &["x", "gam", "rcfg", "xdbg", "tdbg", "xred"],
    params: CtlCxGemma12::NAMES,
    route_slots: 12,
    rcfg: Some(cx),
};

pub const CTL_CY_GEMMA12: Kernel = Kernel {
    source: "ctl_cy_gemma12.csl",
    owns: &["qkw", "rot", "tab", "rcfg", "tdbg"],
    params: CtlCyGemma12::NAMES,
    route_slots: 12,
    rcfg: Some(cy),
};

/// `head_pe.csl` with head groups and the tied embedding.
pub const HEAD_GEMMA12: Kernel = Kernel {
    source: "head_pe.csl",
    owns: &["hw", "hg", "rcfg", "hemb", "tdbg"],
    params: HeadPeGroups::NAMES,
    route_slots: 12,
    rcfg: Some(head),
};

/// A weight PE's flags: column 0, the last unit's (it flushes).
pub const W_COL0: u32 = 1;
pub const W_LAST: u32 = 4;
/// A CX's flags past the shared ones: the last unit's (it flushes and feeds the head), a
/// member of its group's x chain (it sends north, none to a next unit), of the embedding's.
pub const CX_LAST: u32 = 8 | 16;
pub const CX_X_MEMBER: u32 = 64 | CX_NO_OUTPUT;
pub const CX_EMB_MEMBER: u32 = CX_EMB | 128;
/// A CY's flag: the last unit's.
pub const CY_LAST: u32 = 4;

fn last_unit(pe: &Pe, unit: u32) -> bool {
    unit as usize + 1 == pe.prog.floor.units.len()
}

/// A weight PE: the shared words, its flags, its handoff turns of `2 Hc(r) + 1` words.
fn weight(pe: &Pe) -> R<Vec<u32>> {
    let Place::Weight { unit, row, col } = pe.role().place else {
        return Err(format!("{:?}: weight off a weight PE", pe.at));
    };
    let fl = if col == 0 { W_COL0 } else { 0 } | if last_unit(pe, unit) { W_LAST } else { 0 };
    let hn = 2 * dims_of(&pe.prog.plan, row) + 1;
    let relays = [pe.turn(RELAY_PRE), pe.turn(RELAY_POST)];
    common::weight(pe, 18, row, [fl, 0], hn, relays)
}

/// A CX PE: the shared words, its group chain's in / out (parked if absent) and the dims it
/// forwards along it.
fn cx(pe: &Pe) -> R<Vec<u32>> {
    let Place::Cx { first, last, .. } = pe.role().place else {
        return Err(format!("{:?}: cx off a CX PE", pe.at));
    };
    let pk = pe.parks(4)?;
    let (cin, cout) = (pe.colour(CHAIN_IN), pe.colour(CHAIN_OUT));
    let fw = pe.note(FORWARD);
    let mut flags = 0;
    if last {
        flags |= CX_LAST | if cout.is_some() { CX_X_MEMBER } else { 0 };
    }
    if first && cin.is_some() {
        flags |= CX_EMB_MEMBER;
    }
    let (ci, co) = (cin.unwrap_or(pk[3]), cout.unwrap_or(pk[3]));
    let x = CxIo {
        first,
        inn: pe.colour(X_IN),
        out: pe.colour(X_OUT),
        emb: pe.colour(EMB_IN),
        flags,
        finals: vec![ci],
        tail: vec![
            ci,
            co,
            if last { fw } else { 0 },
            if first { fw } else { 0 },
        ],
    };
    common::cx(pe, x, 4, &CT6)
}

/// A CY PE: the shared words (no segment), its flag, four unused words.
fn cy(pe: &Pe) -> R<Vec<u32>> {
    let Place::Cy { unit, .. } = pe.role().place else {
        return Err(format!("{:?}: ctl_cy_gemma12 off a CY PE", pe.at));
    };
    let mut w = common::cy(pe, None)?;
    let fl = if last_unit(pe, unit) { CY_LAST } else { 0 };
    w.extend_from_slice(&[fl, 0, 0, 0, 0]);
    Ok(w)
}

/// A head PE: [`common::head`] with the blocks' notes (a group root takes no relay from the
/// north), then its group's dims.
fn head(pe: &Pe) -> R<Vec<u32>> {
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

/// A head dim's columns a segment PE (rotated pairs side by side).
fn dcf(hd: u32, swa: u32) -> u32 {
    2 * (hd / 2).div_ceil(swa)
}

/// The kernels' shape from the plan: what the meta records too (the family checks them).
pub struct Dims {
    pub k: u32,
    pub km: u32,
    pub qh: u32,
    pub hds: u32,
    pub hdg: u32,
    pub dcs: u32,
    pub dcg: u32,
    pub ic: u32,
    pub nes: u32,
    pub neg: u32,
    pub ra: u32,
    pub win: u32,
    pub hdc: u32,
}

impl Dims {
    pub fn of(plan: &Plan) -> R<Dims> {
        let (s, p) = (&plan.shape, &plan.partition);
        let k = p.layers_per_unit;
        let ra = match p.row_dims.as_slice() {
            [a, b] if a.attention && !b.attention && a.rows.0 == 0 => a.rows.1,
            _ => {
                return Err("gemma12: row runs want attention rows 0..RA then plain rows".into());
            }
        };
        let kinds = |u: u32| -> Vec<bool> {
            s.layers[(u * k) as usize..((u + 1) * k) as usize]
                .iter()
                .map(|l| matches!(l, LayerKind::Global))
                .collect()
        };
        if s.layers.len() < (p.units * k) as usize {
            return Err(format!(
                "gemma12: {} layers for {} units of {k}",
                s.layers.len(),
                p.units
            ));
        }
        if (1..p.units).any(|u| kinds(u) != kinds(0)) {
            return Err("gemma12: the units' layer kinds differ".into());
        }
        let km = kinds(0)
            .iter()
            .enumerate()
            .filter(|(_, g)| **g)
            .fold(0u32, |m, (l, _)| m | (1 << l));
        let window = s
            .layers
            .iter()
            .find_map(|l| match l {
                LayerKind::Sliding { window } => Some(*window),
                _ => None,
            })
            .ok_or("gemma12: no sliding layer")?;
        let swa = p.segments.scores;
        let nseg = p.cols / swa;
        if nseg == 0 || s.attn.kv_heads != nseg || !s.attn.q_heads.is_multiple_of(nseg) {
            return Err(format!(
                "gemma12: {} q / {} kv heads over {nseg} segments",
                s.attn.q_heads, s.attn.kv_heads
            ));
        }
        let (hds, hdg) = (
            s.attn.head_dim,
            s.attn.global_head_dim.unwrap_or(s.attn.head_dim),
        );
        let nes = p
            .kv_entries
            .sliding
            .ok_or("gemma12: no sliding entries (NES)")?;
        let neg = p
            .kv_entries
            .global
            .ok_or("gemma12: no global entries (NEG)")?;
        let hdc = (0..p.head.cols as usize)
            .map(|c| group_dims(plan, c))
            .max()
            .unwrap_or(0);
        Ok(Dims {
            k,
            km,
            qh: s.attn.q_heads / nseg,
            hds,
            hdg,
            dcs: dcf(hds, swa),
            dcg: dcf(hdg, swa),
            ic: s.mlp.intermediate.div_ceil(p.cols),
            nes,
            neg,
            ra,
            win: if window == nes * ra { 0 } else { window },
            hdc,
        })
    }
}

/// Every kind's kernel of a Gemma 12B program, its params from the plan (the weight / CX PEs'
/// `Hc` and `ATT` are the plain rows' here; [`crate::block::row_runs`] sets each
/// PE's).
pub fn kernels(prog: &Program) -> R<Kernels> {
    let plan = &prog.plan;
    let p = &plan.partition;
    let d = Dims::of(plan)?;
    let hcw = p.row_dims[1].dims;
    let (vr, hr) = (p.head.vocab_rows, p.head.rows);
    let mut out = Kernels::new();
    out.insert(
        Kind::WEIGHT,
        KernelBuild::of(
            &WEIGHT_GEMMA12,
            WeightGemma12 {
                K: d.k,
                Hc: hcw,
                Ic: d.ic,
                MV: 1,
                KM: d.km,
                AROW: PLAIN_ROW,
                Qh: d.qh,
                DcS: d.dcs,
                DcG: d.dcg,
                NES: d.nes,
                NEG: d.neg,
                PYW: d.ra,
                HDS: d.hds,
                HDG: d.hdg,
                WIN: d.win,
                WSPEC: 0,
                RING: u32::from(d.win == 0),
                DIAG: 0,
                FIFOW: 32,
                TSTAMP: 0,
                NRM: 1,
                MIXK: 1,
                KEV: 1,
                NR: WEIGHT_GEMMA12.route_slots as u32,
                DONEG: 1,
            },
        )?,
    );
    out.insert(
        Kind::CX,
        KernelBuild::of(
            &CTL_CX_GEMMA12,
            CtlCxGemma12 {
                K: d.k,
                Hc: hcw,
                DIAG: 0,
                FIFOW: 32,
                XKEEP: 0,
                NRM: 1,
                HG: 1,
            },
        )?,
    );
    out.insert(
        Kind::CY,
        KernelBuild::of(
            &CTL_CY_GEMMA12,
            CtlCyGemma12 {
                K: d.k,
                KM: d.km,
                Qh: d.qh,
                DcS: d.dcs,
                DcG: d.dcg,
                Ic: d.ic,
                HR: plan.shape.hidden,
                DIAG: 0,
                FIFOW: 32,
                ATTL: 2,
                TSTAMP: 0,
                HDS: d.hds,
                HDG: d.hdg,
            },
        )?,
    );
    out.insert(
        Kind::HEAD,
        KernelBuild::of(
            &HEAD_GEMMA12,
            HeadPeGroups {
                VR: vr,
                HDc: d.hdc,
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
                VR: vr,
                FIFOW: 32,
                WFORCE: 0,
                WFN: hr,
            },
        )?,
    );
    Ok(out)
}
