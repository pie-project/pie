//! gpt-oss: the attention / MoE weight halves (`weight.csl`, its entry-major attention and routed experts),
//! CX (`cx.csl`, NRM 2), the CY halves (`cy.csl` (MIXK 2, MIXK 3)), the
//! band's head (`head_pe_gptoss.csl`) and control row ([`super::head::HEAD_CTL`]'s source). Their
//! params from the plan ([`kernels`]) and their rcfg writers.

use crate::abi::{FILLER, Kernel, KernelBuild, Kernels, Lit, Params, params};
use crate::block::band::{BAND_IN, BAND_OUT, BAND_V1, BAND_VERT, BUS, BUS_COLOUR, V1_RX, W2_TX};
use crate::block::head::{HEAD_COL_CAST, HEAD_IN, HEAD_ROW_REDUCE, HEAD_ROW_WORD};
use crate::block::moe::{CYA, CYM, ROUTER, WA, WM};
use crate::block::segments::ONORM_CAST;
use crate::block::token_loop::EMB_IN;
use crate::block::units::{RELAY_POST, RELAY_PRE, ROW_CAST, X_IN, X_OUT};
use crate::plan::{LayerKind, Plan};
use crate::program::Program;
use crate::role::{Kind, Place};
use crate::wiring::{LOOPBACK, Slot};

use super::common::{self, CT5, CT6};
use super::head::{HEAD_CTL, HeadCtl};
use super::{Pe, word};

type R<T> = Result<T, String>;

/// Every name the kernels export, in export order: the ABI between the kernels and the host's memcpy.
pub const SYMBOLS: &[(&str, &str)] = &[
    ("w", "u16"),
    ("wx", "u16"),
    ("wr", "f32"),
    ("rcfg", "u32"),
    ("tdbg", "u32"),
    ("x", "f32"),
    ("xdbg", "f32"),
    ("xred", "f32"),
    ("gam", "f32"),
    ("bo", "f32"),
    ("bd", "f32"),
    ("qb", "f32"),
    ("snk", "f32"),
    ("rot", "f32"),
    ("rb", "f32"),
    ("gb", "f32"),
    ("ps", "f32"),
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
    /// `weight.csl` (MIXK 2, rcfg's flags word): layers a unit and the full-attention mask, x's dims a row, the
    /// q / KV shapes a column, the KV entries and rows, the sliding window, the levers' defaults.
    pub struct WeightAtt {
        K: u32, KM: u32, Hc: u32, NQA: u32, QO: u32, Qh: u32, Dc: u32, NE: u32, PYW: u32,
        WIN: u32, PFTZ: u32, DIAG: u32, TSTAMP: u32, FIFOW: u32, MIXK: u32, RCW: u32,
    }
}

params! {
    /// `weight.csl` (MIXK 3, rcfg's flags word): layers a unit, x's dims a row, an expert's columns a PE, the
    /// experts and top-k, the levers' defaults.
    pub struct WeightMoe {
        K: u32, Hc: u32, Ic: u32, NEX: u32, TK: u32, DIAG: u32, TSTAMP: u32, FIFOW: u32, MIXK: u32, RCW: u32,
    }
}

params! {
    /// `cx.csl` (NRM 2): layers a unit, x's dims a row, the experts and top-k, the levers'
    /// defaults.
    pub struct CtlCx { K: u32, Hc: u32, NEX: u32, TK: u32, DIAG: u32, XKEEP: u32, XDLY: u32, NRM: u32 }
}

params! {
    /// `cy.csl` (NRM 2, MIXK 2): the attention shapes, RoPE pairs, the hidden size, the RoPE
    /// table levels, the scores' scale, the levers' defaults.
    pub struct CyAtt {
        K: u32, NQA: u32, Qh: u32, Dc: u32, NROT: u32, HR: u32, NLV: u32, QSC: Lit, CYV: u32,
        DIAG: u32, TSTAMP: u32, FIFOW: u32, NRM: u32, MIXK: u32, EPS: Lit,
    }
}

params! {
    /// `cy.csl` (NRM 2, MIXK 3): the expert shapes, the hidden size, the norm's eps, the levers'
    /// defaults.
    pub struct CyMoe {
        K: u32, Ic: u32, NEX: u32, TK: u32, HR: u32, GLUV: u32, DIAG: u32, TSTAMP: u32, FIFOW: u32,
        NRM: u32, MIXK: u32, EPS: Lit,
    }
}

params! {
    /// `head_pe_gptoss.csl`: vocabulary rows a PE, x's dims a dims-row.
    pub struct HeadPe { VR: u32, HDc: u32 }
}

pub const WEIGHT_ATT: Kernel = Kernel {
    source: "weight.csl",
    owns: &["w", "rcfg", "tdbg"],
    params: WeightAtt::NAMES,
    route_slots: 16,
    rcfg: Some(weight),
};

pub const WEIGHT_MOE: Kernel = Kernel {
    source: "weight.csl",
    owns: &["wx", "wr", "rcfg", "tdbg"],
    params: WeightMoe::NAMES,
    route_slots: 16,
    rcfg: Some(weight),
};

pub const CTL_CX: Kernel = Kernel {
    source: "cx.csl",
    owns: &["x", "gam", "bo", "bd", "rcfg", "xdbg", "tdbg", "xred"],
    params: CtlCx::NAMES,
    route_slots: 12,
    rcfg: Some(cx),
};

pub const CY_ATT: Kernel = Kernel {
    source: "cy.csl",
    owns: &["qb", "snk", "rot", "rcfg", "tdbg"],
    params: CyAtt::NAMES,
    route_slots: 12,
    rcfg: Some(cy),
};

pub const CY_MOE: Kernel = Kernel {
    source: "cy.csl",
    owns: &["rb", "gb", "ps", "rcfg", "tdbg"],
    params: CyMoe::NAMES,
    route_slots: 12,
    rcfg: Some(cy),
};

pub const HEAD_PE: Kernel = Kernel {
    source: "head_pe_gptoss.csl",
    owns: &["hw", "hg", "rcfg", "hemb", "tdbg"],
    params: HeadPe::NAMES,
    route_slots: 12,
    rcfg: Some(head),
};

/// `head_ctl.csl` on the band's control row: never parked on the bus.
pub const HEAD_CTL_BAND: Kernel = Kernel {
    rcfg: Some(hctl),
    ..HEAD_CTL
};

/// The kernel dims the plan implies (`Qh`, `Dc`, `NROT`, `Ic`, `NQA`, `QO`,
/// `XE`).
pub struct Dims {
    pub k: u32,
    pub km: u32,
    pub hcc: u32,
    pub qh: u32,
    pub dc: u32,
    pub nrot: u32,
    pub ic: u32,
    pub nex: u32,
    pub tk: u32,
    pub nqa: u32,
    pub qo: u32,
    pub xe: u32,
    pub nlv: u32,
    pub win: u32,
}

impl Dims {
    pub fn of(plan: &Plan) -> R<Dims> {
        let (s, p) = (&plan.shape, &plan.partition);
        let moe = s.moe.ok_or("gpt-oss plan: no MoE")?;
        let ax = p.attn_cols.ok_or("gpt-oss plan: no attention columns")?;
        let k = p.layers_per_unit;
        let hcc = p.row_dims.first().ok_or("gpt-oss plan: no row dims")?.dims;
        let qh = s.attn.q_heads / s.attn.kv_heads;
        let dc = s.attn.head_dim / p.segments.scores;
        let ic = moe.expert_intermediate / (p.cols - ax);
        let km = (0..k)
            .filter(|&l| matches!(s.layers.get(l as usize), Some(LayerKind::Global)))
            .fold(0, |m, l| m | (1 << l));
        let win = s
            .layers
            .iter()
            .find_map(|l| match l {
                LayerKind::Sliding { window } => Some(*window),
                _ => None,
            })
            .unwrap_or(128);
        Ok(Dims {
            k,
            km,
            hcc,
            qh,
            dc,
            nrot: dc / 2,
            ic,
            nex: moe.experts,
            tk: moe.top_k,
            nqa: qh * dc + 2 * dc,
            qo: qh * dc,
            xe: (hcc / 2) * 2 * ic + (ic / 2) * hcc,
            nlv: plan.contract.rope_levels,
            win,
        })
    }
}

/// Every kind's kernel of a gpt-oss program, its params from the plan with the serving
/// defaults (`DIAG` / `TSTAMP` / `XKEEP` / `XDLY` / `PFTZ` / `WFORCE` 0, `GLUV` 1, `CYV` 2,
/// `FIFOW` 32).
pub fn kernels(prog: &Program) -> R<Kernels> {
    let sc = prog.schedule;
    let plan = &prog.plan;
    let (s, p) = (&plan.shape, &plan.partition);
    let d = Dims::of(plan)?;
    let qsc = 1.0f64 / f64::from(s.attn.head_dim).sqrt();
    let (vr, hr) = (p.head.vocab_rows, s.hidden);
    let mut out = Kernels::new();
    let att = WeightAtt {
        K: d.k,
        KM: d.km,
        Hc: d.hcc,
        NQA: d.nqa,
        QO: d.qo,
        Qh: d.qh,
        Dc: d.dc,
        NE: p.kv_entries.per_row,
        PYW: p.rows,
        WIN: d.win,
        PFTZ: 0,
        DIAG: 0,
        TSTAMP: 0,
        FIFOW: 32,
        MIXK: sc.mixk,
        RCW: 26,
    };
    out.insert(WA, KernelBuild::of(&WEIGHT_ATT, att)?);
    let moe = WeightMoe {
        K: d.k,
        Hc: d.hcc,
        Ic: d.ic,
        NEX: d.nex,
        TK: d.tk,
        DIAG: 0,
        TSTAMP: 0,
        FIFOW: 32,
        MIXK: sc.moe(),
        RCW: 26,
    };
    out.insert(WM, KernelBuild::of(&WEIGHT_MOE, moe)?);
    let cx = CtlCx {
        K: d.k,
        Hc: d.hcc,
        NEX: d.nex,
        TK: d.tk,
        DIAG: 0,
        XKEEP: 0,
        XDLY: 0,
        NRM: sc.nrm,
    };
    out.insert(Kind::CX, KernelBuild::of(&CTL_CX, cx)?);
    let cya = CyAtt {
        K: d.k,
        NQA: d.nqa,
        Qh: d.qh,
        Dc: d.dc,
        NROT: d.nrot,
        HR: hr,
        NLV: d.nlv,
        QSC: Lit::f64(qsc),
        CYV: 2,
        DIAG: 0,
        TSTAMP: 0,
        FIFOW: 32,
        NRM: sc.nrm,
        MIXK: sc.mixk,
        EPS: Lit::f32(s.norm.eps),
    };
    out.insert(CYA, KernelBuild::of(&CY_ATT, cya)?);
    let cym = CyMoe {
        K: d.k,
        Ic: d.ic,
        NEX: d.nex,
        TK: d.tk,
        HR: hr,
        GLUV: 1,
        DIAG: 0,
        TSTAMP: 0,
        FIFOW: 32,
        NRM: sc.nrm,
        MIXK: sc.moe(),
        EPS: Lit::f32(s.norm.eps),
    };
    out.insert(CYM, KernelBuild::of(&CY_MOE, cym)?);
    let hp = HeadPe { VR: vr, HDc: d.hcc };
    out.insert(Kind::HEAD, KernelBuild::of(&HEAD_PE, hp)?);
    let hc = HeadCtl {
        VR: vr,
        FIFOW: 32,
        WFORCE: 0,
        WFN: p.head.cols,
    };
    out.insert(Kind::HEAD_CTL, KernelBuild::of(&HEAD_CTL_BAND, hc)?);
    out.insert(Kind::IDLE, KernelBuild::of(&FILLER, Params::default())?);
    Ok(out)
}

/// `pe` as a band PE sees it: the bus colour reserved.
fn banded<'a>(pe: &Pe<'a>) -> Pe<'a> {
    Pe {
        prog: pe.prog,
        at: pe.at,
        io: pe.io,
        reserved: Some(BUS_COLOUR),
    }
}

fn hcc(pe: &Pe) -> R<u32> {
    let p = &pe.prog.plan.partition;
    Ok(p.row_dims.first().ok_or("gpt-oss: no row dims")?.dims)
}

/// A weight PE: the shared words, its own two the router column flag (the MoE half's first
/// column) and the row multicast, the relays (the handoffs', or the embedding's / band's turn),
/// then whether it takes the embedding.
fn weight(pe: &Pe) -> R<Vec<u32>> {
    let role = pe.role();
    let Place::Weight { row, col, .. } = role.place else {
        return Err(format!(
            "{:?}: a gpt-oss weight kernel off a weight PE",
            pe.at
        ));
    };
    let ax = pe.prog.plan.partition.attn_cols.unwrap_or(0);
    let rwc = u32::from(role.kind == WM && col == ax);
    let turn =
        |s: Slot, inn: Slot, out: Slot| pe.turn(s).or_else(|| pe.colour(inn).zip(pe.colour(out)));
    let relays = [
        turn(RELAY_PRE, V1_RX, W2_TX),
        turn(RELAY_POST, BAND_IN, BAND_OUT),
    ];
    let mx = pe.c(ROW_CAST)?;
    let mut w = common::weight(pe, 16, row, [rwc, mx], hcc(pe)?, relays)?;
    w.push(u32::from(pe.colour(V1_RX).is_some()));
    Ok(w)
}

fn distinct(pe: &Pe, what: &str, q: &[u32]) -> R<()> {
    let mut s = q.to_vec();
    s.sort_unstable();
    s.dedup();
    if s.len() != q.len() {
        return Err(format!("{:?}: {what} queues share a colour {q:?}", pe.at));
    }
    Ok(())
}

/// A CX PE: its row root's halves parked where absent, a parked third queue, x in, the
/// loop-back, the multicast, x out, the flags, a spare colour, the embedding queue.
fn cx(pe: &Pe) -> R<Vec<u32>> {
    let Place::Cx { first, .. } = pe.role().place else {
        return Err(format!("{:?}: cx off a CX PE", pe.at));
    };
    let wl = pe.prog.plan.contract.token_loop;
    let pk = pe.parks(4)?;
    let (inn, out, emb) = (pe.colour(X_IN), pe.colour(X_OUT), pe.colour(EMB_IN));
    let rq = pe.root(crate::block::units::ROW_REDUCE)?;
    let q = [
        rq.lo.map_or(pk[0], word),
        rq.hi.map_or(pk[1], word),
        pk[2],
        inn.unwrap_or(pk[0]),
        pe.c(LOOPBACK)?,
        pe.c(ROW_CAST)?,
        out.unwrap_or(pk[1]),
    ];
    let fl = if first || inn.is_none() {
        common::CX_NO_INPUT
    } else {
        0
    } | if out.is_none() {
        common::CX_NO_OUTPUT
    } else {
        0
    } | if wl && emb.is_some() {
        common::CX_EMB
    } else {
        0
    };
    let q6 = emb.unwrap_or(pk[3]);
    let fin = [q[0], q[1], q[2], q[3], q6];
    distinct(pe, "CX", &fin)?;
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&[fl, pe.spare(&fin, &CT5)?, q6]);
    Ok(w)
}

/// A CY PE: the shared words with its router segment, then a MoE PE's place in its segment.
fn cy(pe: &Pe) -> R<Vec<u32>> {
    let role = pe.role();
    let Place::Cy { col, .. } = role.place else {
        return Err(format!("{:?}: a gpt-oss CY kernel off a CY PE", pe.at));
    };
    let p = &pe.prog.plan.partition;
    let mut w = common::cy(pe, Some((ROUTER, ONORM_CAST)))?;
    w.push(if role.kind == CYM {
        let (ax, swr) = (p.attn_cols.unwrap_or(0), p.segments.router.unwrap_or(1));
        (col - ax) % swr
    } else {
        0
    });
    Ok(w)
}

/// A band head PE: its band column's queues, the band's x in ([`BAND_VERT`], at its source),
/// the bus, the dims-row multicast, the row word, the send, the role, the source flag, a spare
/// colour, the flags (the loop runs, on the bus), the embedding's queues.
fn head(pe: &Pe) -> R<Vec<u32>> {
    let pe = &banded(pe);
    let wl = pe.prog.plan.contract.token_loop;
    let pk = pe.parks(6)?;
    let ro = pe.member(HEAD_ROW_REDUCE)?;
    let src = pe.colour(HEAD_IN).is_some();
    let bus = wl && pe.colour(BUS).is_some();
    let (hv2, hm, hr3) = (pe.c(BAND_VERT)?, pe.c(HEAD_COL_CAST)?, pe.c(HEAD_ROW_WORD)?);
    let he = u32::from(BUS_COLOUR);
    let q = [
        ro.recv1.map_or(pk[0], word),
        ro.recv2.map_or(pk[1], word),
        if src { hv2 } else { pk[2] },
        if bus { he } else { pk[3] },
        if src { pk[5] } else { hm },
        if wl { hr3 } else { pk[4] },
    ];
    distinct(pe, "head", &q)?;
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&[
        word(ro.send),
        hm,
        if ro.role == 3 { 1 } else { ro.role },
        if src { 2 } else { 0 },
        pe.spare(&q, &CT6)?,
        u32::from(wl) | if bus { 2 } else { 0 },
        0,
        if wl { he } else { pk[5] },
        if wl { pe.c(BAND_V1)? } else { pk[5] },
    ]);
    Ok(w)
}

/// A band control PE: [`common::hctl`] at its column's vocabulary offset, the bus reserved.
fn hctl(pe: &Pe) -> R<Vec<u32>> {
    let vr = pe.prog.plan.partition.head.vocab_rows;
    common::hctl(&banded(pe), pe.at.x * vr, pe.prog.plan.contract.token_loop)
}
