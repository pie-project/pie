//! Qwen3.5 (GDN layers + gated full attention), Qwen3 and Llama 3 (dense attention, no output gate):
//! `weight_dense.csl`, `ctl_cx.csl`, `ctl_cy_dense.csl`, with the head's and the relay's
//! ([`super::head`], [`super::common::RELAY`]). Their params from the plan ([`kernels`]) and
//! their rcfg writers.

use crate::abi::{FILLER, Kernel, KernelBuild, Kernels, Lit, Params, params};
use crate::block::gdn::{GDN, Gdn, SOURCE_NONE};
use crate::block::segments::{ONORM, ONORM_CAST};
use crate::block::token_loop::EMB_IN;
use crate::block::units::{RELAY_POST, RELAY_PRE, ROW_CAST, X_IN, X_OUT};
use crate::plan::LayerKind;
use crate::program::Program;
use crate::role::{Kind, Place};

use super::Pe;
use super::common::{self, CT5, CxIo, RELAY};
use super::head::{HEAD_CTL, HEAD_PE, HeadCtl, HeadPe};

type R<T> = Result<T, String>;

/// Every name the Qwen kernels export, with its element type, in export order: the ABI between the kernels and the host's memcpy.
pub const SYMBOLS: &[(&str, &str)] = &[
    ("w", "u16"),
    ("rcfg", "u32"),
    ("x", "f32"),
    ("gam", "@fp16()"),
    ("cw", "f32"),
    ("gpar", "f32"),
    ("gnw", "f32"),
    ("qkw", "f32"),
    ("rot", "f32"),
    ("tab", "u32"),
    ("hw", "u16"),
    ("hg", "f32"),
    ("res", "f32"),
    ("xdbg", "f32"),
    ("tdbg", "u32"),
    ("xred", "f32"),
    ("ring", "u32"),
    ("hemb", "u16"),
    ("lgt", "f32"),
    ("force", "u32"),
    ("nforce", "u32"),
];

params! {
    /// `weight_dense.csl`: layers a unit and the attention layers' mask, x's dims a row, the q /
    /// MLP / GDN / KV shapes a column, the KV rows and entries, the levers' defaults, whether
    /// the q / k heads are normed (`QKN`).
    pub struct WeightDense {
        K: u32, Hc: u32, QO: u32, Ic: u32, MV: u32, KM: u32, NQA: u32, NQG: u32, GDN: u32,
        ATT: u32, DV: u32, DK: u32, Qh: u32, Dc: u32, NE: u32, PYW: u32, HD: u32, DIAG: u32,
        FIFOW: u32, GDEC: u32, ATTL: u32, TSTAMP: u32, QKN: u32,
    }
}

params! {
    /// `ctl_cx.csl`: layers a unit, x's dims a row, the levers' defaults.
    pub struct CtlCx { K: u32, Hc: u32, HOSTX: u32, XG16: u32, DIAG: u32, FIFOW: u32, XKEEP: u32 }
}

params! {
    /// `ctl_cy_dense.csl`: the layer mask and shapes as the weight kernel's, the hidden size
    /// (`HR`), the conv width, the RoPE pairs and table levels, the levers' defaults, the norms'
    /// eps.
    pub struct CtlCyDense {
        K: u32, KM: u32, NQA: u32, NQG: u32, Qh: u32, Dc: u32, DK: u32, DV: u32, Ic: u32,
        HR: u32, CONV: u32, NROT: u32, HK: u32, HV: u32, DIAG: u32, FIFOW: u32, SILU16: u32,
        CONVQK: u32, ATTV: u32, GDEC: u32, ATTL: u32, TSTAMP: u32, GATE: u32, SLW: u32,
        RLV: u32, RBITS: u32, EPS: Lit,
    }
}

pub const WEIGHT_QWEN: Kernel = Kernel {
    source: "weight_dense.csl",
    owns: &["w", "rcfg", "tdbg"],
    params: WeightDense::NAMES,
    route_slots: 16,
    rcfg: Some(weight),
};

pub const CTL_CX: Kernel = Kernel {
    source: "ctl_cx.csl",
    owns: &["x", "gam", "rcfg", "xdbg", "tdbg", "xred"],
    params: CtlCx::NAMES,
    route_slots: 12,
    rcfg: Some(cx),
};

pub const CTL_CY_QWEN: Kernel = Kernel {
    source: "ctl_cy_dense.csl",
    owns: &["cw", "gpar", "gnw", "qkw", "rot", "tab", "rcfg", "tdbg"],
    params: CtlCyDense::NAMES,
    route_slots: 12,
    rcfg: Some(cy),
};

/// A weight PE: the shared words, its own two the GDN source word ([`Gdn::source`]) and the
/// GDN multicast's colour (the row multicast's where it has none), the handoff turns of `Hc`
/// words.
fn weight(pe: &Pe) -> R<Vec<u32>> {
    let Place::Weight { row, col, .. } = pe.role().place else {
        return Err(format!("{:?}: weight_dense off a weight PE", pe.at));
    };
    let plan = &pe.prog.plan;
    let source = Gdn::of(plan).map_or(SOURCE_NONE, |g| g.source(row, col));
    let gdn = match pe.colour(GDN) {
        Some(c) => c,
        None => pe.c(ROW_CAST)?,
    };
    let hc = plan
        .partition
        .row_dims
        .first()
        .ok_or("dense: no row dims")?
        .dims;
    let relays = [pe.turn(RELAY_PRE), pe.turn(RELAY_POST)];
    common::weight(pe, 16, row, [source, gdn], hc, relays)
}

/// A CX PE: the shared words, its x in / out and the token loop's embedding.
fn cx(pe: &Pe) -> R<Vec<u32>> {
    let Place::Cx { first, .. } = pe.role().place else {
        return Err(format!("{:?}: ctl_cx off a CX PE", pe.at));
    };
    let x = CxIo {
        first,
        inn: pe.colour(X_IN),
        out: pe.colour(X_OUT),
        emb: pe.colour(EMB_IN),
        ..CxIo::default()
    };
    common::cx(pe, x, 3, &CT5)
}

/// A CY PE: the shared words with its o-norm segment.
fn cy(pe: &Pe) -> R<Vec<u32>> {
    common::cy(pe, Some((ONORM, ONORM_CAST)))
}

/// Every kind's kernel of a Qwen program, its params from the plan.
pub fn kernels(prog: &Program) -> R<Kernels> {
    let plan = &prog.plan;
    let (s, p) = (&plan.shape, &plan.partition);
    let k = p.layers_per_unit;
    let km: u32 = s.layers[..k as usize]
        .iter()
        .enumerate()
        .filter(|(_, l)| !matches!(l, LayerKind::Gdn))
        .map(|(i, _)| 1 << i)
        .sum();
    let hcc = p.row_dims.first().ok_or("dense: no row dims")?.dims;
    let gate = u32::from(s.attn.output_gate);
    let (hd, kvh) = (s.attn.head_dim, s.attn.kv_heads);
    let qh = s.attn.q_heads / kvh;
    let dc = hd / p.segments.scores;
    // GDN's dims (Qwen3: the stand-ins that shape its unused buffers, DV = Qh Dc)
    let (hk, hv) = match s.gdn {
        Some(g) => (g.key_dim, g.value_dim),
        None => (hd, qh * hd),
    };
    let (dk, dv) = (hk / p.segments.gdn, hv / p.segments.onorm);
    let ic = s.mlp.intermediate / p.cols;
    let nqa = (1 + gate) * qh * dc + 2 * dc;
    let nqg = 2 * dk + 2 * dv + 2;
    let mut out = Kernels::new();
    out.insert(
        Kind::WEIGHT,
        KernelBuild::of(
            &WEIGHT_QWEN,
            WeightDense {
                K: k,
                Hc: hcc,
                QO: qh * dc,
                Ic: ic,
                MV: 1,
                KM: km,
                NQA: nqa,
                NQG: nqg,
                GDN: u32::from(km != (1 << k) - 1),
                ATT: u32::from(km != 0),
                DV: dv,
                DK: dk,
                Qh: qh,
                Dc: dc,
                NE: p.kv_entries.per_row,
                PYW: p.rows,
                HD: hd,
                DIAG: 0,
                FIFOW: 32,
                GDEC: 1,
                ATTL: 2,
                TSTAMP: 0,
                QKN: u32::from(s.attn.qk_norm),
            },
        )?,
    );
    out.insert(
        Kind::CX,
        KernelBuild::of(
            &CTL_CX,
            CtlCx {
                K: k,
                Hc: hcc,
                HOSTX: 0,
                XG16: 0,
                DIAG: 0,
                FIFOW: 32,
                XKEEP: 0,
            },
        )?,
    );
    out.insert(
        Kind::CY,
        KernelBuild::of(
            &CTL_CY_QWEN,
            CtlCyDense {
                K: k,
                KM: km,
                NQA: nqa,
                NQG: nqg,
                Qh: qh,
                Dc: dc,
                DK: dk,
                DV: dv,
                Ic: ic,
                HR: s.hidden,
                CONV: 4,
                NROT: dc / 2,
                HK: hk,
                HV: hv,
                DIAG: 0,
                FIFOW: 32,
                SILU16: 1,
                CONVQK: 1,
                ATTV: 1,
                GDEC: 1,
                ATTL: 2,
                TSTAMP: 0,
                GATE: gate,
                SLW: 0,
                RLV: plan.contract.rope_levels,
                RBITS: plan.contract.rope_bits,
                EPS: Lit::f32(s.norm.eps),
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
            &HEAD_CTL,
            HeadCtl {
                VR: p.head.vocab_rows,
                FIFOW: 32,
                WFORCE: 0,
                WFN: p.head.rows,
            },
        )?,
    );
    if !prog.floor.relays.is_empty() {
        out.insert(Kind::RELAY, KernelBuild::of(&RELAY, Params::default())?);
    }
    out.insert(Kind::IDLE, KernelBuild::of(&FILLER, Params::default())?);
    Ok(out)
}
