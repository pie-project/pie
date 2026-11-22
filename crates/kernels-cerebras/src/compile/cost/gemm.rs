//! The GEMM regime's op-derived time and memory. Every op runs over the whole array, each PE in
//! step with its row's and column's lines, so an op costs its rows' work at the pace of each
//! line's root: line turns (`gemm_turn`), words, `mac32` calls (`gemm_mac_call`, `gemm_mac_elem`),
//! all-reduce hops (`gemm_hop`) and exponents (`gemm_exp`). Each op adds `gemm_op`.

use serde::Serialize;

use super::stream::RegionFeatures;
use crate::compile::stream::gemm::{GOp, GemmProgram};
use crate::tuning;

/// A PE's 32-bit words ([`crate::compile::check::PE_BYTES`]).
pub const PE_WORDS: u32 = (crate::compile::check::PE_BYTES / 4) as u32;
/// The code's words a PE (interpreter, line streams, all-reduces; regions counted apart), fitted
/// to the linked ELFs.
pub const CODE_WORDS: u32 = 4_200;
/// A printed region's code and buffers past its chunks, words.
pub const REGION_WORDS: u32 = 400;
/// memcpy's, the stack's and the runtime's words.
pub const RUNTIME_WORDS: u32 = 900;

/// One op, priced.
#[derive(Clone, Debug, Serialize)]
pub struct OpCost {
    pub op: usize,
    pub what: &'static str,
    pub cycles: f64,
    /// A GEMM's price by term: line turns and pieces, input words, weight block words (and calls),
    /// `mac32` calls and elements.
    pub terms: [f64; 5],
}

/// A GEMM program's ops and its cycles.
#[derive(Clone, Debug, Serialize)]
pub struct GemmCost {
    pub ops: Vec<OpCost>,
    pub cycles: f64,
}

impl GemmCost {
    /// Cycles by kind of op.
    #[must_use]
    pub fn by(&self) -> std::collections::BTreeMap<&'static str, f64> {
        let mut m = std::collections::BTreeMap::new();
        for o in &self.ops {
            *m.entry(o.what).or_default() += o.cycles;
        }
        m
    }
}

/// A `mac32` call of `n` multiply-accumulates.
pub(crate) fn mac(n: u32) -> f64 {
    tuning::get().gemm_mac_call + tuning::get().gemm_mac_elem * f64::from(n)
}

/// An all-reduce of `n` words over `members` PEs.
pub(crate) fn allred(n: u32, members: u32) -> f64 {
    if members <= 1 {
        return 0.0;
    }
    2.0 * f64::from(n) + f64::from(members) * tuning::get().gemm_hop + 2.0 * tuning::get().gemm_call
}

/// The step time a run of `p` is expected to measure: [`time`] by `gemm_measured`.
#[must_use]
pub fn expected(p: &GemmProgram) -> f64 {
    time(p).cycles * tuning::get().gemm_measured
}

/// The program's time.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn time(p: &GemmProgram) -> GemmCost {
    let (c, tb) = (p.c, p.tb);
    let mut ops = Vec::new();
    for (i, op) in p.l.ops.iter().enumerate() {
        let mut terms = [0.0; 5];
        let (what, cy) = match op {
            GOp::In { v, lane, .. } => {
                let rows = if *lane { 1 } else { tb };
                let w = crate::compile::stream::gemm::local_width(&p.l.vals[v], c, 0);
                ("in", f64::from(rows * w) + tuning::get().gemm_call)
            }
            GOp::Out { width, .. } => (
                "out",
                f64::from(tb * width / c) + tuning::get().gemm_call * f64::from(tb),
            ),
            // a kernel's op: its price its module's ([`crate::kernel::GemmForm`])
            GOp::Gemm { .. } | GOp::Attend { .. } => match crate::kernel::gemm_form(op) {
                Some(f) => (f.what_price, (f.price)(p, i, &mut terms)),
                None => ("unpriced", 0.0),
            },
            GOp::Region(ri) => {
                let g = &p.l.regs[*ri];
                let rows = if g.lane { 1 } else { tb };
                let ch = g.chunk;
                let pr = g.printed.as_ref();
                let f = pr.map_or_else(RegionFeatures::default, |pr| {
                    RegionFeatures::of(&pr.csl, ch)
                });
                let heads = pr.map_or(0, |pr| pr.head_reduces);
                let row = f.cycles(ch) - f.all_reduces * tuning::get().stream_all_reduce
                    + f.all_reduces * allred(1, if g.head { p.band } else { c })
                    + f64::from(heads) * allred(1, p.band)
                    + f64::from((g.args.len() + g.outs.len()) as u32)
                        * (tuning::get().gemm_call + f64::from(ch));
                ("region", f64::from(rows) * row)
            }
        };
        ops.push(OpCost {
            op: i,
            what,
            cycles: cy + tuning::get().gemm_op,
            terms,
        });
    }
    let cycles = ops.iter().map(|o| o.cycles).sum();
    GemmCost { ops, cycles }
}

/// A PE's words by part.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Memory {
    pub segment: u32,
    pub arena: u32,
    pub buffers: u32,
    pub regions: u32,
    pub host: u32,
    /// The op buffer (an op's words at a time, off the stream): part of `host`.
    pub table: u32,
    pub code: u32,
    pub total: u32,
}

/// A PE's memory (32-bit words), the busiest PE's.
#[must_use]
pub fn memory(p: &GemmProgram) -> Memory {
    let segment = 0;
    let kb = p.kbm * p.tb;
    let nc = p.qchunk;
    let scratch = p.tb * kb + kb + 64 + p.xbn;
    let arena = p.arena + scratch;
    // the 16-bit buffers (two a word): an input piece, keys, values, queries, a [k | v], P
    let buffers = p.wbn + (p.xbn + 2 + 2 * nc * kb + 3 * p.tb * nc + kb).div_ceil(2) + 2 * p.tb;
    let regions: u32 = p
        .modules
        .iter()
        .enumerate()
        .map(|(mi, _)| {
            let g = p.l.regs.iter().find(|r| r.module == mi);
            g.map_or(0, |g| {
                let ch = g.chunk;
                let pr = g.printed.as_ref();
                let bufs = pr.map_or(0, |pr| pr.csl.matches("]f32 align(16)").count() as u32);
                let tables: u32 = pr.map_or(0, |pr| {
                    pr.tables
                        .iter()
                        .map(|t| {
                            let n = t.of_chunk.iter().map(Vec::len).max().unwrap_or(1) as u32;
                            if t.u16 { n.div_ceil(2) } else { n }
                        })
                        .sum()
                });
                REGION_WORDS + bufs * ch + tables
            })
        })
        .sum();
    let (host, table) = crate::compile::emit::gemm::table(p).map_or((0, 0), |t| {
        let tw = t.ops.iter().map(Vec::len).max().unwrap_or(2).div_ceil(2) as u32;
        (t.hin_words + t.hsc_words + t.hout_words + tw, tw)
    });
    let code = CODE_WORDS + RUNTIME_WORDS;
    let total = segment + arena + buffers + regions + host + code;
    Memory {
        segment,
        arena,
        buffers,
        regions,
        host,
        table,
        code,
        total,
    }
}
