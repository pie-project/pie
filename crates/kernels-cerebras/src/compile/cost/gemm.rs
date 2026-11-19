//! The GEMM regime's op-derived time and memory. Every op runs over the whole array, each PE in
//! step with its row's and column's lines, so an op costs its rows' work at the pace of each
//! line's root: line turns ([`TURN`]), words, `mac32` calls ([`MAC_CALL`], [`MAC_ELEM`]),
//! all-reduce hops ([`HOP`]) and exponents ([`EXP`]). Each op adds [`OP`].

use serde::Serialize;

use super::stream::{ALL_REDUCE, RegionFeatures};
use crate::compile::stream::gemm::{GOp, GemmProgram};

/// A line's turn: two sends, two drains, four route writes, the next member's first word.
pub const TURN: f64 = 150.0;
/// A GEMM's piece past its turn and words (the interpreter's loop, the blocks' setup), fitted to
/// flux2-mini on the simulator.
pub const PIECE: f64 = 280.0;
/// A `mac32` (or `tile32`) call's fixed cycles and a multiply-accumulate's: a call is a row of a
/// block, its MACs the block's (`kc` `@fmachs` of `p` words, pipelined). Fitted to the chip.
pub const MAC_CALL: f64 = 61.75;
pub const MAC_ELEM: f64 = 0.543;
/// A PE's hop of an all-reduce's chain (an `fadds` from the fabric into the fabric).
pub const HOP: f64 = 12.0;
/// A DSD op's fixed cycles.
pub const CALL: f64 = 8.0;
/// A scalar exponent (`math.exp1`).
pub const EXP: f64 = 30.0;
/// A key block's cycles past its modelled terms, fitted to the chip.
pub const KEY_BLOCK: f64 = 11_770.0;
/// An op's fixed cycles: the interpreter, the array's step between ops.
pub const OP: f64 = 200.0;
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
    MAC_CALL + MAC_ELEM * f64::from(n)
}

/// An all-reduce of `n` words over `members` PEs.
pub(crate) fn allred(n: u32, members: u32) -> f64 {
    if members <= 1 {
        return 0.0;
    }
    2.0 * f64::from(n) + f64::from(members) * HOP + 2.0 * CALL
}

/// What a step measures over [`time`]'s price (its terms fitted to the simulator's runs).
pub const MEASURED: f64 = 1.0;

/// The step time a run of `p` is expected to measure: [`time`] by [`MEASURED`].
#[must_use]
pub fn expected(p: &GemmProgram) -> f64 {
    time(p).cycles * MEASURED
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
                ("in", f64::from(rows * w) + CALL)
            }
            GOp::Out { width, .. } => ("out", f64::from(tb * width / c) + CALL * f64::from(tb)),
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
                let row = f.cycles(ch) - f.all_reduces * ALL_REDUCE
                    + f.all_reduces * allred(1, if g.head { p.band } else { c })
                    + f64::from(heads) * allred(1, p.band)
                    + f64::from((g.args.len() + g.outs.len()) as u32) * (CALL + f64::from(ch));
                ("region", f64::from(rows) * row)
            }
        };
        ops.push(OpCost {
            op: i,
            what,
            cycles: cy + OP,
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
