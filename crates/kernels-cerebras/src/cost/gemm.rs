//! The GEMM regime's op-derived time and memory (GENERIC.md M2f). Every op runs over the
//! whole array, each PE in step with its row's and column's lines, so an op costs its rows' work
//! at the pace of each line's root:
//!
//! - **A GEMM**: a turn of the row's line a piece of the input (the root's two sends, a drain
//!   each, its four route writes: [`TURN`]), the piece's words, and a weight block a segment
//!   (its words, three times at its column's root: received, sent north and south), each row's
//!   `mac32`s ([`MAC_CALL`] + [`MAC_ELEM`] an element: `mv.csl` on the simulator); rows of
//!   another class drain the blocks.
//! - **A region**: its printed `row()` ([`RegionFeatures`]) a row, its rows' chunks copied in
//!   and out, a row's sum an all-reduce along the row ([`HOP`] a PE), a head's along its band.
//! - **The attention**: each key block's turns of the column's line and its `[k | v]` pieces,
//!   the scores' `mac32`s, two band all-reduces of the block's scores, the softmax (the block's
//!   max and sum, [`EXP`] an exponent of the PE's share), the values' `mac32`s.
//!
//! Each op adds [`OP`]: the array's step between ops.

use serde::Serialize;

use super::stream::{ALL_REDUCE, RegionFeatures};
use crate::stream::gemm::{GOp, GemmProgram, piece};

/// A line's turn: two sends, two drains, four route writes, the next member's first word.
pub const TURN: f64 = 150.0;
/// A GEMM's piece past its turn and words: the interpreter's piece (its loop, the blocks'
/// setup; a piece's bounds two u32 divisions where the input cuts unevenly), fitted to
/// flux2-mini on the simulator (16 + 16 rows: its entries' 219k cycles, 253 a piece; its step's
/// 969k before the even cut's division was hoisted, 395, about 100 of it the division).
pub const PIECE: f64 = 280.0;
/// A `mac32` instruction's fixed cycles and an element's (16-bit operands, f32 accumulation).
pub const MAC_CALL: f64 = 1.84;
pub const MAC_ELEM: f64 = 0.5;
/// A PE's hop of an all-reduce's chain (an `fadds` from the fabric into the fabric).
pub const HOP: f64 = 12.0;
/// A DSD op's fixed cycles.
pub const CALL: f64 = 8.0;
/// A scalar exponent (`math.exp1`).
pub const EXP: f64 = 30.0;
/// An op's fixed cycles: the interpreter, the array's step between ops.
pub const OP: f64 = 200.0;
/// A PE's 32-bit words.
pub const PE_WORDS: u32 = 12_288;
/// The code's words a PE (the interpreter, the line streams, the all-reduces; the regions
/// counted apart), fitted to the linked ELFs so that the model holds them (klein-4B's compute
/// PE at 1024²: 46,962 bytes linked, 47,008 modelled; flux2-mini's at 16 + 16 rows below its
/// model).
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

fn mac(n: u32) -> f64 {
    MAC_CALL + MAC_ELEM * f64::from(n)
}

/// An all-reduce of `n` words over `members` PEs.
fn allred(n: u32, members: u32) -> f64 {
    if members <= 1 {
        return 0.0;
    }
    2.0 * f64::from(n) + f64::from(members) * HOP + 2.0 * CALL
}

/// The program's time.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn time(p: &GemmProgram) -> GemmCost {
    let (c, tb) = (p.c, p.tb);
    let mut ops = Vec::new();
    for (i, op) in p.l.ops.iter().enumerate() {
        let (what, cy) = match op {
            GOp::In { v, lane, .. } => {
                let rows = if *lane { 1 } else { tb };
                let w = crate::stream::gemm::local_width(&p.l.vals[v], c, 0);
                ("in", f64::from(rows * w) + CALL)
            }
            GOp::Out { width, .. } => ("out", f64::from(tb * width / c) + CALL * f64::from(tb)),
            GOp::Gemm {
                k, y, x32, lane, ..
            } => {
                let rows = if *lane { 1 } else { tb };
                let segs = crate::stream::gemm::local(&p.l.vals[y], c, 0);
                let mut cy = 0.0;
                for cp in 0..c {
                    let (a, b) = piece(cp, c, *k);
                    let kc = b - a;
                    if kc == 0 {
                        continue;
                    }
                    let xw = (rows * kc).div_ceil(2) * if *x32 { 2 } else { 1 };
                    cy += TURN + PIECE + 3.0 * f64::from(xw);
                    for s in &segs {
                        let p = s.4;
                        let wb = f64::from(kc * p / 2);
                        // the block off the column's tree
                        cy += wb + CALL;
                        cy +=
                            f64::from(rows) * f64::from(kc) * mac(p) * if *x32 { 2.0 } else { 1.0 };
                    }
                }
                ("gemm", cy)
            }
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
            GOp::Attend { causal, .. } => {
                let kb = p.kbm * tb;
                let nc = p.qchunk;
                let half = tb * nc;
                let blocks = p.r / p.kbm;
                let sl = kb / p.band.max(1);
                let blk = f64::from(p.kbm)
                    * (TURN
                        + 3.0 * f64::from(half)
                        + f64::from(tb) * (CALL + f64::from(nc))
                        + CALL
                        + f64::from(half))
                    + f64::from(tb) * f64::from(nc) * mac(kb)
                    + 2.0 * allred(tb * kb, p.band)
                    + f64::from(tb * kb) * 1.0
                    + f64::from(tb)
                        * (2.0 * (f64::from(kb) + 8.0 * CALL)
                            + f64::from(sl) * EXP
                            + 3.0 * EXP
                            + 4.0 * CALL
                            + f64::from(kb)
                            + f64::from(kb) * mac(nc))
                    + if *causal {
                        f64::from(tb * kb) * 4.0
                    } else {
                        0.0
                    };
                ("attention", f64::from(blocks) * blk)
            }
        };
        ops.push(OpCost {
            op: i,
            what,
            cycles: cy + OP,
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
    let (host, table) = crate::emit::gemm::table(p).map_or((0, 0), |t| {
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
