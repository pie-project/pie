//! Lowering turns a traced fire into rows, buffers, and a flat launch list.

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::kernels::{self, Backend};
use model_ir::trace::{DType, Dim, ForwardPlan, GuardPred, Op, OpKind, PeelWindow, ValueId};

/// One row of a fire, in engine seriation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Row {
    pub multi_token: bool,
    pub custom_mask: bool,
    pub hooked: bool,
    pub lora: bool,
    /// Truncated at layer `k`, or `None` for full depth.
    pub depth_k: Option<u32>,
    /// The fire steers graph replay with explicit KV write descriptors.
    pub write_desc: bool,
    /// The fire's attached programs read attention scores.
    pub wants_scores: bool,
    /// This row's logits are read.
    pub samples: bool,
}

/// The region-signature bits, as the wire states them.
pub mod region_sig {
    /// The region's members carry multi-token qo windows.
    pub const MULTI_TOKEN: u32 = 1 << 0;
    /// Attention-stage hook programs.
    pub const HOOK: u32 = 1 << 1;
    /// A user (custom) attention mask.
    pub const MASK: u32 = 1 << 2;
    /// A depth truncation; the region's `k` is its `region_k`.
    pub const TRUNCATED: u32 = 1 << 3;
    /// A span-grouped correction (lora) program.
    pub const LORA: u32 = 1 << 4;
    /// The region's hooks write the `attn_page_mask` sink.
    pub const HOOK_PAGE_MASK: u32 = 1 << 5;
    /// `region_k`'s full-depth sentinel.
    pub const MAX_LAYERS_FULL: u32 = u32::MAX;
}

/// Why a step's tables do not describe its rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionDrift {
    /// The three region arrays disagree on how many regions there are.
    Ragged {
        segments: usize,
        sigs: usize,
        ks: usize,
    },
    /// A region names rows the step does not have.
    OutOfRange {
        region: usize,
        end: u32,
        rows: usize,
    },
    /// Two regions claim one row, or no region claims it.
    DoNotTile { at: u32 },
    /// A read-out names a row past the end of the request that named it.
    NotItsRow {
        request: usize,
        /// In that request's own numbering.
        row: u32,
        /// How many rows the request has.
        span: u32,
    },
    /// Read-out rows with no CSR that says which request named them.
    RaggedReadout {
        rows: usize,
        segments: usize,
        requests: usize,
    },
}

/// The fire's read-out table, and the row geometry that gives it meaning.
#[derive(Clone, Copy, Debug, Default)]
pub struct Readouts<'a> {
    /// Rows read, each numbered inside its own request.
    pub indices: &'a [u32],
    /// Request -> readout CSR over `indices`, empty when none are named.
    pub indptr: &'a [u32],
    /// Request -> row CSR for the fire.
    pub qo_indptr: &'a [u32],
}

impl<'a> Readouts<'a> {
    /// The rows request `r` reads, in its own numbering.
    fn of(&self, r: usize) -> &'a [u32] {
        let (Some(&lo), Some(&hi)) = (self.indptr.get(r), self.indptr.get(r + 1)) else {
            return &[];
        };
        if hi < lo {
            return &[];
        }
        self.indices.get(lo as usize..hi as usize).unwrap_or(&[])
    }

    /// Which rows of the fire this table marks as read.
    fn samples(&self, rows: usize) -> Result<Vec<bool>, RegionDrift> {
        let mut samples = vec![false; rows];
        let requests = self.qo_indptr.len().saturating_sub(1);
        if !self.indices.is_empty() && self.indptr.len() != requests + 1 {
            return Err(RegionDrift::RaggedReadout {
                rows: self.indices.len(),
                segments: self.indptr.len().saturating_sub(1),
                requests,
            });
        }
        for r in 0..requests {
            let (lo, hi) = (self.qo_indptr[r], self.qo_indptr[r + 1]);
            let span = hi.saturating_sub(lo);
            if span == 0 {
                continue;
            }
            let named = self.of(r);
            if named.is_empty() {
                samples[(hi - 1) as usize] = true;
                continue;
            }
            for &row in named {
                if row >= span {
                    return Err(RegionDrift::NotItsRow {
                        request: r,
                        row,
                        span,
                    });
                }
                samples[(lo + row) as usize] = true;
            }
        }
        Ok(samples)
    }
}

impl Row {
    /// The row a region's signature describes.
    #[must_use]
    pub const fn from_region(sig: u32, k: u32, samples: bool) -> Self {
        Self {
            multi_token: sig & region_sig::MULTI_TOKEN != 0,
            custom_mask: sig & region_sig::MASK != 0,
            hooked: sig & (region_sig::HOOK | region_sig::HOOK_PAGE_MASK) != 0,
            lora: sig & region_sig::LORA != 0,
            depth_k: if sig & region_sig::TRUNCATED != 0 && k != region_sig::MAX_LAYERS_FULL {
                Some(k)
            } else {
                None
            },
            write_desc: false,
            wants_scores: false,
            samples,
        }
    }
}

/// The rows a step lowers over, from its region table and readout list.
pub fn rows_from_regions(
    rows: usize,
    readouts: Readouts<'_>,
    region_row_indptr: &[u32],
    region_sig_bits: &[u32],
    region_k: &[u32],
) -> Result<Vec<Row>, RegionDrift> {
    let samples = readouts.samples(rows)?;

    let segments = region_row_indptr.len().saturating_sub(1);
    if region_row_indptr.is_empty() {
        // NO TABLE IS NOT "NO AXES", AND READING IT THAT WAY COST A PREFILL
        // ITS ATTENTION KERNEL.
        //
        // A sig of zero says every axis is off, and `MULTI_TOKEN` is one of
        // them: `GuardPred::WindowOne` reads it as "this row's query window is
        // one row", which is what makes a fire take `sdpa_paged_decode`
        // instead of the tiled or mma kernel. So a 2048-token prefill arriving
        // without a table was lowered onto the per-row decode kernel, which
        // re-reads the whole key run once per query row. It is not wrong --
        // the logits are the same and every MLX gate still passes -- it is
        // quadratic where the tiled form is not. Measured on Llama-3.2-1B by
        // `attention_is_a_minority_of_a_long_prefill`: 2.104e-4 ms per token²
        // against the mma kernel's 2.589e-5, attention 52% of the fire instead
        // of 8%.
        //
        // And an empty table is a PRODUCTION case, not just a fixture's
        // shorthand: `engine::scheduler::batch::planned_region_table` declines
        // to build one whenever the attribution CSR is absent or any member
        // owns no wire row, and returns three empty vectors.
        //
        // The width is knowable without the table, because `qo_indptr` is the
        // same fact the table was built from -- that function's own rule is
        // `w[1] - w[0] > 1` over the request's windows. Deriving it here is
        // therefore not a guess standing in for the table; it is the one
        // remaining reader computing what the absent producer would have said.
        // The other axes are NOT derivable -- a hook, a mask, a LoRA and a
        // depth are the scheduler's claims about a request and nothing in the
        // step implies them -- so they stay off, which is what an absent table
        // has always meant for them.
        let mut sig = vec![0u32; rows];
        for r in 0..readouts.qo_indptr.len().saturating_sub(1) {
            let (lo, hi) = (readouts.qo_indptr[r], readouts.qo_indptr[r + 1]);
            if hi <= lo || hi.saturating_sub(lo) == 1 {
                continue;
            }
            for row in lo..hi.min(u32::try_from(rows).unwrap_or(u32::MAX)) {
                sig[row as usize] = region_sig::MULTI_TOKEN;
            }
        }
        return Ok((0..rows)
            .map(|i| Row::from_region(sig[i], region_sig::MAX_LAYERS_FULL, samples[i]))
            .collect());
    }
    if segments != region_sig_bits.len() || segments != region_k.len() {
        return Err(RegionDrift::Ragged {
            segments,
            sigs: region_sig_bits.len(),
            ks: region_k.len(),
        });
    }

    let mut out: Vec<Option<Row>> = vec![None; rows];
    for region in 0..segments {
        let (start, end) = (region_row_indptr[region], region_row_indptr[region + 1]);
        if end as usize > rows || start > end {
            return Err(RegionDrift::OutOfRange { region, end, rows });
        }
        for row in start..end {
            let slot = &mut out[row as usize];
            if slot.is_some() {
                return Err(RegionDrift::DoNotTile { at: row });
            }
            *slot = Some(Row::from_region(
                region_sig_bits[region],
                region_k[region],
                samples[row as usize],
            ));
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.ok_or(RegionDrift::DoNotTile {
                at: u32::try_from(i).unwrap_or(u32::MAX),
            })
        })
        .collect()
}

mod buffers;
mod semantics;
mod shapes;
mod walk;

pub use buffers::*;
pub use shapes::*;
pub use walk::*;
