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
        return Ok((0..rows)
            .map(|i| Row::from_region(0, region_sig::MAX_LAYERS_FULL, samples[i]))
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

#[cfg(test)]
mod region_tests {
    use super::*;

    #[test]
    fn each_axis_bit_lands_on_its_own_field() {
        let base = Row {
            samples: true,
            ..Row::default()
        };
        for (bit, expected) in [
            (
                region_sig::MULTI_TOKEN,
                Row {
                    multi_token: true,
                    ..base
                },
            ),
            (
                region_sig::MASK,
                Row {
                    custom_mask: true,
                    ..base
                },
            ),
            (
                region_sig::HOOK,
                Row {
                    hooked: true,
                    ..base
                },
            ),
            (
                region_sig::HOOK_PAGE_MASK,
                Row {
                    hooked: true,
                    ..base
                },
            ),
            (region_sig::LORA, Row { lora: true, ..base }),
        ] {
            assert_eq!(
                Row::from_region(bit, region_sig::MAX_LAYERS_FULL, true),
                expected,
                "bit {bit:#x} landed on the wrong field"
            );
        }
    }

    #[test]
    fn a_truncation_needs_both_the_bit_and_the_depth() {
        assert_eq!(
            Row::from_region(region_sig::TRUNCATED, 7, true).depth_k,
            Some(7)
        );
        assert_eq!(
            Row::from_region(region_sig::TRUNCATED, region_sig::MAX_LAYERS_FULL, true).depth_k,
            None,
            "the bit plus the sentinel is FULL depth, not a truncation at u32::MAX"
        );
        assert_eq!(
            Row::from_region(0, 7, true).depth_k,
            None,
            "no bit, no truncation"
        );
    }

    fn readouts<'a>(indices: &'a [u32], indptr: &'a [u32], qo: &'a [u32]) -> Readouts<'a> {
        Readouts {
            indices,
            indptr,
            qo_indptr: qo,
        }
    }

    #[test]
    fn an_empty_readout_list_samples_each_requests_last_row() {
        let rows = rows_from_regions(3, readouts(&[], &[], &[0, 3]), &[], &[], &[])
            .expect("legacy discipline");
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, true],
            "one request: its last row"
        );
        let many = rows_from_regions(3, readouts(&[], &[], &[0, 1, 2, 3]), &[], &[], &[])
            .expect("legacy discipline");
        assert_eq!(
            many.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [true, true, true]
        );
        let named = rows_from_regions(3, readouts(&[0, 1], &[0, 2], &[0, 3]), &[], &[], &[])
            .expect("named readout");
        assert_eq!(
            named.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [true, true, false]
        );
    }

    #[test]
    fn a_readout_row_is_numbered_inside_its_own_request() {
        let rows = rows_from_regions(4, readouts(&[2, 0], &[0, 1, 2], &[0, 3, 4]), &[], &[], &[])
            .expect("a fire");
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, true, true]
        );
    }

    #[test]
    fn a_readout_past_its_own_request_is_refused() {
        assert_eq!(
            rows_from_regions(4, readouts(&[2], &[0, 1, 1], &[0, 2, 4]), &[], &[], &[]),
            Err(RegionDrift::NotItsRow {
                request: 0,
                row: 2,
                span: 2
            }),
            "it lands inside the FIRE, so only a per-request bound catches it"
        );
    }

    #[test]
    fn readout_rows_without_a_csr_to_place_them_are_refused() {
        assert_eq!(
            rows_from_regions(4, readouts(&[2, 0], &[], &[0, 3, 4]), &[], &[], &[]),
            Err(RegionDrift::RaggedReadout {
                rows: 2,
                segments: 0,
                requests: 2
            })
        );
    }

    #[test]
    fn a_table_that_does_not_tile_is_refused() {
        let gap = rows_from_regions(3, readouts(&[], &[], &[0, 3]), &[0, 2], &[0], &[u32::MAX]);
        assert_eq!(gap, Err(RegionDrift::DoNotTile { at: 2 }));
        let ragged = rows_from_regions(
            2,
            readouts(&[], &[], &[0, 2]),
            &[0, 1, 2],
            &[0],
            &[u32::MAX, u32::MAX],
        );
        assert!(matches!(ragged, Err(RegionDrift::Ragged { .. })));
        let over = rows_from_regions(2, readouts(&[], &[], &[0, 2]), &[0, 5], &[0], &[u32::MAX]);
        assert!(matches!(over, Err(RegionDrift::OutOfRange { .. })));
    }

    #[test]
    fn a_region_carrying_an_adapter_marks_only_its_own_rows() {
        let rows = rows_from_regions(
            4,
            readouts(&[3], &[0, 1], &[0, 4]),
            &[0, 2, 4],
            &[0, region_sig::LORA],
            &[region_sig::MAX_LAYERS_FULL; 2],
        )
        .expect("a tiling table");
        assert_eq!(
            rows.iter().map(|r| r.lora).collect::<Vec<_>>(),
            [false, false, true, true],
            "the adapter is the second region's, not the fire's"
        );
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, false, true]
        );
    }
}
