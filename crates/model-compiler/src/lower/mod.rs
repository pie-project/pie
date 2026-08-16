//! LOWERING — the traced form to a flat launch list
//! (`.wiki/tart/dsl.md` "What one fire lowers to", migration step 6).
//!
//! ```text
//! per statement: compute extent (rows × layers)
//! match arms    → partition the extent into blocks
//! one Launch per rectangle
//! ```
//!
//! The target the doc states for the driver is a loop with no vocabulary
//! in it at all:
//!
//! ```cpp
//! for (const Launch& L : frame.launches)
//!     KERNELS[L.kernel](args + L.args, L.rows.lo, L.rows.hi,
//!                       L.layers.lo, L.layers.hi, stream);
//! ```
//!
//! **This is what a declared fire runs.** A driver builds the rows, calls
//! [`lower`], and executes the result; the old walk over the region IR
//! was deleted in the cutover's step 3, so there is no second form and no
//! switch between them. The one remaining consumer of the traced form
//! that does NOT come through here is the generated `.inc` — an
//! ahead-of-time emission of the same declaration that also carries the
//! unionized supergraph build.
//!
//! # Three decisions this module makes, from the doc's amendments
//!
//! **Row order is the ENGINE's.** `lower` takes the rows as the
//! scheduler's seriation already ordered them
//! (`crates/engine/src/scheduler/fire_plan.rs`) and does not choose a
//! permutation. Two independent permutation choosers would drift, and
//! the engine's is the one coupled to admission, framing and wave
//! discipline. What `lower` may do is REPORT what an order costs
//! ([`Lowered::rectangles`]), which is useful feedback for the seriation
//! key.
//!
//! **`Uncovered` is an ADMISSION answer, not a runtime fire split.** The
//! doc's sketch routed it to "the scheduler splits the fire", which
//! changes scheduling behaviour, and this project's standing constraint
//! is that runtime scheduling does not change — tart is a driver
//! feature. So [`Uncovered`] is what a group that cannot be served looks
//! like BEFORE it is formed: the engine's `LaunchGrouping::accepts`
//! already refuses unservable combinations, and this is the same answer
//! computed from the trace instead of from a hand-written rule.
//!
//! **`lower` assigns the buffers.** The DSL is pure SSA and carries no
//! buffer notion, so choosing one is a backend job — and it was the job
//! both CUDA executors did as FAMILY CONVENTION ("the normed activation
//! is `ws.norm_y`" in one, `ws.norm_x` in the other), which is what made
//! the executor two files. [`Buffers`] does it once, from the values'
//! own extents and liveness.

use std::collections::BTreeMap;
use std::ops::Range;

use model_ir::kernels::{self, Backend};
use model_ir::trace::{DType, Dim, ForwardPlan, GuardPred, Op, OpKind, PeelWindow, ValueId};

/// One row of a fire, as the engine's seriation ordered them.
///
/// These are exactly the axes the seriation key sorts on
/// (`(devgeo, mask, truncated, Reverse(k), hook, !multi_token,
/// arrival)`), so a run of rows sharing any one of them is contiguous by
/// construction — the sentinel this project promoted from a diagnostic
/// to a guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Row {
    pub multi_token: bool,
    pub custom_mask: bool,
    pub hooked: bool,
    pub lora: bool,
    /// Truncated at layer `k`, or `None` for full depth.
    pub depth_k: Option<u32>,
    /// The fire steers a graph replay, so the KV write takes explicit
    /// descriptors. A fire-wide fact today; a row field here because
    /// that is what it will become.
    pub write_desc: bool,
    /// The fire's attached programs read attention scores.
    pub wants_scores: bool,
    /// This row's logits are read — it is one of the fire's SAMPLED
    /// rows. A pure-decode fire samples every row; a prefill fire
    /// samples the last row of each request and gathers them, which is
    /// what makes the epilogue's row space [`Dim::Requests`] rather than
    /// [`Dim::Tokens`], and what the driver spells `logit_row_indices`.
    pub samples: bool,
}

/// The region-signature bits, as the wire states them.
///
/// Restated here rather than imported from the C ABI, for the reason
/// `driver-metal` gives for its own copy: `Row` must not depend on
/// the descriptor surface, and the VALUES are the contract. Stated beside
/// `Row` so the mapping below can be one function instead of one per
/// driver — the two shells were reading the same six bits into the same
/// eight fields, which is two chances for a bit to be read wrongly and no
/// way to notice.
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
    /// `region_k`'s "the whole model" sentinel — NOT the absence of
    /// [`TRUNCATED`]. A region that sets the bit and states this is full
    /// depth, so both have to be read.
    pub const MAX_LAYERS_FULL: u32 = u32::MAX;
}

/// Why a step's tables do not describe its rows.
///
/// Every variant is drift between the scheduler's tables and the step
/// they describe — a structural disagreement, not a runtime condition, so
/// a driver reports it rather than degrading.
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
    ///
    /// The row still lands inside the FIRE, so nothing else catches it and
    /// the fire runs, handing one request another's distribution.
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
///
/// Carried as one value because the three are unreadable apart:
/// `indices` are numbered inside a REQUEST, `indptr` says which request
/// named each, and `qo_indptr` says where that request's rows begin. A
/// caller that cannot state all three cannot read the field, which is the
/// point -- the previous signature took `indices` alone and every backend
/// that used it read them as rows of the fire.
#[derive(Clone, Copy, Debug, Default)]
pub struct Readouts<'a> {
    /// Rows read, each in ITS OWN REQUEST's numbering.
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
    ///
    /// # Errors
    ///
    /// [`RegionDrift::RaggedReadout`] when rows are named with no CSR to
    /// place them, and [`RegionDrift::NotItsRow`] when a request names a
    /// row it does not have.
    fn samples(&self, rows: usize) -> Result<Vec<bool>, RegionDrift> {
        let mut samples = vec![false; rows];
        let requests = self.qo_indptr.len().saturating_sub(1);
        // An EMPTY table is the decode case and needs no CSR. A table with
        // values in it is unreadable without one, because the values are
        // per request and nothing else says which. Falling back here would
        // be as silent as reading them absolutely.
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
            // A request that names none reads its OWN last row -- not the
            // fire's, which is only right when there is one request.
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
    ///
    /// A bit-for-bit read, not a derivation: the seriation already
    /// decided these axes and the table is its output stated once.
    #[must_use]
    pub const fn from_region(sig: u32, k: u32, samples: bool) -> Self {
        Self {
            multi_token: sig & region_sig::MULTI_TOKEN != 0,
            custom_mask: sig & region_sig::MASK != 0,
            // Both hook bits mean "this row runs hook programs". They
            // differ in what the hook WRITES, which changes the attention
            // path rather than the row's shape, so the row folds them.
            hooked: sig & (region_sig::HOOK | region_sig::HOOK_PAGE_MASK) != 0,
            lora: sig & region_sig::LORA != 0,
            // The bit says truncated; the sentinel says how far. Reading
            // only the bit would truncate the whole fire at `u32::MAX`
            // layers.
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
///
/// # Why this is shared rather than per driver
///
/// It is the only thing that turns the wire's axis bitset into the
/// lowering's guard predicates, and a shell that does not call it gets
/// `Row::default()` — every guard false, silently. `driver-cuda` did
/// exactly that: `vec![Row { samples: true, ..default() }; rows]`, so
/// `HasLora`, `HasCustomMask`, `HasStageHooks` and the depth truncation
/// could never hold no matter what the engine sent, and LoRA looked like
/// a missing feature when the wire had been carrying it all along.
///
/// # Arguments
///
/// [`Readouts`] is the fire's readout table. Its rows are numbered inside
/// the REQUEST that named them, not inside the fire; an EMPTY table means
/// every request reads its OWN last row — the decode case, where the fire
/// exists to produce those tokens — and not "no row is read".
///
/// This took the index list alone and read it as rows of the fire. The two
/// readings agree exactly when there is one request, so every fixture here
/// passed while it was wrong, and on a real batch it handed one request
/// another's logits. `samples` is what the epilogue filters to build
/// the gather, so it is the read-out itself.
///
/// An empty `region_row_indptr` is the legacy discipline: no seriation
/// ran, so the fire is one region of the default point. That is not a
/// refusal.
///
/// # Errors
///
/// [`RegionDrift`] naming which structural fact did not hold.
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


// ── The lowering, by file ──────────────────────────────────────────────
//
// `lower.rs` was 2,522 lines in one module. The cut follows what a reader
// asks for: the ROWS a fire is made of (here), the TYPES a driver reads
// (`shapes`), the WALK that produces them (`walk`), what an unstated
// statement means (`semantic`), and where a value lives (`buffers`).
//
// Every name is re-exported flat, because `model_compiler::lower::Launch`
// is what a driver spells and a file boundary is not an API.
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

    /// Every bit lands on its own field.
    ///
    /// A bit-for-bit read has exactly one failure mode — a bit read into
    /// the wrong field — and it is invisible: the fire runs, takes the
    /// wrong guard arm, and returns plausible tokens. So each bit is set
    /// ALONE and the whole row is compared, which is what catches a
    /// transposition rather than merely a missing read.
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

    /// The truncation bit says WHETHER; `region_k` says how far.
    ///
    /// Reading only the bit truncates the whole fire at `u32::MAX`
    /// layers, which is the ABI's spelling for full depth — so the naive
    /// read produces a depth nobody asked for out of a region that said
    /// "all of it".
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

    /// An empty readout list means EACH REQUEST's last row is read, not none
    /// and not only the fire's.
    #[test]
    fn an_empty_readout_list_samples_each_requests_last_row() {
        let rows = rows_from_regions(3, readouts(&[], &[], &[0, 3]), &[], &[], &[])
            .expect("legacy discipline");
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, true],
            "one request: its last row"
        );
        // Three decoding requests. Read fire-wide this marks only row 2 and
        // leaves the first two requests with a dead readout.
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

    /// A readout row is numbered inside its own request, not inside the fire.
    ///
    /// No single-request fixture can tell the two apart -- `qo_indptr[0]` is
    /// zero, so the offset IS the row -- which is why this helper shipped to
    /// two backends reading it the wrong way.
    #[test]
    fn a_readout_row_is_numbered_inside_its_own_request() {
        // Request 0 owns rows 0..3 and reads its row 2; request 1 owns row 3
        // and reads its row 0. Read absolutely: rows 2 and 0.
        let rows = rows_from_regions(4, readouts(&[2, 0], &[0, 1, 2], &[0, 3, 4]), &[], &[], &[])
            .expect("a fire");
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, true, true]
        );
    }

    /// A row past the end of the request that named it is refused, rather
    /// than read out of the request next door.
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

    /// Readout rows with no CSR to place them are refused, not spread.
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

    /// Regions tile the rows, and a table that does not is drift.
    #[test]
    fn a_table_that_does_not_tile_is_refused() {
        // Row 2 is claimed by nobody.
        let gap = rows_from_regions(3, readouts(&[], &[], &[0, 3]), &[0, 2], &[0], &[u32::MAX]);
        assert_eq!(gap, Err(RegionDrift::DoNotTile { at: 2 }));
        // The three arrays disagree on how many regions there are.
        let ragged = rows_from_regions(
            2,
            readouts(&[], &[], &[0, 2]),
            &[0, 1, 2],
            &[0],
            &[u32::MAX, u32::MAX],
        );
        assert!(matches!(ragged, Err(RegionDrift::Ragged { .. })));
        // A region names a row the step does not have.
        let over = rows_from_regions(2, readouts(&[], &[], &[0, 2]), &[0, 5], &[0], &[u32::MAX]);
        assert!(matches!(over, Err(RegionDrift::OutOfRange { .. })));
    }

    /// A real two-region fire: one plain span and one carrying an adapter.
    ///
    /// The case the CUDA shell could not express at all, and the reason
    /// this function is shared: `HasLora` asks `rows.iter().any(|r|
    /// r.lora)`, so a shell that never filled the field answered NO to a
    /// step that said YES.
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
