//! The frame bridge: a sealed frame's step becomes `&[Row]`.
//!
//! `DriverBackend::launch` hands over a frame — a lane roster, a KV page
//! translation, an admission high-water and an ordered list of steps. The
//! lowering wants a [`ForwardPlan`] and `&[Row]`. Everything between them is
//! what `forward.cpp` did, and **`forward.cpp` is not being ported**: it is
//! the family executor, and the lowering replaces it. What survives is the
//! frame questions it answered, and this is where they get their new answers.
//!
//! # The region table already answered the hard one
//!
//! The plan called this "the largest unknown, and the only piece with no
//! predecessor to read". Most of that turned out to be true of the *device*
//! half. The row half has a predecessor after all, and it is not in the C++:
//! **tart rung ③'s region table**, which the scheduler already computes and
//! the ABI already carries.
//!
//! A region is a run of rows sharing an axis bitset (`PIE_REGION_SIG_*`) and a
//! depth (`region_k`). A [`Row`] is a request's feature point on exactly those
//! axes. So the mapping is a bit-for-bit read, not a derivation:
//!
//! | region bit | row field |
//! |---|---|
//! | `MULTI_TOKEN` | `multi_token` |
//! | `HOOK`, `HOOK_PAGE_MASK` | `hooked` |
//! | `MASK` | `custom_mask` |
//! | `TRUNCATED` | `depth_k = Some(region_k)` |
//! | `LORA` | `lora` |
//!
//! and the region table's row runs are the *seriation* — which is what the
//! polymorphism guards need, since a text that splits on an axis requires that
//! axis's rows be contiguous. The two tasks meet here, exactly as planned.
//!
//! # When the table is empty
//!
//! An empty region table is the legacy discipline: no seriation was run, so
//! every row is the same point. That is not a refusal — it is a fire of one
//! region covering everything, which is precisely the monomorphic case the
//! Metal text serves today.
//!
//! [`ForwardPlan`]: model_ir::trace::ForwardPlan

use model_compiler::lower::Row;
use model_ir::trace::FireClass;

/// The region-signature bits, restated so this module does not depend on the
/// C ABI surface that [Task 9] retires. `driver`'s `local.rs` is where
/// they are defined; the values are the contract.
///
/// [Task 9]: https://example.invalid/
pub mod sig {
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
}

/// A depth operand meaning "the whole model" rather than a truncation.
///
/// The ABI spells a full-depth region with `PIE_MAX_LAYERS_FULL` rather than
/// with the absence of the `TRUNCATED` bit, so both have to be read: a region
/// that sets the bit and states the sentinel is full depth.
pub const MAX_LAYERS_FULL: u32 = u32::MAX;

/// One step's row-shaped facts, in the words the lowering uses.
///
/// Deliberately not a borrow of the ABI struct: this is the whole of what the
/// lowering reads out of a step, and naming it makes the bridge's surface the
/// list of things that have to be answered rather than a pointer into a
/// hundred-field descriptor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Step<'a> {
    /// One token per row of the fire.
    pub token_ids: &'a [u32],
    /// Request → row CSR: request `i` owns rows `[qo_indptr[i], qo_indptr[i+1])`.
    pub qo_indptr: &'a [u32],
    /// Row CSR over the regions (`regions + 1` entries), or empty for the
    /// legacy discipline.
    pub region_row_indptr: &'a [u32],
    /// Axis bitset per region, parallel to the CSR's segments.
    pub region_sig: &'a [u32],
    /// Depth operand per region, parallel to the CSR's segments.
    pub region_k: &'a [u32],
    /// The rows whose logits are read, each in ITS OWN REQUEST's numbering.
    ///
    /// Request `i` names rows relative to `qo_indptr[i]`, so a value here is
    /// an offset into that request's span and not a row of the fire. See
    /// `rows_of` for why this cost five backends a silent bug each.
    pub sampling_indices: &'a [u32],
    /// Request → readout CSR over [`Self::sampling_indices`], or empty when
    /// the fire names no read-outs at all.
    pub sampling_indptr: &'a [u32],
}

/// Why a step could not become rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unbridgeable {
    /// A step with no token rows launches nothing.
    NoRows,
    /// The region table's three arrays disagree about how many regions there
    /// are, so no region can be read without guessing which array is right.
    RaggedRegionTable {
        /// Segments the CSR describes.
        segments: usize,
        /// Entries in `region_sig`.
        sigs: usize,
        /// Entries in `region_k`.
        ks: usize,
    },
    /// The read-out table has values but no CSR that says whose they are.
    ///
    /// Refused rather than guessed. The tempting fallback -- treat the whole
    /// list as every request's, or let each request read its own last row --
    /// is silent and wrong in opposite directions, and the numbering is only
    /// readable through the CSR.
    RaggedReadoutTable {
        /// Read-out rows the table states.
        rows: usize,
        /// Segments the CSR describes.
        segments: usize,
        /// Requests the fire has.
        requests: usize,
    },
    /// A read-out names a row past the end of the request that named it.
    ///
    /// Under the request-local numbering this is the only way the table can
    /// be wrong that the fire-wide bound does not already catch, and it is
    /// the interesting one: the row still lands inside the FIRE, so without
    /// this refusal it silently reads a different conversation's logits.
    NotItsRow {
        /// Which request named it.
        request: usize,
        /// The row it named, in that request's own numbering.
        row: u32,
        /// How many rows that request has.
        span: u32,
    },
    /// A region names rows outside the fire, so the table was built against a
    /// different step.
    RegionOutOfRange {
        /// Which region.
        region: usize,
        /// Where it ends.
        end: u32,
        /// How many rows the fire has.
        rows: usize,
    },
    /// The regions do not tile the fire's rows. Every row belongs to exactly
    /// one region: a gap would leave rows with no feature point, and an
    /// overlap would give one row two.
    RegionsDoNotTile {
        /// The first row the regions disagree about.
        at: u32,
    },
}

/// Which fire class a step is.
///
/// One row per request is a decode; anything wider is a prefill. This is the
/// same test `driver-cuda`'s shell makes, and it is a property of the
/// step rather than a flag the scheduler has to keep consistent with it.
#[must_use]
pub fn fire_class(step: &Step<'_>) -> FireClass {
    let requests = step.qo_indptr.len().saturating_sub(1);
    if requests > 0 && step.token_ids.len() == requests {
        FireClass::Decode
    } else {
        FireClass::Prefill
    }
}

/// The rows a step lowers over.
///
/// # Errors
///
/// [`Unbridgeable`] naming which structural fact did not hold. Every variant is
/// drift between the scheduler's tables and the step they describe, not a
/// runtime condition.
pub fn rows_of(step: &Step<'_>) -> Result<Vec<Row>, Unbridgeable> {
    let n = step.token_ids.len();
    if n == 0 {
        return Err(Unbridgeable::NoRows);
    }
    // The region table arrives in WIRE-row space and is read in TOKEN-row
    // space: the engine counts one wire row per request
    // (`planned_region_table` walks `program_row_indptr`), while the rule
    // below numbers one row per token. `qo_indptr` maps between them -- wire
    // row `i` owns token rows `qo_indptr[i]..qo_indptr[i + 1]` -- so the
    // table is TRANSLATED here rather than reinterpreted. `driver-cuda`'s
    // `fire::launch` does the same thing at the same point and says so.
    //
    // Passing the wire table straight through is not a refusal but a wrong
    // shape: a single-request prefill states `[0, 1]`, which tiles token row
    // 0 and leaves every other row of the fire uncovered, so the rule below
    // refused with `DoNotTile { at: 1 }` and no prefill could lower at all.
    let region_row_indptr: Vec<u32> = step
        .region_row_indptr
        .iter()
        .map(|&wire_row| {
            // Out of range is passed through rather than clamped: a clamp
            // would name the wrong rows, where `u32::MAX` is refused by
            // `rows_from_regions` as the drift it is.
            step.qo_indptr
                .get(wire_row as usize)
                .copied()
                .unwrap_or(u32::MAX)
        })
        .collect();
    // ONE COPY OF THE RULE, and not this crate's own.
    //
    // What stood here was a line-for-line duplicate of
    // `model_compiler::lower::rows_from_regions` -- the same region walk, the
    // same tiling refusals, the same readout flag -- which is exactly what
    // `Row`'s own doc warns against: "two chances for a bit to be read wrongly
    // and no way to notice". It was not hypothetical. BOTH copies read
    // `sampling_indices` as rows of the FIRE when they are numbered inside the
    // request that named them, and correcting one would have left the other.
    // The refusals keep this crate's vocabulary; the rule does not.
    model_compiler::lower::rows_from_regions(
        n,
        model_compiler::lower::Readouts {
            indices: step.sampling_indices,
            indptr: step.sampling_indptr,
            qo_indptr: step.qo_indptr,
        },
        &region_row_indptr,
        step.region_sig,
        step.region_k,
    )
    .map_err(Unbridgeable::from)
}

/// The rows of the FIRE this step reads out, in fire order.
///
/// The one translation between the two numberings, and it exists because
/// there are two READERS. `sampling_indices` arrives numbered inside the
/// request that named it -- request `r`'s value `k` is row `qo_indptr[r] + k`
/// -- and `rows_of` reads it that way to set `Row::samples`, which is what
/// sizes the epilogue's gather. The gather's index list is the other reader:
/// `row_gather` binds `Source::Named(<keys::SamplingIndices as keys::Fact>::KEY)` and indexes the fire's stream
/// with it directly, so it needs the absolute row and nothing else will do.
///
/// Handing it the wire values instead is not a refusal, it is a wrong answer:
/// on `qo_indptr = [0, 2, 5]` with `[1, 2]` the gather takes stream rows 1 and
/// 2, and row 2 is the second request's FIRST token rather than its last. The
/// fire still runs, still produces two finite distributions of the right
/// width, and the second one is the wrong token's --
/// `a_request_prefills_the_same_way_beside_another_one` is what noticed, by
/// running the same request alone and comparing.
///
/// Derived from `Row::samples` rather than recomputed from `qo_indptr`, so
/// the list the gather reads and the count the epilogue sizes on cannot come
/// apart: they are one predicate, read once.
///
/// # Errors
///
/// Whatever [`rows_of`] refuses, and for the same reasons.
pub fn sampled_rows(step: &Step<'_>) -> Result<Vec<u32>, Unbridgeable> {
    Ok(rows_of(step)?
        .iter()
        .enumerate()
        .filter(|(_, r)| r.samples)
        .map(|(i, _)| u32::try_from(i).unwrap_or(u32::MAX))
        .collect())
}

/// A step, lowered: the whole host path from a sealed frame to rectangles.
///
/// The caller supplies the [`ForwardPlan`], and that is the north star's
/// division of labour — **the driver does not choose a text**. A plan comes
/// from the model that was loaded; this turns a step into the rows it lowers
/// over and hands both to the compiler.
///
/// From here `lowering::dispatch::plan` produces the grids and
/// `gpu::bind::encode::encode` runs them, so these three calls are the executor.
///
/// # Errors
///
/// [`Unbridged::Step`] when the step's tables do not describe its rows, or
/// [`Unbridged::Uncovered`] when the text states something the lowering cannot
/// yet flatten — which is a text-side gap, not a frame-side one.
///
/// [`ForwardPlan`]: model_ir::trace::ForwardPlan
pub fn lower_step(
    plan: &model_ir::trace::ForwardPlan,
    step: &Step<'_>,
) -> Result<model_compiler::lower::Lowered, Unbridged> {
    let rows = rows_of(step).map_err(Unbridged::Step)?;
    model_compiler::lower::lower(
        plan,
        &rows,
        model_compiler::lower::Fire {
            // Metal captures nothing: `Stepper` re-encodes every step, so no fire
            // is replayed across a different row split. The CUDA side sets this
            // when a captured graph outlives the split it was captured at.
            captures_across_splits: false,
        },
    )
    .map_err(|why| Unbridged::Uncovered(format!("{why:?}")))
}

impl From<model_compiler::lower::RegionDrift> for Unbridgeable {
    fn from(drift: model_compiler::lower::RegionDrift) -> Self {
        use model_compiler::lower::RegionDrift as D;
        match drift {
            D::Ragged { segments, sigs, ks } => Self::RaggedRegionTable { segments, sigs, ks },
            D::OutOfRange { region, end, rows } => Self::RegionOutOfRange { region, end, rows },
            D::DoNotTile { at } => Self::RegionsDoNotTile { at },
            D::NotItsRow { request, row, span } => Self::NotItsRow { request, row, span },
            D::RaggedReadout {
                rows,
                segments,
                requests,
            } => Self::RaggedReadoutTable {
                rows,
                segments,
                requests,
            },
        }
    }
}

/// Why a step did not reach rectangles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unbridged {
    /// The frame's tables did not describe its rows.
    Step(Unbridgeable),
    /// The text states something the lowering cannot flatten yet. Rendered
    /// rather than typed: `model-compiler`'s refusal is its own vocabulary and
    /// re-spelling it here would be a second copy to keep in step.
    Uncovered(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    fn step<'a>(token_ids: &'a [u32], qo_indptr: &'a [u32]) -> Step<'a> {
        Step {
            token_ids,
            qo_indptr,
            ..Step::default()
        }
    }

    #[test]
    fn one_row_a_request_is_a_decode_and_anything_wider_is_a_prefill() {
        assert_eq!(fire_class(&step(&[1, 2], &[0, 1, 2])), FireClass::Decode);
        assert_eq!(fire_class(&step(&[1, 2, 3], &[0, 3])), FireClass::Prefill);
    }

    #[test]
    fn an_empty_region_table_is_one_region_and_not_a_refusal() {
        // The legacy discipline. This is the case the Metal text serves today,
        // so it has to be a fire rather than an error.
        let rows = rows_of(&step(&[1, 2, 3], &[0, 3])).expect("a fire");
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.depth_k.is_none() && !r.lora));
    }

    #[test]
    fn a_region_signature_becomes_the_row_it_describes() {
        // Two requests of two tokens each, because a region table is stated in
        // WIRE rows -- one per request -- so two regions is two requests. The
        // token coverage is the same [0,2) / [2,4) it always was.
        let s = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 2, 4],
            region_row_indptr: &[0, 1, 2],
            region_sig: &[sig::TRUNCATED, sig::LORA | sig::MASK],
            region_k: &[4, MAX_LAYERS_FULL],
            sampling_indices: &[1],
            sampling_indptr: &[0, 0, 1],
        };
        let rows = rows_of(&s).expect("a fire");
        assert_eq!(
            rows[0].depth_k,
            Some(4),
            "the truncated region carries its k"
        );
        assert!(!rows[0].lora);
        assert_eq!(rows[2].depth_k, None);
        assert!(rows[2].lora && rows[2].custom_mask);
        assert!(rows[3].samples, "the readout row is the one sampling names");
        assert!(!rows[0].samples);
    }

    #[test]
    fn a_truncated_region_at_the_sentinel_is_full_depth_not_a_truncation() {
        // Reading the bit alone would truncate the fire at `u32::MAX` layers,
        // which lowers to a depth no model has and splits every rectangle.
        let s = Step {
            token_ids: &[1],
            qo_indptr: &[0, 1],
            region_row_indptr: &[0, 1],
            region_sig: &[sig::TRUNCATED],
            region_k: &[MAX_LAYERS_FULL],
            ..Step::default()
        };
        assert_eq!(rows_of(&s).expect("a fire")[0].depth_k, None);
    }

    #[test]
    fn both_hook_bits_mean_the_row_is_hooked() {
        for bit in [sig::HOOK, sig::HOOK_PAGE_MASK] {
            let s = Step {
                token_ids: &[1],
                qo_indptr: &[0, 1],
                region_row_indptr: &[0, 1],
                region_sig: &[bit],
                region_k: &[MAX_LAYERS_FULL],
                ..Step::default()
            };
            assert!(rows_of(&s).expect("a fire")[0].hooked);
        }
    }

    #[test]
    fn regions_that_leave_a_row_uncovered_are_refused_by_row() {
        // A row with no feature point would lower as the default, which is a
        // DIFFERENT program from the one the scheduler seriated it into.
        //
        // Stated in wire rows: two requests, and a table that names only the
        // first. Wire row 1 is where the second request's tokens begin, so
        // the gap opens at token row 2.
        let s = Step {
            token_ids: &[1, 2, 3],
            qo_indptr: &[0, 2, 3],
            region_row_indptr: &[0, 1],
            region_sig: &[0],
            region_k: &[MAX_LAYERS_FULL],
            ..Step::default()
        };
        assert_eq!(rows_of(&s), Err(Unbridgeable::RegionsDoNotTile { at: 2 }));
    }

    #[test]
    fn a_region_table_is_read_in_wire_rows_and_not_in_token_rows() {
        // The regression this pins: `planned_region_table` counts ONE row per
        // request, and the rule underneath counts one per token. A
        // single-request prefill therefore states `[0, 1]` however many
        // tokens it carries, and reading that as token rows tiles row 0 and
        // leaves the rest of the fire uncovered.
        //
        // Before the translation this refused every prefill with
        // `DoNotTile { at: 1 }`, which is what `pie run` hit on Metal: no
        // prompt of more than one token could lower at all.
        let s = Step {
            token_ids: &[1, 2, 3, 4, 5],
            qo_indptr: &[0, 5],
            region_row_indptr: &[0, 1],
            region_sig: &[sig::TRUNCATED],
            region_k: &[4],
            ..Step::default()
        };
        let rows = rows_of(&s).expect("one request's prefill is one region");
        assert_eq!(rows.len(), 5, "every token of the fire is a row");
        assert!(
            rows.iter().all(|r| r.depth_k == Some(4)),
            "the region covers all five rows, not just the first"
        );
    }

    #[test]
    fn a_region_table_whose_arrays_disagree_names_all_three_counts() {
        let s = Step {
            token_ids: &[1],
            qo_indptr: &[0, 1],
            region_row_indptr: &[0, 1],
            region_sig: &[0, 0],
            region_k: &[MAX_LAYERS_FULL],
            ..Step::default()
        };
        assert_eq!(
            rows_of(&s),
            Err(Unbridgeable::RaggedRegionTable {
                segments: 1,
                sigs: 2,
                ks: 1
            })
        );
    }

    #[test]
    fn a_step_with_no_rows_launches_nothing() {
        assert_eq!(rows_of(&step(&[], &[0])), Err(Unbridgeable::NoRows));
    }

    #[test]
    fn a_step_naming_no_readout_reads_each_requests_own_last_row() {
        // The decode case: the fire exists to produce those tokens, and a fire
        // whose logits nobody reads would lower with every readout dead.
        //
        // What this asserted before was `!rows[0].samples`: only the LAST row
        // of the whole fire. That is right for one request and wrong for two,
        // and two decoding requests in one fire is the ordinary case -- the
        // first request's readout was simply dead. The fire-wide reading and
        // the per-request one agree exactly when there is one request, which
        // is the entire reason this passed.
        let rows = rows_of(&step(&[1, 2], &[0, 1, 2])).expect("a fire");
        assert!(rows[0].samples, "request 0 reads its own last row");
        assert!(rows[1].samples, "and so does request 1");

        // One request, many rows: still exactly one readout, the last.
        let rows = rows_of(&step(&[1, 2, 3], &[0, 3])).expect("a fire");
        assert_eq!(
            rows.iter().map(|r| r.samples).collect::<Vec<_>>(),
            [false, false, true]
        );
    }

    /// The gather reads the FIRE's rows, not the wire's numbers.
    ///
    /// The two agree at one request and disagree at two, which is why this
    /// shipped: `row_gather` binds `Source::Named(<keys::SamplingIndices as keys::Fact>::KEY)` and indexes the
    /// stream with it, and every fixture with one request read correctly.
    /// On `[0, 2, 5]` the second request's last row is fire row 4 and its own
    /// number for it is 2 -- and 2 is a real row of the fire, belonging to
    /// that same request, so nothing refuses and the wrong token's hidden
    /// state goes to the head.
    #[test]
    fn the_gather_reads_fire_rows_where_the_wire_names_request_rows() {
        let two = Step {
            token_ids: &[1, 2, 3, 4, 5],
            qo_indptr: &[0, 2, 5],
            sampling_indices: &[1, 2],
            sampling_indptr: &[0, 1, 2],
            ..Step::default()
        };
        assert_eq!(sampled_rows(&two), Ok(vec![1, 4]));

        // One request: the two numberings coincide, which is the whole
        // reason the difference could ship.
        let one = Step {
            token_ids: &[1, 2, 3],
            qo_indptr: &[0, 3],
            sampling_indices: &[2],
            sampling_indptr: &[0, 1],
            ..Step::default()
        };
        assert_eq!(sampled_rows(&one), Ok(vec![2]));

        // A decode names nothing, and each request still reads its own last
        // row rather than the fire's.
        let decode = Step {
            token_ids: &[1, 2],
            qo_indptr: &[0, 1, 2],
            ..Step::default()
        };
        assert_eq!(sampled_rows(&decode), Ok(vec![0, 1]));

        // And it refuses exactly what `rows_of` refuses, rather than
        // returning a shorter list.
        assert!(
            sampled_rows(&Step {
                token_ids: &[1, 2, 3],
                qo_indptr: &[0, 3],
                sampling_indices: &[2],
                sampling_indptr: &[],
                ..Step::default()
            })
            .is_err()
        );
    }

    /// A read-out table with no CSR is refused, not spread over the requests.
    #[test]
    fn readout_rows_without_a_csr_to_place_them_are_refused() {
        let s = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 3, 4],
            sampling_indices: &[2, 0],
            sampling_indptr: &[],
            ..Step::default()
        };
        assert_eq!(
            rows_of(&s),
            Err(Unbridgeable::RaggedReadoutTable {
                rows: 2,
                segments: 0,
                requests: 2
            })
        );
        // And an empty table still needs no CSR -- that is the decode case.
        assert!(rows_of(&step(&[1, 2], &[0, 1, 2])).is_ok());
    }

    /// A read-out row is numbered inside its own request, not inside the fire.
    ///
    /// This is the case that cannot be told from the absolute reading by any
    /// single-request fixture, because `qo_indptr[0]` is zero and the offset
    /// IS the row. Two requests is the smallest fire that distinguishes them.
    #[test]
    fn a_readout_row_is_numbered_inside_its_own_request() {
        let s = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 3, 4],
            // Request 0 reads its own row 2; request 1 reads its own row 0.
            sampling_indices: &[2, 0],
            sampling_indptr: &[0, 1, 2],
            ..Step::default()
        };
        assert_eq!(
            rows_of(&s)
                .expect("a fire")
                .iter()
                .map(|r| r.samples)
                .collect::<Vec<_>>(),
            [false, false, true, true],
            "read absolutely this marks rows 0 and 2, which hands request 1 \
             request 0's first token"
        );
    }

    /// A row past the end of the request that named it is refused, rather than
    /// read out of the request next door.
    #[test]
    fn a_readout_past_its_own_request_is_refused() {
        let s = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 2, 4],
            // Request 0 has two rows and names a third.
            sampling_indices: &[2],
            sampling_indptr: &[0, 1, 1],
            ..Step::default()
        };
        assert_eq!(
            rows_of(&s),
            Err(Unbridgeable::NotItsRow {
                request: 0,
                row: 2,
                span: 2
            }),
            "it lands inside the FIRE, so only a per-request bound catches it"
        );
    }

    /// A request that names no read-out in a fire where another does still
    /// reads its own last row.
    #[test]
    fn a_request_naming_none_beside_one_that_does_still_reads_its_last_row() {
        let s = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 3, 4],
            sampling_indices: &[0],
            sampling_indptr: &[0, 1, 1],
            ..Step::default()
        };
        assert_eq!(
            rows_of(&s)
                .expect("a fire")
                .iter()
                .map(|r| r.samples)
                .collect::<Vec<_>>(),
            [true, false, false, true],
            "request 0 names its row 0; request 1 names none and falls back"
        );
    }
}
