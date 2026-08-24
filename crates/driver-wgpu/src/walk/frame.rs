//! A sealed step, in the words the executor reads: which fire class it is, and
//! which of its rows are read out.
//!
//! THE MODULE THAT DECIDED THIS CRATE. `driver-metal`'s copy and
//! `driver-wgpu`'s were identical to the character -- 167 lines of CSR
//! arithmetic, twice -- and wgpu's own note said what to do about it: *"Kept as
//! a copy rather than shared... If a third plane wants it, that is the moment
//! to weigh a move -- and the two copies must not drift before then, which is
//! what the identical test names are for."*
//!
//! `driver-vulkan` is the third plane and it is coming back. The weighing
//! happened, and the answer is that a shared crate for eighty lines of CSR
//! arithmetic is not worth a manifest -- but this module was never alone. It
//! comes with the walk, the bound statement, the resolve pass and the lane, and
//! together those are 1,030 lines a plane does not get to have an opinion about.
//!
//! It names NEITHER plane and takes no type parameter: both functions are about
//! the FRAME -- the scheduler's tables and the numbering they are written in --
//! and a frame is the same object whether the fire that follows it dispatches
//! WGSL, MSL or SPIR-V.
//!
//! What stood here before it was 666 lines of `lowering::frame` that turned a
//! step into `model_compiler::lower::Row`s, because the legacy lowering
//! flattened a text ONCE PER FIRE against those rows. A `Program` is bound at
//! LOAD and picked by fact word, so there are no rows to build and nothing
//! reads a region signature. The two that remain are both about the FRAME
//! rather than about the lowering, which is why they outlived it.

use model_ir::plan::FireClass;

/// One step's row-shaped facts.
///
/// Deliberately not a borrow of the ABI struct: this is the whole of what the
/// executor reads out of a step, and naming it makes the bridge's surface the
/// list of things that have to be answered rather than a pointer into a
/// hundred-field descriptor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Step<'a> {
    /// One token per row of the fire.
    pub token_ids: &'a [u32],
    /// Request → row CSR: request `i` owns rows `[qo_indptr[i], qo_indptr[i+1])`.
    pub qo_indptr: &'a [u32],
    /// The rows whose logits are read, each in ITS OWN REQUEST's numbering.
    ///
    /// Request `i` names rows relative to `qo_indptr[i]`, so a value here is
    /// an offset into that request's span and not a row of the fire. See
    /// [`sampled_rows`] for what that cost.
    pub sampling_indices: &'a [u32],
    /// Request → readout CSR over [`Self::sampling_indices`], or empty when
    /// the fire names no read-outs at all.
    pub sampling_indptr: &'a [u32],
}

/// Why a step's tables do not describe its rows.
///
/// Every variant is DRIFT between the scheduler's tables and the step they
/// describe, not a runtime condition: no retry changes one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unbridgeable {
    /// A step with no token rows launches nothing.
    NoRows,
    /// The read-out table has values but no CSR that says whose they are.
    ///
    /// Refused rather than guessed. The tempting fallbacks — treat the whole
    /// list as every request's, or let each request read its own last row —
    /// are silent and wrong in opposite directions, and the numbering is only
    /// readable through the CSR.
    RaggedReadoutTable {
        /// Read-out rows the table states.
        rows: usize,
        /// Segments the CSR describes.
        segments: usize,
        /// Requests the fire holds.
        requests: usize,
    },
    /// A request named a row outside its own span.
    NotItsRow {
        /// Which request.
        request: usize,
        /// The row it named, in that request's own numbering.
        row: u32,
        /// How many rows that request has.
        span: u32,
    },
}

impl core::fmt::Display for Unbridgeable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoRows => write!(f, "a step with no token rows launches nothing"),
            Self::RaggedReadoutTable {
                rows,
                segments,
                requests,
            } => write!(
                f,
                "{rows} read-out row(s) over {segments} CSR segment(s) for {requests} request(s)"
            ),
            Self::NotItsRow { request, row, span } => {
                write!(f, "request {request} reads row {row} of its own {span}")
            }
        }
    }
}

impl std::error::Error for Unbridgeable {}

/// Which fire class a step is.
///
/// One row per request is a decode; anything wider is a prefill. This is the
/// same test `driver-cuda`'s shell makes, and it is a property of the step
/// rather than a flag the scheduler has to keep consistent with it.
#[must_use]
pub fn fire_class(step: &Step<'_>) -> FireClass {
    let requests = step.qo_indptr.len().saturating_sub(1);
    if requests > 0 && step.token_ids.len() == requests {
        FireClass::Decode
    } else {
        FireClass::Prefill
    }
}

/// The rows of the FIRE this step reads out, in fire order.
///
/// THE ONE TRANSLATION BETWEEN THE TWO NUMBERINGS, and it exists because
/// there are two readers. `sampling_indices` arrives numbered inside the
/// request that named it — request `r`'s value `k` is row `qo_indptr[r] + k`
/// — while the gather binds the fire's `sampling_indices` plane and indexes
/// the logits stream with it directly, so it needs the ABSOLUTE row and
/// nothing else will do.
///
/// Handing it the wire values instead is not a refusal, it is a wrong answer:
/// on `qo_indptr = [0, 2, 5]` with `[1, 2]` the gather takes stream rows 1 and
/// 2, and row 2 is the second request's FIRST token rather than its last. The
/// fire still runs, still produces two finite distributions of the right
/// width, and the second one is the wrong token's.
///
/// # It reads the CSR directly now
///
/// It used to derive the list from `Row::samples`, so that the count the
/// epilogue sized on and the list the gather read could not come apart. There
/// is no `Row`: the epilogue's width is the program's own slot, sized by the
/// walk. So the translation is the CSR's own arithmetic, done once here.
///
/// # Errors
///
/// [`Unbridgeable`] naming which structural fact did not hold.
pub fn sampled_rows(step: &Step<'_>) -> Result<Vec<u32>, Unbridgeable> {
    if step.token_ids.is_empty() {
        return Err(Unbridgeable::NoRows);
    }
    let requests = step.qo_indptr.len().saturating_sub(1);
    if step.sampling_indices.is_empty() {
        return Ok(Vec::new());
    }
    // A read-out table with no CSR is unreadable, and the two plausible
    // guesses are wrong in opposite directions.
    if step.sampling_indptr.len() != requests + 1 {
        return Err(Unbridgeable::RaggedReadoutTable {
            rows: step.sampling_indices.len(),
            segments: step.sampling_indptr.len().saturating_sub(1),
            requests,
        });
    }
    let mut out = Vec::with_capacity(step.sampling_indices.len());
    for r in 0..requests {
        let (from, to) = (
            step.sampling_indptr[r] as usize,
            step.sampling_indptr[r + 1] as usize,
        );
        let base = step.qo_indptr[r];
        let span = step.qo_indptr[r + 1].saturating_sub(base);
        for &row in step
            .sampling_indices
            .get(from..to)
            .ok_or(Unbridgeable::RaggedReadoutTable {
                rows: step.sampling_indices.len(),
                segments: requests,
                requests,
            })?
        {
            if row >= span {
                return Err(Unbridgeable::NotItsRow {
                    request: r,
                    row,
                    span,
                });
            }
            out.push(base + row);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row per request is a decode; anything wider is a prefill.
    #[test]
    fn the_class_is_a_property_of_the_step() {
        let decode = Step {
            token_ids: &[7, 8],
            qo_indptr: &[0, 1, 2],
            ..Step::default()
        };
        assert_eq!(fire_class(&decode), FireClass::Decode);
        let prefill = Step {
            token_ids: &[7, 8, 9],
            qo_indptr: &[0, 3],
            ..Step::default()
        };
        assert_eq!(fire_class(&prefill), FireClass::Prefill);
        // No requests at all is not a decode of nothing.
        assert_eq!(fire_class(&Step::default()), FireClass::Prefill);
    }

    /// THE TRANSLATION, on the example that cost five backends a silent bug.
    #[test]
    fn a_readout_is_translated_into_the_fires_own_numbering() {
        let step = Step {
            token_ids: &[1, 2, 3, 4, 5],
            qo_indptr: &[0, 2, 5],
            // Each request names its own LAST row: request 0's row 1, and
            // request 1's row 2.
            sampling_indices: &[1, 2],
            sampling_indptr: &[0, 1, 2],
        };
        assert_eq!(
            sampled_rows(&step).expect("both rows are inside their requests"),
            vec![1, 4],
            "row 4 is the fire's, and 2 was the request's",
        );
    }

    /// A row outside its own request refuses rather than reading a neighbour's
    /// token.
    #[test]
    fn a_request_cannot_read_another_requests_row() {
        let step = Step {
            token_ids: &[1, 2, 3, 4, 5],
            qo_indptr: &[0, 2, 5],
            sampling_indices: &[2],
            sampling_indptr: &[0, 1, 1],
        };
        assert_eq!(
            sampled_rows(&step),
            Err(Unbridgeable::NotItsRow {
                request: 0,
                row: 2,
                span: 2
            }),
        );
    }

    /// A read-out table with no CSR is refused, not guessed at.
    #[test]
    fn a_readout_table_without_a_csr_refuses() {
        let step = Step {
            token_ids: &[1, 2, 3],
            qo_indptr: &[0, 3],
            sampling_indices: &[2],
            sampling_indptr: &[],
        };
        assert!(matches!(
            sampled_rows(&step),
            Err(Unbridgeable::RaggedReadoutTable { .. })
        ));
    }

    /// A fire that names no read-outs is not a refusal.
    #[test]
    fn a_fire_that_samples_nothing_is_an_empty_list() {
        let step = Step {
            token_ids: &[1, 2, 3],
            qo_indptr: &[0, 3],
            ..Step::default()
        };
        assert_eq!(sampled_rows(&step), Ok(Vec::new()));
    }

    /// A STEP WITH NO TOKEN ROWS LAUNCHES NOTHING, and says so by name.
    ///
    /// This refusal arrived here with the module and it arrived UNTESTED:
    /// `driver-wgpu/tests/citations.rs` had `Unbridgeable::NoRows` on its list
    /// of refusals no test names, filed under "reachable and untested, and this
    /// is the real gap" -- it had been named by a suite that walked a lowered
    /// plan, and that suite went with the walk. A refusal nothing names is one
    /// whose condition could be inverted with every suite still green, and the
    /// condition here is `is_empty()`, which inverts to "only an EMPTY fire
    /// launches".
    #[test]
    fn a_step_with_no_rows_refuses_by_name() {
        assert_eq!(
            sampled_rows(&Step {
                qo_indptr: &[0, 1],
                sampling_indices: &[0],
                sampling_indptr: &[0, 1],
                ..Step::default()
            }),
            Err(Unbridgeable::NoRows),
        );
    }
}
