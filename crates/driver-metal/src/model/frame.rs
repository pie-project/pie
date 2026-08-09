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
//! [`ForwardPlan`]: model_compiler::trace::ForwardPlan

use model_compiler::lower::Row;
use model_compiler::trace::FireClass;

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
    /// The rows whose logits are read.
    pub sampling_indices: &'a [u32],
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

/// The row a region's signature describes.
fn row_of(sig: u32, k: u32, samples: bool) -> Row {
    Row {
        multi_token: sig & sig::MULTI_TOKEN != 0,
        custom_mask: sig & sig::MASK != 0,
        // Both hook bits mean "this row runs hook programs". They differ in
        // what the hook WRITES, which changes the attention path rather than
        // the row's shape, so the row folds them.
        hooked: sig & (sig::HOOK | sig::HOOK_PAGE_MASK) != 0,
        lora: sig & sig::LORA != 0,
        // The bit says truncated; the sentinel says how far. A region that
        // sets the bit and states the sentinel is full depth, and reading only
        // the bit would truncate the whole fire at `u32::MAX` layers.
        depth_k: (sig & sig::TRUNCATED != 0 && k != MAX_LAYERS_FULL).then_some(k),
        write_desc: false,
        wants_scores: false,
        samples,
    }
}

/// The rows a step lowers over.
///
/// # Errors
///
/// [`Unbridgeable`] naming which structural fact did not hold. Every variant is
/// drift between the scheduler's tables and the step they describe, not a
/// runtime condition.
pub fn rows_of(step: &Step<'_>) -> Result<Vec<Row>, Unbridgeable> {    let n = step.token_ids.len();
    if n == 0 {
        return Err(Unbridgeable::NoRows);
    }

    // A row whose logits nobody reads still runs; `samples` is what says its
    // readout is live, and it is per row rather than per region because the
    // sampling CSR is not an axis the seriation groups on.
    let mut samples = vec![false; n];
    for &row in step.sampling_indices {
        if let Some(slot) = samples.get_mut(row as usize) {
            *slot = true;
        }
    }
    // A step that names no readout at all is one whose last row is read —
    // the decode case, where the fire exists to produce that token.
    if step.sampling_indices.is_empty() {
        samples[n - 1] = true;
    }

    let segments = step.region_row_indptr.len().saturating_sub(1);
    if step.region_row_indptr.is_empty() {
        // The legacy discipline: no seriation ran, so the fire is one region
        // of the default point. Not a refusal — this is the monomorphic case.
        return Ok((0..n).map(|i| row_of(0, MAX_LAYERS_FULL, samples[i])).collect());
    }
    if segments != step.region_sig.len() || segments != step.region_k.len() {
        return Err(Unbridgeable::RaggedRegionTable {
            segments,
            sigs: step.region_sig.len(),
            ks: step.region_k.len(),
        });
    }

    let mut rows: Vec<Option<Row>> = vec![None; n];
    for region in 0..segments {
        let start = step.region_row_indptr[region];
        let end = step.region_row_indptr[region + 1];
        if end as usize > n || start > end {
            return Err(Unbridgeable::RegionOutOfRange {
                region,
                end,
                rows: n,
            });
        }
        for row in start..end {
            let slot = &mut rows[row as usize];
            if slot.is_some() {
                return Err(Unbridgeable::RegionsDoNotTile { at: row });
            }
            *slot = Some(row_of(
                step.region_sig[region],
                step.region_k[region],
                samples[row as usize],
            ));
        }
    }
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.ok_or(Unbridgeable::RegionsDoNotTile {
                at: u32::try_from(i).unwrap_or(u32::MAX),
            })
        })
        .collect()
}

/// A step, lowered: the whole host path from a sealed frame to rectangles.
///
/// The caller supplies the [`ForwardPlan`], and that is the north star's
/// division of labour — **the driver does not choose a text**. A plan comes
/// from the model that was loaded; this turns a step into the rows it lowers
/// over and hands both to the compiler.
///
/// From here `model::dispatch::plan` produces the grids and
/// `model::encode::encode` runs them, so these three calls are the executor.
///
/// # Errors
///
/// [`Unbridged::Step`] when the step's tables do not describe its rows, or
/// [`Unbridged::Uncovered`] when the text states something the lowering cannot
/// yet flatten — which is a text-side gap, not a frame-side one.
///
/// [`ForwardPlan`]: model_compiler::trace::ForwardPlan
pub fn lower_step(
    plan: &model_compiler::trace::ForwardPlan,
    step: &Step<'_>,
) -> Result<model_compiler::lower::Lowered, Unbridged> {
    let rows = rows_of(step).map_err(Unbridged::Step)?;
    model_compiler::lower::lower(plan, &rows, model_compiler::lower::Fire {
        // Metal captures nothing: `Stepper` re-encodes every step, so no fire
        // is replayed across a different row split. The CUDA side sets this
        // when a captured graph outlives the split it was captured at.
        captures_across_splits: false,
    })
    .map_err(|why| Unbridged::Uncovered(format!("{why:?}")))
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
        let s = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 4],
            region_row_indptr: &[0, 2, 4],
            region_sig: &[sig::TRUNCATED, sig::LORA | sig::MASK],
            region_k: &[4, MAX_LAYERS_FULL],
            sampling_indices: &[3],
        };
        let rows = rows_of(&s).expect("a fire");
        assert_eq!(rows[0].depth_k, Some(4), "the truncated region carries its k");
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
        let s = Step {
            token_ids: &[1, 2, 3],
            qo_indptr: &[0, 3],
            region_row_indptr: &[0, 2],
            region_sig: &[0],
            region_k: &[MAX_LAYERS_FULL],
            ..Step::default()
        };
        assert_eq!(rows_of(&s), Err(Unbridgeable::RegionsDoNotTile { at: 2 }));
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
    fn a_step_naming_no_readout_reads_its_last_row() {
        // The decode case: the fire exists to produce that token, and a fire
        // whose logits nobody reads would lower with every readout dead.
        let rows = rows_of(&step(&[1, 2], &[0, 1, 2])).expect("a fire");
        assert!(!rows[0].samples);
        assert!(rows[1].samples);
    }
}
