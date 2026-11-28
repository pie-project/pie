//! The top rung a CUDA load lays its fires up to
//! ([`engine::load::top_rung`]): of the buckets the asked fire size's
//! lattice offers, the one the ruling picks over the room the card has
//! after the weight tier, every fire timed by the estimate from the
//! device's rates (its profile's `[rates]`) and the model's counts
//! ([`engine::load::Counts`]). A device whose profile states no rates has
//! nothing to weigh the rungs by: the load is refused, naming it, or, where it
//! skips measuring, takes the narrowest decode rung ([`narrowest`]).

use engine::load::{Counts, Kept, RungFacts, TopRung};
use poem_ir::Trace;

use crate::error::{Fault, Result};

/// Why a rung is no rung of this load: its plan does not compile at it.
pub const NO_PLAN: &str = "its plan does not compile at this budget";

/// Why a rung is no rung of this load: its span leaves no room for one
/// sequence beside it.
pub const NO_ROOM: &str = "no room is left beside it for one sequence";

/// The counts of `trace` over the planes this engine keeps: its token
/// tables and routed banks streamed apart from the dense weights.
pub fn counts(trace: &Trace) -> Result<Counts> {
    let planes = crate::weights::plane_bytes(trace)?;
    let tables = crate::store::layout::streamed_rows(trace);
    let banks = crate::store::layout::streamed_experts(trace);
    let attention = poem_exec::store::kv::attention_counts(trace)?;
    Ok(Counts::of(
        trace,
        Kept {
            plane_bytes: &planes,
            tables: &tables,
            banks: &banks,
        },
        attention,
    ))
}

/// One rung offered: a fire of `tokens`, its compiled plan's span and
/// launches (or why it does not compile: [`NO_PLAN`] and the compile's
/// error), and what must lie beside it (one sequence's cache rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub tokens: u32,
    pub plan: std::result::Result<(u64, u32), &'static str>,
    pub beside: u64,
}

/// Why a rung's plan does not compile, as a refusal names it: [`NO_PLAN`]
/// and the compile's own error. Held for the life of the process (a load's
/// handful of rungs at most).
#[must_use]
pub fn no_plan(why: &str) -> &'static str {
    Box::leak(format!("{NO_PLAN}: {why}").into_boxed_str())
}

/// The rungs `candidates` offer as the ruling weighs them over `live`
/// bytes: a lane a token up to `max_lanes`, a row attending its fire's
/// tokens, a matmul's rows as `computed` states them (the tile the dense
/// product takes at that width: `kernels_cuda::linear::gemm::rows_computed`),
/// infeasible where the plan does not compile or the span leaves no room
/// beside it.
#[must_use]
pub fn rung_facts(
    candidates: &[Candidate],
    max_lanes: u32,
    fire_lanes: u32,
    live: u64,
    computed: impl Fn(u32) -> u32,
) -> Vec<RungFacts> {
    candidates
        .iter()
        .map(|c| {
            let (span, launches) = c.plan.unwrap_or((0, 0));
            RungFacts {
                tokens: c.tokens,
                lanes: c.tokens.min(max_lanes),
                keys: c.tokens,
                launches,
                padded_rows: computed(c.tokens),
                span_bytes: span,
                infeasible: match c.plan {
                    Err(why) => Some(why),
                    Ok(_) if span.saturating_add(c.beside) > live => Some(NO_ROOM),
                    Ok(_) => None,
                },
                fire_lanes,
            }
        })
        .collect()
}

/// The rungs a load weighed its top by, and the estimate it weighed them
/// with, over its live range: what a check of the estimate reads
/// (`engine::load::check`).
#[derive(Debug, Clone)]
pub struct Weighed {
    pub rungs: Vec<RungFacts>,
    pub estimate: engine::load::FireEstimate,
    pub live: u64,
    /// The model's fire top: the widest rung a load that does not keep every
    /// rung offers.
    pub fire_top: u32,
}

/// The widest feasible rung of `rungs`: the top a load that keeps every
/// rung lays (`Knobs::every_rung`), so each can be timed.
#[must_use]
pub fn widest(rungs: &[RungFacts]) -> Option<u32> {
    rungs
        .iter()
        .filter(|r| r.infeasible.is_none())
        .map(|r| r.tokens)
        .max()
}

/// The narrowest feasible rung of `rungs` a decode fire of `lanes` takes (a
/// row a lane), else the widest feasible: the fire top of a load that skips
/// measuring on a card whose profile states no rates to weigh them by.
/// Refused, with the widest rung's reason, where none is feasible.
pub fn narrowest(rungs: &[RungFacts], lanes: u32) -> Result<u32> {
    let feasible = || rungs.iter().filter(|r| r.infeasible.is_none());
    feasible()
        .map(|r| r.tokens)
        .filter(|&tokens| tokens >= lanes)
        .min()
        .or_else(|| widest(rungs))
        .ok_or_else(|| {
            let why = rungs.last().and_then(|r| r.infeasible);
            Fault::Card(format!(
                "no rung of this load is feasible: {}",
                why.unwrap_or("it offers none")
            ))
        })
}

/// The fire top a device's profile states (`engine.fire_rows`, from a
/// measured sweep): the widest feasible rung at or under `stated`, the
/// estimate not weighed. What the room leaves only lowers it, as it lowers
/// any top (`serve::load::room`). Refused, with the widest rung's reason,
/// where none is feasible.
pub fn stated(rungs: &[RungFacts], stated: u32) -> Result<u32> {
    rungs
        .iter()
        .filter(|r| r.infeasible.is_none() && r.tokens <= stated)
        .map(|r| r.tokens)
        .max()
        .ok_or_else(|| {
            let why = rungs.last().and_then(|r| r.infeasible);
            Fault::Card(format!(
                "no rung at or under the stated fire top of {stated} rows \
                 (`engine.fire_rows`) is feasible: {}",
                why.unwrap_or("the load offers none")
            ))
        })
}

/// The rows of the tile each dense bf16 product of `m x n x k` computes
/// under `chip`, as `kernels_cuda::linear::gemm` selects its kernel: few
/// rows a streaming skeleton covers (up to the rows skeleton's 16) stream
/// the weight once a row and pad nothing; past them the tile of cuBLASLt's
/// first heuristic algorithm at the shape (on ROCm, and where cuBLASLt names
/// none, a row's own: nothing padded).
#[must_use]
pub fn row_tile(chip: &profile::Profile) -> engine::roofline::RowTile {
    #[cfg(feature = "cuda")]
    {
        let tuning = kernels_cuda::tuning::Tuning::of(chip);
        std::sync::Arc::new(move |m, n, k| {
            if m <= kernels_cuda::linear::skeleton::ROWS_M {
                1
            } else if kernels_cuda::linear::gemm::skinny_rows(&tuning, m, n, k).is_some() {
                // the skinny skeleton runs it, in mma.sync's 16-row tiles
                16
            } else {
                kernels_cuda::linear::gemm::tile_rows(&tuning, m, n, k).unwrap_or(1)
            }
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = chip;
        std::sync::Arc::new(|_, _, _| 1)
    }
}

/// `chip`'s kernels laid on the estimate's constants: each matmul padded to
/// [`row_tile`] and priced by the waves the vendor library's algorithm
/// fills at its shape (cuBLASLt's / hipBLASLt's `wavesCount`,
/// `engine::roofline::Constants::with_gemm_waves`), and on CUDA the few rows the rows skeleton serves (2 to
/// its 16, `linear::gemm::few_rows_grid`) weighed at `few_row_tflops` under
/// the wave rule over its grid and the device's multiprocessors
/// (`engine::roofline::Constants::with_few_rows`). The key it lacks where
/// the profile states none.
pub fn shape(
    chip: &profile::Profile,
) -> Result<
    impl Fn(
        engine::roofline::Constants,
    ) -> std::result::Result<engine::roofline::Constants, &'static str>
    + use<>,
> {
    let tile = row_tile(chip);
    #[cfg(feature = "cuda")]
    let tuning = kernels_cuda::tuning::Tuning::of(chip);
    #[cfg(feature = "cuda")]
    let cores = kernels_cuda::linear::gemm::multiprocessors().ok_or_else(|| {
        Fault::Card(
            "the device did not answer its multiprocessor count, which the few-rows kernel's \
             wave rule reads"
                .to_owned(),
        )
    })?;
    Ok(move |constants: engine::roofline::Constants| {
        let constants = constants.with_row_tile(tile.clone());
        #[cfg(feature = "cuda")]
        {
            constants
                .with_gemm_waves(std::sync::Arc::new(move |m, n, k| {
                    kernels_cuda::linear::gemm::waves(&tuning, m, n, k)
                }))
                .with_few_rows(
                    std::sync::Arc::new(kernels_cuda::linear::gemm::few_rows_grid),
                    cores,
                )
        }
        #[cfg(feature = "rocm")]
        {
            Ok(constants.with_gemm_waves(std::sync::Arc::new(kernels_cuda::linear::gemm::waves)))
        }
        #[cfg(not(feature = "device"))]
        {
            Ok(constants)
        }
    })
}

/// The fire top of `rungs` (every rung of the lattice a load offers) a
/// prompt of `context` holds, on `device`, whose profile states `rates`,
/// over a live range of `live` bytes, the backend's kernels laid on the
/// estimate by `shape` ([`engine::load::choose_shaped`],
/// [`engine::load::fire_top`]), and the estimate it weighed them by.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    context: u32,
    rates: &profile::rates::Rates,
    shape: impl FnOnce(
        engine::roofline::Constants,
    ) -> std::result::Result<engine::roofline::Constants, &'static str>,
    device: &str,
    counts: &Counts,
) -> std::result::Result<(TopRung, engine::load::FireEstimate), engine::load::Refused> {
    engine::load::choose_shaped(rungs, live, context, rates, device, counts, shape)
}

#[cfg(test)]
mod stated_tests {
    use super::*;

    fn rung(tokens: u32, infeasible: Option<&'static str>) -> RungFacts {
        RungFacts {
            tokens,
            lanes: tokens.min(256),
            keys: tokens,
            launches: 1,
            padded_rows: tokens,
            span_bytes: u64::from(tokens),
            infeasible,
            fire_lanes: 256,
        }
    }

    /// A stated fire top is the widest feasible rung at or under it, never
    /// a wider one, and a rung past the room is passed over for the next.
    #[test]
    fn a_stated_fire_top_is_the_widest_feasible_rung_under_it() {
        let rungs = [
            rung(2048, None),
            rung(4096, None),
            rung(4871, None),
            rung(5312, None),
        ];
        assert_eq!(stated(&rungs, 5120).unwrap(), 4871);
        assert_eq!(stated(&rungs, 4096).unwrap(), 4096);
        let short = [
            rung(2048, None),
            rung(4096, Some("no room")),
            rung(4871, Some("no room")),
        ];
        assert_eq!(stated(&short, 5120).unwrap(), 2048);
        let none = [rung(4096, Some("no room"))];
        let why = stated(&none, 4096).unwrap_err().to_string();
        assert!(why.contains("engine.fire_rows"), "{why}");
    }
}
