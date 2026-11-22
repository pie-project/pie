//! The top rung a CUDA load lays its fires up to
//! ([`engine::load::top_rung`]): of the buckets the asked fire size's
//! lattice offers, the one the ruling picks over the room the card has
//! after the weight tier, every fire timed by the estimate from the
//! device's rates (its profile's `[rates]`) and the model's counts
//! ([`engine::load::Counts`]). A device whose profile states no rates has
//! nothing to weigh the rungs by, and the load is refused, naming it.

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
    impl Fn(engine::roofline::Constants) -> std::result::Result<engine::roofline::Constants, &'static str>
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
            Ok(constants
                .with_gemm_waves(std::sync::Arc::new(kernels_cuda::linear::gemm::waves)))
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
    rates: &profile::fire::Rates,
    shape: impl FnOnce(
        engine::roofline::Constants,
    ) -> std::result::Result<engine::roofline::Constants, &'static str>,
    device: &str,
    counts: &Counts,
) -> Result<(TopRung, engine::load::FireEstimate)> {
    engine::load::choose_shaped(rungs, live, context, rates, device, counts, shape)
        .map_err(|refused| Fault::Card(refused.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn counts() -> Counts {
        Counts {
            weight_bytes: 16 << 30,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![engine::load::GemmShape::dense(
                4096,
                4096,
                16 << 30,
                1 << 30,
            )],
            head_bytes: 0,
            attn_flops_per_row_key: 1 << 12,
            kv_bytes_per_key: 1 << 17,
            attn_head_dim: 128,
            readout_flops: 0,
            readout_depth: 4096,
            state_elements_per_row: 0,
            elementwise_ops: vec![0],
            attn_dispatches: 28,
            scan_dispatches: 0,
            width: 4096,
        }
    }

    /// The L40S device file's rates.
    fn l40s() -> profile::fire::Rates {
        profile::fire::Rates {
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0), (8192, 249.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 50.0), (8192, 100.0)]),
            hbm_gbps: Some(654.0),
            elementwise_gbps: Some(400.0),
            launch_ns: Some(1620),
            ..profile::fire::Rates::default()
        }
    }

    fn candidate(tokens: u32, span_mib: u64) -> Candidate {
        Candidate {
            tokens,
            plan: Ok((span_mib * MIB, 400)),
            beside: 800 * MIB,
        }
    }

    #[test]
    fn a_load_without_rates_is_refused_by_name() {
        let rungs = rung_facts(
            &[candidate(1024, 100), candidate(2048, 200)],
            256,
            256,
            8 << 30,
            |rows| rows,
        );
        let err = choose(
            &rungs,
            8 << 30,
            u32::MAX,
            &profile::fire::Rates::default(),
            Ok,
            "NVIDIA L40S",
            &counts(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("device NVIDIA L40S has no rates in its profile"),
            "{err}"
        );
        assert!(
            err.to_string()
                .contains("`pie tune <backend> --rates` measures it"),
            "{err}"
        );
    }

    #[test]
    fn the_candidate_spans_feed_the_ruling() {
        let candidates = [
            candidate(1024, 300),
            candidate(2048, 600),
            candidate(4096, 1200),
            candidate(8192, 2400),
            Candidate {
                plan: Err(no_plan("the arena does not fit")),
                ..candidate(16384, 0)
            },
        ];
        let live = 3 << 30;
        let rungs = rung_facts(&candidates, 256, 256, live, |rows| rows);
        assert_eq!(rungs[0].lanes, 256);
        assert_eq!(rungs[3].infeasible, Some(NO_ROOM));
        assert_eq!(
            rungs[4].infeasible,
            Some("its plan does not compile at this budget: the arena does not fit")
        );
        assert_eq!(rungs[4].span_bytes, 0);
        let (top, _) = choose(
            &rungs,
            live,
            u32::MAX,
            &l40s(),
            Ok,
            "NVIDIA L40S",
            &counts(),
        )
        .unwrap();
        let estimate = engine::load::estimate(&counts(), &l40s()).unwrap();
        assert_eq!(
            top,
            engine::load::top_rung(&rungs, live, &estimate).unwrap()
        );
        assert_eq!(
            top.span_bytes,
            candidates[top.tokens.trailing_zeros() as usize - 10]
                .plan
                .unwrap()
                .0
        );
        // Every rung refused: the ruling's refusal, each rung named.
        let err = choose(
            &rungs[3..],
            live,
            u32::MAX,
            &l40s(),
            Ok,
            "NVIDIA L40S",
            &counts(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains(&format!("8192 tokens (span 2400 MiB): {NO_ROOM}")),
            "{err}"
        );
    }
}
