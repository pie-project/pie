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
pub const NO_PLAN: &str = "its plan does not compile";

/// Why a rung is no rung of this load: its span leaves no room for one
/// sequence and the bodies' floor beside it.
pub const NO_ROOM: &str = "no room is left beside it for one sequence and the bodies' floor";

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

/// One rung offered: a fire of `tokens`, its span (`u64::MAX` where its
/// plan does not compile), the launches its compiled plan issues, and what
/// must lie beside it (one sequence's cache rows and the bodies' floor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub tokens: u32,
    pub span: u64,
    pub launches: u32,
    pub beside: u64,
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
    live: u64,
    computed: impl Fn(u32) -> u32,
) -> Vec<RungFacts> {
    candidates
        .iter()
        .map(|c| RungFacts {
            tokens: c.tokens,
            lanes: c.tokens.min(max_lanes),
            keys: c.tokens,
            launches: c.launches,
            padded_rows: computed(c.tokens),
            span_bytes: c.span,
            infeasible: if c.span == u64::MAX {
                Some(NO_PLAN)
            } else if c.span.saturating_add(c.beside) > live {
                Some(NO_ROOM)
            } else {
                None
            },
        })
        .collect()
}

/// The rungs a load weighed its top by, and the estimate it weighed them
/// with, over its live range: what a check of the estimate reads
/// (`pie tune --check-estimate`).
#[derive(Debug, Clone)]
pub struct Weighed {
    pub rungs: Vec<RungFacts>,
    pub estimate: engine::load::FireEstimate,
    pub live: u64,
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

/// The top rung of `rungs` on `device`, whose profile states `rates`, over
/// a live range of `live` bytes ([`engine::load::choose`]).
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    rates: &profile::fire::Rates,
    device: &str,
    counts: &Counts,
) -> Result<TopRung> {
    engine::load::choose(rungs, live, rates, device, counts)
        .map(|(top, _)| top)
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
            flops_per_row: 16 << 30,
            attn_flops_per_row_key: 1 << 12,
            kv_bytes_per_key: 1 << 17,
            readout_flops: 0,
            state_elements_per_row: 0,
            elementwise_bytes_per_row: 0,
            width: 4096,
        }
    }

    /// The L40S device file's rates.
    fn l40s() -> profile::fire::Rates {
        profile::fire::Rates {
            bf16_flops_per_ns_by_rows: Some(vec![(4096, 249_000.0)]),
            attn_flops_per_ns: Some(100_000.0),
            bytes_per_ns: Some(654.0),
            launch_ns: Some(1620),
            ..profile::fire::Rates::default()
        }
    }

    fn candidate(tokens: u32, span_mib: u64) -> Candidate {
        Candidate {
            tokens,
            span: span_mib * MIB,
            launches: 400,
            beside: 800 * MIB,
        }
    }

    #[test]
    fn a_load_without_rates_is_refused_by_name() {
        let rungs = rung_facts(
            &[candidate(1024, 100), candidate(2048, 200)],
            256,
            8 << 30,
            |rows| rows,
        );
        let err = choose(
            &rungs,
            8 << 30,
            &profile::fire::Rates::default(),
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
                span: u64::MAX,
                ..candidate(16384, 0)
            },
        ];
        let live = 3 << 30;
        let rungs = rung_facts(&candidates, 256, live, |rows| rows);
        assert_eq!(rungs[0].lanes, 256);
        assert_eq!(rungs[3].infeasible, Some(NO_ROOM));
        assert_eq!(rungs[4].infeasible, Some(NO_PLAN));
        let top = choose(&rungs, live, &l40s(), "NVIDIA L40S", &counts()).unwrap();
        let estimate = engine::load::estimate(&counts(), &l40s()).unwrap();
        assert_eq!(
            top,
            engine::load::top_rung(&rungs, live, &estimate).unwrap()
        );
        assert_eq!(
            top.span_bytes,
            candidates[top.tokens.trailing_zeros() as usize - 10].span
        );
        // Every rung refused: the ruling's refusal, each rung named.
        let err = choose(&rungs[3..], live, &l40s(), "NVIDIA L40S", &counts()).unwrap_err();
        assert!(
            err.to_string()
                .contains(&format!("8192 tokens (span 2400 MiB): {NO_ROOM}")),
            "{err}"
        );
    }
}
