//! The top rung a Metal load lays its fires up to ([`engine::load::top_rung`]):
//! the rungs the requested fire size offers, each infeasible whose span
//! passes what one buffer binds, every fire timed by the estimate from the
//! device's measured rates (its profile's `[rates]`) and the model's counts
//! ([`engine::load::Counts`]). A device whose profile states no rates has
//! nothing to weigh the rungs by, and the load is refused, naming it.

use engine::load::{Counts, Kept, RungFacts, TopRung};
use poem_ir::Trace;

use crate::error::{Fault, Result};

/// Why a rung is no rung of this device: its span lies in the pool's last
/// buffer, and no buffer binds more than one.
pub const PAST_ONE_BUFFER: &str = "past one buffer";

/// The counts of `trace` over the planes this engine keeps: its row-streamed
/// token tables and its streamed expert banks apart from the dense weights
/// (`store::streamed`), each plane at the bytes this engine stores it in.
pub fn counts(trace: &Trace) -> Result<Counts> {
    let planes = crate::weights::plane_bytes(trace)?;
    let tables = crate::store::streamed::streamed_rows(trace);
    let banks = crate::store::streamed::streamed_experts(trace);
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

/// What a load weighed its rungs by: the rungs it offered, the estimate
/// from its profile's rates and its model's counts, and the live range.
#[derive(Debug, Clone)]
pub struct Weighed {
    pub rungs: Vec<RungFacts>,
    pub estimate: engine::load::FireEstimate,
    pub live: u64,
    pub counts: Counts,
    /// The model's fire top on this device (`engine::roofline::fire_top`
    /// over its rates): the widest rung a load that is no check offers.
    pub fire_top: u32,
}

/// One rung offered: a fire of `tokens`, the chunks its span takes, the
/// dispatches a recording of it issues, and the rows its matmuls compute
/// (`tokens` padded to the tile Metal's dense kernel takes at that width,
/// [`kernels_metal::linear::gemm::rows_computed`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub tokens: u32,
    pub chunks: u32,
    pub launches: u32,
    pub padded_rows: u32,
}

/// The rungs `candidates` offer as the ruling weighs them: a lane a token
/// up to `max_lanes`, a row attending its fire's tokens, and infeasible
/// where the span passes the `per_buffer` chunks one buffer binds.
#[must_use]
pub fn rung_facts(
    candidates: &[Candidate],
    chunk_bytes: u64,
    max_lanes: u32,
    per_buffer: u32,
) -> Vec<RungFacts> {
    candidates
        .iter()
        .map(|c| RungFacts {
            tokens: c.tokens,
            lanes: c.tokens.min(max_lanes),
            keys: c.tokens,
            launches: c.launches,
            padded_rows: c.padded_rows,
            span_bytes: u64::from(c.chunks) * chunk_bytes,
            infeasible: (c.chunks > per_buffer).then_some(PAST_ONE_BUFFER),
            // One fire carries every lane the budget seats.
            fire_lanes: max_lanes,
        })
        .collect()
}

/// The top rung of `rungs` no wider than `context` on `device`, whose
/// profile states `rates`, over a live range of `live` bytes
/// ([`engine::load::choose`]).
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    context: u32,
    rates: &profile::rates::Rates,
    device: &str,
    counts: &Counts,
) -> Result<TopRung> {
    engine::load::choose(rungs, live, context, rates, device, counts)
        .map(|(top, _)| top)
        .map_err(|refused| Fault::Residency(refused.to_string()))
}

/// `budget` with fires of at most `top` tokens: its buckets below `top`,
/// then `top`, and as many lanes at most.
#[must_use]
pub fn bounded(budget: &poem_compiler::Budget, top: u32) -> poem_compiler::Budget {
    let mut buckets: Vec<u32> = budget
        .buckets
        .iter()
        .copied()
        .filter(|&b| b < top)
        .collect();
    if !budget.buckets.is_empty() {
        buckets.push(top);
    }
    poem_compiler::Budget {
        max_lanes: budget.max_lanes.min(top),
        max_tokens: top,
        buckets,
        ..budget.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn counts() -> Counts {
        Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![engine::load::GemmShape::dense(4096, 4096, 1 << 30, 1 << 30)],
            head_bytes: 0,
            attn_flops_per_row_key: 1 << 10,
            kv_bytes_per_key: 1 << 14,
            attn_head_dim: 128,
            readout_flops: 0,
            readout_depth: 4096,
            state_elements_per_row: 0,
            elementwise_ops: vec![0],
            attn_dispatches: 28,
            scan_dispatches: 0,
            width: 1024,
        }
    }

    fn candidate(tokens: u32, chunks: u32) -> Candidate {
        Candidate {
            tokens,
            chunks,
            launches: 300,
            padded_rows: tokens.next_multiple_of(32),
        }
    }

    #[test]
    fn without_rates_the_load_is_refused_by_name() {
        let rungs = rung_facts(&[candidate(1024, 100), candidate(2048, 150)], MIB, 256, 200);
        let err = choose(
            &rungs,
            1 << 33,
            u32::MAX,
            &profile::rates::Rates::default(),
            "Apple X",
            &counts(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("device Apple X has no rates in its profile"),
            "{err}"
        );
        let rated = profile::rates::Rates {
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 2.0), (4096, 5.0)])]),
            attn_tflops_by_keys: Some(vec![(2048, 2.0)]),
            hbm_gbps: Some(400.0),
            elementwise_gbps: Some(200.0),
            launch_ns: Some(5000),
            ..profile::rates::Rates::default()
        };
        let top = choose(&rungs, 1 << 33, u32::MAX, &rated, "Apple X", &counts()).unwrap();
        assert!(matches!(top.tokens, 1024 | 2048));
        let none = rung_facts(&[candidate(1024, 300)], MIB, 256, 200);
        let err = choose(&none, 1 << 33, u32::MAX, &rated, "Apple X", &counts()).unwrap_err();
        assert!(
            err.to_string()
                .contains("1024 tokens (span 300 MiB): past one buffer"),
            "{err}"
        );
    }

}
