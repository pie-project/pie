//! The top rung a Metal load lays its fires up to ([`engine::load::top_rung`]):
//! the rungs the requested fire size offers, each infeasible whose span
//! passes what one buffer binds, weighed by the estimate from the device's
//! measured rates (its profile's `[rates]`) and the model's counts. A
//! device whose profile states no rates has nothing to weigh the rungs by,
//! and the load is refused, naming it.

use engine::load::{FireEstimate, FireTimes, NoRung, RungFacts, TopRung, TopWhy};
use poem_ir::{Def, Trace};

use crate::error::{Fault, Result};

/// Why a rung is no rung of this device: its span lies in the pool's last
/// buffer, and no buffer binds more than one.
pub const PAST_ONE_BUFFER: &str = "past one buffer";

/// What a fire of the model reads and computes, the estimate's counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// The weight bytes every fire reads: the planes the device keeps.
    pub weight_bytes: u64,
    /// The flops a token takes through the weights its matmuls read.
    pub flops_per_token: u64,
    /// The flops a row takes a key it attends, summed over the kv rows.
    pub attn_flops_per_row_key: u64,
}

impl Counts {
    /// The counts of `trace` over `weight_bytes` kept: two flops a weight
    /// element a token for every weight a matmul reads (each once), and four
    /// a query element a key (the scores' product and the values') at each
    /// kv row an attention reads, the widest query a row is read by.
    pub fn of(trace: &Trace, weight_bytes: u64) -> Result<Counts> {
        let mut read = std::collections::BTreeSet::new();
        let mut elements = 0u64;
        let mut query = std::collections::BTreeMap::new();
        for node in &trace.nodes {
            for (_, w, _) in crate::legs::matmuls(&node.op) {
                if !read.insert(w) {
                    continue;
                }
                let param = match trace.values.get(w.0 as usize).map(|decl| &decl.def) {
                    Some(Def::Weight(at)) => trace.params.get(*at as usize),
                    _ => None,
                };
                if let Some(param) = param {
                    elements = elements.saturating_add(param.shape.iter().product::<u64>());
                }
            }
            let Some(reader) = poem_exec::store::kv::reads(&node.op) else {
                continue;
            };
            let Some(row) = poem_exec::store::kv::row_of(trace, reader.cache) else {
                continue;
            };
            let width = poem_exec::store::kv::width_of(trace, reader.q)?;
            let seat = query.entry(row).or_insert(0u64);
            *seat = (*seat).max(width);
        }
        Ok(Counts {
            weight_bytes,
            flops_per_token: elements.saturating_mul(2),
            attn_flops_per_row_key: query.values().sum::<u64>().saturating_mul(4),
        })
    }
}

/// The estimate of a fire's time on a device of `rates`: its weights read
/// once, beside its launch, then its tokens' and row-keys' flops. `None`
/// where the device states no compute rate or no bandwidth.
#[must_use]
pub fn estimate(counts: &Counts, rates: &profile::fire::Rates) -> Option<FireEstimate> {
    let flops = rates.bf16_flops_per_ns?;
    let bytes = rates.bytes_per_ns?;
    let read = (counts.weight_bytes as f64 / bytes).ceil() as u64;
    Some(FireEstimate {
        fixed_ns: read.saturating_add(rates.launch_ns.unwrap_or(0)),
        per_token_ns: counts.flops_per_token as f64 / flops,
        per_row_key_ns: counts.attn_flops_per_row_key as f64 / flops,
    })
}

/// The rungs `(tokens, span chunks)` offers as the ruling weighs them: a
/// lane a token up to `max_lanes`, a row attending its fire's tokens, and
/// infeasible where the span passes the `per_buffer` chunks one buffer
/// binds.
#[must_use]
pub fn rung_facts(
    act: &[(u32, u32)],
    chunk_bytes: u64,
    max_lanes: u32,
    per_buffer: u32,
) -> Vec<RungFacts> {
    act.iter()
        .map(|&(tokens, chunks)| RungFacts {
            tokens,
            lanes: tokens.min(max_lanes),
            keys: tokens,
            span_bytes: u64::from(chunks) * chunk_bytes,
            infeasible: (chunks > per_buffer).then_some(PAST_ONE_BUFFER),
        })
        .collect()
}

/// The top rung of `rungs` on `device`, whose profile states `rates`, over
/// a live range of `live` bytes, by the estimate from the rates alone.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    rates: &profile::fire::Rates,
    device: &str,
    counts: &Counts,
) -> Result<TopRung> {
    let Some(estimate) = estimate(counts, rates) else {
        return Err(Fault::Residency(format!(
            "device {device} has no rates in its profile (`[rates] bf16_flops_per_ns` and \
             `bytes_per_ns`), so nothing weighs its top rung; measure them with the \
             `the_device_rates` test and state them in its device file"
        )));
    };
    let times = FireTimes {
        measured: &[],
        estimate,
    };
    engine::load::top_rung(rungs, live, &times).map_err(|no: NoRung| Fault::Residency(no.to_string()))
}

/// The load's one line on its top rung: which, why, and the span it
/// reserves of the live range.
#[must_use]
pub fn line(rung: &TopRung, asked: u32, live: u64, device: &str) -> String {
    let by = match rung.why {
        TopWhy::Argmax { eff, live_share } => format!(
            "by the estimate from {device}'s rates: eff {eff:.3} at a live share of {live_share:.3}"
        ),
        TopWhy::OnlyFeasible => "as the one feasible rung".to_string(),
    };
    let gib = live as f64 / f64::from(1u32 << 30);
    format!(
        "top rung {} of the {asked} asked {by}; span {} MiB of {gib:.1} GiB live",
        rung.tokens,
        rung.span_bytes >> 20
    )
}

/// `budget` with fires of at most `top` tokens: its buckets below `top`,
/// then `top`, and as many lanes at most.
#[must_use]
pub fn bounded(budget: &poem_compiler::Budget, top: u32) -> poem_compiler::Budget {
    let mut buckets: Vec<u32> = budget.buckets.iter().copied().filter(|&b| b < top).collect();
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
            flops_per_token: 1 << 30,
            attn_flops_per_row_key: 1 << 10,
        }
    }

    #[test]
    fn a_span_past_one_buffer_is_no_rung() {
        let rungs = rung_facts(&[(1024, 100), (2048, 300)], MIB, 256, 200);
        assert_eq!(rungs[0].infeasible, None);
        assert_eq!((rungs[0].lanes, rungs[0].span_bytes), (256, 100 * MIB));
        assert_eq!(rungs[1].infeasible, Some(PAST_ONE_BUFFER));
    }

    #[test]
    fn without_rates_the_load_is_refused_by_name() {
        let rungs = rung_facts(&[(1024, 100), (2048, 150)], MIB, 256, 200);
        let err = choose(&rungs, 1 << 33, &profile::fire::Rates::default(), "Apple X", &counts())
            .unwrap_err();
        assert!(
            err.to_string().contains("device Apple X has no rates in its profile"),
            "{err}"
        );
        let rated = profile::fire::Rates {
            bf16_flops_per_ns: Some(5000.0),
            bytes_per_ns: Some(400.0),
            launch_ns: Some(5000),
        };
        let top = choose(&rungs, 1 << 33, &rated, "Apple X", &counts()).unwrap();
        assert!(matches!(top.tokens, 1024 | 2048));
        let none = rung_facts(&[(1024, 300)], MIB, 256, 200);
        let err = choose(&none, 1 << 33, &rated, "Apple X", &counts()).unwrap_err();
        assert!(
            err.to_string().contains("1024 tokens (span 300 MiB): past one buffer"),
            "{err}"
        );
    }

    #[test]
    fn the_bounded_budget_ends_at_the_top() {
        let budget = poem_compiler::Budget {
            max_lanes: 512,
            max_tokens: 4096,
            buckets: vec![64, 512, 2048, 4096],
            ..poem_compiler::Budget::default()
        };
        let cut = bounded(&budget, 1024);
        assert_eq!((cut.max_tokens, cut.max_lanes), (1024, 512));
        assert_eq!(cut.buckets, vec![64, 512, 1024]);
    }
}
