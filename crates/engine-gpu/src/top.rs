//! The top rung a load lays its fires up to ([`engine::load::top_rung`]):
//! the rungs the requested fire size offers, each infeasible whose span
//! passes one binding, weighed by the fire times measured of this model on
//! this device ([`profile::fire::fire_ns`]), else by the estimate from the
//! device's rates ([`profile::fire::rates`]) and the model's counts.

use engine::load::{FireEstimate, FireTimes, NoRung, RungFacts, TimesUsed, TopRung, TopWhy};
use poem_ir::Trace;

use crate::error::{Fault, Result};

/// Why a rung is no rung of this device: its span lies in the pool's last
/// segment, and no segment is past a binding.
pub const PAST_ONE_BINDING: &str = "past one binding";

/// What a fire of the model reads and computes, the estimate's counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// The weight bytes every fire reads: the planes the device keeps.
    pub weight_bytes: u64,
    /// The flops a token takes through the device's weights.
    pub flops_per_token: u64,
    /// The flops a row takes a key it attends, summed over the layers.
    pub attn_flops_per_row_key: u64,
}

impl Counts {
    /// The counts of `trace`: its weights less the token tables and the
    /// expert banks that stream through the pool, two flops a weight
    /// element a token, and four a query element a key (the scores' product
    /// and the values') at each kv row an attention reads.
    pub fn of(trace: &Trace) -> Result<Counts> {
        let planes = crate::weights::plane_bytes(trace)?;
        let mut streamed = crate::store::rows::streamed(trace);
        streamed.extend(crate::store::experts::streamed(trace));
        let mut weight_bytes = 0u64;
        let mut elements = 0u64;
        for (at, param) in trace.params.iter().enumerate() {
            if streamed.contains(&at) {
                continue;
            }
            weight_bytes = weight_bytes.saturating_add(planes[at]);
            if param.shape.len() >= 2 {
                let (rows, width) = crate::weights::rectangle(&param.shape);
                elements = elements.saturating_add(rows.saturating_mul(width));
            }
        }
        // One query width a kv row: a row read by a prefill and a decode
        // attention alike is counted once.
        let mut query = std::collections::BTreeMap::new();
        for node in &trace.nodes {
            let Some(read) = crate::store::kv::reads(&node.op) else {
                continue;
            };
            let Some(row) = crate::store::kv::row_of(trace, read.cache) else {
                continue;
            };
            let width = crate::store::kv::width_of(trace, read.q)?;
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
/// once, beside its launches, then its tokens' and row-keys' flops. `None`
/// where the device states no compute rate or no bandwidth.
#[must_use]
pub fn estimate(counts: &Counts, rates: &profile::fire::DeviceRates) -> Option<FireEstimate> {
    let flops = rates.flops_per_ns.filter(|&r| r > 0.0)?;
    let bytes = rates.bytes_per_ns.filter(|&r| r > 0.0)?;
    let read = (counts.weight_bytes as f64 / bytes).ceil() as u64;
    Some(FireEstimate {
        fixed_ns: read.saturating_add(rates.launch_ns.unwrap_or(0)),
        per_token_ns: counts.flops_per_token as f64 / flops,
        per_row_key_ns: counts.attn_flops_per_row_key as f64 / flops,
    })
}

/// The rungs `(tokens, span bytes)` offers as the ruling weighs them: a
/// lane a token up to `max_lanes`, a row attending its fire's tokens, and
/// infeasible where the span passes `binding`.
#[must_use]
pub fn rung_facts(spans: &[(u32, u64)], max_lanes: u32, binding: u64) -> Vec<RungFacts> {
    spans
        .iter()
        .map(|&(tokens, span_bytes)| RungFacts {
            tokens,
            lanes: tokens.min(max_lanes),
            keys: tokens,
            span_bytes,
            infeasible: (span_bytes > binding).then_some(PAST_ONE_BINDING),
        })
        .collect()
}

/// Where a load's top rung came from.
#[derive(Debug, Clone, PartialEq)]
pub enum Chosen {
    /// The ruling's pick.
    Ruled(TopRung),
    /// Nothing to weigh the rungs by (no measured entry, and no rates to
    /// estimate from): the widest rung one binding holds, of those the
    /// requested fire size offers.
    Requested { tokens: u32 },
}

impl Chosen {
    #[must_use]
    pub fn tokens(&self) -> u32 {
        match self {
            Chosen::Ruled(top) => top.tokens,
            Chosen::Requested { tokens } => *tokens,
        }
    }
}

/// The top rung of `rungs` for the model filed as `model` on the device
/// filed as `device`, over a live range of `live` bytes.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    device: &str,
    model: &str,
    counts: &Counts,
    binding: u64,
) -> Result<Chosen> {
    let measured = profile::fire::fire_ns(device, model);
    let estimate = profile::fire::rates(device).and_then(|rates| estimate(counts, &rates));
    let refuse = |no: NoRung| {
        Fault::Residency(format!(
            "{no}; this device binds no more than {} MiB",
            binding >> 20
        ))
    };
    let covered = !measured.is_empty()
        && rungs
            .iter()
            .filter(|r| r.infeasible.is_none())
            .all(|r| measured.iter().any(|&(tokens, _)| tokens == r.tokens));
    match estimate {
        Some(estimate) => engine::load::top_rung(rungs, live, &FireTimes { measured, estimate })
            .map(Chosen::Ruled)
            .map_err(refuse),
        // A measured table covering every feasible rung needs no estimate.
        None if covered => {
            let unused = FireEstimate {
                fixed_ns: 0,
                per_token_ns: 0.0,
                per_row_key_ns: 0.0,
            };
            engine::load::top_rung(
                rungs,
                live,
                &FireTimes {
                    measured,
                    estimate: unused,
                },
            )
            .map(Chosen::Ruled)
            .map_err(refuse)
        }
        None => rungs
            .iter()
            .filter(|r| r.infeasible.is_none())
            .map(|r| r.tokens)
            .max()
            .map(|tokens| Chosen::Requested { tokens })
            .ok_or_else(|| {
                refuse(NoRung {
                    refused: rungs
                        .iter()
                        .map(|r| {
                            (
                                r.tokens,
                                r.span_bytes,
                                r.infeasible.unwrap_or("not feasible"),
                            )
                        })
                        .collect(),
                })
            }),
    }
}

/// The load's one line on its top rung: which, by what, and the span it
/// reserves of the live range.
#[must_use]
pub fn line(
    chosen: &Chosen,
    top: u32,
    asked: u32,
    span: u64,
    live: u64,
    device: &str,
    model: &str,
) -> String {
    let key = format!("{device} / {model}");
    let by = match chosen {
        Chosen::Ruled(rung) => {
            let times = match rung.times {
                TimesUsed::Measured => format!("the measured fire times of {key}"),
                TimesUsed::Estimate => format!("the estimate (no measured entry for {key})"),
                TimesUsed::PartialMeasuredIgnored => format!(
                    "the estimate (the measured entry for {key} misses a feasible rung, so it \
                     is set aside)"
                ),
            };
            match rung.why {
                TopWhy::Argmax { eff, live_share } => {
                    format!("by {times}: eff {eff:.3} at a live share of {live_share:.3}")
                }
                TopWhy::OnlyFeasible => format!("as the one feasible rung ({times})"),
            }
        }
        Chosen::Requested { tokens } if *tokens < asked => format!(
            "by the requested fire size {asked}, cut to the widest rung one binding holds (no \
             measured entry for {key}, and no compute rate or bandwidth for it to estimate by)"
        ),
        Chosen::Requested { .. } => format!(
            "by the requested fire size (no measured entry for {key}, and no compute rate or \
             bandwidth for it to estimate by)"
        ),
    };
    let gib = live as f64 / f64::from(1u32 << 30);
    format!(
        "top rung {top} {by}; span {} MiB of {gib:.1} GiB live",
        span >> 20
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn a_span_past_one_binding_is_no_rung() {
        let rungs = rung_facts(&[(1024, 100 * MIB), (2048, 300 * MIB)], 256, 200 * MIB);
        assert_eq!(rungs[0].infeasible, None);
        assert_eq!(rungs[0].lanes, 256);
        assert_eq!(rungs[1].infeasible, Some(PAST_ONE_BINDING));
    }

    #[test]
    fn without_times_the_top_is_the_widest_rung_one_binding_holds() {
        let rungs = rung_facts(&[(1024, 100 * MIB), (2048, 300 * MIB)], 256, 200 * MIB);
        let counts = Counts {
            weight_bytes: 1 << 30,
            flops_per_token: 1 << 30,
            attn_flops_per_row_key: 1 << 10,
        };
        let chosen = choose(&rungs, 1 << 33, "no-such-device", "m", &counts, 200 * MIB).unwrap();
        assert_eq!(chosen, Chosen::Requested { tokens: 1024 });
        let none = rung_facts(&[(1024, 300 * MIB)], 256, 200 * MIB);
        let err = choose(&none, 1 << 33, "no-such-device", "m", &counts, 200 * MIB).unwrap_err();
        assert!(
            err.to_string()
                .contains("1024 tokens (span 300 MiB): past one binding")
        );
    }

    #[test]
    fn the_estimate_reads_the_rates() {
        let counts = Counts {
            weight_bytes: 1_000_000,
            flops_per_token: 2_000,
            attn_flops_per_row_key: 40,
        };
        let rates = profile::fire::DeviceRates {
            device: "d",
            flops_per_ns: Some(10.0),
            bytes_per_ns: Some(100.0),
            launch_ns: Some(5),
        };
        assert_eq!(
            estimate(&counts, &rates),
            Some(FireEstimate {
                fixed_ns: 10_005,
                per_token_ns: 200.0,
                per_row_key_ns: 4.0,
            })
        );
        let unstated = profile::fire::DeviceRates {
            bytes_per_ns: None,
            ..rates
        };
        assert_eq!(estimate(&counts, &unstated), None);
    }
}
