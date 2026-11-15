//! The top rung a load lays its fires up to ([`engine::load::top_rung`]):
//! the rungs the requested fire size offers, each infeasible whose span
//! passes one binding, every fire timed by the estimate
//! ([`engine::load::FireEstimate`]) over the device's rates (the profile's
//! `[rates]`) and the model's counts ([`Counts`]).

use engine::load::{FireEstimate, NoRung, RungFacts, TopRung, TopWhy};
use poem_ir::{Def, Dim, Operation, Trace, Ty};

use crate::error::{Fault, Result};

/// Why a rung is no rung of this device: its span lies in the pool's last
/// segment, and no segment is past a binding.
pub const PAST_ONE_BINDING: &str = "past one binding";

/// What a fire of the model reads and computes: the estimate's counts, all
/// from the trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// The weight bytes every fire reads: the planes the device keeps, less
    /// the token tables (a gather reads a row of them a token) and the
    /// routed banks.
    pub weight_bytes: u64,
    /// The bytes of every routed bank.
    pub expert_bytes: u64,
    /// `(k, E)` of the routed banks: the experts a row picks, of how many.
    pub experts: Option<(u32, u32)>,
    /// The matmul flops a row takes: two a dense weight element, and two a
    /// routed element of the `k` of `E` experts a row picks.
    pub flops_per_row: u64,
    /// The flops a row takes a key it attends, summed over the layers: four
    /// a query element (the scores' product and the values').
    pub attn_flops_per_row_key: u64,
    /// The kv bytes a key read takes, summed over the kv rows attentions
    /// read: every plane of the row at its element's size.
    pub kv_bytes_per_key: u64,
}

impl Counts {
    /// The counts of `trace`.
    pub fn of(trace: &Trace) -> Result<Counts> {
        let planes = crate::weights::plane_bytes(trace)?;
        let tables = crate::store::rows::streamed(trace);
        let banks = crate::store::experts::streamed(trace);
        let (mut weight_bytes, mut expert_bytes) = (0u64, 0u64);
        let (mut dense, mut routed) = (0u64, 0u64);
        for (at, param) in trace.params.iter().enumerate() {
            if tables.contains(&at) {
                continue;
            }
            let elements = if param.shape.len() >= 2 {
                let (rows, width) = crate::weights::rectangle(&param.shape);
                rows.saturating_mul(width)
            } else {
                0
            };
            if banks.contains(&at) {
                expert_bytes = expert_bytes.saturating_add(planes[at]);
                routed = routed.saturating_add(elements);
            } else {
                weight_bytes = weight_bytes.saturating_add(planes[at]);
                dense = dense.saturating_add(elements);
            }
        }
        let experts = routing(trace);
        let routed_per_row = match experts {
            Some((k, e)) if e > 0 => routed.saturating_mul(u64::from(k)) / u64::from(e),
            // A bank whose routing does not read: every expert, every row.
            _ => routed,
        };
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
        let kv_bytes_per_key = query
            .keys()
            .filter_map(|&row| match trace.caches.get(row) {
                Some(poem_ir::CacheRow::Kv { planes, dtype, .. }) => Some(
                    planes.iter().sum::<u64>()
                        * poem_compiler::arena::elem_bytes(*dtype).unwrap_or(2),
                ),
                _ => None,
            })
            .sum();
        Ok(Counts {
            weight_bytes,
            expert_bytes,
            experts,
            flops_per_row: dense.saturating_add(routed_per_row).saturating_mul(2),
            attn_flops_per_row_key: query.values().sum::<u64>().saturating_mul(4),
            kv_bytes_per_key,
        })
    }
}

/// `(k, E)` of the trace's routed selects: the routes a row carries (the
/// routes' width) and the experts its bank holds (the bank's first axis).
fn routing(trace: &Trace) -> Option<(u32, u32)> {
    let shape = |value: poem_ir::ValueId| match trace.values.get(value.0 as usize) {
        Some(decl) => match (&decl.def, &decl.ty) {
            (Def::Weight(w), _) => trace.params.get(*w as usize).map(|p| p.shape.clone()),
            (_, Ty::Tensor { shape, .. }) => Some(
                shape
                    .iter()
                    .map(|d| match d {
                        Dim::Const(n) => *n,
                        _ => 0,
                    })
                    .collect(),
            ),
            _ => None,
        },
        None => None,
    };
    trace.nodes.iter().find_map(|node| match &node.op {
        Operation::Linear(poem_ir::Linear::MoeSelect { bank, routes, .. }) => {
            let e = *shape(*bank)?.first()?;
            let k = *shape(*routes)?.get(1)?;
            Some((u32::try_from(k).ok()?, u32::try_from(e).ok()?))
        }
        _ => None,
    })
}

/// The estimate of a fire's time on a device of `rates`. `None` where the
/// device states no compute rate or no bandwidth.
#[must_use]
pub fn estimate(counts: &Counts, rates: &profile::fire::Rates) -> Option<FireEstimate> {
    Some(FireEstimate {
        flops_per_ns: rates.bf16_flops_per_ns?,
        bytes_per_ns: rates.bytes_per_ns?,
        launch_ns: rates.launch_ns.unwrap_or(0) as f64,
        dense_bytes: counts.weight_bytes,
        expert_bytes: counts.expert_bytes,
        experts: counts.experts,
        flops_per_row: counts.flops_per_row as f64,
        attn_flops_per_row_key: counts.attn_flops_per_row_key as f64,
        kv_bytes_per_key: counts.kv_bytes_per_key as f64,
    })
}

/// The rungs `(tokens, span bytes)` offers as the ruling weighs them: a
/// lane a token up to `max_lanes`, a row attending its fire's tokens,
/// `launches` kernels a fire, a matmul's rows padded by `padded`, and
/// infeasible where the span passes `binding`.
#[must_use]
pub fn rung_facts(
    spans: &[(u32, u64)],
    max_lanes: u32,
    binding: u64,
    launches: u32,
    padded: &dyn Fn(u32) -> u32,
) -> Vec<RungFacts> {
    spans
        .iter()
        .map(|&(tokens, span_bytes)| RungFacts {
            tokens,
            lanes: tokens.min(max_lanes),
            keys: tokens,
            launches,
            padded_rows: padded(tokens),
            span_bytes,
            infeasible: (span_bytes > binding).then_some(PAST_ONE_BINDING),
        })
        .collect()
}

/// The top rung of `rungs` for `device`, whose profile is `profile`, over a
/// live range of `live` bytes, by the estimate from the profile's rates. A
/// device whose profile states no rates has nothing to weigh the rungs by,
/// and the load is refused, naming it.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    profile: &profile::Profile,
    device: &str,
    counts: &Counts,
    binding: u64,
) -> Result<(TopRung, FireEstimate)> {
    let Some(estimate) = estimate(counts, &profile.rates()) else {
        return Err(Fault::Residency(format!(
            "device {device} has no rates in its profile (`[rates] bf16_flops_per_ns` and \
             `bytes_per_ns`), so nothing weighs its top rung; `pie tune --rates` measures them"
        )));
    };
    let top = engine::load::top_rung(rungs, live, &estimate).map_err(|no: NoRung| {
        Fault::Residency(format!(
            "{no}; this device binds no more than {} MiB",
            binding >> 20
        ))
    })?;
    Ok((top, estimate))
}

/// The load's one line on its top rung: which, by what, and the span it
/// reserves of the live range; the rates where this load measured them.
#[must_use]
pub fn line_of(
    rung: &TopRung,
    top: u32,
    span: u64,
    live: u64,
    tuned: &profile::fire::Tuned,
) -> String {
    let r = tuned.rates;
    let rates = format!(
        "{:.1} TFLOP/s bf16, {:.0} GB/s, launch {} us",
        r.bf16_flops_per_ns.unwrap_or(0.0) / 1e3,
        r.bytes_per_ns.unwrap_or(0.0),
        r.launch_ns.unwrap_or(0) / 1000
    );
    let by = match rung.why {
        TopWhy::Argmax { eff, live_share } => {
            format!("by the estimate: eff {eff:.3} at a live share of {live_share:.3}")
        }
        TopWhy::OnlyFeasible => "as the one feasible rung".to_string(),
    };
    let gib = live as f64 / f64::from(1u32 << 30);
    let from = if tuned.measured_this_load {
        "rates measured this load"
    } else {
        "the profile's rates"
    };
    format!(
        "top rung {top} {by}; span {} MiB of {gib:.1} GiB live; {from}: {rates}",
        span >> 20
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn facts(spans: &[(u32, u64)], binding: u64) -> Vec<RungFacts> {
        rung_facts(spans, 256, binding, 100, &|r| r)
    }

    #[test]
    fn a_span_past_one_binding_is_no_rung() {
        let rungs = facts(&[(1024, 100 * MIB), (2048, 300 * MIB)], 200 * MIB);
        assert_eq!(rungs[0].infeasible, None);
        assert_eq!((rungs[0].lanes, rungs[0].launches), (256, 100));
        assert_eq!(rungs[1].infeasible, Some(PAST_ONE_BINDING));
    }

    fn counts() -> Counts {
        Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            flops_per_row: 1 << 30,
            attn_flops_per_row_key: 1 << 10,
            kv_bytes_per_key: 1 << 10,
        }
    }

    fn profile(text: &str) -> profile::Profile {
        let registry = profile::Registry::new("test", &[], &[]);
        registry.load_user(text, "test").unwrap()
    }

    #[test]
    fn without_rates_the_load_is_refused_by_name() {
        let rungs = facts(&[(1024, 100 * MIB), (2048, 150 * MIB)], 200 * MIB);
        let bare = profile("api = 1\n");
        let err = choose(&rungs, 1 << 33, &bare, "Card X", &counts(), 200 * MIB).unwrap_err();
        assert!(
            err.to_string()
                .contains("device Card X has no rates in its profile"),
            "{err}"
        );
        assert!(
            err.to_string().contains("`pie tune --rates` measures them"),
            "{err}"
        );
        // No feasible rung at all: the ruling's refusal, with the binding.
        let none = facts(&[(1024, 300 * MIB)], 200 * MIB);
        let rated = profile("api = 1\n[rates]\nbf16_flops_per_ns = 1.0\nbytes_per_ns = 1.0\n");
        let err = choose(&none, 1 << 33, &rated, "Card X", &counts(), 200 * MIB).unwrap_err();
        assert!(
            err.to_string()
                .contains("1024 tokens (span 300 MiB): past one binding"),
            "{err}"
        );
    }

    #[test]
    fn the_estimate_reads_the_rates_and_the_counts() {
        let rates = profile::fire::Rates {
            bf16_flops_per_ns: Some(10.0),
            bytes_per_ns: Some(100.0),
            launch_ns: Some(5),
        };
        let e = estimate(&counts(), &rates).unwrap();
        assert_eq!(
            (e.flops_per_ns, e.bytes_per_ns, e.launch_ns),
            (10.0, 100.0, 5.0)
        );
        assert_eq!(e.dense_bytes, 1 << 30);
        let unstated = profile::fire::Rates {
            bytes_per_ns: None,
            ..rates
        };
        assert_eq!(estimate(&counts(), &unstated), None);
    }
}
