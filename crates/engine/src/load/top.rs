//! What every engine weighs its top rung by ([`super::top_rung`]): the
//! model's counts ([`Counts`], from the trace and what the engine alone
//! knows of where its params lie), the estimate over the device's rates
//! (the profile's `[rates]`), the one refusal of a device without them, and
//! the load's one line on the rung chosen. Each engine offers its own rungs
//! (their spans, launches, padded rows and why it cannot lay one).

use std::fmt;

use poem_ir::{Def, Dim, Operation, Trace, Ty};

use super::rung::{FireEstimate, NoRung, RungFacts, TopRung, TopWhy};

/// What a fire of the model reads and computes: the estimate's counts.
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
    /// The contraction of the largest dense plane: the width its matmul
    /// kernel is picked at, which a fire's rows pad to the tile of.
    pub width: u32,
}

/// What the engine alone knows of a trace's params: the bytes each takes
/// on its device (one a param, in order), which are token tables (a gather
/// reads a row of them a token) and which are routed banks.
#[derive(Debug, Clone, Copy)]
pub struct Kept<'a> {
    pub plane_bytes: &'a [u64],
    pub tables: &'a [usize],
    pub banks: &'a [usize],
}

impl Counts {
    /// The counts of `trace`, its params as `kept` states them, and its
    /// attentions' `(flops a row a key, kv bytes a key)` (all layers
    /// together: `poem_exec::store::kv::attention_counts`).
    #[must_use]
    pub fn of(trace: &Trace, kept: Kept<'_>, attention: (u64, u64)) -> Counts {
        let (mut weight_bytes, mut expert_bytes) = (0u64, 0u64);
        let (mut dense, mut routed) = (0u64, 0u64);
        let (mut largest, mut width) = (0u64, 0u64);
        for (at, param) in trace.params.iter().enumerate() {
            if kept.tables.contains(&at) {
                continue;
            }
            let bytes = kept.plane_bytes.get(at).copied().unwrap_or(0);
            let (rows, cols) = rectangle(&param.shape);
            let elements = if param.shape.len() >= 2 {
                rows.saturating_mul(cols)
            } else {
                0
            };
            if kept.banks.contains(&at) {
                expert_bytes = expert_bytes.saturating_add(bytes);
                routed = routed.saturating_add(elements);
            } else {
                weight_bytes = weight_bytes.saturating_add(bytes);
                dense = dense.saturating_add(elements);
                if elements > largest {
                    largest = elements;
                    width = cols;
                }
            }
        }
        let experts = routing(trace);
        let routed_per_row = match experts {
            Some((k, e)) if e > 0 => routed.saturating_mul(u64::from(k)) / u64::from(e),
            // A bank whose routing does not read: every expert, every row.
            _ => routed,
        };
        let (attn_flops_per_row_key, kv_bytes_per_key) = attention;
        Counts {
            weight_bytes,
            expert_bytes,
            experts,
            flops_per_row: dense.saturating_add(routed_per_row).saturating_mul(2),
            attn_flops_per_row_key,
            kv_bytes_per_key,
            width: u32::try_from(width).unwrap_or(u32::MAX),
        }
    }
}

/// The rectangle a param of `shape` is read as: its first axis the rows,
/// the rest its width.
#[must_use]
pub fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
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
        // The curve's widest point until the estimate reads it by rows.
        flops_per_ns: rates.flops_at(u32::MAX)?,
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

/// Why no top rung was chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum Refused {
    /// The device's profile states no rates: nothing weighs its rungs.
    NoRates { device: String },
    /// No rung offered is feasible.
    NoRung(NoRung),
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::NoRates { device } => write!(
                f,
                "device {device} has no rates in its profile (`[rates] bf16_flops_per_ns_by_rows` \
                 and `bytes_per_ns`), so nothing weighs its top rung; `pie tune --rates` measures \
                 them"
            ),
            Refused::NoRung(no) => no.fmt(f),
        }
    }
}

impl std::error::Error for Refused {}

/// The top rung of `rungs` for `device`, whose profile states `rates`, over
/// a live range of `live` bytes, by the estimate from the rates alone. A
/// device whose profile states no rates has nothing to weigh the rungs by,
/// and the load is refused, naming it.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    rates: &profile::fire::Rates,
    device: &str,
    counts: &Counts,
) -> Result<(TopRung, FireEstimate), Refused> {
    let Some(estimate) = estimate(counts, rates) else {
        return Err(Refused::NoRates {
            device: device.to_owned(),
        });
    };
    let top = super::rung::top_rung(rungs, live, &estimate).map_err(Refused::NoRung)?;
    Ok((top, estimate))
}

/// The load's one line on its top rung: which (`top` tokens), by what, and
/// the `span` it reserves of the `live` range; the rates where this load
/// measured them.
#[must_use]
pub fn line_of(
    rung: &TopRung,
    top: u32,
    span: u64,
    live: u64,
    tuned: &profile::fire::Tuned,
) -> String {
    let r = &tuned.rates;
    let rates = format!(
        "{:.1} TFLOP/s bf16, {:.0} GB/s, launch {} ns",
        r.peak_flops().unwrap_or(0.0) / 1e3,
        r.bytes_per_ns.unwrap_or(0.0),
        r.launch_ns.unwrap_or(0)
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

    fn rung(tokens: u32, span: u64, infeasible: Option<&'static str>) -> RungFacts {
        RungFacts {
            tokens,
            lanes: tokens.min(256),
            keys: tokens,
            launches: 100,
            padded_rows: tokens,
            span_bytes: span,
            infeasible,
        }
    }

    fn counts() -> Counts {
        Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            flops_per_row: 1 << 30,
            attn_flops_per_row_key: 1 << 10,
            kv_bytes_per_key: 1 << 10,
            width: 4096,
        }
    }

    #[test]
    fn without_rates_the_load_is_refused_by_name() {
        let rungs = [rung(1024, 100 * MIB, None), rung(2048, 150 * MIB, None)];
        let bare = profile::fire::Rates::default();
        let err = choose(&rungs, 1 << 33, &bare, "Card X", &counts()).unwrap_err();
        assert!(
            err.to_string()
                .contains("device Card X has no rates in its profile"),
            "{err}"
        );
        assert!(
            err.to_string().contains("`pie tune --rates` measures them"),
            "{err}"
        );
        // No feasible rung at all: the ruling's refusal.
        let none = [rung(1024, 300 * MIB, Some("past one binding"))];
        let rated = profile::fire::Rates {
            bf16_flops_per_ns_by_rows: Some(vec![(4096, 1.0)]),
            bytes_per_ns: Some(1.0),
            ..profile::fire::Rates::default()
        };
        let err = choose(&none, 1 << 33, &rated, "Card X", &counts()).unwrap_err();
        assert!(matches!(err, Refused::NoRung(_)), "{err}");
        assert!(
            err.to_string()
                .contains("1024 tokens (span 300 MiB): past one binding"),
            "{err}"
        );
    }

    #[test]
    fn the_estimate_reads_the_rates_and_the_counts() {
        let rates = profile::fire::Rates {
            bf16_flops_per_ns_by_rows: Some(vec![(16, 1.0), (4096, 10.0)]),
            bytes_per_ns: Some(100.0),
            launch_ns: Some(5),
            ..profile::fire::Rates::default()
        };
        let e = estimate(&counts(), &rates).unwrap();
        assert_eq!(
            (e.flops_per_ns, e.bytes_per_ns, e.launch_ns),
            (10.0, 100.0, 5.0)
        );
        assert_eq!(e.dense_bytes, 1 << 30);
        let unstated = profile::fire::Rates {
            bytes_per_ns: None,
            ..rates.clone()
        };
        assert_eq!(estimate(&counts(), &unstated), None);
    }

    #[test]
    fn a_param_is_read_as_its_first_axis_by_the_rest() {
        assert_eq!(rectangle(&[4096, 8, 128]), (4096, 1024));
        assert_eq!(rectangle(&[7]), (7, 1));
        assert_eq!(rectangle(&[]), (1, 1));
    }

    #[test]
    fn the_line_names_the_rung_its_share_and_the_rates() {
        let top = TopRung {
            tokens: 2048,
            lanes: 256,
            span_bytes: 512 * MIB,
            why: TopWhy::Argmax {
                eff: 0.9,
                live_share: 0.25,
            },
        };
        let tuned = profile::fire::Tuned {
            rates: profile::fire::Rates {
                bf16_flops_per_ns_by_rows: Some(vec![(4096, 249_000.0)]),
                bytes_per_ns: Some(654.0),
                launch_ns: Some(1620),
                ..profile::fire::Rates::default()
            },
            measured_this_load: false,
        };
        assert_eq!(
            line_of(&top, 2048, 512 * MIB, 2 << 30, &tuned),
            "top rung 2048 by the estimate: eff 0.900 at a live share of 0.250; span 512 MiB \
             of 2.0 GiB live; the profile's rates: 249.0 TFLOP/s bf16, 654 GB/s, launch 1620 ns"
        );
    }
}
