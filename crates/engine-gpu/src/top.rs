//! The top rung a load lays its fires up to ([`engine::load::top_rung`]):
//! the rungs the requested fire size offers, each infeasible whose span
//! passes one binding, every fire timed by the estimate
//! ([`engine::load::FireEstimate`]) over the device's rates (the profile's
//! `[rates]`) and the model's counts ([`engine::load::Counts`]).

use engine::load::{FireEstimate, Kept, Refused, RungFacts, TopRung};
use poem_ir::Trace;

use crate::error::{Fault, Result};

pub use engine::load::{Counts, estimate, line_of};

/// Why a rung is no rung of this device: its span lies in the pool's last
/// segment, and no segment is past a binding.
pub const PAST_ONE_BINDING: &str = "past one binding";

/// The counts of `trace` over the planes this device keeps.
pub fn counts(trace: &Trace) -> Result<Counts> {
    let planes = crate::weights::plane_bytes(trace)?;
    let tables = crate::store::rows::streamed(trace);
    let banks = crate::store::experts::streamed(trace);
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
            // One fire carries every lane the budget seats.
            fire_lanes: max_lanes,
        })
        .collect()
}

/// The top rung of `rungs` for `device` (the fire top, `engine::load::fire_top`,
/// over those a prompt of `context` holds), whose profile is `profile`, over a
/// live range of `live` bytes, each matmul's rows padded to the row tile of
/// the kernel the backend picks at its shape (`row_tile`,
/// [`engine::load::choose_tiled`]); a refusal for no feasible rung names the
/// binding.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    context: u32,
    profile: &profile::Profile,
    row_tile: engine::roofline::RowTile,
    device: &str,
    counts: &Counts,
    binding: u64,
) -> Result<(TopRung, FireEstimate)> {
    engine::load::choose_tiled(
        rungs,
        live,
        context,
        &profile.rates(),
        Some(row_tile),
        device,
        counts,
    )
    .map_err(|refused| {
        Fault::Residency(match refused {
            Refused::NoRung(no) => {
                format!("{no}; this device binds no more than {} MiB", binding >> 20)
            }
            refused @ Refused::NoRates { .. } => refused.to_string(),
        })
    })
}

/// A step's submit time as the roofline estimates it (the load's
/// [`FireEstimate`], the one every rung is weighed by), and how a step
/// past the device's bound on one submission is cut into segments of
/// layers, each its own submit, each within the bound.
#[derive(Debug, Clone)]
pub struct SubmitTime {
    estimate: FireEstimate,
    /// The kernels a fire launches: the compiled plan's.
    launches: u32,
    /// The most lanes one fire carries.
    fire_lanes: u32,
    /// The layers the trace states its nodes in.
    layers: u32,
}

impl SubmitTime {
    #[must_use]
    pub fn new(estimate: FireEstimate, launches: u32, fire_lanes: u32, layers: u32) -> Self {
        SubmitTime {
            estimate,
            launches,
            fire_lanes,
            layers: layers.max(1),
        }
    }

    /// The estimated time of a step of `rows` rows over `lanes` lanes, its
    /// widest lane attending `keys` keys, in ns.
    #[must_use]
    pub fn ns(&self, rows: u32, lanes: u32, keys: u32) -> f64 {
        self.estimate.ns(&RungFacts {
            tokens: rows.max(1),
            lanes: lanes.max(1),
            keys: keys.max(1),
            launches: self.launches,
            padded_rows: rows.max(1),
            span_bytes: 0,
            infeasible: None,
            fire_lanes: self.fire_lanes,
        })
    }

    /// The layers one submit of such a step carries under `bound`, each
    /// layer its share of the step's estimate: `None` where the whole step
    /// lies within the bound and is submitted whole, else the most layers
    /// whose share does. A step one layer of which passes the bound alone
    /// is refused by name: no cut submits it within the bound.
    pub fn split(
        &self,
        rows: u32,
        lanes: u32,
        keys: u32,
        bound: std::time::Duration,
    ) -> Result<Option<u32>> {
        let step = self.ns(rows, lanes, keys);
        let bound_ns = bound.as_nanos() as f64;
        if step <= bound_ns {
            return Ok(None);
        }
        let layer = step / f64::from(self.layers);
        if layer > bound_ns {
            return Err(Fault::Ceiling {
                what: "ns one layer of a step is estimated to take, against the device's bound \
                       on one submission",
                need: layer as u64,
                have: bound_ns as u64,
            });
        }
        // At least one layer, by the check above; fewer than all, as the
        // whole step passes the bound.
        let layers = (bound_ns / layer).floor() as u32;
        Ok(Some(layers.clamp(1, self.layers - 1)))
    }
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
            gemm_shapes: vec![engine::load::GemmShape::dense(4096, 4096, 1 << 30, 1 << 30)],
            head_bytes: 0,
            attn_flops_per_row_key: 1 << 10,
            kv_bytes_per_key: 1 << 10,
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

    fn profile(text: &str) -> profile::Profile {
        let registry = profile::Registry::new("test", &[], &[]);
        registry.load_user(text, "test").unwrap()
    }

    #[test]
    fn without_rates_the_load_is_refused_by_name() {
        let rungs = facts(&[(1024, 100 * MIB), (2048, 150 * MIB)], 200 * MIB);
        let bare = profile("api = 1\n");
        let err = choose(
            &rungs,
            1 << 33,
            1 << 20,
            &bare,
            std::sync::Arc::new(|_, _, _| 1),
            "Card X",
            &counts(),
            200 * MIB,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("device Card X has no rates in its profile"),
            "{err}"
        );
        // No feasible rung at all: the ruling's refusal, with the binding.
        let none = facts(&[(1024, 300 * MIB)], 200 * MIB);
        let rated = profile(
            "api = 1\n[rates]\ngemm_tflops_by_rows = [[4096, [[4096, 1.0]]]]\nattn_tflops_by_keys = \
                 [[2048, 1.0]]\nhbm_gbps = 1.0\nelementwise_gbps = 1.0\n",
        );
        let err = choose(
            &none,
            1 << 33,
            1 << 20,
            &rated,
            std::sync::Arc::new(|_, _, _| 1),
            "Card X",
            &counts(),
            200 * MIB,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains(
                "1024 tokens (span 300 MiB): past one binding; this device binds no more than \
                 200 MiB"
            ),
            "{err}"
        );
    }

    /// The estimate at 1 TFLOP/s and 1 GB/s over 28 layers of one 4096
    /// square plane each, the load's: a step's submit time scales with its
    /// rows; one within the bound is submitted whole, one past it in
    /// segments of the most layers whose share lies within it, and one a
    /// layer of which passes it is refused.
    #[test]
    fn a_step_past_the_bound_on_one_submission_is_cut_into_layers_within_it() {
        let rated = profile(
            "api = 1\n[rates]\ngemm_tflops_by_rows = [[4096, [[1, 1.0]]]]\nattn_tflops_by_keys = \
                 [[1, 1.0]]\nhbm_gbps = 1.0\nelementwise_gbps = 1.0\n",
        );
        let estimate = estimate(&counts(), &rated.rates()).expect("the estimate");
        let time = SubmitTime::new(estimate, 100, 256, 28);
        let one = time.ns(1, 1, 1);
        let wide = time.ns(4096, 1, 4096);
        assert!(wide > one, "{wide} ns at 4096 rows, {one} at one");
        let bound = |ns: f64| std::time::Duration::from_nanos(ns as u64);

        assert_eq!(time.split(4096, 1, 4096, bound(wide * 2.0)).unwrap(), None);
        // A third of the step a submit: 28 layers at ~9.33 each, 9 a segment.
        assert_eq!(
            time.split(4096, 1, 4096, bound(wide / 3.0)).unwrap(),
            Some(9)
        );
        // Just under the whole step: every layer but one in the first.
        assert_eq!(
            time.split(4096, 1, 4096, bound(wide * 0.99)).unwrap(),
            Some(27)
        );
        let refused = time.split(4096, 1, 4096, bound(wide / 56.0)).unwrap_err();
        assert!(
            refused.to_string().contains("one layer of a step"),
            "{refused}"
        );
    }
}
