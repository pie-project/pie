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

/// Why a rung is no rung of this device: even cut a layer a submission, a
/// fire's estimated time a layer passes the device's submission limit.
pub const PAST_SUBMIT_LIMIT: &str =
    "a layer of its fire is estimated past the device's submission limit";

/// The rungs of `rungs` a device whose one submission may run no longer
/// than `limit_ns` takes, a fire at each estimated at `ns`: one whose fire,
/// cut into `layers` submissions of a layer each, still passes the limit a
/// layer is marked infeasible ([`PAST_SUBMIT_LIMIT`]); the widest rung's
/// tokens whose fire stays under it whole is returned, a wider fire cut a
/// layer a submission (0 where none stays under it). `None` where no limit
/// is known.
pub fn submit_limit(
    rungs: &mut [RungFacts],
    ns: impl Fn(&RungFacts) -> f64,
    layers: u32,
    limit_ns: Option<u64>,
) -> Option<u32> {
    let limit = limit_ns? as f64;
    let mut whole = 0;
    for rung in rungs.iter_mut().filter(|r| r.infeasible.is_none()) {
        let fire = ns(rung);
        if fire <= limit {
            whole = whole.max(rung.tokens);
        } else if fire / f64::from(layers.max(1)) > limit {
            rung.infeasible = Some(PAST_SUBMIT_LIMIT);
        }
    }
    Some(whole)
}

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

/// The top rung of `rungs` for `device`, whose profile is `profile`, over a
/// live range of `live` bytes ([`engine::load::choose`]); a refusal for no
/// feasible rung names the binding.
pub fn choose(
    rungs: &[RungFacts],
    live: u64,
    profile: &profile::Profile,
    device: &str,
    counts: &Counts,
    binding: u64,
) -> Result<(TopRung, FireEstimate)> {
    engine::load::choose(rungs, live, &profile.rates(), device, counts).map_err(|refused| {
        Fault::Residency(match refused {
            Refused::NoRung(no) => {
                format!("{no}; this device binds no more than {} MiB", binding >> 20)
            }
            refused @ (Refused::NoRates { .. } | Refused::OffGrid { .. }) => refused.to_string(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn facts(spans: &[(u32, u64)], binding: u64) -> Vec<RungFacts> {
        rung_facts(spans, 256, binding, 100, &|r| r)
    }

    /// Fires estimated at 1 ms a token on a device with a 2 s submission
    /// limit and 4 layers: 1024 tokens fit whole, 2048 and 4096 are cut a
    /// layer a submission, and 16384 (4 s a layer) is no rung.
    #[test]
    fn a_rung_estimated_past_the_submit_limit_is_not_chosen() {
        let mut rungs = facts(
            &[(1024, MIB), (2048, MIB), (4096, MIB), (16384, MIB)],
            200 * MIB,
        );
        let ns = |r: &RungFacts| f64::from(r.tokens) * 1e6;
        assert_eq!(submit_limit(&mut rungs, ns, 4, None), None);
        assert!(rungs.iter().all(|r| r.infeasible.is_none()));
        let whole = submit_limit(&mut rungs, ns, 4, Some(2_000_000_000));
        assert_eq!(whole, Some(1024));
        let refused: Vec<u32> = rungs
            .iter()
            .filter(|r| r.infeasible == Some(PAST_SUBMIT_LIMIT))
            .map(|r| r.tokens)
            .collect();
        assert_eq!(refused, vec![16384]);
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
        let err = choose(&rungs, 1 << 33, &bare, "Card X", &counts(), 200 * MIB).unwrap_err();
        assert!(
            err.to_string()
                .contains("device Card X has no rates in its profile"),
            "{err}"
        );
        // No feasible rung at all: the ruling's refusal, with the binding.
        let none = facts(&[(1024, 300 * MIB)], 200 * MIB);
        let rated = profile(
            "api = 1\n[rates]\nbf16_flops_per_ns_by_rows_n_k = [[4096, 4096, 4096, 1.0]]\nattn_flops_per_ns_by_keys_and_dim = [[2048, 128, 1.0]]\nbytes_per_ns = 1.0\n",
        );
        let err = choose(&none, 1 << 33, &rated, "Card X", &counts(), 200 * MIB).unwrap_err();
        assert!(
            err.to_string().contains(
                "1024 tokens (span 300 MiB): past one binding; this device binds no more than \
                 200 MiB"
            ),
            "{err}"
        );
    }
}
