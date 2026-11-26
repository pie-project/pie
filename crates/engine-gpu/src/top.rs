//! The top rung a load lays its fires up to ([`engine::load::top_rung`]):
//! the rungs the requested fire size offers, each infeasible whose span
//! passes one binding, every fire timed by the estimate
//! ([`engine::load::FireEstimate`]) over the device's rates (the profile's
//! `[rates]`) and the model's counts ([`engine::load::Counts`]).

use engine::load::{FireEstimate, Kept, Refused, RungFacts, TopRung};
use poem_ir::Trace;

use crate::error::{Fault, Result};

pub use engine::load::{Counts, estimate, line_of};

pub use crate::store::pool::{PAST_ONE_BINDING, PAST_THE_POOL};

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

/// The rungs `(tokens, span bytes)` offers as the fire-top rule
/// weighs them: a lane a token up to `max_lanes`, a row attending its
/// fire's tokens, `launches` kernels a fire, a matmul's rows padded by
/// `padded`, and infeasible for the reason `why` states of its tokens (the
/// pool's, `store::pool::infeasible`).
#[must_use]
pub fn rung_facts(
    spans: &[(u32, u64)],
    max_lanes: u32,
    why: &dyn Fn(u32) -> Option<&'static str>,
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
            infeasible: why(tokens),
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
#[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn facts(spans: &[(u32, u64)], binding: u64) -> Vec<RungFacts> {
        let past = |tokens: u32| {
            spans
                .iter()
                .any(|&(at, bytes)| at == tokens && bytes > binding)
                .then_some(PAST_ONE_BINDING)
        };
        rung_facts(spans, 256, &past, 100, &|r| r)
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
        // No feasible rung at all: the fire-top rule's refusal, with the binding.
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
}
