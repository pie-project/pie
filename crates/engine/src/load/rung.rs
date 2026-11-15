//! The top rung a load lays its fires up to (the ruling): of the feasible
//! rungs, the one whose throughput, against the best any feasible rung
//! reaches, least gives up for the share of the live range its span takes:
//!
//! `r_top = argmax_r eff(r) · (1 − span(r) / M)`, `eff(r) = (r / t(r)) /
//! max_r' (r' / t(r'))`,
//!
//! `t(r)` a fire's time at rung width `r`, `span(r)` the activation span the
//! rung reserves, `M` the live range kv and activation share at load. No
//! constant weighs either term, and an exact tie goes to the narrower rung.
//! The times are measured ones only where a measured table covers every
//! feasible rung's width; otherwise every rung takes the estimate, so one
//! argmax never weighs a measured time against an estimated one.

use std::fmt;

use poem_ir::Trace;

/// One rung offered: fires of at most `tokens` tokens across at most
/// `lanes` lanes, attending at most `keys` keys a row, whose activation
/// span takes `span_bytes`. `infeasible` names why the device cannot lay it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RungFacts {
    pub tokens: u32,
    pub lanes: u32,
    /// The attention extent of a row at this rung: what the estimate's
    /// row-key term is counted over.
    pub keys: u32,
    pub span_bytes: u64,
    pub infeasible: Option<&'static str>,
}

/// A fire's time at each rung width: a measured table, `(tokens, ns)`, and
/// the estimate every rung takes where the table does not cover them all.
#[derive(Debug, Clone, Copy)]
pub struct FireTimes<'a> {
    pub measured: &'a [(u32, u64)],
    pub estimate: FireEstimate,
}

/// `t(r) = fixed_ns + r · per_token_ns + r · keys(r) · per_row_key_ns`
/// ([`FireEstimate::ns`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FireEstimate {
    pub fixed_ns: u64,
    pub per_token_ns: f64,
    pub per_row_key_ns: f64,
}

impl FireEstimate {
    /// The estimated time of a fire at `rung`: the one place the estimate's
    /// form lives.
    #[must_use]
    pub fn ns(&self, rung: &RungFacts) -> f64 {
        let tokens = f64::from(rung.tokens);
        self.fixed_ns as f64
            + tokens * self.per_token_ns
            + tokens * f64::from(rung.keys) * self.per_row_key_ns
    }
}

/// The rung chosen, and why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopRung {
    pub tokens: u32,
    pub lanes: u32,
    pub span_bytes: u64,
    pub why: TopWhy,
    pub times: TimesUsed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopWhy {
    /// The argmax: the chosen rung's `eff` and the share of the live range
    /// its span takes.
    Argmax { eff: f64, live_share: f64 },
    /// One rung alone is feasible: no weighing to do.
    OnlyFeasible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimesUsed {
    /// The measured table covers every feasible rung's width.
    Measured,
    /// No measured table: the estimate for every rung.
    Estimate,
    /// A measured table that misses some feasible rung's width: the
    /// estimate for every rung, the table set aside whole.
    PartialMeasuredIgnored,
}

/// No rung offered is feasible: each rung, its span and why it is not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoRung {
    /// `(tokens, span bytes, why)` of every rung offered.
    pub refused: Vec<(u32, u64, &'static str)>,
}

impl fmt::Display for NoRung {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.refused.is_empty() {
            return write!(f, "no rung of fires is offered");
        }
        write!(f, "no rung of fires is feasible:")?;
        for (k, (tokens, span, why)) in self.refused.iter().enumerate() {
            let sep = if k == 0 { " " } else { "; " };
            write!(f, "{sep}{tokens} tokens (span {} MiB): {why}", span >> 20)?;
        }
        Ok(())
    }
}

impl std::error::Error for NoRung {}

/// The top rung of `rungs` (the module's ruling), over a live range of
/// `live_capacity_bytes`.
pub fn top_rung(
    rungs: &[RungFacts],
    live_capacity_bytes: u64,
    times: &FireTimes,
) -> Result<TopRung, NoRung> {
    let mut feasible: Vec<&RungFacts> = rungs.iter().filter(|r| r.infeasible.is_none()).collect();
    if feasible.is_empty() {
        return Err(NoRung {
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
        });
    }
    // Narrowest first: a later rung must score strictly more to win.
    feasible.sort_by_key(|r| r.tokens);
    let measured = |rung: &RungFacts| {
        times
            .measured
            .iter()
            .find(|&&(tokens, _)| tokens == rung.tokens)
            .map(|&(_, ns)| ns as f64)
    };
    let covered = feasible.iter().all(|r| measured(r).is_some());
    let used = if !times.measured.is_empty() && covered {
        TimesUsed::Measured
    } else if times.measured.is_empty() {
        TimesUsed::Estimate
    } else {
        TimesUsed::PartialMeasuredIgnored
    };
    let chosen = |rung: &RungFacts, why| TopRung {
        tokens: rung.tokens,
        lanes: rung.lanes,
        span_bytes: rung.span_bytes,
        why,
        times: used,
    };
    if let [only] = feasible[..] {
        return Ok(chosen(only, TopWhy::OnlyFeasible));
    }
    let time = |rung: &RungFacts| match used {
        TimesUsed::Measured => measured(rung).unwrap_or(f64::NAN),
        TimesUsed::Estimate | TimesUsed::PartialMeasuredIgnored => times.estimate.ns(rung),
    };
    // Tokens a nanosecond; a rung timed at nothing is unboundedly fast.
    let throughput: Vec<f64> = feasible
        .iter()
        .map(|r| {
            let t = time(r);
            if t > 0.0 {
                f64::from(r.tokens) / t
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let best = throughput.iter().copied().fold(0.0, f64::max);
    let mut top: Option<(usize, f64, f64, f64)> = None;
    for (k, rung) in feasible.iter().enumerate() {
        let eff = if best.is_infinite() {
            if throughput[k].is_infinite() {
                1.0
            } else {
                0.0
            }
        } else if best > 0.0 {
            throughput[k] / best
        } else {
            0.0
        };
        let live_share = share(rung.span_bytes, live_capacity_bytes);
        let score = eff * (1.0 - live_share);
        if top.is_none_or(|(_, _, _, at)| score > at) {
            top = Some((k, eff, live_share, score));
        }
    }
    let (k, eff, live_share, _) = top.expect("a feasible rung was weighed");
    Ok(chosen(feasible[k], TopWhy::Argmax { eff, live_share }))
}

/// The share of `live` bytes `span` takes.
fn share(span: u64, live: u64) -> f64 {
    if live == 0 {
        return if span == 0 { 0.0 } else { f64::INFINITY };
    }
    span as f64 / live as f64
}

/// The key a model's measured fire times are filed under: its deployment
/// (the architecture as configured, with its weight and kv dtypes:
/// `qwen3-8b-bf16-kv-bf16`), less any tensor-parallel width suffix, as one
/// rank's fire is timed alone.
#[must_use]
pub fn model_class(trace: &Trace) -> &str {
    match trace.name.rsplit_once("-tp") {
        Some((base, width)) if !width.is_empty() && width.bytes().all(|b| b.is_ascii_digit()) => {
            base
        }
        _ => &trace.name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn rung(tokens: u32, span_mib: u64) -> RungFacts {
        RungFacts {
            tokens,
            lanes: tokens.min(256),
            keys: tokens,
            span_bytes: span_mib * MIB,
            infeasible: None,
        }
    }

    const FLAT: FireEstimate = FireEstimate {
        fixed_ns: 0,
        per_token_ns: 0.0,
        per_row_key_ns: 0.0,
    };

    fn estimate(fixed_ns: u64, per_token_ns: f64) -> FireEstimate {
        FireEstimate {
            fixed_ns,
            per_token_ns,
            per_row_key_ns: 0.0,
        }
    }

    #[test]
    fn the_argmax_picks_the_documented_rung() {
        // t(r) = 1000 + r: eff 1/1001·k, …; live 1024 MiB.
        // r=  64: r/t = 64/1064  = .0602, eff .0602/.9107 = .0661, share  8/1024, score .0656
        // r= 512: r/t = 512/1512 = .3386, eff .3718, share  64/1024, score .3486
        // r=4096: r/t = 4096/5096 = .8038, eff .8826, share 128/1024, score .7723
        // r=8192: r/t = 8192/9192 = .8912, eff .9786, share 256/1024, score .7339
        // r=10240: r/t = .9107, eff 1, share 400/1024, score .6094
        let rungs = [
            rung(64, 8),
            rung(512, 64),
            rung(4096, 128),
            rung(8192, 256),
            rung(10240, 400),
        ];
        let times = FireTimes {
            measured: &[],
            estimate: estimate(1000, 1.0),
        };
        let top = top_rung(&rungs, 1024 * MIB, &times).unwrap();
        assert_eq!(top.tokens, 4096);
        assert_eq!(top.span_bytes, 128 * MIB);
        assert_eq!(top.times, TimesUsed::Estimate);
        let TopWhy::Argmax { eff, live_share } = top.why else {
            panic!("{top:?}")
        };
        assert!((eff - (4096.0 / 5096.0) / (10240.0 / 11240.0)).abs() < 1e-12);
        assert!((live_share - 0.125).abs() < 1e-12);
    }

    #[test]
    fn a_small_live_capacity_picks_a_narrow_top_rung() {
        // Qwen3-8B on an MI300X: 16.4 GB of bf16 weights read a fire at
        // about 5.3 TB/s (3.1 ms), 16 GFLOP a token at about 1.3 PFLOP/s
        // (12 µs); the activation span grows toward 7.3 GiB at 8192.
        const GIB: f64 = (1u64 << 30) as f64;
        let span = |gib: f64| (gib * GIB) as u64;
        let rungs: Vec<RungFacts> = [
            (256, 0.23),
            (512, 0.46),
            (1024, 0.91),
            (2048, 1.83),
            (4096, 3.65),
            (8192, 7.3),
        ]
        .iter()
        .map(|&(tokens, gib)| RungFacts {
            span_bytes: span(gib),
            ..rung(tokens, 0)
        })
        .collect();
        let times = FireTimes {
            measured: &[],
            estimate: estimate(3_100_000, 12_000.0),
        };
        // A live range far past the span: the throughput decides, and the
        // widest rung is the most efficient.
        let wide = top_rung(&rungs, 1 << 40, &times).unwrap();
        assert_eq!((wide.tokens, wide.times), (8192, TimesUsed::Estimate));
        // A 26 GiB budget: about 3 GiB of kv beside the 7.3 GiB span.
        let live = span(3.0 + 7.3);
        let narrow = top_rung(&rungs, live, &times).unwrap();
        assert!(narrow.tokens < 8192, "{narrow:?}");
        let score = |rung: &TopRung| match rung.why {
            TopWhy::Argmax { eff, live_share } => eff * (1.0 - live_share),
            TopWhy::OnlyFeasible => panic!("{rung:?}"),
        };
        // The wide top's score at this live range: eff 1, its share 7.3 / 10.3.
        let at_wide = 1.0 - 7.3 / 10.3;
        assert!(score(&narrow) > at_wide, "{narrow:?} against {at_wide}");
        assert_eq!(narrow.tokens, 2048);
    }

    #[test]
    fn a_cliff_in_the_measured_table_moves_the_pick() {
        let rungs = [rung(1024, 16), rung(2048, 32), rung(4096, 64)];
        // Throughput doubling with each width: the widest wins.
        let smooth = [(1024, 1_000_000), (2048, 1_000_000), (4096, 1_000_000)];
        let top = top_rung(
            &rungs,
            1024 * MIB,
            &FireTimes {
                measured: &smooth,
                estimate: FLAT,
            },
        )
        .unwrap();
        assert_eq!((top.tokens, top.times), (4096, TimesUsed::Measured));
        // The widest falls off a cliff (four times as long a fire).
        let cliff = [(1024, 1_000_000), (2048, 1_000_000), (4096, 4_000_000)];
        let top = top_rung(
            &rungs,
            1024 * MIB,
            &FireTimes {
                measured: &cliff,
                estimate: FLAT,
            },
        )
        .unwrap();
        assert_eq!((top.tokens, top.times), (2048, TimesUsed::Measured));
    }

    #[test]
    fn a_partial_table_is_set_aside_for_the_estimate_and_said_so() {
        let rungs = [rung(1024, 16), rung(2048, 32), rung(4096, 64)];
        // Measured alone, 1024 would win; the table misses 4096.
        let partial = [(1024, 1_000), (2048, 1_000_000)];
        let times = FireTimes {
            measured: &partial,
            estimate: estimate(1_000_000, 1.0),
        };
        let top = top_rung(&rungs, 1024 * MIB, &times).unwrap();
        assert_eq!(top.times, TimesUsed::PartialMeasuredIgnored);
        // Under the estimate: 4096 scores 1 · (1 − 1/16) = .9375, 2048
        // .5010 · (1 − 1/32) = .4853, 1024 .2508 · (1 − 1/64): 4096 wins.
        assert_eq!(top.tokens, 4096);
        // Widths past the feasible ones do not make a table partial.
        let mut infeasible = rungs;
        infeasible[2].infeasible = Some("past one binding");
        let covering = [(1024, 1_000), (2048, 1_000_000)];
        let top = top_rung(
            &infeasible,
            1024 * MIB,
            &FireTimes {
                measured: &covering,
                estimate: estimate(1_000_000, 1.0),
            },
        )
        .unwrap();
        assert_eq!((top.tokens, top.times), (1024, TimesUsed::Measured));
    }

    #[test]
    fn an_infeasible_rung_is_never_chosen() {
        let mut rungs = [rung(1024, 16), rung(2048, 32), rung(4096, 64)];
        rungs[2].infeasible = Some("past one binding");
        let times = FireTimes {
            measured: &[],
            estimate: estimate(1_000_000_000, 1.0),
        };
        let top = top_rung(&rungs, 1 << 40, &times).unwrap();
        assert_eq!(top.tokens, 2048);
        rungs[1].infeasible = Some("past one binding");
        let top = top_rung(&rungs, 1 << 40, &times).unwrap();
        assert_eq!((top.tokens, top.why), (1024, TopWhy::OnlyFeasible));
    }

    #[test]
    fn an_exact_tie_goes_to_the_narrower_rung() {
        // Equal throughput, equal span: equal scores.
        let rungs = [rung(4096, 64), rung(2048, 64)];
        let measured = [(2048, 1_000), (4096, 2_000)];
        let top = top_rung(
            &rungs,
            1024 * MIB,
            &FireTimes {
                measured: &measured,
                estimate: FLAT,
            },
        )
        .unwrap();
        assert_eq!(top.tokens, 2048);
        // Every rung timed at nothing: eff 1 each, equal spans.
        let top = top_rung(
            &rungs,
            1024 * MIB,
            &FireTimes {
                measured: &[],
                estimate: FLAT,
            },
        )
        .unwrap();
        assert_eq!(top.tokens, 2048);
    }

    #[test]
    fn no_rung_names_every_reason() {
        let mut rungs = [rung(1024, 3000), rung(2048, 6000)];
        rungs[0].infeasible = Some("past one binding");
        rungs[1].infeasible = Some("past one binding");
        let err = top_rung(
            &rungs,
            1024 * MIB,
            &FireTimes {
                measured: &[],
                estimate: FLAT,
            },
        )
        .unwrap_err();
        assert_eq!(
            err.refused,
            vec![
                (1024, 3000 * MIB, "past one binding"),
                (2048, 6000 * MIB, "past one binding")
            ]
        );
        assert_eq!(
            err.to_string(),
            "no rung of fires is feasible: 1024 tokens (span 3000 MiB): past one binding; \
             2048 tokens (span 6000 MiB): past one binding"
        );
    }

    #[test]
    fn the_estimate_is_its_documented_form() {
        let e = FireEstimate {
            fixed_ns: 100,
            per_token_ns: 2.0,
            per_row_key_ns: 0.5,
        };
        let r = RungFacts {
            tokens: 8,
            lanes: 8,
            keys: 16,
            span_bytes: 0,
            infeasible: None,
        };
        assert_eq!(e.ns(&r), 100.0 + 16.0 + 64.0);
    }
}
