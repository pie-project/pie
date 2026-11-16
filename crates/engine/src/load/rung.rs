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
//!
//! `t(r)` is the estimate alone ([`FireEstimate::ns`]): work over the
//! device's rates (the profile's `[rates]`), counted from what the load
//! knows of the model and the compiled plan. No measured fire time is read
//! at load; measured fire times only test the estimate (`pie tune
//! --check-estimate`).

use std::fmt;

use poem_ir::Trace;

/// One rung offered: fires of at most `tokens` tokens across at most
/// `lanes` lanes, whose activation span takes `span_bytes`. `infeasible`
/// names why the device cannot lay it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RungFacts {
    pub tokens: u32,
    pub lanes: u32,
    /// The keys a row attends at this rung: the attention extent the
    /// estimate's attention and kv terms count over.
    pub keys: u32,
    /// The kernels a fire at this rung issues: the compiled plan's launches.
    pub launches: u32,
    /// The rows a matmul over the fire's tokens computes: `tokens` padded
    /// to the backend's matmul tile at that width, so a narrow fire pays for
    /// the tile it under-fills.
    pub padded_rows: u32,
    pub span_bytes: u64,
    pub infeasible: Option<&'static str>,
}

/// What a fire costs, as work over the device's rates (the one shared
/// formula every engine weighs its top rung by):
///
/// `t(r) = launches(r) · launch_ns
///       + max(weights(r) / bytes_per_ns,
///             padded_rows(r) · flops_per_row / flops(padded_rows(r)))
///       + lanes(r) · readout_flops / flops(lanes(r))
///       + tokens(r) · keys(r) · attn_flops_per_row_key / attn_flops_per_ns
///       + keys(r) · kv_bytes_per_key / bytes_per_ns
///       + tokens(r) · state_elements_per_row / scan_elements_per_ns
///       + tokens(r) · elementwise_bytes_per_row / bytes_per_ns`
///
/// where `weights(r) = dense_bytes + expert_bytes · (1 − (1 − k/E)^tokens)`:
/// a fire reads every dense weight once (the output head's among them), and
/// of a routed bank the experts its rows touch (each row picks `k` of `E`).
/// `flops(m)` is the device's matmul rate at `m` rows, its measured curve
/// interpolated in `log m` ([`profile::fire::flops_at`]). The output head
/// runs once a lane (the readout), not once a row, so its flops are their
/// own term. Sources: the rates are the profile's `[rates]` (measured by
/// `pie tune --rates` at `crate::rates`' lattice, through the backend's own
/// paths); the byte, flop, attention, state and elementwise counts are the
/// trace's ([`super::Counts`]); `launches` and `padded_rows` are the
/// backend's, from its compiled plan and the tile its matmul takes at the
/// rung's rows.
#[derive(Debug, Clone, PartialEq)]
pub struct FireEstimate {
    /// Dense bf16 matmul flops a nanosecond by rows: `(m, rate)`, ascending.
    pub flops_by_rows: Vec<(u32, f64)>,
    /// Attention flops a nanosecond; read only where the model attends.
    pub attn_flops_per_ns: f64,
    /// Recurrent state elements a nanosecond; read only where the model
    /// carries recurrent state.
    pub scan_elements_per_ns: f64,
    /// Device memory bytes a nanosecond.
    pub bytes_per_ns: f64,
    /// What one launch costs beside its work.
    pub launch_ns: f64,
    /// The weight bytes every fire reads.
    pub dense_bytes: u64,
    /// The bytes of every routed expert, all banks together.
    pub expert_bytes: u64,
    /// `(k, E)`: the experts a row picks, of how many; `None` without a
    /// routed bank.
    pub experts: Option<(u32, u32)>,
    /// The matmul flops a row takes: two a weight element it multiplies
    /// (the dense ones but the output head, and the `k` experts' a row
    /// picks).
    pub flops_per_row: f64,
    /// The output head's flops a lane: two a head element.
    pub readout_flops: f64,
    /// The flops a row takes a key it attends, all layers together.
    pub attn_flops_per_row_key: f64,
    /// The kv bytes a key read takes, all layers together.
    pub kv_bytes_per_key: f64,
    /// The recurrent state elements a row steps, all layers together.
    pub state_elements_per_row: f64,
    /// The activation bytes a row's norms and elementwise ops read and
    /// write, all layers together.
    pub elementwise_bytes_per_row: f64,
}

impl FireEstimate {
    /// The weight bytes a fire of `tokens` reads.
    #[must_use]
    pub fn weight_bytes(&self, tokens: u32) -> f64 {
        let touched = match self.experts {
            Some((k, e)) if e > 0 => {
                1.0 - (1.0 - f64::from(k.min(e)) / f64::from(e)).powf(f64::from(tokens))
            }
            _ => 0.0,
        };
        self.dense_bytes as f64 + self.expert_bytes as f64 * touched
    }

    /// The matmul rate at `rows`.
    #[must_use]
    pub fn flops_per_ns(&self, rows: u32) -> f64 {
        profile::fire::flops_at(&self.flops_by_rows, rows).unwrap_or(f64::INFINITY)
    }

    /// The estimate's terms at `rung`, in ns: the one place its form lives.
    #[must_use]
    pub fn terms(&self, rung: &RungFacts) -> Terms {
        let tokens = f64::from(rung.tokens);
        let keys = f64::from(rung.keys);
        // Work over a rate, nothing where there is no work.
        let over = |work: f64, rate: f64| if work > 0.0 { work / rate } else { 0.0 };
        Terms {
            launch: f64::from(rung.launches) * self.launch_ns,
            weights: over(self.weight_bytes(rung.tokens), self.bytes_per_ns),
            flops: over(
                f64::from(rung.padded_rows) * self.flops_per_row,
                self.flops_per_ns(rung.padded_rows),
            ),
            readout: over(
                f64::from(rung.lanes) * self.readout_flops,
                self.flops_per_ns(rung.lanes),
            ),
            attention: over(
                tokens * keys * self.attn_flops_per_row_key,
                self.attn_flops_per_ns,
            ),
            kv: over(keys * self.kv_bytes_per_key, self.bytes_per_ns),
            recurrent: over(
                tokens * self.state_elements_per_row,
                self.scan_elements_per_ns,
            ),
            elementwise: over(tokens * self.elementwise_bytes_per_row, self.bytes_per_ns),
        }
    }

    /// The estimated time of a fire at `rung`.
    #[must_use]
    pub fn ns(&self, rung: &RungFacts) -> f64 {
        self.terms(rung).total()
    }
}

/// The estimate's terms of one fire, in ns ([`FireEstimate::terms`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Terms {
    /// `launches · launch_ns`.
    pub launch: f64,
    /// The weight bytes the fire reads, over the bandwidth.
    pub weights: f64,
    /// The padded rows' matmul flops, over the matmul rate at those rows.
    pub flops: f64,
    /// The output head's flops over the lanes, over the rate at the lanes.
    pub readout: f64,
    /// The attention's flops, over the attention rate.
    pub attention: f64,
    /// The kv bytes the keys read, over the bandwidth.
    pub kv: f64,
    /// The recurrent state the rows step, over the scan rate.
    pub recurrent: f64,
    /// The norms' and elementwise ops' activation bytes, over the bandwidth.
    pub elementwise: f64,
}

impl Terms {
    /// The fire's time: the launches, the larger of the weights' read and
    /// the matmuls' compute, then each other term in turn.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.launch
            + self.weights.max(self.flops)
            + self.readout
            + self.attention
            + self.kv
            + self.recurrent
            + self.elementwise
    }
}

/// The rung chosen, and why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopRung {
    pub tokens: u32,
    pub lanes: u32,
    pub span_bytes: u64,
    pub why: TopWhy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopWhy {
    /// The argmax: the chosen rung's `eff` and the share of the live range
    /// its span takes.
    Argmax { eff: f64, live_share: f64 },
    /// One rung alone is feasible: no weighing to do.
    OnlyFeasible,
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
/// `live_capacity_bytes`, each rung's fire timed by `estimate`.
pub fn top_rung(
    rungs: &[RungFacts],
    live_capacity_bytes: u64,
    estimate: &FireEstimate,
) -> Result<TopRung, NoRung> {
    let times: Vec<f64> = rungs.iter().map(|r| estimate.ns(r)).collect();
    pick(rungs, &times, live_capacity_bytes)
}

/// The ruling's argmax over `rungs` whose fires take `times` (ns, one a
/// rung): what [`top_rung`] weighs by the estimate, and what a check of the
/// estimate weighs by measured times.
pub fn pick(
    rungs: &[RungFacts],
    times: &[f64],
    live_capacity_bytes: u64,
) -> Result<TopRung, NoRung> {
    let mut feasible: Vec<(&RungFacts, f64)> = rungs
        .iter()
        .zip(times.iter().copied())
        .filter(|(r, _)| r.infeasible.is_none())
        .collect();
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
    feasible.sort_by_key(|(r, _)| r.tokens);
    let chosen = |rung: &RungFacts, why| TopRung {
        tokens: rung.tokens,
        lanes: rung.lanes,
        span_bytes: rung.span_bytes,
        why,
    };
    if let [(only, _)] = feasible[..] {
        return Ok(chosen(only, TopWhy::OnlyFeasible));
    }
    let scores = scores(&feasible, live_capacity_bytes);
    let mut top: Option<(usize, f64)> = None;
    for (k, &(_, _, score)) in scores.iter().enumerate() {
        if top.is_none_or(|(_, at)| score > at) {
            top = Some((k, score));
        }
    }
    let (k, _) = top.expect("a feasible rung was weighed");
    let (eff, live_share, _) = scores[k];
    Ok(chosen(feasible[k].0, TopWhy::Argmax { eff, live_share }))
}

/// `(eff, live share, score)` of each of `feasible` (`(rung, ns)`).
fn scores(feasible: &[(&RungFacts, f64)], live: u64) -> Vec<(f64, f64, f64)> {
    // Tokens a nanosecond; a rung timed at nothing is unboundedly fast.
    let throughput: Vec<f64> = feasible
        .iter()
        .map(|&(r, t)| {
            if t > 0.0 {
                f64::from(r.tokens) / t
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let best = throughput.iter().copied().fold(0.0, f64::max);
    feasible
        .iter()
        .zip(&throughput)
        .map(|(&(rung, _), &thr)| {
            let eff = if best.is_infinite() {
                if thr.is_infinite() { 1.0 } else { 0.0 }
            } else if best > 0.0 {
                thr / best
            } else {
                0.0
            };
            let live_share = share(rung.span_bytes, live);
            (eff, live_share, eff * (1.0 - live_share))
        })
        .collect()
}

/// Each feasible rung's score under `times` (ns, one a rung), by tokens:
/// what a check of the estimate compares the two picks by.
#[must_use]
pub fn scored(rungs: &[RungFacts], times: &[f64], live_capacity_bytes: u64) -> Vec<(u32, f64)> {
    let mut feasible: Vec<(&RungFacts, f64)> = rungs
        .iter()
        .zip(times.iter().copied())
        .filter(|(r, _)| r.infeasible.is_none())
        .collect();
    feasible.sort_by_key(|(r, _)| r.tokens);
    feasible
        .iter()
        .zip(scores(&feasible, live_capacity_bytes))
        .map(|(&(r, _), (_, _, score))| (r.tokens, score))
        .collect()
}

/// The share of `live` bytes `span` takes.
fn share(span: u64, live: u64) -> f64 {
    if live == 0 {
        return if span == 0 { 0.0 } else { f64::INFINITY };
    }
    span as f64 / live as f64
}

/// The key a model is named by on a device's checks: its deployment (the
/// architecture as configured, with its weight and kv dtypes:
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
    const GIB: f64 = (1u64 << 30) as f64;

    fn rung(tokens: u32, span_mib: u64) -> RungFacts {
        RungFacts {
            tokens,
            lanes: tokens.min(256),
            keys: tokens,
            launches: 0,
            padded_rows: tokens,
            span_bytes: span_mib * MIB,
            infeasible: None,
        }
    }

    /// A fire of `fixed_ns` beside `per_token_ns` a token: weights read at
    /// a byte a ns, one flop a ns.
    fn estimate(fixed_ns: u64, per_token_ns: f64) -> FireEstimate {
        FireEstimate {
            flops_by_rows: vec![(1, 1.0)],
            attn_flops_per_ns: 0.0,
            scan_elements_per_ns: 0.0,
            bytes_per_ns: 1.0,
            launch_ns: 0.0,
            dense_bytes: fixed_ns,
            expert_bytes: 0,
            experts: None,
            flops_per_row: per_token_ns,
            readout_flops: 0.0,
            attn_flops_per_row_key: 0.0,
            kv_bytes_per_key: 0.0,
            state_elements_per_row: 0.0,
            elementwise_bytes_per_row: 0.0,
        }
    }

    /// `fixed + r · per_token` with the read and the compute summed: the
    /// estimate's `max` of the two holds this where launches carry the
    /// fixed part.
    fn summed(fixed_ns: u64, per_token_ns: f64) -> (FireEstimate, impl Fn(u32, u64) -> RungFacts) {
        let e = FireEstimate {
            launch_ns: fixed_ns as f64,
            dense_bytes: 0,
            ..estimate(0, per_token_ns)
        };
        (e, |tokens, span| RungFacts {
            launches: 1,
            ..rung(tokens, span)
        })
    }

    #[test]
    fn the_argmax_picks_the_documented_rung() {
        // t(r) = 1000 + r: live 1024 MiB.
        // r=  64: r/t = .0602, eff .0661, share  8/1024, score .0656
        // r= 512: r/t = .3386, eff .3718, share  64/1024, score .3486
        // r=4096: r/t = .8038, eff .8826, share 128/1024, score .7723
        // r=8192: r/t = .8912, eff .9786, share 256/1024, score .7339
        // r=10240: r/t = .9107, eff 1, share 400/1024, score .6094
        let (e, at) = summed(1000, 1.0);
        let rungs = [
            at(64, 8),
            at(512, 64),
            at(4096, 128),
            at(8192, 256),
            at(10240, 400),
        ];
        let top = top_rung(&rungs, 1024 * MIB, &e).unwrap();
        assert_eq!((top.tokens, top.span_bytes), (4096, 128 * MIB));
        let TopWhy::Argmax { eff, live_share } = top.why else {
            panic!("{top:?}")
        };
        assert!((eff - (4096.0 / 5096.0) / (10240.0 / 11240.0)).abs() < 1e-12);
        assert!((live_share - 0.125).abs() < 1e-12);
    }

    #[test]
    fn a_narrow_fire_pays_for_the_tile_it_under_fills() {
        // Compute-bound past a read of 100 ns: one flop a row a ns. Unpadded,
        // 16 tokens beat 64 (equal throughput, less span); padded to a
        // 64-row tile, 16 tokens cost what 64 do.
        let e = estimate(100, 10.0);
        let unpadded = [rung(16, 1), rung(64, 4)];
        let times: Vec<f64> = unpadded.iter().map(|r| e.ns(r)).collect();
        assert_eq!(times, [160.0, 640.0]);
        let mut padded = unpadded;
        padded[0].padded_rows = 64;
        assert_eq!(e.ns(&padded[0]), 640.0);
        let top = top_rung(&padded, 1024 * MIB, &e).unwrap();
        assert_eq!(top.tokens, 64);
    }

    #[test]
    fn a_routed_bank_is_read_as_far_as_its_rows_touch_it() {
        let e = FireEstimate {
            dense_bytes: 1000,
            expert_bytes: 64_000,
            experts: Some((8, 64)),
            ..estimate(0, 0.0)
        };
        // One row touches 8 of 64; every row past a few hundred all of them.
        assert!((e.weight_bytes(1) - (1000.0 + 8000.0)).abs() < 1e-9);
        let two = 1.0 - (56.0f64 / 64.0).powi(2);
        assert!((e.weight_bytes(2) - (1000.0 + 64_000.0 * two)).abs() < 1e-9);
        assert!(e.weight_bytes(1024) > 64_999.0);
    }

    #[test]
    fn a_small_live_capacity_picks_a_narrow_top_rung() {
        // Qwen3-8B on an MI300X: 16.4 GB of bf16 weights read a fire at
        // about 5.3 TB/s (3.1 ms), 16 GFLOP a token at about 1.3 PFLOP/s
        // (12 µs); the activation span grows toward 7.3 GiB at 8192.
        let span = |gib: f64| (gib * GIB) as u64;
        let (e, at) = summed(3_100_000, 12_000.0);
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
            ..at(tokens, 0)
        })
        .collect();
        // A live range far past the span: the widest rung is the most
        // efficient.
        assert_eq!(top_rung(&rungs, 1 << 40, &e).unwrap().tokens, 8192);
        // A 26 GiB budget: about 3 GiB of kv beside the 7.3 GiB span.
        let live = span(3.0 + 7.3);
        let narrow = top_rung(&rungs, live, &e).unwrap();
        let TopWhy::Argmax { eff, live_share } = narrow.why else {
            panic!("{narrow:?}")
        };
        // The wide top's score at this live range: eff 1, its share 7.3 / 10.3.
        let at_wide = 1.0 - 7.3 / 10.3;
        assert!(
            eff * (1.0 - live_share) > at_wide,
            "{narrow:?} against {at_wide}"
        );
        assert_eq!(narrow.tokens, 2048);
    }

    #[test]
    fn an_infeasible_rung_is_never_chosen() {
        let (e, at) = summed(1_000_000_000, 1.0);
        let mut rungs = [at(1024, 16), at(2048, 32), at(4096, 64)];
        rungs[2].infeasible = Some("past one binding");
        assert_eq!(top_rung(&rungs, 1 << 40, &e).unwrap().tokens, 2048);
        rungs[1].infeasible = Some("past one binding");
        let top = top_rung(&rungs, 1 << 40, &e).unwrap();
        assert_eq!((top.tokens, top.why), (1024, TopWhy::OnlyFeasible));
    }

    #[test]
    fn an_exact_tie_goes_to_the_narrower_rung() {
        // Equal throughput, equal span: equal scores.
        let rungs = [rung(4096, 64), rung(2048, 64)];
        let top = pick(&rungs, &[2_000.0, 1_000.0], 1024 * MIB).unwrap();
        assert_eq!(top.tokens, 2048);
        // Every rung timed at nothing: eff 1 each, equal spans.
        let top = top_rung(&rungs, 1024 * MIB, &estimate(0, 0.0)).unwrap();
        assert_eq!(top.tokens, 2048);
    }

    #[test]
    fn no_rung_names_every_reason() {
        let mut rungs = [rung(1024, 3000), rung(2048, 6000)];
        rungs[0].infeasible = Some("past one binding");
        rungs[1].infeasible = Some("past one binding");
        let err = top_rung(&rungs, 1024 * MIB, &estimate(0, 0.0)).unwrap_err();
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

    /// Every term at once: 5 launches, 8 tokens over 4 lanes attending 16
    /// keys, the matmuls padded to 32 rows.
    fn documented() -> (FireEstimate, RungFacts) {
        let e = FireEstimate {
            // 1 a ns at 4 rows, 4 at 64: 2 at 16, 2.5 at 32 (log rows).
            flops_by_rows: vec![(4, 1.0), (64, 4.0)],
            attn_flops_per_ns: 0.25,
            scan_elements_per_ns: 2.0,
            bytes_per_ns: 4.0,
            launch_ns: 3.0,
            dense_bytes: 400,
            expert_bytes: 0,
            experts: None,
            flops_per_row: 10.0,
            readout_flops: 6.0,
            attn_flops_per_row_key: 0.5,
            kv_bytes_per_key: 8.0,
            state_elements_per_row: 3.0,
            elementwise_bytes_per_row: 2.0,
        };
        let r = RungFacts {
            tokens: 8,
            lanes: 4,
            keys: 16,
            launches: 5,
            padded_rows: 32,
            span_bytes: 0,
            infeasible: None,
        };
        (e, r)
    }

    #[test]
    fn the_matmul_rate_is_the_curve_at_the_padded_rows() {
        let (e, r) = documented();
        assert!((e.flops_per_ns(32) - 2.5).abs() < 1e-12);
        assert_eq!(e.flops_per_ns(1), 1.0, "clamped below");
        assert_eq!(e.flops_per_ns(4096), 4.0, "clamped above");
        // 32 · 10 flops at 2.5 a ns: compute bound past the 400 / 4 read.
        let t = e.terms(&r);
        assert!((t.flops - 128.0).abs() < 1e-9);
        assert_eq!(t.weights, 100.0);
    }

    #[test]
    fn the_readout_runs_once_a_lane_at_the_lanes_rate() {
        let (e, r) = documented();
        // 4 lanes · 6 flops at 1 a ns (the curve at 4 rows).
        assert_eq!(e.terms(&r).readout, 24.0);
        let wide = RungFacts { lanes: 16, ..r };
        // 16 · 6 at 2 a ns: the readout does not grow with the tokens.
        assert_eq!(e.terms(&wide).readout, 48.0);
        let long = RungFacts { tokens: 64, ..r };
        assert_eq!(e.terms(&long).readout, 24.0);
    }

    #[test]
    fn attention_reads_its_own_rate_and_kv_the_bandwidth() {
        let (e, r) = documented();
        let t = e.terms(&r);
        // 8 · 16 · 0.5 at 0.25 a ns; 16 · 8 bytes at 4 a ns.
        assert_eq!(t.attention, 256.0);
        assert_eq!(t.kv, 32.0);
    }

    #[test]
    fn the_recurrent_state_is_stepped_a_token_at_the_scan_rate() {
        let (e, r) = documented();
        // 8 tokens · 3 elements at 2 a ns.
        assert_eq!(e.terms(&r).recurrent, 12.0);
        // A model with no state reads no scan rate.
        let none = FireEstimate {
            state_elements_per_row: 0.0,
            scan_elements_per_ns: 0.0,
            ..e
        };
        assert_eq!(none.terms(&r).recurrent, 0.0);
    }

    #[test]
    fn the_elementwise_bytes_are_read_at_the_bandwidth_a_token() {
        let (e, r) = documented();
        // 8 tokens · 2 bytes at 4 a ns.
        assert_eq!(e.terms(&r).elementwise, 4.0);
    }

    #[test]
    fn the_estimate_is_its_documented_form() {
        let (e, r) = documented();
        // 5·3 + max(400/4, 32·10/2.5) + 4·6/1 + 8·16·0.5/0.25 + 16·8/4
        //     + 8·3/2 + 8·2/4
        let want = 15.0 + 128.0 + 24.0 + 256.0 + 32.0 + 12.0 + 4.0;
        assert!((e.ns(&r) - want).abs() < 1e-9, "{}", e.ns(&r));
        assert_eq!(e.ns(&r), e.terms(&r).total());
    }
}
