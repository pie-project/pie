//! The top rung a load lays its fires up to (the fire-top rule): of the feasible
//! rungs, the one whose fire turns the most of its time into useful work
//! for the share of the live range its span takes:
//!
//! `r_top = argmax_r eff(r) · (1 − span(r) / M)`, `eff(r) = r / (r + r*)`,
//!
//! `r*` the roofline crossover, the rows at which a fire's weight read
//! takes as long as its matmuls' compute; `span(r)` the activation span the
//! rung reserves, `M` the live range kv and activation share at load. A fire
//! of `r` rows reads its weights once and computes `r` rows over them, so
//! `r / (r + r*)` is the share of its weight time that is compute. Over
//! several weight groups it is the groups' compute over their compute and
//! read together ([`FireEstimate::eff`]), each group's compute at the rows
//! its kernel sees: a dense plane the fire's rows, a routed bank the rows
//! each expert is handed (`r · k / E`), whose weights it reads only for
//! those. No constant weighs either term, and an exact tie goes to the
//! narrower rung.
//!
//! Read and compute are the estimate's alone ([`FireEstimate::terms`]):
//! work over the device's rates (the profile's `[rates]`), counted from what
//! the load knows of the model and the compiled plan. No measured fire time
//! is read at load; measured fire times only test the estimate (`pie tune
//! --check-estimate`), their crossover fitted from them
//! ([`measured_effs`]).

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
    /// The most lanes one fire carries: the engine's own cap (a wider wave
    /// the runtime fires as fires of this many), stated by each engine.
    pub fire_lanes: u32,
}

/// The fire a rung is weighed as, and the one a check of the estimate fires
/// for it: `lanes` lanes of `tokens_per_lane` new tokens each (`rows` in
/// all), each row attending `keys_per_lane` keys, `sampled` rows read out.
/// Decode-shaped (a rung no wider than its lanes, nor than the lanes one
/// fire carries, `RungFacts::fire_lanes`): a lane a token, each on a fresh
/// slot, attending its one key, every row read out. A wider decode wave is
/// fires of the narrower rungs, so a rung past either is prefill-shaped.
/// Prefill-shaped: one prompt of the rung's tokens on a fresh slot,
/// attending the rung's keys, its last row read out. The estimate's terms
/// are computed from it and every backend's check builds its fire from it,
/// so the two cannot weigh different fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckFire {
    pub rows: u32,
    pub lanes: u32,
    pub tokens_per_lane: u32,
    pub keys_per_lane: u32,
    pub sampled: u32,
}

impl RungFacts {
    /// The fire this rung is weighed and checked as ([`CheckFire`]).
    #[must_use]
    pub fn check_fire(&self) -> CheckFire {
        if self.decode_shaped() {
            CheckFire {
                rows: self.tokens,
                lanes: self.tokens,
                tokens_per_lane: 1,
                keys_per_lane: 1,
                sampled: self.tokens,
            }
        } else {
            CheckFire {
                rows: self.tokens,
                lanes: 1,
                tokens_per_lane: self.tokens,
                keys_per_lane: self.keys,
                sampled: 1,
            }
        }
    }

    /// The rows a fire at this rung samples, as the estimate weighs it: a
    /// rung no wider than its lanes is decode-shaped (a lane a token, each
    /// sampled), a wider one prefill-shaped (one prompt, one row sampled).
    #[must_use]
    pub fn sampled(&self) -> u32 {
        self.check_fire().sampled
    }

    /// Whether the estimate weighs this rung decode-shaped: no wider than
    /// its lanes, nor than the lanes one fire carries.
    #[must_use]
    pub fn decode_shaped(&self) -> bool {
        self.tokens <= self.lanes.min(self.fire_lanes)
    }
}

/// What a fire costs, as work over the device's roofline (the one shared
/// formula every engine weighs its top rung by, `crate::roofline`):
///
/// `t(r) = Σ_dispatches (launch_ns + work_d(r))`: each of the fire's
/// dispatches takes its own work and a launch, and a dispatch of the
/// compiled plan the model's counts name no work for takes a launch. A
/// dispatch's work is the longer of its compute and its memory traffic: a
/// matmul plane's flops, its rows padded to its kernel's tile, over the
/// measured matmul rate at its rows and width
/// (`crate::roofline::Constants::gemm_compute_ns`), or where the backend's
/// few-rows kernel serves it at `few_row_tflops` under the wave rule over
/// that kernel's grid (`crate::roofline::Constants::matmul_compute_ns`),
/// or its read; an attention layer's flops over the measured attention rate
/// at its keys, or its kv read; a scan layer's state steps; a per-row op's
/// bytes over the per-row ops' measured bandwidth. `launch_ns` is what a
/// dependent dispatch adds to its work as the backend submits a fire
/// (`crate::rates::launch_from`). The terms, summed over the fire (the
/// work, without the launches):
///
/// `work(r) = Σ_planes max(read, 2 pad(tokens(r)) n k / gemm(tokens(r), √(n k)))
///       + max(head_read, 2 sampled(r) vocab hidden / gemm(sampled(r), √(vocab hidden)))
///       + max(tokens(r) · keys(r) · attn_flops_per_row_key / attn(keys(r)),
///             keys(r) · kv_bytes_per_key / bytes_per_ns)
///       + tokens(r) · state_elements_per_row / scan_rate
///       + tokens(r) · elementwise_bytes_per_row / elementwise_bytes_per_ns`
///
/// A routed bank's planes read the experts a fire's rows touch
/// (`dense + expert_bytes · (1 − (1 − k/E)^tokens)`). The output head runs once a sampled row
/// ([`RungFacts::sampled`]): a rung no wider than its lanes is weighed
/// decode-shaped (a lane a token, every row sampled), a wider rung
/// prefill-shaped (one prompt, its last row sampled). A decode-shaped
/// rung's lanes attend one key each (a fresh slot's own): its score is for
/// picking the top, not a prediction of a long-context decode's time.
/// Sources: the constants are the profile's `[rates]` (measured by `pie tune
/// --rates` at `crate::rates`' shapes); the byte, flop, attention, state and
/// elementwise counts are the trace's ([`super::Counts`]); `launches` are
/// the backend's, from its compiled plan.
#[derive(Debug, Clone, PartialEq)]
pub struct FireEstimate {
    /// The device's rates.
    pub rates: crate::roofline::Constants,
    /// The model's widest head.
    pub attn_head_dim: u32,
    /// The weight bytes every fire reads.
    pub dense_bytes: u64,
    /// The bytes of every routed expert, all banks together.
    pub expert_bytes: u64,
    /// `(k, E)`: the experts a row picks, of how many; `None` without a
    /// routed bank.
    pub experts: Option<(u32, u32)>,
    /// The matmuls a row takes, by shape.
    pub gemm_shapes: Vec<GemmShape>,
    /// The output head's bytes, read every fire.
    pub head_bytes: u64,
    /// The output head's flops a lane: two a head element.
    pub readout_flops: f64,
    /// The output head's depth (the model's hidden width).
    pub readout_depth: u32,
    /// The flops a row takes a key it attends, all layers together.
    pub attn_flops_per_row_key: f64,
    /// The kv bytes a key read takes, all layers together.
    pub kv_bytes_per_key: f64,
    /// The recurrent state elements a row steps, all layers together.
    pub state_elements_per_row: f64,
    /// The dispatches the attention takes (one a layer) and the scan (its
    /// conv and its scan a layer).
    pub attn_dispatches: u32,
    pub scan_dispatches: u32,
    /// The activation bytes a row's per-row ops read and write, one an op
    /// (each a dispatch).
    pub elementwise_ops: Vec<u64>,
}

impl FireEstimate {
    /// The weight bytes a fire of `tokens` reads.
    #[must_use]
    pub fn weight_bytes(&self, tokens: u32) -> f64 {
        self.dense_bytes as f64 + self.expert_bytes as f64 * self.touched(tokens)
    }

    /// Each shape's matmul compute at `rung`, in ns: `(n, k, ns)`, the
    /// rung's rows' flops at that shape over the matmul rate at those rows. Their sum is [`Terms::flops`].
    #[must_use]
    pub fn matmuls(&self, rung: &RungFacts) -> Vec<(u32, u32, f64)> {
        let rows = rung.tokens;
        self.gemm_shapes
            .iter()
            .map(|s| {
                // The rows the kernel computes: a dense plane's padded to
                // its tile; a routed bank's flops a row are its picks'.
                let computed = if s.routed {
                    rows
                } else {
                    self.rates.padded_rows(rows, s.n, s.k)
                };
                let flops = f64::from(computed) * s.flops_per_row as f64;
                // A dense plane the backend's few-rows kernel serves at
                // `few_row_tflops` over its grid; a routed bank's on the
                // grouped kernel the curve stands for.
                let ns = if s.routed {
                    if flops > 0.0 {
                        flops / self.rates.gemm_rate(self.expert_rows(rows), s.n, s.k)
                    } else {
                        0.0
                    }
                } else {
                    self.rates.matmul_compute_ns(rows, s.n, s.k, flops)
                };
                (s.n, s.k, ns)
            })
            .collect()
    }

    /// The estimate's terms at `rung`, in ns: the one place its form lives.
    #[must_use]
    pub fn terms(&self, rung: &RungFacts) -> Terms {
        // The fire the rung is weighed as: its rows, the keys a row
        // attends, and the keys the fire reads (each lane its own).
        let fire = rung.check_fire();
        let tokens = f64::from(fire.rows);
        let keys = f64::from(fire.keys_per_lane);
        let keys_read = if fire.lanes > 1 {
            f64::from(fire.lanes) * keys
        } else {
            keys
        };
        let c = &self.rates;
        // Work over a rate, nothing where there is no work.
        let over = |work: f64, rate: f64| if work > 0.0 { work / rate } else { 0.0 };
        // `count` dispatches sharing `work`, each its share and a launch.
        let floored = |work: f64, count: u32| work + f64::from(count.max(1)) * c.launch_ns;
        let (mut dispatched, mut dispatches) = (0.0, 0u64);
        // The matmuls: each plane a dispatch, its read or its compute.
        let touched = self.touched(rung.tokens);
        let (mut weights, mut flops) = (0.0, 0.0);
        for (s, &(_, _, mm)) in self.gemm_shapes.iter().zip(&self.matmuls(rung)) {
            let bytes = s.bytes as f64 * if s.routed { touched } else { 1.0 };
            let read = over(bytes, c.bytes_per_ns);
            weights += read;
            flops += mm;
            dispatched += floored(read.max(mm), s.planes);
            dispatches += u64::from(s.planes.max(1));
        }
        // The readout: one dispatch, the head's read or its compute.
        // The head's bytes are the readout's alone, not the trunk's read.
        let head_read = over(self.head_bytes as f64, c.bytes_per_ns);
        let readout = c.matmul_compute_ns(
            fire.sampled,
            self.readout_flops as u32 / (2 * self.readout_depth.max(1)),
            self.readout_depth,
            f64::from(fire.sampled) * self.readout_flops,
        );
        if self.readout_flops > 0.0 {
            dispatched += floored(readout.max(head_read), 1);
            dispatches += 1;
        }
        // The attention at the rate its kernel reaches at a row's keys.
        let attention = over(
            tokens * keys * self.attn_flops_per_row_key,
            c.attn_rate(fire.keys_per_lane),
        );
        let kv = over(keys_read * self.kv_bytes_per_key, c.bytes_per_ns);
        if self.attn_flops_per_row_key > 0.0 {
            dispatched += floored(attention.max(kv), self.attn_dispatches);
            dispatches += u64::from(self.attn_dispatches.max(1));
        }
        let recurrent = over(tokens * self.state_elements_per_row, c.scan_elements_per_ns);
        if self.state_elements_per_row > 0.0 {
            dispatched += floored(recurrent, self.scan_dispatches);
            dispatches += u64::from(self.scan_dispatches.max(1));
        }
        let mut elementwise = 0.0;
        for &bytes in &self.elementwise_ops {
            let work = over(tokens * bytes as f64, c.elementwise_bytes_per_ns);
            elementwise += work;
            dispatched += floored(work, 1);
            dispatches += 1;
        }
        // The plan's other dispatches: a launch each.
        dispatched += u64::from(rung.launches).saturating_sub(dispatches) as f64 * c.launch_ns;
        Terms {
            launch: f64::from(rung.launches) * c.launch_ns,
            weights,
            flops,
            readout,
            attention,
            kv,
            recurrent,
            elementwise,
            head_read,
            dispatched,
        }
    }

    /// The rows each routed expert is handed in a fire of `rows`: the
    /// fire's rows times the `k` of `E` experts a row picks, shared by the
    /// `E` (at least one).
    #[must_use]
    pub fn expert_rows(&self, rows: u32) -> u32 {
        match self.experts {
            Some((k, e)) if e > 0 => {
                let handed = u64::from(rows) * u64::from(k.min(e));
                u32::try_from(handed.div_ceil(u64::from(e))).unwrap_or(u32::MAX).max(1)
            }
            _ => rows,
        }
    }

    /// The fire-top rule's efficiency of a fire at `rung`, `r / (r + r*)` over its
    /// weight groups: each group's compute at the device's full matmul rate
    /// at its shape (the top of the curve), over that compute and the
    /// group's weight read together. A group of `r*` rows reads its weights
    /// in the time it computes them at that rate (the roofline crossover);
    /// a fire's compute share over its groups is `r / (r + r*)` with `r*`
    /// their reads over their compute a row. A routed bank's weights are
    /// read as far as the fire's rows touch them and computed only for the
    /// `k` of `E` experts each row picks: its crossover is at the rows each
    /// expert is handed ([`Self::expert_rows`]), `E / k` times its fire's.
    /// A fire with neither is wholly efficient.
    #[must_use]
    pub fn eff(&self, rung: &RungFacts) -> f64 {
        let rows = f64::from(rung.tokens);
        let touched = self.touched(rung.tokens);
        let (mut compute, mut read) = (0.0, 0.0);
        for s in &self.gemm_shapes {
            let full = self.rates.gemm_rate(u32::MAX, s.n, s.k);
            if full > 0.0 && full.is_finite() {
                compute += rows * s.flops_per_row as f64 / full;
            }
            let bytes = s.bytes as f64 * if s.routed { touched } else { 1.0 };
            read += bytes / self.rates.bytes_per_ns;
        }
        let whole = compute + read;
        if whole > 0.0 { compute / whole } else { 1.0 }
    }

    /// The share of the experts a fire of `tokens` touches.
    fn touched(&self, tokens: u32) -> f64 {
        match self.experts {
            Some((k, e)) if e > 0 => {
                1.0 - (1.0 - f64::from(k.min(e)) / f64::from(e)).powf(f64::from(tokens))
            }
            _ => 0.0,
        }
    }

    /// The estimated time of a fire at `rung`.
    #[must_use]
    pub fn ns(&self, rung: &RungFacts) -> f64 {
        self.terms(rung).total()
    }
}

/// The per-shape table a check of the estimate prints: at each of `rungs`
/// (measured at `measured` ns, one a rung), each dense shape's matmul ms at
/// the rung's padded rows and its share of the weight read (its flops'
/// share of the dense ones), then the sums, read + matmul and max(read,
/// matmul) beside the measured fire: whether a fire runs the read and the
/// compute one after the other or together.
#[must_use]
pub fn shape_table(estimate: &FireEstimate, rungs: &[RungFacts], measured: &[f64]) -> String {
    let mut out = String::from(
        "  per (n x k) shape: matmul ms at the rung's padded rows / its read ms; then the sums, \
         read + matmul against max(read, matmul) and the measured fire\n",
    );
    let touched = estimate.touched(0);
    for (r, &meas) in rungs.iter().zip(measured) {
        let t = estimate.terms(r);
        let touched = if estimate.experts.is_some() {
            estimate.touched(r.tokens)
        } else {
            touched
        };
        let mut line = format!("  {:>6}", r.tokens);
        for (&(n, k, ns), s) in estimate.matmuls(r).iter().zip(&estimate.gemm_shapes) {
            let bytes = s.bytes as f64 * if s.routed { touched } else { 1.0 };
            let read = bytes / estimate.rates.bytes_per_ns;
            line.push_str(&format!(" | {n}x{k} {:.2}/{:.2}", ns / 1e6, read / 1e6));
        }
        line.push_str(&format!(
            " | sum {:.2}/{:.2}: read + matmul {:.2}, max {:.2}, measured {:.2}\n",
            t.flops / 1e6,
            t.weights / 1e6,
            (t.flops + t.weights) / 1e6,
            t.flops.max(t.weights) / 1e6,
            meas / 1e6
        ));
        out.push_str(&line);
    }
    out
}

/// One matmul shape of a model: `planes` planes (a dispatch each) of `n`
/// columns at depth `k`, `flops_per_row` a row all together (two a weight
/// element; for a routed bank, two a routed element of the `k` of `E`
/// experts a row picks), and `bytes` on the device (a routed bank's read as
/// far as a fire's rows touch it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmShape {
    pub n: u32,
    pub k: u32,
    pub flops_per_row: u64,
    pub planes: u32,
    pub bytes: u64,
    pub routed: bool,
}

impl GemmShape {
    /// One dense plane of `n x k`, `bytes` on the device.
    #[must_use]
    pub fn dense(n: u32, k: u32, flops_per_row: u64, bytes: u64) -> GemmShape {
        GemmShape {
            n,
            k,
            flops_per_row,
            planes: 1,
            bytes,
            routed: false,
        }
    }
}

/// The estimate's terms of one fire, in ns ([`FireEstimate::terms`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Terms {
    /// `launches · launch_ns`.
    pub launch: f64,
    /// The trunk's weight bytes the fire reads (the head's aside), over the
    /// bandwidth.
    pub weights: f64,
    /// The padded rows' matmul flops, each shape over the rate at those
    /// rows and its depth.
    pub flops: f64,
    /// The output head's flops over the rows sampled, over the rate there.
    pub readout: f64,
    /// The attention's flops, over the attention rate.
    pub attention: f64,
    /// The kv bytes the keys read, over the bandwidth.
    pub kv: f64,
    /// The recurrent state the rows step, over the scan rate.
    pub recurrent: f64,
    /// The norms' and elementwise ops' activation bytes, over the bandwidth.
    pub elementwise: f64,
    /// The output head's bytes, over the bandwidth: the readout's read.
    pub head_read: f64,
    /// The fire's time: each dispatch its work and a launch, summed.
    pub dispatched: f64,
}

impl Terms {
    /// The fire's work: the larger of the weights' read and the matmuls'
    /// compute, then each other term in turn.
    #[must_use]
    pub fn work(&self) -> f64 {
        self.weights.max(self.flops)
            + self.readout.max(self.head_read)
            + self.attention.max(self.kv)
            + self.recurrent
            + self.elementwise
    }

    /// The fire's time: each dispatch its work and a launch, summed.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.dispatched
    }

    /// What the launches add to the fire's work: one a dispatch.
    #[must_use]
    pub fn floor(&self) -> f64 {
        (self.dispatched - self.work()).max(0.0)
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
            // A rung whose plan never compiled has no span to state.
            if *span == 0 {
                write!(f, "{sep}{tokens} tokens: {why}")?;
            } else {
                write!(f, "{sep}{tokens} tokens (span {} MiB): {why}", span >> 20)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for NoRung {}

/// The top rung of `rungs` (the module's rule), over a live range of
/// `live_capacity_bytes`, each rung's fire timed by `estimate`.
pub fn top_rung(
    rungs: &[RungFacts],
    live_capacity_bytes: u64,
    estimate: &FireEstimate,
) -> Result<TopRung, NoRung> {
    let times: Vec<f64> = rungs.iter().map(|r| estimate.ns(r)).collect();
    // Each rung as the fire-top rule weighs it: its estimate, and why it is out
    // where it is (the device's memory, a stated bound on one submission).
    for (rung, &ns) in rungs.iter().zip(&times) {
        tracing::debug!(
            tokens = rung.tokens,
            lanes = rung.lanes,
            span_mib = rung.span_bytes >> 20,
            estimate_ms = ns / 1e6,
            tokens_per_ms = f64::from(rung.tokens) / (ns / 1e6).max(f64::MIN_POSITIVE),
            infeasible = rung.infeasible.unwrap_or("-"),
            "top rung: a rung weighed"
        );
    }
    let effs: Vec<f64> = rungs.iter().map(|r| estimate.eff(r)).collect();
    let feasible: Vec<(&RungFacts, f64)> = rungs
        .iter()
        .zip(effs.iter().copied())
        .filter(|(r, _)| r.infeasible.is_none())
        .collect();
    for (&(rung, _), (eff, live_share, score)) in
        feasible.iter().zip(scores(&feasible, live_capacity_bytes))
    {
        tracing::debug!(
            tokens = rung.tokens,
            eff,
            live_share,
            score,
            "top rung: a feasible rung scored"
        );
    }
    pick(rungs, &effs, live_capacity_bytes)
}

/// The fire top: the fire-top rule's top rung ([`top_rung`]) over the rungs a
/// prompt can hold (no wider than `context`) of every rung the engine's
/// lattice offers, each weighed as its check fire shapes it
/// ([`RungFacts::check_fire`]: decode-shaped no wider than its lanes,
/// prefill-shaped past them), by efficiency times live capacity. One
/// function with the check of the estimate's pick (`super::check`), so a
/// load's bound and the check cannot disagree: a load offers its rungs to
/// it, and none past it can win. The context and memory (a rung
/// infeasible) are its only bounds.
pub fn fire_top(
    rungs: &[RungFacts],
    live_capacity_bytes: u64,
    estimate: &FireEstimate,
    context: u32,
) -> Result<TopRung, NoRung> {
    let held: Vec<RungFacts> = rungs
        .iter()
        .filter(|r| r.tokens <= context)
        .copied()
        .collect();
    top_rung(&held, live_capacity_bytes, estimate)
}

/// The fire-top rule's argmax over `rungs` whose fires are `effs` efficient (one
/// a rung, [`FireEstimate::eff`] or [`measured_effs`]): what [`top_rung`]
/// weighs by the estimate, and what a check of the estimate weighs by
/// measured times.
pub fn pick(
    rungs: &[RungFacts],
    effs: &[f64],
    live_capacity_bytes: u64,
) -> Result<TopRung, NoRung> {
    let mut feasible: Vec<(&RungFacts, f64)> = rungs
        .iter()
        .zip(effs.iter().copied())
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

/// `(eff, live share, score)` of each of `feasible` (`(rung, eff)`).
fn scores(feasible: &[(&RungFacts, f64)], live: u64) -> Vec<(f64, f64, f64)> {
    feasible
        .iter()
        .map(|&(rung, eff)| {
            let live_share = share(rung.span_bytes, live);
            (eff, live_share, eff * (1.0 - live_share))
        })
        .collect()
}

/// The fire-top rule's efficiency of each of `rungs` whose fires measured `times`
/// (ns, one a rung; a rung at nothing unmeasured): `r / (r + r*)`, `r*` the
/// crossover of the line the measured fires fit (least squares, `t = a + b
/// r`: `a` what a fire takes whatever its rows, the weights' read among it,
/// and `b` a row's work), `r* = a / b`. Where the fit leaves no fixed part
/// (`a <= 0`) or no row's work (`b <= 0`), every rung is wholly efficient.
#[must_use]
pub fn measured_effs(rungs: &[RungFacts], times: &[f64]) -> Vec<f64> {
    let points: Vec<(f64, f64)> = rungs
        .iter()
        .zip(times)
        .filter(|&(_, &t)| t > 0.0)
        .map(|(r, &t)| (f64::from(r.tokens), t))
        .collect();
    let n = points.len() as f64;
    let (sx, sy) = points.iter().fold((0.0, 0.0), |(x, y), &(px, py)| (x + px, y + py));
    let (mx, my) = (sx / n.max(1.0), sy / n.max(1.0));
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(xy, xx), &(px, py)| {
        (xy + (px - mx) * (py - my), xx + (px - mx) * (px - mx))
    });
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let a = my - b * mx;
    let crossover = if a > 0.0 && b > 0.0 { a / b } else { 0.0 };
    rungs
        .iter()
        .map(|r| {
            let rows = f64::from(r.tokens);
            if rows + crossover > 0.0 { rows / (rows + crossover) } else { 1.0 }
        })
        .collect()
}

/// Each feasible rung's score under `effs` (one a rung), by tokens: what
/// a check of the estimate compares the two picks by.
#[must_use]
pub fn scored(rungs: &[RungFacts], effs: &[f64], live_capacity_bytes: u64) -> Vec<(u32, f64)> {
    let mut feasible: Vec<(&RungFacts, f64)> = rungs
        .iter()
        .zip(effs.iter().copied())
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
    fn rung(tokens: u32, span_mib: u64) -> RungFacts {
        RungFacts {
            tokens,
            lanes: tokens.min(256),
            keys: tokens,
            launches: 0,
            padded_rows: tokens,
            span_bytes: span_mib * MIB,
            infeasible: None,
            fire_lanes: 4096,
        }
    }

    fn constants(launch_ns: f64, bytes_per_ns: f64) -> crate::roofline::Constants {
        // Flat curves: two flops a ns at every row, half a flop a ns at
        // every key; the per-row ops at the bandwidth.
        crate::roofline::Constants {
            launch_ns,
            gemm_by_rows: vec![(1, vec![(1, 2.0)])],
            attn_by_keys: vec![(1, 0.5)],
            bytes_per_ns,
            elementwise_bytes_per_ns: bytes_per_ns,
            scan_elements_per_ns: 2.0,
            few_row_flops_per_ns: None,
            workgroups_per_core: None,
            row_tile: std::sync::Arc::new(|_, _, _| 1),
            few_rows: None,
            gemm_waves: None,
            gemm_full_by_rows: None,
        }
    }

    /// A fire of `fixed_ns` beside `per_token_ns` a token: weights read at
    /// a byte a ns, `2 · per_token_ns` flops a token at two a ns.
    fn estimate(fixed_ns: u64, per_token_ns: f64) -> FireEstimate {
        FireEstimate {
            rates: constants(0.0, 1.0),
            attn_head_dim: 0,
            dense_bytes: fixed_ns,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![GemmShape::dense(
                1,
                1,
                (2.0 * per_token_ns) as u64,
                fixed_ns,
            )],
            head_bytes: 0,
            readout_flops: 0.0,
            readout_depth: 1,
            attn_flops_per_row_key: 0.0,
            kv_bytes_per_key: 0.0,
            state_elements_per_row: 0.0,
            elementwise_ops: Vec::new(),
            attn_dispatches: 1,
            scan_dispatches: 1,
        }
    }

    #[test]
    fn the_argmax_picks_the_documented_rung() {
        // The weights read in 1000 ns, a row's matmul 1 ns: r* = 1000,
        // eff = r / (r + 1000); live 1024 MiB.
        // r=  64: eff .0602, share   8/1024, score .0597
        // r= 512: eff .3386, share  64/1024, score .3174
        // r=4096: eff .8038, share 128/1024, score .7033
        // r=8192: eff .8912, share 256/1024, score .6684
        // r=10240: eff .9110, share 400/1024, score .5551
        let e = estimate(1000, 1.0);
        let at = rung;
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
        assert!((eff - 4096.0 / 5096.0).abs() < 1e-12);
        assert!((live_share - 0.125).abs() < 1e-12);
    }

    #[test]
    fn an_infeasible_rung_is_never_chosen() {
        let e = estimate(1_000_000_000, 1.0);
        let mut rungs = [rung(1024, 16), rung(2048, 32), rung(4096, 64)];
        rungs[2].infeasible = Some("past one binding");
        assert_eq!(top_rung(&rungs, 1 << 40, &e).unwrap().tokens, 2048);
        rungs[1].infeasible = Some("past one binding");
        let top = top_rung(&rungs, 1 << 40, &e).unwrap();
        assert_eq!((top.tokens, top.why), (1024, TopWhy::OnlyFeasible));
    }

    #[test]
    fn an_exact_tie_goes_to_the_narrower_rung() {
        // Equal efficiency, equal span: equal scores.
        let rungs = [rung(4096, 64), rung(2048, 64)];
        let top = pick(&rungs, &[0.5, 0.5], 1024 * MIB).unwrap();
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
    }

}
