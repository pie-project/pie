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

/// What a fire costs, as work over the device's rates (the one shared
/// formula every engine weighs its top rung by):
///
/// `t(r) = Σ_dispatches (launch_ns + work_d(r))`: each of the fire's
/// dispatches takes its own work and a launch, and a dispatch of the
/// compiled plan the model's counts name no work for takes a launch. A
/// dispatch's work is its op's term over the dispatches the op takes (a
/// matmul plane's read or compute, whichever is longer; an attention
/// layer's flops and kv read; a scan layer's state steps; an elementwise
/// op's bytes). `launch_ns` is what a dependent dispatch adds to its work
/// as the backend submits a fire (`crate::rates::launch_from`): where the
/// backend overlaps its launches it is small and the sum is the work; where
/// it serializes them, each adds it. The terms, summed over the fire (the
/// work, without the launches):
///
/// `work(r) = max(weights(r) / bytes_per_ns,
///             Σ_(n, k) tokens(r) · flops_per_row(n, k) / flops(tokens(r), n, k))
///       + sampled(r) · readout_flops / flops(sampled(r), vocab, hidden)
///       + tokens(r) · keys(r) · attn_flops_per_row_key / attn(keys(r), d)
///       + keys(r) · kv_bytes_per_key / bytes_per_ns
///       + tokens(r) · state_elements_per_row / scan(tokens_per_lane(r))
///       + tokens(r) · elementwise_bytes_per_row / bytes_per_ns`
///
/// Where `weights(r) = dense_bytes + expert_bytes · (1 − (1 − k/E)^tokens)`:
/// a fire reads every dense weight once (the output head's at its readout), and
/// of a routed bank the experts its rows touch (each row picks `k` of `E`).
/// `flops(m, n, k)` is the device's matmul rate at `m` rows, `n` columns
/// and depth `k`: its measured grid read at `m` exactly (the grid's rows
/// are the rung lattice, each timed through the kernel serving runs, its
/// padding and tile in the rate) and interpolated in `log n` and `log k`
/// ([`profile::fire::gemm_at`]), and each of the model's dense
/// shapes ([`super::Counts::gemm_shapes`]) is read at its own columns and
/// depth: a narrow model's matmuls run below a wide one's at the same rows,
/// and a wide shallow leg apart from a narrow deep one. `attn(keys, d)` is the device's attention rate at a row's keys and
/// the model's head width `d`, its measured grid interpolated in `log keys`
/// and `log d` and clamped (a paged prefill streams long keys faster a flop
/// than short, and a wide head's tiles run at their own rate). The output head
/// runs once a lane (the readout), not once a row, so its flops are their
/// own term, over the rows the rung samples ([`RungFacts::sampled`]): a
/// rung no wider than its lanes is weighed decode-shaped (a lane a token,
/// every row sampled), and a wider rung prefill-shaped (one prompt, its
/// last row sampled), since only prompt rows reach past the lanes and
/// decode rides the narrow rungs. A mixed fire (decode lanes riding a
/// prompt's piece) reads those lanes' head beside the prompt's, which this
/// leaves out; a backend whose check shows that matters revisits it. A
/// decode-shaped rung's lanes attend one key each (a fresh slot's own), so
/// its attention counts `tokens` row-keys and its kv read `tokens` keys:
/// its score is for picking the top, not a prediction of a long-context
/// decode's time, which the narrow rungs serve whatever they score.
/// Sources: the rates are the profile's `[rates]` (measured by
/// `pie tune --rates` at `crate::rates`' lattice, through the backend's own
/// paths); the byte, flop, attention, state and elementwise counts are the
/// trace's ([`super::Counts`]); `launches` and `padded_rows` are the
/// backend's, from its compiled plan and the tile its matmul takes at the
/// rung's rows.
#[derive(Debug, Clone, PartialEq)]
pub struct FireEstimate {
    /// Dense bf16 matmul flops a nanosecond by rows, columns and depth:
    /// `(m, n, k, rate)` ([`profile::fire::gemm_grid`]).
    pub flops_grid: Vec<(u32, u32, u32, f64)>,
    /// Attention flops a nanosecond by the keys a row attends and the head
    /// width: `(keys, d, rate)`, a full grid; read only where the model
    /// attends.
    pub attn_grid: Vec<(u32, u32, f64)>,
    /// The model's head width, the attention grid's `d` it is read at.
    pub attn_head_dim: u32,
    /// Recurrent state elements a nanosecond; read only where the model
    /// carries recurrent state.
    pub scan_by_tokens: Vec<(u32, f64)>,
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
        let touched = match self.experts {
            Some((k, e)) if e > 0 => {
                1.0 - (1.0 - f64::from(k.min(e)) / f64::from(e)).powf(f64::from(tokens))
            }
            _ => 0.0,
        };
        self.dense_bytes as f64 + self.expert_bytes as f64 * touched
    }

    /// The matmul rate at `rows`, `n` columns and depth `k`.
    #[must_use]
    pub fn flops_per_ns(&self, rows: u32, n: u32, k: u32) -> f64 {
        profile::fire::gemm_at(&self.flops_grid, rows, n, k).unwrap_or(f64::INFINITY)
    }

    /// Each dense shape's matmul time at `rung`, in ns: `(n, k, ns)`, the
    /// rung's tokens' flops at that shape over the rate there. Their
    /// sum is [`Terms::flops`].
    #[must_use]
    pub fn matmuls(&self, rung: &RungFacts) -> Vec<(u32, u32, f64)> {
        // The rung's tokens: the grid was timed at them through the kernel
        // serving runs, its padding and tile in the rate.
        let rows = rung.tokens;
        self.gemm_shapes
            .iter()
            .map(|s| {
                let work = f64::from(rows) * s.flops_per_row as f64;
                let ns = if work > 0.0 {
                    work / self.flops_per_ns(rows, s.n, s.k)
                } else {
                    0.0
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
        // Work over a rate, nothing where there is no work.
        let over = |work: f64, rate: f64| if work > 0.0 { work / rate } else { 0.0 };
        // `count` dispatches sharing `work`, each its share and a launch.
        let floored = |work: f64, count: u32| work + f64::from(count.max(1)) * self.launch_ns;
        let (mut dispatched, mut dispatches) = (0.0, 0u64);
        // The matmuls: each plane a dispatch, its read or its compute.
        let touched = self.touched(rung.tokens);
        let (mut weights, mut flops) = (0.0, 0.0);
        for (s, &(_, _, mm)) in self.gemm_shapes.iter().zip(&self.matmuls(rung)) {
            let bytes = s.bytes as f64 * if s.routed { touched } else { 1.0 };
            let read = over(bytes, self.bytes_per_ns);
            weights += read;
            flops += mm;
            dispatched += floored(read.max(mm), s.planes);
            dispatches += u64::from(s.planes.max(1));
        }
        // The readout: one dispatch, the head's read or its compute.
        // The head's bytes are the readout's alone, not the trunk's read.
        let head_read = over(self.head_bytes as f64, self.bytes_per_ns);
        let readout = over(
            f64::from(fire.sampled) * self.readout_flops,
            self.flops_per_ns(fire.sampled, self.readout_columns(), self.readout_depth),
        );
        if self.readout_flops > 0.0 {
            dispatched += floored(readout.max(head_read), 1);
            dispatches += 1;
        }
        let attention = over(
            tokens * keys * self.attn_flops_per_row_key,
            // The rate at the keys a row attends: one at a decode lane, the
            // grid's narrowest point.
            profile::fire::flops_at(&self.attn_grid, keys as u32, self.attn_head_dim)
                .unwrap_or(f64::INFINITY),
        );
        let kv = over(keys_read * self.kv_bytes_per_key, self.bytes_per_ns);
        if self.attn_flops_per_row_key > 0.0 {
            dispatched += floored(attention + kv, self.attn_dispatches);
            dispatches += u64::from(self.attn_dispatches.max(1));
        }
        let recurrent = over(
            tokens * self.state_elements_per_row,
            // The rate at a lane's prompt (a decode lane's one token at
            // the curve's narrowest point).
            profile::fire::interpolate(&self.scan_by_tokens, fire.tokens_per_lane)
                .unwrap_or(f64::INFINITY),
        );
        if self.state_elements_per_row > 0.0 {
            dispatched += floored(recurrent, self.scan_dispatches);
            dispatches += u64::from(self.scan_dispatches.max(1));
        }
        let mut elementwise = 0.0;
        for &bytes in &self.elementwise_ops {
            let work = over(tokens * bytes as f64, self.bytes_per_ns);
            elementwise += work;
            dispatched += floored(work, 1);
            dispatches += 1;
        }
        // The plan's other dispatches: a launch each.
        dispatched += u64::from(rung.launches).saturating_sub(dispatches) as f64 * self.launch_ns;
        Terms {
            launch: f64::from(rung.launches) * self.launch_ns,
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

    /// The output head's columns (the vocabulary): its flops a row over two
    /// a depth.
    fn readout_columns(&self) -> u32 {
        let depth = f64::from(self.readout_depth.max(1));
        (self.readout_flops / (2.0 * depth))
            .round()
            .clamp(1.0, f64::from(u32::MAX)) as u32
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
            let read = bytes / estimate.bytes_per_ns;
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
            + self.attention
            + self.kv
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
            fire_lanes: 4096,
        }
    }

    /// A fire of `fixed_ns` beside `per_token_ns` a token: weights read at
    /// a byte a ns, one flop a ns.
    fn estimate(fixed_ns: u64, per_token_ns: f64) -> FireEstimate {
        FireEstimate {
            // The lattices up to 10240 and 16384 rows: every rung below.
            flops_grid: crate::rates::row_lattice(16384)
                .into_iter()
                .chain([10240])
                .map(|m| (m, 1, 1, 1.0))
                .collect(),
            attn_grid: Vec::new(),
            attn_head_dim: 0,
            scan_by_tokens: Vec::new(),
            bytes_per_ns: 1.0,
            launch_ns: 0.0,
            dense_bytes: fixed_ns,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![GemmShape::dense(1, 1, per_token_ns as u64, fixed_ns)],
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

    /// `fixed + r · per_token` with the read and the compute summed: the
    /// estimate's `max` of the two holds this where launches carry the
    /// fixed part.
    fn summed(fixed_ns: u64, per_token_ns: f64) -> (FireEstimate, impl Fn(u32, u64) -> RungFacts) {
        // The fixed part a prefill-shaped fire's one readout: one lane, so
        // every rung samples one row, at the grid's one flop a ns.
        let e = FireEstimate {
            readout_flops: fixed_ns as f64,
            dense_bytes: 0,
            ..estimate(0, per_token_ns)
        };
        (e, |tokens, span| RungFacts {
            lanes: 1,
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
    fn a_narrow_fire_pays_for_the_tile_it_under_fills_in_its_measured_rate() {
        // The grid is timed at each rung's own rows, so a 16-row fire that
        // computes a 64-row tile reads a quarter of 64's rate there: 16
        // tokens cost what 64 do, and the narrow rung does not win.
        let mut e = estimate(100, 10.0);
        for point in e.flops_grid.iter_mut().filter(|p| p.0 == 16) {
            point.3 = 0.25;
        }
        let rungs = [rung(16, 1), rung(64, 4)];
        let times: Vec<f64> = rungs.iter().map(|r| e.ns(r)).collect();
        assert_eq!(times, [640.0, 640.0]);
        let top = top_rung(&rungs, 1024 * MIB, &e).unwrap();
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

    /// Every term at once: 5 launches, 4 tokens a lane each (decode-shaped)
    /// attending 16 keys, the matmuls padded to 32 rows.
    fn documented() -> (FireEstimate, RungFacts) {
        let e = FireEstimate {
            // At depth 64: 1 a ns to 4 rows, then one more each doubling to 5
            // at 64 and past. At depth 256: twice that.
            flops_grid: crate::rates::row_lattice(8192)
                .into_iter()
                .flat_map(|m| {
                    let r = (f64::from(m).log2() - 1.0).clamp(1.0, 5.0);
                    [(m, 64, 64, r), (m, 256, 256, 2.0 * r)]
                })
                .collect(),
            // At width 8: 0.25 a ns at one key, 1 at 16 and past; at width
            // 32 twice that.
            attn_grid: vec![(1, 8, 0.25), (16, 8, 1.0), (1, 32, 0.5), (16, 32, 2.0)],
            attn_head_dim: 8,
            scan_by_tokens: vec![(1, 2.0), (16, 4.0)],
            bytes_per_ns: 4.0,
            launch_ns: 3.0,
            dense_bytes: 200,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![GemmShape::dense(64, 64, 10, 200)],
            head_bytes: 0,
            readout_flops: 6.0,
            readout_depth: 64,
            attn_flops_per_row_key: 0.5,
            kv_bytes_per_key: 8.0,
            state_elements_per_row: 3.0,
            elementwise_ops: vec![2],
            attn_dispatches: 1,
            scan_dispatches: 1,
        };
        let r = RungFacts {
            tokens: 4,
            lanes: 4,
            keys: 16,
            launches: 5,
            padded_rows: 32,
            span_bytes: 0,
            infeasible: None,
            fire_lanes: 4096,
        };
        (e, r)
    }

    #[test]
    fn a_rung_is_weighed_and_checked_as_one_fire() {
        let (_, r) = documented();
        // Decode-shaped: four lanes of a token each, a key each, all read out.
        assert_eq!(
            r.check_fire(),
            CheckFire {
                rows: 4,
                lanes: 4,
                tokens_per_lane: 1,
                keys_per_lane: 1,
                sampled: 4
            }
        );
        // Prefill-shaped: one prompt over the rung's keys, its last row.
        let prompt = RungFacts { tokens: 8, ..r };
        assert_eq!(
            prompt.check_fire(),
            CheckFire {
                rows: 8,
                lanes: 1,
                tokens_per_lane: 8,
                keys_per_lane: 16,
                sampled: 1
            }
        );
    }

    #[test]
    fn a_rung_past_the_lanes_one_fire_carries_is_a_prompt() {
        let (_, r) = documented();
        // 512 rows, but one fire carries 256 lanes: a 512-lane decode is two
        // fires of the 256 rung, so this rung is one prompt of 512.
        let wide = RungFacts {
            tokens: 512,
            lanes: 512,
            keys: 512,
            fire_lanes: 256,
            ..r
        };
        assert_eq!(
            wide.check_fire(),
            CheckFire {
                rows: 512,
                lanes: 1,
                tokens_per_lane: 512,
                keys_per_lane: 512,
                sampled: 1
            }
        );
        // 256 rows: 256 lanes of a token each.
        let at = RungFacts {
            tokens: 256,
            lanes: 256,
            ..wide
        };
        assert_eq!(
            at.check_fire(),
            CheckFire {
                rows: 256,
                lanes: 256,
                tokens_per_lane: 1,
                keys_per_lane: 1,
                sampled: 256
            }
        );
    }

    #[test]
    fn the_matmul_rate_is_the_grid_at_the_rung_tokens() {
        let (e, r) = documented();
        assert_eq!(e.flops_per_ns(4, 64, 64), 1.0);
        assert_eq!(e.flops_per_ns(16, 64, 64), 3.0);
        assert_eq!(e.flops_per_ns(4096, 64, 64), 5.0);
        // Halfway from 64 to 256 in log depth: halfway from 3 to 6 a ns.
        assert!((e.flops_per_ns(16, 128, 128) - 4.5).abs() < 1e-12);
        // Read at the rung's 4 tokens (not its 32 padded rows): 4 · 10 at 1
        // a ns, under the 200 / 4 read.
        let t = e.terms(&r);
        assert!((t.flops - 40.0).abs() < 1e-9);
        assert_eq!(t.weights, 50.0);
    }

    #[test]
    fn a_model_of_two_depths_reads_each_shape_at_its_own_depth() {
        let (e, r) = documented();
        let two = FireEstimate {
            gemm_shapes: vec![
                GemmShape::dense(64, 64, 10, 0),
                GemmShape::dense(256, 256, 10, 0),
            ],
            ..e
        };
        // 4 · 10 at 1 a ns (depth 64), and 4 · 10 at 2 a ns (depth 256).
        let shapes = two.matmuls(&r);
        assert_eq!(shapes.len(), 2);
        assert!((shapes[0].2 - 40.0).abs() < 1e-9);
        assert!((shapes[1].2 - 20.0).abs() < 1e-9);
        assert!((two.terms(&r).flops - 60.0).abs() < 1e-9, "the shapes' sum");
        // The check's table: each shape's matmul and its half of the 200 / 4
        // read, then read + matmul and max beside the measured.
        let table = shape_table(&two, &[r], &[200e6]);
        assert!(
            table.ends_with(
                "       4 | 64x64 0.00/0.00 | 256x256 0.00/0.00 | sum 0.00/0.00: read + matmul \
                 0.00, max 0.00, measured 200.00\n"
            ),
            "{table}"
        );
    }

    #[test]
    fn the_readout_runs_once_a_sampled_row_decode_shaped_or_prefill_shaped() {
        let (e, r) = documented();
        // At the lanes (max_lanes 4): decode-shaped, 4 rows sampled, 4 · 6
        // flops at 1 a ns (the curve at 4 rows).
        assert_eq!((r.tokens, r.lanes, r.sampled()), (4, 4, 4));
        assert_eq!(e.terms(&r).readout, 24.0);
        // At 16 lanes and 16 tokens: 16 · 6 at 3 a ns.
        let wide = RungFacts {
            tokens: 16,
            lanes: 16,
            ..r
        };
        assert!((e.terms(&wide).readout - 32.0).abs() < 1e-9);
        // At twice the lanes: prefill-shaped, one prompt, one row sampled
        // (6 flops at the curve's 1 a ns below 4 rows).
        let prompt = RungFacts { tokens: 8, ..r };
        assert_eq!(prompt.sampled(), 1);
        assert_eq!(e.terms(&prompt).readout, 6.0);
        let long = RungFacts { tokens: 64, ..r };
        assert_eq!(e.terms(&long).readout, 6.0);
    }

    #[test]
    fn attention_reads_its_own_rate_and_kv_the_bandwidth() {
        let (e, r) = documented();
        let t = e.terms(&r);
        // Decode-shaped: 4 lanes of one key each, 4 · 1 · 0.5 at 0.25 a ns,
        // and 4 keys' 8 bytes at 4 a ns.
        assert!(r.decode_shaped());
        assert_eq!(t.attention, 8.0);
        assert_eq!(t.kv, 8.0);
        // Prefill-shaped (one prompt of 8 rows over 16 keys): 8 · 16 · 0.5
        // at the curve's 1 a ns at 16 keys, and 16 keys' 8 bytes at 4 a ns.
        let prompt = RungFacts { tokens: 8, ..r };
        let t = e.terms(&prompt);
        assert_eq!(t.attention, 64.0);
        // 4 keys, halfway from 1 to 16 in log keys: 0.625 a ns.
        let short = RungFacts { keys: 4, ..prompt };
        assert!((e.terms(&short).attention - 8.0 * 4.0 * 0.5 / 0.625).abs() < 1e-9);
        // A model of 32-wide heads reads the grid at its width: 2 a ns at 16
        // keys, half the time; one of 16, halfway in log width, 1.5.
        let wide = FireEstimate {
            attn_head_dim: 32,
            ..e.clone()
        };
        assert_eq!(wide.terms(&prompt).attention, 32.0);
        let mid = FireEstimate {
            attn_head_dim: 16,
            ..e
        };
        assert!((mid.terms(&prompt).attention - 64.0 / 1.5).abs() < 1e-9);
        assert_eq!(t.kv, 32.0);
    }

    #[test]
    fn the_recurrent_state_is_stepped_a_token_at_the_scan_rate() {
        let (e, r) = documented();
        // 4 tokens · 3 elements at 2 a ns.
        assert_eq!(e.terms(&r).recurrent, 6.0);
        // A model with no state reads no scan rate.
        let none = FireEstimate {
            state_elements_per_row: 0.0,
            scan_by_tokens: Vec::new(),
            ..e
        };
        assert_eq!(none.terms(&r).recurrent, 0.0);
    }

    #[test]
    fn the_elementwise_bytes_are_read_at_the_bandwidth_a_token() {
        let (e, r) = documented();
        // 4 tokens · 2 bytes at 4 a ns.
        assert_eq!(e.terms(&r).elementwise, 2.0);
    }

    #[test]
    fn the_heads_bytes_are_the_readouts_alone() {
        let (e, r) = documented();
        let headed = FireEstimate {
            head_bytes: 400,
            ..e.clone()
        };
        // Decode-shaped: the head's 400 bytes read once, at the readout
        // (100 ns past its 24 of compute), not again in the trunk's 50.
        let t = headed.terms(&r);
        assert_eq!((t.weights, t.head_read), (50.0, 100.0));
        assert!((t.total() - (50.0 + 100.0 + 16.0 + 6.0 + 2.0 + 5.0 * 3.0)).abs() < 1e-9);
        assert!((t.work() - (50.0 + 100.0 + 16.0 + 6.0 + 2.0)).abs() < 1e-9);
        // Prefill-shaped (one row sampled), the trunk's compute past its
        // read: the head moves only the readout's dispatch.
        let prompt = RungFacts { tokens: 8, ..r };
        let (with, without) = (headed.terms(&prompt), e.terms(&prompt));
        assert_eq!(with.weights, without.weights);
        assert!(
            (with.total() - without.total() - (100.0 - without.readout)).abs() < 1e-9,
            "{with:?} {without:?}"
        );
    }

    #[test]
    fn each_dispatch_adds_a_launch_to_its_work() {
        let (e, r) = documented();
        // The documented fire's five dispatches: the matmul's read 50 (past
        // its 40 of compute), the readout's 24, the attention's 8 + 8, the
        // scan's 6 and the one elementwise op's 2, each and a 3 ns launch.
        let t = e.terms(&r);
        assert!((t.work() - 98.0).abs() < 1e-9);
        assert!((t.total() - (98.0 + 5.0 * 3.0)).abs() < 1e-9);
        assert!((t.floor() - 15.0).abs() < 1e-9);
        // The plan's dispatches the counts name no work for: a launch each.
        let more = RungFacts {
            launches: 1000,
            ..r
        };
        assert!((e.terms(&more).total() - (98.0 + 1000.0 * 3.0)).abs() < 1e-9);
    }

    #[test]
    fn a_chain_of_short_dispatches_adds_its_gaps() {
        let (e, r) = documented();
        // A hundred per-row ops of one byte a row: 1 ns of work each at 4
        // tokens, and each a 3 ns gap: the gaps add, they do not hide.
        let tiny = FireEstimate {
            elementwise_ops: vec![1; 100],
            ..e
        };
        let wide = RungFacts { launches: 104, ..r };
        let t = tiny.terms(&wide);
        assert!((t.elementwise - 100.0).abs() < 1e-9);
        assert!((t.total() - (t.work() + 104.0 * 3.0)).abs() < 1e-9);
    }

    #[test]
    fn a_backend_with_no_gap_is_its_work() {
        let (e, r) = documented();
        let overlapped = FireEstimate {
            launch_ns: 0.0,
            ..e
        };
        let t = overlapped.terms(&r);
        assert!((t.total() - t.work()).abs() < 1e-9);
        assert_eq!(t.floor(), 0.0);
    }

    #[test]
    fn a_one_token_fire_is_its_launches_and_its_little_work() {
        // The M5's rung 1: one row, every dispatch's work far under a launch.
        let (e, r) = documented();
        let quick = FireEstimate {
            launch_ns: 1000.0,
            ..e
        };
        let one = RungFacts {
            tokens: 1,
            lanes: 1,
            padded_rows: 1,
            launches: 40,
            ..r
        };
        let t = quick.terms(&one);
        assert!(t.work() < 100.0, "{t:?}");
        assert!((t.total() - (40.0 * 1000.0 + t.work())).abs() < 1e-9);
    }

    #[test]
    fn the_estimate_is_its_documented_form() {
        let (e, r) = documented();
        // Each of its five dispatches its work and a 3 ns launch: max(200/4,
        // 4·10/1), 4·6/1, 4·1·0.5/0.25 + 4·8/4, 4·3/2 and 4·2/4.
        let want = 50.0 + 24.0 + (8.0 + 8.0) + 6.0 + 2.0 + 5.0 * 3.0;
        assert!((e.ns(&r) - want).abs() < 1e-9, "{}", e.ns(&r));
        assert_eq!(e.ns(&r), e.terms(&r).total());
    }
}
