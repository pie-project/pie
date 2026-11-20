//! A serving step held to the bound the platform states on one GPU
//! submission ([`crate::rates::stated_submit_bound`]: KGSL's 2 s on
//! Android, none elsewhere). Past it the driver faults the submission and
//! the device is lost (an Adreno 840 on wgpu lost it on a 3843-token
//! prompt's prefill steps of ~480 rows at ~3 to 3.8k keys, each 1 to 2 s).
//!
//! Every step is its own submission on both slang backends (its own command
//! buffer and fence), so a frame's steps never share one: each is admitted
//! alone. A step is estimated by the load's own estimate
//! ([`engine::load::FireEstimate`], the device's measured `[rates]` over the
//! model's counts) at its rows and the keys its longest lane attends (its
//! attention against a longer prefix at the rate measured at those rows
//! against those keys, `[rates] attn_tflops_by_rows_keys`), times
//! the worst ratio of a frame's time to its estimate the device's rates step
//! measured, its clock's swing at least (`[rates] submit_worst_ratio`, saved
//! with the other rates), and as a split step its widest submission, a
//! layer's share of the dispatches with the readout's on the last:
//!
//! - a prompt step (a lane of more than one row) is kept whole only where its
//!   whole estimate is under the bound by at least one layer's: the layer is
//!   the split's own granularity, and a device warmer than any boot measured
//!   (an Adreno 840 ran ~1.25x slower warm) eats into a margin measured
//!   cool; else it is submitted a layer at a time
//!   (`Run::flush_between_layers`), which costs a native Vulkan step nothing
//!   measurable;
//! - a decode step (one row a lane, tiny) is kept whole where its estimate
//!   fits the bound;
//! - a step no layer of which fits is refused by name, naming the bound and
//!   the estimate. The load lowers the rows a prefill fire takes
//!   (`FireLimits::prefill_rows`) to the widest laid rung whose prompt step at
//!   the load's context is admitted ([`SubmitBound::prefill_rows`]), so a
//!   prompt is cut to steps that fit; a step refused past that is one no rung
//!   fits.
//!
//! No margin is chosen here: the bound is the platform's, the estimate and
//! the worst ratio are measurements. Where no bound is stated nothing is
//! estimated (the load keeps no [`SubmitBound`]).
//!
//! An estimate is read off curves timed on a loaded device; a device left
//! idle drops its clock (an L40S idles at 210 MHz), and a step admitted
//! against the loaded clock then runs past its estimate (two such frames
//! ran 26 and 44 ms against a simulated 25 ms bound). So a frame that
//! follows an idle device re-times it first ([`Probe`], [`Clock`]): the
//! smallest shape on the matmul curve ([`PROBE`]) fired alone and timed on
//! the device, its time over its curve time the scale every estimate is
//! read at ([`SubmitBound::scale`]) until a later probe. The device is
//! idle where its queue is empty when the frame's first step goes out
//! (every submission made on it landed, read without waiting:
//! `backend::Context::idle`, `Shell::retime`); a frame queued behind work
//! still running follows no idle and keeps the scale. The probe is
//! recorded at load where the backend keeps recordings, and counted apart
//! from the model's fires (`engine::live::Tally::Probed`).

use engine::load::{FireEstimate, RungFacts};
use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::backend::Buffer as _;
use crate::error::Fault;
use crate::handles::Handles;
use crate::serve::Serve;

/// What a step asks of the device, as the estimate weighs it: `rows` new
/// tokens over `lanes` lanes, the longest lane attending `keys` keys (the
/// slots it holds and its own rows); a prompt step where a lane carries more
/// than one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepRows {
    pub rows: u32,
    pub lanes: u32,
    pub keys: u32,
    pub prompt: bool,
}

impl StepRows {
    /// `step`'s rows, lanes and its longest lane's keys.
    #[must_use]
    pub fn of(step: &engine::fire::Step) -> StepRows {
        StepRows {
            rows: step.rows(),
            lanes: u32::try_from(step.lanes.len()).unwrap_or(u32::MAX),
            keys: step
                .lanes
                .iter()
                .map(|lane| lane.kv.held.saturating_add(lane.rows()))
                .max()
                .unwrap_or(0),
            prompt: step.lanes.iter().any(|lane| lane.rows() > 1),
        }
    }
}

/// How one step is submitted under the bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Admitted {
    /// In one submission, estimated at `ns` (times the worst ratio).
    Whole { ns: f64 },
    /// A layer at a time: `ns` whole, `per_submission` the most one
    /// submission is estimated at, `submissions` of them.
    Layers {
        ns: f64,
        per_submission: f64,
        submissions: u32,
    },
    /// Past the bound even a layer at a time.
    Refused { ns: f64, per_submission: f64 },
}

impl Admitted {
    /// Whether the step is submitted a layer at a time.
    #[must_use]
    pub fn split(&self) -> bool {
        matches!(self, Admitted::Layers { .. })
    }

    /// The step's estimate, ns, at the worst ratio.
    #[must_use]
    pub fn ns(&self) -> f64 {
        match *self {
            Admitted::Whole { ns } | Admitted::Layers { ns, .. } | Admitted::Refused { ns, .. } => ns,
        }
    }
}

/// The rule, over estimates already at the worst ratio: a step of `whole`
/// ns, `per_layer` its widest submission a layer at a time over `layers`.
/// A prompt step is whole only where `whole` is under `bound_ns` by at least
/// `per_layer` (the split's granularity), a decode step where `whole` fits;
/// else in layers where `per_layer` fits; else refused.
#[must_use]
pub fn admit_ns(whole: f64, per_layer: f64, layers: u32, bound_ns: f64, prompt: bool) -> Admitted {
    let headroom = if prompt && layers > 1 { per_layer } else { 0.0 };
    if whole + headroom <= bound_ns {
        Admitted::Whole { ns: whole }
    } else if layers > 1 && per_layer <= bound_ns {
        Admitted::Layers {
            ns: whole,
            per_submission: per_layer,
            submissions: layers,
        }
    } else {
        Admitted::Refused {
            ns: whole,
            per_submission: per_layer,
        }
    }
}

/// The submissions a frame of `steps` (each `(whole, per_layer, prompt)`,
/// ns at the worst ratio) is cut into: each step its own submission, or its
/// layers' where [`admit_ns`] splits it; each entry the step's index and its
/// estimate. `Err` names the first step no submission of which fits.
pub fn frame_submissions(
    steps: &[(f64, f64, bool)],
    layers: u32,
    bound_ns: f64,
) -> Result<Vec<(usize, f64)>, (usize, Admitted)> {
    let mut out = Vec::new();
    for (at, &(whole, per_layer, prompt)) in steps.iter().enumerate() {
        match admit_ns(whole, per_layer, layers, bound_ns, prompt) {
            Admitted::Whole { ns } => out.push((at, ns)),
            Admitted::Layers {
                per_submission,
                submissions,
                ..
            } => out.extend(std::iter::repeat_n((at, per_submission), submissions as usize)),
            refused @ Admitted::Refused { .. } => return Err((at, refused)),
        }
    }
    Ok(out)
}

/// The probe a device is re-timed by ([`Probe`]): the smallest shape on the
/// matmul curve, `rows x width x width` (the curve's narrowest rows at its
/// narrowest width, `engine::rates::GEMM_ROWS` and `GEMM_WIDTHS`).
pub const PROBE: (u32, u32, u32) = (
    engine::rates::GEMM_ROWS[0],
    engine::rates::GEMM_WIDTHS[0],
    engine::rates::GEMM_WIDTHS[0],
);

/// The device's clock as the last probe read it: the probe's time against
/// its time on the curve the estimates are read from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clock {
    /// The probe's time on the curve, ns: what a probe at the clock the
    /// curve was timed at takes.
    pub curve_ns: f64,
    /// The last probe's time over `curve_ns` (1 before any probe).
    pub ratio: f64,
}

impl Clock {
    #[must_use]
    pub fn new(curve_ns: f64) -> Clock {
        Clock {
            curve_ns,
            ratio: 1.0,
        }
    }

    /// A probe took `took_ns`: the ratio from here.
    pub fn probed(&mut self, took_ns: f64) {
        if self.curve_ns > 0.0 && took_ns > 0.0 {
            self.ratio = took_ns / self.curve_ns;
        }
    }

    /// The scale a serving estimate is read at: the ratio, never below 1
    /// (a step is never read faster than its curve measured).
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.ratio.max(1.0)
    }

    /// The next probe's estimate, ns: its curve time at the last ratio.
    #[must_use]
    pub fn probe_ns(&self) -> f64 {
        self.curve_ns * self.ratio
    }
}

/// The probe's matmul ([`PROBE`]) laid on the device with its own operands
/// (`engine::rates::operand_pattern`, as the curve's were), bindings and
/// pipelines: fired alone in a spanned frame, timed by the device's
/// timestamps where it writes them, else by the host's clock around its
/// submit and wait (longer than the device's: the scale only rises by it).
pub struct Probe<B: Serve> {
    handles: Handles<B>,
    pipelines: B::Pipelines,
    _held: Vec<B::Buffer>,
    act: Tensor,
    weight: Tensor,
    out: Tensor,
    tactic: kernels_slang::linear::gemm::Tactic,
    /// A device slowed by this factor, for a test: each fire runs the
    /// matmul this many times (1 serving).
    slowed: u32,
    /// Its work as the backend holds it: recorded at load where the backend
    /// keeps recordings ([`Serve::record_probe`]).
    probed: B::Probed,
}

impl<B: Serve> Probe<B> {
    /// The probe laid on `device` under `tactic` (the curve's at its rows):
    /// its operands, its work recorded where the backend keeps recordings,
    /// and fired once.
    pub fn new(
        device: &B::Context,
        tactic: kernels_slang::linear::gemm::Tactic,
    ) -> crate::error::Result<Self> {
        let (m, n, k) = PROBE;
        let handles = Handles::<B>::new();
        let mut held = Vec::new();
        let pattern = engine::rates::operand_pattern();
        let mut lay = |rows: u32, width: u32, fill: bool| -> crate::error::Result<Tensor> {
            let bytes = u64::from(rows) * u64::from(width) * 2;
            let mut buf = B::Buffer::zeroed(device, bytes, crate::books::owners::TUNING)?;
            let mut at = 0u64;
            while fill && at < bytes {
                let len = (pattern.len() as u64).min(bytes - at);
                buf.write(at, &pattern[..len as usize])?;
                at += len;
            }
            let handle = handles.bind(&buf, 0, buf.bytes())?;
            held.push(buf);
            Ok(Tensor::new(handle, rows, width, Dtype::Bf16))
        };
        let act = lay(m, k, true)?;
        let weight = lay(n, k, true)?;
        let out = lay(m, n, false)?;
        let pipelines = B::Pipelines::default();
        let probed = B::record_probe(device, &|frame| {
            Self::encode(device, frame, &pipelines, &handles, (tactic, act, weight, out), 1)
        })?;
        let probe = Probe {
            handles,
            pipelines,
            _held: held,
            act,
            weight,
            out,
            tactic,
            slowed: 1,
            probed,
        };
        // Fired once untimed: its operands touched, so no probe's time
        // holds that.
        probe.fire(device)?;
        Ok(probe)
    }

    /// The probe's matmul, `times` over, into `frame`.
    fn encode(
        device: &B::Context,
        frame: &B::Frame,
        pipelines: &B::Pipelines,
        handles: &Handles<B>,
        (tactic, act, weight, out): (kernels_slang::linear::gemm::Tactic, Tensor, Tensor, Tensor),
        times: u32,
    ) -> crate::error::Result<()> {
        let sink = B::sink(device, frame, pipelines, handles);
        for _ in 0..times.max(1) {
            kernels_slang::linear::gemm::matmul(&sink, tactic, act, weight, out).map_err(|e| {
                Fault::Device {
                    call: "submit bound: probe",
                    why: e.to_string(),
                }
            })?;
        }
        Ok(())
    }

    /// The probe fired alone, ns: the device's span where it writes
    /// timestamps, else the host's clock around the submit and the wait. A
    /// backend that keeps recordings replays the one laid at load.
    pub fn fire(&self, device: &B::Context) -> crate::error::Result<f64> {
        let started = std::time::Instant::now();
        let work = (self.tactic, self.act, self.weight, self.out);
        let span = B::replay_probe(device, &self.probed, &|frame| {
            Self::encode(device, frame, &self.pipelines, &self.handles, work, self.slowed)
        })?;
        let host = started.elapsed().as_nanos() as f64;
        Ok(span.map_or(host, |ns| ns as f64).max(1.0))
    }

    /// For a test: the device slowed by `factor` from here (each probe runs
    /// its matmul `factor` times), its work recorded again.
    #[doc(hidden)]
    pub fn slow(&mut self, device: &B::Context, factor: u32) -> crate::error::Result<()> {
        self.slowed = factor.max(1);
        let work = (self.tactic, self.act, self.weight, self.out);
        let (pipelines, handles, slowed) = (&self.pipelines, &self.handles, self.slowed);
        self.probed = B::record_probe(device, &|frame| {
            Self::encode(device, frame, pipelines, handles, work, slowed)
        })?;
        Ok(())
    }
}

/// A load's hold on its serving steps under a stated submit bound.
#[derive(Debug, Clone)]
pub struct SubmitBound {
    /// The platform's bound, ns.
    pub bound_ns: f64,
    /// The worst ratio of a frame's time to its estimate the device's rates
    /// step measured, at least 1 (`[rates] submit_worst_ratio`).
    pub worst: f64,
    /// The load's estimate (its rates with the backend's row tile laid).
    pub estimate: FireEstimate,
    /// The compiled plan's launches a fire.
    pub launches: u32,
    /// The model's layers: the submissions a split step takes.
    pub layers: u32,
    /// The most lanes one fire carries.
    pub fire_lanes: u32,
    /// The rungs the load laid, ascending.
    pub rungs: Vec<u32>,
    /// The attention of a prompt step's rows against a longer prefix, flops
    /// a ns: `(rows, [(keys, rate)])` (`[rates] attn_tflops_by_rows_keys`).
    pub prefixed: Vec<(u32, Vec<(u32, f64)>)>,
    /// The scale every estimate is read at: the last probe's
    /// ([`Clock::scale`]), 1 before any.
    pub scale: f64,
}

/// `curve` (ascending in keys) at `keys`: linear in `log keys` between its
/// points, its first point's below them, and past its last along its last
/// segment in `log rate` over `log keys` where that segment falls (a rate
/// falling with keys keeps falling past the keys measured), its last
/// point's where it rises (never read faster than measured).
#[must_use]
pub fn rate_at_keys(curve: &[(u32, f64)], keys: u32) -> Option<f64> {
    let &(x1, y1) = curve.last()?;
    if keys <= x1 || curve.len() < 2 {
        return profile::fire::interpolate(curve, f64::from(keys));
    }
    let (x0, y0) = curve[curve.len() - 2];
    if y1 >= y0 {
        return Some(y1);
    }
    let slope = (y1 / y0).ln() / (f64::from(x1) / f64::from(x0)).ln();
    Some(y1 * (f64::from(keys) / f64::from(x1)).powf(slope))
}

/// `by_rows` (ascending in rows, each a curve by keys) at `rows` against
/// `keys`: each curve read at the keys ([`rate_at_keys`]), linear in `log
/// rows` between the two rows bracketing it; past the widest, the widest's
/// (more rows fill more of the device, never read faster than measured);
/// below the narrowest, the narrowest's time (its rate scaled by the rows:
/// fewer rows against the same keys fill less of a device the narrowest
/// already underfills, and finish no sooner; on an RTX the rate fell in
/// proportion to the rows from 512 down to 128).
#[must_use]
pub fn rate_at(by_rows: &[(u32, Vec<(u32, f64)>)], rows: u32, keys: u32) -> Option<f64> {
    let at: Vec<(u32, f64)> = by_rows
        .iter()
        .filter_map(|(r, curve)| rate_at_keys(curve, keys).map(|rate| (*r, rate)))
        .collect();
    let &(narrowest, rate) = at.first()?;
    if rows < narrowest {
        return Some(rate * f64::from(rows.max(1)) / f64::from(narrowest));
    }
    profile::fire::interpolate(&at, f64::from(rows))
}

impl SubmitBound {
    /// The estimate of `step` at the worst ratio and the clock's scale, ns:
    /// `(whole, per layer)`,
    /// a layer's submission its share of the dispatches but the readout's,
    /// the last carrying the readout too.
    #[must_use]
    pub fn step_ns(&self, step: StepRows) -> (f64, f64) {
        let rung = RungFacts {
            tokens: step.rows.max(1),
            lanes: step.lanes.max(1),
            keys: step.keys.max(step.rows).max(1),
            launches: self.launches,
            padded_rows: step.rows.max(1),
            span_bytes: 0,
            infeasible: None,
            fire_lanes: self.fire_lanes.max(1),
        };
        let terms = self.estimate.terms(&rung);
        let mut whole = terms.dispatched;
        // A prompt step's rows against a longer prefix: its attention at the
        // rate measured at those rows against those keys, where slower than
        // the square curve reads it (a square prompt skips half its
        // rectangle causally, a prefixed step almost none).
        let fire = rung.check_fire();
        if !rung.decode_shaped() && fire.keys_per_lane > fire.rows {
            let flops = f64::from(fire.rows)
                * f64::from(fire.keys_per_lane)
                * self.estimate.attn_flops_per_row_key;
            if let Some(rate) = rate_at(&self.prefixed, fire.rows, fire.keys_per_lane)
                && flops > 0.0
                && rate > 0.0
            {
                let prefixed = flops / rate;
                if prefixed > terms.attention {
                    whole += prefixed.max(terms.kv) - terms.attention.max(terms.kv);
                }
            }
        }
        let head = terms.readout.max(terms.head_read) + self.estimate.rates.launch_ns;
        let per_layer = (whole - head).max(0.0) / f64::from(self.layers.max(1)) + head;
        let worst = self.worst.max(1.0) * self.scale.max(1.0);
        (whole * worst, per_layer * worst)
    }

    /// How `step` is submitted.
    #[must_use]
    pub fn admit(&self, step: StepRows) -> Admitted {
        let (whole, per_layer) = self.step_ns(step);
        admit_ns(whole, per_layer, self.layers, self.bound_ns, step.prompt)
    }

    /// The laid rung a step of `rows` runs under: the narrowest at or past
    /// them.
    #[must_use]
    pub fn rung_of(&self, rows: u32) -> u32 {
        self.rungs
            .iter()
            .copied()
            .find(|&rung| rung >= rows)
            .or(self.rungs.last().copied())
            .unwrap_or(rows)
    }

    /// The rows a prefill fire takes: the widest laid rung whose one prompt
    /// step of its rows at `context` keys is admitted, `None` where the
    /// widest is (no preference); the narrowest where none is (a prompt
    /// step past it is refused by name as it comes).
    #[must_use]
    pub fn prefill_rows(&self, context: u32) -> Option<u32> {
        let fits = |rung: u32| {
            !matches!(
                self.admit(StepRows {
                    rows: rung,
                    lanes: 1,
                    keys: context.max(rung),
                    prompt: rung > 1,
                }),
                Admitted::Refused { .. }
            )
        };
        let widest = *self.rungs.last()?;
        if fits(widest) {
            return None;
        }
        Some(
            self.rungs
                .iter()
                .rev()
                .copied()
                .find(|&rung| fits(rung))
                .unwrap_or(self.rungs[0]),
        )
    }

    /// Why `step`, admitted as `admitted`, is refused: the bound and the
    /// estimate, by name.
    #[must_use]
    pub fn refusal(&self, step: StepRows, admitted: &Admitted) -> String {
        let per = match *admitted {
            Admitted::Refused { per_submission, .. } | Admitted::Layers { per_submission, .. } => {
                per_submission
            }
            Admitted::Whole { ns } => ns,
        };
        format!(
            "a step of {} rows over {} lanes at {} keys is estimated at {:.0} ms whole and {:.0} \
             ms a layer at a time ({} layers, at the worst ratio {:.3} of a frame's time to its \
             estimate the rates step measured, `[rates] submit_worst_ratio`, and the clock's \
             scale {:.3} its last probe read), past the platform's stated bound of {:.0} ms on \
             one GPU submission (engine_gpu::rates::stated_submit_bound); the widest prefill \
             rung this load lays is {} rows",
            step.rows,
            step.lanes,
            step.keys,
            admitted.ns() / 1e6,
            per / 1e6,
            self.layers,
            self.worst,
            self.scale,
            self.bound_ns / 1e6,
            self.rungs.last().copied().unwrap_or(0),
        )
    }
}

/// The model's layers: the distinct layers its nodes state.
#[must_use]
pub fn layers(trace: &poem_ir::Trace) -> u32 {
    let mut seen: Vec<u32> = trace.nodes.iter().filter_map(|node| node.layer).collect();
    seen.sort_unstable();
    seen.dedup();
    u32::try_from(seen.len()).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: f64 = 1e6;
    const BOUND: f64 = 2000.0 * MS;

    /// A synthetic model and device: 28 layers of a 4096-wide plane, an
    /// attention, rates of 0.25 TFLOP/s matmul and attention (a phone's
    /// order), 100 GB/s.
    fn bound(worst: f64) -> SubmitBound {
        let counts = engine::load::Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![engine::load::GemmShape {
                n: 4096,
                k: 4096,
                flops_per_row: 2 * 4096 * 4096 * 28,
                planes: 28,
                bytes: 1 << 30,
                routed: false,
            }],
            head_bytes: 0,
            attn_flops_per_row_key: 4 * 4096 * 28,
            kv_bytes_per_key: 1 << 14,
            attn_head_dim: 128,
            readout_flops: 0,
            readout_depth: 4096,
            state_elements_per_row: 0,
            elementwise_ops: vec![0],
            attn_dispatches: 28,
            scan_dispatches: 0,
            width: 4096,
        };
        let registry = profile::Registry::new("test", &[], &[]);
        let profile = registry
            .load_user(
                "api = 1\n[rates]\nlaunch_ns = 10000\ngemm_tflops_by_rows = [[4096, [[1, 0.25], \
                 [4096, 0.25]]]]\nattn_tflops_by_keys = [[512, 0.25], [8192, 0.25]]\nhbm_gbps = \
                 100.0\nelementwise_gbps = 100.0\n",
                "test",
            )
            .unwrap();
        SubmitBound {
            bound_ns: BOUND,
            worst,
            estimate: engine::load::estimate(&counts, &profile.rates()).unwrap(),
            launches: 200,
            layers: 28,
            fire_lanes: 64,
            rungs: vec![64, 128, 256, 512],
            prefixed: Vec::new(),
            scale: 1.0,
        }
    }

    fn prompt(rows: u32, keys: u32) -> StepRows {
        StepRows {
            rows,
            lanes: 1,
            keys,
            prompt: true,
        }
    }

    /// `b`'s estimate of `step` as [`frame_submissions`] takes it.
    fn shaped(b: &SubmitBound, step: StepRows) -> (f64, f64, bool) {
        let (whole, per_layer) = b.step_ns(step);
        (whole, per_layer, step.prompt)
    }

    /// Two prompt steps of a frame, together past the bound and each under
    /// it by a layer: two submissions, each within the bound. A step past
    /// it whole goes a layer at a time.
    #[test]
    fn a_frame_past_the_bound_together_splits_into_submissions_within_it() {
        let b = bound(1.0);
        let steps = [shaped(&b, prompt(256, 1024)), shaped(&b, prompt(256, 1280))];
        assert!(steps[0].0 + steps[1].0 > BOUND, "{steps:?}");
        assert!(steps.iter().all(|s| s.0 + s.1 <= BOUND), "{steps:?}");
        let subs = frame_submissions(&steps, b.layers, BOUND).unwrap();
        assert_eq!(subs.len(), 2);
        assert!(subs.iter().all(|&(_, ns)| ns <= BOUND));
        let long = shaped(&b, prompt(512, 3800));
        assert!(long.0 > BOUND, "{long:?}");
        let subs = frame_submissions(&[steps[0], long], b.layers, BOUND).unwrap();
        assert_eq!(subs.len(), 1 + 28);
        assert!(subs.iter().all(|&(_, ns)| ns <= BOUND));
        assert!(b.admit(prompt(512, 3800)).split());
    }

    /// A prompt step whose whole estimate fits the bound but not by one
    /// layer's estimate is split; a decode step of the same estimate stays
    /// whole.
    #[test]
    fn a_prompt_step_stays_whole_only_a_layer_under_the_bound() {
        let b = bound(1.0);
        let step = prompt(256, 1024);
        let (whole, per_layer) = b.step_ns(step);
        // A bound the step fits, by less than a layer.
        let near = whole + per_layer / 2.0;
        let held = SubmitBound {
            bound_ns: near,
            ..b.clone()
        };
        assert!(held.admit(step).split(), "{:?}", held.admit(step));
        assert!(matches!(
            admit_ns(whole, per_layer, b.layers, near, false),
            Admitted::Whole { .. }
        ));
        // A layer under it: whole.
        let roomy = SubmitBound {
            bound_ns: whole + per_layer,
            ..b.clone()
        };
        assert!(matches!(roomy.admit(step), Admitted::Whole { .. }));
    }

    /// The margin saved with the rates moves a step past the bound that
    /// fits at a ratio of 1: Android's lost step, 1024 rows estimated at 83%
    /// of the bound at a ratio of 1 (it ran 1.25x that warm), is split at
    /// the 1.7 its rates step measured.
    #[test]
    fn the_measured_worst_ratio_is_the_margin() {
        let at_one = bound(1.0);
        let step = prompt(256, 1024);
        let (whole, per_layer) = at_one.step_ns(step);
        let scaled = SubmitBound {
            bound_ns: whole / 0.83,
            ..at_one.clone()
        };
        assert!(whole + per_layer <= scaled.bound_ns);
        assert!(matches!(scaled.admit(step), Admitted::Whole { .. }));
        let measured = SubmitBound {
            worst: 1.7,
            ..scaled
        };
        assert!(measured.admit(step).split());
    }

    /// A prompt step past the bound even a layer at a time takes a
    /// narrower rung: the load's prefill rows are the widest rung whose
    /// step at the context fits; a step no rung fits is refused by name.
    #[test]
    fn a_step_past_the_bound_takes_a_narrower_rung_or_is_refused_by_name() {
        let b = SubmitBound {
            layers: 1,
            ..bound(1.0)
        };
        let context = 4096;
        assert!(matches!(b.admit(prompt(512, context)), Admitted::Refused { .. }));
        let rows = b.prefill_rows(context).expect("a narrower rung");
        assert!(rows < 512);
        assert!(matches!(b.admit(prompt(rows, context)), Admitted::Whole { .. }));
        // Within the bound at the widest rung: no preference.
        let roomy = SubmitBound {
            bound_ns: 100.0 * BOUND,
            ..b.clone()
        };
        assert_eq!(roomy.prefill_rows(context), None);
        // At a context no rung fits, the narrowest, and its step refused.
        let far = 1 << 22;
        assert_eq!(b.prefill_rows(far), Some(64));
        let step = prompt(64, far);
        let admitted = b.admit(step);
        assert!(matches!(admitted, Admitted::Refused { .. }));
        let why = b.refusal(step, &admitted);
        assert!(why.contains("past the platform's stated bound of 2000 ms"), "{why}");
        assert!(why.contains("is estimated at"), "{why}");
        assert_eq!(
            frame_submissions(&[shaped(&b, step)], b.layers, BOUND).map_err(|(at, _)| at),
            Err(0)
        );
    }

    /// A probe after the device idled scales every step's estimate by its
    /// time over its curve time, never below 1: a step admitted whole at the
    /// loaded clock is split (or refused) once the clock dropped, and is
    /// whole again once a probe reads the clock back.
    #[test]
    fn a_probe_after_idle_scales_every_estimate() {
        let b = bound(1.0);
        let step = prompt(256, 1024);
        let (whole, per_layer) = b.step_ns(step);
        let held = SubmitBound {
            bound_ns: whole + per_layer,
            ..b.clone()
        };
        assert!(matches!(held.admit(step), Admitted::Whole { .. }));
        let mut clock = Clock::new(2.0 * MS);
        assert_eq!(clock.scale(), 1.0);
        assert_eq!(clock.probe_ns(), 2.0 * MS);
        // Idle: the probe ran 3x its curve time.
        clock.probed(6.0 * MS);
        assert_eq!(clock.scale(), 3.0);
        assert_eq!(clock.probe_ns(), 6.0 * MS);
        let dropped = SubmitBound {
            scale: clock.scale(),
            ..held.clone()
        };
        let (scaled, _) = dropped.step_ns(step);
        assert!((scaled - 3.0 * whole).abs() < 1e-6 * whole);
        assert!(!matches!(dropped.admit(step), Admitted::Whole { .. }));
        assert!(dropped.refusal(step, &dropped.admit(step)).contains("scale 3.000"));
        // Faster than its curve: read at the curve, never faster.
        clock.probed(1.0 * MS);
        assert_eq!(clock.scale(), 1.0);
        assert_eq!(clock.ratio, 0.5);
        let back = SubmitBound {
            scale: clock.scale(),
            ..held
        };
        assert!(matches!(back.admit(step), Admitted::Whole { .. }));
    }

    /// Past the keys measured, a falling attention rate keeps falling along
    /// its last segment; a rising one is read at its last point. Between
    /// rows the rate is read in log rows; outside them, at the nearest.
    #[test]
    fn the_prefixed_rate_extrapolates_its_last_segment_falling_only() {
        let falling = [(2048, 0.367e3), (4096, 0.341e3)];
        let at = rate_at_keys(&falling, 8192).unwrap();
        assert!(at < 0.341e3, "{at}");
        assert!((at - 0.341e3 * (0.341f64 / 0.367).powf(1.0)).abs() < 1e-6, "{at}");
        let rising = [(1024, 0.3e3), (4096, 0.4e3)];
        assert_eq!(rate_at_keys(&rising, 8192), Some(0.4e3));
        assert_eq!(rate_at_keys(&rising, 512), Some(0.3e3));
        let by_rows = vec![(256, vec![(4096, 0.2e3)]), (1024, vec![(4096, 0.4e3)])];
        let mid = rate_at(&by_rows, 512, 4096).unwrap();
        assert!((mid - 0.3e3).abs() < 1e-6, "{mid}");
        assert_eq!(rate_at(&by_rows, 2048, 4096), Some(0.4e3));
        // Below the narrowest rows, the narrowest's time.
        assert_eq!(rate_at(&by_rows, 128, 4096), Some(0.1e3));
    }

    /// A prompt step against a long prefix is estimated at the prefixed
    /// rate where it is slower than the square curve: Android's 512 x 3600
    /// step, read as fast as a square prompt, ran ~2x its estimate.
    #[test]
    fn a_prefixed_step_is_estimated_at_the_prefixed_rate() {
        let square = bound(1.0);
        let step = prompt(512, 3600);
        let (at_square, _) = square.step_ns(step);
        // Half the square curve's 0.25 TFLOP/s against a prefix.
        let prefixed = SubmitBound {
            prefixed: vec![(512, vec![(1024, 0.125e3), (8192, 0.125e3)])],
            ..square.clone()
        };
        let (at_prefixed, _) = prefixed.step_ns(step);
        let attention = 512.0 * 3600.0 * square.estimate.attn_flops_per_row_key / 0.25e3;
        assert!(
            (at_prefixed - at_square - attention).abs() < 1e-3 * attention,
            "{at_square} {at_prefixed} {attention}"
        );
        // A square prompt (no prefix) keeps the square curve's term.
        let own = prompt(512, 512);
        assert_eq!(prefixed.step_ns(own), square.step_ns(own));
        // A prefixed rate faster than the square curve's is not read.
        let faster = SubmitBound {
            prefixed: vec![(512, vec![(1024, 1e3), (8192, 1e3)])],
            ..square.clone()
        };
        assert_eq!(faster.step_ns(step), square.step_ns(step));
    }
}
