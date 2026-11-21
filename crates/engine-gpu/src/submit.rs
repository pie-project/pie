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
//!   cool; else it is submitted in runs of consecutive layers, as few as
//!   each fit the bound ([`pack_layers`], `Run::flush_between_layers`): each
//!   submission costs a launch's worth (an Adreno 840's 472-row step ran
//!   ~24% longer as 28 submissions than whole, where 5 would fit);
//! - a decode step (one row a lane, tiny) is kept whole where its estimate
//!   fits the bound;
//! - a step no layer of which fits has its one prompt lane cut by rows
//!   ([`row_pieces`]): its leading pieces fired first, each that lane alone
//!   over its next rows, the most a piece is admitted at
//!   ([`SubmitBound::rows_a_piece`]), each split by layer in turn; only a
//!   step one row of one layer of which is past the bound, or one that
//!   cannot be cut (a mask, media, guest inputs, a device-counted fold,
//!   more than one prompt lane), is refused by name, naming the bound and
//!   the estimate. A recurrent lane's state carries piece to piece. The load lowers the rows a prefill fire takes
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
//!
//! A probe reads the clock only after idle; a device that slows while busy
//! (an Adreno phone's GPU at ~96 C within ~10 s of load, its clock capped
//! 1300 to 826-1000 MHz) runs past estimates taken at full clock with no
//! idle to re-time it. A 472-row prompt step, right after a burst of
//! prefills, was admitted whole at the worst ratio 2.062 and the probe's
//! 1.005 and ran past 2 s: the device was lost. So every serving step's
//! device time is measured once it lands ([`DeviceTime`], from
//! [`FireTiming`], stated at load), read where its landing is already seen
//! (a fence waited, a work-done callback), never by a wait of its own, and
//! its ratio to its estimate at a ratio of 1 kept for one bound's span
//! ([`Measured`]): every estimate is read at the worst of them where it
//! passes the stored ratio, the worst ratio times the probe's scale
//! ([`SubmitBound::ratio_now`]), so a slowing device's next steps are split
//! before they reach the bound.

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
    /// In submissions of consecutive layers, `layers_a_piece` of them each
    /// (the last the rest): `ns` whole, `per_submission` the most one
    /// submission is estimated at, `submissions` of them.
    Layers {
        ns: f64,
        per_submission: f64,
        submissions: u32,
        layers_a_piece: u32,
    },
    /// Past the bound even a layer at a time.
    Refused { ns: f64, per_submission: f64 },
}

impl Admitted {
    /// Whether the step is submitted in more than one submission.
    #[must_use]
    pub fn split(&self) -> bool {
        matches!(self, Admitted::Layers { .. })
    }

    /// The layers each submission takes where the step is split.
    #[must_use]
    pub fn layers_a_piece(&self) -> Option<u32> {
        match *self {
            Admitted::Layers { layers_a_piece, .. } => Some(layers_a_piece),
            _ => None,
        }
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
/// else in submissions of consecutive layers where `per_layer` fits, as
/// few as each fit the bound ([`pack_layers`]); else refused.
#[must_use]
pub fn admit_ns(whole: f64, per_layer: f64, layers: u32, bound_ns: f64, prompt: bool) -> Admitted {
    let headroom = if prompt && layers > 1 { per_layer } else { 0.0 };
    if whole + headroom <= bound_ns {
        Admitted::Whole { ns: whole }
    } else if layers > 1 && per_layer <= bound_ns {
        let (layers_a_piece, per_submission) = pack_layers(whole, per_layer, layers, bound_ns);
        Admitted::Layers {
            ns: whole,
            per_submission,
            submissions: layers.div_ceil(layers_a_piece),
            layers_a_piece,
        }
    } else {
        Admitted::Refused {
            ns: whole,
            per_submission: per_layer,
        }
    }
}

/// The layers a submission of a split step takes, and its estimate: a step
/// of `whole` ns over `layers`, `per_layer` a layer's submission (its
/// share of the dispatches and the readout's, which the last carries),
/// packed from its first layer while a submission's estimate fits
/// `bound_ns`, into the fewest submissions, then evened over them (the
/// same count, each at most as wide). At least one layer a submission,
/// fewer than all.
#[must_use]
pub fn pack_layers(whole: f64, per_layer: f64, layers: u32, bound_ns: f64) -> (u32, f64) {
    if layers < 2 {
        return (1, per_layer);
    }
    // whole = layers x a layer's own + the readout's; per_layer = a
    // layer's own + the readout's.
    let own = ((whole - per_layer) / f64::from(layers - 1)).max(0.0);
    let readout = (per_layer - own).max(0.0);
    let fit = if own > 0.0 {
        (((bound_ns - readout) / own).floor() as u32).clamp(1, layers - 1)
    } else {
        layers - 1
    };
    let submissions = layers.div_ceil(fit);
    let each = layers.div_ceil(submissions);
    (each, f64::from(each) * own + readout)
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
    /// The ratios of the serving fires' measured device times to their
    /// estimates at a ratio of 1, landed within the last bound
    /// ([`Measured`]): an estimate is read at the worst of them where it
    /// passes the worst ratio times the scale ([`SubmitBound::ratio_now`]).
    pub measured: Measured,
    /// The GPU's clock table, where the platform states it
    /// ([`crate::clock`]): every estimate (timed at full clock) is read at
    /// its floor ([`SubmitBound::floor_factor`]), so a fire fits the bound
    /// even where the clock drops to its least inside it.
    pub clock: Option<crate::clock::ClockRange>,
}

/// A serving step's estimate at a ratio of 1, ns, what it was (its frame
/// and step, rows, lanes, and the submissions it went out in) and the clock
/// the GPU ran at as it was admitted, where the platform states it:
/// carried by its flight to its landing ([`crate::serve::Shell::timed`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimated {
    pub ns: f64,
    pub at: engine::StepDone,
    pub rows: u32,
    pub lanes: u32,
    pub pieces: u32,
    pub hz: Option<u64>,
}

/// Where a serving fire's device time is read from under a stated submit
/// bound, stated at load ([`crate::serve::Serve::fire_timing`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireTiming {
    /// The device's timestamps written before and after the fire's work.
    Timestamps,
    /// The host's clock from the fire's submit to its work done, where the
    /// device writes no timestamps: longer than the device's time (queued
    /// behind the fire before it), so a ratio only rises by it.
    SubmitToDone,
}

impl std::fmt::Display for FireTiming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FireTiming::Timestamps => "the device's timestamps around its work",
            FireTiming::SubmitToDone => {
                "the host's clock from its submit to its work done (the device writes no \
                 timestamps)"
            }
        })
    }
}

/// A serving fire's device time, ns, as its backend fills it once the fire
/// landed, read without waiting: nothing until then
/// ([`FireTiming`]). Filled where the fire's completion is already
/// observed (a fence waited, a work-done callback), never by a wait of its
/// own.
#[derive(Debug, Clone, Default)]
pub struct DeviceTime(std::sync::Arc<std::sync::atomic::AtomicU64>);

/// [`DeviceTime`]'s word for a fire landed whose time was not read (its
/// timestamps' readback failed).
const UNREAD: u64 = u64::MAX;

impl DeviceTime {
    /// The fire took `ns` on the device.
    pub fn set(&self, ns: u64) {
        self.0
            .store(ns.clamp(1, UNREAD - 1), std::sync::atomic::Ordering::Release);
    }

    /// The fire landed and its time was not read.
    pub fn unread(&self) {
        self.0.store(UNREAD, std::sync::atomic::Ordering::Release);
    }

    /// `None` until the fire's time is filled; then its ns, `None` inside
    /// where it was not read.
    #[must_use]
    pub fn get(&self) -> Option<Option<u64>> {
        match self.0.load(std::sync::atomic::Ordering::Acquire) {
            0 => None,
            UNREAD => Some(None),
            ns => Some(Some(ns)),
        }
    }
}

/// The ratios of serving fires' measured device times to their estimates
/// (each the estimate at a ratio of 1, before the worst ratio and the
/// scale), each kept from its landing for one bound's span (the span the
/// next submission is held to): every estimate is read at the worst of
/// them where it passes the stored ratio ([`SubmitBound::ratio_now`]). A device that slows under
/// load (a phone's GPU heating to its thermal cap, its clock 1300 to ~830
/// MHz within seconds) runs each fire longer than its estimate taken at
/// load; the next fires are read at that ratio and split before they reach
/// the bound. The worst, not a mean: while the device slows, splitting only
/// grows more conservative. Nothing landed within the span: the stored
/// ratio alone.
///
/// Each ratio is kept with the rung its fire ran under
/// ([`SubmitBound::rung_of`]): where a clock table's floor covers the
/// clock, an estimate's error is its shape's (an Adreno 840's one-lane
/// decode ran ~2.2x its estimate where two lanes ran ~1.0, a cost of width
/// no clock scales), so a step is read at the worst of its own rung's
/// fires; with no table, the worst of every fire's, which then carries the
/// clock across shapes.
#[derive(Debug, Clone, Default)]
pub struct Measured {
    landed: std::collections::VecDeque<(std::time::Instant, u32, f64)>,
}

impl Measured {
    /// A fire under rung `rung` landed at `at`, its measured ratio `ratio`
    /// (at full clock where its clock is known).
    pub fn landed(&mut self, at: std::time::Instant, rung: u32, ratio: f64) {
        if ratio.is_finite() && ratio > 0.0 {
            self.landed.push_back((at, rung, ratio));
        }
    }

    /// Drops what landed before `since`.
    pub fn since(&mut self, since: std::time::Instant) {
        while self.landed.front().is_some_and(|&(at, ..)| at < since) {
            self.landed.pop_front();
        }
    }

    /// The worst ratio kept, of every rung, 0 where none is.
    #[must_use]
    pub fn worst(&self) -> f64 {
        self.landed.iter().map(|&(.., r)| r).fold(0.0, f64::max)
    }

    /// The worst ratio kept of `rung`'s fires, 0 where none is.
    #[must_use]
    pub fn worst_of(&self, rung: u32) -> f64 {
        self.landed
            .iter()
            .filter(|&&(_, at, _)| at == rung)
            .map(|&(.., r)| r)
            .fold(0.0, f64::max)
    }

    /// The fires kept.
    #[must_use]
    pub fn len(&self) -> usize {
        self.landed.len()
    }

    /// The fires kept of `rung`.
    #[must_use]
    pub fn len_of(&self, rung: u32) -> usize {
        self.landed.iter().filter(|&&(_, at, _)| at == rung).count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.landed.is_empty()
    }
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
    /// The estimate of `step` at a ratio of 1, ns: `(whole, per layer)`, a
    /// layer's submission its share of the dispatches but the readout's,
    /// the last carrying the readout too.
    #[must_use]
    pub fn raw_ns(&self, step: StepRows) -> (f64, f64) {
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
        (whole, per_layer)
    }

    /// The ratio of a fire's time to its estimate at a ratio of 1
    /// ([`SubmitBound::raw_ns`]) an estimate is read at: the stored one (the
    /// worst ratio the rates step measured times the last probe's scale),
    /// or the worst one measured over the fires landed within the last
    /// bound ([`Measured`]) where it passes that. One quantity, actual over
    /// estimate, both ways: the stored covers a cold start and the fires
    /// before any lands, the measured a device slowing past it. Never below
    /// 1.
    /// Of every rung's fires: what a step of the widest rung is read at
    /// with no table, the widest any is read at with one.
    #[must_use]
    pub fn ratio_now(&self) -> f64 {
        self.error_now() * self.floor_factor()
    }

    /// The estimate's error now, of every rung's fires: the stored ratio,
    /// or the worst measured where it passes that; at least 1.
    #[must_use]
    pub fn error_now(&self) -> f64 {
        self.stored_ratio().max(self.measured.worst())
    }

    /// The fires whose measured ratios read `step`'s error: its own rung's
    /// where a clock table's floor covers the clock, `None` (every rung's)
    /// where none does ([`Measured`]).
    #[must_use]
    pub fn measured_rung(&self, step: StepRows) -> Option<u32> {
        self.clock.as_ref().map(|_| self.rung_of(step.rows))
    }

    /// `step`'s estimate error: the stored ratio, or the worst measured over
    /// the fires that read it ([`SubmitBound::measured_rung`]) where it
    /// passes that.
    #[must_use]
    pub fn error_for(&self, step: StepRows) -> f64 {
        let measured = match self.measured_rung(step) {
            Some(rung) => self.measured.worst_of(rung),
            None => self.measured.worst(),
        };
        self.stored_ratio().max(measured)
    }

    /// The ratio `step` is read at: its error ([`SubmitBound::error_for`])
    /// times the floor factor.
    #[must_use]
    pub fn ratio_for(&self, step: StepRows) -> f64 {
        self.error_for(step) * self.floor_factor()
    }

    /// How much longer a fire timed at full clock runs at the GPU's least
    /// clock, its table's greatest over its least ([`crate::clock`]); 1
    /// where the platform states no table.
    #[must_use]
    pub fn floor_factor(&self) -> f64 {
        self.clock
            .as_ref()
            .map_or(1.0, crate::clock::ClockRange::floor_factor)
            .max(1.0)
    }

    /// The stored ratio: the worst ratio the rates step measured times the
    /// last probe's scale, each at least 1.
    #[must_use]
    pub fn stored_ratio(&self) -> f64 {
        self.worst.max(1.0) * self.scale.max(1.0)
    }

    /// The estimate of `step` at [`SubmitBound::ratio_for`], ns: `(whole,
    /// per layer)` ([`SubmitBound::raw_ns`]).
    #[must_use]
    pub fn step_ns(&self, step: StepRows) -> (f64, f64) {
        let (whole, per_layer) = self.raw_ns(step);
        let ratio = self.ratio_for(step);
        (whole * ratio, per_layer * ratio)
    }

    /// A serving fire estimated at `estimate_ns` (its whole [`raw_ns`])
    /// landed at `at`, taking `device_ns` on the device; what landed more
    /// than one bound before it is dropped.
    ///
    /// [`raw_ns`]: SubmitBound::raw_ns
    /// The GPU ran it at `hz` where its clock is known
    /// ([`crate::clock::ClockRange::now_hz`]): its ratio is read back to
    /// full clock, the estimate's error, so the floor factor counts the
    /// clock once.
    /// Returns the ratio kept (at full clock where the clock is known).
    /// It ran `rows` rows: kept under their rung ([`Measured`]).
    pub fn landed(
        &mut self,
        at: std::time::Instant,
        rows: u32,
        estimate_ns: f64,
        device_ns: u64,
        hz: Option<u64>,
    ) -> Option<f64> {
        let mut kept = None;
        if estimate_ns > 0.0 && device_ns > 0 {
            let ratio = device_ns as f64 / estimate_ns;
            let error = self
                .clock
                .as_ref()
                .map_or(ratio, |clock| clock.at_full_clock(ratio, hz));
            let rung = self.rung_of(rows);
            self.measured.landed(at, rung, error);
            kept = Some(error);
        }
        self.age(at);
        kept
    }

    /// The probe read `ratio` (its time over its curve time) at `hz`: the
    /// scale estimates are read at from here, read back to full clock
    /// where its clock is known, never below 1.
    pub fn probed(&mut self, ratio: f64, hz: Option<u64>) {
        let at_full = self
            .clock
            .as_ref()
            .map_or(ratio, |clock| clock.at_full_clock(ratio, hz));
        self.scale = at_full.max(1.0);
    }

    /// The clock the GPU runs at now, where the platform states it.
    #[must_use]
    pub fn clock_now(&self) -> Option<u64> {
        self.clock.as_ref().and_then(crate::clock::ClockRange::now_hz)
    }

    /// Drops the measured fires that landed more than one bound before
    /// `now`.
    pub fn age(&mut self, now: std::time::Instant) {
        if let Some(since) =
            now.checked_sub(std::time::Duration::from_nanos(self.bound_ns as u64))
        {
            self.measured.since(since);
        }
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
        let (measured, fires, of) = match self.measured_rung(step) {
            Some(rung) => (
                self.measured.worst_of(rung),
                self.measured.len_of(rung),
                format!("of its rung {rung}"),
            ),
            None => (
                self.measured.worst(),
                self.measured.len(),
                "of every rung".to_string(),
            ),
        };
        format!(
            "a step of {} rows over {} lanes at {} keys is estimated at {:.0} ms whole and {:.0} \
             ms a layer at a time ({} layers, at the ratio {:.3} of a fire's time to its \
             estimate: the worst ratio {:.3} the rates step measured, `[rates] \
             submit_worst_ratio`, times the clock's scale {:.3} its last probe read, or the \
             worst ratio {:.3} measured over the {} fires {} landed within the last bound \
             where it passes that, times the floor factor {:.2} of the GPU's clock table {}), \
             past \
             the \
             platform's stated bound of {:.0} ms on one GPU submission \
             (engine_gpu::rates::stated_submit_bound); the widest prefill rung this load lays \
             is {} rows",
            step.rows,
            step.lanes,
            step.keys,
            admitted.ns() / 1e6,
            per / 1e6,
            self.layers,
            self.ratio_for(step),
            self.worst,
            self.scale,
            measured,
            fires,
            of,
            self.floor_factor(),
            self.clock
                .as_ref()
                .map_or_else(|| "(none read)".to_string(), ToString::to_string),
            self.bound_ns / 1e6,
            self.rungs.last().copied().unwrap_or(0),
        )
    }
}

impl SubmitBound {
    /// The rows a piece of `step`'s one prompt lane (of `prompt_rows` rows)
    /// takes where `step` is past the bound even a layer at a time: the
    /// most rows of it a step of the other lanes' rows and those is
    /// admitted at, every piece attending the step's longest keys (its last
    /// piece's); `None` where even one row's piece is past the bound a
    /// layer at a time (refused by name).
    #[must_use]
    pub fn rows_a_piece(&self, step: StepRows, prompt_rows: u32) -> Option<u32> {
        let other = step.rows.saturating_sub(prompt_rows);
        let fits = |rows: u32| {
            !matches!(
                self.admit(StepRows {
                    rows: other + rows,
                    lanes: step.lanes,
                    keys: step.keys,
                    prompt: true,
                }),
                Admitted::Refused { .. }
            )
        };
        if !fits(1) {
            return None;
        }
        // Admitted rows fall as rows rise: the most that fit, by halves.
        let (mut fit, mut past) = (1u32, prompt_rows.max(1));
        while past - fit > 1 {
            let mid = fit + (past - fit) / 2;
            if fits(mid) {
                fit = mid;
            } else {
                past = mid;
            }
        }
        Some(fit)
    }
}

/// Under `held`, where `step` is past the bound even a layer at a time:
/// the piece to fire before it and the rest, `Ok(None)` where it is
/// admitted as it is. Its widest prompt lane gives the piece: that lane
/// alone over the most of its rows a step of them alone is admitted at,
/// all but its last row at most (so each prompt lane runs as its own
/// sequence of pieces, and the rest keeps every lane, the decode lanes and
/// each prompt lane's last rows); called again on the rest at the ratio
/// read then. `Err` names why it is refused: no prompt lane to cut, one
/// row of one layer past the bound, or a lane a piece cannot carry.
pub fn next_cut(
    held: &SubmitBound,
    step: &engine::fire::Step,
    facts: &poem_ir::Facts,
) -> Result<Option<(engine::fire::Step, engine::fire::Step, u32)>, String> {
    if !matches!(held.admit(StepRows::of(step)), Admitted::Refused { .. }) {
        return Ok(None);
    }
    let Some(at) = (0..step.lanes.len())
        .filter(|&at| step.lanes[at].rows() > 1)
        .max_by_key(|&at| step.lanes[at].rows())
    else {
        return Err("it carries no prompt lane to cut by rows".into());
    };
    let lane = &step.lanes[at];
    let rows = lane.rows();
    let alone = StepRows {
        rows,
        lanes: 1,
        keys: lane.kv.held.saturating_add(rows),
        prompt: true,
    };
    let Some(fit) = held.rows_a_piece(alone, rows) else {
        return Err(format!(
            "one row of one layer of its lane {at} ({rows} rows at {} keys) is past the bound",
            alone.keys
        ));
    };
    let piece_rows = fit.min(rows - 1);
    let (piece, rest) = split_lane(step, at, facts, piece_rows)?;
    Ok(Some((piece, rest, piece_rows)))
}

/// Why `lane` cannot be cut into pieces of fewer rows: what it states that
/// a piece would not carry as the whole lane does. `None` where it can be.
fn uncuttable(lane: &engine::fire::Lane, facts: &poem_ir::Facts) -> Option<&'static str> {
    use poem_ir::Builtin;
    let device_fold = matches!(
        lane.rs,
        engine::fire::RsVerb::Buffer {
            fold: engine::fire::FoldLen::Device,
            ..
        }
    );
    let admitted = matches!(
        lane.rs,
        engine::fire::RsVerb::Buffer {
            lineage: engine::fire::Lineage::Admitted,
            ..
        }
    );
    let checks: [(bool, &'static str); 9] = [
        (lane.kv_less, "it holds no kv slots"),
        (lane.mask.is_some(), "it states a mask"),
        (!lane.inputs.is_empty(), "it feeds guest inputs"),
        (lane.tap.is_some(), "it taps its rows"),
        (lane.reports != 0, "it asks its fold reported"),
        (
            device_fold,
            "its recurrent buffer folds as far as its program says, which no piece knows",
        ),
        (
            admitted,
            "its recurrence replays the slots its mask admits, shared with other lanes",
        ),
        (
            lane.list != engine::fire::ListKind::default(),
            "its program places its rows",
        ),
        (
            [Builtin::Masked, Builtin::Media, Builtin::Drafts, Builtin::Scores, Builtin::Encode]
                .iter()
                .any(|&b| facts.holds(lane.word, b)),
            "its class reads its whole rows (a mask, media, drafts, scores or an encoder)",
        ),
    ];
    checks.iter().find(|(holds, _)| *holds).map(|&(_, why)| why)
}

/// `step` cut so its one prompt lane runs at most `rows` rows a piece: the
/// leading pieces, each that lane alone over its next `rows` rows, and the
/// step with that lane's last piece in its place ([`split_rows`], over and
/// over). `Err` names why it cannot be cut.
pub fn row_pieces(
    step: &engine::fire::Step,
    facts: &poem_ir::Facts,
    rows: u32,
) -> Result<(Vec<engine::fire::Step>, engine::fire::Step), String> {
    let mut leading = Vec::new();
    let mut rest = step.clone();
    while prompt_rows(&rest) > rows.max(1) {
        let (piece, after) = split_rows(&rest, facts, rows)?;
        leading.push(piece);
        rest = after;
    }
    Ok((leading, rest))
}

/// The rows of `step`'s widest lane: its one prompt lane's where it has one.
#[must_use]
pub fn prompt_rows(step: &engine::fire::Step) -> u32 {
    step.lanes.iter().map(engine::fire::Lane::rows).max().unwrap_or(0)
}

/// `step`'s one prompt lane's first `rows` rows split off: that lane alone
/// over them (no attachment, nothing read out: its keys land and the rest
/// reads them), and `step` with the lane's rows past them in its place, its
/// attachments and readout as stated (a readout row shifted). A recurrent
/// lane's state carries on, as a prompt chunked across fires carries it:
/// the first piece starts it as stated, the rest from what the first
/// folded (`RsReset::Held`, no fresh class bit, no slot copy again, a
/// forked record its own); a recurrent buffer's host fold is cut where the
/// first piece's rows reach it, and each snapshot lands in the piece that
/// reaches its row. `Err` names why it cannot be split there.
pub fn split_rows(
    step: &engine::fire::Step,
    facts: &poem_ir::Facts,
    rows: u32,
) -> Result<(engine::fire::Step, engine::fire::Step), String> {
    let prompts: Vec<usize> = (0..step.lanes.len())
        .filter(|&at| step.lanes[at].rows() > 1)
        .collect();
    let [at] = prompts[..] else {
        return Err(format!(
            "it carries {} prompt lanes, not one",
            prompts.len()
        ));
    };
    split_lane(step, at, facts, rows)
}

/// [`split_rows`] of lane `at` of `step`, whatever other prompt lanes it
/// carries: that lane alone over its first `rows` rows, and `step` with the
/// lane's rows past them in its place, the other lanes as stated.
pub fn split_lane(
    step: &engine::fire::Step,
    at: usize,
    facts: &poem_ir::Facts,
    rows: u32,
) -> Result<(engine::fire::Step, engine::fire::Step), String> {
    use engine::fire::Readout;
    if step.lanes.get(at).is_none_or(|lane| lane.rows() < 2) {
        return Err(format!("its lane {at} holds no rows to cut"));
    }
    if !step.media.is_empty() || !step.voxels.is_empty() {
        return Err("it carries media or clips, which its rows read whole".into());
    }
    let lane = &step.lanes[at];
    if let Some(why) = uncuttable(lane, facts) {
        return Err(format!("its prompt lane cannot be cut by rows: {why}"));
    }
    let total = lane.rows();
    let rows = rows.clamp(1, total - 1);
    let readout = match &lane.readout {
        Readout::Last => Readout::Last,
        Readout::None => Readout::None,
        Readout::Rows(read) => {
            if read.iter().any(|&row| row < rows) {
                return Err(format!(
                    "its prompt lane reads out a row before its last piece's (from row {rows})"
                ));
            }
            Readout::Rows(read.iter().map(|&row| row - rows).collect())
        }
    };
    // A recurrent buffer folding `fold` slots past where its record stands
    // (`folded`): the first piece folds what of that its rows reach, so the
    // rest replays no more than the whole fire would have left buffered.
    let target = match &lane.rs {
        engine::fire::RsVerb::Buffer {
            folded,
            fold: engine::fire::FoldLen::Host(fold),
            ..
        } => Some((*folded, folded + fold)),
        _ => None,
    };
    let piece = |from: u32, to: u32, readout: Readout| {
        let mut piece = lane.clone();
        let (a, b) = (from as usize, to as usize);
        piece.tokens = lane.tokens[a..b].to_vec();
        if !lane.positions.is_empty() {
            piece.positions = lane.positions[a..b].to_vec();
        }
        piece.kv.held = lane.kv.held + from;
        // What the piece before wrote is written as this one stages.
        piece.kv.readable = lane.kv.readable.max(lane.kv.held + from);
        piece.readout = readout;
        piece.word = facts.with(lane.word, poem_ir::Builtin::SingleToken, to - from == 1);
        if from > 0 {
            // The recurrent state the piece before folded carries on: this
            // one starts from it, not from zeros, and copies nothing more
            // into it.
            piece.rs_reset = engine::fire::RsReset::Held;
            piece.word = facts.with(piece.word, poem_ir::Builtin::Fresh, false);
            piece.slot_copies = Vec::new();
            // A record the first piece copied onto its own (`cow != run`, a
            // fork) is that one from there on.
            if let engine::fire::LaneRecord::Pieces(pieces) = &mut piece.record {
                for (_, refs) in pieces.iter_mut() {
                    for pair in refs.iter_mut() {
                        pair[0] = pair[1];
                    }
                }
            }
        }
        // Each snapshot in the piece whose rows reach it, at its row there.
        piece.rs_snapshots = lane
            .rs_snapshots
            .iter()
            .filter(|snap| (snap.row > from || (from == 0 && snap.row == 0)) && snap.row <= to)
            .map(|snap| engine::fire::RsSnapshot {
                row: snap.row - from,
                ..snap.clone()
            })
            .collect();
        if let (Some((stands, target)), engine::fire::RsVerb::Buffer { folded, fold, .. }) =
            (target, &mut piece.rs)
        {
            // Where the first piece leaves the record standing.
            let first = target.min(lane.kv.held + rows).max(stands);
            let (from_row, to_row) = if from == 0 { (stands, first) } else { (first, target) };
            *folded = from_row;
            *fold = engine::fire::FoldLen::Host(to_row.saturating_sub(from_row));
        }
        piece
    };
    let first = engine::fire::Step {
        lanes: vec![piece(0, rows, Readout::None)],
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    };
    let mut rest = step.clone();
    rest.lanes[at] = piece(rows, total, readout);
    Ok((first, rest))
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
            measured: Measured::default(),
            clock: None,
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
        let Admitted::Layers { submissions, .. } = b.admit(prompt(512, 3800)) else {
            panic!("split: {:?}", b.admit(prompt(512, 3800)));
        };
        assert!(submissions < 28, "layers packed: {submissions}");
        assert_eq!(subs.len(), 1 + submissions as usize);
        assert!(subs.iter().all(|&(_, ns)| ns <= BOUND));
    }

    /// A step past the bound whole goes out in the fewest submissions of
    /// consecutive layers that each fit it, not a layer at a time: 4.6
    /// bounds' worth over 28 layers takes 5 submissions of 6 layers (the
    /// last 4), each within the bound.
    #[test]
    fn a_split_step_packs_its_layers_into_the_fewest_submissions_that_fit() {
        let b = bound(1.0);
        let step = prompt(256, 1024);
        let (whole, per_layer) = b.step_ns(step);
        let held = SubmitBound {
            bound_ns: whole / 4.6,
            ..b
        };
        let Admitted::Layers {
            per_submission,
            submissions,
            layers_a_piece,
            ..
        } = held.admit(step)
        else {
            panic!("split: {:?}", held.admit(step));
        };
        assert_eq!((submissions, layers_a_piece), (5, 6));
        assert!(per_submission <= held.bound_ns);
        // Packing by the bound alone: one layer more a submission is past it.
        let (each, ns) = pack_layers(whole, per_layer, 28, held.bound_ns);
        assert_eq!((each, ns), (layers_a_piece, per_submission));
        let (wider, _) = pack_layers(whole, per_layer, 28, 2.0 * held.bound_ns);
        assert!(wider > each);
        // A layer's own estimate the bound: a layer a submission.
        assert_eq!(pack_layers(whole, per_layer, 28, per_layer).0, 1);
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

    /// A device slowing under load (a phone's GPU at its thermal cap):
    /// each fire's measured time over its estimate at a ratio of 1 rises
    /// from fire to fire. An estimate is read at the stored ratio (the
    /// rates step's worst times the probe's scale) until a measured ratio
    /// passes it, then at that one: a step that fit whole at load is split
    /// before it reaches the bound. One fast fire after a slow one does not
    /// lower it (the worst, not a mean); once the slow fires are a bound
    /// old, the stored ratio again.
    #[test]
    fn a_device_slowing_fire_by_fire_splits_a_step_that_fit_at_load() {
        let at_one = bound(1.0);
        let step = prompt(256, 1024);
        let (raw, per_layer) = at_one.raw_ns(step);
        // The field's worst ratio; the step whole under it with a layer to
        // spare, by 5%.
        let worst = 2.062;
        let mut b = SubmitBound {
            worst,
            bound_ns: (raw + per_layer) * worst * 1.05,
            ..at_one
        };
        assert!(matches!(b.admit(step), Admitted::Whole { .. }));
        assert_eq!(b.ratio_now(), worst);
        let t0 = std::time::Instant::now();
        let ms = |n: u64| std::time::Duration::from_millis(n);
        let fire = 100.0 * MS;
        let mut read = Vec::new();
        for (at, ratio) in [1.5, 2.0, 2.1, 2.3, 2.6].into_iter().enumerate() {
            b.landed(t0 + ms(at as u64 * 100), 256, fire, (fire * ratio) as u64, None);
            read.push((b.ratio_now(), b.admit(step).split()));
        }
        // Under the stored ratio: read at it, whole (2.1 is past it but
        // within the 5% to spare). Past it: read at the measured, split.
        let want = [(worst, false), (worst, false), (2.1, false), (2.3, true), (2.6, true)];
        for ((got, split), (ratio, splits)) in read.iter().zip(want) {
            assert!((got - ratio).abs() < 1e-6 && *split == splits, "{read:?}");
        }
        let (scaled, _) = b.step_ns(step);
        assert!((scaled - raw * 2.6).abs() < 1e-6 * scaled);
        // A fast fire after: the worst stands.
        b.landed(t0 + ms(500), 256, fire, fire as u64, None);
        assert!(b.admit(step).split());
        assert!((b.ratio_now() - 2.6).abs() < 1e-6);
        let why = b.refusal(step, &b.admit(step));
        assert!(why.contains("at the ratio 2.600"), "{why}");
        assert!(why.contains("worst ratio 2.600 measured over the 6 fires of every rung"), "{why}");
        // A bound past the slow fires: the stored ratio again.
        b.age(t0 + ms(501) + std::time::Duration::from_nanos(b.bound_ns as u64));
        assert!(b.measured.is_empty());
        assert_eq!(b.ratio_now(), worst);
        assert!(matches!(b.admit(step), Admitted::Whole { .. }));
        // A probe's scale raises the stored ratio over a measured one it
        // passes: 2.062 x 1.2 = 2.474 over 2.3.
        b.landed(t0 + ms(10_000), 256, fire, (fire * 2.3) as u64, None);
        let probed = SubmitBound { scale: 1.2, ..b };
        assert!((probed.ratio_now() - worst * 1.2).abs() < 1e-6);
    }

    /// The field's lost fire: a prompt step estimated at ~1643 ms at the
    /// worst ratio (82% of KGSL's 2 s, a layer to spare) went out whole and
    /// the GPU's cap fell 1300 to 578 MHz inside it. With the Adreno 840's
    /// clock table (160 to 1300 MHz) stated, every estimate is read at its
    /// floor: the step splits a layer at a time, each submission within the
    /// bound even at 160 MHz; a decode step stays whole.
    #[test]
    fn a_clock_table_holds_each_submission_to_the_bound_at_its_floor() {
        let at_one = bound(1.0);
        let step = prompt(256, 1024);
        let (raw, _) = at_one.raw_ns(step);
        let worst = 2.062;
        let unclocked = SubmitBound {
            worst,
            bound_ns: raw * worst / 0.8215,
            ..at_one
        };
        let (whole, _) = unclocked.step_ns(step);
        assert!((whole / unclocked.bound_ns - 0.8215).abs() < 1e-9);
        assert!(matches!(unclocked.admit(step), Admitted::Whole { .. }));
        assert_eq!(unclocked.floor_factor(), 1.0);
        let table = crate::clock::ClockRange::stated(160_000_000, 1_300_000_000);
        let clocked = SubmitBound {
            clock: Some(table),
            ..unclocked.clone()
        };
        assert!((clocked.floor_factor() - 8.125).abs() < 1e-9);
        assert!((clocked.ratio_now() - worst * 8.125).abs() < 1e-9);
        let Admitted::Layers {
            per_submission,
            submissions,
            layers_a_piece,
            ..
        } = clocked.admit(step)
        else {
            panic!("split: {:?}", clocked.admit(step));
        };
        // 0.8215 of the bound at full clock, x8.125 at the floor: 6.7
        // bounds' worth, in 7 submissions of 4 layers.
        assert_eq!((submissions, layers_a_piece), (7, 4));
        assert!(per_submission <= clocked.bound_ns, "{per_submission} past the bound");
        let subs = frame_submissions(&[shaped(&clocked, step)], clocked.layers, clocked.bound_ns)
            .expect("every submission fits");
        assert!(subs.iter().all(|&(_, ns)| ns <= clocked.bound_ns));
        let why = clocked.refusal(step, &clocked.admit(step));
        assert!(why.contains("floor factor 8.12"), "{why}");
        // A decode step, tiny, stays whole at the floor.
        let decode = StepRows {
            rows: 1,
            lanes: 1,
            keys: 1024,
            prompt: false,
        };
        assert!(matches!(clocked.admit(decode), Admitted::Whole { .. }));
    }

    /// A step past the bound even a layer at a time (a 512-row prompt at
    /// 4096 keys, a layer of it past the bound) is cut by rows: the most
    /// rows a piece is admitted at, the next row past it refused; a bound
    /// one row of one layer is past: none, refused by name.
    #[test]
    fn a_step_past_the_bound_a_layer_at_a_time_is_cut_by_rows() {
        let b = bound(1.0);
        let step = prompt(512, 4096);
        let (_, per_layer) = b.step_ns(step);
        let tight = SubmitBound {
            bound_ns: per_layer * 0.6,
            ..b.clone()
        };
        assert!(matches!(tight.admit(step), Admitted::Refused { .. }));
        let rows = tight.rows_a_piece(step, 512).expect("a piece of fewer rows fits");
        assert!(rows > 1 && rows < 512, "{rows}");
        let piece = |rows: u32| StepRows {
            rows,
            lanes: 1,
            keys: 4096,
            prompt: true,
        };
        assert!(!matches!(tight.admit(piece(rows)), Admitted::Refused { .. }));
        assert!(matches!(tight.admit(piece(rows + 1)), Admitted::Refused { .. }));
        // One row of one layer past the bound: none.
        let one = tight.step_ns(piece(1)).1;
        let past = SubmitBound {
            bound_ns: one * 0.5,
            ..b
        };
        assert_eq!(past.rows_a_piece(step, 512), None);
    }

    /// A step's one prompt lane cut into pieces of `rows`: the leading
    /// pieces that lane alone, in order, each over its next rows at the
    /// slots its predecessors wrote, nothing read out; the step's last
    /// piece in its place with the readout; a one-row piece's class word
    /// single-token. A lane whose class reads its whole rows is not cut.
    #[test]
    fn a_prompt_lane_cuts_into_pieces_its_successors_read() {
        use engine::fire::{Lane, Readout, Step};
        let mut facts = poem_ir::Facts::default();
        let _ = facts.builtin(poem_ir::Builtin::SingleToken);
        let _ = facts.builtin(poem_ir::Builtin::Masked);
        let mut prompt_lane = Lane {
            slot: 2,
            tokens: (100..110).collect(),
            positions: (5..15).collect(),
            readout: Readout::Last,
            ..Lane::default()
        };
        prompt_lane.kv.held = 5;
        prompt_lane.kv.readable = 5;
        let decode = Lane {
            slot: 1,
            tokens: vec![7],
            positions: vec![40],
            ..Lane::default()
        };
        let step = Step {
            lanes: vec![decode.clone(), prompt_lane.clone()],
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
        };
        let (leading, last) = row_pieces(&step, &facts, 4).expect("cut");
        assert_eq!(leading.len(), 2);
        for (k, piece) in leading.iter().enumerate() {
            let k = k as u32;
            assert!(piece.attachments.is_empty());
            let [lane] = &piece.lanes[..] else { panic!("one lane a piece") };
            assert_eq!(lane.tokens, (100 + 4 * k..104 + 4 * k).collect::<Vec<_>>());
            assert_eq!(lane.positions, (5 + 4 * k..9 + 4 * k).collect::<Vec<_>>());
            assert_eq!((lane.kv.held, lane.kv.readable), (5 + 4 * k, 5 + 4 * k));
            assert_eq!(lane.readout, Readout::None);
            assert!(!facts.holds(lane.word, poem_ir::Builtin::SingleToken));
        }
        assert_eq!(last.lanes[0], decode, "the other lanes as stated");
        assert_eq!(last.lanes[1].tokens, vec![108, 109]);
        assert_eq!(last.lanes[1].kv.held, 13);
        assert_eq!(last.lanes[1].readout, Readout::Last);
        // Pieces of one row: single-token words; a readout row shifts.
        let mut rows_read = step.clone();
        rows_read.lanes[1].readout = Readout::Rows(vec![9]);
        let (leading, last) = row_pieces(&rows_read, &facts, 1).expect("cut by rows of one");
        assert_eq!(leading.len(), 9);
        assert!(facts.holds(leading[0].lanes[0].word, poem_ir::Builtin::SingleToken));
        assert_eq!(last.lanes[1].readout, Readout::Rows(vec![0]));
        rows_read.lanes[1].readout = Readout::Rows(vec![2, 9]);
        assert!(row_pieces(&rows_read, &facts, 4).unwrap_err().contains("reads out a row"));
        // A masked class reads its whole rows: not cut, by name.
        let mut masked = step.clone();
        masked.lanes[1].word = facts.with(0, poem_ir::Builtin::Masked, true);
        assert!(row_pieces(&masked, &facts, 4).unwrap_err().contains("cannot be cut"));
    }

    /// A recurrent lane cut into pieces carries its state piece to piece:
    /// the first piece starts it as stated (fresh, its slot copy), each
    /// after from what the pieces before folded; each snapshot lands in the
    /// piece that reaches its row; a recurrent buffer's host fold is cut
    /// where each piece's rows reach it; a record the first piece copies
    /// (a fork) is its own after. A device-counted fold, or a recurrence
    /// shared with other lanes, is not cut.
    #[test]
    fn a_recurrent_lane_cut_into_pieces_carries_its_state() {
        use engine::fire::{FoldLen, Lane, Lineage, Readout, RsReset, RsSnapshot, RsVerb, Step};
        let mut facts = poem_ir::Facts::default();
        let _ = facts.builtin(poem_ir::Builtin::SingleToken);
        let _ = facts.builtin(poem_ir::Builtin::Fresh);
        let snap = |row: u32, slot: u32| RsSnapshot {
            row,
            slot,
            pieces: Vec::new(),
            fixed: Vec::new(),
        };
        let lane = Lane {
            slot: 0,
            word: facts.with(0, poem_ir::Builtin::Fresh, true),
            tokens: (0..10).collect(),
            readout: Readout::Last,
            rs_reset: RsReset::Fresh,
            rs_snapshots: vec![snap(3, 7), snap(9, 8)],
            slot_copies: vec![(engine::chunk::ShapeId(1), vec![[1, 2]])],
            record: engine::fire::LaneRecord::Pieces(vec![(
                engine::chunk::ShapeId(2),
                vec![[30, 31], [32, 32]],
            )]),
            ..Lane::default()
        };
        let step = |lane: Lane| Step {
            lanes: vec![lane],
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
        };
        let (leading, last) = row_pieces(&step(lane.clone()), &facts, 4).expect("cut");
        let pieces: Vec<&Lane> = leading.iter().map(|s| &s.lanes[0]).chain([&last.lanes[0]]).collect();
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].rs_reset, RsReset::Fresh);
        assert!(facts.holds(pieces[0].word, poem_ir::Builtin::Fresh));
        assert_eq!(pieces[0].slot_copies, lane.slot_copies);
        assert_eq!(pieces[0].rs_snapshots, vec![snap(3, 7)]);
        for piece in &pieces[1..] {
            assert_eq!(piece.rs_reset, RsReset::Held);
            assert!(!facts.holds(piece.word, poem_ir::Builtin::Fresh));
            assert!(piece.slot_copies.is_empty());
        }
        assert!(pieces[1].rs_snapshots.is_empty());
        // The fork's copy at the first piece; its own record after.
        assert_eq!(pieces[0].record, lane.record);
        let own = engine::fire::LaneRecord::Pieces(vec![(
            engine::chunk::ShapeId(2),
            vec![[31, 31], [32, 32]],
        )]);
        assert!(pieces[1..].iter().all(|piece| piece.record == own));
        assert_eq!(pieces[2].rs_snapshots, vec![snap(1, 8)]);
        // A buffer standing at 2 that folds 8: 2..6, 6..10, nothing after.
        let buffer = |fold: FoldLen, lineage: Lineage| RsVerb::Buffer {
            pages: vec![3, 4],
            base: 0,
            folded: 2,
            fold,
            lineage,
        };
        let mut buffered = lane.clone();
        buffered.kv.held = 2;
        buffered.rs = buffer(FoldLen::Host(8), Lineage::Set);
        let (leading, last) = row_pieces(&step(buffered.clone()), &facts, 4).expect("cut");
        let folds: Vec<(u32, FoldLen)> = leading
            .iter()
            .map(|s| &s.lanes[0])
            .chain([&last.lanes[0]])
            .map(|piece| match &piece.rs {
                RsVerb::Buffer { folded, fold, .. } => (*folded, *fold),
                RsVerb::Fold => panic!("a buffer stays a buffer"),
            })
            .collect();
        assert_eq!(
            folds,
            vec![(2, FoldLen::Host(4)), (6, FoldLen::Host(4)), (10, FoldLen::Host(0))]
        );
        buffered.rs = buffer(FoldLen::Device, Lineage::Set);
        assert!(row_pieces(&step(buffered.clone()), &facts, 4).unwrap_err().contains("program says"));
        buffered.rs = buffer(FoldLen::Host(8), Lineage::Admitted);
        assert!(row_pieces(&step(buffered), &facts, 4).unwrap_err().contains("shared"));
    }

    /// A step of two prompt lanes and a decode lane, past the bound even a
    /// layer at a time, is cut lane by lane, then by rows: each piece one
    /// prompt lane alone and admitted, every prompt lane's rows in order
    /// and whole across its pieces and the rest, the decode lane riding the
    /// rest, which is admitted.
    #[test]
    fn a_step_of_two_prompt_lanes_past_the_bound_is_cut_lane_by_lane() {
        use engine::fire::{Lane, Readout, Step};
        let facts = poem_ir::Facts::default();
        let prompt_lane = |slot: u32, base: u32| Lane {
            slot,
            tokens: (base..base + 256).collect(),
            readout: Readout::Last,
            ..Lane::default()
        };
        let mut decode = Lane {
            slot: 2,
            tokens: vec![9],
            readout: Readout::Last,
            ..Lane::default()
        };
        decode.kv.held = 700;
        let step = Step {
            lanes: vec![prompt_lane(0, 0), prompt_lane(1, 1000), decode.clone()],
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
        };
        let b = bound(1.0);
        let (_, per_layer) = b.step_ns(StepRows::of(&step));
        let held = SubmitBound {
            bound_ns: per_layer * 0.4,
            ..b
        };
        assert!(matches!(held.admit(StepRows::of(&step)), Admitted::Refused { .. }));
        let mut pieces = Vec::new();
        let mut rest = step.clone();
        while let Some((piece, after, rows)) = next_cut(&held, &rest, &facts).expect("cut") {
            assert_eq!(piece.lanes.len(), 1, "a piece is one prompt lane alone");
            assert_eq!(piece.lanes[0].rows(), rows);
            assert!(!matches!(held.admit(StepRows::of(&piece)), Admitted::Refused { .. }));
            pieces.push(piece);
            rest = after;
        }
        assert!(!matches!(held.admit(StepRows::of(&rest)), Admitted::Refused { .. }));
        assert!(pieces.iter().any(|p| p.lanes[0].slot == 0));
        assert!(pieces.iter().any(|p| p.lanes[0].slot == 1));
        assert_eq!(rest.lanes.len(), 3, "the rest keeps every lane");
        assert_eq!(rest.lanes[2], decode, "the decode lane rides the rest as stated");
        for (slot, base) in [(0u32, 0u32), (1, 1000)] {
            let tokens: Vec<u32> = pieces
                .iter()
                .filter(|p| p.lanes[0].slot == slot)
                .flat_map(|p| p.lanes[0].tokens.clone())
                .chain(rest.lanes[slot as usize].tokens.clone())
                .collect();
            assert_eq!(tokens, (base..base + 256).collect::<Vec<_>>(), "lane {slot} whole, in order");
            let last = &rest.lanes[slot as usize];
            assert_eq!(last.kv.held + last.rows(), 256);
            assert_eq!(last.readout, Readout::Last);
        }
        // One row of one layer past the bound: refused, by name.
        let one = held.step_ns(StepRows { rows: 1, lanes: 1, keys: 256, prompt: true }).1;
        let past = SubmitBound {
            bound_ns: one * 0.5,
            ..held
        };
        let why = next_cut(&past, &step, &facts).unwrap_err();
        assert!(why.contains("one row of one layer"), "{why}");
    }

    /// Under a clock table's floor a fire's measured ratio reads its own
    /// rung: a slow one-lane decode (the Adreno 840's ~2.2x) leaves a
    /// 512-row prompt's admission at the stored ratio, the refusal naming
    /// the rung; with no table it raises it, every rung's fires read.
    #[test]
    fn a_slow_decode_moves_its_own_rung_under_a_floor_and_every_rung_without() {
        let base = bound(1.0);
        let prompt512 = prompt(512, 512);
        let decode = StepRows {
            rows: 1,
            lanes: 1,
            keys: 600,
            prompt: false,
        };
        let floored = SubmitBound {
            clock: Some(crate::clock::ClockRange::stated(160_000_000, 1_300_000_000)),
            ..base.clone()
        };
        let at = std::time::Instant::now();
        for held in [floored, base] {
            let floor = held.clock.is_some();
            let mut held = held;
            let before = held.step_ns(prompt512);
            assert_eq!(held.ratio_for(prompt512), held.stored_ratio() * held.floor_factor());
            // A decode of one row ran 2.2x its estimate at full clock.
            held.landed(at, 1, 10.0 * MS, (22.0 * MS) as u64, None);
            assert!((held.error_for(decode) - 2.2).abs() < 1e-9, "the decode's own rung");
            assert_eq!(held.measured_rung(decode), floor.then_some(64));
            let after = held.step_ns(prompt512);
            if floor {
                assert_eq!(after, before, "the prompt's rung measured nothing: the stored ratio");
                assert_eq!(held.error_for(prompt512), held.stored_ratio());
                let why = held.refusal(prompt512, &held.admit(prompt512));
                assert!(why.contains("over the 0 fires of its rung 512"), "{why}");
            } else {
                assert!((after.0 / before.0 - 2.2).abs() < 1e-9, "every rung's fires read");
                let why = held.refusal(prompt512, &held.admit(prompt512));
                assert!(why.contains("over the 1 fires of every rung"), "{why}");
            }
        }
    }

    /// A fire's device time reads nothing until its backend fills it, then
    /// its ns, or that it landed unread.
    #[test]
    fn a_device_time_is_nothing_until_filled() {
        let time = DeviceTime::default();
        assert_eq!(time.get(), None);
        time.clone().set(42);
        assert_eq!(time.get(), Some(Some(42)));
        time.unread();
        assert_eq!(time.get(), Some(None));
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
