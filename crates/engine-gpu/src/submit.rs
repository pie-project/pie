//! A serving step held to the bound the platform states on one GPU
//! submission ([`crate::rates::STATED_SUBMIT_BOUND`]: KGSL's 2 s on
//! Android, none elsewhere). Past it the driver faults the submission and
//! the device is lost (an Adreno 840 on wgpu lost it on a 3843-token
//! prompt's prefill steps of ~480 rows at ~3 to 3.8k keys, each 1 to 2 s).
//!
//! Every step is its own submission on both slang backends, so a frame's
//! steps never share one: each is admitted alone. A step is estimated by the
//! load's own estimate ([`engine::load::FireEstimate`], the device's
//! measured `[rates]` over the model's counts) at its rows and the keys its
//! longest lane attends, times the worst ratio of a frame's time to its
//! estimate the device's rates step measured (its clock's swing among them,
//! [`crate::rates::measured_worst`]; 1 where this process measured none):
//!
//! - estimated within the bound: one submission, as without a bound;
//! - past it whole, its layers each within it: submitted a layer at a time
//!   (`Run::flush_between_layers`), each submission a layer's share of the
//!   step's dispatches and the readout's with the last;
//! - past it even a layer at a time: refused by name, naming the bound and
//!   the estimate. The load lowers the rows a prefill fire takes
//!   (`FireLimits::prefill_rows`) to the widest laid rung whose prompt step
//!   at the load's context is admitted ([`SubmitBound::prefill_rows`]), so a
//!   prompt is cut to steps that fit; a step refused past that is one no
//!   rung fits.
//!
//! No margin is chosen here: the bound is the platform's, the estimate and
//! the worst ratio are measurements. Where no bound is stated nothing is
//! estimated (the load keeps no [`SubmitBound`]).

use engine::load::{FireEstimate, RungFacts};

/// What a step asks of the device, as the estimate weighs it: `rows` new
/// tokens over `lanes` lanes, the longest lane attending `keys` keys (the
/// slots it holds and its own rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepRows {
    pub rows: u32,
    pub lanes: u32,
    pub keys: u32,
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
/// ns in one submission where it fits `bound_ns`; else in `layers`
/// submissions where the widest, `per_layer` ns, fits; else refused.
#[must_use]
pub fn admit_ns(whole: f64, per_layer: f64, layers: u32, bound_ns: f64) -> Admitted {
    if whole <= bound_ns {
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

/// The submissions a frame of `steps` (each `(whole, per_layer)` ns at the
/// worst ratio) is cut into: each step its own submission, or its layers'
/// where it is past the bound whole; each entry the step's index and its
/// estimate. `Err` names the first step no submission of which fits.
pub fn frame_submissions(
    steps: &[(f64, f64)],
    layers: u32,
    bound_ns: f64,
) -> Result<Vec<(usize, f64)>, (usize, Admitted)> {
    let mut out = Vec::new();
    for (at, &(whole, per_layer)) in steps.iter().enumerate() {
        match admit_ns(whole, per_layer, layers, bound_ns) {
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

/// A load's hold on its serving steps under a stated submit bound.
#[derive(Debug, Clone)]
pub struct SubmitBound {
    /// The platform's bound, ns.
    pub bound_ns: f64,
    /// The worst ratio of a frame's time to its estimate the device's rates
    /// step measured, at least 1.
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
}

impl SubmitBound {
    /// The estimate of `step` at the worst ratio, ns: `(whole, per layer)`,
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
        let whole = terms.dispatched;
        let head = terms.readout.max(terms.head_read) + self.estimate.rates.launch_ns;
        let per_layer = (whole - head).max(0.0) / f64::from(self.layers.max(1)) + head;
        let worst = self.worst.max(1.0);
        (whole * worst, per_layer * worst)
    }

    /// How `step` is submitted.
    #[must_use]
    pub fn admit(&self, step: StepRows) -> Admitted {
        let (whole, per_layer) = self.step_ns(step);
        admit_ns(whole, per_layer, self.layers, self.bound_ns)
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
             estimate the rates step measured), past the platform's stated bound of {:.0} ms on \
             one GPU submission (engine_gpu::rates::STATED_SUBMIT_BOUND); the widest prefill \
             rung this load lays is {} rows",
            step.rows,
            step.lanes,
            step.keys,
            admitted.ns() / 1e6,
            per / 1e6,
            self.layers,
            self.worst,
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
        }
    }

    fn prompt(rows: u32, keys: u32) -> StepRows {
        StepRows { rows, lanes: 1, keys }
    }

    /// Two prompt steps of a frame, together past the bound and each within
    /// it: two submissions, each within the bound.
    #[test]
    fn a_frame_past_the_bound_together_splits_into_submissions_within_it() {
        let b = bound(1.0);
        let steps = [b.step_ns(prompt(256, 1024)), b.step_ns(prompt(256, 1280))];
        assert!(steps[0].0 + steps[1].0 > BOUND, "{steps:?}");
        let subs = frame_submissions(&steps, b.layers, BOUND).unwrap();
        assert_eq!(subs.len(), 2);
        assert!(subs.iter().all(|&(_, ns)| ns <= BOUND));
        // A step past the bound whole goes a layer at a time.
        let long = b.step_ns(prompt(512, 3800));
        assert!(long.0 > BOUND, "{long:?}");
        let subs = frame_submissions(&[steps[0], long], b.layers, BOUND).unwrap();
        assert_eq!(subs.len(), 1 + 28);
        assert!(subs.iter().all(|&(_, ns)| ns <= BOUND));
        assert!(b.admit(prompt(512, 3800)).split());
    }

    /// The worst ratio the rates step measured moves a step past the bound
    /// that fits at a ratio of 1.
    #[test]
    fn the_measured_worst_ratio_is_the_margin() {
        let at_one = bound(1.0);
        let step = prompt(256, 1024);
        let (whole, _) = at_one.step_ns(step);
        assert!(matches!(at_one.admit(step), Admitted::Whole { .. }));
        let swung = bound(BOUND / whole * 1.01);
        assert!(swung.admit(step).split());
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
            frame_submissions(&[b.step_ns(step)], b.layers, BOUND).map_err(|(at, _)| at),
            Err(0)
        );
    }
}
