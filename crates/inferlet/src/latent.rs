use crate::eta::{
    Channel, ForwardPass, Pipeline, broadcast, gather, max_elem, min_elem, normal, reduce_sum,
    reshape, sqrt,
};
use crate::model::{ScheduleFact, ScheduleKind};
use eta_dsl::Tensor;

pub mod prelude {
    pub use super::{
        DenoiseLoop, Dpm2m, Dpm2mChannels, FlowMatchEuler, apg, at, cfg_combine, dynamic_shift,
        euler_step, noise, rng_state,
    };
    pub use crate::eta::prelude::*;
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlowMatchEuler {
    pub sigmas: Vec<f32>,
    pub train_steps: u32,
}

impl FlowMatchEuler {
    pub fn from_model(steps: u32, rows: Option<u32>) -> Result<FlowMatchEuler, String> {
        let Some(schedule) = crate::model::schedule() else {
            return Err("this model states no schedule; nothing here denoises".to_string());
        };
        FlowMatchEuler::from_schedule(&schedule, steps, rows)
    }

    pub fn from_schedule(
        fact: &ScheduleFact,
        steps: u32,
        rows: Option<u32>,
    ) -> Result<FlowMatchEuler, String> {
        if fact.kind != ScheduleKind::Flow {
            return Err(format!(
                "FlowMatchEuler integrates a flow schedule; this model's is {:?}",
                fact.kind
            ));
        }
        let steps = steps.max(1);
        let mut sigmas: Vec<f32> = if fact.pinned_sigmas.is_empty() {
            let raw: Vec<f32> = (0..steps).map(|i| 1.0 - i as f32 / steps as f32).collect();
            let shift = match rows {
                Some(rows) if fact.shift > 0.0 => dynamic_shift(rows, fact.shift),
                _ => fact.shift,
            };
            raw.into_iter()
                .map(|sigma| shift_sigma(sigma, shift))
                .collect()
        } else {
            resample(&fact.pinned_sigmas, steps as usize)
        };
        sigmas.push(0.0);
        Ok(FlowMatchEuler {
            sigmas,
            train_steps: fact.train_steps.max(1),
        })
    }

    #[must_use]
    pub fn from_sigmas(mut sigmas: Vec<f32>, train_steps: u32) -> FlowMatchEuler {
        if sigmas.last().is_none_or(|&last| last != 0.0) {
            sigmas.push(0.0);
        }
        FlowMatchEuler {
            sigmas,
            train_steps: train_steps.max(1),
        }
    }

    #[must_use]
    pub fn steps(&self) -> u32 {
        u32::try_from(self.sigmas.len().saturating_sub(1)).unwrap_or(u32::MAX)
    }

    #[must_use]
    pub fn dt(&self, i: u32) -> f32 {
        let i = i as usize;
        match (self.sigmas.get(i), self.sigmas.get(i + 1)) {
            (Some(&from), Some(&to)) => to - from,
            _ => 0.0,
        }
    }

    #[must_use]
    pub fn dts(&self) -> Vec<f32> {
        (0..self.steps()).map(|i| self.dt(i)).collect()
    }

    #[must_use]
    pub fn timestep(&self, i: u32) -> f32 {
        self.sigmas.get(i as usize).copied().unwrap_or(0.0) * self.train_steps as f32
    }

    #[must_use]
    pub fn timesteps(&self) -> Vec<f32> {
        (0..self.steps()).map(|i| self.timestep(i)).collect()
    }

    #[must_use]
    pub fn handover(&self, boundary: f32) -> u32 {
        self.sigmas[..self.sigmas.len().saturating_sub(1)]
            .iter()
            .position(|&sigma| sigma <= boundary)
            .map_or(self.steps(), |i| i as u32)
    }
}

fn shift_sigma(sigma: f32, shift: f32) -> f32 {
    if shift <= 0.0 || shift == 1.0 {
        return sigma;
    }
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

#[must_use]
pub fn dynamic_shift(rows: u32, base_mu: f32) -> f32 {
    const BASE_ROWS: f32 = 256.0;
    const MAX_ROWS: f32 = 4096.0;
    const MAX_MU: f32 = 1.15;
    let slope = (MAX_MU - base_mu) / (MAX_ROWS - BASE_ROWS);
    let mu = slope * rows as f32 + (base_mu - slope * BASE_ROWS);
    mu.exp()
}

fn resample(pinned: &[f32], steps: usize) -> Vec<f32> {
    if pinned.len() == steps || pinned.len() < 2 {
        return pinned.to_vec();
    }
    (0..steps)
        .map(|i| {
            let at = i as f32 * (pinned.len() - 1) as f32 / (steps.max(2) - 1) as f32;
            let lo = at.floor() as usize;
            let hi = (lo + 1).min(pinned.len() - 1);
            let frac = at - lo as f32;
            pinned[lo] * (1.0 - frac) + pinned[hi] * frac
        })
        .collect()
}

pub fn euler_step(x: &Tensor, v: &Tensor, dt: &Tensor) -> Tensor {
    let shape = x.shape();
    x + &(v * &broadcast(dt, shape))
}

pub fn cfg_combine(cond: &Tensor, uncond: &Tensor, s: &Tensor) -> Tensor {
    let shape = cond.shape();
    uncond + &(&(cond - uncond) * &broadcast(s, shape))
}

pub fn apg(cond: &Tensor, uncond: &Tensor, s: f32, eta: f32, norm_threshold: f32) -> Tensor {
    let shape = cond.shape();
    let rows = shape.dims()[0];
    let per_row = |x: &Tensor| broadcast(reshape(x, [rows, 1]), shape);
    let mut diff = cond - uncond;
    if norm_threshold > 0.0 {
        let norm = sqrt(reduce_sum(&(&diff * &diff)));
        let scale = min_elem(
            &(&Tensor::constant(norm_threshold) / &max_elem(&norm, 1e-12f32)),
            1.0f32,
        );
        diff = &diff * &per_row(&scale);
    }
    let dot = reduce_sum(&(&diff * cond));
    let cond_sq = max_elem(reduce_sum(&(cond * cond)), 1e-12f32);
    let parallel = cond * &per_row(&(&dot / &cond_sq));
    let orthogonal = &diff - &parallel;
    let guidance = &orthogonal + &(&parallel * eta);
    cond + &(&guidance * (s - 1.0))
}

#[must_use]
pub fn rng_state(seed: u32) -> [u32; 2] {
    [seed, 0]
}

pub fn noise(shape: impl eta_dsl::IntoShape, seed: &Tensor) -> Tensor {
    normal(seed, shape)
}

pub fn at(vector: &Tensor, index: &Tensor) -> Tensor {
    gather(vector, reshape(index, [1]))
}

/// A denoising loop's clock: one pass a step, [`DenoiseLoop::drive`] its epilogue.
pub struct DenoiseLoop {
    pipe: Pipeline,
    step: Channel,
    timestep: Channel,
    ts: Channel,
    dts: Channel,
    fires: u32,
}

impl DenoiseLoop {
    #[must_use]
    pub fn new(sched: &FlowMatchEuler) -> DenoiseLoop {
        let mut ts = vec![sched.timestep(0)];
        ts.extend(sched.timesteps());
        ts.push(0.0);
        let mut dts = vec![0.0];
        dts.extend(sched.dts());
        dts.push(0.0);
        DenoiseLoop {
            pipe: Pipeline::new(),
            step: Channel::from([0u32]),
            timestep: Channel::from([ts[0]]),
            ts: Channel::from(ts),
            dts: Channel::from(dts),
            fires: sched.steps() + 1,
        }
    }

    #[must_use]
    pub fn fires(&self) -> u32 {
        self.fires
    }

    /// The timestep this fire runs at, the `timestep` input of every lane.
    #[must_use]
    pub fn timestep(&self) -> Channel {
        self.timestep
    }

    /// The step each fire takes.
    #[must_use]
    pub fn dts(&self) -> Channel {
        self.dts
    }

    /// `pass`'s epilogue: `body(k)` at fire `k`, then the clock moves on.
    pub fn drive(&self, pass: &ForwardPass, body: impl Fn(&Tensor) + 'static) {
        let (step, timestep, ts) = (self.step, self.timestep, self.ts);
        pass.epilogue(move || {
            let k = step.take();
            body(&k);
            let next = &k + 1u32;
            timestep.take();
            timestep.put(at(&ts.read(), &next));
            step.put(&next);
        });
    }

    pub fn fire(&self, pass: &ForwardPass) -> Result<(), String> {
        pass.submit(&self.pipe)
    }

    pub fn close(&self) {
        self.pipe.close();
    }
}

/// DPM-Solver++ 2M over a flow schedule, as per-fire scalar tables indexed
/// by `DenoiseLoop`'s fire `k` (fire 0 is the seed and steps nowhere; fire
/// `k` steps from `sigmas[k-1]` to `sigmas[k]`). With `x0 = x - sigma_s * v`
/// the update is `x <- ratio * x + gain * D`, `D = (1 + mix) * x0 - mix *
/// x0_prev`; the first step and the step onto sigma 0 are first order.
#[derive(Clone, Debug, PartialEq)]
pub struct Dpm2m {
    pub sigma_s: Vec<f32>,
    pub ratio: Vec<f32>,
    pub gain: Vec<f32>,
    pub mix: Vec<f32>,
}

impl Dpm2m {
    #[must_use]
    pub fn of(sched: &FlowMatchEuler) -> Dpm2m {
        let s = &sched.sigmas;
        let lambda = |sigma: f32| ((1.0 - sigma).max(1e-12) / sigma.max(1e-12)).ln();
        let mut tables = Dpm2m {
            sigma_s: vec![s.first().copied().unwrap_or(1.0)],
            ratio: vec![0.0],
            gain: vec![0.0],
            mix: vec![0.0],
        };
        let mut h_prev: Option<f32> = None;
        for k in 1..s.len() {
            let (ss, st) = (s[k - 1], s[k]);
            let (alpha_s, alpha_t) = (1.0 - ss, 1.0 - st);
            let e = if st <= 0.0 {
                0.0
            } else {
                (st * alpha_s) / (ss * alpha_t)
            };
            let h = if st <= 0.0 {
                f32::INFINITY
            } else {
                lambda(st) - lambda(ss)
            };
            tables.sigma_s.push(ss);
            tables.ratio.push(if ss > 0.0 { st / ss } else { 0.0 });
            tables.gain.push(-alpha_t * (e - 1.0));
            tables.mix.push(match h_prev {
                Some(hp) if h.is_finite() && hp.is_finite() && hp != 0.0 => h / (2.0 * hp),
                _ => 0.0,
            });
            h_prev = Some(h);
        }
        tables
    }

    #[must_use]
    pub fn channels(&self) -> Dpm2mChannels {
        Dpm2mChannels {
            sigma_s: Channel::from(self.sigma_s.clone()),
            ratio: Channel::from(self.ratio.clone()),
            gain: Channel::from(self.gain.clone()),
            mix: Channel::from(self.mix.clone()),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Dpm2mChannels {
    pub sigma_s: Channel,
    pub ratio: Channel,
    pub gain: Channel,
    pub mix: Channel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_every_case() {
        a_flow_schedule_ends_at_zero_and_steps_downhill();
        a_shift_bends_the_sigmas_up();
        pinned_sigmas_are_taken_and_resampled();
        the_dynamic_shift_interpolates_mu();
    }

    fn a_flow_schedule_ends_at_zero_and_steps_downhill() {
        let fact = ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 1.0,
            train_steps: 1000,
            boundary: None,
            pinned_sigmas: Vec::new(),
            stream_shifts: Vec::new(),
        };
        let sched = FlowMatchEuler::from_schedule(&fact, 4, None).unwrap();
        assert_eq!(sched.sigmas, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
        assert_eq!(sched.steps(), 4);
        assert_eq!(sched.dt(0), -0.25);
        assert_eq!(sched.timestep(0), 1000.0);
        assert_eq!(sched.dts().len(), 4);
    }

    fn a_shift_bends_the_sigmas_up() {
        let fact = ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 3.0,
            train_steps: 1000,
            boundary: Some(0.875),
            pinned_sigmas: Vec::new(),
            stream_shifts: Vec::new(),
        };
        let sched = FlowMatchEuler::from_schedule(&fact, 4, None).unwrap();
        assert!(sched.sigmas[1] > 0.75, "{:?}", sched.sigmas);
        assert_eq!(sched.sigmas[0], 1.0);
        assert_eq!(*sched.sigmas.last().unwrap(), 0.0);
        assert!(sched.handover(0.875) >= 1);
    }

    fn pinned_sigmas_are_taken_and_resampled() {
        let fact = ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 1.0,
            train_steps: 1000,
            boundary: None,
            pinned_sigmas: vec![1.0, 0.5],
            stream_shifts: Vec::new(),
        };
        let sched = FlowMatchEuler::from_schedule(&fact, 2, None).unwrap();
        assert_eq!(sched.sigmas, vec![1.0, 0.5, 0.0]);
        let sched = FlowMatchEuler::from_schedule(&fact, 3, None).unwrap();
        assert_eq!(sched.sigmas, vec![1.0, 0.75, 0.5, 0.0]);
    }

    fn the_dynamic_shift_interpolates_mu() {
        assert!((dynamic_shift(256, 0.5) - 0.5f32.exp()).abs() < 1e-5);
        assert!((dynamic_shift(4096, 0.5) - 1.15f32.exp()).abs() < 1e-5);
    }
}
