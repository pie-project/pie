use crate::eta::{
    Channel, KvGeometry, Pipeline, WorkingSet, attention::ForwardPass, broadcast, eq, gather,
    intrinsics, max_elem, min_elem, normal, reduce_sum, reshape, select, sqrt,
};
use crate::model::{AxisRole, PositionConvention, ReadoutKind, ScheduleFact, ScheduleKind};
use eta_dsl::Tensor;

pub mod prelude {
    pub use super::{
        DenoiseLoop, FlowMatchEuler, LaneClock, LaneRows, apg, at, cfg_combine, dynamic_shift,
        encode_ids, encode_ids_rows, encode_text, euler_step, guided_velocity, noise,
        positions_for, positions_grid, resume_or_step, rng_state, seed_or_step,
    };
    pub use crate::eta::attention::prelude::*;
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

    pub fn for_stream(
        fact: &ScheduleFact,
        lane: crate::model::LaneStream,
        steps: u32,
        rows: Option<u32>,
    ) -> Result<FlowMatchEuler, String> {
        let mut fact = fact.clone();
        if let Some(found) = fact
            .stream_shifts
            .iter()
            .find(|shift| shift.lane == lane)
            .map(|shift| shift.shift)
        {
            fact.shift = found;
            fact.pinned_sigmas.clear();
        }
        FlowMatchEuler::from_schedule(&fact, steps, rows)
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

#[must_use]
pub fn guided_velocity(width: u32, s: f32, conditional: bool) -> Tensor {
    let own = intrinsics::velocity(width);
    let peer = intrinsics::peer_velocity(width);
    let scale = Tensor::constant([s]);
    if conditional {
        cfg_combine(&own, &peer, &scale)
    } else {
        cfg_combine(&peer, &own, &scale)
    }
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

#[must_use]
pub fn positions_grid(t: u32, h: u32, w: u32, offsets: [f32; 3]) -> Vec<f32> {
    let mut grid = Vec::with_capacity((t * h * w * 3) as usize);
    for i in 0..t {
        for j in 0..h {
            for k in 0..w {
                grid.push(i as f32 + offsets[0]);
                grid.push(j as f32 + offsets[1]);
                grid.push(k as f32 + offsets[2]);
            }
        }
    }
    grid
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRows {
    Sequence(u32),
    Grid { h: u32, w: u32 },
    Volume { t: u32, h: u32, w: u32 },
    Reference { index: u32, h: u32, w: u32 },
}

#[must_use]
pub fn positions_for(convention: &PositionConvention, lane: LaneRows, text_rows: u32) -> Vec<f32> {
    let axes = convention.axes.len();
    let text_axis = convention.text_axis as usize;
    let origin = convention.text_origin as f32;
    let mut time_offset = 0f32;
    let (t, h, w) = match lane {
        LaneRows::Sequence(rows) => {
            let mut grid = vec![0f32; rows as usize * axes];
            for j in 0..rows as usize {
                if let Some(cell) = grid.get_mut(j * axes + text_axis) {
                    *cell = origin + j as f32;
                }
            }
            return grid;
        }
        LaneRows::Grid { h, w } => (1, h, w),
        LaneRows::Volume { t, h, w } => (t, h, w),
        LaneRows::Reference { index, h, w } => {
            time_offset = (convention.reference_stride.unwrap_or(0) * (index + 1)) as f32;
            (1, h, w)
        }
    };
    let follow = origin + text_rows as f32;
    let mut grid = Vec::with_capacity((t * h * w) as usize * axes);
    for i in 0..t {
        for a in 0..h {
            for b in 0..w {
                for (axis, role) in convention.axes.iter().enumerate() {
                    grid.push(if axis == text_axis && convention.image_follows_text {
                        follow
                    } else {
                        match role {
                            AxisRole::Time => time_offset + i as f32,
                            AxisRole::Height => a as f32,
                            AxisRole::Width => b as f32,
                            AxisRole::Index => 0.0,
                        }
                    });
                }
            }
        }
    }
    grid
}

#[derive(Clone)]
pub struct LaneClock {
    pub step: Channel,
    pub timestep: Channel,
    ts: Channel,
}

impl LaneClock {
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
}

pub struct DenoiseLoop {
    pipes: Vec<Pipeline>,
    dts: Vec<f32>,
    ts: Vec<f32>,
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
            pipes: Vec::new(),
            dts,
            ts,
            fires: sched.steps() + 1,
        }
    }

    #[must_use]
    pub fn fires(&self) -> u32 {
        self.fires
    }

    pub fn lane(&mut self, tag: &str) -> LaneClock {
        self.pipes.push(Pipeline::new());
        LaneClock {
            step: Channel::from([0u32]).named(&format!("{tag}_step")),
            timestep: Channel::from([self.ts[0]]).named(&format!("{tag}_t")),
            ts: Channel::from(self.ts.clone()).named(&format!("{tag}_ts")),
        }
    }

    #[must_use]
    pub fn dts(&self, tag: &str) -> Channel {
        Channel::from(self.dts.clone()).named(&format!("{tag}_dts"))
    }

    pub fn fire(&self, passes: &[&ForwardPass]) -> Result<(), String> {
        if passes.len() != self.pipes.len() {
            return Err(format!(
                "this loop has {} lanes and {} passes were submitted; every lane of a group must \
                 fire in the same step",
                self.pipes.len(),
                passes.len()
            ));
        }
        for (i, pass) in passes.iter().enumerate() {
            pass.submit(&self.pipes[i])
                .map_err(|why| format!("lane {i}: {why}"))?;
        }
        Ok(())
    }

    pub fn close(&self) {
        for pipe in &self.pipes {
            pipe.close();
        }
    }
}

pub fn seed_or_step(
    k: &Tensor,
    x: &Channel,
    velocity: &Tensor,
    dts: &Channel,
    rng: &Channel,
    shape: [u32; 2],
    out: Option<&Channel>,
) {
    let current = x.take();
    let stepped = euler_step(&current, velocity, &at(&dts.read(), k));
    let state = rng.take();
    let fresh = noise(shape, &state);
    let first = broadcast(reshape(eq(k, 0u32), [1, 1]), shape);
    let next = select(&first, &fresh, &stepped);
    x.put(&next);
    if let Some(out) = out {
        out.put(&next);
    }
    rng.put(&state + &Tensor::constant([0u32, 1u32]));
}

#[allow(clippy::too_many_arguments)]
pub fn resume_or_step(
    k: &Tensor,
    x: &Channel,
    init: &Channel,
    sigma0: f32,
    velocity: &Tensor,
    dts: &Channel,
    rng: &Channel,
    shape: [u32; 2],
    out: Option<&Channel>,
) {
    let current = x.take();
    let stepped = euler_step(&current, velocity, &at(&dts.read(), k));
    let state = rng.take();
    let eps = noise(shape, &state);
    let x0 = init.read();
    let a = broadcast(Tensor::constant([1.0 - sigma0]), shape);
    let b = broadcast(Tensor::constant([sigma0]), shape);
    let noised = &(&x0 * &a) + &(&eps * &b);
    let first = broadcast(reshape(eq(k, 0u32), [1, 1]), shape);
    let next = select(&first, &noised, &stepped);
    x.put(&next);
    if let Some(out) = out {
        out.put(&next);
    }
    rng.put(&state + &Tensor::constant([0u32, 1u32]));
}

pub async fn encode_text(prompt: &str, reading: &str) -> Result<Channel, String> {
    let Some(fact) = crate::model::reading(reading) else {
        return Err(format!("this model declares no reading `{reading}`"));
    };
    if !fact.takes_tokens {
        return Err(format!(
            "reading `{reading}` is not a text encoder (it embeds no tokens)"
        ));
    }
    let mut ids = crate::chat::first_user(prompt);
    ids.extend(crate::chat::cue());
    if ids.is_empty() {
        return Err("the prompt tokenizes to nothing".to_string());
    }
    encode_ids(&ids, reading).await
}

pub async fn encode_ids(ids: &[u32], reading: &str) -> Result<Channel, String> {
    let (rows, width) = encode_ids_rows(ids, reading).await?;
    let len = u32::try_from(rows.len() / width.max(1) as usize).unwrap_or(0);
    Ok(Channel::from_shaped([len, width], rows))
}

pub async fn encode_ids_rows(ids: &[u32], reading: &str) -> Result<(Vec<f32>, u32), String> {
    let Some(fact) = crate::model::reading(reading) else {
        return Err(format!("this model declares no reading `{reading}`"));
    };
    if !fact.takes_tokens {
        return Err(format!(
            "reading `{reading}` embeds no tokens, so there is nothing to hand ids to"
        ));
    }
    if fact.readout != ReadoutKind::Hidden {
        return Err(format!(
            "reading `{reading}` reads out {:?}, not the hidden rows an encoder hands over",
            fact.readout
        ));
    }
    let width = fact.readout_width;
    let cap = crate::eta::max_embed_length().max(1);
    let ids: Vec<i32> = ids
        .iter()
        .take(cap)
        .map(|id| i32::try_from(*id).unwrap_or(0))
        .collect();
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    if len == 0 {
        return Err("no ids to encode".to_string());
    }
    let pipe = Pipeline::new();
    let toks = Channel::from(ids);
    let embed_indptr = Channel::from([0u32, len]);
    let readout = Channel::from_iter(0..len);
    let out = Channel::new([len, width], crate::eta::Dtype::F32);

    let pass = crate::eta::attention::ForwardPass::new();
    pass.reading(reading)?;
    pass.embed(&toks, &embed_indptr)?;
    pass.readout(&readout)?;
    let ws = WorkingSet::new();
    if fact.has_kv {
        let page_size = crate::eta::kv_page_size().max(1);
        let pages = len.div_ceil(page_size);
        ws.reserve(pages)?;
        let positions = Channel::from_iter(0..len);
        let page_ids = Channel::from_iter(0..pages);
        let page_indptr = Channel::from([0u32, pages]);
        let w_slot = Channel::from_iter((0..len).map(|p| p / page_size));
        let w_off = Channel::from_iter((0..len).map(|p| p % page_size));
        let kv_len = Channel::from([len]);
        pass.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len,
                pages: &page_ids,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
    }
    let readback = out;
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(width));
    });
    pass.submit(&pipe)?;
    let rows: Vec<f32> = out.take_host().await?;
    pipe.close();
    Ok((rows, width))
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
        a_positions_grid_is_t_major_with_offsets();
        a_grid_then_index_convention_numbers_text_on_its_own_axis();
        a_text_time_prefix_convention_stacks_the_image_behind_the_caption();
        a_reference_grid_rides_its_own_time_offset();
        a_grid_only_convention_leaves_the_time_axis_at_zero();
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

    fn a_positions_grid_is_t_major_with_offsets() {
        let grid = positions_grid(1, 2, 2, [10.0, 0.0, 0.0]);
        assert_eq!(
            grid,
            vec![
                10.0, 0.0, 0.0, 10.0, 0.0, 1.0, 10.0, 1.0, 0.0, 10.0, 1.0, 1.0
            ]
        );
    }

    fn a_grid_then_index_convention_numbers_text_on_its_own_axis() {
        let flux = PositionConvention {
            axes: vec![
                AxisRole::Time,
                AxisRole::Height,
                AxisRole::Width,
                AxisRole::Index,
            ],
            text_axis: 3,
            text_origin: 0,
            image_follows_text: false,
            reference_stride: Some(10),
        };
        assert_eq!(
            positions_for(&flux, LaneRows::Sequence(2), 2),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            positions_for(&flux, LaneRows::Grid { h: 2, w: 2 }, 2),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            ]
        );
    }

    fn a_text_time_prefix_convention_stacks_the_image_behind_the_caption() {
        let z = PositionConvention {
            axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
            text_axis: 0,
            text_origin: 1,
            image_follows_text: true,
            reference_stride: None,
        };
        assert_eq!(
            positions_for(&z, LaneRows::Sequence(2), 2),
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
        );
        assert_eq!(
            positions_for(&z, LaneRows::Grid { h: 1, w: 2 }, 32),
            vec![33.0, 0.0, 0.0, 33.0, 0.0, 1.0]
        );
    }

    fn a_reference_grid_rides_its_own_time_offset() {
        let flux = PositionConvention {
            axes: vec![
                AxisRole::Time,
                AxisRole::Height,
                AxisRole::Width,
                AxisRole::Index,
            ],
            text_axis: 3,
            text_origin: 0,
            image_follows_text: false,
            reference_stride: Some(10),
        };
        assert_eq!(
            positions_for(
                &flux,
                LaneRows::Reference {
                    index: 0,
                    h: 1,
                    w: 2
                },
                7
            ),
            vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0,]
        );
        assert_eq!(
            positions_for(
                &flux,
                LaneRows::Reference {
                    index: 1,
                    h: 1,
                    w: 2
                },
                7
            ),
            vec![20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0,]
        );
        let silent = PositionConvention {
            reference_stride: None,
            ..flux.clone()
        };
        assert_eq!(
            positions_for(
                &silent,
                LaneRows::Reference {
                    index: 0,
                    h: 1,
                    w: 2
                },
                7
            ),
            positions_for(&silent, LaneRows::Grid { h: 1, w: 2 }, 7)
        );
    }

    fn a_grid_only_convention_leaves_the_time_axis_at_zero() {
        let mini = PositionConvention {
            axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
            text_axis: 0,
            text_origin: 0,
            image_follows_text: false,
            reference_stride: None,
        };
        assert_eq!(
            positions_for(&mini, LaneRows::Sequence(2), 2),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            positions_for(&mini, LaneRows::Grid { h: 2, w: 1 }, 2),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }
}
