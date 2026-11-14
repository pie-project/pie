#[derive(Debug, Clone, PartialEq)]
pub struct Generative {
    pub latent: Option<LatentSpace>,
    pub schedule: Option<ScheduleFact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentSpace {
    pub channels: u32,
    pub spatial_compression: u32,
    pub temporal_compression: u32,
    pub first_frames: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleFact {
    pub kind: ScheduleKind,
    pub shift: f32,
    pub train_steps: u32,
    pub boundary: Option<f32>,
    pub pinned_sigmas: Vec<f32>,
    /// The shift a lane whose `stream` switch takes `value` advances at.
    pub stream_shifts: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleKind {
    Flow,
    Epsilon,
    V,
}

impl ScheduleFact {
    /// A flow schedule's sigmas for `steps` steps over `rows` image rows, then 0: the pinned
    /// sigmas resampled to `steps`, else `1 - i / steps` under the (dynamic) shift. What a guest's
    /// `FlowMatchEuler::from_schedule` integrates over (inferlet `latent.rs`); a host loop (a
    /// device runner's) takes the same. Step `i` feeds timestep `sigmas[i] · train_steps` and
    /// moves the latent by `(sigmas[i + 1] - sigmas[i]) · velocity`.
    pub fn flow_sigmas(&self, steps: u32, rows: Option<u32>) -> Result<Vec<f32>, String> {
        if self.kind != ScheduleKind::Flow {
            return Err(format!(
                "a flow schedule integrates; this one is {:?}",
                self.kind
            ));
        }
        let steps = steps.max(1);
        let mut sigmas: Vec<f32> = if self.pinned_sigmas.is_empty() {
            let shift = match rows {
                Some(rows) if self.shift > 0.0 => dynamic_shift(rows, self.shift),
                _ => self.shift,
            };
            (0..steps)
                .map(|i| {
                    let sigma = 1.0 - i as f32 / steps as f32;
                    if shift <= 0.0 || shift == 1.0 {
                        sigma
                    } else {
                        shift * sigma / (1.0 + (shift - 1.0) * sigma)
                    }
                })
                .collect()
        } else {
            let pinned = &self.pinned_sigmas;
            if pinned.len() == steps as usize || pinned.len() < 2 {
                pinned.clone()
            } else {
                (0..steps as usize)
                    .map(|i| {
                        let at = i as f32 * (pinned.len() - 1) as f32 / (steps.max(2) - 1) as f32;
                        let lo = at.floor() as usize;
                        let hi = (lo + 1).min(pinned.len() - 1);
                        let frac = at - lo as f32;
                        pinned[lo] * (1.0 - frac) + pinned[hi] * frac
                    })
                    .collect()
            }
        };
        sigmas.push(0.0);
        Ok(sigmas)
    }
}

/// The shift of a schedule's base `mu` at `rows` image rows (inferlet `dynamic_shift`).
fn dynamic_shift(rows: u32, base_mu: f32) -> f32 {
    const BASE_ROWS: f32 = 256.0;
    const MAX_ROWS: f32 = 4096.0;
    const MAX_MU: f32 = 1.15;
    let slope = (MAX_MU - base_mu) / (MAX_ROWS - BASE_ROWS);
    (slope * rows as f32 + (base_mu - slope * BASE_ROWS)).exp()
}
