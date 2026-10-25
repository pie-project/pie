#[derive(serde::Serialize)]
pub struct GenerativeFacts {
    pub latent: Option<Latent>,
    pub schedule: Option<Schedule>,
    /// The inputs the deployment's guest binds, `name[axis, width]`, and
    /// the switches it sets, `name = value|value|...`: traced, so only `pie
    /// model info` fills them.
    pub inputs: Vec<String>,
    pub switches: Vec<String>,
}

#[derive(serde::Serialize)]
pub struct Latent {
    pub channels: u32,
    pub patch_t: u32,
    pub patch_h: u32,
    pub patch_w: u32,
    pub spatial_compression: u32,
    pub temporal_compression: u32,
}

#[derive(serde::Serialize)]
pub struct Schedule {
    pub kind: String,
    pub shift: f32,
    pub train_steps: u32,
    pub boundary: Option<f32>,
    pub pinned_sigmas: Vec<f32>,
}

pub fn of(deployment: Option<&str>) -> Option<GenerativeFacts> {
    let deployment = runtime::catalog::catalog().parse(deployment?)?;
    let generative = deployment.try_generative().ok()??;
    Some(GenerativeFacts {
        latent: generative.latent.map(|l| Latent {
            channels: l.channels,
            patch_t: l.patch_t,
            patch_h: l.patch_h,
            patch_w: l.patch_w,
            spatial_compression: l.spatial_compression,
            temporal_compression: l.temporal_compression,
        }),
        schedule: generative.schedule.as_ref().map(|s| Schedule {
            kind: match s.kind {
                poem::generative::ScheduleKind::Flow => "flow",
                poem::generative::ScheduleKind::Epsilon => "epsilon",
                poem::generative::ScheduleKind::V => "v-prediction",
            }
            .to_string(),
            shift: s.shift,
            train_steps: s.train_steps,
            boundary: s.boundary,
            pinned_sigmas: s.pinned_sigmas.clone(),
        }),
        inputs: Vec::new(),
        switches: Vec::new(),
    })
}

/// `deployment`'s trace, to read what its guest binds and sets.
pub fn traced(deployment: &str) -> Option<poem::Trace> {
    let deployment = runtime::catalog::catalog().parse(deployment)?;
    deployment.try_trace(poem::Platform::Cuda).ok()
}

/// The switches `trace`'s model declares, `name = value|value|...`, the
/// first value its default.
pub fn switches(trace: &poem::Trace) -> Vec<String> {
    trace
        .facts
        .switches
        .iter()
        .map(|decl| format!("{} = {}", decl.name, decl.values.join("|")))
        .collect()
}

/// The inputs `trace` reads from its guest, `name[axis, width]`.
pub fn inputs(trace: &poem::Trace) -> Vec<String> {
    trace
        .values
        .iter()
        .filter_map(|decl| match decl.def {
            poem::Def::Input(poem::RuntimeInput::Guest {
                id, axis, width, ..
            }) => trace
                .inputs
                .get(id as usize)
                .map(|input| format!("{}[{}, {width}]", input.name, axis.name())),
            _ => None,
        })
        .collect()
}

impl GenerativeFacts {
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(latent) = &self.latent {
            parts.push(format!("latent {}", latent.line()));
        }
        if let Some(schedule) = &self.schedule {
            parts.push(schedule.line());
        }
        parts.join(" · ")
    }

    /// Whether the deployment binds the standard `latents` and `timestep`
    /// inputs `text-to-image` drives (known once traced).
    pub fn text_to_image(&self) -> bool {
        ["latents[", "timestep["]
            .iter()
            .all(|name| self.inputs.iter().any(|input| input.starts_with(name)))
    }
}

impl Latent {
    pub fn line(&self) -> String {
        let patch = if self.patch_t > 1 {
            format!("patch {}x{}x{}", self.patch_t, self.patch_h, self.patch_w)
        } else {
            format!("patch {}x{}", self.patch_h, self.patch_w)
        };
        let temporal = if self.temporal_compression > 1 {
            format!(", /{} frames", self.temporal_compression)
        } else {
            String::new()
        };
        format!(
            "{} ch, {}, /{} px{temporal}",
            self.channels, patch, self.spatial_compression
        )
    }

    pub fn pixels_per_row(&self) -> (u32, u32) {
        (
            self.patch_w * self.spatial_compression,
            self.patch_h * self.spatial_compression,
        )
    }
}

impl Schedule {
    pub fn line(&self) -> String {
        let mut line = format!("{}, shift {:.2}", self.kind, self.shift);
        if !self.pinned_sigmas.is_empty() {
            line.push_str(&format!(", {} pinned sigmas", self.pinned_sigmas.len()));
        }
        if let Some(boundary) = self.boundary {
            line.push_str(&format!(", handover at sigma {boundary:.3}"));
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLUX2_KLEIN: &str = "flux2-klein-4b-bf16-kv-bf16";

    #[test]
    fn facts_every_case() {
        a_generative_row_reports_what_it_can_be_asked_to_draw();
        a_text_row_reports_nothing_here();
    }

    fn a_generative_row_reports_what_it_can_be_asked_to_draw() {
        let mut facts = of(Some(FLUX2_KLEIN)).expect("flux2-klein-4b is a generative row");
        let trace = traced(FLUX2_KLEIN).expect("flux2-klein-4b traces");
        facts.inputs = inputs(&trace);
        facts.switches = switches(&trace);
        assert!(facts.text_to_image(), "{:?}", facts.inputs);
        assert!(
            facts
                .switches
                .contains(&"reading = text|denoise|vae.decode|vae.encode".to_string()),
            "{:?}",
            facts.switches
        );
        let latent = facts.latent.expect("a denoiser has a latent space");
        assert_eq!(latent.pixels_per_row(), (16, 16));
        let schedule = facts.schedule.expect("a denoiser has a schedule");
        assert_eq!(schedule.kind, "flow");
        assert!(
            facts.inputs.contains(&"latents[row, 128]".to_string()),
            "{:?}",
            facts.inputs
        );
    }

    fn a_text_row_reports_nothing_here() {
        let text = runtime::catalog::deployments()
            .find(|deployment| deployment.try_generative().ok().flatten().is_none())
            .expect("the catalog ships text rows");
        assert!(of(Some(&text.name)).is_none());
        assert!(of(None).is_none());
        assert!(of(Some("a row this build does not ship")).is_none());
    }
}
