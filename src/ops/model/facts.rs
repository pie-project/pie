#[derive(serde::Serialize)]
pub struct GenerativeFacts {
    pub readings: Vec<Reading>,
    pub latent: Option<Latent>,
    pub schedule: Option<Schedule>,
    pub max_latent_rows: u32,
}

#[derive(serde::Serialize)]
pub struct Reading {
    pub name: String,
    pub index: u8,
    pub kv: bool,
    pub tokens: bool,
    pub streams: Vec<String>,
    pub ports: Vec<String>,
    pub readout: String,
    pub readout_width: u32,
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

pub fn of(sku: Option<&str>) -> Option<GenerativeFacts> {
    let generative = models::sku(sku?)?.generative.as_ref()?;
    Some(GenerativeFacts {
        readings: generative.readings.iter().map(reading).collect(),
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
                models::ScheduleKind::Flow => "flow",
                models::ScheduleKind::Epsilon => "epsilon",
                models::ScheduleKind::V => "v-prediction",
            }
            .to_string(),
            shift: s.shift,
            train_steps: s.train_steps,
            boundary: s.boundary,
            pinned_sigmas: s.pinned_sigmas.clone(),
        }),
        max_latent_rows: generative.max_rows,
    })
}

fn reading(fact: &models::ReadingFact) -> Reading {
    Reading {
        name: fact.name.to_string(),
        index: fact.index,
        kv: fact.has_kv,
        tokens: fact.takes_tokens,
        streams: fact
            .streams
            .iter()
            .map(|s| stream(*s).to_string())
            .collect(),
        ports: fact
            .ports
            .iter()
            .map(|port| format!("{}[{}]", port.name, port.width))
            .collect(),
        readout: match fact.readout {
            models::ReadoutKind::Logits => "logits",
            models::ReadoutKind::Velocity => "velocity",
            models::ReadoutKind::Hidden => "hidden",
            models::ReadoutKind::Pixels => "pixels",
        }
        .to_string(),
        readout_width: fact.readout_width,
    }
}

fn stream(stream: models::Stream) -> &'static str {
    match stream {
        models::Stream::Text => "text",
        models::Stream::Image => "image",
        models::Stream::Video => "video",
        models::Stream::Audio => "audio",
        models::Stream::Context => "context",
        models::Stream::Reference => "reference",
    }
}

impl GenerativeFacts {
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("{} readings", self.readings.len())];
        if let Some(latent) = &self.latent {
            parts.push(format!("latent {}", latent.line()));
        }
        if let Some(schedule) = &self.schedule {
            parts.push(schedule.line());
        }
        parts.join(" · ")
    }

    pub fn text_to_image(&self) -> bool {
        self.readings
            .iter()
            .any(|r| r.tokens && r.readout == "hidden")
            && self
                .readings
                .iter()
                .any(|r| !r.tokens && r.readout == "velocity")
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
        let facts = of(Some(FLUX2_KLEIN)).expect("flux2-klein-4b is a generative row");
        assert!(
            facts.text_to_image(),
            "klein-4B has a text reading and a denoise reading: {:?}",
            facts
                .readings
                .iter()
                .map(|r| (&r.name, &r.readout, r.tokens))
                .collect::<Vec<_>>()
        );
        let latent = facts.latent.expect("a denoiser has a latent space");
        assert_eq!(latent.pixels_per_row(), (16, 16));
        let schedule = facts.schedule.expect("a denoiser has a schedule");
        assert_eq!(schedule.kind, "flow");
        assert!(facts.max_latent_rows > 0);
    }

    fn a_text_row_reports_nothing_here() {
        let text = models::skus()
            .find(|sku| sku.generative.is_none())
            .expect("the catalog ships text rows");
        assert!(of(Some(&text.name)).is_none());
        assert!(of(None).is_none());
        assert!(of(Some("a row this build does not ship")).is_none());
    }
}
