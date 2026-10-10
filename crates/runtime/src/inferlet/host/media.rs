pub mod decode;

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use media::front::{Budget, EncodedSpan, Fault, Rgb8};
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Clone)]
pub struct Image {
    pub span: Arc<EncodedSpan>,
}

pub struct Video {
    pub frames: Vec<Image>,
    pub timestamps: Vec<f32>,
}

#[derive(Clone)]
pub struct Audio {
    pub span: Arc<EncodedSpan>,
}

/// A still `rgb` as this model reads it under `budget`: framed by the
/// package, cut, and spelled.
fn encode(rgb: &Rgb8, budget: Budget) -> Result<EncodedSpan, String> {
    let m = crate::model::model();
    let Some(spec) = m.image() else {
        return Err(Fault::NoVisionFrontEnd {
            model: m.name().to_string(),
        }
        .to_string());
    };
    let framed = m
        .frame(rgb.h, rgb.w, budget.word())
        .map_err(|why| Fault::Frame(format!("{why:#}")).to_string())?;
    let image = media::front::Image {
        patch: spec.patch,
        block: spec.block,
        mrope: spec.mrope,
    };
    let mut span = image
        .encode(rgb, framed, decode::resize_exact)
        .map_err(|fault| fault.to_string())?;
    spell(&mut span, &spec.prefix, &spec.placeholder, &spec.suffix)?;
    Ok(span)
}

fn spell(
    span: &mut EncodedSpan,
    prefix: &str,
    placeholder: &str,
    suffix: &str,
) -> Result<(), String> {
    let encode = |s: &str| -> Vec<u32> {
        if s.is_empty() {
            Vec::new()
        } else {
            crate::model::model().tokenize(s)
        }
    };
    let pad = encode(placeholder);
    let [id] = pad[..] else {
        return Err(format!(
            "MediaSpelling: this model's tokenizer spells the placeholder \
             '{placeholder}' as {} tokens; a media run is one reserved id repeated, so a \
             span it cannot spell has no run the submission scan could match",
            pad.len()
        ));
    };
    span.spell_with(encode(prefix), id, encode(suffix));
    Ok(())
}

#[must_use]
pub fn span_digest(span: &EncodedSpan) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    h.update(b"pie:media-span:v1");
    for n in [
        span.token_count,
        span.position_span,
        span.rows,
        span.grid.t,
        span.grid.h,
        span.grid.w,
        span.patch_grid.t,
        span.patch_grid.h,
        span.patch_grid.w,
    ] {
        h.update(&n.to_le_bytes());
    }
    h.update(&[u8::from(span.uses_mrope)]);
    h.update(&span.payload);
    for p in &span.positions {
        h.update(&p.to_le_bytes());
    }
    h.finalize().as_bytes().to_vec()
}

fn sample_indices(n: usize, max_frames: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let k = max_frames.clamp(1, n);
    if k == 1 {
        return vec![0];
    }
    if k >= n {
        return (0..n).collect();
    }
    (0..k).map(|i| i * (n - 1) / (k - 1)).collect()
}

impl pie::inferlet::media::Host for ProcessCtx {}

impl pie::inferlet::media::HostImage for ProcessCtx {
    async fn from_bytes(&mut self, bytes: Vec<u8>) -> Result<Result<Resource<Image>, String>> {
        let rgb = match decode::decode(&bytes) {
            Ok(rgb) => rgb,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        let span = match encode(&rgb, Budget::Still) {
            Ok(span) => span,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let image = Image {
            span: Arc::new(span),
        };
        Ok(Ok(self.ctx().table.push(image)?))
    }

    async fn tokens(&mut self, this: Resource<Image>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.span.tokens())
    }

    async fn digest(&mut self, this: Resource<Image>) -> Result<Vec<u8>> {
        Ok(span_digest(&self.ctx().table.get(&this)?.span))
    }

    async fn token_count(&mut self, this: Resource<Image>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.span.token_count)
    }

    async fn position_span(&mut self, this: Resource<Image>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.span.position_span)
    }

    async fn grid(&mut self, this: Resource<Image>) -> Result<pie::inferlet::media::MergedGrid> {
        let g = self.ctx().table.get(&this)?.span.grid;
        Ok(pie::inferlet::media::MergedGrid {
            t: g.t,
            h: g.h,
            w: g.w,
        })
    }

    async fn prefix_tokens(&mut self, this: Resource<Image>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.span.prefix.clone())
    }

    async fn suffix_tokens(&mut self, this: Resource<Image>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.span.suffix.clone())
    }

    async fn drop(&mut self, this: Resource<Image>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::media::HostVideo for ProcessCtx {
    async fn from_bytes(
        &mut self,
        bytes: Vec<u8>,
        max_frames: u32,
    ) -> Result<Result<Resource<Video>, String>> {
        let mut decoded = match media::gif::frames(&bytes) {
            Ok(f) => f,
            Err(e) => return Ok(Err(Fault::Decode(e).to_string())),
        };
        let sel = sample_indices(decoded.len(), max_frames as usize);
        let mut frames = Vec::with_capacity(sel.len());
        let mut timestamps = Vec::with_capacity(sel.len());
        for &i in &sel {
            let picture = &mut decoded[i];
            let (height, width, timestamp) = (picture.height, picture.width, picture.timestamp);
            let frame = match Rgb8::new(height, width, std::mem::take(&mut picture.rgb)) {
                Ok(frame) => frame,
                Err(fault) => return Ok(Err(fault.to_string())),
            };
            let span = match encode(&frame, Budget::VideoFrame) {
                Ok(span) => span,
                Err(refusal) => return Ok(Err(refusal)),
            };
            frames.push(Image {
                span: Arc::new(span),
            });
            timestamps.push(timestamp);
        }
        let video = Video { frames, timestamps };
        Ok(Ok(self.ctx().table.push(video)?))
    }

    async fn frame_count(&mut self, this: Resource<Video>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.frames.len() as u32)
    }

    async fn frame(
        &mut self,
        this: Resource<Video>,
        index: u32,
    ) -> Result<Result<Resource<Image>, String>> {
        let img = {
            let v = self.ctx().table.get(&this)?;
            match v.frames.get(index as usize) {
                Some(f) => f.clone(),
                None => {
                    return Ok(Err(format!(
                        "video frame index {index} out of range ({} frames)",
                        v.frames.len()
                    )));
                }
            }
        };
        Ok(Ok(self.ctx().table.push(img)?))
    }

    async fn timestamp(&mut self, this: Resource<Video>, index: u32) -> Result<f32> {
        Ok(self
            .ctx()
            .table
            .get(&this)?
            .timestamps
            .get(index as usize)
            .copied()
            .unwrap_or(0.0))
    }

    async fn drop(&mut self, this: Resource<Video>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::media::HostAudio for ProcessCtx {
    async fn from_bytes(&mut self, _bytes: Vec<u8>) -> Result<Result<Resource<Audio>, String>> {
        Ok(Err(Fault::NoAudioFrontEnd {
            model: crate::model::model().name().to_string(),
        }
        .to_string()))
    }

    async fn tokens(&mut self, this: Resource<Audio>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.span.tokens())
    }

    async fn digest(&mut self, this: Resource<Audio>) -> Result<Vec<u8>> {
        Ok(span_digest(&self.ctx().table.get(&this)?.span))
    }

    async fn token_count(&mut self, this: Resource<Audio>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.span.token_count)
    }

    async fn position_span(&mut self, this: Resource<Audio>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.span.position_span)
    }

    async fn prefix_tokens(&mut self, this: Resource<Audio>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.span.prefix.clone())
    }

    async fn suffix_tokens(&mut self, this: Resource<Audio>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.span.suffix.clone())
    }

    async fn drop(&mut self, this: Resource<Audio>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
