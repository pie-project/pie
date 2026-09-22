pub mod decode;
pub mod multimodal;

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use models::media::{
    AudioFrontEnd, Budget, Delimiters, EncodedSpan, Fault, Grid, Rgb8, VisionFrontEnd,
};
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

struct AudioAdapter {
    arch_name: &'static str,
}

impl AudioFrontEnd for AudioAdapter {
    fn arch(&self) -> &'static str {
        self.arch_name
    }

    fn delimiters(&self) -> Delimiters {
        let (prefix, suffix) = multimodal::audio_delimiters(self.arch_name);
        Delimiters {
            prefix,
            placeholder: multimodal::audio_placeholder(),
            suffix,
        }
    }

    fn encode_audio(&self, bytes: &[u8]) -> models::media::Result<EncodedSpan> {
        let (mel, n_frames) = multimodal::audio::process_wav_bytes(bytes).map_err(Fault::Decode)?;
        if n_frames == 0 {
            return Err(Fault::Empty("audio: clip decoded to zero frames".into()));
        }
        let n_frames = n_frames as u32;
        let token_count = multimodal::gemma_audio_token_count(n_frames);
        if token_count == 0 {
            return Err(Fault::Empty(
                "audio: clip is shorter than one soft token".into(),
            ));
        }
        Ok(EncodedSpan {
            token_count,
            position_span: token_count,
            grid: Grid::still(1, token_count),
            patch_grid: Grid::still(1, n_frames),
            uses_mrope: false,
            payload: mel,
            rows: n_frames,
            positions: Vec::new(),
            embed_rows: Vec::new(),
            embed_weights: Vec::new(),
            prefix: Vec::new(),
            placeholder: 0,
            suffix: Vec::new(),
        })
    }
}

fn vision_front_end() -> models::media::Result<Box<dyn VisionFrontEnd>> {
    let m = crate::model::model();
    let arch = m.arch_name();
    models::media::vision_front_end(arch).ok_or_else(|| Fault::NoVisionFrontEnd {
        model: m.name().to_string(),
        arch: arch.to_string(),
    })
}

fn audio_front_end() -> models::media::Result<AudioAdapter> {
    let m = crate::model::model();
    let arch = m.arch_name();
    if multimodal::audio_arch_supported(arch) {
        Ok(AudioAdapter { arch_name: arch })
    } else {
        Err(Fault::NoAudioFrontEnd {
            model: m.name().to_string(),
            arch: arch.to_string(),
        })
    }
}

fn spell(span: &mut EncodedSpan, delims: Delimiters) -> Result<(), String> {
    let encode = |s: &str| -> Vec<u32> {
        if s.is_empty() {
            Vec::new()
        } else {
            crate::model::model().tokenize(s)
        }
    };
    let pad = encode(delims.placeholder);
    let [placeholder] = pad[..] else {
        return Err(format!(
            "MediaSpelling: this model's tokenizer spells the placeholder \
             '{}' as {} tokens; a media run is one reserved id repeated, so a \
             span it cannot spell has no run the submission scan could match",
            delims.placeholder,
            pad.len()
        ));
    };
    span.spell_with(encode(delims.prefix), placeholder, encode(delims.suffix));
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
    for f in &span.payload {
        h.update(&f.to_le_bytes());
    }
    for p in &span.positions {
        h.update(&p.to_le_bytes());
    }
    for r in &span.embed_rows {
        h.update(&r.to_le_bytes());
    }
    for w in &span.embed_weights {
        h.update(&w.to_le_bytes());
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
        let front_end = match vision_front_end() {
            Ok(fe) => fe,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        let rgb = match decode::decode(&bytes) {
            Ok(rgb) => rgb,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        let mut span = match front_end.encode(&rgb, Budget::Still, decode::resize_exact) {
            Ok(span) => span,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        if let Err(refusal) = spell(&mut span, front_end.delimiters()) {
            return Ok(Err(refusal));
        }
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
        let front_end = match vision_front_end() {
            Ok(fe) => fe,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        let mut decoded = match media::gif::frames(&bytes) {
            Ok(f) => f,
            Err(e) => return Ok(Err(Fault::Decode(e).to_string())),
        };
        let delims = front_end.delimiters();
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
            let mut span = match front_end.encode(&frame, Budget::VideoFrame, decode::resize_exact)
            {
                Ok(span) => span,
                Err(fault) => return Ok(Err(fault.to_string())),
            };
            if let Err(refusal) = spell(&mut span, delims) {
                return Ok(Err(refusal));
            }
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
    async fn from_bytes(&mut self, bytes: Vec<u8>) -> Result<Result<Resource<Audio>, String>> {
        let adapter = match audio_front_end() {
            Ok(fe) => fe,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        let front_end: &dyn AudioFrontEnd = &adapter;
        let mut span = match front_end.encode_audio(&bytes) {
            Ok(span) => span,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        if let Err(refusal) = spell(&mut span, front_end.delimiters()) {
            return Ok(Err(refusal));
        }
        let audio = Audio {
            span: Arc::new(span),
        };
        Ok(Ok(self.ctx().table.push(audio)?))
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
