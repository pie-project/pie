//! pie:core/media — Image / Video / Audio resources for multimodal input.
//!
//! **Model-agnostic by construction.** The inferlet hands the host raw encoded
//! bytes (a PNG/JPEG, an animated GIF, a WAV); everything model-specific —
//! decode, resize, patchify, normalize, log-mel, video frame-sampling, and the
//! model's own span delimiters — happens here, dispatched off the bound model's
//! arch (`multimodal::Processor` / `multimodal::audio`). The wire payload
//! (`pixels`/`positions`/`patch_grid`/`mel`) is byte-identical to what the old
//! inferlet-side `from_pixels`/`from_mel` produced, so the bridge + driver are
//! unchanged — only the *source* of the bytes moved host-side. See MULTIMODAL.md.

use crate::api::pie;
use crate::instance::InstanceState;
use crate::multimodal::{self, Processor, VisionArch};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Image resource — a preprocessed still image, also used for one video frame.
#[derive(Clone)]
pub struct Image {
    /// Computed layout (token count, position span, merged grid).
    pub span: multimodal::VisualSpan,
    /// Whether the owning model uses M-RoPE (drives the wire side-channel).
    pub uses_mrope: bool,
    /// Preprocessed pixel_values `[n_patch * patch_dim]` (arch-specific layout).
    pub pixels: Vec<f32>,
    /// Per-patch positions `[n_patch * 2]` (x, y).
    pub positions: Vec<u32>,
    /// Pre-merge `(t, h, w)` patch-unit grid for the driver's vision encoder
    /// (`t*h*w == n_patch` for Qwen; `(1,1,token_count)` for Gemma). Sent on the
    /// wire's `image_grids`; distinct from `span.grid` (the merged LLM grid).
    pub patch_grid: multimodal::Grid,
    /// Model-specific delimiter tokens the context places immediately before /
    /// after this span (e.g. Qwen `<|vision_start|>` / `<|vision_end|>`). Empty
    /// when the model needs none. The SDK's `append-image` applies them, so the
    /// inferlet never names them.
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
}

impl Image {
    fn from_processed(
        p: multimodal::ProcessedImage,
        uses_mrope: bool,
        prefix: Vec<u32>,
        suffix: Vec<u32>,
    ) -> Image {
        Image {
            span: p.span,
            uses_mrope,
            pixels: p.pixels,
            positions: p.positions,
            patch_grid: p.patch_grid,
            prefix,
            suffix,
        }
    }
}

/// Video resource — frames decoded + uniformly sampled host-side, each already
/// preprocessed into an [`Image`]. The SDK splices them in order.
pub struct Video {
    pub frames: Vec<Image>,
    /// Per-frame timestamp in seconds, parallel to `frames`.
    pub timestamps: Vec<f32>,
}

/// Audio resource — preprocessed log-mel features (option-equivalent of Image).
pub struct Audio {
    /// Log-mel features `[n_frames * 128]` f32, frame-major.
    pub mel: Vec<f32>,
    /// Number of mel frames.
    pub n_frames: u32,
    /// Audio soft tokens this clip occupies.
    pub token_count: u32,
    /// Model-specific delimiter tokens (e.g. Gemma `<|audio>` / `<audio|>`).
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
}

/// Encode a delimiter string with the model's tokenizer, or `[]` if empty.
fn encode_delim(s: &str) -> Vec<u32> {
    if s.is_empty() {
        Vec::new()
    } else {
        crate::model::model().tokenize(s)
    }
}

/// Uniformly sample up to `max_frames` indices from `0..n` (inclusive of the
/// first and last frame). Returns all indices when `n <= max_frames`.
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

impl pie::core::media::Host for InstanceState {}

impl pie::core::media::HostImage for InstanceState {
    /// Decode + resize + patchify an encoded still image per the bound model.
    async fn from_bytes(
        &mut self,
        bytes: Vec<u8>,
    ) -> Result<Result<Resource<Image>, String>> {
        let (processor, prefix, suffix) = {
            let m = crate::model::model();
            let arch = m.arch_name();
            let varch = match VisionArch::from_arch_name(arch) {
                Some(a) => a,
                None => {
                    return Ok(Err(format!(
                        "model '{}' (arch '{arch}') has no vision front-end",
                        m.name()
                    )));
                }
            };
            let (pre, suf) = multimodal::vision_delimiters(varch);
            (
                Processor::for_arch(varch),
                encode_delim(pre),
                encode_delim(suf),
            )
        };
        let processed = match processor.process_image_bytes(&bytes) {
            Ok(p) => p,
            Err(e) => return Ok(Err(e)),
        };
        let img = Image::from_processed(processed, processor.uses_mrope(), prefix, suffix);
        Ok(Ok(self.ctx().table.push(img)?))
    }

    async fn token_count(&mut self, this: Resource<Image>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.span.token_count)
    }

    async fn position_span(&mut self, this: Resource<Image>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.span.position_span)
    }

    async fn grid(&mut self, this: Resource<Image>) -> Result<(u32, u32, u32)> {
        let g = self.ctx().table.get(&this)?.span.grid;
        Ok((g.t, g.h, g.w))
    }

    async fn prefix_tokens(&mut self, this: Resource<Image>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.prefix.clone())
    }

    async fn suffix_tokens(&mut self, this: Resource<Image>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.suffix.clone())
    }

    async fn drop(&mut self, this: Resource<Image>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::media::HostVideo for InstanceState {
    /// Decode an animated container, uniformly sample `<= max_frames` frames,
    /// and preprocess each per the bound model's per-frame budget.
    async fn from_bytes(
        &mut self,
        bytes: Vec<u8>,
        max_frames: u32,
    ) -> Result<Result<Resource<Video>, String>> {
        let (processor, prefix, suffix) = {
            let m = crate::model::model();
            let arch = m.arch_name();
            let varch = match VisionArch::from_arch_name(arch) {
                Some(a) => a,
                None => {
                    return Ok(Err(format!(
                        "model '{}' (arch '{arch}') has no vision front-end",
                        m.name()
                    )));
                }
            };
            let (pre, suf) = multimodal::vision_delimiters(varch);
            (
                // Video frames use the per-frame budget (Gemma ≤70 tokens/frame).
                Processor::for_arch_video(varch),
                encode_delim(pre),
                encode_delim(suf),
            )
        };
        let decoded = match multimodal::decode_gif_frames(&bytes) {
            Ok(f) => f,
            Err(e) => return Ok(Err(e)),
        };
        let sel = sample_indices(decoded.len(), max_frames as usize);
        let mut frames = Vec::with_capacity(sel.len());
        let mut timestamps = Vec::with_capacity(sel.len());
        for &i in &sel {
            let (img, ts) = &decoded[i];
            let processed = processor.process_image(img);
            frames.push(Image::from_processed(
                processed,
                processor.uses_mrope(),
                prefix.clone(),
                suffix.clone(),
            ));
            timestamps.push(*ts);
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

impl pie::core::media::HostAudio for InstanceState {
    /// Decode (WAV) + resample + log-mel an encoded audio clip per the bound
    /// model. Non-audio models return a clean error.
    async fn from_bytes(
        &mut self,
        bytes: Vec<u8>,
    ) -> Result<Result<Resource<Audio>, String>> {
        let (prefix, suffix) = {
            let m = crate::model::model();
            let arch = m.arch_name();
            if !multimodal::audio_arch_supported(arch) {
                return Ok(Err(format!(
                    "model '{}' (arch '{arch}') has no audio front-end",
                    m.name()
                )));
            }
            let (pre, suf) = multimodal::audio_delimiters(arch);
            (encode_delim(pre), encode_delim(suf))
        };
        let (mel, n_frames) = match multimodal::audio::process_wav_bytes(&bytes) {
            Ok(x) => x,
            Err(e) => return Ok(Err(e)),
        };
        if n_frames == 0 {
            return Ok(Err("audio: clip decoded to zero frames".into()));
        }
        let token_count = multimodal::gemma_audio_token_count(n_frames as u32);
        let audio = Audio {
            mel,
            n_frames: n_frames as u32,
            token_count,
            prefix,
            suffix,
        };
        Ok(Ok(self.ctx().table.push(audio)?))
    }

    async fn token_count(&mut self, this: Resource<Audio>) -> Result<u32> {
        Ok(self.ctx().table.get(&this)?.token_count)
    }

    async fn position_span(&mut self, this: Resource<Audio>) -> Result<u32> {
        // 1-D RoPE: the sequence cursor advances by the soft-token count.
        Ok(self.ctx().table.get(&this)?.token_count)
    }

    async fn prefix_tokens(&mut self, this: Resource<Audio>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.prefix.clone())
    }

    async fn suffix_tokens(&mut self, this: Resource<Audio>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.suffix.clone())
    }

    async fn drop(&mut self, this: Resource<Audio>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
