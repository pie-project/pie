//! pie:core/media - Image resource for multimodal (vision + video) input.
//!
//! Phase 1.2: construct a visual span and answer its geometry
//! (`token-count` / `position-span` / `grid`) synchronously. The vision encoder
//! and the forward-pass splice (`input-image`) land in later phases; the
//! encoded pixel bytes are stashed on the resource now so the wire stage
//! (Phase 1.4) can forward them without re-plumbing the API. See MULTIMODAL.md.

use crate::api::model::Model;
use crate::api::pie;
use crate::instance::InstanceState;
use crate::multimodal::{self, Processor, VisionArch};
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Image resource — a preprocessed still image or video clip.
pub struct Image {
    /// Computed layout (token count, position span, merged grid).
    pub span: multimodal::VisualSpan,
    /// Whether the owning model uses M-RoPE (drives the wire side-channel).
    #[allow(dead_code)] // consumed by the wire/scatter stage.
    pub uses_mrope: bool,
    /// Preprocessed pixel_values `[n_patch * patch_dim]` (option B — the
    /// inferlet patchified). Empty for the legacy `from_bytes` (header-only) path.
    pub pixels: Vec<f32>,
    /// Per-patch positions `[n_patch * 2]` (x, y). Pairs with `pixels`.
    pub positions: Vec<u32>,
    /// Pre-merge `(t, h, w)` patch-unit grid (Qwen3-VL). Sent on the wire's
    /// `image_grids` for the driver's vision encoder (which needs patch units
    /// so `t*h*w == n_patch`). Distinct from `span.grid`, the merged LLM grid
    /// reported to callers via `image.grid()`. `(1,1,token_count)` for Gemma.
    pub patch_grid: multimodal::Grid,
    /// Encoded frame bytes (legacy `from_bytes`/`from_frames` path).
    #[allow(dead_code)]
    pub frames: Vec<Vec<u8>>,
}

impl pie::core::media::Host for InstanceState {}

impl pie::core::media::HostImage for InstanceState {
    async fn from_bytes(
        &mut self,
        model: Resource<Model>,
        bytes: Vec<u8>,
    ) -> Result<Result<Resource<Image>, String>> {
        let (arch, name) = {
            let m = self.ctx().table.get(&model)?;
            (m.model.arch_name().to_string(), m.model.name().to_string())
        };
        let processor = match VisionArch::from_arch_name(&arch).map(Processor::for_arch) {
            Some(p) => p,
            None => {
                return Ok(Err(format!(
                    "model '{name}' (arch '{arch}') has no vision front-end"
                )));
            }
        };
        let (w, h) = match multimodal::image_dimensions(&bytes) {
            Some(d) => d,
            None => return Ok(Err("unsupported or truncated image (expected PNG or JPEG)".into())),
        };
        let span = processor.layout_image(w, h);
        let img = Image {
            span,
            uses_mrope: processor.uses_mrope(),
            pixels: Vec::new(),
            positions: Vec::new(),
            patch_grid: span.grid,
            frames: vec![bytes],
        };
        Ok(Ok(self.ctx().table.push(img)?))
    }

    /// Option B: the inferlet supplies already-patchified `pixels`
    /// (`[n_patch * patch_dim]`) + `positions` (`[n_patch * 2]`). The host
    /// derives the soft-token count and stages the tensors for the wire; the
    /// driver runs the vision encoder over them.
    async fn from_pixels(
        &mut self,
        model: Resource<Model>,
        pixels: Vec<f32>,
        positions: Vec<u32>,
    ) -> Result<Result<Resource<Image>, String>> {
        let (arch, name) = {
            let m = self.ctx().table.get(&model)?;
            (m.model.arch_name().to_string(), m.model.name().to_string())
        };
        let processor = match VisionArch::from_arch_name(&arch).map(Processor::for_arch) {
            Some(p) => p,
            None => {
                return Ok(Err(format!(
                    "model '{name}' (arch '{arch}') has no vision front-end"
                )));
            }
        };
        if positions.len() % 2 != 0 || positions.is_empty() {
            return Ok(Err("from_pixels: positions must be a non-empty list of (x,y) pairs".into()));
        }
        let n_patch = (positions.len() / 2) as u32;
        let pool = processor.pool_factor();
        if n_patch % pool != 0 {
            return Ok(Err(format!(
                "from_pixels: n_patch ({n_patch}) not divisible by pool factor ({pool})"
            )));
        }
        let token_count = n_patch / pool;
        // Build the visual span + patch-unit grid. Gemma uses a 1-D span (the
        // encoder's 2-D RoPE is internal); Qwen3-VL needs the real merged
        // `(t, h, w)` grid so the wire's M-RoPE side-channel + `image.grid()`
        // are correct, plus the pre-merge patch grid for the driver encoder.
        let (span, patch_grid) = match processor {
            Processor::Qwen(c) => {
                // Per-patch positions are (x, y) in patch units. Derive the
                // pre-merge patch grid from their extents, then merge by `merge`.
                let merge = c.merge_size;
                let mut max_x = 0u32;
                let mut max_y = 0u32;
                for p in positions.chunks_exact(2) {
                    max_x = max_x.max(p[0]);
                    max_y = max_y.max(p[1]);
                }
                let patch_w = max_x + 1;
                let patch_h = max_y + 1;
                if patch_w % merge != 0 || patch_h % merge != 0 {
                    return Ok(Err(format!(
                        "from_pixels: Qwen patch grid {patch_w}x{patch_h} not \
                         divisible by merge ({merge})"
                    )));
                }
                let patch_grid = multimodal::Grid { t: 1, h: patch_h, w: patch_w };
                let merged = multimodal::Grid {
                    t: 1,
                    h: patch_h / merge,
                    w: patch_w / merge,
                };
                (
                    multimodal::VisualSpan {
                        token_count,
                        position_span: patch_grid.mrope_position_span(merge),
                        grid: merged,
                    },
                    patch_grid,
                )
            }
            Processor::Gemma(_) => (
                multimodal::VisualSpan {
                    token_count,
                    position_span: token_count,
                    grid: multimodal::Grid { t: 1, h: 1, w: token_count },
                },
                multimodal::Grid { t: 1, h: 1, w: token_count },
            ),
        };
        let img = Image {
            span,
            uses_mrope: processor.uses_mrope(),
            pixels,
            positions,
            patch_grid,
            frames: Vec::new(),
        };
        Ok(Ok(self.ctx().table.push(img)?))
    }

    async fn from_frames(
        &mut self,
        model: Resource<Model>,
        frames: Vec<Vec<u8>>,
        timestamps: Vec<f32>,
    ) -> Result<Result<Resource<Image>, String>> {
        let (arch, name) = {
            let m = self.ctx().table.get(&model)?;
            (m.model.arch_name().to_string(), m.model.name().to_string())
        };
        let processor = match VisionArch::from_arch_name(&arch).map(Processor::for_arch) {
            Some(p) => p,
            None => {
                return Ok(Err(format!(
                    "model '{name}' (arch '{arch}') has no vision front-end"
                )));
            }
        };
        if frames.is_empty() {
            return Ok(Err("from-frames: at least one frame is required".into()));
        }
        if !timestamps.is_empty() && timestamps.len() != frames.len() {
            return Ok(Err(format!(
                "from-frames: {} timestamps for {} frames",
                timestamps.len(),
                frames.len()
            )));
        }
        // Frames are assumed to share a resolution; derive it from the first.
        let (w, h) = match multimodal::image_dimensions(&frames[0]) {
            Some(d) => d,
            None => return Ok(Err("unsupported or truncated frame (expected PNG or JPEG)".into())),
        };
        let span = processor.layout_video(w, h, frames.len() as u32);
        let img = Image {
            span,
            uses_mrope: processor.uses_mrope(),
            pixels: Vec::new(),
            positions: Vec::new(),
            patch_grid: span.grid,
            frames,
        };
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

    async fn drop(&mut self, this: Resource<Image>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

/// Audio resource — a preprocessed audio clip (log-mel features, option B).
/// The inferlet computed the log-mel; the driver runs the gemma4_audio encoder
/// over them and scatters the projected soft-token rows. See audio_frontend.md.
pub struct Audio {
    /// Log-mel features `[n_frames * 128]` f32, frame-major. Staged for the
    /// wire's `audio_features` side-channel.
    pub mel: Vec<f32>,
    /// Number of mel frames.
    pub n_frames: u32,
    /// Audio soft tokens this clip occupies == `gemma_audio_token_count(n_frames)`.
    pub token_count: u32,
}

impl pie::core::media::HostAudio for InstanceState {
    /// Option B: the inferlet supplies precomputed log-mel features
    /// `[n_frames * 128]`. The host derives the soft-token count via the exact
    /// SSCP downsample and stages the features for the wire. Non-Gemma-4 models
    /// return a clean error.
    async fn from_mel(
        &mut self,
        model: Resource<Model>,
        mel: Vec<f32>,
        n_frames: u32,
    ) -> Result<Result<Resource<Audio>, String>> {
        let (arch, name) = {
            let m = self.ctx().table.get(&model)?;
            (m.model.arch_name().to_string(), m.model.name().to_string())
        };
        if !multimodal::audio_arch_supported(&arch) {
            return Ok(Err(format!(
                "model '{name}' (arch '{arch}') has no audio front-end"
            )));
        }
        if n_frames == 0 {
            return Ok(Err("from_mel: n_frames must be > 0".into()));
        }
        // Gemma4AudioFeatureExtractor uses 128 mel bins; the feature blob must
        // be exactly `n_frames * 128` f32.
        const N_MEL: usize = 128;
        let expected = n_frames as usize * N_MEL;
        if mel.len() != expected {
            return Ok(Err(format!(
                "from_mel: mel has {} values, expected n_frames({n_frames}) * 128 = {expected}",
                mel.len()
            )));
        }
        let token_count = multimodal::gemma_audio_token_count(n_frames);
        let audio = Audio {
            mel,
            n_frames,
            token_count,
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

    async fn drop(&mut self, this: Resource<Audio>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
