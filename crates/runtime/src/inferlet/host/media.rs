//! pie:inferlet/media — Image / Video / Audio resources for multimodal input.
//!
//! **Model-agnostic by construction, and the dispatch is the whole of what
//! this file knows.** The inferlet hands the host raw encoded bytes (a
//! PNG/JPEG, an animated GIF, a WAV); this file reads the bound model's
//! `ROWS.arch`, picks a [`media_frontend::VisionFrontEnd`] /
//! [`media_frontend::AudioFrontEnd`] for it or refuses by name, and hands the
//! bytes over. Everything after that — decode, resize, patchify, normalize,
//! log-mel, the interpolation taps of a resampled position table — is the
//! front-end's, behind the trait (media-door.md §4). An inferlet never branches
//! on the model, and neither, past the `match` below, does the runtime.
//!
//! **ONE LEDGER** (media-door.md §0). A span enters the sequence as the token
//! run `tokens()` answers — prefix + pad × `token-count` + suffix, in the bound
//! model's own ids — and as nothing else. The handle crosses a second time
//! beside the tokens, through `forward-pass.media`, carrying only the payload;
//! the correspondence between the two is SCANNED at submit
//! ([`crate::pipeline::media`]), never asserted by the guest.
//!
//! **THE SPELLING IS THE TOKENIZER'S, NOT THE FRONT-END'S.** A front-end names
//! its architecture's delimiters as strings and stops there, because the ids
//! belong to the bound checkpoint's tokenizer and a front-end holds none. This
//! file encodes them, once, at `from_bytes` — which is also what keeps
//! `tokens()` right across two checkpoints of one architecture that renumbered
//! their specials.

pub mod multimodal;

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use media_frontend::{
    AudioFrontEnd, Budget, Delimiters, EncodedSpan, Fault, Grid, VisionFrontEnd,
};
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Image resource — a preprocessed still image, also used for one video frame.
///
/// **THE HANDLE IS THE SPAN AND NOTHING BESIDE IT.** Everything the old struct
/// carried in parallel — pixels, per-patch positions, the pre-merge grid, the
/// delimiter ids, the M-RoPE flag — is a field of [`EncodedSpan`], which is the
/// type the front-end trait answers and the type the submission carries toward
/// the contract. One struct, one place a field can be added, and the wire
/// marshaling MD-C cuts reads the same record the front-end wrote.
#[derive(Clone)]
pub struct Image {
    /// The preprocessed span, shared: a decoded image submitted to two passes
    /// is decoded once and its payload copied never.
    pub span: Arc<EncodedSpan>,
}

/// Video resource — frames decoded + uniformly sampled host-side, each already
/// preprocessed into an [`Image`]. The SDK splices them in order.
pub struct Video {
    pub frames: Vec<Image>,
    /// Per-frame timestamp in seconds, parallel to `frames`.
    pub timestamps: Vec<f32>,
}

/// Audio resource — a preprocessed log-mel span. The same [`EncodedSpan`] as
/// [`Image`], because the two differ in how they are computed and in nothing
/// the sequence can see.
#[derive(Clone)]
pub struct Audio {
    pub span: Arc<EncodedSpan>,
}

/// Lift the local processor's answer into the front-end contract's span.
///
/// **A SEAM WITH NOTHING BEHIND IT ANY MORE** (wave MD-C, the second of MD-B's
/// two pinned seams). `VisionAdapter` stood here — a wrapper that put
/// `multimodal::Processor`, the campaign's test-helper preprocessing, behind
/// [`VisionFrontEnd`] so MD-A's dispatch could be the real dispatch before the
/// real front-ends existed. They exist ([`media_frontend::qwen3_5`],
/// [`media_frontend::gemma4`], wave MD-B), and each is a pinned transcription
/// of its reference processor with goldens; the adapter's `span_of` left
/// `embed_rows` and `embed_weights` EMPTY under a note saying the native grid
/// needs none, which is true of the helper's fixed resize and false of a
/// front-end that resamples the learned position table. Keeping the wrapper
/// would have served every image through the preprocessing that has no golden.
///
/// [`media_frontend::qwen3_5`]: media_frontend::qwen3_5
/// [`media_frontend::gemma4`]: media_frontend::gemma4

/// The bound model's audio processor, behind the front-end trait.
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

    fn encode_audio(&self, bytes: &[u8]) -> media_frontend::Result<EncodedSpan> {
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
            // 1-D RoPE: the sequence cursor advances by the soft-token count.
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

/// **THE DISPATCH** (media-door §4): the bound model's `ROWS.arch` to a vision
/// front-end, or the one refusal this layer knows how to say.
///
/// # Errors
///
/// [`Fault::NoVisionFrontEnd`], naming the model and the arch it was asked
/// about — a text model has no tower and never will, and saying so here is
/// what keeps every layer below this one free of the question.
fn vision_front_end() -> media_frontend::Result<Box<dyn VisionFrontEnd>> {
    let m = crate::model::model();
    let arch = m.arch_name();
    front_end_for(arch).ok_or_else(|| Fault::NoVisionFrontEnd {
        model: m.name().to_string(),
        arch: arch.to_string(),
    })
}

/// **THE MATCH IS THE WHOLE OF WHAT THIS LAYER KNOWS ABOUT ARCHITECTURES**
/// (media-door §4). Past it the runtime names none: it asked for a front-end
/// and got one, or got the one sentence it has to know how to say.
///
/// Boxed because the two arms are two types and every caller wants a
/// `dyn VisionFrontEnd` — the seam itself, in both directions. Split from
/// [`vision_front_end`] so a test can ask the question without a booted model:
/// the dispatch is a property of the ARCH STRING, and the global is only where
/// the string comes from.
pub(crate) fn front_end_for(arch: &str) -> Option<Box<dyn VisionFrontEnd>> {
    match arch {
        media_frontend::qwen3_5::ARCH => Some(Box::new(media_frontend::qwen3_5::Qwen35Vision::new())),
        media_frontend::gemma4::ARCH => Some(Box::new(media_frontend::gemma4::Gemma4Vision::new())),
        _ => None,
    }
}

/// The same, for audio.
fn audio_front_end() -> media_frontend::Result<AudioAdapter> {
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

/// **SPELL THE SPAN IN THE BOUND CHECKPOINT'S OWN IDS.**
///
/// The front-end named its architecture's delimiters as strings; this resolves
/// them through the model's tokenizer, once, at `from_bytes`. The placeholder
/// must resolve to EXACTLY ONE id — it is a reserved special, and a checkpoint
/// whose tokenizer does not carry it cannot spell a run the scan could ever
/// match, so that is refused here rather than diagnosed later as a run of the
/// wrong length.
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

impl pie::inferlet::media::Host for ProcessCtx {}

impl pie::inferlet::media::HostImage for ProcessCtx {
    /// Decode + resize + patchify an encoded still image per the bound model.
    async fn from_bytes(&mut self, bytes: Vec<u8>) -> Result<Result<Resource<Image>, String>> {
        let front_end = match vision_front_end() {
            Ok(fe) => fe,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        // Through the TRAIT and nothing else: this call site is the seam
        // media-door §4 draws, and it must not be able to reach anything an
        // arbitrary front-end does not offer.
        let mut span = match front_end.encode_image(&bytes, Budget::Still) {
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

    /// The span's full spelling, ready to splice (media-door §0).
    async fn tokens(&mut self, this: Resource<Image>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.span.tokens())
    }

    /// The cache statute's key material (media-door §5).
    async fn digest(&mut self, this: Resource<Image>) -> Result<Vec<u8>> {
        Ok(self.ctx().table.get(&this)?.span.digest())
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
    /// Decode an animated container, uniformly sample `<= max_frames` frames,
    /// and preprocess each per the bound model's per-frame budget.
    async fn from_bytes(
        &mut self,
        bytes: Vec<u8>,
        max_frames: u32,
    ) -> Result<Result<Resource<Video>, String>> {
        let front_end = match vision_front_end() {
            Ok(fe) => fe,
            Err(fault) => return Ok(Err(fault.to_string())),
        };
        let decoded = match multimodal::decode_gif_frames(&bytes) {
            Ok(f) => f,
            Err(e) => return Ok(Err(Fault::Decode(e).to_string())),
        };
        let delims = front_end.delimiters();
        let sel = sample_indices(decoded.len(), max_frames as usize);
        let mut frames = Vec::with_capacity(sel.len());
        let mut timestamps = Vec::with_capacity(sel.len());
        for &i in &sel {
            let (img, ts) = &decoded[i];
            // **A DEMUXED FRAME CROSSES AS PIXELS**, which is the one
            // interchange form `encode_rgb8` states and the one that needs no
            // image library on either side of the seam. `to_rgb8` is the
            // decoder's own `do_convert_rgb`: the alpha channel is dropped.
            let rgb = img.to_rgb8();
            let (fw, fh) = (rgb.width(), rgb.height());
            let mut span = match front_end.encode_rgb8(
                rgb.as_raw(),
                fw,
                fh,
                Budget::VideoFrame,
            ) {
                Ok(span) => span,
                Err(fault) => return Ok(Err(fault.to_string())),
            };
            if let Err(refusal) = spell(&mut span, delims) {
                return Ok(Err(refusal));
            }
            frames.push(Image {
                span: Arc::new(span),
            });
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

impl pie::inferlet::media::HostAudio for ProcessCtx {
    /// Decode (WAV) + resample + log-mel an encoded audio clip per the bound
    /// model. Non-audio models are refused by name.
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
        Ok(self.ctx().table.get(&this)?.span.digest())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// media-door §4: the dispatch is on `ROWS.arch` and every arch in the
    /// catalog answers it one way or the other — a front-end, or the refusal
    /// that names itself.
    #[test]
    fn every_catalogued_arch_either_has_a_vision_front_end_or_is_refused_by_name() {
        let mut archs: Vec<&'static str> = crate::model::ROWS.iter().map(|r| r.arch).collect();
        archs.sort_unstable();
        archs.dedup();
        assert!(!archs.is_empty(), "the catalog names no architectures");
        for arch in archs {
            match front_end_for(arch) {
                Some(fe) => {
                    assert_eq!(fe.arch(), arch, "a front-end answered for another arch");
                    let d = fe.delimiters();
                    assert!(
                        !d.placeholder.is_empty(),
                        "{arch} has a vision front-end and no reserved pad to spell a run with"
                    );
                }
                None => {
                    let fault = Fault::NoVisionFrontEnd {
                        model: "m".into(),
                        arch: arch.to_string(),
                    };
                    assert_eq!(fault.name(), "NoVisionFrontEnd");
                    assert!(fault.to_string().starts_with("NoVisionFrontEnd"));
                }
            }
        }
    }

    /// The two archs that DO have towers are the two the campaign built, and
    /// each spells its run with its own reserved pad.
    #[test]
    fn the_two_vision_archs_spell_their_runs_differently() {
        let qwen = front_end_for("qwen3_5").expect("qwen has a tower").delimiters();
        let gemma = front_end_for("gemma4").expect("gemma has a tower").delimiters();
        assert_eq!(qwen.placeholder, "<|image_pad|>");
        assert_eq!(qwen.prefix, "<|vision_start|>");
        // MD-B's own reading of THIS checkpoint's vocabulary: gemma-4 spells
        // its markers `<|x>` / `<|x|>` / `<x|>`, the family `<|turn>` and
        // `<|audio>` already belong to. `<image_soft_token>` was the adapter's
        // guess and is gemma-3's.
        assert_eq!(gemma.placeholder, "<|image|>");
        assert_ne!(qwen.placeholder, gemma.placeholder);
    }

    /// **THE WIT SURFACE, AS LANDED** (media-door.md §2).
    ///
    /// This is a compile-time assertion wearing a test's name: every path
    /// below exists only if `bindgen!` generated it from the WIT, and the
    /// crate does not build if `HostImage`/`HostAudio` are missing `tokens` or
    /// `digest` (E0046) or if `forward-pass` is missing `media`. Written down
    /// rather than left implicit because "the bindings generate" is a claim
    /// the wave makes, and a claim with no named test is a claim nobody
    /// re-checks.
    #[test]
    fn the_wit_surface_carries_tokens_digest_and_the_media_span_variant() {
        // media.image / media.audio: the two new verbs, named by their
        // generated trait-method paths. Naming them is the assertion — a WIT
        // that did not declare them generates no such item and this does not
        // compile.
        let _ = <ProcessCtx as pie::inferlet::media::HostImage>::tokens;
        let _ = <ProcessCtx as pie::inferlet::media::HostImage>::digest;
        let _ = <ProcessCtx as pie::inferlet::media::HostAudio>::tokens;
        let _ = <ProcessCtx as pie::inferlet::media::HostAudio>::digest;
        // forward.forward-pass.media, and the variant it takes — both cases
        // from day one, so audio joins without a verb change.
        let _ = <ProcessCtx as pie::inferlet::forward::HostForwardPass>::media;
        let image_case = pie::inferlet::forward::MediaSpan::Image(Resource::new_borrow(0));
        let audio_case = pie::inferlet::forward::MediaSpan::Audio(Resource::new_borrow(0));
        assert!(matches!(
            image_case,
            pie::inferlet::forward::MediaSpan::Image(_)
        ));
        assert!(matches!(
            audio_case,
            pie::inferlet::forward::MediaSpan::Audio(_)
        ));
    }

    /// An arch with no vision front-end is refused BY NAME, and the sentence
    /// carries both the model and the arch it was asked about.
    #[test]
    fn an_unknown_arch_is_refused_as_no_vision_front_end() {
        assert!(front_end_for("deepseek_v4").is_none());
        let fault = Fault::NoVisionFrontEnd {
            model: "ds-v4".into(),
            arch: "deepseek_v4".into(),
        };
        let said = fault.to_string();
        assert!(said.starts_with("NoVisionFrontEnd"), "{said}");
        assert!(said.contains("ds-v4") && said.contains("deepseek_v4"), "{said}");
    }
}
