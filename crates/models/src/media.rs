//! Media front-end contract: decoded pixels in, one [`EncodedSpan`] out.
//! No image/audio codec lives here; pixels/bytes are decoded by the host and
//! passed in, with [`Resample`] lent in for the one step that needs a library.

use std::fmt;

/// A span's extent in the units its front-end counts in: the merged grid
/// (`t·h·w / merge²` rows, what the LLM's token rectangle sees) or the patch
/// grid (what the tower's rotation stream is indexed by).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Grid {
    /// Temporal extent (1 for a still image, frames for a clip).
    pub t: u32,
    /// Height.
    pub h: u32,
    /// Width.
    pub w: u32,
}

impl Grid {
    /// A still image's grid: one temporal step over `h × w`.
    #[must_use]
    pub const fn still(h: u32, w: u32) -> Grid {
        Grid { t: 1, h, w }
    }

    /// How many cells this grid holds.
    #[must_use]
    pub const fn cells(&self) -> u32 {
        self.t * self.h * self.w
    }
}

/// How much of a soft-token budget this span is allowed. A still image and a
/// video frame get the same preprocessing under different ceilings (e.g.
/// Gemma: 256 vs 70).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Budget {
    /// The model's full per-image soft-token budget.
    #[default]
    Still,
    /// The model's per-frame budget, for one frame of a sampled clip.
    VideoFrame,
}

/// The delimiters a span spells itself with, as strings (the front-end names
/// them; the runtime resolves them to token ids since it holds the
/// tokenizer).
///
/// An empty string means this architecture needs none here. `placeholder`
/// may not be empty: it is the reserved pad the run scan finds a span by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delimiters {
    /// Placed immediately before the placeholder run (e.g. `<|vision_start|>`).
    pub prefix: &'static str,
    /// The reserved per-model pad, repeated `token_count` times. A tokenizer
    /// never emits it from text, which is exactly what makes the scan sound.
    pub placeholder: &'static str,
    /// Placed immediately after the run (e.g. `<|vision_end|>`).
    pub suffix: &'static str,
}

/// A decoded image, 8-bit RGB, row-major HWC — the format a decoder hands
/// back and every front-end patchifies from directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8 {
    /// Height in pixels.
    pub h: u32,
    /// Width in pixels.
    pub w: u32,
    /// `h · w · 3` bytes, row-major, R then G then B per pixel.
    pub data: Vec<u8>,
}

impl Rgb8 {
    /// Wrap already-decoded pixels.
    ///
    /// # Errors
    ///
    /// [`Fault::Empty`] for a zero side; [`Fault::Decode`] for a wrong-length
    /// buffer.
    pub fn new(h: u32, w: u32, data: Vec<u8>) -> Result<Rgb8> {
        if w == 0 || h == 0 {
            return Err(Fault::Empty(format!(
                "a frame of {h} x {w} pixels occupies no rows"
            )));
        }
        let owed = h as usize * w as usize * 3;
        if data.len() != owed {
            return Err(Fault::Decode(format!(
                "a {h} x {w} RGB frame is {owed} bytes and {} arrived",
                data.len()
            )));
        }
        Ok(Rgb8 { h, w, data })
    }
}

/// Resize `src` to exactly `(target_h, target_w)`, lent by the host (does
/// not preserve aspect ratio; the front-end has already computed the target).
pub type Resample = fn(&Rgb8, u32, u32) -> Rgb8;

/// One preprocessed media span — what a front-end answers. Knows nothing
/// about a sequence, lane, or fire; the runtime derives those from these
/// numbers once it knows where the run landed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EncodedSpan {
    /// Hidden-state rows / KV slots this span occupies — the length of the
    /// placeholder run.
    pub token_count: u32,
    /// How far the 1-D sequence cursor advances past this span. Equals
    /// `token_count` under 1-D RoPE; `max(t, h, w)` of the merged grid under
    /// M-RoPE.
    pub position_span: u32,
    /// Extent in merged-token units — what the LLM's rectangle sees.
    pub grid: Grid,
    /// Extent in pre-merge patch units. `cells() == rows` of
    /// [`payload`](EncodedSpan::payload).
    pub patch_grid: Grid,
    /// Does the bound architecture rotate this span with M-RoPE? If false,
    /// the engine reads scalar `(p, p, p)` positions.
    pub uses_mrope: bool,
    /// The payload the tower consumes: patch vectors for vision
    /// (`[rows · patch_dim]`, merge-block-major), log-mel frames for audio
    /// (`[frames · mel_bins]`). Never raw pixels.
    pub payload: Vec<f32>,
    /// How many rows of [`payload`](EncodedSpan::payload) this span
    /// contributes. `payload.len() == rows · row_width`.
    pub rows: u32,
    /// Per payload row, its own coordinate in the span's grid — `(y, x)` for a
    /// vision patch. Two entries per row, or empty when the architecture's
    /// tower reads no position stream.
    pub positions: Vec<u32>,
    /// Which rows of the learned position table each payload row gathers, and
    /// how much of each. `taps` entries per payload row, or both empty when
    /// no resampling is staged.
    pub embed_rows: Vec<i32>,
    /// Beside [`embed_rows`](EncodedSpan::embed_rows), same length.
    pub embed_weights: Vec<f32>,
    /// The delimiter TOKEN IDS before the run — the runtime's encoding of
    /// [`Delimiters::prefix`], filled in after the front-end returns.
    pub prefix: Vec<u32>,
    /// The reserved pad id the run is spelled with — the runtime's encoding of
    /// [`Delimiters::placeholder`].
    pub placeholder: u32,
    /// The delimiter token ids after the run.
    pub suffix: Vec<u32>,
}

impl EncodedSpan {
    /// The span's full spelling: prefix + pad × `token_count` + suffix, in
    /// the bound model's own ids.
    #[must_use]
    pub fn tokens(&self) -> Vec<u32> {
        let mut out =
            Vec::with_capacity(self.prefix.len() + self.token_count as usize + self.suffix.len());
        out.extend_from_slice(&self.prefix);
        out.extend(std::iter::repeat_n(
            self.placeholder,
            self.token_count as usize,
        ));
        out.extend_from_slice(&self.suffix);
        out
    }

    /// Fill the id fields from a tokenizer's answer for this front-end's
    /// [`Delimiters`]. Called once by the runtime after the front-end
    /// returns, since only the runtime holds the tokenizer.
    pub fn spell_with(&mut self, prefix: Vec<u32>, placeholder: u32, suffix: Vec<u32>) {
        self.prefix = prefix;
        self.placeholder = placeholder;
        self.suffix = suffix;
    }
}

/// The refusals this seam can state, each by its own name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The bound model's architecture has no vision front-end registered.
    NoVisionFrontEnd {
        /// The bound model's catalog id.
        model: String,
        /// Its `ROWS.arch`.
        arch: String,
    },
    /// The bound model's architecture has no audio front-end registered.
    NoAudioFrontEnd {
        /// The bound model's catalog id.
        model: String,
        /// Its `ROWS.arch`.
        arch: String,
    },
    /// The bytes did not decode as the container this front-end reads.
    Decode(String),
    /// The bytes decoded but the span is degenerate (zero frames, zero
    /// patches) — a span occupying no rows has no run to match.
    Empty(String),
}

impl Fault {
    /// The fault's own name, for a caller that wants to test the refusal
    /// rather than its prose.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Fault::NoVisionFrontEnd { .. } => "NoVisionFrontEnd",
            Fault::NoAudioFrontEnd { .. } => "NoAudioFrontEnd",
            Fault::Decode(_) => "Decode",
            Fault::Empty(_) => "Empty",
        }
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::NoVisionFrontEnd { model, arch } => write!(
                f,
                "NoVisionFrontEnd: model '{model}' (arch '{arch}') has no vision front-end"
            ),
            Fault::NoAudioFrontEnd { model, arch } => write!(
                f,
                "NoAudioFrontEnd: model '{model}' (arch '{arch}') has no audio front-end"
            ),
            Fault::Decode(why) => write!(f, "Decode: {why}"),
            Fault::Empty(why) => write!(f, "Empty: {why}"),
        }
    }
}

impl std::error::Error for Fault {}

/// What a front-end answers.
pub type Result<T> = std::result::Result<T, Fault>;

/// One architecture's vision preprocessing, implemented once per family
/// against pinned transcriptions of the reference processor.
pub trait VisionFrontEnd: Send + Sync {
    /// The `ROWS.arch` string this front-end answers for.
    fn arch(&self) -> &'static str;

    /// How this architecture wraps a visual span.
    fn delimiters(&self) -> Delimiters;

    /// Resize (through the lent `resample`) + patchify + normalize `src`
    /// under `budget`.
    ///
    /// # Errors
    ///
    /// [`Fault::Empty`] when the pixels encode to a span occupying no rows, or
    /// when the architecture's own resize policy refuses the shape.
    fn encode(&self, src: &Rgb8, budget: Budget, resample: Resample) -> Result<EncodedSpan>;
}

/// One architecture's audio preprocessing. Separate from [`VisionFrontEnd`]
/// since a model may have either tower without the other.
pub trait AudioFrontEnd: Send + Sync {
    /// The `ROWS.arch` string this front-end answers for.
    fn arch(&self) -> &'static str;

    /// How this architecture wraps an audio span.
    fn delimiters(&self) -> Delimiters;

    /// Decode + resample + compute this architecture's log-mel features.
    ///
    /// # Errors
    ///
    /// [`Fault::Decode`] for bytes that are not the container this front-end
    /// reads; [`Fault::Empty`] for a clip that decodes to zero frames.
    fn encode_audio(&self, bytes: &[u8]) -> Result<EncodedSpan>;
}

/// Dispatch from an arch string to its family's vision front-end, or `None`
/// for a family with no served tower.
#[must_use]
pub fn vision_front_end(arch: &str) -> Option<Box<dyn VisionFrontEnd>> {
    match arch {
        crate::qwen_3::media::ARCH => Some(Box::new(crate::qwen_3::media::Qwen35Vision::new())),
        crate::gemma_4::media::ARCH => Some(Box::new(crate::gemma_4::media::Gemma4Vision::new())),
        _ => None,
    }
}

/// A front-end that reads no real picture, for tests above the seam.
#[derive(Clone, Copy, Debug)]
pub struct StubFrontEnd {
    /// The arch this stub claims.
    pub arch: &'static str,
    /// How many rows every span it answers occupies.
    pub token_count: u32,
    /// The delimiters it claims.
    pub delimiters: Delimiters,
}

impl StubFrontEnd {
    /// A stub answering `token_count`-row spans wrapped in Qwen's delimiters.
    #[must_use]
    pub const fn new(arch: &'static str, token_count: u32) -> StubFrontEnd {
        StubFrontEnd {
            arch,
            token_count,
            delimiters: Delimiters {
                prefix: "<|vision_start|>",
                placeholder: "<|image_pad|>",
                suffix: "<|vision_end|>",
            },
        }
    }
}

impl VisionFrontEnd for StubFrontEnd {
    fn arch(&self) -> &'static str {
        self.arch
    }

    fn delimiters(&self) -> Delimiters {
        self.delimiters
    }

    fn encode(&self, src: &Rgb8, budget: Budget, _resample: Resample) -> Result<EncodedSpan> {
        let rows = match budget {
            Budget::Still => self.token_count,
            // Half the budget, so a test can tell the two apart.
            Budget::VideoFrame => self.token_count.div_ceil(2),
        };
        if rows == 0 {
            return Err(Fault::Empty("stub front-end: zero-row span".into()));
        }
        Ok(EncodedSpan {
            token_count: rows,
            position_span: rows,
            grid: Grid::still(1, rows),
            patch_grid: Grid::still(1, rows),
            uses_mrope: false,
            // The pixels themselves, so two different inputs digest differently.
            payload: src.data.iter().map(|&b| f32::from(b)).collect(),
            rows,
            positions: Vec::new(),
            embed_rows: Vec::new(),
            embed_weights: Vec::new(),
            prefix: Vec::new(),
            placeholder: 0,
            suffix: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixels(bytes: &[u8]) -> Rgb8 {
        let mut data = bytes.to_vec();
        data.resize(3, 0);
        Rgb8::new(1, 1, data).expect("one pixel")
    }

    fn spelled(rows: u32, pad: u32) -> EncodedSpan {
        let mut span = StubFrontEnd::new("stub", rows)
            .encode(&pixels(b"abc"), Budget::Still, |src, _, _| src.clone())
            .expect("stub encodes");
        span.spell_with(vec![7], pad, vec![8]);
        span
    }

    #[test]
    fn tokens_are_prefix_then_the_run_then_suffix() {
        let span = spelled(4, 99);
        assert_eq!(span.tokens(), vec![7, 99, 99, 99, 99, 8]);
        assert_eq!(
            span.tokens().len(),
            span.prefix.len() + span.token_count as usize + span.suffix.len()
        );
    }

    #[test]
    fn degenerate_pixels_are_refused_by_name() {
        assert_eq!(
            Rgb8::new(0, 4, Vec::new()).expect_err("zero side").name(),
            "Empty"
        );
        assert_eq!(
            Rgb8::new(2, 2, vec![0; 5]).expect_err("wrong length").name(),
            "Decode"
        );
    }

    #[test]
    fn the_two_vision_archs_spell_their_runs_differently() {
        let qwen = vision_front_end("qwen3_5").expect("qwen has a tower").delimiters();
        let gemma = vision_front_end("gemma4").expect("gemma has a tower").delimiters();
        assert_eq!(qwen.placeholder, "<|image_pad|>");
        assert_eq!(qwen.prefix, "<|vision_start|>");
        assert_eq!(gemma.placeholder, "<|image|>");
        assert_ne!(qwen.placeholder, gemma.placeholder);
        assert!(vision_front_end("deepseek_v4").is_none());
    }

}
