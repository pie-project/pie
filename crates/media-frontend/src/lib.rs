//! **The media front-end contract: encoded bytes in, one [`EncodedSpan`] out.**
//!
//! `.wiki/alto/media-door.md` §4 draws four layers between a guest's `bytes`
//! and the engine's tower, and this crate is the third of them. The guest hands
//! the host a PNG/JPEG/WAV; the runtime's media host dispatches on the bound
//! model's `arch` — and stops there. Everything after the dispatch — decode,
//! resize, patchify, normalize, the interpolation taps of a resampled position
//! table, log-mel — is a property of one architecture's processor, and lives
//! behind [`VisionFrontEnd`] / [`AudioFrontEnd`].
//!
//! **THE TRAIT IS THE SEAM, AND IT IS THE SEAM IN BOTH DIRECTIONS.** Above it,
//! the runtime never names an architecture: it asks for a front-end and gets
//! one or gets [`Fault::NoVisionFrontEnd`], which is the only sentence it has
//! to know how to say. Below it, a front-end never names a WIT resource, a
//! resource table, a submission or a fire: it is a pure function from bytes to
//! a span, which is what makes it testable against a pinned transcription
//! rather than against a running engine.
//!
//! **AND THE SPAN IS ONE STRUCT, NOT TWO.** An image span and an audio span
//! differ in how they are computed and in nothing the sequence can see: both
//! occupy `token_count` rows, both advance the cursor by `position_span`, both
//! carry a payload the tower turns into embedding rows. So [`EncodedSpan`] is
//! the output of both traits, and `media-span`'s variant in the WIT is about
//! which resource the guest is holding, not about which struct crosses here.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

// **THE PER-ARCHITECTURE IMPLEMENTATIONS** (media-door §6, wave MD-B). Each is
// a pinned transcription of one reference processor and nothing else: no
// engine, no runtime, no WIT. `decode` is private because the decode
// dependency is an implementation detail of this crate and of no caller —
// which is the property that lets it be argued and swapped in one file.
mod decode;
pub mod gemma4;
pub mod qwen3_5;

use std::fmt;

/// A span's extent in the units its front-end counts in.
///
/// Read two ways, and both readings are here rather than in two types because
/// they are the same three numbers about the same span: as the MERGED grid it
/// is what the LLM's token rectangle sees (`t·h·w / merge²` rows); as the
/// PATCH grid it is what the tower's rotation stream is indexed by. A front-end
/// answers both on [`EncodedSpan`], and which one a reader wants is a property
/// of the reader.
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

/// **How much of a budget this span is allowed.**
///
/// A still image and one frame of a video are the SAME preprocessing under a
/// different soft-token ceiling (Gemma's video frames get 70 rather than 256),
/// so this is an argument rather than a second trait method: a front-end that
/// does not distinguish them ignores it, and one that does reads it in the one
/// place the ceiling is applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Budget {
    /// The model's full per-image soft-token budget.
    #[default]
    Still,
    /// The model's per-frame budget, for one frame of a sampled clip.
    VideoFrame,
}

/// **The delimiters a span spells itself with, as STRINGS.**
///
/// A front-end knows which special tokens its architecture wraps a span in;
/// it does not know their ids, because the ids are the bound checkpoint's
/// tokenizer's and a front-end holds no tokenizer. So it names them and the
/// runtime encodes them — which is also what keeps `tokens()` honest across two
/// checkpoints of one architecture that renumbered their specials.
///
/// An empty string means "this architecture needs none here" and encodes to no
/// tokens. `placeholder` is the one that may NOT be empty: it is the reserved
/// pad id the run scan finds a span by (media-door §3), and an architecture
/// with no reserved pad has no run for the scan to match.
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

/// **One preprocessed media span — what a front-end answers and the only thing
/// above it reads.**
///
/// Every field is derived from the source bytes and the bound architecture, and
/// nothing here knows about a sequence, a lane or a fire: the runtime derives
/// the anchors, the lane-relative routes and the trunk's token triples from
/// these numbers once it knows where the run landed (media-door §3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EncodedSpan {
    /// Hidden-state rows / KV slots this span occupies — the LENGTH OF THE
    /// PLACEHOLDER RUN, and the number the scan checks a run against.
    pub token_count: u32,
    /// How far the 1-D sequence cursor advances past this span. Equals
    /// `token_count` under 1-D RoPE; `max(t, h, w)` of the merged grid under
    /// M-RoPE, where the next text token's three components all resume.
    pub position_span: u32,
    /// Extent in merged-token units — what the LLM's rectangle sees.
    pub grid: Grid,
    /// Extent in pre-merge patch units — what the tower's rotation stream is
    /// indexed by. `cells() == rows` of [`patches`](EncodedSpan::patches).
    pub patch_grid: Grid,
    /// Does the bound architecture rotate this span with M-RoPE? Decides
    /// whether the runtime owes the contract a token-position triple stream or
    /// may leave it empty (which the engine reads as scalar `(p, p, p)`).
    pub uses_mrope: bool,
    /// The payload the tower consumes: patch VECTORS for vision
    /// (`[rows · patch_dim]`, merge-block-major so the spatial merge is a
    /// view), log-mel frames for audio (`[frames · mel_bins]`). Never pixels,
    /// and never anything a guest could have produced — pixels do not cross
    /// into WASM in either direction.
    pub payload: Vec<f32>,
    /// How many rows of [`payload`](EncodedSpan::payload) this span
    /// contributes. `payload.len() == rows · row_width`.
    pub rows: u32,
    /// Per payload row, its own coordinate in the span's grid — `(y, x)` for a
    /// vision patch. Two entries per row, or empty when the architecture's
    /// tower reads no position stream.
    pub positions: Vec<u32>,
    /// **WHICH ROWS OF THE LEARNED POSITION TABLE EACH ROW GATHERS** and how
    /// much of each (media-door §3's "embed ids/weights"; multimodal §9.2).
    /// `taps` entries per payload row, in `payload`'s own order, or both empty
    /// on the native grid — the cheap path, where the plan declares no
    /// resampling and nothing is staged.
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
    /// **THE SPAN'S FULL SPELLING: prefix + pad × `token_count` + suffix**
    /// (media-door §2), in the bound model's own ids.
    ///
    /// This is what `image.tokens()` answers and what a guest splices into its
    /// context with one `extend`. The guest hardcodes nothing — the principle
    /// media.wit was written to protect — it simply no longer has to assemble
    /// the run itself out of three separate queries and a `repeat`.
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

    /// **A STABLE CONTENT HASH OF THE PREPROCESSED SPAN** (media-door §5, the
    /// cache statute).
    ///
    /// Over the PAYLOAD and its layout, deliberately not over the source
    /// bytes: two encodings of one photograph — a re-JPEG, a re-crop that
    /// resizes back to the same grid — are the same span to the model, and a
    /// digest that separated them would make a correct cache miss look like a
    /// correctness bug. The tokens are NOT folded in, because folding them
    /// would make the digest useless for the thing it exists for: two
    /// different images produce identical token lists, and this is what tells
    /// them apart.
    #[must_use]
    pub fn digest(&self) -> Vec<u8> {
        let mut h = blake3::Hasher::new();
        // Domain separation, so a future audio/vision payload of identical
        // bytes and identical extent cannot collide across modalities.
        h.update(b"pie:media-span:v1");
        for n in [
            self.token_count,
            self.position_span,
            self.rows,
            self.grid.t,
            self.grid.h,
            self.grid.w,
            self.patch_grid.t,
            self.patch_grid.h,
            self.patch_grid.w,
        ] {
            h.update(&n.to_le_bytes());
        }
        h.update(&[u8::from(self.uses_mrope)]);
        for f in &self.payload {
            h.update(&f.to_le_bytes());
        }
        for p in &self.positions {
            h.update(&p.to_le_bytes());
        }
        for r in &self.embed_rows {
            h.update(&r.to_le_bytes());
        }
        for w in &self.embed_weights {
            h.update(&w.to_le_bytes());
        }
        h.finalize().as_bytes().to_vec()
    }

    /// Fill the id fields from a tokenizer's answer for this front-end's
    /// [`Delimiters`]. The runtime calls this once, immediately after the
    /// front-end returns, because it holds the tokenizer and the front-end
    /// does not.
    pub fn spell_with(&mut self, prefix: Vec<u32>, placeholder: u32, suffix: Vec<u32>) {
        self.prefix = prefix;
        self.placeholder = placeholder;
        self.suffix = suffix;
    }
}

/// **The refusals this seam can state, each by its own name.**
///
/// media-door §3's discipline is that every disagreement is refused by name
/// before anything launches, and the first two of them are reachable before a
/// single byte has been decoded: a model with no tower, asked for one.
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

/// **One architecture's vision preprocessing.**
///
/// Implemented once per architecture family, in this crate, against pinned
/// transcriptions of the reference processor. The runtime holds a
/// `dyn VisionFrontEnd` and asks it two questions: how does this architecture
/// spell a span, and what does this image encode to.
pub trait VisionFrontEnd: Send + Sync {
    /// The `ROWS.arch` string this front-end answers for. Used for diagnostics
    /// and for the dispatch table's own consistency test; the dispatch itself
    /// is the runtime's.
    fn arch(&self) -> &'static str;

    /// How this architecture wraps a visual span.
    fn delimiters(&self) -> Delimiters;

    /// Decode + resize + patchify + normalize `bytes` under `budget`.
    ///
    /// # Errors
    ///
    /// [`Fault::Decode`] when the bytes are not an image this front-end reads;
    /// [`Fault::Empty`] when they decode to a span occupying no rows.
    fn encode_image(&self, bytes: &[u8], budget: Budget) -> Result<EncodedSpan>;

    /// **THE SAME ENCODE, ON A FRAME THAT IS ALREADY DECODED.**
    ///
    /// A video arrives as one animated container and leaves as N spans, so its
    /// frames are demuxed once, above this trait, and never re-encoded to be
    /// handed back down. `rgb8` is `height · width · 3` bytes, row-major,
    /// 8 bits per channel — the one interchange form that needs no image
    /// library in this crate's dependency graph.
    ///
    /// Defaulted so a front-end that only reads still images compiles, and
    /// defaulted to a REFUSAL rather than to silence: a video whose frames
    /// quietly encoded as nothing would be a clip the model never saw.
    ///
    /// # Errors
    ///
    /// [`Fault::Decode`] for a buffer that is not `height · width · 3` bytes,
    /// or from a front-end that does not implement this.
    fn encode_rgb8(
        &self,
        rgb8: &[u8],
        width: u32,
        height: u32,
        budget: Budget,
    ) -> Result<EncodedSpan> {
        let _ = (rgb8, width, height, budget);
        Err(Fault::Decode(format!(
            "front-end '{}' reads encoded containers only and was handed a              decoded frame",
            self.arch()
        )))
    }
}

/// **One architecture's audio preprocessing.**
///
/// The same shape as [`VisionFrontEnd`], and separate from it because a model
/// may have either tower without the other: Gemma has both, Qwen has vision
/// only, and a text model has neither. Two traits is what lets the two
/// refusals be two sentences.
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

/// **A front-end that decodes nothing, for the tests above the seam.**
///
/// The run scan, the word stamp and the refusal set are all properties of the
/// runtime and none of them care what an image looked like — they care that a
/// span occupies `n` rows and spells itself with a particular pad. So the
/// tests that exercise them take this rather than a real photograph, which is
/// what keeps them deterministic and keeps a decode dependency out of the
/// runtime's test graph.
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

    fn encode_image(&self, bytes: &[u8], budget: Budget) -> Result<EncodedSpan> {
        if bytes.is_empty() {
            return Err(Fault::Decode("stub front-end: zero bytes".into()));
        }
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
            // The bytes themselves, so two different inputs digest differently
            // — which is the property §5's statute rests on.
            payload: bytes.iter().map(|&b| f32::from(b)).collect(),
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

    fn spelled(rows: u32, pad: u32) -> EncodedSpan {
        let mut span = StubFrontEnd::new("stub", rows)
            .encode_image(b"abc", Budget::Still)
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

    /// media-door §5: the run is the same tokens whatever the picture was, so
    /// the digest is the only thing that tells two spans apart.
    #[test]
    fn two_spans_share_a_token_list_and_not_a_digest() {
        let mut a = StubFrontEnd::new("stub", 4)
            .encode_image(b"one", Budget::Still)
            .expect("a");
        let mut b = StubFrontEnd::new("stub", 4)
            .encode_image(b"two", Budget::Still)
            .expect("b");
        a.spell_with(vec![7], 99, vec![8]);
        b.spell_with(vec![7], 99, vec![8]);
        assert_eq!(a.tokens(), b.tokens(), "the ledger cannot tell them apart");
        assert_ne!(a.digest(), b.digest(), "the statute's key must");
        assert_eq!(a.digest().len(), 32);
    }

    #[test]
    fn a_digest_is_stable_across_two_readings_of_one_span() {
        let span = spelled(3, 5);
        assert_eq!(span.digest(), span.digest());
        assert_eq!(span.digest(), span.clone().digest());
    }

    /// The spelling is the tokenizer's, so the same span under two
    /// checkpoints' special ids answers two token lists and ONE digest.
    #[test]
    fn respelling_moves_the_tokens_and_not_the_digest() {
        let a = spelled(2, 11);
        let b = spelled(2, 22);
        assert_ne!(a.tokens(), b.tokens());
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn a_video_frame_gets_the_frame_budget() {
        let fe = StubFrontEnd::new("stub", 8);
        let still = fe.encode_image(b"x", Budget::Still).expect("still");
        let frame = fe.encode_image(b"x", Budget::VideoFrame).expect("frame");
        assert_eq!(still.token_count, 8);
        assert_eq!(frame.token_count, 4);
    }

    #[test]
    fn every_fault_answers_its_own_name_and_says_it_in_its_prose() {
        let faults = [
            Fault::NoVisionFrontEnd {
                model: "m".into(),
                arch: "a".into(),
            },
            Fault::NoAudioFrontEnd {
                model: "m".into(),
                arch: "a".into(),
            },
            Fault::Decode("why".into()),
            Fault::Empty("why".into()),
        ];
        for fault in &faults {
            assert!(
                fault.to_string().starts_with(fault.name()),
                "{fault} does not lead with {}",
                fault.name()
            );
        }
    }
}
