//! **The media front-end contract: decoded pixels in, one [`EncodedSpan`] out.**
//!
//! `.wiki/alto/media-door.md` §4 draws four layers between a guest's `bytes`
//! and the engine's tower, and this module is the third of them — moved into
//! the catalog crate because it is CATALOG KNOWLEDGE: which architecture
//! preprocesses how is a fact of the family, and it lives beside the family's
//! forward pass, import contract and chat template
//! ([`crate::qwen_3::media`], [`crate::gemma_4::media`]) rather than in a
//! crate of its own with the same families spelled a second way.
//!
//! **THE CODEC IS NOT HERE, AND THAT IS THE CRATE'S OWN RULE.** This crate's
//! manifest states that every dependency is one every consumer of the catalog
//! needs, and a compiler does not need a JPEG decoder — the `serve` feature
//! that once gated an image codec into this crate was removed with the module
//! it gated (M18), and this contract is written so it does not come back. So
//! the seam takes PIXELS: the host decodes (`runtime`'s codec module, the one
//! caller), the front-end computes its architecture's target from the source
//! dims, and the one step that needs an image library — the resample — is
//! LENT through [`Resample`] rather than owned. Everything the front-end does
//! itself is arithmetic transcribed from a reference processor.
//!
//! **THE TRAIT IS THE SEAM, AND IT IS THE SEAM IN BOTH DIRECTIONS.** Above it,
//! the runtime never names an architecture: it asks [`vision_front_end`] and
//! gets one or gets [`Fault::NoVisionFrontEnd`], which is the only sentence it
//! has to know how to say. Below it, a front-end never names a WIT resource, a
//! resource table, a submission or a fire: it is a pure function from pixels
//! to a span, which is what makes it testable against a pinned transcription
//! rather than against a running engine.
//!
//! **AND THE SPAN IS ONE STRUCT, NOT TWO.** An image span and an audio span
//! differ in how they are computed and in nothing the sequence can see: both
//! occupy `token_count` rows, both advance the cursor by `position_span`, both
//! carry a payload the tower turns into embedding rows. So [`EncodedSpan`] is
//! the output of both traits, and `media-span`'s variant in the WIT is about
//! which resource the guest is holding, not about which struct crosses here.

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

/// A decoded image, 8-bit RGB, row-major HWC — the one shape every front-end
/// patchifies from, and the one interchange form that costs this crate no
/// image library: a decoder and a demuxer both already hold exactly this.
///
/// HWC and not CHW because that is the memory order a decoder hands back and
/// the order both patchifiers stride over; a transpose to plane-major would
/// buy one front-end nothing and cost a full copy.
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
    /// Wrap already-decoded pixels, refusing the two shapes that could only
    /// mislead downstream: a zero-sided frame (a span of no pixels occupies no
    /// rows) and a buffer that is not `h · w · 3` bytes.
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

/// **THE LENT RESAMPLE** — resize `src` to exactly `(target_h, target_w)`.
///
/// The one step of a front-end's pipe with no single right answer and no
/// transcription (nobody transcribes a bicubic resampler), so the host that
/// owns the image library lends it here and the choice of kernel is argued
/// where the library is taken (`runtime`'s codec module: Catmull-Rom, which is
/// PIL's `BICUBIC`, which is what `transformers`' PIL backend resizes with).
/// Exact and not fit-inside: the front-end has already computed the target
/// from its own policy, and a resize that preserved aspect ratio a second time
/// would silently disagree with the grid the front-end then patchifies over.
pub type Resample = fn(&Rgb8, u32, u32) -> Rgb8;

/// **One preprocessed media span — what a front-end answers and the only thing
/// above it reads.**
///
/// Every field is derived from the source pixels and the bound architecture,
/// and nothing here knows about a sequence, a lane or a fire: the runtime
/// derives the anchors, the lane-relative routes and the trunk's token triples
/// from these numbers once it knows where the run landed (media-door §3).
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
    /// indexed by. `cells() == rows` of [`payload`](EncodedSpan::payload).
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

    /// Fill the id fields from a tokenizer's answer for this front-end's
    /// [`Delimiters`]. The runtime calls this once, immediately after the
    /// front-end returns, because it holds the tokenizer and the front-end
    /// does not.
    ///
    /// The span's content DIGEST — media-door §5's cache statute — is the
    /// runtime's too (`runtime::inferlet::host::media::span_digest`), for the
    /// same dependency rule that keeps the codec out: the statute needs a
    /// hasher, and the catalog's consumers do not.
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
/// Implemented once per architecture family, in the family's own module
/// beside its forward pass and template, against pinned transcriptions of the
/// reference processor. The runtime holds a `dyn VisionFrontEnd` and asks it
/// two questions: how does this architecture spell a span, and what does this
/// picture encode to.
pub trait VisionFrontEnd: Send + Sync {
    /// The `ROWS.arch` string this front-end answers for. Used for diagnostics
    /// and for the dispatch table's own consistency test; the dispatch itself
    /// is [`vision_front_end`]'s.
    fn arch(&self) -> &'static str;

    /// How this architecture wraps a visual span.
    fn delimiters(&self) -> Delimiters;

    /// Resize (through the lent `resample`) + patchify + normalize `src`
    /// under `budget`.
    ///
    /// One verb for a still image and a demuxed video frame, because past the
    /// decode they are the same pixels: the host decodes its container — a
    /// PNG, or one frame of a clip — and everything after is this
    /// architecture's arithmetic, so the two doors cannot answer two different
    /// spans for one picture.
    ///
    /// # Errors
    ///
    /// [`Fault::Empty`] when the pixels encode to a span occupying no rows, or
    /// when the architecture's own resize policy refuses the shape.
    fn encode(&self, src: &Rgb8, budget: Budget, resample: Resample) -> Result<EncodedSpan>;
}

/// **One architecture's audio preprocessing.**
///
/// The same shape as [`VisionFrontEnd`], and separate from it because a model
/// may have either tower without the other: Gemma has both, Qwen has vision
/// only, and a text model has neither. Two traits is what lets the two
/// refusals be two sentences.
///
/// Bytes rather than pixels, because an audio clip has no lent step: WAV
/// decode and log-mel are hand-rolled arithmetic with no library behind them,
/// so the whole pipe is the implementer's.
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

/// **THE DISPATCH IS A CATALOG FACT** (media-door §4): the arch string to its
/// family's front-end, or `None` for a family with no served tower — the same
/// sentence `catalog!`'s fourth column already writes for a lane's class:
/// nothing outside a family's own module says how that family preprocesses,
/// and no layer above this one names an architecture at all.
///
/// Boxed because the arms are different types and every caller wants a
/// `dyn VisionFrontEnd` — the seam itself, in both directions. The runtime
/// turns `None` into [`Fault::NoVisionFrontEnd`], naming the model it holds
/// and this crate does not.
#[must_use]
pub fn vision_front_end(arch: &str) -> Option<Box<dyn VisionFrontEnd>> {
    match arch {
        crate::qwen_3::media::ARCH => Some(Box::new(crate::qwen_3::media::Qwen35Vision::new())),
        crate::gemma_4::media::ARCH => Some(Box::new(crate::gemma_4::media::Gemma4Vision::new())),
        _ => None,
    }
}

/// **A front-end that reads no real picture, for the tests above the seam.**
///
/// The run scan, the word stamp and the refusal set are all properties of the
/// runtime and none of them care what an image looked like — they care that a
/// span occupies `n` rows and spells itself with a particular pad. So the
/// tests that exercise them take this rather than a real photograph, which is
/// what keeps them deterministic and keeps a codec out of their graph.
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
            // The pixels themselves, so two different inputs digest
            // differently — which is the property §5's statute rests on.
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

    /// The spelling is the tokenizer's: the same span under two checkpoints'
    /// special ids answers two token lists from one front-end.
    #[test]
    fn respelling_moves_the_tokens() {
        let a = spelled(2, 11);
        let b = spelled(2, 22);
        assert_ne!(a.tokens(), b.tokens());
    }

    #[test]
    fn a_video_frame_gets_the_frame_budget() {
        let fe = StubFrontEnd::new("stub", 8);
        let identity: Resample = |src, _, _| src.clone();
        let still = fe
            .encode(&pixels(b"x"), Budget::Still, identity)
            .expect("still");
        let frame = fe
            .encode(&pixels(b"x"), Budget::VideoFrame, identity)
            .expect("frame");
        assert_eq!(still.token_count, 8);
        assert_eq!(frame.token_count, 4);
    }

    /// The two shapes [`Rgb8::new`] refuses, each by the right name.
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

    /// media-door §4: every arch with a served tower answers the dispatch, and
    /// each spells its run with its own reserved pad.
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
