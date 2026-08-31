//! **THE CODEC — the one half of the media pipe that is not the model's.**
//!
//! `model::media` states the rule this module is the other side of: the
//! catalog crate carries only dependencies every consumer of the catalog
//! needs, and a compiler does not need a JPEG decoder. So the front-ends do
//! arithmetic and THIS module — in the one crate that actually serves images —
//! decodes the container and executes the resample the front-end asks for,
//! lent through [`model::media::Resample`].
//!
//! Everything else in the pipe is arithmetic transcribed from a reference
//! processor and gated against it. Decode is the exception: nobody transcribes
//! a JPEG decoder, so the host takes one, and the choice is argued here rather
//! than in a `Cargo.toml` comment nobody reads.
//!
//! **`image` 0.25, `default-features = false`, features `png` / `jpeg` / `gif`.**
//!
//! 1. **It is already in the lockfile, and for this exact code.** `runtime`
//!    took `image` when `model::serve::multimodal` moved into it (M18), which
//!    is the preprocessing the front-ends promote. Zero new crates resolved,
//!    zero new licences to review (`image` and its tree are
//!    MIT OR Apache-2.0, the workspace's own pair).
//! 2. **"`image` vs zune" is not a fork in the road for JPEG — it is a wrapper
//!    choice.** `image` 0.25's baseline/progressive JPEG decoder *is*
//!    `zune-jpeg`; the lockfile carries `zune-jpeg` today because `image` put
//!    it there. Picking zune directly would buy the same decoder and lose PNG
//!    and GIF, which have to come from somewhere anyway.
//! 3. **A hand-rolled png+jpeg pair costs two more crates to audit and still
//!    has no GIF.** media-door §7 keeps video-as-frames alive
//!    (`vid.frame(i)?.tokens()`), and a clip arrives as one container.
//!    Dropping the format that carries it to save a wrapper is a bad trade.
//! 4. **Size is bought with `default-features = false`, not with a smaller
//!    crate.** The default feature set is twenty-odd formats plus `rayon`
//!    plus the exr/avif/webp trees; three explicit features is the same three
//!    codecs a minimal pair would have given, under one name.
//! 5. **Determinism.** All three decoders are scalar and single-threaded here —
//!    no `rayon` (feature off), no runtime CPU-feature dispatch that could hand
//!    two machines two answers, no floating point in the PNG path at all. The
//!    same bytes decode to the same pixels on every run and every host, which
//!    is what [`super::span_digest`] rests on.
//!
//! **THE RESAMPLE IS THE ONE STEP WITH NO SINGLE RIGHT ANSWER, AND THIS IS THE
//! ONE IT TAKES.** Both reference processors resize bicubic with antialiasing,
//! and `transformers` v5.15.1 ships each of them TWICE — a torchvision backend
//! (`Qwen2VLImageProcessor`, `Gemma4ImageProcessor`) and a PIL backend
//! (`Qwen2VLImageProcessorPil`, `Gemma4ImageProcessorPil`). PIL's `BICUBIC` is
//! the Keys cubic at `a = -0.5`, which is the Catmull-Rom spline, which is
//! `image`'s [`FilterType::CatmullRom`] — the same kernel, and the same
//! support-scaled (antialiased) separable resample. So this module tracks
//! UPSTREAM'S PIL BACKEND, deliberately, because that is the backend a pure
//! Rust preprocessor can reproduce; torchvision's uint8 path answers the same
//! kernel through 16-bit fixed-point accumulation and differs from a float
//! evaluation of it by ≤ 1 LSB per channel.
//!
//! Everything downstream of the resize — the patch order, the per-patch vector
//! layout, the normalization, the grid arithmetic, the position table's taps
//! and weights — is bit-pinned against `transformers` and gated as such
//! (`model::qwen_3::media`, `model::gemma_4::media`, and this crate's own
//! whole-pipe tests). The resize is named here so that "approximately right"
//! is a recorded decision about one step rather than an unexamined property of
//! the whole pipe.

use image::imageops::FilterType;
use model::media::{Fault, Rgb8};

/// **THE RESAMPLE FILTER**, and the module docs are its argument: Catmull-Rom
/// is the Keys cubic at `a = -0.5`, which is PIL's `BICUBIC`, which is the
/// kernel `transformers`' PIL image-processor backend resizes with.
const RESAMPLE: FilterType = FilterType::CatmullRom;

/// Decode encoded bytes to 8-bit RGB.
///
/// Alpha is composited away by `to_rgb8`'s own rule (the channel is dropped),
/// which is `do_convert_rgb` in both reference processors.
pub fn decode(bytes: &[u8]) -> model::media::Result<Rgb8> {
    if bytes.is_empty() {
        return Err(Fault::Decode("no bytes: an empty payload is no image".into()));
    }
    let img = image::load_from_memory(bytes)
        .map_err(|e| Fault::Decode(format!("the bytes are not an image this front-end reads: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Rgb8::new(h, w, rgb.into_raw())
}

/// Resample to exactly `(th, tw)` with [`RESAMPLE`] — the function lent to
/// every front-end as its [`model::media::Resample`].
///
/// Exact and not fit-inside: the front-end has already computed the target
/// from its own policy, and a resize that preserved aspect ratio a second
/// time would silently disagree with the grid the caller then patchifies over.
#[must_use]
pub fn resize_exact(src: &Rgb8, th: u32, tw: u32) -> Rgb8 {
    if src.h == th && src.w == tw {
        return src.clone();
    }
    let buf = image::RgbImage::from_raw(src.w, src.h, src.data.clone())
        .expect("an Rgb8 always holds h · w · 3 bytes");
    let out = image::imageops::resize(&buf, tw, th, RESAMPLE);
    Rgb8 {
        h: th,
        w: tw,
        data: out.into_raw(),
    }
}
