//! **THE ONE DEPENDENCY, AND WHY IT IS THIS ONE.**
//!
//! Everything else in this crate is arithmetic transcribed from a reference
//! processor and gated against it. Decode is the exception: nobody transcribes
//! a JPEG decoder, so the crate takes one, and the choice is argued here rather
//! than in a `Cargo.toml` comment nobody reads.
//!
//! **`image` 0.25, `default-features = false`, features `png` / `jpeg` / `gif`.**
//!
//! 1. **It is already in the lockfile, and for this exact code.** `runtime`
//!    took `image` when `model::serve::multimodal` moved into it (M18), which
//!    is the module this crate promotes. So the crate that owns the
//!    preprocessing takes the dependency the preprocessing already had: zero
//!    new crates resolved, zero new licences to review (`image` and its tree
//!    are MIT OR Apache-2.0, the workspace's own pair).
//! 2. **"`image` vs zune" is not a fork in the road for JPEG — it is a wrapper
//!    choice.** `image` 0.25's baseline/progressive JPEG decoder *is*
//!    `zune-jpeg`; the lockfile carries `zune-jpeg 0.5.15` today because
//!    `image` put it there. Picking zune directly would buy the same decoder
//!    and lose PNG and GIF, which have to come from somewhere anyway.
//! 3. **A hand-rolled png+jpeg pair costs two more crates to audit and still
//!    has no GIF.** media-door §7 keeps video-as-frames alive
//!    (`vid.frame(i)?.tokens()`), and a clip arrives as one container. Dropping
//!    the format that carries it to save a wrapper is a bad trade.
//! 4. **Size is bought with `default-features = false`, not with a smaller
//!    crate.** The default feature set is twenty-odd formats plus `rayon`
//!    plus the exr/avif/webp trees; three explicit features is the same three
//!    codecs a minimal pair would have given, under one name.
//! 5. **Determinism.** All three decoders are scalar and single-threaded here —
//!    no `rayon` (feature off), no runtime CPU-feature dispatch that could hand
//!    two machines two answers, no floating point in the PNG path at all. The
//!    same bytes decode to the same pixels on every run and every host, which
//!    is what [`EncodedSpan::digest`](crate::EncodedSpan::digest) rests on.
//!
//! **THE RESAMPLE IS THE ONE STEP WITH NO SINGLE RIGHT ANSWER, AND THIS IS THE
//! ONE IT TAKES.** Both reference processors resize bicubic with antialiasing,
//! and `transformers` v5.15.1 ships each of them TWICE — a torchvision backend
//! (`Qwen2VLImageProcessor`, `Gemma4ImageProcessor`) and a PIL backend
//! (`Qwen2VLImageProcessorPil`, `Gemma4ImageProcessorPil`). PIL's `BICUBIC` is
//! the Keys cubic at `a = -0.5`, which is the Catmull-Rom spline, which is
//! `image`'s [`FilterType::CatmullRom`] — the same kernel, and the same
//! support-scaled (antialiased) separable resample. So this crate tracks
//! UPSTREAM'S PIL BACKEND, deliberately, because that is the backend a pure
//! Rust preprocessor can reproduce; torchvision's uint8 path answers the same
//! kernel through 16-bit fixed-point accumulation and differs from a float
//! evaluation of it by ≤ 1 LSB per channel.
//!
//! Everything downstream of the resize — the patch order, the per-patch vector
//! layout, the normalization, the grid arithmetic, the position table's taps
//! and weights — is bit-pinned against `transformers` and gated as such. The
//! resize is named here so that "approximately right" is a recorded decision
//! about one step rather than an unexamined property of the whole pipe.

use image::imageops::FilterType;

use crate::Fault;

/// **THE RESAMPLE FILTER**, and the module docs are its argument: Catmull-Rom
/// is the Keys cubic at `a = -0.5`, which is PIL's `BICUBIC`, which is the
/// kernel `transformers`' PIL image-processor backend resizes with.
const RESAMPLE: FilterType = FilterType::CatmullRom;

/// A decoded image, 8-bit RGB, row-major HWC — the one shape every front-end
/// in this crate patchifies from.
///
/// HWC and not CHW because that is the memory order a decoder hands back and
/// the order both patchifiers stride over; a transpose to plane-major would
/// buy one front-end nothing and cost a full copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rgb8 {
    /// Height in pixels.
    pub(crate) h: u32,
    /// Width in pixels.
    pub(crate) w: u32,
    /// `h · w · 3` bytes, row-major, R then G then B per pixel.
    pub(crate) data: Vec<u8>,
}

/// Decode encoded bytes to 8-bit RGB.
///
/// Alpha is composited away by `to_rgb8`'s own rule (the channel is dropped),
/// which is `do_convert_rgb` in both reference processors.
pub(crate) fn decode(bytes: &[u8]) -> crate::Result<Rgb8> {
    if bytes.is_empty() {
        return Err(Fault::Decode("no bytes: an empty payload is no image".into()));
    }
    let img = image::load_from_memory(bytes)
        .map_err(|e| Fault::Decode(format!("the bytes are not an image this front-end reads: {e}")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w == 0 || h == 0 {
        return Err(Fault::Empty(format!(
            "the image decoded to {h} x {w}, and a span of no pixels occupies no rows"
        )));
    }
    Ok(Rgb8 {
        h,
        w,
        data: rgb.into_raw(),
    })
}

/// Resample to exactly `(th, tw)` with [`RESAMPLE`].
///
/// Exact and not fit-inside: both front-ends have already computed the target
/// from their own policy, and a resize that preserved aspect ratio a second
/// time would silently disagree with the grid the caller then patchifies over.
pub(crate) fn resize_exact(src: &Rgb8, th: u32, tw: u32) -> Rgb8 {
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

/// **PIXELS THAT NEED NO DECODER** — one already-demuxed frame, as
/// [`VisionFrontEnd::encode_rgb8`](crate::VisionFrontEnd::encode_rgb8) states
/// it: `height · width · 3` bytes, row-major, 8 bits per channel.
///
/// The one interchange form that costs this crate's callers no image library:
/// a demuxer above the seam already holds pixels, and the alternative — re-
/// encoding each frame to PNG so it could go back through [`decode`] — is a
/// full compress/decompress round trip to satisfy a signature.
pub(crate) fn from_rgb8(data: &[u8], w: u32, h: u32) -> crate::Result<Rgb8> {
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
    Ok(Rgb8 {
        h,
        w,
        data: data.to_vec(),
    })
}
