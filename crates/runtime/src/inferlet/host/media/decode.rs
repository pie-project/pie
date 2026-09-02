//! The codec half of the media pipe: decodes the container and executes the
//! resample a front-end asks for, via [`models::media::Resample`]. Front-ends
//! stay codec-free (`models::media` is a catalog crate with minimal deps).
//!
//! Uses `image` 0.25 (`default-features = false`, `png`/`jpeg`/`gif`) with all
//! three decoders scalar and single-threaded (no `rayon`, no runtime
//! CPU-feature dispatch, no float in the PNG path): the same bytes must decode
//! to the same pixels on every run and host, which [`super::span_digest`]
//! rests on.
//!
//! Resample filter is [`FilterType::CatmullRom`], the Keys cubic at
//! `a = -0.5` — the same kernel as `transformers`' PIL image-processor
//! backend (`Qwen2VLImageProcessorPil`, `Gemma4ImageProcessorPil`), chosen
//! over the torchvision backend because a pure-Rust preprocessor can
//! reproduce PIL's float path but not torchvision's fixed-point one (which
//! differs by <= 1 LSB per channel).

use image::imageops::FilterType;
use models::media::{Fault, Rgb8};

/// See module docs for why Catmull-Rom.
const RESAMPLE: FilterType = FilterType::CatmullRom;

/// Decode encoded bytes to 8-bit RGB.
///
/// Alpha is composited away by `to_rgb8`'s own rule (the channel is dropped),
/// which is `do_convert_rgb` in both reference processors.
pub fn decode(bytes: &[u8]) -> models::media::Result<Rgb8> {
    if bytes.is_empty() {
        return Err(Fault::Decode(
            "no bytes: an empty payload is no image".into(),
        ));
    }
    let img = image::load_from_memory(bytes).map_err(|e| {
        Fault::Decode(format!(
            "the bytes are not an image this front-end reads: {e}"
        ))
    })?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Rgb8::new(h, w, rgb.into_raw())
}

/// Resample to exactly `(th, tw)` with [`RESAMPLE`] — the function lent to
/// every front-end as its [`models::media::Resample`].
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
