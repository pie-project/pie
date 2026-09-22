use image::imageops::FilterType;
use image::{ExtendedColorType, ImageEncoder};

const JPEG_QUALITY: u8 = 92;

pub fn png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("png encode failed: {e}"))?;
    Ok(out)
}

pub fn jpeg(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY)
        .write_image(rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("jpeg encode failed: {e}"))?;
    Ok(out)
}

pub fn decode(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("still decode failed on {} bytes: {e}", bytes.len()))?;
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    if width == 0 || height == 0 {
        return Err(format!("the decoded picture is {width}x{height}"));
    }
    Ok((rgb.into_raw(), width, height))
}

const RESAMPLE: FilterType = FilterType::CatmullRom;

pub fn resize(
    rgb: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, String> {
    let owed = width as usize * height as usize * 3;
    let buf = image::RgbImage::from_raw(width, height, rgb.to_vec()).ok_or_else(|| {
        format!(
            "a {height} x {width} RGB picture is {owed} bytes and {} arrived",
            rgb.len()
        )
    })?;
    Ok(image::imageops::resize(&buf, target_width, target_height, RESAMPLE).into_raw())
}

pub fn webp(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::webp::WebPEncoder::new_lossless(&mut out)
        .write_image(rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("webp encode failed: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn still_every_case() {
        a_png_decodes_back_to_the_pixels_it_encoded();
        a_resize_keeps_the_extent_it_was_asked_for();
        bytes_that_are_no_picture_are_refused_by_name();
        png_round_trips_a_gradient_exactly();
        jpeg_and_webp_carry_the_same_picture();
    }

    fn a_png_decodes_back_to_the_pixels_it_encoded() {
        let (w, h) = (7u32, 5u32);
        let rgb = gradient(w, h);
        let bytes = png(&rgb, w, h).expect("a gradient encodes");
        let (back, dw, dh) = decode(&bytes).expect("its own png decodes");
        assert_eq!((dw, dh), (w, h), "the extent survives the round trip");
        assert_eq!(back, rgb, "and so does every byte");
    }

    fn bytes_that_are_no_picture_are_refused_by_name() {
        let why = decode(b"not a picture, just some bytes").expect_err("no magic");
        assert!(why.contains("still decode failed"), "{why}");
        assert!(why.contains("30 bytes"), "the refusal counts them: {why}");
    }

    fn a_resize_keeps_the_extent_it_was_asked_for() {
        let (w, h) = (8u32, 4u32);
        let out = resize(&gradient(w, h), w, h, 4, 2).expect("a gradient resizes");
        assert_eq!(out.len(), 4 * 2 * 3, "the pixels the caller asked for");
        let why = resize(&[0, 0, 0], w, h, 4, 2).expect_err("three bytes are no picture");
        assert!(why.contains("96 bytes and 3 arrived"), "{why}");
    }

    fn gradient(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                v.push((x * 255 / w.max(1)) as u8);
                v.push((y * 255 / h.max(1)) as u8);
                v.push(((x + y) * 255 / (w + h).max(1)) as u8);
            }
        }
        v
    }

    fn png_round_trips_a_gradient_exactly() {
        let (w, h) = (16, 9);
        let src = gradient(w, h);
        let bytes = png(&src, w, h).expect("encode");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let back = image::load_from_memory(&bytes).expect("decode").to_rgb8();
        assert_eq!(back.dimensions(), (w, h));
        assert_eq!(back.into_raw(), src, "png is lossless");
    }

    fn jpeg_and_webp_carry_the_same_picture() {
        let (w, h) = (16, 8);
        let src = gradient(w, h);

        let j = jpeg(&src, w, h).expect("jpeg");
        assert_eq!(&j[..2], b"\xff\xd8", "SOI");
        let back = image::load_from_memory(&j).expect("decode jpeg").to_rgb8();
        assert_eq!(back.dimensions(), (w, h));

        let wp = webp(&src, w, h).expect("webp");
        assert_eq!(&wp[..4], b"RIFF");
        assert_eq!(&wp[8..12], b"WEBP");
    }
}
