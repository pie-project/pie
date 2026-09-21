use image::AnimationDecoder;
use image::codecs::gif::GifDecoder;

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: f32,
}

pub fn frames(bytes: &[u8]) -> Result<Vec<Frame>, String> {
    let decoder =
        GifDecoder::new(std::io::Cursor::new(bytes)).map_err(|e| format!("gif decode: {e}"))?;
    let decoded = decoder
        .into_frames()
        .collect_frames()
        .map_err(|e| format!("gif frames: {e}"))?;
    if decoded.is_empty() {
        return Err("gif has no frames".into());
    }
    let mut frames = Vec::with_capacity(decoded.len());
    let mut shown_ms = 0.0f32;
    for frame in decoded {
        let (num, den) = frame.delay().numer_denom_ms();
        let frame_ms = num as f32 / den as f32;
        let rgb = image::DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8();
        let (width, height) = (rgb.width(), rgb.height());
        frames.push(Frame {
            rgb: rgb.into_raw(),
            width,
            height,
            timestamp: shown_ms / 1000.0,
        });
        shown_ms += frame_ms;
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gif_every_case() {
        an_animation_decodes_to_the_frames_it_was_written_with();
        bytes_that_are_no_animation_are_refused_by_name();
    }

    fn an_animation_decodes_to_the_frames_it_was_written_with() {
        let animation = two_frames();
        let frames = frames(&animation).expect("its own gif decodes");
        assert_eq!(frames.len(), 2, "both frames come back");
        for frame in &frames {
            assert_eq!((frame.width, frame.height), (4, 2), "the extent survives");
            assert_eq!(frame.rgb.len(), 4 * 2 * 3, "RGB, three bytes a pixel");
        }
        assert_eq!(frames[0].timestamp, 0.0, "the first frame opens the clip");
        assert!(
            frames[1].timestamp > 0.0,
            "and the second is shown later: {}",
            frames[1].timestamp
        );
        assert_eq!(&frames[0].rgb[..3], &[255, 0, 0], "the first frame is red");
        assert_eq!(&frames[1].rgb[..3], &[0, 0, 255], "the second is blue");
    }

    fn bytes_that_are_no_animation_are_refused_by_name() {
        let why = frames(b"not a gif, just some bytes").expect_err("no magic");
        assert!(why.contains("gif decode"), "{why}");
    }

    fn two_frames() -> Vec<u8> {
        use image::codecs::gif::GifEncoder;
        use image::{Delay, Frame as GifFrame, RgbaImage};
        let flat = |r, g, b| RgbaImage::from_fn(4, 2, move |_, _| image::Rgba([r, g, b, 255]));
        let mut out = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut out);
            for pixels in [flat(255, 0, 0), flat(0, 0, 255)] {
                let delay = Delay::from_numer_denom_ms(100, 1);
                encoder
                    .encode_frame(GifFrame::from_parts(pixels, 0, 0, delay))
                    .expect("a flat frame encodes");
            }
        }
        out
    }
}
