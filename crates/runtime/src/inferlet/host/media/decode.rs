use models::media::{Fault, Rgb8};

pub fn decode(bytes: &[u8]) -> models::media::Result<Rgb8> {
    if bytes.is_empty() {
        return Err(Fault::Decode(
            "no bytes: an empty payload is no image".into(),
        ));
    }
    let (rgb, width, height) = media::still::decode(bytes).map_err(|e| {
        Fault::Decode(format!(
            "the bytes are not an image this front-end reads: {e}"
        ))
    })?;
    Rgb8::new(height, width, rgb)
}

#[must_use]
pub fn resize_exact(src: &Rgb8, th: u32, tw: u32) -> Rgb8 {
    if src.h == th && src.w == tw {
        return src.clone();
    }
    let data = media::still::resize(&src.data, src.w, src.h, tw, th)
        .expect("an Rgb8 always holds h · w · 3 bytes");
    Rgb8 { h: th, w: tw, data }
}
