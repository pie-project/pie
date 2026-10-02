use super::color::rgb8_to_i420;

pub fn frame_rate(fps: f32) -> (u32, u32) {
    if !fps.is_finite() || fps <= 0.0 {
        return (1, 1);
    }
    let num = (fps * 1000.0).round().max(1.0) as u32;
    (num, 1000)
}

pub fn encode(
    rgb: &[u8],
    width: u32,
    height: u32,
    count: u32,
    fps: f32,
) -> Result<Vec<u8>, String> {
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(format!(
            "y4m is 4:2:0 and needs even dimensions; this handle is {width}x{height}"
        ));
    }
    let (w, h, n) = (width as usize, height as usize, count as usize);
    let frame_bytes = w * h * 3;
    if rgb.len() != frame_bytes * n {
        return Err(format!(
            "y4m: {} bytes for {n} frames of {width}x{height} (expected {})",
            rgb.len(),
            frame_bytes * n
        ));
    }
    let (num, den) = frame_rate(fps);
    let mut out = Vec::with_capacity(64 + n * (6 + w * h * 3 / 2));
    out.extend_from_slice(
        format!("YUV4MPEG2 W{width} H{height} F{num}:{den} Ip A1:1 C420mpeg2\n").as_bytes(),
    );
    for f in 0..n {
        out.extend_from_slice(b"FRAME\n");
        rgb8_to_i420(&rgb[f * frame_bytes..(f + 1) * frame_bytes], w, h, &mut out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y4m_every_case() {
        the_stream_is_a_header_and_then_one_frame_each();
        odd_dimensions_are_refused_by_name();
    }

    fn the_stream_is_a_header_and_then_one_frame_each() {
        let rgb = vec![0u8; 4 * 2 * 3 * 3];
        let out = encode(&rgb, 4, 2, 3, 25.0).expect("encode");
        assert!(out.starts_with(b"YUV4MPEG2 W4 H2 F25000:1000 Ip A1:1 C420mpeg2\n"));
        assert_eq!(out.windows(6).filter(|w| *w == b"FRAME\n").count(), 3);
        let header = out.iter().position(|&b| b == b'\n').unwrap() + 1;
        assert_eq!(out.len() - header, 3 * (6 + 12));
    }

    fn odd_dimensions_are_refused_by_name() {
        let err = encode(&[0u8; 3 * 3 * 3], 3, 3, 1, 1.0).unwrap_err();
        assert!(err.contains("3x3"), "{err}");
    }
}
