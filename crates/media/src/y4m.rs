use super::color::{i420_to_rgb8, rgb8_to_i420};

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

pub fn decode(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32, u32, f32), String> {
    let line = |at: usize| {
        let end = bytes[at..].iter().position(|&b| b == b'\n').map(|i| at + i);
        end.ok_or("y4m: a header line has no end".to_string())
    };
    let end = line(0)?;
    let (mut width, mut height, mut fps) = (0u32, 0u32, 0f32);
    for tag in String::from_utf8_lossy(&bytes[..end]).split(' ').skip(1) {
        let (key, val) = tag.split_at(1);
        match key {
            "W" => width = val.parse().map_err(|_| format!("y4m: bad width `{val}`"))?,
            "H" => {
                height = val
                    .parse()
                    .map_err(|_| format!("y4m: bad height `{val}`"))?
            }
            "F" => {
                let (n, d) = val.split_once(':').unwrap_or((val, "1"));
                fps = n.parse::<f32>().unwrap_or(0.0) / d.parse::<f32>().unwrap_or(1.0);
            }
            "C" if !val.starts_with("420") => {
                return Err(format!("y4m: colorspace `{val}` is not 4:2:0"));
            }
            _ => {}
        }
    }
    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(format!(
            "y4m: a 4:2:0 stream needs even dimensions, not {width}x{height}"
        ));
    }
    let (w, h) = (width as usize, height as usize);
    let plane = w * h * 3 / 2;
    let (mut rgb, mut count, mut at) = (Vec::new(), 0u32, end + 1);
    while at < bytes.len() {
        if !bytes[at..].starts_with(b"FRAME") {
            return Err(format!("y4m: frame {count} does not start with FRAME"));
        }
        at = line(at)? + 1;
        let Some(yuv) = bytes.get(at..at + plane) else {
            return Err(format!("y4m: frame {count} is cut short"));
        };
        i420_to_rgb8(yuv, w, h, &mut rgb);
        at += plane;
        count += 1;
    }
    if count == 0 {
        return Err("y4m: the stream has no frames".to_string());
    }
    Ok((rgb, width, height, count, fps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y4m_every_case() {
        the_stream_is_a_header_and_then_one_frame_each();
        odd_dimensions_are_refused_by_name();
        a_decode_inverts_the_encode();
    }

    fn a_decode_inverts_the_encode() {
        let rgb: Vec<u8> = (0..4 * 2 * 3 * 2)
            .map(|i| [200, 40, 90][i % 3] + (i / 24) as u8 * 30)
            .collect();
        let (back, w, h, n, fps) = decode(&encode(&rgb, 4, 2, 2, 25.0).unwrap()).unwrap();
        assert_eq!((w, h, n, fps), (4, 2, 2, 25.0));
        assert!(
            rgb.iter().zip(&back).all(|(a, b)| a.abs_diff(*b) <= 2),
            "{rgb:?} {back:?}"
        );
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
