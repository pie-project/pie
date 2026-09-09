pub fn encode(samples: &[f32], rate: u32, channels: u32) -> Vec<u8> {
    let channels = channels.max(1) as u16;
    let rate = rate.max(1);
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
    out.extend_from_slice(&(channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn raw_f32(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_every_case() {
        the_header_describes_the_data_that_follows();
        raw_f32_is_the_samples_and_nothing_else();
    }

    fn the_header_describes_the_data_that_follows() {
        let pcm = [0.0f32, 1.0, -1.0, 0.5];
        let out = encode(&pcm, 24_000, 2);
        assert_eq!(&out[..4], b"RIFF");
        assert_eq!(&out[8..12], b"WAVE");
        assert_eq!(out.len(), 44 + 8);
        assert_eq!(u32::from_le_bytes(out[4..8].try_into().unwrap()), 36 + 8);
        assert_eq!(u32::from_le_bytes(out[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u16::from_le_bytes(out[22..24].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(out[40..44].try_into().unwrap()), 8);
        assert_eq!(i16::from_le_bytes(out[46..48].try_into().unwrap()), 32767);
        assert_eq!(i16::from_le_bytes(out[48..50].try_into().unwrap()), -32767);
    }

    fn raw_f32_is_the_samples_and_nothing_else() {
        let pcm = [0.25f32, -0.5];
        assert_eq!(
            raw_f32(&pcm),
            [0.25f32.to_le_bytes(), (-0.5f32).to_le_bytes()].concat()
        );
    }
}
