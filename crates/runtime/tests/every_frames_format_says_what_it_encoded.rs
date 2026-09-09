use runtime::inferlet::{Frames, ImageFormat, Pcm};

fn gradient(w: u32, h: u32, count: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * count * 3) as usize);
    for f in 0..count {
        for y in 0..h {
            for x in 0..w {
                v.push((x * 255 / w) as u8);
                v.push((y * 255 / h) as u8);
                v.push((f * 255 / count) as u8);
            }
        }
    }
    v
}

#[test]
fn every_frames_format_says_what_it_encoded_every_case() {
    a_still_round_trips_through_png_and_raw_and_reads_back_identical();
    a_clip_encodes_as_y4m_and_refuses_the_still_formats_by_name();
    a_handle_whose_bytes_do_not_match_its_extent_is_refused_at_construction();
    pcm_carries_its_own_rate_and_channels_into_the_wav_header();
}

fn a_still_round_trips_through_png_and_raw_and_reads_back_identical() {
    let (w, h) = (16, 8);
    let src = gradient(w, h, 1);
    let still = Frames::from_rgb8(src.clone(), w, h, 1, 0.0).expect("build");
    assert_eq!((still.width, still.height, still.count), (w, h, 1));

    let raw = still.encode(ImageFormat::RawRgb8).expect("raw");
    assert_eq!(raw, src, "raw-rgb8 is the handle's own bytes");

    let png = still.encode(ImageFormat::Png).expect("png");
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory(&png).expect("decode").to_rgb8();
    assert_eq!(decoded.dimensions(), (w, h));
    assert_eq!(
        decoded.into_raw(),
        src,
        "png is lossless, so it must be exact"
    );

    let webp = still.encode(ImageFormat::Webp).expect("webp");
    assert_eq!(&webp[..4], b"RIFF");
    assert_eq!(&webp[8..12], b"WEBP");

    let jpeg = still.encode(ImageFormat::Jpeg).expect("jpeg");
    assert_eq!(&jpeg[..2], b"\xff\xd8");
}

fn a_clip_encodes_as_y4m_and_refuses_the_still_formats_by_name() {
    let (w, h, n) = (8, 4, 3);
    let clip = Frames::from_rgb8(gradient(w, h, n), w, h, n, 24.0).expect("build");

    let y4m = clip.encode(ImageFormat::Y4m).expect("y4m");
    assert!(y4m.starts_with(b"YUV4MPEG2 W8 H4 F24000:1000 "));
    assert_eq!(
        y4m.windows(6).filter(|c| *c == b"FRAME\n").count(),
        n as usize,
        "one FRAME marker per frame"
    );

    for (format, word) in [
        (ImageFormat::Png, "png"),
        (ImageFormat::Jpeg, "jpeg"),
        (ImageFormat::Webp, "webp"),
    ] {
        let err = clip.encode(format).unwrap_err();
        assert!(err.contains(word) && err.contains("3 frames"), "{err}");
    }
}

fn a_handle_whose_bytes_do_not_match_its_extent_is_refused_at_construction() {
    let err = Frames::from_rgb8(vec![0; 10], 4, 4, 1, 0.0).unwrap_err();
    assert!(err.contains("expected 48"), "{err}");
    let err = Frames::from_rgb8(Vec::new(), 0, 4, 1, 0.0).unwrap_err();
    assert!(err.contains("non-zero extent"), "{err}");
}

fn pcm_carries_its_own_rate_and_channels_into_the_wav_header() {
    let samples: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
    let pcm = Pcm::from_f32(samples.clone(), 24_000, 2).expect("build");
    let wav = pcm.encode(runtime::inferlet::AudioFormat::Wav);
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
    assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 2);
    assert_eq!(wav.len(), 44 + samples.len() * 2);

    let err = Pcm::from_f32(vec![0.0; 3], 24_000, 2).unwrap_err();
    assert!(err.contains("do not divide"), "{err}");
}
