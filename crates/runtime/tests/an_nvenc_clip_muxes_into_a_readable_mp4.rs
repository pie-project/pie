#![cfg(feature = "cuda")]

use media::mp4;
use runtime::inferlet::{Frames, ImageFormat};

fn moving_gradient(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * n * 3) as usize);
    for f in 0..n {
        for y in 0..h {
            for x in 0..w {
                v.push(((x + f * 4) % 256) as u8);
                v.push((y % 256) as u8);
                v.push(((x + y + f * 8) % 256) as u8);
            }
        }
    }
    v
}

#[test]
#[ignore = "needs an NVIDIA encoder; run with PIE_NVENC_TEST=1 -- --ignored"]
fn eight_frames_encode_and_the_container_reads_back() {
    if std::env::var("PIE_NVENC_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PIE_NVENC_TEST=1 to run the NVENC gate");
        return;
    }
    let (w, h, n) = (160u32, 96u32, 8u32);
    let clip = Frames::from_rgb8(moving_gradient(w, h, n), w, h, n, 25.0).expect("build handle");

    let bytes = clip
        .encode(ImageFormat::Mp4H264)
        .expect("NVENC encode + mux");
    assert!(
        bytes.len() > 256,
        "an 8-frame mp4 cannot be {} bytes",
        bytes.len()
    );

    let tree = mp4::parse_boxes(&bytes).expect("the muxer's own walker reads its own file");
    let names: Vec<&str> = tree.iter().map(|b| b.name()).collect();
    assert_eq!(names, ["ftyp", "mdat", "moov"]);
    assert_eq!(
        tree.iter().map(|b| b.size).sum::<usize>(),
        bytes.len(),
        "the walk consumed the file exactly"
    );

    let avcc = mp4::find(&tree, "moov/trak/mdia/minf/stbl/stsd/avc1/avcC").expect("avcC");
    let cfg = &bytes[avcc.body..avcc.offset + avcc.size];
    assert_eq!(cfg[0], 1, "configurationVersion");
    assert_eq!(cfg[4], 0xff, "lengthSizeMinusOne = 3");
    assert_eq!(cfg[5] & 0x1f, 1, "exactly one SPS");
    let sps_len = u16::from_be_bytes(cfg[6..8].try_into().unwrap()) as usize;
    assert!(sps_len >= 4, "SPS is {sps_len} bytes");
    assert_eq!(cfg[8] & 0x1f, 7, "the SPS NAL is really a SPS");

    let stsz = mp4::find(&tree, "moov/trak/mdia/minf/stbl/stsz").expect("stsz");
    let body = &bytes[stsz.body..];
    assert_eq!(u32::from_be_bytes(body[8..12].try_into().unwrap()), n);
    for i in 0..n as usize {
        let at = 12 + i * 4;
        let size = u32::from_be_bytes(body[at..at + 4].try_into().unwrap());
        assert!(size > 0, "sample {i} is empty");
    }

    let mdat = tree.iter().find(|b| b.name() == "mdat").unwrap();
    let stco = mp4::find(&tree, "moov/trak/mdia/minf/stbl/stco").expect("stco");
    let offset =
        u32::from_be_bytes(bytes[stco.body + 8..stco.body + 12].try_into().unwrap()) as usize;
    assert_eq!(offset, mdat.body);
    assert!(offset < mdat.offset + mdat.size);

    let mdhd = mp4::find(&tree, "moov/trak/mdia/mdhd").expect("mdhd");
    let m = &bytes[mdhd.body..];
    assert_eq!(u32::from_be_bytes(m[12..16].try_into().unwrap()), 25_000);
}
