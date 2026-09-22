//! The `frames` output path, end to end, with no model in it.
//!
//! `session.send-file` had no inferlet exercising it and `pie run` never wrote
//! what it received, so the whole binary-output half of the guest API was
//! untested (codebase-guest-api.md §4.2). This is the fixture that closes
//! that: it builds pixels with `frames.from-rgb8`, hands the handle to
//! `session.send-frames`, and never touches the encoded bytes itself — which
//! is the property design.md D11 is about, and the one a test can only observe
//! from outside, by the files arriving.
//!
//! No forward pass, deliberately. What is under test is the door, and putting
//! a model behind it would make a failure ambiguous.

use inferlet::Result;
use inferlet::frames::{AudioFormat, Frames, ImageFormat, Pcm};
use inferlet::session;
use serde::Deserialize;

#[derive(Deserialize, Default)]
struct Input {
    width: Option<u32>,
    height: Option<u32>,
    frames: Option<u32>,
    fps: Option<f32>,
    /// `mp4` (NVENC) or `y4m` (uncompressed, works anywhere).
    video: Option<String>,
}

/// A horizontal/vertical gradient whose blue channel advances with the frame,
/// so a decoder can tell the frames apart and a transposed encoder shows up.
fn gradient(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * n * 3) as usize);
    for f in 0..n {
        for y in 0..h {
            for x in 0..w {
                v.push((x * 255 / w.max(1)) as u8);
                v.push((y * 255 / h.max(1)) as u8);
                v.push((f * 255 / n.max(1)) as u8);
            }
        }
    }
    v
}

/// A quarter-second 440 Hz sine, so the wav that comes out is something a
/// person can check by ear rather than only by header.
fn tone(rate: u32, ms: u32) -> Vec<f32> {
    let n = (rate as u64 * ms as u64 / 1000) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            (t * 440.0 * std::f32::consts::TAU).sin() * 0.25
        })
        .collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let width = input.width.unwrap_or(160);
    let height = input.height.unwrap_or(96);
    let count = input.frames.unwrap_or(8);
    let fps = input.fps.unwrap_or(25.0);

    // One still: frame 0 on its own, so the png path gets a `count == 1`
    // handle rather than the clip refusing it.
    let still = Frames::from_rgb8(&gradient(width, height, 1), width, height, 1, 0.0)?;
    session::send_frames(&still, ImageFormat::Png, "gradient.png")?;

    // The clip. The bytes below never exist in this module's memory: the
    // handle goes over, the runtime encodes, the client receives.
    let clip = Frames::from_rgb8(&gradient(width, height, count), width, height, count, fps)?;
    let asked = input.video.as_deref().unwrap_or("mp4");
    let video = match asked {
        "y4m" => ImageFormat::Y4m,
        "mp4" => ImageFormat::Mp4H264,
        other => return Err(format!("unknown video format {other:?}; try 'mp4' or 'y4m'")),
    };
    // A machine with no encoder is a fact about the machine, not a failure of
    // this program: report it and fall back, so the fixture still proves the
    // door works where it can.
    let (video_name, video_note) = match session::send_frames(&clip, video, "gradient") {
        Ok(()) => (
            if matches!(video, ImageFormat::Mp4H264) { "gradient.mp4" } else { "gradient.y4m" },
            String::new(),
        ),
        Err(error) => {
            session::send_frames(&clip, ImageFormat::Y4m, "gradient")?;
            ("gradient.y4m", error)
        }
    };

    // And the audio half of the same statute.
    let pcm = Pcm::from_f32(&tone(24_000, 250), 24_000, 1)?;
    session::send_pcm(&pcm, AudioFormat::Wav, "tone.wav")?;

    // The one place the guest DOES take the bytes: `encode` is the other door,
    // and a probe should prove both exist. Only the length is reported.
    let png_len = still.encode(ImageFormat::Png)?.len();

    Ok(format!(
        "{{\"width\":{width},\"height\":{height},\"frames\":{},\"fps\":{fps},\
          \"files\":[\"gradient.png\",\"{video_name}\",\"tone.wav\"],\
          \"png_bytes\":{png_len},\"video_fallback\":{}}}",
        clip.count(),
        inferlet::serde_json::to_string(&video_note).unwrap_or_else(|_| "\"\"".to_string()),
    ))
}
