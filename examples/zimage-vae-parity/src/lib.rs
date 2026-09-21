//! **A VAE READING, FIRED FROM A GUEST, END TO END.** The pie half of
//! `scripts/imagegen/zimage_vae_parity.py`, and the first thing in the tree
//! that walks the whole of design D8's guest road:
//!
//! 1. the model's own catalog says which reading decodes (`readings()`, the
//!    one with a `Voxels` port and a `Pixels` readout) and how wide its port
//!    is — nothing here hardcodes `z-image`;
//! 2. the golden latent is bound as that port's CHANNEL, whose shape IS the
//!    clip's box (`[h, w, 16]`), so the geometry the port needs and the
//!    numbers it reads travel together and neither is a payload the guest
//!    hands the shell;
//! 3. the epilogue reads the `pixels` seam through `intrinsics::pixels(rows,
//!    3)` — `8h·8w` rows of RGB in `[-1, 1]` — and puts it on one channel;
//! 4. that channel becomes a `frames` handle through
//!    `Channel::take_frames`, and `session::send_frames` streams it to the
//!    client as a PNG. The pixels never enter linear memory on that road.
//!
//! `--pixels_out true` additionally lifts them into the JSON answer, which
//! is what the parity harness diffs against `pixels.f32`; the PNG path and
//! the numbers path read the SAME channel cell, so a run that wants both
//! asks for the numbers and lets this file re-encode them (see `main`).
//!
//! A lane of a VAE reading embeds no token and binds no `[rows, ·]` port: its
//! rows are its clip's voxels, on the third axis. The library states that for the
//! guest — `pass.input` on a voxel port is the only row-stating thing it
//! needs — so nothing below mentions a row count the model did not.

use inferlet::latent::prelude::*;
use inferlet::{frames, session};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    case: Option<String>,
    #[serde(default)]
    case_file: Option<String>,
    #[serde(default)]
    case_0: Option<String>,
    #[serde(default)]
    case_1: Option<String>,
    #[serde(default)]
    case_2: Option<String>,
    #[serde(default)]
    case_3: Option<String>,
    #[serde(default)]
    case_4: Option<String>,
    #[serde(default)]
    case_5: Option<String>,
    #[serde(default)]
    case_6: Option<String>,
    #[serde(default)]
    case_7: Option<String>,
    #[serde(default)]
    case_8: Option<String>,
    #[serde(default)]
    case_9: Option<String>,
    #[serde(default)]
    case_10: Option<String>,
    #[serde(default)]
    case_11: Option<String>,
    #[serde(default)]
    case_12: Option<String>,
    #[serde(default)]
    case_13: Option<String>,
    #[serde(default)]
    case_14: Option<String>,
    #[serde(default)]
    case_15: Option<String>,
    #[serde(default)]
    png: Option<String>,
    #[serde(default)]
    pixels_out: bool,
    #[serde(default)]
    pixels_file: bool,
}

/// The golden clip: the latent as `[h, w, channels]` row-major, and the box
/// the decoder's answer is expected at.
#[derive(Deserialize)]
struct Case {
    /// `[h * w * channels]`, `w` fastest then `channels` — the port
    /// channel's own order.
    latent: Vec<f32>,
    h: u32,
    w: u32,
    channels: u32,
    /// The VAE's spatial compression, so the guest can size the answer
    /// without asking the reference: `8` for every FLUX-class autoencoder.
    /// Cross-checked against `model::latent()` when the family states one.
    compression: u32,
}

#[derive(Serialize)]
struct Output {
    /// Which reading fired, by the family's own name.
    reading: String,
    /// Which port carried the clip.
    port: String,
    /// The clip's box, and the pixels' box.
    latent_h: u32,
    latent_w: u32,
    latent_channels: u32,
    pixel_h: u32,
    pixel_w: u32,
    pixel_channels: u32,
    /// How many pixel rows came back.
    rows: u32,
    /// The pixel rows themselves, `[rows, 3]` row-major in `[-1, 1]`; empty
    /// unless `--pixels_out true`.
    pixels: Vec<f32>,
    /// What was sent to the client, or `None`.
    png: Option<String>,
    /// How many bytes of raw `[rows, 3]` f32 went out as a file (`pie run -o
    /// DIR` writes it as `file-NNNN.bin`); `0` when none did.
    pixels_bytes: u32,
    /// The picture's own arithmetic, so a run that did not ask for the rows
    /// still says something falsifiable.
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    non_finite: u32,
}

/// The family's `vae.decode`: the token-less reading whose readout is
/// `Pixels` and whose input port is a `Voxels` one. Read off the catalog so
/// a family that renames either fails here with its own vocabulary.
struct Decode {
    name: String,
    port: String,
    channels: u32,
    readout_width: u32,
}

fn decode_reading() -> Result<Decode> {
    let candidates: Vec<model::ReadingFact> = model::readings()
        .into_iter()
        .filter(|r| r.readout == model::ReadoutKind::Pixels && !r.takes_tokens)
        .collect();
    if candidates.is_empty() {
        return Err("this model declares no reading that lands pixels; \
                    it has no VAE decoder to fire"
            .into());
    }
    // A VAE states two pixel readings — decode and encode. The DECODER is
    // the one whose voxel port is WIDER than its answer: it reads a latent
    // and lands RGB, where the encoder reads RGB and lands a latent.
    let mut best: Option<(model::ReadingFact, model::PortFact)> = None;
    for reading in candidates {
        let Some(port) = reading
            .ports
            .iter()
            .find(|p| p.kind == model::PortKind::Voxels)
            .cloned()
        else {
            continue;
        };
        if port.width > reading.readout_width {
            best = Some((reading, port));
            break;
        }
    }
    let (reading, port) = best.ok_or_else(|| {
        "no pixels reading of this model reads a voxel port wider than its answer; \
         a decoder takes a latent clip and lands RGB"
            .to_string()
    })?;
    if reading.has_kv {
        return Err(format!(
            "reading `{}` binds a kv space; a VAE tile binds none",
            reading.name
        )
        .into());
    }
    Ok(Decode {
        name: reading.name.clone(),
        port: port.name.clone(),
        channels: port.width,
        readout_width: reading.readout_width,
    })
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let pieces: String = [
        &input.case_0,
        &input.case_1,
        &input.case_2,
        &input.case_3,
        &input.case_4,
        &input.case_5,
        &input.case_6,
        &input.case_7,
        &input.case_8,
        &input.case_9,
        &input.case_10,
        &input.case_11,
        &input.case_12,
        &input.case_13,
        &input.case_14,
        &input.case_15,
    ]
    .into_iter()
    .flatten()
    .map(String::as_str)
    .collect();
    let text = match (&input.case, &input.case_file) {
        (Some(text), _) => text.clone(),
        (None, Some(name)) => std::fs::read_to_string(format!("/scratch/{name}"))
            .map_err(|why| format!("reading /scratch/{name}: {why}"))?,
        (None, None) if !pieces.is_empty() => pieces,
        (None, None) => {
            return Err("pass `case` (json), `case_0..15` (its pieces) or `case_file`".into());
        }
    };
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;

    let d = decode_reading()?;
    if case.channels != d.channels {
        return Err(format!(
            "the case's latent is {} channels wide and reading `{}` reads a {}-channel \
             voxel port",
            case.channels, d.name, d.channels
        )
        .into());
    }
    let want = (case.h as usize) * (case.w as usize) * (case.channels as usize);
    if case.latent.len() != want {
        return Err(format!(
            "`latent` carries {} numbers and the case's box is {}x{}x{} = {want}",
            case.latent.len(),
            case.h,
            case.w,
            case.channels
        )
        .into());
    }
    // The family's own compression wins when it states one.
    let compression = match model::latent() {
        Some(space) if space.spatial_compression > 0 => space.spatial_compression,
        _ => case.compression,
    };
    if compression == 0 {
        return Err("the case states no spatial compression and the model states none".into());
    }
    let (pixel_h, pixel_w) = (case.h * compression, case.w * compression);
    let rows = pixel_h * pixel_w;
    let width = d.readout_width;

    // The port channel IS the clip: its shape is the box (design D8), and the
    // shell reads the committed cell straight into the voxel payload.
    let pipe = Pipeline::new();
    let pass = ForwardPass::new();
    pass.reading(&d.name)?;
    pass.stream(LaneStream::Image)?;
    let clip = Channel::from_shaped([case.h, case.w, case.channels], case.latent.as_slice())
        .named("latent");
    pass.input(&d.port, &clip)?;

    // The answer: one row per OUTPUT voxel, RGB in [-1, 1].
    let out = Channel::new([rows, width], dtype::f32).named("pixels");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(rows, width));
    });
    pass.submit(&pipe).context("the vae.decode lane")?;

    // The cell is taken ONCE, and which door it goes out of is the caller's.
    //
    //   * `--png` alone is the real road: `take_frames` hands the plane
    //     straight to the host encoders and no pixel ever enters this
    //     module's address space.
    //   * `--pixels_file` is the PARITY road: the rows are lifted here and
    //     sent back as raw `[rows, 3]` f32 bytes, which is what the harness
    //     diffs against the reference dump. A 512² picture is 786 432
    //     numbers — far past what an answer document can carry — so this is a
    //     file, not a field.
    //   * `--pixels_out` puts them in the answer instead, for a case small
    //     enough that a person wants to read them.
    //
    // Asking for a lift AND a PNG re-encodes the rows this guest now holds,
    // so the harness can prove the picture and the numbers are the same
    // picture.
    let lift = input.pixels_out || input.pixels_file;
    let mut pixels: Vec<f32> = Vec::new();
    let mut sent = None;
    let mut pixels_bytes = 0u32;
    if lift {
        pixels = out.take_host().await?;
        if input.pixels_file {
            let raw: Vec<u8> = pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
            pixels_bytes = raw.len() as u32;
            session::send_file(&raw);
        }
        if let Some(name) = &input.png {
            let rgb: Vec<u8> = pixels
                .iter()
                .map(|v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            let handle = frames::Frames::from_rgb8(&rgb, pixel_w, pixel_h, 1, 0.0)
                .map_err(|why| format!("frames.from-rgb8: {why}"))?;
            session::send_frames(&handle, frames::ImageFormat::Png, name)
                .map_err(|why| format!("session.send-frames: {why}"))?;
            sent = Some(name.clone());
        }
    } else if let Some(name) = &input.png {
        let handle = out.take_frames(pixel_w, pixel_h, 1, 0.0)?;
        session::send_frames(&handle, frames::ImageFormat::Png, name)
            .map_err(|why| format!("session.send-frames: {why}"))?;
        sent = Some(name.clone());
    } else {
        // Nothing asked for: still take the cell, so the fire settles and the
        // facts below are the fire's own.
        pixels = out.take_host().await?;
    }
    pipe.close();

    let (mean, std, min, max, non_finite) = stats(&pixels);
    Ok(Output {
        reading: d.name,
        port: d.port,
        latent_h: case.h,
        latent_w: case.w,
        latent_channels: case.channels,
        pixel_h,
        pixel_w,
        pixel_channels: width,
        rows,
        pixels: if input.pixels_out { pixels } else { Vec::new() },
        png: sent,
        pixels_bytes,
        mean,
        std,
        min,
        max,
        non_finite,
    })
}

/// Mean, standard deviation, extremes and the non-finite count. All zero for
/// an empty slice — a run that took the zero-copy road holds no rows here,
/// and says so by the `png` field rather than by pretending to a statistic.
fn stats(values: &[f32]) -> (f32, f32, f32, f32, u32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let non_finite = values.iter().filter(|v| !v.is_finite()).count() as u32;
    (mean, var.sqrt(), min, max, non_finite)
}
