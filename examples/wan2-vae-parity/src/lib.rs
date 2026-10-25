//! **THE WAN 2.2 CAUSAL VAE, DECODED THROUGH THE RUNTIME.** The pie half of
//! `scripts/imagegen/wan22_vae_parity.py`.
//!
//! The engine gate `the_wan_2_vae_answers_the_reference` drives the shell
//! from the host; this guest decodes a `T`-frame latent the way
//! `examples/text-to-video` does: one fire per latent frame down ONE
//! pipeline, the `.head` arm first and closed before the later arm opens,
//! the pipeline closed last. Every chunk after the head reads the conv
//! caches the fire before it left, so a runtime that hands a later fire
//! cold caches shows up as a chunk that drifts.
//!
//! The pixels go back to the client as one raw `[F*H*W, 3]` f32 file
//! (`pie run -o DIR` writes it as `file-0000.bin`).

use inferlet::latent::prelude::*;
use inferlet::session;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
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
}

/// The reference latent: `[t, h, w, channels]` row-major, DiT space.
#[derive(Deserialize)]
struct Case {
    latent: Vec<f32>,
    t: u32,
    h: u32,
    w: u32,
    channels: u32,
}

#[derive(Serialize)]
struct Output {
    /// The arm each fire used, in order.
    fires: Vec<String>,
    /// Output frames each fire landed.
    chunk_frames: Vec<u32>,
    pixel_h: u32,
    pixel_w: u32,
    /// Bytes of raw f32 sent as the file.
    pixels_bytes: u32,
}

/// The decode arms, found as `text-to-video` finds them.
struct Arms {
    head: String,
    decode: String,
}

/// What `reading`'s video lane reads out.
fn readout(reading: &str) -> Option<model::Readout> {
    model::lane_of(&[("reading", reading), ("stream", "video")])
        .ok()?
        .readout
}

fn arms() -> Result<Arms> {
    let decoders: Vec<String> = model::switch("reading")
        .map(|decl| decl.values)
        .unwrap_or_default()
        .into_iter()
        .filter(|r| {
            !r.starts_with("vae.encode")
                && readout(r).is_some_and(|readout| readout.kind == model::ReadoutKind::Pixels)
        })
        .collect();
    let head = decoders.iter().find(|r| r.ends_with(".head")).cloned();
    let decode = decoders.iter().find(|r| !r.ends_with(".head")).cloned();
    match (head, decode) {
        (Some(head), Some(decode)) => Ok(Arms { head, decode }),
        _ => Err(
            "this model declares no causal `vae.decode` pair (a `.head` arm and the rest)".into(),
        ),
    }
}

/// The one input a VAE arm's lane reads: a clip on the voxel axis.
fn voxel_input(reading: &str) -> Result<model::InputDecl> {
    match &model::lane_of(&[("reading", reading), ("stream", "video")])?.inputs[..] {
        [input] if input.axis == model::InputAxis::Voxel => Ok(input.clone()),
        other => Err(format!(
            "reading `{reading}` reads {} input(s), and a VAE arm reads one clip on the voxel axis",
            other.len()
        )
        .into()),
    }
}

/// Fire ONE decode arm over ONE latent frame and take its pixels back —
/// `text-to-video`'s `decode_frame`, unchanged: the pass dies with the call.
#[allow(clippy::too_many_arguments)]
async fn decode_frame(
    reading: &str,
    clip: &[f32],
    clip_h: u32,
    clip_w: u32,
    frames: u32,
    pixel_h: u32,
    pixel_w: u32,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let input = voxel_input(reading)?;
    let rows = frames * pixel_h * pixel_w;
    let width = readout(reading)
        .map(|readout| readout.width)
        .ok_or_else(|| format!("reading `{reading}` reads nothing out on its `video` lane"))?;
    let pass = ForwardPass::new();
    pass.switch(0, "reading", reading)?;
    pass.switch(0, "stream", "video")?;
    let cell =
        Channel::from_shaped([clip_h, clip_w, input.width], clip.to_vec()).named("vae_latent");
    pass.input(0, &input.name, &cell)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_pixels");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(0, rows, width));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{reading}` fire"))?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let text: String = [
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
    if text.is_empty() {
        return Err("pass the case as `case_0..15`".into());
    }
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;
    let plane = (case.h * case.w * case.channels) as usize;
    if case.latent.len() != plane * case.t as usize {
        return Err(format!(
            "`latent` carries {} numbers and the box is {}x{}x{}x{}",
            case.latent.len(),
            case.t,
            case.h,
            case.w,
            case.channels
        )
        .into());
    }
    let space = model::latent().ok_or("this model states no latent space")?;
    if space.channels != case.channels {
        return Err(format!(
            "the case is {} channels and the model's latent is {}",
            case.channels, space.channels
        )
        .into());
    }
    let (tc, sc) = (space.temporal_compression, space.spatial_compression);
    let (pixel_h, pixel_w) = (case.h * sc, case.w * sc);
    let a = arms()?;

    // ONE pipeline; the head fire first, its pass closed before the rest.
    let pipe = Pipeline::new();
    let mut pixels: Vec<f32> = Vec::new();
    let mut fires = Vec::new();
    let mut chunk_frames = Vec::new();
    for k in 0..case.t {
        let (arm, frames) = if k == 0 {
            (&a.head, 1)
        } else {
            (&a.decode, tc)
        };
        let at = k as usize * plane;
        let out = decode_frame(
            arm,
            &case.latent[at..at + plane],
            case.h,
            case.w,
            frames,
            pixel_h,
            pixel_w,
            &pipe,
        )
        .await?;
        pixels.extend_from_slice(&out);
        fires.push(arm.clone());
        chunk_frames.push(frames);
    }
    pipe.close();

    let raw: Vec<u8> = pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
    session::send_file(&raw);
    Ok(Output {
        fires,
        chunk_frames,
        pixel_h,
        pixel_w,
        pixels_bytes: raw.len() as u32,
    })
}
