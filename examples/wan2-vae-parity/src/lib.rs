//! The pie half of `scripts/imagegen/wan22_vae_parity.py`: decodes a `T`-frame
//! latent as `text-to-video` does, one pass per latent frame over one working
//! set, and sends the pixels back as one raw `[F*H*W, 3]` f32 file.

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
    fires: Vec<String>,
    chunk_frames: Vec<u32>,
    /// Max |difference| of the last chunk decoded on a fork from its parent's.
    fork_max_abs_diff: f32,
    pixel_h: u32,
    pixel_w: u32,
    pixels_bytes: u32,
}

/// The `pixels` output a reading's video lane reads out, if any.
fn pixels(reading: &str) -> Option<model::OutputDecl> {
    model::lane_of(&[("reading", reading), ("stream", "video")])
        .ok()?
        .outputs
        .into_iter()
        .find(|output| output.name == "pixels")
}

fn decoder() -> Result<String> {
    model::switch("reading")
        .map(|decl| decl.values)
        .unwrap_or_default()
        .into_iter()
        .find(|r| !r.starts_with("vae.encode") && pixels(r).is_some())
        .ok_or_else(|| "this model declares no `vae.decode` reading".into())
}

fn voxel_input(reading: &str) -> Result<model::InputDecl> {
    match &model::lane_of(&[("reading", reading), ("stream", "video")])?.inputs[..] {
        [input] if input.axis == model::InputAxis::Voxel => Ok(input.clone()),
        other => Err(format!(
            "reading `{reading}` reads {} input(s), and a VAE arm reads one clip on the voxel axis",
            other.len()
        )),
    }
}

#[allow(clippy::too_many_arguments)]
async fn decode_frame(
    reading: &str,
    clip: &[f32],
    clip_h: u32,
    clip_w: u32,
    frames: u32,
    pixel_h: u32,
    pixel_w: u32,
    state: &WorkingSet,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let input = voxel_input(reading)?;
    let rows = frames * pixel_h * pixel_w;
    let width = pixels(reading)
        .map(|pixels| pixels.width)
        .ok_or_else(|| format!("reading `{reading}` reads no pixels out on its `video` lane"))?;
    let pass = ForwardPass::new();
    pass.switch(0, "reading", reading)?;
    pass.switch(0, "stream", "video")?;
    pass.state(&[state], StateGeometry::default())?;
    let cell =
        Channel::from_shaped([clip_h, clip_w, input.width], clip.to_vec()).named("vae_latent");
    pass.input(0, &input.name, &cell)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_pixels");
    let readback = out;
    pass.epilogue(move || {
        readback.put(intrinsics::output_rows(0, "pixels", rows));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{reading}` fire"))?;
    out.take_host::<Vec<f32>>().await
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
        ));
    }
    let space = model::latent().ok_or("this model states no latent space")?;
    if space.channels != case.channels {
        return Err(format!(
            "the case is {} channels and the model's latent is {}",
            case.channels, space.channels
        ));
    }
    let (tc, sc) = (space.temporal_compression, space.spatial_compression);
    let (pixel_h, pixel_w) = (case.h * sc, case.w * sc);
    let decode = decoder()?;

    let state = WorkingSet::new();
    let pipe = Pipeline::new();
    let mut pixels: Vec<f32> = Vec::new();
    let mut fires = Vec::new();
    let mut chunk_frames = Vec::new();
    let mut fork_max_abs_diff = 0.0f32;
    for k in 0..case.t {
        // A fresh state lands one frame, every later one `tc`.
        let frames = if k == 0 { 1 } else { tc };
        let at = k as usize * plane;
        // The last chunk also decodes on a fork, which must match its parent.
        let fork = (k + 1 == case.t && k > 0)
            .then(|| state.fork(&pipe))
            .transpose()?;
        let mut outs = Vec::new();
        for ws in fork.iter().chain(std::iter::once(&state)) {
            outs.push(
                decode_frame(
                    &decode,
                    &case.latent[at..at + plane],
                    case.h,
                    case.w,
                    frames,
                    pixel_h,
                    pixel_w,
                    ws,
                    &pipe,
                )
                .await?,
            );
        }
        let out = outs.pop().ok_or("no decode")?;
        let forked = outs.pop();
        if let Some(forked) = forked {
            fork_max_abs_diff = forked
                .iter()
                .zip(&out)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
        }
        pixels.extend_from_slice(&out);
        fires.push(decode.clone());
        chunk_frames.push(frames);
    }
    pipe.close();

    let raw: Vec<u8> = pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
    session::send_file(&raw);
    Ok(Output {
        fires,
        chunk_frames,
        fork_max_abs_diff,
        pixel_h,
        pixel_w,
        pixels_bytes: raw.len() as u32,
    })
}
