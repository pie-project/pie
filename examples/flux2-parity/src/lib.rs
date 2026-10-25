//! The pie half of the FLUX.2 miniature golden. Hands the `flux2-mini` row
//! exactly what `scripts/imagegen/flux2_golden.py --mini` handed diffusers'
//! `Flux2Transformer2DModel` — the target tokens, the two references'
//! tokens, the raw `[32, 192]` text stack, the timestep, the guidance scale
//! and the four rotary coordinates of every row — and reads back the
//! velocity the head predicts on the target rows, as JSON
//! `scripts/imagegen/flux2_parity.py` turns into an `.npz` under the
//! golden's own key names.
//!
//! # THREE LANES, ONE GROUP, ONE PASS
//!
//! `flux_2`'s `denoise` reading runs on three values of its `stream` switch
//! (D2): the text lane (`text`, the `context` input), the target lane
//! (`image`, the `latents` input) and the reference lane (`reference`, the
//! same `latents` input — a reference is more of the same tokens, at rotary
//! `T = 10·(i+1)`). One step is one pass of three lanes — lane 0 the text,
//! lane 1 the target, lane 2 the references — each stating its stream and
//! all in one group, which packs them `[txt ‖ target ‖ refs]` — the
//! reference's own layout. Every lane binds the timestep (and the guidance
//! scale: the miniature has `guidance_embeds`), because the shared
//! modulation is per lane; the lanes read one channel of each. Neither
//! `state` nor `embed` is called: a denoise reading declares no kv space
//! and no tokens.
//!
//! Only the target lane reads out: the head runs on `image` rows
//! alone (`pred[:, :S_img]`), so the epilogue reads its velocity.
//!
//! One step only: the miniature golden is one transformer call at
//! `timestep = 0.5`, not a trajectory.
use inferlet::latent::prelude::*;
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
}

/// One batch element of the reference's fixed inputs, flattened row-major:
/// the target tokens (`hidden_states[:, :S_img]`), the reference tokens
/// (`hidden_states[:, S_img:]`), the text stack, the ids as f32 rows, the
/// timestep as the scheduler holds it (`σ·1000`), the raw guidance scale.
#[derive(Deserialize)]
struct Case {
    /// `[image_rows, channels]`.
    latents: Vec<f32>,
    image_rows: u32,
    channels: u32,
    /// `[reference_rows, channels]`; `reference_rows` may be 0.
    #[serde(default)]
    reference: Vec<f32>,
    #[serde(default)]
    reference_rows: u32,
    /// `[text_rows, context_width]`.
    context: Vec<f32>,
    text_rows: u32,
    context_width: u32,
    /// `[rows, 4]` each.
    text_positions: Vec<f32>,
    image_positions: Vec<f32>,
    #[serde(default)]
    reference_positions: Vec<f32>,
    /// `σ · 1000`.
    timestep: f32,
    /// The raw guidance scale; bound only where the model reads `guidance`.
    #[serde(default)]
    guidance: f32,
}

#[derive(Serialize)]
struct Output {
    /// `[image_rows, channels]` — the reference's `mini.out.0[:, :S_img]`.
    velocity: Vec<f32>,
    image_rows: u32,
    channels: u32,
}

/// What this guest sizes and binds off the model's own lane facts: the
/// rotary axes of `positions`, whether the lanes read `guidance`, and the
/// velocity width.
struct Dims {
    axes: u32,
    guidance: bool,
    velocity_width: u32,
}

/// The lane `stream` of `reading`, checked to read exactly `binds`: what
/// this guest binds on that lane, by the model's own input names. Checked
/// up front so a model whose lane reads anything else fails here in its own
/// vocabulary instead of at `program`.
fn lane_reading(reading: &str, stream: &str, binds: &[&str]) -> Result<model::LaneDecl> {
    let lane = model::lane_of(&[("reading", reading), ("stream", stream)])?;
    if lane.reads_kv || lane.embeds {
        return Err(format!(
            "reading `{reading}`'s `{stream}` lane binds a kv space or tokens; a denoise pass \
             binds neither"
        ));
    }
    let mut reads: Vec<&str> = lane.inputs.iter().map(|i| i.name.as_str()).collect();
    let mut wants = binds.to_vec();
    reads.sort_unstable();
    wants.sort_unstable();
    if reads != wants {
        return Err(format!(
            "reading `{reading}`'s `{stream}` lane reads {reads:?}; this pass binds {wants:?}"
        )
        .into());
    }
    Ok(lane)
}

/// The width of the input `name` a lane reads.
fn width_of(lane: &model::LaneDecl, name: &str) -> Result<u32> {
    lane.inputs
        .iter()
        .find(|i| i.name == name)
        .map(|i| i.width)
        .ok_or_else(|| format!("the lane reads no `{name}`").into())
}

fn dims(reading: &str, references: bool) -> Result<Dims> {
    let image = model::lane_of(&[("reading", reading), ("stream", "image")])?;
    let guidance = image.inputs.iter().any(|i| i.name == "guidance");
    let scalars: &[&'static str] = if guidance {
        &["timestep", "guidance"]
    } else {
        &["timestep"]
    };
    let with = |rows: &'static str| [&[rows, "positions"][..], scalars].concat();
    let image = lane_reading(reading, "image", &with("latents"))?;
    lane_reading(reading, "text", &with("context"))?;
    if references {
        lane_reading(reading, "reference", &with("latents"))?;
    }
    let velocity_width = match &image.readout {
        Some(readout) if readout.kind == model::ReadoutKind::Velocity => readout.width,
        other => {
            return Err(format!(
                "reading `{reading}`'s image lane reads out {other:?}, not a velocity"
            ));
        }
    };
    Ok(Dims {
        axes: width_of(&image, "positions")?,
        guidance,
        velocity_width,
    })
}

/// The scalars every lane of the reading binds, on lane `lane`: the one
/// timestep (and guidance) channel of the pass.
fn scalars(pass: &ForwardPass, lane: u32, t: &Channel, g: Option<&Channel>) -> Result<()> {
    pass.input(lane, "timestep", t)?;
    if let Some(g) = g {
        pass.input(lane, "guidance", g)?;
    }
    Ok(())
}

/// One denoise step: three lanes, one group, one pass, one velocity.
async fn step(case: &Case, dims: &Dims, reading: &str, pipe: &Pipeline) -> Result<Vec<f32>> {
    let group = 0;
    let rows = case.image_rows;
    let width = case.channels;
    let pass = ForwardPass::new();
    let t = Channel::from([case.timestep]).named("t");
    let g = dims
        .guidance
        .then(|| Channel::from([case.guidance]).named("g"));

    // Lane 0, the text lane: the raw `[L, joint_attention_dim]` stack,
    // embedded by the plan (the miniature has no encoder).
    pass.switch(0, "reading", reading)?;
    pass.switch(0, "stream", "text")?;
    pass.group(0, group)?;
    let ctx = Channel::from_shaped(
        [case.text_rows, case.context_width],
        case.context.as_slice(),
    )
    .named("context");
    let txt_pos = Channel::from_shaped([case.text_rows, dims.axes], case.text_positions.as_slice())
        .named("txt_pos");
    pass.input(0, "context", &ctx)?;
    pass.input(0, "positions", &txt_pos)?;
    scalars(&pass, 0, &t, g.as_ref())?;

    // Lane 1, the target lane: the latents in, the velocity out.
    pass.switch(1, "reading", reading)?;
    pass.switch(1, "stream", "image")?;
    pass.group(1, group)?;
    let x = Channel::from_shaped([rows, width], case.latents.as_slice()).named("latents");
    let img_pos =
        Channel::from_shaped([rows, dims.axes], case.image_positions.as_slice()).named("img_pos");
    pass.input(1, "latents", &x)?;
    pass.input(1, "positions", &img_pos)?;
    scalars(&pass, 1, &t, g.as_ref())?;

    // Lane 2, the reference lane: every reference's tokens concatenated, at
    // their own `T` offsets. Same input, same weights as the target; no
    // readout.
    if case.reference_rows > 0 {
        pass.switch(2, "reading", reading)?;
        pass.switch(2, "stream", "reference")?;
        pass.group(2, group)?;
        let r = Channel::from_shaped([case.reference_rows, width], case.reference.as_slice())
            .named("reference");
        let ref_pos = Channel::from_shaped(
            [case.reference_rows, dims.axes],
            case.reference_positions.as_slice(),
        )
        .named("ref_pos");
        pass.input(2, "latents", &r)?;
        pass.input(2, "positions", &ref_pos)?;
        scalars(&pass, 2, &t, g.as_ref())?;
    }

    let out = Channel::new([rows, width], dtype::f32).named("velocity");
    let velocity_width = dims.velocity_width;
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::velocity(1, rows, velocity_width));
    });
    pass.submit(pipe).context("the denoise step")?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
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
            return Err("pass `case` (json), `case_0..7` (its pieces) or `case_file`".into());
        }
    };
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;

    let reading = "denoise";
    let dims = dims(reading, case.reference_rows > 0)?;
    if dims.velocity_width != case.channels {
        return Err(format!(
            "the model reads a {}-wide velocity and the case carries {}-wide tokens",
            dims.velocity_width, case.channels
        )
        .into());
    }
    if !dims.guidance && case.guidance != 0.0 {
        return Err("the case carries a guidance scale and this row reads no `guidance`".into());
    }
    let pipe = Pipeline::new();
    let velocity = step(&case, &dims, reading, &pipe).await?;
    pipe.close();
    Ok(Output {
        velocity,
        image_rows: case.image_rows,
        channels: case.channels,
    })
}
