//! The pie half of the FLUX.2 miniature golden: `scripts/imagegen/flux2_golden.py --mini`'s
//! inputs in, the target's velocity out as JSON for `flux2_parity.py`.
//!
//! One pass, one group, packed `[txt ‖ target ‖ refs]` as the reference packs:
//! lane 0 the text, lane 1 the target (the only readout), lanes 2.. the references.
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

/// `[height, width, channels]`, flattened.
#[derive(Deserialize)]
struct Latent {
    latents: Vec<f32>,
    height: u32,
    width: u32,
}

#[derive(Deserialize)]
struct Case {
    target: Latent,
    channels: u32,
    #[serde(default)]
    references: Vec<Latent>,
    /// `[text_rows, context_width]`.
    context: Vec<f32>,
    text_rows: u32,
    context_width: u32,
    /// `σ · 1000`.
    timestep: f32,
    #[serde(default)]
    guidance: f32,
}

#[derive(Serialize)]
struct Output {
    /// `[height · width, channels]`.
    velocity: Vec<f32>,
    height: u32,
    width: u32,
    channels: u32,
}

struct Dims {
    guidance: bool,
    velocity_width: u32,
}

fn lane_reading(reading: &str, stream: &str, binds: &[&str]) -> Result<model::LaneDecl> {
    let lane = model::lane_of(&[("reading", reading), ("stream", stream)])?;
    if lane.state || lane.embeds {
        return Err(format!(
            "reading `{reading}`'s `{stream}` lane binds a working set or tokens; a denoise pass \
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
        ));
    }
    Ok(lane)
}

fn dims(reading: &str, references: bool) -> Result<Dims> {
    let image = model::lane_of(&[("reading", reading), ("stream", "image")])?;
    let guidance = image.inputs.iter().any(|i| i.name == "guidance");
    let scalars: &[&'static str] = if guidance {
        &["timestep", "guidance"]
    } else {
        &["timestep"]
    };
    let with = |rows: &[&'static str]| [rows, scalars].concat();
    let image = lane_reading(reading, "image", &with(&["latents"]))?;
    lane_reading(reading, "text", &with(&["context"]))?;
    if references {
        lane_reading(reading, "reference", &with(&["latents", "reference"]))?;
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
        guidance,
        velocity_width,
    })
}

fn scalars(pass: &ForwardPass, lane: u32, t: &Channel, g: Option<&Channel>) -> Result<()> {
    pass.input(lane, "timestep", t)?;
    if let Some(g) = g {
        pass.input(lane, "guidance", g)?;
    }
    Ok(())
}

fn latent(at: &Latent, channels: u32, name: &str) -> Result<Channel> {
    let want = (at.height * at.width * channels) as usize;
    if at.latents.len() != want {
        return Err(format!(
            "`{name}` carries {} numbers and {}x{}x{channels} is {want}",
            at.latents.len(),
            at.height,
            at.width
        ));
    }
    Ok(Channel::from_shaped([at.height, at.width, channels], at.latents.as_slice()).named(name))
}

async fn step(case: &Case, dims: &Dims, reading: &str, pipe: &Pipeline) -> Result<Vec<f32>> {
    let group = 0;
    let pass = ForwardPass::new();
    let t = Channel::from([case.timestep]).named("t");
    let g = dims
        .guidance
        .then(|| Channel::from([case.guidance]).named("g"));

    // The raw `[L, joint_attention_dim]` stack: the miniature has no encoder.
    pass.switch(0, "reading", reading)?;
    pass.switch(0, "stream", "text")?;
    pass.group(0, group)?;
    let ctx = Channel::from_shaped(
        [case.text_rows, case.context_width],
        case.context.as_slice(),
    )
    .named("context");
    pass.input(0, "context", &ctx)?;
    scalars(&pass, 0, &t, g.as_ref())?;

    pass.switch(1, "reading", reading)?;
    pass.switch(1, "stream", "image")?;
    pass.group(1, group)?;
    let x = latent(&case.target, case.channels, "latents")?;
    pass.input(1, "latents", &x)?;
    scalars(&pass, 1, &t, g.as_ref())?;

    for (k, reference) in case.references.iter().enumerate() {
        let lane = 2 + k as u32;
        pass.switch(lane, "reading", reading)?;
        pass.switch(lane, "stream", "reference")?;
        pass.group(lane, group)?;
        let r = latent(reference, case.channels, &format!("reference{k}"))?;
        let ordinal = Channel::from([k as f32]).named(&format!("ordinal{k}"));
        pass.input(lane, "latents", &r)?;
        pass.input(lane, "reference", &ordinal)?;
        scalars(&pass, lane, &t, g.as_ref())?;
    }

    let voxels = case.target.height * case.target.width;
    let velocity_width = dims.velocity_width;
    let out = Channel::new([voxels, velocity_width], dtype::f32).named("velocity");
    pass.epilogue(move || {
        out.put(intrinsics::velocity(1, voxels, velocity_width));
    });
    pass.submit(pipe).context("the denoise step")?;
    out.take_host::<Vec<f32>>().await
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
    let dims = dims(reading, !case.references.is_empty())?;
    if dims.velocity_width != case.channels {
        return Err(format!(
            "the model reads a {}-wide velocity and the case's latent has {} channels",
            dims.velocity_width, case.channels
        ));
    }
    if !dims.guidance && case.guidance != 0.0 {
        return Err("the case carries a guidance scale and this row reads no `guidance`".into());
    }
    let pipe = Pipeline::new();
    let velocity = step(&case, &dims, reading, &pipe).await?;
    pipe.close();
    Ok(Output {
        velocity,
        height: case.target.height,
        width: case.target.width,
        channels: case.channels,
    })
}
