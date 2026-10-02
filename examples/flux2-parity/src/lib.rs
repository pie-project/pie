//! The pie half of the FLUX.2 miniature golden. Hands the `flux2-mini` row
//! exactly what `scripts/imagegen/flux2_golden.py --mini` handed diffusers'
//! `Flux2Transformer2DModel` — the target tokens, the two references'
//! tokens, the raw `[32, 192]` text stack, the timestep, the guidance scale
//! and the four rotary coordinates of every row — and reads back the
//! velocity the head predicts on the target rows, as JSON
//! `scripts/imagegen/flux2_parity.py` turns into an `.npz` under the
//! golden's own key names.
//!
//! # THREE LANES, ONE GROUP, ONE FIRE
//!
//! `flux_2`'s `denoise` reading declares three streams (D2): the text lane
//! (`Stream::Text`, the `context` port), the target lane (`Stream::Image`,
//! the `latents` port) and the reference lane (`Stream::Reference`, the
//! same `latents` port — a reference is more of the same tokens, at rotary
//! `T = 10·(i+1)`). One step is three passes, each stating its stream and
//! all in one group, which packs them `[txt ‖ target ‖ refs]` — the
//! reference's own layout. Every lane binds the timestep (and the guidance
//! scale: the miniature has `guidance_embeds`), because the shared
//! modulation is per lane. Neither `attention` nor `embed` is called: a
//! denoise reading declares no kv space and no tokens.
//!
//! Only the target lane reads out: the head runs on `Stream::Image` rows
//! alone (`pred[:, :S_img]`), so only it carries an epilogue.
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
    /// The raw guidance scale; bound only where the model declares the port.
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

/// The port names this family's `denoise` reading declares, read off
/// `model::readings()` rather than typed in, so a renamed port fails here
/// with the model's own vocabulary instead of at the host.
struct Ports {
    latents: String,
    context: String,
    timestep: String,
    guidance: Option<String>,
    positions: String,
    axes: u32,
    velocity_width: u32,
}

fn ports(reading: &model::ReadingFact) -> Result<Ports> {
    let named = |name: &str| -> Result<&model::PortFact> {
        reading
            .ports
            .iter()
            .find(|port| port.name == name)
            .ok_or_else(|| format!("reading `{}` declares no port `{name}`", reading.name).into())
    };
    Ok(Ports {
        latents: named("latents")?.name.clone(),
        context: named("context")?.name.clone(),
        timestep: named("timestep")?.name.clone(),
        guidance: reading
            .ports
            .iter()
            .find(|port| port.name == "guidance")
            .map(|port| port.name.clone()),
        positions: named("positions")?.name.clone(),
        axes: named("positions")?.width,
        velocity_width: reading.readout_width,
    })
}

/// The per-lane scalars every lane of the reading binds: one cell PER PASS,
/// since a seeded channel attaches to one pass only (the runtime's
/// channel-role rule).
fn scalars(pass: &ForwardPass, ports: &Ports, case: &Case, tag: &str) -> Result<()> {
    let t = Channel::from([case.timestep]).named(&format!("t_{tag}"));
    pass.input(&ports.timestep, &t)?;
    if let Some(guidance) = &ports.guidance {
        let g = Channel::from([case.guidance]).named(&format!("g_{tag}"));
        pass.input(guidance, &g)?;
    }
    Ok(())
}

/// One denoise step: three lanes, one group, one fire, one velocity.
/// One pipeline per lane of the group (see [`step`]).
struct Pipes {
    text: Pipeline,
    reference: Pipeline,
    image: Pipeline,
}

/// ONE PIPELINE PER LANE. A pipeline is serial — the scheduler never seats
/// two of its passes in one step — so three passes down one pipeline are
/// three fires of one lane each, the attention group never forms, and the
/// image lane denoises with no caption and no reference. That reads as a
/// plausible answer: on a miniature whose random-init caption is small, it
/// even passed this file's own tolerance, which is how it went unnoticed
/// until the frame gate began refusing a group that could not compose.
async fn step(case: &Case, ports: &Ports, reading: &str, pipes: &Pipes) -> Result<Vec<f32>> {
    let group = 0;
    let rows = case.image_rows;
    let width = case.channels;

    // The text lane: the raw `[L, joint_attention_dim]` stack, embedded by
    // the plan (the miniature has no encoder).
    let text = ForwardPass::new();
    text.reading(reading)?;
    text.stream(LaneStream::Text)?;
    text.group(group)?;
    let ctx = Channel::from_shaped(
        [case.text_rows, case.context_width],
        case.context.as_slice(),
    )
    .named("context");
    let txt_pos =
        Channel::from_shaped([case.text_rows, ports.axes], case.text_positions.as_slice())
            .named("txt_pos");
    text.input(&ports.context, &ctx)?;
    text.input(&ports.positions, &txt_pos)?;
    scalars(&text, ports, case, "txt")?;

    // The reference lane: every reference's tokens concatenated, at their
    // own `T` offsets. Same port, same weights as the target; no readout.
    let reference = (case.reference_rows > 0).then(|| ForwardPass::new());
    if let Some(reference) = &reference {
        reference.reading(reading)?;
        reference.stream(LaneStream::Reference)?;
        reference.group(group)?;
        let r = Channel::from_shaped([case.reference_rows, width], case.reference.as_slice())
            .named("reference");
        let ref_pos = Channel::from_shaped(
            [case.reference_rows, ports.axes],
            case.reference_positions.as_slice(),
        )
        .named("ref_pos");
        reference.input(&ports.latents, &r)?;
        reference.input(&ports.positions, &ref_pos)?;
        scalars(reference, ports, case, "ref")?;
    }

    // The target lane: the latents in, the velocity out.
    let out = Channel::new([rows, width], dtype::f32).named("velocity");
    let image = ForwardPass::new();
    image.reading(reading)?;
    image.stream(LaneStream::Image)?;
    image.group(group)?;
    let x = Channel::from_shaped([rows, width], case.latents.as_slice()).named("latents");
    let img_pos =
        Channel::from_shaped([rows, ports.axes], case.image_positions.as_slice()).named("img_pos");
    image.input(&ports.latents, &x)?;
    image.input(&ports.positions, &img_pos)?;
    scalars(&image, ports, case, "img")?;
    let velocity_width = ports.velocity_width;
    let readback = out.clone();
    image.epilogue(move || {
        readback.put(intrinsics::velocity(velocity_width));
    });

    text.submit(&pipes.text).context("text lane")?;
    if let Some(reference) = &reference {
        reference
            .submit(&pipes.reference)
            .context("reference lane")?;
    }
    image.submit(&pipes.image).context("image lane")?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err(
            "flux_2 is an attention-kind pass with no kv bound on its denoise reading".into(),
        );
    }
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

    let reading = model::readings()
        .into_iter()
        .find(|reading| reading.name == "denoise")
        .ok_or("this model declares no `denoise` reading")?;
    if reading.has_kv || reading.takes_tokens {
        return Err(format!(
            "reading `{}` binds a kv space or tokens; a denoise pass binds neither",
            reading.name
        )
        .into());
    }
    if reading.readout_width != case.channels {
        return Err(format!(
            "the model reads a {}-wide velocity and the case carries {}-wide tokens",
            reading.readout_width, case.channels
        )
        .into());
    }
    let ports = ports(&reading)?;
    if ports.guidance.is_none() && case.guidance != 0.0 {
        return Err(
            "the case carries a guidance scale and this row declares no guidance port".into(),
        );
    }
    let pipes = Pipes {
        text: Pipeline::new(),
        reference: Pipeline::new(),
        image: Pipeline::new(),
    };
    let velocity = step(&case, &ports, &reading.name, &pipes).await?;
    pipes.text.close();
    pipes.reference.close();
    pipes.image.close();
    Ok(Output {
        velocity,
        image_rows: case.image_rows,
        channels: case.channels,
    })
}
