//! The pie half of the LTX-2.5 miniature golden. Hands an `ltx25-mini` row
//! exactly what `scripts/imagegen/ltx2_golden.py --mini` handed the vendored
//! reference — one joint video+audio denoise step, or one connector pass —
//! and reads back what the reference returned, as JSON
//! `scripts/imagegen/ltx2_parity.py` turns into an `.npz` under the golden's
//! own key names.
//!
//! # FOUR LANES, ONE GROUP, ONE FIRE
//!
//! `ltx_2`'s `denoise` reading declares four streams (D2), and a step is one
//! lane of each:
//!
//! | lane | binds | reads back |
//! |---|---|---|
//! | `Video` | `latents`, `positions` (3 axes), `timestep` | `velocity` |
//! | `Audio` | `latents`, `audio_positions` (1 axis), `timestep` | `velocity` |
//! | `Context` | `context` (the video text rows), `timestep` | — |
//! | `Reference` | `audio_context` (the audio text rows), `timestep` | — |
//!
//! The two context lanes carry a timestep because they modulate their OWN
//! rows: the reference scales and shifts the text context by
//! `prompt_scale_shift_table + prompt_adaln_single(t)`, and a modulation
//! vector is computed on the arm whose rows it applies to.
//!
//! **ONE PIPELINE PER LANE.** A group is a fact about a FIRE: the passes
//! join one attention only when they are members of one step. The scheduler
//! seals a frame when every live pipeline has submitted and never seats two
//! passes of one pipeline in one step, so the four lanes go down four
//! pipelines, submitted back to back; the runtime holds a fresh group's
//! first frame for its cohort.
//!
//! # THE CONNECTOR PASS
//!
//! `refine.video` and `refine.audio` are two readings over ONE rectangle of
//! packed trunk rows (`text`, `caption·49` wide) and one column of rope
//! coordinates. They are separate passes on separate pipelines and share
//! nothing but their input, so the case runs them back to back and reports
//! both `hidden` readouts.
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
    #[serde(default)]
    case_8: Option<String>,
    #[serde(default)]
    case_9: Option<String>,
    #[serde(default)]
    case_10: Option<String>,
    #[serde(default)]
    case_11: Option<String>,
}

/// The reference's fixed inputs, flattened row-major. `kind` says which of
/// the two passes to run; the fields the other pass wants are absent.
#[derive(Deserialize)]
struct Case {
    /// `"denoise"` or `"refine"`.
    kind: String,

    // --- denoise ---------------------------------------------------------
    #[serde(default)]
    latents: Vec<f32>,
    #[serde(default)]
    rows: u32,
    #[serde(default)]
    channels: u32,
    #[serde(default)]
    audio_latents: Vec<f32>,
    #[serde(default)]
    audio_rows: u32,
    #[serde(default)]
    context: Vec<f32>,
    #[serde(default)]
    context_rows: u32,
    #[serde(default)]
    context_width: u32,
    #[serde(default)]
    audio_context: Vec<f32>,
    #[serde(default)]
    audio_context_width: u32,
    /// `[rows, 3]`, already normalised to the angle scale.
    #[serde(default)]
    positions: Vec<f32>,
    /// `[audio_rows, 1]`.
    #[serde(default)]
    audio_positions: Vec<f32>,
    #[serde(default)]
    timestep: f32,
    #[serde(default)]
    audio_timestep: f32,

    // --- refine ----------------------------------------------------------
    #[serde(default)]
    text: Vec<f32>,
    #[serde(default)]
    text_rows: u32,
    #[serde(default)]
    text_width: u32,
    /// `[text_rows, 1]`.
    #[serde(default)]
    text_positions: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    kind: String,
    /// `denoise`: the video velocity `[rows, channels]`; `refine`: the video
    /// context `[text_rows, video_width]`.
    video: Vec<f32>,
    /// `denoise`: the audio velocity; `refine`: the audio context.
    audio: Vec<f32>,
    video_rows: u32,
    video_width: u32,
    audio_rows: u32,
    audio_width: u32,
}

/// The port a reading declares under `name`, refused by the model's own
/// vocabulary rather than at the host.
fn named<'a>(reading: &'a model::ReadingFact, name: &str) -> Result<&'a model::PortFact> {
    reading
        .ports
        .iter()
        .find(|port| port.name == name)
        .ok_or_else(|| format!("reading `{}` declares no port `{name}`", reading.name).into())
}

fn reading_named(name: &str) -> Result<model::ReadingFact> {
    model::readings()
        .into_iter()
        .find(|reading| reading.name == name)
        .ok_or_else(|| format!("this model declares no `{name}` reading").into())
}

/// One lane: its pass, on its own pipeline, in one group.
fn lane(reading: &str, stream: LaneStream, group: u32) -> Result<ForwardPass> {
    let pass = ForwardPass::new();
    pass.reading(reading)?;
    pass.stream(stream)?;
    pass.group(group)?;
    Ok(pass)
}

/// One joint denoise step: four lanes, one group, one fire, the velocity of
/// every video and audio row.
async fn denoise(case: &Case) -> Result<(Vec<f32>, Vec<f32>)> {
    let reading = reading_named("denoise")?;
    if reading.has_kv || reading.takes_tokens {
        return Err(format!(
            "reading `{}` binds a kv space or tokens; a denoise pass binds neither",
            reading.name
        )
        .into());
    }
    let latents = named(&reading, "latents")?.name.clone();
    let context = named(&reading, "context")?.name.clone();
    let audio_context = named(&reading, "audio_context")?.name.clone();
    let timestep = named(&reading, "timestep")?.name.clone();
    let positions = named(&reading, "positions")?;
    let axes = positions.width;
    let positions = positions.name.clone();
    let audio_positions = named(&reading, "audio_positions")?.name.clone();
    let width = reading.readout_width;
    if width != case.channels {
        return Err(format!(
            "the model reads a {width}-wide velocity and the case carries {}-wide latents",
            case.channels
        )
        .into());
    }
    let group = 0;

    // The video lane.
    let video = lane(&reading.name, LaneStream::Video, group)?;
    let xv = Channel::from_shaped([case.rows, case.channels], case.latents.as_slice())
        .named("latents_video");
    let pv = Channel::from_shaped([case.rows, axes], case.positions.as_slice())
        .named("positions_video");
    let tv = Channel::from([case.timestep]).named("t_video");
    video.input(&latents, &xv)?;
    video.input(&positions, &pv)?;
    video.input(&timestep, &tv)?;
    let out_v = Channel::new([case.rows, width], dtype::f32).named("velocity_video");
    let read_v = out_v.clone();
    video.epilogue(move || read_v.put(intrinsics::velocity(width)));

    // The audio lane: the same `latents` port at the same index — the two
    // streams' rows are disjoint — and its own one-axis coordinates.
    let audio = lane(&reading.name, LaneStream::Audio, group)?;
    let xa = Channel::from_shaped(
        [case.audio_rows, case.channels],
        case.audio_latents.as_slice(),
    )
    .named("latents_audio");
    let pa = Channel::from_shaped([case.audio_rows, 1], case.audio_positions.as_slice())
        .named("positions_audio");
    let ta = Channel::from([case.audio_timestep]).named("t_audio");
    audio.input(&latents, &xa)?;
    audio.input(&audio_positions, &pa)?;
    audio.input(&timestep, &ta)?;
    let out_a = Channel::new([case.audio_rows, width], dtype::f32).named("velocity_audio");
    let read_a = out_a.clone();
    audio.epilogue(move || read_a.put(intrinsics::velocity(width)));

    // The two text contexts. Each carries a timestep cell of its own: the
    // prompt modulation is computed where it is applied.
    let ctx = lane(&reading.name, LaneStream::Context, group)?;
    let cv = Channel::from_shaped(
        [case.context_rows, case.context_width],
        case.context.as_slice(),
    )
    .named("context_video");
    let tc = Channel::from([case.timestep]).named("t_context");
    ctx.input(&context, &cv)?;
    ctx.input(&timestep, &tc)?;

    let actx = lane(&reading.name, LaneStream::Reference, group)?;
    let ca = Channel::from_shaped(
        [case.context_rows, case.audio_context_width],
        case.audio_context.as_slice(),
    )
    .named("context_audio");
    let tac = Channel::from([case.audio_timestep]).named("t_audio_context");
    actx.input(&audio_context, &ca)?;
    actx.input(&timestep, &tac)?;

    // One pipeline per lane, submitted back to back.
    let (pv_pipe, pa_pipe) = (Pipeline::new(), Pipeline::new());
    let (pc_pipe, pac_pipe) = (Pipeline::new(), Pipeline::new());
    video.submit(&pv_pipe).context("video lane")?;
    audio.submit(&pa_pipe).context("audio lane")?;
    ctx.submit(&pc_pipe).context("video context lane")?;
    actx.submit(&pac_pipe).context("audio context lane")?;

    let v = out_v.take_host::<Vec<f32>>().await?;
    let a = out_a.take_host::<Vec<f32>>().await?;
    for pipe in [pv_pipe, pa_pipe, pc_pipe, pac_pipe] {
        pipe.close();
    }
    Ok((v, a))
}

/// One connector pass per modality over the same packed trunk rows.
async fn refine(case: &Case) -> Result<((Vec<f32>, u32), (Vec<f32>, u32))> {
    let mut out = Vec::new();
    let mut pipes = Vec::new();
    let mut widths = Vec::new();
    for name in ["refine.video", "refine.audio"] {
        let reading = reading_named(name)?;
        let text = named(&reading, "text")?;
        if text.width != case.text_width {
            return Err(format!(
                "reading `{name}` takes {}-wide packed rows and the case carries {}",
                text.width, case.text_width
            )
            .into());
        }
        let text = text.name.clone();
        let positions = named(&reading, "text_positions")?.name.clone();
        let width = reading.readout_width;
        let pass = ForwardPass::new();
        pass.reading(&reading.name)?;
        pass.stream(LaneStream::Text)?;
        let x = Channel::from_shaped([case.text_rows, case.text_width], case.text.as_slice())
            .named(&format!("text_{name}"));
        let p = Channel::from_shaped([case.text_rows, 1], case.text_positions.as_slice())
            .named(&format!("text_positions_{name}"));
        pass.input(&text, &x)?;
        pass.input(&positions, &p)?;
        let answer = Channel::new([case.text_rows, width], dtype::f32).named(name);
        let readback = answer.clone();
        pass.epilogue(move || readback.put(intrinsics::hidden(width)));
        let pipe = Pipeline::new();
        pass.submit(&pipe).context(name)?;
        out.push(answer);
        pipes.push(pipe);
        widths.push(width);
    }
    let video = out[0].take_host::<Vec<f32>>().await?;
    let audio = out[1].take_host::<Vec<f32>>().await?;
    for pipe in pipes {
        pipe.close();
    }
    Ok(((video, widths[0]), (audio, widths[1])))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err("ltx_2 is an attention-kind pass with no kv bound on any reading".into());
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
        &input.case_8,
        &input.case_9,
        &input.case_10,
        &input.case_11,
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
            return Err("pass `case` (json), `case_0..11` (its pieces) or `case_file`".into());
        }
    };
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;

    let max_rows = model::max_latent_rows();
    if max_rows > 0 && case.rows.max(case.text_rows) > max_rows {
        return Err(format!(
            "{} rows exceed the model's {max_rows} latent rows a pass",
            case.rows.max(case.text_rows)
        )
        .into());
    }

    match case.kind.as_str() {
        "denoise" => {
            let (video, audio) = denoise(&case).await?;
            Ok(Output {
                kind: case.kind.clone(),
                video,
                audio,
                video_rows: case.rows,
                video_width: case.channels,
                audio_rows: case.audio_rows,
                audio_width: case.channels,
            })
        }
        "refine" => {
            let ((video, vw), (audio, aw)) = refine(&case).await?;
            Ok(Output {
                kind: case.kind.clone(),
                video,
                audio,
                video_rows: case.text_rows,
                video_width: vw,
                audio_rows: case.text_rows,
                audio_width: aw,
            })
        }
        other => Err(format!("`kind` is `denoise` or `refine`, not `{other}`").into()),
    }
}
