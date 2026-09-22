//! The pie half of the MiniMax H3 miniature golden. Hands a
//! `minimax-h3-mini` row exactly what `scripts/imagegen/h3_golden.py
//! --mini` handed the vendored `MiniMaxH3DiT` — the raw text hidden rows,
//! the refined text rows, the video / audio / reference latent rows, the
//! `(t, h, w)` coordinate of every row, and the step's four unique
//! timesteps — and reads back the refined caption, the video velocity and
//! the audio velocity, as JSON `scripts/imagegen/h3_parity.py` turns into
//! an `.npz` under the golden's own key names.
//!
//! # TWO PASSES, THEN FOUR LANES IN ONE GROUP
//!
//! `minimax_h3` declares three readings (D1). The first pass here is
//! `refine`: one `Stream::Text` lane binding the encoder's raw
//! `[L, text_dim]` rows on the `caption` port, whose `hidden` readout is
//! `condition_proj` + the token-refiner blocks + `final_norm` — the
//! reference's `refine_prompt_embeds`, run ONCE per request.
//!
//! The second is `denoise`: FOUR lanes of ONE attention group —
//! `Stream::Text` (the refined rows, on the `context` port),
//! `Stream::Video` (`latents`), `Stream::Audio` (`audio`) and
//! `Stream::Reference` (`reference`, the clean keyframe latents) — packed
//! into one row sequence and read by one joint attention. Every lane binds
//! `positions` (its own rows' `(t, h, w)`) and `timestep`, and the
//! timestep cell is the SAME four numbers on every lane: the step's unique
//! timesteps, of which each stream's arm slices the column its class names
//! (video and text slot 0, the condition rows slot 1, audio slot 2). That
//! column choice and the modality's third of the adaLN bank are together
//! the reference's `combined = 3·inverse + tag` gather.
//!
//! **ONE PIPELINE PER LANE.** A group is a fact about a FIRE: the passes
//! join one attention only when they are members of one step. The
//! scheduler seals a frame when every live pipeline has submitted and
//! never seats two passes of one pipeline in one step, so the lanes go
//! down one pipeline each, submitted back to back.
//!
//! **Two readouts of different widths.** The video head is 96 wide and
//! rides `velocity`; the audio head is 32 wide and rides `hidden`, because
//! a plan carries one velocity export (`crates/models/src/minimax_h3/
//! forward.rs` says why). The text and reference lanes read back nothing —
//! the reference discards their rows too.
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

/// The reference's fixed inputs, every rectangle flattened row-major in
/// the golden's own row order.
#[derive(Deserialize)]
struct Case {
    /// `[text_rows, text_dim]` — the encoder's raw hidden rows, what the
    /// `refine` reading takes.
    text_hidden: Vec<f32>,
    text_rows: u32,
    text_dim: u32,
    /// `[text_rows, dim]` — the refined rows the golden computed, what the
    /// `denoise` reading's text lane takes (so a `refine` drift does not
    /// hide inside the denoise diff).
    refined_text: Vec<f32>,
    dim: u32,
    /// `[video_rows, video_features]`.
    video: Vec<f32>,
    video_rows: u32,
    video_features: u32,
    /// `[audio_rows, audio_channels]`.
    audio: Vec<f32>,
    audio_rows: u32,
    audio_channels: u32,
    /// `[reference_rows, video_features]`; may be empty.
    reference: Vec<f32>,
    reference_rows: u32,
    /// `[text_rows + video_rows + audio_rows + reference_rows, 3]`, in the
    /// packed order `[text | video | audio | reference]`.
    positions: Vec<f32>,
    /// The step's unique timesteps, exactly `timestep` port wide.
    timesteps: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    /// `[text_rows, dim]` — the `refine` reading's answer.
    refined: Vec<f32>,
    /// `[video_rows, video_features]` — the video head's velocity.
    velocity: Vec<f32>,
    /// `[audio_rows, audio_channels]` — the audio head's, on the `hidden`
    /// seam.
    audio: Vec<f32>,
    text_rows: u32,
    dim: u32,
    video_rows: u32,
    video_features: u32,
    audio_rows: u32,
    audio_channels: u32,
}

/// The port names a reading declares, read off `model::readings()` rather
/// than typed in, so a renamed port fails here with the model's own
/// vocabulary instead of at the host.
fn named<'a>(reading: &'a model::ReadingFact, name: &str) -> Result<&'a model::PortFact> {
    reading
        .ports
        .iter()
        .find(|port| port.name == name)
        .ok_or_else(|| format!("reading `{}` declares no port `{name}`", reading.name).into())
}

fn reading(name: &str) -> Result<model::ReadingFact> {
    model::readings()
        .into_iter()
        .find(|reading| reading.name == name)
        .ok_or_else(|| format!("this model declares no `{name}` reading").into())
}

/// Rows `[from, to)` of the case's position table, as one channel.
fn positions_of(case: &Case, axes: u32, from: u32, to: u32) -> Channel {
    let (a, f, t) = (axes as usize, from as usize, to as usize);
    Channel::from_shaped([to - from, axes], &case.positions[f * a..t * a])
}

/// One denoise lane: its pass, and (when it reads one back) the channel
/// its answer lands in.
struct Lane {
    pass: ForwardPass,
    pipe: Pipeline,
    out: Option<Channel>,
}

/// The `refine` pass: the caption rows in, the refined rows out.
async fn refine(case: &Case) -> Result<Vec<f32>> {
    let fact = reading("refine")?;
    if fact.readout != model::ReadoutKind::Hidden {
        return Err(format!(
            "reading `{}` reads back {:?}; the refiner answers hidden rows",
            fact.name, fact.readout
        )
        .into());
    }
    let caption = named(&fact, "caption")?;
    if caption.width != case.text_dim {
        return Err(format!(
            "the model's caption port is {} wide and the case carries {}-wide rows",
            caption.width, case.text_dim
        )
        .into());
    }
    let pass = ForwardPass::new();
    pass.reading(&fact.name)?;
    pass.stream(LaneStream::Text)?;
    let rows = Channel::from_shaped([case.text_rows, case.text_dim], case.text_hidden.as_slice())
        .named("caption");
    pass.input(&caption.name, &rows)?;
    let out = Channel::new([case.text_rows, fact.readout_width], dtype::f32).named("refined");
    let readback = out.clone();
    let width = fact.readout_width;
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(width));
    });
    let pipe = Pipeline::new();
    pass.submit(&pipe).context("refine pass")?;
    let refined = out.take_host::<Vec<f32>>().await?;
    pipe.close();
    Ok(refined)
}

/// One `denoise` step: the four lanes of one group, each on its own
/// pipeline, and the two velocities.
async fn denoise(case: &Case) -> Result<(Vec<f32>, Vec<f32>)> {
    let fact = reading("denoise")?;
    if fact.has_kv || fact.takes_tokens {
        return Err(format!(
            "reading `{}` binds a kv space or tokens; a denoise pass binds neither",
            fact.name
        )
        .into());
    }
    let timestep = named(&fact, "timestep")?;
    if case.timesteps.len() != timestep.width as usize {
        return Err(format!(
            "the model's timestep port carries {} unique timesteps and the case has {}",
            timestep.width,
            case.timesteps.len()
        )
        .into());
    }
    let axes = named(&fact, "positions")?.width;
    let positions = named(&fact, "positions")?.name.clone();
    let timestep_name = timestep.name.clone();
    let group = 0;

    // The packed order the lanes take: `[text | video | audio |
    // reference]`, by stream code. The case's position table is in it.
    let mut at = 0u32;
    let mut cut = |rows: u32| {
        let from = at;
        at += rows;
        (from, at)
    };
    let text_span = cut(case.text_rows);
    let video_span = cut(case.video_rows);
    let audio_span = cut(case.audio_rows);
    let reference_span = cut(case.reference_rows);
    if at as usize * axes as usize != case.positions.len() {
        return Err(format!(
            "the case's position table is {} long and its {at} rows want {}",
            case.positions.len(),
            at as usize * axes as usize
        )
        .into());
    }

    let mut lanes: Vec<Lane> = Vec::new();
    let lane = |stream: LaneStream,
                tag: &str,
                port: &str,
                rows: u32,
                width: u32,
                values: &[f32],
                span: (u32, u32),
                readout: Option<(u32, bool)>|
     -> Result<Lane> {
        let pass = ForwardPass::new();
        pass.reading(&fact.name)?;
        pass.stream(stream)?;
        pass.group(group)?;
        let cell = Channel::from_shaped([rows, width], values).named(&format!("rows_{tag}"));
        pass.input(port, &cell)?;
        pass.input(
            &positions,
            &positions_of(case, axes, span.0, span.1).named(&format!("pos_{tag}")),
        )?;
        // One timestep cell PER PASS (a seeded channel attaches to one
        // pass only), carrying the same four numbers: the arm slices the
        // column its stream names.
        let t = Channel::from(case.timesteps.as_slice()).named(&format!("t_{tag}"));
        pass.input(&timestep_name, &t)?;
        let out = readout.map(|(width, velocity)| {
            let out = Channel::new([rows, width], dtype::f32).named(&format!("out_{tag}"));
            let readback = out.clone();
            pass.epilogue(move || {
                readback.put(if velocity {
                    intrinsics::velocity(width)
                } else {
                    intrinsics::hidden(width)
                });
            });
            out
        });
        Ok(Lane {
            pass,
            pipe: Pipeline::new(),
            out,
        })
    };

    lanes.push(lane(
        LaneStream::Text,
        "text",
        &named(&fact, "context")?.name,
        case.text_rows,
        case.dim,
        &case.refined_text,
        text_span,
        None,
    )?);
    lanes.push(lane(
        LaneStream::Video,
        "video",
        &named(&fact, "latents")?.name,
        case.video_rows,
        case.video_features,
        &case.video,
        video_span,
        Some((case.video_features, true)),
    )?);
    lanes.push(lane(
        LaneStream::Audio,
        "audio",
        &named(&fact, "audio")?.name,
        case.audio_rows,
        case.audio_channels,
        &case.audio,
        audio_span,
        Some((case.audio_channels, false)),
    )?);
    if case.reference_rows > 0 {
        lanes.push(lane(
            LaneStream::Reference,
            "reference",
            &named(&fact, "reference")?.name,
            case.reference_rows,
            case.video_features,
            &case.reference,
            reference_span,
            None,
        )?);
    }

    for (i, lane) in lanes.iter().enumerate() {
        lane.pass
            .submit(&lane.pipe)
            .context(&format!("denoise lane {i}"))?;
    }
    let mut answers: Vec<Vec<f32>> = Vec::new();
    for lane in &lanes {
        if let Some(out) = &lane.out {
            answers.push(out.take_host::<Vec<f32>>().await?);
        }
    }
    for lane in lanes {
        lane.pipe.close();
    }
    let mut answers = answers.into_iter();
    let velocity = answers.next().ok_or("the video lane read nothing back")?;
    let audio = answers.next().ok_or("the audio lane read nothing back")?;
    Ok((velocity, audio))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err(
            "minimax_h3 is an attention-kind pass with no kv bound on its denoise reading".into(),
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

    let max_rows = model::max_latent_rows();
    let rows = case.text_rows + case.video_rows + case.audio_rows + case.reference_rows;
    if max_rows > 0 && rows > max_rows {
        return Err(format!("{rows} packed rows exceed the model's {max_rows} a pass").into());
    }
    let refined = refine(&case).await?;
    let (velocity, audio) = denoise(&case).await?;
    Ok(Output {
        refined,
        velocity,
        audio,
        text_rows: case.text_rows,
        dim: case.dim,
        video_rows: case.video_rows,
        video_features: case.video_features,
        audio_rows: case.audio_rows,
        audio_channels: case.audio_channels,
    })
}
