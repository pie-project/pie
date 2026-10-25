//! The pie half of the `z-image` goldens. Hands the family exactly what
//! `scripts/imagegen/zimage_golden.py` handed diffusers — the caption rows,
//! the latent (the VAE's own `[h, w, C]`, on the voxel axis) and the
//! timestep — and reads back the refined caption and the velocity, as JSON
//! `scripts/imagegen/zimage_parity.py` turns into an `.npz` under the
//! golden's own key names. The family patchifies and computes every row's
//! rotary position in its graph; the guest states neither, and nothing is
//! padded: a lane carries its own rows.
//!
//! Two fires per case, in the order the reference's forward runs them, down
//! one pipeline:
//!
//! 1. **`refine`** — one `context` lane carrying `caption` (`[L, cap]`); its
//!    epilogue reads `hidden(dim)`, the refined caption.
//! 2. **`denoise`** — a `context` lane (`context` = the refine readout,
//!    `timestep`) and an `image` lane (`latents`, `timestep`) in one group of
//!    ONE pass — lane 0 the context, lane 1 the image; the epilogue reads
//!    the image lane's `velocity` on the voxel axis.
//!
//! The refine readout goes through the host between the two: a parity
//! harness wants both numbers on disk anyway (the refined caption is the
//! first thing to diff when the velocity disagrees).
//!
//! Under `steps` the second fire becomes the whole TRAJECTORY: `steps`
//! denoise fires over the family's pinned sigmas with the latent carried on
//! the device by the image lane's Euler epilogue, started from the case's
//! own latent (the golden's recorded `randn`, not a device draw).
//!
//! The timestep is the family's input convention — the scheduler's
//! `σ·1000`, which the plan flips to the reference's `1000 − t` — so the
//! harness converts the golden's transformer-side `t` first.
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    case: Option<String>,
    #[serde(default)]
    case_file: Option<String>,
    /// `case_0`, `case_1`, …: the case JSON in argv-sized pieces (the
    /// Turbo case is ~3 MB; a 128 KiB argument ceiling makes that ~30).
    #[serde(flatten)]
    pieces: std::collections::BTreeMap<String, inferlet::serde_json::Value>,
    #[serde(default)]
    refine_only: bool,
    /// Run the `text` reading on this prompt first (the flagship row only):
    /// the encoder's rows come back as `text`, and — unless `text_only` —
    /// replace the case's caption rows before `refine`.
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    text_only: bool,
    /// Read the denoise readout off the CONTEXT lane too (a bisect: with
    /// the family tapped on a joint-trunk value, this is its caption rows).
    #[serde(default)]
    ctx_tap: bool,
    /// Run the whole Euler trajectory instead of one step: the case's
    /// `latents` are then the INITIAL noise, and this is the schedule's
    /// step count (the family pins eight for Turbo). 0 keeps one step.
    #[serde(default)]
    steps: u32,
    /// Stop the trajectory after this many steps (a bisect against the
    /// golden's `sched.x{n}`); all of them by default.
    #[serde(default)]
    stop: Option<u32>,
}

/// The reference's fixed inputs, flattened row-major.
#[derive(Deserialize)]
struct Case {
    /// `[height, width, channels]`: the VAE latent.
    latents: Vec<f32>,
    height: u32,
    width: u32,
    channels: u32,
    /// `[caption_rows, caption_width]`.
    caption: Vec<f32>,
    caption_rows: u32,
    caption_width: u32,
    /// The SCHEDULER timestep `σ·1000` the family's `timestep` input takes.
    timestep: f32,
    /// The token rows a tapped deployment's intermediate reads out on; the
    /// velocity is read on the latent's voxels.
    #[serde(default)]
    readout_rows: Option<u32>,
}

#[derive(Serialize)]
struct Output {
    /// `[text_rows, text_width]`: the `text` readout (Qwen3 layer −2) over
    /// the templated prompt; empty without a `prompt`.
    text: Vec<f32>,
    text_rows: u32,
    text_width: u32,
    /// `[caption_rows, dim]`: the refine readout.
    refined: Vec<f32>,
    caption_rows: u32,
    dim: u32,
    /// `[height · width, readout]`: the velocity on the voxel axis, in the
    /// family's sign (the reference's `-model_out`); a tapped deployment's
    /// intermediate is `[readout_rows, readout]` instead.
    velocity: Vec<f32>,
    height: u32,
    width: u32,
    readout: u32,
    /// `[caption_rows, ·]`: the same readout off the context lane, under
    /// `ctx_tap`; empty otherwise.
    ctx: Vec<f32>,
    /// `[height, width, channels]`: the trajectory's latent after its last
    /// fire, under `steps`; empty otherwise. The sigmas it walked and the
    /// timesteps it read, so the report says what schedule ran.
    latent: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

struct Reading {
    name: String,
    readout_width: u32,
}

/// The lane `stream` of `reading`, checked to read exactly `binds`: what
/// this guest binds on that lane, by the model's own input names. Checked
/// up front so a model whose lane reads anything else fails here in its own
/// vocabulary instead of at `program`.
fn lane_reading(reading: &str, stream: &str, binds: &[&str]) -> Result<model::LaneDecl> {
    let lane = model::lane_of(&[("reading", reading), ("stream", stream)])?;
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

/// The token-less reading with `readout`, each of its `lanes` checked to
/// read exactly the inputs this guest binds there.
fn reading(readout: model::ReadoutKind, lanes: &[(&str, &[&str])]) -> Result<Reading> {
    let readings = model::switch("reading")
        .map(|decl| decl.values)
        .unwrap_or_default();
    let (name, readout_width) = readings
        .into_iter()
        .find_map(|name| {
            let decls = lanes
                .iter()
                .map(|(stream, _)| {
                    model::lane_of(&[("reading", name.as_str()), ("stream", stream)])
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok()?;
            if decls.iter().any(|lane| lane.embeds) {
                return None;
            }
            let width = decls.iter().find_map(|lane| {
                lane.readout
                    .as_ref()
                    .filter(|r| r.kind == readout)
                    .map(|r| r.width)
            })?;
            Some((name, width))
        })
        .ok_or_else(|| {
            format!("this model declares no token-less reading with a {readout:?} readout")
        })?;
    for (stream, binds) in lanes {
        let lane = lane_reading(&name, stream, binds)?;
        if lane.reads_kv {
            return Err(format!(
                "reading `{name}` binds a kv space; the parity pass binds none"
            ));
        }
    }
    Ok(Reading {
        name,
        readout_width,
    })
}

/// The `refine` reading: one context lane, the refined caption back.
async fn refine(case: &Case, pipe: &Pipeline) -> Result<(Vec<f32>, u32)> {
    let r = reading(model::ReadoutKind::Hidden, &[("context", &["caption"])])?;
    let rows = case.caption_rows;
    let pass = ForwardPass::new();
    pass.switch(0, "reading", &r.name)?;
    pass.switch(0, "stream", "context")?;
    let caption =
        Channel::from_shaped([rows, case.caption_width], case.caption.as_slice()).named("caption");
    pass.input(0, "caption", &caption)?;
    let width = r.readout_width;
    let out = Channel::new([rows, width], dtype::f32).named("refined");
    pass.epilogue(move || {
        out.put(intrinsics::hidden(0, rows, width));
    });
    pass.submit(pipe).context("refine lane")?;
    let rows_out: Vec<f32> = out.take_host().await?;
    Ok((rows_out, width))
}

/// The `denoise` reading, its two lanes checked against what this guest
/// binds.
fn denoise_reading() -> Result<Reading> {
    reading(
        model::ReadoutKind::Velocity,
        &[
            ("context", &["context", "timestep"]),
            ("image", &["latents", "timestep"]),
        ],
    )
}

/// Binds the `denoise` reading's lanes on `pass`: lane 0 the refined
/// caption, lane 1 the image over `latent`, both in one group and reading
/// `timestep`.
fn denoise_lanes(
    pass: &ForwardPass,
    r: &Reading,
    context: &Channel,
    latent: &Channel,
    timestep: &Channel,
) -> Result<()> {
    for (lane, stream) in [(0, "context"), (1, "image")] {
        pass.switch(lane, "reading", &r.name)?;
        pass.switch(lane, "stream", stream)?;
        pass.group(lane, 0)?;
        pass.input(lane, "timestep", timestep)?;
    }
    pass.input(0, "context", context)?;
    pass.input(1, "latents", latent)?;
    Ok(())
}

/// The rows the denoise readout carries: the latent's voxels for a
/// `channels`-wide velocity, the case's `readout_rows` for a tap.
fn readout_rows(case: &Case, width: u32) -> Result<u32> {
    if width == case.channels {
        return Ok(case.height * case.width);
    }
    case.readout_rows.ok_or_else(|| {
        format!(
            "the denoise reading reads out {width} wide, not the latent's {} channels: a tap, \
             whose rows the case must state (`readout_rows`)",
            case.channels
        )
        .into()
    })
}

fn latent_of(case: &Case, name: &str) -> Channel {
    Channel::from_shaped(
        [case.height, case.width, case.channels],
        case.latents.as_slice(),
    )
    .named(name)
}

/// One denoise step: the refined-caption lane and the image lane in one
/// group of one pass, the velocity back off the image lane.
async fn denoise(
    case: &Case,
    context: &Channel,
    ctx_tap: bool,
    pipe: &Pipeline,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let r = denoise_reading()?;
    // The readout is the velocity, `channels` wide — unless the family is
    // tapped on an intermediate (a bisect), which is `readout_width` wide
    // over the token rows.
    let width = r.readout_width;
    let voxels = readout_rows(case, width)?;
    let caption_rows = case.caption_rows;
    let t = Channel::from([case.timestep]).named("t");
    let x = latent_of(case, "latents");
    let pass = ForwardPass::new();
    denoise_lanes(&pass, &r, context, &x, &t)?;

    let out = Channel::new([voxels, width], dtype::f32).named("velocity");
    let ctx_out = ctx_tap.then(|| Channel::new([caption_rows, width], dtype::f32).named("ctx_tap"));
    pass.epilogue(move || {
        out.put(intrinsics::velocity(1, voxels, width));
        if let Some(tapped) = &ctx_out {
            tapped.put(intrinsics::velocity(0, caption_rows, width));
        }
    });

    pass.submit(pipe).context("the denoise step")?;
    let velocity: Vec<f32> = out.take_host().await?;
    let ctx_rows = match ctx_out {
        Some(out) => out.take_host::<Vec<f32>>().await?,
        None => Vec::new(),
    };
    Ok((velocity, ctx_rows))
}

/// The whole Euler trajectory in one job: `steps` denoise fires over the
/// family's own pinned sigmas ([`FlowMatchEuler::from_model`] — Turbo's
/// eight), the latent carried on the DEVICE between them by the image
/// lane's epilogue (`x <- x + (sigma' - sigma) . velocity`) and read back
/// once per fire so a partial run can be diffed against the reference's own
/// `sched.x{n}`.
///
/// The initial latent is the CASE's, not a device draw: this is a parity
/// harness, and the reference's trajectory starts from a `randn` the golden
/// recorded. `DenoiseLoop` puts a seed fire ahead of the steps whose `dt`
/// is 0, so fire 0 leaves that latent alone and fire `k` integrates step
/// `k - 1`.
async fn trajectory(
    case: &Case,
    context: &Channel,
    steps: u32,
    stop: Option<u32>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let r = denoise_reading()?;
    let width = r.readout_width;
    let voxels = case.height * case.width;
    let lat = [case.height, case.width, case.channels];
    let sched = FlowMatchEuler::from_model(steps, None)?;
    let loops = DenoiseLoop::new(&sched);
    let latent = latent_of(case, "traj_x");
    let pass = ForwardPass::new();
    denoise_lanes(&pass, &r, context, &latent, &loops.timestep())?;

    let out = Channel::new(lat, dtype::f32).named("traj_out");
    let dts = loops.dts();
    loops.drive(&pass, move |k| {
        let next = euler_step(
            &reshape(latent.take(), [voxels, lat[2]]),
            &intrinsics::velocity(1, voxels, width),
            &at(&dts.read(), k),
        );
        let next = reshape(&next, lat);
        latent.put(&next);
        out.put(&next);
    });

    let fires = stop.map_or(loops.fires(), |n| n.min(sched.steps()) + 1);
    let mut last = Vec::new();
    for fire in 0..fires {
        loops
            .fire(&pass)
            .map_err(|why| format!("fire {fire}: {why}"))?;
        last = out.take_host::<Vec<f32>>().await?;
    }
    loops.close();
    Ok((last, sched.sigmas.clone(), sched.timesteps()))
}

/// The `text` reading: the prompt through the family's template
/// (`user(prompt) ++ cue()`, what the reference's `apply_chat_template(
/// add_generation_prompt=True)` renders), one prefill over a fresh working
/// set, `hidden` at every row. `inferlet::latent::encode_text` with the
/// template applied — that helper tokenizes plain text.
async fn text(prompt: &str, pipe: &Pipeline) -> Result<(Vec<f32>, u32, u32)> {
    let readings = model::switch("reading")
        .map(|decl| decl.values)
        .unwrap_or_default();
    let (name, lane) = readings
        .into_iter()
        .find_map(|name| {
            let lane = model::lane_of(&[("reading", name.as_str())]).ok()?;
            let hidden = lane.readout.as_ref().map(|r| r.kind) == Some(model::ReadoutKind::Hidden);
            (lane.embeds && hidden).then_some((name, lane))
        })
        .ok_or("this row declares no token reading with a hidden readout (the miniature has no encoder)")?;
    if !lane.reads_kv {
        return Err(format!(
            "reading `{name}` takes tokens but binds no kv space"
        ));
    }
    let width = lane.readout.map_or(0, |r| r.width);
    let mut ids = inferlet::chat::prefix();
    ids.extend(inferlet::chat::user(prompt));
    ids.extend(inferlet::chat::cue());
    let ids: Vec<i32> = ids
        .into_iter()
        .map(|id| i32::try_from(id).unwrap_or(0))
        .collect();
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    if len == 0 {
        return Err("the prompt tokenizes to nothing".into());
    }
    let ws = WorkingSet::new();

    let toks = Channel::from(ids);
    let embed_indptr = Channel::from([0u32, len]);
    let readout = Channel::from_iter(0..len);
    let out = Channel::new([len, width], dtype::f32).named("text");

    let pass = ForwardPass::new();
    pass.switch(0, "reading", &name)?;
    pass.embed(&toks, &embed_indptr)?;
    pass.readout(&readout)?;
    // The encoder's state is scratch: nothing folds, and it goes with `ws`.
    pass.state(&[&ws], StateGeometry::default())?;
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(0, len, width));
    });
    pass.submit(pipe).context("text lane")?;
    let rows: Vec<f32> = out.take_host().await?;
    Ok((rows, len, width))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.text_only {
        let prompt = input
            .prompt
            .as_deref()
            .ok_or("`text_only` needs a `prompt`")?;
        let pipe = Pipeline::new();
        let (rows, text_rows, text_width) = text(prompt, &pipe).await?;
        pipe.close();
        return Ok(Output {
            text: rows,
            text_rows,
            text_width,
            refined: Vec::new(),
            caption_rows: 0,
            dim: 0,
            velocity: Vec::new(),
            height: 0,
            width: 0,
            readout: 0,
            ctx: Vec::new(),
            latent: Vec::new(),
            sigmas: Vec::new(),
            timesteps: Vec::new(),
        });
    }
    let mut pieces = String::new();
    for i in 0.. {
        match input.pieces.get(&format!("case_{i}")) {
            Some(inferlet::serde_json::Value::String(piece)) => pieces.push_str(piece),
            Some(other) => return Err(format!("`case_{i}` is not a string: {other}").into()),
            None => break,
        }
    }
    let json = match (&input.case, &input.case_file) {
        (Some(text), _) => text.clone(),
        (None, Some(name)) => std::fs::read_to_string(format!("/scratch/{name}"))
            .map_err(|why| format!("reading /scratch/{name}: {why}"))?,
        (None, None) if !pieces.is_empty() => pieces,
        (None, None) => {
            return Err("pass `case` (json), `case_0..N` (its pieces) or `case_file`".into());
        }
    };
    let case: Case =
        inferlet::serde_json::from_str(&json).map_err(|why| format!("case json: {why}"))?;
    let check = |what: &str, have: usize, want: u32| -> Result<()> {
        if have != want as usize {
            return Err(
                format!("`{what}` carries {have} numbers, the case's shape wants {want}").into(),
            );
        }
        Ok(())
    };
    check(
        "latents",
        case.latents.len(),
        case.height * case.width * case.channels,
    )?;
    check(
        "caption",
        case.caption.len(),
        case.caption_rows * case.caption_width,
    )?;

    let pipe = Pipeline::new();
    // The chained check: the encoder's own rows stand in for the golden's
    // caption.
    let mut case = case;
    let (text_rows, text_len, text_width) = match &input.prompt {
        Some(prompt) => {
            let (rows, len, width) = text(prompt, &pipe).await?;
            if width != case.caption_width {
                return Err(format!(
                    "the encoder writes {width}-wide rows and the case's caption is {} wide",
                    case.caption_width
                )
                .into());
            }
            if len != case.caption_rows {
                return Err(format!(
                    "the prompt renders to {len} tokens and the case's caption is {} rows",
                    case.caption_rows
                )
                .into());
            }
            case.caption.copy_from_slice(&rows);
            (rows, len, width)
        }
        None => (Vec::new(), 0, 0),
    };
    let (refined, dim) = refine(&case, &pipe).await?;
    // The refined caption: no pass writes it, so every denoise pass reads it.
    let context =
        Channel::from_shaped([case.caption_rows, dim], refined.as_slice()).named("context");
    let (mut latent, mut sigmas, mut timesteps) = (Vec::new(), Vec::new(), Vec::new());
    let (velocity, ctx_rows) = if input.refine_only {
        (Vec::new(), Vec::new())
    } else if input.steps > 0 {
        let (rows, s, t) = trajectory(&case, &context, input.steps, input.stop).await?;
        latent = rows;
        sigmas = s;
        timesteps = t;
        (Vec::new(), Vec::new())
    } else {
        denoise(&case, &context, input.ctx_tap, &pipe).await?
    };
    let voxels = velocity.len() / (case.height * case.width).max(1) as usize;
    let readout = if voxels == case.channels as usize {
        case.channels
    } else {
        case.readout_rows
            .and_then(|rows| u32::try_from(velocity.len() / rows.max(1) as usize).ok())
            .unwrap_or(0)
    };
    pipe.close();
    Ok(Output {
        text: text_rows,
        text_rows: text_len,
        text_width,
        refined,
        caption_rows: case.caption_rows,
        dim,
        readout,
        velocity,
        height: case.height,
        width: case.width,
        ctx: ctx_rows,
        latent,
        sigmas,
        timesteps,
    })
}
