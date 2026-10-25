//! The pie half of the `z-image` goldens. Hands the family exactly what
//! `scripts/imagegen/zimage_golden.py` handed diffusers — the caption rows,
//! the patchified latent, the timestep and the three rotary coordinates of
//! every row — and reads back the refined caption and the velocity, as
//! JSON `scripts/imagegen/zimage_parity.py` turns into an `.npz` under the
//! golden's own key names.
//!
//! Two fires per case, in the order the reference's forward runs them:
//!
//! 1. **`refine`** — one `Context` lane carrying `caption` (`[L32, cap]`),
//!    `pad` (`[L32, 1]`) and `positions` (`[L32, 3]`); its epilogue reads
//!    `hidden(dim)`, the refined caption with its pad rows.
//! 2. **`denoise`** — an `Image` lane (`latents`, `pad`, `positions`,
//!    `timestep`) and a `Context` lane (`context` = the refine readout,
//!    `positions`, `timestep`) in one group of ONE pass — lane 0 the
//!    context, lane 1 the image; the epilogue reads the image lane's
//!    `velocity`.
//!
//! The refine readout goes through the host between the two: a parity
//! harness wants both numbers on disk anyway (the refined caption is the
//! first thing to diff when the velocity disagrees), and a seeded channel
//! attaches to one pass only, so the second fire seeds its own.
//!
//! Under `steps` the second fire becomes the whole TRAJECTORY: `steps`
//! denoise fires over the family's pinned sigmas with the latent carried on
//! the device by the image lane's Euler epilogue, started from the case's
//! own latent (the golden's recorded `randn`, not a device draw).
//!
//! Row counts, pad flags and positions are the CASE's: the family cannot
//! grow a lane, so the harness pads to the 32-row multiple and states the
//! reference's own coordinates (caption row `j` at `(1 + j, 0, 0)`, pads
//! included; image patch `(a, b)` at `(L32 + 1, a, b)`; image pads at
//! `(0, 0, 0)`). The timestep is the family's input convention — the
//! scheduler's `σ·1000`, which the plan flips to the reference's `1000 −
//! t` — so the harness converts the golden's transformer-side `t` first.
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

/// The reference's fixed inputs, flattened row-major and already padded.
#[derive(Deserialize)]
struct Case {
    /// `[image_rows, patch_features]`: the patchified latent, pad rows
    /// appended (their content is irrelevant; the plan overwrites them).
    latents: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// `[image_rows]`: `0.0` real, `1.0` pad.
    image_pad: Vec<f32>,
    /// `[image_rows, 3]`.
    image_positions: Vec<f32>,
    /// `[caption_rows, caption_width]`, pad rows appended.
    caption: Vec<f32>,
    caption_rows: u32,
    caption_width: u32,
    /// `[caption_rows]`.
    caption_pad: Vec<f32>,
    /// `[caption_rows, 3]`.
    caption_positions: Vec<f32>,
    /// The SCHEDULER timestep `σ·1000` the family's `timestep` input takes.
    timestep: f32,
}

#[derive(Serialize)]
struct Output {
    /// `[text_rows, text_width]`: the `text` readout (Qwen3 layer −2) over
    /// the templated prompt; empty without a `prompt`.
    text: Vec<f32>,
    text_rows: u32,
    text_width: u32,
    /// `[caption_rows, dim]`: the refine readout, pads included.
    refined: Vec<f32>,
    caption_rows: u32,
    dim: u32,
    /// `[image_rows, patch_features]`: the velocity, pads included, in the
    /// family's sign (the reference's `-model_out`).
    velocity: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// `[caption_rows, ·]`: the same readout off the context lane, under
    /// `ctx_tap`; empty otherwise.
    ctx: Vec<f32>,
    /// `[image_rows, patch_features]`: the trajectory's latent after its
    /// last fire, under `steps`; empty otherwise. The sigmas it walked and
    /// the timesteps it read, so the report says what schedule ran.
    latent: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

struct Reading {
    name: String,
    axes: u32,
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

/// The width of the input `name` a lane reads.
fn width_of(lane: &model::LaneDecl, name: &str) -> Result<u32> {
    lane.inputs
        .iter()
        .find(|i| i.name == name)
        .map(|i| i.width)
        .ok_or_else(|| format!("the lane reads no `{name}`").into())
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
    let mut axes = None;
    for (stream, binds) in lanes {
        let lane = lane_reading(&name, stream, binds)?;
        if lane.reads_kv {
            return Err(format!(
                "reading `{name}` binds a kv space; the parity pass binds none"
            ));
        }
        axes.get_or_insert(width_of(&lane, "positions")?);
    }
    Ok(Reading {
        name,
        axes: axes.ok_or("no lane to read")?,
        readout_width,
    })
}

/// The `refine` reading: one context lane, the refined caption back.
async fn refine(case: &Case, pipe: &Pipeline) -> Result<(Vec<f32>, u32)> {
    let r = reading(
        model::ReadoutKind::Hidden,
        &[("context", &["caption", "pad", "positions"])],
    )?;
    let rows = case.caption_rows;
    let pass = ForwardPass::new();
    pass.switch(0, "reading", &r.name)?;
    pass.switch(0, "stream", "context")?;
    let caption =
        Channel::from_shaped([rows, case.caption_width], case.caption.as_slice()).named("caption");
    let pad = Channel::from_shaped([rows, 1], case.caption_pad.as_slice()).named("cap_pad");
    let positions =
        Channel::from_shaped([rows, r.axes], case.caption_positions.as_slice()).named("cap_pos");
    pass.input(0, "caption", &caption)?;
    pass.input(0, "pad", &pad)?;
    pass.input(0, "positions", &positions)?;
    let width = r.readout_width;
    let out = Channel::new([rows, width], dtype::f32).named("refined");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(0, rows, width));
    });
    pass.submit(pipe).context("refine lane")?;
    let rows_out: Vec<f32> = out.take_host().await?;
    Ok((rows_out, width))
}

/// Binds the `denoise` reading's lanes on `pass`: lane 0 the refined
/// caption (pads included, no `pad` input), lane 1 the image over `latent`,
/// both in one group and reading `timestep`.
#[allow(clippy::too_many_arguments)]
fn denoise_lanes(
    pass: &ForwardPass,
    r: &Reading,
    case: &Case,
    refined: &[f32],
    dim: u32,
    latent: &Channel,
    timestep: &Channel,
    tag: &str,
) -> Result<()> {
    let group = 0;
    pass.switch(0, "reading", &r.name)?;
    pass.switch(0, "stream", "context")?;
    pass.group(0, group)?;
    let ctx =
        Channel::from_shaped([case.caption_rows, dim], refined).named(&format!("{tag}context"));
    let ctx_pos = Channel::from_shaped(
        [case.caption_rows, r.axes],
        case.caption_positions.as_slice(),
    )
    .named(&format!("{tag}ctx_pos"));
    pass.input(0, "context", &ctx)?;
    pass.input(0, "positions", &ctx_pos)?;
    pass.input(0, "timestep", timestep)?;

    let rows = case.image_rows;
    pass.switch(1, "reading", &r.name)?;
    pass.switch(1, "stream", "image")?;
    pass.group(1, group)?;
    let pad =
        Channel::from_shaped([rows, 1], case.image_pad.as_slice()).named(&format!("{tag}img_pad"));
    let img_pos = Channel::from_shaped([rows, r.axes], case.image_positions.as_slice())
        .named(&format!("{tag}img_pos"));
    pass.input(1, "latents", latent)?;
    pass.input(1, "pad", &pad)?;
    pass.input(1, "positions", &img_pos)?;
    pass.input(1, "timestep", timestep)?;
    Ok(())
}

/// The `denoise` reading: the refined-caption lane and the image lane in
/// one group of one pass, the velocity back off the image lane.
async fn denoise(
    case: &Case,
    refined: &[f32],
    dim: u32,
    ctx_tap: bool,
    pipe: &Pipeline,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let r = reading(
        model::ReadoutKind::Velocity,
        &[
            ("context", &["context", "positions", "timestep"]),
            ("image", &["latents", "pad", "positions", "timestep"]),
        ],
    )?;
    // The readout is the velocity, `patch_features` wide — unless the family
    // is tapped on an intermediate (a bisect), which is `readout_width` wide.
    let width = r.readout_width;
    let rows = case.image_rows;
    let caption_rows = case.caption_rows;
    let t = Channel::from([case.timestep]).named("t");
    let x =
        Channel::from_shaped([rows, case.patch_features], case.latents.as_slice()).named("latents");
    let pass = ForwardPass::new();
    denoise_lanes(&pass, &r, case, refined, dim, &x, &t, "")?;

    let out = Channel::new([rows, width], dtype::f32).named("velocity");
    let ctx_out = ctx_tap.then(|| Channel::new([caption_rows, width], dtype::f32).named("ctx_tap"));
    let (readback, tapped) = (out.clone(), ctx_out.clone());
    pass.epilogue(move || {
        readback.put(intrinsics::velocity(1, rows, width));
        if let Some(tapped) = &tapped {
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
/// lane's epilogue (`x <- x + (sigma' - sigma) . velocity`, design D4) and
/// read back once per fire so a partial run can be diffed against the
/// reference's own `sched.x{n}`.
///
/// The initial latent is the CASE's, not a device draw: this is a parity
/// harness, and the reference's trajectory starts from a `randn` the golden
/// recorded. `DenoiseLoop` puts a seed fire ahead of the steps whose `dt`
/// is 0, so fire 0 leaves that latent alone and fire `k` integrates step
/// `k - 1`.
async fn trajectory(
    case: &Case,
    refined: &[f32],
    dim: u32,
    steps: u32,
    stop: Option<u32>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let r = reading(
        model::ReadoutKind::Velocity,
        &[
            ("context", &["context", "positions", "timestep"]),
            ("image", &["latents", "pad", "positions", "timestep"]),
        ],
    )?;
    let width = r.readout_width;
    let rows = case.image_rows;
    let sched = FlowMatchEuler::from_model(steps, None)?;
    let loops = DenoiseLoop::new(&sched);
    let latent =
        Channel::from_shaped([rows, case.patch_features], case.latents.as_slice()).named("traj_x");
    let pass = ForwardPass::new();
    denoise_lanes(
        &pass,
        &r,
        case,
        refined,
        dim,
        &latent,
        &loops.timestep(),
        "traj_",
    )?;

    let out = Channel::new([rows, width], dtype::f32).named("traj_out");
    let dts = loops.dts();
    let x = latent.clone();
    let readback = out.clone();
    loops.drive(&pass, move |k| {
        let next = euler_step(
            &x.take(),
            &intrinsics::velocity(1, rows, width),
            &at(&dts.read(), k),
        );
        x.put(&next);
        readback.put(&next);
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
            image_rows: 0,
            patch_features: 0,
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
        case.image_rows * case.patch_features,
    )?;
    check("image_pad", case.image_pad.len(), case.image_rows)?;
    check(
        "image_positions",
        case.image_positions.len(),
        case.image_rows * 3,
    )?;
    check(
        "caption",
        case.caption.len(),
        case.caption_rows * case.caption_width,
    )?;
    check("caption_pad", case.caption_pad.len(), case.caption_rows)?;
    check(
        "caption_positions",
        case.caption_positions.len(),
        case.caption_rows * 3,
    )?;
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && case.image_rows + case.caption_rows > max_rows {
        return Err(format!(
            "{} rows exceed the model's {max_rows} rows a pass",
            case.image_rows + case.caption_rows
        )
        .into());
    }

    let pipe = Pipeline::new();
    // The chained check: the encoder's own rows stand in for the golden's
    // caption (the case's pads and positions must already fit them).
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
            let real = case.caption_pad.iter().filter(|flag| **flag == 0.0).count();
            if real != len as usize {
                return Err(format!(
                    "the prompt renders to {len} tokens and the case flags {real} real caption rows"
                )
                .into());
            }
            case.caption[..rows.len()].copy_from_slice(&rows);
            (rows, len, width)
        }
        None => (Vec::new(), 0, 0),
    };
    let (refined, dim) = refine(&case, &pipe).await?;
    let (mut latent, mut sigmas, mut timesteps) = (Vec::new(), Vec::new(), Vec::new());
    let (velocity, ctx_rows) = if input.refine_only {
        (Vec::new(), Vec::new())
    } else if input.steps > 0 {
        let (rows, s, t) = trajectory(&case, &refined, dim, input.steps, input.stop).await?;
        latent = rows;
        sigmas = s;
        timesteps = t;
        (Vec::new(), Vec::new())
    } else {
        denoise(&case, &refined, dim, input.ctx_tap, &pipe).await?
    };
    // The readout's width: `patch_features` for the velocity, the tapped
    // intermediate's when the family is being bisected.
    let patch_features = if velocity.is_empty() {
        case.patch_features
    } else {
        u32::try_from(velocity.len() / case.image_rows.max(1) as usize).unwrap_or(0)
    };
    pipe.close();
    Ok(Output {
        text: text_rows,
        text_rows: text_len,
        text_width,
        refined,
        caption_rows: case.caption_rows,
        dim,
        velocity,
        image_rows: case.image_rows,
        patch_features,
        ctx: ctx_rows,
        latent,
        sigmas,
        timesteps,
    })
}
