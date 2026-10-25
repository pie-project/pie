//! The pie half of the `mini-dit` golden. Hands the model exactly what
//! `scripts/imagegen/mini_dit_ref.py` handed PyTorch — the patch rows, the
//! caption rows, the cross-attention context, the timestep and the three
//! rotary coordinates per row — and reads back the velocity the head
//! predicts, as JSON `scripts/imagegen/mini_dit_parity.py` turns into an
//! `.npz` under the golden's own key names.
//!
//! Two parity modes: one step against `mini_dit_dump_bf16.npz`'s `velocity`,
//! and the four-step Euler schedule against `mini_dit_euler_bf16.npz`'s
//! `euler.v{0..3}` / `euler.x{1..4}`.
//!
//! And two modes that need no golden because every claim they make is
//! WITHIN one fire's own answers: `--cfg` (classifier-free guidance on the
//! device, six lanes and two groups in one pass — `prototype.md` §1.1) and
//! `--cache` (step caching on the device, the whole Euler schedule as fires
//! with the cache decision carried in a channel from one fire to the next —
//! `.wiki/imagegen/conditional-fire.md`). Their harnesses are
//! `mini_dit_parity.py guidance` and `mini_dit_cache.py`.
//!
//! # THREE LANES, ONE GROUP, ONE PASS
//!
//! `mini-dit`'s `denoise` reading declares three streams (D2), so one step
//! is one pass of three lanes — caption, context, image — each stating its
//! `stream` switch and all in one attention group, which is what lets the
//! caption rows join the image rows' attention and the context rows serve
//! as block 2's keys. Neither `state` nor `embed` is called: a denoise
//! reading declares no kv space and no tokens, and the image lane's row
//! count comes from its latents channel. Group `g` of a pass is lanes `3g`
//! (caption), `3g + 1` (context) and `3g + 2` (image).
//!
//! Only the image lane reads out — the caption stream ends after block 1,
//! and the head is image-only — so the epilogue reads its velocity.
//!
//! # WHY THE EULER LOOP IS ON THE HOST
//!
//! `latent-probe` integrates on the device, which is what a real sampler
//! does (D4) and what a real guest should copy. A parity harness wants the
//! opposite: every step's velocity AND every step's latent, at the
//! reference's own numbers, with nothing folded together. So this program
//! takes the velocity after each fire and steps the latent itself. The
//! arithmetic is `FlowMatchEuler`'s and the sigmas are the model's own
//! (`model::schedule()`), so what is checked is still the schedule the
//! family declares.
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
    euler: bool,
    /// The artifact serves a tapped deployment (`-tap.{key}`): the readout
    /// is an intermediate, at whatever width the model declares.
    #[serde(default)]
    tap: bool,
    /// Under `tap`, a tap whose rectangle spans the caption rows too: read
    /// the caption lane out as well, into `text_tap`.
    #[serde(default)]
    tap_text: bool,
    /// Classifier-free guidance, ON THE DEVICE. Runs the step as SIX lanes
    /// of one pass — two attention groups of three, group 0 holding the
    /// case's context and group 1 holding a zeroed one — and combines the
    /// two branches' velocities inside the pass's epilogue with
    /// `cfg_combine`. Reports the combine at `cfg_scale`
    /// alongside each branch's own answer, so the caller can check the two
    /// identities that need no golden: at `s = 1` the combine IS the
    /// conditional branch, at `s = 0` it IS the unconditional one.
    #[serde(default)]
    cfg: bool,
    /// The guidance scale the `cfg` combine runs at. Default 1.0, the
    /// identity on the conditional branch.
    #[serde(default)]
    cfg_scale: Option<f32>,
    /// STEP CACHING, ON THE DEVICE (`.wiki/imagegen/conditional-fire.md`).
    /// Runs the whole Euler loop on the device — one fire per step, nothing
    /// per step across the host — with the cache decision and the reuse in
    /// the image lane's epilogue. `cache_threshold` absent is the PLAIN
    /// loop (no cache ops in the trace at all), which is the trajectory
    /// every cached run is compared against.
    #[serde(default)]
    cache: bool,
    /// The relative-change threshold a step must fall under to be skipped.
    /// Absent means "no cache path" — the baseline, not "threshold 0".
    #[serde(default)]
    cache_threshold: Option<f32>,
    /// How many Euler steps (fires are `steps + 1`; fire 0 seeds).
    #[serde(default)]
    cache_steps: Option<u32>,
    /// The keyed-RNG seed the device draw starts from, so the baseline and
    /// every cached run start from the same latent.
    #[serde(default)]
    cache_seed: Option<u32>,
    /// `dpm2m`: run the schedule as a DPM-Solver++ 2M device loop from the
    /// case's latent, answering `euler_v` / `euler_x` per step.
    #[serde(default)]
    solver: Option<String>,
    /// `--classes none|all|split|split-swap`: one step under an attention
    /// mask (`forward-pass.mask`) spelled from two classes. `all` states two
    /// classes that both see everything; `split` halves the image grid by
    /// column and the text rows by index into classes 0 and 1 that see only
    /// themselves; `split-swap` is `split` with the class-1 text rows
    /// negated. The context rows attend and are attended by every row.
    #[serde(default)]
    classes: Option<String>,
    /// `--sweep N`: N independent attention groups (N requests' worth of
    /// lanes) in ONE pass. Group 1 repeats group 0's inputs; group g >= 2
    /// scales the latent by `1 + g/4`.
    #[serde(default)]
    sweep: Option<u32>,
}

struct ClassPlan {
    text: Vec<i32>,
    image: Vec<i32>,
    table: Vec<u8>,
    count: u32,
    swap_text: bool,
}

impl ClassPlan {
    fn of(mode: &str, case: &Case) -> Result<Option<ClassPlan>> {
        let rows = case.image_rows as usize;
        let side = (rows as f64).sqrt() as usize;
        if side * side != rows {
            return Err(format!("{rows} image rows are not a square grid"));
        }
        let half_text = case.text_rows as usize / 2;
        let split_text: Vec<i32> = (0..case.text_rows as usize)
            .map(|i| i32::from(i >= half_text))
            .collect();
        let split_image: Vec<i32> = (0..rows).map(|i| i32::from(i % side >= side / 2)).collect();
        Ok(match mode {
            "none" => None,
            "all" => Some(ClassPlan {
                text: split_text,
                image: split_image,
                table: vec![1, 1, 1, 1],
                count: 2,
                swap_text: false,
            }),
            "split" | "split-swap" => Some(ClassPlan {
                text: split_text,
                image: split_image,
                table: vec![1, 0, 0, 1],
                count: 2,
                swap_text: mode == "split-swap",
            }),
            other => return Err(format!("unknown classes mode `{other}`")),
        })
    }
}

impl ClassPlan {
    /// The plan as the pass's mask over its one group, lane by lane
    /// (caption, context, image): a row attends a row where the table lets
    /// their classes, and a context row (no class) attends and is attended
    /// by every row.
    fn mask(&self, context_rows: u32) -> (u32, Vec<bool>) {
        let classes: Vec<Option<i32>> = self
            .text
            .iter()
            .map(|&c| Some(c))
            .chain((0..context_rows).map(|_| None))
            .chain(self.image.iter().map(|&c| Some(c)))
            .collect();
        let n = classes.len();
        let count = self.count as usize;
        let mut cells = Vec::with_capacity(n * n);
        for q in &classes {
            for k in &classes {
                cells.push(match (q, k) {
                    (Some(q), Some(k)) => self.table[*q as usize * count + *k as usize] != 0,
                    _ => true,
                });
            }
        }
        (n as u32, cells)
    }
}

/// One batch element of the reference's fixed inputs, flattened row-major.
/// The harness writes the same numbers `mini_dit_ref.py` fed torch: the
/// patchified latent (never the `[C, H, W]` array), the caption rows, the
/// context rows, one timestep, and the `(t, h, w)` coordinates of every row.
#[derive(Deserialize)]
struct Case {
    /// `[image_rows, patch_features]`, row-major. The reference's `patches`.
    latents: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// `[text_rows, text_width]`.
    text: Vec<f32>,
    text_rows: u32,
    text_width: u32,
    /// `[context_rows, context_width]`.
    context: Vec<f32>,
    context_rows: u32,
    context_width: u32,
    /// `[rows, 3]` each, in the packed order the plan reads (the `text`
    /// lane before the `image` lane).
    text_positions: Vec<f32>,
    image_positions: Vec<f32>,
    /// One step's timestep, or — under `--euler` — the schedule.
    timestep: f32,
    #[serde(default)]
    sigmas: Vec<f32>,
    #[serde(default)]
    t_scale: f32,
    #[serde(default)]
    steps: u32,
}

#[derive(Serialize)]
struct Output {
    /// `[image_rows, patch_features]` — the reference's `final.tokens`
    /// before its unpatchify. The harness unpatchifies.
    velocity: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// The sigmas actually stepped, so a schedule mismatch reads as one.
    #[serde(default)]
    sigmas: Vec<f32>,
    /// `--euler`: the velocity of every step, and the latent after each.
    #[serde(default)]
    euler_v: Vec<Vec<f32>>,
    #[serde(default)]
    euler_x: Vec<Vec<f32>>,
    /// `--tap_text`: the caption lane's rows of the tapped rectangle,
    /// `[text_rows, width]`.
    #[serde(default)]
    text_tap: Vec<f32>,
    #[serde(default)]
    text_rows: u32,
    /// `--cfg`: the guided velocity group 0's epilogue computed on the
    /// device, `[image_rows, patch_features]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_guided: Vec<f32>,
    /// `--cfg`: the conditional branch's own velocity (group 0), for the
    /// `s = 1` identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_cond: Vec<f32>,
    /// `--cfg`: the unconditional branch's own velocity (group 1), for the
    /// `s = 0` identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_uncond: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cfg_scale: Option<f32>,
    /// `--cache`: the latent after every fire, fire 0 (the seed) first.
    /// `[fires][rows * patch_features]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_x: Vec<Vec<f32>>,
    /// `--cache`: per fire, the relative change the epilogue measured —
    /// the metric it decides the NEXT fire on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_rel: Vec<f32>,
    /// `--cache`: per fire, 1.0 if THIS fire reused the cached velocity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_did: Vec<f32>,
    /// `--cache`: the running skip count the DEVICE kept, after every fire.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_count: Vec<f32>,
    /// `--cache`: fire `k`'s Euler `dt` (0 on the seed fire), so the host can
    /// recover each step's used velocity from the trajectory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_dts: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cache_rows: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sweep_velocity: Vec<Vec<f32>>,
}

/// The two widths this guest sizes channels from: the rotary axes of the
/// `positions` input and the readout's velocity width.
struct Dims {
    axes: u32,
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

/// Checks the three lanes against what this guest binds on each and reads
/// the widths off them.
fn dims(reading: &str) -> Result<Dims> {
    lane_reading(reading, "text", &["text", "positions", "timestep"])?;
    lane_reading(reading, "context", &["context"])?;
    let image = lane_reading(reading, "image", &["latents", "positions", "timestep"])?;
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
        velocity_width,
    })
}

/// Binds attention group `group`'s three lanes on `pass`, lanes `3g`
/// (caption: `text`), `3g + 1` (context: `context`) and `3g + 2` (image:
/// latents `x`), every modulating lane reading `t`. Answers the image lane.
#[allow(clippy::too_many_arguments)]
fn bind_group(
    pass: &ForwardPass,
    case: &Case,
    dims: &Dims,
    reading: &str,
    group: u32,
    text: &[f32],
    context: &[f32],
    x: &Channel,
    t: &Channel,
    tag: &str,
) -> Result<u32> {
    let tag = |what: &str| format!("{tag}{what}_g{group}");
    let (caption, ctx_lane, image) = (3 * group, 3 * group + 1, 3 * group + 2);

    // The caption lane. Its rows are fixed random 256-wide embeddings —
    // this family has no text encoder, which is what makes it an M0 fixture
    // and not a model. It modulates, so it reads the timestep too.
    pass.switch(caption, "reading", reading)?;
    pass.switch(caption, "stream", "text")?;
    pass.group(caption, group)?;
    let txt = Channel::from_shaped([case.text_rows, case.text_width], text).named(&tag("text"));
    let txt_pos = Channel::from_shaped([case.text_rows, dims.axes], case.text_positions.as_slice())
        .named(&tag("txt_pos"));
    pass.input(caption, "text", &txt)?;
    pass.input(caption, "positions", &txt_pos)?;
    pass.input(caption, "timestep", t)?;

    // The context lane: block 2's cross-attention keys and values, and
    // nothing else. No positions — the Wan contract gives cross-attention
    // no rope — and no timestep, since its class never modulates.
    pass.switch(ctx_lane, "reading", reading)?;
    pass.switch(ctx_lane, "stream", "context")?;
    pass.group(ctx_lane, group)?;
    let ctx =
        Channel::from_shaped([case.context_rows, case.context_width], context).named(&tag("ctx"));
    pass.input(ctx_lane, "context", &ctx)?;

    // The image lane: the latents in, the velocity out.
    pass.switch(image, "reading", reading)?;
    pass.switch(image, "stream", "image")?;
    pass.group(image, group)?;
    let img_pos = Channel::from_shaped(
        [case.image_rows, dims.axes],
        case.image_positions.as_slice(),
    )
    .named(&tag("img_pos"));
    pass.input(image, "latents", x)?;
    pass.input(image, "positions", &img_pos)?;
    pass.input(image, "timestep", t)?;
    Ok(image)
}

/// `groups` independent attention groups in ONE pass, each its own
/// velocity: group g >= 2 scales the latent by `1 + g/4`.
async fn sweep_fire(case: &Case, dims: &Dims, reading: &str, groups: u32) -> Result<Vec<Vec<f32>>> {
    let rows = case.image_rows;
    let width = dims.velocity_width;
    let pass = ForwardPass::new();
    let t = Channel::from([case.timestep]).named("sweep_t");
    let mut outs: Vec<(u32, Channel)> = Vec::new();
    for group in 0..groups {
        let scale = if group <= 1 {
            1.0
        } else {
            1.0 + group as f32 / 4.0
        };
        let latents: Vec<f32> = case.latents.iter().map(|v| v * scale).collect();
        let x = Channel::from_shaped([rows, case.patch_features], latents.as_slice())
            .named(&format!("sweep_latents_g{group}"));
        let image = bind_group(
            &pass,
            case,
            dims,
            reading,
            group,
            &case.text,
            &case.context,
            &x,
            &t,
            "sweep_",
        )?;
        let out =
            Channel::new([rows, width], dtype::f32).named(&format!("sweep_velocity_g{group}"));
        outs.push((image, out));
    }
    let readbacks = outs.clone();
    pass.epilogue(move || {
        for (image, out) in &readbacks {
            out.put(intrinsics::velocity(*image, rows, width));
        }
    });
    let pipe = Pipeline::new();
    pass.submit(&pipe).context("the sweep pass")?;
    let mut velocities = Vec::with_capacity(groups as usize);
    for (g, (_, out)) in outs.iter().enumerate() {
        velocities.push(
            out.take_host::<Vec<f32>>()
                .await
                .with_context(|| format!("sweep group {g} readback"))?,
        );
    }
    pipe.close();
    Ok(velocities)
}

/// One guided denoise step: SIX lanes, two attention groups, ONE pass.
///
/// Group 0 carries the case's context, group 1 a zeroed one — the ONLY
/// difference between the two branches, which is what makes the guidance
/// real rather than a combine of one answer with itself. The epilogue reads
/// both image lanes' velocities and combines them at `scale`. The two
/// groups do not attend each other, which is what makes them two
/// independent denoisings rather than one wider one.
///
/// Also publishes the unconditional branch's own velocity, so the caller can
/// check the two identities that need no golden: `s = 1` is the conditional
/// branch exactly, `s = 0` is the unconditional one exactly.
async fn guided_step(
    case: &Case,
    dims: &Dims,
    reading: &str,
    scale: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let rows = case.image_rows;
    let width = dims.velocity_width;
    let zeros = vec![0.0f32; case.context.len()];
    let pass = ForwardPass::new();
    let t = Channel::from([case.timestep]).named("cfg_t");
    let x = Channel::from_shaped([rows, case.patch_features], case.latents.as_slice())
        .named("cfg_latents");
    let cond = bind_group(
        &pass,
        case,
        dims,
        reading,
        0,
        &case.text,
        &case.context,
        &x,
        &t,
        "cfg_",
    )?;
    let uncond = bind_group(
        &pass, case, dims, reading, 1, &case.text, &zeros, &x, &t, "cfg_",
    )?;
    let guided = Channel::new([rows, width], dtype::f32).named("cfg_guided");
    let unguided = Channel::new([rows, width], dtype::f32).named("cfg_uncond");
    let (g, u) = (guided.clone(), unguided.clone());
    pass.epilogue(move || {
        let v_u = intrinsics::velocity(uncond, rows, width);
        g.put(&cfg_combine(
            &intrinsics::velocity(cond, rows, width),
            &v_u,
            &Tensor::constant([scale]),
        ));
        u.put(&v_u);
    });
    let pipe = Pipeline::new();
    pass.submit(&pipe).context("the guided pass")?;
    let guided = guided.take_host::<Vec<f32>>().await?;
    let uncond = unguided.take_host::<Vec<f32>>().await?;
    pipe.close();
    Ok((guided, Vec::new(), uncond))
}

/// **STEP CACHING, ON THE DEVICE** — the proof of
/// `.wiki/imagegen/conditional-fire.md` §9.
///
/// The whole Euler loop runs on the device: `steps + 1` fires of one pass,
/// and nothing per step crosses the host except the readback the
/// harness wants. What is new is five lines inside the image lane's epilogue:
///
/// ```text
/// skip_now = skip.take()                       // decided by the LAST fire
/// v_use    = select(skip_now, cache, velocity())
/// x        = x + dt(k) * v_use                 // the step, unchanged
/// cache.put(v_use)                             // the cache holds what was USED
/// skip.put(and(k >= 1, rel(v_use, cache) < threshold))   // decided FOR the NEXT fire
/// ```
///
/// **The decision is made a fire early on purpose.** A predicate that is to
/// gate a fire has to be written before that fire is armed, so it is computed
/// in the previous fire's epilogue and carried in a channel. That channel is
/// exactly the artifact a conditional fire would fan out into a per-lane byte
/// plane and a `graph::set_conditional_byte` would read (design §6.2/§6.3) —
/// so **this guest does not change when the engine half lands**.
///
/// **The cache is a CHANNEL, never the velocity plane.** A skipped fire's
/// plane holds whatever the last fire left there, at row offsets that belong
/// to whichever lanes that fire seated — the stale-rectangle failure
/// `graph/conditional.cuh` already paid for once. Reading `velocity()` and
/// then discarding it under `select` is what makes the arithmetic independent
/// of whether the trunk actually ran.
///
/// `threshold` absent installs no cache ops at all: the plain device loop,
/// which is the trajectory a cached run is compared against bit for bit.
async fn dpm2m_loop(
    case: &Case,
    dims: &Dims,
    reading: &str,
    steps: u32,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>)> {
    let rows = case.image_rows;
    let feats = case.patch_features;
    let width = dims.velocity_width;
    if width != feats {
        return Err(format!(
            "the solver integrates the latent it predicts: the model reads a {width}-wide \
             velocity and the case carries {feats}-wide patch rows"
        )
        .into());
    }
    let shape = [rows, feats];
    let cells = (rows * feats) as usize;
    let sched = match model::schedule() {
        Some(fact) => FlowMatchEuler::from_schedule(&fact, steps, Some(rows))?,
        None if !case.sigmas.is_empty() => {
            FlowMatchEuler::from_sigmas(case.sigmas.clone(), case.t_scale.max(1.0) as u32)
        }
        None => return Err("this model states no schedule; nothing here denoises".into()),
    };
    let tables = Dpm2m::of(&sched).channels();
    let loops = DenoiseLoop::new(&sched);
    let fires = loops.fires();

    let pass = ForwardPass::new();
    let x = Channel::from_shaped(shape, case.latents.as_slice()).named("dpm_latents");
    let x0_prev = Channel::from_shaped(shape, vec![0f32; cells]).named("dpm_x0_prev");
    let image = bind_group(
        &pass,
        case,
        dims,
        reading,
        0,
        &case.text,
        &case.context,
        &x,
        &loops.timestep(),
        "dpm_",
    )?;

    let out_x = Channel::new(shape, dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("dpm_out_x");
    let out_v = Channel::new(shape, dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("dpm_out_v");
    loops.drive(&pass, move |k| {
        let v = intrinsics::velocity(image, rows, width);
        dpm2m_step(k, &x, &x0_prev, &tables, &v, shape, Some(&out_x));
        out_v.put(&v);
    });

    let mut xs: Vec<Vec<f32>> = Vec::with_capacity(fires as usize);
    let mut vs: Vec<Vec<f32>> = Vec::with_capacity(fires as usize);
    for fire in 0..fires {
        loops
            .fire(&pass)
            .map_err(|why| format!("dpm2m fire {fire}: {why}"))?;
        let landed = out_x
            .take_host::<Vec<f32>>()
            .await
            .map_err(|why| format!("latent readback after dpm2m fire {fire}: {why}"))?;
        let v = out_v
            .take_host::<Vec<f32>>()
            .await
            .map_err(|why| format!("velocity readback after dpm2m fire {fire}: {why}"))?;
        if fire > 0 {
            xs.push(landed);
            vs.push(v);
        }
    }
    loops.close();
    Ok((vs, xs, sched.sigmas.clone()))
}

async fn cached_loop(
    case: &Case,
    dims: &Dims,
    reading: &str,
    steps: u32,
    seed: u32,
    threshold: Option<f32>,
) -> Result<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let rows = case.image_rows;
    let feats = case.patch_features;
    let width = dims.velocity_width;
    if width != feats {
        return Err(format!(
            "the cached loop integrates the latent it predicts: the model reads a {width}-wide \
             velocity and the case carries {feats}-wide patch rows"
        )
        .into());
    }
    let shape = [rows, feats];
    let cells = (rows * feats) as usize;

    let sched = match model::schedule() {
        Some(fact) => FlowMatchEuler::from_schedule(&fact, steps, Some(rows))?,
        None if !case.sigmas.is_empty() => {
            FlowMatchEuler::from_sigmas(case.sigmas.clone(), case.t_scale.max(1.0) as u32)
        }
        None => return Err("this model states no schedule; nothing here denoises".into()),
    };
    let loops = DenoiseLoop::new(&sched);
    let dts = loops.dts();
    let fires = loops.fires();

    // One pass of the three lanes. `x` is bound as the image lane's
    // `latents` input AND loop-carried by the epilogue: the input feed reads
    // the committed cell, the epilogue takes it and puts the next one.
    let pass = ForwardPass::new();
    let x = Channel::from_shaped(shape, vec![0f32; cells]).named("cache_latents");
    let image = bind_group(
        &pass,
        case,
        dims,
        reading,
        0,
        &case.text,
        &case.context,
        &x,
        &loops.timestep(),
        "cache_",
    )?;

    let rng = Channel::from(rng_state(seed)).named("cache_rng");
    let out = Channel::new(shape, dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("cache_out");
    // `[rel this fire, did this fire skip, the device's running skip count]`.
    // One channel rather than three: a probe is one row of three numbers.
    // `new`, not `from`: a seeded channel is USED at construction and
    // `.capacity()` must come first.
    let probe = Channel::new([3u32], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("cache_probe");

    match threshold {
        // ── THE CACHED LOOP ────────────────────────────────────────────────
        Some(thr) => {
            let cache = Channel::from_shaped(shape, vec![0f32; cells]).named("cache_velocity");
            let skip = Channel::from([0f32]).named("cache_skip");
            let count = Channel::from([0f32]).named("cache_count");
            loops.drive(&pass, move |k| {
                let v_now = intrinsics::velocity(image, rows, width);
                let v_prev = cache.take();
                // The flag THIS fire acts on was written by the last one.
                let skipping = gt(&skip.take(), 0.5f32);
                let take_cache = broadcast(reshape(&skipping, [1, 1]), shape);
                let v_use = select(&take_cache, &v_prev, &v_now);

                // `seed_or_step`'s arithmetic, unchanged, over `v_use`.
                let current = x.take();
                let stepped = euler_step(&current, &v_use, &at(&dts.read(), k));
                let state = rng.take();
                let fresh = noise(shape, &state);
                let first = broadcast(reshape(eq(k, 0u32), [1, 1]), shape);
                let next = select(&first, &fresh, &stepped);
                x.put(&next);
                out.put(&next);
                rng.put(&state + &Tensor::constant([0u32, 1u32]));
                // The cache holds what was USED, not what the trunk answered:
                // a run of skips must reuse ONE velocity, not chase the plane.
                cache.put(&v_use);

                // The metric, and the decision for the NEXT fire. `k >= 1`
                // because fire 0 seeds and fire 1 has no cache to compare to.
                let flat = |t: &Tensor| reshape(t, [1, rows * feats]);
                let num = reduce_sum(flat(&abs(&(&v_use - &v_prev))));
                let den = max_elem(reduce_sum(flat(&abs(&v_prev))), 1e-12f32);
                let one = Tensor::constant([1.0f32]);
                let zero = Tensor::constant([0.0f32]);
                // Fire 0's cache is the seeded zero cell, so the ratio there
                // is `|v| / 1e-12` — a number, and a meaningless one. It
                // feeds no decision (`k >= 1` is false), so report 0 rather
                // than publish 1e15 into a series a reader has to squint at.
                let rel = select(&eq(k, 0u32), &zero, &(&num / &den));
                let ahead = and(ge(k, 1u32), lt(&rel, thr));
                skip.put(&select(&ahead, &one, &zero));

                let did = select(&skipping, &one, &zero);
                let total = &count.take() + &did;
                count.put(&total);
                let mut row = broadcast(&zero, [3]);
                row = scatter_set(&row, &Tensor::constant([0u32]), &rel);
                row = scatter_set(&row, &Tensor::constant([1u32]), &did);
                row = scatter_set(&row, &Tensor::constant([2u32]), &total);
                probe.put(&row);
            });
        }
        // ── THE BASELINE ───────────────────────────────────────────────────
        // No cache ops in the trace at all. `seed_or_step` itself, so what a
        // threshold-0 run is compared against is the SHIPPED loop body and
        // not a rearrangement of it.
        None => {
            loops.drive(&pass, move |k| {
                seed_or_step(
                    k,
                    &x,
                    &intrinsics::velocity(image, rows, width),
                    &dts,
                    &rng,
                    shape,
                    Some(&out),
                );
                let zero = Tensor::constant([0.0f32]);
                probe.put(&broadcast(&zero, [3]));
            });
        }
    }

    let mut trace: Vec<Vec<f32>> = Vec::with_capacity(fires as usize);
    let mut rel: Vec<f32> = Vec::with_capacity(fires as usize);
    let mut did: Vec<f32> = Vec::with_capacity(fires as usize);
    let mut count: Vec<f32> = Vec::with_capacity(fires as usize);
    for fire in 0..fires {
        loops
            .fire(&pass)
            .map_err(|why| format!("cached fire {fire}: {why}"))?;
        trace.push(
            out.take_host::<Vec<f32>>()
                .await
                .map_err(|why| format!("latent readback after cached fire {fire}: {why}"))?,
        );
        let row = probe
            .take_host::<Vec<f32>>()
            .await
            .map_err(|why| format!("probe readback after cached fire {fire}: {why}"))?;
        rel.push(row.first().copied().unwrap_or(0.0));
        did.push(row.get(1).copied().unwrap_or(0.0));
        count.push(row.get(2).copied().unwrap_or(0.0));
    }
    loops.close();
    Ok((trace, rel, did, count, sched.dts()))
}

/// One denoise step: three lanes, one group, one pass, one velocity.
#[allow(clippy::too_many_arguments)]
async fn step(
    case: &Case,
    dims: &Dims,
    reading: &str,
    pipe: &Pipeline,
    index: u32,
    latents: &[f32],
    timestep: f32,
    text_tap: Option<&mut Vec<f32>>,
    plan: Option<&ClassPlan>,
) -> Result<Vec<f32>> {
    let tag = format!("s{index}_");
    let rows = case.image_rows;
    let text_rows = case.text_rows;
    // The readout's width is the reading's, which is the patch features —
    // or, under the family's tap knob, the tapped rectangle's.
    let width = dims.velocity_width;
    let t = Channel::from([timestep]).named(&format!("{tag}t"));
    let mut text = case.text.clone();
    if let Some(plan) = plan.filter(|plan| plan.swap_text) {
        let w = case.text_width as usize;
        for (i, class) in plan.text.iter().enumerate() {
            if *class == 1 {
                for v in &mut text[i * w..(i + 1) * w] {
                    *v = -*v;
                }
            }
        }
    }
    let x =
        Channel::from_shaped([rows, case.patch_features], latents).named(&format!("{tag}latents"));
    let pass = ForwardPass::new();
    let image = bind_group(
        &pass,
        case,
        dims,
        reading,
        0,
        &text,
        &case.context,
        &x,
        &t,
        &tag,
    )?;
    if let Some(plan) = plan {
        let (n, cells) = plan.mask(case.context_rows);
        let mask = Channel::from_shaped([n, n], cells).named(&format!("{tag}mask"));
        pass.mask(&mask)?;
    }

    let out = Channel::new([rows, width], dtype::f32).named(&format!("{tag}velocity"));
    // The bisection knob's caption readout: the same intrinsic, off the
    // caption lane's own rows of the tapped rectangle.
    let text_out = text_tap
        .as_ref()
        .map(|_| Channel::new([text_rows, width], dtype::f32).named(&format!("{tag}text_tap")));
    let (readback, tapped) = (out.clone(), text_out.clone());
    pass.epilogue(move || {
        // The denoise reading's `logits()`: `[image rows, C·p²]`.
        readback.put(intrinsics::velocity(image, rows, width));
        if let Some(tapped) = &tapped {
            tapped.put(intrinsics::velocity(0, text_rows, width));
        }
    });

    pass.submit(pipe).context("the denoise step")?;
    let velocity = out.take_host::<Vec<f32>>().await?;
    if let (Some(sink), Some(text_out)) = (text_tap, text_out) {
        *sink = text_out.take_host::<Vec<f32>>().await?;
    }
    Ok(velocity)
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
    let dims = dims(reading)?;
    if !(input.tap || input.tap_text) && dims.velocity_width != case.patch_features {
        return Err(format!(
            "the model reads a {}-wide velocity and the case carries {}-wide patch rows",
            dims.velocity_width, case.patch_features
        )
        .into());
    }
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && case.image_rows > max_rows {
        return Err(format!(
            "{} rows exceed the model's {max_rows} latent rows a pass",
            case.image_rows
        )
        .into());
    }
    let class_plan = match input.classes.as_deref() {
        Some(mode) => ClassPlan::of(mode, &case)?,
        None => None,
    };
    let pipe = Pipeline::new();
    let mut out = Output {
        velocity: Vec::new(),
        image_rows: case.image_rows,
        patch_features: case.patch_features,
        sigmas: Vec::new(),
        euler_v: Vec::new(),
        euler_x: Vec::new(),
        text_tap: Vec::new(),
        text_rows: case.text_rows,
        cfg_guided: Vec::new(),
        cfg_cond: Vec::new(),
        cfg_uncond: Vec::new(),
        cfg_scale: None,
        cache_x: Vec::new(),
        cache_rel: Vec::new(),
        cache_did: Vec::new(),
        cache_count: Vec::new(),
        cache_dts: Vec::new(),
        cache_rows: None,
        sweep_velocity: Vec::new(),
    };

    // Step caching runs the whole schedule on the device — never one step.
    if let Some(groups) = input.sweep.filter(|n| *n > 0) {
        let velocities = sweep_fire(&case, &dims, reading, groups).await?;
        out.velocity = velocities.first().cloned().unwrap_or_default();
        out.sweep_velocity = velocities;
        pipe.close();
        return Ok(out);
    }

    if let Some(solver) = input.solver.as_deref() {
        if solver != "dpm2m" {
            return Err(format!("unknown solver `{solver}`; this guest runs `dpm2m`").into());
        }
        let (vs, xs, sigmas) = dpm2m_loop(&case, &dims, reading, case.steps.max(1)).await?;
        out.velocity = vs.last().cloned().unwrap_or_default();
        out.sigmas = sigmas;
        out.euler_v = vs;
        out.euler_x = xs;
        pipe.close();
        return Ok(out);
    }

    if input.cache {
        let steps = input.cache_steps.or(Some(case.steps)).unwrap_or(0).max(1);
        let seed = input.cache_seed.unwrap_or(7);
        let (trace, rel, did, count, dts) =
            cached_loop(&case, &dims, reading, steps, seed, input.cache_threshold).await?;
        // Fire `k` integrates `dts[k]`: 0 on the seed fire.
        let mut fire_dts = vec![0.0f32];
        fire_dts.extend(dts);
        out.velocity = trace.last().cloned().unwrap_or_default();
        out.cache_x = trace;
        out.cache_rel = rel;
        out.cache_did = did;
        out.cache_count = count;
        out.cache_dts = fire_dts;
        out.cache_rows = Some(case.image_rows);
        pipe.close();
        return Ok(out);
    }

    // Guidance runs one step, six lanes, two groups — never a schedule.
    if input.cfg {
        let scale = input.cfg_scale.unwrap_or(1.0);
        // The conditional branch alone FIRST, as the unguided three-lane
        // step: the same rows at the same timestep with the same context.
        let mut text_tap = Vec::new();
        let cond = step(
            &case,
            &dims,
            reading,
            &pipe,
            0,
            &case.latents,
            case.timestep,
            input.tap_text.then_some(&mut text_tap),
            None,
        )
        .await?;
        pipe.close();
        let (guided, _, uncond) = guided_step(&case, &dims, reading, scale).await?;
        out.velocity = cond.clone();
        out.cfg_guided = guided;
        out.cfg_cond = cond;
        out.cfg_uncond = uncond;
        out.cfg_scale = Some(scale);
        return Ok(out);
    }

    if !input.euler {
        let mut text_tap = Vec::new();
        out.velocity = step(
            &case,
            &dims,
            reading,
            &pipe,
            0,
            &case.latents,
            case.timestep,
            input.tap_text.then_some(&mut text_tap),
            class_plan.as_ref(),
        )
        .await?;
        out.text_tap = text_tap;
        pipe.close();
        return Ok(out);
    }

    // The schedule the model declares, checked against the one the golden
    // was generated under: a silent disagreement here would look like a
    // numerics bug in the trunk.
    let steps = case.steps.max(1);
    let sched = match model::schedule() {
        Some(fact) => FlowMatchEuler::from_schedule(&fact, steps, Some(case.image_rows))?,
        None => FlowMatchEuler::from_sigmas(case.sigmas.clone(), case.t_scale.max(1.0) as u32),
    };
    if !case.sigmas.is_empty() {
        let drift = sched
            .sigmas
            .iter()
            .zip(&case.sigmas)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if sched.sigmas.len() != case.sigmas.len() || drift > 1e-6 {
            return Err(format!(
                "the model's schedule is {:?} and the golden's is {:?}",
                sched.sigmas, case.sigmas
            )
            .into());
        }
    }
    out.sigmas = sched.sigmas.clone();

    let mut x = case.latents.clone();
    for i in 0..steps {
        let v = step(
            &case,
            &dims,
            reading,
            &pipe,
            i,
            &x,
            sched.timestep(i),
            None,
            None,
        )
        .await?;
        let dt = sched.dt(i);
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += dt * vi;
        }
        out.euler_v.push(v);
        out.euler_x.push(x.clone());
    }
    out.velocity = out.euler_v.last().cloned().unwrap_or_default();
    pipe.close();
    Ok(out)
}
