//! The pie half of the `mini-dit` golden: `scripts/imagegen/mini_dit_ref.py`'s
//! inputs in, the velocity `[h·w, C]` out as JSON for `mini_dit_parity.py`.
//!
//! Two parity modes: one step against `mini_dit_dump_bf16.npz`'s `velocity`,
//! and the four-step Euler schedule against `mini_dit_euler_bf16.npz`'s
//! `euler.v{0..3}` / `euler.x{1..4}`.
//!
//! And two modes that need no golden because every claim they make is
//! WITHIN one fire's own answers: `--cfg` (classifier-free guidance on the
//! device, six lanes and two groups in one pass) and
//! `--cache` (step caching on the device, the whole Euler schedule as fires
//! with the cache decision carried in a channel from one fire to the next —
//! `.wiki/imagegen/conditional-fire.md`). Their harnesses are
//! `mini_dit_parity.py guidance` and `mini_dit_cache.py`.
//!
//! A step is one pass; attention group `g` is lanes `3g` (caption), `3g + 1`
//! (context) and `3g + 2` (image), and only the image lane reads out.
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
    /// `--classes none|all|split|split-swap`: one step under a two-class
    /// attention mask; `split-swap` also negates the class-1 caption rows.
    #[serde(default)]
    classes: Option<String>,
    /// `--sweep N`: N independent attention groups in ONE pass; group g >= 2
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
        // The mask addresses token rows: the image lane's patches, in grid order.
        let rows = case.image_tokens as usize;
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
    /// Context rows have no class: they attend, and are attended by, every row.
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

/// The reference's inputs, row-major; the latent unpatched `[latent_h, latent_w, channels]`.
#[derive(Deserialize)]
struct Case {
    latents: Vec<f32>,
    latent_h: u32,
    latent_w: u32,
    channels: u32,
    image_tokens: u32,
    /// `[text_rows, text_width]`.
    text: Vec<f32>,
    text_rows: u32,
    text_width: u32,
    /// `[context_rows, context_width]`.
    context: Vec<f32>,
    context_rows: u32,
    context_width: u32,
    /// One step's timestep, or — under `--euler` — the schedule.
    timestep: f32,
    #[serde(default)]
    sigmas: Vec<f32>,
    #[serde(default)]
    t_scale: f32,
    #[serde(default)]
    steps: u32,
}

impl Case {
    fn voxels(&self) -> u32 {
        self.latent_h * self.latent_w
    }
    fn lat(&self) -> [u32; 3] {
        [self.latent_h, self.latent_w, self.channels]
    }
}

#[derive(Serialize)]
struct Output {
    /// `[h·w, C]`, or under a tap the intermediate on the image lane's token rows.
    velocity: Vec<f32>,
    image_rows: u32,
    velocity_width: u32,
    latent_h: u32,
    latent_w: u32,
    channels: u32,
    /// The sigmas actually stepped, so a schedule mismatch reads as one.
    #[serde(default)]
    sigmas: Vec<f32>,
    /// `--euler` / `--solver`: every step's velocity, and the latent after it.
    #[serde(default)]
    euler_v: Vec<Vec<f32>>,
    #[serde(default)]
    euler_x: Vec<Vec<f32>>,
    /// `--tap_text`: the caption lane's rows of the tapped rectangle.
    #[serde(default)]
    text_tap: Vec<f32>,
    #[serde(default)]
    text_rows: u32,
    /// `--cfg`: the guided velocity the epilogue computed on the device.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_guided: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_cond: Vec<f32>,
    /// `--cfg`: the unconditional branch's own velocity (group 1).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_uncond: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cfg_scale: Option<f32>,
    /// `--cache`: the latent after every fire, fire 0 (the seed) first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_x: Vec<Vec<f32>>,
    /// `--cache`: per fire, the relative change the epilogue measured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_rel: Vec<f32>,
    /// `--cache`: per fire, 1.0 if THIS fire reused the cached velocity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_did: Vec<f32>,
    /// `--cache`: the running skip count the DEVICE kept, after every fire.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_count: Vec<f32>,
    /// `--cache`: fire `k`'s Euler `dt` (0 on the seed fire).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cache_dts: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cache_rows: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sweep_velocity: Vec<Vec<f32>>,
}

const STREAMS: [&str; 3] = ["text", "context", "image"];

struct Shared {
    text: Channel,
    context: Channel,
    blank: Channel,
}

impl Shared {
    fn of(case: &Case) -> Shared {
        Shared {
            text: Channel::from_shaped([case.text_rows, case.text_width], case.text.as_slice())
                .named("text"),
            context: Channel::from_shaped(
                [case.context_rows, case.context_width],
                case.context.as_slice(),
            )
            .named("context"),
            blank: Channel::from_shaped(
                [case.context_rows, case.context_width],
                vec![0f32; case.context.len()],
            )
            .named("blank_context"),
        }
    }
}

fn velocity_width(reading: &str) -> Result<u32> {
    for stream in STREAMS {
        let lane = model::lane_of(&[("reading", reading), ("stream", stream)])?;
        if lane.state || lane.embeds {
            return Err(format!(
                "reading `{reading}`'s `{stream}` lane binds a working set or tokens; a denoise \
                 pass binds neither"
            ));
        }
        for input in &lane.inputs {
            if !["text", "context", "latents", "timestep"].contains(&input.name.as_str()) {
                return Err(format!(
                    "reading `{reading}`'s `{stream}` lane reads `{}`; this pass binds text, \
                     context, latents and timestep",
                    input.name
                ));
            }
        }
    }
    let image = model::lane_of(&[("reading", reading), ("stream", "image")])?;
    image
        .outputs
        .iter()
        .find(|output| output.name == "velocity")
        .map(|velocity| velocity.width)
        .ok_or_else(|| format!("reading `{reading}`'s image lane reads out no velocity"))
}

/// Binds attention group `group`'s three lanes; answers the image lane.
fn bind_group(
    pass: &ForwardPass,
    reading: &str,
    group: u32,
    text: &Channel,
    context: &Channel,
    x: &Channel,
    t: &Channel,
) -> Result<u32> {
    for (i, stream) in STREAMS.into_iter().enumerate() {
        let lane = 3 * group + i as u32;
        pass.switch(lane, "reading", reading)?;
        pass.switch(lane, "stream", stream)?;
        pass.group(lane, group)?;
        for input in model::lane_of(&[("reading", reading), ("stream", stream)])?.inputs {
            let ch = match input.name.as_str() {
                "text" => text,
                "context" => context,
                "latents" => x,
                "timestep" => t,
                other => return Err(format!("the `{stream}` lane reads `{other}`")),
            };
            pass.input(lane, &input.name, ch)?;
        }
    }
    Ok(3 * group + 2)
}

async fn sweep_fire(
    case: &Case,
    shared: &Shared,
    width: u32,
    reading: &str,
    groups: u32,
) -> Result<Vec<Vec<f32>>> {
    let rows = case.voxels();
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
        let x = Channel::from_shaped(case.lat(), latents.as_slice())
            .named(&format!("sweep_latents_g{group}"));
        let image = bind_group(&pass, reading, group, &shared.text, &shared.context, &x, &t)?;
        let out =
            Channel::new([rows, width], dtype::f32).named(&format!("sweep_velocity_g{group}"));
        outs.push((image, out));
    }
    let readbacks = outs.clone();
    pass.epilogue(move || {
        for (image, out) in &readbacks {
            out.put(intrinsics::output_rows(*image, "velocity", rows));
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

/// One CFG step: group 0 reads the context, group 1 a zeroed one; answers (guided, uncond).
async fn guided_step(
    case: &Case,
    shared: &Shared,
    width: u32,
    reading: &str,
    scale: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let rows = case.voxels();
    let pass = ForwardPass::new();
    let t = Channel::from([case.timestep]).named("cfg_t");
    let x = Channel::from_shaped(case.lat(), case.latents.as_slice()).named("cfg_latents");
    let cond = bind_group(&pass, reading, 0, &shared.text, &shared.context, &x, &t)?;
    let uncond = bind_group(&pass, reading, 1, &shared.text, &shared.blank, &x, &t)?;
    let guided = Channel::new([rows, width], dtype::f32).named("cfg_guided");
    let unguided = Channel::new([rows, width], dtype::f32).named("cfg_uncond");
    pass.epilogue(move || {
        let v_u = intrinsics::output_rows(uncond, "velocity", rows);
        guided.put(cfg_combine(
            &intrinsics::output_rows(cond, "velocity", rows),
            &v_u,
            &Tensor::constant([scale]),
        ));
        unguided.put(&v_u);
    });
    let pipe = Pipeline::new();
    pass.submit(&pipe).context("the guided pass")?;
    let g = guided.take_host::<Vec<f32>>().await?;
    let u = unguided.take_host::<Vec<f32>>().await?;
    pipe.close();
    Ok((g, u))
}

fn schedule(case: &Case, steps: u32) -> Result<FlowMatchEuler> {
    match model::schedule() {
        Some(fact) => FlowMatchEuler::from_schedule(&fact, steps, Some(case.image_tokens)),
        None if !case.sigmas.is_empty() => Ok(FlowMatchEuler::from_sigmas(
            case.sigmas.clone(),
            case.t_scale.max(1.0) as u32,
        )),
        None => Err("this model states no schedule; nothing here denoises".into()),
    }
}

/// DPM-Solver++ 2M as a device loop; fire 0 seeds and steps nowhere.
async fn dpm2m_loop(
    case: &Case,
    shared: &Shared,
    reading: &str,
    steps: u32,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>)> {
    let (n, c, lat) = (case.voxels(), case.channels, case.lat());
    let flat = [n, c];
    let sched = schedule(case, steps)?;
    let tables = Dpm2m::of(&sched).channels();
    let loops = DenoiseLoop::new(&sched);
    let fires = loops.fires();

    let pass = ForwardPass::new();
    let x = Channel::from_shaped(lat, case.latents.as_slice()).named("dpm_latents");
    let x0_prev = Channel::from_shaped(flat, vec![0f32; (n * c) as usize]).named("dpm_x0_prev");
    let image = bind_group(
        &pass,
        reading,
        0,
        &shared.text,
        &shared.context,
        &x,
        &loops.timestep(),
    )?;
    let out_x = Channel::new(flat, dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("dpm_out_x");
    let out_v = Channel::new(flat, dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("dpm_out_v");
    loops.drive(&pass, move |k| {
        let v = intrinsics::output_rows(image, "velocity", n);
        // `dpm2m_step`'s arithmetic over the flattened latent.
        let scalar = |table: &Channel| broadcast(reshape(at(&table.read(), k), [1, 1]), flat);
        let current = reshape(x.take(), flat);
        let x0 = &current - &(&v * &scalar(&tables.sigma_s));
        let prev = x0_prev.take();
        let mix = scalar(&tables.mix);
        let one = broadcast(Tensor::constant([1.0f32]), flat);
        let d = &(&x0 * &(&one + &mix)) - &(&prev * &mix);
        let stepped = &(&current * &scalar(&tables.ratio)) + &(&d * &scalar(&tables.gain));
        let first = broadcast(reshape(eq(k, 0u32), [1, 1]), flat);
        let next = select(&first, &current, &stepped);
        x.put(reshape(&next, lat));
        x0_prev.put(&x0);
        out_x.put(&next);
        out_v.put(&v);
    });

    let mut xs = Vec::with_capacity(fires as usize);
    let mut vs = Vec::with_capacity(fires as usize);
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

/// Fire `k`'s latent: fire 0 a fresh keyed draw, later fires one Euler step over `v`.
#[allow(clippy::too_many_arguments)]
fn seed_or_euler(
    k: &Tensor,
    x: &Channel,
    v: &Tensor,
    dts: &Channel,
    rng: &Channel,
    flat: [u32; 2],
    lat: [u32; 3],
    out: &Channel,
) {
    let current = reshape(x.take(), flat);
    let stepped = euler_step(&current, v, &at(&dts.read(), k));
    let state = rng.take();
    let fresh = noise(flat, &state);
    let first = broadcast(reshape(eq(k, 0u32), [1, 1]), flat);
    let next = select(&first, &fresh, &stepped);
    x.put(reshape(&next, lat));
    out.put(&next);
    rng.put(&state + &Tensor::constant([0u32, 1u32]));
}

/// Step caching on the device; `threshold` absent is the plain loop, with no cache ops.
async fn cached_loop(
    case: &Case,
    shared: &Shared,
    reading: &str,
    steps: u32,
    seed: u32,
    threshold: Option<f32>,
) -> Result<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let (n, c, lat) = (case.voxels(), case.channels, case.lat());
    let flat = [n, c];
    let cells = (n * c) as usize;
    let sched = schedule(case, steps)?;
    let loops = DenoiseLoop::new(&sched);
    let dts = loops.dts();
    let fires = loops.fires();

    // `x` is both the image lane's `latents` input and the epilogue's loop state.
    let pass = ForwardPass::new();
    let x = Channel::from_shaped(lat, vec![0f32; cells]).named("cache_latents");
    let image = bind_group(
        &pass,
        reading,
        0,
        &shared.text,
        &shared.context,
        &x,
        &loops.timestep(),
    )?;
    let rng = Channel::from(rng_state(seed)).named("cache_rng");
    let out = Channel::new(flat, dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("cache_out");
    // `[rel this fire, did this fire skip, the device's running skip count]`.
    let probe = Channel::new([3u32], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("cache_probe");

    match threshold {
        Some(thr) => {
            let cache = Channel::from_shaped(flat, vec![0f32; cells]).named("cache_velocity");
            let skip = Channel::from([0f32]).named("cache_skip");
            let count = Channel::from([0f32]).named("cache_count");
            loops.drive(&pass, move |k| {
                let v_now = intrinsics::output_rows(image, "velocity", n);
                let v_prev = cache.take();
                // The flag THIS fire acts on was written by the last one.
                let skipping = gt(skip.take(), 0.5f32);
                let v_use = select(broadcast(reshape(&skipping, [1, 1]), flat), &v_prev, &v_now);
                seed_or_euler(k, &x, &v_use, &dts, &rng, flat, lat, &out);
                cache.put(&v_use);

                let row = |t: &Tensor| reshape(t, [1, n * c]);
                let num = reduce_sum(row(&abs(&(&v_use - &v_prev))));
                let den = max_elem(reduce_sum(row(&abs(&v_prev))), 1e-12f32);
                let one = Tensor::constant([1.0f32]);
                let zero = Tensor::constant([0.0f32]);
                let rel = select(eq(k, 0u32), &zero, &(&num / &den));
                let ahead = and(ge(k, 1u32), lt(&rel, thr));
                skip.put(select(&ahead, &one, &zero));

                let did = select(&skipping, &one, &zero);
                let total = &count.take() + &did;
                count.put(&total);
                let mut probed = broadcast(&zero, [3]);
                probed = scatter_set(&probed, Tensor::constant([0u32]), &rel);
                probed = scatter_set(&probed, Tensor::constant([1u32]), &did);
                probed = scatter_set(&probed, Tensor::constant([2u32]), &total);
                probe.put(&probed);
            });
        }
        None => {
            loops.drive(&pass, move |k| {
                seed_or_euler(
                    k,
                    &x,
                    &intrinsics::output_rows(image, "velocity", n),
                    &dts,
                    &rng,
                    flat,
                    lat,
                    &out,
                );
                probe.put(broadcast(Tensor::constant([0.0f32]), [3]));
            });
        }
    }

    let (mut trace, mut rel, mut did, mut count) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
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

#[allow(clippy::too_many_arguments)]
async fn step(
    case: &Case,
    shared: &Shared,
    width: u32,
    rows: u32,
    reading: &str,
    pipe: &Pipeline,
    index: u32,
    latents: &[f32],
    timestep: f32,
    text_tap: Option<&mut Vec<f32>>,
    plan: Option<&ClassPlan>,
) -> Result<Vec<f32>> {
    let tag = format!("s{index}_");
    let text_rows = case.text_rows;
    let t = Channel::from([timestep]).named(&format!("{tag}t"));
    let swapped = plan.filter(|plan| plan.swap_text).map(|plan| {
        let w = case.text_width as usize;
        let mut text = case.text.clone();
        for (i, class) in plan.text.iter().enumerate() {
            if *class == 1 {
                for v in &mut text[i * w..(i + 1) * w] {
                    *v = -*v;
                }
            }
        }
        Channel::from_shaped([case.text_rows, case.text_width], text).named(&format!("{tag}text"))
    });
    let x = Channel::from_shaped(case.lat(), latents).named(&format!("{tag}latents"));
    let pass = ForwardPass::new();
    let image = bind_group(
        &pass,
        reading,
        0,
        swapped.as_ref().unwrap_or(&shared.text),
        &shared.context,
        &x,
        &t,
    )?;
    if let Some(plan) = plan {
        let (n, cells) = plan.mask(case.context_rows);
        let mask = Channel::from_shaped([n, n], cells).named(&format!("{tag}mask"));
        pass.mask(&mask)?;
    }
    let out = Channel::new([rows, width], dtype::f32).named(&format!("{tag}velocity"));
    let text_out = text_tap
        .as_ref()
        .map(|_| Channel::new([text_rows, width], dtype::f32).named(&format!("{tag}text_tap")));
    let tapped = text_out;
    pass.epilogue(move || {
        out.put(intrinsics::output_rows(image, "velocity", rows));
        if let Some(tapped) = &tapped {
            tapped.put(intrinsics::output_rows(0, "velocity", text_rows));
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
    if case.latents.len() != (case.voxels() * case.channels) as usize {
        return Err(format!(
            "`latents` carries {} numbers and a {}x{}x{} latent is {}",
            case.latents.len(),
            case.latent_h,
            case.latent_w,
            case.channels,
            case.voxels() * case.channels
        ));
    }

    let reading = "denoise";
    let width = velocity_width(reading)?;
    let tapped = input.tap || input.tap_text;
    if !tapped && width != case.channels {
        return Err(format!(
            "the model reads out a {width}-wide velocity and the case's latent has {} channels",
            case.channels
        ));
    }
    // A velocity is one row a voxel; a tapped intermediate one row a token.
    let rows = if tapped {
        case.image_tokens
    } else {
        case.voxels()
    };
    let class_plan = match input.classes.as_deref() {
        Some(mode) => ClassPlan::of(mode, &case)?,
        None => None,
    };
    let shared = Shared::of(&case);
    let pipe = Pipeline::new();
    let mut out = Output {
        velocity: Vec::new(),
        image_rows: rows,
        velocity_width: width,
        latent_h: case.latent_h,
        latent_w: case.latent_w,
        channels: case.channels,
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

    if let Some(groups) = input.sweep.filter(|n| *n > 0) {
        let velocities = sweep_fire(&case, &shared, width, reading, groups).await?;
        out.velocity = velocities.first().cloned().unwrap_or_default();
        out.sweep_velocity = velocities;
        pipe.close();
        return Ok(out);
    }

    if let Some(solver) = input.solver.as_deref() {
        if solver != "dpm2m" {
            return Err(format!(
                "unknown solver `{solver}`; this guest runs `dpm2m`"
            ));
        }
        let (vs, xs, sigmas) = dpm2m_loop(&case, &shared, reading, case.steps.max(1)).await?;
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
            cached_loop(&case, &shared, reading, steps, seed, input.cache_threshold).await?;
        let mut fire_dts = vec![0.0f32];
        fire_dts.extend(dts);
        out.velocity = trace.last().cloned().unwrap_or_default();
        out.cache_x = trace;
        out.cache_rel = rel;
        out.cache_did = did;
        out.cache_count = count;
        out.cache_dts = fire_dts;
        out.cache_rows = Some(case.voxels());
        pipe.close();
        return Ok(out);
    }

    if input.cfg {
        let scale = input.cfg_scale.unwrap_or(1.0);
        let mut text_tap = Vec::new();
        let cond = step(
            &case,
            &shared,
            width,
            rows,
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
        let (guided, uncond) = guided_step(&case, &shared, width, reading, scale).await?;
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
            &shared,
            width,
            rows,
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

    let steps = case.steps.max(1);
    let sched = schedule(&case, steps)?;
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
            ));
        }
    }
    out.sigmas = sched.sigmas.clone();

    let mut x = case.latents.clone();
    for i in 0..steps {
        let v = step(
            &case,
            &shared,
            width,
            rows,
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
