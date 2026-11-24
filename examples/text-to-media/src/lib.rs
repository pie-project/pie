//! A model-agnostic text-to-image/video sampler: rectified-flow Euler over the
//! model's latent, by the standard reading, stream and input names.

use inferlet::frames::{Frames, ImageFormat};
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Default)]
#[serde(default)]
struct Input {
    prompt: String,
    /// More than one image a run, denoised together: a prompt each (a list, or one string
    /// split by `|`), seeded `seed + i` and written as `<out>_<i>`.
    #[serde(deserialize_with = "prompt_list")]
    prompts: Vec<String>,
    negative_prompt: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    frames: Option<u32>,
    fps: Option<f32>,
    steps: Option<u32>,
    seed: u32,
    guidance: Option<f32>,
    strength: Option<f32>,
    /// The region to repaint, `x0,y0,x1,y1` in pixels.
    mask_rect: Option<String>,
    /// Solver: `"euler"` (default) or `"dpm2m"` (second-order DPM-Solver++ 2M).
    solver: Option<String>,
    /// Host-driven stepping: drain a `[1]` sync word after every denoise fire.
    host_step: bool,
    /// Skip the `vae.decode` stage and drain a `[1]` sync word after the last denoise fire.
    skip_vae: bool,
    /// Skip sending the encoded media over the session channel.
    no_send: bool,
    format: Option<String>,
    out: Option<String>,
}

/// A list of prompts, or one string of them split by `|` (as `pie run -- --prompts "a|b"`).
fn prompt_list<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Prompts {
        List(Vec<String>),
        Joined(String),
    }
    Ok(match Prompts::deserialize(d)? {
        Prompts::List(v) => v,
        Prompts::Joined(s) => s.split('|').map(|p| p.trim().to_string()).collect(),
    })
}

#[derive(Serialize)]
struct Output {
    file: String,
    size: String,
    frames: u32,
    steps: u32,
    seed: u32,
    guidance: f32,
    cfg: bool,
}

fn lane(reading: &str, stream: &str) -> Result<model::LaneDecl> {
    model::lane_of(&[("reading", reading), ("stream", stream)])
}

fn on(pass: &ForwardPass, at: u32, reading: &str, stream: &str) -> Result<()> {
    pass.switch(at, "reading", reading)?;
    pass.switch(at, "stream", stream)?;
    Ok(())
}

/// Each prompt's hidden rows, zero-padded to the rows `context` takes, and their count: one
/// text pass, a lane and group a prompt.
fn encode(
    pipe: &Pipeline,
    context: &model::InputDecl,
    prompts: &[&str],
) -> Result<Vec<(Channel, Channel)>> {
    let text =
        model::lane_of(&[("reading", "text")]).map_err(|_| "this model has no text reading")?;
    let width = text
        .outputs
        .iter()
        .find(|o| o.name == "hidden")
        .map_or(0, |o| o.width);
    if width != context.width {
        return Err(format!(
            "`text` reads out {width}-wide rows and `context` takes {}",
            context.width
        ));
    }
    let pass = ForwardPass::new();
    let (mut tokens, mut indptr, mut lens) = (Vec::new(), vec![0u32], Vec::new());
    let mut out = Vec::with_capacity(prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let mut ids = inferlet::chat::first_user(prompt);
        ids.extend(inferlet::chat::cue());
        let len = ids.len() as u32;
        let rows = match context.rows {
            None => len,
            Some(model::RowCount::Exact(n)) if len <= n => n,
            Some(model::RowCount::Multiple(n)) => len.next_multiple_of(n),
            Some(rows) => {
                return Err(format!(
                    "the prompt is {len} tokens and `context` takes {rows:?} rows"
                ));
            }
        };
        pass.switch(i as u32, "reading", "text")?;
        if prompts.len() > 1 {
            pass.group(i as u32, i as u32)?;
        }
        tokens.extend(ids.iter().map(|&id| id as i32));
        indptr.push(indptr[i] + len);
        lens.push((len, rows));
        out.push((
            Channel::new([rows, width], dtype::f32).queued(),
            Channel::from([len as f32]),
        ));
    }
    let all = indptr[prompts.len()];
    pass.embed(&Channel::from(tokens), &Channel::from(indptr))?;
    pass.readout(&Channel::from_iter(0..all))?;
    if text.state {
        let sets: Vec<WorkingSet> = prompts.iter().map(|_| WorkingSet::new()).collect();
        pass.state(&sets.iter().collect::<Vec<_>>(), StateGeometry::default())?;
    }
    let hidden: Vec<Channel> = out.iter().map(|o| o.0).collect();
    pass.epilogue(move || {
        for (i, (&(len, rows), h_ch)) in lens.iter().zip(&hidden).enumerate() {
            let r = iota(rows);
            let keep = cast(lt(&r, len), dtype::f32);
            let h = gather(
                intrinsics::output_rows(i as u32, "hidden", len),
                min_elem(&r, len - 1),
            );
            h_ch.put(&h * &broadcast(reshape(&keep, [rows, 1]), [rows, width]));
        }
    });
    pass.submit(pipe)?;
    Ok(out)
}

fn parse_rect(text: &str) -> Result<[u32; 4]> {
    let p: Vec<u32> = text
        .split(',')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    match p[..] {
        [x0, y0, x1, y1] if x0 < x1 && y0 < y1 => Ok([x0, y0, x1, y1]),
        _ => Err(format!(
            "`mask_rect` is `x0,y0,x1,y1` in pixels, not `{text}`"
        )),
    }
}

fn picture(name: &str, width: u32, height: u32) -> Result<Option<Frames>> {
    let Some(bytes) = inferlet::session::file(name) else {
        return Ok(None);
    };
    let media = Frames::decode(&bytes).map_err(|why| format!("file `{name}`: {why}"))?;
    if media.width() != width || media.height() != height {
        return Err(format!(
            "file `{name}` is {}x{}; this run is {width}x{height}",
            media.width(),
            media.height()
        ));
    }
    Ok(Some(media))
}

fn need(value: Option<u32>, name: &str) -> Result<u32> {
    value.ok_or_else(|| format!("`{name}` is required"))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let prompts: Vec<String> = if input.prompts.is_empty() {
        vec![input.prompt.clone()]
    } else {
        input.prompts.clone()
    };
    if prompts.iter().any(String::is_empty) {
        return Err("empty prompt".into());
    }
    let images = prompts.len();
    let space = model::latent().ok_or("this model states no latent space")?;
    let (sc, tc, first, c) = (
        space.spatial_compression,
        space.temporal_compression.max(1),
        space.first_frames.max(1),
        space.channels,
    );
    let video = tc > 1;
    let frames = if video {
        need(input.frames, "frames")?
    } else {
        input.frames.unwrap_or(1)
    };
    if frames < first || (frames - first) % tc != 0 {
        return Err(format!(
            "a clip is {first} frames, then {tc} a latent frame; {frames} is not one"
        ));
    }
    let width = (need(input.width, "width")? / sc).max(1) * sc;
    let height = (need(input.height, "height")? / sc).max(1) * sc;
    let (lt, lh, lw) = ((frames - first) / tc + 1, height / sc, width / sc);
    let pixel_frames = move |j: u32| if j == 0 { first } else { tc };
    let lat = [lt, lh, lw, c];
    let n = lt * lh * lw;

    let init = picture("init", width, height)?;
    if let Some(init) = &init
        && init.count() != frames
    {
        return Err(format!(
            "`init` has {} frames for a {frames}-frame run",
            init.count()
        ));
    }
    let rect = input.mask_rect.as_deref().map(parse_rect).transpose()?;
    let gray = match picture("mask", width, height)? {
        Some(m) => Some(m.encode(ImageFormat::RawRgb8)?),
        None => None,
    };
    let mask: Option<Vec<f32>> = (rect.is_some() || gray.is_some()).then(|| {
        let px = |x: u32, y: u32| match (rect, &gray) {
            (Some([x0, y0, x1, y1]), _) => {
                f32::from(u8::from(x >= x0 && x < x1 && y >= y0 && y < y1))
            }
            (_, Some(g)) => f32::from(g[3 * (y * width + x) as usize]) / 255.0,
            _ => 0.0,
        };
        let cell = |a: u32, b: u32| {
            let sum: f32 = (0..sc * sc)
                .map(|i| px(b * sc + i % sc, a * sc + i / sc))
                .sum();
            sum / (sc * sc) as f32
        };
        let one: Vec<f32> = (0..lh * lw).map(|i| cell(i / lw, i % lw)).collect();
        one.repeat(lt as usize)
    });
    if mask.is_some() && init.is_none() {
        return Err("a mask needs the picture it is cut from: attach `init`".into());
    }
    if images > 1 && (init.is_some() || input.solver.as_deref() == Some("dpm2m")) {
        return Err("a batch of prompts takes no `init`, mask or `dpm2m`".into());
    }

    let fact = model::schedule().ok_or("this model states no schedule")?;
    let steps = match input.steps {
        None if !fact.pinned_sigmas.is_empty() => fact.pinned_sigmas.len() as u32,
        steps => need(steps, "steps")?,
    };
    let mut sched = FlowMatchEuler::from_schedule(&fact, steps, None)?;
    if init.is_some() {
        let strength = input.strength.unwrap_or(0.6);
        if !(0.0 < strength && strength <= 1.0) {
            return Err(format!(
                "`strength` is {strength}; it is a fraction in (0, 1]"
            ));
        }
        let run = ((steps as f32 * strength).round() as u32).clamp(1, steps);
        sched = FlowMatchEuler::from_sigmas(
            sched.sigmas[(steps - run) as usize..].to_vec(),
            sched.train_steps,
        );
    }
    let sigma0 = sched.sigmas[0];
    let mut ts = sched.timesteps();
    ts.push(0.0);

    let latents = lane("denoise", "latents")?;
    let context = lane("denoise", "context")?;
    let context_in = context
        .inputs
        .iter()
        .find(|i| i.name == "context")
        .ok_or("`denoise` reads no `context`")?;
    let guidance = input.guidance.unwrap_or(1.0);
    let negative = input.negative_prompt;
    let distilled = latents.inputs.iter().any(|i| i.name == "guidance");
    if negative.is_some() && distilled {
        return Err(
            "this model is guidance-distilled: pass `guidance` without a negative prompt".into(),
        );
    }
    let cfg = negative.is_some() && guidance > 1.0;

    let pipe = Pipeline::new();
    // each image's prompt rows, and the negative's under guidance: one text pass
    let mut texts: Vec<&str> = prompts.iter().map(String::as_str).collect();
    if let Some(neg) = negative.as_deref().filter(|_| cfg) {
        texts.push(neg);
    }
    let mut encoded = encode(&pipe, context_in, &texts)?;
    let uncond = cfg.then(|| encoded.pop()).flatten();
    let conds: Vec<_> = encoded.into_iter().map(|c| (c, uncond)).collect();

    let start = init
        .map(|init| -> Result<Channel> {
            let vae = lane("vae.encode", "latents")?;
            let ws = WorkingSet::new();
            let start = Channel::new([n, c], dtype::f32).queued();
            let all = Channel::seeded([frames, height * width * 3], dtype::f32);
            all.set_frames(&init)?;
            let mut at = 0;
            for j in 0..lt {
                let take = pixel_frames(j);
                let pass = ForwardPass::new();
                on(&pass, 0, "vae.encode", "latents")?;
                if vae.state {
                    pass.state(&[&ws], StateGeometry::default())?;
                }
                let pixels = Channel::new([take, height, width, 3], dtype::f32).queued();
                pass.input(0, "pixels", &pixels)?;
                pass.prologue(move || {
                    let rows = gather(all.read(), &iota(take) + at);
                    pixels.put(reshape(rows, [take, height, width, 3]));
                });
                at += take;
                pass.epilogue(move || {
                    pixels.take();
                    let base = if j == 0 {
                        broadcast(Tensor::constant([0f32]), [n, c])
                    } else {
                        start.take()
                    };
                    let cells = &iota(lh * lw) + j * lh * lw;
                    let latent = intrinsics::output_rows(0, "pixels", lh * lw);
                    start.put(scatter_set(&base, &cells, &latent));
                });
                pass.submit(&pipe)?;
            }
            Ok(start)
        })
        .transpose()?;

    let xs: Vec<Channel> = (0..images)
        .map(|_| Channel::new(lat, dtype::f32).queued())
        .collect();
    let x = xs[0];
    let use_dpm2m = input.solver.as_deref() == Some("dpm2m");
    let x0_prev = use_dpm2m.then(|| Channel::new([n, c], dtype::f32).queued());
    let dpm_tables = use_dpm2m.then(|| Dpm2m::of(&sched).channels());
    let host_step = input.host_step;
    let skip_vae = input.skip_vae;
    let num_steps = sched.steps();
    let k = Channel::new([1], dtype::u32).queued();
    let t = Channel::new([1], dtype::f32).queued();
    let ts = Channel::from(ts);
    let sigmas = Channel::from(sched.sigmas.clone());
    let guidance_ch = Channel::from([guidance]);
    let mask = mask.map(|m| Channel::from_shaped([n, 1], m.as_slice()));
    let seed = input.seed;
    // (image i's noise keyed by `seed + i`)
    let seeds: Vec<Channel> = (0..images as u32)
        .map(|i| Channel::from([seed + i, 0u32]))
        .collect();
    let seed_ch = seeds[0];
    let key = move |i: &Tensor| scatter_set(seed_ch.read(), Tensor::constant([1u32]), i);
    let xs_pro = xs.clone();
    let seeds_pro = seeds.clone();
    let prologue = move || {
        let zero = Tensor::constant([0u32]);
        for (i, (x, sd)) in xs_pro.iter().zip(&seeds_pro).enumerate() {
            let eps = normal(scatter_set(sd.read(), Tensor::constant([1u32]), &zero), lat);
            x.put(match start {
                Some(start) if i == 0 => {
                    &(&reshape(start.read(), lat) * (1.0 - sigma0)) + &(&eps * sigma0)
                }
                _ => eps,
            });
        }
        if let Some(prev) = &x0_prev {
            prev.put(broadcast(Tensor::constant([0f32]), [n, c]));
        }
        k.put(&zero);
        t.put(at(&ts.read(), &zero));
    };
    let denoise = |first: bool, with_sync: bool| -> Result<(ForwardPass, Option<Channel>)> {
        let pass = ForwardPass::new();
        let step_sync = with_sync.then(|| Channel::new([1], dtype::u32));
        // each image its group (and its negative's): a context lane and a latents lane each
        let groups = conds.iter().zip(&xs).flat_map(|((cond, uncond), x)| {
            [Some((cond, x)), uncond.as_ref().map(|u| (u, x))]
                .into_iter()
                .flatten()
        });
        for (g, ((ctx, len), x)) in groups.enumerate() {
            for (s, (stream, lane)) in [("context", &context), ("latents", &latents)]
                .into_iter()
                .enumerate()
            {
                let at = 2 * g as u32 + s as u32;
                on(&pass, at, "denoise", stream)?;
                pass.group(at, g as u32)?;
                for decl in &lane.inputs {
                    let ch = match decl.name.as_str() {
                        "latents" => x,
                        "timestep" => &t,
                        "guidance" => &guidance_ch,
                        "context" => ctx,
                        "length" => len,
                        other => {
                            return Err(format!(
                                "`denoise` reads `{other}`, which this sampler does not bind"
                            ));
                        }
                    };
                    pass.input(at, &decl.name, ch)?;
                }
            }
        }
        if first {
            pass.prologue(prologue.clone());
        }
        let sync_ep = step_sync;
        let xs_ep = xs.clone();
        pass.epilogue(move || {
            let step = k.take();
            let next = &step + 1u32;
            let sig = sigmas.read();
            // (each image's velocity: its latents lane's, its group's lanes in turn)
            let per = if cfg { 4 } else { 2 };
            for (i, xi) in xs_ep.iter().enumerate().skip(1) {
                let base = per * i as u32;
                let v = intrinsics::output_rows(base + 1, "velocity", n);
                let v = if cfg {
                    cfg_combine(
                        &v,
                        &intrinsics::output_rows(base + 3, "velocity", n),
                        &Tensor::constant([guidance]),
                    )
                } else {
                    v
                };
                let x1 = euler_step(
                    &reshape(xi.take(), [n, c]),
                    &v,
                    &(&at(&sig, &next) - &at(&sig, &step)),
                );
                xi.put(reshape(&x1, lat));
            }
            let v = intrinsics::output_rows(1, "velocity", n);
            let v = if cfg {
                cfg_combine(
                    &v,
                    &intrinsics::output_rows(3, "velocity", n),
                    &Tensor::constant([guidance]),
                )
            } else {
                v
            };
            let mut x1 = if let (Some(prev_ch), Some(tables)) = (&x0_prev, &dpm_tables) {
                let current = reshape(x.take(), [n, c]);
                let scalar = |table: &Channel, idx: &Tensor| {
                    broadcast(reshape(at(&table.read(), idx), [1, 1]), [n, c])
                };
                let x0 = &current - &(&v * &scalar(&tables.sigma_s, &next));
                let prev = prev_ch.take();
                let mix = scalar(&tables.mix, &next);
                let one = broadcast(Tensor::constant([1.0f32]), [n, c]);
                let d = &(&x0 * &(&one + &mix)) - &(&prev * &mix);
                let stepped = &(&current * &scalar(&tables.ratio, &next))
                    + &(&d * &scalar(&tables.gain, &next));
                prev_ch.put(&x0);
                stepped
            } else {
                euler_step(
                    &reshape(x.take(), [n, c]),
                    &v,
                    &(&at(&sig, &next) - &at(&sig, &step)),
                )
            };
            if let (Some(m), Some(start)) = (mask, start) {
                let to = broadcast(reshape(at(&sig, &next), [1, 1]), [n, c]);
                let known =
                    &(&start.read() * &(1.0f32 - &to)) + &(&normal(key(&next), [n, c]) * &to);
                let m = broadcast(m.read(), [n, c]);
                x1 = &(&m * &x1) + &(&(1.0f32 - &m) * &known);
            }
            x.put(reshape(&x1, lat));
            if let Some(sync) = &sync_ep {
                sync.put(&next);
            }
            t.take();
            t.put(at(&ts.read(), &next));
            k.put(&next);
        });
        Ok((pass, step_sync))
    };
    let head = denoise(true, host_step || (skip_vae && num_steps == 1))?;
    let middle = denoise(false, host_step)?;
    let tail = denoise(false, host_step || skip_vae)?;
    for step in 0..num_steps {
        let (pass, sync) = if step == 0 {
            &head
        } else if step + 1 < num_steps {
            &middle
        } else {
            &tail
        };
        pass.submit(&pipe)?;
        if host_step || (skip_vae && step + 1 == num_steps) {
            if let Some(s) = sync {
                let _ = s.take_host::<u32>().await?;
            }
        }
    }

    if skip_vae {
        pipe.close();
        return Ok(Output {
            file: String::new(),
            size: format!("{width}x{height}"),
            frames,
            steps: num_steps,
            seed,
            guidance,
            cfg,
        });
    }

    let vae = lane("vae.decode", "latents")?;
    // each image's frames decoded in turn
    let mut media = Vec::with_capacity(images);
    for &x in &xs {
        let ws = WorkingSet::new();
        let j = Channel::new([1], dtype::u32).queued();
        let decode = |first: bool| -> Result<(ForwardPass, Channel)> {
            let pass = ForwardPass::new();
            on(&pass, 0, "vae.decode", "latents")?;
            if vae.state {
                pass.state(&[&ws], StateGeometry::default())?;
            }
            let frame = Channel::new([lh, lw, c], dtype::f32).queued();
            pass.input(0, "latents", &frame)?;
            pass.prologue(move || {
                let at = if first {
                    Tensor::constant([0u32])
                } else {
                    j.take()
                };
                frame.put(reshape(
                    gather(reshape(x.read(), [lt, lh * lw * c]), &at),
                    [lh, lw, c],
                ));
                j.put(&at + 1u32);
            });
            let rows = pixel_frames(u32::from(!first)) * height * width;
            let pixels = Channel::new([rows, 3], dtype::f32);
            pass.epilogue(move || {
                frame.take();
                pixels.put(intrinsics::output_rows(0, "pixels", rows));
            });
            Ok((pass, pixels))
        };
        let (head, rest) = (decode(true)?, decode(false)?);
        let mut clips = Vec::new();
        for f in 0..lt {
            let (pass, pixels) = if f == 0 { &head } else { &rest };
            pass.submit(&pipe)?;
            clips.push(pixels.take_frames(
                width,
                height,
                pixel_frames(f),
                input.fps.unwrap_or(24.0),
            )?);
        }
        media.push(clips);
    }
    pipe.close();

    let (format, ext) =
        match input
            .format
            .as_deref()
            .unwrap_or(if frames == 1 { "png" } else { "mp4" })
        {
            "png" => (ImageFormat::Png, "png"),
            "mp4" => (ImageFormat::Mp4H264, "mp4"),
            "y4m" => (ImageFormat::Y4m, "y4m"),
            "rgb8" => (ImageFormat::RawRgb8, "rgb"),
            other => {
                return Err(format!(
                    "unknown format `{other}`; try png, mp4, y4m or rgb8"
                ));
            }
        };
    let stem = input.out.as_deref().unwrap_or("out");
    let mut files = Vec::with_capacity(images);
    for (i, clips) in media.iter().enumerate() {
        let file = if images == 1 {
            format!("{stem}.{ext}")
        } else {
            format!("{stem}_{i}.{ext}")
        };
        if !input.no_send {
            let media = Frames::concat(&clips.iter().collect::<Vec<_>>())?;
            inferlet::session::send_frames(&media, format, &file)?;
        }
        files.push(file);
    }
    Ok(Output {
        file: files.join(","),
        size: format!("{width}x{height}"),
        frames,
        steps: num_steps,
        seed,
        guidance,
        cfg,
    })
}
