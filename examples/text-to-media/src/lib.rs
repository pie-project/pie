//! A model-agnostic text-to-image/video sampler: rectified-flow Euler over the
//! model's latent, by the standard reading, stream and input names.

use inferlet::frames::{Frames, ImageFormat};
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Default)]
#[serde(default)]
struct Input {
    prompt: String,
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
    format: Option<String>,
    out: Option<String>,
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

/// The prompt's hidden rows, zero-padded to the rows `context` takes, and their count.
fn encode(pipe: &Pipeline, context: &model::InputDecl, prompt: &str) -> Result<(Channel, Channel)> {
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
    let pass = ForwardPass::new();
    pass.switch(0, "reading", "text")?;
    pass.embed(
        &Channel::from(ids.iter().map(|&id| id as i32).collect::<Vec<_>>()),
        &Channel::from([0u32, len]),
    )?;
    pass.readout(&Channel::from_iter(0..len))?;
    if text.state {
        pass.state(&[&WorkingSet::new()], StateGeometry::default())?;
    }
    let hidden = Channel::new([rows, width], dtype::f32).queued();
    pass.epilogue(move || {
        let i = iota(rows);
        let keep = cast(lt(&i, len), dtype::f32);
        let h = gather(
            intrinsics::output_rows(0, "hidden", len),
            min_elem(&i, len - 1),
        );
        hidden.put(&h * &broadcast(reshape(&keep, [rows, 1]), [rows, width]));
    });
    pass.submit(pipe)?;
    Ok((hidden, Channel::from([len as f32])))
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
    if input.prompt.is_empty() {
        return Err("empty prompt".into());
    }
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
    let cond = encode(&pipe, context_in, &input.prompt)?;
    let uncond = match &negative {
        Some(neg) if cfg => Some(encode(&pipe, context_in, neg)?),
        _ => None,
    };

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

    let x = Channel::new(lat, dtype::f32).queued();
    let k = Channel::new([1], dtype::u32).queued();
    let t = Channel::new([1], dtype::f32).queued();
    let ts = Channel::from(ts);
    let sigmas = Channel::from(sched.sigmas.clone());
    let guidance_ch = Channel::from([guidance]);
    let mask = mask.map(|m| Channel::from_shaped([n, 1], m.as_slice()));
    let seed = input.seed;
    let seed_ch = Channel::from([seed, 0u32]);
    let key = move |i: &Tensor| scatter_set(seed_ch.read(), Tensor::constant([1u32]), i);
    let prologue = move || {
        let zero = Tensor::constant([0u32]);
        let eps = normal(key(&zero), lat);
        x.put(match start {
            None => eps,
            Some(start) => &(&reshape(start.read(), lat) * (1.0 - sigma0)) + &(&eps * sigma0),
        });
        k.put(&zero);
        t.put(at(&ts.read(), &zero));
    };
    let epilogue = move || {
        let step = k.take();
        let next = &step + 1u32;
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
        let sig = sigmas.read();
        let mut x1 = euler_step(
            &reshape(x.take(), [n, c]),
            &v,
            &(&at(&sig, &next) - &at(&sig, &step)),
        );
        if let (Some(m), Some(start)) = (mask, start) {
            let to = broadcast(reshape(at(&sig, &next), [1, 1]), [n, c]);
            let known = &(&start.read() * &(1.0f32 - &to)) + &(&normal(key(&next), [n, c]) * &to);
            let m = broadcast(m.read(), [n, c]);
            x1 = &(&m * &x1) + &(&(1.0f32 - &m) * &known);
        }
        x.put(reshape(&x1, lat));
        t.take();
        t.put(at(&ts.read(), &next));
        k.put(&next);
    };
    let denoise = |first: bool| -> Result<ForwardPass> {
        let pass = ForwardPass::new();
        let groups = [Some(&cond), uncond.as_ref()].into_iter().flatten();
        for (g, (ctx, len)) in groups.enumerate() {
            for (s, (stream, lane)) in [("context", &context), ("latents", &latents)]
                .into_iter()
                .enumerate()
            {
                let at = 2 * g as u32 + s as u32;
                on(&pass, at, "denoise", stream)?;
                pass.group(at, g as u32)?;
                for decl in &lane.inputs {
                    let ch = match decl.name.as_str() {
                        "latents" => &x,
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
            pass.prologue(prologue);
        }
        pass.epilogue(epilogue);
        Ok(pass)
    };
    let (head, rest) = (denoise(true)?, denoise(false)?);
    for step in 0..sched.steps() {
        (if step == 0 { &head } else { &rest }).submit(&pipe)?;
    }

    let vae = lane("vae.decode", "latents")?;
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
    let file = format!("{}.{ext}", input.out.as_deref().unwrap_or("out"));
    let media = Frames::concat(&clips.iter().collect::<Vec<_>>())?;
    inferlet::session::send_frames(&media, format, &file)?;
    Ok(Output {
        file,
        size: format!("{width}x{height}"),
        frames,
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
    })
}
