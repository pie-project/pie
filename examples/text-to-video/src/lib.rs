//! A model-agnostic text-to-video sampler, the sibling of `text-to-image`:
//! it reads the model's switches, lanes, latent space and schedule, binds
//! inputs by the standard names, and decodes one latent frame a fire.

use inferlet::frames::{Frames, ImageFormat};
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    prompt: Option<String>,
    /// Comma-separated token ids, for a model whose tokenizer pie lacks.
    #[serde(default)]
    prompt_ids: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    negative_prompt_ids: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    frames: Option<u32>,
    #[serde(default)]
    fps: Option<f32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    out: Option<String>,
    #[serde(default)]
    strength: Option<f32>,
    /// `init_frame_<f>_<n>` (or `init_frame_<f>`): base64 of init frame `f`,
    /// piece `n` (video2video).
    #[serde(flatten)]
    rest: std::collections::BTreeMap<String, inferlet::serde_json::Value>,
}

impl Input {
    fn init_frames(&self) -> Vec<String> {
        let mut found: std::collections::BTreeMap<u64, std::collections::BTreeMap<u64, &str>> =
            std::collections::BTreeMap::new();
        for (name, value) in &self.rest {
            let Some(rest) = name.strip_prefix("init_frame_") else {
                continue;
            };
            let (frame, piece) = rest.split_once('_').unwrap_or((rest, "0"));
            let (Ok(frame), Ok(piece), Some(text)) =
                (frame.parse::<u64>(), piece.parse::<u64>(), value.as_str())
            else {
                continue;
            };
            found.entry(frame).or_default().insert(piece, text);
        }
        found
            .into_values()
            .map(|pieces| pieces.into_values().collect())
            .collect()
    }
}

#[derive(Serialize)]
struct Output {
    model: String,
    architecture: String,
    reading: String,
    prompt: String,
    prompt_source: String,
    width: u32,
    height: u32,
    frames: u32,
    fps: f32,
    latent_frames: u32,
    latent_h: u32,
    latent_w: u32,
    latent_channels: u32,
    spatial_compression: u32,
    temporal_compression: u32,
    decode_fires: u32,
    decode_reading: String,
    init_frames: u32,
    encode_fires: u32,
    strength: f32,
    steps: u32,
    seed: u32,
    guidance: f32,
    cfg: bool,
    sigmas: Vec<f32>,
    context_rows: u32,
    prompt_rows: u32,
    file: String,
    format: String,
}

fn values(switch: &str) -> Vec<String> {
    model::switch(switch)
        .map(|decl| decl.values)
        .unwrap_or_default()
}

fn lane_switches<'a>(reading: &'a str, stream: Option<&'a str>) -> Vec<(&'a str, &'a str)> {
    let mut switches = vec![("reading", reading)];
    if let Some(stream) = stream {
        switches.push(("stream", stream));
    }
    switches
}

fn lane(reading: &str, stream: Option<&str>) -> Result<model::LaneDecl> {
    model::lane_of(&lane_switches(reading, stream))
}

fn set_switches(pass: &ForwardPass, at: u32, reading: &str, stream: Option<&str>) -> Result<()> {
    for (name, value) in lane_switches(reading, stream) {
        pass.switch(at, name, value)?;
    }
    Ok(())
}

fn input<'a>(lane: &'a model::LaneDecl, name: &str) -> Option<&'a model::InputDecl> {
    lane.inputs.iter().find(|input| input.name == name)
}

/// The output `name` a lane reads out, if it reads it.
fn output<'a>(lane: &'a model::LaneDecl, name: &str) -> Option<&'a model::OutputDecl> {
    lane.outputs.iter().find(|output| output.name == name)
}

fn streams() -> Vec<Option<String>> {
    let streams = values("stream");
    if streams.is_empty() {
        vec![None]
    } else {
        streams.into_iter().map(Some).collect()
    }
}

fn stream_reading(reading: &str, name: &str) -> Option<Option<String>> {
    streams()
        .into_iter()
        .find(|stream| lane(reading, stream.as_deref()).is_ok_and(|l| input(&l, name).is_some()))
}

const STANDARD: [&str; 5] = ["latents", "timestep", "guidance", "context", "length"];

fn bind(
    pass: &ForwardPass,
    at: u32,
    lane: &model::LaneDecl,
    have: &[(&str, &Channel)],
    what: &str,
) -> Result<()> {
    for decl in &lane.inputs {
        let (_, ch) = have
            .iter()
            .find(|(name, _)| *name == decl.name)
            .ok_or_else(|| {
                format!(
                    "{what} reads `{}`; this sampler binds {}",
                    decl.name,
                    STANDARD.join(", ")
                )
            })?;
        pass.input(at, &decl.name, ch)?;
    }
    Ok(())
}

struct Text {
    reading: String,
    width: u32,
    /// The most hidden rows one pass reads out.
    rows: u32,
    state: bool,
}

struct Denoise {
    reading: String,
    latent_stream: Option<String>,
    context_stream: Option<String>,
    latents: model::InputDecl,
    context: model::InputDecl,
    guidance: bool,
}

fn text_reading() -> Result<Text> {
    values("reading")
        .into_iter()
        .find_map(|r| {
            let lane = lane(&r, None).ok()?;
            let hidden = output(&lane, "hidden")?;
            lane.embeds.then(|| Text {
                width: hidden.width,
                rows: hidden.max_rows,
                state: lane.state,
                reading: r.clone(),
            })
        })
        .ok_or_else(|| "this model has no text reading".into())
}

fn denoise_reading() -> Result<Denoise> {
    for reading in values("reading") {
        let Some(latent_stream) = stream_reading(&reading, "latents") else {
            continue;
        };
        let video = lane(&reading, latent_stream.as_deref())?;
        let Some(velocity) = output(&video, "velocity").map(|velocity| velocity.width) else {
            continue;
        };
        if video.embeds {
            continue;
        }
        if video.state {
            return Err(format!(
                "reading `{reading}` binds a working set; a denoise pass binds none"
            ));
        }
        let context_stream = stream_reading(&reading, "context")
            .ok_or_else(|| format!("reading `{reading}` reads no `context` for the prompt rows"))?;
        let prompt = lane(&reading, context_stream.as_deref())?;
        let latents = input(&video, "latents").cloned().expect("found by it");
        if latents.axis != model::InputAxis::Voxel {
            return Err(format!(
                "reading `{reading}` reads `latents` per {:?}; this sampler holds the VAE's latent, on the voxel axis",
                latents.axis
            ));
        }
        if input(&video, "timestep").is_none() {
            return Err(format!("reading `{reading}` reads no `timestep`"));
        }
        if velocity != latents.width {
            return Err(format!(
                "reading `{reading}` reads out {velocity}-wide velocity rows over {}-wide latent cells",
                latents.width
            ));
        }
        return Ok(Denoise {
            context: input(&prompt, "context").cloned().expect("found by it"),
            guidance: input(&video, "guidance").is_some(),
            reading,
            latent_stream,
            context_stream,
            latents,
        });
    }
    Err("this model declares no token-less reading of `latents` reading out a velocity".into())
}

fn padded(rows: Option<model::RowCount>, len: u32) -> Option<u32> {
    match rows {
        None => Some(len),
        Some(model::RowCount::Exact(n)) => (len <= n).then_some(n),
        Some(model::RowCount::Multiple(n)) => len.checked_next_multiple_of(n),
    }
}

fn pad_rows(h: &Tensor, len: u32, rows: u32, width: u32) -> Tensor {
    if rows == len {
        return h.clone();
    }
    let i = iota(rows);
    let src = min_elem(&i, len - 1);
    let keep = cast(lt(&i, len), dtype::f32);
    &gather(h, &src) * &broadcast(reshape(&keep, [rows, 1]), [rows, width])
}

#[derive(Default)]
struct Held {
    passes: Vec<ForwardPass>,
    sets: Vec<WorkingSet>,
}

/// The prompt as the rows `context` takes, and their count before the pad.
fn encode(
    pipe: &Pipeline,
    held: &mut Held,
    text: &Text,
    context: &model::InputDecl,
    ids: &[u32],
) -> Result<(Channel, Channel)> {
    if text.width != context.width {
        return Err(format!(
            "the text reading hands back {}-wide rows and input `{}` takes {}",
            text.width, context.name, context.width
        ));
    }
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    if len == 0 {
        return Err("the prompt tokenizes to nothing".into());
    }
    let cap = text.rows;
    if len > cap {
        return Err(format!(
            "the prompt is {len} tokens and one pass reads out {cap} hidden rows"
        ));
    }
    let rows = padded(context.rows, len).ok_or_else(|| {
        format!(
            "input `{}` takes {:?} context rows and the prompt is {len} tokens",
            context.name, context.rows
        )
    })?;
    let ids: Vec<i32> = ids
        .iter()
        .map(|&id| i32::try_from(id).unwrap_or(0))
        .collect();
    let pass = ForwardPass::new();
    pass.switch(0, "reading", &text.reading)?;
    pass.embed(&Channel::from(ids), &Channel::from([0u32, len]))?;
    pass.readout(&Channel::from_iter(0..len))?;
    if text.state {
        let ws = WorkingSet::new();
        pass.state(&[&ws], StateGeometry::default())?;
        held.sets.push(ws);
    }
    let width = text.width;
    let encoded = Channel::new([rows, width], dtype::f32).queued();
    pass.epilogue(move || {
        encoded.put(pad_rows(
            &intrinsics::output_rows(0, "hidden", len),
            len,
            rows,
            width,
        ));
    });
    pass.submit(pipe)?;
    held.passes.push(pass);
    Ok((encoded, Channel::from([len as f32]).named("length")))
}

fn b64_decode(text: &str) -> Result<Vec<u8>> {
    let code = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    };
    let text = text.trim().trim_end_matches('=').as_bytes();
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for (i, &c) in text.iter().enumerate() {
        let six = code(c).ok_or_else(|| format!("an init frame: byte {i} is not base64"))?;
        acc = (acc << 6) | six;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Ok(bytes)
}

struct Vae {
    reading: String,
    stream: Option<String>,
    state: bool,
    input: model::InputDecl,
    width: u32,
}

fn vae_lane(reading: &str) -> Result<Vae> {
    for stream in streams() {
        let Ok(lane) = lane(reading, stream.as_deref()) else {
            continue;
        };
        if let ([input], Some(pixels)) = (&lane.inputs[..], output(&lane, "pixels"))
            && input.axis == model::InputAxis::Voxel
        {
            return Ok(Vae {
                reading: reading.to_string(),
                stream,
                state: lane.state,
                input: input.clone(),
                width: pixels.width,
            });
        }
    }
    Err(format!(
        "reading `{reading}` reads no one clip on the voxel axis"
    ))
}

impl Vae {
    fn pass(&self, state: &WorkingSet) -> Result<ForwardPass> {
        let pass = ForwardPass::new();
        set_switches(&pass, 0, &self.reading, self.stream.as_deref())?;
        if self.state {
            pass.state(&[state], StateGeometry::default())?;
        }
        Ok(pass)
    }
}

fn template_ids(prompt: &str) -> Vec<u32> {
    let mut ids = inferlet::chat::first_user(prompt);
    ids.extend(inferlet::chat::cue());
    ids
}

fn ids_of(text: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for piece in text.split(|c: char| c == ',' || c.is_whitespace()) {
        if piece.is_empty() {
            continue;
        }
        out.push(
            piece
                .parse::<u32>()
                .map_err(|_| format!("`{piece}` in the id list is not a token id"))?,
        );
    }
    if out.is_empty() {
        return Err("the id list is empty".into());
    }
    Ok(out)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let text = text_reading()?;
    let denoise = denoise_reading()?;
    let readings = values("reading");
    let decode = vae_lane(
        readings
            .iter()
            .find(|r| *r == "vae.decode")
            .ok_or("this model declares no `vae.decode` reading, so its latent has no way out")?,
    )?;

    let space = model::latent().ok_or("this model states no latent space")?;
    let (sc, tc, first, c) = (
        space.spatial_compression.max(1),
        space.temporal_compression.max(1),
        space.first_frames.max(1),
        space.channels,
    );
    if tc == 1 {
        return Err(
            "this model's latent has no temporal compression: `text-to-image` is its driver".into(),
        );
    }
    if c != denoise.latents.width {
        return Err(format!(
            "the latent space has {c} channels and `latents` reads {}",
            denoise.latents.width
        ));
    }
    if decode.input.width != c || decode.width != 3 {
        return Err(format!(
            "`{}` reads {}-wide cells into {}-wide pixels; the latent has {c} channels and a \
             frame is RGB",
            decode.reading, decode.input.width, decode.width
        ));
    }
    let height = (input.height.unwrap_or(480) / sc).max(1) * sc;
    let width = (input.width.unwrap_or(832) / sc).max(1) * sc;
    let (lh, lw) = (height / sc, width / sc);
    let frames = input.frames.unwrap_or(17).max(1);
    if frames < first || !(frames - first).is_multiple_of(tc) {
        let below = first + tc * (frames.saturating_sub(first) / tc);
        return Err(format!(
            "this model's latent frames stand for {first}, then {tc} each, so {frames} frames is \
             not a clip it has: the nearest are {below} and {}",
            below + tc
        ));
    }
    let lt = (frames - first) / tc + 1;
    let pixel_frames = |j: u32| if j == 0 { first } else { tc };
    let lat = [lt, lh, lw, c];
    let cells = lt * lh * lw;
    let plane = lh * lw * c;

    let fact = model::schedule().ok_or("this model states no schedule; nothing here denoises")?;
    let steps = match input.steps.filter(|s| *s > 0) {
        Some(steps) => steps,
        None if !fact.pinned_sigmas.is_empty() => {
            u32::try_from(fact.pinned_sigmas.len()).unwrap_or(8)
        }
        None => 8,
    };
    let full = FlowMatchEuler::from_schedule(&fact, steps, None)?;
    let init_b64 = input.init_frames();
    let has_init = !init_b64.is_empty();
    let strength = input.strength.unwrap_or(0.6);
    if has_init && !(0.0 < strength && strength <= 1.0) {
        return Err(format!(
            "`strength` is {strength}; it is a fraction in (0, 1]"
        ));
    }
    let sched = if has_init {
        let run = ((steps as f32 * strength).round() as u32).clamp(1, steps);
        FlowMatchEuler::from_sigmas(
            full.sigmas[(steps - run) as usize..].to_vec(),
            full.train_steps,
        )
    } else {
        full
    };
    let sigma0 = sched.sigmas[0];
    let mut ts = sched.timesteps();
    ts.push(0.0);
    let fps = input.fps.unwrap_or(24.0);

    let (ids, prompt, prompt_source) = match (&input.prompt_ids, &input.prompt) {
        (Some(ids), _) => {
            let ids = ids_of(ids)?;
            let shown = format!("<{} ids>", ids.len());
            (ids, shown, "prompt-ids")
        }
        (None, Some(text)) if !text.trim().is_empty() => {
            (template_ids(text), text.clone(), "prompt")
        }
        _ => {
            return Err(
                "pass `prompt` or `prompt_ids`: this program films what it is told to".into(),
            );
        }
    };
    let negative = match (&input.negative_prompt_ids, &input.negative_prompt) {
        (Some(ids), _) => Some(ids_of(ids)?),
        (None, Some(text)) if !text.trim().is_empty() => Some(template_ids(text)),
        _ => None,
    };
    let guidance = input.guidance.unwrap_or(1.0);
    if negative.is_some() && denoise.guidance {
        return Err(format!(
            "reading `{}` reads a `guidance` input, so this row is guidance-distilled and a \
             negative prompt has no lane to ride",
            denoise.reading
        ));
    }
    let cfg = negative.is_some() && guidance > 1.0;

    let pipe = Pipeline::new();
    let mut held = Held::default();
    let init: Option<Channel> = if !has_init {
        None
    } else {
        let enc = vae_lane(readings.iter().find(|r| *r == "vae.encode").ok_or(
            "this model declares no `vae.encode` reading, so an init clip has no way in",
        )?)?;
        if enc.width != c {
            return Err(format!(
                "`{}` reads out {}-wide cells, the latent has {c} channels",
                enc.reading, enc.width
            ));
        }
        if init_b64.len() as u32 != frames {
            return Err(format!(
                "the init clip is {} frames and this run is {frames}",
                init_b64.len()
            ));
        }
        let mut stills = Vec::with_capacity(init_b64.len());
        for (f, b64) in init_b64.iter().enumerate() {
            let picture = Frames::decode(&b64_decode(b64)?)
                .map_err(|why| format!("init frame {f} does not decode: {why}"))?;
            if picture.width() != width || picture.height() != height || picture.count() != 1 {
                return Err(format!(
                    "init frame {f} is {}x{}x{} and this run is one {width}x{height} still a frame",
                    picture.width(),
                    picture.height(),
                    picture.count()
                ));
            }
            stills.push(picture);
        }
        let init = Channel::new([lt, plane], dtype::f32).queued().named("init");
        let enc_state = WorkingSet::new();
        let rows = lh * lw;
        let mut at = 0usize;
        for j in 0..lt {
            let take = pixel_frames(j);
            let parts: Vec<&Frames> = stills[at..at + take as usize].iter().collect();
            at += take as usize;
            let chunk = Frames::concat(&parts).map_err(|why| format!("encode chunk {j}: {why}"))?;
            let pass = enc.pass(&enc_state)?;
            let pixels = Channel::seeded([take, height, width, enc.input.width], dtype::f32)
                .named("vae_pixels_in");
            pass.input(0, &enc.input.name, &pixels)?;
            pass.epilogue(move || {
                let base = if j == 0 {
                    broadcast(Tensor::constant([0f32]), [lt, plane])
                } else {
                    init.take()
                };
                let frame = reshape(intrinsics::output_rows(0, "pixels", rows), [1, plane]);
                init.put(scatter_set(&base, Tensor::constant([j]), &frame));
            });
            pixels.set_frames(&chunk)?;
            pass.submit(&pipe)
                .with_context(|| format!("the `{}` fire on chunk {j}", enc.reading))?;
            held.passes.push(pass);
        }
        held.sets.push(enc_state);
        Some(init)
    };

    let x = Channel::new(lat, dtype::f32).queued().named("x");
    let k = Channel::new([1], dtype::u32).queued().named("k");
    let t = Channel::new([1], dtype::f32).queued().named("t");
    let ts = Channel::from(ts).named("ts");
    let sigmas = Channel::from(sched.sigmas.clone()).named("sigmas");
    let seed = input.seed.unwrap_or(0);
    let draw = Channel::from(rng_state(seed)).named("draw");
    let seed_loop = move || {
        let eps = normal(draw.read(), lat);
        let x0 = match &init {
            None => eps,
            Some(init) => {
                &(&reshape(init.read(), lat) * &broadcast(Tensor::constant([1.0 - sigma0]), lat))
                    + &(&eps * &broadcast(Tensor::constant([sigma0]), lat))
            }
        };
        x.put(&x0);
        k.put(Tensor::constant([0u32]));
        t.put(at(&ts.read(), &Tensor::constant([0u32])));
    };

    let context = encode(&pipe, &mut held, &text, &denoise.context, &ids)?;
    let uncond = match &negative {
        Some(neg) if cfg => Some(encode(&pipe, &mut held, &text, &denoise.context, neg)?),
        _ => None,
    };

    let guidance_ch = Channel::from([guidance]).named("guidance");
    let n = cells;
    let denoise_pass = |first: bool| -> Result<ForwardPass> {
        let pass = ForwardPass::new();
        let mut lanes = 0u32;
        for (group, ctx) in [Some(&context), uncond.as_ref()]
            .into_iter()
            .flatten()
            .enumerate()
        {
            for stream in [&denoise.latent_stream, &denoise.context_stream] {
                let at = lanes;
                lanes += 1;
                set_switches(&pass, at, &denoise.reading, stream.as_deref())?;
                pass.group(at, group as u32)?;
                bind(
                    &pass,
                    at,
                    &lane(&denoise.reading, stream.as_deref())?,
                    &[
                        ("latents", &x),
                        ("timestep", &t),
                        ("guidance", &guidance_ch),
                        ("context", &ctx.0),
                        ("length", &ctx.1),
                    ],
                    &format!("reading `{}` on lane {at}", denoise.reading),
                )?;
            }
        }
        if first {
            pass.prologue(seed_loop);
        }
        pass.epilogue(move || {
            let step = k.take();
            let v = if cfg {
                cfg_combine(
                    &intrinsics::output_rows(0, "velocity", n),
                    &intrinsics::output_rows(2, "velocity", n),
                    &Tensor::constant([guidance]),
                )
            } else {
                intrinsics::output_rows(0, "velocity", n)
            };
            let sig = sigmas.read();
            let next = &step + 1u32;
            let dt = &at(&sig, &next) - &at(&sig, &step);
            let stepped = reshape(euler_step(&reshape(x.take(), [n, c]), &v, &dt), lat);
            x.put(&stepped);
            t.take();
            t.put(at(&ts.read(), &next));
            k.put(&next);
        });
        Ok(pass)
    };
    let first_pass = denoise_pass(true)?;
    let step_pass = denoise_pass(false)?;
    for fire in 0..sched.steps() {
        let pass = if fire == 0 { &first_pass } else { &step_pass };
        pass.submit(&pipe)
            .with_context(|| format!("denoise fire {fire}"))?;
    }

    let vae_state = WorkingSet::new();
    let j = Channel::new([1], dtype::u32).queued().named("j");
    let decode_pass = |head: bool| -> Result<(ForwardPass, Channel)> {
        let pass = decode.pass(&vae_state)?;
        let frame = Channel::new([lh, lw, c], dtype::f32)
            .queued()
            .named("vae_latent");
        pass.input(0, &decode.input.name, &frame)?;
        pass.prologue(move || {
            let at = if head {
                Tensor::constant([0u32])
            } else {
                j.take()
            };
            let frames = reshape(x.read(), [lt, lh * lw * c]);
            frame.put(reshape(gather(&frames, &at), [lh, lw, c]));
            j.put(&at + 1u32);
        });
        let rows = pixel_frames(u32::from(!head)) * height * width;
        let out = Channel::new([rows, 3], dtype::f32).named("vae_pixels");
        pass.epilogue(move || {
            frame.take();
            out.put(intrinsics::output_rows(0, "pixels", rows));
        });
        Ok((pass, out))
    };
    let head = decode_pass(true)?;
    let tail = decode_pass(false)?;
    let mut clips = Vec::with_capacity(lt as usize);
    for fire in 0..lt {
        let (pass, px) = if fire == 0 { &head } else { &tail };
        pass.submit(&pipe)
            .with_context(|| format!("the `{}` fire on latent frame {fire}", decode.reading))?;
        clips.push(px.take_frames(width, height, pixel_frames(fire), fps)?);
    }
    pipe.close();
    let video = Frames::concat(&clips.iter().collect::<Vec<_>>())
        .map_err(|why| format!("frames.concat: {why}"))?;

    let name = input.out.clone().unwrap_or_else(|| "video".to_string());
    let asked_format = input.format.as_deref().unwrap_or("mp4");
    let (format, ext) = match asked_format {
        "mp4" | "mp4-h264" => (ImageFormat::Mp4H264, "mp4"),
        "y4m" => (ImageFormat::Y4m, "y4m"),
        "rgb8" | "raw-rgb8" => (ImageFormat::RawRgb8, "rgb"),
        other => {
            return Err(format!(
                "unknown video format {other:?}; try 'mp4', 'y4m' or 'rgb8'"
            ));
        }
    };
    let file = format!("{name}.{ext}");
    inferlet::session::send_frames(&video, format, &file)
        .map_err(|why| format!("session.send-frames({asked_format}): {why}"))?;

    Ok(Output {
        model: model::name(),
        architecture: model::architecture(),
        reading: denoise.reading.clone(),
        prompt,
        prompt_source: prompt_source.to_string(),
        width,
        height,
        frames,
        fps,
        latent_frames: lt,
        latent_h: lh,
        latent_w: lw,
        latent_channels: c,
        spatial_compression: sc,
        temporal_compression: tc,
        decode_fires: lt,
        decode_reading: decode.reading,
        init_frames: init_b64.len() as u32,
        encode_fires: if has_init { lt } else { 0 },
        strength: if has_init { strength } else { 1.0 },
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
        sigmas: sched.sigmas.clone(),
        context_rows: context.0.shape().dims()[0],
        prompt_rows: ids.len() as u32,
        file,
        format: asked_format.to_string(),
    })
}
