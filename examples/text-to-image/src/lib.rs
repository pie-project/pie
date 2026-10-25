//! **THE MODEL-AGNOSTIC `text-to-image` INFERLET**: a prompt in, an image
//! out, with nothing about any family spelled here.
//!
//! What it knows are MODEL FACTS: the readings (the values of the `reading`
//! switch) and what each lane of them binds (`model::lane_of`), the VAE's
//! latent space (`model::latent()`: channels and spatial compression), and
//! the schedule. What it never knows are ARCHITECTURE FACTS: patch sizes,
//! rotary positions, padding, the order of a family's features. The latent
//! it holds is the VAE's own `[h, w, C]`; the denoiser patchifies it and
//! computes its positions in its own graph.
//!
//! # ROLES, BY WHAT THE LANES BIND
//!
//! | role | the fact that says so |
//! |---|---|
//! | text encoder | a lane that `embeds` and reads out `Hidden` |
//! | denoiser | a token-less lane that reads `latents` and reads out `Velocity` |
//! | the latent lane | the denoiser's `stream` whose lane reads `latents` |
//! | the prompt lane | the denoiser's `stream` whose lane reads `context` |
//! | a bridge | a token-less `Hidden` reading turning the encoder's rows into the width `context` takes (Z-Image's `refine`) |
//! | VAE decode / encode | `vae.decode` / `vae.encode`, a `Pixels` readout |
//!
//! The standard input names are `latents`, `timestep`, `guidance` and
//! `context`; a lane that reads any other required input is refused by
//! name, never zero-filled.
//!
//! # ONE PIPELINE, NOTHING BACK
//!
//! Every pass of the job goes down one pipeline, in order: the text encoder
//! (its rows stay on the device, padded there to the rows the `context`
//! input states), the bridge if the model has one, the VAE encoder for
//! img2img, `steps` denoise fires, and the VAE decode. A pipeline runs in
//! submission order, so nothing waits on the host until the decoded picture
//! is handed to the client.
//!
//! The latent `x` is a device-only channel. The last pass before the loop
//! draws it (`normal` on a keyed state), or noises the encoded picture to the
//! schedule's first sigma, and seeds the loop's clock. The denoise epilogue
//! is the sampler: `x <- x + (sigma' - sigma) v`, `v` the guided
//! `u + s (c - u)` when a negative prompt adds a second attention group to
//! the pass.

use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    out: Option<String>,
    /// IMG2IMG: a picture to start from, base64 of an encoded still. It is
    /// run through the family's `vae.encode` reading and noised to the
    /// schedule's `strength` point.
    #[serde(default)]
    init_image: Option<String>,
    /// `init_image_<n>` / `mask_image_<n>`: the same in argv-sized pieces.
    #[serde(flatten)]
    rest: std::collections::BTreeMap<String, inferlet::serde_json::Value>,
    /// The fraction of the schedule to run over the encoded picture.
    #[serde(default)]
    strength: Option<f32>,
    /// Inpainting: the region to repaint, `x0,y0,x1,y1` in pixels.
    #[serde(default)]
    mask_rect: Option<String>,
}

impl Input {
    fn pieces(&self, prefix: &str) -> String {
        let mut found: Vec<(u64, &str)> = self
            .rest
            .iter()
            .filter_map(|(name, value)| {
                let n = name.strip_prefix(prefix)?.parse::<u64>().ok()?;
                Some((n, value.as_str()?))
            })
            .collect();
        found.sort_unstable_by_key(|(n, _)| *n);
        found.into_iter().map(|(_, text)| text).collect()
    }
}

#[derive(Serialize)]
struct Output {
    model: String,
    architecture: String,
    reading: String,
    prompt: String,
    width: u32,
    height: u32,
    /// The VAE latent this run denoised, `[latent_h, latent_w, channels]`.
    latent_h: u32,
    latent_w: u32,
    latent_channels: u32,
    spatial_compression: u32,
    steps: u32,
    seed: u32,
    guidance: f32,
    cfg: bool,
    sigmas: Vec<f32>,
    /// The rows the prompt lane carried, and the readings that made them.
    context_rows: u32,
    context_readings: Vec<String>,
    decode_reading: String,
    file: String,
    /// Inpainting: latent cells the mask touches.
    mask_cells: Option<u32>,
}

/// What a lane reads out.
fn readout_kind(lane: &model::LaneDecl) -> Option<model::ReadoutKind> {
    lane.readout.as_ref().map(|readout| readout.kind)
}

fn values(switch: &str) -> Vec<String> {
    model::switch(switch)
        .map(|decl| decl.values)
        .unwrap_or_default()
}

/// The switches of a lane: `reading`, and `stream` where the model has one.
fn lane_switches<'a>(reading: &'a str, stream: Option<&'a str>) -> Vec<(&'a str, &'a str)> {
    let mut switches = vec![("reading", reading)];
    if let Some(stream) = stream {
        switches.push(("stream", stream));
    }
    switches
}

fn lane(reading: &str, stream: Option<&str>) -> Result<model::LaneDecl> {
    model::lane_of(&lane_switches(reading, stream)).map_err(Into::into)
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

/// The streams a reading's lanes may take: the `stream` switch's values, or
/// the one default lane of a model without that switch.
fn streams() -> Vec<Option<String>> {
    let streams = values("stream");
    if streams.is_empty() {
        vec![None]
    } else {
        streams.into_iter().map(Some).collect()
    }
}

/// The stream of `reading` whose lane reads `name`.
fn stream_reading(reading: &str, name: &str) -> Option<Option<String>> {
    streams()
        .into_iter()
        .find(|stream| lane(reading, stream.as_deref()).is_ok_and(|l| input(&l, name).is_some()))
}

struct Text {
    reading: String,
    width: u32,
    reads_kv: bool,
}

/// A token-less reading that turns `from`-wide rows into `to`-wide hidden
/// rows: its stream and the one input it reads them through.
struct Bridge {
    reading: String,
    stream: Option<String>,
    input: String,
}

struct Denoise {
    reading: String,
    latent_stream: Option<String>,
    context_stream: Option<String>,
    latents: model::InputDecl,
    context: model::InputDecl,
    guidance: bool,
    velocity_width: u32,
}

fn text_reading() -> Result<Text> {
    values("reading")
        .into_iter()
        .find_map(|r| {
            let lane = lane(&r, None).ok()?;
            let readout = lane.readout.as_ref()?;
            (lane.embeds && readout.kind == model::ReadoutKind::Hidden).then(|| Text {
                width: readout.width,
                reads_kv: lane.reads_kv,
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
        let image = lane(&reading, latent_stream.as_deref())?;
        if image.embeds || readout_kind(&image) != Some(model::ReadoutKind::Velocity) {
            continue;
        }
        if image.reads_kv {
            return Err(format!("reading `{reading}` binds a kv space; a denoise pass binds none").into());
        }
        let context_stream = stream_reading(&reading, "context").ok_or_else(|| {
            format!("reading `{reading}` reads no `context`; a text-conditioned sampler has nowhere to put the prompt")
        })?;
        let prompt = lane(&reading, context_stream.as_deref())?;
        let latents = input(&image, "latents").cloned().expect("found by it");
        if latents.axis != model::InputAxis::Voxel {
            return Err(format!("reading `{reading}` reads `latents` per {:?}; this sampler holds the VAE's latent, on the voxel axis", latents.axis).into());
        }
        let context = input(&prompt, "context").cloned().expect("found by it");
        if input(&image, "timestep").is_none() {
            return Err(format!("reading `{reading}` reads no `timestep`").into());
        }
        return Ok(Denoise {
            velocity_width: image.readout.as_ref().map_or(0, |r| r.width),
            guidance: input(&image, "guidance").is_some(),
            reading,
            latent_stream,
            context_stream,
            latents,
            context,
        });
    }
    Err("this model declares no token-less reading of `latents` with a velocity readout, so nothing here denoises".into())
}

/// The reading that turns the encoder's `from`-wide rows into the `to`-wide
/// rows `context` takes, when they differ.
fn bridge_reading(from: u32, to: u32) -> Result<Bridge> {
    for reading in values("reading") {
        for stream in streams() {
            let Ok(lane) = lane(&reading, stream.as_deref()) else {
                continue;
            };
            let hidden = lane.readout.as_ref().is_some_and(|r| r.kind == model::ReadoutKind::Hidden && r.width == to);
            if lane.embeds || lane.reads_kv || !hidden {
                continue;
            }
            if let [only] = &lane.inputs[..]
                && only.axis == model::InputAxis::Row
                && only.width == from
            {
                return Ok(Bridge {
                    input: only.name.clone(),
                    reading,
                    stream,
                });
            }
        }
    }
    Err(format!(
        "the text reading hands back {from}-wide rows, the denoise `context` input takes {to}, and \
         no reading turns one into the other"
    )
    .into())
}

/// `rows` rows of `width` out of `h`'s first `len`, the rest zero: the
/// context pad a model states with `input-decl.rows`, done on the device.
fn pad_rows(h: &Tensor, len: u32, rows: u32, width: u32) -> Tensor {
    if rows == len {
        return h.clone();
    }
    let i = iota(rows);
    let src = min_elem(&i, len - 1);
    let keep = cast(lt(&i, len), dtype::f32);
    &gather(h, &src) * &broadcast(reshape(&keep, [rows, 1]), [rows, width])
}

/// All the passes of one job, kept alive until its picture is out.
#[derive(Default)]
struct Held {
    passes: Vec<ForwardPass>,
    sets: Vec<WorkingSet>,
}

/// Encode `prompt` on `pipe` into a device channel the denoise `context`
/// input takes. `then` runs at the end of the last pass's epilogue.
fn encode(
    pipe: &Pipeline,
    held: &mut Held,
    text: &Text,
    bridge: Option<&Bridge>,
    context: &model::InputDecl,
    prompt: &str,
    then: Option<Box<dyn Fn()>>,
) -> Result<Channel> {
    let mut ids = inferlet::chat::first_user(prompt);
    ids.extend(inferlet::chat::cue());
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    if len == 0 {
        return Err("the prompt tokenizes to nothing".into());
    }
    let cap = max_readout_rows(model::ReadoutKind::Hidden);
    if len as usize > cap {
        return Err(format!("the prompt is {len} tokens and one pass reads out {cap} hidden rows").into());
    }
    let rows = context.rows.unwrap_or(len);
    if len > rows {
        return Err(format!("input `{}` takes {rows} rows and the prompt is {len} tokens", context.name).into());
    }
    let ids: Vec<i32> = ids.iter().map(|&id| i32::try_from(id).unwrap_or(0)).collect();
    let pass = ForwardPass::new();
    pass.switch(0, "reading", &text.reading)?;
    pass.embed(&Channel::from(ids), &Channel::from([0u32, len]))?;
    pass.readout(&Channel::from_iter(0..len))?;
    if text.reads_kv {
        let ws = WorkingSet::new();
        pass.state(&[&ws], StateGeometry::default())?;
        held.sets.push(ws);
    }
    let width = text.width;
    let encoded = Channel::new([rows, width], dtype::f32).queued();
    let (here, later) = match bridge {
        None => (then, None),
        Some(_) => (None, then),
    };
    pass.epilogue(move || {
        encoded.put(pad_rows(&intrinsics::hidden(0, len, width), len, rows, width));
        if let Some(then) = &here {
            then();
        }
    });
    pass.submit(pipe)?;
    held.passes.push(pass);
    let Some(bridge) = bridge else {
        return Ok(encoded);
    };
    let to = context.width;
    let pass = ForwardPass::new();
    set_switches(&pass, 0, &bridge.reading, bridge.stream.as_deref())?;
    pass.input(0, &bridge.input, &encoded)?;
    let bridged = Channel::new([rows, to], dtype::f32).queued();
    pass.epilogue(move || {
        bridged.put(intrinsics::hidden(0, rows, to));
        if let Some(then) = &later {
            then();
        }
    });
    pass.submit(pipe)?;
    held.passes.push(pass);
    Ok(bridged)
}

/// One weight per latent cell: the fraction of its `cell x cell` pixels the
/// rectangle `(x0, y0, x1, y1)` or the gray mask covers.
fn mask_cells(h: u32, w: u32, cell: u32, rect: Option<(u32, u32, u32, u32)>, gray: Option<&[f32]>) -> Vec<f32> {
    let width = (w * cell) as usize;
    let mut out = vec![0f32; (h * w) as usize];
    for a in 0..h {
        for b in 0..w {
            let mut sum = 0f32;
            for y in a * cell..(a + 1) * cell {
                for x in b * cell..(b + 1) * cell {
                    sum += match (rect, gray) {
                        (Some((x0, y0, x1, y1)), _) => f32::from(u8::from(x >= x0 && x < x1 && y >= y0 && y < y1)),
                        (None, Some(px)) => px[y as usize * width + x as usize],
                        (None, None) => 0.0,
                    };
                }
            }
            out[(a * w + b) as usize] = sum / (cell * cell) as f32;
        }
    }
    out
}

fn parse_rect(text: &str) -> Result<(u32, u32, u32, u32)> {
    let parts: Vec<u32> = text
        .split(',')
        .map(|p| p.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|why| format!("`mask_rect` is `x0,y0,x1,y1` in pixels: {why}"))?;
    if parts.len() != 4 || parts[0] >= parts[2] || parts[1] >= parts[3] {
        return Err(format!("`mask_rect` is `x0,y0,x1,y1` with x0 < x1 and y0 < y1, not `{text}`").into());
    }
    Ok((parts[0], parts[1], parts[2], parts[3]))
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
        let six = code(c).ok_or_else(|| format!("byte {i} is not base64"))?;
        acc = (acc << 6) | six;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Ok(bytes)
}

/// The one voxel input a VAE reading's lane reads, and its stream.
fn vae_lane(reading: &str) -> Result<(Option<String>, model::InputDecl, u32)> {
    for stream in streams() {
        let Ok(lane) = lane(reading, stream.as_deref()) else {
            continue;
        };
        if let ([input], Some(readout)) = (&lane.inputs[..], &lane.readout)
            && input.axis == model::InputAxis::Voxel
        {
            return Ok((stream, input.clone(), readout.width));
        }
    }
    Err(format!("reading `{reading}` reads no one clip on the voxel axis").into())
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let prompt = input
        .prompt
        .clone()
        .filter(|p| !p.trim().is_empty())
        .ok_or("pass `prompt`: this program draws what it is told to")?;
    let text = text_reading()?;
    let denoise = denoise_reading()?;
    let bridge = (text.width != denoise.context.width)
        .then(|| bridge_reading(text.width, denoise.context.width))
        .transpose()?;
    let readings = values("reading");
    let decode = readings
        .iter()
        .find(|r| *r == "vae.decode")
        .cloned()
        .ok_or("this model declares no `vae.decode` reading, so its latent has no way out as pixels")?;

    // ---- the latent, the VAE's own ----------------------------------------
    let space = model::latent().ok_or("this model states no latent space")?;
    if space.channels != denoise.latents.width {
        return Err(format!(
            "the latent space has {} channels and `latents` reads {}",
            space.channels, denoise.latents.width
        )
        .into());
    }
    let sc = space.spatial_compression.max(1);
    let height = (input.height.unwrap_or(1024) / sc).max(1) * sc;
    let width = (input.width.unwrap_or(1024) / sc).max(1) * sc;
    let (lh, lw, c) = (height / sc, width / sc, space.channels);
    let lat = [lh, lw, c];
    let cells = lh * lw;

    // ---- the schedule -----------------------------------------------------
    let fact = model::schedule().ok_or("this model states no schedule; nothing here denoises")?;
    let steps = match input.steps.filter(|s| *s > 0) {
        Some(steps) => steps,
        None if !fact.pinned_sigmas.is_empty() => u32::try_from(fact.pinned_sigmas.len()).unwrap_or(8),
        None => 8,
    };
    let full = FlowMatchEuler::from_schedule(&fact, steps, None)?;
    let init_b64 = match (&input.init_image, input.pieces("init_image_")) {
        (Some(_), pieces) if !pieces.is_empty() => {
            return Err("pass `init_image` whole or as `init_image_<n>` pieces, not both".into());
        }
        (Some(b64), _) => Some(b64.clone()),
        (None, pieces) if !pieces.is_empty() => Some(pieces),
        _ => None,
    };
    let strength = input.strength.unwrap_or(0.6);
    if init_b64.is_some() && !(0.0 < strength && strength <= 1.0) {
        return Err(format!("`strength` is {strength}; it is a fraction in (0, 1]").into());
    }
    let sched = match &init_b64 {
        None => full,
        Some(_) => {
            let run = ((steps as f32 * strength).round() as u32).clamp(1, steps);
            FlowMatchEuler::from_sigmas(full.sigmas[(steps - run) as usize..].to_vec(), full.train_steps)
        }
    };
    let sigma0 = sched.sigmas[0];
    let mut ts = sched.timesteps();
    ts.push(0.0);

    // ---- guidance: an input, or a second attention group ------------------
    let guidance = input.guidance.unwrap_or(1.0);
    let negative = input.negative_prompt.clone().filter(|p| !p.trim().is_empty());
    if negative.is_some() && denoise.guidance {
        return Err(format!(
            "reading `{}` reads a `guidance` input, so this row is guidance-distilled and a \
             negative prompt has no lane to ride; pass `guidance` alone",
            denoise.reading
        )
        .into());
    }
    let cfg = negative.is_some() && guidance > 1.0;

    // ---- the mask -----------------------------------------------------------
    let mask_b64 = input.pieces("mask_image_");
    let mask: Option<Vec<f32>> = if input.mask_rect.is_none() && mask_b64.is_empty() {
        None
    } else {
        if init_b64.is_none() {
            return Err("a mask says what to repaint; pass `init_image` for the picture it is cut from".into());
        }
        let rect = input.mask_rect.as_deref().map(parse_rect).transpose()?;
        let gray = match rect {
            Some(_) => None,
            None => {
                let mask = inferlet::frames::Frames::decode(&b64_decode(&mask_b64)?)
                    .map_err(|why| format!("the mask image does not decode: {why}"))?;
                if mask.width() != width || mask.height() != height {
                    return Err(format!("the mask is {}x{} and this run is {width}x{height}", mask.width(), mask.height()).into());
                }
                let rgb = mask
                    .encode(inferlet::frames::ImageFormat::RawRgb8)
                    .map_err(|why| format!("frames.encode(raw-rgb8): {why}"))?;
                Some(rgb.chunks(3).map(|px| f32::from(px[0]) / 255.0).collect::<Vec<f32>>())
            }
        };
        Some(mask_cells(lh, lw, sc, rect, gray.as_deref()))
    };

    // ---- the loop's state, all on the device -------------------------------
    let pipe = Pipeline::new();
    let mut held = Held::default();
    let x = Channel::new(lat, dtype::f32).queued().named("x");
    let k = Channel::new([1], dtype::u32).queued().named("k");
    let t = Channel::new([1], dtype::f32).queued().named("t");
    let ts = Channel::from(ts).named("ts");
    let sigmas = Channel::from(sched.sigmas.clone()).named("sigmas");
    let seed = input.seed.unwrap_or(0);
    let draw = Channel::from(rng_state(seed)).named("draw");
    // The picture an inpainting step keeps, on the device.
    let known = mask.as_ref().map(|_| Channel::new(lat, dtype::f32).queued().named("known"));
    // `x` from `start` (the encoded picture) or from noise alone, and the clock at 0.
    let seed_loop = move |start: Option<Tensor>| {
        let eps = normal(&draw.read(), lat);
        let x0 = match start {
            None => eps,
            Some(start) => {
                if let Some(known) = &known {
                    known.put(&start);
                }
                let shape = start.shape();
                &(&start * &broadcast(Tensor::constant([1.0 - sigma0]), shape))
                    + &(&eps * &broadcast(Tensor::constant([sigma0]), shape))
            }
        };
        x.put(&x0);
        k.put(Tensor::constant([0u32]));
        t.put(at(&ts.read(), &Tensor::constant([0u32])));
    };

    // ---- the prompt, and the picture, encoded on the device ----------------
    let seed_at_encode = init_b64.is_none();
    let first: Option<Box<dyn Fn()>> = seed_at_encode.then(|| Box::new(move || seed_loop(None)) as Box<dyn Fn()>);
    let context = encode(&pipe, &mut held, &text, bridge.as_ref(), &denoise.context, &prompt, first)?;
    let uncond = match &negative {
        Some(neg) if cfg => Some(encode(&pipe, &mut held, &text, bridge.as_ref(), &denoise.context, neg, None)?),
        _ => None,
    };
    if let Some(b64) = &init_b64 {
        let reading = readings
            .iter()
            .find(|r| *r == "vae.encode")
            .ok_or("this model declares no `vae.encode` reading, so a picture has no door in")?;
        let picture = inferlet::frames::Frames::decode(&b64_decode(b64)?)
            .map_err(|why| format!("the init image does not decode: {why}"))?;
        if picture.width() != width || picture.height() != height {
            return Err(format!(
                "the init image is {}x{} and this run is {width}x{height}; resize it first",
                picture.width(),
                picture.height()
            )
            .into());
        }
        let (stream, pixels_in, latent_width) = vae_lane(reading)?;
        if latent_width != c {
            return Err(format!("`{reading}` reads out {latent_width}-wide cells, the latent has {c} channels").into());
        }
        let pass = ForwardPass::new();
        set_switches(&pass, 0, reading, stream.as_deref())?;
        let clip = Channel::seeded([height, width, pixels_in.width], dtype::f32).named("pixels_in");
        pass.input(0, &pixels_in.name, &clip)?;
        pass.epilogue(move || {
            seed_loop(Some(reshape(intrinsics::pixels(0, cells, c), lat)));
        });
        clip.set_frames(&picture)?;
        pass.submit(&pipe)?;
        held.passes.push(pass);
    }

    // ---- the denoise pass: one a step --------------------------------------
    let pass = ForwardPass::new();
    let guidance_ch = Channel::from([guidance]).named("guidance");
    let mut lanes = 0u32;
    for (group, ctx) in [Some(&context), uncond.as_ref()].into_iter().flatten().enumerate() {
        let group = group as u32;
        for (stream, latent) in [(&denoise.latent_stream, true), (&denoise.context_stream, false)] {
            let at = lanes;
            lanes += 1;
            set_switches(&pass, at, &denoise.reading, stream.as_deref())?;
            pass.group(at, group)?;
            for decl in lane(&denoise.reading, stream.as_deref())?.inputs {
                let ch = match decl.name.as_str() {
                    "latents" if latent => &x,
                    "context" if !latent => ctx,
                    "timestep" => &t,
                    "guidance" => &guidance_ch,
                    other => {
                        return Err(format!(
                            "reading `{}` reads `{other}` on lane {at}; this sampler binds latents, \
                             timestep, guidance and context",
                            denoise.reading
                        )
                        .into());
                    }
                };
                pass.input(at, &decl.name, ch)?;
            }
        }
    }
    let (n, vw) = (cells, denoise.velocity_width);
    let mask_ch = mask.as_ref().map(|m| Channel::from_shaped([cells, 1], m.as_slice()).named("mask"));
    let noise = Channel::from(rng_state(seed ^ 0x5eed)).named("noise");
    pass.epilogue(move || {
        let step = k.take();
        let v = if cfg {
            cfg_combine(&intrinsics::velocity(0, n, vw), &intrinsics::velocity(2, n, vw), &Tensor::constant([guidance]))
        } else {
            intrinsics::velocity(0, n, vw)
        };
        let sig = sigmas.read();
        let next = &step + 1u32;
        let dt = &at(&sig, &next) - &at(&sig, &step);
        let mut stepped = euler_step(&reshape(x.take(), [n, c]), &v, &dt);
        if let (Some(m), Some(known)) = (&mask_ch, &known) {
            // Inpainting: the kept cells are the picture noised to the sigma
            // this step lands at, the rest the step.
            let state = noise.take();
            let shape = stepped.shape();
            let to = broadcast(reshape(at(&sig, &next), [1, 1]), shape);
            let one = broadcast(Tensor::constant([1.0f32]), shape);
            let eps = normal(&state, [n, c]);
            let kept = &(&reshape(known.read(), [n, c]) * &(&one - &to)) + &(&eps * &to);
            let m = broadcast(m.read(), shape);
            stepped = &(&m * &stepped) + &(&(&one - &m) * &kept);
            noise.put(&state + &Tensor::constant([0u32, 1u32]));
        }
        x.put(reshape(&stepped, lat));
        t.take();
        t.put(at(&ts.read(), &next));
        k.put(&next);
    });
    for fire in 0..sched.steps() {
        pass.submit(&pipe).with_context(|| format!("denoise fire {fire}"))?;
    }
    // Every pass so far is closed before the decode opens: a float pass
    // holds a seat until it closes, and the decode needs it (until B3).
    drop(pass);
    drop(held);

    // ---- the decode, down the same pipeline ---------------------------------
    let (stream, latent_in, rgb) = vae_lane(&decode)?;
    if latent_in.width != c {
        return Err(format!("`{decode}` reads {}-wide cells, the latent has {c} channels", latent_in.width).into());
    }
    let pass = ForwardPass::new();
    set_switches(&pass, 0, &decode, stream.as_deref())?;
    pass.input(0, &latent_in.name, &x)?;
    let px = Channel::new([height * width, rgb], dtype::f32).named("pixels");
    let pixels = height * width;
    pass.epilogue(move || {
        px.put(intrinsics::pixels(0, pixels, rgb));
    });
    pass.submit(&pipe).context("the decode")?;
    let picture = px.take_frames(width, height, 1, 0.0)?;
    drop(pass);
    pipe.close();
    let name = input.out.unwrap_or_else(|| "image".to_string());
    let file = format!("{}.png", name.trim_end_matches(".png"));
    inferlet::session::send_frames(&picture, inferlet::frames::ImageFormat::Png, &file)
        .map_err(|why| format!("session.send-frames: {why}"))?;

    Ok(Output {
        model: model::name(),
        architecture: model::architecture(),
        reading: denoise.reading.clone(),
        prompt,
        width,
        height,
        latent_h: lh,
        latent_w: lw,
        latent_channels: c,
        spatial_compression: sc,
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
        sigmas: sched.sigmas.clone(),
        context_rows: context.shape().dims()[0],
        context_readings: std::iter::once(text.reading.clone()).chain(bridge.map(|b| b.reading)).collect(),
        decode_reading: decode,
        file,
        mask_cells: mask.map(|m| m.iter().filter(|w| **w > 0.0).count() as u32),
    })
}
