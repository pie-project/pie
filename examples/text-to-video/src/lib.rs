//! **THE MODEL-AGNOSTIC `text-to-video` INFERLET**: a prompt in, a clip
//! out, with nothing about any family spelled here. The sibling of
//! `text-to-image`, and built the same way: the guest knows MODEL FACTS (the
//! readings, what each lane of them binds, the VAE's latent space, the
//! schedule) and never ARCHITECTURE FACTS (patches, positions, padding).
//!
//! The latent it holds is the VAE's own `[t, h, w, C]` volume. Every pass of
//! the denoise goes down one pipeline: the text encoder, whose rows stay on
//! the device (padded there to the rows the `context` input states, Wan's
//! 512), then `steps` denoise fires. The encoder's epilogue draws the latent
//! from a keyed `normal` on the device (or noises the encoded init clip) and
//! seeds the loop's clock; each denoise epilogue integrates one Euler step,
//! guided `u + s (c - u)` when a negative prompt adds a second attention
//! group to the pass. The last fire also lands the latent on a host-read
//! channel: the decode below takes it frame by frame.
//!
//! # THE DECODE
//!
//! A causal video VAE decodes one latent frame a fire on one working set:
//! the first fire on it lands one frame, every later one
//! `temporal-compression` (the model's own branch on its working set being
//! fresh). So `T` latent frames are `1 + tc (T - 1)` output
//! frames, and a `frames` off that lattice is refused with the two nearest.
//! The decode fires are a sequence (the per-conv frame caches); they go down
//! one pipeline, in order.
//!
//! # THE PROMPT
//!
//! `--prompt` renders the bound model's template and tokenizes with its
//! vocabulary. Wan 2.2's umT5 tokenizer is a Unigram model `crates/tokenizer`
//! cannot spell, so `--prompt-ids` hands in ids tokenized elsewhere; the
//! encoder still runs inside pie, and the report says which door was used.

use inferlet::frames::{Frames, ImageFormat};
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    prompt: Option<String>,
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
    /// VIDEO2VIDEO: how much of the schedule to run over the encoded clip,
    /// in `(0, 1]`. `1.0` is every step, which is text-to-video from noise:
    /// `(1 - 1)*x0 + 1*eps` IS the keyed draw, so the init washes out
    /// entirely and the run is the one with no init at all. `0.6` keeps the
    /// first 40% of the trajectory as the clip's own. Only read when an init
    /// clip is given.
    ///
    /// That identity is exact in the arithmetic and only NEARLY exact in the
    /// file, because a `pie run` of this row is not bit-reproducible: two
    /// identical runs land clips a few ten-thousandths apart. What the gate
    /// claims is the shape that survives it — see `gates.py`'s
    /// `v2v_claims`.
    #[serde(default)]
    strength: Option<f32>,
    /// THE INIT CLIP, frame by frame: `init_frame_<f>_<n>` is frame `f`'s
    /// base64-encoded still (PNG, JPEG, GIF, WebP), piece `n`. Two indices
    /// because a clip is a LIST of pictures and one argv argument is capped
    /// at `MAX_ARG_STRLEN` well below a single still. A frame small enough
    /// to travel whole may be `init_frame_<f>` with no piece index.
    ///
    /// The clip must be this run's own `width x height x frames`: a guest
    /// has no resampler, in space or in time.
    #[serde(flatten)]
    rest: std::collections::BTreeMap<String, inferlet::serde_json::Value>,
}

impl Input {
    /// **THE INIT CLIP'S FRAMES, IN FRAME ORDER**, each frame's pieces
    /// concatenated in piece order. The `f`s are taken in sorted order and
    /// renumbered densely: what matters is which frame is first, not the
    /// label the caller gave it.
    fn init_frames(&self) -> Vec<String> {
        let mut found: std::collections::BTreeMap<u64, std::collections::BTreeMap<u64, &str>> =
            std::collections::BTreeMap::new();
        for (name, value) in &self.rest {
            let Some(rest) = name.strip_prefix("init_frame_") else {
                continue;
            };
            // `<f>_<n>`, or a bare `<f>` for a frame that travelled whole.
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
    /// `"prompt"` (the bound tokenizer) or `"prompt-ids"` (handed in).
    prompt_source: String,
    width: u32,
    height: u32,
    frames: u32,
    fps: f32,
    /// The VAE latent this run denoised, `[latent_frames, latent_h, latent_w, channels]`.
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
    /// The rows the context lane carries, and how many are the prompt's.
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
    state: bool,
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
        if video.embeds
            || video.readout.as_ref().map(|r| r.kind) != Some(model::ReadoutKind::Velocity)
        {
            continue;
        }
        if video.state {
            return Err(
                format!("reading `{reading}` binds a working set; a denoise pass binds none").into(),
            );
        }
        let context_stream = stream_reading(&reading, "context")
            .ok_or_else(|| format!("reading `{reading}` reads no `context` for the prompt rows"))?;
        let prompt = lane(&reading, context_stream.as_deref())?;
        let latents = input(&video, "latents").cloned().expect("found by it");
        if latents.axis != model::InputAxis::Voxel {
            return Err(format!("reading `{reading}` reads `latents` per {:?}; this sampler holds the VAE's latent, on the voxel axis", latents.axis).into());
        }
        if input(&video, "timestep").is_none() {
            return Err(format!("reading `{reading}` reads no `timestep`").into());
        }
        return Ok(Denoise {
            context: input(&prompt, "context").cloned().expect("found by it"),
            velocity_width: video.readout.as_ref().map_or(0, |r| r.width),
            guidance: input(&video, "guidance").is_some(),
            reading,
            latent_stream,
            context_stream,
            latents,
        });
    }
    Err("this model declares no token-less reading of `latents` with a velocity readout".into())
}

/// The rows a lane of `len` rows of its own carries under the model's
/// `input-decl.rows`, `None` when they do not fit.
fn padded(rows: Option<model::RowCount>, len: u32) -> Option<u32> {
    match rows {
        None => Some(len),
        Some(model::RowCount::Exact(n)) => (len <= n).then_some(n),
        Some(model::RowCount::Multiple(n)) => len.checked_next_multiple_of(n),
    }
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

/// All the passes of one job, kept alive until the denoise has landed.
#[derive(Default)]
struct Held {
    passes: Vec<ForwardPass>,
    sets: Vec<WorkingSet>,
}

/// Encode `ids` on `pipe` into a device channel the denoise `context`
/// input takes; `then` runs at the end of the pass's epilogue.
fn encode(
    pipe: &Pipeline,
    held: &mut Held,
    text: &Text,
    context: &model::InputDecl,
    ids: &[u32],
    then: Option<Box<dyn Fn()>>,
) -> Result<(Channel, u32)> {
    if text.width != context.width {
        return Err(format!(
            "the text reading hands back {}-wide rows and input `{}` takes {}",
            text.width, context.name, context.width
        )
        .into());
    }
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    if len == 0 {
        return Err("the prompt tokenizes to nothing".into());
    }
    let cap = max_readout_rows(model::ReadoutKind::Hidden);
    if len as usize > cap {
        return Err(
            format!("the prompt is {len} tokens and one pass reads out {cap} hidden rows").into(),
        );
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
            &intrinsics::hidden(0, len, width),
            len,
            rows,
            width,
        ));
        if let Some(then) = &then {
            then();
        }
    });
    pass.submit(pipe)?;
    held.passes.push(pass);
    Ok((encoded, len))
}

/// What `reading`'s lane on `stream` reads out.
fn readout(reading: &str, stream: &str) -> Option<model::Readout> {
    model::lane_of(&[("reading", reading), ("stream", stream)])
        .ok()?
        .readout
}

/// How wide a row `reading`'s lane on `stream` reads out.
fn readout_width(reading: &str, stream: &str) -> Result<u32> {
    readout(reading, stream)
        .map(|readout| readout.width)
        .ok_or_else(|| format!("reading `{reading}` reads nothing out on its `{stream}` lane"))
}

/// Standard base64 (`+/`, `=` padded) into the bytes it spells. The one
/// thing that travels this way is an init clip's frames: argv is text, an
/// encoded still is not, and a scratch-file door would make the clip a path
/// rather than a value — and the sandbox's `/scratch` is per PROCESS, so
/// nothing outside could put a file there anyway.
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

/// The one input a VAE arm's lane reads: a clip on the voxel axis.
fn voxel_input(reading: &str) -> Result<model::InputDecl> {
    match &model::lane_of(&[("reading", reading), ("stream", "video")])?.inputs[..] {
        [input] if input.axis == model::InputAxis::Voxel => Ok(input.clone()),
        other => Err(format!(
            "reading `{reading}` reads {} input(s), and a VAE arm reads one clip on the voxel axis",
            other.len()
        )
        .into()),
    }
}

/// Fire ONE encode arm over ONE PIXEL CHUNK and take the latent frame it
/// lands: the mirror of [`decode_frame`], and the door video2video comes in
/// through.
///
/// The chunk goes in as the input's CHANNEL, seeded straight from a host-held
/// `frames` handle (`Channel::set_frames`, which is a `put` and so must
/// happen after the pass has bound the input — a channel has no ring before
/// that). Its shape IS the clip's box, `[t, H, W, 3]`, so the geometry the
/// voxel axis needs travels with the numbers.
///
/// The answer comes back on the same `pixels` seam a decode uses. For an
/// encoder that seam carries ONE LATENT FRAME — the posterior MEAN, already
/// normalised into the denoiser's own space — whatever the number of pixel
/// frames that went in.
async fn encode_chunk(
    reading: &str,
    chunk: &Frames,
    chunk_frames: u32,
    pixel_h: u32,
    pixel_w: u32,
    clip_h: u32,
    clip_w: u32,
    state: &WorkingSet,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let input = voxel_input(reading)?;
    let rows = clip_h * clip_w;
    let width = readout_width(reading, "video")?;
    let pass = ForwardPass::new();
    pass.switch(0, "reading", reading)?;
    pass.switch(0, "stream", "video")?;
    pass.state(&[state], StateGeometry::default())?;
    // Empty, and SEEDED from the chunk below: the pixels are the host's, and
    // the eta trace tracks readiness STATICALLY — a channel the host seeds
    // has to be declared `seeded`, or the trace refuses it as consumed but
    // never produced.
    let cell = Channel::seeded([chunk_frames, pixel_h, pixel_w, input.width], dtype::f32)
        .named("vae_pixels_in");
    pass.input(0, &input.name, &cell)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_latent_out");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(0, rows, width));
    });
    // THE PIXELS, seeded straight from the host's handle, after the bind.
    cell.set_frames(chunk)?;
    pass.submit(pipe)
        .with_context(|| format!("the `{reading}` fire"))?;
    let landed = out.take_host::<Vec<f32>>().await?;
    let want = (rows * width) as usize;
    if landed.len() != want {
        return Err(format!(
            "reading `{reading}` landed {} numbers over a {chunk_frames}-frame chunk and one \
             latent frame is {clip_h}x{clip_w}x{width} = {want}",
            landed.len()
        )
        .into());
    }
    Ok(landed)
}

/// Fire ONE decode arm over ONE latent frame and take its pixels back.
///
/// The clip's box IS the channel's shape (`[h, w, C]` for one frame), so
/// the geometry the voxel axis needs travels with the numbers. `frames` is
/// how many output frames this arm lands, which the caller knows from the
/// family's `temporal-compression` and which arm this is.
async fn decode_frame(
    reading: &str,
    clip: &[f32],
    clip_h: u32,
    clip_w: u32,
    frames: u32,
    pixel_h: u32,
    pixel_w: u32,
    state: &WorkingSet,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let input = voxel_input(reading)?;
    let want = (clip_h as usize) * (clip_w as usize) * (input.width as usize);
    if clip.len() != want {
        return Err(format!(
            "one latent frame is {} numbers and input `{}` wants {clip_h}x{clip_w}x{} = {want}",
            clip.len(),
            input.name,
            input.width
        )
        .into());
    }
    let rows = frames * pixel_h * pixel_w;
    let width = readout_width(reading, "video")?;
    let pass = ForwardPass::new();
    pass.switch(0, "reading", reading)?;
    pass.switch(0, "stream", "video")?;
    pass.state(&[state], StateGeometry::default())?;
    let cell =
        Channel::from_shaped([clip_h, clip_w, input.width], clip.to_vec()).named("vae_latent");
    pass.input(0, &input.name, &cell)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_pixels");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(0, rows, width));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{reading}` fire"))?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

/// `[-1, 1]` f32 in `(t, h, w, 3)` order to interleaved RGB8, clamped.
fn rgb8(pixels: &[f32]) -> Vec<u8> {
    pixels
        .iter()
        .map(|v| (((v + 1.0) * 127.5).clamp(0.0, 255.0)) as u8)
        .collect()
}

/// The prompt as ids, through the BOUND MODEL's own template and
/// tokenizer: one user turn plus the generation cue, which is what every
/// instruct-encoder pipeline renders before it tokenizes. Only reached on
/// the `--prompt` door; `--prompt-ids` skips both.
fn template_ids(prompt: &str) -> Vec<u32> {
    let mut ids = inferlet::chat::first_user(prompt);
    ids.extend(inferlet::chat::cue());
    ids
}

/// `"1,2,3"` to ids. Refuses a piece that is not a number, by name.
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
    let decode = readings
        .iter()
        .find(|r| *r == "vae.decode")
        .cloned()
        .ok_or(
            "this model declares no `vae.decode` reading, so its latent has no way out as frames",
        )?;
    let encode_reading = readings.iter().find(|r| *r == "vae.encode").cloned();

    // ---- the latent volume, the VAE's own -----------------------------------
    let space = model::latent().ok_or("this model states no latent space")?;
    let (sc, tc, c) = (
        space.spatial_compression.max(1),
        space.temporal_compression.max(1),
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
        )
        .into());
    }
    let height = (input.height.unwrap_or(480) / sc).max(1) * sc;
    let width = (input.width.unwrap_or(832) / sc).max(1) * sc;
    let (lh, lw) = (height / sc, width / sc);
    let frames = input.frames.unwrap_or(17).max(1);
    // A causal VAE's first fire on its working set lands one frame and every
    // later one `tc`: `1 + tc (T - 1)` frames and no other count.
    if (frames - 1) % tc != 0 {
        let below = 1 + tc * ((frames - 1) / tc);
        return Err(format!(
            "this family's VAE lands 1 + {tc}*(T - 1) frames, so {frames} is not a clip it \
             has: the nearest are {below} and {}",
            below + tc
        )
        .into());
    }
    let lt = (frames - 1) / tc + 1;
    let lat = [lt, lh, lw, c];
    let cells = lt * lh * lw;
    let plane = (lh * lw * c) as usize;

    // ---- the schedule -------------------------------------------------------
    let fact = model::schedule().ok_or("this model states no schedule; nothing here denoises")?;
    let steps = match input.steps.filter(|s| *s > 0) {
        Some(steps) => steps,
        None if !fact.pinned_sigmas.is_empty() => {
            u32::try_from(fact.pinned_sigmas.len()).unwrap_or(8)
        }
        None => 8,
    };
    // No dynamic shift: a video row's stated shift is the shift itself.
    let full = FlowMatchEuler::from_schedule(&fact, steps, None)?;
    let init_b64 = input.init_frames();
    let has_init = !init_b64.is_empty();
    let strength = input.strength.unwrap_or(0.6);
    if has_init && !(0.0 < strength && strength <= 1.0) {
        return Err(format!("`strength` is {strength}; it is a fraction in (0, 1]").into());
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

    // ---- the prompt's ids, through whichever door the caller opened -------
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
        )
        .into());
    }
    let cfg = negative.is_some() && guidance > 1.0;

    // ---- video2video: the clip, encoded chunk by chunk ----------------------
    let pipe = Pipeline::new();
    let init: Option<Channel> = if !has_init {
        None
    } else {
        let reading = encode_reading.as_ref().ok_or(
            "this model declares no `vae.encode` reading, so an init clip has no door to come in through",
        )?;
        if init_b64.len() as u32 != frames {
            return Err(format!(
                "the init clip is {} frames and this run is {frames}",
                init_b64.len()
            )
            .into());
        }
        let mut rgb: Vec<Vec<u8>> = Vec::with_capacity(init_b64.len());
        for (f, b64) in init_b64.iter().enumerate() {
            let picture = Frames::decode(&b64_decode(b64)?)
                .map_err(|why| format!("init frame {f} does not decode: {why}"))?;
            if picture.width() != width || picture.height() != height || picture.count() != 1 {
                return Err(format!(
                    "init frame {f} is {}x{}x{} and this run is one {width}x{height} still a frame",
                    picture.width(),
                    picture.height(),
                    picture.count()
                )
                .into());
            }
            rgb.push(
                picture
                    .encode(ImageFormat::RawRgb8)
                    .map_err(|why| format!("init frame {f} does not read back as RGB8: {why}"))?,
            );
        }
        let mut chunks: Vec<u32> = vec![1];
        while chunks.iter().sum::<u32>() < frames {
            chunks.push(tc);
        }
        if chunks.iter().sum::<u32>() != frames || chunks.len() as u32 != lt {
            return Err(format!(
                "a {frames}-frame clip does not chunk into {lt} encode fires at tc = {tc}"
            )
            .into());
        }
        let mut clip: Vec<f32> = Vec::with_capacity(lt as usize * plane);
        // One working set for the whole clip: the conv caches are its state.
        let enc_state = WorkingSet::new();
        let mut at = 0usize;
        for (k, take) in chunks.iter().copied().enumerate() {
            let mut payload: Vec<u8> =
                Vec::with_capacity(take as usize * (height * width * 3) as usize);
            for f in at..at + take as usize {
                payload.extend_from_slice(&rgb[f]);
            }
            at += take as usize;
            let handle = Frames::from_rgb8(&payload, width, height, take, fps)
                .map_err(|why| format!("encode chunk {k}: frames.from-rgb8: {why}"))?;
            let landed = encode_chunk(
                reading, &handle, take, height, width, lh, lw, &enc_state, &pipe,
            )
            .await
            .with_context(|| format!("encoding chunk {k}"))?;
            clip.extend_from_slice(&landed);
        }
        Some(Channel::from_shaped(lat, clip).named("init"))
    };

    // ---- the loop's state, all on the device ---------------------------------
    let mut held = Held::default();
    let x = Channel::new(lat, dtype::f32).queued().named("x");
    let k = Channel::new([1], dtype::u32).queued().named("k");
    let t = Channel::new([1], dtype::f32).queued().named("t");
    let ts = Channel::from(ts).named("ts");
    let sigmas = Channel::from(sched.sigmas.clone()).named("sigmas");
    let seed = input.seed.unwrap_or(0);
    let draw = Channel::from(rng_state(seed)).named("draw");
    let seed_loop = move || {
        let eps = normal(&draw.read(), lat);
        let x0 = match &init {
            None => eps,
            Some(init) => {
                &(&init.read() * &broadcast(Tensor::constant([1.0 - sigma0]), lat))
                    + &(&eps * &broadcast(Tensor::constant([sigma0]), lat))
            }
        };
        x.put(&x0);
        k.put(Tensor::constant([0u32]));
        t.put(at(&ts.read(), &Tensor::constant([0u32])));
    };

    // ---- the prompt, encoded on the device -----------------------------------
    let (context, prompt_rows) = encode(
        &pipe,
        &mut held,
        &text,
        &denoise.context,
        &ids,
        Some(Box::new(seed_loop)),
    )?;
    let uncond = match &negative {
        Some(neg) if cfg => Some(encode(&pipe, &mut held, &text, &denoise.context, neg, None)?.0),
        _ => None,
    };

    // ---- the denoise passes: every fire the same step, the last one also
    // landing the latent for the decode ---------------------------------------
    let guidance_ch = Channel::from([guidance]).named("guidance");
    let out = Channel::new(lat, dtype::f32).named("out");
    let (n, vw) = (cells, denoise.velocity_width);
    let denoise_pass = |land: bool| -> Result<ForwardPass> {
        let pass = ForwardPass::new();
        let mut lanes = 0u32;
        for (group, ctx) in [Some(&context), uncond.as_ref()]
            .into_iter()
            .flatten()
            .enumerate()
        {
            for (stream, latent) in [
                (&denoise.latent_stream, true),
                (&denoise.context_stream, false),
            ] {
                let at = lanes;
                lanes += 1;
                set_switches(&pass, at, &denoise.reading, stream.as_deref())?;
                pass.group(at, group as u32)?;
                for decl in lane(&denoise.reading, stream.as_deref())?.inputs {
                    let ch = match decl.name.as_str() {
                        "latents" if latent => &x,
                        "context" if !latent => ctx,
                        "timestep" => &t,
                        "guidance" => &guidance_ch,
                        other => {
                            return Err(format!(
                                "reading `{}` reads `{other}` on lane {at}; this sampler binds \
                                 latents, timestep, guidance and context",
                                denoise.reading
                            )
                            .into());
                        }
                    };
                    pass.input(at, &decl.name, ch)?;
                }
            }
        }
        pass.epilogue(move || {
            let step = k.take();
            let v = if cfg {
                cfg_combine(
                    &intrinsics::velocity(0, n, vw),
                    &intrinsics::velocity(2, n, vw),
                    &Tensor::constant([guidance]),
                )
            } else {
                intrinsics::velocity(0, n, vw)
            };
            let sig = sigmas.read();
            let next = &step + 1u32;
            let dt = &at(&sig, &next) - &at(&sig, &step);
            let stepped = reshape(euler_step(&reshape(x.take(), [n, c]), &v, &dt), lat);
            x.put(&stepped);
            if land {
                out.put(&stepped);
            }
            t.take();
            t.put(at(&ts.read(), &next));
            k.put(&next);
        });
        Ok(pass)
    };
    let step = denoise_pass(false)?;
    let last = denoise_pass(true)?;
    for fire in 0..sched.steps() {
        let pass = if fire + 1 == sched.steps() {
            &last
        } else {
            &step
        };
        pass.submit(&pipe)
            .with_context(|| format!("denoise fire {fire}"))?;
    }
    let latent = out
        .take_host::<Vec<f32>>()
        .await
        .context("the denoised latent")?;
    // The denoise passes go before the decode opens: they hold seats.
    drop(step);
    drop(last);
    drop(held);
    if let Some(bad) = latent.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "the denoise loop left a non-finite latent (at {bad}); there is nothing to decode"
        )
        .into());
    }

    // ---- the decode: one fire per latent frame, in order ---------------------
    let mut pixels: Vec<f32> = Vec::with_capacity((frames * height * width * 3) as usize);
    // One working set for the whole clip: the conv caches are its state.
    let vae_state = WorkingSet::new();
    for j in 0..lt {
        let at = j as usize * plane;
        let landed = decode_frame(
            &decode,
            &latent[at..at + plane],
            lh,
            lw,
            if j == 0 { 1 } else { tc },
            height,
            width,
            &vae_state,
            &pipe,
        )
        .await?;
        pixels.extend_from_slice(&landed);
    }
    pipe.close();
    let want = (frames * height * width * 3) as usize;
    if pixels.len() != want {
        return Err(format!(
            "the decode landed {} values and a {frames}x{height}x{width} clip is {want}",
            pixels.len()
        )
        .into());
    }

    // ---- the way out ----------------------------------------------------------
    let name = input.out.clone().unwrap_or_else(|| "video".to_string());
    let asked_format = input.format.as_deref().unwrap_or("mp4");
    let (format, ext) = match asked_format {
        "mp4" | "mp4-h264" => (ImageFormat::Mp4H264, "mp4"),
        "y4m" => (ImageFormat::Y4m, "y4m"),
        "rgb8" | "raw-rgb8" => (ImageFormat::RawRgb8, "rgb"),
        other => {
            return Err(
                format!("unknown video format {other:?}; try 'mp4', 'y4m' or 'rgb8'").into(),
            );
        }
    };
    let handle = Frames::from_rgb8(&rgb8(&pixels), width, height, frames, fps)
        .map_err(|why| format!("frames.from-rgb8: {why}"))?;
    let file = format!("{name}.{ext}");
    inferlet::session::send_frames(&handle, format, &file)
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
        decode_reading: decode,
        init_frames: init_b64.len() as u32,
        encode_fires: if has_init { lt } else { 0 },
        strength: if has_init { strength } else { 1.0 },
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
        sigmas: sched.sigmas.clone(),
        context_rows: context.shape().dims()[0],
        prompt_rows,
        file,
        format: asked_format.to_string(),
    })
}
