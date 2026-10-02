//! **THE MODEL-AGNOSTIC `text-to-video` INFERLET** (imagegen design D1, D4,
//! D8, D11, D12): a prompt in, an mp4 out, with nothing about any family
//! spelled here. The sibling of `text-to-image`, and deliberately its
//! sibling rather than a flag on it — three things differ, and each is a
//! fact the host answers:
//!
//! | | image | video |
//! |---|---|---|
//! | the latent | an `h x w` GRID | a `t x h x w` VOLUME (`LaneRows::Volume`, `latent().temporal-compression > 1`) |
//! | the decode | ONE fire over one clip | ONE FIRE PER LATENT FRAME, in order, down one pipeline |
//! | the way out | a still through `take_frames` | a clip assembled from the fires, `frames.from-rgb8`, `mp4-h264` |
//!
//! # HOW IT FINDS ITS READINGS
//!
//! | role | the fact that says so |
//! |---|---|
//! | text encoder | `takes_tokens && readout == Hidden` |
//! | denoiser | `!takes_tokens && readout == Velocity` |
//! | VAE decode | `readout == Pixels`, a name that is not `vae.encode*` |
//! | VAE encode | `readout == Pixels`, a name that IS `vae.encode*` |
//!
//! # THE FRAME ARITHMETIC IS THE FAMILY'S, AND IT REFUSES BY NAME
//!
//! A causal video VAE decodes ONE latent frame at a time through per-conv
//! frame caches and treats the FIRST apart: latent frame 0 lands ONE output
//! frame and every later one lands `temporal-compression`. So `T` latent
//! frames are `1 + tc*(T - 1)` output frames and nothing else — 1, 5, 9,
//! 13, 17, ..., 49, ... at `tc = 4`. A `frames` that is not on that lattice
//! is REFUSED with the two nearest that are, rather than silently rounded:
//! a caller who asked for 48 frames and got 49 has a clip whose length is
//! not the one their timeline expects.
//!
//! A family that declares ONE decode reading (no `.head` arm) is an image
//! VAE or a non-causal one; this program then fires it once over the whole
//! clip and the arithmetic is `tc * T`.
//!
//! # THE DECODE FIRES ARE A SEQUENCE, AND THE ORDER IS LOAD-BEARING
//!
//! Every causal convolution of the decoder holds its last input frames in a
//! `CacheRow::State` slab that the slot carries BETWEEN fires. So the fires
//! are not independent: frame `k`'s pixels depend on frame `k-1`'s having
//! been fired into the same slot first. Two rules follow, and this program
//! keeps both:
//!
//! * ONE PIPELINE for the whole decode. A pipeline is serial, which here is
//!   the point — the opposite of the denoise loop, where one pipeline per
//!   lane is what lets a group compose.
//! * The head arm's pass is closed BEFORE the later arm's is opened. A
//!   float pass is seated on a scratch working set the host mints at bind,
//!   and a seat comes back to the book when its pass closes — so closing
//!   the head pass first is what puts the later arm on the slot the head
//!   arm warmed. Firing them the other way round, or holding both open,
//!   gives the later arm a slot whose slabs are zero, which is the pixels
//!   of a decoder with no history (measured: cos 0.98 and mean |err| 0.125
//!   on `[-1, 1]` against the reference, versus 0.99999 warm).
//!
//! # THE PIXELS DO CROSS, AND THAT IS SAID RATHER THAN HIDDEN
//!
//! `text-to-image` never lets its picture into linear memory: one fire, one
//! channel, `Channel::take_frames` hands the cell to the host's encoders.
//! A clip is several fires and there is no host-side verb that concatenates
//! their cells, so this program takes each chunk with `take_host` and builds
//! ONE handle with `frames.from-rgb8`. The cost is real (a 480x832x17 clip
//! is 81 MB of f32 through the boundary and 20 MB of RGB8 back) and the fix
//! is a seam that appends rather than replaces, not a change here.
//!
//! # VIDEO2VIDEO
//!
//! The video half of img2img, and the same three moves: a clip in through
//! the family's `vae.encode` reading, noised to the schedule's `strength`
//! point, integrated from there. What differs is the ENCODE, and it differs
//! exactly the way the decode does.
//!
//! A causal video VAE encodes ONE chunk at a time through the same per-conv
//! frame caches, and treats the FIRST apart: output frame 0 of each
//! `downsample3d` time convolution is the IDENTITY, not a convolution, so no
//! single strided conv over a whole clip can produce it. So the head arm
//! takes ONE pixel frame and every later chunk takes `temporal-compression`
//! of them, each landing one latent frame, down ONE pipeline in order — the
//! decode loop above read backwards, and refused by the same lattice
//! (`1 + tc*(T - 1)` pixel frames, no other length).
//!
//! Two things this does NOT do, and both are the family's own statement
//! rather than this program's:
//!
//! * It does not rescale. `vae.encode` lands `(mean - latents_mean)/std` —
//!   the exact inverse of what `vae.decode` undoes — so the encoded clip is
//!   already the DENOISER's own space and goes straight into the sampler.
//!   `the_wan_2_vae_encodes_the_reference` is where that is proved: the same
//!   rows fit the RAW posterior mean materially worse.
//! * It does not resize, in space or in time. An init clip that is not this
//!   run's `width x height x frames` is REFUSED by name; a guest has no
//!   resampler and a silent one would be the wrong one.
//!
//! # THE INIT CLIP CROSSES, AND SO DOES THE OUTPUT
//!
//! A clip comes in as `init_frame_<f>_<n>`: frame `f`'s encoded still, piece
//! `n`. Two indices because a clip is a LIST of pictures and one argv
//! argument is capped at `MAX_ARG_STRLEN` far below one still, so both
//! dimensions have to be spelled. The sandbox scratch is not the door:
//! `/scratch` is mounted per PROCESS and removed at teardown, so nothing
//! outside can place a file there for a guest to find.
//!
//! `frames.decode` sniffs ONE still and lands a one-frame handle, and there
//! is no host-side verb that concatenates handles — but a `vae.encode` chunk
//! is FOUR frames on one voxel port, one box. So each still is decoded
//! host-side, read back as raw RGB8 (`encode(raw-rgb8)`), concatenated in the
//! guest, and handed over as one `frames.from-rgb8` handle per chunk, seeded
//! onto the pixel port with `Channel::set_frames`. The pixels cross linear
//! memory once on the way in, exactly as they cross once on the way out, and
//! the fix for both is the same seam that appends rather than replaces.
//!
//! # CFG
//!
//! A model with no `guidance` port takes a negative prompt as a second lane
//! PAIR, and the two velocities combine `u + s(c - u)` ON THE DEVICE. The
//! branches are two attention GROUPS of one fire — they must not attend
//! each other — and each names the other with `ForwardPass::peer`, which is
//! also what gathers them into one cohort so they seal into one fire. BOTH
//! video lanes compute the same combine from their own side (the
//! conditional one reads `(own = c, peer = u)`, the other
//! `(own = u, peer = c)`), so both latents step in lockstep and neither
//! crosses the host between fires.
//!
//! This path used to fail at the first readback with "no cell available",
//! because the host combine took both branches' `out` cells in one turn and
//! one had none committed yet. Moving the combine onto the device removed
//! the turn, and with it the failure.
//!
//! # THE PROMPT, AND THE ROW WHOSE TOKENIZER PIE CANNOT SPELL
//!
//! `--prompt` renders the bound model's template and tokenizes with the
//! bound model's vocabulary. Wan 2.2's text encoder is umT5, whose
//! tokenizer is a SentencePiece **Unigram** model whereas `crates/tokenizer`
//! compiles BPE pipelines alone (`models::wan_2::tokenizer` states this),
//! so that row's artifact carries somebody else's vocabulary and `--prompt`
//! on it would condition the DiT on ids it has never seen. `--prompt-ids`
//! is the door for that: the caller tokenizes with the reference tokenizer
//! and hands the ids over, and the ENCODER still runs inside pie. The
//! report says which door was used, so a green run can never be mistaken
//! for a tokenizer that works.

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
    /// The latent-row volume and one row's width.
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    rows: u32,
    row_width: u32,
    /// `model::latent()`, verbatim.
    latent_channels: u32,
    patch_t: u32,
    patch_h: u32,
    patch_w: u32,
    spatial_compression: u32,
    temporal_compression: u32,
    /// The latent frames the decode fired, and the arms it used.
    latent_frames: u32,
    decode_fires: u32,
    decode_readings: Vec<String>,
    /// VIDEO2VIDEO: the init clip's frames (0 when none was passed), the
    /// `vae.encode` fires that took them and the arms those fires used, and
    /// the fraction of the schedule that was run. `strength` reads 1.0 with
    /// no init, which is what the schedule then is.
    init_frames: u32,
    encode_fires: u32,
    encode_readings: Vec<String>,
    strength: f32,
    /// The encoded clip's own moments, before a step runs over it. Here
    /// because they are the only view a caller has of what `vae.encode`
    /// landed: a run whose init clip came back as garbage says so here and
    /// nowhere else. Both read 0 with no init.
    init_mean: f32,
    init_std: f32,
    steps: u32,
    seed: u32,
    guidance: f32,
    cfg: bool,
    sigmas: Vec<f32>,
    /// The rows the context lane carries, and how many of them are the
    /// prompt's own (the rest are the family's stated zero pad).
    context_rows: u32,
    prompt_rows: u32,
    /// What the client was sent, and in what format.
    file: String,
    format: String,
    /// Health of the final latent — a loop that diverged says so here.
    non_finite: u32,
    mean: f32,
    std: f32,
}

/// The readings a text-to-video job needs, found by role.
struct Roles {
    text: model::ReadingFact,
    denoise: model::ReadingFact,
    /// The first-frame arm, when the family splits its causal decoder in
    /// two (`vae.decode.head`); `None` for a one-arm decoder.
    head: Option<model::ReadingFact>,
    /// The arm every later latent frame goes through, or the only arm.
    decode: model::ReadingFact,
    /// VIDEO2VIDEO's door in. The first-chunk arm, when the family splits
    /// its causal encoder in two (`vae.encode.head`); `None` for a one-arm
    /// encoder, and `None` for both when the family declares no encoder at
    /// all — this program only refuses that when an init clip is passed.
    encode_head: Option<model::ReadingFact>,
    /// The arm every later pixel chunk goes through, or the only arm.
    encode: Option<model::ReadingFact>,
}

fn roles() -> Result<Roles> {
    // **NOT `pass_kind()`.** That verb is derived from `rs_state_size() > 0`,
    // and a video row carries a causal-conv VAE whose every convolution owns
    // a `CacheRow::State` slab — so `wan22-ti2v-5b` answers `recurrent`
    // while a transformer-only miniature answers `attention`. The
    // difference is in the DECODER arms and says nothing about a denoise
    // pass. What a denoise step actually needs is stated on the reading,
    // and is checked below: it binds no kv space and takes no tokens.
    let readings = model::readings();
    let text = readings
        .iter()
        .find(|r| r.takes_tokens && r.readout == model::ReadoutKind::Hidden)
        .cloned()
        .ok_or("this model has no text reading")?;
    let denoise = readings
        .iter()
        .find(|r| !r.takes_tokens && r.readout == model::ReadoutKind::Velocity)
        .cloned()
        .ok_or("this model declares no token-less reading with a velocity readout")?;
    if denoise.has_kv {
        return Err(format!(
            "reading `{}` binds a kv space; a denoise pass binds none",
            denoise.name
        )
        .into());
    }
    // A decode arm lands PIXELS. The encoder lands pixels too — its pixels
    // are its INPUT — and the name is what separates them, in the design's
    // own reading vocabulary and not a family's.
    //
    // `starts_with`, not `!= "vae.encode"`: a causal video encoder is TWO
    // arms and the head one is `vae.encode.head`, which ends in `.head` like
    // a decode head does. Matched on the exact name alone, the encoder's
    // head arm was picked up as the DECODER's, and the first thing that
    // would have said so is a clip of noise.
    let pixels: Vec<model::ReadingFact> = readings
        .iter()
        .filter(|r| r.readout == model::ReadoutKind::Pixels)
        .cloned()
        .collect();
    let (encoders, decoders): (Vec<_>, Vec<_>) = pixels
        .into_iter()
        .partition(|r| r.name.starts_with("vae.encode"));
    let head = decoders.iter().find(|r| r.name.ends_with(".head")).cloned();
    let decode = decoders
        .iter()
        .find(|r| !r.name.ends_with(".head"))
        .cloned()
        .ok_or(
            "this model declares no `vae.decode` reading, so there is no way to turn its \
             latent into frames from inside pie",
        )?;
    let encode_head = encoders.iter().find(|r| r.name.ends_with(".head")).cloned();
    let encode = encoders
        .iter()
        .find(|r| !r.name.ends_with(".head"))
        .cloned();
    Ok(Roles {
        text,
        denoise,
        head,
        decode,
        encode_head,
        encode,
    })
}

/// The denoise reading's ports, sorted into the roles a sampler binds.
struct Ports {
    latents: model::PortFact,
    context: model::PortFact,
    timestep: String,
    guidance: Option<String>,
    convention: model::PositionConvention,
    video_stream: model::LaneStream,
    context_stream: model::LaneStream,
}

fn stream_of(
    streams: &[model::LaneStream],
    want: model::LaneStream,
    fallback: model::LaneStream,
) -> model::LaneStream {
    if streams.iter().any(|s| *s == want) {
        return want;
    }
    streams.first().copied().unwrap_or(fallback)
}

fn ports(reading: &model::ReadingFact) -> Result<Ports> {
    let of_kind = |kind: model::PortKind| reading.ports.iter().find(|p| p.kind == kind).cloned();
    let latents = of_kind(model::PortKind::Latents)
        .ok_or("this denoise reading declares no latents port; there is nothing to denoise")?;
    let context = of_kind(model::PortKind::Context)
        .ok_or("this denoise reading declares no context port for the prompt rows")?;
    let positions = of_kind(model::PortKind::AxisPositions)
        .ok_or("this denoise reading declares no axis-positions port")?;
    let convention = reading.positions.clone().ok_or_else(|| {
        format!(
            "reading `{}` takes positions but states no position convention, so a \
             family-blind guest cannot say where its rows sit",
            reading.name
        )
    })?;
    if convention.axes.len() as u32 != positions.width {
        return Err(format!(
            "reading `{}` states {} axis roles for a {}-wide positions port",
            reading.name,
            convention.axes.len(),
            positions.width
        )
        .into());
    }
    // A video family's rows sit on a TIME axis; without one there is no
    // coordinate for a frame index and this is the wrong driver.
    if !convention.axes.iter().any(|r| *r == model::AxisRole::Time) {
        return Err(format!(
            "reading `{}` states no `time` axis in its position convention; a clip's rows \
             have nowhere to put a frame index",
            reading.name
        )
        .into());
    }
    let guidance = reading
        .ports
        .iter()
        .find(|p| p.kind == model::PortKind::LaneVector && p.name == "guidance")
        .map(|p| p.name.clone());
    let timestep = reading
        .ports
        .iter()
        .find(|p| p.kind == model::PortKind::LaneVector && Some(&p.name) != guidance.as_ref())
        .map(|p| p.name.clone())
        .ok_or_else(|| {
            format!(
                "reading `{}` declares no lane-vector port for the timestep",
                reading.name
            )
        })?;
    let video_stream = stream_of(
        &latents.streams,
        model::LaneStream::Video,
        model::LaneStream::Image,
    );
    let context_stream = stream_of(
        &context.streams,
        model::LaneStream::Context,
        model::LaneStream::Text,
    );
    Ok(Ports {
        latents,
        context,
        timestep,
        guidance,
        convention,
        video_stream,
        context_stream,
    })
}

/// Everything one lane binds, kept alive for the loop's lifetime.
struct Lane {
    pass: ForwardPass,
    #[allow(dead_code)]
    held: Vec<Channel>,
}

#[allow(clippy::too_many_arguments)]
fn lane(
    reading: &model::ReadingFact,
    ports: &Ports,
    stream: model::LaneStream,
    group: u32,
    // The OTHER guidance branch's group, on the CFG path. Named by EVERY
    // lane of both branches, not only the ones that read: naming is what
    // gathers the two groups into one cohort, and a peer in another fire is
    // not on this fire's velocity plane at all.
    peer: Option<u32>,
    rows: u32,
    latents: Option<&Channel>,
    context: Option<&Channel>,
    positions: &[f32],
    clock: &LaneClock,
    guidance: f32,
    tag: &str,
) -> Result<Lane> {
    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.stream(stream)?;
    pass.group(group)?;
    if let Some(peer) = peer {
        pass.peer(peer)?;
    }
    let mut held = Vec::new();
    let mut latents_bound = false;
    let mut context_bound = false;
    let mut timestep_bound = false;
    for port in &reading.ports {
        if !(port.streams.is_empty() || port.streams.iter().any(|s| *s == stream)) {
            continue;
        }
        let name = format!("{tag}_{}", port.name);
        let ch = match port.kind {
            model::PortKind::Latents if !latents_bound && latents.is_some() => {
                latents_bound = true;
                latents.unwrap().clone()
            }
            model::PortKind::Context if !context_bound && context.is_some() => {
                context_bound = true;
                context.unwrap().clone()
            }
            model::PortKind::Latents | model::PortKind::Context => {
                Channel::from_shaped([rows, port.width], vec![0f32; (rows * port.width) as usize])
                    .named(&name)
            }
            model::PortKind::AxisPositions => {
                Channel::from_shaped([rows, port.width], positions.to_vec()).named(&name)
            }
            model::PortKind::LaneVector if port.name == ports.timestep && !timestep_bound => {
                timestep_bound = true;
                clock.timestep.clone()
            }
            model::PortKind::LaneVector if Some(&port.name) == ports.guidance.as_ref() => {
                Channel::from(vec![guidance; port.width as usize]).named(&name)
            }
            model::PortKind::LaneVector => {
                Channel::from(vec![1.0f32; port.width as usize]).named(&name)
            }
            model::PortKind::Voxels => {
                return Err(format!(
                    "reading `{}` declares a voxel port `{}`; this sampler drives latent rows, \
                     not a VAE tile",
                    reading.name, port.name
                )
                .into());
            }
        };
        pass.input(&port.name, &ch)?;
        held.push(ch);
    }
    Ok(Lane { pass, held })
}

/// Unpatchify a denoise reading's final latent into the clip volume a
/// `vae.decode` port takes: `[rows, C*pt*ph*pw]` in `(c, pt, ph, pw)`
/// feature order becomes `[grid_t*pt, grid_h*ph, grid_w*pw, C]` row-major
/// — the `[t, h, w, C]` shape a `Voxels` channel declares (design D8).
///
/// The inverse of the patchify every DiT does at its embed, written out
/// rather than called, because the guest is the only place both halves of
/// the pair are visible: the trunk states the patch and the VAE states the
/// clip, and nothing between them holds the index algebra.
#[allow(clippy::too_many_arguments)]
fn unpatchify(rows: &[f32], grid: [u32; 3], patch: [u32; 3], channels: u32) -> Vec<f32> {
    let (gt, gh, gw) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    let (pt, ph, pw) = (patch[0] as usize, patch[1] as usize, patch[2] as usize);
    let c = channels as usize;
    let (t, h, w) = (gt * pt, gh * ph, gw * pw);
    let row_width = c * pt * ph * pw;
    let mut out = vec![0f32; t * h * w * c];
    for a in 0..gt {
        for b in 0..gh {
            for d in 0..gw {
                let row = ((a * gh + b) * gw + d) * row_width;
                for ci in 0..c {
                    for z in 0..pt {
                        for y in 0..ph {
                            for x in 0..pw {
                                let src = row + ((ci * pt + z) * ph + y) * pw + x;
                                let dst =
                                    (((a * pt + z) * h + (b * ph + y)) * w + (d * pw + x)) * c + ci;
                                out[dst] = rows[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`unpatchify`]: the `[t, h, w, C]` latent clip a
/// `vae.encode` reading LANDS, back into the `[rows, C*pt*ph*pw]` rows a
/// denoise reading's latent port takes, in `(c, pt, ph, pw)` feature order.
///
/// Here for the same reason its pair is: the trunk states the patch and the
/// VAE states the clip, and the guest is the only place both halves of the
/// index algebra are visible.
fn patchify(clip: &[f32], grid: [u32; 3], patch: [u32; 3], channels: u32) -> Vec<f32> {
    let (gt, gh, gw) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    let (pt, ph, pw) = (patch[0] as usize, patch[1] as usize, patch[2] as usize);
    let c = channels as usize;
    let (h, w) = (gh * ph, gw * pw);
    let row_width = c * pt * ph * pw;
    let mut out = vec![0f32; gt * gh * gw * row_width];
    for a in 0..gt {
        for b in 0..gh {
            for d in 0..gw {
                let row = ((a * gh + b) * gw + d) * row_width;
                for ci in 0..c {
                    for z in 0..pt {
                        for y in 0..ph {
                            for x in 0..pw {
                                let dst = row + ((ci * pt + z) * ph + y) * pw + x;
                                let src =
                                    (((a * pt + z) * h + (b * ph + y)) * w + (d * pw + x)) * c + ci;
                                out[dst] = clip[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
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

/// Fire ONE encode arm over ONE PIXEL CHUNK and take the latent frame it
/// lands: the mirror of [`decode_frame`], and the door video2video comes in
/// through.
///
/// The chunk goes in as the port's CHANNEL, seeded straight from a host-held
/// `frames` handle (`Channel::set_frames`, which is a `put` and so must
/// happen after the pass has bound the port — a channel has no ring before
/// that). Its shape IS the clip's box, `[t, H, W, 3]`, so the geometry the
/// voxel axis needs travels with the numbers.
///
/// The answer comes back on the same `pixels` seam a decode uses. For an
/// encoder that seam carries ONE LATENT FRAME — the posterior MEAN, already
/// normalised into the denoiser's own space — whatever the number of pixel
/// frames that went in.
async fn encode_chunk(
    reading: &model::ReadingFact,
    chunk: &Frames,
    chunk_frames: u32,
    pixel_h: u32,
    pixel_w: u32,
    clip_h: u32,
    clip_w: u32,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let port = reading
        .ports
        .iter()
        .find(|p| p.kind == model::PortKind::Voxels)
        .ok_or_else(|| {
            format!(
                "reading `{}` takes pixels but declares no voxel port; there is no chunk to \
                 hand it",
                reading.name
            )
        })?;
    let rows = clip_h * clip_w;
    let width = reading.readout_width;
    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.stream(model::LaneStream::Video)?;
    // Empty, and SEEDED from the chunk below: the pixels are the host's, and
    // the eta trace tracks readiness STATICALLY — a channel the host seeds
    // has to be declared `seeded`, or the trace refuses it as consumed but
    // never produced.
    let cell = Channel::seeded([chunk_frames, pixel_h, pixel_w, port.width], dtype::f32)
        .named("vae_pixels_in");
    pass.input(&port.name, &cell)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_latent_out");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(rows, width));
    });
    // THE PIXELS, seeded straight from the host's handle, after the bind.
    cell.set_frames(chunk)?;
    pass.submit(pipe)
        .with_context(|| format!("the `{}` fire", reading.name))?;
    let landed = out.take_host::<Vec<f32>>().await?;
    let want = (rows * width) as usize;
    if landed.len() != want {
        return Err(format!(
            "reading `{}` landed {} numbers over a {chunk_frames}-frame chunk and one latent \
             frame is {clip_h}x{clip_w}x{width} = {want}",
            reading.name,
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
    reading: &model::ReadingFact,
    clip: &[f32],
    clip_h: u32,
    clip_w: u32,
    frames: u32,
    pixel_h: u32,
    pixel_w: u32,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let port = reading
        .ports
        .iter()
        .find(|p| p.kind == model::PortKind::Voxels)
        .ok_or_else(|| {
            format!(
                "reading `{}` lands pixels but reads no voxel port; there is no clip to hand it",
                reading.name
            )
        })?;
    let want = (clip_h as usize) * (clip_w as usize) * (port.width as usize);
    if clip.len() != want {
        return Err(format!(
            "one latent frame is {} numbers and port `{}` wants {clip_h}x{clip_w}x{} = {want}",
            clip.len(),
            port.name,
            port.width
        )
        .into());
    }
    let rows = frames * pixel_h * pixel_w;
    let width = reading.readout_width;
    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.stream(model::LaneStream::Video)?;
    let cell =
        Channel::from_shaped([clip_h, clip_w, port.width], clip.to_vec()).named("vae_latent");
    pass.input(&port.name, &cell)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_pixels");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(rows, width));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{}` fire", reading.name))?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

/// `[-1, 1]` f32 in `(t, h, w, 3)` order to interleaved RGB8, clamped.
fn rgb8(pixels: &[f32]) -> Vec<u8> {
    pixels
        .iter()
        .map(|v| (((v + 1.0) * 127.5).clamp(0.0, 255.0)) as u8)
        .collect()
}

/// **THE CONTEXT PAD, WHERE THE FAMILY STATES ONE.**
///
/// `port-fact.rows` is a family saying "my context lane is exactly this
/// tall, whatever the prompt was". Wan 2.2 says 512: its reference
/// truncates umT5's answer to the prompt's real length and zero-pads the
/// EMBEDS back to 512, and the transformer attends every one of those keys
/// — the pad rows are not masked away, they go through `text_embedder`
/// into a nonzero constant with real attention mass. Handing such a model a
/// context lane of the prompt's own height is a different model, quietly.
///
/// `None` (FLUX.2, Z-Image) means the encoder's rows are the lane's rows
/// and this hands the channel straight back. A prompt LONGER than the
/// stated height is refused rather than truncated: the encoder already
/// truncated at `max-embed-length` if it had to, and silently dropping rows
/// here would drop the end of a caption.
fn pad_context(mut rows: Vec<f32>, width: u32, port: &model::PortFact) -> Result<Channel> {
    if width != port.width {
        return Err(format!(
            "the text reading hands back {width}-wide rows and port `{}` takes {}",
            port.name, port.width
        )
        .into());
    }
    let have = u32::try_from(rows.len() / width.max(1) as usize).unwrap_or(0);
    let want = port.rows.unwrap_or(have);
    if have > want {
        return Err(format!(
            "port `{}` takes exactly {want} context rows and the prompt encoded to {have}",
            port.name
        )
        .into());
    }
    // Zero rows, not pad-token rows: the reference truncates the encoder's
    // answer and pads the EMBEDS, and zeros are what the DiT was
    // conditioned on. There is no device-side verb that grows a rectangle,
    // and these rows already crossed the host coming out of the encoder.
    rows.resize((want * width) as usize, 0.0);
    Ok(Channel::from_shaped([want, width], rows).named("context"))
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

/// One prompt's two lanes: the encoder rows and the latent they condition.
struct Branch {
    context: Lane,
    image: Lane,
    out: Channel,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let roles = roles()?;
    let ports = ports(&roles.denoise)?;

    // ---- the latent volume, out of `model::latent()` ----------------------
    let space = model::latent().ok_or(
        "this model states no latent space, so its denoise reading cannot be sized from pixels",
    )?;
    let sc = space.spatial_compression.max(1);
    let tc = space.temporal_compression.max(1);
    let (pt, ph, pw) = (
        space.patch_t.max(1),
        space.patch_h.max(1),
        space.patch_w.max(1),
    );
    if tc == 1 {
        return Err(
            "this model's latent space has no temporal compression: it is an image family, and \
             `text-to-image` is its driver"
                .into(),
        );
    }
    let cell_h = sc * ph;
    let cell_w = sc * pw;
    let height = (input.height.unwrap_or(480) / cell_h).max(1) * cell_h;
    let width = (input.width.unwrap_or(832) / cell_w).max(1) * cell_w;
    let grid_h = height / cell_h;
    let grid_w = width / cell_w;

    // THE FRAME LATTICE, REFUSED BY NAME. A causal decoder with a head arm
    // lands `1 + tc*(T - 1)` frames; a one-arm decoder lands `tc*T`.
    let asked = input.frames.unwrap_or(17).max(1);
    let latent_frames = if roles.head.is_some() {
        if (asked - 1) % tc != 0 {
            let below = 1 + tc * ((asked - 1) / tc);
            return Err(format!(
                "this family's VAE lands 1 + {tc}*(T - 1) frames, so {asked} is not a clip it \
                 has: the nearest are {below} and {}",
                below + tc
            )
            .into());
        }
        (asked - 1) / tc + 1
    } else {
        if asked % tc != 0 {
            return Err(format!(
                "this family's VAE lands {tc}*T frames, so {asked} is not a clip it has: the \
                 nearest are {} and {}",
                (asked / tc).max(1) * tc,
                (asked / tc + 1) * tc
            )
            .into());
        }
        asked / tc
    };
    let frames = asked;
    if latent_frames % pt != 0 {
        return Err(format!(
            "{latent_frames} latent frames do not divide into this model's {pt}-deep patches"
        )
        .into());
    }
    let grid_t = latent_frames / pt;
    let rows = grid_t * grid_h * grid_w;
    let row_width = ports.latents.width;
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && rows > max_rows {
        return Err(format!(
            "{width}x{height} at {frames} frames is {rows} latent rows and this model carries \
             {max_rows} a pass"
        )
        .into());
    }

    // ---- the schedule -----------------------------------------------------
    let fact = model::schedule().ok_or("this model states no schedule; nothing here denoises")?;
    let steps = match input.steps.filter(|s| *s > 0) {
        Some(steps) => steps,
        None if !fact.pinned_sigmas.is_empty() => {
            u32::try_from(fact.pinned_sigmas.len()).unwrap_or(8)
        }
        None => 8,
    };
    // **NO DYNAMIC SHIFT.** `FlowMatchEuler`'s `rows` argument turns the
    // stated shift into a base `mu` and bends it by the latent's row count
    // — a resolution heuristic FLUX and Z-Image were trained with. A video
    // family's rows count FRAMES as well as pixels, and neither video row
    // in the tree rescales by them (Wan's scheduler says
    // `use_dynamic_shifting: false`, and its 5.0 is the shift itself), so
    // bending by 1950 rows would put nineteen of twenty steps above sigma
    // 0.58 and leave the last one to do the denoising. `None` takes the
    // stated shift as the fixed one it is.
    let full = FlowMatchEuler::from_schedule(&fact, steps, None)?;

    // VIDEO2VIDEO: `strength` is the FRACTION OF THE TRAJECTORY still to
    // run, so the schedule is truncated to its tail and the encoded clip is
    // noised to that tail's first sigma. `1.0` truncates nothing, which
    // makes `(1 - 1)*x0 + 1*eps` the keyed draw and the run identical to one
    // with no init at all — the identity a caller can check.
    let init_b64 = input.init_frames();
    let has_init = !init_b64.is_empty();
    let strength = input.strength.unwrap_or(0.6);
    if has_init && !(0.0 < strength && strength <= 1.0) {
        return Err(format!("`strength` is {strength}; it is a fraction in (0, 1]").into());
    }
    let cut = if has_init {
        // `steps - round(steps * strength)`: the number of steps SKIPPED.
        // Clamped to leave at least one, since a schedule of no steps is a
        // clip handed straight back.
        let run = ((steps as f32 * strength).round() as u32).clamp(1, steps);
        (steps - run) as usize
    } else {
        0
    };
    let sched = if cut == 0 {
        full.clone()
    } else {
        FlowMatchEuler::from_sigmas(full.sigmas[cut..].to_vec(), full.train_steps)
    };
    let sigma0 = *sched.sigmas.first().unwrap_or(&1.0);
    let fps = input.fps.unwrap_or(24.0);

    // ---- the prompt, through whichever door the caller opened -------------
    let guidance = input.guidance.unwrap_or(1.0);
    let (encoded, prompt, prompt_source) = match (&input.prompt_ids, &input.prompt) {
        (Some(ids), _) => {
            let ids = ids_of(ids)?;
            let shown = format!("<{} ids>", ids.len());
            (
                encode_ids_rows(&ids, &roles.text.name).await?,
                shown,
                "prompt-ids".to_string(),
            )
        }
        (None, Some(text)) if !text.trim().is_empty() => (
            encode_ids_rows(&template_ids(text), &roles.text.name).await?,
            text.clone(),
            "prompt".to_string(),
        ),
        _ => {
            return Err(
                "pass `prompt` or `prompt_ids`: this program films what it is told to".into(),
            );
        }
    };
    let (hidden, hidden_width) = encoded;
    let prompt_rows = u32::try_from(hidden.len() / hidden_width.max(1) as usize).unwrap_or(0);
    let context = pad_context(hidden, hidden_width, &ports.context)?;
    let context_rows = context.shape().dims()[0];
    let negative = match (&input.negative_prompt_ids, &input.negative_prompt) {
        (Some(ids), _) => {
            let (anti, anti_width) = encode_ids_rows(&ids_of(ids)?, &roles.text.name).await?;
            Some(pad_context(anti, anti_width, &ports.context)?)
        }
        (None, Some(text)) if !text.trim().is_empty() => {
            let (anti, anti_width) = encode_ids_rows(&template_ids(text), &roles.text.name).await?;
            Some(pad_context(anti, anti_width, &ports.context)?)
        }
        _ => None,
    };
    if negative.is_some() && ports.guidance.is_some() {
        return Err(format!(
            "reading `{}` declares a `guidance` port, so this row is guidance-distilled and a \
             negative prompt has no lane to ride",
            roles.denoise.name
        )
        .into());
    }
    let cfg = negative.is_some() && guidance > 1.0;
    let uncond = if cfg { negative } else { None };

    // ---- video2video: the clip, encoded chunk by chunk --------------------
    //
    // Before the loop, on a pipeline of its own that closes: each encode
    // pass holds a seat and the denoise lanes need it back.
    //
    // The chunk lattice is the DECODE's read backwards. A family with a head
    // arm takes pixel frame 0 alone and then `tc` frames at a time; a
    // one-arm encoder takes `tc` every time. Either way each chunk lands one
    // latent frame, and the count must come out at `latent_frames` — the
    // same arithmetic that sized the run, so a mismatch here is this
    // program disagreeing with itself and says so.
    let mut encode_used: Vec<String> = Vec::new();
    let init_latent: Option<Vec<f32>> = if !has_init {
        None
    } else {
        let body = roles.encode.as_ref().ok_or(
            "this model declares no `vae.encode` reading, so an init clip has no door to \
             come in through",
        )?;
        if init_b64.len() as u32 != frames {
            return Err(format!(
                "the init clip is {} frames and this run is {frames}; a guest has no \
                 resampler in time and a silent one would be the wrong one",
                init_b64.len()
            )
            .into());
        }
        // Every frame decoded host-side, then read back as raw RGB8 so the
        // guest can concatenate them: a `vae.encode` chunk is FOUR frames on
        // ONE voxel port, and there is no host-side verb that joins handles.
        let mut rgb: Vec<Vec<u8>> = Vec::with_capacity(init_b64.len());
        for (f, b64) in init_b64.iter().enumerate() {
            let bytes = b64_decode(b64)?;
            let picture = Frames::decode(&bytes)
                .map_err(|why| format!("init frame {f} does not decode: {why}"))?;
            if picture.width() != width || picture.height() != height {
                return Err(format!(
                    "init frame {f} is {}x{} and this run is {width}x{height}; resize the \
                     clip before handing it over — a guest has no resampler and a silent \
                     one would be the wrong one",
                    picture.width(),
                    picture.height()
                )
                .into());
            }
            if picture.count() != 1 {
                return Err(format!(
                    "init frame {f} decoded to {} frames; `frames.decode` sniffs a STILL and \
                     a clip comes in one still per frame",
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
        let mut chunks: Vec<u32> = Vec::new();
        if roles.encode_head.is_some() {
            chunks.push(1);
        }
        while chunks.iter().sum::<u32>() < frames {
            chunks.push(tc);
        }
        if chunks.iter().sum::<u32>() != frames || chunks.len() as u32 != latent_frames {
            return Err(format!(
                "a {frames}-frame clip does not chunk into {latent_frames} encode fires at \
                 tc = {tc}{}",
                if roles.encode_head.is_some() {
                    " with a head arm"
                } else {
                    ""
                }
            )
            .into());
        }
        let clip_h = grid_h * ph;
        let clip_w = grid_w * pw;
        let plane = (clip_h * clip_w * space.channels) as usize;
        let mut clip: Vec<f32> = Vec::with_capacity(latent_frames as usize * plane);
        // ONE pipeline for the whole encode, and each arm's pass closed
        // before the next opens: the causal convolutions' `CacheRow::State`
        // slabs make these fires a sequence, exactly as the decode's are.
        let enc_pipe = Pipeline::new();
        let mut at = 0usize;
        for (k, take) in chunks.iter().copied().enumerate() {
            let reading = match (k, roles.encode_head.as_ref()) {
                (0, Some(head)) => head,
                _ => body,
            };
            let mut payload: Vec<u8> =
                Vec::with_capacity(take as usize * (height * width * 3) as usize);
            for f in at..at + take as usize {
                payload.extend_from_slice(&rgb[f]);
            }
            at += take as usize;
            let handle = Frames::from_rgb8(&payload, width, height, take, fps)
                .map_err(|why| format!("encode chunk {k}: frames.from-rgb8: {why}"))?;
            let landed = encode_chunk(
                reading, &handle, take, height, width, clip_h, clip_w, &enc_pipe,
            )
            .await
            .with_context(|| format!("encoding chunk {k}"))?;
            clip.extend_from_slice(&landed);
            // The arms this clip actually used, in first-use order: a
            // one-arm encoder names one, a split one names both, and a
            // single-frame clip on a split encoder names only the head.
            if !encode_used.iter().any(|name| name == &reading.name) {
                encode_used.push(reading.name.clone());
            }
        }
        enc_pipe.close();
        // NO RESCALE. `vae.encode` lands `(mean - latents_mean)/std`, the
        // exact inverse of what `vae.decode` undoes, so these rows are the
        // denoise reading's own space already. All that is left is the
        // trunk's patchify, which the guest owns because only the guest sees
        // both the family's patch and the family's clip.
        Some(patchify(
            &clip,
            [grid_t, grid_h, grid_w],
            [pt, ph, pw],
            space.channels,
        ))
    };

    let (init_mean, init_std) = match &init_latent {
        None => (0.0, 0.0),
        Some(rows) => {
            let n = rows.len().max(1) as f32;
            let mean = rows.iter().sum::<f32>() / n;
            let var = rows.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            if rows.iter().any(|v| !v.is_finite()) {
                return Err("the encoded init clip holds non-finite values".into());
            }
            (mean, var.sqrt())
        }
    };

    // ---- the loop ---------------------------------------------------------
    let mut loops = DenoiseLoop::new(&sched);
    let velocity_width = roles.denoise.readout_width;
    let cells = (rows * row_width) as usize;
    let shape = [rows, row_width];
    let seed = input.seed.unwrap_or(0);
    let volume = LaneRows::Volume {
        t: grid_t,
        h: grid_h,
        w: grid_w,
    };

    let mut branches: Vec<Branch> = Vec::new();
    for (group, ctx) in [Some(&context), uncond.as_ref()]
        .into_iter()
        .flatten()
        .enumerate()
    {
        let group = group as u32;
        let tag = if group == 0 { "cond" } else { "uncond" };
        let ctx_rows = ctx.shape().dims()[0];
        let text_clock = loops.lane(&format!("{tag}_ctx"));
        let image_clock = loops.lane(&format!("{tag}_vid"));
        let context_lane = lane(
            &roles.denoise,
            &ports,
            ports.context_stream,
            group,
            cfg.then(|| 1 - group),
            ctx_rows,
            None,
            Some(ctx),
            &positions_for(&ports.convention, LaneRows::Sequence(ctx_rows), ctx_rows),
            &text_clock,
            guidance,
            &format!("{tag}_ctx"),
        )?;
        let latent = Channel::from_shaped(shape, vec![0f32; cells]).named(&format!("{tag}_x"));
        let image_lane = lane(
            &roles.denoise,
            &ports,
            ports.video_stream,
            group,
            cfg.then(|| 1 - group),
            rows,
            Some(&latent),
            None,
            &positions_for(&ports.convention, volume, ctx_rows),
            &image_clock,
            guidance,
            &format!("{tag}_vid"),
        )?;
        let out = Channel::new(shape, dtype::f32)
            .capacity(channel_capacity() as u32)
            .named(&format!("{tag}_out"));
        text_clock.drive(&context_lane.pass, |_| {});
        let dts = loops.dts(tag);
        let rng = Channel::from(rng_state(seed)).named(&format!("{tag}_rng"));
        let x = latent.clone();
        let readback = out.clone();
        // VIDEO2VIDEO: the encoded clip, seeded once, read by every fire and
        // used by fire 0. Its own channel per branch, since a seeded channel
        // attaches to one pass (the runtime's channel-role rule).
        let init = init_latent
            .as_ref()
            .map(|rows| Channel::from_shaped(shape, rows.as_slice()).named(&format!("{tag}_init")));
        image_clock.drive(&image_lane.pass, move |k| {
            // On the CFG path BOTH branches step with the SAME combine, each
            // computed from its own side: the conditional lane reads
            // `(own = c, peer = u)` and the unconditional one
            // `(own = u, peer = c)`, and `u + s(c - u)` is the same number
            // either way. So the two latents stay in step without either one
            // crossing the host between fires.
            let v = if cfg {
                guided_velocity(velocity_width, guidance, group == 0)
            } else {
                intrinsics::velocity(velocity_width)
            };
            // Fire 0 either draws the latent from the keyed RNG, or —
            // video2video — takes the encoded clip noised to the truncated
            // schedule's first sigma. Every later fire is the same Euler
            // step either way.
            match &init {
                None => seed_or_step(k, &x, &v, &dts, &rng, shape, Some(&readback)),
                Some(init) => {
                    resume_or_step(k, &x, init, sigma0, &v, &dts, &rng, shape, Some(&readback))
                }
            }
        });
        branches.push(Branch {
            context: context_lane,
            image: image_lane,
            out,
        });
    }

    // Both paths are the same loop now: the device seeds its own latent and
    // integrates its own Euler step, guided or not. The guided branches read
    // each other's velocity off the fire's own plane, so nothing crosses the
    // host between steps — which is also why the four-lane path runs at all
    // now: it no longer takes two branches' cells in one turn.
    let mut last: Vec<f32> = Vec::new();
    for fire in 0..loops.fires() {
        let passes: Vec<&ForwardPass> = branches
            .iter()
            .flat_map(|b| [&b.context.pass, &b.image.pass])
            .collect();
        loops
            .fire(&passes)
            .with_context(|| format!("fire {fire}"))?;
        // The conditional branch's latent is the answer; the unconditional
        // one steps in lockstep with it (same combine, computed from its own
        // side) and is read only to keep its channel's cursor moving.
        last = branches[0]
            .out
            .take_host::<Vec<f32>>()
            .await
            .with_context(|| format!("readback after fire {fire}"))?;
        if cfg {
            let _ = branches[1].out.take_host::<Vec<f32>>().await?;
        }
    }
    loops.close();
    // EVERY DENOISE PASS IS CLOSED BEFORE THE FIRST DECODE FIRE. Each holds
    // a seat, and the decode's two arms need the ones this releases.
    drop(branches);

    let n = last.len().max(1) as f32;
    let mean = last.iter().sum::<f32>() / n;
    let var = last.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    if last.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "the denoise loop left {} non-finite values in the latent; there is nothing to \
             decode",
            last.iter().filter(|v| !v.is_finite()).count()
        )
        .into());
    }

    // ---- the decode: one fire per latent frame, in order ------------------
    let clip = unpatchify(
        &last,
        [grid_t, grid_h, grid_w],
        [pt, ph, pw],
        space.channels,
    );
    let clip_h = grid_h * ph;
    let clip_w = grid_w * pw;
    let plane = (clip_h * clip_w * space.channels) as usize;
    let mut pixels: Vec<f32> = Vec::with_capacity((frames * height * width * 3) as usize);
    let mut used: Vec<String> = Vec::new();

    // ONE pipeline for the whole decode: a pipeline is serial, and the
    // frame caches make these fires a sequence.
    let vae_pipe = Pipeline::new();
    let mut first_frame = 0u32;
    if let Some(head) = &roles.head {
        let out = decode_frame(
            head,
            &clip[..plane],
            clip_h,
            clip_w,
            1,
            height,
            width,
            &vae_pipe,
        )
        .await?;
        pixels.extend_from_slice(&out);
        used.push(head.name.clone());
        first_frame = 1;
    }
    // The head arm's pass is closed by now (`decode_frame` owns it and it
    // died with the call), so the later arm binds onto the seat it warmed.
    for k in first_frame..latent_frames {
        let at = k as usize * plane;
        let out = decode_frame(
            &roles.decode,
            &clip[at..at + plane],
            clip_h,
            clip_w,
            tc,
            height,
            width,
            &vae_pipe,
        )
        .await?;
        pixels.extend_from_slice(&out);
    }
    used.push(roles.decode.name.clone());
    vae_pipe.close();

    let want = (frames as usize) * (height as usize) * (width as usize) * 3;
    if pixels.len() != want {
        return Err(format!(
            "the decode landed {} values and a {frames}x{height}x{width} clip is {want}",
            pixels.len()
        )
        .into());
    }

    // ---- the way out ------------------------------------------------------
    let name = input.out.clone().unwrap_or_else(|| "video".to_string());
    let asked_format = input.format.as_deref().unwrap_or("mp4");
    // `rgb8` is the PARITY format — exactly the bytes the handle holds,
    // `count*height*width*3`, no header, no codec. A claim that two runs
    // landed the SAME clip is a claim about pixels, and an H.264 file is a
    // claim about an encoder as well.
    let (format, ext) = match asked_format {
        "mp4" | "mp4-h264" => (ImageFormat::Mp4H264, "mp4"),
        "y4m" => (ImageFormat::Y4m, "y4m"),
        // `.rgb`, not `.rgb8`: the host names a file by the FORMAT's own
        // extension (`image_extension`), and a suggested name that disagrees
        // gets that one appended to it.
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
        reading: roles.denoise.name.clone(),
        prompt,
        prompt_source,
        width,
        height,
        frames,
        fps,
        grid_t,
        grid_h,
        grid_w,
        rows,
        row_width,
        latent_channels: space.channels,
        patch_t: pt,
        patch_h: ph,
        patch_w: pw,
        spatial_compression: sc,
        temporal_compression: tc,
        latent_frames,
        decode_fires: latent_frames,
        decode_readings: used,
        init_mean,
        init_std,
        init_frames: init_b64.len() as u32,
        encode_fires: if has_init { latent_frames } else { 0 },
        encode_readings: encode_used,
        strength: if has_init { strength } else { 1.0 },
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
        sigmas: sched.sigmas.clone(),
        context_rows,
        prompt_rows,
        file,
        format: asked_format.to_string(),
        non_finite: 0,
        mean,
        std: var.sqrt(),
    })
}

