//! **THE MODEL-AGNOSTIC `text-to-image` INFERLET** (imagegen design D1, D4,
//! D11, D12): a prompt in, an image out, with nothing about any family
//! spelled here. Everything this program needs to size and drive the job is
//! a host-answered fact — `model::readings()`, `model::latent()`,
//! `model::schedule()`, `model::max_latent_rows()` — and every port and
//! reading is found by ROLE, never by name and never by `architecture()`.
//!
//! # HOW IT FINDS ITS THREE READINGS
//!
//! | role | the fact that says so |
//! |---|---|
//! | text encoder | `takes_tokens && readout == Hidden` |
//! | denoiser | `!takes_tokens && readout == Velocity` |
//! | VAE decode | a reading named `vae.decode` |
//!
//! A model with no text reading is refused BY NAME ("this model has no text
//! reading") — that is the right answer for the `mini-dit` fixture, whose
//! caption rows are random embeddings and which therefore cannot be told
//! what to draw.
//!
//! # THE STEP LOOP
//!
//! Two lanes of one request (D2), each on its own pipeline, submitted back
//! to back so the wait-all seal composes them into ONE fire: the prompt
//! lane (the stream the reading's `Context` port lists — `Text` for FLUX.2,
//! `Context` for Z-Image) carrying the encoder's rows, and the image lane
//! carrying the latent. Neither `embed` nor `attention` is called: a
//! denoise reading declares no tokens and no KV space.
//!
//! The sampler is an EPILOGUE (D4). Fire 0 seeds the latent with a keyed
//! Normal draw on the device — no 2 MB `Channel::from` through WASM — and
//! every later fire integrates one Euler step `x <- x + (sigma' - sigma).v`
//! over `velocity()`, advancing the loop-carried timestep and step cells.
//! `inferlet::latent::DenoiseLoop` owns the per-lane pipelines and clocks;
//! `seed_or_step` is the epilogue body.
//!
//! # POSITIONS COME FROM A FACT, NOT FROM THE FAMILY
//!
//! Where a lane's rows sit in the rotary space is a family contract — FLUX.2
//! puts the target grid at `(0, h, w, 0)` and text row `j` at `(0, 0, 0, j)`,
//! Z-Image puts the caption at `(1 + j, 0, 0)` and the image BEHIND it at
//! `(L + 1, a, b)` — so a guest that spelled either would be a guest for one
//! family. The reading states it instead (`reading-fact.positions`, a
//! `position-convention` of per-axis roles plus the text axis, its origin
//! and whether the image follows the caption on it), and
//! `latent::positions_for` fills both grids from the same code.
//!
//! # THE WAY OUT
//!
//! Two exits, and which one a model gets is its own fact.
//!
//! The intended one is D11, and it is wired: a model that declares a
//! `vae.decode` reading — a `Voxels` port and a `Pixels` readout — gets its
//! final latent unpatchified into the clip that port takes, fired on the
//! voxel axis, and read back off the `pixels` seam with
//! `intrinsics::pixels()`. `Channel::take_frames` turns that channel's cell
//! into a host-held `frames` handle and `session::send_frames` encodes and
//! streams it to the client, so the picture never enters linear memory.
//!
//! A model whose VAE is TRACED but not declared as a reading (FLUX.2 today:
//! its decoder's mid-block attention has no kernel arm, so its pixels would
//! not be the reference's) still takes the second exit, and the report says
//! which reading it saw.
//!
//! That second exit is the final latent as raw little-endian f32 through
//! `session::send_file_as`, under the name the report also states
//! (`<out>.latent.f32`), with the geometry a decode needs returned as the
//! JSON report — its sidecar. It is named for the same reason the picture
//! is: `pie run -o DIR` writes a file whose name says what it is rather
//! than a numbered `file-0000.bin`, and `scripts/imagegen/decode_latent.py`
//! finishes the job with the diffusers VAE of the same checkpoint.
//!
//! # CFG
//!
//! When the reading declares a `guidance` port the model is guidance-
//! distilled and the scale is just another lane vector; `--negative-prompt`
//! is then refused as meaningless. Without that port, a negative prompt at
//! `guidance > 1` runs a second lane PAIR (its own group, its own prompt
//! rows, its own latent), and the two velocities combine as
//! `u + s(c - u)` — ON THE DEVICE, inside the epilogue.
//!
//! The two branches are two attention GROUPS of one fire. They must not
//! attend each other (they are independent denoisings), and each names the
//! other with `ForwardPass::peer`, which is also what gathers them into one
//! cohort so they seal into one fire — a peer in another fire is not on
//! that fire's velocity plane at all. BOTH image lanes then compute the
//! same combine from their own side: the conditional one reads
//! `(own = c, peer = u)` and the unconditional one `(own = u, peer = c)`,
//! and `u + s(c - u)` is the same number either way. So both latents step
//! in lockstep without either crossing the host, and — unlike the old
//! host-combining path, which had to draw its own noise — the seeded device
//! draw is the same draw the ungated path makes.
//!
//! Neither shipped row exercises the guided path: klein-4B and Z-Image
//! Turbo are both guidance-distilled and take the scale as a port. What
//! proves the combine is `gates.py --only guidance`, on `mini-dit`.

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
    /// IMG2IMG: a picture to start from, base64 of an encoded still (PNG,
    /// JPEG, GIF, WebP). It is run through the family's `vae.encode`
    /// reading, noised to the schedule's `strength` point, and the sampler
    /// integrates from there — so the output keeps the input's composition
    /// and takes the prompt's content.
    #[serde(default)]
    init_image: Option<String>,
    /// The same picture in PIECES: `init_image_0`, `init_image_1`, ...
    /// concatenated in index order. An encoded still is hundreds of
    /// kilobytes and one argv argument is capped far below that
    /// (`MAX_ARG_STRLEN`), so a caller driving this over a command line
    /// splits it; a client with a real request body uses `init_image`.
    ///
    /// The sandbox scratch is NOT the door: `/scratch` is mounted per
    /// PROCESS (`<base>/<process id>`, removed at teardown), so nothing
    /// outside can place a file there for the guest to find.
    #[serde(flatten)]
    rest: std::collections::BTreeMap<String, inferlet::serde_json::Value>,
    /// How much of the schedule to run over the encoded picture, in
    /// `(0, 1]`. `1.0` is every step, which is text-to-image from noise;
    /// `0.6` keeps the first 40% of the trajectory as the picture's own.
    /// Only read when `init_image` is given.
    #[serde(default)]
    strength: Option<f32>,
    /// Inpainting: the region to repaint, `x0,y0,x1,y1` in pixels; the rest
    /// of `init_image` is kept. `mask_image_<n>` pieces carry a PNG mask
    /// instead (white = repaint).
    #[serde(default)]
    mask_rect: Option<String>,
    /// Diagnostic: scale the encoded prompt rows before they reach the
    /// context port (0 = the prompt's shape with none of its content).
    #[serde(default)]
    context_scale: Option<f32>,
}

async fn encode_prompt(prompt: &str, reading: &str, scale: Option<f32>) -> Result<Channel> {
    let Some(scale) = scale else {
        return encode_text(prompt, reading).await.map_err(Into::into);
    };
    let mut ids = inferlet::chat::first_user(prompt);
    ids.extend(inferlet::chat::cue());
    let (mut rows, width) = encode_ids_rows(&ids, reading).await?;
    for v in &mut rows {
        *v *= scale;
    }
    let len = (rows.len() / width.max(1) as usize) as u32;
    Ok(Channel::from_shaped([len, width], rows))
}

impl Input {
    /// The `init_image_<n>` pieces, in `n` order, concatenated.
    fn image_pieces(&self) -> String {
        let mut found: Vec<(u64, &str)> = self
            .rest
            .iter()
            .filter_map(|(name, value)| {
                let n = name.strip_prefix("init_image_")?.parse::<u64>().ok()?;
                Some((n, value.as_str()?))
            })
            .collect();
        found.sort_unstable_by_key(|(n, _)| *n);
        found.into_iter().map(|(_, text)| text).collect()
    }

    fn mask_pieces(&self) -> String {
        let mut found: Vec<(u64, &str)> = self
            .rest
            .iter()
            .filter_map(|(name, value)| {
                let n = name.strip_prefix("mask_image_")?.parse::<u64>().ok()?;
                Some((n, value.as_str()?))
            })
            .collect();
        found.sort_unstable_by_key(|(n, _)| *n);
        found.into_iter().map(|(_, text)| text).collect()
    }
}

/// One weight per latent row: the fraction of the row's pixel cell that the
/// mask covers. `rect` is `(x0, y0, x1, y1)` in pixels; `pixels` is a
/// `width x height` gray mask in `[0, 1]`.
fn mask_rows(
    grid_h: u32,
    grid_w: u32,
    cell_h: u32,
    cell_w: u32,
    rect: Option<(u32, u32, u32, u32)>,
    pixels: Option<&[f32]>,
) -> Vec<f32> {
    let width = (grid_w * cell_w) as usize;
    let mut rows = vec![0f32; (grid_h * grid_w) as usize];
    for a in 0..grid_h {
        for b in 0..grid_w {
            let (y0, x0) = (a * cell_h, b * cell_w);
            let mut sum = 0f32;
            for y in y0..y0 + cell_h {
                for x in x0..x0 + cell_w {
                    sum += match (rect, pixels) {
                        (Some((rx0, ry0, rx1, ry1)), _) => {
                            f32::from(u8::from(x >= rx0 && x < rx1 && y >= ry0 && y < ry1))
                        }
                        (None, Some(px)) => px[y as usize * width + x as usize],
                        (None, None) => 0.0,
                    };
                }
            }
            rows[(a * grid_w + b) as usize] = sum / (cell_h * cell_w) as f32;
        }
    }
    rows
}

fn parse_rect(text: &str) -> Result<(u32, u32, u32, u32)> {
    let parts: Vec<u32> = text
        .split(',')
        .map(|p| p.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|why| format!("`mask_rect` is `x0,y0,x1,y1` in pixels: {why}"))?;
    if parts.len() != 4 || parts[0] >= parts[2] || parts[1] >= parts[3] {
        return Err(format!(
            "`mask_rect` is `x0,y0,x1,y1` with x0 < x1 and y0 < y1, not `{text}`"
        ));
    }
    Ok((parts[0], parts[1], parts[2], parts[3]))
}

fn mask_stat(
    painted: &Option<Vec<f32>>,
    init: &Option<Vec<f32>>,
    last: &[f32],
    row_width: u32,
    pick: impl Fn(f32) -> bool,
    max: bool,
) -> Option<f32> {
    let (mask, init) = (painted.as_ref()?, init.as_ref()?);
    let w = row_width as usize;
    let mut acc = 0f32;
    let mut n = 0usize;
    for (row, weight) in mask.iter().enumerate() {
        if !pick(*weight) {
            continue;
        }
        for i in row * w..(row + 1) * w {
            let d = (last[i] - init[i]).abs();
            if max {
                acc = acc.max(d);
            } else {
                acc += d;
            }
            n += 1;
        }
    }
    Some(if max || n == 0 { acc } else { acc / n as f32 })
}

/// The report, which doubles as the SIDECAR of the raw-latent exit: every
/// number `scripts/imagegen/decode_latent.py` needs to turn the `.f32` blob
/// back into pixels, plus what the run actually did.
#[derive(Serialize)]
struct Output {
    model: String,
    architecture: String,
    reading: String,
    prompt: String,
    /// The pixel size actually rendered, after rounding to the grid.
    width: u32,
    height: u32,
    /// The latent-row grid and one row's width.
    grid_h: u32,
    grid_w: u32,
    rows: u32,
    row_width: u32,
    /// `model::latent()`, verbatim: what a row is in VAE terms.
    latent_channels: u32,
    patch_h: u32,
    patch_w: u32,
    spatial_compression: u32,
    steps: u32,
    seed: u32,
    guidance: f32,
    cfg: bool,
    sigmas: Vec<f32>,
    /// The prompt's row count out of the text reading.
    context_rows: u32,
    /// Whether the pixels went out as an encoded image (a `vae.decode`
    /// reading) or the latent went out raw.
    decoded: bool,
    /// The decode reading this model declares, if any. Present with
    /// `decoded: false` means the model states a pixels reading this run
    /// could not drive — no voxel port to hand the clip to, or a latent the
    /// loop left non-finite.
    decode_reading: Option<String>,
    /// The name the client should give what was sent.
    file: String,
    bytes: u32,
    /// Health of the final latent — a loop that diverged says so here.
    non_finite: u32,
    mean: f32,
    std: f32,
    /// Inpainting: rows the mask touches, the largest change the loop made to
    /// a fully kept row (0 when the kept region is the init bit for bit), and
    /// the mean change over fully repainted rows.
    mask_rows: Option<u32>,
    kept_max_abs: Option<f32>,
    painted_mean_abs: Option<f32>,
    /// Diagnostic (with `context_scale`): the prompt's token ids and each
    /// encoded row's mean magnitude.
    #[serde(skip_serializing_if = "Option::is_none")]
    context_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_head: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_dump: Option<Vec<f32>>,
}

/// The readings a text-to-image job needs, found by role.
struct Roles {
    text: model::ReadingFact,
    denoise: model::ReadingFact,
    decode: Option<model::ReadingFact>,
    encode: Option<model::ReadingFact>,
}

fn roles() -> Result<Roles> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err(
            "a DiT's readings are attention-kind passes with no kv bound; this model's \
                    pass kind is not attention"
                .into(),
        );
    }
    let readings = model::readings();
    if readings.is_empty() {
        return Err("this model declares no readings; there is nothing here to draw with".into());
    }
    let text = readings
        .iter()
        .find(|r| r.takes_tokens && r.readout == model::ReadoutKind::Hidden)
        .cloned()
        .ok_or("this model has no text reading")?;
    let denoise = readings
        .iter()
        .find(|r| !r.takes_tokens && r.readout == model::ReadoutKind::Velocity)
        .cloned()
        .ok_or(
            "this model declares no token-less reading with a velocity readout, so nothing \
                here denoises",
        )?;
    if denoise.has_kv {
        return Err(format!(
            "reading `{}` binds a kv space; a denoise pass binds none",
            denoise.name
        ));
    }
    // The decoder, by role: a `Pixels` readout is what says "this arm ends
    // in pixels", and the one other reading that carries it is the ENCODER,
    // whose pixels are its input and whose readout is latent rows. The name
    // separates them, and it is the design's own vocabulary (D1's reading
    // list), not a family's.
    let decode = readings
        .iter()
        .find(|r| {
            r.name == "vae.decode"
                || (r.readout == model::ReadoutKind::Pixels && r.name != "vae.encode")
        })
        .cloned();
    // And the ENCODER: the reading whose pixels are its INPUT, which is
    // what a voxel port beside a non-pixels readout says. img2img comes in
    // through it.
    let encode = readings
        .iter()
        .find(|r| {
            r.name == "vae.encode"
                || (r.readout != model::ReadoutKind::Pixels
                    && r.ports.iter().any(|p| p.kind == model::PortKind::Voxels))
        })
        .cloned();
    Ok(Roles {
        text,
        denoise,
        decode,
        encode,
    })
}

/// The denoise reading's ports, sorted into the roles a sampler binds. Kind
/// is what says which is which (`PortKind` is the facts vocabulary); the
/// one name read here is `guidance`, which the design lists as a port role
/// of its own and which decides whether the model wants a scale or a second
/// lane pair.
struct Ports {
    latents: model::PortFact,
    context: model::PortFact,
    timestep: String,
    guidance: Option<String>,
    /// The convention `positions_for` fills the grids from.
    convention: model::PositionConvention,
    /// Which stream carries the latent, and which carries the prompt rows.
    image_stream: model::LaneStream,
    context_stream: model::LaneStream,
}

/// The first stream of `streams`, preferring `want` when it is listed;
/// `fallback` when the port lists none (a port every lane binds).
fn stream_of(
    streams: &[model::LaneStream],
    want: model::LaneStream,
    fallback: model::LaneStream,
) -> model::LaneStream {
    if streams.contains(&want) {
        return want;
    }
    streams.first().copied().unwrap_or(fallback)
}

fn ports(reading: &model::ReadingFact) -> Result<Ports> {
    let of_kind = |kind: model::PortKind| reading.ports.iter().find(|p| p.kind == kind).cloned();
    let latents = of_kind(model::PortKind::Latents)
        .ok_or("this denoise reading declares no latents port; there is nothing to denoise")?;
    let context = of_kind(model::PortKind::Context).ok_or(
        "this denoise reading declares no context port; a text-conditioned sampler has nowhere \
         to put the prompt",
    )?;
    let positions = of_kind(model::PortKind::AxisPositions).ok_or(
        "this denoise reading declares no axis-positions port; this sampler builds rotary \
         coordinates and the model takes none",
    )?;
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
    let image_stream = stream_of(
        &latents.streams,
        model::LaneStream::Image,
        model::LaneStream::Image,
    );
    let context_stream = stream_of(
        &context.streams,
        model::LaneStream::Text,
        model::LaneStream::Context,
    );
    Ok(Ports {
        latents,
        context,
        timestep,
        guidance,
        convention,
        image_stream,
        context_stream,
    })
}

/// Everything one lane binds, kept alive for the loop's lifetime: a channel
/// dropped while its pass is live is a port with nothing feeding it.
struct Lane {
    pass: ForwardPass,
    #[allow(dead_code)]
    held: Vec<Channel>,
}

/// Build one lane of a denoise group: state its stream and group, then bind
/// EXACTLY the ports that list its stream (or list none), each from its
/// kind. `latents` is the lane's own rows when it is the image lane;
/// `context` is the encoder's rows when it is the prompt lane. A port of a
/// kind this lane has no value for — Z-Image's `pad` flags, a second
/// context — is bound to zeros of its declared width, which is what "no
/// pad rows / no extra conditioning" means.
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
            // A voxel port is a VAE tile's box on the third row axis
            // (D8), not something a denoise lane carries. If a denoise
            // reading ever declares one, this program is the wrong driver
            // for it and says so rather than feeding it zeros.
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

/// Unpatchify a denoise reading's final latent into the clip a `vae.decode`
/// port takes: `[rows, C·ph·pw]` in `(c, ph, pw)` feature order becomes
/// `[grid_h·ph, grid_w·pw, C]` row-major, which is the `[h, w, C]` shape a
/// `Voxels` port channel declares (design D8).
///
/// The inverse of the patchify every DiT does at its embed, written out
/// rather than called, because the guest is the only place both halves of
/// the pair are visible: the trunk states the patch and the VAE states the
/// clip, and nothing between them holds the index algebra.
fn unpatchify(
    rows: &[f32],
    grid_h: u32,
    grid_w: u32,
    patch_h: u32,
    patch_w: u32,
    channels: u32,
) -> Vec<f32> {
    let (gh, gw) = (grid_h as usize, grid_w as usize);
    let (ph, pw) = (patch_h as usize, patch_w as usize);
    let c = channels as usize;
    let (h, w) = (gh * ph, gw * pw);
    let row_width = c * ph * pw;
    let mut out = vec![0f32; h * w * c];
    for a in 0..gh {
        for b in 0..gw {
            let row = (a * gw + b) * row_width;
            for ci in 0..c {
                for y in 0..ph {
                    for x in 0..pw {
                        let src = row + ci * ph * pw + y * pw + x;
                        let dst = (((a * ph + y) * w) + (b * pw + x)) * c + ci;
                        out[dst] = rows[src];
                    }
                }
            }
        }
    }
    out
}

/// Standard base64 (`+/`, `=` padded) into the bytes it spells. The one
/// thing that travels this way is an init image: argv is text, an encoded
/// still is not, and a scratch-file door would make the picture a path
/// rather than a value.
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
        let six = code(c).ok_or_else(|| format!("the init image: byte {i} is not base64"))?;
        acc = (acc << 6) | six;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Ok(bytes)
}

/// The inverse: a `[h, w, C]` clip back into the `[rows, C.ph.pw]` rows a
/// denoise reading's latent port takes, in `(c, ph, pw)` feature order.
///
/// The pair to [`unpatchify`], and here for the same reason: img2img starts
/// from a picture, so the clip a `vae.encode` reading LANDS has to become
/// the rows the sampler steps, and the guest is the only place both halves
/// of the index algebra are visible.
fn patchify(
    clip: &[f32],
    grid_h: u32,
    grid_w: u32,
    patch_h: u32,
    patch_w: u32,
    channels: u32,
) -> Vec<f32> {
    let (gh, gw) = (grid_h as usize, grid_w as usize);
    let (ph, pw) = (patch_h as usize, patch_w as usize);
    let c = channels as usize;
    let w = gw * pw;
    let row_width = c * ph * pw;
    let mut out = vec![0f32; gh * gw * row_width];
    for a in 0..gh {
        for b in 0..gw {
            let row = (a * gw + b) * row_width;
            for ci in 0..c {
                for y in 0..ph {
                    for x in 0..pw {
                        let dst = row + ci * ph * pw + y * pw + x;
                        let src = (((a * ph + y) * w) + (b * pw + x)) * c + ci;
                        out[dst] = clip[src];
                    }
                }
            }
        }
    }
    out
}

/// Fire the family's `vae.encode` reading over one picture and hand back the
/// latent clip it lands: the mirror of [`decode_to_frames`], and the door
/// img2img comes in through.
///
/// The picture goes in as the port's CHANNEL, seeded straight from the
/// host's `frames` handle (`Channel::set_frames`) so the pixels never enter
/// this module's address space, and the answer comes back on the same
/// `pixels` seam a decode uses — for an encoder that seam carries the
/// POSTERIOR MEAN, `[clip rows, latent channels]`, which is what the
/// reference pipeline's `retrieve_latents` takes when it samples nothing.
async fn encode_from_frames(
    reading: &model::ReadingFact,
    picture: &inferlet::frames::Frames,
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
                "reading `{}` takes pixels but declares no voxel port; there is no \
                 picture to hand it",
                reading.name
            )
        })?;
    let rows = clip_h * clip_w;
    let width = reading.readout_width;
    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.stream(model::LaneStream::Image)?;
    // Empty, and SEEDED from the picture below: the pixels are the host's
    // and never enter linear memory, which is the whole point of the
    // `frames` handle.
    let clip = Channel::seeded([pixel_h, pixel_w, port.width], dtype::f32).named("vae_pixels_in");
    pass.input(&port.name, &clip)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_latent_out");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(rows, width));
    });
    // THE PICTURE, seeded straight from the host's handle.
    clip.set_frames(picture)?;
    pass.submit(pipe).context("the vae.encode lane")?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

/// Fire the family's `vae.decode` reading over one clip and hand the pixels
/// straight to the host's encoders (design D8 into D11).
///
/// The latent goes in as the port's CHANNEL — whose shape IS the clip's box,
/// so the geometry the voxel axis needs travels with the numbers — and the
/// answer comes back on the `pixels` seam, which the epilogue reads with
/// `intrinsics::pixels()` and puts on one channel. `Channel::take_frames`
/// turns that channel's cell into a `frames` handle host-side, so the picture
/// never enters this module's address space on its way to the client.
async fn decode_to_frames(
    reading: &model::ReadingFact,
    latent: &[f32],
    clip_h: u32,
    clip_w: u32,
    pixel_h: u32,
    pixel_w: u32,
    pipe: &Pipeline,
) -> Result<inferlet::frames::Frames> {
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
    if latent.len() != want {
        return Err(format!(
            "the unpatchified latent is {} numbers and port `{}` wants {clip_h}x{clip_w}x{} \
             = {want}",
            latent.len(),
            port.name,
            port.width
        )
        .into());
    }
    let rows = pixel_h * pixel_w;
    let width = reading.readout_width;
    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.stream(model::LaneStream::Image)?;
    let clip = Channel::from_shaped([clip_h, clip_w, port.width], latent).named("vae_latent");
    pass.input(&port.name, &clip)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_pixels");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(rows, width));
    });
    pass.submit(pipe).context("the vae.decode lane")?;
    out.take_frames(pixel_w, pixel_h, 1, 0.0)
        .map_err(Into::into)
}

/// One prompt's two lanes: the encoder rows and the latent they condition.
struct Branch {
    context: Lane,
    image: Lane,
    /// `set` by the host on the CFG path).
    /// the plain path, this fire's velocity on the CFG path.
    out: Channel,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let prompt = input
        .prompt
        .clone()
        .filter(|p| !p.trim().is_empty())
        .ok_or("pass `prompt`: this program draws what it is told to")?;
    let roles = roles()?;
    let ports = ports(&roles.denoise)?;

    // ---- the latent grid, out of `model::latent()` ------------------------
    let space = model::latent().ok_or(
        "this model states no latent space, so its denoise reading cannot be sized from pixels",
    )?;
    let cell_h = space.spatial_compression.max(1) * space.patch_h.max(1);
    let cell_w = space.spatial_compression.max(1) * space.patch_w.max(1);
    let height = (input.height.unwrap_or(1024) / cell_h).max(1) * cell_h;
    let width = (input.width.unwrap_or(1024) / cell_w).max(1) * cell_w;
    let grid_h = height / cell_h;
    let grid_w = width / cell_w;
    let rows = grid_h * grid_w;
    let row_width = ports.latents.width;
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && rows > max_rows {
        return Err(format!(
            "{width}x{height} is {rows} latent rows and this model carries {max_rows} a pass"
        )
        .into());
    }

    // ---- the schedule -----------------------------------------------------
    let fact = model::schedule().ok_or("this model states no schedule; nothing here denoises")?;
    // A DISTILLED ROW STATES ITS OWN STEP COUNT. `pinned_sigmas` is the
    // trajectory the model was distilled onto, so its length is the default;
    // an undistilled row gets eight, which is a policy and not a fact.
    let steps = match input.steps.filter(|s| *s > 0) {
        Some(steps) => steps,
        None if !fact.pinned_sigmas.is_empty() => {
            u32::try_from(fact.pinned_sigmas.len()).unwrap_or(8)
        }
        None => 8,
    };
    let full = FlowMatchEuler::from_schedule(&fact, steps, Some(rows))?;

    // ---- img2img: a picture, encoded, and a schedule cut to `strength` ----
    //
    // The picture comes in as an encoded still and goes out to the host's
    // decoder without entering this module's address space: `Frames::decode`
    // holds it, `Channel::set_frames` seeds the pixel port from it, and the
    // encode lane's epilogue lands the posterior mean. `strength` is the
    // FRACTION OF THE TRAJECTORY still to run, so the schedule is truncated
    // to its tail and the picture is noised to that tail's first sigma.
    let strength = input.strength.unwrap_or(0.6);
    let has_init = input.init_image.is_some() || !input.image_pieces().is_empty();
    if has_init && !(0.0 < strength && strength <= 1.0) {
        return Err(format!("`strength` is {strength}; it is a fraction in (0, 1]").into());
    }
    let cut = if has_init {
        // `steps - round(steps * strength)`: the number of steps SKIPPED.
        // Clamped to leave at least one, since a schedule of no steps is a
        // picture handed straight back.
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

    // ---- guidance: a port, or a second lane pair --------------------------
    let guidance = input.guidance.unwrap_or(1.0);
    let negative = input
        .negative_prompt
        .clone()
        .filter(|p| !p.trim().is_empty());
    if negative.is_some() && ports.guidance.is_some() {
        return Err(format!(
            "reading `{}` declares a `guidance` port, so this row is guidance-distilled and a \
             negative prompt has no lane to ride; pass `guidance` alone",
            roles.denoise.name
        )
        .into());
    }
    let cfg = negative.is_some() && guidance > 1.0;

    // ---- the prompt -------------------------------------------------------
    let context = encode_prompt(&prompt, &roles.text.name, input.context_scale).await?;
    let mut context_ids: Option<Vec<u32>> = None;
    let mut context_rows_dump: Option<Vec<f32>> = None;
    let context_probe: Option<Vec<f32>> = match input.context_scale {
        Some(_) => {
            let mut ids = inferlet::chat::first_user(&prompt);
            ids.extend(inferlet::chat::cue());
            let (rows, width) = encode_ids_rows(&ids, &roles.text.name).await?;
            let w = width as usize;
            println!("{}", inferlet::serde_json::json!({ "context_ids": ids }));
            let mut bytes = Vec::with_capacity(rows.len() * 4);
            for v in &rows {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            inferlet::session::send_file_as(&bytes, "context_dump.f32");
            context_ids = Some(ids);
            context_rows_dump = Some(rows.clone());
            Some(
                rows.chunks(w)
                    .map(|row| row.iter().map(|v| v.abs()).sum::<f32>() / w as f32)
                    .collect(),
            )
        }
        None => None,
    };
    let context_rows = context.shape().dims()[0];
    if context.shape().dims()[1] != ports.context.width {
        return Err(format!(
            "the text reading hands back {}-wide rows and the denoise context port takes {}",
            context.shape().dims()[1],
            ports.context.width
        )
        .into());
    }
    let uncond = match &negative {
        Some(text) if cfg => {
            Some(encode_prompt(text, &roles.text.name, input.context_scale).await?)
        }
        _ => None,
    };

    // ---- the picture, encoded -------------------------------------------
    //
    // Before the loop, on a pipeline of its own that closes: the encode lane
    // holds a seat and the denoise lanes need it back.
    let pieced = input.image_pieces();
    let init_bytes: Option<Vec<u8>> = match (&input.init_image, pieced.is_empty()) {
        (Some(_), false) => {
            return Err("pass `init_image` whole or as `init_image_<n>` pieces, not both".into());
        }
        (Some(b64), true) => Some(b64_decode(b64)?),
        (None, false) => Some(b64_decode(&pieced)?),
        (None, true) => None,
    };
    let init_latent: Option<Vec<f32>> = match &init_bytes {
        None => None,
        Some(bytes) => {
            let reading = roles.encode.as_ref().ok_or_else(|| {
                "this model declares no `vae.encode` reading, so it has no door a picture                  comes in through"
                    .to_string()
            })?;
            let picture = inferlet::frames::Frames::decode(bytes)
                .map_err(|why| format!("the init image does not decode: {why}"))?;
            let (pw, ph) = (picture.width(), picture.height());
            if pw != width || ph != height {
                return Err(format!(
                    "the init image is {pw}x{ph} and this run is {width}x{height}; resize it                      before handing it over — a guest has no resampler and a silent one                      would be the wrong one"
                )
                .into());
            }
            let pipe = Pipeline::new();
            let clip = encode_from_frames(
                reading,
                &picture,
                ph,
                pw,
                grid_h * space.patch_h.max(1),
                grid_w * space.patch_w.max(1),
                &pipe,
            )
            .await?;
            pipe.close();
            // The clip the encoder lands is `[h, w, C]`; the sampler steps
            // `[rows, C.ph.pw]`, so it is patchified the way the trunk's own
            // embed would.
            Some(patchify(
                &clip,
                grid_h,
                grid_w,
                space.patch_h.max(1),
                space.patch_w.max(1),
                space.channels,
            ))
        }
    };

    // ---- the loop ---------------------------------------------------------
    let mask_bytes = input.mask_pieces();
    let painted: Option<Vec<f32>> = if input.mask_rect.is_none() && mask_bytes.is_empty() {
        None
    } else {
        if init_latent.is_none() {
            return Err(
                "a mask says what to repaint; pass `init_image` for the picture it is cut from"
                    .into(),
            );
        }
        let rect = input.mask_rect.as_deref().map(parse_rect).transpose()?;
        let gray: Option<Vec<f32>> = if rect.is_some() || mask_bytes.is_empty() {
            None
        } else {
            let mask = inferlet::frames::Frames::decode(&b64_decode(&mask_bytes)?)
                .map_err(|why| format!("the mask image does not decode: {why}"))?;
            if mask.width() != width || mask.height() != height {
                return Err(format!(
                    "the mask is {}x{} and this run is {width}x{height}",
                    mask.width(),
                    mask.height()
                )
                .into());
            }
            let rgb = mask
                .encode(inferlet::frames::ImageFormat::RawRgb8)
                .map_err(|why| format!("frames.encode(raw-rgb8): {why}"))?;
            Some(rgb.chunks(3).map(|px| f32::from(px[0]) / 255.0).collect())
        };
        Some(mask_rows(
            grid_h,
            grid_w,
            cell_h,
            cell_w,
            rect,
            gray.as_deref(),
        ))
    };

    let mut loops = DenoiseLoop::new(&sched);
    let velocity_width = roles.denoise.readout_width;
    let cells = (rows * row_width) as usize;
    let shape = [rows, row_width];
    let seed = input.seed.unwrap_or(0);

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
        let image_clock = loops.lane(&format!("{tag}_img"));
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
            ports.image_stream,
            group,
            cfg.then(|| 1 - group),
            rows,
            Some(&latent),
            None,
            &positions_for(
                &ports.convention,
                LaneRows::Grid {
                    h: grid_h,
                    w: grid_w,
                },
                ctx_rows,
            ),
            &image_clock,
            guidance,
            &format!("{tag}_img"),
        )?;
        let out = Channel::new(shape, dtype::f32)
            .capacity(channel_capacity() as u32)
            .named(&format!("{tag}_out"));

        // A lane that only modulates advances its clock and nothing else.
        text_clock.drive(&context_lane.pass, |_| {});
        let dts = loops.dts(tag);
        let rng = Channel::from(rng_state(seed)).named(&format!("{tag}_rng"));
        let x = latent.clone();
        let readback = out.clone();
        // IMG2IMG: the encoded picture, seeded once, read by every fire and
        // used by fire 0. Its own channel per branch, since a seeded channel
        // attaches to one pass (the runtime's channel-role rule).
        let init = init_latent
            .as_ref()
            .map(|rows| Channel::from_shaped(shape, rows.as_slice()).named(&format!("{tag}_init")));
        let mask = painted
            .as_ref()
            .map(|m| Channel::from_shaped([rows], m.as_slice()).named(&format!("{tag}_mask")));
        let sigmas = loops.sigmas(tag);
        image_clock.drive(&image_lane.pass, move |k| {
            // On the CFG path BOTH branches step with the SAME combine, each
            // computed from its own side: the conditional lane reads
            // `(own = c, peer = u)` and the unconditional one
            // `(own = u, peer = c)`, and `u + s(c - u)` is the same number
            // either way. So the two latents stay in step without either
            // one crossing the host, and the seeded device draw is the same
            // draw the ungated path makes.
            let v = if cfg {
                guided_velocity(velocity_width, guidance, group == 0)
            } else {
                intrinsics::velocity(velocity_width)
            };
            // Fire 0 either draws the latent from the keyed RNG, or —
            // img2img — takes the encoded picture noised to the truncated
            // schedule's first sigma. Every later fire is the same Euler
            // step either way.
            match (&init, &mask) {
                (None, _) => seed_or_step(k, &x, &v, &dts, &rng, shape, Some(&readback)),
                (Some(init), None) => {
                    resume_or_step(k, &x, init, sigma0, &v, &dts, &rng, shape, Some(&readback))
                }
                (Some(init), Some(mask)) => inpaint_step(
                    k,
                    &x,
                    init,
                    mask,
                    &sigmas,
                    &v,
                    &dts,
                    &rng,
                    shape,
                    Some(&readback),
                ),
            }
        });
        branches.push(Branch {
            context: context_lane,
            image: image_lane,
            out,
        });
    }

    // Both paths are the same loop now: the device seeds its own latent from
    // the keyed RNG and integrates its own Euler step, guided or not. The
    // guided branches read each other's velocity off the fire's own plane,
    // so nothing crosses the host between steps and the same seed draws the
    // same latent either way.
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

    // ---- the way out ------------------------------------------------------
    let n = last.len().max(1) as f32;
    let mean = last.iter().sum::<f32>() / n;
    let var = last.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let name = input.out.unwrap_or_else(|| "image".to_string());
    // THE DECODE BRANCH IS A BRANCH, NOT A REFUSAL. A model that declares a
    // `vae.decode` reading with a voxel port and a pixels readout gets D11's
    // exit: the latent is unpatchified into the clip the port takes, the
    // reading fires on the voxel axis, and the pixels go from the `pixels`
    // seam into the host's PNG encoder without ever entering linear memory.
    // A model that declares no such reading — because its VAE is traced but
    // not stated as a reading, or because it has none — takes the latent
    // exit as before, and the report says which reading it saw.
    let decode_reading = roles.decode.as_ref().map(|r| r.name.clone());
    let (file, decoded, bytes) = match roles.decode.as_ref() {
        Some(reading)
            if reading
                .ports
                .iter()
                .any(|p| p.kind == model::PortKind::Voxels)
                && last.iter().all(|v| v.is_finite()) =>
        {
            let clip = unpatchify(
                &last,
                grid_h,
                grid_w,
                space.patch_h.max(1),
                space.patch_w.max(1),
                space.channels,
            );
            // A pipeline of its own: the denoise loop's are closed, and a
            // VAE tile is one fire that shares nothing with them.
            let vae_pipe = Pipeline::new();
            let handle = decode_to_frames(
                reading,
                &clip,
                grid_h * space.patch_h.max(1),
                grid_w * space.patch_w.max(1),
                height,
                width,
                &vae_pipe,
            )
            .await?;
            vae_pipe.close();
            let png = format!("{name}.png");
            inferlet::session::send_frames(&handle, inferlet::frames::ImageFormat::Png, &png)
                .map_err(|why| format!("session.send-frames: {why}"))?;
            // The bytes never crossed, so there is no length to report; the
            // picture's size is what the client got.
            (png, true, 0)
        }
        _ => {
            let mut bytes = Vec::with_capacity(last.len() * 4);
            for value in &last {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
            // NAMED, like the decode arm above. `send_file` carries bytes and
            // nothing else, so `pie run -o DIR` could only write
            // `file-0000.bin` and the caller had to read the report to learn
            // what it had -- a worse answer on the arm that already needs a
            // second tool to finish the job.
            let file = format!("{}.latent.f32", name.trim_end_matches(".png"));
            inferlet::session::send_file_as(&bytes, &file);
            (file, false, len)
        }
    };

    Ok(Output {
        model: model::name(),
        architecture: model::architecture(),
        reading: roles.denoise.name.clone(),
        prompt,
        width,
        height,
        grid_h,
        grid_w,
        rows,
        row_width,
        latent_channels: space.channels,
        patch_h: space.patch_h,
        patch_w: space.patch_w,
        spatial_compression: space.spatial_compression,
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
        sigmas: sched.sigmas.clone(),
        context_rows,
        decoded,
        decode_reading,
        file,
        bytes,
        non_finite: last.iter().filter(|v| !v.is_finite()).count() as u32,
        mask_rows: painted
            .as_ref()
            .map(|m| m.iter().filter(|w| **w > 0.0).count() as u32),
        kept_max_abs: mask_stat(&painted, &init_latent, &last, row_width, |w| w == 0.0, true),
        painted_mean_abs: mask_stat(
            &painted,
            &init_latent,
            &last,
            row_width,
            |w| w == 1.0,
            false,
        ),
        context_ids,
        context_head: context_probe,
        context_dump: context_rows_dump,
        mean,
        std: var.sqrt(),
    })
}
