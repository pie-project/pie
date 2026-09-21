//! The pie half of the HunyuanImage 3 miniature golden: the two TOKEN-AXIS
//! readings of `hunyuan_image_3`, driven exactly as
//! `scripts/imagegen/hy3_golden.py --mini` drove the reference — and two
//! claims about the thing this family exists to do.
//!
//! # One whole denoise step, in three fires
//!
//! A denoise step of this family is THREE fires (`crates/models/src/
//! hunyuan_image_3/forward.rs` says why the conv image head cannot share
//! the trunk's plan unit), and this program runs all three end to end —
//! each one fed pie's OWN answer to the one before, so the velocity it
//! lands is the step:
//!
//! ```text
//! encode     the prompt prefix, causal, its K/V ARE the sequence -> logits
//! image.in   the noisy latent as a `[h, w, 32]` CLIP on the voxel axis,
//!            through `patch_embed`'s 3x3 convs and its adaptive group
//!            norm                                        -> pixels [h*w, D]
//! denoise    the canvas: those h*w rows through the `latents` port plus
//!            ONE <timestep> row the plan lands from the lane's timestep
//!            vector, bidirectional over [prefix | canvas] against the
//!            pages `encode` wrote                         -> hidden
//! image.out  those rows back on the voxel axis through `final_layer`
//!                                                        -> pixels [h*w, 32]
//! ```
//!
//! The two voxel arms bind their clip as a CHANNEL whose shape is the box
//! (`[h, w, C + 256]`, design D8) and read their seam with
//! `intrinsics::pixels(rows, width)`. The timestep's sinusoid rides in the
//! SAME rectangle, appended to every voxel row: the CUDA shell seats one
//! voxel width a fire and refuses a lane that feeds two, and a voxel-axis
//! lane broadcast (which is what an adaGN vector wants) does not exist.
//! The one host round trip is between
//! `denoise` and `image.out`: the trunk hands back `[h*w + 1, D]` token
//! rows and the voxel arm wants `[h, w, D]` without the `<timestep>` row,
//! and dropping a row is not something this program can spell on the
//! device. A production guest would carry it in an epilogue.
//!
//! # Four fires, three claims
//!
//! ```text
//! A  prefill(prompt)        -> denoise(t0)   the answer the golden gates
//! B  prefill(<cfg>-masked)  -> denoise(t0)   THE PREFIX MATTERS
//! C  A's working set again  -> denoise(t1)   THE PREFIX PAGES ARE FROZEN
//! D  prefill(prompt) fresh  -> denoise(t1)   ... and C is exactly D
//! ```
//!
//! (A, B, C and D below are the DENOISE fires; `image.in` runs once and
//! `image.out` once, over A's rows.)
//!
//! **B is the claim that the canvas is CONDITIONED.** A denoise fire whose
//! image rows silently attended only themselves would still match a
//! random-init golden inside the tolerance — the conditioning is a small
//! part of a small model's answer — and would be catastrophically wrong on
//! the real row. So B re-prefills the reference's own unconditional branch
//! (every prompt token replaced by `<cfg>`, the image-meta tokens kept, so
//! the length, the mask and the rotary positions are identical) and the
//! harness asserts the canvas MOVED by far more than the gate.
//!
//! **C and D are design D10's exactness claim.** After step 0 the prefix
//! K/V is never recomputed: C fires a second denoise over A's pages, whose
//! canvas slots step 0 already wrote and whose prefix pages nothing has
//! touched. D does the same step from a fresh prefill. The two must agree
//! to the last bit, and the harness gates on it.
//!
//! # The canvas lane
//!
//! `h*w + 1` rows, the `<timestep>` row FIRST: the CUDA fire path derives a
//! lane's KV positions as `held .. held + rows` and refuses an explicit
//! list by name, so the lane's rows must BE the sequence's own run. The
//! `special` port flags the row the plan overwrites with `timestep_emb(t)`;
//! every other row takes its `latents` row.
//!
//! The mask is the reference's generalized causal attention, built here as
//! the dense `[rows, keys]` bool rectangle the engine's `AttnMask` port
//! takes: an image row sees every key of `[prefix | <timestep> | canvas]`,
//! the `<timestep>` row sees the prefix and itself. Nothing past the canvas
//! is in the KV at all, so the reference's trailing `<eoi>` needs no
//! masking off.

use inferlet::eta::diffusion::prelude::*;
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
}

/// The golden's inputs, flattened row-major.
#[derive(Deserialize)]
struct Case {
    /// The text prefix's ids: `<bos> … <boi> <img_size> <img_ratio>`.
    prefix: Vec<i32>,
    /// The same span with every PROMPT token replaced by `<cfg>` — the
    /// reference's own unconditional branch, and this program's control.
    prefix_alt: Vec<i32>,
    /// `[prefix.len(), 2]` — `(y, x')`, the `x` already through the
    /// family's rotary scale. The same for both prefixes: `<cfg>` keeps the
    /// length, so it keeps the positions.
    prefix_positions: Vec<f32>,
    /// The `<img>` id and the `<timestep>` id.
    img_id: i32,
    timestep_id: i32,
    /// `h·w`.
    image_rows: u32,
    /// `[image_rows + 1, 2]`, the canvas lane's row order: the `<timestep>`
    /// row, then the image rows in raster order.
    canvas_positions: Vec<f32>,
    /// `[image_rows + 1]`: the SEQUENCE position of each canvas row.
    canvas_sequence: Vec<u32>,
    /// The clip's box: `image_rows == token_h * token_w`.
    token_h: u32,
    token_w: u32,
    /// `[token_h * token_w, latent_channels]` — the noisy latent x_t in
    /// raster order, what `image.in` convolves.
    latent: Vec<f32>,
    latent_channels: u32,
    /// `timestep_embedding(t, 256)`, one row, replicated over the clip's
    /// voxels by this program: the two adaGN embedders read it there
    /// because a voxel-axis lane vector has no broadcast (see
    /// `model::port::TFREQ_VOXELS`).
    tfreq: Vec<f32>,
    hidden: u32,
    /// The scheduler timestep `σ·1000` of step 0, and of the step the two
    /// KV-reuse fires run.
    timestep: f32,
    timestep_next: f32,
}

#[derive(Serialize)]
struct Output {
    /// `[prefix.len()]`: the argmax of the causal pass's logits, and the
    /// logit it won with.
    encode_argmax: Vec<i32>,
    encode_max: Vec<f32>,
    /// `[image_rows + 1, hidden]` in the canvas lane's row order — the
    /// trunk's rows with `ln_f` NOT applied, which is what `final_layer`
    /// consumes. `A` is what the golden gates.
    canvas_hidden: Vec<f32>,
    /// The same fire over the `<cfg>`-masked prefix: the harness asserts
    /// this is FAR from `canvas_hidden`.
    canvas_hidden_uncond: Vec<f32>,
    /// Step 1 over the frozen prefix pages, and step 1 from a fresh
    /// prefill: the harness asserts these agree exactly.
    canvas_hidden_reused: Vec<f32>,
    canvas_hidden_fresh: Vec<f32>,
    /// `[image_rows, hidden]` — `patch_embed(x_t, time_embed(t))` off the
    /// voxel axis, which is what the canvas lane was then fed.
    image_in_rows: Vec<f32>,
    /// `[image_rows, latent_channels]` — the flow-matching velocity, the
    /// whole step's answer.
    velocity: Vec<f32>,
    image_rows: u32,
    hidden: u32,
    latent_channels: u32,
}

fn reading(name: &'static str) -> Result<model::ReadingFact> {
    model::readings()
        .into_iter()
        .find(|reading| reading.name == name)
        .ok_or_else(|| format!("this model declares no `{name}` reading").into())
}

fn port_of(reading: &model::ReadingFact, name: &str) -> Result<String> {
    reading
        .ports
        .iter()
        .find(|port| port.name == name)
        .map(|port| port.name.clone())
        .ok_or_else(|| format!("reading `{}` declares no port `{name}`", reading.name).into())
}

/// The geometry every fire of this program shares.
struct Shape {
    axes: u32,
    prefix_len: u32,
    canvas_len: u32,
    kv_len: u32,
    page_size: u32,
    max_pages: u32,
    /// The sequence index of the `<timestep>` row — the causal bound of the
    /// one row that is not bidirectional.
    t_at: u32,
}

/// One causal `encode` fire over `ids`, whose K/V become the sequence.
/// Answers the per-row argmax and winning logit.
fn prefill(
    case: &Case,
    shape: &Shape,
    encode: &model::ReadingFact,
    ws: &WorkingSet,
    pipe: &Pipeline,
    ids: &[i32],
    tag: &str,
) -> Result<(Channel, Channel)> {
    let n = shape.prefix_len;
    let toks = Channel::from(ids.to_vec()).named(&format!("prefix_ids_{tag}"));
    let indptr = Channel::from([0u32, n]).named(&format!("prefix_indptr_{tag}"));
    let kv_positions = Channel::from_iter(0..n).named(&format!("prefix_kv_positions_{tag}"));
    let rope = Channel::from_shaped([n, shape.axes], case.prefix_positions.as_slice())
        .named(&format!("prefix_positions_{tag}"));
    let pages = Channel::from_iter(0..shape.max_pages).named(&format!("prefix_pages_{tag}"));
    let page_indptr = Channel::from([0u32, n.div_ceil(shape.page_size)])
        .named(&format!("prefix_page_indptr_{tag}"));
    let slot = Channel::from_iter((0..n).map(|p| p / shape.page_size))
        .named(&format!("prefix_w_slot_{tag}"));
    let off = Channel::from_iter((0..n).map(|p| p % shape.page_size))
        .named(&format!("prefix_w_off_{tag}"));
    let kv_len = Channel::from([n]).named(&format!("prefix_kv_len_{tag}"));
    let readout = Channel::from_iter(0..n).named(&format!("prefix_readout_{tag}"));
    let argmax = Channel::new([n], dtype::i32).named(&format!("encode_argmax_{tag}"));
    let max = Channel::new([n], dtype::f32).named(&format!("encode_max_{tag}"));

    let pass = ForwardPass::new();
    pass.reading(&encode.name)?;
    pass.canvas(Mode::Encode)?;
    pass.embed(&toks, &indptr)?;
    pass.input(&port_of(encode, "positions")?, &rope)?;
    pass.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &slot,
            w_off: &off,
            positions: &kv_positions,
            mask: None,
        },
    )?;
    // Every row reads out: the default is the lane's LAST row, and the
    // golden's logits are the whole causal pass.
    pass.readout(&readout)?;
    let (argmax_out, max_out) = (argmax.clone(), max.clone());
    pass.epilogue(move || {
        let logits = intrinsics::logits();
        argmax_out.put(cast(reduce_argmax(&logits), dtype::i32));
        max_out.put(reduce_max(&logits));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{tag}` prefill"))?;
    Ok((argmax, max))
}

/// One `denoise` fire over `ws`'s pages at `timestep`: the canvas lane's
/// `h*w + 1` rows, read back on the `hidden` seam.
#[allow(clippy::too_many_arguments)]
fn denoise_step(
    case: &Case,
    shape: &Shape,
    denoise: &model::ReadingFact,
    ws: &WorkingSet,
    pipe: &Pipeline,
    image_rows: &[f32],
    timestep: f32,
    tag: &str,
) -> Result<Channel> {
    let canvas_len = shape.canvas_len;
    let kv_len = shape.kv_len;
    let mut ids: Vec<i32> = vec![case.timestep_id];
    ids.extend(std::iter::repeat_n(case.img_id, case.image_rows as usize));
    let special: Vec<f32> = (0..canvas_len).map(|row| f32::from(row == 0)).collect();
    // `[canvas_len, hidden]`: one row the plan overwrites (its bytes are
    // never read), then the image head's rows.
    let mut rows = vec![0.0f32; case.hidden as usize];
    rows.extend_from_slice(image_rows);
    // The generalized causal mask, dense `[rows, keys]` row-major: an image
    // row sees every key, the `<timestep>` row sees itself and the prefix.
    let mask: Vec<bool> = (0..canvas_len)
        .flat_map(|row| {
            let bound = if row == 0 { shape.t_at } else { kv_len - 1 };
            (0..kv_len).map(move |key| key <= bound)
        })
        .collect();

    let name = |what: &str| format!("canvas_{what}_{tag}");
    let toks = Channel::from(ids).named(&name("ids"));
    let indptr = Channel::from([0u32, canvas_len]).named(&name("indptr"));
    let latents =
        Channel::from_shaped([canvas_len, case.hidden], rows.as_slice()).named(&name("rows"));
    let flag = Channel::from_shaped([canvas_len, 1], special.as_slice()).named(&name("special"));
    let t = Channel::from([timestep]).named(&name("timestep"));
    let rope = Channel::from_shaped([canvas_len, shape.axes], case.canvas_positions.as_slice())
        .named(&name("positions"));
    let kv_positions = Channel::from(case.canvas_sequence.clone()).named(&name("kv_positions"));
    // Its OWN page list: a seeded channel attaches to one pass only.
    let pages = Channel::from_iter(0..shape.max_pages).named(&name("pages"));
    let page_indptr =
        Channel::from([0u32, kv_len.div_ceil(shape.page_size)]).named(&name("page_indptr"));
    let slot = Channel::from_iter(case.canvas_sequence.iter().map(|p| p / shape.page_size))
        .named(&name("w_slot"));
    let off = Channel::from_iter(case.canvas_sequence.iter().map(|p| p % shape.page_size))
        .named(&name("w_off"));
    let len = Channel::from([kv_len]).named(&name("kv_len"));
    let slab = Channel::from_shaped([canvas_len, kv_len], mask.as_slice()).named(&name("mask"));
    let readout = Channel::from_iter(0..canvas_len).named(&name("readout"));
    let out = Channel::new([canvas_len, case.hidden], dtype::f32).named(&name("hidden"));

    let pass = ForwardPass::new();
    pass.reading(&denoise.name)?;
    pass.canvas(Mode::Denoise)?;
    pass.stream(LaneStream::Image)?;
    pass.embed(&toks, &indptr)?;
    pass.input(&port_of(denoise, "latents")?, &latents)?;
    pass.input(&port_of(denoise, "special")?, &flag)?;
    pass.input(&port_of(denoise, "timestep")?, &t)?;
    pass.input(&port_of(denoise, "positions")?, &rope)?;
    pass.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &slot,
            w_off: &off,
            positions: &kv_positions,
            mask: Some(&slab),
        },
    )?;
    pass.readout(&readout)?;
    let hidden = case.hidden;
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(hidden));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{tag}` denoise step"))?;
    Ok(out)
}

/// One fire of a VOXEL arm: the clip bound as the port's channel, whose
/// shape IS the box (design D8), read back on the `pixels` seam.
///
/// `clip` is `[h*w, channels]` in raster order and the timestep's sinusoid
/// is appended to EVERY row, because the arm reads one packed rectangle:
/// the CUDA shell seats one voxel width a fire and refuses a lane that
/// feeds two. The model text splits the
/// columns back apart (`super::model::port::LATENT_VOXELS`).
fn voxel_arm(
    case: &Case,
    arm: &model::ReadingFact,
    pipe: &Pipeline,
    clip_port: &str,
    clip: &[f32],
    channels: u32,
    tag: &str,
) -> Result<Channel> {
    let rows = case.token_h * case.token_w;
    let freq = u32::try_from(case.tfreq.len()).map_err(|_| "a sinusoid too wide")?;
    let width_in = channels + freq;
    let mut packed = Vec::with_capacity((rows * width_in) as usize);
    for row in 0..rows as usize {
        let at = row * channels as usize;
        packed.extend_from_slice(&clip[at..at + channels as usize]);
        packed.extend_from_slice(&case.tfreq);
    }
    let pass = ForwardPass::new();
    pass.reading(&arm.name)?;
    // A `forward-diffusion` pass states its attention MODE even where the
    // reading declares no KV space and binds none.
    pass.canvas(Mode::Encode)?;
    pass.stream(LaneStream::Image)?;
    let x = Channel::from_shaped([case.token_h, case.token_w, width_in], packed.as_slice())
        .named(&format!("clip_{tag}"));
    pass.input(clip_port, &x)?;
    let width = arm.readout_width;
    let out = Channel::new([rows, width], dtype::f32).named(&format!("pixels_{tag}"));
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(rows, width));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{}` lane", arm.name))?;
    Ok(out)
}

fn parse(input: &Input) -> Result<Case> {
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
    inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}").into())
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Diffusion {
        return Err("hunyuan_image_3 is a `forward-diffusion` row; the bound model is not".into());
    }
    let case = parse(&input)?;

    let encode = reading("encode")?;
    let denoise = reading("denoise")?;
    if !(encode.has_kv && denoise.has_kv) {
        return Err("both token-axis readings of this family bind a kv space".into());
    }
    if denoise.readout_width != case.hidden {
        return Err(format!(
            "the model reads a {}-wide canvas row and the case carries {}",
            denoise.readout_width, case.hidden
        )
        .into());
    }
    if case.prefix.len() != case.prefix_alt.len() {
        return Err("the two prefixes must have one length, one mask and one rope".into());
    }

    let prefix_len = u32::try_from(case.prefix.len()).map_err(|_| "a prefix too long")?;
    let canvas_len = case.image_rows + 1;
    if case.canvas_sequence.len() as u32 != canvas_len {
        return Err("one sequence position per canvas row".into());
    }
    // The KV holds the prefix, the `<timestep>` row and the canvas — and
    // nothing after them, which is what makes the reference's trailing
    // `<eoi>` invisible to the image rows without a mask column.
    let kv_len = case
        .canvas_sequence
        .iter()
        .copied()
        .max()
        .ok_or("an empty canvas")?
        + 1;
    let page_size = kv_page_size();
    let shape = Shape {
        axes: encode
            .ports
            .iter()
            .find(|port| port.name == "positions")
            .map(|port| port.width)
            .ok_or("the `encode` reading declares no `positions` port")?,
        prefix_len,
        canvas_len,
        kv_len,
        page_size,
        max_pages: kv_len.div_ceil(page_size).max(1),
        t_at: case.canvas_sequence[0],
    };

    let reserve = |ws: &WorkingSet| -> Result<()> {
        ws.reserve(shape.max_pages).context("reserve KV")?;
        Ok(())
    };
    let pipe = Pipeline::new();

    // ---- the image head, going in: `patch_embed` on the voxel axis -------
    let image_in = reading("image.in")?;
    let image_out = reading("image.out")?;
    if image_in.readout_width != case.hidden {
        return Err(format!(
            "`image.in` lands {}-wide rows and the case's trunk is {} wide",
            image_in.readout_width, case.hidden
        )
        .into());
    }
    let rows_channel = voxel_arm(
        &case,
        &image_in,
        &pipe,
        &port_of(&image_in, "latent")?,
        &case.latent,
        case.latent_channels,
        "in",
    )?;
    let image_in_rows = rows_channel.take_host::<Vec<f32>>().await?;

    // ---- A: the prompt's prefix, then step 0 -----------------------------
    let ws_a = WorkingSet::new();
    reserve(&ws_a)?;
    let (argmax, max) = prefill(&case, &shape, &encode, &ws_a, &pipe, &case.prefix, "a")?;
    let a = denoise_step(
        &case,
        &shape,
        &denoise,
        &ws_a,
        &pipe,
        &image_in_rows,
        case.timestep,
        "a",
    )?;
    let encode_argmax = argmax.take_host::<Vec<i32>>().await?;
    let encode_max = max.take_host::<Vec<f32>>().await?;
    let canvas_hidden = a.take_host::<Vec<f32>>().await?;

    // ---- B: the SAME step over the unconditional prefix -------------------
    let ws_b = WorkingSet::new();
    reserve(&ws_b)?;
    prefill(&case, &shape, &encode, &ws_b, &pipe, &case.prefix_alt, "b")?;
    let b = denoise_step(
        &case,
        &shape,
        &denoise,
        &ws_b,
        &pipe,
        &image_in_rows,
        case.timestep,
        "b",
    )?;
    let canvas_hidden_uncond = b.take_host::<Vec<f32>>().await?;

    // ---- C: step 1 over A's pages, whose prefix nothing has touched -------
    let c = denoise_step(
        &case,
        &shape,
        &denoise,
        &ws_a,
        &pipe,
        &image_in_rows,
        case.timestep_next,
        "c",
    )?;
    let canvas_hidden_reused = c.take_host::<Vec<f32>>().await?;

    // ---- D: the same step from a fresh prefill ----------------------------
    let ws_d = WorkingSet::new();
    reserve(&ws_d)?;
    prefill(&case, &shape, &encode, &ws_d, &pipe, &case.prefix, "d")?;
    let d = denoise_step(
        &case,
        &shape,
        &denoise,
        &ws_d,
        &pipe,
        &image_in_rows,
        case.timestep_next,
        "d",
    )?;
    let canvas_hidden_fresh = d.take_host::<Vec<f32>>().await?;

    // ---- the image head, coming out: `final_layer` over A's rows ---------
    // The `<timestep>` row is row 0 of the canvas lane and no voxel of the
    // clip; the rest are the clip's rows in raster order.
    let velocity_channel = voxel_arm(
        &case,
        &image_out,
        &pipe,
        &port_of(&image_out, "rows")?,
        &canvas_hidden[case.hidden as usize..],
        case.hidden,
        "out",
    )?;
    let velocity = velocity_channel.take_host::<Vec<f32>>().await?;
    pipe.close();

    Ok(Output {
        encode_argmax,
        encode_max,
        canvas_hidden,
        canvas_hidden_uncond,
        canvas_hidden_reused,
        canvas_hidden_fresh,
        image_in_rows,
        velocity,
        image_rows: case.image_rows,
        hidden: case.hidden,
        latent_channels: case.latent_channels,
    })
}
