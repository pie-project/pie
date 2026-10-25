//! The pie half of the FLUX.2-klein-4B golden (`scripts/imagegen/
//! flux2_golden.py --full`), driven by `scripts/imagegen/flux2_klein_parity.py`.
//! Three readings of the row, in the order the golden was made:
//!
//! 1. **`text`** — the golden's prompt through the family's template
//!    (`chat::first_user` + `chat::cue`: one user turn, the generation
//!    cue, thinking off), one prefill over the Qwen3-4B trunk, `hidden()`
//!    read at every row: the layer-{9,18,27} stack already folded through
//!    `context_embedder`, `[L, 3072]`. The harness folds the golden's raw
//!    `[512, 7680]` stack in numpy and compares the `L` unpadded rows.
//! 2. **one `denoise` step** — the golden's own step-0 inputs (its prompt
//!    embeds folded, its noise, `σ₀ = 1`), the velocity on the image lane.
//! 2b. **one independent step per sigma** (`probe_latents_file`) — the
//!    same one-fire denoise from the reference's OWN latent at every step,
//!    so each step's velocity is checked under the reference's state
//!    rather than under pie's drifting one.
//! 3. **four steps** — the golden's sigmas, Euler in the image lane's
//!    epilogue (`euler_step`), the latent after every step. With `native`,
//!    the same trajectory again over pie's OWN text rows from (1): the
//!    first text-to-image trajectory that is pie's end to end.
//!
//! # THE CASE CROSSES AS FILES
//!
//! A 1024² job is 4096 latent rows and 512 context rows of width 3072 —
//! eight megabytes, past what argv carries. The sandbox mounts a
//! per-process directory at `/scratch` (`fs_scratch_dir/<instance-id>`),
//! created when the instance starts, so the guest announces its instance
//! id (`session::send`) and waits for the harness to drop the case there;
//! the arrays are raw little-endian f32, the answers go back the same way
//! through `session::send_file` (`pie run -o DIR` writes them as
//! `file-NNNN.bin` in send order, which `Output::files` names).
//!
//! # LANES
//!
//! `denoise` is two lanes here (no reference): the text lane
//! (stream `text`, the `context` input at width `dim`) and the target lane
//! (stream `image`, the `latents` input: the VAE-native latent `[h, w, C]`
//! on the voxel axis), one group, both lanes of ONE pass binding the one
//! loop-carried timestep cell. The model patchifies and computes every
//! row's rotary position in its graph; the guest states neither. Only the
//! image lane reads out, on the voxel axis. Every trajectory runs down ONE
//! pipeline and reads the one conditioning channel: a channel no pass
//! writes is shared by every pass that reads it.
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    case_file: Option<String>,
    #[serde(default)]
    wait_secs: Option<u32>,
}

/// The case the harness writes beside its arrays.
#[derive(Deserialize)]
struct Case {
    prompt: String,
    /// `[text_rows, context_width]` f32 LE: the golden's prompt embeds
    /// through `context_embedder` (the `context` input's width).
    context_file: String,
    text_rows: u32,
    context_width: u32,
    /// `[height, width, channels]` f32 LE: the golden's initial noise in
    /// the VAE's own latent space (unpacked by the harness).
    latents_file: String,
    height: u32,
    width: u32,
    channels: u32,
    /// Descending, without the trailing zero.
    sigmas: Vec<f32>,
    /// `[sigmas.len(), height, width, channels]` f32 LE: the reference's own
    /// input latent at every step, each denoised once independently.
    #[serde(default)]
    probe_latents_file: Option<String>,
    /// Also run the trajectory over pie's own text rows.
    #[serde(default)]
    native: bool,
    /// Submit the image lane before the text lane each step (default: text
    /// first). A group packs its lanes in submission order, so this packs
    /// `[img ‖ txt]` instead of `[txt ‖ img]`.
    #[serde(default)]
    image_first: bool,
}

#[derive(Serialize)]
struct Output {
    instance: String,
    token_ids: Vec<u32>,
    text_rows: u32,
    hidden_width: u32,
    /// The latent the velocities and latents are sent in, `[height, width,
    /// channels]` (the velocity's voxels in the same row-major order).
    height: u32,
    width: u32,
    channels: u32,
    sigmas: Vec<f32>,
    /// What `session::send_file` sent, in order: `file-0000.bin` is
    /// `files[0]`.
    files: Vec<String>,
}

/// The velocity width the image lane reads out.
struct Dims {
    velocity_width: u32,
}

/// The lane `stream` of `reading`, checked to read exactly `binds`.
fn lane_reading(reading: &str, stream: &str, binds: &[&str]) -> Result<model::LaneDecl> {
    let lane = model::lane_of(&[("reading", reading), ("stream", stream)])?;
    if lane.state || lane.embeds {
        return Err(format!(
            "reading `{reading}`'s `{stream}` lane binds a working set or tokens; a denoise pass \
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
        ));
    }
    Ok(lane)
}

/// Checks both lanes (klein-4B reads no `guidance`) and reads the velocity width.
fn dims(reading: &str) -> Result<Dims> {
    let image = lane_reading(reading, "image", &["latents", "timestep"])?;
    lane_reading(reading, "text", &["context", "timestep"])?;
    let velocity_width = match &image.readout {
        Some(readout) if readout.kind == model::ReadoutKind::Velocity => readout.width,
        other => {
            return Err(format!(
                "reading `{reading}`'s image lane reads out {other:?}, not a velocity"
            ));
        }
    };
    Ok(Dims { velocity_width })
}

fn read_f32(name: &str, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(format!("/scratch/{name}"))
        .map_err(|why| format!("reading /scratch/{name}: {why}"))?;
    if bytes.len() != count * 4 {
        return Err(format!(
            "/scratch/{name}: {} bytes, expected {} ({count} f32)",
            bytes.len(),
            count * 4
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn send_f32(files: &mut Vec<String>, name: &str, rows: &[f32]) {
    let mut bytes = Vec::with_capacity(rows.len() * 4);
    for x in rows {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    inferlet::session::send_file(&bytes);
    files.push(name.to_string());
}

/// The `text` reading over `ids`: `encode_text`'s pass, with the ids
/// chosen here so the harness can check them against the pipeline's.
async fn text_rows(ids: &[u32], width: u32, pipe: &Pipeline) -> Result<Vec<f32>> {
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    let toks: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let ws = WorkingSet::new();

    let toks = Channel::from(toks).named("ids");
    let embed_indptr = Channel::from([0u32, len]);
    let readout = Channel::from_iter(0..len);
    let out = Channel::new([len, width], dtype::f32).named("hidden");

    let pass = ForwardPass::new();
    pass.switch(0, "reading", "text")?;
    pass.embed(&toks, &embed_indptr)?;
    pass.readout(&readout)?;
    pass.state(&[&ws], StateGeometry::default())?;
    let readback = out;
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(0, len, width));
    });
    pass.submit(pipe).context("text pass")?;
    let rows: Vec<f32> = out.take_host().await?;
    Ok(rows)
}

/// `sigmas.len()` Euler steps from `noise` (`[h, w, C]`): the step-0
/// velocity and the latent after every step.
#[allow(clippy::too_many_arguments)]
async fn trajectory(
    context: &Channel,
    noise: &[f32],
    lat: [u32; 3],
    sigmas: &[f32],
    dims: &Dims,
    reading: &str,
    image_first: bool,
    pipe: &Pipeline,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let steps = u32::try_from(sigmas.len()).map_err(|_| "too many sigmas")?;
    // `ts[k]` is what fire `k` reads; one trailing 0 so the last advance
    // gathers something. `dts[k]` is what fire `k` integrates.
    let mut ts: Vec<f32> = sigmas.iter().map(|s| s * 1000.0).collect();
    ts.push(0.0);
    let dts: Vec<f32> = (0..sigmas.len())
        .map(|i| sigmas.get(i + 1).copied().unwrap_or(0.0) - sigmas[i])
        .collect();
    let ts_ch = Channel::from(ts.clone()).named("ts");
    let dts_ch = Channel::from(dts).named("dts");

    let voxels = lat[0] * lat[1];
    let channels = lat[2];
    let x = Channel::from_shaped(lat, noise).named("latents");
    let velocity_width = dims.velocity_width;
    let (img_lane, txt_lane) = if image_first { (0, 1) } else { (1, 0) };

    let t = Channel::from([ts[0]]).named("t");
    let k = Channel::from([0u32]).named("k");
    let pass = ForwardPass::new();
    pass.switch(txt_lane, "reading", reading)?;
    pass.switch(txt_lane, "stream", "text")?;
    pass.group(txt_lane, 0)?;
    pass.input(txt_lane, "context", context)?;
    pass.input(txt_lane, "timestep", &t)?;
    pass.switch(img_lane, "reading", reading)?;
    pass.switch(img_lane, "stream", "image")?;
    pass.group(img_lane, 0)?;
    pass.input(img_lane, "latents", &x)?;
    pass.input(img_lane, "timestep", &t)?;

    let v_out = Channel::new([voxels, velocity_width], dtype::f32).named("v_out");
    let x_out = Channel::new(lat, dtype::f32).named("x_out");
    pass.epilogue(move || {
        let k_cur = k.take();
        let v = intrinsics::velocity(img_lane, voxels, velocity_width);
        let cur = reshape(x.take(), [voxels, channels]);
        let dt = gather(dts_ch.read(), &k_cur);
        let next = reshape(euler_step(&cur, &v, &dt), lat);
        x.put(&next);
        v_out.put(&v);
        x_out.put(&next);
        let k_next = &k_cur + 1u32;
        t.take();
        t.put(gather(ts_ch.read(), &k_next));
        k.put(&k_next);
    });

    let mut velocity0 = Vec::new();
    let mut latents = Vec::with_capacity(sigmas.len());
    for i in 0..steps {
        pass.submit(pipe)
            .with_context(|| format!("denoise step {i}"))?;
        let v: Vec<f32> = v_out
            .take_host()
            .await
            .with_context(|| format!("velocity readback, step {i}"))?;
        let xn: Vec<f32> = x_out
            .take_host()
            .await
            .with_context(|| format!("latent readback, step {i}"))?;
        if i == 0 {
            velocity0 = v;
        }
        latents.push(xn);
    }
    Ok((velocity0, latents))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let instance = inferlet::runtime::instance_id();
    inferlet::session::send(&format!("instance {instance}"));

    // Wait for the harness to drop the case into this instance's scratch.
    let case_name = input.case_file.unwrap_or_else(|| "case.json".to_string());
    let path = format!("/scratch/{case_name}");
    let deadline = u64::from(input.wait_secs.unwrap_or(120)) * 1_000_000_000;
    let start = inferlet::monotonic_now_ns();
    let text = loop {
        if let Ok(text) = std::fs::read_to_string(&path) {
            break text;
        }
        if inferlet::monotonic_now_ns() - start > deadline {
            return Err(format!("{path}: not written within the wait"));
        }
        inferlet::sleep(std::time::Duration::from_millis(200)).await;
    };
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;

    let text_lane = model::lane_of(&[("reading", "text")])?;
    let hidden_width = match &text_lane.readout {
        Some(readout)
            if text_lane.state
                && text_lane.embeds
                && readout.kind == model::ReadoutKind::Hidden =>
        {
            readout.width
        }
        _ => {
            return Err("reading `text` is not a token + kv reading with a hidden readout".into());
        }
    };
    let denoise = "denoise";
    let dims = dims(denoise)?;
    if dims.velocity_width != case.channels {
        return Err(format!(
            "the model reads a {}-wide velocity and the case's latent has {} channels",
            dims.velocity_width, case.channels
        ));
    }
    if hidden_width != case.context_width {
        return Err(format!(
            "`text` reads out {hidden_width} wide and the case's context is {} wide",
            case.context_width
        ));
    }

    let mut files = Vec::new();
    // Every pass of the job, in order, down one pipeline.
    let pipe = Pipeline::new();

    // 1. The text reading.
    let mut ids = inferlet::chat::first_user(&case.prompt);
    ids.extend(inferlet::chat::cue());
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    let hidden = text_rows(&ids, hidden_width, &pipe).await?;
    send_f32(&mut files, "hidden.f32", &hidden);

    // 2 + 3. The trajectory over the golden's conditioning.
    let lat = [case.height, case.width, case.channels];
    let cells = (case.height * case.width * case.channels) as usize;
    let noise = read_f32(&case.latents_file, cells)?;
    let context = read_f32(
        &case.context_file,
        (case.text_rows * case.context_width) as usize,
    )?;
    // One conditioning channel: no pass writes it, so every trajectory's
    // pass reads the same one.
    let golden =
        Channel::from_shaped([case.text_rows, case.context_width], context).named("context");
    let (velocity0, latents) = trajectory(
        &golden,
        &noise,
        lat,
        &case.sigmas,
        &dims,
        denoise,
        case.image_first,
        &pipe,
    )
    .await?;
    send_f32(&mut files, "velocity0.f32", &velocity0);
    for (i, x) in latents.iter().enumerate() {
        send_f32(&mut files, &format!("latent{}.f32", i + 1), x);
    }

    // One independent pass per step, from the reference's own state.
    if let Some(name) = &case.probe_latents_file {
        let steps = case.sigmas.len();
        let stack = read_f32(name, steps * cells)?;
        for (k, sigma) in case.sigmas.iter().enumerate() {
            let (velocity, _) = trajectory(
                &golden,
                &stack[k * cells..(k + 1) * cells],
                lat,
                &[*sigma],
                &dims,
                denoise,
                case.image_first,
                &pipe,
            )
            .await
            .with_context(|| format!("probe step {k}"))?;
            send_f32(&mut files, &format!("probe_velocity{k}.f32"), &velocity);
        }
    }

    // The same trajectory over pie's own text rows.
    if case.native {
        let native = Channel::from_shaped([len, hidden_width], hidden.clone()).named("native_ctx");
        let (velocity0, latents) = trajectory(
            &native,
            &noise,
            lat,
            &case.sigmas,
            &dims,
            denoise,
            case.image_first,
            &pipe,
        )
        .await?;
        send_f32(&mut files, "native_velocity0.f32", &velocity0);
        for (i, x) in latents.iter().enumerate() {
            send_f32(&mut files, &format!("native_latent{}.f32", i + 1), x);
        }
    }
    pipe.close();

    Ok(Output {
        instance,
        token_ids: ids,
        text_rows: len,
        hidden_width,
        height: case.height,
        width: case.width,
        channels: case.channels,
        sigmas: case.sigmas,
        files,
    })
}
