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
//!    embeds folded, its noise, `σ₀ = 1`, its ids), the velocity on the
//!    image lane.
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
//! (`Stream::Text`, the `context` port at width `dim`) and the target lane
//! (`Stream::Image`, the `latents` port), one group, each on its own
//! pipeline, each binding the timestep as its own loop-carried cell. Only
//! the image lane reads out. Each lane is one pass submitted once a step
//! (`trajectory` on why); the latent channel is loop-carried by the
//! epilogue.
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
    /// through `context_embedder` (the `context` port's width).
    context_file: String,
    text_rows: u32,
    context_width: u32,
    /// `[image_rows, channels]` f32 LE: the golden's initial noise.
    latents_file: String,
    image_rows: u32,
    channels: u32,
    /// `[rows, 4]` f32 LE each.
    text_positions_file: String,
    image_positions_file: String,
    /// Descending, without the trailing zero.
    sigmas: Vec<f32>,
    /// `[sigmas.len(), image_rows, channels]` f32 LE: the reference's OWN
    /// input latent at every step. Given, the guest also runs one
    /// independent one-fire denoise per step off these — the velocity a
    /// step predicts under the reference's own state, which is the model
    /// check the four-step trajectory cannot be (a distilled four-step
    /// schedule amplifies a bf16-floor velocity difference into the final
    /// latent; see the harness's `NUMERICS` note).
    #[serde(default)]
    probe_latents_file: Option<String>,
    /// Also run the trajectory over pie's own text rows.
    #[serde(default)]
    native: bool,
    /// Submit the image lane before the text lane each step (default: text
    /// first). The frame packs by stream, so the order should not matter;
    /// a knob for the harness to prove it.
    #[serde(default)]
    image_first: bool,
}

#[derive(Serialize)]
struct Output {
    instance: String,
    token_ids: Vec<u32>,
    text_rows: u32,
    hidden_width: u32,
    image_rows: u32,
    channels: u32,
    sigmas: Vec<f32>,
    /// What `session::send_file` sent, in order: `file-0000.bin` is
    /// `files[0]`.
    files: Vec<String>,
}

struct Ports {
    latents: String,
    context: String,
    timestep: String,
    positions: String,
    axes: u32,
    velocity_width: u32,
}

fn ports(reading: &model::ReadingFact) -> Result<Ports> {
    let named = |name: &str| -> Result<&model::PortFact> {
        reading
            .ports
            .iter()
            .find(|port| port.name == name)
            .ok_or_else(|| format!("reading `{}` declares no port `{name}`", reading.name).into())
    };
    if reading.ports.iter().any(|port| port.name == "guidance") {
        return Err("this row declares a guidance port; klein-4B embeds no guidance".into());
    }
    Ok(Ports {
        latents: named("latents")?.name.clone(),
        context: named("context")?.name.clone(),
        timestep: named("timestep")?.name.clone(),
        positions: named("positions")?.name.clone(),
        axes: named("positions")?.width,
        velocity_width: reading.readout_width,
    })
}

fn read_f32(name: &str, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(format!("/scratch/{name}"))
        .map_err(|why| format!("reading /scratch/{name}: {why}"))?;
    if bytes.len() != count * 4 {
        return Err(format!(
            "/scratch/{name}: {} bytes, expected {} ({count} f32)",
            bytes.len(),
            count * 4
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
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
async fn text_rows(ids: &[u32], reading: &model::ReadingFact) -> Result<Vec<f32>> {
    let width = reading.readout_width;
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    let toks: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let page_size = kv_page_size().max(1);
    let pages = len.div_ceil(page_size);
    let ws = WorkingSet::new();
    ws.reserve(pages)?;
    let pipe = Pipeline::new();

    let toks = Channel::from(toks).named("ids");
    let embed_indptr = Channel::from([0u32, len]);
    let positions = Channel::from_iter(0..len);
    let page_ids = Channel::from_iter(0..pages);
    let page_indptr = Channel::from([0u32, pages]);
    let w_slot = Channel::from_iter((0..len).map(|p| p / page_size));
    let w_off = Channel::from_iter((0..len).map(|p| p % page_size));
    let kv_len = Channel::from([len]);
    let readout = Channel::from_iter(0..len);
    let out = Channel::new([len, width], dtype::f32).named("hidden");

    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.embed(&toks, &embed_indptr)?;
    pass.readout(&readout)?;
    pass.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &page_ids,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(width));
    });
    pass.submit(&pipe).context("text pass")?;
    let rows: Vec<f32> = out.take_host().await?;
    pipe.close();
    Ok(rows)
}

/// The conditioning one trajectory runs under.
struct Text {
    rows: u32,
    context: Channel,
    positions: Channel,
}

/// `sigmas.len()` Euler steps from `noise`, the velocity at step 0 and the
/// latent after every step read back to the host (one `[rows, channels]`
/// readback a fire; the loop-carried latent never crosses).
///
/// Both lanes are ONE pass each, submitted once a step: a seeded channel's
/// seed rides with the first submit of the pass that binds it, so a fresh
/// pass a step would find the context cell empty. The timestep is a
/// loop-carried cell per lane (each lane's epilogue advances its own by
/// the step index), the way `latent-probe` steps its schedule.
async fn trajectory(
    text: &Text,
    noise: &[f32],
    positions: &[f32],
    rows: u32,
    width: u32,
    sigmas: &[f32],
    ports: &Ports,
    reading: &str,
    group: u32,
    image_first: bool,
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let steps = u32::try_from(sigmas.len()).map_err(|_| "too many sigmas")?;
    // `ts[k]` is what fire `k` reads; one trailing 0 so the last advance
    // gathers something. `dts[k]` is what fire `k` integrates.
    let mut ts: Vec<f32> = sigmas.iter().map(|s| s * 1000.0).collect();
    ts.push(0.0);
    let dts: Vec<f32> = (0..sigmas.len())
        .map(|i| sigmas.get(i + 1).copied().unwrap_or(0.0) - sigmas[i])
        .collect();
    // A seeded channel attaches to one pass, so each lane gets its own
    // copy of the schedule.
    let ts_txt = Channel::from(ts.clone()).named("ts_txt");
    let ts_img = Channel::from(ts.clone()).named("ts_img");
    let dts_ch = Channel::from(dts).named("dts");

    let pipe_txt = Pipeline::new();
    let pipe_img = Pipeline::new();
    let x = Channel::from_shaped([rows, width], noise).named("latents");
    let img_pos = Channel::from_shaped([rows, ports.axes], positions).named("img_pos");
    let velocity_width = ports.velocity_width;

    let t_txt = Channel::from([ts[0]]).named("t_txt");
    let k_txt = Channel::from([0u32]).named("k_txt");
    let txt = ForwardPass::new();
    txt.reading(reading)?;
    txt.stream(LaneStream::Text)?;
    txt.group(group)?;
    txt.input(&ports.context, &text.context)?;
    txt.input(&ports.positions, &text.positions)?;
    txt.input(&ports.timestep, &t_txt)?;
    {
        let (t, k, ts_ch) = (t_txt.clone(), k_txt.clone(), ts_txt);
        txt.epilogue(move || {
            let k_next = &k.take() + 1u32;
            t.take();
            t.put(gather(ts_ch.read(), &k_next));
            k.put(&k_next);
        });
    }

    let t_img = Channel::from([ts[0]]).named("t_img");
    let k_img = Channel::from([0u32]).named("k_img");
    let v_out = Channel::new([rows, velocity_width], dtype::f32).named("v_out");
    let x_out = Channel::new([rows, width], dtype::f32).named("x_out");
    let img = ForwardPass::new();
    img.reading(reading)?;
    img.stream(LaneStream::Image)?;
    img.group(group)?;
    img.input(&ports.latents, &x)?;
    img.input(&ports.positions, &img_pos)?;
    img.input(&ports.timestep, &t_img)?;
    {
        let (xc, vo, xo) = (x.clone(), v_out.clone(), x_out.clone());
        let (t, k, ts_ch) = (t_img.clone(), k_img.clone(), ts_img);
        img.epilogue(move || {
            let k_cur = k.take();
            let v = intrinsics::velocity(velocity_width);
            let cur = xc.take();
            let dt = gather(dts_ch.read(), &k_cur);
            let next = euler_step(&cur, &v, &dt);
            xc.put(&next);
            vo.put(&v);
            xo.put(&next);
            let k_next = &k_cur + 1u32;
            t.take();
            t.put(gather(ts_ch.read(), &k_next));
            k.put(&k_next);
        });
    }

    let mut velocity0 = Vec::new();
    let mut latents = Vec::with_capacity(sigmas.len());
    for i in 0..steps {
        if image_first {
            img.submit(&pipe_img)
                .with_context(|| format!("image lane, step {i}"))?;
            txt.submit(&pipe_txt)
                .with_context(|| format!("text lane, step {i}"))?;
        } else {
            txt.submit(&pipe_txt)
                .with_context(|| format!("text lane, step {i}"))?;
            img.submit(&pipe_img)
                .with_context(|| format!("image lane, step {i}"))?;
        }
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
    pipe_txt.close();
    pipe_img.close();
    let _ = text.rows;
    Ok((velocity0, latents))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err("flux_2 is an attention-kind pass".into());
    }
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
            return Err(format!("{path}: not written within the wait").into());
        }
        inferlet::sleep(std::time::Duration::from_millis(200)).await;
    };
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;

    let text_reading = model::reading("text").ok_or("this model declares no `text` reading")?;
    if !(text_reading.has_kv && text_reading.takes_tokens)
        || text_reading.readout != model::ReadoutKind::Hidden
    {
        return Err("reading `text` is not a token + kv reading with a hidden readout".into());
    }
    let denoise = model::reading("denoise").ok_or("this model declares no `denoise` reading")?;
    if denoise.has_kv || denoise.takes_tokens {
        return Err("reading `denoise` binds a kv space or tokens".into());
    }
    let ports = ports(&denoise)?;
    if ports.velocity_width != case.channels {
        return Err(format!(
            "the model reads a {}-wide velocity and the case carries {}-wide tokens",
            ports.velocity_width, case.channels
        )
        .into());
    }
    let hidden_width = text_reading.readout_width;
    if hidden_width != case.context_width {
        return Err(format!(
            "`text` reads out {hidden_width} wide and the case's context is {} wide",
            case.context_width
        )
        .into());
    }

    let mut files = Vec::new();

    // 1. The text reading.
    let mut ids = inferlet::chat::first_user(&case.prompt);
    ids.extend(inferlet::chat::cue());
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    let hidden = text_rows(&ids, &text_reading).await?;
    send_f32(&mut files, "hidden.f32", &hidden);

    // 2 + 3. The trajectory over the golden's conditioning.
    let rows = case.image_rows;
    let width = case.channels;
    let noise = read_f32(&case.latents_file, (rows * width) as usize)?;
    let img_pos = read_f32(&case.image_positions_file, (rows * ports.axes) as usize)?;
    let context = read_f32(
        &case.context_file,
        (case.text_rows * case.context_width) as usize,
    )?;
    let txt_pos = read_f32(
        &case.text_positions_file,
        (case.text_rows * ports.axes) as usize,
    )?;
    // A SEEDED CHANNEL BINDS TO ONE PASS, so every trajectory below builds
    // its own copy of the conditioning off the same host rows: sharing one
    // `Text` across two trajectories is "seeded but no seed was put before
    // the first fire" on the second one's text pass.
    let conditioning = |tag: &str| Text {
        rows: case.text_rows,
        context: Channel::from_shaped([case.text_rows, case.context_width], context.clone())
            .named(&format!("context_{tag}")),
        positions: Channel::from_shaped([case.text_rows, ports.axes], txt_pos.clone())
            .named(&format!("txt_pos_{tag}")),
    };
    let golden = conditioning("walk");
    let (velocity0, latents) = trajectory(
        &golden,
        &noise,
        &img_pos,
        rows,
        width,
        &case.sigmas,
        &ports,
        &denoise.name,
        0,
        case.image_first,
    )
    .await?;
    send_f32(&mut files, "velocity0.f32", &velocity0);
    for (i, x) in latents.iter().enumerate() {
        send_f32(&mut files, &format!("latent{}.f32", i + 1), x);
    }

    // One independent fire per step, from the reference's own state.
    if let Some(name) = &case.probe_latents_file {
        let steps = case.sigmas.len();
        let stack = read_f32(name, steps * (rows * width) as usize)?;
        let stride = (rows * width) as usize;
        for (k, sigma) in case.sigmas.iter().enumerate() {
            let (velocity, _) = trajectory(
                &conditioning(&format!("probe{k}")),
                &stack[k * stride..(k + 1) * stride],
                &img_pos,
                rows,
                width,
                &[*sigma],
                &ports,
                &denoise.name,
                2 + k as u32,
                case.image_first,
            )
            .await
            .with_context(|| format!("probe step {k}"))?;
            send_f32(&mut files, &format!("probe_velocity{k}.f32"), &velocity);
        }
    }

    // The same trajectory over pie's own text rows: `(0, 0, 0, j)`.
    if case.native {
        let mut pos = Vec::with_capacity((len * ports.axes) as usize);
        for j in 0..len {
            for axis in 0..ports.axes {
                pos.push(if axis + 1 == ports.axes {
                    j as f32
                } else {
                    0.0
                });
            }
        }
        let native = Text {
            rows: len,
            context: Channel::from_shaped([len, hidden_width], hidden.clone()).named("native_ctx"),
            positions: Channel::from_shaped([len, ports.axes], pos).named("native_pos"),
        };
        let (velocity0, latents) = trajectory(
            &native,
            &noise,
            &img_pos,
            rows,
            width,
            &case.sigmas,
            &ports,
            &denoise.name,
            1,
            case.image_first,
        )
        .await?;
        send_f32(&mut files, "native_velocity0.f32", &velocity0);
        for (i, x) in latents.iter().enumerate() {
            send_f32(&mut files, &format!("native_latent{}.f32", i + 1), x);
        }
    }

    Ok(Output {
        instance,
        token_ids: ids,
        text_rows: len,
        hidden_width,
        image_rows: rows,
        channels: width,
        sigmas: case.sigmas,
        files,
    })
}
