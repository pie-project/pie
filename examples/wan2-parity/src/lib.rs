//! The pie half of the Wan 2.2 miniature golden. Hands a `wan22-mini-*`
//! row exactly what `scripts/imagegen/wan22_golden.py --mini` handed
//! diffusers' `WanTransformer3DModel` — the patch rows, the random
//! context rows, the timestep and the three rotary coordinates of every
//! row — and reads back the velocity the head predicts, as JSON
//! `scripts/imagegen/wan22_parity.py` turns into an `.npz` under the
//! golden's own key names.
//!
//! # TWO OR THREE LANES, ONE GROUP, ONE FIRE
//!
//! `wan_2`'s `denoise` reading declares two streams (D2): the context
//! lane (`Stream::Context`, the `context` port — no positions, no
//! timestep: its class runs the text embedder and the cross-attention
//! keys, and modulates nothing) and one or more video lanes
//! (`Stream::Video`: `latents`, `positions`, `timestep`). The golden's
//! scalar-timestep forward is one video lane; its TI2V per-token forward
//! (`timestep_pertoken`: the first latent frame's tokens at 0, the rest
//! at `t`) is TWO video lanes of one group — the conditioning rows at
//! timestep 0 and the rest at `t` — which the family's self-attention
//! packs into one sequence (`crates/models/src/wan_2/forward.rs`). The
//! case says where the cut is (`cond_rows`); every row's rotary
//! coordinates travel with it, so the lane split changes nothing.
//!
//! **ONE PIPELINE PER LANE.** A group is a fact about a FIRE: the passes
//! join one attention only when they are members of one step. The
//! scheduler seals a frame when every live pipeline has submitted and
//! never seats two passes of one pipeline in one step, so the lanes go
//! down one pipeline each, submitted back to back; the runtime holds a
//! fresh group's first frame for its cohort (`FireRequest::cohort`).
//!
//! Every video lane reads out (the head runs on every video row), each
//! into its own channel; the answer is `[cond rows ‖ the rest]`, the
//! case's row order.
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

/// The case arrives as `case` whole, as `case_file` (a name under
/// `/scratch`), or — the usual way — cut into `case_0`, `case_1`, … argv
/// pieces, because Linux caps ONE argument at 128 KiB. How MANY pieces is
/// the harness's business: the real row needs a score of them, so they are
/// read off the argument map by name rather than declared one field each.
#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    case: Option<String>,
    #[serde(default)]
    case_file: Option<String>,
    #[serde(flatten)]
    rest: std::collections::BTreeMap<String, inferlet::serde_json::Value>,
}

impl Input {
    /// The `case_<n>` pieces, in `n` order, concatenated.
    fn pieces(&self) -> String {
        let mut found: Vec<(u64, &str)> = self
            .rest
            .iter()
            .filter_map(|(name, value)| {
                let n = name.strip_prefix("case_")?.parse::<u64>().ok()?;
                Some((n, value.as_str()?))
            })
            .collect();
        found.sort_unstable_by_key(|(n, _)| *n);
        found.into_iter().map(|(_, text)| text).collect()
    }
}

/// One batch element of the reference's fixed inputs, flattened
/// row-major: the patchified latent in `(c, ph, pw)` feature order (never
/// the `[C, T, H, W]` array), the context rows, one timestep, the
/// `(t, h, w)` patch coordinates of every row, and how many leading rows
/// are the conditioning frame's (0 for the scalar-timestep forward).
///
/// Every float rectangle comes either as a JSON array — legible, and what
/// a miniature's case uses — or, when JSON's ten bytes a number would put
/// the case past what argv carries at all, as `*_b64`: standard base64 of
/// the same numbers as little-endian `f32`, which is a third the size.
#[derive(Deserialize)]
struct Case {
    /// `[rows, patch_features]`, row-major.
    #[serde(default)]
    latents: Vec<f32>,
    #[serde(default)]
    latents_b64: Option<String>,
    rows: u32,
    patch_features: u32,
    /// The first `cond_rows` rows form a second video lane at timestep 0.
    #[serde(default)]
    cond_rows: u32,
    /// `[context_rows, context_width]` — but only the LEADING rows need be
    /// given: `wan_2`'s context port takes exactly what the reference hands
    /// the transformer, 512 rows of which the prompt fills the front and
    /// the rest are hard zeros this guest writes (`wan_2/forward.rs`).
    #[serde(default)]
    context: Vec<f32>,
    #[serde(default)]
    context_b64: Option<String>,
    context_rows: u32,
    context_width: u32,
    /// `[rows, 3]`.
    #[serde(default)]
    positions: Vec<f32>,
    #[serde(default)]
    positions_b64: Option<String>,
    /// The scheduler timestep of the non-conditioning rows.
    timestep: f32,
}

/// Standard base64 (`+/`, `=` padded) into the `f32`s it spells, little
/// endian. A case's rectangles are the only thing that travels this way.
fn floats_of_base64(text: &str, what: &str) -> Result<Vec<f32>> {
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
    let text = text.trim_end_matches('=').as_bytes();
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for (i, &c) in text.iter().enumerate() {
        let six = code(c).ok_or_else(|| format!("{what}: byte {i} is not base64"))?;
        acc = (acc << 6) | six;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(((acc >> bits) & 0xff) as u8);
        }
    }
    if bytes.len() % 4 != 0 {
        return Err(format!("{what}: {} bytes is not whole f32s", bytes.len()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

impl Case {
    /// The three rectangles, each in whichever form the case carried it,
    /// with the context zero-padded to the rows the transformer attends.
    fn rectangles(&mut self) -> Result<()> {
        for (b64, into, what) in [
            (&self.latents_b64, &mut self.latents, "latents"),
            (&self.context_b64, &mut self.context, "context"),
            (&self.positions_b64, &mut self.positions, "positions"),
        ] {
            if let Some(text) = b64 {
                *into = floats_of_base64(text, what)?;
            }
        }
        let want = (self.context_rows * self.context_width) as usize;
        if self.context.len() > want {
            return Err(format!(
                "the case gives {} context values and states {}x{}",
                self.context.len(),
                self.context_rows,
                self.context_width
            )
            .into());
        }
        self.context.resize(want, 0.0);
        Ok(())
    }
}

#[derive(Serialize)]
struct Output {
    /// `[rows, patch_features]` in the case's row order — the reference's
    /// output before its unpatchify. The harness unpatchifies.
    velocity: Vec<f32>,
    rows: u32,
    patch_features: u32,
}

/// The port names this family's `denoise` reading declares, read off
/// `model::readings()` rather than typed in, so a renamed port fails here
/// with the model's own vocabulary instead of at the host.
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
    Ok(Ports {
        latents: named("latents")?.name.clone(),
        context: named("context")?.name.clone(),
        timestep: named("timestep")?.name.clone(),
        positions: named("positions")?.name.clone(),
        axes: named("positions")?.width,
        velocity_width: reading.readout_width,
    })
}

/// One video lane over rows `[from, to)` of the case at `timestep`: its
/// pass, and the channel its velocity lands in.
fn video_lane(
    case: &Case,
    ports: &Ports,
    reading: &str,
    group: u32,
    tag: &str,
    from: u32,
    to: u32,
    timestep: f32,
) -> Result<(ForwardPass, Channel)> {
    let rows = to - from;
    let width = case.patch_features;
    let (f, w) = (from as usize, width as usize);
    let latents = &case.latents[f * w..(to as usize) * w];
    let positions = &case.positions[f * ports.axes as usize..(to as usize) * ports.axes as usize];
    let pass = ForwardPass::new();
    pass.reading(reading)?;
    pass.stream(LaneStream::Video)?;
    pass.group(group)?;
    let x = Channel::from_shaped([rows, width], latents).named(&format!("latents_{tag}"));
    let pos =
        Channel::from_shaped([rows, ports.axes], positions).named(&format!("positions_{tag}"));
    // One timestep cell PER PASS: a seeded channel attaches to one pass
    // only (the runtime's channel-role rule).
    let t = Channel::from([timestep]).named(&format!("t_{tag}"));
    pass.input(&ports.latents, &x)?;
    pass.input(&ports.positions, &pos)?;
    pass.input(&ports.timestep, &t)?;
    let out =
        Channel::new([rows, ports.velocity_width], dtype::f32).named(&format!("velocity_{tag}"));
    let readback = out.clone();
    let velocity_width = ports.velocity_width;
    pass.epilogue(move || {
        readback.put(intrinsics::velocity(velocity_width));
    });
    Ok((pass, out))
}

/// One denoise step: the context lane and one or two video lanes, one
/// group, one fire, the velocity of every video row.
async fn step(case: &Case, ports: &Ports, reading: &str) -> Result<Vec<f32>> {
    let group = 0;

    // The context lane: the cross-attention keys and values, and nothing
    // else.
    let context = ForwardPass::new();
    context.reading(reading)?;
    context.stream(LaneStream::Context)?;
    context.group(group)?;
    let ctx = Channel::from_shaped(
        [case.context_rows, case.context_width],
        case.context.as_slice(),
    )
    .named("context");
    context.input(&ports.context, &ctx)?;

    // The conditioning lane (TI2V's clean first frame, timestep 0), if the
    // case cuts one, and the main lane.
    let cond = (case.cond_rows > 0)
        .then(|| video_lane(case, ports, reading, group, "cond", 0, case.cond_rows, 0.0))
        .transpose()?;
    let (main, main_out) = video_lane(
        case,
        ports,
        reading,
        group,
        "main",
        case.cond_rows,
        case.rows,
        case.timestep,
    )?;

    // One pipeline per lane, submitted back to back.
    let ctx_pipe = Pipeline::new();
    let main_pipe = Pipeline::new();
    let cond_pipe = cond.as_ref().map(|_| Pipeline::new());
    context.submit(&ctx_pipe).context("context lane")?;
    if let (Some((pass, _)), Some(pipe)) = (&cond, &cond_pipe) {
        pass.submit(pipe).context("conditioning lane")?;
    }
    main.submit(&main_pipe).context("video lane")?;

    let mut velocity = Vec::with_capacity(case.rows as usize * ports.velocity_width as usize);
    if let Some((_, out)) = &cond {
        velocity.extend(out.take_host::<Vec<f32>>().await?);
    }
    velocity.extend(main_out.take_host::<Vec<f32>>().await?);
    ctx_pipe.close();
    main_pipe.close();
    if let Some(pipe) = cond_pipe {
        pipe.close();
    }
    Ok(velocity)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    // NOT `pass_kind()`. That verb is derived from `rs_state_size() > 0`,
    // and the flagship row carries a VAE whose causal convolutions each own
    // a `CacheRow::State` slab, so `wan22-ti2v-5b` answers `recurrent` while
    // the miniatures (transformer only) answer `attention` — a difference in
    // the DECODER arms that says nothing about this pass. What a denoise
    // step actually needs is stated on the reading, and is checked below:
    // it binds no kv space and takes no tokens.
    let pieces = input.pieces();
    let text = match (&input.case, &input.case_file) {
        (Some(text), _) => text.clone(),
        (None, Some(name)) => std::fs::read_to_string(format!("/scratch/{name}"))
            .map_err(|why| format!("reading /scratch/{name}: {why}"))?,
        (None, None) if !pieces.is_empty() => pieces,
        (None, None) => {
            return Err("pass `case` (json), `case_0..n` (its pieces) or `case_file`".into());
        }
    };
    let mut case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;
    case.rectangles()?;
    if case.cond_rows >= case.rows {
        return Err("the conditioning rows must leave at least one row for the main lane".into());
    }

    let reading = model::readings()
        .into_iter()
        .find(|reading| reading.name == "denoise")
        .ok_or("this model declares no `denoise` reading")?;
    if reading.has_kv || reading.takes_tokens {
        return Err(format!(
            "reading `{}` binds a kv space or tokens; a denoise pass binds neither",
            reading.name
        )
        .into());
    }
    if reading.readout_width != case.patch_features {
        return Err(format!(
            "the model reads a {}-wide velocity and the case carries {}-wide patch rows",
            reading.readout_width, case.patch_features
        )
        .into());
    }
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && case.rows > max_rows {
        return Err(format!(
            "{} rows exceed the model's {max_rows} latent rows a pass",
            case.rows
        )
        .into());
    }
    let ports = ports(&reading)?;
    let velocity = step(&case, &ports, &reading.name).await?;
    Ok(Output {
        velocity,
        rows: case.rows,
        patch_features: case.patch_features,
    })
}
