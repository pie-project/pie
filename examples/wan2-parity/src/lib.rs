//! The pie half of the Wan 2.2 miniature golden. Hands a `wan22-mini-*`
//! row exactly what `scripts/imagegen/wan22_golden.py --mini` handed
//! diffusers' `WanTransformer3DModel` — the latent (the VAE's own
//! `[t, h, w, C]`, on the voxel axis), the context rows and the timestep —
//! and reads back the velocity the head predicts on the same voxels, as
//! JSON `scripts/imagegen/wan22_parity.py` turns into an `.npz` under the
//! golden's own key names. The model patchifies and computes each row's
//! rotary position in its graph.
//!
//! # TWO OR THREE LANES, ONE GROUP, ONE PASS
//!
//! `wan_2`'s `denoise` reading runs on two values of its `stream` switch:
//! the context lane (`context`, the `context` input) and one or more video
//! lanes (`video`: `latents`, `timestep`). The golden's scalar-timestep
//! forward is one video lane; its TI2V per-token forward (the first latent
//! frame at timestep 0, the rest at `t`) is TWO video lanes of one group,
//! one clip each: frame 0, and frames `1..`. A lane computes positions
//! from its own clip, so the second lane's frames would restart at `t = 0`;
//! the case then states the reference's positions and both video lanes take
//! them (`positions = given`).
//!
//! The step is ONE pass: lane 0 is the context lane, then the conditioning
//! video lane when the case cuts one, then the main video lane, all in
//! attention group 0. Every video lane reads out, each into its own channel
//! in the pass's one epilogue; the answer is the clip's voxels in frame
//! order.
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

/// The case arrives as `case` whole, as `case_file` (a name under
/// `/scratch`), or cut into `case_0`, `case_1`, … argv pieces, because Linux
/// caps one argument at 128 KiB.
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

/// One batch element of the reference's fixed inputs, flattened row-major:
/// the latent `[frames, height, width, channels]`, the context rows, one
/// timestep, and how many leading frames are the conditioning frame's (0
/// for the scalar-timestep forward).
///
/// Every float rectangle comes either as a JSON array or, when JSON's ten
/// bytes a number would put the case past what argv carries, as `*_b64`:
/// standard base64 of the same numbers as little-endian `f32`.
#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    latents: Vec<f32>,
    #[serde(default)]
    latents_b64: Option<String>,
    frames: u32,
    height: u32,
    width: u32,
    channels: u32,
    /// The first `cond_frames` frames form a second video lane at timestep 0.
    #[serde(default)]
    cond_frames: u32,
    /// `[context_rows, context_width]` — but only the LEADING rows need be
    /// given: the rest are hard zeros this guest writes (the reference's
    /// zero pad to 512).
    #[serde(default)]
    context: Vec<f32>,
    #[serde(default)]
    context_b64: Option<String>,
    context_rows: u32,
    context_width: u32,
    /// With `cond_frames`: every patch row's `(t, h, w)`, `[rows, 3]` in the
    /// clip's row-major patch order, which both video lanes take.
    #[serde(default)]
    positions: Vec<f32>,
    #[serde(default)]
    positions_b64: Option<String>,
    /// The scheduler timestep of the non-conditioning frames.
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
    /// `[frames · height · width, channels]`: the velocity on the clip's
    /// voxels, frame-major.
    velocity: Vec<f32>,
    frames: u32,
    height: u32,
    width: u32,
    channels: u32,
}

/// The readout's velocity width.
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
        )
        .into());
    }
    Ok(lane)
}

/// Checks the context and video lanes and reads the velocity width.
fn dims(reading: &str, given: bool) -> Result<Dims> {
    lane_reading(reading, "context", &["context"])?;
    let video = if given {
        let lane = model::lane_of(&[
            ("reading", reading),
            ("stream", "video"),
            ("positions", "given"),
        ])?;
        let mut reads: Vec<&str> = lane.inputs.iter().map(|i| i.name.as_str()).collect();
        reads.sort_unstable();
        if reads != ["latents", "positions", "timestep"] {
            return Err(format!(
                "reading `{reading}`'s video lane under `positions = given` reads {reads:?}"
            )
            .into());
        }
        lane
    } else {
        lane_reading(reading, "video", &["latents", "timestep"])?
    };
    let velocity_width = match &video.readout {
        Some(readout) if readout.kind == model::ReadoutKind::Velocity => readout.width,
        other => {
            return Err(format!(
                "reading `{reading}`'s video lane reads out {other:?}, not a velocity"
            ));
        }
    };
    Ok(Dims { velocity_width })
}

/// Binds lane `lane` of `pass` as one video lane over frames `[from, to)`
/// of the case at `timestep`, and answers the voxels it carries.
#[allow(clippy::too_many_arguments)]
fn video_lane(
    pass: &ForwardPass,
    lane: u32,
    case: &Case,
    reading: &str,
    group: u32,
    tag: &str,
    from: u32,
    to: u32,
    timestep: f32,
) -> Result<u32> {
    let plane = case.height * case.width;
    let frame = (plane * case.channels) as usize;
    let latents = &case.latents[from as usize * frame..to as usize * frame];
    pass.switch(lane, "reading", reading)?;
    pass.switch(lane, "stream", "video")?;
    pass.group(lane, group)?;
    let x = Channel::from_shaped([to - from, case.height, case.width, case.channels], latents)
        .named(&format!("latents_{tag}"));
    let t = Channel::from([timestep]).named(&format!("t_{tag}"));
    pass.input(lane, "latents", &x)?;
    pass.input(lane, "timestep", &t)?;
    if !case.positions.is_empty() {
        // Patch rows per frame: the clip's patch grid over one frame.
        let rows = |frames: u32| (case.positions.len() as u32 / 3) / case.frames * frames;
        let (a, b) = (rows(from) as usize * 3, rows(to) as usize * 3);
        pass.switch(lane, "positions", "given")?;
        let pos = Channel::from_shaped([rows(to - from), 3], &case.positions[a..b])
            .named(&format!("positions_{tag}"));
        pass.input(lane, "positions", &pos)?;
    }
    Ok((to - from) * plane)
}

/// One denoise step: the context lane and one or two video lanes, one
/// group, one pass, the velocity of every video voxel.
async fn step(case: &Case, dims: &Dims, reading: &str) -> Result<Vec<f32>> {
    let group = 0;
    let pass = ForwardPass::new();

    // Lane 0, the context lane.
    pass.switch(0, "reading", reading)?;
    pass.switch(0, "stream", "context")?;
    pass.group(0, group)?;
    let ctx = Channel::from_shaped(
        [case.context_rows, case.context_width],
        case.context.as_slice(),
    )
    .named("context");
    pass.input(0, "context", &ctx)?;

    // The conditioning lane (TI2V's clean first frame, timestep 0), if the
    // case cuts one, and the main lane: each video lane's (lane, voxels).
    let mut videos: Vec<(u32, u32)> = Vec::new();
    if case.cond_frames > 0 {
        let voxels = video_lane(&pass, 1, case, reading, group, "cond", 0, case.cond_frames, 0.0)?;
        videos.push((1, voxels));
    }
    let main_lane = videos.len() as u32 + 1;
    let voxels = video_lane(
        &pass,
        main_lane,
        case,
        reading,
        group,
        "main",
        case.cond_frames,
        case.frames,
        case.timestep,
    )?;
    videos.push((main_lane, voxels));

    let velocity_width = dims.velocity_width;
    let outs: Vec<(u32, u32, Channel)> = videos
        .iter()
        .map(|&(lane, voxels)| {
            let out = Channel::new([voxels, velocity_width], dtype::f32)
                .named(&format!("velocity_{lane}"));
            (lane, voxels, out)
        })
        .collect();
    let readbacks = outs.clone();
    pass.epilogue(move || {
        for (lane, voxels, out) in &readbacks {
            out.put(intrinsics::velocity(*lane, *voxels, velocity_width));
        }
    });
    let pipe = Pipeline::new();
    pass.submit(&pipe).context("the denoise step")?;

    let mut velocity = Vec::new();
    for (_, _, out) in &outs {
        velocity.extend(out.take_host::<Vec<f32>>().await?);
    }
    pipe.close();
    Ok(velocity)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
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
    if case.cond_frames >= case.frames {
        return Err("the conditioning frames must leave at least one frame for the main lane".into());
    }
    let want = (case.frames * case.height * case.width * case.channels) as usize;
    if case.latents.len() != want {
        return Err(format!(
            "the case carries {} latent values and {}x{}x{}x{} is {want}",
            case.latents.len(),
            case.frames,
            case.height,
            case.width,
            case.channels
        )
        .into());
    }
    if case.positions.len() % (3 * case.frames as usize) != 0 {
        return Err("the case's positions are not whole (t, h, w) rows per frame".into());
    }

    let reading = "denoise";
    let dims = dims(reading, !case.positions.is_empty())?;
    if dims.velocity_width != case.channels {
        return Err(format!(
            "the model reads a {}-wide velocity and the case's latent has {} channels",
            dims.velocity_width, case.channels
        )
        .into());
    }
    let velocity = step(&case, &dims, reading).await?;
    Ok(Output {
        velocity,
        frames: case.frames,
        height: case.height,
        width: case.width,
        channels: case.channels,
    })
}
