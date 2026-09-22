//! A probe of the float-port loop (imagegen design D1/D3/D4), on a model
//! whose `readings()` declare a token-less reading with a velocity readout
//! (the mini-DiT family, then every DiT).
//!
//! One `forward` pass in that reading, with no `embed` and no `attention`:
//! a `[rows, width]` f32 latent channel bound as `input("latents")`, a
//! `[1]` timestep channel bound as `input("timestep")`, every other port
//! the reading declares bound to a plausible constant (an axis-position
//! grid, an encoded or zero context, unit lane vectors), and an epilogue
//! that does the whole step on the device: fire 0 SEEDS the latent with a
//! keyed Normal draw (the model's velocity over the zero cell is ignored),
//! and every fire `k >= 1` integrates step `k - 1` of a flow schedule,
//! `x <- x + dt · v`, advancing the timestep and step cells for the next
//! submit. Nothing crosses to the host per step but the fire's own `[rows,
//! width]` readback, taken once a fire so the ring never fills; the last
//! one is the answer.
//!
//! It cannot check the model's arithmetic (that is the family's parity
//! harness); it checks that the loop closes: the port feeds land, the
//! velocity comes back at the declared width, the loop-carried cells
//! advance, and the final latent is finite.

use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    reading: Option<String>,
    #[serde(default = "default_steps")]
    steps: u32,
    #[serde(default = "default_rows")]
    rows: u32,
    #[serde(default = "default_seed")]
    seed: u32,
    #[serde(default)]
    prompt: Option<String>,
}

fn default_steps() -> u32 {
    4
}
fn default_rows() -> u32 {
    16
}
fn default_seed() -> u32 {
    7
}

#[derive(Serialize)]
struct Output {
    reading: String,
    rows: u32,
    width: u32,
    steps: u32,
    /// The schedule stepped, `steps + 1` sigmas ending in 0.
    sigmas: Vec<f32>,
    /// Which ports were bound, in the reading's order.
    ports: Vec<String>,
    /// Mean and standard deviation of the final latent.
    mean: f32,
    std: f32,
    /// How many of its values are not finite (0 on a closed loop).
    non_finite: u32,
    /// Its first eight values.
    head: Vec<f32>,
    /// Mean absolute value of the latent after every fire, fire 0 (the
    /// seed) first.
    trace: Vec<f32>,
}

/// The `[rows, axes]` positions of a `1 × 1 × rows` grid, cut to `axes`.
fn positions_for(rows: u32, axes: u32) -> Vec<f32> {
    let grid = positions_grid(1, 1, rows, [0.0, 0.0, 0.0]);
    let mut cut = Vec::with_capacity((rows * axes) as usize);
    for row in grid.chunks(3) {
        for axis in 0..axes as usize {
            cut.push(row.get(axis).copied().unwrap_or(0.0));
        }
    }
    cut
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err(
            "this program drives a stateless `forward` reading; the bound model's pass kind is \
             not attention"
                .into(),
        );
    }
    let readings = model::readings();
    if readings.is_empty() {
        return Err(
            "this model declares no readings; the latent probe wants a latent reading".into(),
        );
    }
    let reading = match &input.reading {
        Some(name) => readings
            .iter()
            .find(|reading| &reading.name == name)
            .cloned()
            .ok_or_else(|| format!("this model declares no reading `{name}`"))?,
        None => readings
            .iter()
            .find(|reading| {
                !reading.takes_tokens && reading.readout == model::ReadoutKind::Velocity
            })
            .cloned()
            .ok_or("this model declares no token-less reading with a velocity readout")?,
    };
    if reading.has_kv || reading.takes_tokens {
        return Err(format!(
            "reading `{}` is a sequence reading (tokens and KV); the probe drives a float lane",
            reading.name
        ));
    }
    if reading.readout != model::ReadoutKind::Velocity {
        return Err(format!(
            "reading `{}` reads out {:?}; the probe integrates a velocity",
            reading.name, reading.readout
        ));
    }
    let latents = reading
        .ports
        .iter()
        .find(|port| port.kind == model::PortKind::Latents)
        .ok_or("a token-less reading declares a latents port")?;
    let rows = input.rows.max(1);
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && rows > max_rows {
        return Err(format!(
            "{rows} rows exceed the model's {max_rows} latent rows a pass"
        ));
    }
    let width = latents.width;
    let velocity_width = reading.readout_width;
    let steps = input.steps.max(1);

    // The schedule: the model's when it states one, else a plain linspace.
    let sched = match model::schedule() {
        Some(fact) => FlowMatchEuler::from_schedule(&fact, steps, None)?,
        None => FlowMatchEuler::from_sigmas(
            (0..steps).map(|i| 1.0 - i as f32 / steps as f32).collect(),
            1000,
        ),
    };
    // Fire k reads `ts[k]` and integrates `dts[k]`; fire 0 is the seed
    // (its model output is discarded, its dt is 0). One trailing 0 so the
    // last fire's advance gathers something.
    let fires = steps + 1;
    let mut ts: Vec<f32> = vec![sched.timestep(0)];
    ts.extend(sched.timesteps());
    ts.push(0.0);
    let mut dts: Vec<f32> = vec![0.0];
    dts.extend(sched.dts());
    dts.push(0.0);

    // Channels. The latent and the control words are loop-carried (the
    // epilogue takes and puts them); the readback is host-read once a fire.
    let cells = (rows * width) as usize;
    let x = Channel::from_shaped([rows, width], vec![0f32; cells]).named("latents");
    let t = Channel::from([ts[0]]).named("timestep");
    let step = Channel::from([0u32]).named("step");
    let rng = Channel::from(rng_state(input.seed)).named("rng");
    // The schedule's vectors are channels, read by step index: the op set
    // carries constants as scalars only.
    let dts_ch = Channel::from(dts).named("dts");
    let ts_ch = Channel::from(ts).named("ts");
    let out = Channel::new([rows, width], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("out");

    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.input(&latents.name, &x)?;
    let mut bound = vec![latents.name.clone()];
    let mut timestep_bound = false;
    let mut extras: Vec<Channel> = Vec::new();
    for port in &reading.ports {
        if port.name == latents.name {
            continue;
        }
        let ch = match port.kind {
            model::PortKind::LaneVector if !timestep_bound && port.width == 1 => {
                // The first lane vector is the timestep; the schedule
                // advances it.
                timestep_bound = true;
                t
            }
            model::PortKind::LaneVector => {
                Channel::from(vec![1.0f32; port.width as usize]).named(&port.name)
            }
            model::PortKind::AxisPositions => {
                Channel::from_shaped([rows, port.width], positions_for(rows, port.width))
                    .named(&port.name)
            }
            model::PortKind::Context => match (&input.prompt, model::reading("text")) {
                (Some(prompt), Some(text)) if text.readout_width == port.width => {
                    encode_text(prompt, "text").await?
                }
                _ => Channel::from_shaped([8, port.width], vec![0f32; 8 * port.width as usize])
                    .named(&port.name),
            },
            model::PortKind::Latents => {
                Channel::from_shaped([rows, port.width], vec![0f32; (rows * port.width) as usize])
                    .named(&port.name)
            }
            // A voxel port is a VAE tile's box on the third row axis (D8),
            // which is not a float lane this probe can plausibly fill.
            model::PortKind::Voxels => {
                return Err(format!(
                    "reading `{}` declares a voxel port `{}`; the latent probe drives rows, \
                     not a VAE tile",
                    reading.name, port.name
                ));
            }
        };
        pass.input(&port.name, &ch)?;
        bound.push(port.name.clone());
        extras.push(ch);
    }
    if !timestep_bound {
        return Err(format!(
            "reading `{}` declares no lane-vector port for the timestep",
            reading.name
        ));
    }

    pass.epilogue(move || {
        let k = step.take();
        let x_cur = x.take();
        let v = intrinsics::velocity(velocity_width);
        let dt = gather(dts_ch.read(), &k);
        let next = euler_step(&x_cur, &v, &dt);
        let r = rng.take();
        let fresh = noise([rows, width], &r);
        let first = broadcast(reshape(eq(&k, 0u32), [1, 1]), [rows, width]);
        let x_next = select(&first, &fresh, &next);
        x.put(&x_next);
        out.put(&x_next);
        let k_next = &k + 1u32;
        t.take();
        t.put(gather(ts_ch.read(), &k_next));
        step.put(&k_next);
        rng.put(&r + &Tensor::constant([0u32, 1u32]));
    });

    let pipe = Pipeline::new();
    let mut trace = Vec::with_capacity(fires as usize);
    let mut last: Vec<f32> = Vec::new();
    for fire in 0..fires {
        pass.submit(&pipe)
            .with_context(|| format!("submit fire {fire}"))?;
        last = out
            .take_host::<Vec<f32>>()
            .await
            .with_context(|| format!("readback after fire {fire}"))?;
        let mean_abs = last.iter().map(|v| v.abs()).sum::<f32>() / last.len().max(1) as f32;
        trace.push(mean_abs);
    }
    pipe.close();
    drop(extras);

    let n = last.len().max(1) as f32;
    let mean = last.iter().sum::<f32>() / n;
    let var = last.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    Ok(Output {
        reading: reading.name.clone(),
        rows,
        width,
        steps,
        sigmas: sched.sigmas.clone(),
        ports: bound,
        mean,
        std: var.sqrt(),
        non_finite: last.iter().filter(|v| !v.is_finite()).count() as u32,
        head: last.iter().take(8).copied().collect(),
        trace,
    })
}
