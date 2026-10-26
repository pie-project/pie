//! A probe of the float-input loop on a model with a token-less velocity reading:
//! one pass on the lane that reads `latents`, the whole flow-Euler step in its
//! epilogue (fire 0 seeds a keyed Normal draw), every other input a constant.
//!
//! It checks that the loop closes, not the model's arithmetic: the input feeds
//! land, the velocity comes back at the declared width, the loop-carried cells
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
    /// Which inputs were bound, `latents` first.
    inputs: Vec<String>,
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

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let readings = model::switch("reading")
        .ok_or("this model declares no `reading` switch; the latent probe wants a latent reading")?
        .values;
    if let Some(name) = &input.reading
        && !readings.contains(name)
    {
        return Err(format!("this model declares no reading `{name}`"));
    }
    let streams: Vec<Option<String>> = match model::switch("stream") {
        Some(decl) => decl.values.into_iter().map(Some).collect(),
        None => vec![None],
    };
    let (reading, stream, lane, latents) = readings
        .iter()
        .filter(|reading| input.reading.as_ref().is_none_or(|name| name == *reading))
        .flat_map(|reading| streams.iter().map(move |stream| (reading, stream)))
        .find_map(|(reading, stream)| {
            let mut switches = vec![("reading", reading.as_str())];
            if let Some(stream) = stream {
                switches.push(("stream", stream.as_str()));
            }
            let lane = model::lane_of(&switches).ok()?;
            let velocity = lane.outputs.iter().any(|output| output.name == "velocity");
            if input.reading.is_none() && (lane.embeds || !velocity) {
                return None;
            }
            let latents = lane.inputs.iter().find(|i| i.name == "latents")?.clone();
            Some((reading.clone(), stream.clone(), lane, latents))
        })
        .ok_or(
            "this model declares no token-less lane reading out a velocity that reads `latents`",
        )?;
    if lane.state || lane.embeds {
        return Err(format!(
            "reading `{reading}` is a sequence reading (tokens and KV); the probe drives a float lane"
        ));
    }
    if !lane.outputs.iter().any(|output| output.name == "velocity") {
        return Err(format!(
            "reading `{reading}` reads out no velocity; the probe integrates one"
        ));
    }
    let rows = input.rows.max(1);
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && rows > max_rows {
        return Err(format!(
            "{rows} rows exceed the model's {max_rows} latent rows a pass"
        ));
    }
    let width = latents.width;
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
    pass.switch(0, "reading", &reading)?;
    if let Some(stream) = &stream {
        pass.switch(0, "stream", stream)?;
    }
    pass.input(0, "latents", &x)?;
    let mut bound = vec![latents.name.clone()];
    let mut extras: Vec<Channel> = Vec::new();
    if !lane.inputs.iter().any(|i| i.name == "timestep") {
        return Err(format!(
            "reading `{reading}`'s `{}` lane reads no `timestep`",
            stream.as_deref().unwrap_or("default")
        ));
    }
    pass.input(0, "timestep", &t)?;
    bound.push("timestep".to_string());
    for decl in &lane.inputs {
        if decl.name == "latents" || decl.name == "timestep" {
            continue;
        }
        let ch = match decl.axis {
            model::InputAxis::Lane => {
                Channel::from(vec![1.0f32; decl.width as usize]).named(&decl.name)
            }
            model::InputAxis::Row => {
                Channel::from_shaped([rows, decl.width], vec![0f32; (rows * decl.width) as usize])
                    .named(&decl.name)
            }
            // A voxel input is a VAE tile's box on the third row axis (D8),
            // which is not a float lane this probe can plausibly fill.
            model::InputAxis::Voxel => {
                return Err(format!(
                    "reading `{}` reads a voxel input `{}`; the latent probe drives rows, \
                     not a VAE tile",
                    reading, decl.name
                ));
            }
        };
        pass.input(0, &decl.name, &ch)?;
        bound.push(decl.name.clone());
        extras.push(ch);
    }

    pass.epilogue(move || {
        let k = step.take();
        let x_cur = x.take();
        let v = intrinsics::output_rows(0, "velocity", rows);
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
        reading,
        rows,
        width,
        steps,
        sigmas: sched.sigmas.clone(),
        inputs: bound,
        mean,
        std: var.sqrt(),
        non_finite: last.iter().filter(|v| !v.is_finite()).count() as u32,
        head: last.iter().take(8).copied().collect(),
        trace,
    })
}
