//! flux2-mini's denoise through the cerebras engine as Pie drives any engine (GENERIC.md M1g):
//! the model loaded from a diffusers checkpoint through the import contract, the wafer program
//! compiled with no capture (its static arrays the checkpoint's, cut by the engine), and the
//! denoise loop a guest program's: its epilogue takes each step's velocity, Euler-steps the
//! latent it holds in a channel over the poem's flow sigmas, and sets the next timestep; each
//! fire's inputs come off the channels its lanes feed from.
//!
//! - The static arrays the engine makes from the checkpoint are, word for word, the ones the
//!   compile writes from engine-cpu's capture of the whole step.
//! - On the host fabric (`stream_fabric = "host"`: the program's rows as engine-cpu's forward
//!   over the checkpoint makes them), the guest's 2-step loop lands bit for bit where the host's
//!   own loop does, and within the bf16 tolerance of diffusers'.
//! - `flux_2_minis_denoise_on_the_fabric` (ignored) runs the same request on the simulator or a
//!   system: `READING_PROGRAM` (the no-capture compile, linked in `READING_LINK`),
//!   `READING_OUT` (the latents after each step, `latent<i>.f32`), against
//!   `READING_GOLDEN` (diffusers' `diff_latent<n>.f32`, engine-cpu's `cpu_latent<i>.f32`).

mod common;

use std::path::{Path, PathBuf};

use common::{CHANNELS, MINI, grid, rows, trace};
use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration, ChannelSeed};
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step, StepVoxels};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, InstanceId, ProgramRegistration};
use engine_cerebras::reading::{Denoise, Reading};
use engine_cpu::Forward;
use eta_dsl::prelude::*;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use poem_ir::{Request, Trace};

/// diffusers' latent after the 2-step loop at 64 rows a lane.
const LOOP_GOLDEN: &[u8] = include_bytes!("flux_2_mini_loop2_latent.f32");

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn rel_cos(a: &[f32], b: &[f32]) -> (f64, f64) {
    let dot = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| f64::from(*p) * f64::from(*q))
            .sum::<f64>()
    };
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    (
        (dot(&d, &d) / dot(b, b)).sqrt(),
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt(),
    )
}

/// The program compiled with no capture into `dir`.
fn compile(trace: &Trace, dir: &Path, gold: Option<&Path>) {
    let o = kernels_cerebras::stream::Options {
        reading: "denoise".into(),
        rows: rows() as u32,
        round16: true,
        upto: None,
        taps: Vec::new(),
    };
    kernels_cerebras::stream::compile(trace, &o, dir, gold).expect("the program compiles");
}

/// The step engine-cpu's test runs, its lanes the reading's classes.
fn step_of<'a>(
    trace: &Trace,
    inputs: &'a [(String, Vec<f32>)],
) -> engine_cpu::forward::StreamStep<'a> {
    let word = |stream: &str| {
        let r = Request::new(rows() as u32, false)
            .switched("reading", "denoise")
            .switched("stream", stream);
        trace.facts.word(&r)
    };
    engine_cpu::forward::StreamStep {
        lanes: vec![
            engine_cpu::forward::StreamLane {
                word: word("context"),
                rows: rows(),
                clip: None,
            },
            engine_cpu::forward::StreamLane {
                word: word("latents"),
                rows: (grid()[1] * grid()[2] / 4) as usize,
                clip: Some(grid()),
            },
        ],
        inputs: inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect(),
    }
}

/// The static arrays the engine cuts from the checkpoint are the capture compile's, word for
/// word; the no-capture compile states the same arrays, with no words of a value on file.
#[test]
fn flux_2_minis_statics_are_the_checkpoints() {
    let trace = trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (gold, with, bare, ckpt) = (
        tmp.path().join("gold"),
        tmp.path().join("with"),
        tmp.path().join("bare"),
        tmp.path().join("ckpt"),
    );
    // the capture compile: every array's words from engine-cpu's step
    let mut forward = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(&trace, n, c)
    })
    .expect("flux2-mini loads");
    let inputs = common::inputs();
    let step = step_of(&trace, &inputs);
    capture(&mut forward, &trace, &step, &gold);
    compile(&trace, &with, Some(&gold));
    compile(&trace, &bare, None);
    let (with, bare) = (
        Reading::open(&with).expect("the capture compile's host.json"),
        Reading::open(&bare).expect("the bare compile's host.json"),
    );
    assert_eq!(with.h2d.len(), bare.h2d.len());

    // the checkpoint, through the import contract, as the engine lands it
    common::checkpoint(&trace, &ckpt);
    let contract = common::contract_for(&trace, &ckpt).expect("the contract reads the checkpoint");
    let landed = engine_cerebras::prepare::land(&trace, &contract, &ckpt).expect("it lands");
    let mut weights = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        let p =
            <engine_cerebras::prepare::Landed as kernels_cerebras::tiles::Params>::get(&landed, n)
                .unwrap_or_else(|| panic!("{n} landed"));
        p.bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(c)
            .map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16))
            .collect()
    })
    .expect("flux2-mini loads from its checkpoint");
    let statics = bare
        .statics(&mut weights, &step)
        .expect("the statics compute");
    let (mut spans, mut files) = (0, 0);
    for ((a, b), got) in with.h2d.iter().zip(&bare.h2d).zip(&statics) {
        assert_eq!((&a.sym, a.rect.x, a.rect.y), (&b.sym, b.rect.x, b.rect.y));
        let Some(got) = got else {
            assert!(b.entry.is_some(), "`{}`: no words and no entry", b.sym);
            continue;
        };
        let want = kernels_cerebras::npy::read_f32_or_i32(a.file.as_deref().expect("on file"))
            .expect("an h2d file")
            .1;
        let want: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, &want, "`{}` at ({}, {})", b.sym, b.rect.x, b.rect.y);
        if b.statics.is_some() {
            assert!(b.file.is_none(), "`{}`: a value's words on file", b.sym);
            spans += 1;
        } else {
            files += 1;
        }
    }
    eprintln!("{spans} arrays from the checkpoint, {files} the program's own");
    assert!(spans > 0 && files > 0);
}

/// Every node's values of the step, as `wafer-compile stream --values` reads a capture.
fn capture(
    forward: &mut Forward,
    trace: &Trace,
    step: &engine_cpu::forward::StreamStep<'_>,
    dir: &Path,
) {
    use engine_cpu::forward::Values;
    use poem_ir::Operands;
    let all: Vec<usize> = (0..trace.nodes.len()).collect();
    let (_, caps) = forward.stream_capture(step, &all).expect("the step runs");
    std::fs::create_dir_all(dir).expect("a capture directory");
    let mut index = Vec::new();
    for &node in &all {
        let (mut ins, mut outs) = (0, 0);
        let mut files = Vec::new();
        for c in caps.iter().filter(|c| c.node == node) {
            let k = if c.input { &mut ins } else { &mut outs };
            let name = format!("n{node}_{}{k}.npy", if c.input { "in" } else { "out" });
            *k += 1;
            let v: Vec<f32> = match &c.values {
                Values::F32(v) => v.clone(),
                Values::I32(v) => v.iter().map(|&x| x as f32).collect(),
            };
            kernels_cerebras::npy::write_f32(&dir.join(&name), &[c.rows, c.width], &v)
                .expect("a capture file");
            files.push(format!("\"{name}\": {}", c.value.0));
        }
        if !files.is_empty() {
            index.push(format!(
                "\"{node}\": {{\"op\": \"{}\", \"values\": {{{}}}}}",
                trace.nodes[node].op.name(),
                files.join(", ")
            ));
        }
    }
    std::fs::write(
        dir.join("nodes.json"),
        format!("{{{}}}\n", index.join(",\n")),
    )
    .expect("the capture's index");
}

/// The guest's denoise loop: the latent, the timestep, the guidance and the context in
/// channels; an epilogue a step that takes the latents lane's velocity, Euler-steps the latent
/// from sigma `k` to `k + 1` and sets the next timestep. Each step's latent also goes to `out`.
struct Guest {
    traced: eta_dsl::Traced,
    /// Each channel's gid and, for a seeded one, its first cell.
    seeds: Vec<(u64, Vec<u8>)>,
    latent: u64,
    timestep: u64,
    guidance: u64,
    context: u64,
    out: u64,
}

fn velocity_decl(voxels: u32) -> OutputDecl {
    let _ = voxels;
    OutputDecl {
        name: "velocity".into(),
        axis: Axis::Voxel,
        width: CHANNELS as u32,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: true,
    }
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn guest(sigmas: &[f32], train: u32, latent0: &[f32], context: &[f32]) -> Guest {
    let voxels = grid()[1] * grid()[2];
    let c = CHANNELS as u32;
    let x = Channel::from(latent0.to_vec());
    let t = Channel::from([sigmas[0] * train as f32]);
    let g = Channel::from([4.0f32]);
    let ctx = Channel::from(context.to_vec());
    let k = Channel::from([0u32]);
    let sig = Channel::from(sigmas.to_vec());
    let out = Channel::new([voxels * c], eta_dsl::dtype::f32);
    let seeds = vec![
        (x.gid(), bytes(latent0)),
        (t.gid(), bytes(&[sigmas[0] * train as f32])),
        (g.gid(), bytes(&[4.0])),
        (ctx.gid(), bytes(context)),
        (k.gid(), 0u32.to_le_bytes().to_vec()),
        (sig.gid(), bytes(sigmas)),
    ];
    let ids = (x.gid(), t.gid(), g.gid(), ctx.gid(), out.gid());
    let (ex, et, eo) = (x.clone(), t.clone(), out.clone());
    let mut b = Builder::new(32, 16);
    b.outputs(vec![
        vec![velocity_decl(voxels)],
        vec![velocity_decl(voxels)],
    ]);
    // the channels the lanes feed the model's inputs from
    for fed in [&x, &t, &g, &ctx] {
        b.bind_input(fed);
    }
    b.stage(Stage::Epilogue, move || {
        let step = k.take();
        let next = &step + 1u32;
        let v = intrinsics::output_rows(1, "velocity", voxels);
        let s = sig.read();
        let dt = &gather(&s, &next) - &gather(&s, &step);
        let x0 = reshape(ex.take(), [voxels, c]);
        let x1 = &x0 + &(&v * &broadcast(reshape(&dt, [1, 1]), [voxels, c]));
        ex.put(reshape(&x1, [voxels * c]));
        eo.put(reshape(&x1, [voxels * c]));
        et.take();
        et.put(&gather(&s, &next) * (train as f32));
        k.put(&next);
    });
    let traced = b.build().expect("the guest's loop traces");
    Guest {
        traced,
        seeds,
        latent: ids.0,
        timestep: ids.1,
        guidance: ids.2,
        context: ids.3,
        out: ids.4,
    }
}

/// What a run of the guest's loop gives: each step's latent.
struct Ran {
    latents: Vec<Vec<f32>>,
}

/// The engine loaded from `ckpt` with `program` (and its link), the guest's loop of `steps`
/// steps submitted a step a frame, as the runtime submits a pass.
fn run_loop(
    program: &Path,
    link: Option<PathBuf>,
    ckpt: &Path,
    fabric: Option<&str>,
    steps: u32,
) -> Ran {
    let trace = trace();
    let cache = tempfile::tempdir().expect("a cache directory");
    let mut cfg = engine_cerebras::Config {
        cache_dir: Some(cache.path().to_path_buf()),
        num_threads: 8,
        ..engine_cerebras::Config::default()
    };
    cfg.debug = Some(engine_cerebras::config::Debug {
        program: Some(program.to_path_buf()),
        link,
        stream_fabric: fabric.map(str::to_string),
        ..engine_cerebras::config::Debug::default()
    });
    let mut engine = engine_cerebras::CerebrasEngine::new(cfg, common::contract_for);
    let image_rows = grid()[1] * grid()[2] / 4;
    let loaded = engine
        .load(LoadRequest {
            trace: trace.clone(),
            checkpoint: Checkpoint::Path(ckpt.to_path_buf()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 2 * rows() as u32,
                buckets: Vec::new(),
                max_adapters: 0,
                page_size: 16,
                max_context: 4 * rows() as u32,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: Some(grid()[1] * grid()[2]),
                max_clips: Some(1),
            },
            residency: Residency::default(),
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("flux2-mini loads on the cerebras engine");

    // the guest: the poem's schedule as the runtime states it to an inferlet
    let d = Denoise::of(MINI, steps, Some(image_rows)).expect("a flow schedule");
    let g = guest(
        &d.sigmas,
        d.train_steps,
        &common::latents(),
        &common::context(),
    );
    let container = g.traced.container().clone();
    let decls = vec![vec![velocity_decl(0)], vec![velocity_decl(0)]];
    let bound = eta_ir::validate::bind(container.clone(), decls, EngineCaps::default())
        .expect("the guest binds on the velocity lanes");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let program_id = engine
        .register_program(&ProgramRegistration {
            program_hash: bound.hash,
            launch,
            ..Default::default()
        })
        .expect("the guest registers");
    let order = g.traced.channel_order().to_vec();
    let dense = |gid: u64| order.iter().position(|&o| o == gid).expect("a channel") as u32;
    let mut channels = Vec::new();
    for (i, decl) in container.channels.iter().enumerate() {
        let id = ChannelId(100 + i as u64);
        engine
            .register_channel(&ChannelRegistration {
                id,
                shape: decl.shape.dims().to_vec(),
                dtype: decl.dtype,
                host_role: decl.host_role,
                seeded: decl.seeded,
                extern_dir: None,
                capacity: decl.capacity.max(1),
                extern_name: Vec::new(),
            })
            .expect("a channel registers");
        channels.push(id);
    }
    let seeds = g
        .seeds
        .iter()
        .map(|(gid, b)| ChannelSeed {
            channel: dense(*gid),
            bytes: b.clone(),
        })
        .collect();
    let total = (rows() + image_rows as usize) as u32;
    let instance: InstanceId = engine
        .bind_instance(&InstanceBinding {
            program: program_id,
            channels: channels.clone(),
            seeds,
            geometry: GeometryClass::Host,
            device_resolved: Default::default(),
            extents: engine::program::BindExtents {
                row_count: total,
                token_count: total,
                sampled_rows: total,
                query_len: total,
                ..engine::program::BindExtents::default()
            },
        })
        .expect("the guest binds")
        .id;

    // a lane a class; each feeds the inputs its class reads, off the guest's channels
    let caps = loaded.caps;
    let word = |stream: &str| {
        trace.facts.word(
            &Request::new(rows() as u32, false)
                .switched("reading", "denoise")
                .switched("stream", stream),
        )
    };
    let feeds = |w: u64| -> Vec<InputFeed> {
        let id = |name: &str| {
            trace
                .inputs
                .iter()
                .position(|i| i.name == name)
                .expect("an input") as u32
        };
        caps.inputs
            .iter()
            .filter(|i| i.read_by(w))
            .map(|i| InputFeed {
                id: id(&i.name),
                channel: channels[dense(match i.name.as_str() {
                    "latents" => g.latent,
                    "timestep" => g.timestep,
                    "guidance" => g.guidance,
                    "context" => g.context,
                    other => panic!("the guest binds no `{other}`"),
                }) as usize],
            })
            .collect()
    };
    let lane = |stream: &str, n: u32| {
        let w = word(stream);
        Lane {
            slot: 0,
            word: w,
            tokens: vec![0; n as usize],
            readout: Readout::Rows((0..n).collect()),
            group: Some(0),
            inputs: feeds(w),
            kv_less: true,
            ..Lane::default()
        }
    };
    let mut latents = Vec::new();
    for _ in 0..d.steps() {
        let frame = FrameSubmission::of(Step {
            lanes: vec![lane("context", rows() as u32), lane("latents", image_rows)],
            attachments: vec![Attachment {
                lane: 0,
                lanes: 2,
                instance,
                adapter_only: false,
            }],
            media: Vec::new(),
            voxels: vec![StepVoxels {
                lane: 1,
                clips: vec![grid()],
                payload: Vec::new(),
            }],
        });
        let t0 = std::time::Instant::now();
        let mut ticket = engine.submit(&frame).expect("the step runs");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        let cell = engine
            .take_channel(instance, dense(g.out))
            .expect("the latent channel reads")
            .expect("the epilogue put the step's latent");
        latents.push(f32s(&cell));
        eprintln!("step {}: {} s", latents.len() - 1, t0.elapsed().as_secs());
    }
    Ran { latents }
}

/// The host's own loop over the same schedule, each velocity engine-cpu's whole step.
fn host_loop(steps: u32) -> Vec<Vec<f32>> {
    let trace = trace();
    let mut forward = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(&trace, n, c)
    })
    .expect("flux2-mini loads");
    let d = Denoise::of(MINI, steps, Some(grid()[1] * grid()[2] / 4)).expect("a flow schedule");
    let mut inputs = common::inputs();
    let mut out = Vec::new();
    let (_, vs) = d
        .run(&mut inputs, 2, |_, ins| {
            let step = step_of(&trace, ins);
            let outs = forward.stream(&step)?;
            Ok(match outs.into_iter().find(|o| o.name == "velocity") {
                Some(engine_cpu::forward::Output {
                    values: engine_cpu::forward::Values::F32(v),
                    ..
                }) => v,
                _ => return Err("no velocity".into()),
            })
        })
        .expect("the host's loop runs");
    let mut x = common::latents();
    for (i, v) in vs.iter().enumerate() {
        let dt = d.sigmas[i + 1] - d.sigmas[i];
        for (x, v) in x.iter_mut().zip(v) {
            *x += dt * v;
        }
        out.push(x.clone());
    }
    out
}

/// The guest's loop through the engine on the host fabric: the host's loop bit for bit, and
/// diffusers' within the bf16 tolerance.
#[test]
fn flux_2_minis_denoise_runs_through_the_engine() {
    let trace = trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (prog, ckpt) = (tmp.path().join("prog"), tmp.path().join("ckpt"));
    compile(&trace, &prog, None);
    common::checkpoint(&trace, &ckpt);
    let ran = run_loop(&prog, None, &ckpt, Some("host"), 2);
    let want = host_loop(2);
    for (i, (got, want)) in ran.latents.iter().zip(&want).enumerate() {
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the latent after step {i}"
        );
    }
    if rows() == 64 {
        let (rel, cos) = rel_cos(ran.latents.last().expect("a latent"), &f32s(LOOP_GOLDEN));
        eprintln!(
            "flux2-mini through the engine, 2 steps, vs diffusers: rel L2 {rel:.3e}, cosine {cos:.8}"
        );
        assert!(rel <= 1e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
    }
}

/// The same request on the fabric (the simulator, or the system at `READING_CMADDR`): the
/// program `READING_PROGRAM` compiled with no capture (`wafer-compile stream`), linked in
/// `READING_LINK`, its checkpoint written to `READING_OUT/ckpt`; each step's latent written to
/// `READING_OUT/latent<i>.f32` and held to `READING_GOLDEN`'s `diff_latent<n>.f32` (diffusers,
/// after the last step) and `cpu_latent<i>.f32` (engine-cpu's loop).
#[test]
#[ignore = "runs on the fabric simulator; set READING_PROGRAM, READING_LINK, READING_OUT"]
fn flux_2_minis_denoise_on_the_fabric() {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let (prog, link, out) = (
        PathBuf::from(var("READING_PROGRAM")),
        PathBuf::from(var("READING_LINK")),
        PathBuf::from(var("READING_OUT")),
    );
    let steps: u32 = std::env::var("READING_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    // the engine's log (each step's cycles off the stamps, the static arrays' write)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .try_init();
    let trace = trace();
    let ckpt = out.join("ckpt");
    common::checkpoint(&trace, &ckpt);
    let ran = run_loop(&prog, Some(link), &ckpt, None, steps);
    let golden = std::env::var("READING_GOLDEN").ok().map(PathBuf::from);
    let mut ok = true;
    for (i, x) in ran.latents.iter().enumerate() {
        std::fs::write(out.join(format!("latent{}.f32", i + 1)), bytes(x)).expect("a latent");
        for (name, what) in [
            (format!("cpu_latent{}.f32", i + 1), "engine-cpu"),
            (format!("diff_latent{}.f32", i + 1), "diffusers"),
        ] {
            let Some(p) = golden
                .as_ref()
                .map(|g| g.join(&name))
                .filter(|p| p.is_file())
            else {
                continue;
            };
            let (rel, cos) = rel_cos(x, &f32s(&std::fs::read(&p).expect("a golden")));
            ok &= rel <= 1e-2 && cos >= 0.999;
            println!(
                "READING_ENGINE latent after {} vs {what}: rel L2 {rel:.3e}, cosine {cos:.8}",
                i + 1
            );
        }
    }
    assert!(ok, "a latent is past the bf16 tolerance");
}
