//! The guest's denoise loop over flux2-mini's `denoise` reading, driven through
//! `CerebrasEngine::submit` as the runtime drives a pass.

use engine::caps::Capabilities;
use engine::channel::{ChannelId, ChannelRegistration, ChannelSeed};
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step, StepVoxels};
use engine::program::{InstanceBinding, InstanceId, ProgramRegistration};
use engine::{Channels, Engine, Programs};
use engine_cerebras::reading::Denoise;
use eta_dsl::prelude::*;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use poem_ir::Request;

use super::{CHANNELS, MINI, grid, rows, trace};

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The guest's denoise loop: latent, timestep, guidance and context in channels; each step's
/// epilogue Euler-steps the latent by the velocity and sets the next timestep (latent to `out`).
pub struct Guest {
    traced: eta_dsl::Traced,
    /// Each channel's gid and, for a seeded one, its first cell.
    seeds: Vec<(u64, Vec<u8>)>,
    latent: u64,
    timestep: u64,
    guidance: u64,
    context: u64,
    out: u64,
}

pub fn velocity_decl(voxels: u32) -> OutputDecl {
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

pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn guest(sigmas: &[f32], train: u32, latent0: &[f32], context: &[f32]) -> Guest {
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
pub struct Ran {
    pub latents: Vec<Vec<f32>>,
}

/// The guest's loop of `steps` steps over the text rows `context` on an engine `load`ed with
/// flux2-mini, submitted a step a frame.
pub fn loop_on(
    engine: &mut engine_cerebras::CerebrasEngine,
    caps: &Capabilities,
    steps: u32,
    context: &[f32],
) -> Ran {
    loop_on_rows(engine, caps, steps, context, rows() as u32)
}

/// [`loop_on`] with a context lane of `context_rows` rows (fewer than the program's class).
pub fn loop_on_rows(
    engine: &mut engine_cerebras::CerebrasEngine,
    caps: &Capabilities,
    steps: u32,
    context: &[f32],
    context_rows: u32,
) -> Ran {
    let trace = trace();
    let image_rows = grid()[1] * grid()[2] / 4;
    // the guest: the poem's schedule as the runtime states it to an inferlet
    let d = Denoise::of(MINI, steps, Some(image_rows)).expect("a flow schedule");
    let g = guest(&d.sigmas, d.train_steps, &super::latents(), context);
    let container = g.traced.container().clone();
    let decls = vec![vec![velocity_decl(0)], vec![velocity_decl(0)]];
    let bound = eta_ir::validate::bind(container.clone(), decls, EngineCaps::default())
        .expect("the guest binds on the velocity lanes");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let program_id = engine
        .register(&ProgramRegistration {
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
            .open(&ChannelRegistration {
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
        .bind(&InstanceBinding {
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
            lanes: vec![lane("context", context_rows), lane("latents", image_rows)],
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
            .take(instance, dense(g.out))
            .expect("the latent channel reads")
            .expect("the epilogue put the step's latent");
        latents.push(f32s(&cell));
        eprintln!("step {}: {} s", latents.len() - 1, t0.elapsed().as_secs());
    }
    Ran { latents }
}
