//! A guest reads a lane output stacked over two layers and an i32 readout
//! output by name through the fire, against host references.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use checkpoint::file::emit::{self, Object};
use checkpoint::serving::Stamp;
use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, ProgramRegistration};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, ForwardHybrid, Input, Request, StateRows, Trace, Weight, ops};
use poem_ir::InputAxis;

use crate::backend;

pub(super) const NAME: &str = "outputs-mini";
pub(super) const WIDTH: u32 = 16;
pub(super) const VOCAB: u32 = 16;
pub(super) const LAYERS: u32 = 2;
const THETA: f32 = 10_000.0;

pub(super) struct Outputs;

impl ForwardHybrid for Outputs {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let t = inputs.input("t", Dtype::F32, InputAxis::Lane, 1, None);
        for layer in 0..LAYERS {
            let scale = (layer + 1) as f32;
            let emb = ops::elemwise::sinusoid(&t, WIDTH, THETA, true, scale);
            poem::output("stack", &emb, None);
        }
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let x = ops::gather::gather_rows(&x, &inputs.readout_rows());
        poem::output("logits", &x, None);
        poem::output("ids", &ops::sampling::argmax(&[&x]), None);
    }
}

/// The miniature's table read alone: a lane of it feeds no guest input, so
/// it fires without an attachment and the host reads its rows.
pub(super) struct Rows;

impl ForwardHybrid for Rows {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let x = ops::gather::gather_rows(&x, &inputs.readout_rows());
        poem::output("logits", &x, None);
    }
}

fn bf(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}

pub(super) fn planes() -> BTreeMap<String, (Vec<u64>, Vec<f32>)> {
    let mut state = 0x5eed_u64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let mut planes = BTreeMap::new();
    let values = (0..VOCAB * WIDTH).map(|_| bf(unit())).collect();
    planes.insert(
        "embed".to_string(),
        (vec![u64::from(VOCAB), u64::from(WIDTH)], values),
    );
    planes
}

pub(super) fn write(path: &Path, planes: &BTreeMap<String, (Vec<u64>, Vec<f32>)>) {
    let bytes: Vec<(String, Vec<u64>, Vec<u8>)> = planes
        .iter()
        .map(|(name, (shape, values))| {
            let raw = values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            (name.clone(), shape.clone(), raw)
        })
        .collect();
    let objects: Vec<Object<'_>> = bytes
        .iter()
        .map(|(name, shape, raw)| Object::leaf(name, shape.clone(), ztensor::Leaf::BF16, raw))
        .collect();
    emit::write(
        path,
        &Stamp::of(backend::NAME, NAME),
        &BTreeMap::new(),
        4096,
        &objects,
        |o, p, _| panic!("{o}/{p} is not streamed here"),
    )
    .expect("the artifact writes");
}

pub(super) fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, backend::PLATFORM)
        .map_err(|why| why.to_string())
}

fn epilogue() -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded: false,
    };
    TraceContainer {
        names: vec!["ids".to_string(), "stack".to_string()],
        channels: vec![
            decl(Shape::matrix(1, 1), EtaDtype::I32, HostRole::Reader),
            decl(
                Shape::matrix(LAYERS, WIDTH),
                EtaDtype::F32,
                HostRole::Reader,
            ),
            decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, 1),
                    dtype: EtaDtype::I32,
                },
                Op::ChanPut { chan: 0, value: 0 },
                Op::Output {
                    index: 1,
                    lane: 0,
                    shape: Shape::matrix(LAYERS, WIDTH),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// The miniature loaded one frame in flight, an instance of the epilogue
/// bound, and `t` published.
struct Bound {
    engine: backend::Engine,
    instance: engine::InstanceId,
    channels: Vec<ChannelId>,
    t_input: u32,
    /// The word of a pass of [`ROWS`] tokens.
    word: u64,
    planes: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    dir: std::path::PathBuf,
    /// The buffers the device had created when the load was ready.
    ready: u64,
}

const T: f32 = 0.375;

/// The buffers `engine`'s device has created.
fn creations(engine: &mut backend::Engine) -> u64 {
    engine
        .shell_mut()
        .expect("loaded")
        .device()
        .books()
        .creations()
}

/// The tokens of a pass.
const ROWS: u32 = 3;

/// A model loaded one frame in flight over the miniature's table.
pub(super) struct Loaded {
    pub(super) engine: backend::Engine,
    pub(super) caps: engine::load::Loaded,
    pub(super) trace: Trace,
    pub(super) planes: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    pub(super) dir: std::path::PathBuf,
}

/// `None` without a device. `tag` names the scratch directory.
pub(super) fn loaded(tag: &str, model: &impl ForwardHybrid) -> Option<Loaded> {
    loaded_in(tag, model, None)
}

/// [`loaded`] in a pool of `chunk`-byte chunks where `chunk` says.
pub(super) fn loaded_in(
    tag: &str,
    model: &impl ForwardHybrid,
    chunk: Option<u64>,
) -> Option<Loaded> {
    let boot = backend::DeviceBoot {
        chunk_bytes: chunk,
        ..backend::DeviceBoot::default()
    };
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return None;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-{tag}-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    let planes = planes();
    write(&path, &planes);

    let trace = poem::trace_hybrid(NAME, model, backend::PLATFORM);
    let mut engine = backend::Engine::new(boot, contract_for);
    let caps = engine
        .load(LoadRequest {
            trace: trace.clone(),
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 16,
                buckets: vec![16],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: backend::ALLOWANCE,
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("the miniature loads");
    Some(Loaded {
        engine,
        caps,
        trace,
        planes,
        dir,
    })
}

/// `None` without a device. `tag` names the scratch directory.
fn bound(tag: &str) -> Option<Bound> {
    let Loaded {
        mut engine,
        caps: loaded,
        trace,
        planes,
        dir,
    } = loaded(tag, &Outputs)?;
    let ready = creations(&mut engine);
    let facts = trace.facts.clone();
    let t_input = trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;

    let mut decls: Vec<OutputDecl> = Vec::new();
    for decl in loaded.caps.classes.classes.iter().flat_map(|c| &c.outputs) {
        if decls.iter().any(|seen| seen.name == decl.name) {
            continue;
        }
        decls.push(OutputDecl {
            name: decl.name.clone(),
            axis: match decl.axis {
                InputAxis::Lane => Axis::Lane,
                InputAxis::Row => Axis::Row,
                InputAxis::Voxel => Axis::Voxel,
            },
            width: decl.width,
            dtype: match decl.dtype {
                poem_ir::Dtype::I32 => EtaDtype::I32,
                _ => EtaDtype::F32,
            },
            layers: decl.layers,
            readout: decl.readout,
        });
    }
    let bound = eta_ir::validate::bind(epilogue(), vec![decls], EngineCaps::default())
        .expect("the epilogue binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0x0e7,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots =
        super::slots::Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channels: Vec<ChannelId> = [
        (vec![1, 1], EtaDtype::I32, HostRole::Reader),
        (vec![LAYERS, WIDTH], EtaDtype::F32, HostRole::Reader),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
    ]
    .into_iter()
    .enumerate()
    .map(|(at, (shape, dtype, host_role))| {
        let id = ChannelId(at as u64 + 1);
        slots.register(
            &mut engine,
            ChannelRegistration {
                id,
                shape,
                dtype: ChanDType::Concrete(dtype),
                host_role,
                seeded: false,
                extern_dir: None,
                capacity: 2,
                extern_name: Vec::new(),
                slots: Vec::new(),
            },
        );
        id
    })
    .collect();
    let rows = ROWS;
    let binding = slots.binding(
        &engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            extents: engine::program::BindExtents {
                row_count: 1,
                token_count: rows,
                sampled_rows: 1,
                query_len: rows,
                ..engine::program::BindExtents::default()
            },
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            ring_slots: Vec::new(),
            adapter_slots: Vec::new(),
            ..Default::default()
        },
        &channels,
    );
    let instance = engine
        .bind_instance(&binding)
        .expect("the instance binds")
        .id;
    assert!(
        engine
            .publish_channel(instance, 2, &T.to_le_bytes())
            .expect("the timestep publishes"),
        "the ring had room"
    );
    Some(Bound {
        engine,
        instance,
        channels,
        t_input,
        word: facts.word(&Request::new(rows, false)),
        planes,
        dir,
        ready,
    })
}

#[test]
fn a_guest_reads_lane_layer_and_id_outputs_by_name() {
    let Some(Bound {
        mut engine,
        instance,
        channels,
        t_input,
        word,
        planes,
        dir,
        ..
    }) = bound("outputs")
    else {
        return;
    };
    let t = T;
    let tokens = vec![3u32, 11, 7];
    let lane = Lane {
        slot: 0,
        word,
        tokens: tokens.clone(),
        readout: Readout::Last,
        kv_less: true,
        inputs: vec![InputFeed {
            id: t_input,
            channel: channels[2],
        }],
        ..Lane::default()
    };
    let mut ticket = engine
        .submit(&FrameSubmission::of(Step {
            lanes: vec![lane],
            attachments: vec![Attachment {
                lane: 0,
                lanes: 1,
                instance,
                adapter_only: false,
            }],
            media: Vec::new(),
            voxels: Vec::new(),
        }))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");

    let take = |engine: &mut backend::Engine, dense: u32| {
        engine
            .take_channel(instance, dense)
            .expect("the channel reads")
            .expect("the channel holds a cell")
    };
    let row = &planes["embed"].1[(tokens[2] * WIDTH) as usize..][..WIDTH as usize];
    let want_id = (0..row.len())
        .max_by(|&a, &b| row[a].total_cmp(&row[b]))
        .expect("a row") as i32;
    let ids = take(&mut engine, 0);
    assert_eq!(ids, want_id.to_le_bytes(), "the i32 readout output");

    let half = (WIDTH / 2) as usize;
    let got: Vec<f32> = take(&mut engine, 1)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(got.len(), (LAYERS * WIDTH) as usize, "both layers' rows");
    for layer in 0..LAYERS as usize {
        let at = t * (layer + 1) as f32;
        for c in 0..WIDTH as usize {
            let angle = at * (-THETA.ln() * (c % half) as f32 / half as f32).exp();
            let want = if c < half { angle.cos() } else { angle.sin() };
            let have = got[layer * WIDTH as usize + c];
            assert!(
                (have - want).abs() <= 2e-2 * want.abs().max(1.0),
                "layer {layer} column {c}: device {have} against host {want}"
            );
        }
    }
    eprintln!("ids {ids:?} and both stacked layers read back");
    assert_eq!(
        engine.live().tallied(engine::live::Tally::Unrecorded),
        0,
        "a fire a guest reads built a pipeline the load did not"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Nothing is allocated after the load is ready (ruling (v)): a bind lays
/// the instance's tables and scratch in pool slots the runtime took, and
/// its passes stage and land through the ring the load laid; three fires
/// create no buffer. The negative control: one buffer made by hand counts.
#[test]
fn a_bind_and_its_fires_create_no_buffer() {
    let Some(Bound {
        mut engine,
        instance,
        channels,
        t_input,
        word,
        dir,
        ready,
        ..
    }) = bound("creations")
    else {
        return;
    };
    for token in [3u32, 11, 7] {
        let lane = Lane {
            slot: 0,
            word,
            tokens: vec![token, token, token],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed {
                id: t_input,
                channel: channels[2],
            }],
            ..Lane::default()
        };
        let mut ticket = engine
            .submit(&FrameSubmission::of(Step {
                lanes: vec![lane],
                attachments: vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        for dense in [0, 1] {
            engine
                .take_channel(instance, dense)
                .expect("the channel reads")
                .expect("the channel holds a cell");
        }
    }
    assert_eq!(
        creations(&mut engine),
        ready,
        "a bind and three fires created buffers after the load was ready"
    );
    let made = <backend::Buffer as engine_gpu::Buffer<backend::Api>>::zeroed(
        engine.shell_mut().expect("loaded").device(),
        256,
        engine_gpu::books::owners::TEST,
    )
    .expect("a buffer by hand");
    assert_eq!(creations(&mut engine), ready + 1, "the books see a creation");
    drop(made);
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A frame of as many steps as the table has rows, one frame in flight (the
/// frame settles inside `submit`), every step's rows read by the host: each
/// step's rows come back, the first ones among them.
#[test]
fn a_frame_of_many_steps_settles_every_step_one_frame_in_flight() {
    let Some(Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    }) = loaded("many-steps", &Rows)
    else {
        return;
    };
    let word = trace.facts.word(&Request::new(ROWS, false));
    let argmax = |values: &[f32]| {
        (0..values.len())
            .max_by(|&a, &b| values[a].total_cmp(&values[b]))
            .expect("a row")
    };
    // Each step reads another token last: its row is its own.
    let last: Vec<u32> = (0..VOCAB).collect();
    let steps: Vec<Step> = last
        .iter()
        .map(|&token| Step {
            lanes: vec![Lane {
                slot: 0,
                word,
                tokens: vec![3, 11, token],
                readout: Readout::Last,
                kv_less: true,
                ..Lane::default()
            }],
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
        })
        .collect();
    let mut ticket = engine
        .submit(&FrameSubmission { steps })
        .expect("a frame of many steps fires and settles");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    assert_eq!(ticket.steps.len(), last.len());
    for (step, &token) in ticket.steps.iter().zip(&last) {
        let row = &planes["embed"].1[(token * WIDTH) as usize..][..WIDTH as usize];
        let got = &step.readouts[0].values;
        assert_eq!(got.len(), WIDTH as usize, "token {token}'s step read a row");
        assert_eq!(
            argmax(got),
            argmax(row),
            "token {token}'s step read its own row: {got:?} against {row:?}"
        );
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A sampler of `n` candidates a draw (a Gumbel draw over `n` keys): its
/// value of `n` f32 is the widest of its scratch.
fn wide_draw(n: u32) -> TraceContainer {
    use eta_dsl::prelude::*;
    let leak = |channel: Channel| -> &'static Channel { Box::leak(Box::new(channel)) };
    let rng = leak(Channel::from([0x7ce1u32, 0]));
    let out = leak(Channel::new([1], dtype::i32).capacity(8));
    let mut b = Builder::new(VOCAB, 16);
    b.stage(Stage::Epilogue, move || {
        let r = rng.take();
        let token = reshape(reduce_argmax(gumbel(&r, [n])), [1]);
        rng.put(&r + iota(2));
        out.put(&token);
    });
    out.note_host_take();
    b.build().expect("the draw traces").container().clone()
}

/// Each lane's tokens over three fires of four lanes, an instance a lane,
/// each seeded apart, in a pool of `chunk`-byte chunks where `chunk` says;
/// and the run of chunks a piece of the sampler's state takes.
fn wide_draws(tag: &str, chunk: Option<u64>, n: u32) -> Option<(Vec<Vec<i32>>, u32)> {
    let Loaded {
        mut engine,
        caps,
        trace,
        dir,
        ..
    } = loaded_in(tag, &Rows, chunk)?;
    let bound = eta_ir::validate::bind(wide_draw(n), vec![Vec::new()], EngineCaps::default())
        .expect("the draw binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0x51de,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the draw registers");
    let chunks = &caps.caps.chunks;
    // The instances' tables and runs of slots from chunk `top - 40` up (a
    // chunk each where they fill one), their own cells in the top chunk,
    // and whole chunks below that, adjacent, for runs past a chunk.
    let top = chunks.segment_chunks;
    assert!(top > 41, "segment 0 holds the test's chunks: {top}");
    let mut slots = super::slots::Slots::of(chunks, top - 40, top - 1);
    slots.chunks = (top - 26..top - 2).collect();
    let extents = engine::program::BindExtents::default();
    let run = engine
        .guest_state(program, &extents)
        .expect("the draw's state")
        .chunks_a_piece();
    let lanes = 4u32;
    let mut instances = Vec::new();
    for lane in 0..lanes {
        let channels = vec![ChannelId(100 + u64::from(lane) * 2), ChannelId(101 + u64::from(lane) * 2)];
        let binding = slots.binding(
            &engine,
            InstanceBinding {
                program,
                channels: channels.clone(),
                seeds: vec![engine::channel::ChannelSeed {
                    channel: 0,
                    bytes: [0x7ce1u32 + lane, 0].iter().flat_map(|w| w.to_le_bytes()).collect(),
                }],
                geometry: GeometryClass::Host,
                extents,
                ..Default::default()
            },
            &[],
        );
        instances.push(engine.bind_instance(&binding).expect("the draw binds").id);
    }
    let word = trace.facts.word(&Request::new(1, false));
    let mut tokens = vec![Vec::new(); lanes as usize];
    for token in [3u32, 11, 7] {
        let step = Step {
            lanes: (0..lanes)
                .map(|slot| Lane {
                    slot,
                    word,
                    tokens: vec![token],
                    readout: Readout::Last,
                    kv_less: true,
                    ..Lane::default()
                })
                .collect(),
            attachments: instances
                .iter()
                .enumerate()
                .map(|(lane, &instance)| Attachment {
                    lane: lane as u32,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                })
                .collect(),
            media: Vec::new(),
            voxels: Vec::new(),
        };
        let mut ticket = engine
            .submit(&FrameSubmission::of(step))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        for (lane, &instance) in instances.iter().enumerate() {
            let cell = engine
                .take_channel(instance, 1)
                .expect("the token reads")
                .expect("a token");
            tokens[lane].push(i32::from_le_bytes(cell[..4].try_into().expect("an i32")));
        }
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
    Some((tokens, run))
}

/// A sampler whose widest value passes the pool's chunk lies in pieces of
/// a run of adjacent chunks (`GuestState::run` past one): every lane draws
/// the token it draws where one chunk holds it.
#[test]
fn a_sampler_in_runs_of_chunks_draws_what_one_chunk_draws() {
    // 100,000 candidates: 400 KB of f32, and the 1.6 MB of whole-value
    // temporaries beside them, past a 1 MiB chunk.
    let n = 100_000;
    let Some((one, run_one)) = wide_draws("draw-wide", None, n) else {
        return;
    };
    let (runs, run) = wide_draws("draw-runs", Some(1 << 20), n).expect("the device bound once");
    eprintln!("tokens {one:?}, runs of {run} chunk(s) against {run_one}");
    assert_eq!(run_one, 1, "the default chunk holds the draw's value");
    assert!(run > 1, "a 1 MiB chunk holds no 1.6 MB temporary area: a run of {run}");
    assert_eq!(runs, one, "every lane's tokens");
    assert!(
        one.iter().flatten().any(|&t| t != one[0][0]),
        "the lanes draw apart, so a lane reading another's state would show"
    );
}

/// A top-p sampler over `n` candidates (a Gumbel draw of logits, then
/// `nucleus_sample` at p = 0.95): the shape of the standard top-p guest.
fn nucleus_draw(n: u32) -> TraceContainer {
    use eta_dsl::prelude::*;
    let leak = |channel: Channel| -> &'static Channel { Box::leak(Box::new(channel)) };
    let rng = leak(Channel::from([0x51deu32, 0]));
    let top_p = leak(Channel::from([0.95f32]));
    let out = leak(Channel::new([1], dtype::i32).capacity(8));
    let mut b = Builder::new(VOCAB, 16);
    b.stage(Stage::Epilogue, move || {
        let r = rng.take();
        let p = top_p.take();
        let logits = gumbel(&r, [n]);
        let token = reshape(nucleus_sample(&logits, reshape(&p, []), &r), [1]);
        rng.put(&r + iota(2));
        top_p.put(&p);
        out.put(&token);
    });
    out.note_host_take();
    b.build().expect("the sampler traces").container().clone()
}

/// The device whose cold compile of [`nucleus_draw`]'s stages the bound
/// below was measured on, and that compile at the sitting (863185d4e, the
/// private heap's single scratch binding) on each backend, in ms; the bound
/// is a quarter past it. Run with the driver's shader cache off
/// (`__GL_SHADER_DISK_CACHE=0`) so the compile is cold: measured so on
/// 2026-11-17, this test's registration and first bind at 863185d4e.
const COLD_DEVICE: &str = "NVIDIA RTX PRO 4500 Blackwell";
const COLD_AT_BASE_MS: &[(&str, u128)] = &[("vulkan", 20_979), ("wgpu", 4_547)];

/// A top-p sampler's stages compile, at its registration and first bind,
/// within a quarter past what the sitting's compile of the same program
/// took: the guest's addressing adds no code a driver spends its compile
/// on (a stage binds the one segment its state lies in, and an access is
/// an index of it).
#[test]
fn a_top_p_sampler_compiles_cold_within_a_quarter_past_the_sitting() {
    let Some(Loaded {
        mut engine,
        caps,
        dir,
        ..
    }) = loaded("cold-compile", &Rows)
    else {
        return;
    };
    let device = backend::Context::bind(&backend::DeviceBoot::default(), backend::ALLOWANCE)
        .map(|context| context.chip().name)
        .unwrap_or_default();
    let started = std::time::Instant::now();
    let bound = eta_ir::validate::bind(nucleus_draw(151_936), vec![Vec::new()], EngineCaps::default())
        .expect("the sampler binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0x70_9b,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the sampler registers");
    let chunks = &caps.caps.chunks;
    let top = chunks.segment_chunks;
    let mut slots = super::slots::Slots::of(chunks, top - 40, top - 1);
    slots.chunks = (top - 26..top - 2).collect();
    let binding = slots.binding(
        &engine,
        InstanceBinding {
            program,
            channels: vec![ChannelId(300), ChannelId(301), ChannelId(302)],
            geometry: GeometryClass::Host,
            ..Default::default()
        },
        &[],
    );
    engine.bind_instance(&binding).expect("the sampler binds its instance");
    let took = started.elapsed().as_millis();
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("COLD_COMPILE {} {device}: {took} ms", backend::NAME);
    let base = COLD_AT_BASE_MS
        .iter()
        .find(|(name, _)| *name == backend::NAME)
        .map_or(0, |&(_, ms)| ms);
    if device == COLD_DEVICE && base > 0 {
        assert!(
            took * 4 <= base * 5,
            "the cold compile took {took} ms, past a quarter over the sitting's {base} ms"
        );
    }
}
