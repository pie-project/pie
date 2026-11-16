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
}

const T: f32 = 0.375;

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
    let boot = backend::DeviceBoot::default();
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
