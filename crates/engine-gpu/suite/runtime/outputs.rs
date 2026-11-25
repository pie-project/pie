//! A guest reads a lane output stacked over two layers and an i32 readout
//! output by name through the fire, against host references. The miniature,
//! its epilogue and the helpers here are the other runtime tests' too.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use checkpoint::file::emit::{self, Object};
use checkpoint::serving::Stamp;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{BindExtents, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::{ChannelId, Engine};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, ForwardHybrid, Input, Request, StateRows, Trace, Weight, ops};
use poem_ir::InputAxis;

use super::slots::Slots;
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
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let mut planes = BTreeMap::new();
    let values = (0..VOCAB * WIDTH).map(|_| bf(unit())).collect();
    planes.insert("embed".to_string(), (vec![u64::from(VOCAB), u64::from(WIDTH)], values));
    planes
}

pub(super) fn write(path: &Path, planes: &BTreeMap<String, (Vec<u64>, Vec<f32>)>) {
    let bytes: Vec<(String, Vec<u64>, Vec<u8>)> = planes
        .iter()
        .map(|(name, (shape, values))| {
            let raw = values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
            (name.clone(), shape.clone(), raw)
        })
        .collect();
    let objects: Vec<Object<'_>> = bytes
        .iter()
        .map(|(name, shape, raw)| Object::leaf(name, shape.clone(), ztensor::Leaf::BF16, raw))
        .collect();
    let stamp = Stamp::of(backend::NAME, NAME);
    emit::write(path, &stamp, &BTreeMap::new(), 4096, &objects, |o, p, _| panic!("{o}/{p} is not streamed here"))
        .expect("the artifact writes");
}

pub(super) fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, backend::PLATFORM)
        .map_err(|why| why.to_string())
}

/// The outputs `loaded` declares, as a guest binds them.
pub(super) fn output_decls(loaded: &engine::load::Loaded) -> Vec<OutputDecl> {
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
            dtype: if decl.dtype == poem_ir::Dtype::I32 { EtaDtype::I32 } else { EtaDtype::F32 },
            layers: decl.layers,
            readout: decl.readout,
        });
    }
    decls
}

/// `program` registered as `program_hash` against the outputs `decls`.
pub(super) fn registration_of(decls: &[OutputDecl], program: TraceContainer, program_hash: u64) -> ProgramRegistration {
    let bound = eta_ir::validate::bind(program, vec![decls.to_vec()], EngineCaps::default())
        .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    ProgramRegistration {
        program_hash,
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    }
}

pub(super) fn epilogue() -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded: false,
    };
    let output = |index: u16, shape: Shape, dtype: EtaDtype| Op::Output { index, lane: 0, shape, dtype };
    TraceContainer {
        names: vec!["ids".to_string(), "stack".to_string()],
        channels: vec![
            decl(Shape::matrix(1, 1), EtaDtype::I32, HostRole::Reader),
            decl(Shape::matrix(LAYERS, WIDTH), EtaDtype::F32, HostRole::Reader),
            decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                output(0, Shape::matrix(1, 1), EtaDtype::I32),
                Op::ChanPut { chan: 0, value: 0 },
                output(1, Shape::matrix(LAYERS, WIDTH), EtaDtype::F32),
                // Negated twice on the device, so the pass dispatches steps
                // of the closed set and reads back what it read out.
                Op::Neg(1),
                Op::Neg(2),
                Op::ChanPut { chan: 1, value: 3 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

pub(super) const T: f32 = 0.375;

/// The tokens of a pass.
pub(super) const ROWS: u32 = 3;

pub(super) const EXTENTS: BindExtents =
    BindExtents { row_count: 1, token_count: ROWS, sampled_rows: 1, query_len: ROWS, key_len: 1 };

/// A host channel of `shape` cells of `dtype`, `capacity` deep.
pub(super) fn channel(id: ChannelId, shape: Vec<u32>, dtype: EtaDtype, host_role: HostRole, capacity: u32) -> ChannelRegistration {
    ChannelRegistration {
        id,
        shape,
        dtype: ChanDType::Concrete(dtype),
        host_role,
        seeded: false,
        extern_dir: None,
        capacity,
        extern_name: Vec::new(),
        slots: Vec::new(),
    }
}

/// A model loaded one frame in flight over the miniature's table, a fire
/// carrying up to four lanes of [`ROWS`] tokens.
pub(super) struct Loaded {
    pub(super) engine: backend::Engine,
    pub(super) caps: engine::load::Loaded,
    pub(super) trace: Trace,
    pub(super) planes: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    pub(super) dir: std::path::PathBuf,
}

/// `None` without a device. `tag` names the scratch directory.
pub(super) fn loaded(tag: &str, model: &impl ForwardHybrid) -> Option<Loaded> {
    loaded_with(tag, model, None, crate::gpu::device_boot())
}

/// [`loaded`] on `boot`, with a host tier of `host` bytes where given.
pub(super) fn loaded_with(
    tag: &str,
    model: &impl ForwardHybrid,
    host: Option<u64>,
    boot: backend::DeviceBoot,
) -> Option<Loaded> {
    loaded_flight(tag, model, LANES, host, boot, 1)
}

const LANES: u32 = 4;

/// [`loaded_with`] for `lanes` lanes, `frames` frames in flight.
pub(super) fn loaded_flight(
    tag: &str,
    model: &impl ForwardHybrid,
    lanes: u32,
    host: Option<u64>,
    boot: backend::DeviceBoot,
    frames: u8,
) -> Option<Loaded> {
    let tokens = (lanes * ROWS).next_power_of_two().max(16);
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("pie-{}-{tag}-{}", backend::NAME, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    let planes = planes();
    write(&path, &planes);
    let trace = poem::trace_hybrid(NAME, model, backend::PLATFORM);
    let mut engine = backend::Engine::new(crate::gpu::fires_stated(boot), contract_for);
    let caps = engine
        .load(LoadRequest {
            trace: trace.clone(),
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: lanes,
                max_tokens: tokens,
                buckets: vec![tokens],
                page_size: 16,
                // The fire top is a rung a prompt of the context holds
                // (`engine::load::fire_top`): a context of the fire's tokens.
                max_context: tokens.max(64),
                slots: lanes,
                ..Budgets::default()
            },
            residency: Residency { device: crate::gpu::share(), host_kv_budget: host, ..Residency::default() },
            ordinal: 0,
            frames_in_flight: frames,
            steps: Default::default(),
        })
        .expect("the miniature loads");
    Some(Loaded { engine, caps, trace, planes, dir })
}

/// The miniature loaded, the epilogue registered, and the slots its guests
/// take.
pub(super) struct Bound {
    pub(super) loaded: Loaded,
    pub(super) program: ProgramId,
    pub(super) slots: Slots,
    pub(super) t_input: u32,
    pub(super) word: u64,
}

/// `None` without a device.
pub(super) fn bound(tag: &str) -> Option<Bound> {
    let mut loaded = loaded(tag, &Outputs)?;
    let decls = output_decls(&loaded.caps);
    let program = engine::Programs::register(&mut loaded.engine, &registration_of(&decls, epilogue(), 0x0e7))
        .expect("the program registers");
    let chunks = &loaded.caps.caps.chunks;
    let slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let t_input = loaded.trace.inputs.iter().position(|input| input.name == "t").expect("the plan reads `t`") as u32;
    let word = loaded.trace.facts.word(&Request::new(ROWS, false));
    Some(Bound { loaded, program, slots, t_input, word })
}

impl Bound {
    /// Binds an instance of the epilogue over three channels of its own
    /// (`ids` and `stack` the host reads, `t` it writes) from id `first`,
    /// and publishes `t`.
    pub(super) fn bind(&mut self, first: u64) -> (InstanceId, Vec<ChannelId>, InstanceBinding) {
        let engine = &mut self.loaded.engine;
        let channels: Vec<ChannelId> = [
            (vec![1, 1], EtaDtype::I32, HostRole::Reader),
            (vec![LAYERS, WIDTH], EtaDtype::F32, HostRole::Reader),
            (vec![1, 1], EtaDtype::F32, HostRole::Writer),
        ]
        .into_iter()
        .enumerate()
        .map(|(at, (shape, dtype, host_role))| {
            let id = ChannelId(first + at as u64);
            self.slots.register(engine, channel(id, shape, dtype, host_role, 2));
            id
        })
        .collect();
        let binding = InstanceBinding {
            program: self.program,
            channels: channels.clone(),
            geometry: GeometryClass::Host,
            extents: EXTENTS,
            ..InstanceBinding::default()
        };
        let binding = self.slots.binding(engine, binding, &channels);
        let instance = engine::Programs::bind(engine, &binding).expect("the instance binds").id;
        assert!(
            engine::Channels::publish(engine, instance, 2, &T.to_le_bytes()).expect("the timestep publishes"),
            "the ring had room"
        );
        (instance, channels, binding)
    }

    /// One pass of `instance` reading `token` last: its `ids` and `stack`
    /// cells, `t` published again for the next.
    pub(super) fn pass(&mut self, instance: InstanceId, channels: &[ChannelId], token: u32) -> (i32, Vec<f32>) {
        let engine = &mut self.loaded.engine;
        let lane = Lane {
            slot: 0,
            word: self.word,
            tokens: vec![3, 11, token],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed { id: self.t_input, channel: channels[2] }],
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
                    scratch: ::engine::ScratchPlace::None,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
                shape: None,
            }))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        let mut take = |dense: u32| {
            engine::Channels::take(engine, instance, dense)
                .expect("the channel reads")
                .expect("the channel holds a cell")
        };
        let ids = i32::from_le_bytes(take(0).try_into().expect("one i32"));
        let stack = take(1).as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        engine::Channels::publish(engine, instance, 2, &T.to_le_bytes()).expect("the timestep publishes");
        (ids, stack)
    }

    /// The argmax of embedding row `token`.
    pub(super) fn argmax(&self, token: u32) -> i32 {
        let row = &self.loaded.planes["embed"].1[(token * WIDTH) as usize..][..WIDTH as usize];
        (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).expect("a row") as i32
    }
}

#[test]
fn a_guest_reads_lane_layer_and_id_outputs_by_name() {
    let Some(mut bound) = bound("outputs") else {
        return;
    };
    let (instance, channels, _) = bound.bind(1);
    let (ids, got) = bound.pass(instance, &channels, 7);
    assert_eq!(ids, bound.argmax(7), "the i32 readout output");
    assert_eq!(got.len(), (LAYERS * WIDTH) as usize, "both layers' rows");
    let half = (WIDTH / 2) as usize;
    for layer in 0..LAYERS as usize {
        let at = T * (layer + 1) as f32;
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
    let Bound { loaded: Loaded { engine, dir, .. }, .. } = bound;
    assert_eq!(
        engine.live().tallied(engine::live::Tally::BuiltInFire),
        0,
        "a fire a guest reads built a pipeline the load did not"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
