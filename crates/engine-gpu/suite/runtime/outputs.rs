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
            dtype: match decl.dtype {
                poem_ir::Dtype::I32 => EtaDtype::I32,
                _ => EtaDtype::F32,
            },
            layers: decl.layers,
            readout: decl.readout,
        });
    }
    decls
}

/// `program` registered as `program_hash` against the outputs `decls`.
pub(super) fn registration_of(
    decls: &[OutputDecl],
    program: TraceContainer,
    program_hash: u64,
) -> ProgramRegistration {
    let bound = eta_ir::validate::bind(program, vec![decls.to_vec()], EngineCaps::default())
        .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    ProgramRegistration {
        program_hash,
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    }
}

/// The epilogue registered against the outputs `loaded` declares.
pub(super) fn epilogue_registration(loaded: &engine::load::Loaded) -> ProgramRegistration {
    registration_of(&output_decls(loaded), epilogue(), 0x0e7)
}

pub(super) fn epilogue() -> TraceContainer {
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

/// The miniature loaded one frame in flight, an instance of the epilogue
/// bound, and `t` published.
pub(super) struct Bound {
    pub(super) engine: backend::Engine,
    pub(super) instance: engine::InstanceId,
    pub(super) channels: Vec<ChannelId>,
    pub(super) t_input: u32,
    /// The word of a pass of [`ROWS`] tokens.
    pub(super) word: u64,
    pub(super) planes: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    pub(super) dir: std::path::PathBuf,
    /// The outputs the load declares.
    pub(super) decls: Vec<OutputDecl>,
}

pub(super) const T: f32 = 0.375;

/// The tokens of a pass.
pub(super) const ROWS: u32 = 3;

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
    loaded_lanes(tag, model, 4)
}

/// [`loaded`], a fire carrying up to `lanes` lanes of [`ROWS`] tokens.
pub(super) fn loaded_lanes(tag: &str, model: &impl ForwardHybrid, lanes: u32) -> Option<Loaded> {
    loaded_bound(tag, model, lanes, None)
}

/// [`loaded_lanes`] on a device whose platform states `bound` on one GPU
/// submission (the boot's, `test-limits`, as Android's KGSL states 2 s):
/// every serving step held to it (`engine_gpu::submit`).
pub(super) fn loaded_bound(
    tag: &str,
    model: &impl ForwardHybrid,
    lanes: u32,
    bound: Option<std::time::Duration>,
) -> Option<Loaded> {
    let boot = crate::backend::DeviceBoot {
        submit_bound: bound,
        ..crate::gpu::device_boot()
    };
    loaded_boot(tag, model, lanes, boot)
}

/// What `engine`'s load left to measure behind serving (the dense settle's
/// ranges, a stale rate), run to its end as the runtime's idle lane runs
/// it: one slice a call, no other lane submitting. The slices it ran.
pub(super) fn measure_behind(engine: &mut impl ::engine::Engine) -> usize {
    let mut slices = 0;
    let behind = ::engine::BehindServing {
        others: &|| 0,
        arrived: &|| false,
        seats: None,
        alone: true,
    };
    while engine
        .measure_behind(&behind)
        .expect("measuring behind serving")
    {
        slices += 1;
    }
    slices
}

/// [`loaded_lanes`] on `boot`.
pub(super) fn loaded_boot(
    tag: &str,
    model: &impl ForwardHybrid,
    lanes: u32,
    boot: crate::backend::DeviceBoot,
) -> Option<Loaded> {
    loaded_flight(tag, model, lanes, boot, 1)
}

/// [`loaded_boot`] with `frames` frames in flight (a served load's two: the
/// next fired before the last lands).
pub(super) fn loaded_flight(
    tag: &str,
    model: &impl ForwardHybrid,
    lanes: u32,
    boot: crate::backend::DeviceBoot,
    frames: u8,
) -> Option<Loaded> {
    let tokens = (lanes * ROWS).next_power_of_two().max(16);
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
                max_lanes: lanes,
                max_tokens: tokens,
                buckets: vec![tokens],
                max_adapters: 0,
                page_size: 16,
                // The fire top is a rung a prompt of the context holds
                // (`engine::load::fire_top`): a context of the fire's tokens.
                max_context: tokens.max(64),
                context_declared: false,
                slots: lanes,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: crate::gpu::share(),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: frames,
            steps: Default::default(),
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
pub(super) fn bound(tag: &str) -> Option<Bound> {
    bound_with(tag, |_, _| {})
}

/// [`bound`], with `before` run on the engine and the registered program
/// before the instance binds.
fn bound_with(
    tag: &str,
    before: impl FnOnce(&mut backend::Engine, engine::program::ProgramId),
) -> Option<Bound> {
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

    let decls = output_decls(&loaded);
    let program = engine
        .register_program(&registration_of(&decls, epilogue(), 0x0e7))
        .expect("the program registers");
    before(&mut engine, program);
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
            scratch: engine::ScratchPlace::None,
            pair: engine::ScratchPlace::None,
            pair_generation: 0,
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
        decls,
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
                scratch: ::engine::ScratchPlace::None,
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
        engine.live().tallied(engine::live::Tally::BuiltInFire),
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

/// What the engine tallies of a serving fire (`engine::live::Tally`): a
/// guest program registered and bound after the load, and its fire, build
/// no pipeline, since the load built every guest step's and every kernel
/// its walks named, so `BuiltInFire` reads 0; where the backend records,
/// the first fire records and the second replays, and neither counts
/// `Unrecorded`; where it keeps no recording, each fire walks eagerly and
/// counts `Eager`. The negative controls: a kernel pipeline the load did not
/// build, built after it, is counted `BuiltInFire` by the next fire; a
/// guest step's pipeline a bind builds after the load, before any fire of
/// it, is counted `BuiltAtRegister` and not `BuiltInFire`.
#[test]
fn a_pipeline_built_after_the_load_is_counted() {
    use engine::live::Tally;
    use engine_gpu::serve::Serve;
    let Some(Bound {
        mut engine,
        instance,
        channels,
        t_input,
        word,
        dir,
        ..
    }) = bound("tally")
    else {
        return;
    };
    let live = engine.live();
    let built = |live: &engine::live::Live| live.tallied(Tally::BuiltInFire);
    let fire = |engine: &mut backend::Engine| {
        let lane = Lane {
            slot: 0,
            word,
            tokens: vec![3, 11, 7],
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
                    scratch: ::engine::ScratchPlace::None,
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
        // The next fire's timestep, where the ring has room for it (a
        // full ring still holds this one's).
        engine
            .publish_channel(instance, 2, &T.to_le_bytes())
            .expect("the timestep publishes");
    };

    fire(&mut engine);
    fire(&mut engine);
    assert_eq!(
        built(&live),
        0,
        "a guest program bound after the load and its fires built no pipeline"
    );
    assert_eq!(
        live.tallied(Tally::Unrecorded),
        0,
        "where the backend records, the second fire replays what the first recorded"
    );
    if !<backend::Api as Serve>::RECORDS {
        assert_eq!(
            live.tallied(Tally::Eager),
            2,
            "a backend that keeps no recording walks each fire eagerly, as it states"
        );
    }
    assert_eq!(live.tallied(Tally::Refused), 0);

    // A kernel no walk of the load named, built now, as a fire would.
    let shell = engine.shell_mut().expect("loaded");
    let before = shell.compiled();
    let unnamed = kernels_slang::CENSUS.iter().find(|&&(key, _, _, unit)| {
        <backend::Api as Serve>::build(shell.device(), shell.pipelines(), unit, key).is_ok()
            && shell.compiled() > before
    });
    let Some(&(key, ..)) = unnamed else {
        panic!("the load built every kernel of the census, so none is left to build after it");
    };
    fire(&mut engine);
    assert_eq!(
        built(&live),
        1,
        "`{key}`, built after the load, is counted by the fire after it"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);

    // A guest step's pipeline the load built, taken out before the bind:
    // the bind that dispatches it builds it while serving, and that counts.
    let Some(Bound { engine, dir, .. }) = bound_with("tally-guest", |engine, program| {
        use engine_gpu::program::Programs as _;
        let package = engine
            .programs
            .package(program)
            .cloned()
            .expect("the program is registered");
        let first = package
            .plans
            .iter()
            .flat_map(|plan| {
                eta_compiler::codegen::wgsl::emit_launch_steps("main", plan)
                    .expect("the stage emits")
                    .steps
            })
            .next()
            .expect("a stage of the program takes a step")
            .call;
        engine.programs.forget_step(first);
    }) else {
        return;
    };
    assert_eq!(
        engine.live().tallied(Tally::BuiltAtRegister),
        1,
        "a bind that built a guest step's pipeline after the load counts it at its registration"
    );
    assert_eq!(
        engine.live().tallied(Tally::BuiltInFire),
        0,
        "a guest step's pipeline built at its bind, before any fire of it, is no fire's build"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
