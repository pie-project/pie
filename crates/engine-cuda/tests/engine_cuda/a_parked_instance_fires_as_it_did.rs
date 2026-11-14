#![cfg(feature = "device")]

//! An instance's `Instance` slots park on the host tier and come back to
//! other device slots as the runtime moves them under pressure
//! (`new-memory/STATE-ROLES.md` 2: a non-replayable slot parks, never
//! drops). Its item is a run of several slots, addressed as one span, so it
//! parks as a run and comes back as a run; after `rebind_slots` its ring
//! reads the cells it read before.

use engine::Engine;
use engine::chunk::{SlotCopy, SlotRole};
use engine::fire::{Attachment, FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, ProgramRegistration, Rebind};
use engine::transfer::MemoryDomain;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use eta_ir::validate::bind;
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, ops};
use poem_ir::InputAxis;

use crate::common_two_axis::{Weights, contract_for};

const C: u32 = 4;
const P: [u32; 3] = [1, 2, 2];
/// `[cells axis 0 | cells axis 3]` of each text row.
const WIDTH: u32 = 6 + 7;
/// Rows of the one lane: a ring of three cells of `ROWS x WIDTH` f32 is
/// past one 4 KiB `Instance` slot, so the instance's item is a run.
const ROWS: u32 = 32;

struct Cells;

impl ForwardHybrid for Cells {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        inputs
            .recorder()
            .declare_switch("stream", &["text", "image"]);
        let img = inputs.on(!poem::fact::switch_is("stream", "text"));
        let grid = img.grid();
        let latents = img.input("latents", Dtype::Bf16, InputAxis::Voxel, C, None);
        let patched = ops::conv::patchify(&latents, &grid, P, &img.token_grid(P));
        let tgrid = inputs.token_grid(P);
        let positions = inputs.positions();
        let first = ops::conv::cells(&tgrid, &positions, 0);
        let fourth = ops::conv::cells(&tgrid, &positions, 3);
        let out = ops::gather::concat_cols(&[&first, &fourth]);
        poem::output("velocity", &out, None);
        let back = ops::conv::unpatchify(&patched, &img.token_grid(P), P, &grid);
        let (pixels, pixel_grid) = ops::conv::upsample_nearest(&back, &grid, [1, 1, 1], false);
        poem::output("pixels", &pixels, Some(&pixel_grid));
    }
}

fn epilogue() -> TraceContainer {
    let plane = Shape::matrix(ROWS, WIDTH);
    TraceContainer {
        names: vec!["velocity".to_string()],
        channels: vec![ChannelDecl {
            shape: plane,
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: plane,
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn a_parked_instance_fires_as_it_did() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid(crate::common_two_axis::NAME, &Cells, Platform::Cuda);
    let facts = trace.facts.clone();
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 1).write(dir.path());
    let mut engine = engine_cuda::open(engine_cuda::DeviceBoot::default(), contract_for)
        .expect("the engine opens");
    let loaded = engine
        .load(LoadRequest {
            trace,
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 64,
                buckets: vec![32, 64],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: Some(128),
                max_clips: Some(4),
            },
            residency: Residency {
                device_kv_budget: Some(crate::POOL),
                host_kv_budget: Some(64 << 20),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("the plan loads");
    let shapes = &loaded.caps.chunks.shapes;
    let instance_shape = shapes
        .iter()
        .find(|shape| shape.role == SlotRole::Instance)
        .expect("the load states an Instance shape");
    let mut guest = crate::guest::Guest::on(&mut engine, &loaded.caps.chunks);

    let velocity = OutputDecl {
        name: "velocity".into(),
        axis: Axis::Voxel,
        width: WIDTH,
        dtype: EtaDtype::F32,
        layers: 0,
        readout: false,
    };
    let bound =
        bind(epilogue(), vec![vec![velocity]], EngineCaps::default()).expect("the epilogue binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0x9a4c,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let program_facts = engine.program_facts(program).expect("the program's facts");
    assert!(
        program_facts.instance.contiguous && program_facts.instance.slots > 1,
        "the instance's item is a run of slots: {:?}",
        program_facts.instance
    );
    let (slots, rings) = guest.slots(&program_facts, |_| false);
    let instance = engine
        .bind_instance(&InstanceBinding {
            program,
            channels: Vec::new(),
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            device_resolved: Default::default(),
            extents: engine::program::BindExtents {
                row_count: ROWS,
                token_count: ROWS,
                sampled_rows: ROWS,
                query_len: ROWS,
                ..engine::program::BindExtents::default()
            },
            instance_slots: slots.clone(),
            ring_slots: rings.clone(),
        })
        .expect("the instance binds")
        .id;

    let word = facts.word(&Request::new(ROWS, false).switched("stream", "text"));
    // Each fire on a slot of its own: its rows start at position 0, so the
    // two fires land the same cells.
    let fire = |engine: &mut dyn Engine, slot: u32| -> Vec<u8> {
        let lane = Lane {
            slot,
            word,
            tokens: vec![0; ROWS as usize],
            readout: Readout::Rows((0..ROWS).collect()),
            group: Some(0),
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
        engine
            .take_channel(instance, 0)
            .expect("the channel reads")
            .expect("the channel holds a cell")
    };
    let before = fire(&mut engine, 0);

    // Park the run on a run of the host tier, then unpark it to a run of
    // other device slots: past the bound ones, in the same chunk.
    let device = MemoryDomain::CudaDevice(0);
    let count = slots.len() as u32;
    let host: Vec<u32> = (1..=count).collect();
    let moved: Vec<u32> = slots.iter().map(|&slot| slot + count).collect();
    assert_eq!(
        moved[0] / instance_shape.slots_per_chunk,
        moved[moved.len() - 1] / instance_shape.slots_per_chunk,
        "the run comes back inside one chunk"
    );
    engine
        .copy_slots(&SlotCopy {
            shape: instance_shape.id,
            src: device,
            dst: MemoryDomain::HostPinned,
            slots: slots.iter().copied().zip(host.iter().copied()).collect(),
            runs: Vec::new(),
        })
        .expect("the instance's slots park on the host tier");
    engine
        .copy_slots(&SlotCopy {
            shape: instance_shape.id,
            src: MemoryDomain::HostPinned,
            dst: device,
            slots: host.iter().copied().zip(moved.iter().copied()).collect(),
            runs: Vec::new(),
        })
        .expect("the instance's slots unpark to other device slots");
    engine
        .rebind_slots(Rebind::Instance {
            id: instance,
            instance: moved,
            rings,
        })
        .expect("the instance rebinds to where its slots lie now");

    let after = fire(&mut engine, 1);
    assert_eq!(
        after, before,
        "the unparked instance's ring reads the cell the parked one read"
    );
}
