//! **A DEVICE-ONLY RING TWO PASSES SHARE**, which is design §5's
//! *"draft→verify chaining is free: a device-only private ring shared by ≤8
//! attachments, ordered by the pipeline FIFO"*.
//!
//! # What was broken
//!
//! A channel declared [`HostRole::None`] had no endpoint at all: `Rings`
//! allocated its cells inside whichever `Session` bound it and its counters
//! lived only in that session's prediction. That is right for a ring ONE pass
//! owns and silently wrong for the shape the design names by hand — the
//! prefill epilogue's put landed in one session's slab and the decode's take
//! read the other's, forever empty. Every `text-completion-bench` run died on
//! it, at the first decode, with
//!
//! ```text
//! instance N's epilogue blocked on channel 0 AFTER the gate admitted it,
//! so something advanced its cursors between the two
//! ```
//!
//! — a fault whose sentence was true and whose diagnosis was not: nothing had
//! advanced anything, and the gate and the fire agreed with each other
//! perfectly. They were both reading a ring nobody had written.
//!
//! # The three claims
//!
//! 1. **THE VALUE CROSSES.** One instance's epilogue puts a cell the host
//!    handed it; a second instance's epilogue, in a later fire, takes that
//!    cell and publishes it where the host can read it. The number that comes
//!    out is the number that went in.
//! 2. **THE DENSE SLOT IS NOT THE RING.** The two programs declare the shared
//!    channel at DIFFERENT dense slots — 1 in the putter, 0 in the taker —
//!    which is the arrangement the old design could not have served even with
//!    a shared registry, because the control kernels index the registry by
//!    dense slot. The ring is addressed through its endpoint's own pointers
//!    instead, so the slot numbers never have to agree.
//! 3. **EIGHT SEATS, AND THE NINTH IS A REFUSAL BY NAME.** The bound is the
//!    design's; past it there is no ordering argument, so there is no ring.
//!
//! Claim 1 is the one that would have caught the bug: before the fix the take
//! finds an empty ring and the fire is refused, so this gate fails at the
//! second fire rather than on a comparison.
//!
//! ```text
//! cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_shared_device_ring -- --nocapture
//! ```

#![cfg(feature = "_engine-cuda")]

mod common;

use engine::{
    Attachment, BindExtents, Boundary, Budgets, ChannelRegistration, FrameSubmission,
    InstanceBinding, KvDelta, Lane, Readout, RsReset, RsVerb, Step,
};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype, Shape};
use model_ir::Platform;
use runtime::engine::backend::open;

/// The catalog row this gate serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The cell the putter is handed and the taker must produce. Arbitrary, and
/// arbitrary is the point: it is not derivable from the model, so a ring that
/// produced it produced it because the bytes crossed.
const CROSSES: i32 = 0x0BAD_F00D_u32 as i32;

/// Channel ids, in the engine's own numbering. The gate mints them because it
/// drives `register_channel` directly, which is the runtime's job in a
/// deployment.
const IN_A: u64 = 101;
const SHARED: u64 = 102;
const OUT_B: u64 = 103;
/// A second shared ring, for the eight-seat claim alone.
const CROWDED: u64 = 104;

/// **THE PUTTER**: take the host's cell, put it into the shared ring.
///
/// The shared channel is dense slot **1** here and dense slot **0** in the
/// taker, which is claim 2: one ring, two dense numberings, and nothing
/// downstream is allowed to care.
fn putter() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![
            // 0: the host's cell. `Writer` + a take is what makes this pass
            // wait for the host to publish.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 1,
                host_role: HostRole::Writer,
                seeded: false,
            },
            // 1: THE SHARED RING. No host end at either side — the cell never
            // leaves the device between here and the taker.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanPut { chan: 1, value: 0 },
            ],
        }],
    }
}

/// **THE TAKER**: take the shared ring's cell, publish it to the host.
fn taker() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![
            // 0: THE SHARED RING, at a different dense slot than the putter's.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: false,
            },
            // 1: where the host reads what crossed.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanPut { chan: 1, value: 0 },
            ],
        }],
    }
}

/// **THE CROWD**: one loop-carried shared ring and nothing else, so that eight
/// binds need eight instances and no other channel.
fn loop_carried() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(Dtype::I32),
            capacity: 1,
            host_role: HostRole::None,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
    }
}

/// One container, all the way to what `Engine::register_program` takes.
fn registration(
    container: TraceContainer,
    profile: &ModelProfile,
) -> engine::ProgramRegistration {
    let bound = eta_ir::validate::bind(container, profile.clone())
        .unwrap_or_else(|why| panic!("the gate's program does not bind: {why:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = eta_compiler::codegen::program::Backend::Cuda;
    let emitted = eta_compiler::codegen::program::emit_program(backend, &stages, &bound);
    for kernel in &emitted {
        assert!(kernel.error.is_empty(), "did not emit: {}", kernel.error);
    }
    engine::ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted,
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

/// One i32 cell, as the wire carries it.
fn cell(value: i32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// A device-only channel's registration, which is the one shape this gate is
/// about.
fn device_only(id: u64) -> ChannelRegistration {
    ChannelRegistration {
        id,
        shape: vec![1],
        dtype: ChanDType::Concrete(Dtype::I32),
        host_role: HostRole::None,
        seeded: false,
        extern_dir: None,
        capacity: 1,
        extern_name: Vec::new(),
    }
}

fn host_channel(id: u64, role: HostRole) -> ChannelRegistration {
    ChannelRegistration {
        host_role: role,
        ..device_only(id)
    }
}

/// One prefill lane, which is only ever the thing the epilogue hangs off.
fn fire(tokens: &[u32], attachment: Attachment) -> FrameSubmission {
    let classify =
        runtime::engine::load::classify(SKU).expect("this build ships the gate's SKU");
    FrameSubmission::of(Step {
        lanes: vec![Lane {
            slot: 0,
            word: classify(&models::Request::new(tokens.len() as u32, false)),
            tokens: tokens.to_vec(),
            positions: Vec::new(),
            kv: KvDelta {
                held: 0,
                pages: vec![0, 1],
                ..KvDelta::default()
            },
            mask: None,
            adapter: None,
            drafts: false,
            captures_scores: false,
            rs: RsVerb::Fold,
            rs_reset: RsReset::Inferred,
            channels: Vec::new(),
            readout: Readout::Last,
        }],
        attachments: vec![attachment],
        media: Vec::new(),
    })
}

#[test]
fn a_device_only_ring_carries_a_cell_from_one_instance_to_another() {
    use engine::Engine;

    if !engine_cuda::device::present() {
        eprintln!("skipping the shared-ring gate: no CUDA device on this machine");
        return;
    }
    let Ok(checkpoint) = common::resolve_qwen35_snapshot() else {
        eprintln!("skipping the shared-ring gate: no Qwen3.5-0.8B snapshot in the HF cache");
        return;
    };
    let checkpoint = std::path::PathBuf::from(checkpoint);
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let mut engine = open::cuda(b"[model]\ndevice = \"cuda:0\"\n").expect("the cuda seam opens");
    let budgets = Budgets {
        max_lanes: 4,
        max_tokens: 256,
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: 16,
        max_context: 512,
        slots: 4,
        max_patches: None,
        max_images: None,
    };
    let request = runtime::engine::load::request(
        &checkpoint,
        Platform::Cuda,
        budgets,
        engine::Residency::uncapped(),
        0,
        1,
    )
    .expect("the checkpoint identifies and its SKU traces");
    assert_eq!(request.trace.name, SKU);
    let loaded = engine.load(request).expect("the checkpoint lands");
    let profile = loaded.caps.profile.clone();

    // ── THE CHANNELS. The shared one is registered ONCE and named by BOTH
    //    binds below; that single registration is where its ring is cut.
    for plan in [
        host_channel(IN_A, HostRole::Writer),
        device_only(SHARED),
        host_channel(OUT_B, HostRole::Reader),
    ] {
        engine
            .register_channel(&plan)
            .unwrap_or_else(|why| panic!("registering channel {}: {why}", plan.id));
    }

    let put_program = engine
        .register_program(&registration(putter(), &profile))
        .expect("the putter compiles");
    let take_program = engine
        .register_program(&registration(taker(), &profile))
        .expect("the taker compiles");

    // **THE DENSE SLOTS DISAGREE ON PURPOSE** (claim 2): the shared ring is
    // the putter's channel 1 and the taker's channel 0.
    let putting = engine
        .bind_instance(&InstanceBinding {
            program: put_program,
            channels: vec![IN_A, SHARED],
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            extents: BindExtents::default(),
        })
        .expect("the putter binds");
    let taking = engine
        .bind_instance(&InstanceBinding {
            program: take_program,
            channels: vec![SHARED, OUT_B],
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            extents: BindExtents::default(),
        })
        .expect("the taker binds onto the same ring");
    assert_ne!(putting.id, taking.id, "two instances, one ring");

    let tokens = tokenizer.encode("The capital of France is");

    // ── FIRE 1. The host publishes a cell; the putter's epilogue moves it
    //    into the shared ring and nothing else.
    assert!(
        engine
            .publish_channel(putting.id, 0, &cell(CROSSES))
            .expect("publishing the putter's input"),
        "the putter's input ring has room on its first fire"
    );
    let mut ticket = engine
        .submit(&fire(
            &tokens,
            Attachment {
                lane: 0,
                instance: putting.id,
                at: Boundary::Epilogue,
            },
        ))
        .expect("the putting fire runs");
    engine.settle_frame(&mut ticket).expect("and settles");

    // ── FIRE 2. The taker's epilogue finds the ring FULL — which it can only
    //    be if the two instances addressed one ring — takes the cell, and
    //    publishes it where the host reads.
    //
    //    Before the fix this line is where the gate fails: the taker's own
    //    copy of the ring is empty, `NeedsFull` is unmet, and the fire is
    //    refused with "blocked AFTER the gate admitted it".
    let mut ticket = engine
        .submit(&fire(
            &tokens,
            Attachment {
                lane: 0,
                instance: taking.id,
                at: Boundary::Epilogue,
            },
        ))
        .unwrap_or_else(|why| {
            panic!(
                "the taking fire was refused, so the cell the putter wrote never reached \
                 the ring the taker reads: {why}"
            )
        });
    engine.settle_frame(&mut ticket).expect("and settles");

    let published = engine
        .take_channel(taking.id, 1)
        .expect("taking the taker's output")
        .expect("the taker published, so its epilogue ran and committed");
    let crossed = i32::from_le_bytes([published[0], published[1], published[2], published[3]]);
    assert_eq!(
        crossed, CROSSES,
        "the cell that came out of the shared ring is not the cell that went in"
    );
    eprintln!("shared ring carried {crossed:#x} from instance {} to {}", putting.id, taking.id);

    // ── CLAIM 3: eight seats, and the ninth refused by name.
    engine
        .register_channel(&device_only(CROWDED))
        .expect("the crowded ring registers");
    let crowd_program = engine
        .register_program(&registration(loop_carried(), &profile))
        .expect("the loop-carried program compiles");
    let seat = |engine: &mut Box<dyn Engine>| {
        engine.bind_instance(&InstanceBinding {
            program: crowd_program,
            channels: vec![CROWDED],
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            extents: BindExtents::default(),
        })
    };
    for at in 1..=8 {
        seat(&mut engine).unwrap_or_else(|why| panic!("attachment {at} of 8 was refused: {why}"));
    }
    let ninth = seat(&mut engine);
    let said = match &ninth {
        Err(why) => why.to_string(),
        Ok(bound) => panic!("a ninth attachment bound as instance {}", bound.id),
    };
    assert!(
        said.contains('8'),
        "the refusal must name the bound a caller has to size against: {said}"
    );
    eprintln!("ninth attachment refused: {said}");
}
