//! Registration is a guest program's ready: `register_program` has every
//! pipeline its stages dispatch through built, off the call, and tells the
//! registration when they are; nothing of the program is built at a bind,
//! and a second registration of the program starts no build.

use std::sync::mpsc::{Receiver, channel};

use engine::Engine;
use engine::ProgramBuilt;
use engine::live::Tally;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::ProgramRegistration;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem_ir::InputAxis;

use super::outputs::{NAME, Outputs, contract_for, planes, write};
use crate::backend;

/// One stage with a step to dispatch: the pass's id, doubled on the
/// device, where the host reads it.
fn doubled() -> TraceContainer {
    TraceContainer {
        names: vec!["ids".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::matrix(1, 1),
            dtype: ChanDType::Concrete(EtaDtype::I32),
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
                    shape: Shape::matrix(1, 1),
                    dtype: EtaDtype::I32,
                },
                Op::Add(0, 0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// A registration's `built`, and where what it is told arrives: an `Err`
/// to receive once it is dropped untold.
fn told() -> (ProgramBuilt, Receiver<engine::Result<()>>) {
    let (tell, told) = channel();
    let built: ProgramBuilt = Box::new(move |built| {
        let _ = tell.send(built);
    });
    (built, told)
}

#[test]
fn a_registration_is_told_once_its_program_is_built_before_any_bind() {
    let boot = backend::DeviceBoot::default();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-register-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    write(&path, &planes());

    let trace = poem::trace_hybrid(NAME, &Outputs, backend::PLATFORM);
    let mut engine = backend::Engine::new(boot, contract_for);

    // Before the load there is no device to build on: the call refuses and
    // tells nobody.
    let (built, untold) = told();
    let early = engine
        .register_program(&ProgramRegistration::default(), built)
        .expect_err("a registration before the load is refused");
    assert!(early.to_string().contains("before load"), "{early}");
    assert!(untold.recv().is_err(), "dropped untold");

    let loaded = engine
        .load(LoadRequest {
            trace,
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
    let bound = eta_ir::validate::bind(doubled(), vec![decls], EngineCaps::default())
        .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let registration = ProgramRegistration {
        program_hash: 0x0b17,
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    };

    // What the call does on the caller's thread, measured apart: every
    // stage lowered to what the backend's pipelines take.
    let lowering = std::time::Instant::now();
    let lowered = engine_gpu::guest::Compiled::<backend::Api>::of(&registration.launch)
        .expect("the program lowers");
    let lowering = lowering.elapsed();
    let steps: Vec<usize> = (lowered.all().iter())
        .map(|stage| stage.entries.len())
        .collect();
    assert!(
        steps.iter().any(|&steps| steps > 0),
        "the program dispatches a step: {steps:?}"
    );

    let started = std::time::Instant::now();
    let (built, first) = told();
    let program = engine
        .register_program(&registration, built)
        .expect("the program registers");
    let called = started.elapsed();
    first
        .recv()
        .expect("a registration that returned an id is told")
        .expect("the program builds");
    eprintln!(
        "{}: lowering alone {lowering:?}; register_program returned in {called:?}, told built \
         after {:?}",
        backend::NAME,
        started.elapsed()
    );
    assert!(first.recv().is_err(), "told once");

    // No instance is bound, and every pipeline its stages dispatch through
    // is there: one for each step.
    let pipelines = engine
        .programs
        .built_pipelines(program)
        .expect("its registration built its pipelines");
    assert_eq!(pipelines, steps, "a pipeline for every step of every stage");
    assert_eq!(engine.programs.builds(), 1);

    // The program again: its id, told at once, and no build started.
    let (built, again) = told();
    assert_eq!(
        engine
            .register_program(&registration, built)
            .expect("the program registers again"),
        program
    );
    assert!(
        matches!(again.try_recv(), Ok(Ok(()))),
        "a program already built is told so inside the call"
    );
    assert_eq!(
        engine.programs.builds(),
        1,
        "a second registration starts no build"
    );

    // The hash names that program and no other: refused by the call, and
    // told nothing.
    let (built, untold) = told();
    let other = engine
        .register_program(
            &ProgramRegistration {
                program_hash: registration.program_hash,
                ..Default::default()
            },
            built,
        )
        .expect_err("another program under a registered hash is refused");
    assert!(
        other.to_string().contains("another launch package"),
        "{other}"
    );
    assert!(untold.recv().is_err(), "dropped untold");
    assert_eq!(engine.programs.builds(), 1);
    assert_eq!(
        engine.live().tallied(Tally::Unbuilt),
        0,
        "nothing met a program unbuilt"
    );

    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
