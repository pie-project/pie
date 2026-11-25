#![cfg(feature = "device")]
//! A host writer's cell that only a port names is taken by the fire that
//! consumes it (its head moves, and a fire with none behind it is blocked)
//! and not copied to the device: what lies there stays, and no kernel of
//! the program reads it. A cell an op reads is copied whole.

use std::sync::Arc;

use engine::{ChannelId, ProgramRegistration};
use engine_cuda::device::Context;
use engine_cuda::program::Endpoint;
use engine_cuda::program::session::{Fired, Launched};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Port, Stage};
use eta_ir::types::{Dtype, Shape};

/// Rows of the port's cell: no whole number of 16-byte words.
const ROWS: u32 = 4871;
const TOKENS: u32 = 0;
const VALUE: u32 = 1;
const OUT: u32 = 2;
/// A byte neither cell holds.
const MARK: u8 = 0xa5;

fn program() -> TraceContainer {
    let decl = |shape, host_role| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(Dtype::U32),
        capacity: 1,
        host_role,
        seeded: false,
    };
    TraceContainer {
        channels: vec![
            decl(Shape::vector(ROWS), HostRole::Writer),
            decl(Shape::vector(3), HostRole::Writer),
            decl(Shape::vector(3), HostRole::Reader),
        ],
        ports: vec![PortBinding {
            port: Port::EmbedTokens,
            source: PortSource::Channel(TOKENS),
        }],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(VALUE),
                Op::ChanPut {
                    chan: OUT,
                    value: 0,
                },
            ],
        }],
        ..TraceContainer::default()
    }
}

#[test]
fn a_cell_no_op_reads_is_taken_and_not_copied() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let bound = crate::bound::logits(program(), 16).expect("the program binds");
    let stages = compile_bound(&bound);
    let context = Context::bind(0, None).expect("a device context");
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let registration = ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    };
    let (mut plane, _fence) = crate::guest::alone_plane_fenced();
    let program = plane.register(&context, &registration).expect("registers");
    let stream = context.stream();
    let endpoint = |role, rows: u32| {
        Arc::new(Endpoint::open(role, 4 * rows, 1, None, stream).expect("the endpoint opens"))
    };
    let endpoints = [
        Some(endpoint(HostRole::Writer, ROWS)),
        Some(endpoint(HostRole::Writer, 3)),
        Some(endpoint(HostRole::Reader, 3)),
    ];
    let instance = plane
        .bind(
            program,
            &[],
            Extents {
                row_count: 1,
                sampled_rows: 1,
                query_len: 1,
                ..Extents::default()
            },
            GeometryClass::Host,
            eta_ir::registry::PortMask::NONE,
            false,
            &endpoints,
            &[ChannelId(1), ChannelId(2), ChannelId(3)],
            crate::guest::alone_slots(&plane, program),
            &crate::guest::alone_span,
            stream,
        )
        .expect("an instance binds");

    for round in 0..3u32 {
        let session = plane.instance_mut(instance).expect("the instance");
        let tokens: Vec<u8> = (0..ROWS)
            .flat_map(|row| (row + round).to_le_bytes())
            .collect();
        let value: Vec<u8> = (7..10)
            .flat_map(|v: u32| (v + round).to_le_bytes())
            .collect();
        for (channel, wire) in [(TOKENS, &tokens), (VALUE, &value)] {
            assert!(
                session.publish(channel, wire, stream).expect("publishes"),
                "round {round}: channel {channel} has room again"
            );
        }
        let head = session.cursor(TOKENS).expect("the channel").head;
        let (at, bytes) = session.cell(TOKENS as usize, head).expect("the cell");
        let garbage = vec![MARK.wrapping_add(round as u8); bytes as usize];
        engine_cuda::device::alloc::stage_raw(stream, at, &garbage).expect("the cell is marked");

        let fired = plane.fire(&context, instance).expect("the fire runs");
        assert!(matches!(fired, Fired::Committed), "{fired:?}");
        let session = plane.instance_mut(instance).expect("the instance");
        assert_eq!(
            session.take(OUT).expect("reads").as_deref(),
            Some(&value[..]),
            "the cell an op takes is copied whole"
        );
        assert_eq!(session.cursor(TOKENS).expect("the channel").head, head + 1);
        let mut held = vec![0u8; bytes as usize];
        engine_cuda::device::alloc::copy_d2h_on(stream, at, &mut held).expect("the cell reads");
        assert_eq!(held, garbage, "the port's cell is not copied");
    }
    // Back-pressure: with no cell behind its port a fire does not fly.
    let session = plane.instance_mut(instance).expect("the instance");
    let value: Vec<u8> = [1u32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert!(session.publish(VALUE, &value, stream).expect("publishes"));
    let launched = plane.stage(instance, Stage::Epilogue).expect("stages");
    assert!(
        matches!(launched, Launched::Refused(Fired::Blocked(TOKENS))),
        "{launched:?}"
    );
}
