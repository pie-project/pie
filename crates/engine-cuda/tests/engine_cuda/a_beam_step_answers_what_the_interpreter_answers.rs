//! The guest-program contract (`eta_compiler::contract`) on this engine: each
//! case's program registered, bound with its seeds, fired over its logits,
//! and every channel the host reads exactly what the interpreter puts there.

use engine::{ChannelId, ProgramRegistration};
use engine_cuda::device::{Buffer, Context};
use engine_cuda::program::launch::INTRINSIC_STORAGE_F32;
use engine_cuda::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::contract::{Case, beam_stage};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::registry::GeometryClass;

/// What the device puts in each of `case`'s readers.
fn fired(case: &Case) -> Vec<Vec<u8>> {
    let context = Context::bind(0, None).expect("a device context");
    let stages = compile_bound(&case.bound);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen");
    let registration = ProgramRegistration {
        program_hash: 0xbea0_0000 ^ u64::from(case.rows),
        emitted_kernels: emit_program(backend, &stages, &case.bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
        ..Default::default()
    };
    let mut plane = crate::guest::alone_plane();
    let program = plane
        .register(&context, &registration)
        .expect("the program compiles for the device");
    let extents = Extents {
        row_count: case.rows,
        sampled_rows: case.rows,
        query_len: case.rows,
        ..Extents::default()
    };
    let count = case.bound.container.channels.len();
    let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
    let instance = plane
        .bind(
            program,
            &case.wire_seeds(),
            extents,
            GeometryClass::Host,
            eta_ir::registry::PortMask::NONE,
            false,
            &vec![None; count],
            &ids,
            crate::guest::alone_slots(&plane, program),
            &crate::guest::alone_span,
            context.stream(),
        )
        .expect("the instance binds");
    let mut logits = Buffer::zeroed(case.logits.len() * 4).expect("a logits plane");
    let bytes: Vec<u8> = case.logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    logits.write(0, &bytes).expect("the logits upload");
    plane
        .bind_output(
            instance,
            "logits",
            0,
            logits.ptr(),
            INTRINSIC_STORAGE_F32,
            case.vocab,
            case.vocab,
            0,
        )
        .expect("the logits bind");
    let fired = plane.fire(&context, instance).expect("the fire runs");
    assert!(matches!(fired, Fired::Committed), "{}: {fired:?}", case.name);
    let session = plane.instance_mut(instance).expect("the instance");
    case.readers
        .iter()
        .map(|&channel| {
            session
                .take(channel)
                .expect("a channel reads")
                .expect("the channel holds a cell")
        })
        .collect()
}

#[test]
fn a_beam_step_answers_what_the_interpreter_answers() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    for case in beam_stage() {
        let want = case.expected();
        let got = fired(&case);
        for ((channel, got), want) in case.readers.iter().zip(&got).zip(&want) {
            assert_eq!(got, want, "{}: channel {channel}", case.name);
        }
    }
}
