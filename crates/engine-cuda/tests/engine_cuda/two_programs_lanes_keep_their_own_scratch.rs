//! A guest lane's scratch is its instance's state (`engine::GuestState`):
//! two programs' lanes in one fire, each addressing its scratch where its
//! binding places it (`engine::ScratchPlace`), draw what the interpreter
//! draws. The negative control places every instance's scratch on the same
//! slots, and the lanes then write over one another.

use engine::chunk::SlotRole;
use engine::{ChannelId, ProgramId, ProgramRegistration, ScratchPlace};
use engine_cuda::device::{Buffer, Context};
use engine_cuda::program::session::{Fired, Launched};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::Value;
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::validate::BoundTrace;

use crate::a_lane_sampler_draws_what_the_interpreter_draws::{
    KNOB, LOGITS, Noise, STATE, TOKEN, VOCAB, host_token, sampler,
};

/// The pool's `Instance` slot, as the CUDA load lays it.
const SLOT: u64 = 4 << 10;
/// Instances of each program in the fire.
const EACH: u32 = 3;

fn extents() -> Extents {
    Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    }
}

/// Draws one fire carrying `EACH` lanes of a top-p and of a top-k program,
/// each instance's scratch at `place(program, index, slots)`, and counts
/// the lanes that drew the interpreter's token.
fn draw(place: impl Fn(usize, u32, u32) -> u32) -> (usize, usize) {
    let context = Context::bind(0, None).expect("a device context");
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let pool = Buffer::zeroed(256 << 20).expect("the pool's buffer");
    let base = Box::leak(Box::new(pool)).ptr();
    // Slots of the `Instance` role in the one buffer; any other item a
    // buffer of its own.
    let span = move |role: SlotRole, refs: &[u32], bytes: u64| -> engine_cuda::Result<u64> {
        match (role, refs.first()) {
            (SlotRole::Instance, Some(&first)) => Ok(base + u64::from(first) * SLOT),
            _ => crate::guest::alone_span(role, refs, bytes),
        }
    };
    let mut plane = crate::guest::alone_plane();
    let mut noise = Noise(0x0dd5_ca1e_7e57_0001);
    let mut programs: Vec<(BoundTrace, ProgramId)> = Vec::new();
    for rank in [false, true] {
        let bound = crate::bound::logits(sampler(rank), VOCAB).expect("the sampler binds");
        let stages = compile_bound(&bound);
        let program = plane
            .register(
                &context,
                &ProgramRegistration {
                    program_hash: bound.hash,
                    emitted_kernels: emit_program(backend, &stages, &bound),
                    emitter_version: backend.emitter_version(),
                    region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(
                        &stages,
                    ),
                    launch: eta_compiler::codegen::launch::build(&bound, &stages),
                    ..Default::default()
                },
            )
            .expect("the sampler compiles");
        programs.push((bound, program));
    }
    // Each pair's state past every instance's scratch.
    let mut pairs = 60_000u32;
    let mut lanes = Vec::new();
    for (index, (bound, program)) in programs.iter().enumerate() {
        let state = plane
            .guest_state(*program, extents())
            .expect("the program's guest state");
        let slots = |bytes: u64| u32::try_from(bytes.div_ceil(SLOT)).expect("slots");
        let pair = ScratchPlace::Run {
            first: pairs,
            slots: slots(state.per_pair),
        };
        pairs += slots(state.per_pair);
        let count = bound.container.channels.len();
        let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
        for at in 0..EACH {
            let logits: Vec<f32> = (0..VOCAB).map(|_| noise.unit() * 6.0 - 3.0).collect();
            let knob = if index == 1 {
                Value::I32(vec![40])
            } else {
                Value::F32(vec![0.9])
            };
            let seeds = vec![
                (STATE, Value::U32(vec![5 + at, 9 + index as u32])),
                (KNOB, knob),
                (LOGITS, Value::F32(logits)),
            ];
            let want = host_token(bound, &seeds);
            let wire: Vec<(u32, Vec<u8>)> =
                seeds.iter().map(|(c, v)| (*c, v.to_le_bytes())).collect();
            let per = slots(state.per_instance);
            let instance = plane
                .bind(
                    *program,
                    &wire,
                    extents(),
                    GeometryClass::Host,
                    eta_ir::registry::PortMask::NONE,
                    false,
                    &vec![None; count],
                    &ids,
                    engine_cuda::program::Slots {
                        scratch: ScratchPlace::Run {
                            first: place(index, at, per),
                            slots: per,
                        },
                        pair: pair.clone(),
                        ..crate::guest::alone_slots(&plane, *program)
                    },
                    &span,
                    context.stream(),
                )
                .expect("the sampler binds an instance");
            lanes.push((instance, want));
        }
    }
    // One wave: every lane of both programs in one fire.
    for (instance, _) in &lanes {
        let launched = plane
            .stage(*instance, Stage::Epilogue)
            .expect("the lane stages");
        assert!(matches!(launched, Launched::Airborne), "{launched:?}");
    }
    plane.fly(&context).expect("the wave flies");
    plane.land(&context).expect("the wave lands");
    context.synchronize().expect("the fire settles");
    let mut agree = 0;
    for (instance, want) in &lanes {
        let fired = plane.settle_launched(*instance).expect("the lane settles");
        assert!(matches!(fired, Fired::Committed), "{fired:?}");
        let cell = plane
            .instance_mut(*instance)
            .expect("the instance")
            .take(TOKEN)
            .expect("the token reads")
            .expect("a token");
        let got = i32::from_le_bytes(cell[..4].try_into().unwrap());
        if got == *want {
            agree += 1;
        } else {
            eprintln!("instance {instance}: the device drew {got}, the interpreter {want}");
        }
    }
    (agree, lanes.len())
}

#[test]
fn two_programs_lanes_keep_their_own_scratch() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    // Each instance its own run, the programs' interleaved.
    let (agree, total) =
        draw(|program, at, slots| (at * 2 + program as u32) * slots.next_multiple_of(16));
    assert_eq!(
        agree, total,
        "a lane drew another than the interpreter's token"
    );
}

/// The negative control: every instance's scratch on the same slots. The
/// lanes write over one another's, and draw what the interpreter does not.
#[test]
fn lanes_placed_on_one_scratch_write_over_one_another() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let (agree, total) = draw(|_, _, _| 0);
    eprintln!("overlapping scratch: {agree} of {total} lanes drew the interpreter's token");
    assert!(
        agree < total,
        "lanes sharing their scratch all drew their own token"
    );
}
