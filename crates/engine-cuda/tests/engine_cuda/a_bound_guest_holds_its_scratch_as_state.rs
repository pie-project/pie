#![cfg(feature = "device")]
//! A guest program's scratch is state, not the fire's: each instance holds
//! its lane's value scratch and the pair its tables, both in `Instance`
//! slots, and a fire of 64 bound lanes lays nothing of the guest in the
//! activation's span, which stays what the rungs state.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine::frame::Shell as FrameShell;
use engine_cuda::serve::{Attached, StepView};
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
/// A decode row a lane: 64 lanes are a fire of 64 rows.
const ROWS: u32 = 1;
const LANES: u32 = 64;

struct Micro;

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(KV_ROW, Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", plane, WIDTH));
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, 0.125, None, None, None, false, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        ..crate::common::boot(
            trace,
            &contract,
            path,
            Budget::new(LANES, LANES * ROWS),
            64,
            LANES,
        )
    })
    .expect("the shell loads");
    for slot in 0..LANES {
        shell.open(slot).expect("the slot opens");
    }
    shell
}

/// An epilogue putting the lane's argmax.
fn epilogue() -> TraceContainer {
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
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
                    shape: Shape::matrix(1, VOCAB),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn register(shell: &mut Shell) -> engine::ProgramId {
    let bound = crate::bound::logits(epilogue(), VOCAB).expect("the epilogue binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    shell
        .register_program(&ProgramRegistration {
            program_hash: 0x9e57,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers")
}

fn fire(
    shell: &mut Shell,
    word: u64,
    tokens: &[Vec<u32>],
    attachments: &[Attached],
) -> Vec<Vec<f32>> {
    let seated: Vec<Seated<'_>> = tokens
        .iter()
        .enumerate()
        .map(|(lane, tokens)| Seated {
            reports: 10 + lane as u64,
            ..Seated::of(Lane {
                slot: lane as u32,
                word,
                tokens,
            })
        })
        .collect();
    let prepared = FrameShell::prepare(
        shell,
        StepView {
            lanes: &seated,
            attachments,
            media: &[],
            voxels: &[],
            shape: None,
        },
        None,
    )
    .expect("the step prepares");
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the step enqueues");
    let mut settled = shell.settle_step(enqueued, None).expect("the step settles");
    shell.read_out(&mut settled).expect("the step reads out");
    std::mem::take(&mut settled.logits)
}

#[test]
fn a_bound_guest_holds_its_scratch_as_state() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-state-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4243).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = (0..LANES)
        .map(|lane| (0..ROWS).map(|row| (lane * 7 + row * 13) % VOCAB).collect())
        .collect();

    let mut plain = shell(trace.clone(), &path);
    let want = fire(&mut plain, word, &tokens, &[]);
    drop(plain);

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };

    // Each instance holds its lane's scratch, the pair its tables.
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    assert!(state.instance_bytes() > 0, "{state:?}");
    assert!(state.per_pair > 0, "{state:?}");

    let rungs = shell.chunk_facts().expect("the chunk facts").act_chunks;
    let activation = shell.footprint().1;
    let instances: Vec<engine::InstanceId> = (0..LANES)
        .map(|_| {
            let (scratch, pair) = guest.places((program, extents), &state);
            assert!(
                matches!(
                    (&scratch, &pair),
                    (engine::ScratchPlace::Run { .. }, engine::ScratchPlace::Run { .. })
                ),
                "a small guest's state lies in runs of Instance slots: {scratch:?} {pair:?}"
            );
            let slots = guest.slots(
                &shell.program_facts(program).expect("the program's facts"),
                |_| false,
            );
            shell
                .bind_program(
                    program,
                    &[],
                    extents,
                    GeometryClass::Host,
                    PortMask::NONE,
                    &[None],
                    &[engine::ChannelId(1)],
                    engine_cuda::program::Slots {
                        instance: slots.0,
                        rings: slots.1,
                        scratch,
                        pair,
                        pair_generation: 0,
                    },
                )
                .expect("the instance binds")
        })
        .collect();
    // 64 bound lanes ask the same state, and nothing of the span.
    assert_eq!(
        shell
            .guest_state(program, extents)
            .expect("the guest state"),
        state
    );
    assert_eq!(
        shell.chunk_facts().expect("the chunk facts").act_chunks,
        rungs,
        "the rungs' activation is what it was before any guest bound"
    );
    assert_eq!(
        shell.footprint().1,
        activation,
        "nothing backed past the span"
    );

    let attachments: Vec<Attached> = instances
        .iter()
        .enumerate()
        .map(|(lane, &instance)| Attached {
            lane: lane as u32,
            lanes: 1,
            instance,
            adapter_only: false,
        })
        .collect();
    let got = fire(&mut shell, word, &tokens, &attachments);
    shell.drain().expect("the device drains");
    assert_eq!(
        shell.footprint().1,
        activation,
        "the fire laid nothing past the span"
    );

    for (lane, instance) in instances.iter().enumerate() {
        let off = got[lane]
            .iter()
            .zip(&want[lane])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            off <= 1e-2,
            "lane {lane} answers {off} away from the plain fire"
        );
        let cell = shell
            .program_instance(*instance)
            .expect("the instance settles")
            .expect("the instance")
            .take(0)
            .expect("its channel reads")
            .expect("its epilogue put a cell");
        let argmax = i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"));
        // Its own lane's greatest logit (bf16 logits tie: either index).
        let most = got[lane].iter().copied().fold(f32::MIN, f32::max);
        assert_eq!(
            got[lane][argmax as usize],
            most,
            "lane {lane}'s epilogue put {argmax}, not an argmax of its own logits"
        );
    }

    // The pair's state is given back at its last close.
    for instance in instances {
        shell
            .close_program_instance(instance)
            .expect("the instance closes");
    }
}

/// A pair's state lies in slots the runtime may give back while its
/// instances stay bound (a park sheds them; a last close the engine defers
/// frees them at once), and another owner may write them before the pair is
/// taken again at the very same refs. Taken again, the slots are another
/// taking (`pair_generation`): the bind lays the pair's state over whatever
/// the other owner left, and its fire answers as before. The same live
/// slots bound again lay nothing.
#[test]
fn a_pair_taken_again_at_the_same_refs_is_laid_again() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-pair-retaken", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4247).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = (0..LANES)
        .map(|lane| (0..ROWS).map(|row| (lane * 5 + row * 11) % VOCAB).collect())
        .collect();

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    assert!(state.per_pair > 0, "{state:?}");

    let bind = |shell: &mut Shell,
                guest: &mut crate::guest::Guest,
                pair: &engine::ScratchPlace,
                generation: u64| {
        let (scratch, _) = guest.places((program, extents), &state);
        let slots = guest.slots(
            &shell.program_facts(program).expect("the program's facts"),
            |_| false,
        );
        shell
            .bind_program(
                program,
                &[],
                extents,
                GeometryClass::Host,
                PortMask::NONE,
                &[None],
                &[engine::ChannelId(1)],
                engine_cuda::program::Slots {
                    instance: slots.0,
                    rings: slots.1,
                    scratch,
                    pair: pair.clone(),
                    pair_generation: generation,
                },
            )
            .expect("the instance binds")
    };
    // Each instance's epilogue puts an argmax of its own lane's logits.
    let answers = |shell: &mut Shell, instance: engine::InstanceId| {
        let got = fire(
            shell,
            word,
            &tokens,
            &[Attached {
                lane: 0,
                lanes: 1,
                instance,
                adapter_only: false,
            }],
        );
        shell.drain().expect("the device drains");
        let cell = shell
            .program_instance(instance)
            .expect("the instance settles")
            .expect("the instance")
            .take(0)
            .expect("its channel reads")
            .expect("its epilogue put a cell");
        let argmax = i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"));
        let most = got[0].iter().copied().fold(f32::MIN, f32::max);
        got[0].get(argmax as usize).copied() == Some(most)
    };

    let (_, pair) = guest.places((program, extents), &state);
    let engine::ScratchPlace::Run { first, slots } = pair else {
        panic!("a small guest's pair lies in a run of Instance slots: {pair:?}");
    };
    let first_bound = bind(&mut shell, &mut guest, &pair, 1);
    assert!(answers(&mut shell, first_bound), "laid at its first bind");
    assert_eq!(shell.pair_stale(program, extents), Some(false));

    // The same live slots bound again: nothing is laid again.
    let again = bind(&mut shell, &mut guest, &pair, 1);
    assert_eq!(
        shell.pair_stale(program, extents),
        Some(false),
        "the same taking of the slots is not laid again"
    );
    shell
        .close_program_instance(again)
        .expect("the instance closes");

    // The pair's slots go back while its instance stays bound, another
    // owner writes them, and they are taken again at the same refs.
    let refs: Vec<u32> = (first..first + slots).collect();
    let at = shell
        .guest_span(engine::chunk::SlotRole::Instance, &refs, state.per_pair)
        .expect("the pair's run is a guest item's span");
    let foreign = vec![0xA5u8; usize::try_from(state.per_pair).expect("bytes")];
    engine_cuda::device::alloc::stage_raw(shell.stream(), at, &foreign)
        .expect("another owner writes the slots");
    let retaken = bind(&mut shell, &mut guest, &pair, 2);
    assert_eq!(
        shell.pair_stale(program, extents),
        Some(true),
        "another taking of the same refs is laid again"
    );
    assert!(
        answers(&mut shell, retaken),
        "its fire runs on the pair's own state, not the other owner's words"
    );

    for instance in [first_bound, retaken] {
        shell
            .close_program_instance(instance)
            .expect("the instance closes");
    }
}

/// A guest's state changes owner clean: instance A runs over its scratch,
/// which holds A's bytes as it closes; its scratch is handed on to instance
/// B, and B's scratch reads zero, never A's bytes. The pair's slots given
/// back, written by another owner and taken again read zero at the new
/// taking; an unpark's fresh room reads zero though garbage lay there.
#[test]
fn a_guests_state_reads_zero_when_its_slots_change_owner() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-state-owner", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4251).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = vec![vec![7]];

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    let span_of = |shell: &Shell, place: &engine::ScratchPlace, bytes: u64| {
        let engine::ScratchPlace::Run { first, slots } = place else {
            panic!("a small guest's state lies in a run of Instance slots: {place:?}");
        };
        let refs: Vec<u32> = (*first..first + slots).collect();
        shell
            .guest_span(engine::chunk::SlotRole::Instance, &refs, bytes)
            .expect("a guest item's span")
    };
    let foreign = |shell: &Shell, place: &engine::ScratchPlace, bytes: u64| {
        let bytes = vec![0xA5u8; usize::try_from(bytes).expect("bytes")];
        engine_cuda::device::alloc::stage_raw(shell.stream(), span_of(shell, place, bytes.len() as u64), &bytes)
            .expect("another owner writes the slots");
    };
    let zero = |shell: &mut Shell, place: &engine::ScratchPlace, bytes: u64| {
        shell.drain().expect("the device drains");
        let mut out = vec![0x5Au8; usize::try_from(bytes).expect("bytes")];
        engine_cuda::device::alloc::copy_d2h(span_of(shell, place, bytes), &mut out)
            .expect("the room reads");
        out.iter().all(|&byte| byte == 0)
    };
    let bind = |shell: &mut Shell,
                guest: &mut crate::guest::Guest,
                scratch: &engine::ScratchPlace,
                pair: &engine::ScratchPlace,
                generation: u64| {
        let slots = guest.slots(
            &shell.program_facts(program).expect("the program's facts"),
            |_| false,
        );
        let id = shell
            .bind_program(
                program,
                &[],
                extents,
                GeometryClass::Host,
                PortMask::NONE,
                &[None],
                &[engine::ChannelId(1)],
                engine_cuda::program::Slots {
                    instance: slots.0.clone(),
                    rings: slots.1.clone(),
                    scratch: scratch.clone(),
                    pair: pair.clone(),
                    pair_generation: generation,
                },
            )
            .expect("the instance binds");
        (id, slots)
    };
    let attach = |instance| {
        [Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }]
    };

    let (scratch, pair) = guest.places((program, extents), &state);
    let (a, _) = bind(&mut shell, &mut guest, &scratch, &pair, 1);
    fire(&mut shell, word, &tokens, &attach(a));
    shell.drain().expect("the device drains");
    // Whatever A's launches left, A's bytes are in its scratch as it closes.
    foreign(&shell, &scratch, state.instance_bytes());
    shell.close_program_instance(a).expect("A closes");
    // The pair's slots went back with A's close and another owner wrote them.
    foreign(&shell, &pair, state.per_pair);

    let (b, b_slots) = bind(&mut shell, &mut guest, &scratch, &pair, 2);
    assert!(
        zero(&mut shell, &scratch, state.instance_bytes()),
        "B's scratch holds none of A's bytes"
    );
    assert!(
        zero(&mut shell, &pair, state.per_pair),
        "the pair's new taking holds none of the other owner's bytes"
    );
    fire(&mut shell, word, &tokens, &attach(b));
    shell.drain().expect("the device drains");

    // An unpark: fresh room holding garbage reads zero once B is there.
    let (fresh, fresh_pair) = guest.places((program, "unparked"), &state);
    foreign(&shell, &fresh, state.instance_bytes());
    foreign(&shell, &fresh_pair, state.per_pair);
    // Its items stay where they lie; its state moves.
    shell
        .rebind_program(
            b,
            engine_cuda::program::Slots {
                instance: b_slots.0,
                rings: b_slots.1,
                scratch: fresh.clone(),
                pair: fresh_pair.clone(),
                pair_generation: 3,
            },
        )
        .expect("the unpark rebinds");
    assert!(
        zero(&mut shell, &fresh, state.instance_bytes()),
        "the unparked scratch holds none of the room's last bytes"
    );
    assert!(
        zero(&mut shell, &fresh_pair, state.per_pair),
        "the unparked pair holds none of the room's last bytes"
    );
    fire(&mut shell, word, &tokens, &attach(b));
    shell.drain().expect("the device drains");
    shell.close_program_instance(b).expect("B closes");
}

/// A park releases what it shed (`engine::Rebind::Parked`): instance A
/// parks with its tables kept in place, its scratch and its pair given
/// back; instance B binds into exactly that scratch and pair without an
/// overlap refusal; A's unpark rebinds it to fresh room, claimed again,
/// and it fires there. Unparked without the park said, B's bind would be
/// refused by name on A's stale claim.
#[test]
fn a_parked_instance_s_shed_room_is_another_s_and_its_unpark_claims_again() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-park-claims", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4253).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = vec![vec![7]];

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    let bind = |shell: &mut Shell,
                guest: &mut crate::guest::Guest,
                scratch: &engine::ScratchPlace,
                pair: &engine::ScratchPlace,
                generation: u64| {
        let slots = guest.slots(
            &shell.program_facts(program).expect("the program's facts"),
            |_| false,
        );
        let id = shell.bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &[None],
            &[engine::ChannelId(1)],
            engine_cuda::program::Slots {
                instance: slots.0.clone(),
                rings: slots.1.clone(),
                scratch: scratch.clone(),
                pair: pair.clone(),
                pair_generation: generation,
            },
        );
        (id, slots)
    };
    let attach = |instance| {
        [Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }]
    };

    let (scratch, pair) = guest.places((program, extents), &state);
    let (a, a_slots) = bind(&mut shell, &mut guest, &scratch, &pair, 1);
    let a = a.expect("A binds");
    fire(&mut shell, word, &tokens, &attach(a));
    shell.drain().expect("the device drains");

    shell.park_program(a, true).expect("A parks, its tables kept in place");
    let (b, _) = bind(&mut shell, &mut guest, &scratch, &pair, 2);
    let b = b.expect("B binds on the scratch and pair A's park gave back, no overlap");

    let (fresh, fresh_pair) = guest.places((program, "unparked"), &state);
    shell
        .rebind_program(
            a,
            engine_cuda::program::Slots {
                instance: a_slots.0,
                rings: a_slots.1,
                scratch: fresh,
                pair: fresh_pair,
                pair_generation: 3,
            },
        )
        .expect("A's unpark claims fresh room");
    fire(&mut shell, word, &tokens, &attach(a));
    fire(&mut shell, word, &tokens, &attach(b));
    shell.drain().expect("the device drains");
    for instance in [a, b] {
        shell
            .close_program_instance(instance)
            .expect("the instance closes");
    }
}

/// An evacuation under a kept park (`engine::Rebind::Parked { kept }`):
/// instance A parks with its tables kept in place, and the chunk holding
/// them moves. A's claim moves with them: an instance bound at A's own refs
/// (now the new bytes) is refused by name, and one bound in the chunk now
/// lying on A's old bytes binds. A's unpark rebinds it at the same refs and
/// it answers as before the move. Unmoved, A's claim would stand on the old
/// bytes: the first bind would pass over A's tables and the second be
/// refused on a stale claim.
#[test]
fn an_evacuation_moves_a_kept_park_s_claim_and_tables() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-kept-evac", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4261).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = vec![vec![11]];

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let committed = segments * facts.segment_chunks;
    let mut guest = crate::guest::Guest::of(&facts, committed);
    let per_chunk = facts
        .shapes
        .iter()
        .find(|shape| shape.role == engine::chunk::SlotRole::Instance)
        .expect("the load states the Instance shape")
        .slots_per_chunk;
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    let bind = |shell: &mut Shell,
                instance: Vec<u32>,
                rings: Vec<engine::program::RingSlots>,
                scratch: &engine::ScratchPlace,
                pair: &engine::ScratchPlace,
                generation: u64| {
        shell.bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &[None],
            &[engine::ChannelId(1)],
            engine_cuda::program::Slots {
                instance,
                rings,
                scratch: scratch.clone(),
                pair: pair.clone(),
                pair_generation: generation,
            },
        )
    };
    let attach = |instance| {
        [Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }]
    };
    let argmax = |shell: &mut Shell, instance| {
        let cell = shell
            .program_instance(instance)
            .expect("the instance settles")
            .expect("the instance")
            .take(0)
            .expect("its channel reads")
            .expect("its epilogue put a cell");
        i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"))
    };

    let (scratch, pair) = guest.places((program, extents), &state);
    let a_slots = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let a = bind(&mut shell, a_slots.0.clone(), a_slots.1.clone(), &scratch, &pair, 1)
        .expect("A binds");
    let before = fire(&mut shell, word, &tokens, &attach(a));
    shell.drain().expect("the device drains");
    let put = argmax(&mut shell, a);

    shell.park_program(a, true).expect("A parks, its tables kept in place");
    // A's tables' chunk onto the last chunk backed: the chunk that lay
    // there takes the bytes A's tables left.
    let tables = a_slots.0[0] / per_chunk;
    // The last chunk of state backed (chunk 0 is the null chunk).
    let onto = committed.min(facts.state_chunks() + 1) - 1;
    assert_ne!(tables, onto);
    shell
        .evacuate(&engine::chunk::Evacuation {
            moves: vec![(tables, onto)],
        })
        .expect("the chunk moves under the kept park");

    // At A's own refs, the new bytes: A's.
    let (c_scratch, c_pair) = guest.places((program, "c"), &state);
    let c_rings = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let refused = bind(&mut shell, a_slots.0.clone(), c_rings.1.clone(), &c_scratch, &c_pair, 4);
    let why = refused.expect_err("an instance at A's refs overlaps A's kept tables where they moved");
    assert!(
        why.to_string().contains(&format!("instance {}", a.0)),
        "refused naming A: {why}"
    );
    // In the chunk now lying on A's old bytes: free of A.
    let displaced: Vec<u32> = a_slots.0.iter().map(|&r| onto * per_chunk + r % per_chunk).collect();
    let (b_scratch, b_pair) = guest.places((program, "b"), &state);
    let b = bind(&mut shell, displaced, c_rings.1, &b_scratch, &b_pair, 5)
        .expect("B binds on the bytes A's tables left, no stale claim");

    let (fresh, fresh_pair) = guest.places((program, "unparked"), &state);
    shell
        .rebind_program(
            a,
            engine_cuda::program::Slots {
                instance: a_slots.0,
                rings: a_slots.1,
                scratch: fresh,
                pair: fresh_pair,
                pair_generation: 3,
            },
        )
        .expect("A's unpark claims its moved tables");
    let after = fire(&mut shell, word, &tokens, &attach(a));
    shell.drain().expect("the device drains");
    assert_eq!(after, before, "A answers as before the move");
    assert_eq!(argmax(&mut shell, a), put, "A's epilogue puts what it put");
    for instance in [a, b] {
        shell
            .close_program_instance(instance)
            .expect("the instance closes");
    }
}

/// A pair whose every member is parked while an evacuation moves its runs
/// is laid again where they lie now: instance A parks (its pair's members
/// all parked), the chunk holding the pair moves, and another chunk moves
/// onto the bytes it left. B then binds at the same pair and taking, and A
/// comes back at it; both fire and answer as A did before the move.
/// Unmoved, the pair's tables would stand at the old bytes, another chunk's
/// now, and the fires would read them.
#[test]
fn a_pair_whose_members_are_all_parked_is_laid_again_where_a_move_left_it() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-pair-relay", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4271).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = vec![vec![13]];

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let committed = segments * facts.segment_chunks;
    let mut guest = crate::guest::Guest::of(&facts, committed);
    let per_chunk = facts
        .shapes
        .iter()
        .find(|shape| shape.role == engine::chunk::SlotRole::Instance)
        .expect("the load states the Instance shape")
        .slots_per_chunk;
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    let bind = |shell: &mut Shell,
                slots: (Vec<u32>, Vec<engine::program::RingSlots>),
                scratch: &engine::ScratchPlace,
                pair: &engine::ScratchPlace| {
        shell.bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &[None],
            &[engine::ChannelId(1)],
            engine_cuda::program::Slots {
                instance: slots.0,
                rings: slots.1,
                scratch: scratch.clone(),
                pair: pair.clone(),
                pair_generation: 1,
            },
        )
    };
    let attach = |instance| {
        [Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }]
    };
    // The cell the instance's epilogue put: what reads the pair's tables.
    let put = |shell: &mut Shell, instance| {
        let cell = shell
            .program_instance(instance)
            .expect("the instance settles")
            .expect("the instance")
            .take(0)
            .expect("its channel reads")
            .expect("its epilogue put a cell");
        i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"))
    };

    let (scratch, pair) = guest.places((program, extents), &state);
    let engine::ScratchPlace::Run { first, .. } = pair else {
        panic!("a small guest's pair lies in a run of Instance slots: {pair:?}");
    };
    let a_slots = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let a = bind(&mut shell, a_slots.clone(), &scratch, &pair).expect("A binds");
    let want = fire(&mut shell, word, &tokens, &attach(a));
    shell.drain().expect("the device drains");
    let wanted = put(&mut shell, a);

    shell.park_program(a, false).expect("A parks: every member of its pair parked");
    // The pair's chunk onto the last chunk of state backed, then another
    // chunk onto the bytes it left: the old bytes are another chunk's.
    let held = first / per_chunk;
    let last = committed.min(facts.state_chunks() + 1) - 1;
    let other = last - 1;
    assert!(held != last && held != other, "the pair's chunk is not a test's spare");
    shell
        .evacuate(&engine::chunk::Evacuation { moves: vec![(held, last)] })
        .expect("the pair's chunk moves while every member is parked");
    shell
        .evacuate(&engine::chunk::Evacuation { moves: vec![(other, held)] })
        .expect("another chunk moves onto the bytes it left");

    // B joins the pair at its taking; A comes back at it.
    let (b_scratch, _) = guest.places((program, "b"), &state);
    let b_slots = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let b = bind(&mut shell, b_slots, &b_scratch, &pair).expect("B binds at the pair's taking");
    let got_b = fire(&mut shell, word, &tokens, &attach(b));
    shell.drain().expect("the device drains");
    let b_put = put(&mut shell, b);
    let (a_scratch, _) = guest.places((program, "a again"), &state);
    shell
        .rebind_program(
            a,
            engine_cuda::program::Slots {
                instance: a_slots.0,
                rings: a_slots.1,
                scratch: a_scratch,
                pair: pair.clone(),
                pair_generation: 1,
            },
        )
        .expect("A comes back at the pair's taking");
    let got_a = fire(&mut shell, word, &tokens, &attach(a));
    shell.drain().expect("the device drains");
    assert_eq!(got_b, want, "B's logits are A's before the move");
    assert_eq!(got_a, want, "A's logits are its own before the move");
    assert_eq!(b_put, wanted, "B's epilogue reads the pair where the move left it");
    assert_eq!(put(&mut shell, a), wanted, "A's epilogue reads the pair where the move left it");
    for instance in [a, b] {
        shell
            .close_program_instance(instance)
            .expect("the instance closes");
    }
}

/// An evacuation is all or nothing (`Engine::evacuate`): one whose last
/// check fails moves nothing. Instance A's state chunk is evacuated, and
/// the evacuation also names a channel whose slots cannot be spanned, the
/// check that once ran after the copy and the table change. It is refused,
/// and A's state lies where it lay, its bytes as they were; the same moves
/// with no such channel then go through, the bytes with them, and A fires
/// where they went.
#[test]
fn an_evacuation_refused_late_moves_nothing() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-evacuation-atomic", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4257).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = vec![vec![7]];

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let committed = (segments * facts.segment_chunks).min(facts.state_chunks() + 1);
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    let (scratch, pair) = guest.places((program, extents), &state);
    let engine::ScratchPlace::Run { first, slots } = scratch else {
        panic!("a small guest's state lies in a run of Instance slots: {scratch:?}");
    };
    let refs: Vec<u32> = (first..first + slots).collect();
    let bytes = state.instance_bytes();
    let span = |shell: &Shell| {
        shell
            .guest_span(engine::chunk::SlotRole::Instance, &refs, bytes)
            .expect("A's scratch spans")
    };
    let read = |shell: &mut Shell| {
        shell.drain().expect("the device drains");
        let mut out = vec![0u8; usize::try_from(bytes).expect("bytes")];
        engine_cuda::device::alloc::copy_d2h(span(shell), &mut out).expect("the scratch reads");
        out
    };

    let (instance, rings) = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let a = shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &[None],
            &[engine::ChannelId(1)],
            engine_cuda::program::Slots {
                instance,
                rings,
                scratch: scratch.clone(),
                pair,
                pair_generation: 1,
            },
        )
        .expect("A binds");
    let attach = [Attached {
        lane: 0,
        lanes: 1,
        instance: a,
        adapter_only: false,
    }];
    fire(&mut shell, word, &tokens, &attach);
    shell.drain().expect("the device drains");
    // A's state, marked: what an evacuation copies, or leaves.
    let marked: Vec<u8> = (0..bytes).map(|at| (at % 251) as u8 ^ 0x3C).collect();
    engine_cuda::device::alloc::stage_raw(shell.stream(), span(&shell), &marked)
        .expect("A's state is marked");

    let per_chunk = facts
        .shapes
        .iter()
        .find(|shape| shape.role == engine::chunk::SlotRole::Instance)
        .expect("an Instance shape")
        .slots_per_chunk;
    let held = first / per_chunk;
    let free = committed - 1;
    assert_ne!(held, free, "A's chunk and the free one differ");
    let evacuation = engine::chunk::Evacuation {
        moves: vec![(held, free)],
    };
    let before = span(&shell);
    let refused = shell.evacuate_with(&evacuation, &[(vec![u32::MAX - 1], 64)]);
    assert!(refused.is_err(), "a channel with no span refuses the evacuation");
    assert_eq!(span(&shell), before, "the refused evacuation left the table");
    assert_eq!(read(&mut shell), marked, "and A's bytes where they lay");

    shell
        .evacuate_with(&evacuation, &[])
        .expect("the same moves go through alone");
    assert_ne!(span(&shell), before, "A's state chunk moved");
    assert_eq!(read(&mut shell), marked, "its bytes with it");
    fire(&mut shell, word, &tokens, &attach);
    shell.drain().expect("A fires where its state went");
    shell.close_program_instance(a).expect("A closes");
}
