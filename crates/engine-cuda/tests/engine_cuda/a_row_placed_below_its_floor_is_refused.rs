#![cfg(feature = "device")]
//! A device-geometry lane is handed its working set's whole page table to
//! read through, the shared prefix's pages among them; a row its program
//! places below the lane's frozen prefix (`KvDelta::folded`), or past what
//! it may write (`KvDelta::write_end`), refuses that lane by name before
//! any of it runs, and the shared page another lane reads is unchanged.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine::chunk::{SlotKind, SlotRole};
use engine_cuda::serve::Attached;
use engine_cuda::{Boot, Fault, Graphs, Lane, Seated, Shell};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};
use poem_compiler::Budget;

pub(crate) const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
pub(crate) const PAGE: u32 = 16;

pub(crate) struct Micro;

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

pub(crate) fn shell(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        page_size: PAGE,
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        ..crate::common::boot(trace, &contract, path, Budget::new(2, 64), 256, 2)
    })
    .expect("the shell loads");
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    shell
}

fn const_port(port: Port, words: &[u32]) -> PortBinding {
    PortBinding {
        port,
        source: PortSource::Const {
            dtype: EtaDtype::U32,
            shape: Shape::vector(words.len() as u32),
            data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        },
    }
}

/// A program whose ports state a lane of one row for each of `slots`:
/// token `token`, landing at that slot of the lane's working set, reading
/// `kv_len` slots; its epilogue puts lane 0's logits in a ring the host
/// reads.
pub(crate) fn placing(token: u32, slots: &[u32], kv_len: u32) -> TraceContainer {
    let lanes = slots.len() as u32;
    let mut ports = vec![
        const_port(Port::EmbedTokens, &vec![token; slots.len()]),
        const_port(Port::EmbedIndptr, &(0..=lanes).collect::<Vec<u32>>()),
        const_port(Port::Slots, slots),
        const_port(Port::KvLen, &vec![kv_len; slots.len()]),
    ];
    ports.sort_by_key(|binding| binding.port);
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::matrix(1, VOCAB),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports,
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, VOCAB),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// Binds `slots`' placing program in `DeviceGeometry`, its channels from
/// `first_channel` on.
pub(crate) fn bind(
    shell: &mut Shell,
    guest: &mut crate::guest::Guest,
    slots: &[u32],
    first_channel: u64,
) -> engine::InstanceId {
    let bound = crate::bound::logits(placing(9, slots, PAGE + 1), VOCAB).expect("the ports bind");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0xf100_0000 + first_channel,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let count = bound.container.channels.len();
    let ids: Vec<engine::ChannelId> = (0..count as u64)
        .map(|at| engine::ChannelId(first_channel + at))
        .collect();
    let slots = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let extents = Extents {
        row_count: 1,
        token_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let (scratch, pair) = guest.places(
        (program, extents),
        &shell
            .guest_state(program, extents)
            .expect("the program's guest state"),
    );
    shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::DeviceGeometry,
            PortMask::NONE,
            &vec![None; count],
            &ids,
            engine_cuda::program::Slots {
                instance: slots.0,
                rings: slots.1,
                scratch,
                pair,
                pair_generation: 0,
            },
        )
        .expect("the device-geometry instance binds")
}

/// The reader: lane 0, its prefix in `shared`, decoding one token at slot
/// `PAGE` into `own`. Its logits.
pub(crate) fn read(shell: &mut Shell, word: u64, pages: &[u32]) -> Vec<f32> {
    let tokens = [21u32];
    let mut rows = shell
        .fire_media(
            &[Seated {
                pages,
                held: Some(PAGE),
                ..Seated::of(Lane {
                    slot: 0,
                    word,
                    tokens: &tokens,
                })
            }],
            &[],
            &[],
        )
        .expect("the reader's decode lands");
    rows.swap_remove(0)
}

/// The placing lane, lane 0 of its fire on slot 1, reading through
/// `translation`, its rows to land in `[folded, write_end)`.
fn place(
    shell: &mut Shell,
    word: u64,
    instance: engine::InstanceId,
    translation: &[u32],
    folded: u32,
) -> Result<Vec<Vec<f32>>, Fault> {
    let tokens = [0u32];
    shell.fire_media(
        &[Seated {
            translation,
            folded,
            write_end: 2 * PAGE,
            ..Seated::of(Lane {
                slot: 1,
                word,
                tokens: &tokens,
            })
        }],
        &[Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }],
        &[],
    )
}

/// A loaded micro model with two slots open, whose slot 0 wrote a page
/// of prompt rows into `shared`.
pub(crate) struct Fixture {
    pub shell: Shell,
    pub guest: crate::guest::Guest,
    /// The shared prefix's page, the reader's own and the placing lane's.
    pub shared: u32,
    pub own: u32,
    pub placer: u32,
    /// A one-row decode's word.
    pub word: u64,
    /// The reader's logits over `[shared, own]` before anything else ran.
    pub want: Vec<f32>,
    pub _dir: tempfile::TempDir,
}

pub(crate) fn fixture() -> Fixture {
    let trace = poem::trace_hybrid("write-floor-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 2718).write(dir.path());
    let mut shell = shell(trace.clone(), &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    shell.take_refs();
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    // The kv list's refs in chunk 1, the first past the null chunk: the
    // guest's own chunks lie past the middle of what is committed.
    let list = facts
        .shapes
        .iter()
        .find(|shape| {
            !matches!(shape.role, SlotRole::Instance | SlotRole::Channel)
                && matches!(shape.kind, SlotKind::List { window: None, .. })
        })
        .expect("a kv list");
    let refs = |at: u32| list.slots_per_chunk + at;
    let (shared, own, placer) = (refs(0), refs(1), refs(2));

    // The shared prefix: a page of rows another request wrote.
    let prompt: Vec<u32> = (0..PAGE).map(|at| (at * 7 + 3) % VOCAB).collect();
    shell
        .fire_media(
            &[Seated {
                pages: &[shared],
                held: Some(0),
                ..Seated::of(Lane {
                    slot: 0,
                    word: trace.facts.word(&Request::new(PAGE, false)),
                    tokens: &prompt,
                })
            }],
            &[],
            &[],
        )
        .expect("the shared prefix lands");
    let word = trace.facts.word(&Request::new(1, false));
    let reader = [shared, own];
    let want = read(&mut shell, word, &reader);
    assert_eq!(
        read(&mut shell, word, &reader),
        want,
        "the reader answers alike"
    );

    Fixture {
        shell,
        guest,
        shared,
        own,
        placer,
        word,
        want,
        _dir: dir,
    }
}

#[test]
fn a_row_placed_below_its_floor_is_refused() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let Fixture {
        mut shell,
        mut guest,
        shared,
        own,
        placer,
        word,
        want,
        _dir,
    } = fixture();
    let reader = [shared, own];

    // Frozen through the shared page, the placing lane may write its own
    // page alone: a row it places at slot 3, or past its write end, is its
    // refusal, by name, and nothing of it runs.
    let table = [shared, placer];
    let below = bind(&mut shell, &mut guest, &[3], 1);
    let refused = place(&mut shell, word, below, &table, PAGE).expect_err("a row below its floor");
    assert!(
        matches!(refused, Fault::KvWrite { lane: 0, .. }),
        "the lane is refused by name: {refused}"
    );
    assert_eq!(refused.lane(), Some(0), "the lane alone, not the frame");
    let why = refused.to_string();
    assert!(
        why.contains("kv slot 3") && why.contains("frozen prefix"),
        "{why}"
    );
    assert_eq!(
        read(&mut shell, word, &reader),
        want,
        "the shared page is unchanged"
    );

    let past = bind(&mut shell, &mut guest, &[2 * PAGE], 11);
    let refused = place(&mut shell, word, past, &table, PAGE).expect_err("a row past its end");
    assert!(
        matches!(refused, Fault::KvWrite { lane: 0, .. }),
        "{refused}"
    );

    // A row at its floor lands in its own page; the reader's is untouched.
    let at_floor = bind(&mut shell, &mut guest, &[PAGE], 21);
    place(&mut shell, word, at_floor, &table, PAGE).expect("a row at its floor lands");
    assert_eq!(
        read(&mut shell, word, &reader),
        want,
        "its own page is not the reader's"
    );

    // The same row at slot 3, from a lane the page is its own (frozen
    // through nothing), lands there and moves what the reader answers:
    // the refusal above is what kept it.
    place(&mut shell, word, below, &table, 0).expect("a lane that owns the page writes it");
    assert_ne!(
        read(&mut shell, word, &reader),
        want,
        "a row written at slot 3 moves the reader"
    );
}
