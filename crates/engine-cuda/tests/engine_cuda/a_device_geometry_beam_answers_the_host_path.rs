#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine::fire::{Masking, Readout};
use engine::runahead::Runahead;
use engine_cuda::serve::Attached;
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
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

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const PROMPT: u32 = 20;
/// The pool every beam's mask spans: the prompt, then a slot a beam.
const KEYS: u32 = 32;
const MAX_BEAMS: u32 = 5;

/// One attention layer whose masked lanes read by their masks.
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
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, true, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: poem::Trace, path: &Path, runahead: Runahead) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(MAX_BEAMS, 64),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: MAX_BEAMS,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    for slot in 0..MAX_BEAMS {
        shell.open(slot).expect("the slot opens");
    }
    shell
}

const TOKENS: u32 = 0;
const MASK: u32 = 1;
const POSITIONS: u32 = 2;
const SLOTS: u32 = 3;

/// A beam step bound in the device-geometry class over `beams` lanes: its
/// tokens, its `[beams, KEYS]` ancestry mask, its positions and the slots
/// its rows land at, each re-put by its epilogue so the next fire finds the
/// same cell. The host stages none of them: the device places each lane.
fn beam_step(beams: u32) -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 4,
        host_role: HostRole::None,
        seeded: true,
    };
    let carried = |chan: u32, value: u32| {
        [
            Op::ChanTake(chan),
            Op::ChanPut { chan, value },
        ]
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            decl(Shape::vector(beams), EtaDtype::I32),
            decl(Shape::matrix(beams, KEYS), EtaDtype::Bool),
            decl(Shape::vector(beams), EtaDtype::U32),
            decl(Shape::vector(beams), EtaDtype::U32),
        ],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(TOKENS),
            },
            PortBinding {
                port: Port::Positions,
                source: PortSource::Channel(POSITIONS),
            },
            PortBinding {
                port: Port::Slots,
                source: PortSource::Channel(SLOTS),
            },
            PortBinding {
                port: Port::AttnMask,
                source: PortSource::Channel(MASK),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: [
                carried(TOKENS, 0),
                carried(MASK, 1),
                carried(POSITIONS, 2),
                carried(SLOTS, 3),
            ]
            .concat(),
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// Beam `i` writes slot `PROMPT + i` at depth `PROMPT` and sees the prompt
/// and its own cell, not its siblings'.
fn ancestry(beam: u32) -> Vec<bool> {
    (0..KEYS)
        .map(|key| key < PROMPT || key == PROMPT + beam)
        .collect()
}

fn token(beam: u32) -> u32 {
    (beam * 11 + 7) % VOCAB
}

/// What the program's cells hold: each beam's token (`None`: a skip, -1)
/// and slot.
struct Cells {
    tokens: Vec<Option<u32>>,
    slots: Vec<u32>,
}

impl Cells {
    fn of(beams: u32) -> Cells {
        Cells {
            tokens: (0..beams).map(|beam| Some(token(beam))).collect(),
            slots: (0..beams).map(|beam| PROMPT + beam).collect(),
        }
    }
}

fn bind(shell: &mut Shell, beams: u32, cells: &Cells) -> engine::InstanceId {
    let tokens: Vec<u8> = cells
        .tokens
        .iter()
        .flat_map(|token| token.map_or(-1, |token| token as i32).to_le_bytes())
        .collect();
    let ancestry: Vec<bool> = (0..beams).flat_map(ancestry).collect();
    // A bool cell rides the wire a bit a cell.
    let mut mask = vec![0u8; ancestry.len().div_ceil(8)];
    for (at, _) in ancestry.iter().enumerate().filter(|(_, kept)| **kept) {
        mask[at / 8] |= 1 << (at % 8);
    }
    let positions: Vec<u8> = (0..beams).flat_map(|_| PROMPT.to_le_bytes()).collect();
    let landing: Vec<u8> = cells.slots.iter().flat_map(|slot| slot.to_le_bytes()).collect();
    bind_with(
        shell,
        beam_step(beams),
        0xde60 + u64::from(beams),
        &[
            (TOKENS, tokens),
            (MASK, mask),
            (POSITIONS, positions),
            (SLOTS, landing),
        ],
        beams,
        true,
    )
}

/// Registers and binds `container` in the device-geometry class over `rows`
/// rows, its channels seeded with `seeds`, its rings the device's alone
/// (`shared`, as the runtime binds them) or the instance's own.
fn bind_with(
    shell: &mut Shell,
    container: TraceContainer,
    hash: u64,
    seeds: &[(u32, Vec<u8>)],
    beams: u32,
    shared: bool,
) -> engine::InstanceId {
    let bound = crate::bound::logits(container, VOCAB).expect("the program binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: hash,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let count = bound.container.channels.len();
    let ids: Vec<engine::ChannelId> = (1..=count as u64).map(engine::ChannelId).collect();
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    // Rings only the device touches, as the runtime binds a program's
    // channels: the host's view of one is a mirror its puts fill as they land.
    let adopted: Vec<Option<std::sync::Arc<engine_cuda::program::Endpoint>>> = bound
        .container
        .channels
        .iter()
        .map(|channel| {
            if !shared {
                return None;
            }
            let ChanDType::Concrete(dtype) = channel.dtype else {
                panic!("the beam step's channels are concrete");
            };
            let numel = channel.shape.numel() as usize;
            let cell = engine_cuda::program::launch::native_cell_bytes(dtype, numel) as u32;
            let span = engine_cuda::program::Endpoint::cell_span(cell, channel.capacity);
            let slots =
                guest.channel(&shell.guest_footprint(engine::chunk::SlotRole::Channel, span));
            let at = shell
                .guest_span(engine::chunk::SlotRole::Channel, &slots, span)
                .expect("the ring's slots");
            Some(std::sync::Arc::new(
                engine_cuda::program::Endpoint::open(
                    HostRole::None,
                    cell,
                    channel.capacity,
                    Some((at, slots)),
                    shell.stream(),
                )
                .expect("the ring opens"),
            ))
        })
        .collect();
    let slots = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| shared,
    );
    let extents = Extents {
        row_count: beams,
        token_count: beams,
        sampled_rows: beams,
        query_len: beams,
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
            seeds,
            extents,
            GeometryClass::DeviceGeometry,
            PortMask::DEVICE_GEOMETRY,
            &adopted,
            &ids,
            engine_cuda::program::Slots {
                instance: slots.0,
                rings: slots.1,
                scratch,
                pair,
                pair_generation: 0,
            },
        )
        .expect("the instance binds")
}

struct Pool {
    pages: Vec<u32>,
    depth: [u32; 1],
    tokens: Vec<[u32; 1]>,
    masks: Vec<Masking>,
    /// What the runtime hands a lane whose rows the device places: a row's
    /// placeholder token.
    placeholder: [u32; 1],
}

impl Pool {
    fn new(beams: u32) -> Pool {
        Pool {
            pages: (0..64 / PAGE).collect(),
            depth: [PROMPT],
            tokens: (0..beams).map(|beam| [token(beam)]).collect(),
            masks: (0..beams)
                .map(|beam| engine::ports::from_dense(&ancestry(beam), KEYS as usize))
                .collect(),
            placeholder: [0],
        }
    }

    /// The beams as the host places them: each in its own seat over the
    /// pool's pages at its slot, stating its mask.
    fn host_lanes(&self, trace: &poem::Trace) -> Vec<Seated<'_>> {
        let word = trace.facts.word(&Request::new(1, true));
        (0..self.tokens.len())
            .map(|beam| Seated {
                pages: &self.pages,
                held: Some(PROMPT + beam as u32),
                positions: &self.depth,
                mask: Some(&self.masks[beam]),
                ..Seated::of(Lane {
                    slot: beam as u32,
                    word,
                    tokens: &self.tokens[beam],
                })
            })
            .collect()
    }

    /// The beams as a device-geometry fire hands them: the working set's
    /// pages, a placeholder row each, nothing of where it lands.
    fn placed_lanes(&self, trace: &poem::Trace) -> Vec<Seated<'_>> {
        let word = trace.facts.word(&Request::new(1, true));
        (0..self.tokens.len())
            .map(|beam| Seated {
                translation: &self.pages,
                ..Seated::of(Lane {
                    slot: beam as u32,
                    word,
                    tokens: &self.placeholder,
                })
            })
            .collect()
    }
}

fn prefill(shell: &mut Shell, trace: &poem::Trace, pages: &[u32]) {
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
    let seated = Seated {
        pages,
        held: Some(0),
        ..Seated::of(Lane {
            slot: 0,
            word: trace.facts.word(&Request::new(PROMPT, false)),
            tokens: &prompt,
        })
    };
    shell
        .fire_rows(&[seated], &[Readout::Last])
        .expect("the prompt prefills");
}

fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0f32, f32::max)
}

fn setup(beams: u32, runahead: Runahead) -> (Shell, poem::Trace, tempfile::TempDir) {
    let trace = poem::trace_hybrid("devgeo-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 3307).write(dir.path());
    let mut shell = shell(trace.clone(), &path, runahead);
    prefill(&mut shell, &trace, &Pool::new(beams).pages);
    (shell, trace, dir)
}

/// Beams bound in the device-geometry class, their slots, positions,
/// tokens and masks in cells only the device writes, are placed on the
/// device: each fire answers what the same beams answer placed by the host,
/// staged ahead or not.
#[test]
fn a_device_geometry_beam_answers_the_host_path() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    for beams in [2, MAX_BEAMS] {
        for runahead in [Runahead::F1, Runahead::of(2)] {
            let (mut shell, trace, _dir) = setup(beams, runahead);
            let pool = Pool::new(beams);
            let want: Vec<Readout> = (0..beams).map(|_| Readout::Last).collect();
            let host = shell
                .fire_rows(&pool.host_lanes(&trace), &want)
                .expect("the host-placed beams fire");
            let instance = bind(&mut shell, beams, &Cells::of(beams));
            let attached = [Attached {
                lane: 0,
                lanes: beams,
                instance,
                adapter_only: false,
            }];
            for fire in 0..2 {
                let got = shell
                    .fire_media(&pool.placed_lanes(&trace), &attached, &[])
                    .expect("the device-placed beams fire");
                let off = worst(&got, &host);
                assert!(
                    off <= 1e-5,
                    "{beams} beams, {runahead:?}, fire {fire}: the device's placement answers \
                     {off} away from the host's"
                );
            }
        }
    }
}

/// A device-geometry fire is prepared and queued while the fire before it
/// has not run: its stream is held until both are queued, so a prepare that
/// read a cell on the host (or waited for the fire before) would wait on
/// the held stream. The second answers what the host-placed beams answer.
#[test]
fn a_device_geometry_fire_is_staged_while_the_one_before_runs() {
    use engine::frame::Shell as FrameShell;
    use engine_cuda::serve::StepView;

    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let beams = MAX_BEAMS;
    let (mut shell, trace, _dir) = setup(beams, Runahead::of(2));
    let pool = Pool::new(beams);
    let want: Vec<Readout> = (0..beams).map(|_| Readout::Last).collect();
    let host = shell
        .fire_rows(&pool.host_lanes(&trace), &want)
        .expect("the host-placed beams fire");
    let instance = bind(&mut shell, beams, &Cells::of(beams));
    let attached = [Attached {
        lane: 0,
        lanes: beams,
        instance,
        adapter_only: false,
    }];
    let lanes = pool.placed_lanes(&trace);
    shell
        .fire_media(&lanes, &attached, &[])
        .expect("the device-placed beams fire");

    let (open, gate) = std::sync::mpsc::channel::<()>();
    let held_too_long = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let told = std::sync::Arc::clone(&held_too_long);
    engine_cuda::device::ctx::host_fn_on(
        shell.stream(),
        Box::new(move || {
            if gate.recv_timeout(std::time::Duration::from_secs(10)).is_err() {
                told.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }),
    )
    .expect("the gate is queued on the compute stream");
    let step = || StepView {
        lanes: &lanes,
        attachments: &attached,
        media: &[],
        voxels: &[],
    };
    let fire = |shell: &mut Shell| {
        let prepared = FrameShell::prepare(shell, step(), None).expect("the beams prepare");
        let enqueued = FrameShell::enqueue(shell, prepared).expect("the beams enqueue");
        FrameShell::settle(shell, enqueued).expect("the beams settle")
    };
    let _first = fire(&mut shell);
    let mut second = fire(&mut shell);
    let _ = open.send(());
    shell
        .read_out(&mut second)
        .expect("the second fire reads out");
    assert!(
        !held_too_long.load(std::sync::atomic::Ordering::SeqCst),
        "preparing two device-geometry fires waited on the stream the gate held: the second \
         was not staged while the first was still to run"
    );
    let off = worst(&second.logits, &host);
    assert!(
        off <= 1e-5,
        "a device-geometry fire staged ahead answers {off} away from the host-placed beams"
    );
}

/// A row whose token is -1 is skipped: it reads no embedding, writes no key
/// (a lane reading its slot sees what was there before), reads nothing out,
/// and the beams beside it answer as they do without it.
#[test]
fn a_skipped_row_writes_nothing_and_reads_nothing_out() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let beams = 2;
    let (mut shell, trace, _dir) = setup(beams, Runahead::of(2));
    let pool = Pool::new(beams);
    let want: Vec<Readout> = (0..beams).map(|_| Readout::Last).collect();
    let host = shell
        .fire_rows(&pool.host_lanes(&trace), &want)
        .expect("the host-placed beams fire");
    // What a lane reading beam 1's slot sees, beam 1's key landed.
    let reader = || {
        let word = trace.facts.word(&Request::new(1, true));
        Seated {
            pages: &pool.pages,
            held: Some(PROMPT + beams),
            positions: &pool.depth,
            ..Seated::of(Lane {
                slot: 0,
                word,
                tokens: &pool.tokens[0],
            })
        }
    };
    let before = shell
        .fire_rows(&[reader()], &[Readout::Last])
        .expect("the reader fires");
    let mut cells = Cells::of(beams);
    cells.tokens[1] = None;
    let instance = bind(&mut shell, beams, &cells);
    let attached = [Attached {
        lane: 0,
        lanes: beams,
        instance,
        adapter_only: false,
    }];
    let got = shell
        .fire_media(&pool.placed_lanes(&trace), &attached, &[])
        .expect("the beams fire, one skipped");
    let off = worst(&got[..1], &host[..1]);
    assert!(off <= 1e-5, "the beam beside a skipped row answers {off} away");
    assert!(
        got[1].iter().all(|&logit| logit == 0.0),
        "a skipped row reads something out"
    );
    let after = shell
        .fire_rows(&[reader()], &[Readout::Last])
        .expect("the reader fires again");
    assert_eq!(
        worst(&after, &before),
        0.0,
        "a skipped row wrote a key over its slot"
    );
}

/// A slot past what the lane's working set reserved is refused on the
/// device: the lane faults by name and nothing of it is written.
#[test]
fn a_placed_slot_past_the_reserve_faults_its_lane() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let beams = 2;
    let (mut shell, trace, _dir) = setup(beams, Runahead::of(2));
    let pool = Pool::new(beams);
    let mut cells = Cells::of(beams);
    // The mask's keys bound the lane at KEYS slots.
    cells.slots[1] = KEYS;
    let instance = bind(&mut shell, beams, &cells);
    let attached = [Attached {
        lane: 0,
        lanes: beams,
        instance,
        adapter_only: false,
    }];
    let refused = shell
        .fire_media(&pool.placed_lanes(&trace), &attached, &[])
        .expect_err("a slot past the reserve is refused");
    assert!(
        format!("{refused}").contains("lane 1's rows land where the device says"),
        "{refused}"
    );
}

/// A program attached over one lane whose geometry ports describe two: a
/// pass's ports never reach past its own lanes, so the fire is refused by
/// name, placed on the device or read on the host, before anything is
/// staged; the lane beside it (another tenant's) reads what it read before.
#[test]
fn ports_describing_lanes_past_the_attachment_are_refused() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let decl = |shape: Shape, dtype: EtaDtype| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 4,
        host_role: HostRole::None,
        seeded: true,
    };
    let (indptr, tokens, positions, slots) = (0u32, 1u32, 2u32, 3u32);
    let container = || TraceContainer {
        names: Vec::new(),
        channels: vec![
            decl(Shape::vector(3), EtaDtype::U32),
            decl(Shape::vector(2), EtaDtype::I32),
            decl(Shape::vector(2), EtaDtype::U32),
            decl(Shape::vector(2), EtaDtype::U32),
        ],
        ports: vec![
            PortBinding {
                port: Port::EmbedIndptr,
                source: PortSource::Channel(indptr),
            },
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(tokens),
            },
            PortBinding {
                port: Port::Positions,
                source: PortSource::Channel(positions),
            },
            PortBinding {
                port: Port::Slots,
                source: PortSource::Channel(slots),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: [indptr, tokens, positions, slots]
                .iter()
                .flat_map(|&chan| [Op::ChanTake(chan), Op::ChanPut { chan, value: chan }])
                .collect(),
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let words = |values: &[u32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let seeds = vec![
        (indptr, words(&[0, 1, 2])),
        (tokens, words(&[token(0), token(1)])),
        (positions, words(&[PROMPT, PROMPT])),
        (slots, words(&[PROMPT, PROMPT + 1])),
    ];
    let (mut shell, trace, _dir) = setup(2, Runahead::of(2));
    let pool = Pool::new(2);
    let word = trace.facts.word(&Request::new(1, true));
    // The neighbour: another tenant's lane over pages of its own, reading
    // the slot the program would place its second lane at.
    let other: Vec<u32> = (64 / PAGE..128 / PAGE).collect();
    prefill(&mut shell, &trace, &other);
    let neighbour = || Seated {
        pages: &other,
        held: Some(PROMPT + 2),
        positions: &pool.depth,
        ..Seated::of(Lane {
            slot: 1,
            word,
            tokens: &pool.tokens[0],
        })
    };
    let before = shell
        .fire_rows(&[neighbour()], &[Readout::Last])
        .expect("the neighbour fires");
    // Placed on the device (no report asked) and read on the host (one asked).
    for (reports, shared) in [(0u64, true), (1, false)] {
        let instance = bind_with(&mut shell, container(), 0xde70 + reports, &seeds, 2, shared);
        let attached = [Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }];
        let lanes = [
            Seated {
                translation: &pool.pages,
                reports,
                ..Seated::of(Lane {
                    slot: 0,
                    word,
                    tokens: &pool.placeholder,
                })
            },
            neighbour(),
        ];
        let refused = shell
            .fire_media(&lanes, &attached, &[])
            .expect_err("ports past the attachment are refused");
        let said = format!("{refused}");
        assert!(
            said.contains("describes 2") || said.contains("carries 2 token(s)"),
            "reports {reports}: {said}"
        );
    }
    let after = shell
        .fire_rows(&[neighbour()], &[Readout::Last])
        .expect("the neighbour fires again");
    assert_eq!(worst(&after, &before), 0.0, "the neighbour's cache was written");
}
