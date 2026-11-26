//! A beam-style program bound in `GeometryClass::DeviceGeometry`: two beams
//! forked off one prompt share one working set, each lane its translation
//! table, and the program states on its ports each beam's token, its RoPE
//! position (its depth, not its slot), the working-set slot its row lands at
//! (`fill + beam`) and the ancestry mask over the cells past the folded
//! prompt. The positions and slots are loop-carried on the device (each
//! pass puts the next fire's), so a fire staged behind the last resolves
//! what that pass put once it landed. Each beam's logits, read back through
//! the guest's `logits` output, answer what the host path answers for the
//! same beams stated on the lanes, for two decode steps fired one after the other
//! (engine-xla's `a_device_geometry_beam_answers_the_host_path`).

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration, ChannelSeed};
use engine::chunk::{ChunkFacts, Growth, ShapeId};
use engine::fire::{
    Attachment, FrameSubmission, KvDelta, Lane, ListKind, Mask, Masking, Readout, Step,
};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, InstanceId, ProgramRegistration};
use engine_gpu::store::layout::LIST;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Literal, Shape};
use poem::Request;
use poem_ir::InputAxis;

use super::outputs::contract_for;
use super::slots::Slots;
use crate::backend;
use crate::model_logits::fixture;

const MODEL: &str = "qwen3-micro-text";
const PAGE: u32 = 16;
/// The prompt every beam shares, folded: one page.
const PROMPT: u32 = PAGE;
const BEAMS: u32 = 2;
const STEPS: u32 = 2;
/// The mask's cells: the slots past the prompt, a beam's a step.
const CELLS: u32 = BEAMS * STEPS;

/// The beams' tokens and parents at each step: at step 1 both beams descend
/// from beam 1, so beam 0's ancestry is not its own lane's.
const TOKENS: [[u32; 2]; 2] = [[11, 42], [5, 29]];
const PARENTS: [[usize; 2]; 2] = [[0, 1], [1, 1]];

fn decl(shape: Shape, dtype: EtaDtype, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded,
    }
}

/// Channel 0 the beams' tokens and 3 their mask, from the host; 1 their
/// positions and 2 their slots, seeded and carried on the device (each
/// pass puts the next fire's: a step deeper, `BEAMS` slots on); 4 the
/// logits, to the host.
fn beam_program(vocab: u32) -> TraceContainer {
    let indptr: Vec<u8> = (0..=BEAMS).flat_map(|w| w.to_le_bytes()).collect();
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![
            decl(Shape::vector(BEAMS), EtaDtype::I32, HostRole::Writer, false),
            decl(Shape::vector(BEAMS), EtaDtype::U32, HostRole::None, true),
            decl(Shape::vector(BEAMS), EtaDtype::U32, HostRole::None, true),
            decl(
                Shape::matrix(BEAMS, CELLS),
                EtaDtype::Bool,
                HostRole::Writer,
                false,
            ),
            decl(
                Shape::matrix(BEAMS, vocab),
                EtaDtype::F32,
                HostRole::Reader,
                false,
            ),
        ],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(0),
            },
            PortBinding {
                port: Port::EmbedIndptr,
                source: PortSource::Const {
                    dtype: EtaDtype::U32,
                    shape: Shape::vector(BEAMS + 1),
                    data: indptr,
                },
            },
            PortBinding {
                port: Port::Positions,
                source: PortSource::Channel(1),
            },
            PortBinding {
                port: Port::AttnMask,
                source: PortSource::Channel(3),
            },
            PortBinding {
                port: Port::Slots,
                source: PortSource::Channel(2),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                // 0..=3: the fire's ports.
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::ChanTake(2),
                Op::ChanTake(3),
                // 4: the logits, to the host.
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(BEAMS, vocab),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 4, value: 4 },
                // 5..=7: the next fire's positions, a step deeper.
                Op::Const(Literal::U32(1)),
                Op::Broadcast {
                    value: 5,
                    shape: Shape::vector(BEAMS),
                },
                Op::Add(1, 6),
                Op::ChanPut { chan: 1, value: 7 },
                // 8..=10: the next fire's slots, `BEAMS` on.
                Op::Const(Literal::U32(BEAMS)),
                Op::Broadcast {
                    value: 8,
                    shape: Shape::vector(BEAMS),
                },
                Op::Add(2, 9),
                Op::ChanPut { chan: 2, value: 10 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A bool cell's wire bytes: a bit a cell, the first the lowest.
fn bits(cells: &[bool]) -> Vec<u8> {
    let mut bytes = vec![0u8; cells.len().div_ceil(8)];
    for (at, &cell) in cells.iter().enumerate() {
        if cell {
            bytes[at / 8] |= 1 << (at % 8);
        }
    }
    bytes
}

/// Each beam's mask at each step: its parent's cells, and its own at
/// column `step * BEAMS + beam` (slot `PROMPT + column`).
fn masks() -> Vec<Vec<Vec<bool>>> {
    let mut held: Vec<Vec<bool>> = vec![vec![false; CELLS as usize]; BEAMS as usize];
    let mut out = Vec::new();
    for (step, parents) in PARENTS.iter().enumerate() {
        held = (0..BEAMS as usize)
            .map(|beam| {
                let mut row = held[parents[beam]].clone();
                row[step * BEAMS as usize + beam] = true;
                row
            })
            .collect();
        out.push(held.clone());
    }
    out
}

/// A row of mask cells as the runs a lane states: unset first.
fn runs(row: &[bool]) -> Mask {
    Mask::new(engine_gpu::program::mask_runs(row), row.len() as u64)
}

/// Slot refs of one shape out of chunks of the test's own.
struct Refs {
    per: u32,
    next: u32,
}

impl Refs {
    fn of(facts: &ChunkFacts, id: ShapeId, chunk: u32) -> Refs {
        let per = facts
            .shape(id)
            .unwrap_or_else(|| panic!("{MODEL} lays out shape {id}"))
            .slots_per_chunk;
        Refs {
            per,
            next: chunk * per,
        }
    }

    fn take(&mut self) -> u32 {
        self.next += 1;
        self.next - 1
    }
}

struct Rig {
    engine: backend::Engine,
    asynchronous: bool,
    heard: mpsc::Receiver<(engine::StepDone, engine::StepOutcome)>,
}

impl Rig {
    /// Fires `lanes` (attached to `instance` where given) and settles the
    /// frame; each lane's readout where it reads one.
    fn fire(&mut self, lanes: Vec<Lane>, instance: Option<InstanceId>) -> Vec<Vec<f32>> {
        let mut ticket = self.submit(lanes, instance);
        self.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        ticket.steps[0]
            .readouts
            .iter()
            .map(|readout| readout.values.clone())
            .collect()
    }

    fn submit(&mut self, lanes: Vec<Lane>, instance: Option<InstanceId>) -> engine::FrameTicket {
        let width = lanes.len() as u32;
        self.engine
            .submit(&FrameSubmission::of(Step {
                attachments: instance
                    .map(|instance| Attachment {
                        lane: 0,
                        lanes: width,
                        instance,
                        adapter_only: false,
                        scratch: engine::ScratchPlace::None,
                    })
                    .into_iter()
                    .collect(),
                lanes,
                media: Vec::new(),
                voxels: Vec::new(),
                shape: None,
            }))
            .expect("the frame fires")
    }

    /// The next logits cell the program put, once its pass landed.
    fn logits(&mut self, instance: InstanceId) -> Vec<f32> {
        for _ in 0..600 {
            if let Some(cell) = engine::Channels::take(&mut self.engine, instance, 4)
                .expect("the logits channel reads")
            {
                return cell
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect();
            }
            // A pass recorded to land later reports its step once it has.
            if self.asynchronous {
                let _ = self.heard.recv_timeout(Duration::from_millis(100));
            }
        }
        panic!("the guest's logits never landed")
    }
}

fn answers_the_host_path(frames: u8) {
    let boot = backend::DeviceBoot::default();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let fixture = fixture(MODEL, None);
    let mut engine = backend::Engine::new(boot, contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: Checkpoint::Path(fixture.container.clone()),
            budgets: Budgets {
                max_lanes: 4,
                // The overreach step fires a prompt beside a beam lane.
                max_tokens: 2 * PAGE,
                buckets: vec![PAGE, 2 * PAGE],
                max_adapters: 0,
                page_size: PAGE,
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
            frames_in_flight: frames,
            steps: Default::default(),
        })
        .expect("the miniature loads");
    assert_eq!(loaded.caps.geometry, GeometryClass::DeviceGeometry);
    assert!(
        loaded
            .caps
            .ports
            .covers(PortMask::DEVICE_GEOMETRY.with(Port::AttnMask)),
        "{:?}",
        loaded.caps.ports
    );
    let asynchronous = engine::Completion::asynchronous(&engine);
    let (tx, heard) = mpsc::channel();
    let tx = Mutex::new(tx);
    engine::Completion::on_complete(
        &mut engine,
        Arc::new(move |at, outcome| {
            let _ = tx.lock().expect("the sink's sender").send((at, outcome));
        }),
    );

    // Chunks of the test's own above the committed ones: the guest's two,
    // then the kv pages.
    let facts = loaded.caps.chunks.clone();
    let segment = facts.segment_chunks;
    // A pool committed whole at load (wgpu's) takes the test's chunks past
    // its seats; one that grows commits them past what the load did.
    let (first, commit) = if facts.segments.growth == Growth::Fixed {
        let seats = 1 + engine
            .loaded()
            .expect("the shell stands")
            .pool_layout()
            .seat_chunks();
        let first = seats.div_ceil(segment) * segment;
        (first, (first + 3).div_ceil(segment))
    } else {
        let first = facts.segments.committed * segment;
        (first, facts.segments.committed + 3u32.div_ceil(segment))
    };
    assert!(
        commit <= facts.segments.max,
        "the test's chunks span {commit} segments, the pool has {}",
        facts.segments.max
    );
    engine::Pool::commit(&mut engine, commit).expect("the segments commit");
    let mut slots = Slots::of(&facts, first, first + 1);
    let mut refs = Refs::of(&facts, LIST, first + 2);
    assert!(refs.per >= 5, "a chunk holds the test's pages");

    let masked = fixture.trace.facts.word(&Request::new(1, true));
    let plain = fixture.trace.facts.word(&Request::new(PROMPT, false));
    let mut rig = Rig {
        engine,
        asynchronous,
        heard,
    };

    // Two working sets, each the prompt then the beams' cells: one the host
    // path's, one the device path's.
    let prompt: Vec<u32> = (0..PROMPT).map(|at| (at * 53 + 5) % 256).collect();
    let sets: [Vec<u32>; 2] = [
        vec![refs.take(), refs.take()],
        vec![refs.take(), refs.take()],
    ];
    let mut alone = Vec::new();
    for (slot, set) in sets.iter().enumerate() {
        alone = rig.fire(
            vec![Lane {
                slot: slot as u32,
                word: plain,
                tokens: prompt.clone(),
                readout: Readout::Last,
                tables: vec![(LIST, vec![set[0]])],
                ..Lane::default()
            }],
            None,
        );
    }
    let masks = masks();

    // The host path: each beam states its table, extent, position and mask.
    let mut want = Vec::new();
    for step in 0..STEPS as usize {
        let lanes = (0..BEAMS as usize)
            .map(|beam| {
                let slot = PROMPT + (step as u32) * BEAMS + beam as u32;
                Lane {
                    slot: beam as u32,
                    word: masked,
                    tokens: vec![TOKENS[step][beam]],
                    positions: vec![PROMPT + step as u32],
                    readout: Readout::Last,
                    tables: vec![(LIST, sets[0].clone())],
                    kv: KvDelta {
                        held: slot,
                        folded: PROMPT,
                        ..KvDelta::default()
                    },
                    mask: Some(Masking::Extent(runs(&masks[step][beam]))),
                    ..Lane::default()
                }
            })
            .collect();
        want.push(rig.fire(lanes, None));
    }

    let width = want[0][0].len() as u32;
    assert!(width > 0, "the host path reads its logits");

    // The device path: the same beams, stated on the program's ports.
    // The outputs the load states: the program reads its `logits`.
    let mut outputs: Vec<OutputDecl> = Vec::new();
    for decl in loaded.caps.classes.classes.iter().flat_map(|c| &c.outputs) {
        if outputs.iter().any(|seen| seen.name == decl.name) {
            continue;
        }
        outputs.push(OutputDecl {
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
    let bound = eta_ir::validate::bind(beam_program(width), vec![outputs], EngineCaps::default())
        .expect("the beam program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine::Programs::register(
        &mut rig.engine,
        &ProgramRegistration {
            program_hash: 0xbea3_9e0,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        },
    )
    .expect("the beam program registers");
    let channels: Vec<ChannelId> = bound
        .container
        .channels
        .iter()
        .enumerate()
        .map(|(at, decl)| {
            let id = ChannelId((2 << 20) + at as u64);
            slots.register(
                &mut rig.engine,
                ChannelRegistration {
                    id,
                    shape: decl.shape.dims()[..decl.shape.rank()].to_vec(),
                    dtype: decl.dtype.clone(),
                    host_role: decl.host_role,
                    seeded: decl.seeded,
                    extern_dir: None,
                    capacity: decl.capacity,
                    extern_name: Vec::new(),
                    slots: Vec::new(),
                },
            );
            id
        })
        .collect();
    let binding = slots.binding(
        &rig.engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: vec![
                ChannelSeed {
                    channel: 1,
                    bytes: words(&[PROMPT; BEAMS as usize]),
                },
                ChannelSeed {
                    channel: 2,
                    bytes: words(&(PROMPT..PROMPT + BEAMS).collect::<Vec<_>>()),
                },
            ],
            geometry: GeometryClass::DeviceGeometry,
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            ring_slots: Vec::new(),
            adapter_slots: Vec::new(),
            extents: engine::program::BindExtents {
                row_count: BEAMS,
                token_count: BEAMS,
                sampled_rows: BEAMS,
                query_len: 1,
                ..engine::program::BindExtents::default()
            },
            scratch: engine::ScratchPlace::None,
            pair: engine::ScratchPlace::None,
            pair_generation: 0,
        },
        &channels,
    );
    let instance = engine::Programs::bind(&mut rig.engine, &binding)
        .expect("a device-geometry instance binds")
        .id;

    // Both steps' host cells first, then the two fires: the
    // second resolves the positions and slots the first's pass puts.
    for step in 0..STEPS as usize {
        let tokens: Vec<u32> = TOKENS[step].to_vec();
        let mask: Vec<bool> = masks[step].concat();
        for (channel, cell) in [(0u32, words(&tokens)), (3, bits(&mask))] {
            assert!(
                engine::Channels::publish(&mut rig.engine, instance, channel, &cell)
                    .expect("the cell publishes"),
                "the ring had room"
            );
        }
    }
    let device_lanes = || -> Vec<Lane> {
        (0..BEAMS)
            .map(|beam| Lane {
                slot: 2 + beam,
                word: masked,
                tokens: vec![0],
                readout: Readout::Last,
                tables: vec![(LIST, sets[1].clone())],
                list: ListKind::Translation,
                kv: KvDelta {
                    folded: PROMPT,
                    // The working set's beam cells are the lanes' own.
                    write_end: PROMPT + CELLS,
                    ..KvDelta::default()
                },
                ..Lane::default()
            })
            .collect()
    };
    // Each frame's readouts are asked for before the next submits; the
    // second still resolves what the first's pass puts, landed or on its
    // way (two frames in flight).
    for _ in 0..STEPS {
        let mut ticket = rig.submit(device_lanes(), Some(instance));
        rig.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
    }
    for (step, want) in want.iter().enumerate() {
        let got = rig.logits(instance);
        assert_eq!(got.len(), (BEAMS * width) as usize, "a logits row a beam");
        for beam in 0..BEAMS as usize {
            let row = &got[beam * width as usize..][..width as usize];
            assert_eq!(want[beam].len(), width as usize);
            let apart = row
                .iter()
                .zip(&want[beam])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                apart <= 1e-3,
                "step {step} beam {beam}: the device path's logits are {apart} apart from the \
                 host path's"
            );
        }
        eprintln!("{frames} frame(s), step {step}: every beam the host path's");
    }

    // Ports describing more lanes than the attachment owns: the constant
    // `embed_indptr` describes both beams and the attachment owns one. Its
    // lane is refused by name before any of its geometry applies, and the
    // neighbour lane, another request's prompt on a set of its own, answers
    // as it does alone.
    for (channel, cell) in [(0u32, words(&TOKENS[0])), (3, bits(&masks[0].concat()))] {
        assert!(
            engine::Channels::publish(&mut rig.engine, instance, channel, &cell)
                .expect("the cell publishes"),
            "the ring had room"
        );
    }
    let neighbour = Lane {
        slot: 0,
        word: plain,
        tokens: prompt.clone(),
        readout: Readout::Last,
        tables: vec![(LIST, vec![refs.take()])],
        ..Lane::default()
    };
    let _ = rig.heard.try_iter().count();
    let mut ticket = rig
        .engine
        .submit(&FrameSubmission::of(Step {
            attachments: vec![Attachment {
                lane: 0,
                lanes: 1,
                instance,
                adapter_only: false,
                scratch: engine::ScratchPlace::None,
            }],
            lanes: vec![device_lanes().remove(0), neighbour],
            media: Vec::new(),
            voxels: Vec::new(),
            shape: None,
        }))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    let outcome = (0..600)
        .find_map(|_| {
            rig.heard
                .recv_timeout(Duration::from_millis(100))
                .ok()
                .filter(|(at, _)| at.frame == ticket.id)
                .map(|(_, outcome)| outcome)
        })
        .expect("the step that refused a lane tells its outcome");
    let lanes: Vec<u32> = outcome.faulted.iter().map(|fault| fault.lane).collect();
    assert_eq!(
        lanes,
        vec![0],
        "the attachment's lane alone faults: {:?}",
        outcome.faulted
    );
    let why = format!("{:?}", outcome.faulted[0]);
    assert!(why.contains("describes 2"), "{why}");
    // A step whose guest pass is recorded to land later leaves its rows on
    // the device (Vulkan, two frames in flight): no readout to compare
    // there, and the one-frame run compares them.
    let readouts = &ticket.steps[0].readouts;
    if readouts.iter().all(|readout| readout.values.is_empty()) {
        assert!(
            asynchronous,
            "a step settled inside submit reads its rows out"
        );
    } else {
        let got = &readouts
            .last()
            .expect("the neighbour reads its logits")
            .values;
        let apart = got
            .iter()
            .zip(&alone[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            got.len() == alone[0].len() && apart <= 1e-3,
            "the neighbour's logits are {apart} apart from its prompt's alone ({} values \
             against {})",
            got.len(),
            alone[0].len()
        );
    }
    eprintln!("{frames} frame(s): an overreaching embed_indptr refused: {why}");
    // The refused lane wrote nothing its working set holds: the device
    // path's set still answers beam 0's first step as the host path did.
    let again = rig.fire(
        vec![Lane {
            slot: 0,
            word: masked,
            tokens: vec![TOKENS[0][0]],
            positions: vec![PROMPT],
            readout: Readout::Last,
            tables: vec![(LIST, sets[1].clone())],
            kv: KvDelta {
                held: PROMPT,
                folded: PROMPT,
                ..KvDelta::default()
            },
            mask: Some(Masking::Extent(runs(&masks[0][0]))),
            ..Lane::default()
        }],
        None,
    );
    let apart = again[0]
        .iter()
        .zip(&want[0][0])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        apart <= 1e-3,
        "after the refused lane, the device path's set answers {apart} apart: it wrote there"
    );

    engine::Programs::close(&mut rig.engine, instance).expect("the instance closes");
}

/// One frame in flight: each pass runs behind its forward inside `submit`.
#[test]
fn a_device_geometry_beam_answers_the_host_path() {
    answers_the_host_path(1);
}

/// Two in flight: a pass recorded to land later, and the next fire's ports
/// resolved once it landed.
#[test]
fn a_device_geometry_beam_staged_behind_its_pass_answers_the_host_path() {
    answers_the_host_path(2);
}
