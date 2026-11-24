#![cfg(target_vendor = "apple")]

//! A capturing lane's score rows are its reader instance's state: a key row
//! of the load's context and its page row, held in the reader's `Instance`
//! slots after its tables. A score reader registered and bound after ready
//! (the load's walk done and sealed), two scored decodes, a park of its
//! slots to fresh refs elsewhere (the old ones blanked) and an unpark, two
//! more: each decode's rows and logits are a never-parked session's, the
//! rows lie in the instance's slots wherever they moved, and no fire
//! builds a pipeline after ready (the reader's own are built at its
//! registration). A capturing lane no program is attached at, and an
//! instance attached at more than one lane, are refused by name.

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated};
use engine_metal::{Lane, Shell};
use poem::Request;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};

const MODEL: &str = "qwen3-micro-text";
/// The load's context: a capturing lane's key row's width.
const CONTEXT: u32 = 256;
const PROMPT: u32 = 20;
const DECODES: u32 = 4;
/// The decode before which the parked session parks and unparks.
const PARK_AT: u32 = 2;

/// An epilogue that sums lane 0's `attn.scores` row into a reader channel.
pub(super) fn reader(context: u32) -> ProgramRegistration {
    let scores = OutputDecl {
        name: "attn.scores".into(),
        axis: Axis::Lane,
        width: context,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: false,
    };
    let container = TraceContainer {
        names: vec!["attn.scores".into()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            // A cell each fire: the test reads the rows, not the ring.
            capacity: 8,
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
                    shape: Shape::matrix(1, context),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceSum(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let bound = eta_ir::validate::bind(
        container,
        vec![vec![scores]],
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the reader binds");
    let stages = compile_bound(&bound);
    let backend = Backend::parse("metal").expect("the metal codegen backend");
    ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    }
}

/// A score reader of the load's context bound for a lane of `rows` rows
/// (a prompt, or a decode of one), its tables in `Instance` chunk `chunk`
/// and its ring in `Channel` chunk `chunk + 2`: the instance and the
/// slots it holds.
pub(super) fn bind_reader(
    shell: &mut Shell,
    program: engine::ProgramId,
    rows: u32,
    chunk: u32,
) -> (engine::InstanceId, Vec<u32>, Vec<engine::RingSlots>) {
    bind_reader_of_pair(shell, program, rows, chunk, 0)
}

/// [`bind_reader`] as an instance of the `pair`-th pair the test binds: a
/// reader at other rows than another's is another pair, its state laid
/// apart (`crate::guest_places_of_pair`).
pub(super) fn bind_reader_of_pair(
    shell: &mut Shell,
    program: engine::ProgramId,
    rows: u32,
    chunk: u32,
    pair: u32,
) -> (engine::InstanceId, Vec<u32>, Vec<engine::RingSlots>) {
    let facts = shell.program_facts(program).expect("the reader's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    // Counted from the first chunk no slot holds: past the engine's tables
    // run, which no slot names, and the seats' runs.
    let chunk = layout.free_chunk() + chunk - 1;
    let first = chunk * instance.per_chunk;
    let slots: Vec<u32> = (first..first + facts.instance.slots).collect();
    let mut next = (chunk + 2) * channel.per_chunk;
    let rings: Vec<engine::RingSlots> = facts
        .rings
        .iter()
        .map(|ring| {
            let slots = (next..next + ring.footprint.slots).collect();
            next += ring.footprint.slots;
            engine::RingSlots {
                channel: ring.channel,
                slots,
            }
        })
        .collect();
    let extents = eta_exec::Extents {
        row_count: rows,
        token_count: rows,
        sampled_rows: 1,
        query_len: rows,
        ..eta_exec::Extents::default()
    };
    let (scratch, pair) = crate::guest_places_of_pair(shell, program, &extents, chunk, pair);
    let id = shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &[engine::ChannelId(0x5c0f_0000 + u64::from(chunk))],
            &slots,
            &rings,
            &scratch,
            &pair,
            0,
        )
        .expect("the reader binds");
    (id, slots, rings)
}

/// The reader `id` attached at lane `lane`.
pub(super) fn at_lane(lane: u32, id: engine::InstanceId) -> Attached {
    Attached {
        lane,
        lanes: 1,
        instance: id,
        adapter_only: false,
    }
}

/// A scored session's decodes: each one's key row, page row and logits.
type Decodes = Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>;

/// The load, walked and sealed: ready.
fn ready(fixture: &Fixture) -> Shell {
    let mut shell = load_with(fixture, CONTEXT);
    shell.warm_pipelines().expect("the walk runs");
    shell.seal_pipelines().expect("the fixed pipelines build");
    shell
}

fn session(fixture: &Fixture, park: bool) -> Decodes {
    let mut shell = ready(fixture);
    let trace = fixture.trace.clone();
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % 256).collect();
    shell.open(0).expect("the slot opens");
    shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(0),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&trace, PROMPT),
                tokens: &prompt,
            })
        }])
        .expect("the prompt prefills");

    // Registered and bound after ready.
    let program = shell
        .register_program(&reader(CONTEXT))
        .expect("the reader registers");
    let compiled = shell.compiled();
    let facts = shell.program_facts(program).expect("the reader's facts");
    let instance = shell.pool_layout().instance.expect("an instance shape");
    let row_bytes =
        (u64::from(CONTEXT) + u64::from(CONTEXT.div_ceil(shell.paging().page_size))) * 4;
    assert!(
        facts.instance.bytes >= row_bytes,
        "the reader's instance holds its score rows: {} bytes, the rows {row_bytes}",
        facts.instance.bytes
    );
    let (id, slots, rings) = bind_reader(&mut shell, program, 1, 1);
    let attached = [at_lane(0, id)];

    let scored = trace
        .facts
        .word(&Request::new(1, false).capturing_scores(true));
    let mut decodes = Decodes::new();
    for step in 0..DECODES {
        if park && step == PARK_AT {
            // Parked and unparked: its slots copied to fresh refs past
            // them, the old ones blanked from a slot nothing wrote (the
            // chunk's last), the instance rebound there.
            let fresh: Vec<u32> = slots.iter().map(|&s| s + slots.len() as u32).collect();
            let blank = (shell.pool_layout().free_chunk() + 1) * instance.per_chunk - 1;
            assert!(
                fresh.last() < Some(&blank),
                "the fresh run and the blank slot lie apart"
            );
            let copy = |pairs: Vec<(u32, u32)>| engine::chunk::SlotCopy {
                shape: engine_metal::store::layout::INSTANCE,
                src: engine::MemoryDomain::Device { ordinal: 0, reach: engine::DeviceReach::HostShared },
                dst: engine::MemoryDomain::Device { ordinal: 0, reach: engine::DeviceReach::HostShared },
                slots: pairs,
                runs: Vec::new(),
            };
            shell
                .copy_slots(&copy(
                    slots.iter().copied().zip(fresh.iter().copied()).collect(),
                ))
                .expect("the slots park");
            shell
                .copy_slots(&copy(slots.iter().map(|&s| (blank, s)).collect()))
                .expect("the old slots blank");
            shell
                .rebind_slots(&engine::Rebind::Instance {
                    id,
                    instance: fresh,
                    rings: rings.clone(),
                    adapter: Vec::new(),
                    scratch: engine::ScratchPlace::None,
                    pair: engine::ScratchPlace::None,
                    pair_generation: 0,
                })
                .expect("the reader rebinds");
        }
        let token = [(step * 13 + 5) % 256];
        let logits = shell
            .fire_attached(
                &[Seated {
                    replay_stated: None,
                    held: Some(PROMPT + step),
                    ..Seated::of(Lane {
                        slot: 0,
                        word: scored,
                        tokens: &token,
                    })
                }],
                &attached,
            )
            .unwrap_or_else(|why| panic!("scored decode {step} fires: {why}"));
        let rows = shell.score_rows().expect("the rows read back");
        assert_eq!(rows.len(), 1, "one capturing lane, one reader's rows");
        let (source, keys, pages) = rows.into_iter().next().unwrap();
        assert_eq!(source, 0);
        assert_eq!(
            keys.len(),
            CONTEXT as usize,
            "a key row of the load's context"
        );
        decodes.push((keys, pages, logits.into_iter().next().unwrap_or_default()));
    }
    assert_eq!(
        shell.take_unbuilt_pipelines(),
        0,
        "no fire was refused an unbuilt pipeline"
    );
    assert_eq!(
        shell.compiled(),
        compiled,
        "a scored fire after ready built a pipeline (built after ready)"
    );

    // A capturing lane no program is attached at is refused by name (the
    // runtime stamps the capture only on the fire that runs the program,
    // and Step::validate_captures refuses the rest at submit), and so is
    // an instance attached at more lanes than its one lane's rows.
    let refused = shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(PROMPT + DECODES),
            ..Seated::of(Lane {
                slot: 0,
                word: scored,
                tokens: &[7],
            })
        }])
        .expect_err("a capturing lane with no reader is refused");
    assert!(
        refused.to_string().contains("no reader is attached"),
        "refused by name: {refused}"
    );
    let refused = shell
        .fire_attached(
            &[Seated {
                replay_stated: None,
                held: Some(PROMPT + DECODES),
                ..Seated::of(Lane {
                    slot: 0,
                    word: scored,
                    tokens: &[9],
                })
            }],
            &[Attached {
                lanes: 2,
                ..at_lane(0, id)
            }],
        )
        .expect_err("an instance attached at two lanes holds one lane's rows");
    assert!(
        refused
            .to_string()
            .contains("an instance holds one lane's score rows"),
        "refused by name: {refused}"
    );
    shell.close_program_instance(id).expect("the reader closes");
    decodes
}

#[test]
fn a_score_reader_holds_its_rows_through_a_park() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of(MODEL, None);
    let walked = session(&fixture, false);
    let parked = session(&fixture, true);
    for (step, ((keys, pages, logits), (want_keys, want_pages, want_logits))) in
        parked.iter().zip(&walked).enumerate()
    {
        let mass: f32 = keys.iter().sum();
        assert!(
            mass > 0.5,
            "decode {step}'s key row holds the layers' mass: {mass}"
        );
        assert_eq!(
            keys, want_keys,
            "decode {step}'s key row is the never-parked one's"
        );
        assert_eq!(
            pages, want_pages,
            "decode {step}'s page row is the never-parked one's"
        );
        assert_eq!(
            logits, want_logits,
            "decode {step}'s logits are the never-parked one's"
        );
    }
}
