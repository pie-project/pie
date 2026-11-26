//! One program owning eight lanes, the served text-completion bench's batch
//! (one inferlet, eight prompts): each lane's instance takes its token from
//! the cell its last pass put on the device (a decode envelope's
//! `embed_tokens`), so every step after a lane's first feeds it on the
//! device and checks the id it embedded, and each step's passes run behind
//! its forward, two frames in flight. The runtime cut the batch as it does
//! there: seven lanes in one frame, the eighth in the next, so a wide step
//! and a narrow one are in flight together. Every step settles, and every
//! lane's tokens are the host's argmax chain. Every odd round a ninth lane
//! whose token the host states steps among the wide frame's seven: what a
//! fed fire reads its ids by (a streamed table's misses) serves both.
//!
//! A wide step's landing (its seat read, then its fed rows, two maps) can
//! come behind the narrow step fired after it. The landing took the
//! narrow step from the mailbox and left it in flight, as only the front
//! of the flights was ever taken; the next harvest then waited for it with
//! nothing left to post it (wgpu, Qwen3-0.6B, eight lanes of one program,
//! the first fire after a one-lane warmup).

use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration, ChannelSeed};
use engine::fire::{Attachment, FrameSubmission, FrameTicket, Lane, Readout, Step};
use engine::program::{InstanceBinding, InstanceId};
use engine::{StepDone, StepOutcome};
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Port, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::Request;

use super::outputs::{Loaded, Rows, VOCAB, WIDTH, loaded_flight, output_decls, registration_of};
use super::slots::Slots;

/// The lanes the one program owns.
const LANES: u32 = 8;
/// The rounds the lanes decode: a frame of the first seven lanes, then a
/// frame of the eighth, each lane a token a round.
const STEPS: u32 = 24;
/// The lane the host feeds, in the wide frame of every odd round.
const HOSTED: u32 = LANES;
/// The lanes of a round's first frame.
const WIDE: u32 = 7;

/// A greedy sampler, as a served one is laid: the lane's `logits` row's
/// argmax put back into its own `embed_tokens` cell (the device's, the host
/// never reads it) and into a ring the host reads.
fn sampler() -> TraceContainer {
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(EtaDtype::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            },
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(EtaDtype::I32),
                capacity: STEPS + 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
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
                    shape: Shape::vector(2),
                    data: [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, WIDTH),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// The same sampler over a token the host states: no token cell, its one
/// channel the ring.
fn hosted() -> TraceContainer {
    let mut program = sampler();
    program.channels.remove(0);
    program.ports.clear();
    let ops = &mut program.stages[0].ops;
    ops.truncate(2);
    ops.push(Op::ChanPut { chan: 0, value: 1 });
    program
}

/// The token the host states for the hosted lane at `round`.
fn stated(round: u32) -> u32 {
    (round * 11 + 7) % VOCAB
}

/// The host's argmax of the table's row `token` (the first of equals).
fn argmax(table: &[f32], token: u32) -> u32 {
    let row = &table[(token * WIDTH) as usize..][..WIDTH as usize];
    let mut best = 0usize;
    for (at, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = at;
        }
    }
    best as u32
}

#[test]
fn one_program_driving_eight_lanes_settles_every_step() {
    let boot = crate::gpu::device_boot();
    let Some(Loaded {
        mut engine,
        caps,
        trace,
        planes,
        dir,
    }) = loaded_flight("one-owner", &Rows, LANES + 1, None, boot, 2)
    else {
        return;
    };
    let table = planes["embed"].1.clone();
    let (tx, heard) = mpsc::channel::<(StepDone, StepOutcome)>();
    let tx = Mutex::new(tx);
    engine::Completion::on_complete(
        &mut engine,
        Arc::new(move |at, outcome| {
            let _ = tx.lock().expect("the sink's sender").send((at, outcome));
        }),
    );
    let asynchronous = engine::Completion::asynchronous(&engine);

    let program = engine::Programs::register(
        &mut engine,
        &registration_of(&output_decls(&caps), sampler(), 0x0e8),
    )
    .expect("the sampler registers");
    let from_host = engine::Programs::register(
        &mut engine,
        &registration_of(&output_decls(&caps), hosted(), 0x0e9),
    )
    .expect("the hosted sampler registers");
    let chunks = &caps.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let seeds: Vec<u32> = (0..LANES).map(|lane| (lane * 5 + 3) % VOCAB).collect();
    let instances: Vec<InstanceId> = seeds
        .iter()
        .map(Some)
        .chain([None])
        .enumerate()
        .map(|(lane, seed)| {
            let made = if seed.is_some() { sampler() } else { hosted() };
            let channels: Vec<ChannelId> = (0..made.channels.len() as u64)
                .map(|at| ChannelId(((lane as u64 + 1) << 8) + at))
                .collect();
            for (at, &id) in channels.iter().enumerate() {
                let decl = &made.channels[at];
                slots.register(
                    &mut engine,
                    ChannelRegistration {
                        id,
                        shape: vec![1],
                        dtype: decl.dtype,
                        host_role: decl.host_role,
                        seeded: decl.seeded,
                        extern_dir: None,
                        capacity: decl.capacity,
                        extern_name: Vec::new(),
                        slots: Vec::new(),
                    },
                );
            }
            let binding = slots.binding(
                &engine,
                InstanceBinding {
                    program: if seed.is_some() { program } else { from_host },
                    channels: channels.clone(),
                    seeds: seed
                        .map(|&seed| ChannelSeed {
                            channel: 0,
                            bytes: (seed as i32).to_le_bytes().to_vec(),
                        })
                        .into_iter()
                        .collect(),
                    geometry: if seed.is_some() {
                        GeometryClass::DecodeEnvelope
                    } else {
                        GeometryClass::Host
                    },
                    extents: engine::program::BindExtents {
                        row_count: 1,
                        token_count: 1,
                        sampled_rows: 1,
                        query_len: 1,
                        ..engine::program::BindExtents::default()
                    },
                    device_resolved: Default::default(),
                    instance_slots: Vec::new(),
                    ring_slots: Vec::new(),
                    adapter_slots: Vec::new(),
                    scratch: engine::ScratchPlace::None,
                    pair: engine::ScratchPlace::None,
                    pair_generation: 0,
                },
                &channels,
            );
            engine::Programs::bind(&mut engine, &binding)
                .expect("a lane's sampler binds")
                .id
        })
        .collect();

    let word = trace.facts.word(&Request::new(1, false));
    let step = |lanes: &[u32], round: u32| Step {
        lanes: lanes
            .iter()
            .map(|&slot| Lane {
                slot,
                word,
                // The envelope's token: the instance's cell, not this one.
                tokens: vec![if slot == HOSTED { stated(round) } else { 0 }],
                readout: Readout::Last,
                kv_less: true,
                ..Lane::default()
            })
            .collect(),
        attachments: lanes
            .iter()
            .enumerate()
            .map(|(at, &lane)| Attachment {
                lane: at as u32,
                lanes: 1,
                instance: instances[lane as usize],
                adapter_only: false,
                scratch: engine::ScratchPlace::None,
            })
            .collect(),
        media: Vec::new(),
        voxels: Vec::new(),
        shape: None,
    };

    // The failure is a hang inside a submit or a settle, where nothing
    // returns to be asked: a watchdog names it and ends the run.
    let (finished, watched) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        if let Err(mpsc::RecvTimeoutError::Timeout) = watched.recv_timeout(Duration::from_secs(300))
        {
            eprintln!(
                "one_owner: eight lanes of one program, a frame of {WIDE} then one of {}: no \
                 step settled five minutes on; the steps in flight hung",
                LANES - WIDE
            );
            std::process::abort();
        }
    });

    // As the runtime fires them: each frame submitted while the last is in
    // flight, none settled by the host but the last; each step is told
    // through the completion sink once its forward and its passes landed. A
    // hang there is the failure, so the wait is bounded and names it.
    let mut fired = Vec::new();
    let mut last: Option<FrameTicket> = None;
    for round in 0..STEPS {
        let mut wide: Vec<u32> = (0..WIDE).collect();
        if round % 2 == 1 {
            wide.insert(3, HOSTED);
        }
        let narrow: Vec<u32> = (WIDE..LANES).collect();
        for lanes in [wide, narrow] {
            let ticket = engine
                .submit(&FrameSubmission::of(step(&lanes, round)))
                .expect("the frame fires");
            fired.push(ticket.id);
            last = Some(ticket);
        }
    }
    let mut last = last.expect("frames were fired");
    engine
        .settle_frame(&mut last)
        .expect("the last frame settles");
    let mut seen: Vec<(StepDone, StepOutcome)> = heard.try_iter().collect();
    for id in &fired {
        while asynchronous && !seen.iter().any(|(at, _)| at.frame == *id) {
            let told = heard
                .recv_timeout(Duration::from_secs(120))
                .unwrap_or_else(|_| {
                    panic!("frame {id:?}: no completion two minutes on; the step hung")
                });
            seen.push(told);
        }
    }
    for (at, outcome) in &seen {
        assert!(
            outcome.faulted.is_empty(),
            "{at:?}: lanes faulted: {:?}",
            outcome.faulted
        );
    }
    drop(finished);

    for (lane, (&instance, &seed)) in instances.iter().zip(&seeds).enumerate() {
        let mut want = seed;
        for at in 0..STEPS {
            want = argmax(&table, want);
            let cell = engine::Channels::take(&mut engine, instance, 1)
                .expect("the token ring reads")
                .unwrap_or_else(|| panic!("lane {lane}: no token at step {at}"));
            let got = i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"));
            assert_eq!(got, want as i32, "lane {lane}, step {at}: the argmax chain");
        }
    }
    for round in (1..STEPS).step_by(2) {
        let want = argmax(&table, stated(round));
        let cell = engine::Channels::take(&mut engine, instances[HOSTED as usize], 0)
            .expect("the token ring reads")
            .unwrap_or_else(|| panic!("the hosted lane: no token at round {round}"));
        let got = i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"));
        assert_eq!(got, want as i32, "the hosted lane, round {round}");
    }
    eprintln!(
        "{STEPS} rounds of {LANES} lanes of one program, {WIDE} a frame and then the rest, \
         settled, every token the host's"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
