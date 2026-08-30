//! **THE INTER-FIRE CARRY, DEVICE-SIDE, WITH THE HOST INTERPRETER AS THE
//! GOLDEN** (alto design §5, articles 3 and 4; wave F2a).
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test a_carried_cell_crosses_two_fires_without_touching_the_host -- --nocapture
//! ```
//!
//! # What this is for
//!
//! `program_parity` proves the device half computes what the host interpreter
//! computes, fire for fire, over the whole corpus. What it cannot see is the
//! thing F2a changed underneath it: **where a cell lives between two fires**.
//! Until this wave the answer was "wherever the host put it" — a fire
//! synchronised, the host read a four-byte commit word back off the device,
//! advanced its own cursors, and the runtime pumped cells across the contract
//! with an H2D per cell one way and a `Vec` per cell the other (survey §7,
//! invariants I1/I2/I3/I5). Now the answer is "in the device slab, addressed
//! by a prediction the host counted and a kernel checked".
//!
//! The subject is the smallest program with a real inter-fire dependency:
//!
//! ```text
//!   prologue   read(link) ─────────────▶ put(seen)      what fire N-1 left
//!   epilogue   take(link) ─ +1 ─┬──────▶ put(link)      what fire N leaves
//!                               └──────▶ put(out)
//! ```
//!
//! `link` is declared [`HostRole::None`]. Fire N's epilogue publishes into it
//! and fire N+1's PROLOGUE reads the very cell — so the value that makes each
//! fire's answer depend on the last one crosses the fire boundary entirely on
//! the device.
//!
//! # The three claims
//!
//! **(a) Byte-identical to the host-pump path.** `eta_exec::step` is
//! the same interpreter `program_parity` diffs against and it is the OLD
//! path's semantics exactly: host-owned cursors, host-owned rings, a commit
//! the host decides. Both halves run the same program over the same fires and
//! every cell of every ring, plus both cursors, must agree — so "the device
//! commit path did not change any answer" is asserted rather than assumed.
//!
//! **(b) Zero host copies of the carried cell.** Asserted by construction and
//! by API absence, which is the only honest way to assert an absence:
//!
//! ```text
//! * `link` is `HostRole::None`, so `Session::bind` opens NO endpoint for it
//!   — no pinned mirror and no pinned counters exist for anyone to read or
//!   write, and `Rings::read_cell`/`write_cell` fall through to the device
//!   slab (`program/launch.rs`)
//! * the runtime's pump filters on `HostRole::Writer` / `HostRole::Reader`
//!   (`runtime::engine::channel`), so a private channel is not a channel it
//!   has a door for
//! * `Session::publish` / `take` — the two doors `Engine::publish_channel`
//!   and `take_channel` reach — are never called on `link` in this test, and
//!   `Plane` exposes no other way to move its bytes
//! ```
//!
//! What the fire path DOES touch of `link` is its ticket-free slot in the
//! bump's `taken`/`put` lists, which is device state moved by a device
//! kernel. The one host read of a `link` cell in this file is the final
//! `peek`, which exists to state the carry's value and is not on any fire's
//! path.
//!
//! **(c) A wrong prediction dummy-runs, and leaves nothing.** The property
//! the whole protocol rests on is invisible while the host and the device
//! agree, so [`Session::skew_prediction`] makes them disagree on purpose:
//! one fire states a tail its ring is not at, `channel::pull_validate` clears
//! the commit word, every stage still launches and every stage's kernel
//! early-returns on that word, `channel::commit_bump` moves nothing and
//! `channel::scatter_publish` publishes nothing. Afterwards every ring,
//! every cursor and every guest-visible cell is exactly what it was — and the
//! refusal is LOUD, because the admission check had already passed and
//! article 4 says a surviving refusal is a contract violation rather than a
//! retry.

#![cfg(feature = "_cuda")]

use std::collections::BTreeMap;

use engine_cuda::device::{Context, present};
use engine_cuda::program::{Disk, Fired, Plane};
use eta_compiler::codegen::launch::LaunchChannel;
use eta_exec::{
    Boundaries, ExecPlan, HostOp, InterpInstance, PassInputs, StepOutcome, Value,
    adopt_launch_package_with, concrete_dtype, encode_wire, host_take, make_host_instance, step,
    wire_cell_bytes,
};
use eta_ir::container::HostRole;
use eta_ir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype, Literal, Shape};

/// The channels, in the package's declaration order.
const LINK: u32 = 0;
const SEEN: u32 = 1;
const OUT: u32 = 2;

/// What the ring starts holding, and therefore what fire 1's prologue reads.
const SEED: i32 = 41;

/// How many fires the carry is followed over. Four is past the point where
/// `link`'s capacity-one ring has wrapped several times, which is what a
/// prediction that drifted by one would show up in.
const FIRES: usize = 4;

/// The fire whose prediction is deliberately wrong.
const MISPREDICT_AT: usize = 2;

// ─────────────────────────────────────────────────────────────────────────
// The subject
// ─────────────────────────────────────────────────────────────────────────

/// The carry program. See the module header for the shape.
fn carry() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![
            // LINK — the inter-fire carry. `HostRole::None` is the whole
            // point: no host end, no pinned mirror, no door.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            },
            // SEEN — what the PROLOGUE read out of `link`, i.e. what the
            // previous fire's epilogue left there.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 8,
                host_role: HostRole::Reader,
                seeded: false,
            },
            // OUT — what THIS fire's epilogue put into `link`.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 8,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![
            StageProgram {
                stage: Stage::Prologue,
                // A READ and not a take: the head does not move, so the
                // epilogue below still finds the cell to consume. This is the
                // "fire N+1 takes what fire N's epilogue put" edge.
                ops: vec![Op::ChanRead(LINK), Op::ChanPut { chan: SEEN, value: 0 }],
            },
            StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(LINK),
                    Op::Const(Literal::I32(1)),
                    Op::Add(0, 1),
                    Op::ChanPut {
                        chan: LINK,
                        value: 2,
                    },
                    Op::ChanPut {
                        chan: OUT,
                        value: 2,
                    },
                ],
            },
        ],
    }
}

/// The subject, all the way to what `Plane::register` takes.
fn registration() -> engine::program::ProgramRegistration {
    let bound = eta_ir::validate::bind(carry(), ModelProfile::dummy())
        .unwrap_or_else(|why| panic!("the carry program does not bind: {why:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = eta_compiler::codegen::program::Backend::Cuda;
    let emitted = eta_compiler::codegen::program::emit_program(backend, &stages, &bound);
    for kernel in &emitted {
        assert!(
            kernel.error.is_empty(),
            "the carry program did not emit: {}",
            kernel.error
        );
    }
    engine::program::ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted,
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

fn numel(declared: &LaunchChannel) -> usize {
    declared
        .shape
        .iter()
        .map(|&dim| dim as usize)
        .product::<usize>()
        .max(1)
}

/// One I32 cell, as both halves spell it on the wire.
fn wire(value: i32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

// ─────────────────────────────────────────────────────────────────────────
// The gate
// ─────────────────────────────────────────────────────────────────────────

/// Skip at RUN time, saying what was missing — an `#[ignore]`d test on the
/// one box that could run it is a test nobody runs.
fn device_or_skip() -> Option<Context> {
    if !present() {
        println!("skipped: no CUDA runtime");
        return None;
    }
    match Context::bind(0) {
        Ok(context) => Some(context),
        Err(why) => {
            println!("skipped: no device — {why}");
            None
        }
    }
}

#[test]
fn a_carried_cell_crosses_two_fires_without_touching_the_host() {
    let Some(context) = device_or_skip() else {
        return;
    };
    let registration = registration();
    let package = registration.launch.clone();
    let plan: ExecPlan = adopt_launch_package_with(package.clone(), Boundaries::CUDA)
        .expect("the carry program adopts");

    assert_eq!(
        package.channels[LINK as usize].host_role,
        HostRole::None,
        "(b) the carried channel has NO host end — this is the claim the rest \
         of the test rests on, and it is a property of the declaration rather \
         than of anything the fire does",
    );

    // ── The golden: the host interpreter, seeded the same way. This IS the
    //    pre-F2a path's semantics — host-owned rings, host-owned cursors, a
    //    commit the host decides — so agreeing with it is claim (a).
    let seed = wire(SEED);
    let host_seed: BTreeMap<u32, Value> = BTreeMap::from([(LINK, Value::I32(vec![SEED]))]);
    let mut interp: InterpInstance = make_host_instance(&plan, &BTreeMap::new(), &host_seed);

    let scratch = std::env::temp_dir().join(format!("pie-carry-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let mut plane = Plane::new(Disk::at(scratch));
    let program = plane
        .register(&context, &registration)
        .expect("the carry program compiles");
    let instance = plane
        .bind(
            program,
            &[(LINK, seed.clone())],
            eta_exec::Extents::default(),
            GeometryClass::Host,
            &[],
            &[],
        )
        .expect("the carry program binds");

    let inputs = PassInputs::none();
    let mut carried: Vec<i32> = Vec::new();

    for round in 0..FIRES {
        // ── (c) THE DELIBERATELY WRONG PREDICTION, on the round the header
        //    names. `OUT`'s tail is the ENGINE's counter, so shifting the
        //    prediction by one states a tail the guest's pinned word is not
        //    at — which is exactly the staleness `pull_validate` exists to
        //    catch, and exactly what fire N+1 would suffer under run-ahead if
        //    fire N had been refused.
        if round == MISPREDICT_AT {
            let before = ring_snapshot(&plane, instance, &package);
            plane
                .instance_mut(instance)
                .expect("bound")
                .skew_prediction(OUT, 0, 1);
            let refusal = plane
                .fire(&context, instance)
                .expect_err("a prediction the ring denies is a contract violation, not an outcome");
            let text = format!("{refusal}");
            assert!(
                text.contains("not where this fire predicted"),
                "(c) the refusal says what happened: {text}"
            );
            assert!(
                text.contains("article 4") || text.contains("contract violation"),
                "(c) and why it is loud rather than a retry: {text}"
            );
            // Put the prediction back where the ring actually is, then assert
            // the dummy run left NOTHING: every cell of every ring, both
            // cursors of every channel.
            plane
                .instance_mut(instance)
                .expect("bound")
                .skew_prediction(OUT, 0, -1);
            assert_eq!(
                ring_snapshot(&plane, instance, &package),
                before,
                "(c) the refused fire launched every stage and published none of \
                 it: no cell, no cursor and no full byte may have moved",
            );
            // And the host half must not have taken this fire either — it is
            // a fire that did not happen on both sides.
            continue;
        }

        let expected = step(&mut interp, &plan, &inputs);
        assert_eq!(
            expected,
            StepOutcome::Committed,
            "round {round}: the golden commits, so the device must too"
        );
        let fired = plane
            .fire(&context, instance)
            .unwrap_or_else(|why| panic!("round {round}: {why}"));
        assert_eq!(
            fired,
            Fired::Committed,
            "round {round}: the device half's outcome is the golden's"
        );

        // ── (a) EVERY RING, SLOT FOR SLOT, PLUS BOTH CURSORS. Comparing only
        //    what was drained would miss a program that wrote the right value
        //    into the wrong slot.
        compare_rings(&plane, instance, &interp, &package, round);

        // ── Drain both readers, and compare what came out.
        for channel in [SEEN, OUT] {
            let declared = &package.channels[channel as usize];
            let dtype = concrete_dtype(declared.dtype);
            let lanes = numel(declared);
            let (op, value) = host_take(&interp, &plan, channel);
            let device = plane
                .instance_mut(instance)
                .expect("bound")
                .take(channel)
                .unwrap_or_else(|why| panic!("round {round}: taking channel {channel}: {why}"));
            match (op, value, device) {
                (HostOp::Ok, Some(value), Some(bytes)) => {
                    let mut golden = vec![0u8; wire_cell_bytes(dtype, lanes)];
                    encode_wire(&value, &mut golden);
                    assert_eq!(
                        golden, bytes,
                        "round {round}: channel {channel} published different bytes"
                    );
                    if channel == SEEN {
                        carried.push(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                    }
                }
                (op, value, device) => panic!(
                    "round {round}: channel {channel} — host {op:?}/{}, device {}",
                    value.is_some(),
                    device.is_some()
                ),
            }
        }
    }

    // ── (b) THE CARRY ITSELF. Each fire's prologue read what the PREVIOUS
    //    fire's epilogue put, and neither end of that edge is a host copy:
    //    the cell was written by an emitted kernel into the device slab and
    //    read by an emitted kernel out of it, one fire apart, with a
    //    `channel::commit_bump` in between and nothing else.
    //
    //    The refused round is skipped above and contributes no value, which
    //    is itself part of the claim: a dummy run leaves the carry exactly
    //    where it found it, so the chain continues rather than restarts.
    let mut want = SEED;
    for (index, seen) in carried.iter().enumerate() {
        assert_eq!(
            *seen, want,
            "the carry broke at observation {index}: the prologue read {seen} \
             where the previous epilogue left {want}"
        );
        want += 1;
    }
    assert_eq!(
        carried.len(),
        FIRES - 1,
        "one round was refused on purpose and published nothing"
    );

    // The final resting value, read once, off the fire path — stated so that
    // the carry's arithmetic is pinned by a number and not only by a chain.
    let resting = plane
        .instance(instance)
        .expect("bound")
        .peek(LINK, plane.instance(instance).expect("bound").cursor(LINK).expect("carried").head)
        .expect("the link ring's committed cell");
    assert_eq!(
        resting,
        wire(SEED + (FIRES as i32 - 1)),
        "the device slab holds the carry the fires left there"
    );

    plane.close_instance(instance).expect("closes once");
}

/// Every ring's cells and cursors, as the parity comparison and the
/// dummy-run comparison both need them.
fn ring_snapshot(
    plane: &Plane,
    instance: u64,
    package: &eta_compiler::codegen::launch::LaunchPackage,
) -> Vec<(u64, u64, Vec<Vec<u8>>)> {
    let session = plane.instance(instance).expect("bound");
    package
        .channels
        .iter()
        .enumerate()
        .map(|(index, declared)| {
            let channel = index as u32;
            let cursor = session.cursor(channel).expect("carried");
            let slots = u64::from(declared.capacity.max(1)) + 1;
            let cells = (0..slots)
                .map(|slot| session.peek(channel, slot).expect("peeks"))
                .collect();
            (cursor.head, cursor.tail, cells)
        })
        .collect()
}

/// Both halves' rings, slot for slot, plus both cursors.
fn compare_rings(
    plane: &Plane,
    instance: u64,
    interp: &InterpInstance,
    package: &eta_compiler::codegen::launch::LaunchPackage,
    round: usize,
) {
    let session = plane.instance(instance).expect("bound");
    for (index, declared) in package.channels.iter().enumerate() {
        let channel = index as u32;
        let dtype = concrete_dtype(declared.dtype);
        let lanes = numel(declared);
        let ring = &interp.channels[index];
        let cursor = session.cursor(channel).expect("carried");

        assert_eq!(
            ring.head(),
            cursor.head,
            "round {round}: channel {channel}'s heads diverged"
        );
        assert_eq!(
            ring.tail(),
            cursor.tail,
            "round {round}: channel {channel}'s tails diverged"
        );
        for slot in 0..u64::from(declared.capacity.max(1)) + 1 {
            let mut host = vec![0u8; wire_cell_bytes(dtype, lanes)];
            encode_wire(&ring.decode_sequence(slot), &mut host);
            let device = session.peek(channel, slot).expect("peeks");
            assert_eq!(
                host, device,
                "round {round}: channel {channel} slot {slot} — the host holds \
                 {host:02x?} and the device holds {device:02x?}"
            );
        }
    }
}
