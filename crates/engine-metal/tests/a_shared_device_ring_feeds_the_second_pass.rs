//! **A DEVICE-ONLY RING TWO PASSES SHARE**, which is design §5's
//! *"draft→verify chaining is free: a device-only private ring shared by ≤8
//! attachments, ordered by the pipeline FIFO"* — the Metal twin of
//! `tests/gpu/tests/cuda_shared_device_ring.rs` (`27de300fa`).
//!
//! # What was broken
//!
//! A channel declared [`HostRole::None`] had its ring cut inside whichever
//! [`Session`](engine_metal::program::Session) bound it — `Rings::allocate`
//! carves one slab per instance and the cursors lived in that instance's own
//! `Vec<Cursor>` — and `api::register_channel` was `Unsupported` by design.
//! That is right for a ring ONE pass owns and silently wrong for the shape the
//! design names by hand: the prefill epilogue's put landed in one instance's
//! copy and the decode pass's `embed` read the other's, forever empty.
//! `text-completion-bench` died at its first decode frame, every request, in
//! about thirty milliseconds, with
//!
//! ```text
//! channel 0 is empty and its program takes from it (needs a cell, holds 0 of 1)
//! ```
//!
//! — a true sentence about a ring nobody had written.
//!
//! # The four claims
//!
//! 1. **THE VALUE CROSSES.** One instance's epilogue puts a cell the host
//!    handed it; a second instance's epilogue, in a later fire, takes that
//!    cell and publishes it where the host can read it. The number that comes
//!    out is the number that went in — and before the putter has fired, the
//!    taker is `Blocked` on the ring rather than reading a stale cell, which
//!    is the same gate that used to refuse forever.
//! 2. **THE DENSE SLOT IS NOT THE RING.** The two programs declare the shared
//!    channel at DIFFERENT dense slots — 1 in the putter, 0 in the taker.
//!    `InstanceBinding::channels` is where a global id and a dense slot meet,
//!    so the slot numbers never have to agree.
//! 3. **EIGHT SEATS, AND THE NINTH IS A REFUSAL BY NAME** — the design's
//!    bound, and a seat given back at close.
//! 4. **THE COHORT IS WHAT THE FENCE WIDENS TO.** On this plane the consumer
//!    resolves its descriptor ports on the HOST, at `serve::stage`, so a
//!    shared ring needs `serve::fence_instances` to land the PRODUCER's
//!    airborne flight and not only its own. That widening reads
//!    [`Plane::cohort`], and this pins it: the taker's cohort is the putter,
//!    and an instance that shares nothing has none.
//!
//! Claim 1 is the one that would have caught the bug: before the fix the take
//! finds an empty ring and the fire is refused forever, so this gate fails at
//! the second fire rather than on a comparison.
//!
//! **NO CHECKPOINT AND NO MODEL.** `Plane` needs a device and nothing else
//! (its own header says so), which is what lets four claims about a ring be
//! made without a forward pass. What that costs is the fence itself: the
//! attached path — `stage_into`, a model fire's command buffer,
//! `settle_launched` from the harvest — needs a load, so claim 4 gates the
//! SET the fence computes and `serve_smoke` plus the bench arm gate the wait.
//!
//! ```text
//! cargo test -p engine-metal --test a_shared_device_ring_feeds_the_second_pass \
//!   -- --nocapture
//! ```
//!
//! [`HostRole::None`]: eta_ir::container::HostRole::None
//! [`Plane::cohort`]: engine_metal::program::Plane::cohort

#![cfg(target_vendor = "apple")]

use std::sync::{Mutex, MutexGuard, PoisonError};

use engine::program::ProgramRegistration;
use engine_metal::device::{Context, present};
use engine_metal::program::{ChannelShape, Fired, MAX_ATTACHMENTS, Plane};
use eta_exec::Extents;
use eta_ir::Dtype;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::Shape;

/// The cell the putter is handed and the taker must produce. Arbitrary, and
/// arbitrary is the point: it is not derivable from anything either program
/// computes, so a ring that produced it produced it because the bytes crossed.
const CROSSES: i32 = 0x0BAD_F00D_u32 as i32;

/// Channel ids in the engine's own numbering. The gate mints them because it
/// drives `Plane::register_channel` directly, which is the runtime's job in a
/// deployment.
const IN_A: u64 = 101;
const SHARED: u64 = 102;
const OUT_B: u64 = 103;
/// A second shared ring, for the eight-seat claim alone.
const CROWDED: u64 = 104;

/// One device at a time, for the reason every other Metal gate in this
/// directory states it: the tests in one binary run on threads and a device is
/// not a per-test resource.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device_or_skip(what: &str) -> Option<Context> {
    if !present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return None;
    }
    match Context::bind() {
        Ok(context) => Some(context),
        Err(error) => {
            println!("SKIP {what}: the device does not bind ({error})");
            None
        }
    }
}

/// One `[1]`-shaped `i32` channel, which is every channel in this gate — the
/// bench guest's `tok_in` is exactly this shape and so is its `g0`.
fn scalar(host_role: HostRole, capacity: u32) -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(Dtype::I32),
        capacity,
        host_role,
        seeded: false,
    }
}

/// What [`Plane::register_channel`] is told the shared ring's geometry is —
/// the same declaration both programs carry, since a ring cut at one width and
/// addressed at another is a wrong cell and never a fault.
fn shared_shape() -> ChannelShape {
    ChannelShape {
        capacity: 1,
        numel: 1,
        dtype: Dtype::I32,
    }
}

/// **THE PUTTER**: take the host's cell, put it into the shared ring.
///
/// The shared channel is dense slot **1** here and dense slot **0** in the
/// taker, which is claim 2: one ring, two dense numberings, and nothing
/// downstream is allowed to care.
fn putter() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![
            // 0: the host's cell. `Writer` plus a take is what makes this pass
            // wait for the host to publish.
            scalar(HostRole::Writer, 1),
            // 1: THE SHARED RING. No host end at either side — the cell never
            // leaves the device between here and the taker.
            scalar(HostRole::None, 1),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
        }],
    }
}

/// **THE TAKER**: take the shared ring's cell, publish it where the host reads.
fn taker() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![
            // 0: THE SAME RING, at a different dense slot.
            scalar(HostRole::None, 1),
            // 1: what the host comes for.
            scalar(HostRole::Reader, 8),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
        }],
    }
}

/// A container, planned and emitted for this backend — the four lines every
/// gate in this directory spells for itself.
fn registration(container: TraceContainer) -> ProgramRegistration {
    let bound = eta_ir::validate::bind(container, ModelProfile::dummy())
        .unwrap_or_else(|why| panic!("the subject does not bind: {why:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = eta_compiler::codegen::program::Backend::Metal;
    let emitted = eta_compiler::codegen::program::emit_program(backend, &stages, &bound);
    ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted,
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

fn cell(value: i32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn decode(bytes: &[u8]) -> i32 {
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// **CLAIM 1 AND CLAIM 2**: the cell crosses two instances that number the
/// same ring differently, and the taker is refused rather than served
/// something stale before it has been filled.
#[test]
fn a_shared_device_ring_feeds_the_second_pass() {
    let _guard = serialized();
    let Some(context) = device_or_skip("a shared device ring") else {
        return;
    };

    let mut plane = Plane::new();
    plane
        .register_channel(&context, SHARED, shared_shape())
        .expect("the shared ring is cut once, before any bind");

    let put_program = plane
        .register(&context, &registration(putter()))
        .expect("the putter compiles on this device");
    let take_program = plane
        .register(&context, &registration(taker()))
        .expect("the taker compiles on this device");

    // The two bindings, and the whole of claim 2: `SHARED` is at dense 1 in
    // one list and dense 0 in the other.
    let putting = plane
        .bind(
            &context,
            put_program,
            &[],
            Extents::default(),
            GeometryClass::Host,
            &[IN_A, SHARED],
        )
        .expect("the putter binds");
    let taking = plane
        .bind(
            &context,
            take_program,
            &[],
            Extents::default(),
            GeometryClass::Host,
            &[SHARED, OUT_B],
        )
        .expect("the taker binds");

    // **BEFORE ANYTHING IS PUT, THE TAKER IS BLOCKED ON THE RING** — the same
    // refusal the broken plane gave forever. It is here so that a fix which
    // merely made the taker read *something* would fail rather than pass.
    assert_eq!(
        plane.fire(&context, taking).expect("a blocked fire is not an error"),
        Fired::Blocked(0),
        "an empty shared ring refuses the taker by the slot the TAKER numbers it at"
    );

    plane
        .instance_mut(putting)
        .expect("bound")
        .publish(0, &cell(CROSSES))
        .expect("the host's cell fits");
    assert_eq!(
        plane.fire(&context, putting).expect("the putter fires"),
        Fired::Committed,
        "the putter takes the host's cell and puts it into the shared ring"
    );

    // **THE PUT IS VISIBLE THROUGH THE OTHER INSTANCE**, which is the whole
    // claim: one ring, two sessions, one depth.
    assert_eq!(
        plane.instance(taking).expect("bound").depth(0),
        1,
        "the taker's own view of the ring holds the putter's cell"
    );

    assert_eq!(
        plane.fire(&context, taking).expect("the taker fires"),
        Fired::Committed,
        "the taker's ring is full now, so its readiness admits it"
    );
    let landed = plane
        .instance_mut(taking)
        .expect("bound")
        .take(1)
        .expect("taking the taker's host channel")
        .expect("the taker published a cell");
    assert_eq!(
        decode(&landed),
        CROSSES,
        "the number that came out is the number that went in"
    );

    // And the ring is empty again, both counters having moved: the take
    // advanced the head the putter's tail had run ahead of.
    assert_eq!(plane.instance(putting).expect("bound").depth(1), 0);
    assert_eq!(plane.instance(taking).expect("bound").depth(0), 0);

    plane.close_instance(putting).expect("the putter closes");
    plane.close_instance(taking).expect("the taker closes");
}

/// **CLAIM 3**: eight attachments are a bound and the ninth is a refusal that
/// names it — and a seat given back is a seat available again.
#[test]
fn a_shared_ring_seats_eight_attachments_and_refuses_the_ninth() {
    let _guard = serialized();
    let Some(context) = device_or_skip("a shared ring's eight seats") else {
        return;
    };

    let mut plane = Plane::new();
    plane
        .register_channel(&context, CROWDED, shared_shape())
        .expect("the ring is cut");
    let program = plane
        .register(&context, &registration(taker()))
        .expect("the taker compiles on this device");

    let mut seated = Vec::new();
    for seat in 1..=MAX_ATTACHMENTS {
        seated.push(
            plane
                .bind(
                    &context,
                    program,
                    &[],
                    Extents::default(),
                    GeometryClass::Host,
                    &[CROWDED, OUT_B],
                )
                .unwrap_or_else(|why| panic!("seat {seat} is inside the bound: {why}")),
        );
    }
    let ninth = plane.bind(
        &context,
        program,
        &[],
        Extents::default(),
        GeometryClass::Host,
        &[CROWDED, OUT_B],
    );
    let why = format!(
        "{}",
        ninth.expect_err("the ninth attachment is refused, not served")
    );
    assert!(
        why.contains(&MAX_ATTACHMENTS.to_string()),
        "the refusal names the bound: {why}"
    );

    // **THE SEAT COMES BACK**, so a pipeline that closes and rebuilds passes
    // does not walk its ring up to the bound one rebuild at a time.
    let closing = seated.pop().expect("eight were seated");
    plane.close_instance(closing).expect("it closes");
    let again = plane
        .bind(
            &context,
            program,
            &[],
            Extents::default(),
            GeometryClass::Host,
            &[CROWDED, OUT_B],
        )
        .expect("the seat that was just given back");
    plane.close_instance(again).expect("it closes");
    for instance in seated {
        plane.close_instance(instance).expect("it closes");
    }
}

/// **CLAIM 4**: the set `serve::fence_instances` widens to.
///
/// A shared ring's counters advance at the harvest, and on this plane the
/// consumer reads its ports on the HOST at `stage` — so the fence has to land
/// the flight that holds the PRODUCER, not only the ones that hold the
/// consumer. This is the set that makes that exact.
#[test]
fn the_cohort_of_a_shared_ring_is_the_other_attachment() {
    let _guard = serialized();
    let Some(context) = device_or_skip("a shared ring's cohort") else {
        return;
    };

    let mut plane = Plane::new();
    plane
        .register_channel(&context, SHARED, shared_shape())
        .expect("the ring is cut");
    let put_program = plane
        .register(&context, &registration(putter()))
        .expect("the putter compiles");
    let take_program = plane
        .register(&context, &registration(taker()))
        .expect("the taker compiles");

    let putting = plane
        .bind(
            &context,
            put_program,
            &[],
            Extents::default(),
            GeometryClass::Host,
            &[IN_A, SHARED],
        )
        .expect("the putter binds");
    let taking = plane
        .bind(
            &context,
            take_program,
            &[],
            Extents::default(),
            GeometryClass::Host,
            &[SHARED, OUT_B],
        )
        .expect("the taker binds");
    // A third instance of the SAME program, naming an id this plane never
    // registered — so its device-only ring is cut inside its own session,
    // exactly as every ring on this plane used to be. It shares a program
    // with the taker and a ring with nobody, which is what makes this the
    // difference between "same program" and "same ring".
    let alone = plane
        .bind(
            &context,
            take_program,
            &[],
            Extents::default(),
            GeometryClass::Host,
            &[CROWDED, OUT_B],
        )
        .expect("the unrelated instance binds");

    assert_eq!(
        plane.cohort(&[taking]),
        vec![putting],
        "the fence for the taker has to reach whoever fills its ring"
    );
    assert_eq!(
        plane.cohort(&[putting]),
        vec![taking],
        "and the relation is symmetric: either end can be the one waited on"
    );
    assert!(
        plane.cohort(&[alone]).is_empty(),
        "an instance whose rings are its own costs the fence nothing"
    );
    assert!(
        plane.cohort(&[putting, taking]).is_empty(),
        "a fire that already names both ends adds no one"
    );

    for instance in [putting, taking, alone] {
        plane.close_instance(instance).expect("it closes");
    }
}
