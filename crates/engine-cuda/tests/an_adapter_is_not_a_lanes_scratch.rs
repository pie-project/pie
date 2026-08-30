//! **A-5'S SECOND HALF**: an adapter's weights are not in a lane's scratch
//! (alto adapter §6.1, the named trap).
//!
//! # The trap this closes
//!
//! §6.1 promoted the blob store into wave 1 because a channel is not a weight
//! transport: the machinery re-pays a cell EVERY FIRE. Wave A1/A2 landed the
//! bytes once, at instance bind, and
//! `the_fire_path_cannot_reach_the_adapter_store` proves no fire can even
//! reach them. That is the READ. The ALLOCATION outlived it.
//!
//! The `lora` sink is a `sink_call` in a `Library(SecondParty)` region, and
//! `program::compile` skips it — there is no body to compile and this shell
//! launches nothing in its place. Its operands are not skipped: each is an
//! ordinary `chan_read` in an ordinary generated region, so the stage's value
//! table still declared `[layers, rank, hidden]` and `[layers, hidden, rank]`,
//! `Prepared::build` still cut them per lane, and `commit_lanes` still
//! re-zeroed them at every fire — for weights nothing reads. Worse than the
//! slots themselves: `eta_exec::layout` sizes the SHARED temporary block off
//! the widest value times four, so an adapter plane cost its own bytes over
//! again, four times, on top.
//!
//! ```text
//! (a) the sink's scratch does not scale with layers x rank x hidden
//! (b) and the naive budget — every declared value described — DOES,
//!     which is what says this file would fail without the fix
//! (c) an adapter-carrying prologue costs a lane a rounding error, absolutely
//! (d) the planes describe as EMPTY, by name
//! (e) a value the sink AND a launched op both read STAYS: correctness is
//!     not traded for the saving
//! ```
//!
//! Everything here is host arithmetic over a launch package — no device, no
//! `cudaMalloc`, no GPU in the machine.
//!
//! ```text
//! cargo test -p engine-cuda --test an_adapter_is_not_a_lanes_scratch
//! ```

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
use eta_compiler::plan::compile_bound;
use eta_exec::{Extents, describe, layout};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::bind;

use engine_cuda::program::{describe_values, scratch_bytes};

// ── the fixture ──────────────────────────────────────────────────────────

/// qwen35-d0.8b's adapter geometry, which is what `tests/inferlets/lora-probe`
/// seeds: 24 layers, hidden 1024, the bank's own rank 16.
const LAYERS: u32 = 24;
const HIDDEN: u32 = 1024;
const RANK: u32 = 16;

/// How many site bits the trace-known placement constant carries. Small, and
/// nothing here depends on the number.
const SITES: u32 = 4;

/// A prologue that states one low-rank adapter, exactly as the surface lowers
/// `Pass::adapter`: three peeked channels and the `lora` sink over them.
///
/// `also_read` adds a launched op — a negate whose result is put back onto a
/// channel — over the `A` plane, which is claim (e)'s control: a value the
/// sink shares with something that RUNS is a value the fire still has to
/// carry.
fn lora_prologue(layers: u32, rank: u32, hidden: u32, also_read: bool) -> TraceContainer {
    let chan = |shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: true,
    };
    let a = Shape::new(&[layers, rank, hidden]).expect("A is [layers, rank, hidden]");
    let b = Shape::new(&[layers, hidden, rank]).expect("B is out-major, per §6.3");
    let mut channels = vec![chan(a.clone()), chan(b), chan(Shape::vector(SITES))];
    let mut ops = vec![Op::ChanRead(0), Op::ChanRead(1), Op::ChanRead(2)];
    if also_read {
        channels.push(ChannelDecl {
            seeded: false,
            ..chan(a)
        });
        ops.push(Op::Neg(0));
        ops.push(Op::ChanPut { chan: 3, value: 3 });
    }
    ops.push(Op::SinkCall {
        name: 0,
        args: vec![0, 1, 2],
    });
    TraceContainer {
        names: vec!["lora".to_string()],
        channels,
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops,
        }],
        externs: Vec::new(),
    }
}

/// The package a container compiles to.
fn package(container: TraceContainer) -> LaunchPackage {
    let bound = bind(container, ModelProfile::dummy()).expect("the lora prologue binds");
    let stages = compile_bound(&bound);
    eta_compiler::codegen::launch::build(&bound, &stages)
}

/// The one stage whose plan declares the sink — §6.4's "the plan says
/// WHETHER", which is the same fact `adapter::sink_of` keys on.
fn adapter_stage(package: &LaunchPackage) -> &LaunchStagePlan {
    package
        .plans
        .iter()
        .find(|plan| plan.needs.lora)
        .expect("a prologue that states an adapter declares `needs.lora`")
}

/// What one lane's scratch WOULD cost if every value the plan declares were
/// materialised — the budget this shell cut before the fix, and the number
/// claim (b) is about.
fn naive_bytes(plan: &LaunchStagePlan, extents: Extents) -> u64 {
    let descriptors: Vec<_> = plan
        .value_types
        .iter()
        .map(|value| describe(value, &extents).expect("a static shape resolves"))
        .collect();
    layout(&descriptors).expect("the naive budget fits").total
}

/// The per-lane scratch of an adapter prologue at one geometry.
fn adapter_scratch(layers: u32, rank: u32, hidden: u32) -> u64 {
    let package = package(lora_prologue(layers, rank, hidden, false));
    scratch_bytes(adapter_stage(&package), Extents::default()).expect("the stage's scratch")
}

// ── the claims ───────────────────────────────────────────────────────────

/// **(a)** The sink's scratch does not scale with the adapter's geometry.
///
/// Three geometries a hundred thousand elements apart, one number. A rank is
/// TRACE-KNOWN — a different rank is a different traced program — so this is
/// three programs, not one program at three sizes, and the claim is that the
/// shell's per-lane cost cannot tell them apart.
#[test]
fn the_sinks_scratch_does_not_scale_with_layers_rank_hidden() {
    let served = adapter_scratch(LAYERS, RANK, HIDDEN);
    let wider = adapter_scratch(LAYERS, RANK * 4, HIDDEN);
    let deeper = adapter_scratch(LAYERS * 4, RANK, HIDDEN);
    let trivial = adapter_scratch(1, 1, 1);
    assert_eq!(
        (served, wider, deeper),
        (trivial, trivial, trivial),
        "an adapter's planes land ONCE at instance bind (alto adapter §6.1); a \
         per-lane scratch that grows with `layers x rank x hidden` is the \
         per-fire payload the ruling exists to refuse — {LAYERS}x{RANK}x{HIDDEN} \
         cost {served} bytes a lane and a 1x1x1 adapter cost {trivial}"
    );
}

/// **(b)** And the naive budget does — which is what makes (a) a gate.
///
/// A test that passes for the wrong reason is worse than none: if the planes
/// were somehow not in the value table at all, (a) would hold trivially. So
/// this asserts the OTHER direction on the very same plans — describe every
/// declared value, as `Prepared::build` used to, and the number moves by
/// megabytes.
#[test]
fn the_naive_budget_is_the_one_that_scales() {
    let extents = Extents::default();
    let served = package(lora_prologue(LAYERS, RANK, HIDDEN, false));
    let trivial = package(lora_prologue(1, 1, 1, false));
    let served = naive_bytes(adapter_stage(&served), extents);
    let trivial = naive_bytes(adapter_stage(&trivial), extents);
    assert!(
        served > trivial + (1 << 20),
        "the values this shell no longer carries are really there and really \
         large — a qwen35 adapter budgets {served} bytes a lane naively against \
         a 1x1x1 adapter's {trivial}; if these were equal, the claim above \
         would be about nothing"
    );
    // Both planes, plus four temporaries an element off the widest of them.
    let planes = u64::from(LAYERS) * u64::from(RANK) * u64::from(HIDDEN) * 4;
    assert!(
        served >= planes * 2,
        "the naive budget carries A and B and a temporary block sized off the \
         widest of them: at least {} bytes, and it said {served}",
        planes * 2
    );
}

/// **(c)** Stated absolutely, and not only against another adapter: a lane's
/// scratch for an adapter-carrying prologue is a rounding error.
///
/// Three empty values at [`eta_exec::SCRATCH_ALIGN`] apiece, a dummy slot, and
/// a temporary block sized off a single element. Four kibibytes is generous
/// and the point is the order of magnitude, not the constant: 1.5 MiB of `A`,
/// 1.5 MiB of `B` and 6 MiB of temporaries per lane per fire is what this
/// number replaces.
#[test]
fn an_adapter_carrying_prologue_costs_a_lane_a_rounding_error() {
    let bytes = adapter_scratch(LAYERS, RANK, HIDDEN);
    assert!(
        bytes <= 4096,
        "a prologue whose only content is an adapter the engine reads at BIND \
         has nothing to carry per lane, and this one carries {bytes} bytes"
    );
}

/// **(d)** And the descriptors say so by name.
///
/// The saving is not a subtraction in a size computation — it is a
/// DESCRIPTOR, published to the device, that says the value has no elements.
/// That is what makes it safe: the emitted `chan_read` copies
/// `descriptors[o0].len` elements, so the copy that would have filled the slot
/// copies nothing, and there is no second place where the two could disagree.
#[test]
fn the_planes_describe_as_empty() {
    let package = package(lora_prologue(LAYERS, RANK, HIDDEN, false));
    let plan = adapter_stage(&package);
    let descriptors = describe_values(plan, Extents::default()).expect("the descriptor row");
    assert_eq!(
        descriptors.len(),
        plan.value_types.len(),
        "every value still has a descriptor — the row is indexed by value id, \
         so a dropped value is described as empty and never removed"
    );
    assert!(
        descriptors.iter().all(|value| value.len == 0),
        "the three values of an adapter prologue are the sink's own operands \
         and nothing this shell launches reads one; every descriptor should \
         say so: {descriptors:?}"
    );
}

/// **(e)** A value the sink shares with an op that RUNS is still carried.
///
/// The saving is for values nothing launched reads. `A` here also feeds a
/// negate whose result is put onto a channel — a generated region this shell
/// compiles and launches — so the fire has to materialise it, and the cost
/// comes back. A rule that dropped it anyway would hand a live kernel a
/// zero-length buffer.
#[test]
fn a_plane_a_launched_op_also_reads_is_still_carried() {
    let shared = package(lora_prologue(LAYERS, RANK, HIDDEN, true));
    let shared = scratch_bytes(adapter_stage(&shared), Extents::default()).expect("the scratch");
    let alone = adapter_scratch(LAYERS, RANK, HIDDEN);
    let plane = u64::from(LAYERS) * u64::from(RANK) * u64::from(HIDDEN) * 4;
    assert!(
        shared >= alone + plane,
        "`A` feeds the sink AND a negate that is launched, so it is a value \
         this fire really does have to carry — {shared} bytes a lane against \
         {alone} when only the sink reads it, and one plane is {plane}"
    );
}
