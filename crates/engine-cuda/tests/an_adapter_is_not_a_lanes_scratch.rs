//! Pins that an adapter's weights are not a lane's per-fire scratch.

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
use eta_compiler::plan::compile_bound;
use eta_exec::{Extents, describe, layout};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::bind;

use engine_cuda::program::{describe_values, scratch_bytes};

// the fixture

/// qwen35-d0.8b's adapter geometry.
const LAYERS: u32 = 24;
const HIDDEN: u32 = 1024;
const RANK: u32 = 16;

/// Site bits the placement constant carries.
const SITES: u32 = 4;

/// A prologue with one low-rank adapter: three peeked channels and the
/// `lora` sink over them. `also_read` adds a launched op reading `A` too.
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

/// The stage whose plan declares the sink (`needs.lora`).
fn adapter_stage(package: &LaunchPackage) -> &LaunchStagePlan {
    package
        .plans
        .iter()
        .find(|plan| plan.needs.lora)
        .expect("a prologue that states an adapter declares `needs.lora`")
}

/// What one lane's scratch would cost if every declared value materialised
/// — the pre-fix budget.
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

// the claims

/// (a) The sink's scratch does not scale with the adapter's geometry.
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

/// (b) The naive budget does scale — which is what makes (a) a real gate.
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

/// (c) An adapter-carrying prologue costs a lane a rounding error, in
/// absolute bytes.
#[test]
fn an_adapter_carrying_prologue_costs_a_lane_a_rounding_error() {
    let bytes = adapter_scratch(LAYERS, RANK, HIDDEN);
    assert!(
        bytes <= 4096,
        "a prologue whose only content is an adapter the engine reads at BIND \
         has nothing to carry per lane, and this one carries {bytes} bytes"
    );
}

/// (d) The value descriptors say the adapter planes are empty.
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

/// (e) A value the sink shares with a launched op is still carried.
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
