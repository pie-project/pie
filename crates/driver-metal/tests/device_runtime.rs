//! The program compile, against a real device and real (tiny) kernels.
//!
//! What is worth proving here is the machinery around the compiler, not MSL:
//! that the caches answer before the compiler is reached, that a
//! deterministic reject is remembered and a retryable one is not, that two
//! programs share a compiled stage, that a host refusal is data rather than
//! failure, and that a second process replays the whole batch from the
//! archive.

#![allow(clippy::print_stdout)]

use std::rc::Rc;

use driver_api::local::{
    PIE_KERNEL_COMMIT, PIE_KERNEL_FUSED, PIE_KERNEL_GROUPED, PIE_KERNEL_READINESS,
    PIE_KERNEL_SINGLETON, PIE_VALUE_INTRINSIC,
};
use driver_api::plan::{
    EmittedKernel, LaunchChannel, LaunchOp, LaunchPackage, LaunchPlanValue, LaunchRegion,
    LaunchStage, LaunchStagePlan, LaunchValue,
};
use driver_metal::channel::{ExecPlan, Failure, Versions, adopt_launch_package};
use driver_metal::Error;
use driver_metal::device::{Archives, Context};
use driver_metal::program::{ORDINAL_BASE, Runtime};
use tensor_ir::op::{intrinsic_tags, tags};
use tensor_ir::registry::Stage;

const VERSIONS: Versions = Versions {
    compiler: 1,
    region_plan: 1,
    lane_table: 3,
    emitter: 36,
};

struct Fixture {
    context: Context,
    runtime: Runtime,
    /// Holds the kernels dir and the archive dir alive.
    _dir: tempfile::TempDir,
}

fn fixture() -> Option<Fixture> {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => return None,
        Err(e) => panic!("context: {e}"),
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let kernels = dir.path().join("kernels");
    std::fs::create_dir_all(kernels.join("ptir")).expect("kernels dir");
    std::fs::write(
        kernels.join("ptir/ptir_rng.generated.metal"),
        "// rng preamble for tests\n",
    )
    .expect("rng preamble");
    let archives = Archives::new(Some(dir.path().join("archives")));
    let runtime = Runtime::new(kernels, archives).expect("runtime");
    Some(Fixture {
        context,
        runtime,
        _dir: dir,
    })
}

/// A unary op with one result and no channel.
fn exp_op() -> LaunchOp {
    LaunchOp {
        code: u16::from(tags::EXP),
        result_count: 1,
        channel: u32::MAX,
        args: vec![0],
        ..LaunchOp::default()
    }
}

/// One epilogue stage with a single singleton region.
fn one_stage_package(signature: u64, identity: u64) -> LaunchPackage {
    LaunchPackage {
        stages: vec![LaunchStage {
            kind: Stage::Epilogue as u8,
            ..LaunchStage::default()
        }],
        plans: vec![LaunchStagePlan {
            signature_hash: signature,
            identity,
            ops: vec![exp_op()],
            source_ops: vec![vec![0]],
            value_types: vec![LaunchPlanValue::default()],
            singleton: vec![LaunchRegion::default()],
            ..LaunchStagePlan::default()
        }],
        ..LaunchPackage::default()
    }
}

fn plan_of(package: LaunchPackage) -> ExecPlan {
    adopt_launch_package(package).expect("well-formed package")
}

/// A kernel that compiles: an empty entry point named `entry`.
fn kern(kind: u32, stage: u32, region: u32, entry: &str) -> EmittedKernel {
    EmittedKernel {
        kind,
        stage_index: stage,
        region_index: region,
        entry_name: entry.to_owned(),
        source: format!("kernel void {entry}() {{}}\n"),
        error: String::new(),
    }
}

/// A host refusal: no source, a reason.
fn refusal(kind: u32, stage: u32, region: u32, error: &str) -> EmittedKernel {
    EmittedKernel {
        kind,
        stage_index: stage,
        region_index: region,
        entry_name: String::new(),
        source: String::new(),
        error: error.to_owned(),
    }
}

/// The full emission for [`one_stage_package`]: the singleton region, its
/// grouped-singleton twin, the grouped effect pair at region 0 and the
/// per-program pair at region 1.
fn kernels() -> Vec<EmittedKernel> {
    vec![
        kern(PIE_KERNEL_SINGLETON, 0, 0, "s0"),
        kern(PIE_KERNEL_GROUPED, 0, 0, "g0"),
        kern(PIE_KERNEL_READINESS, 0, 0, "gr"),
        kern(PIE_KERNEL_COMMIT, 0, 0, "gc"),
        kern(PIE_KERNEL_READINESS, 0, 1, "r1"),
        kern(PIE_KERNEL_COMMIT, 0, 1, "c1"),
    ]
}

#[test]
fn compiling_the_same_program_twice_compiles_once_and_hits_memory_after() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(one_stage_package(0x51, 0xD1));

    let program = f
        .runtime
        .compile(&f.context, 0xAB, &plan, VERSIONS, &kernels())
        .expect("the program compiles");
    // Six kernels: singleton, grouped singleton, both effect pairs.
    assert_eq!(f.runtime.stats().compilations, 6);
    assert_eq!(program.stages.len(), 1);
    let stage = &program.stages[0].executable;
    assert_eq!(stage.regions.len(), 1);
    assert!(stage.fused.as_ref().is_ok_and(Vec::is_empty));
    assert!(stage.grouped.as_ref().is_ok_and(Vec::is_empty));
    assert_eq!(stage.grouped_singleton.len(), 1);

    // Ordinals are allocated in walk order, from the base that clears the
    // forward DAG namespace.
    assert_eq!(stage.regions[0].ordinal, ORDINAL_BASE);
    assert_eq!(program.readiness_ordinal, ORDINAL_BASE + 1);
    assert_eq!(program.commit_ordinal, ORDINAL_BASE + 2);

    let again = f
        .runtime
        .compile(&f.context, 0xAB, &plan, VERSIONS, &kernels())
        .expect("the second compile hits");
    assert!(
        Rc::ptr_eq(&program, &again),
        "a memory hit must return the same executable, not a rebuild"
    );
    assert_eq!(f.runtime.stats().compilations, 6, "nothing recompiled");
    assert_eq!(f.runtime.stats().memory_hits, 1);
}

#[test]
fn a_plan_that_cannot_execute_is_rejected_once_and_replayed_after() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    // A hidden-state intrinsic marks the plan non-executable at adoption.
    let mut package = one_stage_package(0x52, 0xD2);
    package.values = vec![LaunchValue {
        id: 0,
        source: PIE_VALUE_INTRINSIC,
        intrinsic: intrinsic_tags::HIDDEN as u8,
        shape: vec![1],
        ..LaunchValue::default()
    }];
    let plan = plan_of(package);

    let failure = f
        .runtime
        .compile(&f.context, 0xCD, &plan, VERSIONS, &kernels())
        .expect_err("a non-executable plan cannot compile");
    assert!(matches!(failure, Failure::Deterministic { .. }));
    assert_eq!(f.runtime.negative_entries(), 1);

    let replay = f
        .runtime
        .compile(&f.context, 0xCD, &plan, VERSIONS, &kernels())
        .expect_err("the remembered answer replays");
    assert_eq!(replay.reason(), failure.reason());
    assert_eq!(f.runtime.stats().negative_hits, 1);
    assert_eq!(
        f.runtime.stats().compilations,
        0,
        "no kernel was ever handed to the compiler"
    );
}

#[test]
fn a_compile_error_is_retryable_and_poisons_nothing() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(one_stage_package(0x53, 0xD3));

    let mut broken = kernels();
    broken[0].source = "kernel void s0( {".to_owned();
    let failure = f
        .runtime
        .compile(&f.context, 0xEF, &plan, VERSIONS, &broken)
        .expect_err("the source does not compile");
    let Failure::Retryable { reason } = &failure else {
        panic!("a compiler error is retryable, got {failure:?}");
    };
    assert!(
        reason.contains("Metal M1 compile failed"),
        "the failure does not name the kernel: {reason}"
    );
    assert_eq!(
        f.runtime.negative_entries(),
        0,
        "a retryable failure written to the negative cache becomes permanent"
    );
    assert_eq!(f.runtime.stage_entries(), 0, "nothing was installed");

    // The same hash with fixed sources compiles — and from the ordinal base,
    // proving the failed attempt consumed nothing.
    let program = f
        .runtime
        .compile(&f.context, 0xEF, &plan, VERSIONS, &kernels())
        .expect("the retry succeeds");
    assert_eq!(
        program.stages[0].executable.regions[0].ordinal,
        ORDINAL_BASE
    );
}

#[test]
fn two_programs_sharing_a_stage_share_its_compiled_executable() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(one_stage_package(0x54, 0xD4));

    let first = f
        .runtime
        .compile(&f.context, 1, &plan, VERSIONS, &kernels())
        .expect("first program");
    let second = f
        .runtime
        .compile(&f.context, 2, &plan, VERSIONS, &kernels())
        .expect("second program");

    assert!(
        Rc::ptr_eq(&first.stages[0].executable, &second.stages[0].executable),
        "the stage cache must hand the second program the first one's stage"
    );
    // The second compile needed only its own effect pair — the stage hit in
    // memory, the grouped pair is shared across programs — and even that
    // pair came from the first program's archive, because the two programs
    // share a signature and therefore an archive key.
    assert_eq!(f.runtime.stats().compilations, 6);
    assert_eq!(f.runtime.stats().persistent_hits, 2);
    assert_eq!(f.runtime.stage_entries(), 1);
}

#[test]
fn a_missing_singleton_kernel_is_a_deterministic_reject() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(one_stage_package(0x55, 0xD5));
    let missing: Vec<EmittedKernel> = kernels()
        .into_iter()
        .filter(|k| !(k.kind == PIE_KERNEL_SINGLETON && k.region_index == 0))
        .collect();

    let failure = f
        .runtime
        .compile(&f.context, 0x11, &plan, VERSIONS, &missing)
        .expect_err("an ABI hole is not retryable");
    assert!(matches!(failure, Failure::Deterministic { .. }));
    assert!(
        failure.reason().contains("missing singleton kernel"),
        "the reason does not say what is missing: {}",
        failure.reason()
    );
}

#[test]
fn a_fused_refusal_is_data_and_the_program_still_compiles() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut package = one_stage_package(0x56, 0xD6);
    package.plans[0].fused = vec![LaunchRegion::default()];
    let plan = plan_of(package);

    let mut emission = kernels();
    emission.push(refusal(PIE_KERNEL_FUSED, 0, 0, "stage is too wide to fuse"));
    // The grouped-fused twin sits past the singleton block: region 1.
    emission.push(refusal(PIE_KERNEL_GROUPED, 0, 1, "and too wide to group"));

    let program = f
        .runtime
        .compile(&f.context, 0x22, &plan, VERSIONS, &emission)
        .expect("a refusal is not a failure");
    let stage = &program.stages[0].executable;
    assert_eq!(
        stage.fused.as_ref().err().map(String::as_str),
        Some("stage is too wide to fuse")
    );
    assert_eq!(
        stage.grouped.as_ref().err().map(String::as_str),
        Some("and too wide to group")
    );
    assert_eq!(
        stage.grouped_singleton.len(),
        1,
        "the fallback path must still exist"
    );
}

#[test]
fn more_channel_slots_than_a_lane_can_bind_is_refused() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut package = one_stage_package(0x57, 0xD7);
    package.channels = vec![
        LaunchChannel {
            capacity: 1,
            ..LaunchChannel::default()
        };
        30
    ];
    let plan = plan_of(package);

    let failure = f
        .runtime
        .compile(&f.context, 0x33, &plan, VERSIONS, &kernels())
        .expect_err("30 channels cannot bind");
    assert!(
        failure.reason().contains("at most 29"),
        "the reason does not name the bound: {}",
        failure.reason()
    );
}

#[test]
fn a_new_emitter_version_recompiles_rather_than_reusing_the_stage() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(one_stage_package(0x58, 0xD8));

    let first = f
        .runtime
        .compile(&f.context, 5, &plan, VERSIONS, &kernels())
        .expect("first emitter version");
    let bumped = Versions {
        emitter: VERSIONS.emitter + 1,
        ..VERSIONS
    };
    let second = f
        .runtime
        .compile(&f.context, 6, &plan, bumped, &kernels())
        .expect("second emitter version");

    assert!(
        !Rc::ptr_eq(&first.stages[0].executable, &second.stages[0].executable),
        "a stage compiled by one emitter must not serve kernels for another"
    );
    // Four fresh kernels: the stage pair recompiled, the per-program pair
    // always compiles, and the grouped pair is reused from the first program.
    assert_eq!(f.runtime.stats().compilations, 6 + 4);
    assert_eq!(f.runtime.stage_entries(), 2);
}

#[test]
fn a_second_runtime_replays_the_whole_program_from_the_archive() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let Fixture {
        context,
        mut runtime,
        _dir,
    } = f;
    let plan = plan_of(one_stage_package(0x59, 0xD9));
    runtime
        .compile(&context, 0x44, &plan, VERSIONS, &kernels())
        .expect("first process compiles and archives");
    assert_eq!(runtime.stats().compilations, 6);

    // A "restart": a fresh runtime over the same directories.
    let kernels_dir = _dir.path().join("kernels");
    let archives = Archives::new(Some(_dir.path().join("archives")));
    let mut second = Runtime::new(kernels_dir, archives).expect("second runtime");
    second
        .compile(&context, 0x44, &plan, VERSIONS, &kernels())
        .expect("second process replays");
    assert_eq!(
        second.stats().persistent_hits,
        6,
        "the archive must serve every pipeline of the batch"
    );
    assert_eq!(second.stats().compilations, 0);
}
