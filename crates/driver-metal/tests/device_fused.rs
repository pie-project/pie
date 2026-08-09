//! The M2 placed path, against a real device.
//!
//! A placed command splits a fire around a forward: readiness and prologue
//! regions before, epilogue regions and commit after, all inside the
//! *target* executor's step. The tests stand a trivial forward in the
//! middle and prove the driver's half: the fused region really runs inside
//! the placed step, finish tracks whether anything was encoded rather than
//! reading a zero fill as a fault, and the target's argument tables are
//! given back.

#![allow(clippy::print_stdout)]

use driver_api::local::{
    PIE_KERNEL_COMMIT, PIE_KERNEL_FUSED, PIE_KERNEL_GROUPED, PIE_KERNEL_READINESS,
    PIE_KERNEL_SINGLETON,
};
use driver_api::plan::{
    EmittedKernel, LaunchOp, LaunchPackage, LaunchPlanValue, LaunchRegion, LaunchStage,
    LaunchStagePlan,
};
use driver_metal::channel::{ExecPlan, StatusOutcome, Versions, adopt_launch_package};
use driver_metal::Error;
use driver_metal::device::{Archives, Context, Externals, Pool, Stepper, Tables};
use driver_metal::program::{DeviceInputs, Prepare, Runtime};
use tensor_ir::op::tags;
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
    pool: Pool,
    externals: Externals,
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
    std::fs::write(kernels.join("ptir/ptir_rng.generated.metal"), "// rng\n").expect("rng");
    let runtime =
        Runtime::new(kernels, Archives::new(Some(dir.path().join("archives")))).expect("runtime");
    Some(Fixture {
        context,
        runtime,
        pool: Pool::new(64 << 20),
        externals: Externals::new(),
        _dir: dir,
    })
}

/// One epilogue stage with one op, one fused region over it.
fn package(signature: u64) -> LaunchPackage {
    LaunchPackage {
        stages: vec![LaunchStage {
            kind: Stage::Epilogue as u8,
            ..LaunchStage::default()
        }],
        plans: vec![LaunchStagePlan {
            signature_hash: signature,
            identity: signature ^ 0xF0F0,
            ops: vec![LaunchOp {
                code: u16::from(tags::EXP),
                result_count: 1,
                channel: u32::MAX,
                args: vec![0],
                ..LaunchOp::default()
            }],
            source_ops: vec![vec![0]],
            value_types: vec![LaunchPlanValue::default()],
            singleton: vec![LaunchRegion::default()],
            fused: vec![LaunchRegion::default()],
            ..LaunchStagePlan::default()
        }],
        ..LaunchPackage::default()
    }
}

fn plan_of(p: LaunchPackage) -> ExecPlan {
    adopt_launch_package(p).expect("well-formed")
}

fn kern(kind: u32, stage: u32, region: u32, entry: &str, body: &str) -> EmittedKernel {
    EmittedKernel {
        kind,
        stage_index: stage,
        region_index: region,
        entry_name: entry.to_owned(),
        source: body.to_owned(),
        error: String::new(),
    }
}

/// The emission: singleton and fused twins of the op, a grouped pair of
/// stand-ins, effect kernels that honour the state machine.
fn kernels() -> Vec<EmittedKernel> {
    vec![
        kern(
            PIE_KERNEL_SINGLETON,
            0,
            0,
            "s0",
            "kernel void s0(device uint* status [[buffer(0)]]) { status[2] = 1u; }",
        ),
        // The fused twin stamps a different word so the test can tell which
        // path ran.
        kern(
            PIE_KERNEL_FUSED,
            0,
            0,
            "f0",
            "kernel void f0(device uint* status [[buffer(0)]]) {
                 if (status[0] != 0u) return;
                 status[2] = 0xF00Du;
             }",
        ),
        kern(PIE_KERNEL_GROUPED, 0, 0, "g0", "kernel void g0() {}"),
        kern(PIE_KERNEL_GROUPED, 0, 1, "g1", "kernel void g1() {}"),
        kern(PIE_KERNEL_READINESS, 0, 0, "gr", "kernel void gr() {}"),
        kern(PIE_KERNEL_COMMIT, 0, 0, "gc", "kernel void gc() {}"),
        kern(
            PIE_KERNEL_READINESS,
            0,
            1,
            "r1",
            "kernel void r1(device uint* status [[buffer(0)]]) { status[0] = 0u; }",
        ),
        kern(
            PIE_KERNEL_COMMIT,
            0,
            1,
            "c1",
            "kernel void c1(device uint* status [[buffer(0)]]) {
                 if (status[0] == 0u) status[0] = 4u;
             }",
        ),
    ]
}

#[test]
fn a_placed_fire_runs_inside_the_targets_step_and_commits() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(package(0x81));
    let program = f
        .runtime
        .compile(&f.context, 0x81, &plan, VERSIONS, &kernels())
        .expect("compiles");
    let Prepare::Ready(fire) = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &[], &[])
        .expect("prepare")
    else {
        panic!("ready");
    };

    // The "target" is the forward executor's gear; here, a fresh set.
    let mut tables = Tables::new();
    let mut command = f
        .runtime
        .prepare_m2(
            &f.context,
            &f.pool,
            &f.externals,
            &fire,
            &DeviceInputs::default(),
        )
        .expect("prepare_m2");

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    stepper
        .run(|step| {
            command.encode_pre(&f.context, &mut tables, step)?;
            // The model forward would be encoded here.
            command.encode_post(&f.context, &mut tables, step)
        })
        .expect("the placed step ran");

    let (outcome, report) = command.finish(&mut tables);
    assert_eq!(outcome, StatusOutcome::Committed, "report: {report:?}");
    let status = fire.status().expect("status parses");
    assert_eq!(
        status.reserved0, 0xF00D,
        "the fused region did not run inside the placed step"
    );
    assert!(
        tables.is_empty(),
        "finish must hand the target its tables back"
    );
}

#[test]
fn a_command_never_encoded_reports_never_dispatched_not_a_fault() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(package(0x82));
    let program = f
        .runtime
        .compile(&f.context, 0x82, &plan, VERSIONS, &kernels())
        .expect("compiles");
    let Prepare::Ready(fire) = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &[], &[])
        .expect("prepare")
    else {
        panic!("ready");
    };

    let mut tables = Tables::new();
    let command = f
        .runtime
        .prepare_m2(
            &f.context,
            &f.pool,
            &f.externals,
            &fire,
            &DeviceInputs::default(),
        )
        .expect("prepare_m2");
    // The forward was refused; nothing was ever encoded.
    let (outcome, report) = command.finish(&mut tables);

    assert_eq!(outcome, StatusOutcome::Failed);
    let report = report.expect("a failure carries its account");
    assert!(
        report.contains("never encoded"),
        "the zero fill was read as a GPU fault: {report}"
    );
}

#[test]
fn a_stage_without_a_fused_executable_cannot_be_placed() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(package(0x83));
    let mut emission = kernels();
    // The host refuses the fused kernel; the stage drops to singleton.
    emission.retain(|k| k.kind != PIE_KERNEL_FUSED);
    emission.push(EmittedKernel {
        kind: PIE_KERNEL_FUSED,
        stage_index: 0,
        region_index: 0,
        entry_name: String::new(),
        source: String::new(),
        error: "too wide to fuse".to_owned(),
    });
    let program = f
        .runtime
        .compile(&f.context, 0x83, &plan, VERSIONS, &emission)
        .expect("a refusal is not a compile failure");
    let Prepare::Ready(fire) = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &[], &[])
        .expect("prepare")
    else {
        panic!("ready");
    };

    let refusal = f
        .runtime
        .prepare_m2(
            &f.context,
            &f.pool,
            &f.externals,
            &fire,
            &DeviceInputs::default(),
        )
        .expect_err("an unfusable stage cannot ride a forward");
    assert!(
        refusal.to_string().contains("too wide to fuse"),
        "the refusal must carry the host's reason: {refusal}"
    );
}
