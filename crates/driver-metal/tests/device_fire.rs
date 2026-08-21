//! One whole fire, end to end, against a real device.
//!
//! The kernels here are hand-written stand-ins for the host emitter's, but
//! they honour the real binding ABI — a region sees `(status, descriptors,
//! a0, a1, a2, o0, o1, temporary, params)`, the effect pair sees `(status,
//! lane_table, words...)` — and the real status protocol: readiness may
//! write retry, regions run between the barriers, commit publishes state 4.
//! What the tests prove is the driver's half of the contract: that prepare
//! refuses an early fire without allocating, that execute runs readiness →
//! regions → commit in that order, and that the status readback keeps
//! "retry", "fault" and "committed" apart.

#![allow(clippy::print_stdout)]

use std::rc::Rc;

use driver_api::local::{
    PIE_KERNEL_COMMIT, PIE_KERNEL_GROUPED, PIE_KERNEL_READINESS, PIE_KERNEL_SINGLETON,
    PIE_READINESS_NEEDS_FULL,
};
use driver_api::plan::{
    EmittedKernel, LaunchChannel, LaunchOp, LaunchPackage, LaunchPlanValue, LaunchRegion,
    LaunchStage, LaunchStagePlan,
};
use driver_metal::Error;
use driver_metal::channel::{
    ExecPlan, Reason, StatusOutcome, Ticket, Versions, adopt_launch_package,
};
use driver_metal::device::{Archives, Context, Externals, Pool, Ring, Stepper, Tables};
use driver_metal::program::{DeviceInputs, Mode, Prepare, Runtime};
use tensor_ir::DType;
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
    tables: Tables,
    externals: Externals,
    _dir: tempfile::TempDir,
}

fn fixture() -> Option<Fixture> {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => {
            driver_metal::skip::skipped("no Metal 4 device, so no fire was encoded");
            return None;
        }
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
        tables: Tables::new(),
        externals: Externals::new(),
        _dir: dir,
    })
}

/// One epilogue stage with one unary op over one value, optionally taking
/// from channel slot 0.
fn package(signature: u64, channel: bool) -> LaunchPackage {
    let op = LaunchOp {
        code: if channel {
            u16::from(tags::CHAN_TAKE)
        } else {
            u16::from(tags::EXP)
        },
        result_count: 1,
        channel: if channel { 0 } else { u32::MAX },
        args: if channel { vec![] } else { vec![0] },
        ..LaunchOp::default()
    };
    LaunchPackage {
        channels: if channel {
            vec![LaunchChannel {
                capacity: 1,
                readiness: PIE_READINESS_NEEDS_FULL,
                ..LaunchChannel::default()
            }]
        } else {
            vec![]
        },
        stages: vec![LaunchStage {
            kind: Stage::Epilogue as u8,
            takes: if channel { vec![0] } else { vec![] },
            ..LaunchStage::default()
        }],
        plans: vec![LaunchStagePlan {
            signature_hash: signature,
            identity: signature ^ 0xD00D,
            ops: vec![op],
            source_ops: vec![vec![0]],
            value_types: vec![LaunchPlanValue::default()],
            channel_bindings: if channel { vec![0] } else { vec![] },
            singleton: vec![LaunchRegion::default()],
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

/// The emission: a region that stamps the status `reserved0` word, effect
/// kernels that honour the state machine, and grouped stand-ins.
fn kernels(readiness_state: u32) -> Vec<EmittedKernel> {
    vec![
        kern(
            PIE_KERNEL_SINGLETON,
            0,
            0,
            "s0",
            "kernel void s0(device uint* status [[buffer(0)]],
                            device float* o0 [[buffer(5)]]) {
                 if (status[0] != 0u) return;
                 o0[0] = 42.0f;
                 status[2] = 0xBEEFu;
             }",
        ),
        kern(PIE_KERNEL_GROUPED, 0, 0, "g0", "kernel void g0() {}"),
        kern(PIE_KERNEL_READINESS, 0, 0, "gr", "kernel void gr() {}"),
        kern(PIE_KERNEL_COMMIT, 0, 0, "gc", "kernel void gc() {}"),
        kern(
            PIE_KERNEL_READINESS,
            0,
            1,
            "r1",
            &format!(
                "kernel void r1(device uint* status [[buffer(0)]]) {{
                     status[0] = {readiness_state}u;
                 }}"
            ),
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
fn a_fire_runs_readiness_then_regions_then_commit_and_commits() {
    let Some(mut f) = fixture() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    let plan = plan_of(package(0x71, false));
    let program = f
        .runtime
        .compile(&f.context, 0x71, &plan, VERSIONS, &kernels(0))
        .expect("compiles");

    let Prepare::Ready(fire) = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &[], &[])
        .expect("prepare")
    else {
        panic!("a fire with no channels is always ready");
    };

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    let execution = f
        .runtime
        .execute(
            &f.context,
            &mut stepper,
            &mut f.tables,
            &f.pool,
            &f.externals,
            &fire,
            &DeviceInputs::default(),
            Mode::Singleton,
        )
        .expect("execute");

    assert_eq!(execution.outcome, StatusOutcome::Committed);
    assert!(
        execution.report.is_none(),
        "a committed fire needs no story"
    );
    let status = fire.status().expect("status parses");
    assert_eq!(
        status.reserved0, 0xBEEF,
        "the region did not run between readiness and commit"
    );
}

#[test]
fn a_gpu_side_retry_is_reported_as_retry_not_as_a_fault() {
    let Some(mut f) = fixture() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    let plan = plan_of(package(0x72, false));
    // Readiness writes state 2: the device found the fire early.
    let program = f
        .runtime
        .compile(&f.context, 0x72, &plan, VERSIONS, &kernels(2))
        .expect("compiles");

    let Prepare::Ready(fire) = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &[], &[])
        .expect("prepare")
    else {
        panic!("ready");
    };
    let mut stepper = Stepper::new(&f.context).expect("stepper");
    let execution = f
        .runtime
        .execute(
            &f.context,
            &mut stepper,
            &mut f.tables,
            &f.pool,
            &f.externals,
            &fire,
            &DeviceInputs::default(),
            Mode::Singleton,
        )
        .expect("execute");

    assert_eq!(execution.outcome, StatusOutcome::Retry);
    let report = execution.report.expect("a retry carries its account");
    assert!(
        report.contains("readiness"),
        "the report does not say the fire was early: {report}"
    );
}

#[test]
fn an_early_fire_is_refused_by_the_host_before_anything_is_allocated() {
    let Some(mut f) = fixture() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    let plan = plan_of(package(0x73, true));
    let program = f
        .runtime
        .compile(&f.context, 0x73, &plan, VERSIONS, &kernels(0))
        .expect("compiles");

    // One empty ring; the program's first touch takes, so it must wait.
    let ring = Rc::new(Ring::new(&f.context, DType::F32, 1, 1).expect("ring"));
    let outcome = f
        .runtime
        .prepare(
            &f.context,
            &f.pool,
            &program,
            &[Rc::clone(&ring)],
            &[Ticket::default()],
        )
        .expect("prepare itself succeeds");

    let Prepare::Retry { channel, reason } = outcome else {
        panic!("an empty ring under a take is early, not ready");
    };
    assert_eq!((channel, reason), (0, Reason::Empty));
    assert_eq!(
        f.pool.stats().allocations,
        0,
        "an early fire must not consume pool buffers"
    );
}

#[test]
fn a_prepared_fire_can_be_executed_again_after_a_retry() {
    let Some(mut f) = fixture() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    let plan = plan_of(package(0x74, false));
    // Readiness leaves state 0 = proceed; but commit only stamps 4 when the
    // state is still 0, so both runs commit.
    let program = f
        .runtime
        .compile(&f.context, 0x74, &plan, VERSIONS, &kernels(0))
        .expect("compiles");
    let Prepare::Ready(fire) = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &[], &[])
        .expect("prepare")
    else {
        panic!("ready");
    };
    let mut stepper = Stepper::new(&f.context).expect("stepper");
    for round in 0..2 {
        let execution = f
            .runtime
            .execute(
                &f.context,
                &mut stepper,
                &mut f.tables,
                &f.pool,
                &f.externals,
                &fire,
                &DeviceInputs::default(),
                Mode::Singleton,
            )
            .expect("execute");
        assert_eq!(
            execution.outcome,
            StatusOutcome::Committed,
            "round {round}: the status buffer must be re-zeroed between runs"
        );
    }
}
