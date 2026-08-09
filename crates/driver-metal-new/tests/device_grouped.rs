//! The M3 grouped path, against a real device.
//!
//! The stand-in kernels walk the real lane table: the commit kernel reads
//! each lane's record and writes state 4 through its `commit_slot`, and the
//! region kernel resolves its lane through `lane_indices` — the same
//! pointer-chasing the shipped kernels do. What the tests prove is the
//! grouping itself: two fires sharing a stage identity become ONE region
//! dispatch, every lane commits, a never-encoded group says so instead of
//! reporting sixty-four faults, and the composition refusals (early lane,
//! aliased channel) name their reason.

#![allow(clippy::print_stdout)]

use std::rc::Rc;

use driver_api::local::{
    PIE_KERNEL_COMMIT, PIE_KERNEL_GROUPED, PIE_KERNEL_READINESS, PIE_KERNEL_SINGLETON,
    PIE_READINESS_NEEDS_EMPTY,
};
use driver_api::plan::{
    EmittedKernel, LaunchChannel, LaunchOp, LaunchPackage, LaunchPlanValue, LaunchRegion,
    LaunchStage, LaunchStagePlan,
};
use driver_metal_new::pipeline::{
    ExecPlan, NO_TICKET, StatusOutcome, Ticket, Versions, adopt_launch_package,
};
use driver_metal_new::{
    Archives, Context, DeviceInputs, Error, Externals, LaneCandidate, Pool, Prepare, PreparedFire,
    Region, Ring, Runtime, Stepper, Tables,
};
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

/// One epilogue stage, one op; `channel` adds a put into channel slot 0.
fn package(signature: u64, channel: bool) -> LaunchPackage {
    let op = if channel {
        LaunchOp {
            code: u16::from(tags::CHAN_PUT),
            result_count: 0,
            channel: 0,
            args: vec![0],
            ..LaunchOp::default()
        }
    } else {
        LaunchOp {
            code: u16::from(tags::EXP),
            result_count: 1,
            channel: u32::MAX,
            args: vec![0],
            ..LaunchOp::default()
        }
    };
    LaunchPackage {
        channels: if channel {
            vec![LaunchChannel {
                capacity: 1,
                readiness: PIE_READINESS_NEEDS_EMPTY,
                ..LaunchChannel::default()
            }]
        } else {
            vec![]
        },
        stages: vec![LaunchStage {
            kind: Stage::Epilogue as u8,
            ..LaunchStage::default()
        }],
        plans: vec![LaunchStagePlan {
            signature_hash: signature,
            identity: signature ^ 0x9A9A,
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

/// Stand-ins that walk the real lane table: the region stamps each lane's
/// status `reserved0` through the record's `commit_slot`, and the grouped
/// commit writes state 4 the same way.
fn kernels() -> Vec<EmittedKernel> {
    vec![
        kern(
            PIE_KERNEL_SINGLETON,
            0,
            0,
            "s0",
            "kernel void s0(device uint* status [[buffer(0)]]) {}",
        ),
        kern(
            PIE_KERNEL_GROUPED,
            0,
            0,
            "g0",
            "kernel void g0(const device uchar* lane [[buffer(0)]],
                            const device uint* lane_indices [[buffer(8)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint local [[thread_position_in_threadgroup]]) {
                 if (local != 0u) return;
                 uint target = lane_indices[tg];
                 const device ulong* record =
                     (const device ulong*)(lane + 16 + target * 96);
                 device uint* status = (device uint*)record[7];
                 status[2] = 0xABCDu;
             }",
        ),
        kern(
            PIE_KERNEL_READINESS,
            0,
            0,
            "gr",
            "kernel void gr(const device uchar* lane [[buffer(0)]]) {}",
        ),
        kern(
            PIE_KERNEL_COMMIT,
            0,
            0,
            "gc",
            "kernel void gc(const device uchar* lane [[buffer(0)]],
                            const device uchar* meta [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
                 const device uint* header = (const device uint*)lane;
                 if (gid >= header[1]) return;
                 const device ulong* record =
                     (const device ulong*)(lane + 16 + gid * 96);
                 device uint* status = (device uint*)record[7];
                 if (status[0] == 0u) status[0] = 4u;
             }",
        ),
        kern(
            PIE_KERNEL_READINESS,
            0,
            1,
            "r1",
            "kernel void r1(device uint* status [[buffer(0)]]) {}",
        ),
        kern(
            PIE_KERNEL_COMMIT,
            0,
            1,
            "c1",
            "kernel void c1(device uint* status [[buffer(0)]]) {}",
        ),
    ]
}

/// Compile and prepare one channel-free fire.
fn ready_fire(f: &mut Fixture, hash: u64) -> Rc<PreparedFire> {
    let plan = plan_of(package(0x91, false));
    let program = f
        .runtime
        .compile(&f.context, hash, &plan, VERSIONS, &kernels())
        .expect("compiles");
    let Prepare::Ready(fire) = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &[], &[])
        .expect("prepare")
    else {
        panic!("ready");
    };
    fire
}

#[test]
fn two_fires_sharing_a_stage_become_one_region_dispatch() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let first = ready_fire(&mut f, 1);
    let second = ready_fire(&mut f, 2);

    let mut tables = Tables::new();
    let mut group = f
        .runtime
        .prepare_m3(
            &f.context,
            &f.pool,
            &f.externals,
            vec![
                LaneCandidate {
                    fire: first,
                    inputs: DeviceInputs::default(),
                    retry_ineligible: false,
                },
                LaneCandidate {
                    fire: second,
                    inputs: DeviceInputs::default(),
                    retry_ineligible: false,
                },
            ],
        )
        .expect("prepare_m3");

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    stepper
        .run(|step| {
            group.encode_pre(&f.context, &mut tables, step)?;
            group.encode_post(&f.context, &mut tables, step)
        })
        .expect("the group step ran");

    let (outcomes, report, stats) = group.finish(&mut tables);
    assert_eq!(
        outcomes,
        vec![StatusOutcome::Committed, StatusOutcome::Committed],
        "report: {report:?}"
    );
    assert!(report.is_none());
    assert_eq!(stats.lanes, 2);
    assert_eq!(
        stats.body_launches, 1,
        "two lanes sharing an identity and bucket must be ONE dispatch"
    );
    assert_eq!(stats.singleton_fallback_launches, 1);
    assert_eq!((stats.readiness_launches, stats.commit_launches), (1, 1));
    assert!(
        stats.post_forward_critical_ns > 0,
        "the host-clock fallback must measure the post-forward span"
    );
    assert!(tables.is_empty(), "finish must return the target's tables");
}

#[test]
fn a_group_never_encoded_says_so_once_instead_of_faulting_every_lane() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let fire = ready_fire(&mut f, 3);
    let mut tables = Tables::new();
    let group = f
        .runtime
        .prepare_m3(
            &f.context,
            &f.pool,
            &f.externals,
            vec![LaneCandidate {
                fire,
                inputs: DeviceInputs::default(),
                retry_ineligible: false,
            }],
        )
        .expect("prepare_m3");

    let (outcomes, report, _) = group.finish(&mut tables);
    assert_eq!(outcomes, vec![StatusOutcome::Failed]);
    assert!(
        report.is_some_and(|r| r.contains("never encoded")),
        "the zero fill must not be read as a lane fault"
    );
}

#[test]
fn a_lane_that_went_stale_aborts_the_group_composition() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    // A put program whose ring must have room; fill the ring so the fire
    // prepared against the empty ring is stale by group time.
    let plan = plan_of(package(0x92, true));
    let program = f
        .runtime
        .compile(&f.context, 4, &plan, VERSIONS, &kernels())
        .expect("compiles");
    let ring = Rc::new(Ring::new(&f.context, DType::F32, 1, 1).expect("ring"));
    let ticket = Ticket {
        expected_head: NO_TICKET,
        expected_tail: 0,
    };
    let Prepare::Ready(fire) = f
        .runtime
        .prepare(
            &f.context,
            &f.pool,
            &program,
            &[Rc::clone(&ring)],
            &[ticket],
        )
        .expect("prepare")
    else {
        panic!("an empty ring has room for a put");
    };

    // The ring moves after composition: another producer published.
    // SAFETY: host-only test mutation of the tail word; nothing is in
    // flight.
    unsafe {
        ring.words()
            .contents()
            .cast::<u64>()
            .as_ptr()
            .add(1)
            .write(1);
    }

    let refusal = f
        .runtime
        .prepare_m3(
            &f.context,
            &f.pool,
            &f.externals,
            vec![LaneCandidate {
                fire,
                inputs: DeviceInputs::default(),
                retry_ineligible: false,
            }],
        )
        .expect_err("a stale lane cannot ride a group");
    assert!(
        refusal.to_string().contains("definitive host readiness"),
        "the abort must name the readiness check: {refusal}"
    );
}

#[test]
fn two_lanes_sharing_a_ring_are_refused_as_an_alias() {
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let plan = plan_of(package(0x93, true));
    let program = f
        .runtime
        .compile(&f.context, 5, &plan, VERSIONS, &kernels())
        .expect("compiles");
    let ring = Rc::new(Ring::new(&f.context, DType::F32, 1, 1).expect("ring"));
    let ticket = Ticket {
        expected_head: NO_TICKET,
        expected_tail: 0,
    };
    let mut fires = Vec::new();
    for _ in 0..2 {
        let Prepare::Ready(fire) = f
            .runtime
            .prepare(
                &f.context,
                &f.pool,
                &program,
                &[Rc::clone(&ring)],
                &[ticket],
            )
            .expect("prepare")
        else {
            panic!("ready");
        };
        fires.push(fire);
    }

    let refusal = f
        .runtime
        .prepare_m3(
            &f.context,
            &f.pool,
            &f.externals,
            fires
                .into_iter()
                .map(|fire| LaneCandidate {
                    fire,
                    inputs: DeviceInputs::default(),
                    retry_ineligible: false,
                })
                .collect(),
        )
        .expect_err("one ring under two lanes is an ordering hazard");
    assert!(
        refusal.to_string().contains("alias"),
        "the refusal must name the alias: {refusal}"
    );
}
