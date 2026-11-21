//! Where the device states a bound on one submission, a guest stage the
//! load's measured rates cost past it is refused at its registration by
//! name (`validate::bind_within` with `Budget::submit`): its fire would run
//! until the driver faults the submission, and every tenant's fire in it.

use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, Stage};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::{Budget, SubmitBound, ValidateError, bind_within, footprints};

/// An epilogue scattering `slots` floats into a `n`-wide base taken off
/// channels: the closed set's scatter visits every index for every base
/// element, `n × slots`.
fn scatter(n: u32, slots: u32) -> TraceContainer {
    let chan = |shape, dtype| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role: HostRole::Writer,
        seeded: false,
    };
    TraceContainer {
        channels: vec![
            chan(Shape::vector(n), Dtype::F32),
            chan(Shape::vector(slots), Dtype::I32),
            chan(Shape::vector(slots), Dtype::F32),
            ChannelDecl {
                host_role: HostRole::Reader,
                ..chan(Shape::vector(n), Dtype::F32)
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::ChanTake(2),
                Op::ScatterAdd {
                    base: 0,
                    idx: 1,
                    vals: 2,
                },
                Op::ChanPut { chan: 3, value: 3 },
            ],
        }],
        ..TraceContainer::default()
    }
}

/// A phone's: KGSL's 2 s bound, a 10 us launch, 50 GB/s.
const PHONE: SubmitBound = SubmitBound {
    bound_ns: 2_000_000_000,
    launch_ns: 10_000.0,
    elementwise_bytes_per_ns: 50.0,
};

fn within(submit: Option<SubmitBound>) -> Budget {
    Budget {
        room_bytes: u64::MAX,
        submit,
    }
}

#[test]
fn a_stage_past_the_submit_bound_is_refused_by_name() {
    // A sampler's scatter: a vocabulary's base, a few slots. It fits.
    let small = bind_within(scatter(151_936, 16), vec![], EngineCaps::default(), within(Some(PHONE)))
        .expect("a vocabulary scatter of 16 slots fits 2 s");
    let footprint = footprints(&small).unwrap()[0];
    assert_eq!(footprint.launches, 1, "only the scatter launches");
    assert!(footprint.work >= 151_936 * 4 * 16);

    // A million slots into a million-wide base: 4 TB of visits at 50 GB/s.
    let refused = bind_within(
        scatter(1 << 20, 1 << 20),
        vec![],
        EngineCaps::default(),
        within(Some(PHONE)),
    )
    .unwrap_err();
    let ValidateError::StagePastSubmit {
        stage,
        cost_ns,
        bound_ns,
    } = refused.clone()
    else {
        panic!("{refused}");
    };
    assert_eq!((stage, bound_ns), (Stage::Epilogue, PHONE.bound_ns));
    assert!(cost_ns > 80_000_000_000.0, "{cost_ns}");
    assert!(refused.to_string().contains("submission"), "{refused}");

    // A device stating no bound holds the stage to none.
    assert!(
        bind_within(scatter(1 << 20, 1 << 20), vec![], EngineCaps::default(), within(None)).is_ok()
    );
}
