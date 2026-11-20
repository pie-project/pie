//! Under a stated bound on one GPU submission (`engine_gpu::submit`), with
//! a bound simulated here: every fire has an estimate before it goes out
//! (the probe, the smallest shape on the matmul curve, timed first, and
//! every larger shape grown from the ones measured, `engine_gpu::rates::
//! grown`), and a frame that follows the device idle re-times its clock
//! first: the probe fired alone, its time over its first the scale every
//! estimate is read at. Between two loads of growing matmuls the device is
//! slowed by [`SLOW`] (each fire, the probe's too, runs its matmul that many
//! times: a clock dropped while it idled, as an L40S's drops to 210 MHz):
//! re-timed, no submission's time on the device passes the bound; the same
//! frames admitted at the clock before the drop do (the L40S's ran 26 and 44
//! ms against a simulated 25 ms bound).

use engine::rates::Admission;
use engine_gpu::rates::{Fired, Work, grown};
use engine_gpu::submit::{Clock, PROBE, Probe};
use kernels_slang::Tensor;
use kernels_slang::linear::gemm::{self, Tactic};
use poem_ir::Dtype;

use crate::backend::Api;
use crate::gpu::Gpu;

/// How much slower the device runs after it idled: past twice [`MARGIN`],
/// so a frame filled to within half of its admitted room at the clock
/// before runs past the bound at the clock after.
const SLOW: u32 = 8;

/// The cubes a load walks up, smallest first.
const SIDES: [u32; 5] = [128, 256, 512, 1024, 2048];

/// The simulated bound, in probe times: the probe's first time is all the
/// test knows of the device before it fires anything else.
const BOUND_PROBES: f64 = 16.0;

/// The margin the test admits a submission at (`Admission::worst`): what a
/// fire's time may run past its estimate by on this device, for its noise.
const MARGIN: f64 = 2.0;

/// One submission: what it was and its estimate and time, ns.
#[derive(Debug)]
struct Submitted {
    what: String,
    estimate: f64,
    took: f64,
}

/// Two loads of growing cubes, the device slowed by [`SLOW`] between them,
/// each fire admitted under the simulated bound by its estimate. Each fire
/// is a frame of its own that waits for itself, so every one follows the
/// device idle: where `retime`, each re-times the clock first (as a rates
/// step's frames do); else every estimate is read at the clock the probe
/// first read. `None` where no device is asked.
fn loads(retime: bool) -> Option<(f64, Vec<Submitted>)> {
    let mut gpu = Gpu::open()?;
    let tactic = Tactic {
        tier: kernels_slang::Tier::Baseline,
        vector: 0,
        wide: false,
    };
    let mut probe = Probe::<Api>::new(&gpu.device, tactic).expect("the probe lays");
    // The probe first: the one fire before anything is measured.
    let first = probe.fire(&gpu.device).expect("the probe fires");
    // Waited for, the probe leaves the queue empty: the device reads idle.
    assert!(
        engine_gpu::backend::Context::idle(&gpu.device),
        "the device is idle once every submission landed"
    );
    let mut clock = Clock::new(first);
    let (m, n, k) = PROBE;
    let probed = Fired {
        work: Work::matmul(m, n, k),
        ns: first,
    };
    let bound = BOUND_PROBES * first;
    let admission = Admission {
        settled: true,
        worst: MARGIN,
        ..Admission::new(Some(std::time::Duration::from_nanos(bound as u64)))
    };
    let cubes: Vec<(u32, Tensor, Tensor, Tensor)> = SIDES
        .iter()
        .map(|&side| {
            let t = |gpu: &mut Gpu| {
                Tensor::new(
                    gpu.zeros((side * side) as usize),
                    side,
                    side,
                    Dtype::Bf16,
                )
            };
            (side, t(&mut gpu), t(&mut gpu), t(&mut gpu))
        })
        .collect();
    let slow = std::cell::Cell::new(1u32);
    let mut measured = vec![probed];
    let mut log = Vec::new();
    for load in 0..2 {
        if load == 1 {
            // The device idled and its clock dropped.
            slow.set(SLOW);
            probe.slow(&gpu.device, SLOW).expect("the probe re-recorded");
        }
        // The device is idle before each frame: re-timed first.
        let retimed = |clock: &mut Clock| {
            if retime {
                clock.probed(probe.fire(&gpu.device).expect("the probe fires"));
            }
        };
        // A frame of the widest cube's fires its estimate admits one of, as
        // many as it fits: the first submission after the drop, estimated
        // from what was measured before it.
        let frame = |clock: &mut Clock, measured: &mut Vec<Fired>, log: &mut Vec<Submitted>| {
            retimed(clock);
            let (side, act, w, y, per, fires) = cubes
                .iter()
                .rev()
                .find_map(|&(side, act, w, y)| {
                    let per = grown(measured, Some(probed), Work::matmul(side, side, side))?
                        * clock.ratio;
                    admission
                        .fires(64, Some(per))
                        .map(|fires| (side, act, w, y, per, fires))
                })
                .unwrap_or_else(|| panic!("no cube fits the bound: {log:#?}"));
            let took = gpu.timed(|c| {
                (0..fires * slow.get()).try_for_each(|_| gemm::matmul(c, tactic, act, w, y))
            });
            log.push(Submitted {
                what: format!("load {load}: a frame of {fires} {side}^3"),
                estimate: per * f64::from(fires),
                took,
            });
            measured.push(Fired {
                work: Work::matmul(side, side, side),
                ns: took / f64::from(fires) / clock.ratio,
            });
        };
        if load == 1 {
            frame(&mut clock, &mut measured, &mut log);
        }
        // The cubes walked up, smallest first, each estimated before it
        // goes out.
        for &(side, act, w, y) in &cubes {
            retimed(&mut clock);
            let work = Work::matmul(side, side, side);
            // Grown from what was measured, at the clock's ratio now: a
            // shape with no estimate is never fired under the bound.
            let estimate = grown(&measured, Some(probed), work).map(|ns| ns * clock.ratio);
            assert!(estimate.is_some(), "every cube is estimated");
            if !admission.fits(estimate) {
                break;
            }
            let took = gpu.timed(|c| {
                (0..slow.get()).try_for_each(|_| gemm::matmul(c, tactic, act, w, y))
            });
            log.push(Submitted {
                what: format!("load {load}: {side}^3"),
                estimate: estimate.unwrap_or(0.0),
                took,
            });
            measured.push(Fired {
                work,
                ns: took / clock.ratio,
            });
        }
        if load == 0 {
            frame(&mut clock, &mut measured, &mut log);
        }
    }
    Some((bound, log))
}

#[test]
fn a_frame_after_the_device_idled_is_re_timed_and_stays_under_the_bound() {
    // The control first: the same loads, the second admitted at the clock
    // the first ran at.
    let Some((bound, stale)) = loads(false) else {
        eprintln!("submit bound: no device asked");
        return;
    };
    let over: Vec<&Submitted> = stale.iter().filter(|s| s.took > bound).collect();
    assert!(
        !over.is_empty(),
        "without re-timing, a submission after the clock dropped runs past the {:.2} ms \
         bound: {stale:#?}",
        bound / 1e6
    );
    let (bound, retimed) = loads(true).expect("the device opened once");
    for s in &retimed {
        assert!(
            s.took <= bound,
            "{} took {:.2} ms against its estimate {:.2} ms, past the simulated {:.2} ms \
             bound: {retimed:#?}",
            s.what,
            s.took / 1e6,
            s.estimate / 1e6,
            bound / 1e6
        );
    }
    eprintln!(
        "submit bound {:.2} ms: re-timed {:?}; stale, over: {:?}",
        bound / 1e6,
        retimed
            .iter()
            .map(|s| (s.what.as_str(), s.took / 1e6))
            .collect::<Vec<_>>(),
        over.iter()
            .map(|s| (s.what.as_str(), s.took / 1e6))
            .collect::<Vec<_>>()
    );
}

/// Under a stated submit bound every serving step walks into a timed frame
/// (`Serve::walk_frame`), and its device time is read once it lands, where
/// its landing is already seen (`engine_gpu::submit::Measured`): the
/// backend's timestamps around its work, or (wgpu on an adapter without
/// timestamp queries) the host's clock from its submit to its work done.
/// A fire of a cube four times the probe's work reads a time past half
/// the probe's and within the host's clock around it, walked once or kept
/// and replayed, each fire writing the one arm's timestamps laid before.
#[test]
fn a_timed_step_reads_its_device_time_once_it_lands() {
    let Some(mut gpu) = Gpu::open() else {
        return;
    };
    let tactic = Tactic {
        tier: kernels_slang::Tier::Baseline,
        vector: 0,
        wide: false,
    };
    // 1024^3: four times the probe's 256 x 1024 x 1024.
    let side = 1024;
    let t = |gpu: &mut Gpu| {
        Tensor::new(gpu.zeros((side * side) as usize), side, side, Dtype::Bf16)
    };
    let (act, w, y) = (t(&mut gpu), t(&mut gpu), t(&mut gpu));
    let probe = Probe::<Api>::new(&gpu.device, tactic).expect("the probe lays");
    let probed = probe.fire(&gpu.device).expect("the probe fires");
    // Warmed: the cube's pipeline built and its operands touched.
    gpu.run(|c| gemm::matmul(c, tactic, act, w, y));
    // One arm's timestamps, laid once and written by every fire after,
    // walked and kept alike, one in flight at a time.
    let stamps = gpu.stamps();
    for keep in [false, true, false, true] {
        let started = std::time::Instant::now();
        let ns = gpu.fired(&stamps, keep, |c| gemm::matmul(c, tactic, act, w, y));
        let host = started.elapsed().as_nanos() as u64;
        let ns = ns.unwrap_or_else(|| panic!("a timed step's device time is read (keep {keep})"));
        assert!(ns > 0 && ns <= host, "keep {keep}: {ns} ns on the device, {host} ns on the host");
        // Four times the probe's work: read under half its time, the read
        // is not the cube's.
        assert!(
            ns as f64 >= probed * 0.5,
            "keep {keep}: the cube read {ns} ns, the probe {probed} ns"
        );
    }
}
