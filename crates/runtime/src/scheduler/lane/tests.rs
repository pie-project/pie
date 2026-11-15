//! The lane parks as soon as nothing is queued: a request posted from the
//! empty check on wakes it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::{Lane, LaneRequest, LaneTurn, LaunchReplyRx};
use crate::scheduler::stats::SchedulerStats;

/// A request no engine has to serve.
fn request() -> LaneRequest {
    let (response, _) = crate::rt::channel::bounded(1);
    LaneRequest::Shutdown { response }
}

/// Each request is posted once the lane has taken the last and gone back to
/// wait: the lane is then between its empty checks and its park, or parked.
/// It takes every one, on either queue.
#[test]
fn a_lane_about_to_park_or_parked_takes_every_request_posted() {
    const REQUESTS: usize = 20_000;
    let (launch_tx, launch_rx) = crate::rt::channel::unbounded::<LaneRequest>();
    let (control_tx, control_rx) = crate::rt::channel::unbounded::<LaneRequest>();
    let taken = Arc::new(AtomicUsize::new(0));
    let lane = {
        let taken = taken.clone();
        std::thread::spawn(move || {
            let mut turn = LaneTurn::default();
            while Lane::next_request(&launch_rx, &control_rx, &mut turn).is_ok() {
                taken.fetch_add(1, Ordering::SeqCst);
            }
        })
    };
    let until = std::time::Instant::now() + Duration::from_secs(60);
    for posted in 0..=REQUESTS {
        // The lane has taken every earlier request.
        while taken.load(Ordering::SeqCst) < posted {
            assert!(
                std::time::Instant::now() < until,
                "request {posted} was never taken: a lost wake"
            );
            std::hint::spin_loop();
        }
        if posted == REQUESTS {
            break;
        }
        let queue = if posted % 2 == 0 {
            &launch_tx
        } else {
            &control_tx
        };
        queue.send(request()).expect("the lane's queue is open");
    }
    drop((launch_tx, control_tx));
    lane.join().expect("the lane ends once both queues close");
}

/// Answers every fire with an empty ticket.
struct Fires(Arc<AtomicUsize>);

impl ::engine::Engine for Fires {
    fn kind(&self) -> &'static str {
        "fires"
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(::engine::FrameTicket {
            id: ::engine::FrameId(0),
            steps: Vec::new(),
        })
    }
}

fn spawn(engine_idx: usize, fired: &Arc<AtomicUsize>) -> Lane {
    Lane::spawn(
        engine_idx,
        Some(Box::new(Fires(fired.clone()))),
        Arc::new(SchedulerStats::default()),
    )
}

fn fire(lane: &Lane) -> LaunchReplyRx {
    lane.fire(
        crate::engine::FrameFire {
            steps: vec![crate::engine::StepFire {
                submission: ::engine::Step {
                    lanes: vec![::engine::Lane {
                        tokens: vec![0],
                        ..::engine::Lane::default()
                    }],
                    attachments: Vec::new(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                },
                terminal_cells: Vec::new(),
                completions: Vec::new(),
                instances: Vec::new(),
            }],
        },
        false,
    )
}

fn answer(reply: LaunchReplyRx) -> Result<(), String> {
    crate::rt::block_on(async { crate::rt::time::timeout(Duration::from_secs(10), reply).await })
        .expect("a fire is answered")
        .expect("a fire is answered")
        .map(|_| ())
}

/// The nanoseconds the lane thread of `engine_idx` has run on a CPU
/// (`/proc/self/task/<tid>/schedstat`, first field).
#[cfg(target_os = "linux")]
fn lane_cpu_ns(engine_idx: usize) -> u64 {
    let name = format!("pie-engine-{engine_idx}\n");
    let task = std::fs::read_dir("/proc/self/task")
        .expect("this process's threads")
        .filter_map(|entry| entry.ok())
        .find(|entry| {
            std::fs::read_to_string(entry.path().join("comm")).is_ok_and(|comm| comm == name)
        })
        .expect("the lane's thread");
    std::fs::read_to_string(task.path().join("schedstat"))
        .expect("the lane thread's schedstat")
        .split_whitespace()
        .next()
        .and_then(|ns| ns.parse().ok())
        .expect("its on-CPU nanoseconds")
}

/// A lane with nothing queued is parked: it runs on no CPU while it waits,
/// and a fire posted then is answered.
#[cfg(target_os = "linux")]
#[test]
fn a_lane_with_nothing_queued_runs_on_no_cpu_and_wakes_for_a_fire() {
    const ENGINE: usize = 9106;
    let fired = Arc::new(AtomicUsize::new(0));
    let mut lane = spawn(ENGINE, &fired);
    // The wait starts right after a fire, where a lane that spun for its
    // next request would be spinning.
    answer(fire(&lane)).expect("the lane serves");
    let began = std::time::Instant::now();
    let before = lane_cpu_ns(ENGINE);
    std::thread::sleep(Duration::from_millis(200));
    let ran = Duration::from_nanos(lane_cpu_ns(ENGINE) - before);
    let waited = began.elapsed();
    assert!(
        ran * 2 < waited,
        "the lane ran {ran:?} of the {waited:?} it had nothing to do"
    );
    answer(fire(&lane)).expect("a parked lane wakes for a fire");
    assert_eq!(fired.load(Ordering::SeqCst), 2);
    lane.shutdown();
}

/// Each fire is posted once the last is answered, so the lane is parking
/// or parked when it comes: every one is answered.
#[test]
fn a_lane_that_parks_between_fires_answers_every_fire() {
    const ENGINE: usize = 9107;
    const FIRES: usize = 2_000;
    let fired = Arc::new(AtomicUsize::new(0));
    let mut lane = spawn(ENGINE, &fired);
    for _ in 0..FIRES {
        answer(fire(&lane)).expect("a fire posted to a parking lane is answered");
    }
    assert_eq!(fired.load(Ordering::SeqCst), FIRES);
    lane.shutdown();
}
