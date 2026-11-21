use std::sync::{Arc, Mutex};

use crate::device::{Buffer, OnDone};
use engine_gpu::error::Fault;
use engine_gpu::serve::ReadSpan;
use engine_gpu::settle::{Airborne, Done, merged};

#[cfg(feature = "wgpu")]
pub type Guests = crate::program::Guests;
#[cfg(not(feature = "wgpu"))]
pub enum Guests {}

pub(crate) struct Seat {
    pub seq: u64,
    pub arm: usize,
    pub layout: Vec<ReadSpan>,
    pub seat: Buffer,
    pub mirror: Buffer,
    /// The fire's pixel plane, its output and each lane's (byte, element
    /// count) span of it.
    pub pixels: Option<Pixels>,
    pub guests: Option<Guests>,
    /// Where the fire's beam fault words lie in the readback, past its rows.
    pub beam_tail: Option<engine_gpu::serve::BeamTail>,
}

pub(crate) type Pixels = (Buffer, String, Vec<(u64, u32)>);

pub(crate) struct Landed {
    pub seq: u64,
    pub arm: usize,
    pub rows: Vec<Vec<f32>>,
    pub failed: bool,
}

#[derive(Default)]
pub(crate) struct Post {
    pub landed: Vec<Landed>,
    pub waker: Option<std::task::Waker>,
}

pub(crate) type Mailbox = Arc<Mutex<Post>>;

type Rows = Vec<Vec<f32>>;

/// The readback's rows, and the beam fault words past them where the fire
/// laid any (`tail`), in one read.
fn read_seat(
    layout: Vec<ReadSpan>,
    tail: Option<engine_gpu::serve::BeamTail>,
    seat: &Buffer,
    on_read: impl FnOnce(Result<(Rows, Vec<u32>), Fault>) + Send + 'static,
) {
    let rows_end = layout
        .iter()
        .filter(|span| span.hosted)
        .map(|span| span.at + span.bytes())
        .max()
        .unwrap_or(0);
    let end = tail.map_or(rows_end, |tail| {
        rows_end.max(tail.at + u64::from(tail.words) * 4)
    });
    seat.read_async(0, end, move |raw| {
        on_read(raw.map(|raw| {
            let words = tail.map_or_else(Vec::new, |tail| {
                let from = usize::try_from(tail.at).unwrap_or(usize::MAX);
                raw.get(from..from + tail.words as usize * 4)
                    .unwrap_or(&[])
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect()
            });
            (engine_gpu::serve::rows_from(&raw, &layout), words)
        }));
    });
}

struct Landing {
    seq: u64,
    arm: usize,
    layout: Vec<ReadSpan>,
    seat: Buffer,
    pixels: Option<Pixels>,
    read: Result<Rows, String>,
    guests: Option<Guests>,
    mailbox: Mailbox,
    counts: Airborne,
    done: Option<Done>,
    fed: Option<engine_gpu::settle::FedCheck>,
}

impl Landing {
    /// Runs the step's guest passes, each on its own: the lanes of each
    /// pass that faulted.
    fn run_guests(&mut self, worker: &Worker) -> Vec<engine::LaneFault> {
        let Some(guests) = self.guests.take() else {
            return Vec::new();
        };
        if self.read.is_err() {
            return Vec::new();
        }
        #[cfg(feature = "wgpu")]
        {
            let jobs = guests.jobs(&self.seat, &self.layout, self.pixels.as_ref());
            tracing::debug!(seq = self.seq, guests = jobs.len(), "landing: guests run");
            let mut faulted = engine::StepOutcome::committed();
            for crate::program::Faulted {
                lanes,
                instance,
                refusal,
            } in worker.run_all(jobs)
            {
                faulted.fault_lanes(lanes, Some(instance), &refusal.to_string());
            }
            faulted.faulted
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (&self.layout, &self.seat, &self.pixels, worker);
            match guests {}
        }
    }

    fn finish(self, guest_faults: Vec<engine::LaneFault>) {
        let Landing {
            seq,
            arm,
            read,
            mailbox,
            counts,
            done,
            fed,
            ..
        } = self;
        tracing::debug!(seq, arm, ok = read.is_ok(), "landing: rows back");
        // What the fire embedded is read back before the step is posted:
        // its arm may be taken again then.
        if let Some(fed) = &fed {
            fed.landed(read.is_ok());
        }
        let (rows, refused, failed) = match read {
            Ok(rows) => (rows, None, false),
            Err(why) => (Vec::new(), Some(why), true),
        };
        if let Some(why) = &refused {
            tracing::warn!(seq, arm, %why, "landing: the step was lost");
        }
        for fault in &guest_faults {
            tracing::warn!(seq, arm, lane = fault.lane, why = %fault.reason, "landing: a guest pass faulted its lane");
        }
        let parked = {
            let mut post = mailbox.lock().expect("mailbox lock poisoned");
            post.landed.push(Landed {
                seq,
                arm,
                rows,
                failed,
            });
            post.waker.take()
        };
        if let Some(waker) = parked {
            waker.wake();
        }
        counts.leave();
        tracing::debug!(
            seq,
            has_sink = done.is_some(),
            "landing: posted to the mailbox"
        );
        if let Some(done) = done {
            match refused {
                Some(why) => done.tell(Some(why), "wgpu command buffer"),
                None => (done.sink)(done.at, merged(done.settled(), guest_faults)),
            }
        }
    }
}

/// The landing thread and the guest runners, all started at load: made only
/// by [`Worker::new`].
#[derive(Clone)]
pub(crate) struct Worker {
    inner: Arc<Mutex<WorkerState>>,
}

#[cfg(feature = "wgpu")]
type Runnable = Box<dyn FnOnce() + Send + 'static>;

struct WorkerState {
    tx: crate::device::host::channel::Sender<Landing>,

    queued: usize,

    #[cfg(feature = "wgpu")]
    runners: Runners,
}

/// The threads guest passes run on beside the landing's own, at least one:
/// a job `at` runs on runner `at` mod their count.
#[cfg(feature = "wgpu")]
struct Runners(Vec<crate::device::host::channel::Sender<Runnable>>);

#[cfg(feature = "wgpu")]
impl Runners {
    fn start(count: std::num::NonZeroUsize) -> Runners {
        Runners(
            (0..count.get())
                .map(|at| {
                    let (tx, rx) = crate::device::host::channel::unbounded::<Runnable>();
                    crate::device::host::thread::spawn(format!("wgpu guest {at}"), move || {
                        while let Ok(run) = rx.recv() {
                            run();
                        }
                    })
                    .expect("spawn a guest runner");
                    tx
                })
                .collect(),
        )
    }

    fn of(&self, at: usize) -> crate::device::host::channel::Sender<Runnable> {
        self.0[at % self.0.len()].clone()
    }
}

/// The guest runners: one for each of the host's cores (a guest job holds
/// its own instance and no lane, so jobs past them queue).
#[cfg(feature = "wgpu")]
#[must_use]
pub(crate) fn runners() -> std::num::NonZeroUsize {
    std::thread::available_parallelism().unwrap_or(std::num::NonZeroUsize::MIN)
}

/// The threads that run guest passes and so widen: the runners, the
/// landing's (a step's one job runs there), and the engine's (prologues).
#[cfg(feature = "wgpu")]
#[must_use]
pub(crate) fn guest_threads() -> usize {
    runners().get() + 2
}

impl Worker {
    /// The landing thread and its guest runners, all started now (at load).
    pub(crate) fn new() -> Worker {
        let (tx, rx) = crate::device::host::channel::unbounded::<Landing>();
        let worker = Worker {
            inner: Arc::new(Mutex::new(WorkerState {
                tx,
                queued: 0,
                #[cfg(feature = "wgpu")]
                runners: Runners::start(runners()),
            })),
        };
        worker.start(rx);
        worker
    }

    #[cfg(feature = "wgpu")]
    fn run_all(&self, mut jobs: Vec<crate::program::Job>) -> Vec<crate::program::Faulted> {
        if jobs.len() == 1 {
            return jobs.pop().expect("one job")().err().into_iter().collect();
        }
        let (tx, rx) = crate::device::host::channel::unbounded();
        let expected = jobs.len();
        for (at, job) in jobs.into_iter().enumerate() {
            let tx = tx.clone();
            let runnable: Runnable = Box::new(move || {
                let _ = tx.send(job());
            });
            if let Err(crate::device::host::channel::SendError(runnable)) =
                self.runner(at).send(runnable)
            {
                runnable();
            }
        }
        drop(tx);
        let mut faulted = Vec::new();
        for _ in 0..expected {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(fault)) => faulted.push(fault),
                Err(_) => break,
            }
        }
        faulted
    }

    #[cfg(feature = "wgpu")]
    fn runner(&self, at: usize) -> crate::device::host::channel::Sender<Runnable> {
        self.inner.lock().expect("lock poisoned").runners.of(at)
    }
    fn land(&self, landing: Landing) {
        let inline = {
            let mut state = self.inner.lock().expect("lock poisoned");
            if state.queued == 0 && landing.guests.is_none() {
                Some(landing)
            } else {
                match state.tx.send(landing) {
                    Ok(()) => {
                        state.queued += 1;
                        None
                    }
                    Err(crate::device::host::channel::SendError(landing)) => Some(landing),
                }
            }
        };
        if let Some(landing) = inline {
            landing.finish(Vec::new());
        }
    }

    fn start(&self, rx: crate::device::host::channel::Receiver<Landing>) {
        let worker = self.clone();
        crate::device::host::thread::spawn("wgpu landing".into(), move || {
            while let Ok(mut landing) = rx.recv() {
                let fault = landing.run_guests(&worker);
                landing.finish(fault);
                worker.inner.lock().expect("lock poisoned").queued -= 1;
            }
        })
        .expect("spawn the landing worker");
    }
}

pub(crate) fn on_done(
    seat: Seat,
    mailbox: Mailbox,
    counts: Airborne,
    done: Option<Done>,
    fed: Option<engine_gpu::settle::FedCheck>,
    worker: Worker,
) -> OnDone {
    let Seat {
        seq,
        arm,
        layout,
        seat,
        mirror,
        pixels,
        guests,
        beam_tail,
    } = seat;
    let read_layout = layout.clone();
    let supplies = fed.clone().filter(engine_gpu::settle::FedCheck::reads_beams);
    let verdict = mirror.clone();
    let land = move |read: Result<Rows, String>| {
        worker.land(Landing {
            seq,
            arm,
            layout,
            seat,
            pixels,
            read,
            guests,
            mailbox,
            counts,
            done,
            fed,
        });
    };
    Box::new(move |refused: Option<String>| {
        tracing::debug!(seq, refused = refused.is_some(), "landing: work done");
        match refused {
            Some(why) => land(Err(why)),
            None => read_seat(read_layout, beam_tail, &mirror, move |read| {
                let read = read.map(|(rows, words)| {
                    if let Some(fed) = &supplies
                        && beam_tail.is_some()
                    {
                        fed.supply(beam_tail.map_or(0, |tail| tail.word), words);
                    }
                    rows
                });
                let read = read.and_then(|rows| verdict.queue_verdict().map(|()| rows));
                land(read.map_err(|fault| fault.to_string()));
            }),
        }
    })
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use engine::InstanceId;
    use engine_gpu::program::Refusal;

    use super::Worker;
    use crate::program::{Faulted, Job};

    /// A step's guest passes run each on its own: one faulting does not
    /// stop the others, and each that faulted is told with its own lanes
    /// and instance, not the first fault for the whole step.
    #[test]
    fn each_guest_pass_faults_its_own_lanes() {
        let ran = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let jobs: Vec<Job> = (0..4u32)
            .map(|lane| {
                let ran = std::sync::Arc::clone(&ran);
                let job: Job = Box::new(move || {
                    ran.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    if lane == 1 || lane == 3 {
                        Err(Faulted {
                            lanes: lane..lane + 1,
                            instance: InstanceId(100 + u64::from(lane)),
                            refusal: Refusal::Program(format!("lane {lane}: guard")),
                        })
                    } else {
                        Ok(())
                    }
                });
                job
            })
            .collect();
        let mut faulted = Worker::new().run_all(jobs);
        faulted.sort_by_key(|fault| fault.lanes.start);
        assert_eq!(ran.load(std::sync::atomic::Ordering::Acquire), 4, "every pass ran");
        let told: Vec<(u32, u64)> = faulted
            .iter()
            .map(|fault| (fault.lanes.start, fault.instance.0))
            .collect();
        assert_eq!(told, [(1, 101), (3, 103)]);
    }
}
