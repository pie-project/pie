//! How a step's answer lands. Nothing blocks on the GPU (the browser cannot,
//! and native follows it), so the readout seat is read by callback — the queue reports the work done,
//! a staging copy is mapped — and the rows are posted to a mailbox the shell
//! empties the next time it is touched, or parks on when it needs a step that
//! is still airborne (`Shell::harvest_one`). The callback owns clones of all
//! it reads, so it never needs the shell.
//!
//! A step with attached guests runs them here too, on the device, before its
//! rows are posted and its sink told: the landing is handed to a worker green
//! thread (`Worker`), which can park on the guest stage's readback the way
//! the engine lane would have, while the lane itself goes on submitting the
//! next frame. Landings pass through the worker in the order they arrive, so
//! guests of consecutive steps run in step order and the mailbox sees the
//! steps in sequence; a landing with no guests skips the worker when it is
//! idle.

use std::sync::{Arc, Mutex};

use crate::device::{Buffer, OnDone};
use crate::error::Fault;
use crate::settle::{Airborne, Done};

/// The attached guests a step runs from its landing; a build without a
/// device has none to run.
#[cfg(feature = "wgpu")]
pub type Guests = crate::program::Guests;
#[cfg(not(feature = "wgpu"))]
pub enum Guests {}

/// One step's readout seat, cloned for its callback: the seat the guests
/// widen from, and the mirror of its rows the callback maps (with the draft
/// rows' mirror and width, when the model drafts).
pub(crate) struct Seat {
    pub seq: u64,
    pub arm: usize,
    pub layout: Vec<(u32, u32)>,
    pub width: u32,
    pub seat: Buffer,
    pub mirror: Buffer,
    pub draft: Option<(u32, Buffer)>,
    pub guests: Option<Guests>,
}

/// A step whose rows are back, or whose read failed (`failed`): either way
/// its arm is free again.
pub(crate) struct Landed {
    pub seq: u64,
    pub arm: usize,
    pub layout: Vec<(u32, u32)>,
    pub rows: Vec<Vec<f32>>,
    pub drafts: Vec<Vec<f32>>,
    pub failed: bool,
}

/// What the callbacks post and the shell empties, plus the waker of an engine
/// lane parked in `harvest` waiting for the next landing.
#[derive(Default)]
pub(crate) struct Post {
    pub landed: Vec<Landed>,
    pub waker: Option<std::task::Waker>,
}

pub(crate) type Mailbox = Arc<Mutex<Post>>;

type Rows = (Vec<Vec<f32>>, Vec<Vec<f32>>);

fn read_seat(
    layout: Vec<(u32, u32)>,
    width: u32,
    seat: &Buffer,
    draft: Option<(u32, Buffer)>,
    on_read: impl FnOnce(Result<Rows, Fault>) + Send + 'static,
) {
    let total: usize = layout.iter().map(|&(_, n)| n as usize).sum();
    let width = width as usize;
    seat.read_async(0, (total * width * 2) as u64, move |raw| {
        let rows = match raw {
            Ok(raw) => crate::serve::rows_from(&raw, &layout, width),
            Err(fault) => return on_read(Err(fault)),
        };
        let Some((draft_width, draft)) = draft else {
            return on_read(Ok((rows, Vec::new())));
        };
        let draft_width = draft_width as usize;
        draft.read_async(0, (total * draft_width * 2) as u64, move |raw| {
            on_read(raw.map(|raw| (rows, crate::serve::rows_from(&raw, &layout, draft_width))));
        });
    });
}

/// One step's landing as the worker takes it: the rows as read (or the
/// refusal), the guests still to run on them, and whom to tell after.
struct Landing {
    seq: u64,
    arm: usize,
    layout: Vec<(u32, u32)>,
    width: u32,
    seat: Buffer,
    mtp_width: u32,
    read: Result<Rows, String>,
    guests: Option<Guests>,
    mailbox: Mailbox,
    counts: Airborne,
    done: Option<Done>,
}

impl Landing {
    /// Runs the step's guests on its rows, from the worker's green thread. A
    /// step whose read failed has nothing for them: they are dropped unrun,
    /// which frees their gates.
    fn run_guests(&mut self, worker: &Worker) -> Option<String> {
        let guests = self.guests.take()?;
        let Ok((rows, drafts)) = &self.read else {
            return None;
        };
        #[cfg(feature = "wgpu")]
        {
            let jobs = guests.jobs(
                rows,
                drafts,
                &self.seat,
                &self.layout,
                self.width,
                self.mtp_width,
            );
            tracing::debug!(seq = self.seq, guests = jobs.len(), "landing: guests run");
            worker
                .run_all(jobs)
                .err()
                .map(|refusal| refusal.to_string())
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (rows, drafts, self.width, &self.seat, self.mtp_width, worker);
            match guests {}
        }
    }

    /// Posts the rows to the mailbox and tells the sink; `guest_fault` is a
    /// guest's refusal, which faults the step without failing its read.
    fn finish(self, guest_fault: Option<String>) {
        let Landing {
            seq,
            arm,
            layout,
            read,
            mailbox,
            counts,
            done,
            ..
        } = self;
        tracing::debug!(seq, arm, ok = read.is_ok(), "landing: rows back");
        let (rows, drafts, refused, failed) = match read {
            Ok((rows, drafts)) => (rows, drafts, guest_fault, false),
            Err(why) => (Vec::new(), Vec::new(), Some(why), true),
        };
        if let Some(why) = &refused {
            tracing::warn!(seq, arm, failed, %why, "landing: the step faulted");
        }
        let parked = {
            let mut post = mailbox
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            post.landed.push(Landed {
                seq,
                arm,
                layout,
                rows,
                drafts,
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
            let outcome = match refused {
                None => engine::StepOutcome::Committed,
                Some(why) => engine::StepOutcome::Faulted(format!(
                    "wgpu command buffer for frame {} step {}: {why}",
                    done.at.frame, done.at.step
                )),
            };
            (done.sink)(done.at, outcome);
        }
    }
}

/// The landing worker: one green thread, started on first need, that takes
/// landings in arrival order, runs each step's guests and finishes it. A
/// step's guests run at once, one per runner green thread (started as the
/// widest step needs them and kept), so their device round trips overlap.
#[derive(Clone, Default)]
pub(crate) struct Worker {
    inner: Arc<Mutex<WorkerState>>,
}

/// Work for a runner thread.
#[cfg(feature = "wgpu")]
type Runnable = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct WorkerState {
    tx: Option<crate::device::host::channel::Sender<Landing>>,

    /// Landings handed to the thread and not yet finished.
    queued: usize,

    #[cfg(feature = "wgpu")]
    runners: Vec<crate::device::host::channel::Sender<Runnable>>,
}

impl Worker {
    /// Runs a step's guests together and waits for every one of them; the
    /// first refusal is the verdict. One guest runs right here, on the
    /// worker's own thread.
    #[cfg(feature = "wgpu")]
    fn run_all(&self, mut jobs: Vec<crate::program::Job>) -> Result<(), crate::program::Refusal> {
        if jobs.len() == 1 {
            return jobs.pop().expect("one job")();
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
        let mut verdict = Ok(());
        for _ in 0..expected {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(refusal)) => {
                    if verdict.is_ok() {
                        verdict = Err(refusal);
                    }
                }
                Err(_) => break,
            }
        }
        verdict
    }

    /// The `at`th runner thread, started if it is not yet.
    #[cfg(feature = "wgpu")]
    fn runner(&self, at: usize) -> crate::device::host::channel::Sender<Runnable> {
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.runners.len() <= at {
            let (tx, rx) = crate::device::host::channel::unbounded::<Runnable>();
            crate::device::host::thread::spawn(
                format!("wgpu guest {}", state.runners.len()),
                move || {
                    while let Ok(run) = rx.recv() {
                        run();
                    }
                },
            )
            .expect("spawn a guest runner");
            state.runners.push(tx);
        }
        state.runners[at].clone()
    }
    /// Finishes a landing: inline when nothing is queued and there are no
    /// guests to run, else behind whatever the thread still holds.
    fn land(&self, landing: Landing) {
        let inline = {
            let mut state = self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.queued == 0 && landing.guests.is_none() {
                Some(landing)
            } else {
                let tx = state.tx.get_or_insert_with(|| self.start());
                match tx.send(landing) {
                    Ok(()) => {
                        state.queued += 1;
                        None
                    }
                    // The thread is gone (its receiver dropped); finish here
                    // rather than lose the step, guests unrun.
                    Err(crate::device::host::channel::SendError(landing)) => Some(landing),
                }
            }
        };
        if let Some(landing) = inline {
            landing.finish(None);
        }
    }

    fn start(&self) -> crate::device::host::channel::Sender<Landing> {
        let (tx, rx) = crate::device::host::channel::unbounded::<Landing>();
        let worker = self.clone();
        crate::device::host::thread::spawn("wgpu landing".into(), move || {
            while let Ok(mut landing) = rx.recv() {
                let fault = landing.run_guests(&worker);
                landing.finish(fault);
                worker
                    .inner
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .queued -= 1;
            }
        })
        .expect("spawn the landing worker");
        tx
    }
}

/// The completion chain for one step: work done, rows read, guests run, the
/// mailbox posted, the sink told.
pub(crate) fn on_done(
    seat: Seat,
    mailbox: Mailbox,
    counts: Airborne,
    done: Option<Done>,
    worker: Worker,
) -> OnDone {
    let Seat {
        seq,
        arm,
        layout,
        width,
        seat,
        mirror,
        draft,
        guests,
    } = seat;
    let mtp_width = draft.as_ref().map_or(0, |&(width, _)| width);
    let read_layout = layout.clone();
    // The step's verdict: the map resolves after its work ran, and a refused
    // submission has reached the error sink by then.
    let verdict = mirror.clone();
    let land = move |read: Result<Rows, String>| {
        worker.land(Landing {
            seq,
            arm,
            layout,
            width,
            seat,
            mtp_width,
            read,
            guests,
            mailbox,
            counts,
            done,
        });
    };
    Box::new(move |refused: Option<String>| {
        tracing::debug!(seq, refused = refused.is_some(), "landing: work done");
        match refused {
            Some(why) => land(Err(why)),
            None => read_seat(read_layout, width, &mirror, draft, move |read| {
                let read = read.and_then(|rows| verdict.queue_verdict().map(|()| rows));
                land(read.map_err(|fault| fault.to_string()));
            }),
        }
    })
}
