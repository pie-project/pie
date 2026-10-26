use std::sync::{Arc, Mutex};

use crate::device::{Buffer, OnDone};
use crate::error::Fault;
use crate::serve::ReadSpan;
use crate::settle::{Airborne, Done};

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

fn read_seat(
    layout: Vec<ReadSpan>,
    seat: &Buffer,
    on_read: impl FnOnce(Result<Rows, Fault>) + Send + 'static,
) {
    let end = layout
        .iter()
        .filter(|span| span.host)
        .map(|span| span.at + span.bytes())
        .max()
        .unwrap_or(0);
    seat.read_async(0, end, move |raw| {
        on_read(raw.map(|raw| crate::serve::rows_from(&raw, &layout)));
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
}

impl Landing {
    fn run_guests(&mut self, worker: &Worker) -> Option<String> {
        let guests = self.guests.take()?;
        if self.read.is_err() {
            return None;
        }
        #[cfg(feature = "wgpu")]
        {
            let jobs = guests.jobs(&self.seat, &self.layout, self.pixels.as_ref());
            tracing::debug!(seq = self.seq, guests = jobs.len(), "landing: guests run");
            worker
                .run_all(jobs)
                .err()
                .map(|refusal| refusal.to_string())
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (&self.layout, &self.seat, &self.pixels, worker);
            match guests {}
        }
    }

    fn finish(self, guest_fault: Option<String>) {
        let Landing {
            seq,
            arm,
            read,
            mailbox,
            counts,
            done,
            ..
        } = self;
        tracing::debug!(seq, arm, ok = read.is_ok(), "landing: rows back");
        let (rows, refused, failed) = match read {
            Ok(rows) => (rows, guest_fault, false),
            Err(why) => (Vec::new(), Some(why), true),
        };
        if let Some(why) = &refused {
            tracing::warn!(seq, arm, failed, %why, "landing: the step faulted");
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

#[derive(Clone, Default)]
pub(crate) struct Worker {
    inner: Arc<Mutex<WorkerState>>,
}

#[cfg(feature = "wgpu")]
type Runnable = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct WorkerState {
    tx: Option<crate::device::host::channel::Sender<Landing>>,

    queued: usize,

    #[cfg(feature = "wgpu")]
    runners: Vec<crate::device::host::channel::Sender<Runnable>>,
}

impl Worker {
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

    #[cfg(feature = "wgpu")]
    fn runner(&self, at: usize) -> crate::device::host::channel::Sender<Runnable> {
        let mut state = self.inner.lock().expect("lock poisoned");
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
    fn land(&self, landing: Landing) {
        let inline = {
            let mut state = self.inner.lock().expect("lock poisoned");
            if state.queued == 0 && landing.guests.is_none() {
                Some(landing)
            } else {
                let tx = state.tx.get_or_insert_with(|| self.start());
                match tx.send(landing) {
                    Ok(()) => {
                        state.queued += 1;
                        None
                    }
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
                worker.inner.lock().expect("lock poisoned").queued -= 1;
            }
        })
        .expect("spawn the landing worker");
        tx
    }
}

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
        seat,
        mirror,
        pixels,
        guests,
    } = seat;
    let read_layout = layout.clone();
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
        });
    };
    Box::new(move |refused: Option<String>| {
        tracing::debug!(seq, refused = refused.is_some(), "landing: work done");
        match refused {
            Some(why) => land(Err(why)),
            None => read_seat(read_layout, &mirror, move |read| {
                let read = read.and_then(|rows| verdict.queue_verdict().map(|()| rows));
                land(read.map_err(|fault| fault.to_string()));
            }),
        }
    })
}
