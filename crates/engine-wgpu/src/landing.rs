use std::sync::{Arc, Mutex};

use crate::device::{Buffer, OnDone};
use crate::error::Fault;
use crate::settle::{Airborne, Done};

#[cfg(feature = "wgpu")]
pub type Guests = crate::program::Guests;
#[cfg(not(feature = "wgpu"))]
pub enum Guests {}

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

pub(crate) struct Landed {
    pub seq: u64,
    pub arm: usize,
    pub layout: Vec<(u32, u32)>,
    pub rows: Vec<Vec<f32>>,
    pub drafts: Vec<Vec<f32>>,
    pub failed: bool,
}

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
