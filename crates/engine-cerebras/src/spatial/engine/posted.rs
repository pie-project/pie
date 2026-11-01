//! The asynchronous engine (`[cerebras.spatial] async`, on by default): a
//! frame's steps are posted, not run, and each lands on the completion sink
//! once its epilogue ran. Steps run in order on the lane's thread, as far as
//! the wafer has answered them; while the host runs a token loop's answered
//! rows through their epilogues, the wafer runs the loop on (a
//! continuation over the slots the lanes' posted fires hold pages for).
//! Rows a continuation answered that a lane's stream does not send (a
//! sampled token, a stop) are dropped as a synchronous loop's are, the lane
//! falling back to a launch of its own.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, mpsc};

use engine::fire::{FrameId, Step};
use engine::{StepDone, StepOutcome};

use super::Spatial;
use crate::spatial::{serve2d, wafer};

/// A step posted and not yet run.
pub(super) struct Posted {
    pub(super) at: StepDone,
    pub(super) step: Step,
}

/// A continuation on the wafer: per lane (its sequence slot, its loop's
/// epoch, the first slot it writes), all `steps` iterations.
pub(super) struct InFlight {
    lanes: Vec<(u32, u64, u32)>,
    steps: usize,
    pages: Vec<Vec<u32>>,
    answer: mpsc::Receiver<wafer::BatchAnswer>,
    landing: Arc<Mutex<Landing>>,
}

/// Whether a continuation landed, and the step waiting on it (reported
/// landed by the wafer's thread, which runs no epilogue: the lane's thread
/// runs it when that report reaches it, before it publishes the step).
#[derive(Default)]
struct Landing {
    done: bool,
    waiter: Option<StepDone>,
}

/// The continuation lanes worth one launch, from each candidate's
/// iterations (most first): those with at least half the most, all run the
/// fewest of them; at most `lanes`.
fn launch_of(mut candidates: Vec<(usize, usize)>, lanes: usize) -> (Vec<usize>, usize) {
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let Some(&(_, most)) = candidates.first() else {
        return (Vec::new(), 0);
    };
    let taken: Vec<(usize, usize)> = candidates
        .into_iter()
        .filter(|&(_, n)| 2 * n >= most)
        .take(lanes.max(1))
        .collect();
    let steps = taken.iter().map(|&(_, n)| n).min().unwrap_or(0);
    (taken.into_iter().map(|(i, _)| i).collect(), steps)
}

impl Spatial {
    /// Whether frames settle asynchronously: `async` when set, else on for
    /// a build whose launches carry one lane (B1, no pipeline), the shape
    /// it is measured on. A pipelined or multi-lane build packs several
    /// lanes' continuations into one launch, and a 12-stream pipe program
    /// stalled on that on a CS-3 (a launch never answered); set `async` to
    /// run it there anyway.
    pub(super) fn asynchronous(&self) -> bool {
        self.spatial.asynchronous.unwrap_or_else(|| {
            self.model
                .as_ref()
                .is_some_and(|w| w.paged.is_some() && w.launch_lanes <= 1)
        })
    }

    /// Whether this engine runs asynchronously now: the runtime asked
    /// [`Spatial::asynchronous`] and gave it a completion sink.
    pub(super) fn settling(&self) -> bool {
        self.sink.is_some()
    }

    /// Runs posted steps in order as far as the wafer has answered them,
    /// reporting each landed; posts a continuation whenever the wafer is
    /// free.
    pub(super) fn pump(&mut self) {
        loop {
            self.land(false);
            self.post_continuation();
            let Some(head) = self.posted.front() else {
                return;
            };
            if let Some(flight) = &self.in_flight
                && self.waits_on(flight, &head.step)
            {
                let at = head.at;
                let mut landing = flight.landing.lock().expect("the landing lock poisoned");
                if landing.done {
                    continue;
                }
                landing.waiter = Some(at);
                self.reported.insert((at.frame.0, at.step));
                return;
            }
            let Some(Posted { at, step }) = self.posted.pop_front() else {
                return;
            };
            let outcome = match self.fire_step(&step) {
                Ok(_) => StepOutcome::Committed,
                Err(error) => {
                    tracing::error!(frame = at.frame.0, step = at.step, %error, "spatial: a posted step failed");
                    StepOutcome::Faulted(error.to_string())
                }
            };
            if !self.reported.remove(&(at.frame.0, at.step))
                && let Some(sink) = &self.sink
            {
                sink(at, outcome);
            }
        }
    }

    /// Runs every posted step, waiting on the wafer as it must: what a
    /// request other than a fire sees is every fire before it run.
    pub(super) fn drain(&mut self) {
        while !self.posted.is_empty() {
            self.pump();
            if !self.posted.is_empty() {
                self.land(true);
            }
        }
    }

    /// Whether `step` must wait on `flight`: a lane of it writes a slot the
    /// continuation runs that its loop has not answered yet.
    fn waits_on(&self, flight: &InFlight, step: &Step) -> bool {
        step.lanes.iter().any(|lane| {
            flight.lanes.iter().any(|&(slot, epoch, start)| {
                slot == lane.slot
                    && self.paged_ahead.get(&slot).is_some_and(|run| {
                        run.epoch == epoch
                            && run.end() <= lane.kv.held
                            && lane.kv.held < start + flight.steps as u32
                    })
            })
        })
    }

    /// Takes the continuation's answer once the wafer has it (`wait`: waits
    /// for it): each lane's rows join its loop unless the loop was dropped
    /// or replaced meanwhile.
    pub(super) fn land(&mut self, wait: bool) {
        let Some(flight) = &self.in_flight else {
            return;
        };
        let answer = if wait {
            flight
                .answer
                .recv()
                .unwrap_or_else(|_| Err("the wafer thread went away".into()))
        } else {
            match flight.answer.try_recv() {
                Ok(answer) => answer,
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => Err("the wafer thread went away".into()),
            }
        };
        let Some(flight) = self.in_flight.take() else {
            return;
        };
        let (argmax, _) = match answer {
            Ok(out) => out,
            Err(error) => {
                tracing::error!(%error, "spatial: a token loop's continuation failed");
                for &(slot, epoch, _) in &flight.lanes {
                    if self
                        .paged_ahead
                        .get(&slot)
                        .is_some_and(|r| r.epoch == epoch)
                    {
                        self.paged_ahead.remove(&slot);
                    }
                }
                return;
            }
        };
        for (k, (&(slot, epoch, start), pages)) in
            flight.lanes.iter().zip(&flight.pages).enumerate()
        {
            let Some(run) = self.paged_ahead.get_mut(&slot) else {
                continue;
            };
            if run.epoch != epoch || run.end() != start {
                continue;
            }
            let Some(answers) = argmax.get(k) else {
                continue;
            };
            let answers = &answers[..answers.len().min(flight.steps)];
            let fed: Vec<u32> = std::iter::once(run.last)
                .chain(answers.iter().copied())
                .take(answers.len())
                .collect();
            run.extend(answers);
            let slots: Vec<u32> = (start..start + fed.len() as u32).collect();
            self.kv.write_paged(slot, pages, &slots, &fed);
            self.ahead_runs += 1;
        }
    }

    /// Posts a continuation when the wafer is free: each lane's loop run on
    /// from its end, over the slots the lane's latest posted fire holds
    /// pages for (CAP − 1 iterations at most), when a posted fire of the
    /// lane writes past the loop.
    fn post_continuation(&mut self) {
        if self.in_flight.is_some() || self.posted.is_empty() || self.paged_ahead.is_empty() {
            return;
        }
        let Some(w) = self.model.as_ref() else {
            return;
        };
        if w.paged.is_none() {
            return;
        }
        let (cap, most) = (w.cap.saturating_sub(1), w.launch_lanes);
        let size = self.kv.page_size.max(1);
        // Per lane: (sequence slot, its loop, the pages, iterations).
        let mut lanes: Vec<(u32, u32, Vec<u32>, usize)> = Vec::new();
        for (&slot, run) in &self.paged_ahead {
            let end = run.end();
            let latest = self
                .posted
                .iter()
                .flat_map(|p| p.step.lanes.iter())
                .filter(|l| {
                    l.slot == slot && l.kv.held >= end && l.kv.pages.starts_with(&run.pages)
                })
                .max_by_key(|l| l.kv.held);
            let Some(lane) = latest else { continue };
            let room = (lane.kv.pages.len() as u32 * size).saturating_sub(end) as usize;
            let steps = room.min(cap);
            if steps > 0 {
                lanes.push((slot, end, lane.kv.pages.clone(), steps));
            }
        }
        let (taken, steps) = launch_of(
            lanes.iter().enumerate().map(|(i, l)| (i, l.3)).collect(),
            most,
        );
        if taken.is_empty() || steps == 0 {
            return;
        }
        let mut batch = Vec::with_capacity(taken.len());
        let mut flight_lanes = Vec::with_capacity(taken.len());
        let mut pages = Vec::with_capacity(taken.len());
        for &i in &taken {
            let (slot, end, lane_pages, _) = &lanes[i];
            let run = &self.paged_ahead[slot];
            batch.push(serve2d::BatchLane {
                pos: end.wrapping_add(run.shift),
                slot: *end,
                pages: lane_pages.clone(),
                visible: None,
                forced: vec![run.last],
            });
            flight_lanes.push((*slot, run.epoch, *end));
            pages.push(lane_pages.clone());
        }
        let landing = Arc::new(Mutex::new(Landing::default()));
        let told = Arc::clone(&landing);
        let sink = self.sink.clone();
        let landed = Box::new(move || {
            let waiter = {
                let mut landing = told.lock().expect("the landing lock poisoned");
                landing.done = true;
                landing.waiter.take()
            };
            if let (Some(at), Some(sink)) = (waiter, sink) {
                sink(at, StepOutcome::Committed);
            }
        });
        let (reply, answer) = mpsc::channel();
        tracing::info!(
            lanes = batch.len(),
            steps,
            "spatial: a token loop's continuation posted"
        );
        let Some(w) = self.model.as_mut() else {
            return;
        };
        if let Err(error) = w.post_batch(batch, steps, reply, landed) {
            tracing::error!(%error, "spatial: posting a token loop's continuation");
            return;
        }
        self.in_flight = Some(InFlight {
            lanes: flight_lanes,
            steps,
            pages,
            answer,
            landing,
        });
    }
}

/// A frame's posted steps, in order.
pub(super) fn post(queue: &mut VecDeque<Posted>, frame: FrameId, steps: &[Step]) {
    for (index, step) in steps.iter().enumerate() {
        queue.push_back(Posted {
            at: StepDone {
                frame,
                step: index as u32,
            },
            step: step.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::launch_of;

    #[test]
    fn a_continuation_launches_the_lanes_that_run_about_as_far() {
        assert_eq!(launch_of(Vec::new(), 4), (Vec::new(), 0));
        assert_eq!(launch_of(vec![(0, 30)], 4), (vec![0], 30));
        // A lane near its pages' end would hold the rest back: it waits.
        assert_eq!(
            launch_of(vec![(0, 30), (1, 2), (2, 20)], 4),
            (vec![0, 2], 20)
        );
        // At most a launch's lanes, the farthest first.
        assert_eq!(
            launch_of(vec![(0, 10), (1, 12), (2, 11)], 2),
            (vec![1, 2], 11)
        );
    }
}
