//! Expert-parallel ranks in one process fire in lockstep: dispatch and
//! combine meet every rank, so every rank fires every step, a rank with no
//! frame of its own an idle step at the step's bucket. Each rank keeps its
//! own scheduler and submits on its own; [`Rendezvous`] turns those
//! submissions into one sequence of steps.
//!
//! A step forms when its first frame arrives. It waits for each rank that
//! has announced a next frame (`Engine::expect_fire`) and has not yet
//! brought it, up to a deadline; a rank that announced nothing is idle and
//! is not waited for. Without the wait two busy ranks a moment apart would
//! split every step in two, each idling through the other's. The first
//! arrival leads: it is handed every arrived frame and the step's bucket
//! (the largest asked, which Dispatch's token count must agree on), fires
//! the step on every rank, and posts each joined rank its result.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// What a rank brings to a step: its frame and the token bucket it asks.
pub struct Arrival<F> {
    pub frame: F,
    pub bucket: u32,
}

/// The step a leader fires: every rank's frame (`None`: the rank idles) and
/// the bucket every rank fires at.
pub struct Formed<F> {
    pub step: u64,
    pub bucket: u32,
    pub frames: Vec<Option<F>>,
}

/// A rank's part in a step it brought a frame to.
pub enum Part<F, R> {
    /// Fire `Formed` on every rank, then [`Rendezvous::finish`] it.
    Lead(Formed<F>),
    /// The leader fired it; this is this rank's result.
    Joined(R),
}

struct State<F, R> {
    /// The step arrivals join; it forms on its first.
    open: u64,
    /// A step whose leader has not closed it yet: later arrivals open the next.
    forming: bool,
    /// Steps fired and finished: a step closes only once the one before it
    /// has, so two steps never fire at once.
    fired: u64,
    announced: Vec<bool>,
    arrived: Vec<Option<Arrival<F>>>,
    /// Results of fired steps, by rank, until the rank takes its own.
    results: Vec<Option<(u64, R)>>,
    poisoned: Option<String>,
}

pub struct Rendezvous<F, R> {
    state: Mutex<State<F, R>>,
    changed: Condvar,
    wait: Duration,
}

impl<F, R> Rendezvous<F, R> {
    #[must_use]
    pub fn new(ranks: usize, wait: Duration) -> Self {
        Self {
            state: Mutex::new(State {
                open: 0,
                forming: false,
                fired: 0,
                announced: vec![false; ranks],
                arrived: (0..ranks).map(|_| None).collect(),
                results: (0..ranks).map(|_| None).collect(),
                poisoned: None,
            }),
            changed: Condvar::new(),
            wait,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State<F, R>> {
        self.state.lock().expect("lockstep state lock poisoned")
    }

    /// `rank` will bring a frame soon: a forming step waits for it.
    pub fn announce(&self, rank: usize) {
        self.lock().announced[rank] = true;
        self.changed.notify_all();
    }

    /// Brings `rank`'s frame to the open step. The first rank to arrive
    /// leads it; the others wait for the leader's result.
    pub fn arrive(&self, rank: usize, arrival: Arrival<F>) -> Result<Part<F, R>, String> {
        let mut s = self.lock();
        // A rank leads or waits for its result before it submits again.
        debug_assert!(
            s.arrived[rank].is_none(),
            "rank {rank} arrived twice in a step"
        );
        if let Some(why) = &s.poisoned {
            return Err(why.clone());
        }
        s.announced[rank] = false;
        s.arrived[rank] = Some(arrival);
        let step = s.open;
        self.changed.notify_all();
        if s.forming {
            return self.joined(s, rank, step);
        }
        s.forming = true;
        let deadline = Instant::now() + self.wait;
        loop {
            if let Some(why) = &s.poisoned {
                return Err(why.clone());
            }
            let waiting = s
                .announced
                .iter()
                .zip(&s.arrived)
                .any(|(&announced, arrived)| announced && arrived.is_none());
            let now = Instant::now();
            let previous = s.fired == step;
            if previous && (!waiting || now >= deadline) {
                break;
            }
            // The step before still firing: keep gathering until it is done.
            let until = if previous {
                deadline - now
            } else {
                Duration::from_millis(50)
            };
            s = self
                .changed
                .wait_timeout(s, until)
                .expect("lockstep state lock poisoned")
                .0;
        }
        let frames: Vec<Option<Arrival<F>>> = s.arrived.iter_mut().map(Option::take).collect();
        s.open += 1;
        s.forming = false;
        self.changed.notify_all();
        let bucket = frames.iter().flatten().map(|a| a.bucket).max().unwrap_or(0);
        Ok(Part::Lead(Formed {
            step,
            bucket,
            frames: frames.into_iter().map(|a| a.map(|a| a.frame)).collect(),
        }))
    }

    fn joined(
        &self,
        mut s: MutexGuard<'_, State<F, R>>,
        rank: usize,
        step: u64,
    ) -> Result<Part<F, R>, String> {
        loop {
            if let Some(why) = &s.poisoned {
                return Err(why.clone());
            }
            if s.results[rank].as_ref().is_some_and(|(at, _)| *at == step) {
                let (_, result) = s.results[rank].take().expect("checked above");
                return Ok(Part::Joined(result));
            }
            s = self.changed.wait(s).expect("lockstep state lock poisoned");
        }
    }

    /// The leader of `step` posts each joined rank's result (`None` for the
    /// leader's own and for idle ranks).
    pub fn finish(&self, step: u64, results: Vec<Option<R>>) {
        let mut s = self.lock();
        s.fired = step + 1;
        for (slot, result) in s.results.iter_mut().zip(results) {
            if let Some(result) = result {
                *slot = Some((step, result));
            }
        }
        self.changed.notify_all();
    }

    /// The group stops: every waiting and later arrival answers `why`.
    pub fn poison(&self, why: &str) {
        let mut s = self.lock();
        if s.poisoned.is_none() {
            s.poisoned = Some(why.to_owned());
        }
        self.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn lead_or_join(r: &Rendezvous<u32, u32>, rank: usize, frame: u32, bucket: u32) -> u32 {
        match r
            .arrive(rank, Arrival { frame, bucket })
            .expect("not poisoned")
        {
            Part::Lead(formed) => {
                // The stand-in fire: each rank's result is its frame, plus
                // the bucket so a joined rank sees the step's.
                let results = formed
                    .frames
                    .iter()
                    .enumerate()
                    .map(|(at, f)| f.filter(|_| at != rank).map(|f| f * 1000 + formed.bucket))
                    .collect();
                r.finish(formed.step, results);
                frame * 1000 + formed.bucket
            }
            Part::Joined(result) => result,
        }
    }

    /// A rank alone leads at once: nobody announced, nobody is waited for,
    /// and the others idle in its step.
    #[test]
    fn a_lone_frame_fires_without_waiting_for_idle_ranks() {
        let r = Rendezvous::<u32, u32>::new(4, Duration::from_secs(30));
        let began = Instant::now();
        match r
            .arrive(
                2,
                Arrival {
                    frame: 7,
                    bucket: 64,
                },
            )
            .unwrap()
        {
            Part::Lead(formed) => {
                assert_eq!(formed.bucket, 64);
                assert_eq!(formed.frames, vec![None, None, Some(7), None]);
            }
            Part::Joined(_) => panic!("the first arrival leads"),
        }
        assert!(
            began.elapsed() < Duration::from_secs(5),
            "it waited on idle ranks"
        );
    }

    /// Announced ranks are waited for, so two busy ranks share one step at
    /// the larger bucket instead of idling through each other's.
    #[test]
    fn announced_ranks_share_a_step_at_the_largest_bucket() {
        let r = Arc::new(Rendezvous::<u32, u32>::new(2, Duration::from_secs(30)));
        r.announce(1);
        let late = {
            let r = Arc::clone(&r);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(50));
                lead_or_join(&r, 1, 2, 256)
            })
        };
        let first = lead_or_join(&r, 0, 1, 32);
        assert_eq!(first, 1256, "the leader fired at the joined rank's bucket");
        assert_eq!(
            late.join().unwrap(),
            2256,
            "the joined rank's frame fired in the same step"
        );
    }

    /// An announced rank that never comes holds the step only to its
    /// deadline.
    #[test]
    fn a_rank_that_announces_and_never_comes_holds_the_step_to_its_deadline() {
        let r = Rendezvous::<u32, u32>::new(2, Duration::from_millis(100));
        r.announce(1);
        let began = Instant::now();
        let Part::Lead(formed) = r
            .arrive(
                0,
                Arrival {
                    frame: 1,
                    bucket: 8,
                },
            )
            .unwrap()
        else {
            panic!("the first arrival leads");
        };
        assert!(began.elapsed() >= Duration::from_millis(100));
        assert_eq!(formed.frames, vec![Some(1), None]);
    }

    /// Steps keep their order under many ranks submitting many frames: each
    /// rank sees every one of its frames fire exactly once.
    #[test]
    fn every_frame_fires_once_across_many_steps() {
        let ranks = 4;
        let r = Arc::new(Rendezvous::<u32, u32>::new(ranks, Duration::from_millis(5)));
        let threads: Vec<_> = (0..ranks)
            .map(|rank| {
                let r = Arc::clone(&r);
                thread::spawn(move || {
                    (0..200u32)
                        .map(|i| {
                            r.announce(rank);
                            let frame = (rank as u32) * 1_000 + i;
                            lead_or_join(&r, rank, frame, 16) / 1000
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for (rank, t) in threads.into_iter().enumerate() {
            let got = t.join().unwrap();
            let want: Vec<u32> = (0..200).map(|i| (rank as u32) * 1_000 + i).collect();
            assert_eq!(got, want, "rank {rank}'s frames");
        }
    }

    /// A step forms while the one before it fires, but closes only after:
    /// no two leaders fire at once.
    #[test]
    fn no_two_steps_fire_at_once() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let ranks = 4;
        let r = Arc::new(Rendezvous::<u32, u32>::new(ranks, Duration::from_millis(1)));
        let firing = Arc::new(AtomicBool::new(false));
        let threads: Vec<_> = (0..ranks)
            .map(|rank| {
                let (r, firing) = (Arc::clone(&r), Arc::clone(&firing));
                thread::spawn(move || {
                    for i in 0..100u32 {
                        match r
                            .arrive(
                                rank,
                                Arrival {
                                    frame: i,
                                    bucket: 8,
                                },
                            )
                            .unwrap()
                        {
                            Part::Lead(formed) => {
                                assert!(
                                    !firing.swap(true, Ordering::SeqCst),
                                    "two steps fired at once"
                                );
                                thread::sleep(Duration::from_micros(200));
                                firing.store(false, Ordering::SeqCst);
                                let results = formed.frames.iter().map(|f| f.map(|_| 0)).collect();
                                r.finish(formed.step, results);
                            }
                            Part::Joined(_) => {}
                        }
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
    }

    #[test]
    fn a_poisoned_group_answers_every_waiting_rank() {
        let r = Arc::new(Rendezvous::<u32, u32>::new(2, Duration::from_secs(30)));
        r.announce(1);
        let waiting = {
            let r = Arc::clone(&r);
            thread::spawn(move || {
                r.arrive(
                    0,
                    Arrival {
                        frame: 1,
                        bucket: 8,
                    },
                )
                .err()
            })
        };
        thread::sleep(Duration::from_millis(50));
        r.poison("rank 1 refused");
        assert_eq!(waiting.join().unwrap().as_deref(), Some("rank 1 refused"));
        assert!(
            r.arrive(
                1,
                Arrival {
                    frame: 2,
                    bucket: 8
                }
            )
            .is_err()
        );
    }
}
