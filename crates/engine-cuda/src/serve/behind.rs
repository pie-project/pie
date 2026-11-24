//! What a load measures behind serving (`engine::Engine::measure_behind`):
//! each roofline `[rates]` key it serves from a figure stored under a stale
//! key (`profile::fire::Rates::stale`), timed again where the engine's lane
//! would otherwise sleep, after ready, one key a call, by the same rates
//! step a load runs (`crate::rates::measure_behind`), so the figure stored
//! is the one a load would have measured. The step reads
//! `BehindServing::arrived` before every timed burst and stops there once a
//! request has arrived (yielded, timed again later). A key is kept only
//! where it had the device alone at a steady clock: the stream idle as it
//! starts, no other lane submitting while it runs (the runtime's submission
//! count), and its clock's warmed before read and after read agreeing
//! within their spread (`engine::rates::clock_moved`); otherwise it is
//! discarded and timed again in a later call. A kept key is stored for the
//! next load; this load serves on by the figures it loaded with.
//!
//! A decode product bank served by stale figures (`Routes::estimating`) is
//! not timed again here: its figures are a decode body's times, which only
//! a load's arming (or `pie tune`, or a re-export) measures, and timed any
//! other way its picks differ. It serves by its stale figures, said.

use std::collections::VecDeque;

use kernels_cuda::linear::routes::Shape;

use super::Shell;
use crate::error::Result;

/// What is left to time behind serving, and what has been.
#[derive(Default)]
pub(crate) struct Behind {
    /// The roofline keys served by a stale figure (`Tuned::behind`).
    rates: VecDeque<String>,
    /// The launch's cost the rates step takes off each rate.
    launch_ns: Option<u64>,
    discarded: u64,
    yielded: u64,
    /// Where the user's profile lies (`None`: nothing is stored).
    profiles: Option<std::path::PathBuf>,
}

impl Behind {
    /// The roofline keys `chip` serves by a stale figure.
    pub(crate) fn of(chip: &profile::Profile) -> Behind {
        Behind {
            rates: chip.tuned().behind.iter().cloned().collect(),
            launch_ns: chip.rates().launch_ns,
            ..Behind::default()
        }
    }
}

impl Shell {
    /// The decode product banks this load's picks read a stale figure of
    /// (`Routes::behind`, noted by its arming), said: they serve by those
    /// figures and are not timed again behind serving. The profile kept
    /// roofline keys are stored in lies at `profiles`.
    pub(super) fn queue_behind(
        &mut self,
        noted: std::collections::BTreeMap<Shape, kernels_cuda::linear::routes::Wanted>,
        profiles: Option<&std::path::Path>,
    ) {
        self.behind.profiles = profiles.map(std::path::Path::to_path_buf);
        if !noted.is_empty() {
            let banks: Vec<String> = noted
                .keys()
                .map(|s| format!("{:?} {}x{} m{}", s.kind, s.rows, s.k, s.m))
                .collect();
            eprintln!(
                "engine-cuda: rates: `[rates] product_ns`: {} decode product bank(s) served by \
                 their figures stored under a stale key (a decode body's times; `pie tune` or a \
                 re-export refreshes them, nothing times them before ready or behind serving): \
                 {}",
                banks.len(),
                banks.join(", ")
            );
        }
    }

    /// Whether a roofline key is left to time behind serving.
    #[must_use]
    pub fn measures_behind(&self) -> bool {
        !self.behind.rates.is_empty()
    }

    /// One roofline key behind serving, where the stream is idle; whether
    /// it fired.
    pub fn measure_behind(&mut self, behind: &engine::BehindServing<'_>) -> Result<bool> {
        if self.behind.rates.is_empty() || !crate::device::ctx::idle(self.device.stream())? {
            return Ok(false);
        }
        let (device, keys, launch_ns) = (&self.device, &self.keys, self.behind.launch_ns);
        let mut kept = None;
        let outcome = one_point(
            &mut self.behind.rates,
            &mut self.behind.discarded,
            behind.others,
            |key: &String| {
                crate::rates::measure_behind(device, keys, key, launch_ns, behind.arrived).map(
                    |timed| {
                        let moved = timed.moved;
                        (timed, moved)
                    },
                )
            },
            |key, timed| kept = Some((key, timed)),
        );
        match outcome {
            Err(why) if why.contains(crate::rates::YIELDED) => self.behind.yielded += 1,
            Err(why) => {
                // Not timed here: served by its stale figure for this load,
                // timed again by the next.
                if let Some(key) = self.behind.rates.pop_front() {
                    eprintln!(
                        "engine-cuda: rates behind serving: `[rates] {key}` not timed ({why}); \
                         this load serves on by its stale figure"
                    );
                }
            }
            Ok(Some(why)) => eprintln!(
                "engine-cuda: rates behind serving: `[rates] {}` discarded ({why}); timed again \
                 later",
                self.behind.rates.front().map_or("", String::as_str)
            ),
            Ok(None) => {}
        }
        if let Some((key, timed)) = kept {
            let saved = crate::rates::save_behind(
                self.behind.profiles.as_deref(),
                self.device.name(),
                &timed.rates,
            );
            eprintln!(
                "engine-cuda: rates behind serving: `[rates] {key}` timed and {} (this load \
                 serves on by its stale figure); {} key(s) left ({} discarded, {} yielded so far)",
                match saved {
                    Ok(Some(path)) => format!("stored to {path} for the next load"),
                    Ok(None) => "not stored: no profile directory".to_string(),
                    Err(why) => format!("not stored: {why}"),
                },
                self.behind.rates.len(),
                self.behind.discarded,
                self.behind.yielded
            );
        }
        Ok(true)
    }
}

/// One call's point (a roofline key): the front of `queue` timed by `time` (what it timed,
/// and how far the device's clock moved under it, if it did). Kept where
/// it had the device alone at a steady clock (no submission of another
/// lane counted by `others` while it ran, its clock unmoved): taken off the
/// queue and handed to `keep`, which stores it. Otherwise discarded: left
/// at the front, counted in `discarded`, timed again on a later call; the
/// reason returned. Nothing on an empty queue. A point `time` did not
/// finish (yielded to an arrival, or faulted) is returned as it is, left at
/// the front.
fn one_point<P, T, E>(
    queue: &mut VecDeque<P>,
    discarded: &mut u64,
    others: &dyn Fn() -> u64,
    time: impl FnOnce(&P) -> std::result::Result<(T, Option<f64>), E>,
    keep: impl FnOnce(P, T),
) -> std::result::Result<Option<String>, E> {
    let Some(point) = queue.front() else {
        return Ok(None);
    };
    let submitted = others();
    let (timed, moved) = time(point)?;
    let why = if others() != submitted {
        Some("another lane submitted while it ran".to_string())
    } else {
        moved.map(|ratio| format!("the device's clock moved {ratio:.3}x under it"))
    };
    if why.is_some() {
        *discarded += 1;
        return Ok(why);
    }
    let point = queue.pop_front().expect("the front read above");
    keep(point, timed);
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each call times the front point once; a point another lane's
    /// submission overlapped, or one its clock moved under, is discarded
    /// and timed again on the next call; one that had the device alone at
    /// a steady clock is stored and leaves the queue.
    #[test]
    fn one_point_a_call_discarded_on_overlap_or_clock_move_stored_on_pass() {
        let submits = std::cell::Cell::new(0u64);
        let others = || submits.get();
        let mut queue: VecDeque<u32> = [1, 2].into();
        let mut discarded = 0;
        let mut stored = Vec::new();
        let mut timed = Vec::new();
        let mut call = |overlap: bool, moved: Option<f64>, queue: &mut VecDeque<u32>, discarded: &mut u64| {
            one_point(
                queue,
                discarded,
                &others,
                |&p| {
                    timed.push(p);
                    if overlap {
                        submits.set(submits.get() + 1);
                    }
                    Ok::<_, ()>((p * 10, moved))
                },
                |p, t| stored.push((p, t)),
            )
            .unwrap()
        };
        let why = call(true, None, &mut queue, &mut discarded);
        assert!(why.unwrap().contains("another lane"), "overlap discards");
        let why = call(false, Some(1.2), &mut queue, &mut discarded);
        assert!(why.unwrap().contains("clock moved"), "a clock move discards");
        assert_eq!(call(false, None, &mut queue, &mut discarded), None);
        assert_eq!(call(false, None, &mut queue, &mut discarded), None);
        assert_eq!(call(false, None, &mut queue, &mut discarded), None, "nothing left");
        assert_eq!(timed, [1, 1, 1, 2], "one point a call, a discarded one again");
        assert_eq!(stored, [(1, 10), (2, 20)]);
        assert_eq!(discarded, 2);
        assert!(queue.is_empty());
    }

    /// A request arriving between two timed bursts abandons the point: no
    /// burst runs after it, nothing is stored, the point stays at the
    /// front and is timed whole on a later call.
    #[test]
    fn an_arrival_between_bursts_yields_the_point() {
        let arrivals = std::cell::Cell::new(false);
        let dispatched = std::cell::Cell::new(0u32);
        // A point of four bursts, `arrived` read before each.
        let point = |arrive_after: Option<u32>| -> std::result::Result<(f64, Option<f64>), &'static str> {
            let mut sum = 0.0;
            for at in 0..4 {
                if arrive_after == Some(at) {
                    arrivals.set(true);
                }
                if arrivals.get() {
                    return Err(crate::rates::YIELDED);
                }
                dispatched.set(dispatched.get() + 1);
                sum += 1.0;
            }
            Ok((sum, None))
        };
        let mut queue: VecDeque<u32> = [1].into();
        let (mut discarded, mut stored) = (0, Vec::new());
        let out = one_point(&mut queue, &mut discarded, &|| 0, |_| point(Some(2)), |p, t| stored.push((p, t)));
        assert_eq!(out, Err(crate::rates::YIELDED));
        assert_eq!(dispatched.get(), 2, "no burst once the request arrived");
        assert_eq!((queue.len(), discarded, stored.len()), (1, 0, 0));
        arrivals.set(false);
        let out = one_point(&mut queue, &mut discarded, &|| 0, |_| point(None), |p, t| stored.push((p, t)));
        assert_eq!(out, Ok(None));
        assert_eq!(stored, [(1, 4.0)], "timed whole again, then stored");
        assert!(queue.is_empty());
    }

    /// A fault leaves the point where it was.
    #[test]
    fn a_fault_leaves_the_queue() {
        let mut queue: VecDeque<u32> = [7].into();
        let mut discarded = 0;
        let out = one_point(&mut queue, &mut discarded, &|| 0, |_| Err::<(u32, Option<f64>), _>("no"), |_, _| ());
        assert_eq!(out, Err("no"));
        assert_eq!(queue, [7]);
        assert_eq!(discarded, 0);
    }
}
