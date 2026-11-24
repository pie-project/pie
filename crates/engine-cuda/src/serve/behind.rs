//! What a load measures behind serving (`engine::Engine::measure_behind`):
//! each `[rates]` figure it serves from a figure stored under a stale key
//! (`profile::fire::Rates::stale`), timed again where the engine's lane
//! would otherwise sleep, after ready, one point a call: a roofline rate
//! (`crate::rates::measure_behind`) or one decode product on one kernel
//! (`Shell::time_point_behind`). A point is kept only where it had the
//! device alone at a steady clock: the stream idle as it starts, no other
//! lane submitting while it runs (the runtime's submission count), and its
//! clock's warmed before read and after read agreeing within their spread
//! (`engine::rates::clock_moved`); otherwise it is discarded and timed
//! again in a later call. Each kept point is stored for the next load as it
//! is taken; this load serves on by the figures it loaded with.

use std::collections::VecDeque;

use kernels_cuda::linear::routes::Shape;

use super::Shell;
use crate::error::Result;

/// One product point timed behind serving: the decode body's median ns a
/// fire and the rounds' interquartile spread, between its clock's reads.
pub(crate) struct Timed {
    pub ns: f64,
    pub spread: f64,
    pub before: engine::rates::ClockRead,
    pub after: engine::rates::ClockRead,
}

/// What is left to time behind serving, and what has been.
#[derive(Default)]
pub(crate) struct Behind {
    /// The roofline keys served by a stale figure (`Tuned::behind`).
    rates: VecDeque<String>,
    /// The launch's cost the rates step takes off each rate.
    launch_ns: Option<u64>,
    /// Each decode product a pick read an estimate of, on each kernel no
    /// figure under a matching key states.
    points: VecDeque<(Shape, u32)>,
    /// The points at the load.
    pending: usize,
    stored: usize,
    discarded: u64,
    /// The longest call: what an arrival waited behind at most.
    longest: std::time::Duration,
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

    /// Whether anything is left.
    pub(crate) fn left(&self) -> bool {
        !self.rates.is_empty() || !self.points.is_empty()
    }
}

impl Shell {
    /// The decode products this load's picks read an estimate of, noted by
    /// its arming (`Routes::behind`): each kernel whose figure is one, at
    /// rows a decode body carries, queued to time behind serving, the
    /// profile they are stored in at `profiles`.
    pub(super) fn queue_behind(
        &mut self,
        noted: std::collections::BTreeMap<Shape, kernels_cuda::linear::routes::Wanted>,
        profiles: Option<&std::path::Path>,
    ) {
        self.behind.profiles = profiles.map(std::path::Path::to_path_buf);
        if self.routes_measuring != engine::rates::Measuring::Time || !self.records_bodies() {
            return;
        }
        let Some(served) = self.device.ctx().routes().cloned() else {
            return;
        };
        let mut shapes: Vec<_> = noted.into_iter().filter(|(s, _)| self.timable(s.m)).collect();
        shapes.sort_by_key(|(s, _)| (s.m, s.rows, s.k));
        for (shape, wanted) in shapes {
            for kernel in wanted.candidates {
                if !served.measured_at(shape, kernel) {
                    self.behind.points.push_back((shape, kernel));
                }
            }
        }
        self.behind.pending = self.behind.points.len();
        if self.behind.pending > 0 {
            eprintln!(
                "engine-cuda: rates: `[rates] product_ns`: {} decode product point(s) served by \
                 the figures stored under their stale key; timed again behind serving, one a \
                 call, stored for the next load",
                self.behind.pending
            );
        }
    }

    /// Whether anything is left to time behind serving.
    #[must_use]
    pub fn measures_behind(&self) -> bool {
        self.behind.left()
    }

    /// One point behind serving (a roofline key first, then a decode
    /// product), where the stream is idle; whether it fired. `others`
    /// reads the runtime's count of every lane's submissions.
    pub fn measure_behind(&mut self, others: &dyn Fn() -> u64) -> Result<bool> {
        if !self.behind.left() || !crate::device::ctx::idle(self.device.stream())? {
            return Ok(false);
        }
        let began = std::time::Instant::now();
        if self.behind.rates.is_empty() {
            self.point_behind(others)?;
        } else {
            self.rate_behind(others);
        }
        self.behind.longest = self.behind.longest.max(began.elapsed());
        if !self.behind.left() {
            let b = &self.behind;
            eprintln!(
                "engine-cuda: rates behind serving done: {} of {} product point(s) stored for \
                 the next load ({} discarded and timed again); the longest call, what an \
                 arrival waited behind at most, {:.1} ms",
                b.stored,
                b.pending,
                b.discarded,
                b.longest.as_secs_f64() * 1e3
            );
        }
        Ok(true)
    }

    fn rate_behind(&mut self, others: &dyn Fn() -> u64) {
        let (device, keys, launch_ns) = (&self.device, &self.keys, self.behind.launch_ns);
        let mut kept = None;
        let outcome = one_point(
            &mut self.behind.rates,
            &mut self.behind.discarded,
            others,
            |key: &String| {
                crate::rates::measure_behind(device, keys, key, launch_ns)
                    .map(|timed| {
                        let moved = timed.moved;
                        (timed, moved)
                    })
            },
            |key, timed| kept = Some((key, timed)),
        );
        match outcome {
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
        let Some((key, timed)) = kept else {
            return;
        };
        let saved = crate::rates::save_behind(
            self.behind.profiles.as_deref(),
            self.device.name(),
            &timed.rates,
        );
        eprintln!(
            "engine-cuda: rates behind serving: `[rates] {key}` timed and {} (this load serves \
             on by its stale figure); {} key(s) left",
            said(saved),
            self.behind.rates.len()
        );
    }

    fn point_behind(&mut self, others: &dyn Fn() -> u64) -> Result<()> {
        let mut points = std::mem::take(&mut self.behind.points);
        let mut discarded = self.behind.discarded;
        let mut kept = None;
        let outcome = one_point(
            &mut points,
            &mut discarded,
            others,
            |&(shape, kernel): &(Shape, u32)| {
                self.time_point_behind(shape, kernel).map(|timed| {
                    let moved = engine::rates::clock_moved(timed.before, timed.after);
                    (timed, moved)
                })
            },
            |point, timed| kept = Some((point, timed)),
        );
        self.behind.points = points;
        self.behind.discarded = discarded;
        if let Some(why) = outcome? {
            if let Some(&(shape, kernel)) = self.behind.points.front() {
                eprintln!(
                    "engine-cuda: rates behind serving: product {} discarded ({why}); timed \
                     again later",
                    point_said(shape, kernel)
                );
            }
        }
        let Some(((shape, kernel), timed)) = kept else {
            return Ok(());
        };
        self.behind.stored += 1;
        let measured = vec![(shape, kernel, timed.ns, timed.spread)];
        let saved = crate::rates::save_product_times(
            self.behind.profiles.as_deref(),
            self.device.name(),
            &self.keys,
            &self.products,
            &measured,
        );
        // What the next point's save joins: this load's picks stay as they
        // are (the served table is not touched).
        self.products = crate::rates::joined_product_times(&self.products, &measured);
        eprintln!(
            "engine-cuda: rates behind serving: product {}: step {:.1} us, {} (this load \
             serves on by its stale figure); {} point(s) left",
            point_said(shape, kernel),
            timed.ns / 1e3,
            said(saved),
            self.behind.points.len()
        );
        Ok(())
    }
}

fn point_said(shape: Shape, kernel: u32) -> String {
    format!("{:?} {}x{} m{} on {kernel:08x}", shape.kind, shape.rows, shape.k, shape.m)
}

fn said(saved: std::result::Result<Option<String>, String>) -> String {
    match saved {
        Ok(Some(path)) => format!("stored to {path} for the next load"),
        Ok(None) => "not stored: no profile directory".to_string(),
        Err(why) => format!("not stored: {why}"),
    }
}

/// One call's point: the front of `queue` timed by `time` (what it timed,
/// and how far the device's clock moved under it, if it did). Kept where
/// it had the device alone at a steady clock (no submission of another
/// lane counted by `others` while it ran, its clock unmoved): taken off the
/// queue and handed to `keep`, which stores it. Otherwise discarded: left
/// at the front, counted in `discarded`, timed again on a later call; the
/// reason returned. Nothing on an empty queue. A fault of `time` is
/// returned as it is, the queue untouched.
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
