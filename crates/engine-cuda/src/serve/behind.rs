//! What a load measures behind serving (`engine::Engine::measure_behind`):
//! each `[rates]` figure it serves from a figure stored under a stale key
//! (`profile::fire::Rates::stale`), timed again where the engine's lane
//! would otherwise sleep, after ready, one point a call: a roofline rate
//! (`crate::rates::measure_behind`) or one decode product on one kernel
//! (`crate::rates::time_product`). A product point is launched directly,
//! one eager dispatch at a time on the lane's stream, each timed by events:
//! nothing is recorded or compiled, and serving's recorded bodies are not
//! touched. A request that arrives between two dispatches abandons the
//! point (yielded), so it waits at most one dispatch. A point is kept only
//! where it had the device alone at a steady clock: the stream idle as it
//! starts, no other lane submitting while it runs (the runtime's submission
//! count), and its clock's warmed before read and after read agreeing
//! within their spread (`engine::rates::clock_moved`); otherwise it is
//! discarded and timed again in a later call. A bank's points are stored
//! for the next load together, once every candidate of it is kept, so its
//! figures are all of one method; this load serves on by the figures it
//! loaded with.

use std::collections::VecDeque;

use kernels_cuda::linear::routes::Shape;

use super::Shell;
use crate::error::Result;

/// One decode product to time behind serving: its bank, a candidate
/// kernel, and the served weight it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Point {
    shape: Shape,
    kernel: u32,
    weight: u64,
}

/// What is left to time behind serving, and what has been.
#[derive(Default)]
pub(crate) struct Behind {
    /// The roofline keys served by a stale figure (`Tuned::behind`).
    rates: VecDeque<String>,
    /// The launch's cost the rates step takes off each rate.
    launch_ns: Option<u64>,
    /// Each candidate of each decode product bank a pick read an estimate
    /// of, a bank's candidates together.
    points: VecDeque<Point>,
    /// The kept points of the bank at the front, stored once it is whole.
    bank: Vec<(Shape, u32, f64, f64)>,
    /// The points at the load.
    pending: usize,
    stored: usize,
    discarded: u64,
    yielded: u64,
    /// The longest call.
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
    /// The decode product banks this load's picks read an estimate of,
    /// noted by its arming (`Routes::behind`), each with every candidate
    /// (timed alike, so a bank's figures are of one method), queued to
    /// time behind serving over the first weight the bank met; stored in
    /// the profile at `profiles`.
    pub(super) fn queue_behind(
        &mut self,
        noted: std::collections::BTreeMap<Shape, kernels_cuda::linear::routes::Wanted>,
        profiles: Option<&std::path::Path>,
    ) {
        self.behind.profiles = profiles.map(std::path::Path::to_path_buf);
        if self.routes_measuring != engine::rates::Measuring::Time {
            return;
        }
        let mut banks: Vec<_> = noted
            .into_iter()
            .filter(|(shape, _)| shape.kind == kernels_cuda::linear::routes::Kind::Plain)
            .filter_map(|(shape, wanted)| Some((shape, *wanted.weights.first()?, wanted.candidates)))
            .collect();
        banks.sort_by_key(|(s, ..)| (s.m, s.rows, s.k));
        for (shape, weight, candidates) in banks {
            for kernel in candidates {
                self.behind.points.push_back(Point { shape, kernel, weight });
            }
        }
        self.behind.pending = self.behind.points.len();
        if self.behind.pending > 0 {
            eprintln!(
                "engine-cuda: rates: `[rates] product_ns`: {} decode product point(s) served by \
                 the figures stored under their stale key; timed again behind serving, one a \
                 call, a dispatch at a time, stored for the next load",
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
    /// product), where the stream is idle; whether it fired.
    pub fn measure_behind(&mut self, behind: &engine::BehindServing<'_>) -> Result<bool> {
        if !self.behind.left() || !crate::device::ctx::idle(self.device.stream())? {
            return Ok(false);
        }
        let began = std::time::Instant::now();
        if self.behind.rates.is_empty() {
            self.point_behind(behind);
        } else {
            self.rate_behind(behind.others);
        }
        self.behind.longest = self.behind.longest.max(began.elapsed());
        if !self.behind.left() {
            let b = &self.behind;
            eprintln!(
                "engine-cuda: rates behind serving done: {} of {} product point(s) stored for \
                 the next load ({} discarded and timed again, {} yielded to an arrival); the \
                 longest call {:.1} ms",
                b.stored,
                b.pending,
                b.discarded,
                b.yielded,
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
                    .map_err(crate::rates::Unfinished::Fault)
            },
            |key, timed| kept = Some((key, timed)),
        );
        match outcome {
            Err(why) => {
                // Not timed here: served by its stale figure for this load,
                // timed again by the next.
                if let Some(key) = self.behind.rates.pop_front() {
                    eprintln!(
                        "engine-cuda: rates behind serving: `[rates] {key}` not timed ({why:?}); \
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

    fn point_behind(&mut self, behind: &engine::BehindServing<'_>) {
        let Some(served) = self.device.ctx().routes().cloned() else {
            self.behind.points.clear();
            return;
        };
        let device = &self.device;
        let mut kept = None;
        let outcome = one_point(
            &mut self.behind.points,
            &mut self.behind.discarded,
            behind.others,
            |point: &Point| {
                crate::rates::time_product(
                    device,
                    &served,
                    point.shape,
                    point.kernel,
                    point.weight,
                    behind.arrived,
                )
                .map(|timed| {
                    let moved = engine::rates::clock_moved(timed.before, timed.after);
                    (timed, moved)
                })
            },
            |point, timed| kept = Some((point, timed)),
        );
        match outcome {
            Err(crate::rates::Unfinished::Yielded) => self.behind.yielded += 1,
            Err(crate::rates::Unfinished::Fault(why)) => {
                // The device refused the bank's dispatch: the bank keeps
                // its stale figures for this load and is timed by the next.
                if let Some(point) = self.behind.points.front().copied() {
                    eprintln!(
                        "engine-cuda: rates behind serving: product {} not timed ({why}); its \
                         bank serves on by its stale figures",
                        point_said(point.shape, point.kernel)
                    );
                    self.behind.points.retain(|p| p.shape != point.shape);
                    self.behind.bank.clear();
                }
            }
            Ok(Some(why)) => {
                if let Some(point) = self.behind.points.front() {
                    eprintln!(
                        "engine-cuda: rates behind serving: product {} discarded ({why}); timed \
                         again later",
                        point_said(point.shape, point.kernel)
                    );
                }
            }
            Ok(None) => {}
        }
        let Some((point, timed)) = kept else {
            return;
        };
        self.behind.bank.push((point.shape, point.kernel, timed.ns, timed.spread));
        if self.behind.points.front().is_some_and(|next| next.shape == point.shape) {
            return;
        }
        // The bank is whole: its candidates stored together.
        let measured = std::mem::take(&mut self.behind.bank);
        self.behind.stored += measured.len();
        let saved = crate::rates::save_product_times(
            self.behind.profiles.as_deref(),
            self.device.name(),
            &self.keys,
            &self.products,
            &measured,
            profile::fire::PRODUCT_DISPATCH,
        );
        // What the next bank's save joins: this load's picks stay as they
        // are (the served table is not touched).
        self.products = crate::rates::joined_product_times(&self.products, &measured, profile::fire::PRODUCT_DISPATCH);
        let times: Vec<String> = measured
            .iter()
            .map(|&(_, kernel, ns, _)| format!("{kernel:08x} {:.1} us", ns / 1e3))
            .collect();
        eprintln!(
            "engine-cuda: rates behind serving: product {:?} {}x{} m{}: {}; {} (this load \
             serves on by its stale figures); {} point(s) left",
            point.shape.kind,
            point.shape.rows,
            point.shape.k,
            point.shape.m,
            times.join(" / "),
            said(saved),
            self.behind.points.len()
        );
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

    /// A request arriving between two dispatches abandons the point: no
    /// dispatch runs after it, nothing is stored, the point stays at the
    /// front and is timed whole on a later call.
    #[test]
    fn an_arrival_between_dispatches_yields_the_point() {
        use crate::rates::Unfinished;
        let arrivals = std::cell::Cell::new(false);
        let dispatched = std::cell::Cell::new(0u32);
        // A point of four dispatches, `arrived` read before each.
        let point = |arrive_after: Option<u32>| -> std::result::Result<(f64, Option<f64>), Unfinished> {
            let mut sum = 0.0;
            for at in 0..4 {
                if arrive_after == Some(at) {
                    arrivals.set(true);
                }
                if arrivals.get() {
                    return Err(Unfinished::Yielded);
                }
                dispatched.set(dispatched.get() + 1);
                sum += 1.0;
            }
            Ok((sum, None))
        };
        let mut queue: VecDeque<u32> = [1].into();
        let (mut discarded, mut stored) = (0, Vec::new());
        let out = one_point(&mut queue, &mut discarded, &|| 0, |_| point(Some(2)), |p, t| stored.push((p, t)));
        assert_eq!(out, Err(Unfinished::Yielded));
        assert_eq!(dispatched.get(), 2, "no dispatch once the request arrived");
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
