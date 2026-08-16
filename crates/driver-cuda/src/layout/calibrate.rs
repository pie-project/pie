//! Sweep calibration shapes and time synthetic forward steps.
//!
//! Calibration boots build the feasible ceiling. The ladder only walks
//! downward because that arena can run smaller shapes, but not larger ones.

use super::profile_cache::ShapeSample;
use super::profile_key::ProfileShape;

/// One point on the ladder: a forward-buffer shape to time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    /// Token capacity of the forward buffer.
    pub max_forward_tokens: i32,
    /// Request capacity of the forward buffer.
    pub max_forward_requests: i32,
}

impl Point {
    /// Tokens per request for the synthetic batch, floored at one.
    #[must_use]
    pub const fn tokens_per_request(self) -> i32 {
        let n = self.max_forward_tokens / self.max_forward_requests;
        if n < 1 { 1 } else { n }
    }

    /// Tokens the synthetic batch actually carries.
    #[must_use]
    pub const fn batch_tokens(self) -> i32 {
        self.max_forward_requests * self.tokens_per_request()
    }
}

/// Where a sweep starts: the rectangle the calibration boot built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ceiling {
    /// `CudaMemoryPlan::max_workspace_tokens` of the built arena.
    pub max_forward_tokens: i32,
    /// `CudaMemoryPlan::max_requests` of the built arena.
    pub max_forward_requests: i32,
}

/// Token floor below which fixed per-fire cost dominates.
const FLOOR_TOKENS: i32 = 32;

/// Request-count floor.
const FLOOR_REQUESTS: i32 = 1;

/// Largest-first powers-of-two ladder over tokens and requests.
///
/// Axes vary independently: tokens probe prefill capacity; requests probe decode.
#[must_use]
pub fn ladder(ceiling: Ceiling) -> Vec<Point> {
    let axis = |top: i32, floor: i32| {
        let mut v = Vec::new();
        let mut x = top.max(floor);
        while x > floor {
            v.push(x);
            x /= 2;
        }
        v.push(floor);
        v.dedup();
        v
    };
    let tokens = axis(ceiling.max_forward_tokens, FLOOR_TOKENS);
    let requests = axis(ceiling.max_forward_requests, FLOOR_REQUESTS);
    let mut out = Vec::with_capacity(tokens.len() * requests.len());
    for &t in &tokens {
        for &r in &requests {
            // Whole-request batches cannot distinguish r > t from r == t.
            if r <= t {
                out.push(Point {
                    max_forward_tokens: t,
                    max_forward_requests: r,
                });
            }
        }
    }
    out
}

/// What the driver supplies: one timed step at one shape.
pub trait StepTimer {
    /// Return wall time in milliseconds, or `None` for a declined shape.
    fn step_ms(&mut self, point: Point) -> Option<f64>;
}

/// A finished sweep: the shape that won and every point that was measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// The winner, ready for [`ProfileCache::store`](super::profile_cache::ProfileCache::store).
    pub shape: ProfileShape,
    /// Measured points in ladder order.
    pub samples: Vec<ShapeSample>,
}

/// Timed repeats per point; the first is discarded as warm-up.
pub const REPEATS: usize = 3;

/// Time each ladder point and pick the fastest measured shape.
///
/// Returns `None` if every shape is declined. `template` keeps the policy,
/// page size, and arena budget that are not varied by calibration.
pub fn sweep(
    ceiling: Ceiling,
    template: &ProfileShape,
    timer: &mut dyn StepTimer,
) -> Option<Calibration> {
    let mut samples = Vec::new();
    for point in ladder(ceiling) {
        let Some(sample) = measure(point, timer) else {
            continue;
        };
        samples.push(sample);
    }
    let best = samples.iter().enumerate().fold(None, |best, (i, s)| {
        match best {
            // Strict comparison keeps ties on the earlier, larger shape.
            Some((_, v)) if s.tokens_per_s <= v => best,
            _ => Some((i, s.tokens_per_s)),
        }
    })?;
    let won = &samples[best.0];
    Some(Calibration {
        shape: ProfileShape {
            max_forward_tokens: won.max_forward_tokens,
            max_forward_requests: won.max_forward_requests,
            ..template.clone()
        },
        samples,
    })
}

/// One point, timed [`REPEATS`] times with the first discarded.
fn measure(point: Point, timer: &mut dyn StepTimer) -> Option<ShapeSample> {
    let mut kept = Vec::with_capacity(REPEATS - 1);
    for i in 0..REPEATS {
        let ms = timer.step_ms(point)?;
        if i > 0 {
            kept.push(ms);
        }
    }
    if kept.is_empty() {
        return None;
    }
    let n = kept.len() as f64;
    let mean = kept.iter().sum::<f64>() / n;
    // Population deviation: kept repeats are the full measurement set.
    let var = kept.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    Some(ShapeSample {
        max_forward_tokens: point.max_forward_tokens,
        max_forward_requests: point.max_forward_requests,
        tokens_per_request: point.tokens_per_request(),
        step_ms: mean,
        step_ms_stddev: var.sqrt(),
        tokens_per_s: if mean > 0.0 {
            f64::from(point.batch_tokens()) * 1000.0 / mean
        } else {
            0.0
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template() -> ProfileShape {
        ProfileShape {
            policy_profile: "balanced".to_owned(),
            kv_page_size: 16,
            max_forward_tokens: 0,
            max_forward_requests: 0,
            budget_bytes: 40 * 1024 * 1024 * 1024,
        }
    }

    /// A timer whose step time is a stated function of the shape.
    struct Fake<F: FnMut(Point) -> Option<f64>>(F);
    impl<F: FnMut(Point) -> Option<f64>> StepTimer for Fake<F> {
        fn step_ms(&mut self, p: Point) -> Option<f64> {
            (self.0)(p)
        }
    }

    #[test]
    fn the_ladder_starts_at_the_ceiling_and_only_goes_down() {
        let c = Ceiling {
            max_forward_tokens: 1024,
            max_forward_requests: 64,
        };
        let l = ladder(c);
        assert_eq!(
            l[0],
            Point {
                max_forward_tokens: 1024,
                max_forward_requests: 64
            },
            "the shape the planner would have built is measured first"
        );
        assert!(
            l.iter()
                .all(|p| p.max_forward_tokens <= 1024 && p.max_forward_requests <= 64),
            "a sweep cannot run a shape the arena was not built for"
        );
        assert!(
            l.iter()
                .all(|p| p.max_forward_requests <= p.max_forward_tokens),
            "more requests than tokens is not a distinguishable batch"
        );
    }

    #[test]
    fn the_ladder_halves_each_axis_to_its_floor() {
        let l = ladder(Ceiling {
            max_forward_tokens: 256,
            max_forward_requests: 8,
        });
        let tokens: Vec<i32> = {
            let mut v: Vec<i32> = l.iter().map(|p| p.max_forward_tokens).collect();
            v.dedup();
            v
        };
        assert_eq!(
            tokens,
            vec![256, 128, 64, 32],
            "powers of two down to the floor"
        );
        let requests: Vec<i32> = {
            let mut v: Vec<i32> = l
                .iter()
                .filter(|p| p.max_forward_tokens == 256)
                .map(|p| p.max_forward_requests)
                .collect();
            v.dedup();
            v
        };
        assert_eq!(requests, vec![8, 4, 2, 1]);
    }

    #[test]
    fn a_ceiling_at_the_floor_is_still_one_point() {
        let l = ladder(Ceiling {
            max_forward_tokens: 16,
            max_forward_requests: 1,
        });
        assert_eq!(l.len(), 1, "a degenerate ceiling measures itself");
        assert_eq!(l[0].max_forward_tokens, FLOOR_TOKENS);
    }

    #[test]
    fn the_first_repeat_is_discarded() {
        // The warm-up outlier should not move the mean.
        let mut seen = 0;
        let mut timer = Fake(|_| {
            seen += 1;
            Some(if seen == 1 { 100.0 } else { 10.0 })
        });
        let s = measure(
            Point {
                max_forward_tokens: 64,
                max_forward_requests: 1,
            },
            &mut timer,
        )
        .expect("measured");
        assert!((s.step_ms - 10.0).abs() < 1e-9, "the warm-up did not count");
        assert!((s.step_ms_stddev - 0.0).abs() < 1e-9);
    }

    #[test]
    fn the_spread_is_reported() {
        let mut i = 0;
        let mut timer = Fake(|_| {
            i += 1;
            Some(match i {
                1 => 99.0,
                2 => 8.0,
                _ => 12.0,
            })
        });
        let s = measure(
            Point {
                max_forward_tokens: 64,
                max_forward_requests: 1,
            },
            &mut timer,
        )
        .expect("measured");
        assert!((s.step_ms - 10.0).abs() < 1e-9);
        assert!(
            (s.step_ms_stddev - 2.0).abs() < 1e-9,
            "population, not sample"
        );
    }

    #[test]
    fn the_winner_is_the_fastest_measured_shape_not_the_biggest() {
        // Throughput can peak below the ceiling.
        let c = Ceiling {
            max_forward_tokens: 512,
            max_forward_requests: 16,
        };
        let mut timer = Fake(|p: Point| {
            let t = f64::from(p.batch_tokens());
            // Tokens/s peaks at 128.
            Some(if t <= 128.0 { t } else { t * t / 128.0 })
        });
        let cal = sweep(c, &template(), &mut timer).expect("a winner");
        assert_eq!(
            cal.shape.max_forward_tokens, 128,
            "the peak, not the ceiling"
        );
        assert_eq!(
            cal.shape.policy_profile, "balanced",
            "the template carries through"
        );
        assert_eq!(cal.shape.kv_page_size, 16);
        assert_eq!(
            cal.samples.len(),
            ladder(c).len(),
            "every point is kept for the audit"
        );
    }

    #[test]
    fn a_tie_goes_to_the_bigger_shape() {
        // Equal throughput keeps the larger shape.
        let c = Ceiling {
            max_forward_tokens: 128,
            max_forward_requests: 1,
        };
        let mut timer = Fake(|p: Point| Some(f64::from(p.batch_tokens())));
        let cal = sweep(c, &template(), &mut timer).expect("a winner");
        assert_eq!(cal.shape.max_forward_tokens, 128);
    }

    #[test]
    fn a_declined_shape_is_skipped_not_fatal() {
        let c = Ceiling {
            max_forward_tokens: 256,
            max_forward_requests: 4,
        };
        let mut timer = Fake(|p: Point| {
            // The widest shape is unavailable.
            (p.max_forward_tokens < 256).then(|| f64::from(p.batch_tokens()))
        });
        let cal = sweep(c, &template(), &mut timer).expect("a winner");
        assert!(cal.shape.max_forward_tokens < 256);
        assert!(
            cal.samples.iter().all(|s| s.max_forward_tokens < 256),
            "a shape that could not run is not a sample"
        );
    }

    #[test]
    fn a_ladder_nobody_can_run_stores_nothing() {
        let mut timer = Fake(|_| None);
        assert!(
            sweep(
                Ceiling {
                    max_forward_tokens: 256,
                    max_forward_requests: 4
                },
                &template(),
                &mut timer
            )
            .is_none(),
            "an empty measurement must not overwrite the analytic pick"
        );
    }
}
