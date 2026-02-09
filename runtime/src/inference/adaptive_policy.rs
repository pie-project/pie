//! Adaptive throughput scheduling policy.
//!
//! Uses EMA-based arrival rate estimation and latency modeling to decide
//! when to fire batches so as to maximize throughput while bounding latency.

use std::time::{Duration, Instant};

use super::scheduler::{BatchStats, Decision, SchedulingPolicy};

// =============================================================================
// Adaptive Throughput Policy
// =============================================================================

/// Adaptive scheduling policy that uses EMA-based arrival rate estimation
/// and latency modeling to maximize throughput.
///
/// This is the default policy — it waits for more requests when the
/// estimated throughput improvement from one more request exceeds the
/// cost of waiting.
pub(super) struct AdaptiveThroughputPolicy {
    arrival_estimator: ArrivalRateEstimator,
    latency_model: LatencyModel,
    max_batch_size: usize,
    /// Minimum batch size before considering throughput optimization.
    min_batch_for_optimization: usize,
    /// Maximum wait time before forcing a batch fire (safety limit).
    max_wait_time: Duration,
    /// Time when current batch started accumulating.
    batch_start_time: Option<Instant>,
}

impl AdaptiveThroughputPolicy {
    pub fn new(
        max_batch_size: usize,
        max_wait_time: Duration,
        min_batch_for_optimization: usize,
    ) -> Self {
        Self {
            arrival_estimator: ArrivalRateEstimator::new(0.3),
            latency_model: LatencyModel::new(0.2, max_batch_size),
            max_batch_size,
            min_batch_for_optimization,
            max_wait_time,
            batch_start_time: None,
        }
    }
}

impl SchedulingPolicy for AdaptiveThroughputPolicy {
    fn on_arrival(&mut self, arrival_time: Instant) {
        self.arrival_estimator.record_arrival(arrival_time);
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, stats: &BatchStats) {
        self.latency_model
            .record_latency(stats.batch_size, stats.total_tokens, stats.latency);
    }

    fn on_fired(&mut self) {
        self.batch_start_time = None;
    }

    fn decide(
        &self,
        current_batch_size: usize,
        current_total_tokens: usize,
        in_flight_batches: usize,
    ) -> Decision {
        // Always fire if at capacity
        if current_batch_size >= self.max_batch_size {
            return Decision::Fire;
        }

        // Safety: fire if we've waited too long
        if let Some(start) = self.batch_start_time {
            let waited = start.elapsed();
            if waited >= self.max_wait_time {
                return Decision::Fire;
            }

            // If no batches are in flight, fire to keep GPU busy
            if in_flight_batches == 0 {
                return Decision::Fire;
            }

            // Skip optimization for small batches when pipeline is full
            if current_batch_size < self.min_batch_for_optimization {
                let remaining = self.max_wait_time.saturating_sub(waited);
                return Decision::Wait(remaining);
            }

            // Throughput optimization
            let current_latency =
                self.latency_model
                    .estimate_latency(current_batch_size, current_total_tokens);
            let current_throughput = current_batch_size as f64 / current_latency;

            if let Some(expected_wait) = self.arrival_estimator.expected_wait_time() {
                let wait_secs = expected_wait.as_secs_f64();
                let avg_tokens = if current_batch_size > 0 {
                    current_total_tokens as f64 / current_batch_size as f64
                } else {
                    1.0
                };
                let future_tokens = current_total_tokens + avg_tokens as usize;
                let future_latency =
                    self.latency_model
                        .estimate_latency(current_batch_size + 1, future_tokens);
                let future_throughput =
                    (current_batch_size + 1) as f64 / (future_latency + wait_secs);

                // Fire if waiting would decrease throughput
                if current_throughput >= future_throughput {
                    return Decision::Fire;
                }

                let remaining = self.max_wait_time.saturating_sub(waited);
                Decision::Wait(remaining.min(expected_wait))
            } else {
                // No arrival rate data yet — fire immediately
                Decision::Fire
            }
        } else {
            // No batch start time means batch just appeared — fire
            Decision::Fire
        }
    }
}

// =============================================================================
// Arrival Rate Estimation
// =============================================================================

/// EMA-based arrival rate estimator.
///
/// Tracks inter-arrival times using an exponential moving average (EMA)
/// and derives an estimated arrival rate (λ = 1 / EMA).
///
/// **Note:** This is *not* a true memoryless Poisson model — the EMA
/// reflects smoothed historical inter-arrival times, so estimates lag
/// behind sudden traffic changes. The `expected_wait_time()` value is
/// the smoothed mean inter-arrival time, not a statistically rigorous
/// prediction.
struct ArrivalRateEstimator {
    /// Last request arrival time.
    last_arrival: Option<Instant>,
    /// EMA of inter-arrival time (seconds).
    ema_inter_arrival: f64,
    /// EMA alpha factor.
    alpha: f64,
}

impl ArrivalRateEstimator {
    fn new(alpha: f64) -> Self {
        Self {
            last_arrival: None,
            ema_inter_arrival: 0.0,
            alpha,
        }
    }

    /// Record a new request arrival and update the EMA.
    fn record_arrival(&mut self, arrival_time: Instant) {
        if let Some(last) = self.last_arrival {
            let delta = arrival_time.duration_since(last).as_secs_f64();
            if delta > 0.0 {
                if self.ema_inter_arrival == 0.0 {
                    self.ema_inter_arrival = delta;
                } else {
                    self.ema_inter_arrival =
                        self.alpha * delta + (1.0 - self.alpha) * self.ema_inter_arrival;
                }
            }
        }
        self.last_arrival = Some(arrival_time);
    }

    /// Get estimated arrival rate (requests per second).
    fn arrival_rate(&self) -> Option<f64> {
        if self.ema_inter_arrival > 0.0 {
            Some(1.0 / self.ema_inter_arrival)
        } else {
            None
        }
    }

    /// Estimated time until the next request arrives, based on the
    /// smoothed mean inter-arrival time (`1/λ`).
    ///
    /// This is *not* a statistically rigorous Poisson prediction —
    /// it simply returns the current EMA of inter-arrival times.
    fn expected_wait_time(&self) -> Option<Duration> {
        self.arrival_rate()
            .map(|rate| Duration::from_secs_f64(1.0 / rate))
    }
}

// =============================================================================
// Latency Modeling
// =============================================================================

/// Two-level latency model with EMA smoothing.
///
/// Maintains a lookup table keyed by `(batch_size, avg_tokens_per_request)`
/// for fast estimation, with a linear fallback model for unseen combinations.
struct LatencyModel {
    /// Latency table: key is (batch_size, avg_tokens_per_request), value is EMA latency.
    table: std::collections::HashMap<(usize, usize), f64>,
    /// EMA alpha for updating latency estimates.
    alpha: f64,
    /// Base latency (constant overhead).
    base_latency: f64,
    /// Per-token latency coefficient.
    per_token_latency: f64,
}

impl LatencyModel {
    fn new(alpha: f64, _max_batch_size: usize) -> Self {
        Self {
            table: std::collections::HashMap::new(),
            alpha,
            base_latency: 0.01,       // 10ms base overhead
            per_token_latency: 0.001, // 1ms per token (initial estimate)
        }
    }

    /// Quantize avg tokens per request into a bucket for table lookup.
    fn token_bucket(batch_size: usize, total_tokens: usize) -> usize {
        if batch_size == 0 {
            return 0;
        }
        let avg = total_tokens / batch_size;
        // Quantize to buckets of 64 tokens
        avg / 64
    }

    /// Record an observed latency for a batch.
    fn record_latency(&mut self, batch_size: usize, total_tokens: usize, latency: Duration) {
        let latency_secs = latency.as_secs_f64();

        // Update table entry with EMA, keyed by (batch_size, token_bucket)
        let key = (batch_size, Self::token_bucket(batch_size, total_tokens));
        let entry = self.table.entry(key).or_insert(0.0);
        if *entry == 0.0 {
            *entry = latency_secs;
        } else {
            *entry = self.alpha * latency_secs + (1.0 - self.alpha) * *entry;
        }

        // Also update linear model coefficients
        if total_tokens > 0 && latency_secs > 0.0 {
            let estimated_per_token =
                (latency_secs - self.base_latency).max(0.0) / total_tokens as f64;
            self.per_token_latency =
                self.alpha * estimated_per_token + (1.0 - self.alpha) * self.per_token_latency;
        }
    }

    /// Estimate latency for a given batch size and total tokens.
    fn estimate_latency(&self, batch_size: usize, total_tokens: usize) -> f64 {
        // First try table lookup with token bucket
        let key = (batch_size, Self::token_bucket(batch_size, total_tokens));
        if let Some(&latency) = self.table.get(&key) {
            if latency > 0.0 {
                return latency;
            }
        }

        // Fallback: linear model
        (self.base_latency + self.per_token_latency * total_tokens as f64).max(self.base_latency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- LatencyModel ---------------------------------------------------------

    #[test]
    fn latency_model_fallback_is_linear() {
        let model = LatencyModel::new(0.2, 32);
        // No observations → uses base + per_token_latency * total_tokens
        let est = model.estimate_latency(4, 100);
        let expected = 0.01 + 0.001 * 100.0;
        assert!((est - expected).abs() < 1e-9, "got {est}, expected {expected}");
    }

    #[test]
    fn latency_model_table_overrides_fallback() {
        let mut model = LatencyModel::new(0.2, 32);
        // Record an observation for batch_size=4, 256 tokens (bucket = 256/4/64 = 1)
        model.record_latency(4, 256, Duration::from_millis(50));
        // Now lookup should use the table entry, not the linear fallback
        let est = model.estimate_latency(4, 256);
        assert!((est - 0.050).abs() < 1e-9, "first observation should be exact: {est}");

        // After a second observation, EMA should blend
        model.record_latency(4, 256, Duration::from_millis(100));
        let est2 = model.estimate_latency(4, 256);
        let expected = 0.2 * 0.1 + 0.8 * 0.05; // alpha * new + (1-alpha) * old
        assert!((est2 - expected).abs() < 1e-9, "EMA blend: got {est2}, expected {expected}");
    }

    #[test]
    fn latency_model_different_buckets_are_independent() {
        let mut model = LatencyModel::new(0.5, 32);
        model.record_latency(4, 256, Duration::from_millis(50));
        model.record_latency(4, 1024, Duration::from_millis(200));
        // Different token buckets → different table entries
        let est_small = model.estimate_latency(4, 256);
        let est_large = model.estimate_latency(4, 1024);
        assert!(est_large > est_small, "larger token counts should have higher latency");
    }

    // -- ArrivalRateEstimator -------------------------------------------------

    #[test]
    fn arrival_rate_none_before_second_arrival() {
        let mut est = ArrivalRateEstimator::new(0.3);
        let now = Instant::now();
        est.record_arrival(now);
        assert!(est.arrival_rate().is_none(), "need ≥2 arrivals for a rate");
        assert!(est.expected_wait_time().is_none());
    }

    #[test]
    fn arrival_rate_converges_to_steady_state() {
        let mut est = ArrivalRateEstimator::new(0.5); // high alpha → fast convergence
        let start = Instant::now();
        // Simulate 10 arrivals at 100ms intervals
        for i in 0..10 {
            est.record_arrival(start + Duration::from_millis(i * 100));
        }
        let rate = est.arrival_rate().expect("should have a rate after 10 arrivals");
        // At 100ms inter-arrival → 10 req/sec
        assert!((rate - 10.0).abs() < 1.0, "rate should converge near 10 req/s, got {rate}");
        let wait = est.expected_wait_time().unwrap();
        assert!(wait.as_millis() > 80 && wait.as_millis() < 120,
            "expected ~100ms wait, got {}ms", wait.as_millis());
    }
}
