//! Adaptive batch scheduling for inference.
//!
//! This module provides a throughput-optimizing scheduler that decides when to fire
//! batches based on arrival rate estimation and latency modeling.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::kvcache::NodeId;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for adaptive batch scheduling.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// EMA decay factor for arrival rate estimation (0 < alpha < 1).
    /// Higher values weight recent observations more heavily.
    pub arrival_rate_ema_alpha: f64,
    /// EMA decay factor for latency estimation.
    pub latency_ema_alpha: f64,
    /// Minimum batch size before considering throughput optimization.
    pub min_batch_for_optimization: usize,
    /// Maximum wait time before forcing a batch fire (safety limit).
    pub max_wait_time: Duration,
    /// Maximum number of concurrent in-flight batches.
    pub max_in_flight_batches: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            arrival_rate_ema_alpha: 0.3,
            latency_ema_alpha: 0.2,
            min_batch_for_optimization: 8,
            max_wait_time: Duration::from_millis(50),
            max_in_flight_batches: 3,
        }
    }
}

// =============================================================================
// Arrival Rate Estimation
// =============================================================================

/// EMA-based arrival rate estimator modeling request arrivals as Poisson process.
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

    /// Estimate expected wait time for next request (1/λ).
    fn expected_wait_time(&self) -> Option<Duration> {
        self.arrival_rate()
            .map(|rate| Duration::from_secs_f64(1.0 / rate))
    }
}

// =============================================================================
// Latency Modeling
// =============================================================================

/// Table-based latency model with linear interpolation.
struct LatencyModel {
    /// Latency table: index is batch_size, value is EMA latency.
    table: Vec<f64>,
    /// EMA alpha for updating latency estimates.
    alpha: f64,
    /// Base latency (constant overhead).
    base_latency: f64,
    /// Per-token latency coefficient.
    per_token_latency: f64,
}

impl LatencyModel {
    fn new(alpha: f64, max_batch_size: usize) -> Self {
        Self {
            table: vec![0.0; max_batch_size + 1],
            alpha,
            base_latency: 0.01,       // 10ms base overhead
            per_token_latency: 0.001, // 1ms per token (initial estimate)
        }
    }

    /// Record an observed latency for a batch.
    fn record_latency(&mut self, batch_size: usize, total_tokens: usize, latency: Duration) {
        let latency_secs = latency.as_secs_f64();

        // Update table entry with EMA
        if batch_size < self.table.len() {
            if self.table[batch_size] == 0.0 {
                self.table[batch_size] = latency_secs;
            } else {
                self.table[batch_size] =
                    self.alpha * latency_secs + (1.0 - self.alpha) * self.table[batch_size];
            }
        }

        // Also update linear model coefficients
        if total_tokens > 0 && latency_secs > 0.0 {
            let estimated_per_token = (latency_secs - self.base_latency).max(0.0) / total_tokens as f64;
            self.per_token_latency =
                self.alpha * estimated_per_token + (1.0 - self.alpha) * self.per_token_latency;
        }
    }

    /// Estimate latency for a given batch size and total tokens.
    fn estimate_latency(&self, batch_size: usize, total_tokens: usize) -> f64 {
        // First try exact table lookup
        if batch_size < self.table.len() && self.table[batch_size] > 0.0 {
            return self.table[batch_size];
        }

        // Fallback: linear model
        (self.base_latency + self.per_token_latency * total_tokens as f64).max(self.base_latency)
    }
}

// =============================================================================
// Adaptive Scheduler
// =============================================================================

/// Adaptive scheduler that decides when to fire batches.
///
/// Uses arrival rate estimation and latency modeling to optimize throughput.
pub struct AdaptiveScheduler {
    arrival_estimator: ArrivalRateEstimator,
    latency_model: LatencyModel,
    config: SchedulerConfig,
    /// Time when current batch started accumulating.
    batch_start_time: Option<Instant>,
    /// Aggregate metrics for monitoring
    total_tokens_processed: u64,
    total_batches_completed: u64,
    total_latency_ms: u64,
    metrics_window_start: Instant,
}

impl AdaptiveScheduler {
    /// Create a new adaptive scheduler.
    pub fn new(config: SchedulerConfig, max_batch_size: usize) -> Self {
        Self {
            arrival_estimator: ArrivalRateEstimator::new(config.arrival_rate_ema_alpha),
            latency_model: LatencyModel::new(config.latency_ema_alpha, max_batch_size),
            config,
            batch_start_time: None,
            total_tokens_processed: 0,
            total_batches_completed: 0,
            total_latency_ms: 0,
            metrics_window_start: Instant::now(),
        }
    }

    /// Record a request arrival.
    pub fn on_request_arrival(&mut self, arrival_time: Instant) {
        self.arrival_estimator.record_arrival(arrival_time);
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    /// Record completed batch latency.
    pub fn on_batch_complete(&mut self, batch_size: usize, total_tokens: usize, latency: Duration) {
        self.latency_model.record_latency(batch_size, total_tokens, latency);
        self.total_tokens_processed += total_tokens as u64;
        self.total_batches_completed += 1;
        self.total_latency_ms += latency.as_millis() as u64;
    }
    
    /// Get aggregate metrics. Returns (tokens_per_second, avg_latency_ms).
    pub fn get_aggregate_metrics(&self) -> (f64, f64) {
        let elapsed_secs = self.metrics_window_start.elapsed().as_secs_f64();
        let tokens_per_sec = if elapsed_secs > 0.0 {
            self.total_tokens_processed as f64 / elapsed_secs
        } else { 0.0 };
        let avg_latency_ms = if self.total_batches_completed > 0 {
            self.total_latency_ms as f64 / self.total_batches_completed as f64
        } else { 0.0 };
        (tokens_per_sec, avg_latency_ms)
    }

    /// Reset batch timing after firing.
    pub fn on_batch_fired(&mut self) {
        self.batch_start_time = None;
    }

    /// Decide whether to fire now or wait for more requests.
    pub fn should_fire(
        &self,
        current_batch_size: usize,
        current_total_tokens: usize,
        max_batch_size: usize,
        max_batch_tokens: usize,
        in_flight_batches: usize,
    ) -> bool {
        // Always fire if at capacity
        if current_batch_size >= max_batch_size || current_total_tokens >= max_batch_tokens {
            return true;
        }

        // Safety: fire if we've waited too long
        if let Some(start) = self.batch_start_time {
            if start.elapsed() >= self.config.max_wait_time {
                return true;
            }
        }

        // If no batches are in flight, fire to keep GPU busy
        if in_flight_batches == 0 {
            return true;
        }

        // Skip optimization for small batches when pipeline is full
        if current_batch_size < self.config.min_batch_for_optimization {
            return false;
        }

        // Throughput optimization
        let current_latency = self.latency_model.estimate_latency(current_batch_size, current_total_tokens);
        let current_throughput = current_batch_size as f64 / current_latency;

        if let Some(expected_wait) = self.arrival_estimator.expected_wait_time() {
            let wait_secs = expected_wait.as_secs_f64();
            let avg_tokens_per_request = if current_batch_size > 0 {
                current_total_tokens as f64 / current_batch_size as f64
            } else {
                1.0
            };
            let future_tokens = current_total_tokens + avg_tokens_per_request as usize;
            let future_latency = self.latency_model.estimate_latency(current_batch_size + 1, future_tokens);
            let future_throughput = (current_batch_size + 1) as f64 / (future_latency + wait_secs);

            // Fire if waiting would decrease throughput
            if current_throughput >= future_throughput {
                return true;
            }
        } else {
            // No arrival rate data yet - fire
            return true;
        }

        false
    }
}

// =============================================================================
// Multi-Node Scheduler
// =============================================================================

/// Multi-node scheduler managing independent schedulers for each compute node.
pub struct MultiNodeScheduler {
    schedulers: HashMap<NodeId, AdaptiveScheduler>,
    config: SchedulerConfig,
    max_batch_size: usize,
}

impl MultiNodeScheduler {
    pub fn new(config: SchedulerConfig, max_batch_size: usize, num_nodes: usize) -> Self {
        let mut schedulers = HashMap::new();
        for i in 0..num_nodes {
            schedulers.insert(i as NodeId, AdaptiveScheduler::new(config.clone(), max_batch_size));
        }
        
        Self {
            schedulers,
            config,
            max_batch_size,
        }
    }

    pub fn on_request_arrival(&mut self, node_id: NodeId, arrival_time: Instant) {
        if let Some(sched) = self.schedulers.get_mut(&node_id) {
            sched.on_request_arrival(arrival_time);
        }
    }

    pub fn on_batch_complete(&mut self, node_id: NodeId, batch_size: usize, total_tokens: usize, latency: Duration) {
        if let Some(sched) = self.schedulers.get_mut(&node_id) {
            sched.on_batch_complete(batch_size, total_tokens, latency);
        }
    }

    pub fn on_batch_fired(&mut self, node_id: NodeId) {
        if let Some(sched) = self.schedulers.get_mut(&node_id) {
            sched.on_batch_fired();
        }
    }

    pub fn should_fire(
        &mut self,
        node_id: NodeId,
        current_batch_size: usize,
        current_total_tokens: usize,
        max_batch_size: usize,
        max_batch_tokens: usize,
        in_flight_batches: usize,
    ) -> bool {
        if let Some(sched) = self.schedulers.get_mut(&node_id) {
            sched.should_fire(
                current_batch_size,
                current_total_tokens,
                max_batch_size,
                max_batch_tokens,
                in_flight_batches
            )
        } else {
            false
        }
    }
    
    /// Get aggregate metrics across all nodes.
    pub fn get_aggregate_metrics(&self) -> (f64, f64) {
        let mut total_tps = 0.0;
        let mut total_lat = 0.0;
        let mut count = 0;
        for sched in self.schedulers.values() {
            let (tps, lat) = sched.get_aggregate_metrics();
            total_tps += tps;
            if lat > 0.0 {
                total_lat += lat;
                count += 1;
            }
        }
        let avg_lat = if count > 0 { total_lat / count as f64 } else { 0.0 };
        (total_tps, avg_lat)
    }
}

/// Shared scheduler state wrapped in Arc<Mutex> for thread-safe access.
pub type SharedScheduler = Arc<Mutex<MultiNodeScheduler>>;
