//! Mock-driver helpers for integration tests.
//!
//! These helpers keep the existing harness source compiling on top of direct
//! driver-backend registration.

use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pie_engine::driver::{
    DriverBackend, DriverSpec, LaunchPlan, SchedulerLimits, register_driver_backend,
};
use pie_engine::scheduler::worker::BatchScheduler;

/// Launch observer for harness probes, wired onto the dummy driver's launch
/// path via [`launch_observer`]: it sees every accepted launch's forward
/// geometry, and a sleeping implementation keeps that fire outstanding (the
/// forced-overlap latency the concurrency suites rely on).
pub trait Behavior: Send + Sync + 'static {
    fn observe_launch(&self, _req: &LaunchPlan) {}
}

/// Adapt a [`Behavior`] to the dummy driver's [`LaunchObserver`]: rebuild the
/// observed forward geometry as a [`LaunchPlan`] and forward it to
/// `observe_launch` synchronously on the launch path.
pub fn launch_observer(behavior: Arc<dyn Behavior>) -> pie_driver_dummy_lib::LaunchObserver {
    pie_driver_dummy_lib::LaunchObserver(Arc::new(move |obs| {
        let (kv_page_indices, kv_page_indptr) = if obs.kv_translation.is_empty() {
            (obs.kv_page_indices.clone(), obs.kv_page_indptr.clone())
        } else {
            (
                obs.kv_translation.clone(),
                obs.kv_translation_indptr.clone(),
            )
        };
        let plan = LaunchPlan {
            token_ids: obs.token_ids.clone(),
            qo_indptr: obs.qo_indptr.clone(),
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens: obs.kv_last_page_lens.clone(),
            ..LaunchPlan::default()
        };
        behavior.observe_launch(&plan);
    }))
}

/// Helper: derive the request count from `qo_indptr`.
pub fn num_requests(req: &LaunchPlan) -> u32 {
    req.qo_indptr.len().saturating_sub(1) as u32
}

pub fn total_tokens(req: &LaunchPlan) -> usize {
    req.token_ids.len()
}

/// Marker retained for harness configuration.
pub struct EchoBehavior(pub u32);

impl Behavior for EchoBehavior {}

fn mock_hash_uniform(seed_eff: u64, j: u32) -> f32 {
    pie_ptir::rng::hash_uniform(seed_eff, j)
}

/// Deterministic pseudo-random logits row seeded by request id, in ~[-4, 4].
pub fn synthetic_logits(req_id: u64, vocab: usize) -> Vec<f32> {
    let seed = req_id ^ pie_ptir::rng::seed_eff(0);
    (0..vocab as u32)
        .map(|j| (mock_hash_uniform(seed, j) * 2.0 - 1.0) * 4.0)
        .collect()
}

/// Returns sequential tokens starting from `start`.
pub struct CounterBehavior {
    next: AtomicU32,
}

impl CounterBehavior {
    pub fn new(start: u32) -> Self {
        Self {
            next: AtomicU32::new(start),
        }
    }
}

impl Behavior for CounterBehavior {
    fn observe_launch(&self, req: &LaunchPlan) {
        self.next
            .fetch_add(num_requests(req), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Observer wrapper that adds simulated latency.
pub struct DelayedBehavior<B: Behavior> {
    pub inner: B,
    pub latency: Duration,
}

impl<B: Behavior> Behavior for DelayedBehavior<B> {
    fn observe_launch(&self, req: &LaunchPlan) {
        std::thread::sleep(self.latency);
        self.inner.observe_launch(req);
    }
}

/// Observer wrapper that stops forwarding after N calls.
pub struct FailAfterBehavior<B: Behavior> {
    pub inner: B,
    remaining: AtomicU32,
}

impl<B: Behavior> FailAfterBehavior<B> {
    pub fn new(inner: B, success_count: u32) -> Self {
        Self {
            inner,
            remaining: AtomicU32::new(success_count),
        }
    }
}

impl<B: Behavior> Behavior for FailAfterBehavior<B> {
    fn observe_launch(&self, req: &LaunchPlan) {
        if self
            .remaining
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
            != 0
        {
            self.inner.observe_launch(req);
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordedCall {
    pub device_idx: usize,
    pub method: &'static str,
    pub num_requests: usize,
    pub total_tokens: usize,
    pub timestamp: Instant,
}

#[derive(Debug, Default)]
pub struct CallRecorder {
    calls: Mutex<Vec<RecordedCall>>,
}

impl CallRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, call: RecordedCall) {
        self.calls.lock().unwrap().push(call);
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    pub fn calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    pub fn wait_for_calls(&self, n: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.call_count() >= n {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn register_dummy_driver(
    num_kv_pages: usize,
    driver_idx: usize,
    behavior: Arc<dyn Behavior>,
    vocab_size: u32,
    operation_log: Arc<Mutex<Vec<String>>>,
) -> (usize, BatchScheduler) {
    let (backend, _) = DriverBackend::dummy(pie_driver_dummy_lib::DummyDriverOptions {
        total_pages: num_kv_pages as u32,
        kv_page_size: 16,
        swap_pool_size: 0,
        // MUST match the engine model's `vocab_size()` (the tokenizer
        // fixture's vocab): guests declare `logits` as
        // `[rows, output-vocab-size]` and the dummy validates the decl
        // against this capability at program bind.
        vocab_size,
        max_model_len: 8192,
        arch_name: "test-dummy".into(),
        activation_dtype: "f32".into(),
        snapshot_dir: String::new(),
        max_forward_tokens: 4096,
        max_forward_requests: 32,
        max_page_refs: num_kv_pages.max(1) as u32,
        has_mtp_logits: true,
        has_mtp_drafts: true,
        has_value_head: true,
        callback_delay_ms: 0,
        reject_launches: false,
        reject_launches_remaining: 0,
        fail_launches_after_accept: false,
        retry_launches_remaining: 0,
        operation_log: Some(operation_log),
        launch_observer: Some(launch_observer(behavior)),
    })
    .expect("create dummy driver backend");
    let limits = SchedulerLimits {
        max_forward_requests: 32,
        max_forward_tokens: 4096,
        max_page_refs: num_kv_pages.max(1),
    };
    let driver_id = register_driver_backend(
        DriverSpec {
            num_kv_pages,
            limits,
            device_geometry_port_mask: pie_driver_abi::PIE_DEVICE_GEOMETRY_PORTS
                | pie_driver_abi::PIE_DEVICE_PORT_ATTN_MASK,
        },
        backend,
    );
    let scheduler = BatchScheduler::new(driver_id, driver_idx, 16, limits, 30);
    (driver_id, scheduler)
}

/// Minimal backend wrapper that allocates direct native dummy drivers.
pub struct MockBackend {
    driver_ids: Vec<usize>,
    _schedulers: Vec<BatchScheduler>,
    recorder: Arc<CallRecorder>,
}

impl MockBackend {
    pub fn new(
        num_devices: usize,
        behavior: Arc<dyn Behavior>,
        vocab_size: u32,
        operation_log: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        let recorder = Arc::new(CallRecorder::new());
        let (driver_ids, schedulers) = (0..num_devices)
            .map(|driver_idx| {
                register_dummy_driver(
                    64,
                    driver_idx,
                    behavior.clone(),
                    vocab_size,
                    Arc::clone(&operation_log),
                )
            })
            .unzip();
        Self {
            driver_ids,
            _schedulers: schedulers,
            recorder,
        }
    }

    pub fn driver_ids(&self) -> &[usize] {
        &self.driver_ids
    }

    pub fn recorder(&self) -> &CallRecorder {
        &self.recorder
    }
}
