use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::AttachmentError;

#[derive(Debug, Clone)]
pub struct PolicyEngineConfig {
    pub max_component_bytes: usize,
    pub max_manifest_bytes: usize,
    pub max_package_bytes: usize,
    pub max_memory_bytes: usize,
    pub max_fuel: u64,
    pub max_deadline_ms: u64,
    pub max_input_bytes: u64,
    pub max_output_bytes: u64,
    pub max_map_calls: u32,
    pub max_map_bytes: u64,
    pub max_maps: usize,
    pub max_map_entries: u64,
    pub max_map_storage_bytes: u64,
    pub max_staged_mutations: u32,
    pub max_feedback_records: u32,
    pub max_telemetry_records: u32,
    pub max_telemetry_bytes: u64,
    pub telemetry_buffer_records: usize,
    pub max_concurrent_invocations: u32,
    pub max_conflict_retries: u32,
    pub epoch_tick: Option<Duration>,
}

impl Default for PolicyEngineConfig {
    fn default() -> Self {
        Self {
            max_component_bytes: 4 * 1024 * 1024,
            max_manifest_bytes: 256 * 1024,
            max_package_bytes: 5 * 1024 * 1024,
            max_memory_bytes: 16 * 1024 * 1024,
            max_fuel: 10_000_000,
            max_deadline_ms: 100,
            max_input_bytes: 4 * 1024 * 1024,
            max_output_bytes: 4 * 1024 * 1024,
            max_map_calls: 1024,
            max_map_bytes: 4 * 1024 * 1024,
            max_maps: 64,
            max_map_entries: 1_000_000,
            max_map_storage_bytes: 64 * 1024 * 1024,
            max_staged_mutations: 1024,
            max_feedback_records: 4096,
            max_telemetry_records: 1024,
            max_telemetry_bytes: 256 * 1024,
            telemetry_buffer_records: 4096,
            max_concurrent_invocations: 128,
            max_conflict_retries: 2,
            epoch_tick: Some(Duration::from_millis(1)),
        }
    }
}

impl PolicyEngineConfig {
    pub fn deterministic_replay() -> Self {
        Self {
            epoch_tick: None,
            ..Self::default()
        }
    }
}

#[derive(Clone)]
pub struct PolicyEngine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    engine: wasmtime::Engine,
    config: PolicyEngineConfig,
    stop: Arc<AtomicBool>,
    ticker: Mutex<Option<JoinHandle<()>>>,
    active_invocations: std::sync::atomic::AtomicU32,
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(ticker) = self.ticker.lock().unwrap().take() {
            ticker.thread().unpark();
            let _ = ticker.join();
        }
    }
}

impl PolicyEngine {
    pub fn new(config: PolicyEngineConfig) -> Result<Self, AttachmentError> {
        if config.epoch_tick.is_some_and(|tick| tick.is_zero()) {
            return Err(AttachmentError::EngineConfig(
                "epoch_tick must be non-zero".into(),
            ));
        }
        if config.telemetry_buffer_records == 0 {
            return Err(AttachmentError::EngineConfig(
                "telemetry_buffer_records must be non-zero".into(),
            ));
        }
        if config.max_concurrent_invocations == 0 {
            return Err(AttachmentError::EngineConfig(
                "max_concurrent_invocations must be non-zero".into(),
            ));
        }

        let mut wasmtime_config = wasmtime::Config::new();
        wasmtime_config.consume_fuel(true);
        wasmtime_config.epoch_interruption(true);
        wasmtime_config.wasm_threads(false);
        let engine = wasmtime::Engine::new(&wasmtime_config)
            .map_err(|error| AttachmentError::Compile(error.to_string()))?;

        let stop = Arc::new(AtomicBool::new(false));
        let ticker_engine = engine.clone();
        let ticker_stop = Arc::clone(&stop);
        let ticker = config
            .epoch_tick
            .map(|tick| {
                std::thread::Builder::new()
                    .name("plex-epoch".into())
                    .spawn(move || {
                        while !ticker_stop.load(Ordering::Acquire) {
                            std::thread::park_timeout(tick);
                            ticker_engine.increment_epoch();
                        }
                    })
                    .map_err(|error| {
                        AttachmentError::EngineConfig(format!(
                            "failed to start epoch ticker: {error}"
                        ))
                    })
            })
            .transpose()?;

        Ok(Self {
            inner: Arc::new(EngineInner {
                engine,
                config,
                stop,
                ticker: Mutex::new(ticker),
                active_invocations: std::sync::atomic::AtomicU32::new(0),
            }),
        })
    }

    pub(crate) fn raw(&self) -> &wasmtime::Engine {
        &self.inner.engine
    }

    pub(crate) fn config(&self) -> &PolicyEngineConfig {
        &self.inner.config
    }

    pub(crate) fn deadline_ticks(&self, deadline_ms: u64) -> u64 {
        let Some(epoch_tick) = self.inner.config.epoch_tick else {
            return u64::MAX;
        };
        let tick_ns = epoch_tick.as_nanos().max(1);
        let deadline_ns = u128::from(deadline_ms).saturating_mul(1_000_000);
        let ticks = deadline_ns.saturating_add(tick_ns - 1) / tick_ns;
        u64::try_from(ticks.max(1)).unwrap_or(u64::MAX)
    }

    pub(crate) fn uses_realtime_epochs(&self) -> bool {
        self.inner.config.epoch_tick.is_some()
    }

    pub(crate) fn try_acquire(&self) -> Option<InvocationPermit> {
        self.inner
            .active_invocations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < self.inner.config.max_concurrent_invocations).then_some(active + 1)
            })
            .ok()
            .map(|_| InvocationPermit {
                engine: self.inner.clone(),
            })
    }
}

pub(crate) struct InvocationPermit {
    engine: Arc<EngineInner>,
}

impl Drop for InvocationPermit {
    fn drop(&mut self) {
        self.engine
            .active_invocations
            .fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invocation_permits_bound_global_concurrency() {
        let engine = PolicyEngine::new(PolicyEngineConfig {
            max_concurrent_invocations: 1,
            epoch_tick: None,
            ..PolicyEngineConfig::default()
        })
        .unwrap();
        let permit = engine.try_acquire().unwrap();
        assert!(engine.try_acquire().is_none());
        drop(permit);
        assert!(engine.try_acquire().is_some());
    }
}
