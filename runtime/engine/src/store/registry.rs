//! Per-(model, driver) store registry (kv_refact.md, `store/registry.rs`).
//!
//! Maps a model/driver pair to its owning `KvStore` and `RsStore` so
//! `pipeline::fire` and the WIT host resources resolve handles without each
//! component holding a direct store reference. Mirrors the retired
//! `arena::registry` shape: an append-only static keyed by `model_idx`
//! (lock-step with bootstrap model registration), each entry a `Vec` indexed
//! by driver ordinal.
//!
//! ## Locking discipline (required)
//! Lock a store **synchronously** for prepare/publication/settlement and release it
//! **before** awaiting the driver — never across an `await`:
//! ```text
//! prepare: let mut kv = registry.kv.lock();   // sync
//!          kv.prepare_write(..)               // sync
//!          drop(kv);                          // unlock
//!          driver copies + launch, await      // no lock held
//! settle: let mut kv = registry.kv.lock();    // re-lock
//!         kv.settle(..)                       // sync
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};
use std::time::Instant;

use super::kv::KvStore;
use super::rs::RsStore;

const KV_LOCK_TRACE_CAPACITY: usize = 1_048_576;
const KV_LOCK_TRACE_DEFAULT_THRESHOLD_US: u64 = 250;

#[derive(Clone, Copy)]
struct KvLockTraceRecord {
    t_acquire_us: u64,
    wait_ns: u64,
    hold_ns: u64,
    tag: &'static str,
}

struct KvLockTrace {
    records: crossbeam_queue::ArrayQueue<KvLockTraceRecord>,
    dropped: AtomicU64,
    dumped: AtomicBool,
    threshold_ns: u64,
    output: PathBuf,
}

impl KvLockTrace {
    fn new() -> Self {
        let threshold_us = std::env::var("PIE_KV_LOCK_TRACE_THRESHOLD_US")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(KV_LOCK_TRACE_DEFAULT_THRESHOLD_US);
        let output = std::env::var_os("PIE_KV_LOCK_TRACE_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(format!("/tmp/pie-kv-lock-trace-{}.csv", std::process::id()))
            });
        Self {
            records: crossbeam_queue::ArrayQueue::new(KV_LOCK_TRACE_CAPACITY),
            dropped: AtomicU64::new(0),
            dumped: AtomicBool::new(false),
            threshold_ns: threshold_us.saturating_mul(1_000),
            output,
        }
    }

    fn record(&self, record: KvLockTraceRecord) {
        if record.wait_ns <= self.threshold_ns && record.hold_ns <= self.threshold_ns {
            return;
        }
        if self.records.push(record).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn dump(&self) -> anyhow::Result<()> {
        if self.dumped.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        if let Some(parent) = self.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut records = Vec::with_capacity(self.records.len());
        while let Some(record) = self.records.pop() {
            records.push(record);
        }
        records.sort_unstable_by_key(|record| record.t_acquire_us);
        let mut output = BufWriter::new(File::create(&self.output)?);
        writeln!(output, "t_acquire_us,wait_ns,hold_ns,tag")?;
        for record in records {
            writeln!(
                output,
                "{},{},{},{}",
                record.t_acquire_us, record.wait_ns, record.hold_ns, record.tag
            )?;
        }
        output.flush()?;
        let metadata = serde_json::json!({
            "schema": 1,
            "capacity": KV_LOCK_TRACE_CAPACITY,
            "threshold_ns": self.threshold_ns,
            "dropped": self.dropped.load(Ordering::Acquire),
            "output": self.output,
        });
        let metadata_path = PathBuf::from(format!("{}.meta.json", self.output.display()));
        serde_json::to_writer_pretty(File::create(metadata_path)?, &metadata)?;
        Ok(())
    }
}

fn kv_lock_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PIE_KV_LOCK_TRACE").is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

fn kv_lock_trace() -> &'static KvLockTrace {
    static TRACE: LazyLock<KvLockTrace> = LazyLock::new(KvLockTrace::new);
    &TRACE
}

#[inline(always)]
pub fn with_kv_lock<T>(
    store: &Mutex<KvStore>,
    tag: &'static str,
    operation: impl FnOnce(&mut KvStore) -> T,
) -> T {
    if !kv_lock_trace_enabled() {
        let mut guard = store.lock().unwrap();
        return operation(&mut guard);
    }

    let wait_started = Instant::now();
    let mut guard = store.lock().unwrap();
    let wait_ns = u64::try_from(wait_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let t_acquire_us = crate::scheduler::fire_timing_now_us();
    let hold_started = Instant::now();
    let result = operation(&mut guard);
    let hold_ns = u64::try_from(hold_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    drop(guard);
    kv_lock_trace().record(KvLockTraceRecord {
        t_acquire_us,
        wait_ns,
        hold_ns,
        tag,
    });
    result
}

pub fn dump_kv_lock_trace() -> anyhow::Result<()> {
    if kv_lock_trace_enabled() {
        kv_lock_trace().dump()
    } else {
        Ok(())
    }
}

/// The typed stores for one (model, driver).
#[derive(Clone)]
pub struct Stores {
    pub kv: Arc<Mutex<KvStore>>,
    pub rs: Arc<Mutex<RsStore>>,
    /// Tokens per KV page for this model/driver.
    pub kv_page_size: u32,
}

static REGISTRY: LazyLock<boxcar::Vec<RwLock<Vec<Option<Stores>>>>> =
    LazyLock::new(boxcar::Vec::new);

/// Test convenience: register a model with no host swap pages. Production
/// bootstrap always sizes swap explicitly via [`register_model_with_swap`].
#[cfg(test)]
pub fn register_model(kv_page_size: u32, num_kv_pages: &[usize], num_rs_slots: &[usize]) -> usize {
    register_model_with_swap(
        kv_page_size,
        num_kv_pages,
        &vec![0; num_kv_pages.len()],
        num_rs_slots,
    )
}

/// Register a model's per-driver stores at bootstrap. Capacities come from
/// the driver-preallocated static pools. Returns the assigned model index.
pub fn register_model_with_swap(
    kv_page_size: u32,
    num_kv_pages: &[usize],
    num_host_pages: &[usize],
    num_rs_slots: &[usize],
) -> usize {
    let stores: Vec<Option<Stores>> = (0..num_kv_pages.len())
        .map(|d| {
            let kv = Arc::new(Mutex::new(KvStore::new_with_swap(
                num_kv_pages[d] as u32,
                num_host_pages.get(d).copied().unwrap_or(0) as u32,
                rand::random::<[u8; 32]>(),
            )));
            Some(Stores {
                kv,
                rs: Arc::new(Mutex::new(RsStore::new(
                    num_rs_slots.get(d).copied().unwrap_or(0) as u32,
                ))),
                kv_page_size,
            })
        })
        .collect();
    REGISTRY.push(RwLock::new(stores))
}

pub fn register_driver_with_swap(
    model_idx: usize,
    driver_idx: usize,
    kv_page_size: u32,
    base_page: u32,
    num_kv_pages: usize,
    num_host_pages: usize,
    num_rs_slots: usize,
) -> anyhow::Result<()> {
    let model = REGISTRY
        .get(model_idx)
        .ok_or_else(|| anyhow::anyhow!("store registry: unknown model {model_idx}"))?;
    let mut stores = model.write().unwrap();
    if stores.len() <= driver_idx {
        stores.resize_with(driver_idx + 1, || None);
    }
    anyhow::ensure!(
        stores[driver_idx].is_none(),
        "store registry: driver {driver_idx} is already registered for model {model_idx}"
    );
    if let Some(existing) = stores.iter().flatten().next() {
        anyhow::ensure!(
            existing.kv_page_size == kv_page_size,
            "store registry: KV page size {kv_page_size} does not match model {model_idx} page size {}",
            existing.kv_page_size
        );
    }
    let kv = Arc::new(Mutex::new(KvStore::new_with_swap_range(
        base_page,
        num_kv_pages as u32,
        num_host_pages as u32,
        rand::random::<[u8; 32]>(),
    )));
    stores[driver_idx] = Some(Stores {
        kv,
        rs: Arc::new(Mutex::new(RsStore::new(num_rs_slots as u32))),
        kv_page_size,
    });
    Ok(())
}

pub fn unregister_driver(model_idx: usize, driver_idx: usize) -> anyhow::Result<()> {
    let model = REGISTRY
        .get(model_idx)
        .ok_or_else(|| anyhow::anyhow!("store registry: unknown model {model_idx}"))?;
    let mut stores = model.write().unwrap();
    let slot = stores.get_mut(driver_idx).ok_or_else(|| {
        anyhow::anyhow!("store registry: unknown driver {driver_idx} for model {model_idx}")
    })?;
    anyhow::ensure!(
        slot.take().is_some(),
        "store registry: driver {driver_idx} for model {model_idx} is already unregistered"
    );
    Ok(())
}

/// The stores for `(model_idx, driver_idx)`; cheap `Arc` clones. Panics if
/// never registered — a bootstrap wiring bug, not a runtime condition.
pub fn get(model_idx: usize, driver_idx: usize) -> Stores {
    try_get(model_idx, driver_idx).unwrap_or_else(|| {
        panic!("store registry: no stores for model {model_idx} driver {driver_idx}")
    })
}

pub fn try_get(model_idx: usize, driver_idx: usize) -> Option<Stores> {
    REGISTRY
        .get(model_idx)?
        .read()
        .unwrap()
        .get(driver_idx)
        .cloned()
        .flatten()
}

pub fn all_for_model(model_idx: usize) -> Vec<Stores> {
    REGISTRY
        .get(model_idx)
        .map(|stores| stores.read().unwrap().iter().flatten().cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_store_slots_unregister_without_reusing_driver_ids() {
        let model = register_model(16, &[8], &[0]);
        register_driver_with_swap(model, 1, 16, 10, 4, 0, 0).unwrap();
        assert!(try_get(model, 1).is_some());
        unregister_driver(model, 1).unwrap();
        assert!(try_get(model, 1).is_none());
        register_driver_with_swap(model, 2, 16, 20, 4, 0, 0).unwrap();
        assert!(try_get(model, 2).is_some());
    }

    #[test]
    fn dynamic_store_slots_allow_global_driver_id_gaps() {
        let model = register_model(16, &[8], &[0]);
        register_driver_with_swap(model, 4, 16, 40, 4, 0, 0).unwrap();
        assert!(try_get(model, 1).is_none());
        assert!(try_get(model, 3).is_none());
        assert!(try_get(model, 4).is_some());
        assert_eq!(all_for_model(model).len(), 2);
    }
}
