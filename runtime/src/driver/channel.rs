//! Per-driver channel + spec registry.
//!
//! One [`DriverChannel`] per driver, keyed by [`DriverId`]. Embedded
//! drivers install a channel before starting the driver thread;
//! subprocess drivers fall through to lazy shmem attach on first
//! request. The same channel carries every wire payload kind (Forward,
//! Copy, Adapter, Health) — the driver-side handler dispatches on the
//! [`pie_bridge::RequestPayload`] variant.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use dashmap::DashMap;

use super::{DriverChannel, DriverId, DriverRequest, DriverSpec, ShmemChannel};
use pie_bridge::{ForwardResponse, RequestPayload, ResponsePayload};

fn shmem_name(driver_idx: usize) -> String {
    format!("/pie_shmem_g{driver_idx}")
}

// Channel-level spin budgets are configured at startup via
// `serve::start_subprocess_group` (which pre-installs the channel
// with the user's `[model.driver].ipc_profile` / `spin_budget_us`). The lazy-attach
// fallback below uses the bridge's default budget; in practice no
// production path hits it because subprocess + embedded drivers
// both pre-install.

static CHANNELS: LazyLock<DashMap<DriverId, Arc<dyn DriverChannel>>> = LazyLock::new(DashMap::new);
static SPECS: LazyLock<DashMap<DriverId, DriverSpec>> = LazyLock::new(DashMap::new);
static NEXT_DRIVER_ID: AtomicUsize = AtomicUsize::new(0);

/// Install a channel for the given driver. Used by the embedded path
/// to register an in-process channel before the driver thread starts.
pub fn install_channel(driver_idx: DriverId, channel: Arc<dyn DriverChannel>) {
    CHANNELS.insert(driver_idx, channel);
}

/// Register the driver's static spec under a specific id.
pub fn install_spec(driver_idx: DriverId, spec: DriverSpec) {
    SPECS.insert(driver_idx, spec);
}

/// Allocate the next sequential driver id and register its spec.
/// Returns the id, which callers use to install a channel or address
/// the driver in `fire_batch`/`copy_*`/`load_adapter`.
pub fn register_driver(spec: DriverSpec) -> DriverId {
    let id = NEXT_DRIVER_ID.fetch_add(1, Ordering::Relaxed);
    SPECS.insert(id, spec);
    id
}

/// Return the driver's static configuration.
pub async fn get_spec(driver_idx: DriverId) -> Result<DriverSpec> {
    SPECS
        .get(&driver_idx)
        .map(|s| s.clone())
        .ok_or_else(|| anyhow!("no DriverSpec for driver {driver_idx}"))
}

pub(crate) fn get_channel(driver_idx: DriverId) -> Result<Arc<dyn DriverChannel>> {
    if let Some(c) = CHANNELS.get(&driver_idx) {
        return Ok(c.clone());
    }
    // Subprocess drivers may not have a pre-installed channel; lazy-attach
    // to the shmem region named `/pie_shmem_g{idx}`. The spin budget
    // here is a fallback default — in practice both subprocess and
    // embedded drivers pre-install via `install_channel` with the
    // user's `[model.driver].ipc_profile` / `spin_budget_us`, so this
    // branch is rarely hit. 1000 µs matches `InProcChannel::new()`'s
    // default.
    let name = shmem_name(driver_idx);
    let channel = ShmemChannel::open(&name, 1_000)?;
    let arc: Arc<dyn DriverChannel> = Arc::new(channel);
    CHANNELS.insert(driver_idx, arc.clone());
    Ok(arc)
}

/// Run a closure with a channel handle. Convenience for sync notify-style
/// calls in `ops::copy_*`.
pub(crate) fn with_channel<F, R>(driver_idx: DriverId, f: F) -> Result<R>
where
    F: FnOnce(&Arc<dyn DriverChannel>) -> Result<R>,
{
    let ch = get_channel(driver_idx)?;
    f(&ch)
}

/// Fire a batched forward pass. Wraps the request as a
/// [`pie_bridge::RequestPayload::Forward`] and unwraps the matching
/// [`pie_bridge::ResponsePayload::Forward`].
pub async fn fire_batch(
    driver_idx: DriverId,
    req: pie_bridge::ForwardRequest,
) -> Result<ForwardResponse> {
    let ch = get_channel(driver_idx)?;
    let resp = ch
        .submit(DriverRequest {
            driver_id: driver_idx,
            payload: RequestPayload::Forward(req),
        })
        .await?;
    match resp.payload {
        ResponsePayload::Forward(r) => Ok(r),
        ResponsePayload::Status(s) => Err(anyhow!(
            "fire_batch: driver returned cold-path status {} (driver bug)",
            s.status,
        )),
    }
}

/// Synchronous sibling of [`fire_batch`]. Used by the per-driver
/// scheduler loop, which runs on a dedicated OS thread (not a tokio
/// task) and so doesn't have an async context to `.await` in. Every
/// production `DriverChannel` already has a sync inner; this just
/// routes through the trait's `submit_sync`.
pub fn fire_batch_sync(
    driver_idx: DriverId,
    req: pie_bridge::ForwardRequest,
) -> Result<ForwardResponse> {
    let ch = get_channel(driver_idx)?;
    let resp = ch.submit_sync(DriverRequest {
        driver_id: driver_idx,
        payload: RequestPayload::Forward(req),
    })?;
    match resp.payload {
        ResponsePayload::Forward(r) => Ok(r),
        ResponsePayload::Status(s) => Err(anyhow!(
            "fire_batch_sync: driver returned cold-path status {} (driver bug)",
            s.status,
        )),
    }
}

/// Abort every active driver channel. Called by the supervisor when it
/// observes a driver exit.
pub fn abort_all_driver_channels() {
    for entry in CHANNELS.iter() {
        entry.value().abort();
    }
}
