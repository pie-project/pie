//! Cold-path method wrappers built directly on the wire schema. Each
//! function constructs a `pie_bridge::RequestPayload` (Copy or Adapter),
//! wraps it in a `DriverRequest` with the target driver_id, and ships
//! it through the unified channel — no pie-internal mirror enum.

use std::path::PathBuf;

use anyhow::Result;

use pie_bridge::{
    AdapterOp, AdapterRequest, CopyDir, CopyRequest, CopyResource, RequestPayload, ResponsePayload,
};

use super::channel::with_channel;
use super::{DriverId, DriverRequest};

fn copy_request(
    driver_idx: DriverId,
    resource: CopyResource,
    dir: CopyDir,
    srcs: Vec<u32>,
    dsts: Vec<u32>,
) -> DriverRequest {
    DriverRequest {
        driver_id: driver_idx,
        payload: RequestPayload::Copy(CopyRequest {
            dir,
            srcs,
            dsts,
            resource,
        }),
    }
}

fn adapter_request(
    driver_idx: DriverId,
    op: AdapterOp,
    adapter_id: u64,
    path: String,
) -> DriverRequest {
    DriverRequest {
        driver_id: driver_idx,
        payload: RequestPayload::Adapter(AdapterRequest {
            op,
            adapter_id,
            path,
        }),
    }
}

/// GPU → CPU page copy (fire-and-forget).
pub fn copy_d2h(driver_idx: DriverId, gpu_phys_ids: &[u32], cpu_pages: &[u32]) -> Result<()> {
    with_channel(driver_idx, |ch| {
        ch.notify(copy_request(
            driver_idx,
            CopyResource::Kv,
            CopyDir::D2H,
            gpu_phys_ids.to_vec(),
            cpu_pages.to_vec(),
        ))
    })
}

/// CPU → GPU page copy (fire-and-forget).
pub fn copy_h2d(driver_idx: DriverId, gpu_phys_ids: &[u32], cpu_pages: &[u32]) -> Result<()> {
    with_channel(driver_idx, |ch| {
        ch.notify(copy_request(
            driver_idx,
            CopyResource::Kv,
            CopyDir::H2D,
            cpu_pages.to_vec(),
            gpu_phys_ids.to_vec(),
        ))
    })
}

/// GPU → GPU page copy (fire-and-forget).
pub fn copy_d2d(driver_idx: DriverId, src_phys_ids: &[u32], dst_phys_ids: &[u32]) -> Result<()> {
    with_channel(driver_idx, |ch| {
        ch.notify(copy_request(
            driver_idx,
            CopyResource::Kv,
            CopyDir::D2D,
            src_phys_ids.to_vec(),
            dst_phys_ids.to_vec(),
        ))
    })
}

/// CPU → CPU page copy (fire-and-forget).
pub fn copy_h2h(driver_idx: DriverId, src_slots: &[u32], dst_slots: &[u32]) -> Result<()> {
    with_channel(driver_idx, |ch| {
        ch.notify(copy_request(
            driver_idx,
            CopyResource::Kv,
            CopyDir::H2H,
            src_slots.to_vec(),
            dst_slots.to_vec(),
        ))
    })
}

/// GPU → GPU recurrent-state slot copy (fire-and-forget).
pub fn copy_rs_d2d(driver_idx: DriverId, src_slots: &[u32], dst_slots: &[u32]) -> Result<()> {
    with_channel(driver_idx, |ch| {
        ch.notify(copy_request(
            driver_idx,
            CopyResource::Rs,
            CopyDir::D2D,
            src_slots.to_vec(),
            dst_slots.to_vec(),
        ))
    })
}

/// Load a LoRA adapter from a safetensors path. Awaits the driver's
/// status; non-zero status surfaces as an error.
pub async fn load_adapter(driver_idx: DriverId, adapter_id: u64, path: PathBuf) -> Result<()> {
    // Empty string when the path isn't valid UTF-8 — the driver side
    // treats empty as "no path" (it would reject a Load with no path
    // anyway).
    let path_str = path.to_str().map(|s| s.to_string()).unwrap_or_default();
    submit_adapter_op(
        adapter_request(driver_idx, AdapterOp::Load, adapter_id, path_str),
        "load_adapter",
    )
    .await
}

/// Save a LoRA adapter — driver-side no-op today; the dispatch path
/// exists so a future driver can opt in by replacing its stub.
pub async fn save_adapter(driver_idx: DriverId, adapter_id: u64) -> Result<()> {
    submit_adapter_op(
        adapter_request(driver_idx, AdapterOp::Save, adapter_id, String::new()),
        "save_adapter",
    )
    .await
}

/// Zeroth-order initialization for an adapter — driver-side no-op.
pub async fn zo_initialize_adapter(driver_idx: DriverId, adapter_id: u64) -> Result<()> {
    submit_adapter_op(
        adapter_request(driver_idx, AdapterOp::ZoInit, adapter_id, String::new()),
        "zo_initialize_adapter",
    )
    .await
}

/// Zeroth-order update for an adapter — driver-side no-op.
pub async fn zo_update_adapter(driver_idx: DriverId, adapter_id: u64) -> Result<()> {
    submit_adapter_op(
        adapter_request(driver_idx, AdapterOp::ZoUpdate, adapter_id, String::new()),
        "zo_update_adapter",
    )
    .await
}

async fn submit_adapter_op(req: DriverRequest, method_name: &'static str) -> Result<()> {
    let ch = super::channel::get_channel(req.driver_id)?;
    let resp = ch.submit(req).await?;
    match resp.payload {
        ResponsePayload::Status(s) if s.status == 0 => Ok(()),
        ResponsePayload::Status(s) => Err(anyhow::anyhow!(
            "{method_name} returned status {}",
            s.status
        )),
        ResponsePayload::Forward(_) => Err(anyhow::anyhow!(
            "{method_name} received forward response (driver bug)"
        )),
    }
}
