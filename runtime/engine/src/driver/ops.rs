//! Cold-path method wrappers built directly on the wire schema. Each
//! function constructs a `pie_driver_abi::RequestPayload` (Copy or Adapter),
//! wraps it in a `DriverRequest` with the target driver_id, and ships
//! it through the unified channel — no pie-internal mirror enum.

use std::path::PathBuf;

use anyhow::Result;

use pie_driver_abi::{
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

/// CSM native audio output (pie:core/audio-out). Submits the prompt token ids +
/// `max_frames` as a JSON `AdapterRequest` with `op = GenerateAudio`; the CSM
/// driver runs the full generation (backbone prefill + per-frame depth loop +
/// Mimi decode), writes raw little-endian f32 PCM to a scratch file, and
/// returns `Status` = number of Mimi frames produced (negative on error). We
/// then read the PCM back and return it as f32 samples (24 kHz mono).
/// Returns an error if the model is not CSM or generation failed.
pub async fn generate_audio(
    driver_idx: DriverId,
    prompt: &[u32],
    max_frames: u32,
) -> Result<Vec<f32>> {
    // Unique scratch path the driver writes raw f32 PCM to. The embedded driver
    // shares this process's filesystem, so a temp file is the simplest reliable
    // handoff for the variable-length PCM (the Adapter Status response only
    // carries an i32 — see the AdapterOp::GenerateAudio doc).
    let pid = std::process::id();
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let out_path = std::env::temp_dir().join(format!("pie_csm_audio_{pid}_{nonce}.f32"));
    let out_path_str = out_path.to_string_lossy().to_string();

    let req_json = serde_json::json!({
        "prompt": prompt,
        "max_frames": max_frames,
        "out_path": out_path_str,
    })
    .to_string();

    let req = adapter_request(driver_idx, AdapterOp::GenerateAudio, 0, req_json);
    let ch = super::channel::get_channel(driver_idx)?;
    let resp = ch.submit(req).await?;
    let status = match resp.payload {
        ResponsePayload::Status(s) => s.status,
        ResponsePayload::Forward(_) => {
            let _ = std::fs::remove_file(&out_path);
            return Err(anyhow::anyhow!(
                "generate_audio received forward response (driver bug)"
            ));
        }
    };
    if status < 0 {
        let _ = std::fs::remove_file(&out_path);
        return Err(anyhow::anyhow!(
            "generate_audio failed in driver (status {status}); model may not be CSM"
        ));
    }
    // Read back the raw f32 PCM the driver wrote.
    let bytes = std::fs::read(&out_path).map_err(|e| {
        anyhow::anyhow!("generate_audio: reading PCM scratch {out_path_str}: {e}")
    })?;
    let _ = std::fs::remove_file(&out_path);
    let n = bytes.len() / 4;
    let mut pcm = Vec::with_capacity(n);
    for i in 0..n {
        let b = [bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]];
        pcm.push(f32::from_le_bytes(b));
    }
    Ok(pcm)
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
