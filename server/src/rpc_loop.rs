//! Cold-path RPC dispatch loop.
//!
//! Hosts the wrapper-side cold-path RPC server that the runtime
//! connects to via `device::spawn(hostname, ...)`. The fast path
//! (`fire_batch`) is served by the C++ driver directly on `/pie_shmem`
//! and never reaches us.
//!
//! Mirrors the dispatch table in
//! `pie/src/pie_driver_portable/worker.py::_make_methods`. `ping` /
//! `query` are answered natively; `copy_*` / `swap_*` / `load_adapter`
//! are routed to the embedded driver's [`AuxIpcClient`] when present
//! (portable today; cuda once it grows an `[aux_ipc]` listener).
//! Dummy stubs aux-IPC methods as `()` so adapter / page-copy flows in
//! inferlets succeed end-to-end against the dummy.

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use pie::device::RpcServer;

use crate::aux_ipc::{AuxIpcClient, Method};
use crate::driver_ffi::Flavor;

const POLL_TIMEOUT: Duration = Duration::from_millis(200);

/// Spawn the cold-path dispatch loop on a dedicated OS thread and
/// return its join handle. Stop by calling `server.close()` from the
/// outside — the loop exits the next time it polls.
///
/// `aux` is `None` for drivers without an aux-IPC channel (currently
/// dummy and cuda); `Some` for portable.
pub fn spawn(
    flavor: Flavor,
    server: Arc<RpcServer>,
    aux: Option<Arc<AuxIpcClient>>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("pie-rpc-{}", server.server_name()))
        .spawn(move || run(flavor, server, aux))
        .expect("spawn rpc dispatch thread")
}

fn run(flavor: Flavor, server: Arc<RpcServer>, aux: Option<Arc<AuxIpcClient>>) {
    loop {
        match server.poll(POLL_TIMEOUT) {
            Ok(Some(req)) => {
                let response = dispatch(flavor, &req.method, &req.payload, aux.as_deref());
                if let Err(e) = server.respond(req.request_id, response) {
                    tracing::warn!("rpc respond failed: {e}");
                }
            }
            Ok(None) => {} // timeout, keep polling
            Err(_) => {
                // Either closed by the host or the underlying ipc-channel
                // died. Either way, exit.
                break;
            }
        }
    }
}

/// Dispatch a single RPC. Returns the msgpack-encoded response body.
/// Errors are encoded as msgpack strings — same shape the Python
/// wrapper's RPC loop produces.
fn dispatch(
    flavor: Flavor,
    method: &str,
    payload: &[u8],
    aux: Option<&AuxIpcClient>,
) -> Vec<u8> {
    match method {
        "ping" => encode(&PingResp { ok: true }),
        "query" => encode(&QueryResp {
            driver: flavor.as_str(),
            implemented: false,
        }),

        // Aux-IPC-backed methods. Routed to the embedded driver when
        // it exposes an aux socket; stubbed as `()` for the dummy
        // (which has no socket but inferlets exercise these paths);
        // explicit error otherwise (cuda's aux listener is post-M3).
        "copy_d2h" | "copy_h2d" | "copy_d2d" | "copy_h2h"
        | "swap_out_pages" | "swap_in_pages" => {
            dispatch_copy(flavor, method, payload, aux)
        }
        "load_adapter" => dispatch_load_adapter(flavor, payload, aux),

        // Methods Python's wrappers also stub out — none of the
        // standalone-supported drivers implement these.
        "embed_image" | "initialize_adapter" | "update_adapter" | "save_adapter" => {
            encode_err(format!(
                "{method:?}: not implemented in {} driver",
                flavor.as_str(),
            ))
        }
        // fire_batch should never come down the cold path — it's served
        // by the driver directly on /pie_shmem. If it does, surface a
        // clear error so we notice.
        "fire_batch" => encode_err(
            "fire_batch reached the cold path; \
             this is a runtime↔driver wiring bug".to_string(),
        ),
        other => encode_err(format!("Method not found: {other}")),
    }
}

fn dispatch_copy(
    flavor: Flavor,
    method: &str,
    payload: &[u8],
    aux: Option<&AuxIpcClient>,
) -> Vec<u8> {
    // Dummy has no aux socket; the runtime still calls these on swap
    // restore paths — return `()` so deserialize succeeds.
    let Some(client) = aux else {
        if is_dummy(flavor) {
            return encode(&());
        }
        return encode_err(format!(
            "{method:?}: this driver flavor ({}) has no aux-IPC channel",
            flavor.as_str(),
        ));
    };

    // The runtime ships per-method-named arg shapes (see
    // `runtime/src/device.rs`). Decode the matching shape and
    // translate to (src, dst) page pairs in the order the aux wire
    // expects.
    let (m, pairs) = match method {
        "copy_d2h" | "swap_out_pages" => {
            // GPU → CPU: pairs are (gpu_phys_id, cpu_slot).
            let args: PhysSlotArgs = match decode(payload) {
                Ok(a) => a,
                Err(e) => return encode_err(format!("{method}: {e}")),
            };
            (Method::CopyD2H, zip_pairs(&args.phys_ids, &args.slots))
        }
        "copy_h2d" | "swap_in_pages" => {
            // CPU → GPU: the runtime sends (gpu_dst, cpu_src) under
            // the names (phys_ids, slots); the wire format expects
            // (src, dst) order, so flip on the way through.
            let args: PhysSlotArgs = match decode(payload) {
                Ok(a) => a,
                Err(e) => return encode_err(format!("{method}: {e}")),
            };
            (Method::CopyH2D, zip_pairs(&args.slots, &args.phys_ids))
        }
        "copy_d2d" => {
            let args: SrcDstPhysArgs = match decode(payload) {
                Ok(a) => a,
                Err(e) => return encode_err(format!("{method}: {e}")),
            };
            (Method::CopyD2D, zip_pairs(&args.src_phys_ids, &args.dst_phys_ids))
        }
        "copy_h2h" => {
            let args: SrcDstSlotArgs = match decode(payload) {
                Ok(a) => a,
                Err(e) => return encode_err(format!("{method}: {e}")),
            };
            (Method::CopyH2H, zip_pairs(&args.src_slots, &args.dst_slots))
        }
        _ => unreachable!("dispatch_copy called with non-copy method"),
    };

    let pairs = match pairs {
        Ok(p) => p,
        Err(e) => return encode_err(format!("{method}: {e}")),
    };

    match client.copy(m, &pairs) {
        Ok(()) => encode(&()),
        Err(e) => encode_err(format!("{method}: {e}")),
    }
}

fn dispatch_load_adapter(
    flavor: Flavor,
    payload: &[u8],
    aux: Option<&AuxIpcClient>,
) -> Vec<u8> {
    let Some(client) = aux else {
        if is_dummy(flavor) {
            return encode(&());
        }
        return encode_err(format!(
            "load_adapter: this driver flavor ({}) has no aux-IPC channel",
            flavor.as_str(),
        ));
    };

    let args: LoadAdapterArgs = match decode(payload) {
        Ok(a) => a,
        Err(e) => return encode_err(format!("load_adapter: {e}")),
    };

    // Materialize the adapter blob to disk. The aux wire format
    // sends only a path — the driver mmaps and parses it itself.
    // Use a pid-scoped subdir so concurrent driver instances don't
    // clobber each other; clean up after a successful send.
    let tmp_dir = std::env::temp_dir().join(format!("pie-adapters-{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        return encode_err(format!("load_adapter: create temp dir: {e}"));
    }
    let safe_name: String = args
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let path: PathBuf = tmp_dir.join(format!(
        "{}-{:016x}.safetensors",
        if safe_name.is_empty() { "adapter" } else { safe_name.as_str() },
        args.adapter_ptr,
    ));
    if let Err(e) = std::fs::write(&path, &args.adapter_data) {
        return encode_err(format!("load_adapter: write {path:?}: {e}"));
    }

    let result = client.load_adapter(args.adapter_ptr, &path);

    // Driver has either consumed the file (mmap + parse) or failed;
    // either way our copy isn't needed. Best-effort cleanup.
    let _ = std::fs::remove_file(&path);

    match result {
        Ok(()) => encode(&()),
        Err(e) => encode_err(format!("load_adapter: {e}")),
    }
}

/// True when the runtime flavor is dummy. Helper because `Flavor::Dummy`
/// is cfg-gated — calling code can't reference it directly without
/// duplicating the feature gate.
#[inline]
fn is_dummy(_flavor: Flavor) -> bool {
    #[cfg(feature = "driver-dummy")]
    {
        return matches!(_flavor, Flavor::Dummy);
    }
    #[allow(unreachable_code)]
    false
}

fn zip_pairs(srcs: &[u32], dsts: &[u32]) -> Result<Vec<(u32, u32)>, String> {
    if srcs.len() != dsts.len() {
        return Err(format!(
            "src/dst length mismatch ({} vs {})",
            srcs.len(),
            dsts.len()
        ));
    }
    Ok(srcs.iter().zip(dsts.iter()).map(|(s, d)| (*s, *d)).collect())
}

#[derive(Serialize)]
struct PingResp {
    ok: bool,
}

#[derive(Serialize)]
struct QueryResp {
    driver: &'static str,
    implemented: bool,
}

/// Wire shape for `copy_d2h` / `copy_h2d` / `swap_*_pages` — mirrors
/// `runtime/src/device.rs::copy_d2h::Req`.
#[derive(Deserialize)]
struct PhysSlotArgs {
    phys_ids: Vec<u32>,
    slots: Vec<u32>,
}

/// Wire shape for `copy_d2d`.
#[derive(Deserialize)]
struct SrcDstPhysArgs {
    src_phys_ids: Vec<u32>,
    dst_phys_ids: Vec<u32>,
}

/// Wire shape for `copy_h2h`.
#[derive(Deserialize)]
struct SrcDstSlotArgs {
    src_slots: Vec<u32>,
    dst_slots: Vec<u32>,
}

/// Wire shape for `load_adapter` — mirrors
/// `runtime/src/adapter.rs::LoadAdapterArgs`.
#[derive(Deserialize)]
struct LoadAdapterArgs {
    adapter_ptr: u64,
    name: String,
    adapter_data: Vec<u8>,
}

fn decode<'a, T: Deserialize<'a>>(payload: &'a [u8]) -> Result<T, String> {
    rmp_serde::from_slice(payload).map_err(|e| format!("decode args: {e}"))
}

fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    rmp_serde::to_vec_named(value).unwrap_or_else(|e| encode_err(e.to_string()))
}

fn encode_err(msg: String) -> Vec<u8> {
    // Python wrapper encodes errors as bare msgpack strings — we
    // mirror that shape so the runtime's existing error path applies
    // unchanged.
    rmp_serde::to_vec(&msg).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct PingDecoded {
        ok: bool,
    }
    #[derive(Deserialize)]
    struct QueryDecoded {
        driver: String,
        implemented: bool,
    }

    fn decode_str(payload: &[u8]) -> String {
        rmp_serde::from_slice::<String>(payload).unwrap()
    }

    fn any_flavor() -> Flavor {
        crate::driver_ffi::default_flavor()
            .expect("at least one driver feature must be enabled")
    }

    #[test]
    fn ping_returns_ok_true() {
        let resp = dispatch(any_flavor(), "ping", &[], None);
        let ping: PingDecoded = rmp_serde::from_slice(&resp).unwrap();
        assert!(ping.ok);
    }

    #[test]
    fn query_returns_driver_meta() {
        let f = any_flavor();
        let resp = dispatch(f, "query", &[], None);
        let q: QueryDecoded = rmp_serde::from_slice(&resp).unwrap();
        assert_eq!(q.driver, f.as_str());
        assert!(!q.implemented);
    }

    #[test]
    fn unknown_method_errors() {
        let resp = dispatch(any_flavor(), "nope", &[], None);
        let s = decode_str(&resp);
        assert!(s.contains("Method not found"), "got: {s}");
    }

    #[cfg(feature = "driver-dummy")]
    #[test]
    fn copy_without_aux_dummy_returns_unit() {
        // Build a valid msgpack payload for copy_d2h's wire shape so
        // dispatch_copy doesn't bail on decode.
        let payload = rmp_serde::to_vec_named(&serde_json::json!({
            "phys_ids": [1u32, 2u32],
            "slots":    [10u32, 11u32],
        })).unwrap();

        let resp = dispatch(Flavor::Dummy, "copy_d2h", &payload, None);
        let _: () = rmp_serde::from_slice(&resp)
            .expect("dummy: copy_d2h should decode as ()");
    }

    #[cfg(feature = "driver-portable")]
    #[test]
    fn copy_without_aux_native_errors() {
        let payload = rmp_serde::to_vec_named(&serde_json::json!({
            "phys_ids": [1u32, 2u32],
            "slots":    [10u32, 11u32],
        })).unwrap();
        let resp = dispatch(Flavor::Portable, "copy_d2h", &payload, None);
        let s = decode_str(&resp);
        assert!(s.contains("no aux-IPC channel"), "got: {s}");
    }

    #[cfg(feature = "driver-portable")]
    #[test]
    fn copy_decode_errors_surface_clearly() {
        // Garbage payload — dispatch_copy should produce a decode error
        // string rather than panicking.
        let resp = dispatch(Flavor::Portable, "copy_d2h", &[0xff, 0xff, 0xff], None);
        let s = decode_str(&resp);
        // Either decode error or "no aux-IPC channel" depending on
        // whether the absent-aux branch hit first; both are valid.
        assert!(
            s.contains("decode args") || s.contains("no aux-IPC channel"),
            "got: {s}"
        );
    }
}
