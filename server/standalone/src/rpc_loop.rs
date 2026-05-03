//! Cold-path RPC dispatch loop.
//!
//! Hosts the wrapper-side cold-path RPC server that the runtime
//! connects to via `device::spawn(hostname, ...)`. The fast path
//! (`fire_batch`) is served by the C++ driver directly on `/pie_shmem`
//! and never reaches us.
//!
//! Mirrors the dispatch table in
//! `pie/src/pie_driver_portable/worker.py::_make_methods`. v0 supports
//! `ping` / `query` natively; the rest currently return clean
//! "not yet wired" errors. The aux-IPC `copy_*` / `load_adapter`
//! handlers land alongside the aux-IPC client (post-M2 work).

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde::Serialize;

use pie::device::RpcServer;

use crate::driver_ffi;

const POLL_TIMEOUT: Duration = Duration::from_millis(200);

/// Spawn the cold-path dispatch loop on a dedicated OS thread and
/// return its join handle. Stop by calling `server.close()` from the
/// outside — the loop exits the next time it polls.
pub fn spawn(server: Arc<RpcServer>) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("pie-rpc-{}", server.server_name()))
        .spawn(move || run(server))
        .expect("spawn rpc dispatch thread")
}

fn run(server: Arc<RpcServer>) {
    loop {
        match server.poll(POLL_TIMEOUT) {
            Ok(Some(req)) => {
                let response = dispatch(&req.method, &req.payload);
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
fn dispatch(method: &str, _payload: &[u8]) -> Vec<u8> {
    match method {
        "ping" => encode(&PingResp { ok: true }),
        "query" => encode(&QueryResp {
            driver: driver_ffi::FLAVOR,
            implemented: false,
        }),
        // Aux-IPC-backed methods. The dummy driver doesn't have an aux
        // socket — it accepts these as no-ops so adapter / page-copy
        // flows in inferlets succeed end-to-end against the dummy. The
        // C++ drivers wait on the aux-IPC client (post-M2 work).
        "copy_d2h" | "copy_h2d" | "copy_d2d" | "copy_h2h"
        | "swap_out_pages" | "swap_in_pages" | "load_adapter" => {
            if driver_ffi::FLAVOR == "dummy" {
                encode(&())
            } else {
                encode_err(format!(
                    "{method:?}: not yet wired in standalone server (aux-IPC client pending)"
                ))
            }
        }
        // Truly never-implemented, mirror Python's stubs.
        "embed_image" | "initialize_adapter" | "update_adapter" | "save_adapter" => {
            encode_err(format!(
                "{method:?}: not implemented in {} driver (post-v1)",
                driver_ffi::FLAVOR,
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

#[derive(Serialize)]
struct PingResp {
    ok: bool,
}

#[derive(Serialize)]
struct QueryResp {
    driver: &'static str,
    implemented: bool,
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
    use serde::Deserialize;

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

    #[test]
    fn ping_returns_ok_true() {
        let resp = dispatch("ping", &[]);
        let ping: PingDecoded = rmp_serde::from_slice(&resp).unwrap();
        assert!(ping.ok);
    }

    #[test]
    fn query_returns_driver_meta() {
        let resp = dispatch("query", &[]);
        let q: QueryDecoded = rmp_serde::from_slice(&resp).unwrap();
        assert_eq!(q.driver, driver_ffi::FLAVOR);
        assert!(!q.implemented);
    }

    #[test]
    fn unknown_method_errors() {
        let resp = dispatch("nope", &[]);
        let s = decode_str(&resp);
        assert!(s.contains("Method not found"), "got: {s}");
    }

    #[test]
    fn aux_ipc_methods_match_flavor_capability() {
        let resp = dispatch("copy_d2h", &[]);
        if driver_ffi::FLAVOR == "dummy" {
            // Dummy stubs aux-IPC methods as `()` so the runtime's
            // `device::call::<_, ()>` deserialize succeeds.
            let _: () = rmp_serde::from_slice(&resp)
                .expect("dummy: copy_d2h should decode as ()");
        } else {
            let s = decode_str(&resp);
            assert!(s.contains("aux-IPC client pending"), "got: {s}");
        }
    }
}
