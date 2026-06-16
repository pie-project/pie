//! In-process shmem round-trip tests over the rkyv-archived payload.

#![cfg(feature = "ipc")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use pie_bridge::SCHEMA_HASH;
use pie_bridge::ipc::{ShmemClient, ShmemServer};
use pie_bridge::wire::{WireError, encode_request, encode_response, parse_request, parse_response};
use pie_bridge::{
    AdapterBinding, ForwardRequest, ForwardResponse, Frame, RequestPayload, ResponseFrame,
    ResponsePayload, Sampler, StatusResponse,
};

const SPIN_BUDGET_US: u64 = 100;

fn build_health_frame(driver_id: u32) -> Vec<u8> {
    encode_request(&Frame {
        driver_id,
        payload: RequestPayload::Health,
    })
    .unwrap()
}

fn build_status_response(driver_id: u32, status: i32) -> Vec<u8> {
    encode_response(&ResponseFrame {
        driver_id,
        aborted: false,
        payload: ResponsePayload::Status(StatusResponse { status }),
    })
    .unwrap()
}

fn unique_name(tag: &str) -> String {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("/pie_bridge_test_{tag}_{}_{n}", std::process::id())
}

#[test]
fn handshake_schema_hash_mismatch_rejected() {
    let name = unique_name("hash_mismatch");
    let _server = ShmemServer::create(&name, 2, 4096, 4096, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();
    let bad_hash = [0xFFu8; 8];
    let result = ShmemClient::open(&name, SPIN_BUDGET_US, bad_hash);
    let err = result.err().expect("expected schema_hash mismatch");
    assert!(format!("{err}").contains("schema_hash"));
}

#[test]
fn roundtrip_health_request() {
    let name = unique_name("health");
    let server = ShmemServer::create(&name, 2, 4096, 4096, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();
    let server_arc = Arc::new(server);
    let stop = Arc::new(AtomicBool::new(false));

    let server_thread = {
        let server = server_arc.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(lease) = server.poll() {
                    let frame = parse_request(lease.payload()).expect("parse");
                    let driver_id: u32 = frame.driver_id.into();
                    let resp = build_status_response(driver_id, 0);
                    lease.commit(&resp).expect("commit");
                } else {
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
        })
    };

    let client = ShmemClient::open(&name, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();
    let payload = build_health_frame(7);
    let resp_bytes = client.roundtrip(1, &payload).expect("roundtrip");

    let frame = parse_response(&resp_bytes).expect("parse response");
    let dr: u32 = frame.driver_id.into();
    assert_eq!(dr, 7);
    assert!(!frame.aborted);

    stop.store(true, Ordering::Relaxed);
    server_arc.stop();
    server_thread.join().unwrap();
}

#[test]
fn lease_drop_writes_abort() {
    let name = unique_name("abort");
    let server = ShmemServer::create(&name, 2, 4096, 4096, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();
    let server_arc = Arc::new(server);
    let stop = Arc::new(AtomicBool::new(false));

    let server_thread = {
        let server = server_arc.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(lease) = server.poll() {
                    drop(lease);
                } else {
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
        })
    };

    let client = ShmemClient::open(&name, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();
    let payload = build_health_frame(99);
    let resp_bytes = client.roundtrip(1, &payload).expect("roundtrip");
    let result = parse_response(&resp_bytes);
    assert!(matches!(result.err(), Some(WireError::HandlerAborted)));

    stop.store(true, Ordering::Relaxed);
    server_arc.stop();
    server_thread.join().unwrap();
}

#[test]
fn multiple_concurrent_clients() {
    let name = unique_name("concurrent");
    let server = ShmemServer::create(&name, 4, 4096, 4096, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();
    let server_arc = Arc::new(server);
    let stop = Arc::new(AtomicBool::new(false));

    let server_thread = {
        let server = server_arc.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(lease) = server.poll() {
                    let frame = parse_request(lease.payload()).expect("parse");
                    let driver_id: u32 = frame.driver_id.into();
                    lease
                        .commit(&build_status_response(driver_id, 0))
                        .expect("commit");
                } else {
                    std::thread::sleep(Duration::from_micros(100));
                }
            }
        })
    };

    let client = Arc::new(ShmemClient::open(&name, SPIN_BUDGET_US, SCHEMA_HASH).unwrap());
    let mut handles = vec![];
    for id in 0..8u32 {
        let client = client.clone();
        handles.push(std::thread::spawn(move || {
            let payload = build_health_frame(id);
            let resp = client.roundtrip(id, &payload).expect("roundtrip");
            let frame = parse_response(&resp).expect("parse");
            let dr: u32 = frame.driver_id.into();
            assert_eq!(dr, id);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    server_arc.stop();
    server_thread.join().unwrap();
}

/// Full IPC e2e for a non-trivial Forward request. Builds a request
/// with realistic SoA fields + multiple sampler variants + adapter
/// bindings + spec drafts, ships it through shmem, has the server
/// parse + verify every field, build a Forward response with per-
/// request tokens, ship back. Verifies bytes-on-wire correctness for
/// the realistic Python driver shape.
#[test]
fn forward_request_full_roundtrip_through_shmem() {
    let name = unique_name("fwd_full");
    let server =
        ShmemServer::create(&name, 2, 1 << 20, 1 << 20, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();
    let server_arc = Arc::new(server);
    let stop = Arc::new(AtomicBool::new(false));

    let server_thread = {
        let server = server_arc.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(lease) = server.poll() {
                    let frame = parse_request(lease.payload()).expect("server parse");
                    let driver_id: u32 = frame.driver_id.into();
                    assert_eq!(driver_id, 7);
                    match &frame.payload {
                        pie_bridge::ArchivedRequestPayload::Forward(fr) => {
                            assert_eq!(fr.token_ids.len(), 5);
                            assert_eq!(fr.position_ids.len(), 5);
                            assert_eq!(fr.qo_indptr.len(), 3);
                            assert_eq!(fr.samplers.len(), 2);
                            assert_eq!(fr.adapter_bindings.len(), 2);
                            assert_eq!(fr.spec_token_ids.len(), 3);
                        }
                        _ => panic!("expected forward variant"),
                    }
                    let resp = ResponseFrame {
                        driver_id,
                        aborted: false,
                        payload: ResponsePayload::Forward(ForwardResponse {
                            num_requests: 2,
                            tokens_indptr: vec![0, 2, 3],
                            tokens: vec![100, 200, 300],
                            ..Default::default()
                        }),
                    };
                    let bytes = encode_response(&resp).expect("encode resp");
                    lease.commit(&bytes).expect("commit");
                } else {
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
        })
    };

    let client = ShmemClient::open(&name, SPIN_BUDGET_US, SCHEMA_HASH).unwrap();

    let req = ForwardRequest {
        token_ids: vec![10, 20, 30, 40, 50],
        position_ids: vec![0, 1, 2, 0, 1],
        qo_indptr: vec![0, 3, 5],
        kv_page_indptr: vec![0, 1, 2],
        kv_page_indices: vec![100, 200],
        kv_last_page_lens: vec![3, 2],
        samplers: vec![
            Sampler::Multinomial {
                temperature: 0.7,
                seed: 42,
            },
            Sampler::TopK {
                temperature: 0.5,
                k: 40,
            },
        ],
        sampler_indptr: vec![0, 1, 2],
        adapter_bindings: vec![
            AdapterBinding {
                adapter_id: 101,
                seed: -1,
            },
            AdapterBinding {
                adapter_id: -1,
                seed: -7,
            },
        ],
        spec_token_ids: vec![900, 901, 902],
        spec_position_ids: vec![5, 6, 7],
        spec_indptr: vec![0, 1, 3],
        context_ids: vec![0xCAFE, 0xBABE],
        single_token_mode: false,
        has_user_mask: false,
        ..Default::default()
    };
    let payload = encode_request(&Frame {
        driver_id: 7,
        payload: RequestPayload::Forward(req),
    })
    .unwrap();

    let resp_bytes = client.roundtrip(1, &payload).expect("roundtrip");

    let resp_archived = parse_response(&resp_bytes).expect("parse resp");
    let dr: u32 = resp_archived.driver_id.into();
    assert_eq!(dr, 7);
    assert!(!resp_archived.aborted);
    match &resp_archived.payload {
        pie_bridge::ArchivedResponsePayload::Forward(fr) => {
            let nr: u32 = fr.num_requests.into();
            assert_eq!(nr, 2);
            assert_eq!(fr.tokens.len(), 3);
            let t0: u32 = fr.tokens[0].into();
            let t1: u32 = fr.tokens[1].into();
            let t2: u32 = fr.tokens[2].into();
            assert_eq!((t0, t1, t2), (100, 200, 300));
        }
        _ => panic!("expected forward response variant"),
    }

    stop.store(true, Ordering::Relaxed);
    server_arc.stop();
    server_thread.join().unwrap();
}
