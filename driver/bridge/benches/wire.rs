//! Wire-format benchmarks — rkyv encode/decode hot paths.
//!
//! Measures the rkyv-mediated read and write costs at sizes matching
//! realistic batches:
//!   * `forward_encode_<n>` — `rkyv::to_bytes` of a `Frame::Forward`
//!     carrying `n` tokens × 1 sampler/request.
//!   * `forward_decode_<n>` — `rkyv::access` then per-field slice reads.
//!   * `forward_roundtrip_<n>` — encode + access + read every field.
//!
//! Run: `cargo bench -p pie-bridge --features cabi --bench wire`

#![allow(clippy::useless_vec)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use pie_bridge::wire::{encode_request, parse_request};
use pie_bridge::{AdapterBinding, ForwardRequest, Frame, RequestPayload, Sampler};

// ----- Workload generator ----------------------------------------------------

fn make_frame(n_tokens: usize) -> Frame {
    let token_ids: Vec<u32> = (0..n_tokens).map(|i| i as u32).collect();
    let position_ids: Vec<u32> = (0..n_tokens).map(|i| i as u32).collect();
    // One request with all tokens; one sampler.
    let qo_indptr = vec![0u32, n_tokens as u32];
    let samplers = vec![Sampler::TopKTopP {
        temperature: 0.7,
        k: 50,
        p: 0.9,
    }];
    let sampler_indptr = vec![0u32, 1];
    let adapter_bindings = vec![AdapterBinding {
        adapter_id: -1,
        seed: -1,
    }];
    let req = ForwardRequest {
        token_ids,
        position_ids,
        qo_indptr,
        samplers,
        sampler_indptr,
        adapter_bindings,
        single_token_mode: false,
        has_user_mask: false,
        ..Default::default()
    };
    Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(req),
    }
}

// ----- Benchmarks ------------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_forward");
    for &n in &[16usize, 256, 4096, 16_384] {
        group.bench_function(format!("tokens={n}"), |b| {
            let frame = make_frame(n);
            b.iter(|| {
                let bytes = encode_request(black_box(&frame)).unwrap();
                black_box(bytes);
            });
        });
    }
    group.finish();
}

fn bench_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_forward");
    for &n in &[16usize, 256, 4096, 16_384] {
        let bytes = encode_request(&make_frame(n)).unwrap();
        group.bench_function(format!("tokens={n}"), |b| {
            b.iter(|| {
                let a = parse_request(black_box(&bytes)).unwrap();
                black_box(a);
            });
        });
    }
    group.finish();
}

fn bench_read_all_fields(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_all_fields_forward");
    for &n in &[16usize, 256, 4096, 16_384] {
        let bytes = encode_request(&make_frame(n)).unwrap();
        group.bench_function(format!("tokens={n}"), |b| {
            b.iter(|| {
                let a = parse_request(black_box(&bytes)).unwrap();
                // Force read of every Vec field — same workload as a C++
                // handler that walks the request.
                let driver_id = a.driver_id;
                let payload = &a.payload;
                let kind = match payload {
                    pie_bridge::ArchivedRequestPayload::Forward(fr) => {
                        black_box(&fr.token_ids);
                        black_box(&fr.position_ids);
                        black_box(&fr.qo_indptr);
                        black_box(&fr.samplers);
                        1u8
                    }
                    _ => 0u8,
                };
                black_box((driver_id, kind));
            });
        });
    }
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_forward");
    for &n in &[16usize, 256, 4096, 16_384] {
        group.bench_function(format!("tokens={n}"), |b| {
            b.iter_batched(
                || make_frame(n),
                |frame| {
                    let bytes = encode_request(&frame).unwrap();
                    let archived = parse_request(&bytes).unwrap();
                    black_box(archived);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ----- Response-side: `pie_build_response_frame` --------------------------
//
// Measures the writer hot path (Tier 1 wrote into the caller's buffer
// via rkyv::api::high::to_bytes_in). Compared to encode_forward, this
// includes the desc → native conversion overhead.

use pie_bridge::{
    PIE_RESPONSE_PAYLOAD_FORWARD, PieForwardResponseDesc, PieResponseFrameDesc,
    PieResponsePayloadDesc, PieStatusResponseDesc, pie_build_response_frame,
};

fn empty_fwd_resp_desc() -> PieForwardResponseDesc {
    PieForwardResponseDesc {
        num_requests: 1,
        ..Default::default()
    }
}

fn bench_build_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_response");
    for &n in &[16usize, 256, 4096, 16_384] {
        let tokens: Vec<u32> = (0..n).map(|i| i as u32).collect();
        let tokens_indptr = vec![0u32, n as u32];
        let mut fwd = empty_fwd_resp_desc();
        fwd.tokens_ptr = tokens.as_ptr();
        fwd.tokens_len = tokens.len();
        fwd.tokens_indptr_ptr = tokens_indptr.as_ptr();
        fwd.tokens_indptr_len = tokens_indptr.len();
        let desc = PieResponseFrameDesc {
            driver_id: 0,
            aborted: 0,
            payload: PieResponsePayloadDesc {
                kind: PIE_RESPONSE_PAYLOAD_FORWARD,
                forward: fwd,
                status: PieStatusResponseDesc { status: 0 },
            },
        };
        let mut buf = vec![0u8; 1 << 20];
        group.bench_function(format!("tokens={n}"), |b| {
            b.iter(|| {
                let written = unsafe {
                    pie_build_response_frame(
                        black_box(&desc as *const _),
                        buf.as_mut_ptr(),
                        buf.len(),
                    )
                };
                black_box(written);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_access,
    bench_read_all_fields,
    bench_roundtrip,
    bench_build_response,
);
criterion_main!(benches);
