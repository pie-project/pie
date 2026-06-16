//! Direct-FFI view emission benchmarks — measures the cost of
//! materializing a `PieForwardRequestView<'a>` from a native
//! `Frame`. This is the in-process C++ driver hot path (no rkyv).
//!
//! Compares against `wire.rs::encode_forward` so we can quantify the
//! "skip rkyv" speedup for in-process callers.
//!
//! Run: `cargo bench -p pie-bridge --features cabi --bench direct_ffi`

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use pie_bridge::{
    AdapterBinding, ForwardRequest, Frame, RequestPayload, Sampler, pie_forward_request_view,
    pie_frame_view,
};

fn make_frame(n_tokens: usize) -> Frame {
    let token_ids: Vec<u32> = (0..n_tokens).map(|i| i as u32).collect();
    let position_ids: Vec<u32> = (0..n_tokens).map(|i| i as u32).collect();
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

fn bench_forward_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_view");
    for &n in &[16usize, 256, 4096, 16_384] {
        let frame = make_frame(n);
        let inner = match &frame.payload {
            RequestPayload::Forward(r) => r,
            _ => unreachable!(),
        };
        group.bench_function(format!("tokens={n}"), |b| {
            b.iter(|| {
                let v = pie_forward_request_view(black_box(inner));
                black_box(v);
            });
        });
    }
    group.finish();
}

fn bench_frame_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_view");
    for &n in &[16usize, 256, 4096, 16_384] {
        let frame = make_frame(n);
        group.bench_function(format!("tokens={n}"), |b| {
            b.iter(|| {
                let v = pie_frame_view(black_box(&frame));
                black_box(v);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_forward_view, bench_frame_view);
criterion_main!(benches);
