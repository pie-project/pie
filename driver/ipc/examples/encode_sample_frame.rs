//! One-shot helper that emits a pre-encoded `Frame` to stdout (binary).
//! Used by `python/tests/bench_zero_copy.py` to produce a realistic
//! input payload without rebuilding the schema in Python.
//!
//! Usage: `cargo run --example encode_sample_frame -- 16384 > /tmp/frame.bin`

use std::io::Write;

use pie_ipc::{
    AdapterBinding, ForwardRequest, Frame, RequestPayload, Sampler, wire::encode_request,
};

fn main() {
    let n_tokens: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16_384);
    let token_ids: Vec<u32> = (0..n_tokens).map(|i| i as u32).collect();
    let position_ids: Vec<u32> = (0..n_tokens).map(|i| i as u32).collect();
    let mut req = ForwardRequest {
        token_ids,
        position_ids,
        qo_indptr: vec![0u32, n_tokens as u32],
        sampler_indptr: vec![0u32, 1],
        adapter_bindings: vec![AdapterBinding {
            adapter_id: -1,
            seed: -1,
        }],
        single_token_mode: false,
        has_user_mask: false,
        ..Default::default()
    };
    req.set_samplers(&[Sampler::TopKTopP {
        temperature: 0.7,
        k: 50,
        p: 0.9,
    }]);
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&frame).expect("encode");
    std::io::stdout().write_all(&bytes).expect("write stdout");
    eprintln!(
        "encoded Frame for n_tokens={n_tokens}: {} bytes",
        bytes.len()
    );
}
