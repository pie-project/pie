//! Compiling a checkpoint is linear in its tensor count.
//!
//! It was not. Six `iter().find()` calls inside per-instruction loops — two in
//! `plan/build.rs`, one in the arena pass, two in the coalescing pass, one in
//! the backend lowering — made `compile` quadratic, and a 32k-tensor checkpoint
//! took 2.1 s to plan. Every one of them was looking up a *dense* id, so the
//! fix was `LoadPlan::buffer` and friends, which index directly and refuse a
//! table whose ids are not dense.
//!
//! This asserts the shape of the curve rather than a wall-clock number, so it
//! means the same thing on a slow machine as on a fast one. The quadratic
//! version scaled at 3.5x per doubling; the linear one at ~2.2x.

use std::time::{Duration, Instant};

use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_loader::contract::{Expr, ModelContract, TensorContract};
use pie_loader::plan::{StorageTarget, compile};
use pie_loader::types::{CheckpointFormat, DType, Encoding, FileId, TensorId};

const SPAN: u64 = 8192;

fn fixture(n: u32) -> (CheckpointMetadata, ModelContract) {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "probe.safetensors".to_string(),
            size_bytes: u64::from(n) * SPAN,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: (0..n)
            .map(|i| RawTensor {
                id: TensorId(i),
                name: format!("model.layers.{i}.weight"),
                file_id: FileId(0),
                file_offset: u64::from(i) * SPAN,
                span_bytes: SPAN,
                shape: vec![64, 64],
                encoding: Encoding::Raw(DType::BF16),
            })
            .collect(),
    };
    let contract = ModelContract {
        alignment: 256,
        tensors: (0..n)
            .map(|i| {
                TensorContract::new(
                    format!("w{i}"),
                    Expr::src(format!("model.layers.{i}.weight")),
                    vec![64, 64],
                    Encoding::Raw(DType::BF16),
                )
            })
            .collect(),
        groups: Vec::new(),
    };
    (metadata, contract)
}

/// Best of three: this runs alongside whatever else the test binary is doing,
/// and a single sample would make a scheduling hiccup look like a regression.
fn compile_time(n: u32) -> Duration {
    let (metadata, contract) = fixture(n);
    (0..3)
        .map(|_| {
            let at = Instant::now();
            let plan = compile(&metadata, &contract, StorageTarget::default()).unwrap();
            assert_eq!(plan.tensors.len(), n as usize);
            at.elapsed()
        })
        .min()
        .unwrap()
}

#[test]
fn compiling_a_checkpoint_is_linear_in_its_tensor_count() {
    // Deliberately generous. The point is to catch a return to quadratic, which
    // is 4x per doubling, not to police a 10% regression.
    const LIMIT: f64 = 3.0;

    let base = compile_time(8_000).max(Duration::from_millis(1));
    let doubled = compile_time(16_000);
    let quadrupled = compile_time(32_000);

    let per_doubling = doubled.as_secs_f64() / base.as_secs_f64();
    let over_four = quadrupled.as_secs_f64() / base.as_secs_f64();
    assert!(
        per_doubling < LIMIT && over_four < LIMIT * LIMIT,
        "compile time is superlinear: 8k={base:?}, 16k={doubled:?} ({per_doubling:.1}x), \
         32k={quadrupled:?} ({over_four:.1}x)"
    );
}
