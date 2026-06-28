//! eval-mock executor test (lane L7 / hotel).
//!
//! Proves the host-side eval-mock path: a sampling-program riding the bridge
//! `ForwardRequest` carrier is executed through the canonical CPU `eval`
//! interpreter over deterministic synthetic logits, returning a token — the
//! 4090-free plumbing gate that lights up golf's `program_token` decode.
//!
//! This drives the device-level `SamplingProgramBehavior` directly (no GPU,
//! no model). The numeric-parity gate on real logits is echo/delta's lane.

mod common;
use common::mock_device::{run_program_mock, synthetic_logits, SamplingProgramBehavior};

use pie_driver_abi::{ForwardRequest, SamplingBinding, SamplingProgramSubmission};
use pie_sampling_ir::bytecode;
use pie_sampling_ir::types::*;

const VOCAB: u32 = 4096;

fn argmax_program() -> SamplingProgram {
    // Binding-free v4: Input(0) = [VOCAB] f32 logits; ReduceArgmax → i32 token.
    SamplingProgram {
        inputs: vec![InputDecl::new(Shape::vector(VOCAB), DType::F32)],
        ops: vec![Op::Input(0), Op::ReduceArgmax(0)],
        outputs: vec![OutputDecl::new(1, OutputKind::Token)],
    }
}

fn cpu_argmax(v: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0u32;
    for (j, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            bi = j as u32;
        }
    }
    bi
}

fn request_with_program(bytecode: Vec<u8>, n_requests: u32) -> ForwardRequest {
    let mut req = ForwardRequest {
        qo_indptr: (0..=n_requests).collect(),
        ..Default::default()
    };
    req.push_sampling_program(&SamplingProgramSubmission {
        bytecode,
        inputs: Vec::new(),
        bindings: vec![SamplingBinding::Logits],
        late_keys: Vec::new(),
        late_inputs: Vec::new(),
    });
    req
}

/// `run_program_mock` over an argmax program == CPU argmax of the same logits.
#[test]
fn run_program_mock_argmax() {
    let bytecode = bytecode::encode(&argmax_program());
    let sub = SamplingProgramSubmission { bytecode, inputs: Vec::new(), bindings: vec![SamplingBinding::Logits], late_keys: Vec::new(), late_inputs: Vec::new() };
    for req_id in 0..16u64 {
        let logits = synthetic_logits(req_id, VOCAB as usize);
        let tok = run_program_mock(&sub, &logits).expect("eval-mock");
        assert_eq!(tok, cpu_argmax(&logits), "req {req_id}");
    }
}

/// Full device path: a program-bearing `ForwardRequest` → `SamplingProgramBehavior`
/// → `ForwardResponse` with one argmax token per request (== eval over synthetic
/// logits seeded by request index). This is exactly what the host reconstructs
/// into `SlotOutput::Token` for golf's `program_token(h[0])`.
#[test]
fn sampling_program_behavior_emits_argmax_tokens() {
    use common::mock_device::Behavior;

    const N: u32 = 4;
    let bytecode = bytecode::encode(&argmax_program());
    let req = request_with_program(bytecode, N);
    let beh = SamplingProgramBehavior { fallback: 999 };
    let resp = beh.handle_fire_batch(&req);

    assert_eq!(resp.tokens.len(), N as usize, "one token per request");
    assert_eq!(resp.tokens_indptr, (0..=N).collect::<Vec<_>>());
    for r in 0..N as u64 {
        let want = cpu_argmax(&synthetic_logits(r, VOCAB as usize));
        assert_eq!(resp.tokens[r as usize], want, "request {r}");
    }
}

/// A request with no program falls back (no panic, no eval).
#[test]
fn behavior_without_program_falls_back() {
    use common::mock_device::Behavior;

    let req = ForwardRequest { qo_indptr: vec![0, 1, 2], ..Default::default() };
    let beh = SamplingProgramBehavior { fallback: 7 };
    let resp = beh.handle_fire_batch(&req);
    assert_eq!(resp.tokens, vec![7, 7]);
}

// A mirostat-shaped program: outputs [Token, Scalar]. Token = argmax(logits),
// Scalar = max logit (any scalar reduction — we only test the marshaling path).
fn token_and_scalar_program() -> SamplingProgram {
    // mirostat shape: argmax → token (i32), max-logit → scalar S (f32).
    SamplingProgram {
        inputs: vec![InputDecl::new(Shape::vector(VOCAB), DType::F32)],
        ops: vec![Op::Input(0), Op::ReduceArgmax(0), Op::ReduceMax(0)],
        outputs: vec![
            OutputDecl::new(1, OutputKind::Token),  // argmax → token
            OutputDecl::new(2, OutputKind::Scalar), // max logit → scalar S
        ],
    }
}

/// Multi-output marshaling: a `[Token, Scalar]` program (mirostat shape) must
/// populate BOTH the tokens channel and the entropies (scalar) channel of the
/// `ForwardResponse`, so the host reconstructs `[Token, Entropy(S)]` and golf's
/// `program_scalar(h[1])` returns S — enabling the mirostat μ-update e2e.
#[test]
fn sampling_program_behavior_marshals_token_and_scalar() {
    use common::mock_device::Behavior;

    const N: u32 = 4;
    let bytecode = bytecode::encode(&token_and_scalar_program());
    let req = request_with_program(bytecode, N);
    let beh = SamplingProgramBehavior { fallback: 0 };
    let resp = beh.handle_fire_batch(&req);

    // Token channel: one argmax token per request.
    assert_eq!(resp.tokens.len(), N as usize, "one token per request");
    // Scalar channel: one entropy/scalar value per request (== max logit).
    assert_eq!(resp.entropies.len(), N as usize, "one scalar per request");
    assert_eq!(resp.entropies_indptr, (0..=N).collect::<Vec<_>>(), "1 scalar per req");

    for r in 0..N as u64 {
        let logits = synthetic_logits(r, VOCAB as usize);
        assert_eq!(resp.tokens[r as usize], cpu_argmax(&logits), "token req {r}");
        let want_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((resp.entropies[r as usize] - want_max).abs() < 1e-4, "scalar req {r}");
    }
}

/// `run_program_mock_outputs` returns every declared output paired with its kind.
#[test]
fn run_program_mock_outputs_returns_all() {
    use common::mock_device::run_program_mock_outputs;
    use pie_sampling_ir::eval::Value;
    use pie_sampling_ir::OutputKind;

    let bytecode = bytecode::encode(&token_and_scalar_program());
    let sub = SamplingProgramSubmission {
        bytecode,
        inputs: Vec::new(),
        bindings: vec![SamplingBinding::Logits],
        late_keys: Vec::new(),
        late_inputs: Vec::new(),
    };
    let logits = synthetic_logits(3, VOCAB as usize);
    let outs = run_program_mock_outputs(&sub, &logits).expect("eval");

    assert_eq!(outs.len(), 2, "two declared outputs");
    assert!(matches!(outs[0], (OutputKind::Token, Value::I32(_))));
    assert!(matches!(outs[1], (OutputKind::Scalar, Value::F32(_))));
}
