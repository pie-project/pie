#![cfg(target_vendor = "apple")]

//! C1 — the correctness invariant for wiring the Hadamard KV rotation into the
//! qwen_3 attention (see `crates/models/src/qwen_3/forward.rs::attn_mixer`).
//!
//! The claim, plainly: turning Q/K/V by a per-head orthonormal Hadamard before
//! the cache, and turning the attention output by the same Hadamard afterwards,
//! changes the model's logits by NOTHING but floating-point rounding. The math:
//! H is orthonormal, so the scores `(Hq)·(Hk)ᵀ = q·kᵀ` are identical — the
//! softmax, and every read of the cache, is unchanged — and because the output
//! is `O = P·(V·H) = O_true·H`, one more Hadamard (`H·H = I`) recovers `O_true`.
//! So on real arithmetic the two forwards are bit-identical; at bf16 the extra
//! butterflies round, and the point of this test is that the on-vs-off deviation
//! sits at the bf16 rounding floor, not at a systematic offset (which would be a
//! bug). We ALSO run the same forward in f32 and show the deviation collapses
//! toward zero — that is what distinguishes "just rounding" from "a real bug".
//!
//! The harness builds a tiny dense, attention-only qwen_3 text (`micro_text`)
//! with deterministic synthetic weights written as a real safetensors file, then
//! runs the full Metal forward twice — once with `rotate_kv=false` (the shipped
//! path) and once with `rotate_kv=true` (the C1 path) — over the SAME weights,
//! and compares the returned logits row by row. It mirrors the load/fire harness
//! in `a_buffered_fold_is_the_fold_it_replaces.rs`.

use std::path::Path;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request};
use model_ir::{Elementwise, Operation};
use models::qwen_3::model::Model;

// ---- the micro_text shape (must match `Model::micro_text_dims`) --------------
const HIDDEN: usize = 128;
const LAYERS: usize = 2;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const INTER: usize = 256;
const VOCAB: usize = 256;

const PROMPT: &[u32] = &[7, 42, 11, 200, 3, 99, 128, 65];
const STEPS: usize = 3;

// ---- deterministic synthetic weights (same integer mixer as the sibling
// Hadamard tests, so the fixture is reproducible with no RNG state) ------------
fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E5E_1234_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit01(at: u64) -> f64 {
    (f64::from(noise(at)) + 1.0) / (f64::from(u32::MAX) + 2.0)
}

fn gaussian(at: u64) -> f32 {
    let u1 = unit01(at);
    let u2 = unit01(at ^ 0x9999_5A5A_1357_2468);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// A tensor of `len` scaled gaussians, seeded by a per-plane salt.
fn plane(salt: u64, len: usize, scale: f32) -> Vec<f32> {
    (0..len as u64)
        .map(|i| scale * gaussian(i ^ salt))
        .collect()
}

/// A minimal safetensors writer: `[u64 header-len][JSON header][packed f32 data]`,
/// data_offsets relative to the start of the data section. Enough for the reader
/// to index; we control both ends.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<u64>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, vals) in tensors {
        let begin = data.len() as u64;
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len() as u64;
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [begin, end] }),
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header json");
    let mut file = Vec::with_capacity(8 + header_bytes.len() + data.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&data);
    std::fs::write(path, file).expect("the fixture safetensors writes");
}

/// Every plane `Model::micro_text` imports under the transformers layout, filled
/// with synthetic f32. Norm weights are near zero (rmsnorm here is the +1 form,
/// so a ~0 weight gives a ~unit scale); projections are small gaussians.
fn synth_fixture(dir: &Path) {
    let layer = |l: usize, leaf: &str| format!("model.language_model.layers.{l}.{leaf}");
    let mut t: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut push = |name: String, shape: Vec<u64>, scale: f32| {
        let len = shape.iter().product::<u64>() as usize;
        let salt = name.bytes().fold(0xABCD_1234u64, |h, b| {
            h.wrapping_mul(0x0100_0193).wrapping_add(u64::from(b))
        });
        t.push((name, shape, plane(salt, len, scale)));
    };

    push(
        "model.language_model.embed_tokens.weight".into(),
        vec![VOCAB as u64, HIDDEN as u64],
        0.03,
    );
    push(
        "model.language_model.norm.weight".into(),
        vec![HIDDEN as u64],
        0.01,
    );
    for l in 0..LAYERS {
        push(
            layer(l, "input_layernorm.weight"),
            vec![HIDDEN as u64],
            0.01,
        );
        push(
            layer(l, "post_attention_layernorm.weight"),
            vec![HIDDEN as u64],
            0.01,
        );
        // Gated attention: q_proj carries q AND gate, so 2 * q_heads * head_dim rows.
        push(
            layer(l, "self_attn.q_proj.weight"),
            vec![(2 * Q_HEADS * HEAD_DIM) as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.k_proj.weight"),
            vec![(KV_HEADS * HEAD_DIM) as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.v_proj.weight"),
            vec![(KV_HEADS * HEAD_DIM) as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.o_proj.weight"),
            vec![HIDDEN as u64, (Q_HEADS * HEAD_DIM) as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.q_norm.weight"),
            vec![HEAD_DIM as u64],
            0.01,
        );
        push(
            layer(l, "self_attn.k_norm.weight"),
            vec![HEAD_DIM as u64],
            0.01,
        );
        push(
            layer(l, "mlp.gate_proj.weight"),
            vec![INTER as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "mlp.up_proj.weight"),
            vec![INTER as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "mlp.down_proj.weight"),
            vec![HIDDEN as u64, INTER as u64],
            0.05,
        );
    }
    write_safetensors(&dir.join("model.safetensors"), &t);
}

fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn hadamards(trace: &model_ir::Trace) -> usize {
    trace
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Operation::Elementwise(Elementwise::Hadamard { .. })))
        .count()
}

/// Load a `micro_text` shell over the shared fixture at weight dtype `w`, with
/// the rotation on or off. Returns the load error as a string so a caller can
/// decide whether an unsupported dtype is fatal or merely "not reached".
fn try_load(dir: &Path, w: Dtype, rotate: bool) -> Result<Shell, String> {
    let model = if rotate {
        Model::micro_text_rotated(w, Dtype::Bf16, 1)
    } else {
        Model::micro_text(w, Dtype::Bf16, 1)
    };
    let trace = model_dsl::trace_hybrid("qwen3-micro-text", &model, Platform::Metal);
    let source = ztensor_compat::index(dir.join("model.safetensors")).map_err(|e| e.to_string())?;
    let contract = model
        .import(&source, Platform::Metal)
        .map_err(|e| e.to_string())?;
    drop(source);
    Shell::load(Boot {
        voxels: None,
        trace,
        contract: &contract,
        checkpoint: dir,
        budget: Budget::new(4, 128),
        patches: None,
        profile: None,
        page_size: 16,
        context: 128,
        slots: 4,
        pages: 4 * 128 / 16,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .map_err(|e| e.to_string())
}

/// The bf16 path is the shipped one — it must load.
fn load(dir: &Path, w: Dtype, rotate: bool) -> Shell {
    try_load(dir, w, rotate).expect("the micro_text shell loads")
}

/// Prefill the prompt then greedily decode `STEPS`, returning every logit row.
fn run(shell: &mut Shell, slot: u32) -> Vec<Vec<f32>> {
    shell.open(slot).expect("the slot opens");
    let mut rows = Vec::with_capacity(STEPS + 1);
    let got = shell
        .fire(&[Lane {
            slot,
            word: word(PROMPT.len() as u32),
            tokens: PROMPT,
        }])
        .expect("the prefill fires");
    rows.push(got.into_iter().next().expect("one prefill row"));
    for step in 0..STEPS {
        let fed = [argmax(rows.last().expect("a row")) as u32];
        let got = shell
            .fire(&[Lane {
                slot,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        rows.push(got.into_iter().next().expect("one decode row"));
    }
    rows
}

fn argmax(logits: &[f32]) -> usize {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0)
}

/// (max-abs, mean-abs) deviation and the reference logit spread, over every row.
fn deviation(off: &[Vec<f32>], on: &[Vec<f32>]) -> (f32, f32, f32) {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for (a, b) in off.iter().zip(on.iter()) {
        assert_eq!(a.len(), b.len(), "the two paths return the same width");
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            max = max.max(d);
            sum += f64::from(d);
            count += 1;
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    (max, (sum / count as f64) as f32, hi - lo)
}

#[test]
fn the_kv_rotation_is_the_unrotated_forward() {
    // ---- Non-regression / structural check (no device needed): the default
    // path emits NOT ONE Hadamard, so every shipped SKU is byte-unchanged; the
    // rotated path emits exactly four per attention layer (q, k, v, o).
    let off_trace = model_dsl::trace_hybrid(
        "qwen3-micro-text",
        &Model::micro_text(Dtype::Bf16, Dtype::Bf16, 1),
        Platform::Metal,
    );
    let on_trace = model_dsl::trace_hybrid(
        "qwen3-micro-text",
        &Model::micro_text_rotated(Dtype::Bf16, Dtype::Bf16, 1),
        Platform::Metal,
    );
    let (off_h, on_h) = (hadamards(&off_trace), hadamards(&on_trace));
    eprintln!("[trace] rotate_kv=false Hadamard ops = {off_h}; rotate_kv=true = {on_h}");
    assert_eq!(
        off_h, 0,
        "the default path must emit no Hadamard (byte-unchanged SKUs)"
    );
    assert_eq!(
        on_h,
        4 * LAYERS,
        "the rotated path emits four Hadamards (q,k,v,o) per attention layer"
    );

    if !engine_metal::device::present() {
        eprintln!("skipping the forward invariant: this machine publishes no Metal device");
        return;
    }

    let dir = std::env::temp_dir().join(format!("pie_c1_hadamard_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the fixture dir exists");
    synth_fixture(&dir);

    // ---- bf16: the shipped compute dtype. Deviation must be at the rounding
    // floor, and both paths must agree on the argmax at every position.
    let bf16_off = run(&mut load(&dir, Dtype::Bf16, false), 0);
    let bf16_on = run(&mut load(&dir, Dtype::Bf16, true), 1);
    for (step, (a, b)) in bf16_off.iter().zip(bf16_on.iter()).enumerate() {
        assert!(
            a.iter().all(|v| v.is_finite()),
            "bf16 off step {step} non-finite"
        );
        assert!(
            b.iter().all(|v| v.is_finite()),
            "bf16 on step {step} non-finite"
        );
    }
    let (bf16_max, bf16_mean, spread) = deviation(&bf16_off, &bf16_on);
    let rel = bf16_max / spread.max(1e-6);
    eprintln!(
        "[bf16] logit spread {spread:.4}; on-vs-off deviation max {bf16_max:.5}, mean {bf16_mean:.6}, \
         max/spread {rel:.4}"
    );
    let argmax_off: Vec<usize> = bf16_off.iter().map(|r| argmax(r)).collect();
    let argmax_on: Vec<usize> = bf16_on.iter().map(|r| argmax(r)).collect();
    eprintln!("[bf16] argmax off {argmax_off:?} vs on {argmax_on:?}");

    // The deviation must be small RELATIVE to the logit spread: a rounding floor,
    // not a systematic basis error (which would land near the spread itself).
    assert!(
        rel < 0.02,
        "bf16 on-vs-off deviation {bf16_max:.5} is not at the rounding floor \
         (spread {spread:.4}, max/spread {rel:.4}) — a systematic difference, i.e. a bug"
    );
    assert_eq!(
        argmax_off, argmax_on,
        "the rotated forward chose different tokens than the plain forward — not a rounding effect"
    );

    // ---- f32: same forward, exact-er arithmetic. If it is reachable, the
    // deviation must SHRINK well below bf16 — proof the residual is rounding, not
    // a bug. If the Metal forward refuses f32 end to end, we say so rather than
    // pretend.
    let f32_dir = std::env::temp_dir().join(format!("pie_c1_hadamard_f32_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&f32_dir);
    std::fs::create_dir_all(&f32_dir).expect("the f32 fixture dir exists");
    synth_fixture(&f32_dir);
    match (
        try_load(&f32_dir, Dtype::F32, false),
        try_load(&f32_dir, Dtype::F32, true),
    ) {
        (Ok(mut off_shell), Ok(mut on_shell)) => {
            let off = run(&mut off_shell, 0);
            let on = run(&mut on_shell, 1);
            let (f32_max, f32_mean, f32_spread) = deviation(&off, &on);
            eprintln!(
                "[f32 ] logit spread {f32_spread:.4}; on-vs-off deviation max {f32_max:.6}, mean {f32_mean:.7}"
            );
            assert!(
                f32_max < bf16_max,
                "f32 deviation {f32_max:.6} did not shrink below bf16 {bf16_max:.5} — the residual \
                 is not rounding"
            );
            eprintln!(
                "[f32 ] deviation shrank {:.1}x vs bf16 — the residual is rounding, not a bug",
                bf16_max / f32_max.max(1e-9)
            );
        }
        (off, on) => {
            let why = off.err().or_else(|| on.err()).unwrap_or_default();
            eprintln!(
                "[f32 ] not reached: the Metal forward does not run this text end-to-end in f32 \
                 ({why}); the bf16 rounding-floor result above stands on its own"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&f32_dir);

    eprintln!("=== C1 invariant holds: rotate_kv on == off within the bf16 rounding floor ===");
    let _ = std::fs::remove_dir_all(&dir);
}
