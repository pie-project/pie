#![cfg(target_vendor = "apple")]

//! C2d — the rotated + packed-4-bit KV codec, served end to end.
//!
//! C0 showed the rotation helps a low-bit quantizer; C1 wired the rotation into
//! qwen attention and showed it is the un-rotated forward at the bf16 rounding
//! floor; C2a/b/c built and unit-tested the `KvU4` codec's pack/write/read
//! kernels in isolation. This test proves the pieces COMPOSE: a real small-model
//! Metal forward run through the serve path, over a SKU that turns the rotation
//! ON and stores the KV cache as the packed `KvU4` codec, against the shipped
//! bf16-KV baseline over the SAME weights.
//!
//! Three forwards share every weight and differ only in the KV path:
//!
//!   (baseline) rotate_kv = false, KV = Bf16  — the shipped path, ground truth.
//!   (rot-only) rotate_kv = true,  KV = Bf16  — rotation, no quant. This MUST
//!               reproduce C1's rounding floor: the Hadamard is orthonormal, so
//!               the only difference from baseline is bf16 rounding of the extra
//!               butterflies. If this is NOT tiny, the wiring is broken and the
//!               packed-path deviation below is meaningless.
//!   (rot+u4)   rotate_kv = true,  KV = KvU4  — the product. Its deviation from
//!               baseline is rotation (rounding) PLUS 4-bit quantization loss;
//!               it is larger than rot-only, dominated by quant, and — crucially
//!               — the argmax stays identical at every position (a token flip on
//!               this tiny model would be a wiring bug, not quant noise).
//!
//! We ALSO measure the memory win straight from the allocation layer
//! (`pool_demand` over each SKU's real trace): a packed head is `head_dim/2 + 2`
//! bytes vs `head_dim * 2` for bf16, so the cache is ~3.9x smaller at head_dim
//! 256 and ~3.88x at 128. Both head_dims are exercised — they are the two blocks
//! the `KvU4` write/read kernels ship (`SDPA_KV_U4_WIDTHS = [128, 256]`).

use std::path::Path;

use engine_metal::store::kv::Paging;
use engine_metal::store::pool_demand;
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request};
use models::qwen_3::model::Model;

// ---- the micro_text shape (head_dim is the free axis this test sweeps) -------
const HIDDEN: usize = 128;
const LAYERS: usize = 2;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const INTER: usize = 256;
const VOCAB: usize = 256;

const PROMPT: &[u32] = &[7, 42, 11, 200, 3, 99, 128, 65];
const STEPS: usize = 3;

// ---- deterministic synthetic weights (same integer mixer as the C1 test) -----
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

fn plane(salt: u64, len: usize, scale: f32) -> Vec<f32> {
    (0..len as u64)
        .map(|i| scale * gaussian(i ^ salt))
        .collect()
}

/// A minimal safetensors writer: `[u64 header-len][JSON header][packed f32 data]`.
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

/// Every plane `micro_text` imports under the transformers layout, at `head_dim`,
/// filled with synthetic f32 (same salting/scales as the C1 fixture, generalized
/// to a configurable head_dim).
fn synth_fixture(dir: &Path, head_dim: usize) {
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
            vec![(2 * Q_HEADS * head_dim) as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.k_proj.weight"),
            vec![(KV_HEADS * head_dim) as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.v_proj.weight"),
            vec![(KV_HEADS * head_dim) as u64, HIDDEN as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.o_proj.weight"),
            vec![HIDDEN as u64, (Q_HEADS * head_dim) as u64],
            0.05,
        );
        push(
            layer(l, "self_attn.q_norm.weight"),
            vec![head_dim as u64],
            0.01,
        );
        push(
            layer(l, "self_attn.k_norm.weight"),
            vec![head_dim as u64],
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

/// Which SKU to load: the plain baseline, or the rotated path at a given KV dtype.
#[derive(Clone, Copy)]
enum Sku {
    /// rotate_kv = false — the shipped ground truth.
    PlainBf16,
    /// rotate_kv = true, KV Bf16 — rotation only, no quant.
    RotatedBf16,
    /// rotate_kv = true, KV KvU4 — the product.
    RotatedU4,
}

fn model_of(sku: Sku, head_dim: u32) -> Model {
    match sku {
        Sku::PlainBf16 => Model::micro_text_hd(Dtype::Bf16, Dtype::Bf16, 1, head_dim),
        Sku::RotatedBf16 => Model::micro_text_rotated_hd(Dtype::Bf16, Dtype::Bf16, 1, head_dim),
        Sku::RotatedU4 => Model::micro_text_rotated_hd(Dtype::Bf16, Dtype::KvU4, 1, head_dim),
    }
}

fn load(dir: &Path, sku: Sku, head_dim: u32) -> Shell {
    let model = model_of(sku, head_dim);
    let trace = model_dsl::trace_hybrid("qwen3-micro-text", &model, Platform::Metal);
    let source = ztensor_compat::index(dir.join("model.safetensors")).expect("the fixture indexes");
    let contract = model
        .import(&source, Platform::Metal)
        .expect("the fixture imports");
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
    .expect("the micro_text shell loads")
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
fn deviation(base: &[Vec<f32>], other: &[Vec<f32>]) -> (f32, f32, f32) {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for (a, b) in base.iter().zip(other.iter()) {
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

fn argmaxes(rows: &[Vec<f32>]) -> Vec<usize> {
    rows.iter().map(|r| argmax(r)).collect()
}

/// KV-cache bytes the store's demand accounting reserves for `sku` at `head_dim`,
/// summed over the SKU's real trace (both layers, key and value planes). This is
/// the allocation layer — the same `row_stride` fork `reserve` uses — so the
/// number is what the pool would allocate, not a hand estimate.
fn kv_cache_bytes(sku: Sku, head_dim: u32, paging: Paging) -> u64 {
    let model = model_of(sku, head_dim);
    let trace = model_dsl::trace_hybrid("qwen3-micro-text", &model, Platform::Metal);
    pool_demand(&trace, paging).expect("the KV cache sizes")
}

#[test]
fn the_rotated_packed_kv_serves() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }

    // head_dim 256 (the shipping flagship geometry) and 128 — the two blocks the
    // packed KvU4 write/read kernels ship (SDPA_KV_U4_WIDTHS = [128, 256]). The
    // micro_text's own head_dim is 64, which the packed kernels do NOT serve, so
    // this test uses the parameterized micro_text_*_hd SKUs at 128 and 256.
    for &head_dim in &[256u32, 128] {
        serve_case(head_dim);
    }
    eprintln!(
        "=== C2d: the rotated packed-4-bit KV SKU serves — argmax-identical, ~3.9x smaller cache ==="
    );
}

fn serve_case(head_dim: u32) {
    eprintln!("---- head_dim {head_dim} ----");

    // ---- memory: straight from the allocation layer, needs no device --------
    // Any paging gives the same ratio (both dtypes scale by the same cell count);
    // we use the shell's paging so the byte figures are the ones it reserves.
    let paging = Paging::of(16, 128, 4, 4 * 128 / 16).expect("paging");
    let bf16_bytes = kv_cache_bytes(Sku::PlainBf16, head_dim, paging);
    let u4_bytes = kv_cache_bytes(Sku::RotatedU4, head_dim, paging);
    let ratio = u4_bytes as f64 / bf16_bytes as f64;
    // packed head = head_dim/2 + 2 bytes; bf16 head = head_dim * 2 bytes.
    let want = (f64::from(head_dim) / 2.0 + 2.0) / (f64::from(head_dim) * 2.0);
    let reduction = 1.0 / ratio;
    eprintln!(
        "[mem  ] bf16 KV = {bf16_bytes} B, KvU4 KV = {u4_bytes} B, ratio {ratio:.5} \
         (expected {want:.5}), reduction {reduction:.3}x"
    );
    assert!(
        (ratio - want).abs() < 1e-9,
        "packed cache must be (head_dim/2+2)/(head_dim*2) of bf16 at head_dim {head_dim}: \
         got {ratio}, want {want}"
    );
    let expect_reduction = 1.0 / want;
    assert!(
        (reduction - expect_reduction).abs() < 1e-3,
        "reduction {reduction:.3}x is not the expected {expect_reduction:.3}x at head_dim {head_dim}"
    );

    // ---- the three forwards over the SAME weights ---------------------------
    let dir =
        std::env::temp_dir().join(format!("pie_c2d_serve_hd{head_dim}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the fixture dir exists");
    synth_fixture(&dir, head_dim as usize);

    let baseline = run(&mut load(&dir, Sku::PlainBf16, head_dim), 0);
    let rot_only = run(&mut load(&dir, Sku::RotatedBf16, head_dim), 1);
    let rot_u4 = run(&mut load(&dir, Sku::RotatedU4, head_dim), 2);
    let _ = std::fs::remove_dir_all(&dir);

    for (label, rows) in [
        ("baseline", &baseline),
        ("rot-only", &rot_only),
        ("rot+u4", &rot_u4),
    ] {
        for (step, r) in rows.iter().enumerate() {
            assert!(
                r.iter().all(|v| v.is_finite()),
                "{label} step {step} produced a non-finite logit"
            );
        }
    }

    let am_base = argmaxes(&baseline);
    let am_rot = argmaxes(&rot_only);
    let am_u4 = argmaxes(&rot_u4);

    // ---- rotation-only: MUST be the bf16 rounding floor (else the wiring is
    // broken and the packed number below is meaningless). This reproduces C1.
    let (rot_max, rot_mean, spread) = deviation(&baseline, &rot_only);
    let rot_rel = rot_max / spread.max(1e-6);
    eprintln!(
        "[rot  ] spread {spread:.4}; vs baseline max {rot_max:.5}, mean {rot_mean:.6}, \
         max/spread {rot_rel:.4}"
    );
    eprintln!("[rot  ] argmax base {am_base:?} vs rot {am_rot:?}");
    assert_eq!(
        am_base, am_rot,
        "rotation-only chose different tokens than baseline — a wiring bug, not rounding"
    );
    assert!(
        rot_rel < 0.02,
        "rotation-only deviation {rot_max:.5} (max/spread {rot_rel:.4}) is not at the bf16 \
         rounding floor — the rotation wiring is broken, so the packed result is meaningless"
    );

    // ---- rot + 4-bit: rotation rounding PLUS 4-bit quant loss. Larger than
    // rot-only, dominated by quant, and argmax-identical to baseline.
    let (u4_max, u4_mean, _) = deviation(&baseline, &rot_u4);
    let u4_rel = u4_max / spread.max(1e-6);
    eprintln!("[u4   ] vs baseline max {u4_max:.5}, mean {u4_mean:.6}, max/spread {u4_rel:.4}");
    eprintln!("[u4   ] argmax base {am_base:?} vs u4 {am_u4:?}");
    assert_eq!(
        am_base, am_u4,
        "the rotated 4-bit-KV forward chose different tokens than baseline at head_dim {head_dim} \
         — a token flip on this tiny model is a wiring bug, not quant noise"
    );
    assert!(
        u4_max >= rot_max,
        "the 4-bit deviation {u4_max:.5} is not larger than the rotation-only floor {rot_max:.5} \
         — the quant path is doing nothing"
    );
    // The ceiling: measured on THIS fixture, then set a hair above. Observed
    // max/spread is ~8.4% at head_dim 256 and ~11.8% at 128 (mean ~4.5% / ~6%) —
    // higher than a real model would show because the synthetic KV is
    // unstructured Gaussians the Hadamard cannot de-spike (C0: the rotation helps
    // BECAUSE of outliers), so this is close to raw 4-bit-on-noise error. It is
    // dominated by quant, NOT a systematic basis offset — the rotation-only floor
    // above is only ~0.5%, so essentially the whole gap is quantization, and
    // every argmax held at every position. A basis/wiring bug would instead sit
    // near the spread itself and would have flipped a token.
    assert!(
        u4_rel < 0.13,
        "the 4-bit deviation max/spread {u4_rel:.4} at head_dim {head_dim} exceeds the measured \
         quant ceiling — investigate before trusting it"
    );
}
