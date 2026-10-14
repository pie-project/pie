#![cfg(feature = "cuda")]

//! A paged prefill answers what the host's attention says and what FA2
//! lands: on a tcgen05 device `attention.prefill` runs the tcgen05 kernel,
//! while the log-sum-exp form stays on FA2, so the two read the same pool
//! side by side. Requests carry cached prefixes, page tables in falling
//! runs (as the pool's stack hands them out), partial last pages, grouped
//! query heads and a sliding window.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn;
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

struct Case {
    head_dim: usize,
    kv_heads: usize,
    group: usize,
    /// (new tokens, cached prefix) a request.
    requests: &'static [(usize, usize)],
    /// A sliding window: the keys a row reads, its own included.
    window: Option<u32>,
    /// The softmax scale; `None` is 1/sqrt(head_dim).
    scale: Option<f32>,
}

#[allow(clippy::too_many_lines)]
fn check(case: &Case) {
    let (d, kvh, g) = (case.head_dim, case.kv_heads, case.group);
    let qh = kvh * g;
    let reqs = case.requests;
    let mut qo_indptr = vec![0i32];
    let mut kv_indptr = vec![0i32];
    let mut last = Vec::new();
    let mut kv_len = Vec::new();
    for &(new, prefix) in reqs {
        qo_indptr.push(qo_indptr.last().unwrap() + new as i32);
        let len = new + prefix;
        let pages = len.div_ceil(PAGE);
        kv_indptr.push(kv_indptr.last().unwrap() + pages as i32);
        last.push((len - (pages - 1) * PAGE) as i32);
        kv_len.push(len as i32);
    }
    let pages = *kv_indptr.last().unwrap() as usize;
    // Pages in reverse, past a spare one, as the pool's stack hands them out:
    // falling runs, broken every ten pages by a swapped pair.
    let mut indices: Vec<i32> = (0..pages).map(|p| (pages - p) as i32).collect();
    for p in (3..pages.saturating_sub(1)).step_by(10) {
        indices.swap(p, p + 1);
    }
    let cells = (pages + 1) * PAGE;
    let rows = *qo_indptr.last().unwrap() as usize;

    let mut lcg = Lcg::seeded(0x7c5f);
    let (q_raw, q) = lcg.row(rows * qh * d);
    let (k_raw, k) = lcg.row(cells * kvh * d);
    let (v_raw, v) = lcg.row(cells * kvh * d);

    let mut gpu = Gpu::open();
    // sm_100's profile decision; off, `attention.prefill` is FA2's.
    let ctx = gpu.ctx().with_tuning(Tuning {
        prefill_tc5: true,
        ..Tuning::DEFAULT
    });
    let Some(device) = Device::probe(&ctx) else {
        eprintln!("skipped: the device would not say its shape");
        return;
    };
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let qo_at = gpu.up(&qo_indptr);
    let kvp_at = gpu.up(&kv_indptr);
    let idx_at = gpu.up(&indices);
    let last_at = gpu.up(&last);
    let ws_bytes = 64 << 20;
    let workspace = Workspace {
        int_ptr: gpu.zeros(ws_bytes),
        int_bytes: ws_bytes,
        float_ptr: gpu.zeros(ws_bytes),
        float_bytes: ws_bytes,
    };
    let o_tc = gpu.zeros(rows * qh * d * 2);
    let o_fa = gpu.zeros(rows * qh * d * 2);
    let lse = gpu.zeros(rows * qh * 4);

    let n = reqs.len() as u32;
    let shape = Shape {
        num_requests: n,
        lane_offset: 0,
        num_q_heads: qh as u32,
        num_kv_heads: kvh as u32,
        head_dim: d as u32,
        page_size: PAGE as u32,
        hnd_layout: false,
    };
    let live = Live {
        requests: n,
        lane_offset: 0,
        row_offset: 0,
        rows: rows as u32,
    };
    let built = plan::plan_prefill(
        &qo_indptr,
        &kv_indptr,
        &kv_len,
        rows as u32,
        shape,
        live,
        case.window,
        true,
        false,
        None,
        &device,
        Toggles::of(Tuning::DEFAULT),
        workspace,
    )
    .expect("the plan builds");
    built.stage(&ctx).expect("the plan uploads");
    let pool = KvPool {
        keys: Tensor::new(k_at, cells as u32, (kvh * d) as u32, Dtype::Bf16),
        values: Tensor::new(v_at, cells as u32, (kvh * d) as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(idx_at, pages as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(kvp_at, n + 1, 1, Dtype::I32),
        last_page_lens: Tensor::new(last_at, n, 1, Dtype::I32),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: (kvh * d) as i64,
        head_stride: d as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: pages as i32,
        pages_in_batch: pages as i32,
    };
    let q_t = RaggedTensor {
        data: Tensor::new(q_at, rows as u32, (qh * d) as u32, Dtype::Bf16),
        indptr: Tensor::new(qo_at, n + 1, 1, Dtype::I32),
    };
    let sm_scale = case.scale.unwrap_or(1.0 / (d as f32).sqrt());
    let mut o = Tensor::new(o_tc, rows as u32, (qh * d) as u32, Dtype::Bf16);
    attn::prefill(
        &ctx,
        q_t,
        &built,
        &pool,
        case.window,
        d as u32,
        kvh as u32,
        sm_scale,
        &mut o,
    )
    .expect("the prefill fires");
    let mut o2 = Tensor::new(o_fa, rows as u32, (qh * d) as u32, Dtype::Bf16);
    let mut lse_t = Tensor::new(lse, rows as u32, qh as u32, Dtype::F32);
    attn::prefill_lse(
        &ctx,
        q_t,
        &built,
        &pool,
        case.window,
        d as u32,
        kvh as u32,
        sm_scale,
        &mut o2,
        &mut lse_t,
    )
    .expect("the FA2 prefill fires");
    // The decision off, `attention.prefill` is FA2's, bit for bit.
    let o_off = gpu.zeros(rows * qh * d * 2);
    let mut o3 = Tensor::new(o_off, rows as u32, (qh * d) as u32, Dtype::Bf16);
    attn::prefill(
        &ctx.with_tuning(Tuning::DEFAULT),
        q_t,
        &built,
        &pool,
        case.window,
        d as u32,
        kvh as u32,
        sm_scale,
        &mut o3,
    )
    .expect("the FA2 prefill fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_tc, rows * qh * d);
    let fa2: Vec<u16> = gpu.down(o_fa, rows * qh * d);
    let off: Vec<u16> = gpu.down(o_off, rows * qh * d);
    assert_eq!(off, fa2, "with the decision off the prefill is FA2's");

    let mut worst_ref = 0f32;
    let mut worst_fa2 = 0f32;
    for (r, &(new, prefix)) in reqs.iter().enumerate() {
        let cell =
            |kp: usize| indices[kv_indptr[r] as usize + kp / PAGE] as usize * PAGE + kp % PAGE;
        for t in 0..new {
            let row = qo_indptr[r] as usize + t;
            let reach = prefix + t + 1;
            let first = case.window.map_or(0, |w| reach.saturating_sub(w as usize));
            for h in 0..qh {
                let kh = h / g;
                let qr = &q[(row * qh + h) * d..(row * qh + h + 1) * d];
                let scores: Vec<f32> = (first..reach)
                    .map(|kp| {
                        let kr = &k[(cell(kp) * kvh + kh) * d..(cell(kp) * kvh + kh + 1) * d];
                        qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * sm_scale
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let l: f32 = w.iter().sum();
                for c in 0..d {
                    let want: f32 = (first..reach)
                        .map(|kp| w[kp - first] * v[(cell(kp) * kvh + kh) * d + c])
                        .sum::<f32>()
                        / l;
                    let at = (row * qh + h) * d + c;
                    let tc = from_bf16(got[at]);
                    worst_ref = worst_ref.max((tc - want).abs());
                    worst_fa2 = worst_fa2.max((tc - from_bf16(fa2[at])).abs());
                }
            }
        }
    }
    eprintln!("head {d}, group {g}: worst against the host {worst_ref}, against FA2 {worst_fa2}");
    assert!(
        worst_ref < 2e-2,
        "the prefill strays {worst_ref} from the host's attention"
    );
    assert!(
        worst_fa2 < 2e-2,
        "the prefill strays {worst_fa2} from FA2's"
    );
}

#[test]
fn the_tcgen05_prefill_answers_fa2_and_the_reference() {
    check(&Case {
        head_dim: 128,
        kv_heads: 2,
        group: 4,
        requests: &[(70, 0), (33, 17), (200, 40), (129, 0)],
        window: None,
        scale: None,
    });
    check(&Case {
        head_dim: 256,
        kv_heads: 2,
        group: 6,
        requests: &[(70, 0), (65, 33), (200, 17)],
        window: None,
        scale: None,
    });
    check(&Case {
        head_dim: 256,
        kv_heads: 4,
        group: 2,
        requests: &[(300, 0), (90, 120)],
        window: Some(100),
        scale: None,
    });
    // Gemma-4-31B's sliding layers: a 1024-key window over a 256-wide head
    // of two query heads a KV head, at a unit scale (its queries and keys
    // are normed), so scores run far past a 1/sqrt(d) head's.
    check(&Case {
        head_dim: 256,
        kv_heads: 2,
        group: 2,
        requests: &[(329, 0), (1300, 0)],
        window: Some(1024),
        scale: Some(1.0),
    });
    // Long enough that most key tiles are whole to every row of a block,
    // their pages one falling run: one box an atom of the head.
    check(&Case {
        head_dim: 128,
        kv_heads: 1,
        group: 4,
        requests: &[(600, 0), (200, 300)],
        window: None,
        scale: None,
    });
    check(&Case {
        head_dim: 256,
        kv_heads: 1,
        group: 6,
        requests: &[(500, 0)],
        window: None,
        scale: None,
    });
    check(&Case {
        head_dim: 64,
        kv_heads: 3,
        group: 1,
        requests: &[(150, 0), (90, 12)],
        window: None,
        scale: None,
    });
}

/// The tcgen05 prefill's rate beside FA2's (the log-sum-exp form) over
/// causal 4096-token requests at Qwen3.8-27B's full-attention head (256
/// wide, six query heads a key head) and a 128-wide one with eight. FLOPs
/// count the causal half. vLLM's prefill on B200 (FlashInfer's trtllm-gen
/// FMHA) runs Qwen3.8-27B's 32 x 4096 at about 790 TFLOPS by the same
/// count. Run it alone on an idle GPU:
///
///     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
///         the_tcgen05_prefill_reports -- --ignored --nocapture
#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
#[allow(clippy::too_many_lines)]
fn the_tcgen05_prefill_reports_its_rate_beside_fa2() {
    use crate::the_tcgen05_gemm_lands_and_paces_what_cublaslt_does::{FLUSH, Timer};
    use kernels_cuda::cudarc::runtime::sys as rt;

    let mut gpu = Gpu::open();
    // sm_100's profile decision; off, `attention.prefill` is FA2's.
    let ctx = gpu.ctx().with_tuning(Tuning {
        prefill_tc5: true,
        ..Tuning::DEFAULT
    });
    let Some(device) = Device::probe(&ctx) else {
        eprintln!("skipped: the device would not say its shape");
        return;
    };
    let stream: rt::cudaStream_t = ctx.stream().cast();
    let timer = Timer {
        flush: gpu.zeros(FLUSH),
        stream,
    };
    println!("| head | group | kv heads | requests x tokens | tc5 TFLOPS | FA2 TFLOPS |");
    println!("|---:|---:|---:|---|---:|---:|");
    for (d, g, kvh, n) in [
        (256usize, 6usize, 4usize, 1usize),
        (256, 6, 4, 8),
        (256, 6, 4, 32),
        (128, 8, 8, 1),
        (128, 8, 8, 8),
    ] {
        let len = 4096usize;
        let qh = kvh * g;
        let rows = n * len;
        let pages_per = len / PAGE;
        let pages = n * pages_per;
        let cells = pages * PAGE;
        let qo_indptr: Vec<i32> = (0..=n).map(|r| (r * len) as i32).collect();
        let kv_indptr: Vec<i32> = (0..=n).map(|r| (r * pages_per) as i32).collect();
        let kv_len = vec![len as i32; n];
        let last = vec![PAGE as i32; n];
        let indices: Vec<i32> = (0..pages as i32).collect();
        let mut lcg = Lcg::seeded(0x5eed);
        let (q_raw, _) = lcg.row(rows * qh * d);
        let (k_raw, _) = lcg.row(cells * kvh * d);
        let (v_raw, _) = lcg.row(cells * kvh * d);
        let q_at = gpu.up(&q_raw);
        let k_at = gpu.up(&k_raw);
        let v_at = gpu.up(&v_raw);
        let qo_at = gpu.up(&qo_indptr);
        let kvp_at = gpu.up(&kv_indptr);
        let idx_at = gpu.up(&indices);
        let last_at = gpu.up(&last);
        let ws_bytes = 256 << 20;
        let workspace = Workspace {
            int_ptr: gpu.zeros(ws_bytes),
            int_bytes: ws_bytes,
            float_ptr: gpu.zeros(ws_bytes),
            float_bytes: ws_bytes,
        };
        let o_at = gpu.zeros(rows * qh * d * 2);
        let lse_at = gpu.zeros(rows * qh * 4);
        let shape = Shape {
            num_requests: n as u32,
            lane_offset: 0,
            num_q_heads: qh as u32,
            num_kv_heads: kvh as u32,
            head_dim: d as u32,
            page_size: PAGE as u32,
            hnd_layout: false,
        };
        let live = Live {
            requests: n as u32,
            lane_offset: 0,
            row_offset: 0,
            rows: rows as u32,
        };
        let built = plan::plan_prefill(
            &qo_indptr,
            &kv_indptr,
            &kv_len,
            rows as u32,
            shape,
            live,
            None,
            true,
            false,
            None,
            &device,
            Toggles::of(Tuning::DEFAULT),
            workspace,
        )
        .expect("the plan builds");
        built.stage(&ctx).expect("the plan uploads");
        let pool = KvPool {
            keys: Tensor::new(k_at, cells as u32, (kvh * d) as u32, Dtype::Bf16),
            values: Tensor::new(v_at, cells as u32, (kvh * d) as u32, Dtype::Bf16),
            bf16_keys: Tensor::ABSENT,
            bf16_values: Tensor::ABSENT,
            key_scales: Tensor::ABSENT,
            value_scales: Tensor::ABSENT,
            page_indices: Tensor::new(idx_at, pages as u32, 1, Dtype::I32),
            page_indptr: Tensor::new(kvp_at, n as u32 + 1, 1, Dtype::I32),
            last_page_lens: Tensor::new(last_at, n as u32, 1, Dtype::I32),
            row_valid: Tensor::ABSENT,
            env_min: Tensor::ABSENT,
            env_max: Tensor::ABSENT,
            has_envelopes: false,
            page_size: PAGE as i32,
            seq_stride: (kvh * d) as i64,
            head_stride: d as i64,
            layout: 0,
            scheme_byte: 0,
            block_size: 0,
            max_pages_per_request: pages_per as i32,
            pages_in_batch: pages as i32,
        };
        let q_t = RaggedTensor {
            data: Tensor::new(q_at, rows as u32, (qh * d) as u32, Dtype::Bf16),
            indptr: Tensor::new(qo_at, n as u32 + 1, 1, Dtype::I32),
        };
        let sm_scale = 1.0 / (d as f32).sqrt();
        let tc5 = timer.median(|_| {
            let mut o = Tensor::new(o_at, rows as u32, (qh * d) as u32, Dtype::Bf16);
            attn::prefill(
                &ctx, q_t, &built, &pool, None, d as u32, kvh as u32, sm_scale, &mut o,
            )
            .expect("the prefill fires");
        });
        let fa2 = timer.median(|_| {
            let mut o = Tensor::new(o_at, rows as u32, (qh * d) as u32, Dtype::Bf16);
            let mut lse = Tensor::new(lse_at, rows as u32, qh as u32, Dtype::F32);
            attn::prefill_lse(
                &ctx, q_t, &built, &pool, None, d as u32, kvh as u32, sm_scale, &mut o, &mut lse,
            )
            .expect("the FA2 prefill fires");
        });
        // Causal: row t reads t + 1 keys.
        let flops = 4.0 * (n * qh * d) as f64 * (len * (len + 1) / 2) as f64;
        println!(
            "| {d} | {g} | {kvh} | {n} x {len} | {:.0} | {:.0} |",
            flops / tc5 * 1e-6,
            flops / fa2 * 1e-6
        );
    }
}
