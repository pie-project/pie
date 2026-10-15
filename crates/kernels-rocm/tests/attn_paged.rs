//! Paged attention against a host reference: decode and causal prefill,
//! split and unsplit schedules, NHD and HND pages, with real plans staged
//! through the schedulers the engine runs.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, plan};
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

struct Case {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    page_size: u32,
    hnd: bool,
}

/// A paged cache holding `kv_lens[r]` tokens for request r, pages handed
/// out in a shuffled order so a wrong page walk shows. Returns the pool and
/// the host keys/values as [request][token][head][d].
struct Cache {
    pool: KvPool,
    keys: Vec<Vec<f32>>,
    vals: Vec<Vec<f32>>,
    page_indptr: Vec<i32>,
}

fn cache(gpu: &Gpu, c: &Case, kv_lens: &[u32], seed: u64) -> Cache {
    let (h, d, ps) = (
        c.kv_heads as usize,
        c.head_dim as usize,
        c.page_size as usize,
    );
    let pages_of: Vec<usize> = kv_lens
        .iter()
        .map(|&n| (n as usize).div_ceil(ps).max(1))
        .collect();
    let total_pages: usize = pages_of.iter().sum::<usize>() + 3;
    let order: Vec<i32> = (0..total_pages as i32).rev().collect();
    let mut page_indptr = vec![0i32];
    let mut page_indices = Vec::new();
    let mut next = 0;
    for &p in &pages_of {
        for _ in 0..p {
            page_indices.push(order[next]);
            next += 1;
        }
        page_indptr.push(page_indices.len() as i32);
    }
    let last: Vec<i32> = kv_lens
        .iter()
        .zip(&pages_of)
        .map(|(&n, &p)| (n as usize - (p - 1) * ps) as i32)
        .collect();

    let mut kp = vec![0f32; total_pages * ps * h * d];
    let mut vp = vec![0f32; total_pages * ps * h * d];
    let mut keys = Vec::new();
    let mut vals = Vec::new();
    for (r, &n) in kv_lens.iter().enumerate() {
        let k: Vec<f32> = values(n as usize * h * d, seed + r as u64, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let v: Vec<f32> = values(n as usize * h * d, seed + 100 + r as u64, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        for t in 0..n as usize {
            let page = page_indices[page_indptr[r] as usize + t / ps] as usize;
            let slot = t % ps;
            for hh in 0..h {
                for j in 0..d {
                    let at = if c.hnd {
                        ((page * h + hh) * ps + slot) * d + j
                    } else {
                        ((page * ps + slot) * h + hh) * d + j
                    };
                    kp[at] = k[(t * h + hh) * d + j];
                    vp[at] = v[(t * h + hh) * d + j];
                }
            }
        }
        keys.push(k);
        vals.push(v);
    }
    let rows = (total_pages * ps) as u32;
    let pool = KvPool {
        keys: gpu.tensor(&kp, rows, (h * d) as u32, Dtype::Bf16),
        values: gpu.tensor(&vp, rows, (h * d) as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: gpu.column(&page_indices),
        page_indptr: gpu.column(&page_indptr),
        last_page_lens: gpu.column(&last),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: ps as i32,
        seq_stride: if c.hnd { d as i64 } else { (h * d) as i64 },
        head_stride: if c.hnd { (ps * d) as i64 } else { d as i64 },
        layout: i32::from(c.hnd),
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: *pages_of.iter().max().unwrap() as i32,
        pages_in_batch: page_indices.len() as i32,
    };
    Cache {
        pool,
        keys,
        vals,
        page_indptr,
    }
}

/// Row `token` of a request with `qo_len` queries over `kv` keys: causal,
/// optionally windowed (keys at or after the query's position − window + 1).
#[allow(clippy::too_many_arguments)]
fn reference_row(
    c: &Case,
    q: &[f32],
    keys: &[f32],
    vals: &[f32],
    kv: usize,
    qo_len: usize,
    token: usize,
    window: Option<u32>,
) -> (Vec<f32>, Vec<f32>) {
    let (hq, hk, d) = (c.q_heads as usize, c.kv_heads as usize, c.head_dim as usize);
    let g = hq / hk;
    let scale = 1.0 / (d as f64).sqrt();
    let pos = kv - qo_len + token;
    let lo = window.map_or(0, |w| (pos + 1).saturating_sub(w as usize));
    let mut out = vec![0f32; hq * d];
    let mut lse = vec![0f32; hq];
    for head in 0..hq {
        let kh = head / g;
        let s: Vec<f64> = (lo..=pos)
            .map(|t| {
                (0..d)
                    .map(|j| f64::from(q[head * d + j]) * f64::from(keys[(t * hk + kh) * d + j]))
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let m = s.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let z: f64 = e.iter().sum();
        for j in 0..d {
            let acc: f64 = (lo..=pos)
                .zip(&e)
                .map(|(t, w)| w * f64::from(vals[(t * hk + kh) * d + j]))
                .sum();
            out[head * d + j] = (acc / z) as f32;
        }
        lse[head] = ((m + z.ln()) * std::f64::consts::LOG2_E) as f32;
    }
    (out, lse)
}

fn workspace(gpu: &Gpu) -> plan::Workspace {
    const BYTES: usize = 256 << 20;
    let int = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    let float = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    plan::Workspace {
        int_ptr: int.ptr,
        int_bytes: BYTES,
        float_ptr: float.ptr,
        float_bytes: BYTES,
    }
}

fn close(what: &str, have: &[f32], want: &[f32], tol: f32) {
    for (i, (h, w)) in have.iter().zip(want).enumerate() {
        assert!((h - w).abs() <= tol, "{what}[{i}] = {h}, want {w}");
    }
}

fn run_decode(gpu: &Gpu, c: &Case, kv_lens: &[u32], window: Option<u32>, max_grid: u32) {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let n = kv_lens.len() as u32;
    let cache = cache(gpu, c, kv_lens, 7);
    let qv: Vec<f32> = values((n * c.q_heads * c.head_dim) as usize, 3, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let q = RaggedTensor {
        data: gpu.tensor(&qv, n, c.q_heads * c.head_dim, Dtype::Bf16),
        indptr: gpu.column(&(0..=n as i32).collect::<Vec<_>>()),
    };
    let shape = plan::Shape {
        num_requests: n,
        lane_offset: 0,
        num_q_heads: c.q_heads,
        num_kv_heads: c.kv_heads,
        head_dim: c.head_dim,
        page_size: c.page_size,
        hnd_layout: c.hnd,
    };
    let live = plan::Live {
        requests: n,
        lane_offset: 0,
        row_offset: 0,
        rows: n,
    };
    let kv_len: Vec<i32> = kv_lens.iter().map(|&x| x as i32).collect();
    let built = plan::plan_decode(
        &cache.page_indptr,
        &kv_len,
        shape,
        live,
        window,
        false,
        max_grid,
        plan::Toggles::default(),
        &device,
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    let mut o = gpu.zeros(n, c.q_heads * c.head_dim, Dtype::Bf16);
    let mut lse = gpu.zeros(n, c.q_heads, Dtype::F32);
    attn::decode_lse(
        &ctx,
        q,
        &built,
        &cache.pool,
        window,
        c.head_dim,
        0.0,
        &mut o,
        &mut lse,
    )
    .unwrap();
    eprintln!(
        "decode d{} hnd={} window={window:?} split={} items={}",
        c.head_dim, c.hnd, built.info.split_kv, built.info.padded_batch_size
    );
    let (got, got_lse) = (gpu.read(&o), gpu.read(&lse));
    let row = (c.q_heads * c.head_dim) as usize;
    for r in 0..n as usize {
        let (want, want_lse) = reference_row(
            c,
            &qv[r * row..(r + 1) * row],
            &cache.keys[r],
            &cache.vals[r],
            kv_lens[r] as usize,
            1,
            0,
            window,
        );
        let tag = format!(
            "decode d{} g{} hnd={} split={} req {r}",
            c.head_dim,
            c.q_heads / c.kv_heads,
            c.hnd,
            built.info.split_kv
        );
        close(&tag, &got[r * row..(r + 1) * row], &want, 1.5e-2);
        close(
            &format!("{tag} lse"),
            &got_lse[r * c.q_heads as usize..(r + 1) * c.q_heads as usize],
            &want_lse,
            1e-3,
        );
    }
}

fn run_prefill(gpu: &Gpu, c: &Case, qo_lens: &[u32], prefix: &[u32], stride: usize) {
    run_prefill_window(gpu, c, qo_lens, prefix, stride, None, false);
}

fn run_prefill_window(
    gpu: &Gpu,
    c: &Case,
    qo_lens: &[u32],
    prefix: &[u32],
    stride: usize,
    window: Option<u32>,
    graph: bool,
) {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let n = qo_lens.len() as u32;
    let kv_lens: Vec<u32> = qo_lens.iter().zip(prefix).map(|(q, p)| q + p).collect();
    let cache = cache(gpu, c, &kv_lens, 11);
    let total: u32 = qo_lens.iter().sum();
    let qv: Vec<f32> = values((total * c.q_heads * c.head_dim) as usize, 5, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let mut qo_indptr = vec![0i32];
    for &l in qo_lens {
        qo_indptr.push(qo_indptr.last().unwrap() + l as i32);
    }
    let q = RaggedTensor {
        data: gpu.tensor(&qv, total, c.q_heads * c.head_dim, Dtype::Bf16),
        indptr: gpu.column(&qo_indptr),
    };
    let shape = plan::Shape {
        num_requests: n,
        lane_offset: 0,
        num_q_heads: c.q_heads,
        num_kv_heads: c.kv_heads,
        head_dim: c.head_dim,
        page_size: c.page_size,
        hnd_layout: c.hnd,
    };
    let live = plan::Live {
        requests: n,
        lane_offset: 0,
        row_offset: 0,
        rows: total,
    };
    let kv_len: Vec<i32> = kv_lens.iter().map(|&x| x as i32).collect();
    let built = plan::plan_prefill(
        &qo_indptr,
        &cache.page_indptr,
        &kv_len,
        total,
        shape,
        live,
        window,
        true,
        graph,
        None,
        &device,
        plan::Toggles::default(),
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    let mut o = gpu.zeros(total, c.q_heads * c.head_dim, Dtype::Bf16);
    attn::prefill(
        &ctx,
        q,
        &built,
        &cache.pool,
        window,
        c.head_dim,
        c.kv_heads,
        0.0,
        &mut o,
    )
    .unwrap();
    eprintln!(
        "prefill d{} hnd={} split={} items={} tile={}",
        c.head_dim, c.hnd, built.info.split_kv, built.info.padded_batch_size, built.info.cta_tile_q
    );
    let got = gpu.read(&o);
    let row = (c.q_heads * c.head_dim) as usize;
    for r in 0..n as usize {
        // Every `stride`-th row, and always the last (the longest reach).
        let last = qo_lens[r] as usize - 1;
        for t in (0..qo_lens[r] as usize).filter(|t| t % stride == 0 || *t == last) {
            let at = qo_indptr[r] as usize + t;
            let (want, _) = reference_row(
                c,
                &qv[at * row..(at + 1) * row],
                &cache.keys[r],
                &cache.vals[r],
                kv_lens[r] as usize,
                qo_lens[r] as usize,
                t,
                window,
            );
            let tag = format!(
                "prefill d{} g{} hnd={} split={} window={window:?} graph={graph} req {r} token {t}",
                c.head_dim,
                c.q_heads / c.kv_heads,
                c.hnd,
                built.info.split_kv
            );
            close(&tag, &got[at * row..(at + 1) * row], &want, 1.5e-2);
        }
    }
}

#[test]
fn attn_paged() {
    let Some(gpu) = Gpu::open() else { return };
    let qwen = Case {
        q_heads: 8,
        kv_heads: 2,
        head_dim: 256,
        page_size: 16,
        hnd: false,
    };
    let small = Case {
        q_heads: 4,
        kv_heads: 2,
        head_dim: 128,
        page_size: 32,
        hnd: true,
    };

    for c in [&qwen, &small] {
        // A large grid: one item per request. A one-block grid: every
        // request's kv split into chunks and folded.
        run_decode(&gpu, c, &[1, 37, 300, 1000], None, 4096);
        run_decode(&gpu, c, &[1, 37, 300, 1000], None, 1);
        run_decode(&gpu, c, &[5, 300], Some(64), 4096);
        // Fresh prompts and prompts over a cached prefix.
        run_prefill(&gpu, c, &[1, 70, 200], &[0, 0, 0], 1);
        run_prefill(&gpu, c, &[33, 5, 130], &[100, 17, 900], 1);
        // Enough query tiles to fill the device: the schedule does not split.
        run_prefill(&gpu, c, &[2600, 2600], &[0, 300], 97);
    }
}

/// The head widths and groups past Qwen's: Gemma 4's global layers (512
/// wide, group 8) and sliding ones (256 wide, group 2, a 1024 window), and
/// DeepSeek-V4's single shared head (512 wide, group 64).
#[test]
fn attn_paged_wide_heads() {
    let Some(gpu) = Gpu::open() else { return };
    let global = Case {
        q_heads: 32,
        kv_heads: 4,
        head_dim: 512,
        page_size: 16,
        hnd: false,
    };
    let sliding = Case {
        q_heads: 32,
        kv_heads: 16,
        head_dim: 256,
        page_size: 16,
        hnd: false,
    };
    let mqa = Case {
        q_heads: 64,
        kv_heads: 1,
        head_dim: 512,
        page_size: 16,
        hnd: false,
    };
    run_decode(&gpu, &global, &[1, 37, 300, 1000], None, 4096);
    run_decode(&gpu, &global, &[1, 37, 300, 1000], None, 1);
    // (DeepSeek-V4 decodes through its pool reader, not this one: only the
    // prefill takes its 64-head group.)
    for c in [&global, &mqa] {
        run_prefill(&gpu, c, &[1, 70, 200], &[0, 0, 0], 1);
        run_prefill(&gpu, c, &[33, 5, 130], &[100, 17, 900], 1);
    }
    run_decode(&gpu, &sliding, &[5, 300, 2000], Some(1024), 4096);
    run_prefill(&gpu, &sliding, &[1200, 70], &[0, 1500], 1);
    // DeepSeek-V4's sliding window (128 keys) as its decode reads it, one
    // query row a request over caches past and short of the window (whole,
    // unsplit), a wide batch of them (split), and windowed prefills whose
    // tiles straddle tokens.
    run_prefill_window(
        &gpu,
        &mqa,
        &[1, 1, 1, 1],
        &[0, 40, 127, 900],
        1,
        Some(128),
        false,
    );
    run_prefill_window(&gpu, &mqa, &[1; 64], &[600; 64], 7, Some(128), false);
    run_prefill_window(
        &gpu,
        &mqa,
        &[3, 70, 200],
        &[300, 0, 31],
        1,
        Some(128),
        false,
    );
    run_prefill_window(&gpu, &global, &[33, 130], &[100, 900], 1, Some(128), false);
    run_prefill_window(
        &gpu,
        &sliding,
        &[1200, 70],
        &[0, 1500],
        3,
        Some(1024),
        false,
    );
    // Windows long enough to split: the partials' rows count each request's
    // chunks over the window's reach.
    run_prefill_window(&gpu, &mqa, &[1, 2], &[2000, 700], 1, Some(512), false);
    // A captured plan pads its grid past the work, split or whole.
    run_prefill_window(&gpu, &mqa, &[1, 1, 1], &[0, 40, 900], 1, Some(128), true);
    run_prefill_window(&gpu, &mqa, &[1, 2], &[2000, 700], 1, Some(512), true);
    run_prefill_window(&gpu, &sliding, &[1, 3], &[3000, 5000], 1, Some(1024), false);
}

/// A decode planned captured, as the engine plans a step, over a bucket of
/// `kv_lens.len()` lanes of which `live` run (the window's rows; the rest
/// must stay untouched), at the device's own grid. A bucket whose blocks
/// fill the grid reads each request whole (no split); a small one splits
/// and folds its partials under the window.
fn run_decode_captured(
    gpu: &Gpu,
    c: &Case,
    kv_lens: &[u32],
    window: Option<u32>,
    live: std::ops::Range<u32>,
) {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let n = kv_lens.len() as u32;
    let cache = cache(gpu, c, kv_lens, 11);
    let qv: Vec<f32> = values((n * c.q_heads * c.head_dim) as usize, 5, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let q = RaggedTensor {
        data: gpu.tensor(&qv, n, c.q_heads * c.head_dim, Dtype::Bf16),
        indptr: gpu.column(&(0..=n as i32).collect::<Vec<_>>()),
    };
    let lanes = live.end - live.start;
    let shape = plan::Shape {
        // The bucket the fire states, from its first lane.
        num_requests: n - live.start,
        lane_offset: live.start,
        num_q_heads: c.q_heads,
        num_kv_heads: c.kv_heads,
        head_dim: c.head_dim,
        page_size: c.page_size,
        hnd_layout: c.hnd,
    };
    let seat = plan::Live {
        requests: lanes,
        lane_offset: live.start,
        row_offset: live.start,
        rows: lanes,
    };
    let kv_len: Vec<i32> = kv_lens[live.start as usize..live.end as usize]
        .iter()
        .map(|&x| x as i32)
        .collect();
    let built = plan::plan_decode(
        &cache.page_indptr[live.start as usize..],
        &kv_len,
        shape,
        seat,
        window,
        true,
        kernels_rocm::attn::paged::decode_max_grid_size(c.kv_heads, device.num_sm),
        plan::Toggles::default(),
        &device,
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    let mut o = gpu.zeros(n, c.q_heads * c.head_dim, Dtype::Bf16);
    let mut lse = gpu.zeros(n, c.q_heads, Dtype::F32);
    let win = gpu.ints(&[
        lanes as i32,
        live.start as i32,
        lanes as i32,
        live.start as i32,
    ]);
    ctx.arm_stage(win.ptr);
    attn::decode_lse(
        &ctx,
        q,
        &built,
        &cache.pool,
        window,
        c.head_dim,
        0.0,
        &mut o,
        &mut lse,
    )
    .unwrap();
    ctx.disarm_stage();
    let (got, got_lse) = (gpu.read(&o), gpu.read(&lse));
    let row = (c.q_heads * c.head_dim) as usize;
    let tag = |r: usize| {
        format!(
            "captured decode d{} g{} bucket {n} live {live:?} split={} req {r}",
            c.head_dim,
            c.q_heads / c.kv_heads,
            built.info.split_kv
        )
    };
    for r in 0..n as usize {
        let have = &got[r * row..(r + 1) * row];
        if !live.contains(&(r as u32)) {
            assert!(
                have.iter().all(|&x| x == 0.0),
                "{}: a dead row is untouched",
                tag(r)
            );
            continue;
        }
        let (want, want_lse) = reference_row(
            c,
            &qv[r * row..(r + 1) * row],
            &cache.keys[r],
            &cache.vals[r],
            kv_lens[r] as usize,
            1,
            0,
            window,
        );
        close(&tag(r), have, &want, 1.5e-2);
        close(
            &format!("{} lse", tag(r)),
            &got_lse[r * c.q_heads as usize..(r + 1) * c.q_heads as usize],
            &want_lse,
            1e-3,
        );
    }
}

/// Gemma 4's decode as a step captures it, at 1, 7, 128 and 512 lanes and
/// in padded buckets whose window holds a run of their lanes.
#[test]
fn attn_paged_captured_decode_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let global = Case {
        q_heads: 32,
        kv_heads: 4,
        head_dim: 512,
        page_size: 16,
        hnd: false,
    };
    let sliding = Case {
        q_heads: 32,
        kv_heads: 16,
        head_dim: 256,
        page_size: 16,
        hnd: false,
    };
    let lens = |n: u32, top: u32| -> Vec<u32> { (0..n).map(|r| 1 + (r * 7919) % top).collect() };
    for (c, window) in [(&sliding, Some(1024u32)), (&global, None)] {
        for (n, top) in [(1u32, 1500u32), (7, 1500), (128, 1300), (512, 160)] {
            run_decode_captured(&gpu, c, &lens(n, top), window, 0..n);
        }
        // A few lanes of a small bucket (split, folded under the window)
        // and runs inside wide ones (whole).
        run_decode_captured(&gpu, c, &lens(8, 1500), window, 2..7);
        run_decode_captured(&gpu, c, &lens(128, 1300), window, 7..107);
        run_decode_captured(&gpu, c, &lens(512, 160), window, 100..228);
    }
}

/// Gemma 4's prefill as a step captures it: chunks whose query tiles fill
/// the grid run unsplit (no partials to fold), captured or not; a short
/// chunk over a long prefix still splits.
#[test]
fn attn_paged_prefill_rows_fill_the_grid() {
    let Some(gpu) = Gpu::open() else { return };
    let global = Case {
        q_heads: 32,
        kv_heads: 4,
        head_dim: 512,
        page_size: 16,
        hnd: false,
    };
    let sliding = Case {
        q_heads: 32,
        kv_heads: 16,
        head_dim: 256,
        page_size: 16,
        hnd: false,
    };
    for graph in [false, true] {
        run_prefill_window(
            &gpu,
            &sliding,
            &[1500, 1100],
            &[0, 700],
            113,
            Some(1024),
            graph,
        );
        run_prefill_window(&gpu, &global, &[1300, 1300], &[0, 500], 211, None, graph);
        run_prefill_window(&gpu, &global, &[7], &[2000], 1, None, graph);
    }
}

/// Wide groups whose decode weighs its values on the matrix core: MiMo's and
/// MiniMax-M3's sixteen query heads a kv head (256 wide, after the padding),
/// Qwen3.8-Flash-Next's twelve, and eight 128-wide ones in HND pages; whole
/// and split schedules, and a sliding window.
#[test]
fn attn_paged_wide_groups() {
    let Some(gpu) = Gpu::open() else { return };
    for c in [
        Case { q_heads: 64, kv_heads: 4, head_dim: 256, page_size: 16, hnd: false },
        Case { q_heads: 24, kv_heads: 2, head_dim: 256, page_size: 16, hnd: false },
        Case { q_heads: 16, kv_heads: 2, head_dim: 128, page_size: 32, hnd: true },
    ] {
        run_decode(&gpu, &c, &[1, 37, 300, 1000], None, 4096);
        run_decode(&gpu, &c, &[1, 37, 300, 1000], None, 1);
        run_decode(&gpu, &c, &[5, 300, 2000], Some(128), 4096);
    }
}
