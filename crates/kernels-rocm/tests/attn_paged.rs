//! Paged attention against a host reference: decode and causal prefill,
//! split and unsplit schedules, NHD and HND pages, bf16 and e4m3 pools, with
//! real plans staged through the schedulers the engine runs.

#![cfg(feature = "rocm")]

mod common;

use common::spread::{agrees, out};
use common::{Gpu, KV, e4m3_code, e4m3_value, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, plan};
use kernels_rocm::{KvPool, RaggedTensor};

struct Case {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    page_size: u32,
    hnd: bool,
    /// The pool holds OCP e4m3 codes under a unit scale.
    fp8: bool,
}

impl Case {
    const fn new(q_heads: u32, kv_heads: u32, head_dim: u32, page_size: u32, hnd: bool) -> Self {
        Self {
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            hnd,
            fp8: false,
        }
    }

    const fn e4m3(self) -> Self {
        Self { fp8: true, ..self }
    }
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
    let plane = |x: &[f32]| {
        if c.fp8 {
            let codes: Vec<u8> = x.iter().map(|&v| e4m3_code(v)).collect();
            gpu.upload(&codes, rows, (h * d) as u32, Dtype::E4m3)
        } else {
            gpu.tensor(x, rows, (h * d) as u32, Dtype::Bf16)
        }
    };
    if c.fp8 {
        for x in keys.iter_mut().chain(vals.iter_mut()) {
            x.iter_mut().for_each(|v| *v = e4m3_value(e4m3_code(*v)));
        }
    }
    let pool = KvPool {
        keys: plane(&kp),
        values: plane(&vp),
        page_indices: gpu.column(&page_indices),
        page_indptr: gpu.column(&page_indptr),
        last_page_lens: gpu.column(&last),
        page_size: ps as i32,
        seq_stride: if c.hnd { d as i64 } else { (h * d) as i64 },
        head_stride: if c.hnd { (ps * d) as i64 } else { d as i64 },
        layout: i32::from(c.hnd),
        max_pages_per_request: *pages_of.iter().max().unwrap() as i32,
        pages_in_batch: page_indices.len() as i32,
        live: [0, 0],
        ..KV
    }
    .with_packed_stride();
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
        None,
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
            "decode d{} g{} hnd={} fp8={} split={} req {r}",
            c.head_dim,
            c.q_heads / c.kv_heads,
            c.hnd,
            c.fp8,
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
    let tag = format!(
        "decode d{} hnd={} fp8={} window={window:?} split={}",
        c.head_dim, c.hnd, c.fp8, built.info.split_kv
    );
    agrees(
        gpu,
        &tag,
        &[cache.pool],
        &[],
        &[out(&o), out(&lse)],
        |_, pools, _| {
            let (mut o, mut lse) = (o, lse);
            attn::decode_lse(
                &ctx, q, &built, &pools[0], window, c.head_dim, 0.0, &mut o, &mut lse, None,
            )
            .unwrap();
        },
    );
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
                "prefill d{} g{} hnd={} fp8={} split={} window={window:?} graph={graph} req {r} token {t}",
                c.head_dim,
                c.q_heads / c.kv_heads,
                c.hnd,
                c.fp8,
                built.info.split_kv
            );
            close(&tag, &got[at * row..(at + 1) * row], &want, 1.5e-2);
        }
    }
    let tag = format!(
        "prefill d{} hnd={} fp8={} window={window:?} graph={graph} split={}",
        c.head_dim, c.hnd, c.fp8, built.info.split_kv
    );
    agrees(gpu, &tag, &[cache.pool], &[], &[out(&o)], |_, pools, _| {
        let mut o = o;
        attn::prefill(
            &ctx, q, &built, &pools[0], window, c.head_dim, c.kv_heads, 0.0, &mut o,
        )
        .unwrap();
    });
}

/// A windowed run: the case, its query and cached lengths, its splits, its
/// window, and whether a captured plan pads the grid.
type WindowRun<'a> = (&'a Case, &'a [u32], &'a [u32], usize, u32, bool);

/// An Inkling run: query and cached lengths, the window, the page floor, the
/// (sink, scale) pair and the split budget.
type InklingRun = (
    &'static [u32],
    &'static [u32],
    Option<u32>,
    u32,
    (u32, f32),
    u32,
);

#[test]
fn attn_paged() {
    let Some(gpu) = Gpu::open() else { return };
    let qwen = Case::new(8, 2, 256, 16, false);
    let small = Case::new(4, 2, 128, 32, true);

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
/// wide, group 8) and sliding ones (256 wide, group 2, a 1024 window),
/// DeepSeek-V4's single shared head (512 wide, group 64) and 64-wide heads.
#[test]
fn attn_paged_wide_heads() {
    let Some(gpu) = Gpu::open() else { return };
    let global = Case::new(32, 4, 512, 16, false);
    let sliding = Case::new(32, 16, 256, 16, false);
    let mqa = Case::new(64, 1, 512, 16, false);
    run_decode(&gpu, &global, &[1, 37, 300, 1000], None, 4096);
    run_decode(&gpu, &global, &[1, 37, 300, 1000], None, 1);
    // (DeepSeek-V4 decodes through its pool reader, not this one: only the
    // prefill takes its 64-head group.)
    // Muse Spark's 64-wide heads walk the wide kernel too.
    let narrow = Case::new(16, 2, 64, 16, false);
    for c in [&global, &mqa, &narrow] {
        run_prefill(&gpu, c, &[1, 70, 200], &[0, 0, 0], 1);
        run_prefill(&gpu, c, &[33, 5, 130], &[100, 17, 900], 1);
    }
    run_decode(&gpu, &sliding, &[5, 300, 2000], Some(1024), 4096);
    run_prefill(&gpu, &sliding, &[1200, 70], &[0, 1500], 1);
    // DeepSeek-V4's 128-key window as its decode reads it (whole, then a
    // split batch), windowed prefills whose tiles straddle tokens, windows
    // long enough to split, and captured plans padding their grid.
    let windows: &[WindowRun] = &[
        (&mqa, &[1, 1, 1, 1], &[0, 40, 127, 900], 1, 128, false),
        (&mqa, &[1; 64], &[600; 64], 7, 128, false),
        (&mqa, &[3, 70, 200], &[300, 0, 31], 1, 128, false),
        (&sliding, &[1200, 70], &[0, 1500], 3, 1024, false),
        (&mqa, &[1, 2], &[2000, 700], 1, 512, false),
        (&mqa, &[1, 1, 1], &[0, 40, 900], 1, 128, true),
        (&mqa, &[1, 2], &[2000, 700], 1, 512, true),
    ];
    for &(c, qo, prefix, stride, window, graph) in windows {
        run_prefill_window(&gpu, c, qo, prefix, stride, Some(window), graph);
    }
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
        None,
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
    let global = Case::new(32, 4, 512, 16, false);
    let sliding = Case::new(32, 16, 256, 16, false);
    let lens = |n: u32, top: u32| -> Vec<u32> { (0..n).map(|r| 1 + (r * 7919) % top).collect() };
    for (c, window) in [(&sliding, Some(1024u32)), (&global, None)] {
        // Whole buckets; a few lanes of a small one (split, folded under
        // the window) and a run inside a wide one (whole).
        run_decode_captured(&gpu, c, &lens(7, 1500), window, 0..7);
        run_decode_captured(&gpu, c, &lens(512, 160), window, 0..512);
        run_decode_captured(&gpu, c, &lens(8, 1500), window, 2..7);
        run_decode_captured(&gpu, c, &lens(512, 160), window, 100..228);
    }
}

/// Gemma 4's prefill as a step captures it: chunks whose query tiles fill
/// the grid run unsplit (no partials to fold), captured or not; a short
/// chunk over a long prefix still splits.
#[test]
fn attn_paged_prefill_rows_fill_the_grid() {
    let Some(gpu) = Gpu::open() else { return };
    let global = Case::new(32, 4, 512, 16, false);
    let sliding = Case::new(32, 16, 256, 16, false);
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
        Case::new(64, 4, 256, 16, false),
        Case::new(24, 2, 256, 16, false),
        Case::new(16, 2, 128, 32, true),
    ] {
        run_decode(&gpu, &c, &[1, 37, 300, 1000], None, 4096);
        run_decode(&gpu, &c, &[1, 37, 300, 1000], None, 1);
        run_decode(&gpu, &c, &[5, 300, 2000], Some(128), 4096);
    }
}

/// An e4m3 pool read as it is, by every decode and prefill walk: the
/// matrix-core decode (and its matrix-core values at a group of eight),
/// the column-split decode at 512-wide heads, the vector decode of a wide
/// group, the tiled prefill and the wide one; and the append's codes.
#[test]
fn attn_paged_e4m3() {
    let Some(gpu) = Gpu::open() else { return };
    let narrow = Case::new(8, 2, 128, 16, false).e4m3();
    let pvm = Case::new(16, 2, 128, 32, true).e4m3();
    let global = Case::new(32, 4, 512, 16, false).e4m3();
    let small = Case::new(8, 2, 64, 16, false).e4m3();
    for c in [&narrow, &pvm, &global] {
        run_decode(&gpu, c, &[1, 37, 300, 1000], None, 4096);
        run_decode(&gpu, c, &[1, 37, 300, 1000], None, 1);
    }
    run_decode(
        &gpu,
        &Case::new(32, 1, 128, 16, false).e4m3(),
        &[5, 300],
        Some(64),
        4096,
    );
    for c in [&narrow, &pvm, &global, &small] {
        run_prefill(&gpu, c, &[1, 70, 200], &[0, 0, 0], 1);
        run_prefill(&gpu, c, &[33, 5, 130], &[100, 17, 900], 3);
    }

    // The append: bf16 rows to their codes, ties to even, saturated, −0
    // as 0x00, through explicit pages and slots.
    let ctx = gpu.ctx();
    let (rows, width) = (5u32, 2 * 64u32);
    let mut x: Vec<f32> = values((rows * width) as usize, 9, 300.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    x[..6].copy_from_slice(&[-0.0, 1000.0, -1000.0, 0.0009765625, -0.001, 448.0]);
    let pool = KvPool {
        keys: gpu.upload(&vec![0x5au8; 64 * width as usize], 64, width, Dtype::E4m3),
        values: gpu.upload(&vec![0x5au8; 64 * width as usize], 64, width, Dtype::E4m3),
        page_size: 16,
        seq_stride: i64::from(width),
        head_stride: 64,
        live: [0, 0],
        ..KV
    }
    .with_packed_stride();
    let slots = [(3u32, 1u32), (0, 15), (2, 0), (3, 2), (1, 7)];
    let k = RaggedTensor {
        data: gpu.tensor(&x, rows, width, Dtype::Bf16),
        indptr: gpu.column(&[0, rows as i32]),
    };
    let page = gpu.upload(&slots.map(|s| s.0), rows, 1, Dtype::U32);
    let offset = gpu.upload(&slots.map(|s| s.1), rows, 1, Dtype::U32);
    attn::kv_append(&ctx, k, k.data, &pool, page, offset).unwrap();
    let (kc, vc) = (
        gpu.read_as::<u8>(&pool.keys),
        gpu.read_as::<u8>(&pool.values),
    );
    for (r, &(pg, off)) in slots.iter().enumerate() {
        let at = ((pg * 16 + off) * width) as usize;
        let want: Vec<u8> = x[r * width as usize..(r + 1) * width as usize]
            .iter()
            .map(|&v| e4m3_code(v))
            .collect();
        assert_eq!(&kc[at..at + width as usize], &want[..], "row {r} key codes");
        assert_eq!(
            &vc[at..at + width as usize],
            &want[..],
            "row {r} value codes"
        );
    }
}

/// `reference_row` under a relative bias: the bias at each key's distance
/// inside `extent` and the logits grown by `1 + α·ln((pos + 1) / floor)`.
#[allow(clippy::too_many_arguments)]
fn reference_rel_row(
    c: &Case,
    q: &[f32],
    keys: &[f32],
    vals: &[f32],
    kv: usize,
    qo_len: usize,
    token: usize,
    window: Option<u32>,
    bias: &[f32],
    extent: usize,
    log: (u32, f32),
) -> Vec<f32> {
    let (hq, hk, d) = (c.q_heads as usize, c.kv_heads as usize, c.head_dim as usize);
    let g = hq / hk;
    let scale = 1.0 / (d as f64).sqrt();
    let pos = kv - qo_len + token;
    let lo = window.map_or(0, |w| (pos + 1).saturating_sub(w as usize));
    let grow = if log.1 != 0.0 && log.0 != 0 && pos + 1 > log.0 as usize {
        1.0 + f64::from(log.1) * ((pos + 1) as f64 / f64::from(log.0)).ln()
    } else {
        1.0
    };
    let mut out = vec![0f32; hq * d];
    for head in 0..hq {
        let kh = head / g;
        let s: Vec<f64> = (lo..=pos)
            .map(|t| {
                let mut x = (0..d)
                    .map(|j| f64::from(q[head * d + j]) * f64::from(keys[(t * hk + kh) * d + j]))
                    .sum::<f64>()
                    * scale;
                if pos - t < extent {
                    x += f64::from(bias[head * extent + pos - t]);
                }
                x * grow
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
    }
    out
}

/// Inkling's relative-bias attention through the plans the engine stages:
/// decode rows (all one query a request) on the matrix-core decode, split
/// and whole, and prefill rows on the tiled walk; windows and the log
/// growth past its floor.
#[allow(clippy::too_many_arguments)]
fn run_rel(
    gpu: &Gpu,
    c: &Case,
    qo_lens: &[u32],
    prefix: &[u32],
    window: Option<u32>,
    extent: u32,
    log: (u32, f32),
    max_grid: u32,
) {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let n = qo_lens.len() as u32;
    let decode = qo_lens.iter().all(|&l| l == 1);
    let kv_lens: Vec<u32> = qo_lens.iter().zip(prefix).map(|(q, p)| q + p).collect();
    let cache = cache(gpu, c, &kv_lens, 13);
    let total: u32 = qo_lens.iter().sum();
    let qv: Vec<f32> = values((total * c.q_heads * c.head_dim) as usize, 5, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let bias = values((total * c.q_heads * extent) as usize, 9, 2.0);
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
    let rel = attn::RelBias {
        bias: gpu.tensor(&bias, total, c.q_heads * extent, Dtype::F32),
        extent,
        log_floor: log.0,
        log_alpha: log.1,
    };
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    let mut o = gpu.zeros(total, c.q_heads * c.head_dim, Dtype::Bf16);
    if decode {
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
        attn::decode_rel(
            &ctx,
            q,
            &built,
            &cache.pool,
            rel,
            window,
            c.head_dim,
            scale,
            &mut o,
        )
        .unwrap();
    } else {
        let built = plan::plan_prefill(
            &qo_indptr,
            &cache.page_indptr,
            &kv_len,
            total,
            shape,
            live,
            window,
            true,
            false,
            None,
            &device,
            plan::Toggles::default(),
            workspace(gpu),
        )
        .unwrap();
        built.stage(&ctx).unwrap();
        attn::prefill_rel(
            &ctx,
            q,
            &built,
            &cache.pool,
            rel,
            window,
            c.head_dim,
            c.kv_heads,
            scale,
            &mut o,
        )
        .unwrap();
    }
    let got = gpu.read(&o);
    let row = (c.q_heads * c.head_dim) as usize;
    let brow = (c.q_heads * extent) as usize;
    for r in 0..n as usize {
        for t in 0..qo_lens[r] as usize {
            let at = qo_indptr[r] as usize + t;
            let want = reference_rel_row(
                c,
                &qv[at * row..(at + 1) * row],
                &cache.keys[r],
                &cache.vals[r],
                kv_lens[r] as usize,
                qo_lens[r] as usize,
                t,
                window,
                &bias[at * brow..(at + 1) * brow],
                extent as usize,
                log,
            );
            let tag = format!(
                "rel {} d{} g{} hnd={} window={window:?} req {r} token {t}",
                if decode { "decode" } else { "prefill" },
                c.head_dim,
                c.q_heads / c.kv_heads,
                c.hnd
            );
            close(&tag, &got[at * row..(at + 1) * row], &want, 1.5e-2);
        }
    }
}

#[test]
fn attn_paged_relative_bias() {
    let Some(gpu) = Gpu::open() else { return };
    // Inkling-Small: 32 query heads over 8, 128 wide; its global layers grow
    // past the floor, its local ones read a 128-key window.
    let ink = Case::new(32, 8, 128, 16, false);
    let wide = Case::new(16, 1, 256, 16, true);
    // Decode rows, whole and split; a window; prefill rows, grown and windowed.
    #[rustfmt::skip]
    let cases: &[InklingRun] = &[
        (&[1, 1, 1, 1], &[0, 36, 299, 1000], None, 16, (64, 0.1), 4096),
        (&[1, 1, 1, 1], &[0, 36, 299, 1000], None, 16, (64, 0.1), 1),
        (&[1, 1], &[200, 2000], Some(128), 300, (0, 0.0), 4096),
        (&[70, 3, 200], &[0, 40, 900], None, 16, (64, 0.1), 4096),
        (&[150, 9], &[300, 0], Some(128), 300, (100, 0.3), 4096),
    ];
    for c in [&ink, &wide] {
        for &(qo, prefix, window, extent, log, grid) in cases {
            run_rel(&gpu, c, qo, prefix, window, extent, log, grid);
        }
    }
}

/// The masked prefill reads each request's two segments as the custom mask
/// states them: its committed run (keys below its origin) whole, cut at the
/// row's position unless its word is wide, and its buffer (keys from its
/// origin) by the bits of its plane, `qo · (kv − origin) + (key − origin)`
/// from byte `indptr[req]`, least bit first.
#[test]
fn attn_paged_masked_prefill() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    for c in [
        Case::new(8, 4, 128, 16, false),
        Case::new(16, 4, 256, 16, false),
        Case::new(8, 2, 64, 16, false),
    ] {
        // A causal request over a prefix whose committed run ends at 45, and
        // a wide one that reads only its buffer's bits.
        let qo_lens = [37u32, 5];
        let prefix = [60u32, 0];
        let origins = [45u32, 0];
        let wide = [false, true];
        let n = qo_lens.len() as u32;
        let kv_lens: Vec<u32> = qo_lens.iter().zip(&prefix).map(|(q, p)| q + p).collect();
        let cache = cache(&gpu, &c, &kv_lens, 23);
        let total: u32 = qo_lens.iter().sum();
        let qv: Vec<f32> = values((total * c.q_heads * c.head_dim) as usize, 7, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let mut qo_indptr = vec![0i32];
        for &l in &qo_lens {
            qo_indptr.push(qo_indptr.last().unwrap() + l as i32);
        }
        // Each request's plane of bits, and what a row reads.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut bit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            !state.is_multiple_of(3)
        };
        let mut bytes: Vec<u8> = Vec::new();
        let mut indptr: Vec<i64> = Vec::new();
        let mut reads: Vec<Vec<Vec<bool>>> = Vec::new();
        for r in 0..n as usize {
            let (kv, qo, origin) = (
                kv_lens[r] as usize,
                qo_lens[r] as usize,
                origins[r] as usize,
            );
            let width = kv - origin;
            let mut plane = vec![false; qo * width];
            for t in 0..qo {
                for j in 0..width {
                    // Every row keeps its own key, so none reads nothing.
                    plane[t * width + j] = bit() || origin + j == kv - qo + t;
                }
            }
            indptr.push(bytes.len() as i64);
            let mut packed = vec![0u8; plane.len().div_ceil(8)];
            for (i, &b) in plane.iter().enumerate() {
                if b {
                    packed[i / 8] |= 1 << (i % 8);
                }
            }
            bytes.extend(packed);
            reads.push(
                (0..qo)
                    .map(|t| {
                        (0..kv)
                            .map(|key| {
                                let causal_ok = wide[r] || key <= kv - qo + t;
                                let bits_ok = key < origin || plane[t * width + key - origin];
                                causal_ok && bits_ok
                            })
                            .collect()
                    })
                    .collect(),
            );
        }
        indptr.push(bytes.len() as i64);
        let words: Vec<u32> = origins
            .iter()
            .zip(&wide)
            .map(|(&o, &w)| o | if w { 1 << 31 } else { 0 })
            .collect();
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
        let mut built = plan::plan_prefill(
            &qo_indptr,
            &cache.page_indptr,
            &kv_len,
            total,
            shape,
            live,
            None,
            false,
            false,
            Some(gpu.upload(&indptr, n + 1, 1, Dtype::I64)),
            &device,
            plan::Toggles::default(),
            workspace(&gpu),
        )
        .unwrap();
        built.mask_origin = Some(gpu.upload(&words, n, 1, Dtype::U32));
        built.stage(&ctx).unwrap();
        let mask = gpu.upload(&bytes, 1, bytes.len() as u32, Dtype::U8);
        let mut o = gpu.zeros(total, c.q_heads * c.head_dim, Dtype::Bf16);
        attn::masked(
            &ctx,
            q,
            &built,
            mask,
            &cache.pool,
            None,
            c.head_dim,
            0.0,
            &mut o,
        )
        .unwrap();
        let got = gpu.read(&o);
        let (hq, hk, d) = (c.q_heads as usize, c.kv_heads as usize, c.head_dim as usize);
        let (g, row, scale) = (hq / hk, hq * d, 1.0 / (d as f64).sqrt());
        for r in 0..n as usize {
            for t in 0..qo_lens[r] as usize {
                let at = qo_indptr[r] as usize + t;
                let qrow = &qv[at * row..(at + 1) * row];
                let keys: Vec<usize> = (0..kv_lens[r] as usize)
                    .filter(|&k| reads[r][t][k])
                    .collect();
                let mut want = vec![0f32; row];
                for head in 0..hq {
                    let kh = head / g;
                    let s: Vec<f64> = keys
                        .iter()
                        .map(|&k| {
                            (0..d)
                                .map(|j| {
                                    f64::from(qrow[head * d + j])
                                        * f64::from(cache.keys[r][(k * hk + kh) * d + j])
                                })
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    let m = s.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for j in 0..d {
                        let acc: f64 = keys
                            .iter()
                            .zip(&e)
                            .map(|(&k, w)| w * f64::from(cache.vals[r][(k * hk + kh) * d + j]))
                            .sum();
                        want[head * d + j] = (acc / z) as f32;
                    }
                }
                let tag = format!(
                    "masked prefill d{d} g{g} req {r} token {t} wide {}",
                    wide[r]
                );
                close(&tag, &got[at * row..(at + 1) * row], &want, 1.5e-2);
            }
        }
    }
}

/// A captured prefill body over a 256-lane carve of 2048 rows (Qwen3-30B-A3B's
/// heads), its grid padded past the live work items, captured at one live
/// request and replayed at 200 and then at one again: the live count is
/// staged each fire (as the mask is), never baked at capture, and each
/// replay matches the unpadded eager plan of the same fire bit for bit. A
/// grid swizzled over its padded width hands some chiplets only empty
/// blocks (`prefill_item`); over the live width it is the unpadded grid.
#[test]
fn attn_paged_padded_prefill_replays_as_unpadded() {
    const LANES: u32 = 256;
    const ROWS: u32 = 2048;
    const MANY: usize = 200;
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let c = Case::new(32, 4, 128, 16, false);
    // The many-request fire: one long request and 199 short ones; the
    // one-request fire is the long request's cache with all the rows.
    let mut qo_many = vec![8u32; MANY];
    qo_many[0] = ROWS - 8 * (MANY as u32 - 1);
    let kv_lens: Vec<u32> = (0..MANY as u32)
        .map(|i| if i == 0 { 5048 } else { 8 + (i * 37) % 900 })
        .collect();
    let cache = cache(&gpu, &c, &kv_lens, 23);
    let width = c.q_heads * c.head_dim;
    let qv: Vec<f32> = values((ROWS * width) as usize, 9, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let qdata = gpu.tensor(&qv, ROWS, width, Dtype::Bf16);
    let indptr = |lens: &[u32]| {
        let mut v = vec![0i32];
        for &l in lens {
            v.push(v.last().unwrap() + l as i32);
        }
        v
    };
    let (one, many) = (indptr(&[ROWS]), indptr(&qo_many));
    // One query indptr column over the carve's lanes for every captured
    // fire, rewritten in place, as the engine's fire inputs are.
    let qcol = gpu.column(&[0; LANES as usize + 1]);
    let write = |at: &[i32]| {
        let mut v = at.to_vec();
        v.resize(LANES as usize + 1, *at.last().unwrap());
        gpu.write(qcol.ptr, &v);
    };
    let plan_of = |at: &[i32], lanes: u32, graph: bool, ws: plan::Workspace| {
        let n = (at.len() - 1) as u32;
        plan::plan_prefill(
            at,
            &cache.page_indptr[..=n as usize],
            &[],
            ROWS,
            plan::Shape {
                num_requests: lanes,
                lane_offset: 0,
                num_q_heads: c.q_heads,
                num_kv_heads: c.kv_heads,
                head_dim: c.head_dim,
                page_size: c.page_size,
                hnd_layout: c.hnd,
            },
            plan::Live {
                requests: n,
                lane_offset: 0,
                row_offset: 0,
                rows: ROWS,
            },
            None,
            true,
            graph,
            None,
            &device,
            plan::Toggles::default(),
            ws,
        )
        .unwrap()
    };
    let mut o = gpu.zeros(ROWS, width, Dtype::Bf16);
    let bytes = (ROWS * width * 2) as usize;
    let fire = |indptr, built: &plan::PrefillPlan, o: &mut kernels_rocm::Tensor| {
        let q = RaggedTensor {
            data: qdata,
            indptr,
        };
        attn::prefill(
            &ctx,
            q,
            built,
            &cache.pool,
            None,
            c.head_dim,
            c.kv_heads,
            0.0,
            o,
        )
        .unwrap();
    };
    // The unpadded eager plans of each fire.
    let mut eager = Vec::new();
    for at in [&many, &one] {
        let n = (at.len() - 1) as u32;
        let built = plan_of(at, n, false, workspace(&gpu));
        built.stage(&ctx).unwrap();
        gpu.fill(o.ptr, 0, bytes);
        fire(gpu.column(at), &built, &mut o);
        eprintln!(
            "eager {n} requests: grid {} split={}",
            built.info.padded_batch_size, built.info.split_kv
        );
        eager.push(gpu.read(&o));
    }
    assert_ne!(eager[0], eager[1], "the two fires attend differently");
    // One captured layout serves both fires: the carve's, not the keys'.
    let ws = workspace(&gpu);
    let (cap_many, cap_one) = (
        plan_of(&many, LANES, true, ws),
        plan_of(&one, LANES, true, ws),
    );
    assert_eq!(
        cap_many.info, cap_one.info,
        "one body's layout for every fire"
    );
    assert!(cap_one.info.block_live_offset.is_some());
    eprintln!(
        "captured over {LANES} lanes: grid {} split={}",
        cap_one.info.padded_batch_size, cap_one.info.split_kv
    );
    // Warm (compile) the captured plan's launch, then capture it at one
    // live request.
    write(&one);
    cap_one.stage(&ctx).unwrap();
    fire(qcol, &cap_one, &mut o);
    let graph = gpu.capture(|| fire(qcol, &cap_one, &mut o));
    for (at, built, want, tag) in [
        (&many, &cap_many, &eager[0], "200 live requests"),
        (&one, &cap_one, &eager[1], "one live request"),
    ] {
        write(at);
        built.stage(&ctx).unwrap();
        gpu.fill(o.ptr, 0, bytes);
        graph.replay();
        let got = gpu.read(&o);
        let off = got
            .iter()
            .zip(want.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            off,
            0,
            "replay at {tag}: {off} of {} values differ from unpadded",
            got.len()
        );
    }
}

/// One captured prefill body of Qwen3-0.6B's heads over a 256-lane carve of
/// 2048 rows, its layout split, replayed per fire: one request of all the
/// rows over a long prefix, cut into chunks to fill the device, and 256
/// requests of eight rows, whose tiles fill it, staged whole (direct). The
/// chunks are each fire's, staged with it: the whole replay matches its
/// unpadded eager plan bit for bit (the fold leaves; a fold over the split
/// replay's stale partials would not), and the cut replay matches the same
/// plan fired without the graph bit for bit and its eager plan (cut into
/// fewer chunks, a different summation order) within bf16's rounding.
#[test]
fn attn_paged_captured_prefill_splits_per_fire() {
    const LANES: u32 = 256;
    const ROWS: u32 = 2048;
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let c = Case::new(16, 8, 128, 16, false);
    let kv_lens: Vec<u32> = (0..LANES)
        .map(|i| if i == 0 { 8048 } else { 8 + (i * 37) % 900 })
        .collect();
    let cache = cache(&gpu, &c, &kv_lens, 31);
    let width = c.q_heads * c.head_dim;
    let qv: Vec<f32> = values((ROWS * width) as usize, 13, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let qdata = gpu.tensor(&qv, ROWS, width, Dtype::Bf16);
    let indptr = |lens: &[u32]| {
        let mut v = vec![0i32];
        for &l in lens {
            v.push(v.last().unwrap() + l as i32);
        }
        v
    };
    let (one, many) = (indptr(&[ROWS]), indptr(&vec![8; LANES as usize]));
    let qcol = gpu.column(&[0; LANES as usize + 1]);
    let write = |at: &[i32]| {
        let mut v = at.to_vec();
        v.resize(LANES as usize + 1, *at.last().unwrap());
        gpu.write(qcol.ptr, &v);
    };
    let plan_of = |at: &[i32], lanes: u32, graph: bool, ws: plan::Workspace| {
        let n = (at.len() - 1) as u32;
        plan::plan_prefill(
            at,
            &cache.page_indptr[..=n as usize],
            &[],
            ROWS,
            plan::Shape {
                num_requests: lanes,
                lane_offset: 0,
                num_q_heads: c.q_heads,
                num_kv_heads: c.kv_heads,
                head_dim: c.head_dim,
                page_size: c.page_size,
                hnd_layout: c.hnd,
            },
            plan::Live {
                requests: n,
                lane_offset: 0,
                row_offset: 0,
                rows: ROWS,
            },
            None,
            true,
            graph,
            None,
            &device,
            plan::Toggles::default(),
            ws,
        )
        .unwrap()
    };
    let mut o = gpu.zeros(ROWS, width, Dtype::Bf16);
    let bytes = (ROWS * width * 2) as usize;
    let fire = |indptr, built: &plan::PrefillPlan, o: &mut kernels_rocm::Tensor| {
        let q = RaggedTensor {
            data: qdata,
            indptr,
        };
        attn::prefill(
            &ctx,
            q,
            built,
            &cache.pool,
            None,
            c.head_dim,
            c.kv_heads,
            0.0,
            o,
        )
        .unwrap();
    };
    let exact = |tag: &str, got: &[f32], want: &[f32]| {
        let off = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(off, 0, "{tag}: {off} of {} values differ", got.len());
    };
    let mut eager = Vec::new();
    for at in [&one, &many] {
        let n = (at.len() - 1) as u32;
        let built = plan_of(at, n, false, workspace(&gpu));
        built.stage(&ctx).unwrap();
        gpu.fill(o.ptr, 0, bytes);
        fire(gpu.column(at), &built, &mut o);
        eprintln!(
            "eager {n} requests: grid {} split={}",
            built.info.padded_batch_size, built.info.split_kv
        );
        eager.push(gpu.read(&o));
    }
    let ws = workspace(&gpu);
    let (cut, whole) = (
        plan_of(&one, LANES, true, ws),
        plan_of(&many, LANES, true, ws),
    );
    assert_eq!(cut.info, whole.info, "one body's layout for every fire");
    assert!(cut.info.split_kv && cut.info.block_direct_offset.is_some());
    // The cut fire, fired as planned without the graph.
    write(&one);
    cut.stage(&ctx).unwrap();
    gpu.fill(o.ptr, 0, bytes);
    fire(qcol, &cut, &mut o);
    let cut_walked = gpu.read(&o);
    let graph = gpu.capture(|| fire(qcol, &cut, &mut o));
    for (at, built, tag) in [
        (&many, &whole, "whole"),
        (&one, &cut, "cut"),
        (&many, &whole, "whole after a cut"),
    ] {
        write(at);
        built.stage(&ctx).unwrap();
        gpu.fill(o.ptr, 0, bytes);
        graph.replay();
        let got = gpu.read(&o);
        if tag.starts_with("whole") {
            exact(
                &format!("{tag} replay against its eager plan"),
                &got,
                &eager[1],
            );
        } else {
            exact(&format!("{tag} replay against its walk"), &got, &cut_walked);
            close(
                &format!("{tag} replay against its eager plan"),
                &got,
                &eager[0],
                1.5e-2,
            );
        }
    }
    eprintln!(
        "captured over {LANES} lanes: grid {} split",
        cut.info.padded_batch_size
    );
}
