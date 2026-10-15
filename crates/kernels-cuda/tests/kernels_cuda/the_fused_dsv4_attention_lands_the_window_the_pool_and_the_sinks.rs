#![cfg(feature = "cuda")]

//! DeepSeek-V4's attention at decode widths in one launch
//! (`attn::dsv4::decode`): a row's 128-position window of the shared
//! key/value plane, its compressed rows (all it sees, or an indexer's picks
//! with ids past the visible ones and an early end), folded together with
//! the sinks, against an f32 walk over the same bf16 rows; at row counts that
//! split a row's keys over clusters of 8, 4, 2 and 1 blocks.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::dsv4;
use kernels_cuda::tensor::{KvPool, Tensor};

const PAGE: usize = 64;
const HEADS: usize = 64;
const D: usize = 512;
const WINDOW: usize = 128;

fn pool_of(gpu: &mut Gpu, keys: u64, cells: usize, pages: &[u32], indptr: &[u32]) -> KvPool {
    KvPool {
        keys: Tensor::new(keys, cells as u32, D as u32, Dtype::Bf16),
        values: Tensor::new(keys, cells as u32, D as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(gpu.up(pages), pages.len() as u32, 1, Dtype::U32),
        page_indptr: Tensor::new(gpu.up(indptr), indptr.len() as u32, 1, Dtype::U32),
        last_page_lens: Tensor::ABSENT,
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: D as i64,
        head_stride: D as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: (pages.len() / 2) as i32,
        pages_in_batch: pages.len() as i32,
    }
}

#[derive(Clone, Copy)]
enum Pool {
    None,
    Dense(usize),
    Selected(usize, usize),
}

fn check(rows: usize, pool: Pool, bench: bool) {
    // Two requests of 2304 positions each, their pages interleaved; the
    // window and the compressed rows are two planes on the same tables.
    let span = 2304usize;
    let per_req = span / PAGE;
    let pages: Vec<u32> = (0..2 * per_req as u32)
        .map(|i| {
            if i < per_req as u32 {
                2 * i
            } else {
                2 * (i - per_req as u32) + 1
            }
        })
        .collect();
    let indptr: Vec<u32> = vec![0, per_req as u32, 2 * per_req as u32];
    let cells = 2 * per_req * PAGE;
    let slot = |req: usize, pos: usize| -> usize {
        pages[indptr[req] as usize + pos / PAGE] as usize * PAGE + pos % PAGE
    };

    let base: [i32; 8] = [2, 3, 50, 700, 2047, 2100, 1500, 127];
    let positions: Vec<i32> = (0..rows)
        .map(|r| base[r % 8] + (r / 8) as i32 % 5)
        .collect();
    let reqs: Vec<i32> = (0..rows).map(|r| (r % 2) as i32).collect();

    let mut lcg = Lcg::seeded(0xd5f4 + rows as u64);
    let (q_bits, q) = lcg.row(rows * HEADS * D);
    let (w_bits, wk) = lcg.row(cells * D);
    let (e_bits, ek) = lcg.row(cells * D);
    let sinks: Vec<f32> = (0..HEADS).map(|h| (h as f32 * 0.37).sin() * 3.0).collect();

    let (ratio, top_k) = match pool {
        Pool::None => (0, 0),
        Pool::Dense(r) => (r, 0),
        Pool::Selected(r, k) => (r, k),
    };
    let mut sel: Vec<i32> = Vec::new();
    if let Pool::Selected(..) = pool {
        for (r, &pos) in positions.iter().enumerate() {
            let visible = ((pos + 1) as usize) / ratio;
            for n in 0..top_k {
                let id = match (r + n) % 9 {
                    0 => (visible + n % 7) as i32,
                    _ => ((n * 37 + r * 11) % (visible.max(1) + 3)) as i32,
                };
                sel.push(id);
            }
            if r % 8 == 3 || r % 8 == 5 {
                for n in top_k / 2..top_k {
                    sel[r * top_k + n] = if n == top_k / 2 { -1 } else { 0 };
                }
            }
        }
    }

    // Sharper than 1/sqrt(d), so a few keys carry a head.
    let scale = 4.0 / (D as f32).sqrt();
    let mut want = vec![0f32; rows * HEADS * D];
    for r in 0..if bench { 0 } else { rows } {
        let pos = positions[r] as usize;
        let req = reqs[r] as usize;
        let mut keys: Vec<&[f32]> = Vec::new();
        for p in pos + 1 - (pos + 1).min(WINDOW)..=pos {
            let s = slot(req, p);
            keys.push(&wk[s * D..(s + 1) * D]);
        }
        if let Some(visible) = (pos + 1).checked_div(ratio) {
            let ids: Vec<i64> = match pool {
                Pool::Selected(..) if visible > 0 => sel[r * top_k..(r + 1) * top_k]
                    .iter()
                    .take_while(|&&c| c >= 0)
                    .map(|&c| i64::from(c))
                    .collect(),
                Pool::Selected(..) => Vec::new(),
                _ => (0..visible as i64).collect(),
            };
            for c in ids {
                if c >= 0 && (c as usize) < visible {
                    let s = slot(req, (c as usize + 1) * ratio - 1);
                    keys.push(&ek[s * D..(s + 1) * D]);
                }
            }
        }
        for h in 0..HEADS {
            let qh = &q[(r * HEADS + h) * D..(r * HEADS + h + 1) * D];
            let scores: Vec<f32> = keys
                .iter()
                .map(|k| qh.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale)
                .collect();
            let top = scores.iter().copied().fold(sinks[h], f32::max);
            let ps: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let den = ps.iter().sum::<f32>() + (sinks[h] - top).exp();
            let out = &mut want[(r * HEADS + h) * D..(r * HEADS + h + 1) * D];
            for (p, k) in ps.iter().zip(&keys) {
                for (o, v) in out.iter_mut().zip(k.iter()) {
                    *o += p * v / den;
                }
            }
        }
    }

    let mut gpu = Gpu::open();
    let q_t = Tensor::new(
        gpu.up(&q_bits),
        rows as u32,
        (HEADS * D) as u32,
        Dtype::Bf16,
    );
    let wkeys = gpu.up(&w_bits);
    let ekeys = gpu.up(&e_bits);
    let pos_t = Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&reqs), rows as u32, 1, Dtype::I32);
    let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
    let window_pool = pool_of(&mut gpu, wkeys, cells, &pages, &indptr);
    let entries = pool_of(&mut gpu, ekeys, cells, &pages, &indptr);
    let sel_t = if sel.is_empty() {
        None
    } else {
        Some((
            Tensor::new(gpu.up(&sel), rows as u32, top_k as u32, Dtype::I32),
            top_k as u32,
        ))
    };
    let o_at = gpu.zeros(rows * HEADS * D * 2);
    let (flush, po_at, plse_at) = if bench {
        (
            gpu.zeros(FLUSH),
            gpu.zeros(rows * HEADS * D * 2),
            gpu.zeros(rows * HEADS * 4),
        )
    } else {
        (0, 0, 0)
    };
    let pooled = (ratio > 0).then_some(dsv4::Pooled {
        entries: &entries,
        ratio: ratio as u32,
        selection: sel_t,
    });
    let fire = || {
        let mut o = Tensor::new(o_at, rows as u32, (HEADS * D) as u32, Dtype::Bf16);
        dsv4::decode(
            &gpu.ctx(),
            q_t,
            pos_t,
            req_t,
            &window_pool,
            WINDOW as u32,
            pooled,
            sink_t,
            HEADS as u32,
            D as u32,
            scale,
            &mut o,
        )
        .expect("the fused attention fires");
    };
    if bench {
        let fused = time(&gpu, flush, &fire);
        let sink_only = time(&gpu, flush, &|| {
            let mut po = Tensor::new(po_at, rows as u32, (HEADS * D) as u32, Dtype::Bf16);
            let plse = Tensor::new(plse_at, rows as u32, HEADS as u32, Dtype::F32);
            kernels_cuda::attn::sink(&gpu.ctx(), &mut po, plse, sink_t, D as u32).expect("sink");
        });
        println!("| sink | {rows} rows | {sink_only:.1} us |");
        let five = time(&gpu, flush, &|| {
            for _ in 0..5 {
                fire();
            }
        });
        println!("| dsv4 fused x5 | {rows} rows | ratio {ratio} | top_k {top_k} | {five:.1} us |");
        // The pool walk alone on the tensor cores, as the four launches run it.
        let pool_only = if ratio > 0 {
            let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
                pool_mma: true,
                ..kernels_cuda::tuning::Tuning::DEFAULT
            });
            time(&gpu, flush, &|| {
                let mut po = Tensor::new(po_at, rows as u32, (HEADS * D) as u32, Dtype::Bf16);
                let mut plse = Tensor::new(plse_at, rows as u32, HEADS as u32, Dtype::F32);
                match sel_t {
                    Some((sel, k)) => kernels_cuda::attn::pool::attention_lse_selected(
                        &ctx,
                        q_t,
                        pos_t,
                        req_t,
                        sel,
                        &entries,
                        ratio as u32,
                        k,
                        HEADS as u32,
                        D as u32,
                        scale,
                        &mut po,
                        &mut plse,
                    ),
                    None => kernels_cuda::attn::pool::attention_lse(
                        &ctx,
                        q_t,
                        pos_t,
                        req_t,
                        &entries,
                        ratio as u32,
                        HEADS as u32,
                        D as u32,
                        scale,
                        &mut po,
                        &mut plse,
                    ),
                }
                .expect("the pool walk fires");
            })
        } else {
            0.0
        };
        println!(
            "| pool_lse_mma | {rows} rows | ratio {ratio} | top_k {top_k} | {pool_only:.1} us |"
        );
        println!("| dsv4 fused | {rows} rows | ratio {ratio} | top_k {top_k} | {fused:.1} us |");
        return;
    }
    fire();
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, rows * HEADS * D);

    let label = format!("{rows} rows, ratio {ratio}, top_k {top_k}");
    let mut worst = 0f32;
    let mut at = 0;
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let err = (from_bf16(g) - w).abs() / w.abs().max(0.05);
        if err > worst {
            worst = err;
            at = i;
        }
    }
    assert!(
        worst <= 4e-2,
        "{label}: row {} head {} col {}: {} vs {} ({worst} relative)",
        at / (HEADS * D),
        at / D % HEADS,
        at % D,
        from_bf16(got[at]),
        want[at]
    );
}

#[test]
fn the_fused_dsv4_attention_lands_the_window_the_pool_and_the_sinks() {
    check(8, Pool::Selected(4, 512), false);
    check(8, Pool::Dense(128), false);
    check(8, Pool::None, false);
    check(24, Pool::Selected(4, 512), false);
    check(40, Pool::Dense(4), false);
    check(80, Pool::Selected(4, 64), false);
}

/// The fused launch's device time at DeepSeek-V4-Flash's decode widths.
#[test]
#[ignore = "a timing report"]
fn the_fused_dsv4_attention_reports_its_time() {
    for rows in [1, 8, 32, 64, 128] {
        check(rows, Pool::Selected(4, 512), true);
        check(rows, Pool::Dense(128), true);
        check(rows, Pool::None, true);
    }
}

const FLUSH: usize = 256 << 20;

/// The median device time of `fire` captured in a graph (so the host's
/// launch cost is out of it), each replay after the L2 is flushed.
fn time(gpu: &Gpu, flush: u64, fire: &dyn Fn()) -> f64 {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let stream: rt::cudaStream_t = gpu.ctx().stream().cast();
    fire();
    gpu.sync();
    let mut times = Vec::new();
    unsafe {
        let mut graph: rt::cudaGraph_t = core::ptr::null_mut();
        let mut exec: rt::cudaGraphExec_t = core::ptr::null_mut();
        rt::cudaStreamBeginCapture(
            stream,
            rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
        );
        fire();
        rt::cudaStreamEndCapture(stream, &raw mut graph);
        rt::cudaGraphInstantiate(&raw mut exec, graph, 0);
        let (mut a, mut b): (rt::cudaEvent_t, rt::cudaEvent_t) =
            (core::ptr::null_mut(), core::ptr::null_mut());
        rt::cudaEventCreate(&raw mut a);
        rt::cudaEventCreate(&raw mut b);
        let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
        rt::cudaMallocHost(&raw mut host, FLUSH);
        rt::cudaMemsetAsync(flush as *mut core::ffi::c_void, 1, FLUSH, stream);
        for _ in 0..21 {
            // `DSV4_TIME_WARM` leaves the L2 as the last replay left it.
            if std::env::var_os("DSV4_TIME_WARM").is_none() {
                rt::cudaMemcpyAsync(
                    host,
                    flush as *const core::ffi::c_void,
                    FLUSH,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream,
                );
            }
            rt::cudaEventRecord(a, stream);
            rt::cudaGraphLaunch(exec, stream);
            rt::cudaEventRecord(b, stream);
            rt::cudaEventSynchronize(b);
            let mut ms = 0f32;
            rt::cudaEventElapsedTime(&raw mut ms, a, b);
            times.push(f64::from(ms) * 1e3);
        }
        rt::cudaEventDestroy(a);
        rt::cudaEventDestroy(b);
        rt::cudaFreeHost(host);
        rt::cudaGraphExecDestroy(exec);
        rt::cudaGraphDestroy(graph);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}
