#![cfg(feature = "cuda")]

//! DeepSeek-V4's attention at decode widths in one launch
//! (`attn::dsv4::decode`): a row's 128-position window of the shared
//! key/value plane, its compressed rows (all it sees, or an indexer's picks
//! with ids past the visible ones and an early end), folded together with
//! the sinks, against an f32 walk over the same rows; at row counts that
//! split a row's keys over clusters of 8, 4, 2 and 1 blocks. The planes are
//! bf16, or DeepSeek's fp8 rows as `attn::dsv4::append` writes them, read
//! back against what the bytes say (and the bytes against the bf16 rows).

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::dsv4;
use kernels_cuda::tensor::{KvPool, Tensor};

const PAGE: usize = 64;
const HEADS: usize = 64;
const D: usize = 512;
const WINDOW: usize = 128;

fn pool_of(
    gpu: &mut Gpu,
    keys: u64,
    cells: usize,
    pages: &[u32],
    indptr: &[u32],
    fp8: bool,
) -> KvPool {
    let (width, dtype) = if fp8 {
        (dsv4::DS_ROW, Dtype::U8)
    } else {
        (D as u32, Dtype::Bf16)
    };
    KvPool {
        keys: Tensor::new(keys, cells as u32, width, dtype),
        values: Tensor::new(keys, cells as u32, width, dtype),
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
        seq_stride: i64::from(width),
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

fn check(rows: usize, pool: Pool, bench: bool, fp8: bool) {
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
    let mut gpu = Gpu::open();
    let wkeys = gpu.up(&w_bits);
    let ekeys = gpu.up(&e_bits);
    let window_bf16 = pool_of(&mut gpu, wkeys, cells, &pages, &indptr, false);
    let entries_bf16 = pool_of(&mut gpu, ekeys, cells, &pages, &indptr, false);
    let (window_pool, entries, wk, ek) = if fp8 {
        // Every cell written through the fp8 append, then read back.
        let at_page: Vec<u32> = (0..cells as u32).map(|c| c / PAGE as u32).collect();
        let at_off: Vec<u32> = (0..cells as u32).map(|c| c % PAGE as u32).collect();
        let at_page = Tensor::new(gpu.up(&at_page), cells as u32, 1, Dtype::U32);
        let at_off = Tensor::new(gpu.up(&at_off), cells as u32, 1, Dtype::U32);
        let bytes = cells * dsv4::DS_ROW as usize;
        let mut planes = Vec::new();
        for (src, exact) in [(wkeys, &wk), (ekeys, &ek)] {
            let plane = gpu.zeros(bytes);
            let pool = pool_of(&mut gpu, plane, cells, &pages, &indptr, true);
            let rows_t = Tensor::new(src, cells as u32, D as u32, Dtype::Bf16);
            dsv4::append(&gpu.ctx(), rows_t, &pool, at_page, at_off).expect("the fp8 append fires");
            gpu.sync();
            let raw: Vec<u8> = gpu.down(plane, bytes);
            let read = ds_read(&raw, cells);
            for (i, (&got, &want)) in read.iter().zip(exact.iter()).enumerate() {
                // A code is within half its step of the value: 1/16 of it,
                // or half the subnormal step under the tile's scale.
                let tile_max = exact[i / 64 * 64..i / 64 * 64 + 64]
                    .iter()
                    .fold(1e-4f32, |m, v| m.max(v.abs()));
                let bound = if i % D >= dsv4::DS_NOPE as usize {
                    0.0
                } else {
                    want.abs() / 16.0 + tile_max / 448.0 * 2.0 * 2f32.powi(-9)
                };
                assert!(
                    (got - want).abs() <= bound + 1e-6,
                    "cell {} dim {}: the fp8 row reads {got} for {want}",
                    i / D,
                    i % D
                );
            }
            planes.push((pool, read));
        }
        let (e_pool, e_read) = planes.pop().expect("two planes");
        let (w_pool, w_read) = planes.pop().expect("two planes");
        (w_pool, e_pool, w_read, e_read)
    } else {
        (window_bf16, entries_bf16, wk, ek)
    };
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

    let q_t = Tensor::new(
        gpu.up(&q_bits),
        rows as u32,
        (HEADS * D) as u32,
        Dtype::Bf16,
    );
    let pos_t = Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&reqs), rows as u32, 1, Dtype::I32);
    let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
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
            None,
            &mut o,
        )
        .expect("the fused attention fires");
    };
    if bench {
        let fused = time(&gpu, flush, &fire);
        // The pool walk alone on the tensor cores, as the four launches run it.
        let pool_only = if ratio > 0 && !fp8 {
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
                        &entries_bf16,
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
                        &entries_bf16,
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
        println!(
            "| dsv4 fused | {rows} rows | ratio {ratio} | top_k {top_k} | fp8 {fp8} | {fused:.1} us |"
        );
        return;
    }
    fire();
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, rows * HEADS * D);

    let label = format!("{rows} rows, ratio {ratio}, top_k {top_k}, fp8 {fp8}");
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

    // The output's tail turned back with it lands what turning it after does
    // (`o_at` holds the unturned output; turned in place here).
    let turn = dsv4::Turn {
        dim: 64,
        theta: 10000.0,
        yarn: Some(kernels_cuda::elemwise::rope::Yarn {
            factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 65536,
        }),
    };
    let o2_at = gpu.zeros(rows * HEADS * D * 2);
    let mut o2 = Tensor::new(o2_at, rows as u32, (HEADS * D) as u32, Dtype::Bf16);
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
        Some(turn),
        &mut o2,
    )
    .expect("the turned attention fires");
    let mut o = Tensor::new(o_at, rows as u32, (HEADS * D) as u32, Dtype::Bf16);
    kernels_cuda::elemwise::rope::partial_last(
        &gpu.ctx(),
        &mut o,
        pos_t,
        64,
        D as u32,
        10000.0,
        true,
        true,
        turn.yarn,
    )
    .expect("the turn after fires");
    gpu.sync();
    let after: Vec<u16> = gpu.down(o_at, rows * HEADS * D);
    let with: Vec<u16> = gpu.down(o2_at, rows * HEADS * D);
    for (i, (&a, &b)) in with.iter().zip(&after).enumerate() {
        let (a, b) = (from_bf16(a), from_bf16(b));
        assert!(
            (a - b).abs() <= b.abs() / 64.0 + 1e-2,
            "{label}: turned element {i} is {a} against {b}"
        );
    }
}

#[test]
fn the_fused_dsv4_attention_lands_the_window_the_pool_and_the_sinks() {
    for fp8 in [false, true] {
        check(1, Pool::Selected(4, 512), false, fp8);
        check(8, Pool::Selected(4, 512), false, fp8);
        check(8, Pool::Dense(128), false, fp8);
        check(8, Pool::None, false, fp8);
        check(16, Pool::Dense(4), false, fp8);
        check(40, Pool::Selected(4, 64), false, fp8);
        // More keys than a chunk gathers at once.
        check(40, Pool::Dense(2), false, fp8);
    }
}

/// The fused launch's device time at DeepSeek-V4-Flash's decode widths.
#[test]
#[ignore = "a timing report"]
fn the_fused_dsv4_attention_reports_its_time() {
    for rows in [1, 8, 32, 64, 128] {
        for fp8 in [false, true] {
            check(rows, Pool::Selected(4, 512), true, fp8);
            check(rows, Pool::Dense(128), true, fp8);
            check(rows, Pool::None, true, fp8);
        }
    }
}

/// An e4m3 code's value.
fn e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((code >> 3) & 0xf);
    let man = f32::from(code & 7);
    sign * if exp == 0 {
        man / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// What DeepSeek's fp8 rows say, page by page: every cell's 512 values.
fn ds_read(raw: &[u8], cells: usize) -> Vec<f32> {
    let row = dsv4::DS_ROW as usize;
    let data = dsv4::DS_DATA as usize;
    let nope = dsv4::DS_NOPE as usize;
    let mut out = vec![0f32; cells * D];
    for cell in 0..cells {
        let page = cell / PAGE * PAGE * row;
        let off = cell % PAGE;
        let bytes = &raw[page + off * data..page + (off + 1) * data];
        let scales = &raw[page + PAGE * data + off * 8..page + PAGE * data + off * 8 + 8];
        assert_eq!(scales[7], 0, "cell {cell}: the scale row's pad");
        for d in 0..D {
            out[cell * D + d] = if d < nope {
                e4m3(bytes[d]) * 2f32.powi(i32::from(scales[d / 64]) - 127)
            } else {
                let at = nope + (d - nope) * 2;
                from_bf16(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
            };
        }
    }
    out
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
