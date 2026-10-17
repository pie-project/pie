#![cfg(feature = "cuda")]

//! `attn::dsv4_prefill`: each row's 128-key window and its compressed
//! entries (all visible, or an index's selection) in one softmax with the
//! sinks, against the same arithmetic in f64. Rows pair up within a request
//! and across a request's end, and the fire ends on a lone row.

use crate::common::host::rows_read;
use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::dsv4;
use kernels_cuda::attn::dsv4::Rows;
use kernels_cuda::attn::dsv4_prefill::{self, Pooled};
use kernels_cuda::tensor::{KvPool, Tensor};

const PAGE: usize = 64;
const HEADS: usize = 64;
const D: usize = 512;

fn pool_of(gpu: &mut Gpu, keys: u64, cells: usize, pages: &[u32], indptr: &[u32]) -> KvPool {
    KvPool {
        keys: Tensor::new(keys, cells as u32, D as u32, Dtype::Bf16),
        values: Tensor::ABSENT,
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
        max_pages_per_request: 32,
        pages_in_batch: pages.len() as i32,
    }
}

/// Two requests' pages interleaved: request r's page i is pool page
/// 2 i + r, `per` pages each.
fn paging(per: usize) -> (Vec<u32>, Vec<u32>) {
    let mut pages = Vec::new();
    for r in 0..2u32 {
        pages.extend((0..per as u32).map(|i| 2 * i + r));
    }
    (pages, vec![0, per as u32, 2 * per as u32])
}

/// `src`'s cells written as DeepSeek's quantized rows of format `fmt`
/// through the append: the u8 pool, and what its rows read as.
fn rows_plane(
    gpu: &mut Gpu,
    src: u64,
    cells: usize,
    pages: &[u32],
    indptr: &[u32],
    fmt: Rows,
) -> (KvPool, Vec<f32>) {
    let bytes = cells * fmt.bytes() as usize;
    let plane = gpu.zeros(bytes);
    let pool = KvPool {
        keys: Tensor::new(plane, cells as u32, fmt.bytes(), Dtype::U8),
        values: Tensor::new(plane, cells as u32, fmt.bytes(), Dtype::U8),
        seq_stride: i64::from(fmt.bytes()),
        ..pool_of(gpu, plane, cells, pages, indptr)
    };
    let at_page: Vec<u32> = (0..cells as u32).map(|c| c / PAGE as u32).collect();
    let at_off: Vec<u32> = (0..cells as u32).map(|c| c % PAGE as u32).collect();
    let at_page = Tensor::new(gpu.up(&at_page), cells as u32, 1, Dtype::U32);
    let at_off = Tensor::new(gpu.up(&at_off), cells as u32, 1, Dtype::U32);
    let rows_t = Tensor::new(src, cells as u32, D as u32, Dtype::Bf16);
    dsv4::append(&gpu.ctx(), rows_t, &pool, at_page, at_off).expect("the fp8 append fires");
    gpu.sync();
    let raw: Vec<u8> = gpu.down(plane, bytes);
    let read = rows_read(&raw, cells, PAGE, fmt);
    (pool, read)
}

/// With `graph_safe` the pass a CUDA graph records (every quantized row
/// widened as its tiles land, nothing staged).
fn check(
    ratio: usize,
    top_k: Option<usize>,
    fmts: Option<(Rows, Rows)>,
    floored: bool,
    graph_safe: bool,
) {
    let fp8 = fmts.is_some();
    let per = 18;
    let cells = 2 * per * PAGE;
    let (pages, indptr) = paging(per);
    let cell = |req: usize, pos: usize| pages[req * per + pos / PAGE] as usize * PAGE + pos % PAGE;
    // Request 0's rows 600..608, request 1's 1000..1005: the pair (607,
    // 1000) straddles the two, and the last row stands alone.
    let rows: Vec<(usize, usize)> = (600..608)
        .map(|p| (0, p))
        .chain((1000..1005).map(|p| (1, p)))
        .collect();
    let n = rows.len();
    let mut lcg = Lcg::seeded(0xd5f4 + ratio as u64);
    let (q_bits, q) = lcg.row(n * HEADS * D);
    let (w_bits, mut w) = lcg.row(cells * D);
    let (e_bits, mut e) = lcg.row(cells * D);
    let sinks: Vec<f32> = (0..HEADS).map(|h| (h as f32 - 32.0) / 16.0).collect();
    // Each row's selection: ascending ids it sees, neighbours sharing most.
    let mut sel = vec![-1i32; n * top_k.unwrap_or(0)];
    if let Some(k) = top_k {
        for (r, &(_, pos)) in rows.iter().enumerate() {
            let visible = (pos + 1) / ratio;
            let mut ids: Vec<i32> = (0..visible as i32)
                .filter(|id| (id * 7 + (r as i32 / 3)) % 3 != 0)
                .collect();
            ids.truncate(k - (r % 4));
            for (i, id) in ids.iter().enumerate() {
                sel[r * k + i] = *id;
            }
        }
    }

    let mut gpu = Gpu::open();
    let positions: Vec<i32> = rows.iter().map(|&(_, p)| p as i32).collect();
    let requests: Vec<i32> = rows.iter().map(|&(r, _)| r as i32).collect();
    // A replay floor a row: below its window, inside it, or at the row.
    let floors: Vec<i32> = rows
        .iter()
        .enumerate()
        .map(|(r, &(_, p))| {
            if floored {
                (p as i32 - (r as i32 * 37) % 180).max(0)
            } else {
                0
            }
        })
        .collect();
    let q_t = Tensor::new(gpu.up(&q_bits), n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let pos_t = Tensor::new(gpu.up(&positions), n as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&requests), n as u32, 1, Dtype::I32);
    let floor_t = floored.then(|| Tensor::new(gpu.up(&floors), n as u32, 1, Dtype::I32));
    let w_at = gpu.up(&w_bits);
    let win = pool_of(&mut gpu, w_at, cells, &pages, &indptr);
    let e_at = gpu.up(&e_bits);
    let ent = pool_of(&mut gpu, e_at, cells, &pages, &indptr);
    let (win, ent) = if let Some((wf, pf)) = fmts {
        let (wp, wr) = rows_plane(&mut gpu, w_at, cells, &pages, &indptr, wf);
        let (ep, er) = rows_plane(&mut gpu, e_at, cells, &pages, &indptr, pf);
        (w, e) = (wr, er);
        (wp, ep)
    } else {
        (win, ent)
    };
    let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
    let sel_t = top_k.map(|k| {
        (
            Tensor::new(gpu.up(&sel), n as u32, k as u32, Dtype::I32),
            k as u32,
        )
    });
    let o_at = gpu.zeros(n * HEADS * D * 2);
    let mut o = Tensor::new(o_at, n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let scale = 1.0 / (D as f32).sqrt();
    let pooled = Pooled {
        entries: &ent,
        ratio: ratio as u32,
        selection: sel_t,
    };
    assert!(
        dsv4_prefill::takes(HEADS as u32, D as u32, Some(128), &win, &pooled),
        "the kernel takes DeepSeek-V4-Flash's shape"
    );
    let ctx = gpu.ctx();
    dsv4_prefill::prefill(
        &ctx,
        "attention.pool_attention",
        q_t,
        pos_t,
        req_t,
        floor_t,
        graph_safe,
        &win,
        &pooled,
        sink_t,
        scale,
        None,
        &mut o,
    )
    .expect("the prefill attention fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, n * HEADS * D);

    // The output's tail turned back with it lands what turning it after does.
    {
        let turn = kernels_cuda::attn::dsv4::Turn {
            dim: 64,
            theta: 10000.0,
            yarn: Some(kernels_cuda::elemwise::rope::Yarn {
                factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 65536,
            }),
        };
        let o2_at = gpu.zeros(n * HEADS * D * 2);
        let mut o2 = Tensor::new(o2_at, o.rows, o.width, Dtype::Bf16);
        dsv4_prefill::prefill(
            &ctx,
            "attention.pool_attention",
            q_t,
            pos_t,
            req_t,
            floor_t,
            graph_safe,
            &win,
            &pooled,
            sink_t,
            scale,
            Some(turn),
            &mut o2,
        )
        .expect("the turned prefill attention fires");
        kernels_cuda::elemwise::rope::partial_last(
            &ctx, &mut o, pos_t, 64, D as u32, 10000.0, true, true, turn.yarn,
        )
        .expect("the turn after fires");
        gpu.sync();
        let after: Vec<u16> = gpu.down(o_at, n * HEADS * D);
        let with: Vec<u16> = gpu.down(o2_at, n * HEADS * D);
        for (i, (&a, &b)) in with.iter().zip(&after).enumerate() {
            let (a, b) = (from_bf16(a), from_bf16(b));
            assert!(
                (a - b).abs() <= b.abs() / 64.0 + 1e-2,
                "ratio {ratio} fp8 {fp8}: turned element {i} is {a} against {b}"
            );
        }
    }

    let mut worst = 0f64;
    for (r, &(req, pos)) in rows.iter().enumerate() {
        // The keys this row reads: its window, then its entries.
        let mut keys: Vec<&[f32]> = Vec::new();
        for p in pos.saturating_sub(127).max(floors[r] as usize)..=pos {
            let c = cell(req, p);
            keys.push(&w[c * D..(c + 1) * D]);
        }
        let visible = (pos + 1) / ratio;
        let ids: Vec<usize> = match top_k {
            Some(k) => sel[r * k..(r + 1) * k]
                .iter()
                .take_while(|&&id| id >= 0)
                .map(|&id| id as usize)
                .collect(),
            None => (0..visible).collect(),
        };
        for id in ids {
            let c = cell(req, (id + 1) * ratio - 1);
            keys.push(&e[c * D..(c + 1) * D]);
        }
        for h in 0..HEADS {
            let qh = &q[(r * HEADS + h) * D..(r * HEADS + h + 1) * D];
            let s: Vec<f64> = keys
                .iter()
                .map(|k| {
                    qh.iter()
                        .zip(k.iter())
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            let lse = m + z.ln();
            let sinked = 1.0 / (1.0 + (f64::from(sinks[h]) - lse).exp());
            for d in 0..D {
                let want: f64 = s
                    .iter()
                    .zip(&keys)
                    .map(|(x, k)| (x - m).exp() / z * f64::from(k[d]))
                    .sum::<f64>()
                    * sinked;
                let g = f64::from(from_bf16(got[(r * HEADS + h) * D + d]));
                worst = worst.max((g - want).abs() / want.abs().max(0.02));
            }
        }
    }
    assert!(
        worst <= 4e-2,
        "ratio {ratio}, top_k {top_k:?}, fp8 {fp8}, floored {floored}, graph-safe {graph_safe}: an output element differs by {worst} relative"
    );
}

#[test]
fn deepseeks_prefill_attention_lands_what_its_four_ops_land() {
    let (ds, v41) = (Some((Rows::Ds, Rows::Ds)), Some((Rows::Mx8, Rows::Nv4)));
    // (ratio, top_k, rows, floored, graph_safe): a CSA layer (ratio 4, 64
    // picks; 512 walk several tiles), an HCA layer (ratio 128, every entry),
    // every entry at ratio 4 (a page's sixteen as one box), V4.1's MXFP8
    // window over NVFP4 entries at ratios 2 and 1; windows held above replay
    // floors; and the pass a graph records.
    for (ratio, top_k, fmts, floored, graph_safe) in [
        (4, Some(64), None, false, false),
        (4, Some(64), ds, false, false),
        (4, Some(512), None, false, false),
        (128, None, None, false, false),
        (128, None, ds, false, false),
        (4, None, None, false, false),
        (2, Some(512), v41, false, false),
        (1, Some(512), v41, false, false),
        (2, None, v41, false, false),
        (2, Some(512), v41, true, false),
        (4, Some(64), None, true, false),
        (4, Some(64), ds, false, true),
        (128, None, ds, false, true),
        (2, Some(512), v41, true, true),
    ] {
        check(ratio, top_k, fmts, floored, graph_safe);
    }
}

/// `values` written over the device bytes at `at`.
fn write<T: Copy>(at: u64, values: &[T]) {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let _legacy = crate::common::legacy();
    // SAFETY: `at` holds at least `values`' bytes, allocated by the test.
    let code = unsafe {
        rt::cudaMemcpy(
            at as usize as *mut core::ffi::c_void,
            values.as_ptr().cast(),
            core::mem::size_of_val(values),
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    assert_eq!(code, rt::cudaError::cudaSuccess, "cudaMemcpy H2D");
}

/// The pass as a CUDA graph records it (`graph_safe`), replayed against a
/// page table that has grown since: the replay lands what a fresh launch
/// over the grown table does, bit for bit, over V4's rows and V4.1's. A
/// staged pass sizes its planes by the table it was recorded over, which
/// the grown table outruns.
#[test]
fn a_recorded_pass_reads_the_page_tables_its_replay_states() {
    if !kernels_cuda::linear::tc5::supported() {
        return;
    }
    for (fmts, ratio) in [((Rows::Ds, Rows::Ds), 4usize), ((Rows::Mx8, Rows::Nv4), 4)] {
        let most = 40usize;
        let cells = most * PAGE;
        // The request's pages, scattered over the plane.
        let pages: Vec<u32> = (0..most as u32).map(|i| (i * 7) % most as u32).collect();
        let indptr_full = vec![0u32, most as u32];
        let mut lcg = Lcg::seeded(0x9a);
        let n = 6usize;
        let (q_bits, _) = lcg.row(n * HEADS * D);
        let (w_bits, _) = lcg.row(cells * D);
        let (e_bits, _) = lcg.row(cells * D);
        let mut gpu = Gpu::open();
        let w_at = gpu.up(&w_bits);
        let e_at = gpu.up(&e_bits);
        let (win, _) = rows_plane(&mut gpu, w_at, cells, &pages, &indptr_full, fmts.0);
        let (ent, _) = rows_plane(&mut gpu, e_at, cells, &pages, &indptr_full, fmts.1);
        // The tables the pass reads, rewritten between replays.
        let table_at = gpu.zeros(most * 4);
        let indptr_at = gpu.zeros(2 * 4);
        let pos_at = gpu.zeros(n * 4);
        let req_t = Tensor::new(gpu.up(&vec![0i32; n]), n as u32, 1, Dtype::I32);
        let q_t = Tensor::new(gpu.up(&q_bits), n as u32, (HEADS * D) as u32, Dtype::Bf16);
        let pos_t = Tensor::new(pos_at, n as u32, 1, Dtype::I32);
        let sinks: Vec<f32> = (0..HEADS).map(|h| (h as f32 - 32.0) / 16.0).collect();
        let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
        let o_at = gpu.zeros(n * HEADS * D * 2);
        let fresh_at = gpu.zeros(n * HEADS * D * 2);
        let scale = 1.0 / (D as f32).sqrt();
        // A table of `held` pages and rows at its last positions.
        let table = |held: usize| {
            write(table_at, &pages[..held]);
            write(indptr_at, &[0u32, held as u32]);
            let end = held * PAGE - 5;
            let positions: Vec<i32> = (0..n).map(|r| (end - n + r) as i32).collect();
            write(pos_at, &positions);
            let view = |pool: &KvPool| KvPool {
                page_indices: Tensor::new(table_at, held as u32, 1, Dtype::U32),
                page_indptr: Tensor::new(indptr_at, 2, 1, Dtype::U32),
                pages_in_batch: held as i32,
                ..*pool
            };
            (view(&win), view(&ent))
        };
        let fire = |w: &KvPool, e: &KvPool, at: u64| {
            let pooled = Pooled {
                entries: e,
                ratio: ratio as u32,
                selection: None,
            };
            let mut o = Tensor::new(at, n as u32, (HEADS * D) as u32, Dtype::Bf16);
            dsv4_prefill::prefill(
                &gpu.ctx(),
                "attention.pool_attention",
                q_t,
                pos_t,
                req_t,
                None,
                true,
                w,
                &pooled,
                sink_t,
                scale,
                None,
                &mut o,
            )
            .expect("the pass fires");
        };
        let (w, e) = table(11);
        let exec = crate::common::captured(&gpu.ctx(), || fire(&w, &e, o_at));
        for held in [11usize, 23, most] {
            let (w, e) = table(held);
            crate::common::replay(&gpu.ctx(), exec);
            fire(&w, &e, fresh_at);
            gpu.sync();
            let replayed: Vec<u16> = gpu.down(o_at, n * HEADS * D);
            let fresh: Vec<u16> = gpu.down(fresh_at, n * HEADS * D);
            assert!(
                fresh.iter().any(|&v| v != 0),
                "{fmts:?}: the pass lands something"
            );
            assert_eq!(
                replayed, fresh,
                "{fmts:?}: the pass recorded over 11 pages, replayed over {held}"
            );
        }
        crate::common::forget(exec);
    }
}

/// A layer with a window and no pool: the pass reads the window alone and
/// lands what the decode's one launch does, over bf16 rows and fp8 ones.
#[test]
fn a_window_only_layer_lands_its_window_alone() {
    let per = 18;
    let cells = 2 * per * PAGE;
    let (pages, indptr) = paging(per);
    // Rows near a request's start (fewer than the window) and past it.
    let rows: Vec<(usize, usize)> = (0..5)
        .map(|p| (0, p))
        .chain((600..608).map(|p| (0, p)))
        .chain((1000..1005).map(|p| (1, p)))
        .collect();
    let n = rows.len();
    let mut lcg = Lcg::seeded(0x0a11);
    let (q_bits, _) = lcg.row(n * HEADS * D);
    let (w_bits, _) = lcg.row(cells * D);
    let sinks: Vec<f32> = (0..HEADS).map(|h| (h as f32 - 32.0) / 16.0).collect();
    let mut gpu = Gpu::open();
    let positions: Vec<i32> = rows.iter().map(|&(_, p)| p as i32).collect();
    let requests: Vec<i32> = rows.iter().map(|&(r, _)| r as i32).collect();
    let q_t = Tensor::new(gpu.up(&q_bits), n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let pos_t = Tensor::new(gpu.up(&positions), n as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&requests), n as u32, 1, Dtype::I32);
    let w_at = gpu.up(&w_bits);
    let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
    let scale = 1.0 / (D as f32).sqrt();
    for fp8 in [false, true] {
        let win = if fp8 {
            rows_plane(&mut gpu, w_at, cells, &pages, &indptr, Rows::Ds).0
        } else {
            pool_of(&mut gpu, w_at, cells, &pages, &indptr)
        };
        assert!(dsv4_prefill::takes_window(
            HEADS as u32,
            D as u32,
            Some(128),
            &win
        ));
        let o_at = gpu.zeros(n * HEADS * D * 2);
        let want_at = gpu.zeros(n * HEADS * D * 2);
        let mut o = Tensor::new(o_at, n as u32, (HEADS * D) as u32, Dtype::Bf16);
        let mut want = Tensor::new(want_at, n as u32, (HEADS * D) as u32, Dtype::Bf16);
        let ctx = gpu.ctx();
        dsv4_prefill::window_only(
            &ctx,
            "attention.lse_sink",
            q_t,
            pos_t,
            req_t,
            false,
            &win,
            sink_t,
            scale,
            &mut o,
        )
        .expect("the window-only pass fires");
        dsv4::decode(
            &ctx,
            q_t,
            pos_t,
            req_t,
            None,
            &win,
            128,
            None,
            sink_t,
            HEADS as u32,
            D as u32,
            scale,
            None,
            &mut want,
        )
        .expect("the decode's launch fires");
        gpu.sync();
        let got: Vec<u16> = gpu.down(o_at, n * HEADS * D);
        let wanted: Vec<u16> = gpu.down(want_at, n * HEADS * D);
        for (i, (&a, &b)) in got.iter().zip(&wanted).enumerate() {
            let (a, b) = (from_bf16(a), from_bf16(b));
            assert!(
                (a - b).abs() <= b.abs() / 32.0 + 2e-2,
                "fp8 {fp8}: element {i} (row {}) is {a} against {b}",
                i / (HEADS * D)
            );
        }
    }
}
