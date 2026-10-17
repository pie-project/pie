#![cfg(feature = "cuda")]

//! DeepSeek-V4's attention at prefill widths in one tcgen05 pass: each row's
//! 128-key window and its compressed entries (every visible one, or an
//! index's ascending selection) in one softmax, the sinks over it, against
//! the same arithmetic in f64. Rows pair up within a request and across a
//! request's end, and the fire ends on a lone row; and with each row's
//! window held above a replay floor.

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

fn check(ratio: usize, top_k: Option<usize>) {
    check_with(ratio, top_k, true, None, false);
    // DeepSeek's fp8 rows, which only the block-a-row kernel reads.
    check_with(ratio, top_k, true, Some((Rows::Ds, Rows::Ds)), false);
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

/// An e2m1 code's value.
fn e2m1(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

/// `src`'s cells written as DeepSeek's fp8 rows through the fp8 append:
/// the u8 pool, and what its rows read as.
fn fp8_plane(
    gpu: &mut Gpu,
    src: u64,
    cells: usize,
    pages: &[u32],
    indptr: &[u32],
) -> (KvPool, Vec<f32>) {
    rows_plane(gpu, src, cells, pages, indptr, Rows::Ds)
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
    let row = fmt.bytes() as usize;
    let data = dsv4::DS_DATA as usize;
    let nope = dsv4::DS_NOPE as usize;
    let bytes = cells * row;
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
    let mut read = vec![0f32; cells * D];
    if fmt != Rows::Ds {
        // V4.1's rows: MXFP8 (a ue8m0 a 32) or NVFP4 (an e4m3 a 16).
        let (data, scale) = if fmt == Rows::Mx8 {
            (512, 16)
        } else {
            (256, 32)
        };
        for c in 0..cells {
            let base = c / PAGE * PAGE * row;
            let off = c % PAGE;
            let (b, sc) = (base + off * data, base + PAGE * data + off * scale);
            for d in 0..D {
                read[c * D + d] = if fmt == Rows::Mx8 {
                    e4m3(raw[b + d]) * 2f32.powi(i32::from(raw[sc + d / 32]) - 127)
                } else {
                    let byte = raw[b + d / 2];
                    e2m1(if d % 2 == 0 { byte & 0xf } else { byte >> 4 }) * e4m3(raw[sc + d / 16])
                };
            }
        }
        return (pool, read);
    }
    for c in 0..cells {
        let base = c / PAGE * PAGE * row;
        let off = c % PAGE;
        let b = &raw[base + off * data..base + (off + 1) * data];
        let sc = &raw[base + PAGE * data + off * 8..base + PAGE * data + off * 8 + 8];
        for d in 0..D {
            read[c * D + d] = if d < nope {
                e4m3(b[d]) * 2f32.powi(i32::from(sc[d / 64]) - 127)
            } else {
                let at = nope + (d - nope) * 2;
                from_bf16(u16::from_le_bytes([b[at], b[at + 1]]))
            };
        }
    }
    (pool, read)
}

fn check_with(
    ratio: usize,
    top_k: Option<usize>,
    sparse: bool,
    fmts: Option<(Rows, Rows)>,
    floored: bool,
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
                "ratio {ratio} sparse {sparse} fp8 {fp8}: turned element {i} is {a} against {b}"
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
        "ratio {ratio}, top_k {top_k:?}, sparse {sparse}, fp8 {fp8}, floored {floored}: an output element differs by {worst} relative"
    );
}

#[test]
fn deepseeks_prefill_attention_lands_what_its_four_ops_land() {
    // A CSA layer: ratio 4, a selection of 64 a row; an HCA layer: ratio
    // 128, every visible entry.
    check(4, Some(64));
    // Selections longer than a tile: the merge walks them over several.
    check(4, Some(512));
    check(128, None);
    // Every visible entry at ratio 4: a page's sixteen entries land as one box.
    check(4, None);
    // DeepSeek-V4.1's FP4 cache: an MXFP8 window over NVFP4 compressed rows,
    // at its ratios 2 (selected) and 1.
    let v41 = Some((Rows::Mx8, Rows::Nv4));
    check_with(2, Some(512), true, v41, false);
    check_with(1, Some(512), true, v41, false);
    check_with(2, None, true, v41, false);
    // Windows held above their rows' replay floors (V4.1's bounded replay).
    check_with(2, Some(512), true, v41, true);
    check_with(1, Some(512), true, v41, true);
    check_with(4, Some(64), true, None, true);
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
            fp8_plane(&mut gpu, w_at, cells, &pages, &indptr).0
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

/// DeepSeek-V4-Flash at a prefill's width: 2048 rows of one 4096-position
/// request, each reading its 128-key window and, at a CSA layer, 512
/// selected entries, or at an HCA layer every visible entry; the output's
/// tail turned back as the model's attention does.
#[test]
#[ignore = "a measurement; run it by name under the GPU lock"]
fn deepseeks_prefill_attention_paces_a_prefill() {
    for (ratio, k) in [(4usize, Some(512usize)), (128, None)] {
        pace(ratio, k);
    }
}

fn pace(ratio: usize, top_k: Option<usize>) {
    let (n, span) = (2048usize, 4096usize);
    let k = top_k.unwrap_or(0);
    let per = span / PAGE;
    let (pages, indptr) = paging(per);
    let cells = 2 * per * PAGE;
    let mut lcg = Lcg::seeded(0x7a11);
    let (q_bits, _) = lcg.row(n * HEADS * D);
    let (w_bits, _) = lcg.row(cells * D);
    let positions: Vec<i32> = (0..n as i32).map(|r| (span - n) as i32 + r).collect();
    let requests = vec![0i32; n];
    let mut sel = vec![-1i32; n * k.max(1)];
    for (r, &pos) in positions.iter().enumerate().filter(|_| k > 0) {
        let visible = (pos as usize + 1) / ratio;
        // The same 512 ids a row but for one in sixteen, as neighbouring
        // rows' indexes mostly agree.
        let ids: Vec<i32> = (0..visible as i32)
            .filter(|id| {
                (id * 2_654_435_761u32 as i32).rem_euclid(1024)
                    < (k as i32 * 1024 / visible as i32).min(1024)
                    || (*id as usize + r).is_multiple_of(16)
            })
            .take(k)
            .collect();
        sel[r * k..r * k + ids.len()].copy_from_slice(&ids);
    }
    let mut gpu = Gpu::open();
    let q_t = Tensor::new(gpu.up(&q_bits), n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let pos_t = Tensor::new(gpu.up(&positions), n as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&requests), n as u32, 1, Dtype::I32);
    let w_at = gpu.up(&w_bits);
    let win = pool_of(&mut gpu, w_at, cells, &pages, &indptr);
    let (win8, _) = fp8_plane(&mut gpu, w_at, cells, &pages, &indptr);
    let sinks = vec![0f32; HEADS];
    let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
    let sel_t = Tensor::new(gpu.up(&sel), n as u32, k.max(1) as u32, Dtype::I32);
    let o_at = gpu.zeros(n * HEADS * D * 2);
    let mut o = Tensor::new(o_at, n as u32, (HEADS * D) as u32, Dtype::Bf16);
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
    for fp8 in [false, true] {
        let win = if fp8 { win8 } else { win };
        let pooled = Pooled {
            entries: &win,
            ratio: ratio as u32,
            selection: top_k.map(|k| (sel_t, k as u32)),
        };
        let ctx = gpu.ctx();
        let mut fire = || {
            dsv4_prefill::prefill(
                &ctx,
                "attention.pool_attention",
                q_t,
                pos_t,
                req_t,
                None,
                &win,
                &pooled,
                sink_t,
                0.044,
                Some(turn),
                &mut o,
            )
            .expect("the prefill attention fires");
        };
        fire();
        gpu.sync();
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            fire();
        }
        gpu.sync();
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
        let entries: usize = positions
            .iter()
            .map(|&p| top_k.unwrap_or(usize::MAX).min((p as usize + 1) / ratio))
            .sum();
        let flops = ((n * 128 + entries) * HEADS * D * 4) as f64;
        println!(
            "| {n} rows, ratio {ratio}, window + {} entries a row, fp8 {fp8} | {us:.0} us | {:.1} TFLOP/s useful |",
            entries / n,
            flops / us / 1e6
        );
    }
}
