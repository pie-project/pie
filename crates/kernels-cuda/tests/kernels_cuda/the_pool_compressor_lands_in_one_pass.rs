#![cfg(feature = "cuda")]

//! DeepSeek-V4's compressor at its boundaries in one pass
//! (`pool::compress`, under `kernel.elemwise.rows_vec`): the gathered entry,
//! its scaled RMS norm, its tail rotated at the boundary's rope position and
//! written into the entries pool, against the gather, the norm and rotation
//! and the append the unfused ops fire: at both ratios, the main latent (512
//! wide, bf16 and DeepSeek's fp8 pool) and the index keys (128), on a staged
//! window too. The pools land alike within rounding (the fp8 one as its rows
//! read).

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::{dsv4, pool};
use kernels_cuda::elemwise::rope::{self, Yarn};
use kernels_cuda::tensor::{KvPool, Tensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;
const REQUESTS: usize = 2;
const PAGES_PER: usize = 12;

fn pool_of(
    gpu: &mut Gpu,
    keys: u64,
    pages: &[u32],
    indptr: &[u32],
    width: u32,
    dtype: Dtype,
) -> KvPool {
    KvPool {
        keys: Tensor::new(keys, (pages.len() * PAGE) as u32, width, dtype),
        values: Tensor::new(keys, (pages.len() * PAGE) as u32, width, dtype),
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
        head_stride: i64::from(width),
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: PAGES_PER as i32,
        pages_in_batch: pages.len() as i32,
    }
}

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

/// The entries pool as it reads after one compressor pass, `one` choosing
/// the one pass.
fn pass(one: bool, ratio: usize, d: usize, fp8: bool, window: Option<(u32, u32)>) -> Vec<f32> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * d;
    let positions = PAGES_PER * PAGE;
    let total = REQUESTS * PAGES_PER;
    let cells = total * PAGE;
    let pages: Vec<u32> = (0..total).map(|i| ((7 * i) % total) as u32).collect();
    let indptr: Vec<u32> = (0..=REQUESTS).map(|r| (r * PAGES_PER) as u32).collect();
    let (mut bpos, mut breq, mut brope) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..REQUESTS {
        for p in 0..positions {
            let closes = (p + 1) % ratio == 0;
            bpos.push(if closes { p as i32 } else { -1 });
            breq.push(r as i32);
            brope.push(if closes { (p + 1 - ratio) as i32 } else { 0 });
        }
    }
    let rows = bpos.len();
    let mut lcg = Lcg::seeded(0xc0e + (ratio * d) as u64);
    let (kv_bits, _) = lcg.row(cells * width);
    let score_bits: Vec<u16> = (0..cells * width)
        .map(|_| to_bf16(lcg.unit() * 4.0))
        .collect();
    let ape: Vec<f32> = (0..ratio * width).map(|_| lcg.unit()).collect();
    let w: Vec<u16> = (0..d).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let mut gpu = Gpu::open();
    let kv = gpu.up(&kv_bits);
    let score = gpu.up(&score_bits);
    let state = pool_of(&mut gpu, kv, &pages, &indptr, width as u32, Dtype::Bf16);
    let out_row = if fp8 { dsv4::DS_ROW as usize } else { d * 2 };
    let out_at = gpu.zeros(cells * out_row);
    let out = if fp8 {
        pool_of(&mut gpu, out_at, &pages, &indptr, dsv4::DS_ROW, Dtype::U8)
    } else {
        pool_of(&mut gpu, out_at, &pages, &indptr, d as u32, Dtype::Bf16)
    };
    let ctx = gpu.ctx().with_tuning(Tuning {
        rows_vec: one,
        ..Tuning::DEFAULT
    });
    if let Some((live, base)) = window {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let bpos_t = Tensor::new(gpu.up(&bpos), rows as u32, 1, Dtype::I32);
    let breq_t = Tensor::new(gpu.up(&breq), rows as u32, 1, Dtype::I32);
    let brope_t = Tensor::new(gpu.up(&brope), rows as u32, 1, Dtype::I32);
    let ape_t = Tensor::new(gpu.up(&ape), ratio as u32, width as u32, Dtype::F32);
    let w_t = Tensor::new(gpu.up(&w), d as u32, 1, Dtype::Bf16);
    let state_kv = Tensor::new(kv, cells as u32, width as u32, Dtype::Bf16);
    let state_score = Tensor::new(score, cells as u32, width as u32, Dtype::Bf16);
    let yarn = Some(Yarn {
        factor: 16.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position: 65536,
    });
    let landed = pool::compress(
        &ctx,
        bpos_t,
        breq_t,
        &state,
        d as u32,
        ratio as u32,
        state_kv,
        state_score,
        ape_t,
        w_t,
        1e-6,
        brope_t,
        64,
        10000.0,
        true,
        yarn,
        &out,
    )
    .expect("the compressor fires");
    assert_eq!(landed, one, "the one pass lands where the profile says");
    if !landed {
        let entries = gpu.zeros(rows * d * 2);
        let latent = gpu.zeros(rows * d * 2);
        let mut e_t = Tensor::new(entries, rows as u32, d as u32, Dtype::Bf16);
        pool::gather(
            &ctx,
            bpos_t,
            breq_t,
            &state,
            d as u32,
            ratio as u32,
            state_kv,
            state_score,
            ape_t,
            &mut e_t,
        )
        .expect("the gather fires");
        let mut l_t = Tensor::new(latent, rows as u32, d as u32, Dtype::Bf16);
        rope::norm_rope_last(
            &ctx,
            e_t,
            Some(w_t),
            d as u32,
            1e-6,
            brope_t,
            64,
            10000.0,
            true,
            false,
            yarn,
            &mut l_t,
        )
        .expect("the norm and rotation fire");
        if fp8 {
            dsv4::store_entries(&ctx, l_t, bpos_t, breq_t, &out).expect("the fp8 store fires");
        } else {
            pool::kv_append(
                &ctx,
                l_t,
                bpos_t,
                breq_t,
                &out,
                Tensor::ABSENT,
                Tensor::ABSENT,
            )
            .expect("the append fires");
        }
    }
    gpu.sync();
    if !fp8 {
        let raw: Vec<u16> = gpu.down(out_at, cells * d);
        return raw.iter().map(|&b| from_bf16(b)).collect();
    }
    let raw: Vec<u8> = gpu.down(out_at, cells * out_row);
    let (row, data, nope) = (
        dsv4::DS_ROW as usize,
        dsv4::DS_DATA as usize,
        dsv4::DS_NOPE as usize,
    );
    let mut read = vec![0f32; cells * d];
    for c in 0..cells {
        let base = c / PAGE * PAGE * row;
        let off = c % PAGE;
        let b = &raw[base + off * data..base + (off + 1) * data];
        let sc = &raw[base + PAGE * data + off * 8..base + PAGE * data + off * 8 + 8];
        for k in 0..d {
            read[c * d + k] = if k < nope {
                e4m3(b[k]) * 2f32.powi(i32::from(sc[k / 64]) - 127)
            } else {
                let at = nope + (k - nope) * 2;
                from_bf16(u16::from_le_bytes([b[at], b[at + 1]]))
            };
        }
    }
    read
}

fn check(ratio: usize, d: usize, fp8: bool, window: Option<(u32, u32)>) {
    let old = pass(false, ratio, d, fp8, window);
    let new = pass(true, ratio, d, fp8, window);
    let tol = if fp8 { 8.0 } else { 64.0 };
    let mut written = 0;
    for (i, (&g, &w)) in new.iter().zip(&old).enumerate() {
        assert!(
            (g - w).abs() <= w.abs() / tol + 2e-2,
            "ratio {ratio} d {d} fp8 {fp8} {window:?}: element {i} reads {g} against {w}"
        );
        written += usize::from(w != 0.0);
    }
    assert!(written > 0, "entries are written");
}

#[test]
fn the_pool_compressor_lands_in_one_pass() {
    for (ratio, d) in [(4, 512), (128, 512), (4, 128)] {
        check(ratio, d, false, None);
    }
    check(4, 512, true, None);
    check(128, 512, true, None);
    check(4, 512, false, Some((150, 40)));
}
