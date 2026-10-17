#![cfg(feature = "cuda")]

//! DeepSeek-V4's compressor in one pass (`pool::compress` under
//! `rows_vec`: gather, scaled RMS norm, tail rotation and append) against
//! the unfused ops, at both ratios, the 512-wide latent (bf16 and fp8 pools)
//! and the 128-wide index keys, on a staged window too.

use crate::common::host::rows_read;
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
        None,
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
    rows_read(&raw, cells, PAGE, dsv4::Rows::Ds)
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
    for (ratio, d, fp8, window) in [
        (4, 512, false, None),
        (128, 512, false, None),
        (4, 128, false, None),
        (4, 512, true, None),
        (128, 512, true, None),
        (4, 512, false, Some((150, 40))),
    ] {
        check(ratio, d, fp8, window);
    }
}

/// A decode's fire, one row a request: the compressor's state written in the
/// one pass lands the state slab and the entries pool the separate state
/// write, then the pass, land.
fn decode_pass(fused: bool, ratio: usize, d: usize) -> (Vec<u16>, Vec<u16>) {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * d;
    let requests = 8usize;
    let per = 12usize;
    let total = requests * per;
    let cells = total * PAGE;
    let pages: Vec<u32> = (0..total).map(|i| ((5 * i) % total) as u32).collect();
    let indptr: Vec<u32> = (0..=requests).map(|r| (r * per) as u32).collect();
    // Request r's new row at position p_r; every other request closes an
    // entry with it.
    let blocks = (per * PAGE / ratio).max(1);
    let pos: Vec<usize> = (0..requests)
        .map(|r| {
            let close = ((r / 2) % blocks + 1) * ratio - 1;
            if r % 2 == 0 { close } else { close - 1 }
        })
        .collect();
    let bpos: Vec<i32> = pos
        .iter()
        .map(|&p| if (p + 1) % ratio == 0 { p as i32 } else { -1 })
        .collect();
    let breq: Vec<i32> = (0..requests as i32).collect();
    let brope: Vec<i32> = pos
        .iter()
        .map(|&p| {
            if (p + 1) % ratio == 0 {
                (p + 1 - ratio) as i32
            } else {
                0
            }
        })
        .collect();
    let slot = |r: usize, p: usize| pages[r * per + p / PAGE] as usize * PAGE + p % PAGE;
    let w_page: Vec<u32> = (0..requests)
        .map(|r| (slot(r, pos[r]) / PAGE) as u32)
        .collect();
    let w_off: Vec<u32> = (0..requests)
        .map(|r| (slot(r, pos[r]) % PAGE) as u32)
        .collect();
    let mut lcg = Lcg::seeded(0x57a7e + (ratio * d) as u64);
    let (kv_bits, _) = lcg.row(cells * width);
    let score_bits: Vec<u16> = (0..cells * width)
        .map(|_| to_bf16(lcg.unit() * 4.0))
        .collect();
    let (new_kv, _) = lcg.row(requests * width);
    let new_score: Vec<u16> = (0..requests * width)
        .map(|_| to_bf16(lcg.unit() * 4.0))
        .collect();
    let ape: Vec<f32> = (0..ratio * width).map(|_| lcg.unit()).collect();
    let w: Vec<u16> = (0..d).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let mut gpu = Gpu::open();
    let kv = gpu.up(&kv_bits);
    let score = gpu.up(&score_bits);
    let state = pool_of(&mut gpu, kv, &pages, &indptr, width as u32, Dtype::Bf16);
    let out_at = gpu.zeros(cells * d * 2);
    let out = pool_of(&mut gpu, out_at, &pages, &indptr, d as u32, Dtype::Bf16);
    let ctx = gpu.ctx().with_tuning(Tuning {
        rows_vec: true,
        ..Tuning::DEFAULT
    });
    let rows = requests as u32;
    let t32 = |gpu: &mut Gpu, v: &[i32]| Tensor::new(gpu.up(v), rows, 1, Dtype::I32);
    let bpos_t = t32(&mut gpu, &bpos);
    let breq_t = t32(&mut gpu, &breq);
    let brope_t = t32(&mut gpu, &brope);
    let wp = Tensor::new(gpu.up(&w_page), rows, 1, Dtype::U32);
    let wo = Tensor::new(gpu.up(&w_off), rows, 1, Dtype::U32);
    let new_kv_t = Tensor::new(gpu.up(&new_kv), rows, width as u32, Dtype::Bf16);
    let new_score_t = Tensor::new(gpu.up(&new_score), rows, width as u32, Dtype::Bf16);
    let ape_t = Tensor::new(gpu.up(&ape), ratio as u32, width as u32, Dtype::F32);
    let w_t = Tensor::new(gpu.up(&w), d as u32, 1, Dtype::Bf16);
    let state_kv = Tensor::new(kv, cells as u32, width as u32, Dtype::Bf16);
    let state_score = Tensor::new(score, cells as u32, width as u32, Dtype::Bf16);
    if !fused {
        pool::state_write(
            &ctx,
            new_kv_t,
            new_score_t,
            &state,
            wp,
            wo,
            d as u32,
            ratio as u32,
            state_kv,
            state_score,
        )
        .expect("the state write fires");
    }
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
        None,
        &out,
        fused.then_some(pool::StateRows {
            kv: new_kv_t,
            score: new_score_t,
            write_page: wp,
            write_offset: wo,
        }),
    )
    .expect("the compressor fires");
    assert!(landed, "the one pass lands");
    gpu.sync();
    (gpu.down(kv, cells * width), gpu.down(out_at, cells * d))
}

#[test]
fn a_decodes_compressor_state_lands_in_the_one_pass() {
    for (ratio, d) in [(4, 512), (128, 512), (4, 128)] {
        let two = decode_pass(false, ratio, d);
        let one = decode_pass(true, ratio, d);
        assert!(
            one.0 == two.0,
            "ratio {ratio} d {d}: the state slabs differ"
        );
        assert!(one.1 == two.1, "ratio {ratio} d {d}: the entries differ");
        assert!(
            one.1.iter().any(|&b| b != 0),
            "ratio {ratio} d {d}: entries are written"
        );
    }
}
