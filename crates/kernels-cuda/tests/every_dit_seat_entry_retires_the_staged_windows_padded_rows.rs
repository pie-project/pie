#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::elemwise::modulate::{self, NormKind};
use kernels_cuda::elemwise::{RopeForm, activation, binary, rope_axes, sinusoid};
use kernels_cuda::jit::Ctx;
use kernels_cuda::layout;
use kernels_cuda::tensor::Tensor;

const BASE: u32 = 2;
const LIVE: u32 = 3;
const BUCKET: u32 = 5;
const PLANES: usize = 8;

const WIDTH: usize = 64;
const HEAD_DIM: usize = 32;
const EPS: f32 = 1e-6;

const SENTINEL: u8 = 0xa5;

#[derive(Clone, Copy)]
struct Pair {
    staged: u64,
    shifted: u64,
}

fn destinations(gpu: &mut Gpu, row_bytes: usize) -> Pair {
    let filled = vec![SENTINEL; PLANES * row_bytes];
    Pair {
        staged: gpu.up(&filled),
        shifted: gpu.up(&filled),
    }
}

fn agree(gpu: &Gpu, pair: Pair, row_bytes: usize, name: &str) {
    let staged: Vec<u8> = gpu.down(pair.staged, PLANES * row_bytes);
    let shifted: Vec<u8> = gpu.down(pair.shifted, PLANES * row_bytes);
    assert_eq!(
        staged, shifted,
        "{name}: the staged window is not the shifted rectangle"
    );
    let mut written = 0;
    for row in 0..PLANES {
        let span = &staged[row * row_bytes..(row + 1) * row_bytes];
        let inside = row >= BASE as usize && row < (BASE + LIVE) as usize;
        if inside {
            written += usize::from(span.iter().any(|&b| b != SENTINEL));
        } else {
            assert!(
                span.iter().all(|&b| b == SENTINEL),
                "{name}: row {row} is outside the window and moved"
            );
        }
    }
    assert_eq!(
        written as u32, LIVE,
        "{name}: the window's own rows were not written"
    );
}

fn arm(gpu: &mut Gpu, ctx: &Ctx) {
    let at = gpu.up(&[LIVE, BASE, 0u32, 0]);
    ctx.arm_stage(at);
}

fn view(plane: u64, shift: u32, rows: u32, width: usize, dtype: Dtype, elem: usize) -> Tensor {
    Tensor::new(
        plane + u64::from(shift) * (width * elem) as u64,
        rows,
        width as u32,
        dtype,
    )
}

fn bf16(plane: u64, shift: u32, rows: u32, width: usize) -> Tensor {
    view(plane, shift, rows, width, Dtype::Bf16, 2)
}

#[test]
fn every_dit_seat_entry_retires_the_staged_windows_padded_rows_every_case() {
    the_modulations_retire_their_padded_rows();
    the_deferred_residual_pair_retires_its_padded_rows_on_both_outputs();
    the_timestep_embedding_retires_its_padded_rows();
    the_axial_rotation_retires_its_padded_rows();
    the_row_permutations_retire_their_padded_rows();
    the_bare_pointwise_ops_retire_their_padded_rows();
}

fn the_modulations_retire_their_padded_rows() {
    let mut lcg = Lcg::seeded(0x5ea7);
    let (x_raw, _) = lcg.row(PLANES * WIDTH);
    let (y_raw, _) = lcg.row(PLANES * WIDTH);
    let (m_raw, _) = lcg.row(PLANES * 2 * WIDTH);
    let (w_raw, _) = lcg.row(WIDTH);

    for form in 0..3 {
        let mut gpu = Gpu::open();
        let x = gpu.up(&x_raw);
        let y = gpu.up(&y_raw);
        let m = gpu.up(&m_raw);
        let w = gpu.up(&w_raw);
        let pair = destinations(&mut gpu, WIDTH * 2);
        let ctx = gpu.ctx();
        let weight = Tensor::new(w, WIDTH as u32, 1, Dtype::Bf16);

        let fire = |shift: u32, rows: u32, dst: u64| {
            let x = bf16(x, shift, rows, WIDTH);
            let y = bf16(y, shift, rows, WIDTH);
            let m2 = view(m, shift, rows, 2 * WIDTH, Dtype::Bf16, 2);
            let m1 = bf16(m, shift, rows, WIDTH);
            let mut o = bf16(dst, shift, rows, WIDTH);
            match form {
                0 => modulate::scale_shift(&ctx, x, m2, None, &mut o),
                1 => modulate::gated_residual_add(&ctx, x, m1, y, None, &mut o),
                _ => modulate::norm_modulate(
                    &ctx,
                    x,
                    m2,
                    None,
                    NormKind::RmsNorm { weight, eps: EPS },
                    &mut o,
                ),
            }
            .expect("fires");
        };
        arm(&mut gpu, &ctx);
        fire(0, BUCKET, pair.staged);
        ctx.disarm_stage();
        fire(BASE, LIVE, pair.shifted);
        gpu.sync();
        agree(&gpu, pair, WIDTH * 2, &format!("modulation form {form}"));
    }
}

fn the_deferred_residual_pair_retires_its_padded_rows_on_both_outputs() {
    let mut lcg = Lcg::seeded(0x5ea72);
    let (r_raw, _) = lcg.row(PLANES * WIDTH);
    let (y_raw, _) = lcg.row(PLANES * WIDTH);
    let (g_raw, _) = lcg.row(PLANES * WIDTH);
    let (m_raw, _) = lcg.row(PLANES * 2 * WIDTH);

    let mut gpu = Gpu::open();
    let r = gpu.up(&r_raw);
    let y = gpu.up(&y_raw);
    let g = gpu.up(&g_raw);
    let m = gpu.up(&m_raw);
    let residual = destinations(&mut gpu, WIDTH * 2);
    let normed = destinations(&mut gpu, WIDTH * 2);
    let ctx = gpu.ctx();

    let fire = |shift: u32, rows: u32, r_out: u64, o_out: u64| {
        modulate::gated_residual_norm_modulate(
            &ctx,
            bf16(r, shift, rows, WIDTH),
            bf16(g, shift, rows, WIDTH),
            bf16(y, shift, rows, WIDTH),
            view(m, shift, rows, 2 * WIDTH, Dtype::Bf16, 2),
            None,
            NormKind::LayerNormNoAffine { eps: EPS },
            &mut bf16(r_out, shift, rows, WIDTH),
            &mut bf16(o_out, shift, rows, WIDTH),
        )
        .expect("fires");
    };
    arm(&mut gpu, &ctx);
    fire(0, BUCKET, residual.staged, normed.staged);
    ctx.disarm_stage();
    fire(BASE, LIVE, residual.shifted, normed.shifted);
    gpu.sync();
    agree(&gpu, residual, WIDTH * 2, "the pair's residual");
    agree(&gpu, normed, WIDTH * 2, "the pair's normed row");
}

fn the_timestep_embedding_retires_its_padded_rows() {
    let mut lcg = Lcg::seeded(0x5ea73);
    let t: Vec<f32> = (0..PLANES).map(|_| lcg.unit() * 4.0 + 5.0).collect();

    let mut gpu = Gpu::open();
    let t_at = gpu.up(&t);
    let pair = destinations(&mut gpu, WIDTH * 4);
    let ctx = gpu.ctx();

    let fire = |shift: u32, rows: u32, dst: u64| {
        sinusoid(
            &ctx,
            view(t_at, shift, rows, 1, Dtype::F32, 4),
            WIDTH as u32,
            10_000.0,
            true,
            1.0,
            &mut view(dst, shift, rows, WIDTH, Dtype::F32, 4),
        )
        .expect("fires");
    };
    arm(&mut gpu, &ctx);
    fire(0, BUCKET, pair.staged);
    ctx.disarm_stage();
    fire(BASE, LIVE, pair.shifted);
    gpu.sync();
    agree(&gpu, pair, WIDTH * 4, "sinusoid");
}

fn the_axial_rotation_retires_its_padded_rows() {
    let mut lcg = Lcg::seeded(0x5ea74);
    let (x_raw, _) = lcg.row(PLANES * WIDTH);
    let positions: Vec<f32> = (0..PLANES * 2).map(|i| i as f32 * 0.75).collect();

    let mut gpu = Gpu::open();
    let x = gpu.up(&x_raw);
    let pos = gpu.up(&positions);
    let pair = destinations(&mut gpu, WIDTH * 2);
    let ctx = gpu.ctx();

    let fire = |shift: u32, rows: u32, dst: u64| {
        rope_axes(
            &ctx,
            bf16(x, shift, rows, WIDTH),
            view(pos, shift, rows, 2, Dtype::F32, 4),
            [16, 16, 0, 0],
            [256.0, 10_000.0, 0.0, 0.0],
            RopeForm::Interleaved,
            32,
            HEAD_DIM as u32,
            &mut bf16(dst, shift, rows, WIDTH),
        )
        .expect("fires");
    };
    arm(&mut gpu, &ctx);
    fire(0, BUCKET, pair.staged);
    ctx.disarm_stage();
    fire(BASE, LIVE, pair.shifted);
    gpu.sync();
    agree(&gpu, pair, WIDTH * 2, "rope_axes");
}

fn the_row_permutations_retire_their_padded_rows() {
    let mut lcg = Lcg::seeded(0x5ea75);
    let (x_raw, _) = lcg.row(PLANES * WIDTH);
    let staged_map: Vec<i32> = (0..PLANES)
        .map(|i| {
            let (base, live) = (BASE as usize, LIVE as usize);
            if i >= base && i < base + live {
                (base + (i - base + 1) % live) as i32
            } else {
                i as i32
            }
        })
        .collect();
    let shifted_map: Vec<i32> = staged_map[BASE as usize..]
        .iter()
        .map(|&v| v - BASE as i32)
        .collect();

    for pack in [true, false] {
        let mut gpu = Gpu::open();
        let x = gpu.up(&x_raw);
        let staged_at = gpu.up(&staged_map);
        let shifted_at = gpu.up(&shifted_map);
        let pair = destinations(&mut gpu, WIDTH * 2);
        let ctx = gpu.ctx();

        let fire = |shift: u32, rows: u32, map: u64, dst: u64| {
            let x = bf16(x, shift, rows, WIDTH);
            let map = Tensor::new(map, rows, 1, Dtype::I32);
            let mut o = bf16(dst, shift, rows, WIDTH);
            if pack {
                layout::pack_rows(&ctx, x, map, &mut o)
            } else {
                layout::unpack_rows(&ctx, x, map, &mut o)
            }
            .expect("fires");
        };
        arm(&mut gpu, &ctx);
        fire(0, BUCKET, staged_at, pair.staged);
        ctx.disarm_stage();
        fire(BASE, LIVE, shifted_at, pair.shifted);
        gpu.sync();
        agree(
            &gpu,
            pair,
            WIDTH * 2,
            if pack { "pack_rows" } else { "unpack_rows" },
        );
    }
}

fn the_bare_pointwise_ops_retire_their_padded_rows() {
    let mut lcg = Lcg::seeded(0x5ea76);
    let (x_raw, _) = lcg.row(PLANES * WIDTH);
    let (y_raw, _) = lcg.row(PLANES * WIDTH);

    for which in 0..5 {
        let mut gpu = Gpu::open();
        let x = gpu.up(&x_raw);
        let y = gpu.up(&y_raw);
        let pair = destinations(&mut gpu, WIDTH * 2);
        let ctx = gpu.ctx();

        let fire = |shift: u32, rows: u32, dst: u64| {
            let x = bf16(x, shift, rows, WIDTH);
            let y = bf16(y, shift, rows, WIDTH);
            let mut o = bf16(dst, shift, rows, WIDTH);
            match which {
                0 => binary::add(&ctx, x, y, &mut o),
                1 => binary::mul(&ctx, x, y, &mut o),
                2 => activation::silu(&ctx, x, &mut o),
                3 => activation::gelu_tanh(&ctx, x, &mut o),
                _ => activation::tanh(&ctx, x, &mut o),
            }
            .expect("fires");
        };
        arm(&mut gpu, &ctx);
        fire(0, BUCKET, pair.staged);
        ctx.disarm_stage();
        fire(BASE, LIVE, pair.shifted);
        gpu.sync();
        agree(&gpu, pair, WIDTH * 2, &format!("pointwise op {which}"));
    }
}
