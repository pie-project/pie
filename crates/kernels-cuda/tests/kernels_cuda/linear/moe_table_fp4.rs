//! The FP4 and affine-u4 routed selects through an expert table: every arm
//! this card runs (per route, lean, grouped WMMA and fp32, the register-
//! dequant MMA kernels and their GLU, NVFP4's, mlx's affine u4, W4A8 on the
//! block-scaled mma.sync with and without TMA, W4A4) lands the contiguous
//! bank's bits through its own table and through its experts scattered
//! across a pool, and other bits through a table with two experts swapped.

use super::moe_table::{routes, same_bits};
use crate::common::expert_pool::Plane;
use crate::common::linear_ref::{t, tab};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, F32, I32, Nvfp4, U8};
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::gemm_sm100::Glu;
use kernels_cuda::linear::moe::{self, ExpertTable, GluBank, GroupSeat};
use kernels_cuda::tuning::Tuning;

const SEAT: GroupSeat = GroupSeat::RESIDENT;

/// Which bank and launcher a case fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum K {
    /// MXFP4 through `matmul_select_quant` (no bias) or `_bias`.
    Mx,
    MxBias,
    /// MXFP4 with a GLU through `select_glu`.
    MxGlu,
    /// NVFP4 through `select_nvfp4`, and with a GLU through `select_glu`.
    Nv,
    NvGlu,
    /// mlx affine u4 (group 64) through `matmul_select_quant`.
    Affine,
    /// W4A8 on sm_120a's block-scaled mma.sync (`select_mxfp4_w4a8`).
    W4a8,
    /// W4A4 (`select_nvfp4_w4a4`).
    W4a4,
}

/// Random bytes in `lo..=hi`.
fn bytes(lcg: &mut Lcg, count: usize, lo: u8, hi: u8) -> Vec<u8> {
    let span = f32::from(hi - lo) + 0.99;
    tab(count, |_| lo + (((lcg.unit() + 1.0) * 0.5) * span) as u8)
}

/// Random bf16 in `lo..lo + span`.
fn bf16s(lcg: &mut Lcg, count: usize, lo: f32, span: f32) -> Vec<u16> {
    tab(count, |_| to_bf16(lo + span * (lcg.unit() + 1.0) * 0.5))
}

/// Random f32 in `lo..lo + span`.
fn f32s(lcg: &mut Lcg, count: usize, lo: f32, span: f32) -> Vec<f32> {
    tab(count, |_| lo + span * (lcg.unit() + 1.0) * 0.5)
}

fn w4a8_here() -> bool {
    kernels_cuda::jit::cache::features().warp_block_scale
}

/// One case: the bank, its tuning, `[experts, tokens, top, n, k]`, whether a
/// route reads its token's row.
fn case(g: &mut Gpu, kind: K, tuning: Tuning, [ex, tokens, top, n, k]: [usize; 5], by_token: bool) {
    let what = format!("{kind:?} {ex}x{tokens}x{top} {n}x{k} by_token {by_token}");
    let mut lcg = Lcg::seeded(0xf4 ^ (ex * 131 + tokens * 7 + n + k) as u64);
    let glu = matches!(kind, K::MxGlu | K::NvGlu);
    let rows = if glu { 2 * n } else { n };
    let count = tokens * top;
    let rows_x = if by_token { tokens } else { count };
    let (xr, _) = lcg.row(rows_x * k);
    let x = t(g.up(&xr), rows_x, k, Bf16);
    let r = t(g.up(&routes(tokens, top, ex)), tokens, top, I32);
    let y = g.zeros(count * n * 2);
    let ctx: Ctx = g.ctx().with_tuning(tuning);
    // Codes: e2m1 nibbles, any byte; u4 likewise.
    let code_row = rows * k / 2;
    let codes = g.up(&bytes(&mut lcg, ex * code_row, 0, 255));
    let nv = matches!(kind, K::Nv | K::NvGlu | K::W4a4);
    let affine = kind == K::Affine;
    // Scales: e8m0 a 32 near 2^-5 (MXFP4); e4m3 a 16 in [0.5, 2) (NVFP4);
    // bf16 factor and zero a 64 (affine).
    let (scale_row, scales) = if affine {
        let row = rows * k / 64 * 2;
        (row, g.up(&bf16s(&mut lcg, ex * row / 2, 0.01, 0.02)))
    } else if nv {
        let row = rows * k / 16;
        (row, g.up(&bytes(&mut lcg, ex * row, 0x30, 0x3f)))
    } else {
        let row = rows * k / 32;
        (row, g.up(&bytes(&mut lcg, ex * row, 120, 126)))
    };
    // The third plane: the bias (MXFP4, bf16 a row), the tensor scale
    // (NVFP4, f32 a row), the zeros (affine, bf16 a 64).
    let (third_row, third) = match kind {
        K::MxBias | K::MxGlu => (rows * 2, g.up(&bf16s(&mut lcg, ex * rows, -0.5, 1.0))),
        K::Nv | K::NvGlu | K::W4a4 => (rows * 4, g.up(&f32s(&mut lcg, ex * rows, 0.01, 0.01))),
        K::Affine => (
            scale_row,
            g.up(&bf16s(&mut lcg, ex * scale_row / 2, -0.05, 0.1)),
        ),
        _ => (0, 0),
    };
    let mut planes = vec![
        Plane::of(codes, code_row as u64).aligned(code_row as u64),
        Plane::of(scales, scale_row as u64),
        Plane::of(third, third_row as u64),
    ];
    // W4A4's input scale an expert: the table's copy apart from the one its
    // rows quantize under (`[0]`, the experts' global), which stays put.
    let inputs = f32s(&mut lcg, ex, 0.004, 0.003);
    let (input_q, input_t) = (g.up(&inputs), g.up(&inputs));
    if kind == K::W4a4 {
        planes.push(Plane::of(input_t, 4));
    }
    let y_bytes = count * n * 2;
    let mut took = true;
    same_bits(g, &what, ex, &planes, (y, y_bytes), |table: ExpertTable| {
        let mut yt = t(y, count, n, Bf16);
        let mx = t(codes, ex, code_row, U8);
        let sc = t(scales, ex, scale_row, U8);
        let fired = match kind {
            K::Mx => moe::matmul_select_quant(&ctx, x, mx, sc, None, r, &mut yt, SEAT, table)
                .map(|()| true),
            K::MxBias => {
                let bias = t(third, ex, rows, Bf16);
                moe::matmul_select_bias(&ctx, x, mx, sc, bias, r, &mut yt, SEAT, table)
                    .map(|()| true)
            }
            K::Affine => {
                let zs = t(third, ex, third_row, U8);
                moe::matmul_select_quant(&ctx, x, mx, sc, Some(zs), r, &mut yt, SEAT, table)
                    .map(|()| true)
            }
            K::Nv => {
                let (c, ts) = (t(codes, ex, code_row, Nvfp4), t(third, ex, third_row, U8));
                moe::select_nvfp4(&ctx, x, c, sc, ts, r, &mut yt, SEAT, table).map(|()| true)
            }
            K::MxGlu | K::NvGlu => {
                let bank = if nv {
                    GluBank::Nvfp4 {
                        codes: t(codes, ex, code_row, Nvfp4),
                        scales: sc,
                        tensor_scales: t(third, ex, third_row, U8),
                    }
                } else {
                    GluBank::Mxfp4 {
                        codes: mx,
                        scales: sc,
                        bias: Some(t(third, ex, rows, Bf16)),
                    }
                };
                let alpha = Glu::SwigluClampAlpha {
                    limit: 1e4,
                    alpha: 1.702,
                };
                moe::select_glu(&ctx, x, bank, r, &mut yt, alpha, table)
            }
            K::W4a8 => moe::select_mxfp4_w4a8(&ctx, x, mx, sc, r, &mut yt, 128, table),
            K::W4a4 => {
                let (c, ts) = (t(codes, ex, code_row, Nvfp4), t(third, ex, third_row, U8));
                let is = t(input_q, ex, 1, F32);
                moe::select_nvfp4_w4a4(&ctx, x, c, sc, ts, is, r, &mut yt, table)
            }
        };
        took &= fired.unwrap_or_else(|why| panic!("{what}: {why}"));
    });
    assert!(took, "{what}: the arm declined a table");
}

#[test]
fn an_fp4_select_lands_the_same_bits_through_any_table() {
    let mut g = Gpu::open();
    let d = Tuning::DEFAULT;
    let lean = Tuning {
        mxfp4_select_lean: true,
        ..d
    };
    let fp32 = Tuning {
        mxfp4_wmma: false,
        affine_wmma: false,
        ..d
    };
    let mma = Tuning {
        mxfp4_mma_until: 65_536,
        mxfp4_mma_from: 1,
        ..d
    };
    let nv_mma = Tuning {
        nvfp4_mma_until: 65_536,
        ..d
    };
    let (f, tk) = (false, true);
    let mut cases = vec![
        // MXFP4 per route (and lean), grouped WMMA and fp32, a bias.
        (K::Mx, d, [8, 3, 4, 96, 1024], tk),
        (K::Mx, lean, [8, 3, 4, 96, 1024], f),
        (K::MxBias, d, [8, 3, 4, 96, 1024], tk),
        (K::MxBias, d, [32, 3, 4, 2880, 2880], tk),
        (K::MxBias, d, [8, 64, 4, 256, 256], tk),
        (K::MxBias, fp32, [8, 48, 4, 200, 160], tk),
        // The register-dequant MMA kernel: one 8-route block (K split),
        // two, the staged kernel, the eight-warp ring, a GLU.
        (K::MxBias, mma, [8, 1, 4, 200, 2880], tk),
        (K::Mx, mma, [8, 12, 4, 384, 2880], tk),
        (K::Mx, mma, [8, 120, 4, 256, 512], f),
        (K::Mx, mma, [8, 400, 4, 320, 1024], f),
        (K::MxGlu, mma, [8, 16, 4, 192, 2880], tk),
        (K::MxGlu, mma, [8, 80, 4, 192, 2880], tk),
        // NVFP4 per route (and lean), grouped, the MMA kernel and its GLU.
        (K::Nv, d, [16, 3, 10, 640, 2560], tk),
        (K::Nv, lean, [16, 3, 10, 640, 2560], tk),
        (K::Nv, d, [16, 48, 10, 200, 640], f),
        (K::Nv, nv_mma, [16, 2, 10, 256, 640], f),
        (K::Nv, nv_mma, [16, 160, 10, 256, 640], tk),
        (K::NvGlu, nv_mma, [16, 48, 10, 320, 2560], tk),
        // mlx affine u4: per route, grouped WMMA and fp32.
        (K::Affine, d, [16, 1, 8, 256, 1024], tk),
        (K::Affine, d, [16, 3, 8, 704, 704], f),
        (K::Affine, fp32, [16, 3, 8, 704, 704], f),
    ];
    if w4a8_here() {
        let w4a8 = Tuning {
            mxfp4_w4a8_from: 1,
            ..mma
        };
        let tma = Tuning {
            mxfp4_w4a8_tma: true,
            ..w4a8
        };
        let ws = Tuning {
            mxfp4_w4a8_ws: true,
            ..tma
        };
        let w4a4 = Tuning {
            nvfp4_w4a4_from: 1,
            ..d
        };
        cases.extend([
            (K::W4a8, w4a8, [8, 400, 4, 320, 1024], f),
            (K::W4a8, w4a8, [8, 5, 4, 256, 512], tk),
            (K::W4a8, tma, [8, 400, 4, 320, 1024], f),
            (K::W4a8, tma, [16, 300, 6, 512, 4096], tk),
            (K::W4a8, ws, [8, 400, 4, 320, 1024], f),
            (K::W4a8, ws, [8, 5, 4, 256, 512], tk),
            (K::W4a4, w4a4, [8, 64, 4, 256, 512], tk),
            (K::W4a4, w4a4, [16, 300, 6, 512, 1024], f),
        ]);
    }
    for (kind, tuning, shape, by_token) in cases {
        case(&mut g, kind, tuning, shape, by_token);
    }
}
