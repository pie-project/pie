//! Affine-quantized dense projections (MLX planes: 2-, 4- and 8-bit codes,
//! a factor pair a group, post or pre offset) against a host reference:
//! the streaming decode kernel, the matrix-core GEMM, the decoded plane and
//! its dense arm; then the load's casts.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, f16, from_f16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::linear::affine::{self, OffsetKind, tiled};
use kernels_rocm::linear::moe::GroupSeat;
use kernels_rocm::linear::{cast, dequant};

/// The activation and factor element type.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Elem {
    Bf16,
    F16,
}

impl Elem {
    fn dtype(self) -> Dtype {
        match self {
            Self::Bf16 => Dtype::Bf16,
            Self::F16 => Dtype::F16,
        }
    }
    fn round(self, v: f32) -> f32 {
        match self {
            Self::Bf16 => round_bf16(v),
            Self::F16 => from_f16(f16(v)),
        }
    }
    fn bits(self, v: f32) -> u16 {
        match self {
            Self::Bf16 => common::bf16(v),
            Self::F16 => f16(v),
        }
    }
}

/// A quantized [n, k] weight: its planes as the host knows them.
struct Weight {
    bits: u32,
    group: u32,
    offset: OffsetKind,
    n: u32,
    k: u32,
    codes: Vec<u8>,
    scales: Vec<f32>,
    /// The offset plane's values: real biases, integer zero points, or none.
    offs: Vec<f32>,
}

impl Weight {
    fn new(bits: u32, group: u32, offset: OffsetKind, n: u32, k: u32, e: Elem, seed: u64) -> Self {
        let bytes = n as usize * k as usize * bits as usize / 8;
        let codes: Vec<u8> = values(bytes, seed, 1.0)
            .iter()
            .map(|v| ((v + 1.0) * 128.0) as u8)
            .collect();
        let groups = n as usize * k as usize / group as usize;
        let scales: Vec<f32> = values(groups, seed + 1, 1.0)
            .iter()
            .map(|v| e.round(0.02 + 0.01 * v))
            .collect();
        let top = ((1u32 << bits) - 1) as f32;
        let offs = match offset {
            OffsetKind::Post => values(groups, seed + 2, 0.1)
                .iter()
                .map(|&v| e.round(v - 0.5 * top * 0.02))
                .collect(),
            OffsetKind::PreInt => values(groups, seed + 2, 1.0)
                .iter()
                .map(|v| ((v + 1.0) * 0.5 * top).round())
                .collect(),
            OffsetKind::PreReal => values(groups, seed + 2, 1.0)
                .iter()
                .map(|v| e.round((v + 1.0) * 0.5 * top))
                .collect(),
            OffsetKind::PreConst => Vec::new(),
        };
        Self {
            bits,
            group,
            offset,
            n,
            k,
            codes,
            scales,
            offs,
        }
    }

    fn code(&self, row: usize, i: usize) -> f32 {
        let per_byte = (8 / self.bits) as usize;
        let byte = self.codes[row * (self.k as usize) / per_byte + i / per_byte];
        let shift = self.bits as usize * (i % per_byte);
        f32::from((byte >> shift) & ((1u16 << self.bits) - 1) as u8)
    }

    /// The weight a code stands for, in f32 as the device decodes it.
    fn at(&self, row: usize, i: usize) -> f32 {
        let fx = row * (self.k / self.group) as usize + i / self.group as usize;
        let (c, s) = (self.code(row, i), self.scales[fx]);
        match self.offset {
            OffsetKind::Post => c.mul_add(s, self.offs[fx]),
            OffsetKind::PreInt | OffsetKind::PreReal => s * (c - self.offs[fx]),
            OffsetKind::PreConst => s * (c - (1u32 << (self.bits - 1)) as f32),
        }
    }

    /// The device planes: codes, scales and the offset plane, as bytes.
    fn planes(&self, gpu: &Gpu, e: Elem) -> (Tensor, Tensor, Option<Tensor>) {
        let groups = self.k / self.group;
        let codes = gpu.upload(&self.codes, self.n, self.k * self.bits / 8, Dtype::U8);
        let bits: Vec<u16> = self.scales.iter().map(|&v| e.bits(v)).collect();
        let scales = gpu.upload(&bits, self.n, 2 * groups, Dtype::U8);
        let offs = match self.offset {
            OffsetKind::Post | OffsetKind::PreReal => {
                let bits: Vec<u16> = self.offs.iter().map(|&v| e.bits(v)).collect();
                Some(gpu.upload(&bits, self.n, 2 * groups, Dtype::U8))
            }
            OffsetKind::PreInt => {
                let zs: Vec<u8> = self.offs.iter().map(|&v| v as u8).collect();
                Some(gpu.upload(&zs, self.n, groups, Dtype::U8))
            }
            OffsetKind::PreConst => None,
        };
        (codes, scales, offs)
    }
}

/// y = x · Wᵀ in f64 over the f32 weights, and each output's scale: the
/// root sum of squares of its terms, which sets what rounding the weights
/// to the activation's type may move it by.
fn reference(x: &[f32], w: &Weight, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let (n, k) = (w.n as usize, w.k as usize);
    let wf: Vec<f32> = (0..n * k).map(|i| w.at(i / k, i % k)).collect();
    let mut y = vec![0.0; rows * n];
    let mut scale = vec![0.0; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let (mut dot, mut sq) = (0f64, 0f64);
            for i in 0..k {
                let t = f64::from(x[r * k + i]) * f64::from(wf[c * k + i]);
                dot += t;
                sq += t * t;
            }
            y[r * n + c] = dot as f32;
            scale[r * n + c] = sq.sqrt() as f32;
        }
    }
    (y, scale)
}

/// Within two steps of the output type of the reference, plus 1% of the
/// terms' root sum of squares (a weight rounded to bf16 moves its term by
/// 2^-9 at most, and the moves add as a random walk).
fn check(what: &str, have: &[f32], want: &[f32], scale: &[f32], e: Elem) {
    assert_eq!(have.len(), want.len(), "{what}: lengths");
    let mantissa = if e == Elem::F16 { 1024.0 } else { 128.0 };
    for (i, ((h, w), s)) in have.iter().zip(want).zip(scale).enumerate() {
        let step = (w.abs().max(1e-30)).log2().floor().exp2() / mantissa;
        let tol = 2.0 * step + 1e-2 * s + 1e-6;
        assert!((h - w).abs() <= tol, "{what}[{i}] = {h}, want {w} (±{tol})");
    }
}

struct Case {
    e: Elem,
    bits: u32,
    group: u32,
    offset: OffsetKind,
    rows: u32,
    n: u32,
    k: u32,
}

fn run(gpu: &Gpu, c: &Case, seed: u64) {
    let ctx = gpu.ctx();
    let w = Weight::new(c.bits, c.group, c.offset, c.n, c.k, c.e, seed);
    let (codes, scales, offs) = w.planes(gpu, c.e);
    let x: Vec<f32> = values((c.rows * c.k) as usize, seed + 7, 1.0)
        .into_iter()
        .map(|v| c.e.round(v))
        .collect();
    let xt = gpu.tensor(&x, c.rows, c.k, c.e.dtype());
    let mut y = gpu.zeros(c.rows, c.n, c.e.dtype());
    affine::matmul(
        &ctx,
        xt,
        codes,
        scales,
        c.offset,
        offs,
        c.e.dtype(),
        &mut y,
        GroupSeat::RESIDENT,
    )
    .unwrap();
    let (want, scale) = reference(&x, &w, c.rows as usize);
    check(
        &format!(
            "{:?} u{} g{} {:?} {}x{}x{}",
            c.e, c.bits, c.group, c.offset, c.rows, c.n, c.k
        ),
        &gpu.read(&y),
        &want,
        &scale,
        c.e,
    );
}

#[test]
fn linear_quant_matmul() {
    let Some(gpu) = Gpu::open() else { return };
    use Elem::{Bf16, F16};
    use OffsetKind::{Post, PreConst, PreInt, PreReal};
    let case = |e, bits, group, offset, rows, n, k| Case {
        e,
        bits,
        group,
        offset,
        rows,
        n,
        k,
    };
    let cases = [
        // The decode kernel at each activation-row tiling, a ragged N, and
        // a K that ends inside a wavefront's piece.
        case(Bf16, 4, 64, Post, 1, 1024, 2048),
        case(Bf16, 4, 64, Post, 3, 1003, 2880),
        case(Bf16, 4, 64, Post, 8, 512, 4096),
        case(Bf16, 4, 64, Post, 13, 260, 1024),
        case(Bf16, 4, 64, Post, 16, 256, 2048),
        // The GEMM: a short prefill (narrow tiles), a ragged N and M.
        case(Bf16, 4, 64, Post, 17, 1024, 1024),
        case(Bf16, 4, 64, Post, 100, 1003, 2048),
        case(Bf16, 4, 64, Post, 300, 384, 512),
        // Past 16 rows a K in no whole 64-code stages walks the decode
        // kernel a launch row at a time.
        case(Bf16, 4, 32, Post, 40, 256, 1056),
        // Every width and offset arm, both kernels.
        case(Bf16, 8, 64, PreInt, 2, 512, 1024),
        case(Bf16, 8, 64, PreInt, 70, 512, 1024),
        case(Bf16, 2, 64, PreConst, 5, 512, 2048),
        case(Bf16, 2, 64, PreConst, 64, 512, 2048),
        case(Bf16, 4, 32, PreReal, 4, 640, 1024),
        case(Bf16, 4, 32, PreReal, 33, 640, 1024),
        case(Bf16, 4, 128, Post, 2, 512, 2048),
        case(Bf16, 2, 32, Post, 1, 256, 1024),
        case(Bf16, 8, 32, Post, 1, 256, 1024),
        case(Bf16, 4, 16, PreConst, 3, 256, 512),
        // f16 activations and factors.
        case(F16, 4, 64, Post, 1, 512, 1024),
        case(F16, 8, 64, PreInt, 6, 256, 1024),
        case(F16, 2, 64, PreReal, 50, 256, 1024),
        case(F16, 4, 64, Post, 130, 512, 1024),
    ];
    for (i, c) in cases.iter().enumerate() {
        run(&gpu, c, 11 + i as u64);
    }
}

/// The streaming kernel past one tile a block, with a wavefront holding
/// more than one piece (one of them dead), and a ragged N; the tile kernel
/// at each activation tile, K sliced over the grid, ragged M and N.
#[test]
fn linear_quant_stream_and_tile() {
    let Some(gpu) = Gpu::open() else { return };
    use Elem::{Bf16, F16};
    use OffsetKind::{Post, PreConst, PreInt, PreReal};
    let case = |e, bits, group, offset, rows, n, k| Case {
        e,
        bits,
        group,
        offset,
        rows,
        n,
        k,
    };
    let cases = [
        // Streaming: tiles past the persistent blocks, both activation
        // tiles, a wavefront of two pieces with one dead, ragged N.
        case(Bf16, 4, 64, Post, 1, 12288, 512),
        case(Bf16, 4, 64, Post, 3, 12288, 512),
        case(Bf16, 4, 64, Post, 6, 12288, 512),
        case(Bf16, 4, 64, Post, 1, 256, 8704),
        // K past what a block holds, sliced over the grid.
        case(Bf16, 4, 64, Post, 2, 320, 17408),
        case(Bf16, 4, 64, Post, 7, 320, 17408),
        case(Bf16, 4, 64, Post, 8, 1000, 1536),
        case(Bf16, 8, 64, PreInt, 2, 4096, 1024),
        case(Bf16, 4, 128, PreConst, 4, 2048, 2048),
        case(F16, 4, 32, PreReal, 5, 1024, 1024),
        // Tile: 16-, 32- and 64-row activation tiles, K sliced, ragged.
        case(Bf16, 4, 64, Post, 9, 256, 4096),
        case(Bf16, 4, 64, Post, 16, 1000, 2048),
        case(Bf16, 4, 64, Post, 24, 256, 4096),
        case(Bf16, 4, 64, Post, 32, 640, 1024),
        case(Bf16, 4, 64, Post, 48, 256, 4096),
        case(Bf16, 4, 64, Post, 77, 1000, 1024),
        case(Bf16, 4, 64, Post, 200, 256, 2048),
        case(Bf16, 8, 64, Post, 20, 512, 1024),
        case(Bf16, 4, 32, PreReal, 40, 256, 1024),
        case(F16, 4, 64, Post, 24, 512, 1024),
    ];
    for (i, c) in cases.iter().enumerate() {
        run(&gpu, c, 101 + i as u64);
    }
}

/// A window caps the rows a launch lands; a streaming seat moves the planes
/// the fire reads to wherever the tier put them.
#[test]
fn linear_quant_window_and_seat() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let e = Elem::Bf16;
    for (rows, live, n, k) in [(8u32, 5u32, 512u32, 1024u32), (40, 20, 256, 1024)] {
        let w = Weight::new(4, 64, OffsetKind::Post, n, k, e, 3);
        let (codes, scales, offs) = w.planes(&gpu, e);
        let x: Vec<f32> = values((rows * k) as usize, 4, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let xt = gpu.tensor(&x, rows, k, e.dtype());
        let (want, scale) = reference(&x, &w, rows as usize);

        // The planes the fire binds are decoys; the seat names the real ones.
        let decoy = |t: Tensor| {
            gpu.upload(
                &vec![0u8; (t.rows * t.width) as usize],
                t.rows,
                t.width,
                Dtype::U8,
            )
        };
        let (dc, ds, db) = (decoy(codes), decoy(scales), decoy(offs.unwrap()));
        let seat = gpu.upload(
            &[codes.ptr, scales.ptr, offs.unwrap().ptr, 0],
            1,
            32,
            Dtype::U8,
        );
        let win = gpu.ints(&[live, 0, 0, 0].map(|v| v as i32));
        let mut y = gpu.zeros(rows, n, e.dtype());
        ctx.arm_stage(win.ptr);
        affine::matmul(
            &ctx,
            xt,
            dc,
            ds,
            OffsetKind::Post,
            Some(db),
            Dtype::Bf16,
            &mut y,
            GroupSeat {
                cell: seat.ptr,
                hits: 0,
            },
        )
        .unwrap();
        ctx.disarm_stage();
        let got = gpu.read(&y);
        let cut = (live * n) as usize;
        check(
            &format!("windowed seat {rows}x{n}x{k}"),
            &got[..cut],
            &want[..cut],
            &scale[..cut],
            e,
        );
        assert!(
            got[cut..].iter().all(|v| *v == 0.0),
            "rows past the window stay untouched"
        );
    }
}

/// The plane decoded to bf16 lands each weight rounded once; the dense arm
/// multiplies it.
#[test]
fn linear_quant_decoded() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (bits, group, offset) in [
        (4, 64, OffsetKind::Post),
        (8, 32, OffsetKind::PreInt),
        (2, 64, OffsetKind::PreConst),
        (4, 128, OffsetKind::PreReal),
    ] {
        let (n, k) = (96u32, 512u32);
        let w = Weight::new(bits, group, offset, n, k, Elem::Bf16, 21);
        let (codes, scales, offs) = w.planes(&gpu, Elem::Bf16);
        let dst = gpu.zeros(n, k, Elem::Bf16.dtype());
        dequant::decode_into(
            &ctx,
            "linear.matmul",
            codes,
            scales,
            offset,
            offs,
            Dtype::Bf16,
            dst.ptr,
            n,
            k,
        )
        .unwrap();
        let got = gpu.read(&dst);
        for (i, h) in got.iter().enumerate() {
            let want = round_bf16(w.at(i / k as usize, i % k as usize));
            assert_eq!(
                *h, want,
                "u{bits} g{group} {offset:?} decoded[{i}] = {h}, want {want}"
            );
        }

        let rows = 48u32;
        let x: Vec<f32> = values((rows * k) as usize, 22, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let xt = gpu.tensor(&x, rows, k, Elem::Bf16.dtype());
        let mut y = gpu.zeros(rows, n, Elem::Bf16.dtype());
        dequant::matmul_via_dense(
            &ctx,
            xt,
            codes,
            scales,
            offset,
            offs,
            Dtype::Bf16,
            &mut y,
            GroupSeat::RESIDENT,
        )
        .unwrap();
        let (want, scale) = reference(&x, &w, rows as usize);
        check(
            &format!("via dense u{bits} g{group} {offset:?}"),
            &gpu.read(&y),
            &want,
            &scale,
            Elem::Bf16,
        );
    }
}

#[test]
fn linear_quant_casts() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (rows, width) = (37u32, 1000u32);
    let src = values((rows * width) as usize, 31, 3.0);
    let s = gpu.tensor(&src, rows, width, Dtype::F32);
    let mut dst = gpu.zeros(rows, width, Dtype::Bf16);
    cast::cast_fp32_to(&ctx, s, &mut dst).unwrap();
    let want: Vec<f32> = src.iter().map(|&v| round_bf16(v)).collect();
    assert_eq!(
        gpu.read(&dst),
        want,
        "cast f32 → bf16 rounds to nearest even"
    );

    let buf: Vec<f32> = values((rows * width) as usize, 32, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let l: Vec<f32> = values(width as usize, 33, 2.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let mut b = gpu.tensor(&buf, rows, width, Dtype::Bf16);
    let lt = gpu.tensor(&l, 1, width, Dtype::Bf16);
    cast::scale_rows(&ctx, lt, &mut b).unwrap();
    let want: Vec<f32> = buf
        .iter()
        .enumerate()
        .map(|(i, v)| round_bf16(v * l[i % width as usize]))
        .collect();
    assert_eq!(gpu.read(&b), want, "scale_rows");
}

/// The tile-order repack of a four-bit plane (`U4g64tiled`), as the host
/// spells it: word (band, step, 4·col + q, kt) holds nibble s + 4h from row
/// 16·band + col + 8·(s ≥ 2), code 64·step + 16·kt + 2q + 8·(s & 1) + h.
fn tile_codes(w: &Weight) -> Vec<u32> {
    let (n, k) = (w.n as usize, w.k as usize);
    let bands = n.div_ceil(16);
    let mut out = Vec::with_capacity(bands * k / 64 * 128);
    for band in 0..bands {
        for step in 0..k / 64 {
            for word in 0..32 {
                for kt in 0..4 {
                    let (col, q) = (word / 4, word % 4);
                    let mut v = 0u32;
                    for s in 0..4 {
                        for h in 0..2 {
                            let row = band * 16 + col + if s >= 2 { 8 } else { 0 };
                            let i = step * 64 + kt * 16 + 2 * q + 8 * (s & 1) + h;
                            let c = if row < n { w.code(row, i) as u32 } else { 0 };
                            v |= c << (4 * (s + 4 * h));
                        }
                    }
                    out.push(v);
                }
            }
        }
    }
    out
}

/// The repack, then the tiled GEMM and decode arms over it.
#[test]
fn linear_quant_tiled() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let e = Elem::Bf16;
    for (group, n, k) in [(64u32, 1000u32, 1024u32), (32, 256, 2048), (128, 512, 512)] {
        let w = Weight::new(4, group, OffsetKind::Post, n, k, e, 41);
        let (codes, scales, offs) = w.planes(&gpu, e);
        let biases = offs.unwrap();
        let n_pad = tiled::repacked_rows(n);
        let mut tc = gpu.upload(
            &vec![0u8; (n_pad * k / 2) as usize],
            n_pad,
            k / 2,
            Dtype::U8,
        );
        let mut ts = gpu.upload(
            &vec![0u8; (n_pad * scales.width) as usize],
            n_pad,
            scales.width,
            Dtype::U8,
        );
        let mut tb = gpu.upload(
            &vec![0u8; (n_pad * scales.width) as usize],
            n_pad,
            scales.width,
            Dtype::U8,
        );
        tiled::repack(&ctx, codes, scales, biases, &mut tc, &mut ts, &mut tb).unwrap();
        let have: Vec<u32> = gpu.read_as(&tc);
        assert_eq!(have, tile_codes(&w), "the repacked codes, g{group} {n}x{k}");
        let have: Vec<u16> = gpu.read_as(&ts);
        let groups = (k / group) as usize;
        for (i, h) in have.iter().enumerate() {
            let (j, g, band) = (i % 16, i / 16 % groups, i / 16 / groups);
            let row = band * 16 + j;
            let want = if row < n as usize {
                e.bits(w.scales[row * groups + g])
            } else {
                0
            };
            assert_eq!(*h, want, "the repacked scale [{band}][{g}][{j}]");
        }

        for (rows, live) in [
            (1u32, None),
            (5, None),
            (16, Some(11u32)),
            (40, None),
            (300, Some(260)),
        ] {
            let x: Vec<f32> = values((rows * k) as usize, 42, 1.0)
                .into_iter()
                .map(round_bf16)
                .collect();
            let xt = gpu.tensor(&x, rows, k, e.dtype());
            let mut y = gpu.zeros(rows, n, e.dtype());
            let stage = live.map(|v| gpu.ints(&[v as i32, 0, 0, 0]));
            if let Some(s) = &stage {
                ctx.arm_stage(s.ptr);
            }
            let entry = if rows > 16 {
                tiled::matmul
            } else {
                tiled::matmul_gemv
            };
            entry(&ctx, xt, tc, ts, tb, &mut y, GroupSeat::RESIDENT).unwrap();
            ctx.disarm_stage();
            let (want, scale) = reference(&x, &w, rows as usize);
            let cut = (live.unwrap_or(rows) * n) as usize;
            let got = gpu.read(&y);
            check(
                &format!("tiled g{group} {rows}x{n}x{k}"),
                &got[..cut],
                &want[..cut],
                &scale[..cut],
                e,
            );
            assert!(
                got[cut..].iter().all(|v| *v == 0.0),
                "rows past the window stay untouched"
            );
        }
        // Both GEMM tiles at a row count the gemv also takes.
        for tuple in [tiled::LONG, tiled::SHORT] {
            let rows = 16;
            let x: Vec<f32> = values((rows * k) as usize, 43, 1.0)
                .into_iter()
                .map(round_bf16)
                .collect();
            let xt = gpu.tensor(&x, rows, k, e.dtype());
            let mut y = gpu.zeros(rows, n, e.dtype());
            tiled::matmul_with(&ctx, xt, tc, ts, tb, &mut y, GroupSeat::RESIDENT, tuple).unwrap();
            let (want, scale) = reference(&x, &w, rows as usize);
            check(
                &format!("tiled {tuple:?} g{group}"),
                &gpu.read(&y),
                &want,
                &scale,
                e,
            );
        }
    }
}
