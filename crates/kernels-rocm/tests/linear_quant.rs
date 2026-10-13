//! Affine-quantized dense projections (MLX planes: 2-, 4- and 8-bit codes,
//! a factor pair a group, post or pre offset) against a host reference:
//! the streaming decode kernel, the matrix-core GEMM, the decoded plane and
//! its dense arm; then the load's casts.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, from_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::hip::Buffer;
use kernels_rocm::linear::moe::GroupSeat;
use kernels_rocm::linear::quant::{self, OffsetKind};
use kernels_rocm::linear::tiled;

/// f32 → f16 bits, round to nearest even (normal range and zero).
fn f16(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp < 1 {
        // Below the normal range: the test flushes it to zero, on both sides.
        return sign;
    }
    assert!(exp < 31, "{v} is past the test's f16 range");
    let mant = b & 0x7f_ffff;
    let mut h = (exp as u32) << 10 | mant >> 13;
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f32::from(h & 0x3ff);
    if exp == 0 {
        return sign * mant * (-24f32).exp2();
    }
    sign * (1.0 + mant / 1024.0) * ((exp - 15) as f32).exp2()
}

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
    fn widen(self, h: u16) -> f32 {
        match self {
            Self::Bf16 => from_bf16(h),
            Self::F16 => from_f16(h),
        }
    }
}

/// Device buffers the test holds past its fires.
struct Held<'g> {
    gpu: &'g Gpu,
    buffers: Vec<Buffer>,
}

impl<'g> Held<'g> {
    fn new(gpu: &'g Gpu) -> Self {
        Self {
            gpu,
            buffers: Vec::new(),
        }
    }
    fn bytes<T: Copy>(&mut self, host: &[T], rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let buffer = Buffer::from_slice(host).unwrap();
        let ptr = buffer.device_ptr();
        self.buffers.push(buffer);
        Tensor::new(ptr, rows, width, dtype)
    }
    fn elems(&mut self, values: &[f32], rows: u32, width: u32, e: Elem) -> Tensor {
        let bits: Vec<u16> = values.iter().map(|&v| e.bits(v)).collect();
        self.bytes(&bits, rows, width, e.dtype())
    }
    fn zeros(&mut self, rows: u32, width: u32, e: Elem) -> Tensor {
        self.bytes(&vec![0u16; (rows * width) as usize], rows, width, e.dtype())
    }
    fn read(&self, t: &Tensor, e: Elem) -> Vec<f32> {
        self.gpu.sync();
        let held = self
            .buffers
            .iter()
            .find(|b| b.device_ptr() == t.ptr)
            .expect("a held buffer");
        held.read::<u16>().unwrap()[..t.elements() as usize]
            .iter()
            .map(|&h| e.widen(h))
            .collect()
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
        let bytes = (n * k * bits / 8) as usize;
        let codes: Vec<u8> = values(bytes, seed, 1.0)
            .iter()
            .map(|v| ((v + 1.0) * 128.0) as u8)
            .collect();
        let groups = (n * k / group) as usize;
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
    fn planes(&self, held: &mut Held, e: Elem) -> (Tensor, Tensor, Option<Tensor>) {
        let groups = self.k / self.group;
        let codes = held.bytes(&self.codes, self.n, self.k * self.bits / 8, Dtype::U8);
        let bits: Vec<u16> = self.scales.iter().map(|&v| e.bits(v)).collect();
        let scales = held.bytes(&bits, self.n, 2 * groups, Dtype::U8);
        let offs = match self.offset {
            OffsetKind::Post | OffsetKind::PreReal => {
                let bits: Vec<u16> = self.offs.iter().map(|&v| e.bits(v)).collect();
                Some(held.bytes(&bits, self.n, 2 * groups, Dtype::U8))
            }
            OffsetKind::PreInt => {
                let zs: Vec<u8> = self.offs.iter().map(|&v| v as u8).collect();
                Some(held.bytes(&zs, self.n, groups, Dtype::U8))
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
    let mut held = Held::new(gpu);
    let w = Weight::new(c.bits, c.group, c.offset, c.n, c.k, c.e, seed);
    let (codes, scales, offs) = w.planes(&mut held, c.e);
    let x: Vec<f32> = values((c.rows * c.k) as usize, seed + 7, 1.0)
        .into_iter()
        .map(|v| c.e.round(v))
        .collect();
    let xt = held.elems(&x, c.rows, c.k, c.e);
    let mut y = held.zeros(c.rows, c.n, c.e);
    quant::matmul(
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
        &held.read(&y, c.e),
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

/// A window caps the rows a launch lands; a streaming seat moves the planes
/// the fire reads to wherever the tier put them.
#[test]
fn linear_quant_window_and_seat() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let e = Elem::Bf16;
    for (rows, live, n, k) in [(8u32, 5u32, 512u32, 1024u32), (40, 20, 256, 1024)] {
        let mut held = Held::new(&gpu);
        let w = Weight::new(4, 64, OffsetKind::Post, n, k, e, 3);
        let (codes, scales, offs) = w.planes(&mut held, e);
        let x: Vec<f32> = values((rows * k) as usize, 4, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let xt = held.elems(&x, rows, k, e);
        let (want, scale) = reference(&x, &w, rows as usize);

        // The planes the fire binds are decoys; the seat names the real ones.
        let decoy = |held: &mut Held, t: Tensor| {
            held.bytes(
                &vec![0u8; (t.rows * t.width) as usize],
                t.rows,
                t.width,
                Dtype::U8,
            )
        };
        let (dc, ds, db) = (
            decoy(&mut held, codes),
            decoy(&mut held, scales),
            decoy(&mut held, offs.unwrap()),
        );
        let seat = held.bytes(
            &[codes.ptr, scales.ptr, offs.unwrap().ptr, 0],
            1,
            32,
            Dtype::U8,
        );
        let win = gpu.ints(&[live, 0, 0, 0].map(|v| v as i32));
        let mut y = held.zeros(rows, n, e);
        ctx.arm_stage(win.ptr);
        quant::matmul(
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
        let got = held.read(&y, e);
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
        let mut held = Held::new(&gpu);
        let w = Weight::new(bits, group, offset, n, k, Elem::Bf16, 21);
        let (codes, scales, offs) = w.planes(&mut held, Elem::Bf16);
        let dst = held.zeros(n, k, Elem::Bf16);
        quant::decode_into(
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
        let got = held.read(&dst, Elem::Bf16);
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
        let xt = held.elems(&x, rows, k, Elem::Bf16);
        let mut y = held.zeros(rows, n, Elem::Bf16);
        quant::matmul_via_dense(
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
            &held.read(&y, Elem::Bf16),
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
    quant::cast_fp32_to(&ctx, s, &mut dst).unwrap();
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
    quant::scale_rows(&ctx, lt, &mut b).unwrap();
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
        let mut held = Held::new(&gpu);
        let w = Weight::new(4, group, OffsetKind::Post, n, k, e, 41);
        let (codes, scales, offs) = w.planes(&mut held, e);
        let biases = offs.unwrap();
        let n_pad = tiled::repacked_rows(n);
        let mut tc = held.bytes(
            &vec![0u8; (n_pad * k / 2) as usize],
            n_pad,
            k / 2,
            Dtype::U8,
        );
        let mut ts = held.bytes(
            &vec![0u8; (n_pad * scales.width) as usize],
            n_pad,
            scales.width,
            Dtype::U8,
        );
        let mut tb = held.bytes(
            &vec![0u8; (n_pad * scales.width) as usize],
            n_pad,
            scales.width,
            Dtype::U8,
        );
        tiled::repack(&ctx, codes, scales, biases, &mut tc, &mut ts, &mut tb).unwrap();
        gpu.sync();
        let have: Vec<u32> = held
            .buffers
            .iter()
            .find(|b| b.device_ptr() == tc.ptr)
            .unwrap()
            .read()
            .unwrap();
        assert_eq!(have, tile_codes(&w), "the repacked codes, g{group} {n}x{k}");
        let have: Vec<u16> = held
            .buffers
            .iter()
            .find(|b| b.device_ptr() == ts.ptr)
            .unwrap()
            .read()
            .unwrap();
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
            let xt = held.elems(&x, rows, k, e);
            let mut y = held.zeros(rows, n, e);
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
            let got = held.read(&y, e);
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
            let xt = held.elems(&x, rows, k, e);
            let mut y = held.zeros(rows, n, e);
            tiled::matmul_with(&ctx, xt, tc, ts, tb, &mut y, GroupSeat::RESIDENT, tuple).unwrap();
            let (want, scale) = reference(&x, &w, rows as usize);
            check(
                &format!("tiled {tuple:?} g{group}"),
                &held.read(&y, e),
                &want,
                &scale,
                e,
            );
        }
    }
}

/// Decode and prefill throughput on cold planes: enough copies cycled that
/// the set outruns the 256 MiB last-level cache, timed in one graph. A bf16
/// dense projection of the same bytes runs beside each shape for scale.
/// `--ignored` to run.
#[test]
#[ignore = "benchmark"]
fn bench_linear_quant() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let e = Elem::Bf16;
    for (bits, n, k) in [
        (4u32, 4096u32, 4096u32),
        (4, 14336, 4096),
        (4, 4096, 14336),
        (8, 4096, 4096),
        (2, 4096, 4096),
    ] {
        let mut held = Held::new(&gpu);
        let w = Weight::new(bits, 64, OffsetKind::Post, n, k, e, 5);
        let bytes = f64::from(n * k * bits / 8 + 2 * 2 * n * (k / 64));
        let copies = (512.0 * 1024.0 * 1024.0 / bytes).ceil() as usize;
        let planes: Vec<_> = (0..copies).map(|_| w.planes(&mut held, e)).collect();
        for rows in [1u32, 4, 8, 16, 64, 512, 2048] {
            let x = values((rows * k) as usize, 6, 1.0);
            let xt = held.elems(&x, rows, k, e);
            let mut y = held.zeros(rows, n, e);
            let mut at = 0;
            let us = gpu.time_graph(2 * copies as u32, || {
                let (codes, scales, offs) = planes[at % copies];
                at += 1;
                quant::matmul(
                    &ctx,
                    xt,
                    codes,
                    scales,
                    OffsetKind::Post,
                    offs,
                    Dtype::Bf16,
                    &mut y,
                    GroupSeat::RESIDENT,
                )
                .unwrap();
            });
            let flops = 2.0 * f64::from(rows) * f64::from(n) * f64::from(k);
            eprintln!(
                "u{bits} {n}x{k} rows {rows:>4}: {us:8.1} us  {:5.2} TB/s  {:6.1} TFLOP/s",
                bytes / us * 1e-6,
                flops / us * 1e-6
            );
        }
        // The tile-order arm of the same four-bit planes.
        if bits == 4 {
            let n_pad = tiled::repacked_rows(n);
            let tiles: Vec<_> = planes
                .iter()
                .map(|&(codes, scales, offs)| {
                    let mut tc = held.bytes(
                        &vec![0u8; (n_pad * k / 2) as usize],
                        n_pad,
                        k / 2,
                        Dtype::U8,
                    );
                    let mut ts = held.bytes(
                        &vec![0u8; (n_pad * scales.width) as usize],
                        n_pad,
                        scales.width,
                        Dtype::U8,
                    );
                    let mut tb = held.bytes(
                        &vec![0u8; (n_pad * scales.width) as usize],
                        n_pad,
                        scales.width,
                        Dtype::U8,
                    );
                    tiled::repack(
                        &ctx,
                        codes,
                        scales,
                        offs.unwrap(),
                        &mut tc,
                        &mut ts,
                        &mut tb,
                    )
                    .unwrap();
                    (tc, ts, tb)
                })
                .collect();
            for rows in [1u32, 4, 8, 16, 64, 512, 2048] {
                let x = values((rows * k) as usize, 6, 1.0);
                let xt = held.elems(&x, rows, k, e);
                let mut y = held.zeros(rows, n, e);
                let mut at = 0;
                let entry = if rows > 16 {
                    tiled::matmul
                } else {
                    tiled::matmul_gemv
                };
                let us = gpu.time_graph(2 * copies as u32, || {
                    let (tc, ts, tb) = tiles[at % copies];
                    at += 1;
                    entry(&ctx, xt, tc, ts, tb, &mut y, GroupSeat::RESIDENT).unwrap();
                });
                let flops = 2.0 * f64::from(rows) * f64::from(n) * f64::from(k);
                eprintln!(
                    "  tiled {n}x{k} rows {rows:>4}: {us:8.1} us  {:5.2} TB/s  {:6.1} TFLOP/s",
                    bytes / us * 1e-6,
                    flops / us * 1e-6
                );
            }
        }
        // The bf16 reference: a [n·bits/16, k] weight, the same bytes.
        let rows_bf16 = n * bits / 16;
        let ws: Vec<_> = (0..copies)
            .map(|_| {
                gpu.tensor(
                    &values((rows_bf16 * k) as usize, 2, 0.05),
                    rows_bf16,
                    k,
                    Dtype::Bf16,
                )
            })
            .collect();
        let a = gpu.tensor(&values(k as usize, 1, 1.0), 1, k, Dtype::Bf16);
        let mut y = gpu.zeros(1, rows_bf16, Dtype::Bf16);
        let mut at = 0;
        let us = gpu.time_graph(2 * copies as u32, || {
            kernels_rocm::linear::gemm::matmul(&ctx, a, ws[at % copies], &mut y).unwrap();
            at += 1;
        });
        eprintln!(
            "  bf16 {rows_bf16}x{k} rows    1: {us:8.1} us  {:5.2} TB/s",
            f64::from(rows_bf16 * k * 2) / us * 1e-6
        );
    }
}
