//! The routed swiglu MLP kept by expert over hipBLASLt's grouped GEMM
//! kernels (`hip::lt_grouped`) against its steps. Run these on a ROCm
//! bump: they are what holds the argument contract.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::hip::{self, lt_grouped};
use kernels_rocm::linear::moe;
use kernels_rocm::tuning::Tuning;

fn f16_bits(v: f32) -> u16 {
    // Round-to-nearest-even f32 → f16 for the normal range the tests use.
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    let mut h = (u32::from(sign) << 16 >> 16) | ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h as u16
}

fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    if exp == 0 {
        sign * man * 2f32.powi(-24)
    } else {
        sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
    }
}

fn bits(dtype: Dtype, v: f32) -> u16 {
    match dtype {
        Dtype::Bf16 => common::bf16(v),
        Dtype::F16 => f16_bits(v),
        _ => unreachable!(),
    }
}

fn widen(dtype: Dtype, b: u16) -> f32 {
    match dtype {
        Dtype::Bf16 => common::from_bf16(b),
        Dtype::F16 => from_f16(b),
        _ => unreachable!(),
    }
}

/// Device planes of 16-bit values, read back as they are.
#[derive(Default)]
struct Planes {
    held: Vec<hip::Buffer>,
}

impl Planes {
    fn up(&mut self, v: &[u16], rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let b = hip::Buffer::from_slice(v).unwrap();
        let t = Tensor::new(b.device_ptr(), rows, width, dtype);
        self.held.push(b);
        t
    }
    fn ints(&mut self, v: &[i32], rows: u32, width: u32) -> Tensor {
        let b = hip::Buffer::from_slice(v).unwrap();
        let t = Tensor::new(b.device_ptr(), rows, width, Dtype::I32);
        self.held.push(b);
        t
    }
    fn read(&self, gpu: &Gpu, t: &Tensor) -> Vec<u16> {
        gpu.sync();
        let b = self.held.iter().find(|b| b.device_ptr() == t.ptr).unwrap();
        b.read::<u16>().unwrap()[..t.elements() as usize].to_vec()
    }
}

/// Distinct top-k routes over `experts`, every `dead`-th one -1.
fn routes(tokens: usize, top_k: usize, experts: usize, seed: u64, dead: usize) -> Vec<i32> {
    let keys = values(tokens * experts, seed, 1.0);
    let mut out = Vec::with_capacity(tokens * top_k);
    let mut ids: Vec<usize> = (0..experts).collect();
    for t in 0..tokens {
        let row = &keys[t * experts..(t + 1) * experts];
        ids.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
        out.extend(ids[..top_k].iter().map(|&e| e as i32));
    }
    if dead > 0 {
        for i in (5..out.len()).step_by(dead) {
            out[i] = -1;
        }
    }
    // Under test, experts 3 and 9 take nothing: empty groups among the
    // full ones.
    if dead > 0 {
        for e in &mut out {
            if *e == 3 || *e == 9 {
                *e += 1;
            }
        }
    }
    out
}

#[test]
fn the_library_opens_at_a_known_version() {
    let Some(_gpu) = Gpu::open() else { return };
    let Ok(version) = lt_grouped::version() else {
        eprintln!("no hipBLASLt; skipping");
        return;
    };
    eprintln!("hipBLASLt version {version}");
    for dtype in [Dtype::Bf16, Dtype::F16] {
        match lt_grouped::library(dtype) {
            Ok(lib) => {
                assert!(!lib.solutions.is_empty());
                eprintln!("{dtype:?}: {} grouped solutions", lib.solutions.len());
            }
            Err(why) => eprintln!("{dtype:?}: not offered: {why}"),
        }
    }
}

/// One routed swiglu MLP's planes: the bank pair, the routes and weights,
/// and the steps' intermediate planes.
struct Mlp {
    p: moe::RoutedMlp,
    y: Tensor,
}

#[allow(clippy::too_many_arguments)]
fn mlp(
    planes: &mut Planes,
    dt: Dtype,
    tokens: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
    dead: usize,
) -> Mlp {
    let r = routes(tokens, top_k, experts, 11, dead);
    let count = r.len();
    let up = |v: Vec<f32>| -> Vec<u16> {
        v.iter()
            .map(|&v| bits(dt, if v == 0.0 { 1e-3 } else { v }))
            .collect()
    };
    let x = planes.up(
        &up(values(tokens * hidden, 1, 1.0)),
        tokens as u32,
        hidden as u32,
        dt,
    );
    let gate_up = planes.up(
        &up(values(experts * 2 * inter * hidden, 2, 0.05)),
        experts as u32,
        (2 * inter * hidden) as u32,
        dt,
    );
    let down = planes.up(
        &up(values(experts * hidden * inter, 3, 0.05)),
        experts as u32,
        (hidden * inter) as u32,
        dt,
    );
    let rt = planes.ints(&r, tokens as u32, top_k as u32);
    let w: Vec<f32> = values(count, 4, 0.5)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    let wb = hip::Buffer::from_slice(&w).unwrap();
    let weights = Tensor::new(wb.device_ptr(), tokens as u32, top_k as u32, Dtype::F32);
    planes.held.push(wb);
    let zeros = |rows: usize, w: usize| vec![0u16; rows * w];
    let packed = planes.up(
        &zeros(count, 2 * inter),
        count as u32,
        (2 * inter) as u32,
        dt,
    );
    let h = planes.up(&zeros(count, inter), count as u32, inter as u32, dt);
    let routed = planes.up(&zeros(count, hidden), count as u32, hidden as u32, dt);
    let y = planes.up(&zeros(tokens, hidden), tokens as u32, hidden as u32, dt);
    Mlp {
        p: moe::RoutedMlp {
            x,
            routes: rt,
            weights,
            gate_up,
            gate_up_experts: moe::ExpertTable::RESIDENT,
            down,
            down_experts: moe::ExpertTable::RESIDENT,
            intermediate: inter as u32,
            limit: None,
            fan: top_k as u32,
            packed,
            h,
            routed,
        },
        y,
    }
}

/// The tuning a path's profile would name: the hipBLASLt sorted MLP from
/// `sorted` routes an expert, the native expert-ordered MLP from `ordered`,
/// at the tiles given.
fn tuning(sorted: u32, ordered: u32, tiles: Option<(u32, u32)>) -> Tuning {
    let mut t = Tuning::DEFAULT;
    t.moe_lt_grouped = sorted > 0;
    t.moe_lt_mlp_from = sorted;
    t.moe_ordered_from = ordered;
    if let Some((deep, shallow)) = tiles {
        t.moe_lt_tile_deep = deep;
        t.moe_lt_tile_shallow = shallow;
    }
    t
}

#[test]
fn the_sorted_routed_mlp_answers_the_steps() {
    let Some(gpu) = Gpu::open() else { return };
    if lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    let dt = Dtype::Bf16;
    let mut planes = Planes::default();
    let (tokens, hidden) = (600usize, 320usize);
    let m = mlp(&mut planes, dt, tokens, 4, 8, hidden, 192, 9);
    let mut want = m.y;
    let mut have = planes.up(&vec![0; tokens * hidden], tokens as u32, hidden as u32, dt);
    let steps = gpu.ctx().with_tuning(tuning(0, 0, None));
    moe::mlp_swiglu(&steps, &m.p, &mut want).unwrap();
    // The leg takes contractions from `lt_deep_k`; these are short of the
    // profile's.
    let sorted = gpu.ctx().with_tuning(Tuning {
        moe_lt_deep_k: 64,
        ..tuning(1, 0, None)
    });
    let fired = moe::lt_grouped_fires();
    moe::mlp_swiglu(&sorted, &m.p, &mut have).unwrap();
    assert_eq!(
        moe::lt_grouped_fires(),
        fired + 2,
        "the sorted MLP did not run"
    );
    let (want, have) = (planes.read(&gpu, &want), planes.read(&gpu, &have));
    for (i, (&h, &w)) in have.iter().zip(&want).enumerate() {
        let (h, w) = (widen(dt, h), widen(dt, w));
        assert!(
            (h - w).abs() <= w.abs() / 32.0 + 0.02,
            "y[{}][{}] = {h}, steps {w}",
            i / hidden,
            i % hidden
        );
    }
}
