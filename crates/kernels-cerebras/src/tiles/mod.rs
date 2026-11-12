//! The tiles a plan places: the model's params, as the trace names them, cut into the arrays the
//! program's bindings copy onto the wafer (`<name>.npy`, what the engine's placer reads). Each
//! layout states its cut as a [`spec::Spec`] ([`spec_of`]: [`snake`], [`stacked`], [`runs`],
//! [`split`]) and [`cut::cut`] makes it; the params come from the checkpoint through the engine's import contract, so no
//! checkpoint name appears here.
//!
//! Rounding is numpy's: f32 -> f16 to nearest even (`astype(float16)`), bf16 -> f32 a 16-bit
//! shift, f32 -> bf16 a truncation.

pub mod cut;
pub mod runs;
pub mod snake;
pub mod spec;
pub mod split;
pub mod stacked;

use std::path::{Path, PathBuf};

use poem_ir::Dtype;

use super::compose::Layout;
use super::plan::Plan;

type R<T> = Result<T, String>;

/// One param of the trace as the checkpoint landed it.
#[derive(Clone, Copy, Debug)]
pub struct Param<'a> {
    pub shape: &'a [u64],
    pub dtype: Dtype,
    pub bytes: &'a [u8],
}

/// The trace's params, by name.
pub trait Params: Sync {
    fn get(&self, name: &str) -> Option<Param<'_>>;
}

impl<'a> Param<'a> {
    pub fn len(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values as f32 (bf16 widened, f32 as stored).
    pub fn f32(&self) -> R<Vec<f32>> {
        match self.dtype {
            Dtype::Bf16 => Ok(self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect()),
            Dtype::F32 => Ok(self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()),
            d => Err(format!("a {d:?} param where bf16 or f32 is cut")),
        }
    }

    /// The values as bf16 bits (f32 truncated).
    pub fn bf16_bits(&self) -> R<Vec<u16>> {
        match self.dtype {
            Dtype::Bf16 => Ok(self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect()),
            Dtype::F32 => Ok(self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| (u32::from_le_bytes(*b) >> 16) as u16)
                .collect()),
            d => Err(format!("a {d:?} param where bf16 or f32 is cut")),
        }
    }
}

/// `x` as IEEE half bits, rounded to nearest even (subnormals kept, overflow to infinity).
pub fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if man != 0 {
                0x200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut h = m >> shift;
        if rest > half || (rest == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// An element type of a staged array, as numpy names it.
pub trait Store: Copy {
    const DESCR: &'static str;
    fn put(self, out: &mut Vec<u8>);
}

impl Store for u16 {
    const DESCR: &'static str = "<u2";
    fn put(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Store for f32 {
    const DESCR: &'static str = "<f4";
    fn put(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

/// The `.npy` bytes of `data` at `shape`, header as numpy 1.x writes it (version 1.0, padded to
/// 64 bytes).
pub fn npy<T: Store>(shape: &[usize], data: &[T]) -> Vec<u8> {
    let dims = match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut h = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {dims}, }}",
        T::DESCR
    );
    let pad = 64 - (10 + h.len() + 1) % 64;
    h.push_str(&" ".repeat(pad % 64));
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for &v in data {
        v.put(&mut out);
    }
    out
}

/// Where a cut is written: `<dir>/<name>.npy`.
pub struct Out {
    pub dir: PathBuf,
}

impl Out {
    pub fn new(dir: &Path) -> R<Out> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Out {
            dir: dir.to_path_buf(),
        })
    }

    pub fn save<T: Store>(&self, name: &str, shape: &[usize], data: &[T]) -> R<()> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(format!("{name}: {} values for shape {shape:?}", data.len()));
        }
        let p = self.dir.join(format!("{name}.npy"));
        std::fs::write(&p, npy(shape, data)).map_err(|e| format!("{}: {e}", p.display()))
    }
}

/// The cut of the model's params into the tiles `plan`'s bindings place, as data ([`spec`]), by
/// its layout; `shapes` gives a param's shape where the trace holds it.
pub fn spec_of(plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<spec::Spec> {
    match Layout::of_shape(&plan.shape) {
        Layout::Snake => snake::tiles(plan, shapes),
        Layout::Stacked => stacked::tiles(plan, shapes),
        Layout::Runs => runs::tiles(plan, shapes),
        Layout::Split => split::tiles(plan, shapes),
    }
}

/// Cuts `params` for `plan` into `out`.
pub fn stage(plan: &Plan, params: &dyn Params, out: &Path) -> R<()> {
    let spec = spec_of(plan, &|n| params.get(n).map(|p| p.shape.to_vec()))?;
    cut::cut(&spec, params, &Out::new(out)?)
}

/// `2^k` exactly (subnormal down to `2^-149`, infinity past `2^127`).
pub fn exp2i(k: i32) -> f32 {
    match k {
        128.. => f32::INFINITY,
        -126..=127 => f32::from_bits(((k + 127) as u32) << 23),
        -149..=-127 => f32::from_bits(1 << (k + 149)),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::f16;

    /// Round to nearest even, subnormals, overflow, as numpy's astype(float16).
    #[test]
    fn f16_rounds_as_numpy() {
        for (x, h) in [
            (1.0f32, 0x3c00u16),
            (-2.0, 0xc000),
            (65504.0, 0x7bff),
            (65520.0, 0x7c00),
            (6.1035156e-05, 0x0400),
            (5.9604645e-08, 0x0001),
            (2.9802322e-08, 0x0000),
            (1.0009766, 0x3c01),
            (f32::from_bits(0x3f80_1000), 0x3c00),
            (f32::from_bits(0x3f80_3000), 0x3c02),
            (0.0, 0),
        ] {
            assert_eq!(f16(x), h, "{x}");
        }
    }
}
