//! Multimodal rotary embeddings: each rotated pair reads one of three
//! position axes `(t, h, w)` from an i32 `[rows, 3]` stream. Reference:
//! kernels-wgpu `rope/mrope.wgsl` (interleaved and blocked agree with
//! kernels-cuda `rope_mrope*`; CUDA refuses split).
#![allow(clippy::too_many_arguments)]

use crate::error::{Error, refuse};
use crate::hlo::{Func, Val};
use crate::tensor::{Operand, Stored};

use super::rope::{Turn, heads, int_positions, inv_freq, rotate_all};

pub const AXES: u32 = 3;

const OP: &str = "elemwise.rope_mrope";

fn validate(
    q: impl Stored,
    k: impl Stored,
    positions: impl Stored,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
) -> Result<(), Error> {
    int_positions(OP, positions)?;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the rotated prefix {rotary_dim} is not a whole number of pairs within the {head_dim}-wide head"
            ),
        ));
    }
    heads(OP, q.width(), head_dim)?;
    heads(OP, k.width(), head_dim)?;
    let stated: u32 = sections.iter().sum();
    if stated > head_dim / 2 {
        return Err(refuse(
            OP,
            format!(
                "the sections {sections:?} name {stated} pairs and a {head_dim}-wide head has {}",
                head_dim / 2
            ),
        ));
    }
    Ok(())
}

fn blocked_pairs(sections: [u32; AXES as usize], rotary_dim: u32) -> Result<u32, Error> {
    let total: u32 = sections.iter().sum();
    if total == 0 {
        return Err(refuse(
            OP,
            format!("the sections {sections:?} name no frequency pair"),
        ));
    }
    Ok((rotary_dim / 2).min(total))
}

/// Pair `i` of the first `r/2` reads axis `h` when `i % 3 == 1 && i < 3·s1`,
/// `w` when `i % 3 == 2 && i < 3·s2`, else `t`; pairs `(i, i + r/2)` at
/// `theta^(-2i/r)`: the rotated prefix is its own neox head, as upstream's
/// `apply_rotary_pos_emb` turns `x[..r]` (Qwen3.5's `r = d/4`), as the GPU
/// kernels do.
pub fn interleaved(
    f: &mut Func,
    q: Operand,
    k: Operand,
    positions: Operand,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<[Option<Val>; 2], Error> {
    validate(q, k, positions, sections, rotary_dim, head_dim)?;
    let half = (rotary_dim / 2) as usize;
    let mut t = Turn::new(head_dim as usize, AXES as usize);
    for i in 0..half {
        let axis = match i % 3 {
            1 if i < 3 * sections[1] as usize => 1,
            2 if i < 3 * sections[2] as usize => 2,
            _ => 0,
        };
        t.pair(i, i + half, axis, inv_freq(theta, i, rotary_dim));
    }
    let v = rotate_all(f, OP, &[q, k], positions, AXES, &t)?;
    Ok([v[0], v[1]])
}

/// Pairs `(i, i + d/2)` for `i < min(r/2, Σs)`, the sections taking pairs
/// in order `t, h, w`, each at `theta^(-2·within/Σs)`.
pub fn blocked(
    f: &mut Func,
    q: Operand,
    k: Operand,
    positions: Operand,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<[Option<Val>; 2], Error> {
    validate(q, k, positions, sections, rotary_dim, head_dim)?;
    let pairs = blocked_pairs(sections, rotary_dim)? as usize;
    let total: u32 = sections.iter().sum();
    let half = (head_dim / 2) as usize;
    let (s0, s1) = (sections[0] as usize, sections[1] as usize);
    let mut t = Turn::new(head_dim as usize, AXES as usize);
    for i in 0..pairs {
        let (axis, within) = if i < s0 {
            (0, i)
        } else if i < s0 + s1 {
            (1, i - s0)
        } else {
            (2, i - s0 - s1)
        };
        t.pair(i, i + half, axis, inv_freq(theta, within, total));
    }
    let v = rotate_all(f, OP, &[q, k], positions, AXES, &t)?;
    Ok([v[0], v[1]])
}

/// Each section owns `2·s` contiguous channels, rotated as its own neox half
/// pair `(before·2 + j, before·2 + j + s)` at `theta^(-j/s)`.
pub fn split(
    f: &mut Func,
    q: Operand,
    k: Operand,
    positions: Operand,
    sections: [u32; AXES as usize],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<[Option<Val>; 2], Error> {
    validate(q, k, positions, sections, rotary_dim, head_dim)?;
    let pairs = blocked_pairs(sections, rotary_dim)? as usize;
    let (s0, s1) = (sections[0] as usize, sections[1] as usize);
    let s: [usize; 3] = sections.map(|v| v as usize);
    let mut t = Turn::new(head_dim as usize, AXES as usize);
    for i in 0..pairs {
        let (axis, within, before) = if i < s0 {
            (0, i, 0)
        } else if i < s0 + s1 {
            (1, i - s0, s0)
        } else {
            (2, i - s0 - s1, s0 + s1)
        };
        let width = s[axis];
        if width == 0 {
            continue;
        }
        let lo = 2 * before + within;
        // theta^(-within/width), as `exp2(-(within/width) · log2 theta)`.
        let f = theta.powf(-(within as f32) / width as f32);
        t.pair(lo, lo + width, axis, f);
    }
    let v = rotate_all(f, OP, &[q, k], positions, AXES, &t)?;
    Ok([v[0], v[1]])
}
