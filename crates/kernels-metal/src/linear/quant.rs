//! `quant`: the jit-stamped affine qmm/qmv point namespace — entry names,
//! instantiation stamps, and their grids. Parked machinery: nothing selects
//! these points today — the caller that did died with the baker, and the one
//! quantized bank the moe entry fires is mxfp4, stamped in source
//! (`linear::moe::matmul_select_bias`), not composed here.
//!
//! Entry names are composed at runtime (group size x bit width x tile), so
//! they are interned to `&'static str` — the currency of [`Fire`] and of the
//! driver's pipeline cache.
//!
//! [`Fire`]: crate::encode::Fire

// MENLO-SEAM: reviving this namespace takes a declared bank repr reaching
// the driver — an affine `(codes, scales, biases)` three-plane weight row
// with its group size and bit width; the driver's `WeightRow::Planes` seats
// only mxfp4's two planes.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use kernels::KernelError;

use crate::encode::refuse;

/// Every qmm point a plane may fire, for the driver's warm-up census.
#[must_use]
pub fn composed() -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    for form in ["", "_bias", "_residual", "_routed"] {
        for &gs in &[32, 64, 128] {
            for &b in &[4, 8] {
                for &bm in &[16, 32, 64] {
                    for &bn in &[16, 32, 64] {
                        let p = qmm_point("quant.qmm_t", form, "", gs, b, bm, bn)
                            .expect("an axis point, by construction");
                        out.push(("linear/quant_qmm_t.metal", p.entry));
                    }
                }
            }
        }
    }
    out
}

/// One selected qmm instantiation: the entry symbol and the jit stamp that
/// conjures it when the shader source does not spell it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub entry: &'static str,

    pub stamp: &'static str,
}

const QMM_BK: i32 = 32;

pub fn qmm_point(
    op: &'static str,
    form: &str,
    stamp: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<Point, KernelError> {
    check(op, &[32, 64, 128], group, "group size")?;
    check(op, &[4, 8], bits, "bit width")?;
    check(op, &[16, 32, 64], bm, "row tile")?;
    check(op, &[16, 32, 64], bn, "column tile")?;
    let entry = symbol(&format!(
        "affine_qmm_t{form}_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    ));
    Ok(Point {
        entry,
        stamp: if stamp.is_empty() {
            ""
        } else {
            symbol(&format!(
                "{stamp}(\"{entry}\", {group}, {bits}, {bm}, {QMM_BK}, {bn})"
            ))
        },
    })
}

pub fn qmm_name(
    op: &'static str,
    form: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<&'static str, KernelError> {
    Ok(qmm_point(op, form, "", group, bits, bm, bn)?.entry)
}

pub fn qmm_precast_name(
    op: &'static str,
    before: &str,
    after: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, KernelError> {
    check(op, &[16, 32, 64], bm, "row tile")?;
    check(op, &[16, 32, 64], bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t{before}_fp16_precast{after}_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

pub fn qmv_wide_strided_name(op: &'static str, bits: i32) -> Result<&'static str, KernelError> {
    check(op, &[4, 8], bits, "bit width")?;
    Ok(symbol(&format!(
        "affine_qmv_wide_strided_bfloat16_gs_64_b_{bits}_v_4_kl_8"
    )))
}

pub fn qmv_name(
    op: &'static str,
    form: &str,
    group: i32,
    bits: i32,
) -> Result<&'static str, KernelError> {
    check(op, &[32, 64, 128], group, "group size")?;
    check(op, &[4, 8], bits, "bit width")?;
    Ok(symbol(&format!(
        "affine_qmv_{form}_bfloat16_gs_{group}_b_{bits}"
    )))
}

fn check(
    op: &'static str,
    points: &[i32],
    v: i32,
    what: &'static str,
) -> Result<(), KernelError> {
    points
        .contains(&v)
        .then_some(())
        .ok_or_else(|| refuse(op, format!("no point is stamped at {what} {v}")))
}

pub fn qmm_grid(
    op: &'static str,
    n: i32,
    bn: i32,
    m: i32,
    bm: i32,
    split_k: i32,
) -> Result<[u32; 3], KernelError> {
    if n <= 0 {
        return Err(refuse(op, "the column count is zero"));
    }
    if m <= 0 {
        return Err(refuse(op, "the row count is zero"));
    }
    if bn <= 0 || bm <= 0 {
        return Err(refuse(op, "the tile is zero"));
    }
    if split_k <= 0 {
        return Err(refuse(op, "the k split is zero"));
    }
    if m % bm != 0 {
        return Err(refuse(
            op,
            format!(
                "the row count is {m}, not a multiple of {bm}: the tile must \
                 divide it because no entrypoint takes m and the shader reads \
                 it from the grid"
            ),
        ));
    }
    if n % bn != 0 {
        return Err(refuse(
            op,
            format!(
                "the column count is {n}, not a multiple of {bn}: `quant_qmm_t.metal` \
                 states `M % BM == 0, N % BN == 0 and K % BK == 0` as the \
                 condition under which the driver may select it at all, and \
                 `load_unsafe` is the only path its hot loop takes"
            ),
        ));
    }
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, KernelError> {
        groups
            .checked_mul(local)
            .ok_or_else(|| refuse(op, format!("{what} will not launch at {groups} groups")))
    };
    Ok([
        lanes(
            n.unsigned_abs().div_ceil(bn.unsigned_abs()),
            32,
            "the column tiles",
        )?,
        lanes(m.unsigned_abs() / bm.unsigned_abs(), 2, "the row tiles")?,
        lanes(split_k.unsigned_abs(), 2, "the k splits")?,
    ])
}

pub fn qmv_grid(op: &'static str, vecs: i32, out_vec_size: i32) -> Result<[u32; 3], KernelError> {
    if vecs <= 0 {
        return Err(refuse(op, "the vectors are zero"));
    }
    if out_vec_size <= 0 {
        return Err(refuse(op, "the output vector is zero"));
    }
    let x = vecs
        .unsigned_abs()
        .checked_mul(32)
        .ok_or_else(|| refuse(op, format!("{vecs} vectors will not launch")))?;
    Ok([x, out_vec_size.unsigned_abs().div_ceil(4), 1])
}

/// Interns a composed name, so a runtime-built entry can live in a
/// `&'static str` field. Names are few (the axis grid above) and re-selected
/// every fire, so the leak is bounded and the map earns its keep.
fn symbol(name: &str) -> &'static str {
    static INTERNED: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = INTERNED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), leaked);
    leaked
}
