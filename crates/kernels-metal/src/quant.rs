use kernels::plane::Refusal;

#[must_use]
pub fn composed() -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    for form in ["", "_bias", "_residual", "_routed"] {
        for &gs in &[32, 64, 128] {
            for &b in &[4, 8] {
                for &bm in &[16, 32, 64] {
                    for &bn in &[16, 32, 64] {
                        let p = qmm_point(form, "", gs, b, bm, bn)
                            .expect("an axis point, by construction");
                        out.push(("quant/qmm_t.metal", p.entry));
                    }
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub entry: &'static str,

    pub stamp: &'static str,
}

const QMM_BK: i32 = 32;

pub(crate) fn qmm_point(
    form: &str,
    stamp: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<Point, Refusal> {
    check(&[32, 64, 128], group, "the group size")?;
    check(&[4, 8], bits, "the bit width")?;
    check(&[16, 32, 64], bm, "the row tile")?;
    check(&[16, 32, 64], bn, "the column tile")?;
    let entry = kernels::intern::symbol(&format!(
        "affine_qmm_t{form}_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    ));
    Ok(Point {
        entry,
        stamp: if stamp.is_empty() {
            ""
        } else {
            kernels::intern::symbol(&format!(
                "{stamp}(\"{entry}\", {group}, {bits}, {bm}, {QMM_BK}, {bn})"
            ))
        },
    })
}

pub fn qmm_name(
    form: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Refusal> {
    Ok(qmm_point(form, "", group, bits, bm, bn)?.entry)
}

pub fn qmm_precast_name(
    before: &str,
    after: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Refusal> {
    check(&[16, 32, 64], bm, "the row tile")?;
    check(&[16, 32, 64], bn, "the column tile")?;
    Ok(kernels::intern::symbol(&format!(
        "affine_qmm_t{before}_fp16_precast{after}_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

pub fn qmv_wide_strided_name(bits: i32) -> Result<&'static str, Refusal> {
    check(&[4, 8], bits, "the bit width")?;
    Ok(kernels::intern::symbol(&format!(
        "affine_qmv_wide_strided_bfloat16_gs_64_b_{bits}_v_4_kl_8"
    )))
}

pub fn qmv_name(form: &str, group: i32, bits: i32) -> Result<&'static str, Refusal> {
    check(&[32, 64, 128], group, "the group size")?;
    check(&[4, 8], bits, "the bit width")?;
    Ok(kernels::intern::symbol(&format!(
        "affine_qmv_{form}_bfloat16_gs_{group}_b_{bits}"
    )))
}

fn check(points: &[i32], v: i32, what: &'static str) -> Result<(), Refusal> {
    points.contains(&v).then_some(()).ok_or(Refusal::Narrow {
        what,
        at: i64::from(v),
    })
}

pub fn qmm_grid(n: i32, bn: i32, m: i32, bm: i32, split_k: i32) -> Result<[u32; 3], Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "the column count",
        });
    }
    if m <= 0 {
        return Err(Refusal::Empty {
            what: "the row count",
        });
    }
    if bn <= 0 || bm <= 0 {
        return Err(Refusal::Empty { what: "the tile" });
    }
    if split_k <= 0 {
        return Err(Refusal::Empty {
            what: "the k split",
        });
    }
    if m % bm != 0 {
        return Err(Refusal::Misaligned {
            what: "the row count, which the tile must divide because no \
                   entrypoint takes m and the shader reads it from the grid",
        });
    }
    if n % bn != 0 {
        return Err(Refusal::Misaligned {
            what: "the column count, which the tile must divide: `qmm_t.metal` \
                   states `M % BM == 0, N % BN == 0 and K % BK == 0` as the \
                   condition under which the driver may select it at all, and \
                   `load_unsafe` is the only path its hot loop takes",
        });
    }
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, Refusal> {
        groups.checked_mul(local).ok_or(Refusal::Grid {
            what,
            at: i64::from(groups),
        })
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

pub fn qmv_grid(vecs: i32, out_vec_size: i32) -> Result<[u32; 3], Refusal> {
    if vecs <= 0 {
        return Err(Refusal::Empty {
            what: "the vectors",
        });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "the output vector",
        });
    }
    let x = vecs.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "the vectors",
        at: i64::from(vecs),
    })?;
    Ok([x, out_vec_size.unsigned_abs().div_ceil(4), 1])
}
