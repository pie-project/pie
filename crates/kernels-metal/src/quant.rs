use kernels::routine::Refusal;

/// Every `(file, entrypoint)` this crate STAMPS rather than finds declared.
///
/// # Why the crate answers this at all
///
/// Because something downstream asks *"is this string an entrypoint this tree
/// can produce"*, and for these four forms the shader no longer knows. It
/// declares what it CAN stamp; the host decides which points to stamp, so the
/// host is the only party that can enumerate them. `kernels_metal::kernel_of`
/// is the asker: it maps an instantiated symbol out of a model text back to
/// the declaration that names it, and its census used to come entirely from
/// `build.rs` expanding the `.metal`'s own instantiation lists.
///
/// # It is not a table
///
/// It is the product of the axes through [`qmm_point`] -- the same function a
/// fire calls -- so a name here is a name a fire can reach, by construction.
/// That is the difference between this and the fifty-four-line list it
/// replaced in `moe.rs`.
///
/// The forms NOT here are the ones `quant/qmm_t.metal` still instantiates
/// itself (`_splitk`, `_strided`, the `_fp16_precast` family) and the matvecs.
/// Those stay in `build.rs`'s census until they move too.
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

/// An entrypoint this crate can reach, and the line that makes it exist.
///
/// Two renderings of one set of numbers. [`Self::stamp`] is composed by the
/// same call that composes [`Self::entry`] and embeds it, so the name a fire
/// asks for and the name the shader exports cannot disagree -- which is the
/// whole of what `quant/qmm_t.metal`'s deleted instantiation list, `moe.rs`'s
/// table of the same names, and a fixture comparing the two were for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    /// The symbol a fire names.
    pub entry: &'static str,
    /// The `PIE_STAMP_*(..)` call that declares it, or empty where the file
    /// still declares the form itself. See [`kernels::routine::Fire::stamp`].
    pub stamp: &'static str,
}

/// The K tile every point of this family is stamped at.
///
/// Not an axis: `instantiate_qmm_t`'s `bk` argument was 32 at all fifty-four
/// call sites. A parameter for a constant reads as a choice a caller has.
const QMM_BK: i32 = 32;

/// The affine matmul's point on its four axes.
///
/// `form` is the variant's own infix -- `""`, `"_bias"`, `"_residual"`,
/// `"_routed"` -- and `stamp` names the `#define` in `quant/qmm_t.metal` that
/// stamps that form. **An empty `stamp` means the file still declares the form
/// itself**, which is the four `_splitk` and `_strided` variants: they have
/// their own `#define instantiate_*` blocks with their own call lists, and
/// they keep working untouched. That is what makes this migration a family at
/// a time rather than a file at once.
///
/// # The axis checks stay, and mean something sharper now
///
/// They used to mirror the shader's own call list -- a second copy of it, in
/// Rust. There is no list to mirror any more, so what they are is the bound on
/// what this host will ASK to be stamped: a tile the template cannot serve is
/// now a Metal compile error at load rather than a `Refusal` at the fire, and
/// this is what keeps that from being reachable by a stray number.
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
    let entry = kernels::jit::symbol(&format!(
        "affine_qmm_t{form}_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    ));
    Ok(Point {
        entry,
        stamp: if stamp.is_empty() {
            ""
        } else {
            kernels::jit::symbol(&format!(
                "{stamp}(\"{entry}\", {group}, {bits}, {bm}, {QMM_BK}, {bn})"
            ))
        },
    })
}

/// [`qmm_point`] for a form the file still declares, discarding the stamp.
pub fn qmm_name(
    form: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Refusal> {
    Ok(qmm_point(form, "", group, bits, bm, bn)?.entry)
}

/// The precast form's name, which is stamped at ONE group size and ONE bit
/// width.
///
/// `_fp16_precast` means the activation was cast to half before the tile loop,
/// and `quant/qmm_t.metal` instantiates it only at `gs_64_b_4` — so the two
/// axes that vary are the tiles, and the other two are in the name.
pub fn qmm_precast_name(
    before: &str,
    after: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Refusal> {
    check(&[16, 32, 64], bm, "the row tile")?;
    check(&[16, 32, 64], bn, "the column tile")?;
    Ok(kernels::jit::symbol(&format!(
        "affine_qmm_t{before}_fp16_precast{after}_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

/// The wide strided matvec's name: four outputs per lane, eight k per lane,
/// at one group size.
pub fn qmv_wide_strided_name(bits: i32) -> Result<&'static str, Refusal> {
    check(&[4, 8], bits, "the bit width")?;
    Ok(kernels::jit::symbol(&format!(
        "affine_qmv_wide_strided_bfloat16_gs_64_b_{bits}_v_4_kl_8"
    )))
}

/// The matvec's name on its two axes. `form` is the variant's infix —
/// `fast`, `fast_residual`, `tail`, `tail_bias`.
pub fn qmv_name(form: &str, group: i32, bits: i32) -> Result<&'static str, Refusal> {
    check(&[32, 64, 128], group, "the group size")?;
    check(&[4, 8], bits, "the bit width")?;
    Ok(kernels::jit::symbol(&format!(
        "affine_qmv_{form}_bfloat16_gs_{group}_b_{bits}"
    )))
}

fn check(points: &[i32], v: i32, what: &'static str) -> Result<(), Refusal> {
    points.contains(&v).then_some(()).ok_or(Refusal::Narrow {
        what,
        at: i64::from(v),
    })
}

/// The affine matmul's grid: column tiles by row tiles by k splits, each
/// scaled by the simdgroup shape the tile loop runs in.
///
/// THE TILES MUST DIVIDE, and the refusals below say why in the shader's own
/// words: `qmm_t.metal` states `M % BM == 0, N % BN == 0 and K % BK == 0` as
/// the condition under which a driver may select it at all.
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

/// The matvec's grid: one simdgroup per vector, four outputs per lane.
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

#[cfg(test)]
mod stamping {
    use super::*;
    use std::collections::BTreeSet;

    /// A STAMP CARRIES THE NAME IT DECLARES, which is the one thing the two
    /// renderings must share.
    ///
    /// The whole argument for composing the stamp beside the entry is that
    /// they cannot drift. This is that claim, checked rather than asserted in
    /// a comment -- and it is checked over the PRODUCT, because a fold that
    /// was right at one point and wrong at another is the defect the
    /// fifty-four-line tables actually shipped.
    #[test]
    fn every_stamp_declares_the_entry_it_is_paired_with() {
        for form in ["", "_bias", "_residual", "_routed"] {
            for &gs in &[32, 64, 128] {
                for &b in &[4, 8] {
                    for &bm in &[16, 32, 64] {
                        for &bn in &[16, 32, 64] {
                            let p = qmm_point(form, "PIE_STAMP_x", gs, b, bm, bn)
                                .expect("an axis point");
                            assert!(
                                p.stamp.contains(&format!("\"{}\"", p.entry)),
                                "{} does not declare {}",
                                p.stamp,
                                p.entry
                            );
                        }
                    }
                }
            }
        }
    }

    /// TWO POINTS ARE TWO NAMES. An entry that two coordinates share is an
    /// aliasing: the grid is built for one shape and the pipeline computes
    /// another, which is how gemma4's logits came back all zero.
    #[test]
    fn distinct_points_compose_distinct_entries() {
        let composed = composed();
        let distinct: BTreeSet<&str> = composed.iter().map(|(_, name)| *name).collect();
        assert_eq!(distinct.len(), composed.len(), "two points share one name");
        assert_eq!(
            composed.len(),
            4 * 3 * 2 * 3 * 3,
            "the product is the census"
        );
    }

    /// THE STAMP IS THE FILE'S OWN LANGUAGE, not a signature restated here.
    ///
    /// What the host composes is a macro CALL; the `#define` it names holds
    /// the device parameter list, in `quant/qmm_t.metal`, written once. A
    /// stamp that spelled `const device uint32_t*` would have moved the ABI
    /// into Rust, which is the thing the instantiation lists were deleted for.
    #[test]
    fn a_stamp_is_a_macro_call_and_carries_no_signature() {
        let p = qmm_point("", "PIE_STAMP_qmm_t", 64, 4, 32, 32).expect("an axis point");
        assert_eq!(
            p.stamp,
            "PIE_STAMP_qmm_t(\"affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32\", 64, 4, 32, 32, 32)"
        );
    }

    /// AN UNMIGRATED FORM STAMPS NOTHING, and says so with an empty string --
    /// `Fire::stamp`'s "the file already declares it". `_splitk` and
    /// `_strided` keep their own `instantiate_*` lists in the shader.
    #[test]
    fn a_form_the_file_still_declares_carries_no_stamp() {
        let p = qmm_point("_splitk", "", 64, 4, 32, 32).expect("an axis point");
        assert!(p.stamp.is_empty());
        assert_eq!(
            p.entry,
            "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_32_bn_32"
        );
    }
}
