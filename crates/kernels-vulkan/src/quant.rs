//! The dense quantised matmuls, and the family this plane claims NOTHING
//! of.
//!
//! `kernels::points::Gemm` declares three points — `matmul`, `lm_head`,
//! `attention_landing` — and every one of them states its weight as ONE
//! slot: `w: Const<Self::Tensor<T>>`, an address and no rectangle. There is
//! no matmul on this plane that can be reached from one address.
//!
//! Every entrypoint in `quant/qmm_t.slang` and `quant/qmv.slang` binds
//! `StructuredBuffer<uint> w` for the packed codes, then `scales` and
//! `biases` at the activation element, and picks its module on the
//! `(group, bits)` pair — six affine combinations before the tile shape is
//! chosen. There is no dense bf16 gemm here at all: a `_bfloat16` in these
//! names is the ACTIVATION's element, never the weight's.
//!
//! That is the `Bank<R: Repr>` gap, and it is on baker's ledger by name
//! (`.wiki/baker-todo.md`: "`moe.matmul_select_bias` — grouped mxfp4+bias
//! gemm (real kernel work) + `Bank<R: Repr>` floor type + blocks+scales
//! import verb"). [`crate::points::Bank`] is this crate's stub for it, and
//! [`crate::points::Staged::bank`] is where a body says which three planes
//! it wanted; [`crate::moe`]'s two routed matmuls are already written
//! against both, so the day the floor carries a bank the `Gemm` family is
//! three short bodies and not a design question.
//!
//! `kernels-cuda` does not claim this family either, for a different
//! reason — its dense gemm goes through cuBLAS and the migration order in
//! `.wiki/baker.md` puts `Gemm` after `MLP`. So an unclaimed `Gemm` here is
//! not this plane falling behind; it is the same row on two ledgers.
//!
//! No routine in this file was ever superseded by an impl body, because
//! none could be. Thirty-one of them stood here and all thirty-one are
//! gone with the driver that fired them by name; what is kept is the two
//! grids below, which is the arithmetic those launches carried that a
//! future `Gemm` body cannot read off a shader.
//!
//! # The three pipelines the launches encoded
//!
//! * **split-K.** `affine_qmm_t_splitk*` writes PARTIALS at the k-split
//!   depth and `qmm_splitk_reduce_bfloat16` folds them — two launches, and
//!   the second is elementwise over the result rather than tiled. The
//!   `_f32` sibling of each is the same cut with an f32 accumulator plane.
//!   Only `bn = 32` is stamped for the split forms.
//! * **fp16 precast.** `cast_qmm_input_bfloat16_to_float16` casts the
//!   activation to half FIRST, then an `_fp16_precast` gemm reads it —
//!   again two launches, and the precast family is stamped at `gs_64_b_4`
//!   alone, so the group and bit axes are constants in its names.
//! * **the wide matvec.** `affine_qmv_wide_strided_bfloat16_gs_64_b_*_v_4_kl_8`
//!   does four outputs per lane over eight k per lane, so its vector count
//!   is the row count ceil-divided by four before it reaches [`qmv_grid`].
//!
//! # There is no name table here any more, and that is the point
//!
//! A `composable()` stood above, rebuilding the entrypoint names out of
//! their axes, and nothing anywhere read it. It produced 291 names;
//! `quant/qmm_t.slang` and `quant/qmv.slang` stamp 483 between them, so
//! the second copy had ALREADY drifted from the first and no test could
//! have caught it. The authority is `// pie:instantiate`, which `build.rs`
//! walks into `module::CENSUS`, and `kernels_vulkan::entrypoints()` is
//! that census.

use kernels::routine::Refusal;

/// The affine matmul's grid: column tiles by row tiles by k splits, each
/// multiplied out by the extent the 32x2x2 workgroup covers on that axis.
///
/// `quant/qmm_t.slang` declares `[numthreads(32, 2, 2)]` and a dispatch on
/// this plane is in TOTAL THREADS, so a tile count becomes lanes by that
/// axis's group extent. `split_k` is 1 for every non-split form.
pub fn qmm_grid(n: i32, bn: i32, m: i32, bm: i32, split_k: i32) -> Result<[u32; 3], Refusal> {
    let tiles = |extent: i32, tile: i32, what: &'static str| -> Result<u32, Refusal> {
        if extent <= 0 {
            return Err(Refusal::Empty { what });
        }
        if tile <= 0 {
            return Err(Refusal::Empty { what: "the tile" });
        }
        u32::try_from(extent)
            .map(|e| e.div_ceil(tile.unsigned_abs()))
            .map_err(|_| Refusal::Grid {
                what,
                at: i64::from(extent),
            })
    };
    if split_k <= 0 {
        return Err(Refusal::Empty {
            what: "the k split",
        });
    }
    let x = tiles(n, bn, "the column count")?;
    let y = tiles(m, bm, "the row count")?;
    let z = split_k.unsigned_abs();
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, Refusal> {
        groups.checked_mul(local).ok_or(Refusal::Grid {
            what,
            at: i64::from(groups),
        })
    };
    Ok([
        lanes(x, 32, "the column tiles")?,
        lanes(y, 2, "the row tiles")?,
        lanes(z, 2, "the k splits")?,
    ])
}

/// The matvec's grid: 64 lanes per vector, eight output rows per group.
///
/// `quant/qmv.slang` declares `[numthreads(PIE_LANES, 2, 1)]`, so the y
/// extent is the output width over the eight rows a group covers, times the
/// two the group is deep. The `vecs` a caller passes is the row count for
/// the plain matvecs and the row count over four for the wide strided one,
/// which does four outputs per lane.
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
    let x = vecs.unsigned_abs().checked_mul(64).ok_or(Refusal::Grid {
        what: "the vectors",
        at: i64::from(vecs),
    })?;
    let y = out_vec_size
        .unsigned_abs()
        .div_ceil(8)
        .checked_mul(2)
        .ok_or(Refusal::Grid {
            what: "the output rows",
            at: i64::from(out_vec_size),
        })?;
    Ok([x, y, 1])
}
