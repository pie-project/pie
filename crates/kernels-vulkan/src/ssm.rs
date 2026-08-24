//! The gated-delta mixer, and the family this plane claims NOTHING of.
//!
//! Seven points stand in `kernels::points::Ssm` and every one of them is a
//! measured backlog row here. That is not an oversight and it is not for
//! want of kernels — `ssm/gdn_core.slang` and `ssm/gdn_prep.slang` between
//! them stamp sixteen entrypoints and the whole qwen3.5 mixer runs on them.
//! It is a DECOMPOSITION disagreement, and it is worth writing down once
//! rather than discovering three times.
//!
//! # What the text states
//!
//! `model/src/qwen_3_5/forward.rs` states three statements per mixer:
//!
//! ```text
//! qkv   = ssm.causal_conv1d(qkv, conv_w, conv_state, conv_kernel)
//! gates = ssm.gdn_prep(ba, dt_bias, a_log)
//! core  = ssm.gated_delta(qkv, z, gates, delta_state, k_heads, v_heads, k_dim, v_dim)
//! ```
//!
//! So `ssm.gated_delta`'s `qkv` is POST-convolution and its `gates` is the
//! packed `[g_log | beta]` row `ssm.gdn_prep` already cooked out of the
//! `[b | a]` projection. `kernels-cuda` claims all three at those exact
//! boundaries (W10: "`ssm.gdn_prep` is now ONE launch of a new
//! `qwen_gdn_ba_gates`").
//!
//! # What this plane's kernels are
//!
//! `gdn_core_slotted_bfloat16` takes `mixed` — the projection BEFORE the
//! convolution, with q/k/v at `q_off`/`k_off`/`v_off` inside a `conv_dim`
//! slice — plus `conv_w`, `conv_b`, `a_log`, `dt_bias` and the two gate
//! halves, and does the convolution, the l2 norms, the decay cook and the
//! recurrence in ONE launch. `gdn_prep_slotted_bfloat16` is the same thing
//! cut after the cook, writing three compact f32 planes (`pre_q`, `pre_k`,
//! `pre_gate`); `gdn_core_recurrent_slotted_bfloat16` reads those three
//! back and runs the scan — and still does the convolution itself, from
//! `mixed`.
//!
//! Only `gdn_core_recurrent_prefill_bfloat16_l_*_v_*` is a pure scan, and
//! it reads the three compact planes rather than the packed row and the
//! packed gates the declaration carries.
//!
//! So the cut this plane makes is `[projection → everything]` or
//! `[projection → cooked planes] + [cooked planes → state]`, and the cut
//! the floor declares is `[conv] + [ba → gates] + [qkv, gates → state]`.
//! Not one of the three declared points has an entrypoint whose OPERANDS
//! are its operands:
//!
//! * `ssm.causal_conv1d` / `_chunked` — no standalone convolution exists
//!   here; every arm fuses it into the mixer.
//! * `ssm.gdn_prep` — declares `ba`, `dt_bias`, `a_log` and no conv state.
//!   Every prep arm here reads `conv_state` and writes `new_conv_state`.
//! * `ssm.gated_delta` / `_chunked` — declares the post-conv packed row and
//!   the packed decay row. Every scan arm here either does the conv itself
//!   or reads three COMPACT f32 planes nothing declares.
//! * `ssm.kda_step` / `_chunked` — kimi's rule. No entrypoint at all.
//!
//! # The sixteen entrypoints, and the launch each takes
//!
//! `ssm/gdn_core.slang` stamps `gdn_core_bfloat16` and its `PIE_SLOTTED`
//! sibling; `ssm/gdn_prep.slang` stamps the other fourteen — three
//! `gdn_prep` arms (plain, `_slotted`, `_prefill`), two
//! `gdn_core_recurrent` arms (plain, `_slotted`) and the nine-way
//! `PIE_SCAN` family. `_slotted` is one extra binding, the `slots` plane
//! out of [`crate::views::RecurrentView`], which turns a dense row index
//! into a state-slab slot; the arms without it address the slab by row.
//!
//! The four recurrence arms take [`gdn_grid`], the three prep arms take
//! [`prep_grid`], and the nine chunked scans take [`scan_grid`].
//!
//! The grid helpers below are `pub` and callerless on purpose: they are
//! what a claim body would otherwise have to re-derive, and the body that
//! will state them cannot be written until the cut above is settled.
//!
//! # What would close it
//!
//! Either a `PIE_POSTCONV` instantiation of `gdn_prep.slang` that skips the
//! convolution and cuts a packed `[g_log | beta]` row (which makes
//! `gdn_prep` and a standalone conv both claimable), or the declarations
//! grow a fused point this plane could claim whole. The first is shader
//! work of a few lines and keeps the plan plane-agnostic; the second moves
//! one plane's fusion onto the floor, which `.wiki/baker.md` reserves for
//! tier-2 — an inherent method on `Ctx`, which is where a fused vulkan GDN
//! mixer belongs the day a text may name one.
//!
//! There is no `#[claims] impl Ssm for Ctx<'_>` below, and the absence is
//! the statement: seven unclaimed points, seven backlog rows, one reason.

use kernels::routine::Refusal;

/// The chunked scan's arm, as an index into a nine-name list.
///
/// NINE TILINGS AND NOT A PRODUCT, which is why this is a match and not two
/// axis checks: `ssm/gdn_prep.slang` stamps its `PIE_SCAN` family at the
/// `(PIE_LANES, PIE_VROWS)` pairs its shared-memory budget admits —
/// `(16,1) (16,2) (16,4) (32,2) (32,4) (32,8) (4,1) (8,1) (8,2)`, in that
/// order — and the combinations between them were never compiled.
pub fn scan_point(lanes: i32, vrows: i32) -> Result<usize, Refusal> {
    match (lanes, vrows) {
        (16, 1) => Ok(0),
        (16, 2) => Ok(1),
        (16, 4) => Ok(2),
        (32, 2) => Ok(3),
        (32, 4) => Ok(4),
        (32, 8) => Ok(5),
        (4, 1) => Ok(6),
        (8, 1) => Ok(7),
        (8, 2) => Ok(8),
        _ => Err(Refusal::Narrow {
            what: "no gdn scan is compiled for this (LANES, VROWS)",
            at: i64::from(lanes) * 100 + i64::from(vrows),
        }),
    }
}

/// Every (row, head) pair this fire has, which is the z extent of every
/// gated-delta launch.
pub fn head_rows(rows: i32, v_heads: i32) -> Result<u32, Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    let n = u64::from(rows.unsigned_abs()) * u64::from(v_heads.unsigned_abs());
    u32::try_from(n).map_err(|_| Refusal::Grid {
        what: "rows * v_heads",
        at: i64::try_from(n).unwrap_or(i64::MAX),
    })
}

/// The recurrence's grid: one 32-lane group per value channel, per
/// (row, head).
///
/// The mixer walks the state matrix a channel at a time, so y is the value
/// dimension and z is every (row, head) pair there is.
pub fn gdn_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([32, v_dim.unsigned_abs(), z])
}

/// The prologue's grid: one 32-lane group per (row, head), no channel axis.
///
/// The cook is per head and not per channel — it produces `pre_q`, `pre_k`
/// and `pre_gate`, which are compact — so the axis [`gdn_grid`] spends on
/// the value dimension collapses to one here.
pub fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([32, 1, head_rows(rows, v_heads)?])
}

/// The chunked scan's grid, from the tiling the arm was compiled at.
///
/// EACH WORKGROUP OWNS `(32 / lanes) * vrows` CHANNELS — the lane width
/// divides the 32-wide group into that many independent scans and each
/// carries `vrows` of them — so the channel axis is the value dimension
/// over that product, and the heads ride z. This is the one grid on this
/// plane that a shader name alone does not give you: the name states
/// `(lanes, vrows)` and this states what they buy.
pub fn scan_grid(v_dim: i32, v_heads: i32, lanes: i32, vrows: i32) -> Result<[u32; 3], Refusal> {
    scan_point(lanes, vrows)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    let per_group = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    Ok([
        32,
        v_dim.unsigned_abs().div_ceil(per_group),
        v_heads.unsigned_abs(),
    ])
}
