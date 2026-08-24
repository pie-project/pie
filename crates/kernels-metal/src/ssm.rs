use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;

use crate::plane::{self, Handle};
use crate::routine::{Bind, Const, Ctx, Fire, In, Out, bf16};

fn head_rows(rows: i32, v_heads: i32) -> Result<u32, Refusal> {
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

/// The recurrence's grid: one simdgroup per value channel, per (row, head).
///
/// `gdn_core.metal` walks the state matrix a channel at a time, so the y
/// extent is the value dimension and the z extent is every (row, head) pair
/// there is.
pub fn core_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([32, v_dim.unsigned_abs(), z])
}

/// The prologue's grid: one simdgroup per (row, head), no channel axis.
pub fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([32, 1, head_rows(rows, v_heads)?])
}

/// The recurrence's threadgroup: FOUR simdgroups deep.
///
/// The four cooperate on one channel's state row; the prologue takes
/// [`simd_group`] because it has no state row to share.
pub const fn core_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 4, 1]
}

/// One simdgroup, the width the grid's x extent already states.
pub const fn simd_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

/// The chunked scan's arm, as an index into a nine-name list.
///
/// NINE TILINGS AND NOT A PRODUCT, which is why this is a match and not two
/// axis checks: `ssm/gdn_core.metal` instantiates its prefill scan at the
/// `(lane width, rows per lane group)` pairs its shared-memory budget admits —
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
        (4 | 8 | 16 | 32, _) => Err(Refusal::Narrow {
            what: "scan rows per lane group, at this lane width",
            at: i64::from(vrows),
        }),
        _ => Err(Refusal::Narrow {
            what: "scan lane width",
            at: i64::from(lanes),
        }),
    }
}

/// The chunked scan's grid, from the tiling the arm was compiled at.
///
/// EACH THREADGROUP OWNS `(32 / lanes) * vrows` CHANNELS — the lane width
/// divides the simdgroup into that many independent scans and each carries
/// `vrows` of them — so the channel axis is the value dimension over that
/// product, and the heads ride z.
pub fn scan_grid(v_dim: i32, v_heads: i32, lanes: i32, vrows: i32) -> Result<[u32; 3], Refusal> {
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    if lanes <= 0 || vrows <= 0 {
        return Err(Refusal::Empty {
            what: "the scan tiling",
        });
    }
    let per_y = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    if per_y == 0 {
        return Err(Refusal::Empty {
            what: "the scan tiling",
        });
    }
    Ok([
        32,
        v_dim.unsigned_abs().div_ceil(per_y),
        v_heads.unsigned_abs(),
    ])
}

/// The threadgroup a per-head prologue runs in.
///
/// One lane per value head, capped at the widest threadgroup Metal will take.
/// The grid is `[v_heads, rows, 1]` and Metal dispatches EXACTLY the threads
/// asked for, so the cap only splits the row across threadgroups — it never
/// leaves a head unwritten.
fn head_lanes(v_heads: i32) -> Result<u32, Refusal> {
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    Ok(v_heads.unsigned_abs().min(256))
}

/// The `Ssm` family, claimed — one point of seven, and the one is the one
/// W10 rewrote.
///
/// Six points stay on the floor's default body, and the six absences are two
/// seams and a family:
///
/// * `ssm.causal_conv1d` / `ssm.causal_conv1d_chunked` — SEAM: THE
///   CONVOLUTION IS NOT A LAUNCH ON THIS PLANE. Both `gdn_prep*` and both
///   `gdn_core*` run their own depthwise conv inline (over the channels each
///   one owns) and write their half of `new_conv_state` as they go; no
///   `.metal` entrypoint takes a `[C, K]` bank, a conv-state ring and a
///   stated width and does that and nothing else. Cuda's
///   `causal_conv1d_update_batched` / `_prefill_batched` are the shape these
///   want.
/// * `ssm.gated_delta` / `ssm.gated_delta_chunked` — SEAM: the metal
///   recurrences read the PRE-W10 STAGING. `gdn_core_recurrent_slotted` and
///   `gdn_core_recurrent_prefill` take `pre_q`, `pre_k` and `pre_gate` — the
///   three f32 scratch planes `gdn_prep_slotted` wrote — while the points
///   state the packed post-convolution `qkv`, the gate row and the packed
///   `[g_log | beta]` decay row, and expect the cut from those to the
///   recurrence's compact planes to happen INSIDE the launch. That cut is
///   cuda's `qwen_gdn_v_gates` (`GdnShape::stage`) and this tree has no
///   counterpart; a body that offset into the packed rows by hand would be
///   claiming a row stride of `v_heads` for bytes whose stride is
///   `2 * v_heads` — true at one token, false at two, which is the exact
///   defect W10 was written to remove.
/// * `ssm.kda_step` / `ssm.kda_chunked` — SEAM: kimi's delta attention, and
///   the `.metal` tree carries no KDA kernel in any form.
#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {
    /// Qwen's gated-delta prologue: the packed `[b | a]` projection in, the
    /// packed `[g_log | beta]` decay row out.
    ///
    /// ONE LAUNCH, AND EXACTLY THE DECLARATION'S SLOTS — which is what makes
    /// this a claim and not a delegation. `ssm/gdn_prep.metal` still stamps
    /// the pre-W10 `gdn_prep_slotted` arm that answered this point's name
    /// through the retired `canon` row, and it is a different statement: it
    /// takes the post-mixer row this declaration has no slot for, reaches a
    /// recurrent view this declaration does not name, and writes five
    /// rectangles where this one states one.
    ///
    /// `qwen_gdn_ba_gates` is the arithmetic with the packing kept, ported
    /// from `kernels-cuda`'s `ssm/gated_delta_net_prep.cuh` slot for slot. It
    /// reads the projection as the matmul wrote it and writes the decay row
    /// as the two recurrence points read it.
    ///
    /// # `v_heads` is read, not stated
    ///
    /// The declaration states no scalar, and it does not need to: the operand
    /// IS `[b | a]`, so the value-head count is half its width. A `Const`
    /// restating it could disagree with the rectangle it divides.
    fn gdn_prep<T: Scalar>(
        &self,
        ba: In<Handle<T>>,
        dt_bias: Const<Handle<T>>,
        a_log: Const<Handle<f32>>,
        gates: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`ssm.gdn_prep`, at an element this plane does not stamp";
        let ba = plane::input::<T, bf16>(ba, WHAT)?;
        let gates = plane::result::<f32, f32>(gates, "`ssm.gdn_prep`'s decay row")?;
        if ba.width % 2 != 0 {
            return Err(Refusal::Narrow {
                what: "the `[b | a]` projection's row, which halves into the value heads",
                at: i64::from(ba.width),
            });
        }
        let v_heads = ba.width / 2;
        // THE RESULT IS THE OPERAND'S SHAPE ON f32 — the width rule says so
        // and the kernel strides both by the same `2 * v_heads`, so a
        // rectangle that disagreed would be written past rather than
        // partially.
        if gates.width != ba.width || gates.rows != ba.rows {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the projection it is derived from",
                at: i64::from(gates.width),
            });
        }
        let lanes = head_lanes(v_heads)?;
        let rows = u32::try_from(ba.rows).map_err(|_| Refusal::Empty { what: "rows" })?;
        if rows == 0 {
            return Err(Refusal::Empty { what: "rows" });
        }
        self.fire(
            Fire::at("ssm/gdn_prep.metal", "qwen_gdn_ba_gates_bfloat16").apply(Grid::of(
                [v_heads.unsigned_abs(), rows, 1],
                [lanes, 1, 1],
            )),
            &[
                ba.arg(),
                plane::weight::<f32, f32>(a_log, "`ssm.gdn_prep`'s decay bank")?.arg(),
                plane::weight::<T, bf16>(dt_bias, WHAT)?.arg(),
                gates.arg(),
                v_heads.arg(),
            ],
        )
    }
}
