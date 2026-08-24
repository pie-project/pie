use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;

use crate::plane::{self, Handle};
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};

fn router_lanes(n_experts: u32) -> Result<u32, Refusal> {
    if n_experts == 0 {
        return Err(Refusal::Empty { what: "n_experts" });
    }
    Ok(n_experts.min(1024).div_ceil(32) * 32)
}

fn route_rows(width: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let w = width.unsigned_abs();
    Ok(([w, rows.unsigned_abs(), 1], [w.min(256), 1, 1]))
}

/// The routed MATVEC's grid: one simdgroup per row, four outputs per lane,
/// one z plane per routed slot.
///
/// A routed matvec runs the whole fan-out in one launch — `slots` is the
/// experts-per-token, and the z extent is what tells each threadgroup which
/// slot of the permutation it owns.
pub fn routed_qmv_grid(rows: i32, out_vec_size: i32, slots: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "out_vec_size",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty {
            what: "slots_per_row",
        });
    }
    let x = rows.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "rows * the simdgroup width",
        at: i64::from(rows) * 32,
    })?;
    Ok([
        x,
        out_vec_size.unsigned_abs().div_ceil(4),
        slots.unsigned_abs(),
    ])
}

/// The two widths a routed matvec states, from the rectangle and the fan-out.
///
/// THE OUTPUT WIDTH IS PER SLOT, not per row: the result rectangle holds
/// `slots` outputs side by side, so the width the kernel is told is the
/// rectangle's over the fan-out — and a width the fan-out does not divide is a
/// rectangle laid out for a different routing.
pub fn routed_qmv_widths(
    x_slot_stride: i32,
    y_width: i32,
    slots: i32,
) -> Result<(i32, i32), Refusal> {
    if x_slot_stride <= 0 {
        return Err(Refusal::Empty {
            what: "x_slot_stride",
        });
    }
    if y_width <= 0 {
        return Err(Refusal::Empty {
            what: "out_vec_size",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty {
            what: "slots_per_row",
        });
    }
    if !y_width.unsigned_abs().is_multiple_of(slots.unsigned_abs()) {
        return Err(Refusal::Narrow {
            what: "an output width the slot count does not divide",
            at: i64::from(y_width),
        });
    }
    Ok((x_slot_stride, y_width / slots))
}

/// The routed MATMUL's grid: `[32 * column tiles, 2 * row tiles, 2]`.
///
/// The tiles must DIVIDE, and for `quant/qmm_t.metal`'s reason rather than a
/// convenience: no entrypoint takes `m`, the shader reads it back off the
/// grid, and `load_unsafe` is the only path its hot loop takes.
pub fn routed_qmm_grid(rows: i32, n: i32, tile_m: i32, tile_n: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if n <= 0 {
        return Err(Refusal::Empty { what: "n" });
    }
    let (m, bn) = (tile_m.unsigned_abs(), tile_n.unsigned_abs());
    if m == 0 || !rows.unsigned_abs().is_multiple_of(m) {
        return Err(Refusal::Narrow {
            what: "rows the row tile does not divide",
            at: i64::from(rows),
        });
    }
    if bn == 0 || !n.unsigned_abs().is_multiple_of(bn) {
        return Err(Refusal::Narrow {
            what: "an output width the column tile does not divide",
            at: i64::from(n),
        });
    }
    Ok([
        32 * (n.unsigned_abs() / bn),
        2 * (rows.unsigned_abs() / m),
        2,
    ])
}

/// The routed matmul's arm, as an index into a nine-name list.
///
/// THREE TILES BY THREE TILES, row-major in `(tile_m, tile_n)` over
/// `16, 32, 64` each — the shape every `*_qmm_t_routed*_bm_<m>_bn_<n>` list in
/// `moe/route.metal`'s instantiations is written in.
pub fn tile_point(tile_m: i32, tile_n: i32) -> Result<usize, Refusal> {
    let axis = |v: i32, what: &'static str| match v {
        16 => Ok(0),
        32 => Ok(1),
        64 => Ok(2),
        _ => Err(Refusal::Narrow {
            what,
            at: i64::from(v),
        }),
    };
    Ok(axis(tile_m, "the routed qmm's row tile")? * 3
        + axis(tile_n, "the routed qmm's column tile")?)
}

/// The sorted-order expert combine, and one of the two launches this plane
/// answers by SYMBOL rather than by point — see [`crate::CANON`] for the walk
/// that reads the pair and what dropping it would cost every metal lane.
#[allow(clippy::too_many_arguments)]
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: In<Tensor<bf16>>,
    expert_weights: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv: In<Tensor<i32>>,
    width: Const<u32>,
    experts_per_token: Const<u32>,
    out_pitch: Const<u32>,
    tokens: Const<i32>,
) -> Result<(), Refusal> {
    let y_width = y.width;
    let tokens = *tokens;
    let (lanes, group) = route_rows(y_width, tokens)?;
    ctx.fire(
        Fire::at("moe/route.metal", "combine_sorted").apply(Grid::of(lanes, group)),
        &[
            y.arg(),
            expert_weights.arg(),
            out.arg(),
            inv.arg(),
            width.arg(),
            experts_per_token.arg(),
            out_pitch.arg(),
        ],
    )
}

/// `softmax_over_all = 0`: the softmax denominator is the SELECTED logits, so
/// the `k` weights sum to one.
///
/// THE SAME NUMBERS CUDA'S ROUTER WRITES, and the two spellings meet because
/// softmax is shift-invariant: `::pie::moe::topk_softmax` normalises over
/// every expert, selects, and then divides the `k` survivors by their own
/// sum, which is `exp(x_i - m) / sum_{j in K} exp(x_j - m)` — the expression
/// this arm computes directly. `1` is the other reading (`norm_topk_prob:
/// false`), where the weights sum to less than one and scale the routed FFN's
/// contribution down with them; no point in this family states it.
const SOFTMAX_OVER_SELECTED: u32 = 0;

/// The `Moe` family, claimed — the router and the combine.
///
/// `moe.topk_softmax` LANDED HERE BY A SHADER EDIT, for the reason
/// `norm.rmsnorm_gated` did next door. The point declares
/// `weights: Out<Tensor<f32>>` — a router weight is a probability and every
/// fold that reads one multiplies in float — while `moe/route.metal` wrote
/// `device T*`, the activation element. `router_topk` is now templated over
/// the weight element as well as the logit element, with
/// `<bfloat, bfloat, ..>` keeping the two names and the ABI the legacy driver
/// fires and `<bfloat, float, false>` stamped beside them as
/// `router_topk_f32w_bfloat16`, which is the arm this body names. THE SHADER
/// IS METAL-COMPILE-UNVERIFIED: nothing in this checkout can build a `.metal`
/// file.
///
/// Five points stay on the floor's default body:
///
/// * `moe.topk_sigmoid` / `moe.topk_sqrt_softplus` — SEAM: no `.metal` router
///   scores with a sigmoid or with `sqrt(softplus(x))`, and neither takes the
///   correction bias, the renormalisation flag or the routed scaling factor
///   those two declare. `router_topk_scaled` is the softmax one with a
///   per-expert gain, which is gemma's decision and not either of these.
/// * `moe.weighted_sum` — SEAM: `combine_sorted` takes an
///   `inv: In<Tensor<i32>>`, the inverse of the permutation `route_sort`
///   built, and the declaration states three operands with no permutation
///   among them. This plane folds a SORTED fan-out where cuda folds a
///   token-batched one, so the point wants either a sort-free combine or a
///   slot for the permutation. The weight element was the second half of this
///   gap and is closed — `combine_sorted` could take the f32 plane the same
///   way `router_topk` now does — but the permutation is a kernel, not a
///   crossing.
/// * `moe.matmul_select` / `moe.matmul_select_bias` — SEAM: every routed GEMM
///   in `moe/route.metal` is QUANTIZED (`qmv_routed`, `qmm_t_routed`, the
///   mxfp4 pair): they take blocks, scales and biases where the declaration
///   states one `Const<Tensor<T>>`, and they take the sort's permutation and
///   its tile table beside them. The `Bank<R: Repr>` gap `layout.embed`
///   names, with a permutation on top. [`routed_qmv_grid`],
///   [`routed_qmm_grid`] and [`tile_point`] are the schedules those arms
///   ride, kept for the body that will state them.
#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    /// The top-`k` softmax router: one threadgroup per row, one lane per
    /// expert.
    ///
    /// THE PITCH IS THE WIDTH. `router_topk` reads a row at
    /// `logits_pitch != 0 ? logits_pitch : n_experts`, because a router
    /// reading a SLICE of a wider activation has a pitch that is not its
    /// expert count. A mark on this plane is a dense rectangle by the
    /// strideless-mark law, so the pitch this body states is the rectangle's
    /// own width — and the stated expert count has to be that width or one of
    /// the two is wrong about the same row.
    fn topk_softmax<T: Scalar>(
        &self,
        logits: In<Handle<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<Handle<i32>>,
        weights: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.topk_softmax`, at an element this plane does not stamp";
        let logits = plane::input::<T, bf16>(logits, WHAT)?;
        let routes = plane::result::<i32, i32>(routes, "`moe.topk_softmax`'s route plane")?;
        let weights = plane::result::<f32, f32>(weights, "`moe.topk_softmax`'s weight plane")?;
        let width = plane::stated(experts, "the expert count this router states")?;
        let k = plane::stated(top_k, "the fan-out this router states")?;
        if logits.width != width {
            return Err(Refusal::Narrow {
                what: "the router's row is not the expert count the statement states",
                at: i64::from(logits.width),
            });
        }
        // BOTH RESULTS STRIDE BY `k`, which the kernel reads off the stated
        // fan-out and not off either rectangle, so a rectangle that disagreed
        // would be written past rather than partially.
        if routes.width != k || weights.width != k {
            return Err(Refusal::Narrow {
                what: "a routed result is not the fan-out the statement states",
                at: i64::from(routes.width),
            });
        }
        let lanes = router_lanes(experts)?;
        let rows = u32::try_from(logits.rows).map_err(|_| Refusal::Empty { what: "rows" })?;
        self.fire(
            Fire::at("moe/route.metal", "router_topk_f32w_bfloat16")
                .apply(Grid::of([lanes, rows, 1], [lanes, 1, 1])),
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                // The per-expert gain is gemma's, and the SCALED
                // instantiation is the one that dereferences it; this arm
                // binds the slot because an argument table with a hole in it
                // is not something an encoder can be asked for.
                self.absent()?,
                experts.arg(),
                top_k.arg(),
                SOFTMAX_OVER_SELECTED.arg(),
                // THE PITCH AT THE KIND THE SHADER TAKES IT: a
                // `constant uint&`, and this rectangle's width is the stated
                // expert count checked above.
                experts.arg(),
            ],
        )
    }

    /// `y = routed + shared * sigmoid(gate)`. The declaration hands the
    /// gate over as the `[tokens, 1]` column the statement already
    /// projected, which is exactly what `shared_expert_combine` reads —
    /// this is the point the two shader planes' shape decided, and cuda is
    /// the one that cannot claim it.
    fn sigmoid_gate_add<T: Scalar>(
        &self,
        routed: In<Handle<T>>,
        shared: In<Handle<T>>,
        gate: In<Handle<T>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.sigmoid_gate_add`, at an element this plane does not stamp";
        let routed = plane::input::<T, bf16>(routed, WHAT)?;
        let (lanes, group) = route_rows(routed.width, routed.rows)?;
        self.fire(
            Fire::at("moe/route.metal", "shared_expert_combine").apply(Grid::of(lanes, group)),
            &[
                routed.arg(),
                plane::input::<T, bf16>(shared, WHAT)?.arg(),
                plane::input::<T, bf16>(gate, WHAT)?.arg(),
                plane::result::<T, bf16>(y, WHAT)?.arg(),
                routed.width.arg(),
            ],
        )
    }
}
