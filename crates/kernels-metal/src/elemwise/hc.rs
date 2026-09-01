//! `Hc`: hyper-connections — residual streams expanded, mixed by learned
//! gates, and folded back layer by layer. One entry per IR variant, all over
//! `elemwise/hc.metal`, which is `kernels-cuda/kernels/elemwise/hc.cuh` ported
//! organ for organ.
//!
//! The stream count `M` rides the row width — a `[N, M·H]` rectangle beside a
//! `[N, H]` one — and the mixers hold their `M` (or `M²`) coefficients in
//! register and threadgroup arrays, which is why [`MAX_HC_MULT`] is a hard
//! refusal and not a shape check.
//!
//! **THE SINKHORN COUNT IS A HALF-SWEEP PLUS `n - 1` FULL ONES.** The shader
//! seeds with a row softmax, does ONE column normalization, and then runs
//! `sinkhorn - 1` alternating row/column sweeps — 20 stated iterations are 19
//! loop passes. [`the_sinkhorn_count_is_the_seed_plus_one_less_sweep`] pins
//! that arithmetic against a host mirror of the same loop, because an
//! off-by-one there changes the combiner and nothing downstream complains.
//!
//! **AND THE MIX ROW IS THE PROJECTION NOW.** `hc_gates` reads its operand at
//! a stride of `2M + M²` and always has — that is the reference's mix row,
//! `rmsnorm(streams) @ hc_fn` — but for as long as no plane produced one, the
//! model text handed it `normed` itself, the `[N, M·H]` f32 rmsnorm of the
//! stream row, and both shells read its leading `2M + M²` floats. [`project`]
//! is that missing GEMM: it fires the dynamic hyper plane the model text used
//! to intern (`{attn,ffn}_hc.fn`) and lands a real `[N, 2M + M²]` mix row, so
//! the stride the gate reads is now the width of what it is given. The gate
//! entry below is unchanged — it never had to change; what changed is what
//! reaches it.
//!
//! The TRUNK head (`model.hc_head.*`) is still interned: its plane reduces
//! the streams once before the LM head, and this file has no entry for that
//! collapse (see the note where `Hc::Collapse` was deleted).

use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/hc.metal";

/// The threadgroup the per-row points are stamped for, and the shader's own
/// `BLOCK` template argument.
const BLOCK: u32 = 256;

/// The largest stream count the mixers unroll into register (`hc_fold`'s
/// read-before-write array) and threadgroup gate vectors — the shader's
/// `HC_MAX_MULT`.
const MAX_HC_MULT: u32 = 8;

/// The stream fan `M`: how many `hidden`-wide streams the wide row holds.
/// The variants that state a count beside their rows assert agreement at
/// their own entries; `fold` states none.
fn stream_fan(op: &'static str, wide: u32, hidden: u32) -> Result<u32, Error> {
    nonzero(op, "the hidden width", hidden)?;
    if wide == 0 || wide % hidden != 0 {
        return Err(refuse(
            op,
            format!(
                "the {wide}-wide row is not a whole number of {hidden}-wide \
                 hyper-connection streams"
            ),
        ));
    }
    let fan = wide / hidden;
    if fan > MAX_HC_MULT {
        return Err(refuse(
            op,
            format!(
                "the stream count is {fan}, above the {MAX_HC_MULT} the mixers unroll into \
                 register and threadgroup arrays"
            ),
        ));
    }
    Ok(fan)
}

/// One threadgroup per row: `BLOCK` threads apiece, laid on the x axis the way
/// this plane's other per-row points lay them (`dispatchThreads` counts
/// THREADS, not groups).
fn per_row(op: &'static str, rows: u32) -> Result<Grid, Error> {
    let rows = nonzero(op, "rows", rows)?;
    let lanes = rows
        .checked_mul(BLOCK)
        .ok_or_else(|| refuse(op, format!("the grid will not launch: {rows} rows x {BLOCK}")))?;
    Ok(Grid::of([lanes, 1, 1], [BLOCK, 1, 1]))
}

/// Tiles `x` across `streams` residual streams.
pub fn expand(ctx: &Ctx<'_>, x: Tensor, streams: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_expand";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "hc_expand_bfloat16" });
    debug_assert_eq!(y.rows, x.rows, "the expansion lands one wide row per row");
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

/// RMS-normalises the wide stream row and widens it to f32 — the mix
/// coefficients derived downstream are too sensitive for a bf16 round-trip.
pub fn rmsnorm_f32(ctx: &Ctx<'_>, streams: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_rmsnorm_f32";
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "hc_rmsnorm_f32_bfloat16" });
    debug_assert_eq!(y.dtype, dtype::Dtype::F32, "`{OP}` widens to f32");
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the normed rectangle is the stream rectangle"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(per_row(OP, y.rows)?),
        &[
            streams.arg(),
            y.arg_mut(),
            stated(OP, nonzero(OP, "the normed row's width", y.width)?)?.arg(),
            eps.arg(),
        ],
    )
}

/// The per-token mix row: `mixes = normed · hc_fn^T`, `[N, M·H]` against a
/// `[2M + M², M·H]` plane, all f32.
///
/// **NOT `linear.matmul`, AND THE SHADER'S OWN NOTE SAYS WHY** — the dense
/// gemm on this plane instantiates bf16 only, both operands here are f32, and
/// the fp32 discipline this file's header states is the whole reason the mix
/// row is f32 in the first place. One threadgroup per `(row, column)`.
pub fn project(
    ctx: &Ctx<'_>,
    normed: Tensor,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_project";
    debug_assert!(
        normed.dtype == dtype::Dtype::F32
            && hc_fn.dtype == dtype::Dtype::F32
            && mixes.dtype == dtype::Dtype::F32,
        "`{OP}` projects an f32 row through an f32 plane into an f32 row"
    );
    debug_assert_eq!(
        mixes.rows, normed.rows,
        "the projection lands one mix row per stream row"
    );
    let fan = nonzero(OP, "the stream row this projection contracts", normed.width)?;
    if hc_fn.width != fan {
        return Err(refuse(
            OP,
            format!(
                "the dynamic plane contracts {} and the stream row is {fan} wide",
                hc_fn.width
            ),
        ));
    }
    if stream_count == 0 || stream_count > MAX_HC_MULT {
        return Err(refuse(
            OP,
            format!(
                "the stream count is {stream_count}, not one of the {MAX_HC_MULT} the \
                 mixers unroll"
            ),
        ));
    }
    // The mix row the gate splits: `M` pre weights, `M` post weights and the
    // `M x M` combiner, in that order — the one width both ends agree on.
    let mix_hc = 2 * stream_count + stream_count * stream_count;
    if mixes.width != mix_hc || hc_fn.rows != mix_hc {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {mix_hc} wide; the plane lands {} \
                 rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
    // One threadgroup per `(row, column)`, laid on ONE axis: the shader's
    // point derives the pair, because a metal entry's position attributes
    // must all be scalars or all be vectors of one width and its three lane
    // indices are scalars.
    let lanes = rows
        .checked_mul(mix_hc)
        .and_then(|points| points.checked_mul(BLOCK))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("the grid will not launch: {rows} rows x {mix_hc} x {BLOCK}"),
            )
        })?;
    ctx.fire(
        Fire::at(FILE, "hc_project").apply(Grid::of([lanes, 1, 1], [BLOCK, 1, 1])),
        &[
            normed.arg(),
            hc_fn.arg(),
            mixes.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, mix_hc)?.arg(),
        ],
    )
}

/// Splits the mix row, Sinkhorn-normalises the combiner, and collapses the
/// `M` streams into the layer's input — one threadgroup per token, the gate
/// matrices landing beside it.
///
/// `normed` is the mix row [`project`] lands, `[N, 2M + M²]` — the stride the
/// shader reads it at, which is now the width of what it is handed.
#[allow(clippy::too_many_arguments)]
pub fn gates(
    ctx: &Ctx<'_>,
    normed: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    x: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_gates";
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "hc_gates_bfloat16" });
    debug_assert_eq!(normed.dtype, dtype::Dtype::F32, "`{OP}` reads an f32 mix row");
    debug_assert_eq!(scale.dtype, dtype::Dtype::F32, "`{OP}` reads f32 mix scales");
    debug_assert_eq!(base.dtype, dtype::Dtype::F32, "`{OP}` reads f32 mix bases");
    debug_assert!(
        post_mix.dtype == dtype::Dtype::F32 && comb_mix.dtype == dtype::Dtype::F32,
        "`{OP}` lands f32 gate matrices"
    );
    let fan = stream_fan(OP, streams.width, x.width)?;
    debug_assert_eq!(
        fan, stream_count,
        "the row's stream fan is the count the statement states"
    );
    debug_assert!(
        post_mix.width == fan && comb_mix.width == fan * fan,
        "the gate matrices are `[N, M]` and `[N, M, M]`"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(per_row(OP, x.rows)?),
        &[
            normed.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            post_mix.arg_mut(),
            comb_mix.arg_mut(),
            x.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, nonzero(OP, "the hidden width", x.width)?)?.arg(),
            gate_eps.arg(),
            alpha.arg(),
            stated(OP, sinkhorn)?.arg(),
        ],
    )
}

/// Mixes the layer's output back into the streams under the gate matrices —
/// across the wide row, which is why each thread owns its whole `(n, h)`
/// column and reads every stream before writing any.
pub fn fold(
    ctx: &Ctx<'_>,
    x: Tensor,
    streams: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_fold";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "hc_fold_bfloat16" });
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the fold lands the stream rectangle it mixes"
    );
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert!(
        post_mix.dtype == dtype::Dtype::F32 && comb_mix.dtype == dtype::Dtype::F32,
        "`{OP}` reads f32 gate matrices"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            elementwise_rows(OP, x.width, y.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            y.arg_mut(),
            stated(OP, y.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

// `collapse` went with `Hc::Collapse`: no plane could fire it honestly (review R5).

/// **THE GATED-RESIDUAL FLAVOR (qwen4).** Three entries, and the whole
/// difference from the four above is the gate: a low-rank GEMM chain produces
/// per-element logits, a sigmoid of them mixes, and there is no Birkhoff
/// projection and no f32 gate plane anywhere in it. The GEMMs are ordinary
/// linear nodes; these three say only the arithmetic no other op says.
/// [`reference`] states each of them a second time in host f32.
///
/// `y[h] = mean_s( σ(gates[s·H + h]) · normed[s·H + h] )` — the `M` streams
/// collapsed to the one row a sublayer reads.
pub fn mix(
    ctx: &Ctx<'_>,
    gates: Tensor,
    normed: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_mix";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "hc_mix_bfloat16" });
    let fan = stream_fan(OP, normed.width, y.width)?;
    if fan != streams {
        return Err(refuse(
            OP,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    debug_assert!(
        gates.width == normed.width && gates.rows == normed.rows,
        "`{OP}` gates the normed rectangle element for element"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            gates.arg(),
            normed.arg(),
            y.arg_mut(),
            stated(OP, y.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, y.width)?.arg(),
        ],
    )
}

/// `hyper[s·H + h] += 2·σ(gates[s]/M) · o[h]`, in place on the wide row —
/// the sublayer's output written back into every stream at its own depth
/// weight. The gate is per STREAM here and per element in [`mix`], which is
/// the asymmetry the two halves of a hyper-connection are built on.
pub fn inject(
    ctx: &Ctx<'_>,
    o: Tensor,
    gates: Tensor,
    streams: u32,
    hyper: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_inject";
    let entry = dtype_dispatch!(OP, hyper.dtype, { Bf16 => "hc_inject_bfloat16" });
    let fan = stream_fan(OP, hyper.width, o.width)?;
    if fan != streams {
        return Err(refuse(
            OP,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    debug_assert_eq!(
        gates.width, fan,
        "`{OP}` reads one gate logit per stream, not one per element"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            elementwise_rows(OP, o.width, hyper.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            o.arg(),
            gates.arg(),
            hyper.arg_mut(),
            stated(OP, hyper.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, o.width)?.arg(),
        ],
    )
}

/// **THE PLE GATE.** Per `(row, stream)`, `y = σ(sgn(d)·√max(|d|, 1e-6)) · v`
/// where `d` is the key·query dot over the stream's `H` values, scaled by
/// `1/√H`. One threadgroup per `(row, stream)`, flattened the way the grouped
/// norms flatten — group `b` is row `b / M`, stream `b % M`.
pub fn ple_gate(
    ctx: &Ctx<'_>,
    key: Tensor,
    query: Tensor,
    value: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.ple_gate";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "ple_gate_bfloat16" });
    let fan = stream_fan(OP, y.width, value.width)?;
    if fan != streams {
        return Err(refuse(
            OP,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    debug_assert!(
        key.width == y.width && query.width == y.width,
        "`{OP}` dots the key and query over the stream row it gates"
    );
    let groups = y.rows.checked_mul(fan).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {} rows x {fan} streams", y.rows),
        )
    })?;
    ctx.fire(
        Fire::at(FILE, entry).apply(per_row(OP, groups)?),
        &[
            key.arg(),
            query.arg(),
            value.arg(),
            y.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, value.width)?.arg(),
        ],
    )
}

/// The hyper-connection arithmetic, in host f32 — the deviceless mirror the
/// pins below are written against.
///
/// Nothing in this crate calls these; they exist so the three facts a reader
/// of `hc.metal` has to take on trust (the Sinkhorn loop count, the two
/// different gate curves, the fold's algebra) are *stated twice*, in two
/// languages, and a test can disagree with one of them.
pub mod reference {
    /// The pre gate: `σ(logit) + eps`, a width weight in `~0..1`.
    #[must_use]
    pub fn pre_gate(mix: f32, scale: f32, base: f32, eps: f32) -> f32 {
        1.0 / (1.0 + (-(mix * scale + base)).exp()) + eps
    }

    /// The post gate: `alpha · σ(logit)` — the model's `alpha` is 2, so this
    /// is the reference's `2·sigmoid` and NOT the pre gate with a different
    /// epsilon. The two curves are the gotcha the MLX oracle names.
    #[must_use]
    pub fn post_gate(mix: f32, scale: f32, base: f32, alpha: f32) -> f32 {
        alpha / (1.0 + (-(mix * scale + base)).exp())
    }

    /// The combiner, from raw `M x M` logits to (approximately) doubly
    /// stochastic — row-major, `comb[i * m + j]`.
    ///
    /// **THE COUNT.** A row softmax seeds it, ONE column normalization
    /// follows, and only then `iters - 1` alternating row/column sweeps run:
    /// `iters = 20` is 19 passes of the loop, which is what
    /// `sinkhorn_iters - 1` says in both shaders and what `v4mlx/hc.py`
    /// spells `range(sinkhorn_iters - 1)`.
    #[must_use]
    pub fn sinkhorn(logits: &[f32], m: usize, iters: u32, eps: f32) -> Vec<f32> {
        let mut comb = vec![0.0f32; m * m];
        for i in 0..m {
            let row = &logits[i * m..(i + 1) * m];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - max_v).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                comb[i * m + j] = e / sum + eps;
            }
        }
        col_normalize(&mut comb, m, eps);
        for _ in 0..iters.saturating_sub(1) {
            row_normalize(&mut comb, m, eps);
            col_normalize(&mut comb, m, eps);
        }
        comb
    }

    fn row_normalize(comb: &mut [f32], m: usize, eps: f32) {
        for i in 0..m {
            let sum: f32 = (0..m).map(|j| comb[i * m + j]).sum::<f32>() + eps;
            for j in 0..m {
                comb[i * m + j] /= sum;
            }
        }
    }

    fn col_normalize(comb: &mut [f32], m: usize, eps: f32) {
        for j in 0..m {
            let sum: f32 = (0..m).map(|i| comb[i * m + j]).sum::<f32>() + eps;
            for i in 0..m {
                comb[i * m + j] /= sum;
            }
        }
    }

    /// `hc_gates`' collapse: `x[h] = Σ_i pre[i] · streams[i][h]`.
    #[must_use]
    pub fn collapse(pre: &[f32], streams: &[f32], m: usize, h: usize) -> Vec<f32> {
        (0..h)
            .map(|k| (0..m).map(|i| pre[i] * streams[i * h + k]).sum())
            .collect()
    }

    /// `hc_fold`'s algebra: `y[j][h] = post[j]·x[h] + Σ_i comb[i][j]·r[i][h]`.
    #[must_use]
    pub fn fold(x: &[f32], streams: &[f32], post: &[f32], comb: &[f32], m: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * h];
        for j in 0..m {
            for k in 0..h {
                let mut acc = post[j] * x[k];
                for i in 0..m {
                    acc += comb[i * m + j] * streams[i * h + k];
                }
                out[j * h + k] = acc;
            }
        }
        out
    }

    /// The weightless RMS norm the wide stream row is widened through.
    #[must_use]
    pub fn rmsnorm(row: &[f32], eps: f32) -> Vec<f32> {
        let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = (mean + eps).sqrt().recip();
        row.iter().map(|v| v * inv).collect()
    }

    fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    // ---- the gated-residual flavor (qwen4) --------------------------------

    /// [`super::mix`]: `y[h] = mean_s( σ(gates[s·H + h]) · normed[s·H + h] )`,
    /// over one row of `m · h` values.
    #[must_use]
    pub fn mix(gates: &[f32], normed: &[f32], m: usize, h: usize) -> Vec<f32> {
        (0..h)
            .map(|k| {
                let acc: f32 = (0..m)
                    .map(|s| normed[s * h + k] * sigmoid(gates[s * h + k]))
                    .sum();
                acc / m as f32
            })
            .collect()
    }

    /// [`super::inject`]: `hyper[s·H + h] += 2·σ(gates[s]/M) · o[h]`, returning
    /// the wide row rather than editing it. The gate is per STREAM here and
    /// per element in [`mix`], and that asymmetry is the point of stating both.
    #[must_use]
    pub fn inject(o: &[f32], gates: &[f32], hyper: &[f32], m: usize, h: usize) -> Vec<f32> {
        let mut out = hyper.to_vec();
        for s in 0..m {
            let g = 2.0 * sigmoid(gates[s] / m as f32);
            for k in 0..h {
                out[s * h + k] += g * o[k];
            }
        }
        out
    }

    /// [`super::ple_gate`]: per stream, the key·query dot over `H` scaled by
    /// `1/√H`, damped to its SIGNED square root with the magnitude clamped at
    /// `1e-6`, sigmoided, and spent on the value row. `sign(0)` is zero and not
    /// the clamp floor, which is the one place a careless port rounds up.
    #[must_use]
    pub fn ple_gate(key: &[f32], query: &[f32], value: &[f32], m: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * h];
        for s in 0..m {
            let dot: f32 = (0..h).map(|i| key[s * h + i] * query[s * h + i]).sum::<f32>()
                / (h as f32).sqrt();
            let magnitude = dot.abs().max(1e-6).sqrt();
            let damped = if dot > 0.0 {
                magnitude
            } else if dot < 0.0 {
                -magnitude
            } else {
                0.0
            };
            let gate = sigmoid(damped);
            for i in 0..h {
                out[s * h + i] = gate * value[i];
            }
        }
        out
    }

    /// `elementwise.rmsnorm_grouped_plus_one`: moments per `group`-wide slice,
    /// gain `weight + 1` off a bank that spans the whole row.
    #[must_use]
    pub fn rmsnorm_grouped_plus_one(row: &[f32], weight: &[f32], group: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0; row.len()];
        for (g, slice) in row.chunks_exact(group).enumerate() {
            let mean: f32 = slice.iter().map(|v| v * v).sum::<f32>() / group as f32;
            let inv = (mean + eps).sqrt().recip();
            for (i, v) in slice.iter().enumerate() {
                out[g * group + i] = v * inv * (weight[g * group + i] + 1.0);
            }
        }
        out
    }

    /// `elementwise.silu_scaled`: `silu(s·x)`, element for element.
    #[must_use]
    pub fn silu_scaled(x: &[f32], s: f32) -> Vec<f32> {
        x.iter()
            .map(|v| {
                let v = v * s;
                v * sigmoid(v)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::reference;
    use super::*;
    use crate::encode::ArgValue;
    use crate::probe::Probe;
    use dtype::Dtype;

    /// dsv4's own geometry, shrunk on the row axis only: four streams, a
    /// hidden width wide enough that the per-row points loop.
    const M: u32 = 4;
    const H: u32 = 512;
    const ROWS: u32 = 3;
    const SINKHORN: u32 = 20;

    fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::Bf16)
    }

    fn f32t(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::F32)
    }

    // ---- the marshalling pins ---------------------------------------------

    /// The expansion is ONE THREAD PER INPUT ELEMENT, each writing its own `M`
    /// outputs — the grid covers the narrow `[N, H]` side, never the wide one.
    #[test]
    fn expand_covers_the_narrow_side_and_states_the_fan() {
        let probe = Probe::default();
        expand(&probe, bf16(1, ROWS, H), M, bf16(2, ROWS, M * H)).expect("the expansion enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.file, "elemwise/hc.metal");
        assert_eq!(f.entrypoint, "hc_expand_bfloat16");
        assert_eq!(f.lanes, [H, ROWS, 1]);
        assert_eq!(f.group, [256, 1, 1]);
        assert_eq!(a[0], ArgValue::Buffer(1));
        assert_eq!(a[1], ArgValue::BufferMut(2));
        assert_eq!(a[2], ArgValue::I32(ROWS as i32));
        assert_eq!(a[3], ArgValue::I32(M as i32));
        assert_eq!(a[4], ArgValue::I32(H as i32));
    }

    /// The widening norm is one threadgroup per row over the WIDE width, and
    /// `dispatchThreads` counts threads, so the x extent is `rows · BLOCK`.
    #[test]
    fn rmsnorm_is_a_threadgroup_per_row_over_the_wide_width() {
        let probe = Probe::default();
        rmsnorm_f32(&probe, bf16(1, ROWS, M * H), 1e-6, f32t(2, ROWS, M * H))
            .expect("the norm enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.entrypoint, "hc_rmsnorm_f32_bfloat16");
        assert_eq!(f.lanes, [ROWS * 256, 1, 1]);
        assert_eq!(f.group, [256, 1, 1]);
        assert_eq!(a[0], ArgValue::Buffer(1));
        assert_eq!(a[1], ArgValue::BufferMut(2));
        assert_eq!(a[2], ArgValue::I32((M * H) as i32));
        assert_eq!(a[3], ArgValue::F32(1e-6));
    }

    /// The gate point's operand order is `hc.cuh`'s, and the sinkhorn count
    /// travels as a stated extent rather than a baked one.
    #[test]
    fn gates_marshals_the_cuh_operand_order() {
        let probe = Probe::default();
        gates(
            &probe,
            f32t(1, ROWS, M * H),
            bf16(2, ROWS, M * H),
            f32t(3, 1, 3),
            f32t(4, 1, M * 2 + M * M),
            M,
            1e-6,
            2.0,
            SINKHORN,
            bf16(5, ROWS, H),
            f32t(6, ROWS, M),
            f32t(7, ROWS, M * M),
        )
        .expect("the gates enqueue");
        let (f, a) = probe.only();
        assert_eq!(f.entrypoint, "hc_gates_bfloat16");
        assert_eq!(f.lanes, [ROWS * 256, 1, 1]);
        assert_eq!(f.group, [256, 1, 1]);
        assert_eq!(a[0], ArgValue::Buffer(1)); // the mix row (normed)
        assert_eq!(a[1], ArgValue::Buffer(3)); // scale
        assert_eq!(a[2], ArgValue::Buffer(4)); // base
        assert_eq!(a[3], ArgValue::Buffer(2)); // the residual streams
        assert_eq!(a[4], ArgValue::BufferMut(6)); // post_mix
        assert_eq!(a[5], ArgValue::BufferMut(7)); // comb_mix
        assert_eq!(a[6], ArgValue::BufferMut(5)); // the layer input
        assert_eq!(a[7], ArgValue::I32(M as i32));
        assert_eq!(a[8], ArgValue::I32(H as i32));
        assert_eq!(a[9], ArgValue::F32(1e-6));
        assert_eq!(a[10], ArgValue::F32(2.0));
        assert_eq!(a[11], ArgValue::I32(SINKHORN as i32));
    }

    /// The fold's grid is the NARROW side too: a thread owning the whole
    /// `(n, h)` column is what lets it read every stream before writing any.
    #[test]
    fn fold_gives_each_thread_a_whole_column() {
        let probe = Probe::default();
        fold(
            &probe,
            bf16(1, ROWS, H),
            bf16(2, ROWS, M * H),
            f32t(3, ROWS, M),
            f32t(4, ROWS, M * M),
            bf16(5, ROWS, M * H),
        )
        .expect("the fold enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.entrypoint, "hc_fold_bfloat16");
        assert_eq!(f.lanes, [H, ROWS, 1]);
        assert_eq!(a[0], ArgValue::Buffer(1));
        assert_eq!(a[1], ArgValue::Buffer(2));
        assert_eq!(a[2], ArgValue::Buffer(3));
        assert_eq!(a[3], ArgValue::Buffer(4));
        assert_eq!(a[4], ArgValue::BufferMut(5));
        assert_eq!(a[5], ArgValue::I32(ROWS as i32));
        assert_eq!(a[6], ArgValue::I32(M as i32));
        assert_eq!(a[7], ArgValue::I32(H as i32));
    }

    /// A fan past the unrolled maximum is a REFUSAL and not a clamp: the
    /// shader's arrays are `HC_MAX_MULT` long and a wider row would walk off
    /// them.
    #[test]
    fn a_fan_past_the_unrolled_maximum_is_refused_by_name() {
        let probe = Probe::default();
        let err = expand(&probe, bf16(1, ROWS, H), 9, bf16(2, ROWS, 9 * H))
            .expect_err("nine streams do not fit the mixers");
        match err {
            Error::Backend { op, detail } => {
                assert_eq!(op, "elementwise.hc_expand");
                assert!(detail.contains("stream count is 9"), "{detail}");
            }
            other => panic!("expected a backend refusal, got {other:?}"),
        }
        assert!(probe.fires().is_empty(), "a refused entry fired anyway");
    }

    /// A wide row that is not a whole number of hidden-wide streams is the
    /// other refusal — the fan is DERIVED from the two widths, so a mismatch
    /// has no honest answer.
    #[test]
    fn a_row_that_is_not_whole_streams_is_refused_by_name() {
        let probe = Probe::default();
        let err = fold(
            &probe,
            bf16(1, ROWS, H),
            bf16(2, ROWS, M * H + 1),
            f32t(3, ROWS, M),
            f32t(4, ROWS, M * M),
            bf16(5, ROWS, M * H + 1),
        )
        .expect_err("a ragged wide row has no fan");
        assert!(matches!(err, Error::Backend { op, .. } if op == "elementwise.hc_fold"));
    }

    /// f16 is not a dtype this plane stamps — the CUDA twin instantiates it,
    /// this one does not, and the refusal says which op and which dtype.
    #[test]
    fn an_unstamped_dtype_is_refused_by_name() {
        let probe = Probe::default();
        let err = expand(
            &probe,
            Tensor::new(1, ROWS, H, Dtype::F16),
            M,
            Tensor::new(2, ROWS, M * H, Dtype::F16),
        )
        .expect_err("f16 is unstamped here");
        assert!(matches!(
            err,
            Error::DtypeUnsupported { op, dtype } if op == "elementwise.hc_expand" && dtype == Dtype::F16
        ));
    }

    // ---- the arithmetic pins ----------------------------------------------

    /// **THE OFF-BY-ONE, AND HOW MUCH IT IS WORTH.**
    /// `sinkhorn(_, _, n, _)` is the seed column-norm plus `n - 1` sweeps, so
    /// `n == 1` is the SEED ALONE — no loop pass at all — and that is the
    /// sharpest statement of the shader's `iter < sinkhorn - 1` bound.
    ///
    /// **HOW MUCH ONE SWEEP IS WORTH DEPENDS ENTIRELY ON THE MATRIX**, and
    /// that is the fact worth writing down. Sinkhorn-Knopp is a contraction
    /// whose rate is the logits' own: a mix row whose row and column maxima
    /// already agree converges to fp32's last bit in two sweeps, and for it 19,
    /// 20 and 21 iterations are the SAME matrix — the documented trap costs
    /// nothing. A row whose maxima COMPETE for the same column is still moving
    /// at twenty, and there one sweep is worth `2e-4`, three orders above fp32
    /// rounding. So the count is pinned on the slow matrix, where an
    /// off-by-one is a real difference, and the fast one is measured beside it
    /// to say why a test written on the wrong fixture would have proved
    /// nothing. `engine-metal/tests/hc_on_device.rs` sweeps the card the same
    /// way, on the same two shapes.
    #[test]
    fn the_sinkhorn_count_is_the_seed_plus_one_less_sweep() {
        let m = 4;
        let eps = 1e-6;
        let spread = |a: &[f32], b: &[f32]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };

        // Rows 1 and 3 both peak in column 2, so the alternating normalization
        // has a genuine transport problem to solve and takes its time.
        let slow: Vec<f32> = (0..m * m).map(|k| ((k * 7) % 11) as f32 - 5.0).collect();
        // Every row peaks in its own column: the seed is already almost
        // doubly stochastic and two sweeps finish it.
        let fast: Vec<f32> = (0..m * m).map(|k| ((k * 13) % 17) as f32 * 0.5 - 4.0).collect();

        // One "iteration" is the seed alone: softmax, `+ eps`, one column
        // normalization, and nothing else.
        let mut seeded = vec![0.0f32; m * m];
        for i in 0..m {
            let row = &slow[i * m..(i + 1) * m];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - max_v).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..m {
                seeded[i * m + j] = exps[j] / sum + eps;
            }
        }
        for j in 0..m {
            let s: f32 = (0..m).map(|i| seeded[i * m + j]).sum::<f32>() + eps;
            for i in 0..m {
                seeded[i * m + j] /= s;
            }
        }
        for (a, b) in reference::sinkhorn(&slow, m, 1, eps).iter().zip(&seeded) {
            assert!((a - b).abs() < 1e-6, "one iteration is the seed alone");
        }

        // The early counts are far apart under either matrix.
        assert!(
            spread(&reference::sinkhorn(&slow, m, 1, eps), &reference::sinkhorn(&slow, m, 2, eps))
                > 1e-3,
            "1 and 2 iterations agree too closely to pin the bound"
        );

        // AT THE SHIPPED COUNT: still moving on the slow matrix, finished on
        // the fast one. Both halves are the claim.
        let slow_step = spread(
            &reference::sinkhorn(&slow, m, 19, eps),
            &reference::sinkhorn(&slow, m, 20, eps),
        );
        let fast_step = spread(
            &reference::sinkhorn(&fast, m, 19, eps),
            &reference::sinkhorn(&fast, m, 20, eps),
        );
        assert!(
            slow_step > 1e-5,
            "one sweep at twenty is worth {slow_step:.3e} on the slow matrix — if this ever \
             fell to rounding, the count would stop being observable and the pin would be \
             passing for the wrong reason"
        );
        assert!(
            fast_step < 1e-6,
            "the fast matrix has not converged by twenty ({fast_step:.3e}), so the contrast \
             this test draws is not real"
        );
    }

    /// Twenty iterations land on the Birkhoff polytope — approximately, and
    /// the approximation is one-sided. This is the property the port exists to
    /// preserve (the mHC paper's whole reason for the projection: a combiner
    /// that is doubly stochastic conserves the residual's magnitude) and it is
    /// what an fp16 accumulation would lose first.
    ///
    /// **THE COLUMNS ARE EXACT AND THE ROWS ARE NOT**, because the loop ENDS
    /// on a column normalization. On a fast-converging matrix both hold to
    /// `1e-5`; on a slow one the columns still hold and the rows are out by
    /// `6e-3` at twenty iterations. Anything downstream that needs the row sums
    /// is reading a property this family does not guarantee at the shipped
    /// count.
    #[test]
    fn twenty_iterations_land_on_the_birkhoff_polytope() {
        let m = 4;
        let deviation = |logits: &[f32]| {
            let comb = reference::sinkhorn(logits, m, 20, 1e-6);
            let rows = (0..m)
                .map(|i| ((0..m).map(|j| comb[i * m + j]).sum::<f32>() - 1.0).abs())
                .fold(0.0f32, f32::max);
            let cols = (0..m)
                .map(|j| ((0..m).map(|i| comb[i * m + j]).sum::<f32>() - 1.0).abs())
                .fold(0.0f32, f32::max);
            (rows, cols)
        };

        let fast: Vec<f32> = (0..m * m).map(|k| ((k * 13) % 17) as f32 * 0.5 - 4.0).collect();
        let (rows, cols) = deviation(&fast);
        assert!(rows < 1e-5, "the converged matrix's rows are out by {rows:.3e}");
        assert!(cols < 1e-5, "the converged matrix's columns are out by {cols:.3e}");

        let slow: Vec<f32> = (0..m * m).map(|k| ((k * 7) % 11) as f32 - 5.0).collect();
        let (rows, cols) = deviation(&slow);
        assert!(
            cols < 1e-5,
            "the columns are normalized LAST and must hold regardless: {cols:.3e}"
        );
        assert!(
            rows > 1e-4,
            "the slow matrix's rows are already within {rows:.3e} of one, so the asymmetry \
             this test documents is not real at twenty iterations"
        );
    }

    /// **THE TWO GATES ARE DIFFERENT CURVES.** `pre` is `σ + eps` and tops out
    /// at one; `post` is `α·σ` and tops out at `α`, which the model states as
    /// 2. Reading the post plane through the pre curve halves every depth
    /// weight, and nothing downstream would say so.
    #[test]
    fn the_pre_gate_is_sigmoid_and_the_post_gate_is_twice_one() {
        let eps = 1e-6;
        assert!((reference::pre_gate(0.0, 1.0, 0.0, eps) - (0.5 + eps)).abs() < 1e-6);
        assert!((reference::post_gate(0.0, 1.0, 0.0, 2.0) - 1.0).abs() < 1e-6);
        // Saturated: the pre gate approaches 1, the post gate approaches 2.
        assert!((reference::pre_gate(40.0, 1.0, 0.0, eps) - (1.0 + eps)).abs() < 1e-5);
        assert!((reference::post_gate(40.0, 1.0, 0.0, 2.0) - 2.0).abs() < 1e-5);
        // And both read `mix * scale + base`, so the scale and the base are
        // not interchangeable with each other.
        assert!(
            (reference::pre_gate(2.0, 3.0, 1.0, 0.0) - reference::pre_gate(7.0, 1.0, 0.0, 0.0))
                .abs()
                < 1e-6
        );
    }

    /// **EXPAND THEN FOLD IS THE IDENTITY WHEN THE GATES SAY IT IS.** With the
    /// combiner at the identity and the post gate at zero, the fold hands back
    /// exactly the streams it was given — which pins the fold's index algebra
    /// (`comb[i*M + j]` contracting `i`, the stream axis outer) rather than its
    /// transpose, since the identity is the one matrix that cannot tell them
    /// apart... so the round trip is measured at a NON-symmetric combiner too.
    #[test]
    fn the_fold_round_trips_the_expansion_under_the_gates_that_say_so() {
        let (m, h) = (4usize, 8usize);
        let x: Vec<f32> = (0..h).map(|k| (k as f32) * 0.25 - 1.0).collect();
        let expanded: Vec<f32> = (0..m).flat_map(|_| x.clone()).collect();

        // (a) identity combiner, zero post gate: the streams come back whole.
        let mut identity = vec![0.0f32; m * m];
        for i in 0..m {
            identity[i * m + i] = 1.0;
        }
        let back = reference::fold(&x, &expanded, &vec![0.0; m], &identity, m, h);
        assert_eq!(back, expanded, "an identity fold moved the streams");

        // (b) a NON-symmetric combiner, where `comb[i*m + j]` and its
        //     transpose disagree — the pin that the contraction is over `i`.
        let mut comb = vec![0.0f32; m * m];
        comb[0 * m + 1] = 1.0; // stream 0 flows into stream 1, not the reverse
        for i in 1..m {
            comb[i * m + i] = 1.0;
        }
        let streams: Vec<f32> = (0..m * h).map(|k| k as f32).collect();
        let moved = reference::fold(&vec![0.0; h], &streams, &vec![0.0; m], &comb, m, h);
        // Stream 1 received stream 0 AND kept its own; stream 0 received
        // nothing.
        for k in 0..h {
            assert_eq!(moved[0 * h + k], 0.0, "stream 0 should have emptied");
            assert_eq!(
                moved[1 * h + k],
                streams[0 * h + k] + streams[1 * h + k],
                "stream 1 should hold both"
            );
        }
    }

    /// The collapse is the pre gate's weighted sum over streams, and it is the
    /// `pre`-weighted mean the reference writes as `Σ_stream pre · x` — no
    /// division by `M` anywhere, which is what separates this family from the
    /// qwen4 `hc_mix` flavour that DOES average.
    #[test]
    fn the_collapse_is_a_weighted_sum_and_not_a_mean() {
        let (m, h) = (4usize, 3usize);
        let streams: Vec<f32> = (0..m * h).map(|k| (k + 1) as f32).collect();
        let ones = vec![1.0f32; m];
        let got = reference::collapse(&ones, &streams, m, h);
        for k in 0..h {
            let want: f32 = (0..m).map(|i| streams[i * h + k]).sum();
            assert!((got[k] - want).abs() < 1e-6, "the collapse averaged");
        }
    }

    /// The widening norm is weightless and divides by the row's own RMS: a
    /// row scaled by `c` normalizes to the same vector.
    #[test]
    fn the_widening_norm_is_scale_free() {
        let row: Vec<f32> = (1..=16).map(|k| k as f32 * 0.125).collect();
        let scaled: Vec<f32> = row.iter().map(|v| v * 7.0).collect();
        let a = reference::rmsnorm(&row, 1e-12);
        let b = reference::rmsnorm(&scaled, 1e-12);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "the norm carried the row's scale");
        }
        let mean: f32 = a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32;
        assert!((mean - 1.0).abs() < 1e-4, "the normed row's RMS is {mean}");
    }

    // ---- the gated-residual flavor (qwen4) --------------------------------

    /// The mix's grid is the NARROW side — one thread per `(row, h)` of the
    /// collapsed row — and its three stated ints are `(rows, fan, hidden)`.
    /// A grid over the wide row would run `M` times and write each output `M`
    /// times, which is a race no assert on the numbers would name.
    #[test]
    fn the_mix_covers_the_collapsed_row_and_states_the_fan() {
        let probe = Probe::default();
        mix(&probe, bf16(1, ROWS, M * H), bf16(2, ROWS, M * H), M, bf16(3, ROWS, H))
            .expect("the mix enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.file, "elemwise/hc.metal");
        assert_eq!(f.entrypoint, "hc_mix_bfloat16");
        assert_eq!(f.lanes, [H, ROWS, 1]);
        assert_eq!(a[0], ArgValue::Buffer(1));
        assert_eq!(a[1], ArgValue::Buffer(2));
        assert_eq!(a[2], ArgValue::BufferMut(3));
        assert_eq!(a[3..], [
            ArgValue::I32(ROWS as i32),
            ArgValue::I32(M as i32),
            ArgValue::I32(H as i32),
        ]);
    }

    /// The injection edits the WIDE row in place and its grid is still the
    /// narrow side: each thread owns one `(row, h)` column of every stream.
    #[test]
    fn the_injection_edits_the_wide_row_from_the_narrow_grid() {
        let probe = Probe::default();
        inject(&probe, bf16(1, ROWS, H), bf16(2, ROWS, M), M, bf16(3, ROWS, M * H))
            .expect("the injection enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.entrypoint, "hc_inject_bfloat16");
        assert_eq!(f.lanes, [H, ROWS, 1]);
        assert_eq!(a[2], ArgValue::BufferMut(3), "the wide row is the one written");
    }

    /// The PLE gate is one THREADGROUP per `(row, stream)` — `rows · M` of
    /// them, `BLOCK` threads apiece — because its dot is a reduction and not
    /// an element map.
    #[test]
    fn the_ple_gate_is_one_group_per_row_and_stream() {
        let probe = Probe::default();
        ple_gate(
            &probe,
            bf16(1, ROWS, M * H),
            bf16(2, ROWS, M * H),
            bf16(3, ROWS, H),
            M,
            bf16(4, ROWS, M * H),
        )
        .expect("the gate enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.entrypoint, "ple_gate_bfloat16");
        assert_eq!(f.lanes, [ROWS * M * BLOCK, 1, 1]);
        assert_eq!(f.group, [BLOCK, 1, 1]);
        assert_eq!(a[4], ArgValue::I32(M as i32));
        assert_eq!(a[5], ArgValue::I32(H as i32), "the gate spends the VALUE row's width");
    }

    /// A bank one group wide would gain every stream by the same plane and
    /// land the right spread around `M − 1` wrong centres, with no NaN to
    /// notice it by. The entry refuses it instead.
    #[test]
    fn the_grouped_norms_bank_has_to_span_the_row() {
        let probe = Probe::default();
        let short = crate::elemwise::norm::rmsnorm_grouped_plus_one(
            &probe,
            bf16(1, ROWS, M * H),
            bf16(2, 1, H),
            H,
            1e-6,
            bf16(3, ROWS, M * H),
        );
        assert!(short.is_err(), "a one-group bank passed for a {M}-group row");
        crate::elemwise::norm::rmsnorm_grouped_plus_one(
            &probe,
            bf16(1, ROWS, M * H),
            bf16(2, 1, M * H),
            H,
            1e-6,
            bf16(3, ROWS, M * H),
        )
        .expect("a full-width bank is the one this op gains by");
        let (f, a) = probe.only();
        assert_eq!(f.entrypoint, "rms_grouped_row_bfloat16");
        assert_eq!(
            a.last(),
            Some(&ArgValue::I32(M as i32)),
            "the shader picks its weight plane by `gid % groups`, so the count is stated"
        );
    }

    // ---- the arithmetic pins ----------------------------------------------

    /// The mix AVERAGES over the fan where [`reference::collapse`] sums — the
    /// one difference between this flavour and the sinkhorn one, and the one a
    /// port sharing a body would lose.
    #[test]
    fn the_gated_mix_is_a_mean_and_the_gate_is_per_element() {
        let (m, h) = (4usize, 3usize);
        let normed: Vec<f32> = (0..m * h).map(|k| (k + 1) as f32).collect();
        // A gate of +inf-ish logits is σ ≈ 1, so the mix is the plain mean.
        let wide = vec![40.0f32; m * h];
        let got = reference::mix(&wide, &normed, m, h);
        for k in 0..h {
            let want: f32 = (0..m).map(|i| normed[i * h + k]).sum::<f32>() / m as f32;
            assert!((got[k] - want).abs() < 1e-4, "the mix summed where it means");
        }
        // And a logit of zero is σ = ½, which halves — a per-element gate, so
        // gating ONE stream changes ONE stream's contribution and nothing else.
        let mut half = wide.clone();
        half[0] = 0.0;
        let got = reference::mix(&half, &normed, m, h);
        let full = reference::mix(&wide, &normed, m, h);
        assert!((full[0] - got[0]) > 1e-3, "the element gate did nothing");
        assert!((full[1] - got[1]).abs() < 1e-5, "the element gate reached a neighbour");
    }

    /// The injection's gate is `2·σ(logit/M)` — TWO at saturation and ONE at a
    /// zero logit, and the divide by `M` is inside the sigmoid and not outside.
    #[test]
    fn the_injection_gate_saturates_at_two_and_divides_inside() {
        let (m, h) = (4usize, 2usize);
        let o = vec![1.0f32; h];
        let zero = vec![0.0f32; m * h];
        let saturated = reference::inject(&o, &vec![400.0; m], &zero, m, h);
        assert!((saturated[0] - 2.0).abs() < 1e-4, "the alpha is two");
        let neutral = reference::inject(&o, &vec![0.0; m], &zero, m, h);
        assert!((neutral[0] - 1.0).abs() < 1e-6, "σ(0) is a half, doubled is one");
        // `logit/M` inside: at M = 4 a logit of 4 is σ(1), not σ(4).
        let inside = reference::inject(&o, &vec![4.0; m], &zero, m, h);
        let want = 2.0 / (1.0 + (-1.0f32).exp());
        assert!((inside[0] - want).abs() < 1e-5, "the fan divided outside the sigmoid");
    }

    /// `sign(0)` is ZERO and not the clamp floor: a dot of exactly zero gates
    /// at σ(0) = ½, and the `1e-6` clamp is only ever reached from a nonzero
    /// side. A port that wrote `sqrt(max(|d|, 1e-6))` without the sign would
    /// gate a zero dot at σ(1e-3), which is half a permille away and invisible
    /// in any band this file could set on a whole row.
    #[test]
    fn the_ple_gates_damping_carries_the_sign_and_zero_is_zero() {
        let (m, h) = (1usize, 4usize);
        let value = vec![1.0f32; h];
        let zeros = vec![0.0f32; h];
        let at_zero = reference::ple_gate(&zeros, &zeros, &value, m, h);
        assert!((at_zero[0] - 0.5).abs() < 1e-7, "a zero dot did not gate at a half");

        let pos: Vec<f32> = vec![1.0; h];
        let up = reference::ple_gate(&pos, &pos, &value, m, h);
        let down = reference::ple_gate(&pos, &pos.iter().map(|v| -v).collect::<Vec<_>>(), &value, m, h);
        assert!(up[0] > 0.5 && down[0] < 0.5, "the damping dropped the sign");
        assert!(
            ((up[0] - 0.5) - (0.5 - down[0])).abs() < 1e-6,
            "the two sides are not the same curve reflected"
        );
        // `dot = (Σ k·q)/√H` = 4/2 = 2, damped to √2.
        let want = 1.0 / (1.0 + (-(2.0f32).sqrt()).exp());
        assert!((up[0] - want).abs() < 1e-6, "the dot was not scaled by 1/√H");
    }

    /// The grouped norm takes its moments PER GROUP and its gain per element
    /// off a full-width bank: scaling one stream renormalizes that stream and
    /// leaves the others where they were.
    #[test]
    fn the_grouped_norm_normalizes_each_stream_on_its_own() {
        let (m, g) = (4usize, 8usize);
        let row: Vec<f32> = (1..=m * g).map(|k| k as f32 * 0.25).collect();
        let weight = vec![0.0f32; m * g];
        let flat = reference::rmsnorm_grouped_plus_one(&row, &weight, g, 1e-12);
        let mut scaled = row.clone();
        for v in scaled.iter_mut().take(g) {
            *v *= 9.0;
        }
        let after = reference::rmsnorm_grouped_plus_one(&scaled, &weight, g, 1e-12);
        for k in 0..m * g {
            assert!(
                (flat[k] - after[k]).abs() < 1e-4,
                "scaling stream 0 moved element {k}, so the moments are not per group"
            );
        }
        // And the `+ 1`: a zero weight is a UNIT gain, not a zero one.
        let rms: f32 = flat[..g].iter().map(|v| v * v).sum::<f32>() / g as f32;
        assert!((rms - 1.0).abs() < 1e-4, "a zero weight zeroed the row: {rms}");
    }

    /// `silu(s·x)` and not `s·silu(x)`: the scale is INSIDE, so it moves the
    /// curve's knee and not just its height.
    #[test]
    fn the_scaled_silu_scales_inside_the_curve() {
        let x = [1.0f32, -1.0, 0.0, 4.0];
        let inside = reference::silu_scaled(&x, 2.0);
        let outside: Vec<f32> = reference::silu_scaled(&x, 1.0).iter().map(|v| v * 2.0).collect();
        assert!((inside[2] - 0.0).abs() < 1e-7, "silu(0) is zero either way");
        assert!(
            (inside[0] - outside[0]).abs() > 1e-2,
            "the scale landed outside the curve, where it is a plain gain"
        );
        let want = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((inside[0] - want).abs() < 1e-6);
    }
}
