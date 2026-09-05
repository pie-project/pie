#![allow(clippy::too_many_arguments)]

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::Tensor;

const BLOCK: u32 = 256;

fn even_hidden(op: &'static str, hidden: u32) -> Result<u32, Error> {
    if !hidden.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the hidden width {hidden} is odd: a bf16 word holds a pair"),
        ));
    }
    Ok(hidden / 2)
}

const MAX_HC_MULT: u32 = 8;

fn stream_fan(op: &'static str, wide: u32, hidden: u32) -> Result<u32, Error> {
    nonzero(op, "the hidden width", hidden)?;
    if wide == 0 || !wide.is_multiple_of(hidden) {
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
                 register and shared arrays"
            ),
        ));
    }
    Ok(fan)
}

fn per_row(op: &'static str, rows: u32) -> Result<Grid, Error> {
    let rows = nonzero(op, "rows", rows)?;
    let lanes = rows.checked_mul(BLOCK).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {rows} rows x {BLOCK}"),
        )
    })?;
    Ok(Grid::of([lanes, 1, 1], [BLOCK, 1, 1]))
}

pub fn expand(ctx: &Ctx<'_>, x: Tensor, streams: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_expand";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "hc_expand_bf16" });
    debug_assert_eq!(y.rows, x.rows, "the expansion lands one wide row per row");
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    let words = even_hidden(OP, x.width)?;
    ctx.fire(
        Fire::at("elemwise/hc_expand.wgsl", entry).apply(Grid::of(
            elementwise_rows(OP, words, x.rows)?,
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

pub fn rmsnorm_f32(ctx: &Ctx<'_>, streams: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_rmsnorm_f32";
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "hc_rmsnorm_f32_bf16" });
    debug_assert_eq!(y.dtype, dtype::Dtype::F32, "`{OP}` widens to f32");
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the normed rectangle is the stream rectangle"
    );
    ctx.fire(
        Fire::at("elemwise/hc_rmsnorm.wgsl", entry).apply(per_row(OP, y.rows)?),
        &[
            streams.arg(),
            y.arg_mut(),
            stated(OP, nonzero(OP, "the normed row's width", y.width)?)?.arg(),
            eps.arg(),
        ],
    )
}

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

    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    if mixes.width != mix_hc || (mix_hc != layer_row && mix_hc != stream_count) {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide and a trunk collapse row \
                 is {stream_count}; the plane lands {} rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
    let points = rows.checked_mul(mix_hc).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {rows} rows x {mix_hc}"),
        )
    })?;
    ctx.fire(
        Fire::at("elemwise/hc_project.wgsl", "hc_project").apply(per_row(OP, points)?),
        &[
            normed.arg(),
            hc_fn.arg(),
            mixes.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, mix_hc)?.arg(),
        ],
    )
}

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
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "hc_gates_bf16" });
    debug_assert_eq!(
        normed.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads an f32 mix row"
    );
    debug_assert_eq!(
        scale.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads f32 mix scales"
    );
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
    even_hidden(OP, x.width)?;
    ctx.fire(
        Fire::at("elemwise/hc_gates.wgsl", entry).apply(per_row(OP, x.rows)?),
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

pub fn fold(
    ctx: &Ctx<'_>,
    x: Tensor,
    streams: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_fold";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "hc_fold_bf16" });
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the fold lands the stream rectangle it mixes"
    );
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert!(
        post_mix.dtype == dtype::Dtype::F32 && comb_mix.dtype == dtype::Dtype::F32,
        "`{OP}` reads f32 gate matrices"
    );
    let words = even_hidden(OP, x.width)?;
    ctx.fire(
        Fire::at("elemwise/hc_fold.wgsl", entry).apply(Grid::of(
            elementwise_rows(OP, words, y.rows)?,
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

pub fn collapse(
    ctx: &Ctx<'_>,
    mixes: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    hc_eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_collapse";
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "hc_collapse_bf16" });
    debug_assert_eq!(
        mixes.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads an f32 mix row"
    );
    debug_assert_eq!(
        scale.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads an f32 mix scale"
    );
    debug_assert_eq!(base.dtype, dtype::Dtype::F32, "`{OP}` reads f32 mix bases");
    debug_assert_eq!(y.dtype, streams.dtype, "`{OP}` lands the streams' element");
    let fan = stream_fan(OP, streams.width, y.width)?;
    debug_assert_eq!(
        fan, stream_count,
        "the row's stream fan is the count the statement states"
    );
    if mixes.width != fan || mixes.rows != y.rows {
        return Err(refuse(
            OP,
            format!(
                "the trunk collapse folds {fan} streams under a {}-wide mix row over {} of {} rows",
                mixes.width, mixes.rows, y.rows
            ),
        ));
    }
    even_hidden(OP, y.width)?;
    ctx.fire(
        Fire::at("elemwise/hc_collapse.wgsl", entry).apply(per_row(OP, y.rows)?),
        &[
            mixes.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            y.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, nonzero(OP, "the hidden width", y.width)?)?.arg(),
            hc_eps.arg(),
        ],
    )
}

pub fn mix(
    ctx: &Ctx<'_>,
    gates: Tensor,
    normed: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_mix";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "hc_mix_bf16" });
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
    let words = even_hidden(OP, y.width)?;
    ctx.fire(
        Fire::at("elemwise/hc_mix.wgsl", entry).apply(Grid::of(
            elementwise_rows(OP, words, y.rows)?,
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

pub fn inject(
    ctx: &Ctx<'_>,
    o: Tensor,
    gates: Tensor,
    streams: u32,
    hyper: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_inject";
    let entry = dtype_dispatch!(OP, hyper.dtype, { Bf16 => "hc_inject_bf16" });
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
    let words = even_hidden(OP, o.width)?;
    ctx.fire(
        Fire::at("elemwise/hc_inject.wgsl", entry).apply(Grid::of(
            elementwise_rows(OP, words, hyper.rows)?,
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

pub fn ple_gate(
    ctx: &Ctx<'_>,
    key: Tensor,
    query: Tensor,
    value: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.ple_gate";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "ple_gate_bf16" });
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
    even_hidden(OP, value.width)?;
    let groups = y.rows.checked_mul(fan).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {} rows x {fan} streams", y.rows),
        )
    })?;
    ctx.fire(
        Fire::at("elemwise/hc_ple_gate.wgsl", entry).apply(per_row(OP, groups)?),
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

pub mod reference {
    #[must_use]
    pub fn pre_gate(mix: f32, scale: f32, base: f32, eps: f32) -> f32 {
        1.0 / (1.0 + (-(mix * scale + base)).exp()) + eps
    }

    #[must_use]
    pub fn post_gate(mix: f32, scale: f32, base: f32, alpha: f32) -> f32 {
        alpha / (1.0 + (-(mix * scale + base)).exp())
    }

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

    #[must_use]
    pub fn collapse(pre: &[f32], streams: &[f32], m: usize, h: usize) -> Vec<f32> {
        (0..h)
            .map(|k| (0..m).map(|i| pre[i] * streams[i * h + k]).sum())
            .collect()
    }

    #[must_use]
    pub fn fold(
        x: &[f32],
        streams: &[f32],
        post: &[f32],
        comb: &[f32],
        m: usize,
        h: usize,
    ) -> Vec<f32> {
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

    #[must_use]
    pub fn rmsnorm(row: &[f32], eps: f32) -> Vec<f32> {
        let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = (mean + eps).sqrt().recip();
        row.iter().map(|v| v * inv).collect()
    }

    fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

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

    #[must_use]
    pub fn ple_gate(key: &[f32], query: &[f32], value: &[f32], m: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0; m * h];
        for s in 0..m {
            let dot: f32 = (0..h)
                .map(|i| key[s * h + i] * query[s * h + i])
                .sum::<f32>()
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

    #[must_use]
    pub fn rmsnorm_grouped_plus_one(
        row: &[f32],
        weight: &[f32],
        group: usize,
        eps: f32,
    ) -> Vec<f32> {
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
