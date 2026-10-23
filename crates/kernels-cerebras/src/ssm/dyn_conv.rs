//! The block draft's dynamic conv: a stateless causal depthwise conv whose
//! taps move per row. Reference: kernels-xla `ssm::block_dyn_conv`.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::linear::gemm::{ARRAY_WORDS, max_pes, pe_words};
use crate::program::Shard;
use crate::tensor::{RaggedTensor, Tensor};

/// Row `t` of a lane is `Σ_{k ≤ t} (base[side·taps + k, c] + coeff[t,
/// (side·taps + k)·groups + c / group]) · x[t − k, c]`, the lane's rows
/// before its first reading nothing. Every PE holds every row (a lane needs
/// its history); the channels split over PEs in whole groups, the
/// coefficients' `2·taps` runs of groups with them.
pub fn block_dyn_conv(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    coeff: Tensor,
    base: Tensor,
    side: u32,
    taps: u32,
    group: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.dyn_conv";
    expect(OP, x.data, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, x.indptr, &[Dtype::I32])?;
    expect(OP, coeff, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, base, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    let channels = x.data.width;
    if channels == 0 || taps == 0 || group == 0 || x.data.rows == 0 {
        return Err(refuse(
            OP,
            "the channel count, tap count, group and rows are nonzero",
        ));
    }
    if side > 1 {
        return Err(refuse(
            OP,
            format!("side {side} is stated, and the projection carries two"),
        ));
    }
    if !channels.is_multiple_of(group) {
        return Err(refuse(
            OP,
            format!("{channels} channels are not a whole number of groups of {group}"),
        ));
    }
    let groups = channels / group;
    if coeff.width != 2 * taps * groups || coeff.rows != x.data.rows {
        return Err(refuse(
            OP,
            "the coefficients are not two sides of taps over groups per row",
        ));
    }
    if base.rows != 2 * taps || base.width != channels {
        return Err(refuse(
            OP,
            "the base kernel is not two sides of taps over channels",
        ));
    }
    if y.rows != x.data.rows || y.width != channels {
        return Err(refuse(OP, "the convolution lands the rows it convolves"));
    }
    let lanes = x.indptr.elements().saturating_sub(1);
    let rows = u64::from(x.data.rows);
    let budget = pe_words();
    let words = |cg: u32| {
        let cb = u64::from(channels / cg);
        let gl = u64::from(groups / cg);
        let coeff_local = rows * 2 * u64::from(taps) * gl;
        let largest = (rows * cb).max(coeff_local);
        (
            rows * cb * 2 + 2 * u64::from(taps) * cb + coeff_local + lanes + 1,
            largest,
        )
    };
    let cg = (1..=groups)
        .filter(|cg| groups.is_multiple_of(*cg) && *cg <= max_pes())
        .find(|cg| {
            let (total, largest) = words(*cg);
            total <= budget && largest <= ARRAY_WORDS
        })
        .ok_or_else(|| {
            refuse(
                OP,
                format!(
                    "{} rows of {channels} channels do not fit a PE's {budget} even one group of {group} at a time",
                    x.data.rows
                ),
            )
        })?;
    let cb = channels / cg;
    ctx.emit(&mut |cx| {
        let xb = cx.read(x.data)?;
        let ib = cx.read(x.indptr)?;
        let cb_ = cx.read(coeff)?;
        let bb = cx.read(base)?;
        let yb = cx.write(y)?;
        if cg > 1 {
            cx.over(cg);
            for b in [&xb, &bb, &yb] {
                cx.shard(b, Shard::Cols(cg));
            }
            cx.shard(
                &cb_,
                Shard::Tile {
                    rows: 1,
                    cols: cg,
                    segments: 2 * taps,
                },
            );
        }
        cx.call(
            Kernel::BlockDynConv,
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(cb_),
                Arg::Ptr(bb),
                Arg::Ptr(ib),
                Arg::Ptr(yb),
                Arg::Int(lanes as i64),
                Arg::Int(i64::from(cb)),
                Arg::Int(i64::from(taps)),
                Arg::Int(i64::from(group)),
                Arg::Int(i64::from(side)),
            ],
        );
        Ok(())
    })
}
