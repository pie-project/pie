//! The block draft's dynamic conv (a stateless causal depthwise conv whose
//! taps move per row) and its candidate selector's walk. References:
//! kernels-xla `attn::ssm::block_dyn_conv`, `attn::ple::selector_walk`.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::linear::gemm::{ARRAY_WORDS, max_pes, pe_words};
use crate::program::{HostOp, Shard};
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
    const OP: &str = "attention.block_dyn_conv";
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
        cx.library("k_block_dyn_conv");
        cx.call(
            "k_block_dyn_conv",
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

/// The drafter's greedy walk over each lane's candidate rows: row `t` scores
/// its `k` candidates `unary[t, c] + Σ_d pred[prev, d] · hp[t, d] · succ[cand, d]`
/// (the bilinear term only when both ids are in the vocabulary; `hp` 1 when
/// absent), picks the first best, and the pick is the next row's `prev`
/// (the lane's first `prev` is `tokens[begin]`). With `first` 1 the anchor
/// row takes its first candidate unscored. Rows outside every lane keep
/// `picks`. A sequential walk over a few rows: the host runs it.
pub fn selector_walk(
    ctx: &Ctx<'_>,
    cand: RaggedTensor,
    unary: Tensor,
    hp: Option<Tensor>,
    tokens: Tensor,
    pred: Tensor,
    succ: Tensor,
    first: u32,
    picks: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.selector_walk";
    if first > 1 {
        return Err(refuse(
            OP,
            "a span's anchor is row 0 and its first mask row 1",
        ));
    }
    for t in [cand.data, cand.indptr, tokens, picks] {
        expect(OP, t, &[Dtype::I32])?;
    }
    expect(OP, unary, &[Dtype::F32])?;
    expect(OP, pred, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, succ, &[Dtype::Bf16, Dtype::F32])?;
    if let Some(h) = hp {
        expect(OP, h, &[Dtype::Bf16, Dtype::F32])?;
    }
    let (rows, k, rank, vocab) = (cand.data.rows, cand.data.width, pred.width, pred.rows);
    if k == 0 || rank == 0 || vocab == 0 || cand.indptr.elements() < 2 {
        return Err(refuse(
            OP,
            "the candidates a slot, the codebooks' rank and vocabulary are nonzero, the lanes a CSR",
        ));
    }
    if succ.width != rank || hp.is_some_and(|h| h.width != rank) || pred.rows != succ.rows {
        return Err(refuse(
            OP,
            "the codebooks and the projected hidden disagree on rank or vocabulary",
        ));
    }
    if unary.rows != rows
        || unary.width != k
        || picks.rows != rows
        || tokens.rows != rows
        || hp.is_some_and(|h| h.rows != rows)
    {
        return Err(refuse(
            OP,
            "unary, hp, tokens and picks carry one row per candidate row",
        ));
    }
    // One PE walks every lane when the codebooks (vocabulary-sized) and the
    // rows fit it; else the host walks.
    let words = [
        cand.data.elements(),
        cand.indptr.elements(),
        unary.elements(),
        hp.map_or(0, |h| h.elements()),
        tokens.elements(),
        pred.elements(),
        succ.elements(),
        picks.elements(),
    ];
    let on_pe = words.iter().all(|w| *w <= crate::linear::gemm::ARRAY_WORDS)
        && words.iter().sum::<u64>() + 64 <= crate::linear::gemm::pe_words();
    ctx.emit(&mut |cx| {
        let cb = cx.read(cand.data)?;
        let ib = cx.read(cand.indptr)?;
        let ub = cx.read(unary)?;
        let hb = match hp {
            Some(h) => Some(cx.read(h)?),
            None => None,
        };
        let tb = cx.read(tokens)?;
        let pb = cx.read(pred)?;
        let sb = cx.read(succ)?;
        cx.read(picks)?;
        let kb = cx.write(picks)?;
        if !on_pe {
            cx.host(HostOp::SelectorWalk {
                cand: cb.name,
                indptr: ib.name,
                unary: ub.name,
                hp: hb.map(|b| b.name),
                tokens: tb.name,
                pred: pb.name,
                succ: sb.name,
                picks: kb.name,
                rows,
                k,
                rank,
                vocab,
                first,
            });
            return Ok(());
        }
        let (hp_ptr, has_hp) = match &hb {
            Some(h) => (Arg::Ptr(h.clone()), true),
            None => (Arg::Dummy("f32"), false),
        };
        cx.library("k_selector_walk");
        cx.call(
            "k_selector_walk",
            vec![
                Arg::Ptr(cb),
                Arg::Ptr(ib),
                Arg::Int(cand.indptr.elements().saturating_sub(1) as i64),
                Arg::Ptr(ub),
                hp_ptr,
                Arg::Bool(has_hp),
                Arg::Ptr(tb),
                Arg::Ptr(pb),
                Arg::Ptr(sb),
                Arg::Ptr(kb),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(k)),
                Arg::Int(i64::from(rank)),
                Arg::Int(i64::from(vocab)),
                Arg::Int(i64::from(first)),
            ],
        );
        Ok(())
    })
}
