//! The block draft's candidate selector walk. Reference: kernels-xla
//! `sampling::selector_walk`.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::HostOp;
use crate::plan::Held;
use crate::tensor::{RaggedTensor, Tensor};

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
    const OP: &str = "sampling.selector_walk";
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
    let on_pe = ctx.plan().holds(Held::arrays(&words, 64));
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
        cx.call(
            Kernel::SelectorWalk,
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
