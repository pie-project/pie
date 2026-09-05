//! DFlash2's candidate selector, walked from each request's anchor —
//! `Attention::SelectorWalk`. One kernel, `attn/selector_walk.metal`.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Tensor};

const FILE: &str = "attn/selector_walk.metal";

/// Threads a request's walk runs at: sixteen lanes for each of up to
/// sixteen candidates.
const THREADS: u32 = 256;

/// The most candidates a slot may carry — the threadgroup's lane budget.
const MAX_K: u32 = THREADS / 16;

/// `picks[row]` for every row of every request: the anchor row's first
/// candidate, then the chain-argmax walk of
/// `unary[c] + ⟨pred[prev] ⊙ hp[row], succ[cand[c]]⟩` slot by slot.
///
/// `cand` is `[rows, k]` i32 with the request CSR, `unary` `[rows, k]` f32,
/// `hp` `[rows, rank]`, `tokens` `[rows]` i32 (the anchor is each span's
/// first), `pred` / `succ` `[vocab, rank]` in `hp`'s element type.
///
/// # Errors
///
/// Refuses a dtype the kernel is not stamped for, more than sixteen
/// candidates, codebooks that disagree with `hp` on the rank or with each
/// other on the vocabulary, and a `unary` or `picks` of the wrong shape.
#[allow(clippy::too_many_arguments)]
pub fn walk(
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
    let entry = dtype_dispatch!(OP, pred.dtype, { Bf16 => "selector_walk_bfloat16" });
    if succ.dtype != pred.dtype || hp.is_some_and(|h| h.dtype != pred.dtype) {
        return Err(refuse(
            OP,
            format!(
                "the codebooks are {:?} / {:?} and the projected hidden {:?}; the kernel reads one element type",
                pred.dtype, succ.dtype, hp.map(|h| h.dtype)
            ),
        ));
    }
    if first > 1 {
        return Err(refuse(OP, format!("the first slot row is {first}; a span's anchor is row 0 and its first mask row 1")));
    }
    if cand.data.dtype != Dtype::I32 || cand.indptr.dtype != Dtype::I32 || tokens.dtype != Dtype::I32 {
        return Err(refuse(OP, "the candidates, the CSR and the tokens are i32"));
    }
    if unary.dtype != Dtype::F32 || picks.dtype != Dtype::I32 {
        return Err(refuse(OP, "the unary logits are f32 and the picks i32"));
    }
    let lanes = match cand.indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => lanes,
        _ => return Err(refuse(OP, "the request CSR this fire names spans no request")),
    };
    let k = nonzero(OP, "candidates a slot", cand.data.width)?;
    if k > MAX_K {
        return Err(refuse(OP, format!("{k} candidates a slot; the walk lanes out at {MAX_K}")));
    }
    let rank = nonzero(OP, "the codebooks' rank", pred.width)?;
    if succ.width != rank || hp.is_some_and(|h| h.width != rank) {
        return Err(refuse(
            OP,
            format!(
                "the codebooks are {} / {} wide and the projected hidden {:?}",
                pred.width, succ.width, hp.map(|h| h.width)
            ),
        ));
    }
    if pred.rows != succ.rows {
        return Err(refuse(
            OP,
            format!("the codebooks disagree on the vocabulary: {} against {} rows", pred.rows, succ.rows),
        ));
    }
    let vocab = nonzero(OP, "the codebooks' vocabulary", pred.rows)?;
    if unary.rows != cand.data.rows || unary.width != k {
        return Err(refuse(OP, "the unary logits are not one row of `k` per candidate row"));
    }
    if picks.rows != cand.data.rows
        || tokens.rows != cand.data.rows
        || hp.is_some_and(|h| h.rows != cand.data.rows)
    {
        return Err(refuse(OP, "hp, tokens and picks carry one row per candidate row"));
    }
    // With no hidden term the seat is bound to the codebook (never read: the
    // kernel branches on `has_hp`), so the binding is never nil.
    let hp_arg = hp.map_or_else(|| pred.arg(), |h| h.arg());
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([THREADS, lanes, 1], [THREADS, 1, 1])),
        &[
            cand.data.arg(),
            cand.indptr.arg(),
            unary.arg(),
            hp_arg,
            tokens.arg(),
            pred.arg(),
            succ.arg(),
            picks.arg_mut(),
            stated(OP, k)?.arg(),
            stated(OP, rank)?.arg(),
            stated(OP, vocab)?.arg(),
            stated(OP, u32::from(hp.is_some()))?.arg(),
            stated(OP, first)?.arg(),
        ],
    )
}
