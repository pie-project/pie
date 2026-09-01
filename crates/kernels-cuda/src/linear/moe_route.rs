//! `MoeRoute`: the router that reads no logits.
//!
//! One entry, `linear.moe_hash_route`, and it sits in a file of its own for
//! the reason its device text does (`kernels/linear/moe_route.cuh`): it
//! shares nothing with the four ranked routers next door in [`super::moe`] —
//! no expert staging, no shuffle reduction, no `renormalize`/`scaling` pair —
//! and the only thing it has in common with them is the shape of the answer.
//! The metal plane draws the same line between `linear/moe_route.metal` and
//! its neighbours.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe_route.cuh";

/// One thread per (token row, slot); the gather is two loads wide and wants
/// the lanes, not the block shape.
const BLOCK: u32 = 256;

/// **ROUTING BY LOOKUP, NOT BY A GATE.**
///
/// `tid2eid` is `[vocab, top_k]` I64: for every token id it NAMES the
/// `top_k` experts that id routes to, at the uniform weight `1/top_k`. No
/// router logits are computed at all — this reads the token ids and the table
/// and lands the same `routes` I32 / `weights` F32 pair every gate in
/// [`super::moe`] lands, so the sorted-MoE path behind it — [`super::moe::matmul_select`],
/// [`super::moe::weighted_sum`] — cannot tell the two apart.
///
/// DeepSeek-V4-Flash's first `num_hash_layers` layers route this way
/// (`ffn.gate.tid2eid`); every later layer carries the `noaux_tc` correction
/// bias and takes [`super::moe::topk_sqrt_softplus`]. **Substituting a
/// softmax gate for this answers DIFFERENT experts** — which is why the arm
/// this replaced in the CUDA shell refused by name rather than falling back.
///
/// **THE TABLE IS I64 AND THE ROUTES ARE I32**, and the narrowing is the
/// gather's, in the one place a 64-bit lookup meets the 32-bit route plane
/// every downstream kernel already reads an expert id from. `tid2eid` is not
/// a weight representation the trace can intern, and an expert count never
/// approaches `2^31`.
///
/// **THE TOKEN IDS ARE A 32-BIT COLUMN**, the same `RuntimeInput::Tokens`
/// stream [`crate::layout::embed`] gathers by, and an out-of-range id falls
/// to table row 0 exactly as that gather's does — so a token id at the vocab
/// boundary reads the last table row rather than off the end.
///
/// # Errors
///
/// A refusal for a zero fan-out, a zero vocabulary, an empty row count, and
/// for a `tokens x top_k` lane count that does not fit a 32-bit grid.
pub fn hash_route(
    ctx: &Ctx,
    ids: Tensor,
    tid2eid: Tensor,
    vocab: u32,
    top_k: u32,
    routes: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_hash_route";
    debug_assert!(
        matches!(ids.dtype, Dtype::I32 | Dtype::U32),
        "`{OP}` gathers by a 32-bit token id column"
    );
    debug_assert_eq!(tid2eid.dtype, Dtype::I64, "`{OP}` reads the i64 hash table");
    debug_assert_eq!(
        tid2eid.width, top_k,
        "the hash table names `top_k` experts per token id"
    );
    // The route/weight planes carry the same shape and element a ranked
    // router lands, restated here because this file does not share that
    // file's private validator.
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` lands i32 routes");
    debug_assert_eq!(weights.dtype, Dtype::F32, "`{OP}` lands f32 route weights");
    debug_assert!(
        routes.width == top_k && weights.width == top_k,
        "a routed result is the fan-out the statement states"
    );
    debug_assert_eq!(
        routes.rows, weights.rows,
        "a routed result lands one row per token row"
    );
    debug_assert_eq!(
        ids.rows, routes.rows,
        "the token ids handed over are the rows this route lands"
    );

    let rows = nonzero(OP, "rows", routes.rows)?;
    let top_k = nonzero(OP, "the fan-out this router states", top_k)?;
    let vocab = nonzero(OP, "the vocabulary this table spans", vocab)?;
    // The grid counts (token row, slot) pairs; the kernel divides back to a
    // token row, and the seat's guard is stated in those.
    let lanes = rows.checked_mul(top_k).ok_or_else(|| {
        refuse(
            OP,
            format!("the gather will not launch: {rows} tokens x {top_k} fan-out"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::hash_route_gather").apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            tid2eid.arg(),
            routes.arg(),
            weights.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, vocab)?.arg(),
            stated(OP, top_k)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
