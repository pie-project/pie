//! Per-layer embedding n-gram ids (`gather.ngram_ids`,
//! `gather.ngram_ids_chunked`) from `kernels/gather/ngram.slang`: tokens
//! hashed with the ids before them into table rows, each lane's window
//! carried in a recurrent state of `ngram - 1` i32s a slot.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const NGRAM: &str = "gather/ngram.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 64;

/// The longest n-gram the window holds.
const MAX_NGRAM: usize = 4;

/// The most heads a token hashes into.
const MAX_HEADS: usize = 32;

struct Shape {
    ngram: u32,
    heads: u32,
    heads_per_ngram: u32,
    eos: u32,
}

fn shape(
    op: &'static str,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
) -> Result<Shape, Error> {
    if mults.is_empty() || mults.len() > MAX_NGRAM {
        return Err(refuse(
            op,
            format!(
                "{} multipliers do not fit the {MAX_NGRAM}-gram ceiling",
                mults.len()
            ),
        ));
    }
    if primes.len() != offsets.len() || primes.is_empty() || primes.len() > MAX_HEADS {
        return Err(refuse(
            op,
            format!(
                "{} primes against {} offsets do not fit the {MAX_HEADS}-head ceiling",
                primes.len(),
                offsets.len()
            ),
        ));
    }
    nonzero(
        op,
        "the heads per n-gram this statement states",
        heads_per_ngram,
    )?;
    let expected = (mults.len() - 1) * heads_per_ngram as usize;
    if primes.len() != expected {
        return Err(refuse(
            op,
            format!(
                "{} heads against {} n-gram orders of {heads_per_ngram}",
                primes.len(),
                mults.len() - 1
            ),
        ));
    }
    Ok(Shape {
        ngram: mults.len() as u32,
        heads: primes.len() as u32,
        heads_per_ngram,
        eos,
    })
}

fn hash_plane(op: &'static str, hash: Tensor, shape: &Shape) -> Result<(), Error> {
    if hash.dtype != Dtype::U64 {
        return Err(refuse(
            op,
            format!(
                "the hash constants arrive as {:?} and this plane reads u64",
                hash.dtype
            ),
        ));
    }
    let want = u64::from(shape.ngram) + 2 * u64::from(shape.heads);
    let held = u64::from(hash.rows) * u64::from(hash.width);
    if held != want {
        return Err(refuse(
            op,
            format!(
                "the hash plane holds {held} constants and this hashing states {want} \
                 ({} multipliers, {} primes, {} offsets)",
                shape.ngram, shape.heads, shape.heads
            ),
        ));
    }
    Ok(())
}

fn id_planes(op: &'static str, ids: Tensor, ngram_ids: Tensor) -> Result<(), Error> {
    if ids.dtype != Dtype::I32 || ngram_ids.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the hasher reads i32 token ids and lands i32 table rows, not {:?} into {:?}",
                ids.dtype, ngram_ids.dtype
            ),
        ));
    }
    Ok(())
}

fn lanes_of(op: &'static str, indptr: Tensor) -> Result<u32, Error> {
    if indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the token CSR's boundaries are {:?}, and this hasher walks an i32 indptr",
                indptr.dtype
            ),
        ));
    }
    match indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => Ok(lanes),
        _ => Err(refuse(op, "the token CSR this fire names spans no request")),
    }
}

/// Each decoded token's n-gram table rows, one token a lane, the lane's
/// recurrent window rolled forward.
pub fn ngram_ids(
    ctx: &Ctx<'_>,
    ids: Tensor,
    state: Tensor,
    slots: Tensor,
    hash: Tensor,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.ngram_ids";
    id_planes(OP, ids, ngram_ids)?;
    let shape = shape(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    hash_plane(OP, hash, &shape)?;
    debug_assert_eq!(
        ngram_ids.width, shape.heads,
        "one output column per hashed head"
    );
    debug_assert_eq!(
        state.width,
        shape.ngram - 1,
        "the window a lane keeps is the n-gram context, one i32 per trailing id"
    );
    let rows = nonzero(OP, "rows", ids.rows)?;
    ctx.fire(
        Fire::at(NGRAM, "gather_ngram_update").apply(Grid::of([rows, 1, 1], [GROUP, 1, 1])),
        &[
            ids.arg(),
            state.arg_mut(),
            slots.arg(),
            hash.arg(),
            ngram_ids.arg_mut(),
            stated(OP, shape.ngram)?.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.heads_per_ngram)?.arg(),
            stated(OP, shape.eos)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}

/// Each request's run of tokens hashed in order, one request of the i32
/// CSR `indptr` a lane.
pub fn ngram_ids_chunked(
    ctx: &Ctx<'_>,
    ids: Tensor,
    indptr: Tensor,
    state: Tensor,
    slots: Tensor,
    hash: Tensor,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.ngram_ids_chunked";
    id_planes(OP, ids, ngram_ids)?;
    let shape = shape(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    hash_plane(OP, hash, &shape)?;
    debug_assert_eq!(
        ngram_ids.width, shape.heads,
        "one output column per hashed head"
    );
    debug_assert_eq!(
        state.width,
        shape.ngram - 1,
        "the window a lane keeps is the n-gram context, one i32 per trailing id"
    );
    nonzero(OP, "rows", ids.rows)?;
    let lanes = lanes_of(OP, indptr)?;
    ctx.fire(
        Fire::at(NGRAM, "gather_ngram_chunked").apply(Grid::of([lanes, 1, 1], [GROUP, 1, 1])),
        &[
            ids.arg(),
            indptr.arg(),
            state.arg_mut(),
            slots.arg(),
            hash.arg(),
            ngram_ids.arg_mut(),
            stated(OP, shape.ngram)?.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.heads_per_ngram)?.arg(),
            stated(OP, shape.eos)?.arg(),
            stated(OP, lanes)?.arg(),
        ],
    )
}

/// The hash plane a hasher reads: `mults | primes | offsets`.
#[must_use]
pub fn hash_constants(mults: &[u64], primes: &[u64], offsets: &[u64]) -> Vec<u64> {
    let mut plane = Vec::with_capacity(mults.len() + primes.len() + offsets.len());
    plane.extend_from_slice(mults);
    plane.extend_from_slice(primes);
    plane.extend_from_slice(offsets);
    plane
}
