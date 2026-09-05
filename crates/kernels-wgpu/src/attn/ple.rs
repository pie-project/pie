#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/ple.wgsl";

const GROUP: u32 = 64;

const MAX_NGRAM: usize = 4;

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

pub fn ngram_ids(
    ctx: &Ctx<'_>,
    ids: Tensor,
    state: &RecurrentPool,
    hash: Tensor,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids";
    id_planes(OP, ids, ngram_ids)?;
    let shape = shape(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    hash_plane(OP, hash, &shape)?;
    debug_assert_eq!(
        ngram_ids.width, shape.heads,
        "one output column per hashed head"
    );
    debug_assert_eq!(
        state.state.width,
        shape.ngram - 1,
        "the window a lane keeps is the n-gram context, one i32 per trailing id"
    );
    let rows = nonzero(OP, "rows", ids.rows)?;
    ctx.fire(
        Fire::at(FILE, "ple_ngram_ids_update").apply(Grid::of([rows, 1, 1], [GROUP, 1, 1])),
        &[
            ids.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
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

pub fn ngram_ids_chunked(
    ctx: &Ctx<'_>,
    ids: RaggedTensor,
    state: &RecurrentPool,
    hash: Tensor,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids_chunked";
    id_planes(OP, ids.data, ngram_ids)?;
    let shape = shape(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    hash_plane(OP, hash, &shape)?;
    debug_assert_eq!(
        ngram_ids.width, shape.heads,
        "one output column per hashed head"
    );
    debug_assert_eq!(
        state.state.width,
        shape.ngram - 1,
        "the window a lane keeps is the n-gram context, one i32 per trailing id"
    );
    nonzero(OP, "rows", ids.data.rows)?;
    let lanes = lanes_of(OP, ids.indptr)?;
    ctx.fire(
        Fire::at(FILE, "ple_ngram_ids_chunked").apply(Grid::of([lanes, 1, 1], [GROUP, 1, 1])),
        &[
            ids.data.arg(),
            ids.indptr.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
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

#[must_use]
pub fn hash_constants(mults: &[u64], primes: &[u64], offsets: &[u64]) -> Vec<u64> {
    let mut plane = Vec::with_capacity(mults.len() + primes.len() + offsets.len());
    plane.extend_from_slice(mults);
    plane.extend_from_slice(primes);
    plane.extend_from_slice(offsets);
    plane
}

pub mod reference {
    pub struct Hash<'a> {
        pub eos: i32,

        pub mults: &'a [u64],
        pub primes: &'a [u64],
        pub offsets: &'a [u64],
        pub heads_per_ngram: usize,
    }

    impl Hash<'_> {
        #[must_use]
        pub fn ngram(&self) -> usize {
            self.mults.len()
        }

        #[must_use]
        pub fn span(&self) -> usize {
            self.mults.len() - 1
        }

        #[must_use]
        pub fn heads(&self) -> usize {
            self.primes.len()
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn mask_window(h: &Hash, window: &mut [i32]) {
        let mut crossed = false;
        for p in 1..h.ngram() {
            if crossed {
                window[p] = h.eos;
            }
            if window[p] == h.eos {
                crossed = true;
            }
        }
    }

    #[must_use]
    #[allow(clippy::needless_range_loop)]
    pub fn hash_row(h: &Hash, window: &[i32]) -> Vec<i32> {
        let mut out = vec![0i32; h.heads()];
        for order in 2..=h.ngram() {
            let mut mixed = (window[0] as i64 as u64).wrapping_mul(h.mults[0]);
            for p in 1..order {
                mixed ^= (window[p] as i64 as u64).wrapping_mul(h.mults[p]);
            }
            let base = (order - 2) * h.heads_per_ngram;
            for k in 0..h.heads_per_ngram {
                let head = base + k;
                out[head] = (mixed % h.primes[head] + h.offsets[head]) as i32;
            }
        }
        out
    }

    #[must_use]
    pub fn cell(state_cell: i32, eos: i32) -> i32 {
        if state_cell == 0 { eos } else { state_cell - 1 }
    }

    #[must_use]
    pub fn step(h: &Hash, id: i32, state: &mut [i32]) -> Vec<i32> {
        let span = h.span();
        let mut window = vec![0i32; h.ngram()];
        window[0] = id;
        for p in 1..=span {
            window[p] = cell(state[span - p], h.eos);
        }
        mask_window(h, &mut window);
        let out = hash_row(h, &window);
        for p in 0..span.saturating_sub(1) {
            state[p] = state[p + 1];
        }
        state[span - 1] = id + 1;
        out
    }

    #[must_use]
    pub fn walk(h: &Hash, ids: &[i32], state: &mut [i32]) -> Vec<i32> {
        let span = h.span();
        let mut out = Vec::with_capacity(ids.len() * h.heads());
        for t in 0..ids.len() {
            let mut window = vec![0i32; h.ngram()];
            window[0] = ids[t];
            for p in 1..=span {
                window[p] = if t >= p {
                    ids[t - p]
                } else {
                    cell(state[span - (p - t)], h.eos)
                };
            }
            mask_window(h, &mut window);
            out.extend(hash_row(h, &window));
        }

        let mut next = vec![0i32; span];
        for p in 0..span {
            let src = ids.len() as isize - span as isize + p as isize;
            next[p] = if src >= 0 {
                ids[src as usize] + 1
            } else {
                state[p + ids.len()]
            };
        }
        state[..span].copy_from_slice(&next);
        out
    }
}

pub fn ngram_ids_committed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    indptr: Tensor,
    committed: &crate::attn::ssm::Committed,
    state: &RecurrentPool,
    hash: Tensor,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids_committed";
    id_planes(OP, ids, ngram_ids)?;
    let shape = shape(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    hash_plane(OP, hash, &shape)?;
    nonzero(OP, "extended rows", ids.rows)?;
    let lanes = lanes_of(OP, indptr)?;
    ctx.fire(
        Fire::at(FILE, "ple_ngram_ids_committed").apply(Grid::of([lanes, 1, 1], [GROUP, 1, 1])),
        &[
            ids.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            state.state.arg_mut(),
            hash.arg(),
            ngram_ids.arg_mut(),
            stated(OP, committed.lane0)?.arg(),
            stated(OP, shape.ngram)?.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.heads_per_ngram)?.arg(),
            stated(OP, shape.eos)?.arg(),
            stated(OP, lanes)?.arg(),
        ],
    )
}
