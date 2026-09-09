use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, nonzero, refuse, stated};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/ple.metal";

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

#[allow(clippy::too_many_arguments)]
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
    if ids.dtype != Dtype::I32 || ngram_ids.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the hasher reads i32 token ids and lands i32 table rows, not {:?} into {:?}",
                ids.dtype, ngram_ids.dtype
            ),
        ));
    }
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
        Fire::at(FILE, "ple_ngram_ids_update").apply([rows, 1, 1]),
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
        ],
    )
}

#[allow(clippy::too_many_arguments)]
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
    if ids.data.dtype != Dtype::I32 || ngram_ids.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the hasher reads i32 token ids and lands i32 table rows, not {:?} into {:?}",
                ids.data.dtype, ngram_ids.dtype
            ),
        ));
    }
    if ids.indptr.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the token CSR's boundaries are {:?}, and this hasher walks an i32 indptr",
                ids.indptr.dtype
            ),
        ));
    }
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
    let lanes = match ids.indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => lanes,
        _ => return Err(refuse(OP, "the token CSR this fire names spans no request")),
    };
    ctx.fire(
        Fire::at(FILE, "ple_ngram_ids_chunked").apply([lanes, 1, 1]),
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

#[allow(clippy::too_many_arguments)]
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
    if ids.dtype != Dtype::I32 || ngram_ids.dtype != Dtype::I32 || indptr.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the hasher reads i32 ids over an i32 CSR and lands i32 rows, not {:?}/{:?}/{:?}",
                ids.dtype, indptr.dtype, ngram_ids.dtype
            ),
        ));
    }
    let shape = shape(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    hash_plane(OP, hash, &shape)?;
    nonzero(OP, "extended rows", ids.rows)?;
    let lanes = match indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => lanes,
        _ => {
            return Err(refuse(
                OP,
                "the window CSR this fire names spans no request",
            ));
        }
    };
    ctx.fire(
        Fire::at(FILE, "ple_ngram_ids_committed").apply([lanes, 1, 1]),
        &[
            ids.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            state.state.arg_mut(),
            hash.arg(),
            ngram_ids.arg_mut(),
            stated(OP, shape.ngram)?.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.heads_per_ngram)?.arg(),
            stated(OP, shape.eos)?.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::probe::Probe;

    const MULTS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

    const PRIMES: [u64; 4] = [20_000_003, 20_000_023, 20_000_033, 20_000_047];

    const OFFSETS: [u64; 4] = [0, 20_000_003, 40_000_026, 60_000_059];

    const EOS: i32 = 248_044;

    fn i32t(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::I32)
    }

    fn pool() -> RecurrentPool {
        let bank = i32t(10, 8, 2);
        RecurrentPool {
            state: bank,
            slots: Tensor::new(11, 1, 8, Dtype::U32),
            conv_state: bank,
            new_conv_state: bank,
        }
    }

    fn plane() -> Tensor {
        Tensor::new(12, 1, (MULTS.len() + 2 * PRIMES.len()) as u32, Dtype::U64)
    }

    #[test]
    fn ple_every_case() {
        the_constants_plane_is_multipliers_then_primes_then_offsets();
        a_hash_plane_the_shape_does_not_describe_is_refused();
        a_head_count_the_orders_do_not_cover_is_refused();
    }

    fn the_constants_plane_is_multipliers_then_primes_then_offsets() {
        let plane = hash_constants(&MULTS, &PRIMES, &OFFSETS);
        assert_eq!(plane.len(), 3 + 4 + 4);
        assert_eq!(&plane[..3], &MULTS);
        assert_eq!(&plane[3..7], &PRIMES);
        assert_eq!(&plane[7..], &OFFSETS);
    }

    fn a_hash_plane_the_shape_does_not_describe_is_refused() {
        let probe = Probe::default();
        let short = Tensor::new(12, 1, 4, Dtype::U64);
        let why = ngram_ids(
            &probe,
            i32t(1, 6, 1),
            &pool(),
            short,
            EOS as u32,
            &MULTS,
            &PRIMES,
            &OFFSETS,
            2,
            i32t(2, 6, 4),
        )
        .expect_err("a four-constant plane cannot hold eleven");
        assert!(why.to_string().contains("11"), "{why}");
        assert!(probe.fires().is_empty(), "nothing was encoded");
    }

    fn a_head_count_the_orders_do_not_cover_is_refused() {
        let probe = Probe::default();
        let why = ngram_ids(
            &probe,
            i32t(1, 6, 1),
            &pool(),
            plane(),
            EOS as u32,
            &MULTS,
            &PRIMES,
            &OFFSETS,
            3,
            i32t(2, 6, 4),
        )
        .expect_err("four heads are not three per order");
        assert!(why.to_string().contains("4 heads against 2"), "{why}");
    }
}
