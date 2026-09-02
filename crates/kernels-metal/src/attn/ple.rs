//! `Ple`: qwen4's n-gram hasher — token ids in, hashed table rows out, with a
//! per-lane window of trailing ids as the one piece of sequence state.
//!
//! Ported from `kernels-cuda/kernels/attn/ple.cuh` and
//! `kernels-cuda/src/attn/ple.rs`; what differs on this plane is where the
//! hash constants live. This plane's `ArgValue` has no by-value blob seat,
//! so the constants ride one `u64` plane the shell lays down and writes once
//! at load (`engine_metal::scratch`), in the field order of the CUDA
//! aggregate with its fixed-size arrays cut to the lengths the node states:
//!
//! ```text
//! [ mults[0..ngram] ][ primes[0..heads] ][ offsets[0..heads] ]
//! ```
//!
//! Everything else is the same arithmetic in the same order, and
//! [`reference`] states it a second time in host Rust so the pins below can
//! disagree with one of the two.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, nonzero, refuse, stated};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/ple.metal";

/// The ceilings `ple.metal` declares its per-thread window and output arrays
/// at. Mirrored from `ple.cuh`'s `PLE_MAX_NGRAM` / `PLE_MAX_HEADS`, and
/// refused here rather than overrun there.
const MAX_NGRAM: usize = 4;

const MAX_HEADS: usize = 32;

/// The shape a hashing states, checked once for both arms.
///
/// `hash_arg` on the CUDA plane, minus the marshalling: the numbers do not
/// travel through here, only their counts.
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
    nonzero(op, "the heads per n-gram this statement states", heads_per_ngram)?;
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

/// The `u64` plane the constants were written into, checked against the shape
/// the node states — a plane of the wrong extent is a wrong hash and not a
/// coarse one, because every read past its end is a table row nothing carved.
fn hash_plane(op: &'static str, hash: Tensor, shape: &Shape) -> Result<(), Error> {
    if hash.dtype != Dtype::U64 {
        return Err(refuse(
            op,
            format!("the hash constants arrive as {:?} and this plane reads u64", hash.dtype),
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

/// Decode form: one new token per lane, hashed against the lane's window,
/// which then shifts by one.
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

/// Prefill form: walks the fire's request boundaries, one thread per request —
/// `ssm::causal_conv1d_chunked`'s own shape, and for its reason.
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

/// The plane the hash constants are laid down in, as both this crate's
/// shaders and `engine_metal::scratch` read it: multipliers, then primes,
/// then offsets, all `u64`. One function, two readers: the shell writes the
/// bytes at load and the shader indexes into them at every fire.
#[must_use]
pub fn hash_constants(mults: &[u64], primes: &[u64], offsets: &[u64]) -> Vec<u64> {
    let mut plane = Vec::with_capacity(mults.len() + primes.len() + offsets.len());
    plane.extend_from_slice(mults);
    plane.extend_from_slice(primes);
    plane.extend_from_slice(offsets);
    plane
}

/// The hash, in host Rust: `ple.metal`'s arithmetic restated so a box with
/// no GPU can hold the shader to it. Worth stating twice because the hash's
/// output is a table row — a hash that is off by one indexes a different
/// embedding, so every pin below is an equality rather than a band.
pub mod reference {
    /// A hashing, as the node states it.
    pub struct Hash<'a> {
        pub eos: i32,
        /// One multiplier per n-gram position; `mults.len()` is the n-gram
        /// size, and the window is one shorter.
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

        /// The window a lane keeps: `ngram − 1` trailing ids.
        #[must_use]
        pub fn span(&self) -> usize {
            self.mults.len() - 1
        }

        #[must_use]
        pub fn heads(&self) -> usize {
            self.primes.len()
        }
    }

    /// The eos-segmentation rule: a previous id is replaced by eos when a
    /// nearer previous id is eos — the window crossed a sequence boundary.
    /// The index loop mirrors the shader's own.
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

    /// The window `[t, p1, p2, …]` (newest first), hashed for every head.
    /// Order `g + 2` folds the newest `g + 2` ids and lands the
    /// `heads_per_ngram` heads at `g · heads_per_ngram`.
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

    /// The lane state's own convention: a cell holds `id + 1`, so a zeroed
    /// slot reads as "no history" and lands on eos.
    #[must_use]
    pub fn cell(state_cell: i32, eos: i32) -> i32 {
        if state_cell == 0 { eos } else { state_cell - 1 }
    }

    /// The decode arm: one token against the lane's window, which then
    /// shifts. `state` is `span` cells, updated in place.
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

    /// The chunked arm: one request's tokens in order, each hashed against
    /// the fire's own rows where reachable and the lane's state otherwise,
    /// with the state advanced once at the end.
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
        // The new window: the last `span` ids of (state ++ segment), staged
        // whole before any of it is written back.
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

#[cfg(test)]
mod tests {
    
    use super::*;
    
    use crate::probe::Probe;

    // ------------------------------------------------------------------
    // The shipped hashing, at four heads.
    //
    // **THE CONSTANTS ARE THE CHECKPOINT'S OWN.** `models::qwen_4`'s
    // `hash_constants` derives them from `seed: 1234` and the sixteen primes
    // past twenty million, and
    // `model/tests/the_qwen4_text_reads_the_two_bit_miniature.rs` holds that
    // derivation against `Qwen3.8-Flash-Next`'s published
    // `layer_multipliers` / `ngram_heads_vocab_sizes` / `ngram_heads_offsets`
    // buffers. So the numbers below are not a fixture: they are the shipped
    // model's, cut to the first four heads (`heads_per_ngram = 2` over the
    // two n-gram orders a `ngram_size = 3` has), and the offsets are that
    // cut's own prefix sums.
    // ------------------------------------------------------------------

    const MULTS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

    const PRIMES: [u64; 4] = [20_000_003, 20_000_023, 20_000_033, 20_000_047];

    const OFFSETS: [u64; 4] = [0, 20_000_003, 40_000_026, 60_000_059];

    /// `Qwen3.8-Flash-Next`'s own `eos_token_id`.
    const EOS: i32 = 248_044;

    // ---- the arithmetic pins ------------------------------------------

    // ---- the marshalling pins -----------------------------------------

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

    /// The constants plane is laid down in ONE order and both this crate's
    /// shaders and the shell's writer read it in that order.
    #[test]
    fn the_constants_plane_is_multipliers_then_primes_then_offsets() {
        let plane = hash_constants(&MULTS, &PRIMES, &OFFSETS);
        assert_eq!(plane.len(), 3 + 4 + 4);
        assert_eq!(&plane[..3], &MULTS);
        assert_eq!(&plane[3..7], &PRIMES);
        assert_eq!(&plane[7..], &OFFSETS);
    }

    /// A plane of the wrong extent is refused rather than read past: an
    /// out-of-range prime is a table row nothing carved.
    #[test]
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

    /// The head count and the n-gram orders have to agree, or a head sits at
    /// an order nothing hashed for it.
    #[test]
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
            // Four heads over two orders is two per order, not three.
            3,
            i32t(2, 6, 4),
        )
        .expect_err("four heads are not three per order");
        assert!(why.to_string().contains("4 heads against 2"), "{why}");
    }
}


/// The committed form (`engine_metal::rs`): [`ngram_ids_chunked`] over the
/// extended row run, advancing each lane's window only over its `commit`.
/// `ids` is the extended run; `ngram_ids` is the op's OWN rectangle (the
/// lane's rows at the window CSR's offsets), since the gathered-table cut
/// reads it the instant this kernel is enqueued.
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
        _ => return Err(refuse(OP, "the window CSR this fire names spans no request")),
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
