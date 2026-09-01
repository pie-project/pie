//! The PLE n-gram hasher (`attention.ple_ngram_ids`, qwen4): token ids in,
//! hashed table rows out, with a per-lane window of trailing ids as the one
//! piece of sequence state. The hash constants ride one by-value aggregate
//! (`ArgValue::Bytes`) — they are trace constants, derived from the config's
//! seed, and no checkpoint plane is read to know them.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, nonzero, refuse, stated};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const FILE: &str = "attn/ple.cuh";

const BLOCK: u32 = 128;

/// The device-side aggregate, mirrored field for field from `ple.cuh`'s
/// `PleHash`. `#[repr(C)]` because the bytes cross the launch ABI as one
/// parameter.
#[repr(C)]
struct PleHash {
    mults: [u64; 4],
    primes: [u64; 32],
    offsets: [u64; 32],
    ngram: i32,
    heads: i32,
    heads_per_ngram: i32,
    eos: i32,
}

fn hash_arg(
    op: &'static str,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
) -> Result<PleHash, Error> {
    if mults.is_empty() || mults.len() > 4 {
        return Err(refuse(
            op,
            format!("{} multipliers do not fit the 4-gram ceiling", mults.len()),
        ));
    }
    if primes.len() != offsets.len() || primes.is_empty() || primes.len() > 32 {
        return Err(refuse(
            op,
            format!(
                "{} primes against {} offsets do not fit the 32-head ceiling",
                primes.len(),
                offsets.len()
            ),
        ));
    }
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
    let mut h = PleHash {
        mults: [0; 4],
        primes: [0; 32],
        offsets: [0; 32],
        ngram: mults.len() as i32,
        heads: primes.len() as i32,
        heads_per_ngram: heads_per_ngram as i32,
        eos: eos as i32,
    };
    h.mults[..mults.len()].copy_from_slice(mults);
    h.primes[..primes.len()].copy_from_slice(primes);
    h.offsets[..offsets.len()].copy_from_slice(offsets);
    Ok(h)
}

/// Decode form: one new token per lane, hashed against the lane's window,
/// which then shifts by one.
#[allow(clippy::too_many_arguments)]
pub fn ngram_ids(
    ctx: &Ctx,
    ids: Tensor,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    ngram_ids: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` reads i32 token ids");
    debug_assert_eq!(ngram_ids.dtype, Dtype::I32, "`{OP}` lands i32 table rows");
    let h = hash_arg(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    debug_assert_eq!(
        ngram_ids.width,
        primes.len() as u32,
        "one output column per hashed head"
    );
    let rows = nonzero(OP, "rows", ids.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ple_ngram_ids_update")
            .apply(Launch::flat(rows, BLOCK)),
        &[
            ids.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
            ngram_ids.arg(),
            stated(OP, rows)?.arg(),
            ArgValue::Bytes {
                ptr: std::ptr::from_ref(&h).cast(),
                len: size_of::<PleHash>(),
            },
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// Prefill form: walks the fire's ambient request boundaries, as the chunked
/// convolution does.
#[allow(clippy::too_many_arguments)]
pub fn ngram_ids_chunked(
    ctx: &Ctx,
    ids: RaggedTensor,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    ngram_ids: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids_chunked";
    debug_assert_eq!(ids.data.dtype, Dtype::I32, "`{OP}` reads i32 token ids");
    debug_assert_eq!(ngram_ids.dtype, Dtype::I32, "`{OP}` lands i32 table rows");
    let h = hash_arg(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    let lanes = nonzero(OP, "request lanes", ids.indptr.rows.saturating_sub(1))?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::ple_ngram_ids_chunked")
            .apply(Launch::grid([lanes, 1, 1], [BLOCK, 1, 1])),
        &[
            ids.data.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            ids.indptr.arg(),
            state.slot_stride_elems.arg(),
            ngram_ids.arg(),
            // The three rs seats and the segment origin, exactly as the
            // chunked convolution binds them: state advances only over the
            // committed prefix of a fold-predicated row.
            state.write_state.arg(),
            state.write_state_mask.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Bytes {
                ptr: std::ptr::from_ref(&h).cast(),
                len: size_of::<PleHash>(),
            },
            // **THE STAGED-GEOMETRY SEAT, READ ON THE LANE AXIS** (the
            // chunked-arm wave). One block per REQUEST, so this arm spends
            // `win[2]` where the decode form above spends `win[0]`, and
            // `win[3]` to name the fire lane of its window-local request `r` —
            // the tables `Run::recurrent_absolute` hands over whole. `win[1]`
            // shifts the token and table planes; the window's own CSR stays on
            // the launch-local ordinal. Passed UNCONDITIONALLY, which is what
            // puts the name on `engine_cuda::SHIFTED`.
            ctx.stage(),
        ],
    )
}
