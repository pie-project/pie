//! Per-layer-embedding n-gram hashing (Gemma 3n PLE, DeepSeek Engram): each
//! token's id and the ids before it (the lane's kept window first) hash to
//! one table row per head. Reference: kernels-xla `attn::ple`.
//!
//! The kept window is a recurrent state of `ngram - 1` i32 cells per slot,
//! each `id + 1` (0: nothing yet, read as `eos`). One PE hashes every
//! lane (`k_ple_ngram`); the host does when the tables outgrow a PE.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::program::HostOp;
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const MAX_NGRAM: usize = 4;
const MAX_HEADS: usize = 32;

fn ngram(
    ctx: &Ctx<'_>,
    op: &'static str,
    ids: Tensor,
    indptr: Option<Tensor>,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Tensor>,
    out: Tensor,
) -> Result<(), Error> {
    if mults.len() < 2 || mults.len() > MAX_NGRAM {
        return Err(refuse(
            op,
            format!(
                "{} multipliers do not fit the 2..={MAX_NGRAM}-gram range",
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
    if heads_per_ngram == 0 || primes.len() != (mults.len() - 1) * heads_per_ngram as usize {
        return Err(refuse(
            op,
            format!(
                "{} heads against {} n-gram orders of {heads_per_ngram}",
                primes.len(),
                mults.len() - 1
            ),
        ));
    }
    if primes.contains(&0) {
        return Err(refuse(op, "a hash prime is zero"));
    }
    if map.is_some() {
        return Err(refuse(op, "an id map (i64 table) is not taken here yet"));
    }
    expect(op, ids, &[Dtype::I32])?;
    expect(op, out, &[Dtype::I32])?;
    expect(op, state.state, &[Dtype::I32])?;
    expect(op, state.slots, &[Dtype::I32])?;
    let span = mults.len() - 1;
    if out.width as usize != primes.len() || out.rows != ids.rows || ids.rows == 0 {
        return Err(refuse(
            op,
            "the landing is one column per hashed head, one row per id",
        ));
    }
    if state.state.width as usize != span {
        return Err(refuse(
            op,
            "the window a lane keeps is the n-gram context, one i32 per trailing id",
        ));
    }
    match indptr {
        Some(ip) => {
            expect(op, ip, &[Dtype::I32])?;
            if ip.elements() < 2 {
                return Err(refuse(op, "the lanes are a CSR of at least one lane"));
            }
        }
        None => {
            if state.slots.elements() < u64::from(ids.rows) {
                return Err(refuse(op, "the slots ride the fire's rows"));
            }
        }
    }
    let rows = ids.rows;
    let (mults, primes, offsets) = (mults.to_vec(), primes.to_vec(), offsets.to_vec());
    // One PE hashes every lane (a few integer hashes a row); a fire whose
    // tables outgrow a PE hashes on the host.
    let words = ids.elements()
        + indptr.map_or(0, |ip| ip.elements())
        + state.slots.elements()
        + state.state.elements()
        + out.elements()
        + 3 * (mults.len() + primes.len()) as u64;
    let on_pe = words <= crate::linear::gemm::pe_words();
    ctx.emit(&mut |cx| {
        let ib = cx.read(ids)?;
        let ipb = match indptr {
            Some(ip) => Some(cx.read(ip)?),
            None => None,
        };
        let sb = cx.read(state.slots)?;
        cx.read(state.state)?;
        let slab = cx.write(state.state)?;
        let ob = cx.write(out)?;
        if !on_pe {
            cx.host(HostOp::PleNgramIds {
                ids: ib.name,
                indptr: ipb.map(|b| b.name),
                slots: sb.name,
                state: slab.name,
                out: ob.name,
                rows,
                eos,
                mults: mults.clone(),
                primes: primes.clone(),
                offsets: offsets.clone(),
                heads_per_ngram,
            });
            return Ok(());
        }
        let table = |cx: &mut crate::cx::Cx<'_>, hint: &str, vals: &[u64]| -> String {
            let body: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
            cx.table(hint, "u64", &body)
        };
        let mt = table(cx, "ple_mults", &mults);
        let pt = table(cx, "ple_primes", &primes);
        let ot = table(cx, "ple_offsets", &offsets);
        let window = cx.scratch_of("ple_window", MAX_NGRAM as u64, "i32");
        let past = cx.scratch_of("ple_past", MAX_NGRAM as u64, "i32");
        let (lanes, per_row, indptr_ptr) = match &ipb {
            Some(ip) => (ip.len().saturating_sub(1), false, Arg::Ptr(ip.clone())),
            None => (u64::from(rows), true, Arg::Dummy("i32")),
        };
        let span = mults.len() - 1;
        cx.library("k_ple_ngram");
        cx.call(
            "k_ple_ngram",
            vec![
                Arg::Ptr(ib),
                indptr_ptr,
                Arg::Int(lanes as i64),
                Arg::Bool(per_row),
                Arg::Ptr(sb),
                Arg::Ptr(slab.clone()),
                Arg::Int(i64::from(slab.rows)),
                Arg::Ptr(ob),
                Arg::Int(i64::from(rows)),
                Arg::Int(i64::from(eos)),
                Arg::Scratch(mt.clone(), "u64"),
                Arg::Scratch(pt.clone(), "u64"),
                Arg::Scratch(ot.clone(), "u64"),
                Arg::Int(span as i64),
                Arg::Int(primes.len() as i64),
                Arg::Int(i64::from(heads_per_ngram)),
                Arg::Scratch(window.clone(), "i32"),
                Arg::Scratch(past.clone(), "i32"),
            ],
        );
        Ok(())
    })
}

/// One row per lane: hash `[id, window..]`, then push `id` into the window.
pub fn ngram_ids(
    ctx: &Ctx<'_>,
    ids: Tensor,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Tensor>,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    ngram(
        ctx,
        "attention.ple_ngram_ids",
        ids,
        None,
        state,
        eos,
        mults,
        primes,
        offsets,
        heads_per_ngram,
        map,
        ngram_ids,
    )
}

/// Every row of a query CSR's lanes; `state.slots` is per row.
pub fn ngram_ids_chunked(
    ctx: &Ctx<'_>,
    ids: RaggedTensor,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Tensor>,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    ngram(
        ctx,
        "attention.ple_ngram_ids_chunked",
        ids.data,
        Some(ids.indptr),
        state,
        eos,
        mults,
        primes,
        offsets,
        heads_per_ngram,
        map,
        ngram_ids,
    )
}
