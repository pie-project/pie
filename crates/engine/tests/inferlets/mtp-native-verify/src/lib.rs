//! **MTP Stage 2 — PTIR-native explicit draft→verify→accept** (bravo), on the
//! `inferlet::ptir` bridge. The speculative decode is a single traced eDSL
//! epilogue per window — the target VERIFY (match + bonus tail) AND the next
//! window's DRAFTING (native MTP argmax) in ONE lowered graph:
//!
//!   `picked = logits[k+1, vocab].argmax()`         // [k+1] target greedy (k verify + bonus)
//!   `head   = gather(picked, lanes_k)`              // picked[0..k]
//!   `hit    = head.eq(draft)`                       // [k] bool — verify vs the EMBEDDED drafts
//!   `n_acc  = reduce_sum(cumprod(hit))`              // scalar accepted count 0..k
//!   `keep   = broadcast(n_acc).ge(lanes_k1)`          // [k+1] i <= n_acc
//!   commit  = select(keep, picked, -1)                // accepted prefix + BONUS@n_acc, then -1
//!   drafts' = argmax(mtp_logits(k))                   // [k] FRESH drafts — NEXT window's proposals
//!
//! `draft` is read DEVICE-ALIAS off the SAME embedded window tokens (`toks.read()`,
//! non-consuming peek — rows `1..=k`, the previous step's MTP proposals fed as
//! this window's input), not a separately submitted draft channel; each fresh
//! pass seeds its `toks` channel from the host's running commit/draft bookkeeping
//! (the ptir-bridge equivalent of the deleted
//! `sampling::program::mtp_native_verify` + `resolve_bindings` blob-submit
//! surface — there is no lower-level "device-resident retain" primitive on the
//! current `inferlet::ptir` surface, so drafts round-trip through host state
//! each iteration, exactly mirroring this inferlet's ORIGINAL host-round-trip
//! dataflow, contrasted with `mtp-specdecode`'s attempted device residency).
//!
//! Bootstrap fire: `prompt + (k-1)` fillers ⇒ the last k positions carry k
//! logit rows ⇒ both `logits` (target) and `mtp_logits` (drafts) get REAL k-row
//! data (no anchor-row collapse) — yields the seed (row-0 target argmax) + the
//! first REAL drafts (mtp argmax) for window 1.
//!
//! **The accept step is fold-commit, because this is a LINEAR model.** The
//! verify fire cannot fold: how many of its k drafts are real is decided by
//! its own logits, and a fold is irreversible — there is no recurrent-state
//! equivalent of dropping a KV slot. So each window runs in two fires:
//!
//!   1. verify — `fold_len = 0`: the window's pre-recurrence activations land
//!      in the RS buffer and the folded boundary does not move, leaving the
//!      whole window abandonable while its logits are inspected.
//!   2. commit — `fold_len = clen`: with the accepted length now known, the
//!      boundary advances through exactly the accepted prefix plus its bonus
//!      token, replaying the buffered activations instead of recomputing the
//!      in-projection. Freeing the remaining slabs abandons the rejected tail,
//!      which never touched the recurrent state.
//!
//! The window is also ONE request row of `k+1` causal tokens, not the `k+1`
//! -request staircase this inferlet used to build. The two are equivalent for
//! attention, but a linear model carries one recurrent state per REQUEST, so
//! the staircase would demand `k+1` working sets holding divergent copies of a
//! single sequence's state.
//!
//! JSON/plain input: optional draft window `k` (default 4).

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

const PROMPT: &str = "The quick brown fox jumps over";
const MAX_TOKENS: u32 = 16;
const PAGE_T: u32 = 16;

/// Committed length of a sentinel `[k+1]` tail = the count before the first
/// `-1` (accepted prefix + the bonus at lane `n_acc`), always ≥ 1.
fn committed_len(tail: &[i32]) -> usize {
    tail.iter().take_while(|&&t| t >= 0).count()
}

fn bind_single_sequence<B>(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    token_count: u32,
    pool_pages: u32,
    readout: &[u32],
    rs: &[RsWorkingSet],
    rs_geom: RsGeometry<'_, B>,
) -> Result<()>
where
    B: std::ops::RangeBounds<u32>,
{
    let embed_indptr = Channel::from([0u32, token_count]).named("embed_indptr");
    let positions = Channel::from_iter(0..token_count).named("positions");
    let pages = Channel::from_iter(0..pool_pages).named("pages");
    let page_indptr = Channel::from([0u32, token_count.div_ceil(PAGE_T)]).named("page_indptr");
    let w_slot = Channel::from_iter((0..token_count).map(|p| p / PAGE_T)).named("w_slot");
    let w_off = Channel::from_iter((0..token_count).map(|p| p % PAGE_T)).named("w_off");
    let readout = Channel::from(readout).named("readout");
    pass.embed(toks, &embed_indptr)?;
    pass.readout(&readout)?;
    pass.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        rs,
        rs_geom,
    )
}

/// One request row of `count` tokens starting at absolute position
/// `first_pos`, reading out the rows named by `readout` (row indices within
/// the window).
///
/// This replaces the old `k+1`-REQUEST staircase. On a pure-attention model
/// the staircase and a single causal row are equivalent — row `i` saw
/// `seq_len + i` keys either way — but a linear model cannot express the
/// staircase at all: it carries one recurrent state per REQUEST, so `k+1`
/// rows would need `k+1` working sets holding `k+1` divergent copies of one
/// sequence's state. One row of `k+1` causal tokens is the shape that has a
/// single state to advance, which is the shape the buffer is built for.
fn bind_window<B>(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    first_pos: u32,
    count: u32,
    pool_pages: u32,
    readout: &[u32],
    rs: &[RsWorkingSet],
    rs_geom: RsGeometry<'_, B>,
) -> Result<()>
where
    B: std::ops::RangeBounds<u32>,
{
    // `count == 0` is a row that spans NO TOKENS: "compute nothing, only move
    // the recurrent boundary". The emptiness lives in the token CSR, not in
    // the channels -- the IR has no zero-sized tensor -- so every per-token
    // channel carries one unreferenced element and the indptr says zero.
    let pad = count.max(1);
    let embed_indptr = Channel::from([0u32, count]).named("w_embed_indptr");
    let positions = Channel::from_iter(first_pos..first_pos + pad).named("w_positions");
    let pages = Channel::from_iter(0..pool_pages).named("w_pages");
    let page_indptr = Channel::from([0u32, (first_pos + count).div_ceil(PAGE_T).min(pool_pages)])
        .named("w_page_indptr");
    let w_slot =
        Channel::from_iter((first_pos..first_pos + pad).map(|p| p / PAGE_T)).named("w_w_slot");
    let w_off =
        Channel::from_iter((first_pos..first_pos + pad).map(|p| p % PAGE_T)).named("w_w_off");
    // A fold fire reads nothing, and `Channel::from([])` cannot express
    // that (an empty shape has no vector length). Omitting `readout` entirely
    // is the honest encoding of "this fire samples no rows" — and the driver
    // requires it, since a fold returns before the output projection.
    if !readout.is_empty() {
        let readout = Channel::from(readout).named("w_readout");
        pass.readout(&readout)?;
    }
    pass.embed(toks, &embed_indptr)?;
    pass.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        rs,
        rs_geom,
    )
}

/// Bootstrap fire over `prompt + (k-1)` fillers: yields the seed (row-0 target
/// argmax at the prompt's REAL last position) + the first REAL `[k]` drafts
/// (native MTP argmax) for window 1. No verify (nothing to verify yet).
async fn bootstrap(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    pipeline: &Pipeline,
    prompt: &[u32],
    k: u32,
    max_pages: u32,
) -> Result<(i32, Vec<i32>)> {
    let l = prompt.len() as u32;
    let mut window: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    window.extend(std::iter::repeat(0i32).take((k - 1) as usize));
    let n = l + k - 1;

    let toks = Channel::from(window).named("b_toks");
    let seed_out = Channel::new([1], dtype::i32).named("b_seed");
    let drafts_out = Channel::new([k], dtype::i32).named("b_drafts");

    // k read-out rows (the last k positions) ⇒ intrinsics::logits() AND
    // intrinsics::mtp_logits(k) both declare [k, vocab] — real k-row data.
    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd = ForwardPass::new();
    let kv_len = Channel::from([n]).named("b_kv_len");
    bind_single_sequence(
        &fwd,
        ws,
        &toks,
        &kv_len,
        n,
        max_pages,
        &readout,
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    fwd.epilogue(move || {
        let picked = reduce_argmax(intrinsics::logits()); // [k] target argmax
        let seed = gather(&picked, Tensor::constant(vec![0u32])); // [1] row-0
        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let drafts = reduce_argmax(mtp); // [k] fresh drafts
        seed_out.put(&seed);
        drafts_out.put(&drafts);
    });

    fwd.submit(pipeline).context("bootstrap submit")?;
    let seed = seed_out
        .take_host::<Vec<i32>>()
        .await?
        .first()
        .copied()
        .ok_or_else(|| "bootstrap: empty seed".to_string())?;
    let drafts = drafts_out.take_host::<Vec<i32>>().await?;
    Ok((seed, drafts))
}

/// One `[k+1]`-wide verify window: embed `[seed, draft]` (a freshly seeded
/// channel each iteration) at positions derived from the pre-envelope
/// `seq_len` cursor, verify `draft` (device-alias
/// peeked off the SAME embedded tokens) against the target's per-row argmax,
/// and draft the NEXT window natively off `mtp_logits`. Returns `(commit
/// [k+1], next_drafts [k])`.
///
/// CONSTRUCTS the pass without submitting it. When `clen_out` is present the
/// epilogue also publishes the committed length on device, and the commit fire
/// that consumes it must already have claimed that channel -- so this fire
/// cannot submit itself.
///
/// The returned fourth channel ECHOES that same device-computed length to the
/// host. It is a separate channel, not the descriptor one: a channel a later
/// pass claims as a descriptor declares `HostRole::None`, so it cannot also be
/// a terminal host-read output. The echo exists purely so the loop can assert,
/// WITHIN one run, that the number the driver folded is the number the host
/// would have chosen -- an equality that holds regardless of which decode
/// trajectory this particular run took.
#[allow(clippy::too_many_arguments)]
fn build_verify(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    k: u32,
    seed: i32,
    draft: &[i32],
    seq_len: u32,
    max_pages: u32,
    fold_len: u32,
    clen_out: Option<&Channel>,
) -> Result<(ForwardPass, Channel, Channel, Option<Channel>)> {
    let kp1 = k + 1;
    let mut window: Vec<i32> = vec![seed];
    window.extend_from_slice(draft);

    let toks = Channel::from(window).named("v_toks");
    let commit_out = Channel::new([kp1], dtype::i32).named("v_commit");
    let drafts_out = Channel::new([k], dtype::i32).named("v_drafts");
    let commit_out_h = commit_out.clone();
    let drafts_out_h = drafts_out.clone();
    let clen_sink = clen_out.cloned();
    let clen_echo = clen_out.map(|_| Channel::new([1], dtype::u32).named("v_clen_echo"));
    let clen_echo_h = clen_echo.clone();

    let fwd = ForwardPass::new();
    let kv_len = Channel::from([seq_len + kp1]).named("v_kv_len");
    let readout: Vec<u32> = (0..kp1).collect();
    // This fire does NOT fold its own tokens. The window is `seed + k
    // drafts`, and how many of those k are real is not known until this
    // fire's own logits come back. Folding them would advance the recurrent
    // state through a tail that may be rejected, and a fold is irreversible.
    // So the window's pre-recurrence activations are parked in the buffer
    // with its own boundary held still.
    //
    // `fold_len` is the PREVIOUS window's accepted prefix, which by now IS
    // known — it sits behind this fire's new tokens in the buffer, the
    // rejected tail between them having been discarded before the fire was
    // built. Folding it here is what makes the commit fire unnecessary: one
    // fire per window instead of two. (§10.2.9's fold-behind shape:
    // `0 < n <= b` with `t > 0`.)
    let fold_len = Channel::from([fold_len]).named("v_fold_len");
    bind_window(
        &fwd,
        ws,
        &toks,
        &kv_len,
        seq_len,
        kp1,
        max_pages,
        &readout,
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: Some(&fold_len),
            buffer: ..,
        },
    )
    .context("verify binding")?;
    fwd.epilogue(move || {
        // Device-alias read: peek the embedded window (NOT a resubmitted draft
        // channel) and gather rows 1..=k as the verify operand.
        let win = toks.read(); // [k+1] i32
        let draft_v = gather(&win, Tensor::constant((1..=k).collect::<Vec<u32>>())); // [k]
        let picked = reduce_argmax(intrinsics::logits()); // [k+1] target (k verify + bonus)
        let head = gather(&picked, iota(k)); // [k] picked[0..k]
        let hit = eq(&head, &draft_v); // [k] bool
        let ones = broadcast(1.0f32, [k]);
        let zeros = broadcast(0.0f32, [k]);
        let run = cumprod(select(&hit, &ones, &zeros)); // [k]
        let n_acc = cast(reduce_sum(run), dtype::u32); // accepted-prefix length
        let keep = ge(broadcast(&n_acc, [kp1]), iota(kp1)); // [k+1] i <= n_acc
        let neg1 = broadcast(-1i32, [kp1]);
        let commit = select(&keep, &picked, &neg1); // accepted prefix + bonus + -1s

        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let next_drafts = reduce_argmax(mtp); // [k] fresh drafts — NEXT window

        // The committed length, on device. `n_acc` above is already the whole
        // computation; the host's `committed_len` is just this number read out
        // of a sentinel tail. Publishing it directly is what lets the commit
        // fire be traced BEFORE anyone knows its value.
        if let Some(sink) = clen_sink {
            let clen = cast(&n_acc + 1u32, dtype::u32);
            if let Some(echo) = clen_echo {
                echo.put(&clen);
            }
            sink.put(&clen);
        }

        commit_out.put(&commit);
        drafts_out.put(&next_drafts);
    });

    Ok((fwd, commit_out_h, drafts_out_h, clen_echo_h))
}

/// The same commit, with the accepted length never read back to the host.
///
/// This is the point of the whole fold-commit path. `commit_window` above
/// cannot be TRACED until `clen` is known, so the host has to await the verify
/// fire between two fires that are otherwise back to back. Here `fold_len` is
/// a channel the verify epilogue fills, the host plans against its own UPPER
/// BOUND -- the row's whole live buffer -- and the driver substitutes the real
/// value and clamps it.
///
/// Two consequences shape this function.
///
/// **It carries NO TOKENS AT ALL.** It used to re-run the full window, because
/// the accepted prefix is not host-known here and so its length could not
/// appear in the KV geometry -- and re-running was free, the verify fire
/// having already written KV for exactly that span. But a row spanning no
/// tokens says the same thing without the pretence: this fire computes
/// nothing, it only moves the boundary. That is now also the ONLY way to say
/// it, since the planner reads a pure replay off the empty row rather than
/// off `fold-len <= buffered`.
///
/// **It is CONSTRUCTED before the verify fire is submitted.** A channel a
/// later pass consumes as a descriptor must be claimed by that pass first, or
/// the producer infers it as a terminal host-read output and the two
/// declarations conflict at bind. Construction order, not annotation.
fn build_commit(
    rs: &RsWorkingSet,
    ws: &WorkingSet,
    seq_len: u32,
    max_pages: u32,
    fold_len: &Channel,
) -> Result<ForwardPass> {
    let toks = Channel::from([0i32]).named("c_toks");

    let fwd = ForwardPass::new();
    let kv_len = Channel::from([seq_len]).named("c_kv_len");
    bind_window(
        &fwd,
        ws,
        &toks,
        &kv_len,
        seq_len,
        0,
        max_pages,
        &[],
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: Some(fold_len),
            buffer: ..,
        },
    )
    .context("commit binding")?;
    fwd.epilogue(|| {});
    Ok(fwd)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // `k` or `k:device`. The device mode keeps the accepted length on the GPU
    // instead of reading it back to choose the commit's fold length; both
    // modes must decode the identical string, which is what the harness pins.
    let (k_str, device) = match input.trim().split_once(':') {
        Some((k, "device")) => (k, true),
        _ => (input.trim(), false),
    };
    let k: u32 = k_str.trim().parse().unwrap_or(4).max(2);
    if wit_model::pass_kind() == wit_model::ForwardKind::Attention {
        // The buffer this inferlet's accept step depends on only exists on a
        // linear model. On a pure-attention model a rejected tail is discarded
        // by dropping KV slots and none of the fold-commit machinery applies.
        return Ok("skipped: mtp-native-verify needs a linear model".to_string());
    }
    let ws = WorkingSet::new();
    let rs = RsWorkingSet::new();
    let rs_page = rs.buffer_page_size();
    if rs_page == 0 {
        return Err("linear model reports no RS buffer page size".to_string());
    }
    let mut prompt = wit_model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let max_pages = (prompt.len() as u32 + MAX_TOKENS + k + 1).div_ceil(PAGE_T);
    ws.reserve(max_pages).context("ws.reserve")?;

    // ONE pipeline for the whole stream (R4-4): the bootstrap and every
    // verify window continue the same sequential decode, so all their fires
    // submit here. The loop is acceptance-driven (the last submit is not
    // knowable at submit time), so the stream ends with a close after the
    // final drain instead of a final-submit marker.
    let pipeline = Pipeline::new();

    // Bootstrap: real seed + real first drafts off the prompt's REAL last position.
    let (seed0, draft0) = bootstrap(&ws, &rs, &pipeline, &prompt, k, max_pages).await?;
    let mut seq_len: u32 = prompt.len() as u32 + k - 1;

    let mut committed: Vec<u32> = prompt.clone();
    committed.push(seed0 as u32);
    let mut seed = seed0;
    let mut draft = draft0;
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut clen_agreements = 0usize;
    let mut clen_nontrivial = 0usize;
    let mut generated: u32 = 1;

    // North-star spec-decode loop: verify the embedded drafts against the
    // target while folding the PREVIOUS window's accepted prefix → discard
    // the rejected tail → take the fresh native-MTP drafts as the next
    // window's proposals → repeat.
    //
    // The fold and the discard are the two ends of the buffer, and having
    // both is what collapses a window to ONE fire. `fold-len` moves the
    // folded boundary right through the tokens the previous window earned;
    // `discard-buffered` moves the live end left over the tokens it did not.
    // Until the second existed, dropping the rejected tail meant
    // `free_buffer`, which empties the buffer wholesale — so the accepted
    // prefix had to be folded away first, in a fire of its own, because its
    // length is not knowable until the verify has run.
    //
    // The DEVICE path cannot fuse, and the reason is worth stating: its fold
    // length is never read back, so the store holds only an upper bound on
    // the live buffer and refuses to plan a replay against it. A fused fire
    // must replay the accepted prefix it is folding. Device-residency buys
    // back-to-back enqueue WITHOUT a host read; reading back — which this
    // loop does anyway, for the drafts — buys the fusion. They are
    // alternatives, not a ladder.
    let window_slabs = (k + 1).div_ceil(rs_page);
    let kp1 = k + 1;
    // The previous window's accepted prefix, still buffered, waiting for the
    // next fire to fold it. Zero on the first pass: nothing precedes it.
    let mut pending_fold: u32 = 0;
    while generated < MAX_TOKENS {
        if device {
            // Two fires, an empty buffer between windows.
            rs.alloc_buffer(window_slabs).context("rs.alloc_buffer")?;
        } else {
            // One fire. The buffer never empties: it carries the previous
            // window's accepted prefix into this fire, which folds it while
            // appending its own tokens past it. Reserve only what the live
            // content plus this window actually needs -- `advance_fold`
            // releases the head pages the fold covers, so the reservation
            // stays bounded instead of growing once per window.
            let live = pending_fold + kp1 + rs_page;
            while rs.buffer_size() * rs_page < live {
                rs.alloc_buffer(window_slabs.max(1))
                    .context("rs.alloc_buffer")?;
            }
        }

        // The device path builds the COMMIT first -- it must claim `clen`
        // before the verify fire that fills it is submitted -- then enqueues
        // both fires back to back with nothing awaited in between. The host
        // path has no commit to build.
        let clen_ch = device.then(|| Channel::new([1], dtype::u32).named("v_clen"));
        let (verify, commit_out, drafts_out, clen_echo) = build_verify(
            &ws,
            &rs,
            k,
            seed,
            &draft,
            seq_len,
            max_pages,
            pending_fold,
            clen_ch.as_ref(),
        )?;
        // The commit is CONSTRUCTED here -- after the verify pass is built but
        // before it is submitted. F8 only requires the consumer to claim the
        // channel before the producer is submitted; building it earlier would
        // also take its KV pages out of the working set ahead of the verify's,
        // which is a different trace.
        let device_commit = match &clen_ch {
            Some(ch) => Some(build_commit(&rs, &ws, seq_len, max_pages, ch)?),
            None => None,
        };

        verify.submit(&pipeline).context("verify submit")?;

        if let Some(c) = device_commit.as_ref() {
            c.submit(&pipeline).context("device commit submit")?;
        }

        let commit = commit_out.take_host::<Vec<i32>>().await?;
        let drafts = drafts_out.take_host::<Vec<i32>>().await?;
        let clen = committed_len(&commit); // n_acc accepted + 1 bonus (≥ 1)

        // The device path's whole claim is that the number the DRIVER folded is
        // the number the host would have chosen. Assert it here, in the same
        // run: `clen_echo` is the very expression that fed the `fold-len`
        // descriptor, so an equality failure means the boundary moved somewhere
        // the host never sanctioned.
        //
        // This is a WITHIN-run invariant on purpose. Comparing a host-mode
        // decode against a device-mode decode token-for-token does not work on
        // this model: the drafts feed back into the next window, and a single
        // argmax tie broken the other way by ordinary bf16 reduction-order
        // noise forks the whole trajectory. Three identical host-mode launches
        // in one engine boot produce different token streams.
        if let Some(echo) = clen_echo {
            let seen = echo.take_host::<Vec<u32>>().await?;
            match seen.first().copied() {
                Some(v) if v as usize == clen => clen_agreements += 1,
                other => {
                    return Err(format!(
                        "device fold length {other:?} disagrees with the host's \
                         committed length {clen} at seq_len {seq_len}"
                    ));
                }
            }
            if clen > 1 {
                clen_nontrivial += 1;
            }
        }
        let n_acc = clen.saturating_sub(1);
        accepted_lengths.push(n_acc);
        let commit_toks: Vec<u32> = commit.iter().take(clen).map(|&t| t as u32).collect();

        // Only NOW is the accepted length known, so only now can anything be
        // said about this window's tokens.
        if device {
            // The commit fire has already folded `clen` from the device-side
            // channel. Everything still buffered is the rejected tail, and
            // freeing it wholesale is also what restores the store's exact
            // occupancy after a device-resident fold.
            let remaining = rs.buffer_size();
            if remaining > 0 {
                rs.free_buffer(&(0..remaining).collect::<Vec<_>>())
                    .context("free_buffer")?;
            }
        } else {
            // The rejected tail never touched the recurrent state, so it is
            // enough to say it never happened. The accepted prefix stays
            // buffered -- unfolded -- and the NEXT window's fire folds it
            // while writing its own tokens over the slots just released.
            let rejected = kp1 - clen as u32;
            if rejected > 0 {
                rs.discard_buffered(rejected).context("discard_buffered")?;
            }
            pending_fold = clen as u32;
        }

        seq_len += clen as u32;
        committed.extend(&commit_toks);
        generated += clen.max(1) as u32;

        draft = drafts;
        seed = *committed.last().unwrap_or(&0) as i32;
    }
    // Every window's takes have drained: this close cancels nothing.
    pipeline.close();

    let total_acc: usize = accepted_lengths.iter().sum();
    let steps = accepted_lengths.len();
    let mean_acc = if steps > 0 {
        total_acc as f64 / steps as f64
    } else {
        0.0
    };
    // The point of the fusion, stated as a number: one fire per window on the
    // host path, two on the device path.
    let fires = if device { 2 * steps } else { steps };
    let result = format!(
        "mtp-native-verify: mode={} k={k} steps={steps} fires={fires} \
         accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} fold_len_checked={clen_agreements} \
         fold_len_nontrivial={clen_nontrivial} (PTIR-native verify+draft: verify vs embedded \
         drafts, next-drafts from mtp_logits argmax, [k+1] bonus tail, all traced)",
        if device { "device" } else { "host" },
        committed.len()
    );
    eprintln!("{result}");
    Ok(result)
}
