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

/// Decode a `[k]`/`[k+1]` i32 host vector.
async fn get_u32(t: inferlet::ptir::Taken) -> Result<Vec<u32>> {
    t.get::<u32>()
        .await
        .map_err(|e| format!("tensor take: {e}"))
}

async fn get_i32(t: inferlet::ptir::Taken) -> Result<Vec<i32>> {
    t.get::<i32>()
        .await
        .map_err(|e| format!("tensor take: {e}"))
}

/// Committed length of a sentinel `[k+1]` tail = the count before the first
/// `-1` (accepted prefix + the bonus at lane `n_acc`), always ≥ 1.
fn committed_len(tail: &[i32]) -> usize {
    tail.iter().take_while(|&&t| t >= 0).count()
}

fn bind_single_sequence(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    token_count: u32,
    pool_pages: u32,
    readout: &[u32],
) -> Result<()> {
    let embed_indptr = Channel::from(vec![0u32, token_count]).named("embed_indptr");
    let positions = Channel::from((0..token_count).collect::<Vec<_>>()).named("positions");
    let pages = Channel::from((0..pool_pages).collect::<Vec<_>>()).named("pages");
    let page_indptr = Channel::from(vec![0u32, token_count.div_ceil(PAGE_T)]).named("page_indptr");
    let w_slot =
        Channel::from((0..token_count).map(|p| p / PAGE_T).collect::<Vec<_>>()).named("w_slot");
    let w_off =
        Channel::from((0..token_count).map(|p| p % PAGE_T).collect::<Vec<_>>()).named("w_off");
    let readout = Channel::from(readout.to_vec()).named("readout");
    pass.embed(toks, &embed_indptr)?;
    pass.readout(&readout)?;
    pass.attention(
        ws,
        ..,
        ..,
        kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
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
fn bind_window(
    pass: &ForwardPass,
    ws: &WorkingSet,
    toks: &Channel,
    kv_len: &Channel,
    first_pos: u32,
    count: u32,
    pool_pages: u32,
    readout: &[u32],
) -> Result<()> {
    let embed_indptr = Channel::from(vec![0u32, count]).named("w_embed_indptr");
    let positions =
        Channel::from((first_pos..first_pos + count).collect::<Vec<_>>()).named("w_positions");
    let pages = Channel::from((0..pool_pages).collect::<Vec<_>>()).named("w_pages");
    let page_indptr =
        Channel::from(vec![0u32, (first_pos + count).div_ceil(PAGE_T).min(pool_pages)])
            .named("w_page_indptr");
    let w_slot = Channel::from(
        (first_pos..first_pos + count)
            .map(|p| p / PAGE_T)
            .collect::<Vec<_>>(),
    )
    .named("w_w_slot");
    let w_off = Channel::from(
        (first_pos..first_pos + count)
            .map(|p| p % PAGE_T)
            .collect::<Vec<_>>(),
    )
    .named("w_w_off");
    // A fold fire reads nothing, and `Channel::from(vec![])` cannot express
    // that (an empty shape has no vector length). Omitting `readout` entirely
    // is the honest encoding of "this fire samples no rows" — and the driver
    // requires it, since a fold returns before the output projection.
    if !readout.is_empty() {
        let readout = Channel::from(readout.to_vec()).named("w_readout");
        pass.readout(&readout)?;
    }
    pass.embed(toks, &embed_indptr)?;
    pass.attention(
        ws,
        ..,
        ..,
        kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
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
    let kv_len = Channel::from(vec![n]).named("b_kv_len");
    bind_single_sequence(&fwd, ws, &toks, &kv_len, n, max_pages, &readout)?;
    fwd.recurrent(std::slice::from_ref(rs))?;
    fwd.epilogue(move || {
        let picked = reduce_argmax(intrinsics::logits()); // [k] target argmax
        let seed = gather(&picked, Tensor::constant(vec![0u32])); // [1] row-0
        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let drafts = reduce_argmax(mtp); // [k] fresh drafts
        seed_out.put(&seed);
        drafts_out.put(&drafts);
    });

    fwd.submit(pipeline)
        .map_err(|e| format!("bootstrap submit: {e}"))?;
    let seed = get_i32(seed_out.take())
        .await?
        .first()
        .copied()
        .ok_or_else(|| "bootstrap: empty seed".to_string())?;
    let drafts = get_i32(drafts_out.take()).await?;
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
    let kv_len = Channel::from(vec![seq_len + kp1]).named("v_kv_len");
    let readout: Vec<u32> = (0..kp1).collect();
    bind_window(&fwd, ws, &toks, &kv_len, seq_len, kp1, max_pages, &readout)?;

    // BUFFER, do not fold. The verify window is `seed + k drafts`, and how
    // many of those k are real is not known until this fire's own logits come
    // back. Folding them would advance the recurrent state through a tail that
    // may be rejected, and a fold is irreversible — unlike KV, there are no
    // slots to discard. So the window's pre-recurrence activations are parked
    // in the buffer with the folded boundary held still, and the NEXT fire
    // moves it through exactly the accepted prefix.
    let fold_len = Channel::from(vec![0u32]).named("v_fold_len");
    fwd.recurrent_with(std::slice::from_ref(rs), &fold_len, ..)
    .map_err(|e| format!("verify recurrent binding: {e}"))?;
    fwd.epilogue(move || {
        // Device-alias read: peek the embedded window (NOT a resubmitted draft
        // channel) and gather rows 1..=k as the verify operand.
        let win = toks.read().tensor(); // [k+1] i32
        let draft_v = gather(&win, Tensor::constant((1..=k).collect::<Vec<u32>>())); // [k]
        let picked = reduce_argmax(intrinsics::logits()); // [k+1] target (k verify + bonus)
        let head = gather(&picked, iota(k)); // [k] picked[0..k]
        let hit = eq(&head, &draft_v); // [k] bool
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        let run = cumprod(select(&hit, &ones, &zeros)); // [k]
        let n_acc = cast(reduce_sum(run), DType::U32); // accepted-prefix length
        let keep = ge(broadcast(&n_acc, [kp1]), iota(kp1)); // [k+1] i <= n_acc
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1); // accepted prefix + bonus + -1s

        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let next_drafts = reduce_argmax(mtp); // [k] fresh drafts — NEXT window

        // The committed length, on device. `n_acc` above is already the whole
        // computation; the host's `committed_len` is just this number read out
        // of a sentinel tail. Publishing it directly is what lets the commit
        // fire be traced BEFORE anyone knows its value.
        if let Some(sink) = clen_sink {
            let clen = cast(add(&n_acc, 1u32), dtype::u32);
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
/// **It re-runs the FULL window, not the accepted prefix.** The prefix is not
/// host-known here, so its length cannot appear in the KV geometry. That costs
/// nothing: the verify fire already wrote KV for this exact span with these
/// exact tokens at these exact positions, so re-writing it is bit-identical,
/// and the rejected tail is overwritten by the next window regardless. The
/// recurrent side ignores these tokens completely and replays the buffered
/// activations, folding only as far as the device says.
///
/// **It is CONSTRUCTED before the verify fire is submitted.** A channel a
/// later pass consumes as a descriptor must be claimed by that pass first, or
/// the producer infers it as a terminal host-read output and the two
/// declarations conflict at bind. Construction order, not annotation.
fn build_commit(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    window: &[i32],
    seq_len: u32,
    max_pages: u32,
    fold_len: &Channel,
) -> Result<ForwardPass> {
    let count = window.len() as u32;
    let toks = Channel::from(window.to_vec()).named("c_toks");

    let fwd = ForwardPass::new();
    let kv_len = Channel::from(vec![seq_len + count]).named("c_kv_len");
    bind_window(&fwd, ws, &toks, &kv_len, seq_len, count, max_pages, &[])?;

    fwd.recurrent_with(std::slice::from_ref(rs), fold_len, ..)
        .map_err(|e| format!("commit recurrent binding: {e}"))?;
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
    if !wit_model::is_linear() {
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
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

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
    // target with the fold SUPPRESSED → fold exactly the accepted prefix and
    // abandon the rest → take the fresh native-MTP drafts as the NEXT
    // window's proposals → repeat.
    let window_slabs = (k + 1).div_ceil(rs_page);
    while generated < MAX_TOKENS {
        // Each window buffers k+1 tokens and ends with an empty buffer, so
        // the slabs are reserved per window rather than held across the loop.
        // They must be reserved fresh: appending onto a buffer that still held
        // the previous window's tail would ask the recurrence to read what is
        // already buffered, which it cannot do.
        rs.alloc_buffer(window_slabs)
            .map_err(|e| format!("rs.alloc_buffer: {e}"))?;

        let window: Vec<i32> = std::iter::once(seed).chain(draft.iter().copied()).collect();

        // The device path builds the COMMIT first -- it must claim `clen`
        // before the verify fire that fills it is submitted -- then enqueues
        // both fires back to back with nothing awaited in between. The host
        // path cannot: its commit is untraceable until `clen` is known.
        let clen_ch = device.then(|| Channel::new([1], dtype::u32).named("v_clen"));
        let (verify, commit_out, drafts_out, clen_echo) = build_verify(
            &ws, &rs, k, seed, &draft, seq_len, max_pages, clen_ch.as_ref(),
        )?;
        // The commit is CONSTRUCTED here -- after the verify pass is built but
        // before it is submitted. F8 only requires the consumer to claim the
        // channel before the producer is submitted; building it earlier would
        // also take its KV pages out of the working set ahead of the verify's,
        // which is a different trace.
        let device_commit = match &clen_ch {
            Some(ch) => Some(build_commit(
                &ws, &rs, &window, seq_len, max_pages, ch,
            )?),
            None => None,
        };

        verify
            .submit(&pipeline)
            .map_err(|e| format!("verify submit: {e}"))?;

        if let Some(c) = device_commit.as_ref() {
            c.submit(&pipeline)
                .map_err(|e| format!("device commit submit: {e}"))?;
        }

        let commit = get_i32(commit_out.take()).await?;
        let drafts = get_i32(drafts_out.take()).await?;
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
            let seen = get_u32(echo.take()).await?;
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

        // Only NOW is the accepted length known, so only now can the folded
        // boundary move. It advances through exactly `clen` of the k+1
        // buffered tokens; the rejected tail is abandoned by freeing its
        // slabs, having never touched the recurrent state.
        //
        // The commit re-runs the WINDOW's first `clen` tokens, which is the
        // span the verify fire already wrote KV for. `commit_toks` is the
        // window shifted by one (each entry is what the target predicted
        // AFTER the corresponding window token), so feeding it here would
        // corrupt exactly the prefix being committed.
        if !device {
            // The host path can only trace its commit NOW, with `clen` in
            // hand, which is exactly the round-trip the device path removes.
            let fold = Channel::from(vec![clen as u32]).named("c_fold_len");
            let pass = build_commit(&ws, &rs, &window[..clen], seq_len, max_pages, &fold)?;
            pass.submit(&pipeline)
                .map_err(|e| format!("commit submit: {e}"))?;
        }
        let remaining = rs.buffer_size();
        if remaining > 0 {
            rs.free_buffer(&(0..remaining).collect::<Vec<_>>())
                .map_err(|e| format!("free_buffer: {e}"))?;
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
    let result = format!(
        "mtp-native-verify: mode={} k={k} steps={steps} accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} fold_len_checked={clen_agreements} \
         fold_len_nontrivial={clen_nontrivial} (PTIR-native verify+draft: verify vs embedded \
         drafts, next-drafts from mtp_logits argmax, [k+1] bonus tail, all traced)",
        if device { "device" } else { "host" },
        committed.len()
    );
    eprintln!("{result}");
    Ok(result)
}
