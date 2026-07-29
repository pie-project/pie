//! **FOLD-COMMIT on a linear model** — the speculative-commit shape
//! `model.wit`'s `is-linear()` exists to select.
//!
//! A linear/SSM model folds tokens into its recurrent state IRREVERSIBLY, so
//! the KV trick of discarding rejected slots does not exist for it. The
//! answer the runtime declares is fold-commit: run the uncertain tokens with
//! the fold SUPPRESSED, holding their pre-recurrence activations in the RS
//! working set's buffered slots, and afterwards fold only the accepted
//! prefix. A rejected tail is simply never folded — abandoning it costs one
//! `free-buffer`.
//!
//! This inferlet drives all three modes end to end:
//!
//!  1. **prefill — `fold`** (the default): materializes the folded state.
//!     Buffering and folding both read it, so this must come first.
//!  2. **speculate — `fold_len = 0`**: one chunk of `SPEC_TOKENS` tokens
//!     appended to the buffer; the folded boundary does not move, so the
//!     chunk stays abandonable.
//!  3. **commit — `fold_len = accepted`**: advances the boundary through the
//!     accepted prefix and drops the fully covered head slots.
//!
//! Input: the number of speculative tokens to accept (default all of them),
//! e.g. `"2"`. Output names the three phases so a harness can assert them.
//!
//! The special input `"chain"` accepts 2 and then buffers a SECOND chunk onto
//! the surviving unfolded tail, without emptying the buffer in between. That
//! second append is the buffer READ PATH: its tokens must recur from
//! `folded ⊕ replay(buffer)`, not from the folded state alone. The runtime
//! refuses it today, and `cuda_gdn_foldcommit.rs` pins that refusal — the mode
//! exists so the read path has an executable acceptance test rather than a
//! prose description.

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};

/// Tokens written in the buffered (speculative) chunk.
const SPEC_TOKENS: u32 = 4;

/// One arm of the `inside` comparison: a fresh context, prefilled with the
/// prompt, then some number of fires over the SAME four continuation tokens.
/// Everything is parameterised so the two arms differ only in how the fires
/// are cut, never in what they compute.
struct Arm {
    ws: WorkingSet,
    rs: RsWorkingSet,
    page_size: u32,
    rs_page: u32,
    max_pages: u32,
    pos: u32,
}

impl Arm {
    /// Prefill `prompt` with the default fold, returning the greedy token.
    async fn open(prompt: &[u32], max_pages: u32, pipe: &Pipeline) -> Result<(Self, i32)> {
        let ws = WorkingSet::new();
        let rs = RsWorkingSet::new();
        let page_size = ws.page_size();
        ws.reserve(max_pages)
            .map_err(|e| format!("ws.reserve: {e}"))?;
        let mut arm = Arm {
            ws,
            rs,
            page_size,
            rs_page: inferlet::model::rs_buffer_page_size().max(1),
            max_pages,
            pos: 0,
        };
        let g0 = arm.fire(prompt, None, pipe, "prefill").await?;
        arm.pos = prompt.len() as u32;
        Ok((arm, g0.unwrap_or(0)))
    }

    /// One fire over `toks`.
    ///
    /// `buffered` is `Some((buffer_start, fold_len))`: the fire scatters its
    /// tokens into the buffer starting at logical token `buffer_start` and
    /// folds `fold_len` of the resulting buffer. `fold_len == 0` is a pure
    /// append; `0 < fold_len < buffer_start + toks.len()` is the boundary
    /// landing strictly INSIDE this fire's own tokens. `None` is the plain
    /// in-forward fold that touches no buffer at all.
    async fn fire(
        &self,
        toks: &[u32],
        buffered: Option<(u32, u32)>,
        pipe: &Pipeline,
        tag: &str,
    ) -> Result<Option<i32>> {
        let t = toks.len() as u32;
        let base = self.pos;
        let end = base + t;
        let ps = self.page_size;
        let ch = |v: Vec<u32>| Channel::from(v);

        let fwd = ForwardPass::new();
        fwd.embed(
            &Channel::from(toks.iter().map(|&x| x as i32).collect::<Vec<_>>()),
            &ch(vec![0, t]),
        )?;
        fwd.attention(
            &self.ws,
            ..,
            ..,
            &ch(vec![end]),
            &ch((0..self.max_pages).collect()),
            &ch(vec![0, end.div_ceil(ps)]),
            &ch((base..end).map(|p| p / ps).collect()),
            &ch((base..end).map(|p| p % ps).collect()),
            &ch((base..end).collect()),
            None,
        )?;
        match buffered {
            None => fwd.recurrent(std::slice::from_ref(&self.rs))?,
            Some((start, fold_len)) => {
                let rp = self.rs_page;
                // The write span is [start, start + t) in LOGICAL buffer
                // tokens; the runtime resolves it against the physical head.
                let first = start / rp;
                let last = (start + t - 1) / rp;
                let pages: Vec<u32> = (first..=last).collect();
                let count = pages.len() as u32;
                fwd.recurrent_with(
                    std::slice::from_ref(&self.rs),
                    &ch(vec![fold_len]),
                    ..,
                    ..,
                    &ch(vec![t]),
                    &ch(pages),
                    &ch(vec![0, count]),
                    &ch((start..start + t).map(|x| x / rp - first).collect()),
                    &ch((start..start + t).map(|x| x % rp).collect()),
                )
                .map_err(|e| format!("{tag} recurrent binding: {e}"))?;
            }
        }
        let out = Channel::new([1], dtype::i32).named("arm_out");
        let sink = out.clone();
        fwd.epilogue(move || {
            let t = reduce_argmax(intrinsics::logits());
            sink.put(&t);
        });
        fwd.submit(pipe)
            .map_err(|e| format!("{tag} submit: {e}"))?;
        Ok(Some(
            out.take()
                .get::<i32>()
                .await
                .map_err(|e| format!("{tag} take: {e}"))?[0],
        ))
    }
}

/// A fold running THROUGH a non-empty buffer, in the same fire that fills it.
///
/// Arm A appends two tokens onto a buffer that already holds two, and folds
/// all four in that one fire. The buffered pair has to be replayed ahead of
/// the new pair (the read path), and the folded boundary has to land past
/// both -- so this is a write and a fold at once, over the extended layout.
///
/// Arm B computes the same four tokens the long way: fold the first two
/// outright, then fold the last two outright. Both arms must agree, and
/// because the comparison is drawn AFTER the fold -- each arm continues from
/// its own resulting state -- agreement pins the folded state itself, not
/// merely the logits along the way.
async fn fold_inside_new_tokens(prompt: &[u32]) -> Result<String> {
    const T: usize = 4;
    const HALF: u32 = 2;

    let n = prompt.len() as u32;
    let pipe = Pipeline::new();
    let probe = WorkingSet::new();
    let max_pages = (n + 2 * T as u32 + 1).div_ceil(probe.page_size());
    drop(probe);

    let (mut a, g0) = Arm::open(prompt, max_pages, &pipe).await?;
    let (mut b, g0b) = Arm::open(prompt, max_pages, &pipe).await?;
    if g0 != g0b {
        return Err(format!("arms disagree on the prefill token: {g0} vs {g0b}"));
    }
    let cont: Vec<u32> = vec![g0 as u32; T];

    // Arm A: buffer the first pair, then append the second pair AND fold all
    // four -- the write-and-fold through a non-empty buffer.
    a.rs.alloc_buffer((T as u32).div_ceil(a.rs_page))
        .map_err(|e| format!("A alloc_buffer: {e}"))?;
    a.fire(&cont[..HALF as usize], Some((0, 0)), &pipe, "A-buffer")
        .await?;
    a.pos += HALF;
    let a_last = a
        .fire(
            &cont[HALF as usize..],
            Some((HALF, T as u32)),
            &pipe,
            "A-append-and-fold",
        )
        .await?
        .unwrap_or(0);
    a.pos += HALF;

    // Arm B: the same four tokens, folded two fires at a time.
    b.fire(&cont[..HALF as usize], None, &pipe, "B-fold-1").await?;
    b.pos += HALF;
    let b_last = b
        .fire(&cont[HALF as usize..], None, &pipe, "B-fold-2")
        .await?
        .unwrap_or(0);
    b.pos += HALF;

    // Now compare the STATES the two arms arrived at, by continuing each one
    // token further. If arm A's fold had missed the buffered pair (or replayed
    // it twice) this is where it shows.
    let next = vec![b_last as u32];
    let a_next = a.fire(&next, None, &pipe, "A-next").await?.unwrap_or(0);
    let b_next = b.fire(&next, None, &pipe, "B-next").await?.unwrap_or(0);

    pipe.close();

    let agree = a_last == b_last && a_next == b_next;
    let result = format!(
        "foldthrough tokens={T} buffered={HALF} a_last={a_last} b_last={b_last} \
         a_next={a_next} b_next={b_next} agree={}",
        if agree { "yes" } else { "no" }
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    if !agree {
        return Err(result);
    }
    Ok(result)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // `chain` folds a strict prefix so the buffer keeps an unfolded tail, then
    // appends a second chunk onto it.
    let chain = input.trim() == "chain";
    let accepted: u32 = if chain {
        SPEC_TOKENS / 2
    } else {
        input.trim().parse().unwrap_or(SPEC_TOKENS).min(SPEC_TOKENS)
    };
    let chunks: u32 = if chain { 2 } else { 1 };

    if !wit_model::is_linear() {
        return Ok("skipped: fold-commit needs a linear model".to_string());
    }

    if input.trim() == "inside" {
        let prompt = wit_model::encode("hello world");
        let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
        return fold_inside_new_tokens(&prompt).await;
    }

    let ws = WorkingSet::new();
    let page_size = ws.page_size();
    let rs = RsWorkingSet::new();
    let buffer_page = rs.buffer_page_size();
    if buffer_page == 0 {
        return Err("linear model reports no RS buffer page size".to_string());
    }

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + chunks * SPEC_TOKENS + 1).div_ceil(page_size);
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // Buffered slots for the speculative chunk. Reserved logically here;
    // the buffering fire materializes them on first write.
    let slabs = (chunks * SPEC_TOKENS).div_ceil(buffer_page);
    rs.alloc_buffer(slabs)
        .map_err(|e| format!("rs.alloc_buffer: {e}"))?;

    // ───────────────── 1. PREFILL — mode `fold` (default) ─────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from((0..n).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
    let w_off_p = Channel::from((0..n).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.attention(
        &ws,
        ..,
        ..,
        &kv_len_p,
        &pages_p,
        &page_indptr_p,
        &w_slot_p,
        &w_off_p,
        &positions_p,
        None,
    )?;
    fwd_p.recurrent(std::slice::from_ref(&rs))?;
    fwd_p.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        g0_ch.put(&t);
    });

    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];

    // ────────────── 2. SPECULATE — `fold_len = 0`, nothing folds ──────────
    // One SPEC_TOKENS-wide fire. Its activations land in the buffered slots;
    // the folded state does not move, so nothing here is committed yet.
    let spec_toks = Channel::from(vec![g0; SPEC_TOKENS as usize]).named("spec_toks");
    let spec_indptr = Channel::from(vec![0u32, SPEC_TOKENS]).named("spec_indptr");
    let spec_positions =
        Channel::from((n..n + SPEC_TOKENS).collect::<Vec<_>>()).named("spec_positions");
    let spec_pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("spec_pages");
    let spec_page_indptr =
        Channel::from(vec![0u32, (n + SPEC_TOKENS).div_ceil(page_size)]).named("spec_page_indptr");
    let spec_w_slot = Channel::from(
        (n..n + SPEC_TOKENS)
            .map(|p| p / page_size)
            .collect::<Vec<_>>(),
    )
    .named("spec_w_slot");
    let spec_w_off = Channel::from(
        (n..n + SPEC_TOKENS)
            .map(|p| p % page_size)
            .collect::<Vec<_>>(),
    )
    .named("spec_w_off");
    let spec_out = Channel::new([1], dtype::i32).named("spec_out");

    let fwd_s = ForwardPass::new();
    fwd_s.embed(&spec_toks, &spec_indptr)?;
    let spec_kv_len = Channel::from(vec![n + SPEC_TOKENS]).named("spec_kv_len");
    fwd_s.attention(
        &ws,
        ..,
        ..,
        &spec_kv_len,
        &spec_pages,
        &spec_page_indptr,
        &spec_w_slot,
        &spec_w_off,
        &spec_positions,
        None,
    )?;
    // The buffered geometry: `SPEC_TOKENS` tokens appended at the buffer tail,
    // page-major from slab zero. `fold_len = 0` holds the folded boundary
    // still, so nothing here is committed -- that is what makes this fire
    // abandonable.
    let rs_page = inferlet::model::rs_buffer_page_size().max(1);
    let rs_pages = SPEC_TOKENS.div_ceil(rs_page);
    let spec_rs_len = Channel::from(vec![SPEC_TOKENS]).named("spec_rs_len");
    let spec_rs_pages = Channel::from((0..rs_pages).collect::<Vec<_>>()).named("spec_rs_pages");
    let spec_rs_indptr = Channel::from(vec![0u32, rs_pages]).named("spec_rs_indptr");
    let spec_rs_w_slot =
        Channel::from((0..SPEC_TOKENS).map(|t| t / rs_page).collect::<Vec<_>>())
            .named("spec_rs_w_slot");
    let spec_rs_w_off = Channel::from((0..SPEC_TOKENS).map(|t| t % rs_page).collect::<Vec<_>>())
        .named("spec_rs_w_off");
    let spec_fold_len = Channel::from(vec![0u32]).named("spec_fold_len");
    fwd_s
        .recurrent_with(
            std::slice::from_ref(&rs),
            &spec_fold_len,
            ..,
            ..,
            &spec_rs_len,
            &spec_rs_pages,
            &spec_rs_indptr,
            &spec_rs_w_slot,
            &spec_rs_w_off,
        )
        .map_err(|e| format!("speculative recurrent binding: {e}"))?;
    fwd_s.epilogue(move || {
        let t = reduce_argmax(intrinsics::logits());
        spec_out.put(&t);
    });
    fwd_s
        .submit(&pipe)
        .map_err(|e| format!("speculative submit: {e}"))?;
    let drafted = spec_out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("spec_out take: {e}"))?[0];

    // ─────────────── 3. COMMIT — `fold_len = accepted` ────────────────────
    // Replays only the accepted prefix into the folded state. No logits, and
    // none are possible: a fold fire returns from the linear layers before the
    // output projection, so the driver refuses one that declares sample rows.
    // Hence the empty readout and the absent epilogue — and nothing to await,
    // since pipeline order already puts the fold ahead of whatever reads next.
    let mut committed = 0u32;
    if accepted > 0 {
        let commit_toks = Channel::from(vec![g0; accepted as usize]).named("commit_toks");
        let commit_indptr = Channel::from(vec![0u32, accepted]).named("commit_indptr");
        let commit_positions =
            Channel::from((n..n + accepted).collect::<Vec<_>>()).named("commit_positions");
        let commit_pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("commit_pages");
        let commit_page_indptr = Channel::from(vec![0u32, (n + accepted).div_ceil(page_size)])
            .named("commit_page_indptr");
        let commit_w_slot =
            Channel::from((n..n + accepted).map(|p| p / page_size).collect::<Vec<_>>())
                .named("commit_w_slot");
        let commit_w_off =
            Channel::from((n..n + accepted).map(|p| p % page_size).collect::<Vec<_>>())
                .named("commit_w_off");

        let fwd_c = ForwardPass::new();
        fwd_c.embed(&commit_toks, &commit_indptr)?;
        let commit_kv_len = Channel::from(vec![n + accepted]).named("commit_kv_len");
        fwd_c.attention(
            &ws,
            ..,
            ..,
            &commit_kv_len,
            &commit_pages,
            &commit_page_indptr,
            &commit_w_slot,
            &commit_w_off,
            &commit_positions,
            None,
        )?;
        // The commit replays `accepted` buffered tokens, so it reads exactly
        // the span the speculate fire wrote.
        let commit_rs_pages_n = accepted.div_ceil(rs_page);
        let commit_rs_len = Channel::from(vec![accepted]).named("commit_rs_len");
        let commit_rs_pages =
            Channel::from((0..commit_rs_pages_n).collect::<Vec<_>>()).named("commit_rs_pages");
        let commit_rs_indptr =
            Channel::from(vec![0u32, commit_rs_pages_n]).named("commit_rs_indptr");
        let commit_rs_w_slot =
            Channel::from((0..accepted).map(|t| t / rs_page).collect::<Vec<_>>())
                .named("commit_rs_w_slot");
        let commit_rs_w_off =
            Channel::from((0..accepted).map(|t| t % rs_page).collect::<Vec<_>>())
                .named("commit_rs_w_off");
        let commit_fold_len = Channel::from(vec![accepted]).named("commit_fold_len");
        fwd_c
            .recurrent_with(
                std::slice::from_ref(&rs),
                &commit_fold_len,
                ..,
                ..,
                &commit_rs_len,
                &commit_rs_pages,
                &commit_rs_indptr,
                &commit_rs_w_slot,
                &commit_rs_w_off,
            )
            .map_err(|e| format!("commit recurrent binding: {e}"))?;
        // An EMPTY epilogue, not an absent one. The commit samples nothing —
        // it replays buffered activations and stops at the recurrence — but a
        // pass with no stages has no PTIR program at all, and registration
        // fails with PIE_STATUS_INVALID_ARGUMENT. An empty epilogue is the
        // minimal well-formed program that produces no logits.
        fwd_c.epilogue(|| {});
        fwd_c
            .submit(&pipe)
            .map_err(|e| format!("commit submit: {e}"))?;
        committed = accepted;
    }

    // ────────── 4. CHAIN — a SECOND chunk onto the unfolded tail ──────────
    // The commit folded `accepted` of the `SPEC_TOKENS` buffered tokens, so the
    // buffer still holds `SPEC_TOKENS - accepted` of them. Appending here means
    // the new tokens sit at [F + tail, ...), and their recurrence has to start
    // from `folded ⊕ replay(buffer)` — the buffer READ PATH. Every recurrence
    // today initializes from `recurrent_state[slot]`, the state at F, so the
    // runtime refuses this fire rather than running it and silently pretending
    // the tail is not there.
    //
    // When the read path lands this becomes a real two-chunk speculation and
    // the harness flips from asserting the refusal to asserting the value.
    let mut chained = "n/a";
    if chain {
        let tail = SPEC_TOKENS - committed;
        let base = n + SPEC_TOKENS;
        let c2_toks = Channel::from(vec![drafted; SPEC_TOKENS as usize]).named("c2_toks");
        let c2_indptr = Channel::from(vec![0u32, SPEC_TOKENS]).named("c2_indptr");
        let c2_positions =
            Channel::from((base..base + SPEC_TOKENS).collect::<Vec<_>>()).named("c2_positions");
        let c2_pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("c2_pages");
        let c2_page_indptr = Channel::from(vec![0u32, (base + SPEC_TOKENS).div_ceil(page_size)])
            .named("c2_page_indptr");
        let c2_w_slot = Channel::from(
            (base..base + SPEC_TOKENS)
                .map(|p| p / page_size)
                .collect::<Vec<_>>(),
        )
        .named("c2_w_slot");
        let c2_w_off = Channel::from(
            (base..base + SPEC_TOKENS)
                .map(|p| p % page_size)
                .collect::<Vec<_>>(),
        )
        .named("c2_w_off");
        let c2_out = Channel::new([1], dtype::i32).named("c2_out");

        let fwd2 = ForwardPass::new();
        fwd2.embed(&c2_toks, &c2_indptr)?;
        let c2_kv_len = Channel::from(vec![base + SPEC_TOKENS]).named("c2_kv_len");
        fwd2.attention(
            &ws,
            ..,
            ..,
            &c2_kv_len,
            &c2_pages,
            &c2_page_indptr,
            &c2_w_slot,
            &c2_w_off,
            &c2_positions,
            None,
        )?;
        // The new chunk starts at buffer token `tail`, immediately after what
        // the commit left unfolded — that offset is exactly what makes this the
        // read path rather than a fresh chunk.
        let c2_rs_pages = (tail + SPEC_TOKENS).div_ceil(rs_page);
        let c2_rs_len = Channel::from(vec![SPEC_TOKENS]).named("c2_rs_len");
        let c2_rs_pages_ch =
            Channel::from((0..c2_rs_pages).collect::<Vec<_>>()).named("c2_rs_pages");
        let c2_rs_indptr = Channel::from(vec![0u32, c2_rs_pages]).named("c2_rs_indptr");
        let c2_rs_w_slot = Channel::from(
            (tail..tail + SPEC_TOKENS)
                .map(|t| t / rs_page)
                .collect::<Vec<_>>(),
        )
        .named("c2_rs_w_slot");
        let c2_rs_w_off = Channel::from(
            (tail..tail + SPEC_TOKENS)
                .map(|t| t % rs_page)
                .collect::<Vec<_>>(),
        )
        .named("c2_rs_w_off");
        let c2_fold_len = Channel::from(vec![0u32]).named("c2_fold_len");
        fwd2.recurrent_with(
            std::slice::from_ref(&rs),
            &c2_fold_len,
            ..,
            ..,
            &c2_rs_len,
            &c2_rs_pages_ch,
            &c2_rs_indptr,
            &c2_rs_w_slot,
            &c2_rs_w_off,
        )
        .map_err(|e| format!("chain recurrent binding: {e}"))?;
        fwd2.epilogue(move || {
            let t = reduce_argmax(intrinsics::logits());
            c2_out.put(&t);
        });
        fwd2.submit(&pipe)
            .map_err(|e| format!("chain submit: {e}"))?;
        c2_out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("c2_out take: {e}"))?;
        chained = "ok";
    }

    // Whatever was buffered but not folded is abandoned by dropping its
    // slots — the reject half of fold-commit, and the whole point of the
    // buffer: no folded state was ever perturbed by the rejected tail.
    let remaining = rs.buffer_size();
    if remaining > 0 {
        rs.free_buffer(&(0..remaining).collect::<Vec<_>>())
            .map_err(|e| format!("free_buffer: {e}"))?;
    }

    pipe.close();

    let result = format!(
        "foldcommit prefill=1 buffered={SPEC_TOKENS} committed={committed} \
         abandoned={} g0={g0} drafted={drafted} chained={chained}",
        SPEC_TOKENS - committed
    );
    eprintln!("[GDN_FOLDCOMMIT] {result}");
    Ok(result)
}
