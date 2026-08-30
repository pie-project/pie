//! Speculative decoding with a model's native multi-token-prediction head:
//! draft `k` tokens from the head, verify them against the target's own
//! logits in the same fire, and commit only the accepted prefix.
//!
//! # Why the loop turns on the HOST
//!
//! This pass was once a single loop-carried fire whose epilogue computed the
//! next round's `kv_len`, `positions`, `w_slot`, `w_off` and `page_indptr`
//! in-graph and put them back into the channels the next fire would read. That
//! is not servable and never was: the page CSR and the arena rectangles are
//! carved on the HOST before a launch, so the fire path answers `KvLen is not
//! host-derivable` and the program had never run end to end (its own README
//! said so). See `.wiki/alto/multimodal.md` §10.3.
//!
//! The shape that works is `cacheback-speculative-decoding`'s, which passes
//! the census: geometry is host-known, one fire per verify round, and the
//! round's accepted count — which this loop already read back every round to
//! decide what to emit — is what the next round's geometry is computed from.
//! What stays in the graph is the part worth having: the verify itself, so the
//! host learns a COUNT and never a comparison.
//!
//! # The rule
//!
//! ```text
//! window = [x, d_1 .. d_k]              pending correct token, then drafts
//! truth  = argmax(logits at each row)   row i is the truth after window[..=i]
//! m      = |{ i : d_{i+1} == truth_i, and every draft before it matched }|
//! commit = window[0 ..= m]              the correction plus the accepted run
//! next   = [truth_m, fresh drafts]
//! ```
//!
//! Only `m + 1` tokens advance the KV length. The `k - m` rejected cells sit
//! above it and the next round writes over them, which is why nothing has to
//! be retracted: shape decides slots, the length decides what is real.
//!
//! # What this is a gate for
//!
//! The greedy output must equal the greedy non-speculative output BYTE FOR
//! BYTE, and that is true for any draft head — verification keeps the target's
//! own argmax — which is what makes a synthetic head a fair test of the
//! MECHANISM (campaign M-4, and `tests/inferlets/test_eagle.py` asks it).
//! The acceptance counters say how much of the mechanism actually ran; a loop
//! that drafted nothing would satisfy the identity by doing no speculation.
//!
//! # And it is RED for a reason outside this program (`.wiki/alto/multimodal.md` §17)
//!
//! The default SKU is a HYBRID text — three layers in four are gated-delta
//! recurrences — and a recurrence is a FOLD, not an addressed cell. "Rejected
//! drafts sit above the advanced length and are overwritten by the next fire"
//! is true of the paged KV and has no meaning for the state: every row of a
//! verify window is folded into it and nothing retracts the rejected ones. On
//! a non-recurrent SKU (measured: gemma-4-E4B) the same fire shapes answer
//! BIT-IDENTICALLY at every `pad` width and every pad token, which is what
//! says the geometry, the pages, the mask and the KV rewrite are all right.
//!
//! `pad_token`, `no_drafts` and `peak_trace` are the instruments that said so
//! and are kept for the wave that fixes it: `no_drafts` runs the loop against
//! a SKU with no draft head at all, `pad_token` decides what a passenger row
//! carries, and `peak_trace` exports row 0's max logit as an f32 so a
//! perturbation is visible BEFORE it flips an argmax — round 0 identical and
//! round 1 not is what separates "the rows see each other" from "the last
//! round left something behind".

use inferlet::chat;
use inferlet::eta::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// **`k = 0` IS THE CONTROLLED A/B, AND IT IS NOT A DISABLED FEATURE.**
    /// The window is one row, nothing is drafted, nothing is verified, and the
    /// loop degenerates to sequential greedy decoding — through the SAME
    /// geometry, the same fire and the same commit arithmetic. So `k = 0`
    /// against `k = 4` separates "the loop's rows and pages are right" from
    /// "the speculation on top of them is right", which no comparison against
    /// a second program can do. `cacheback-speculative-decoding` states the
    /// same trick for the same reason under the name `draft_length`.
    #[serde(default = "default_k")]
    k: u32,
    /// **WHICH MASK THE VERIFY FIRE BINDS**, and it exists so that this
    /// program can falsify its own reading of the door rather than trust it.
    ///
    ///   `causal`  the per-row staircase: row `i` sees `j <= base + i`.
    ///             Semantically the derived bound, so a LIVE mask changes
    ///             nothing and an inert one changes nothing either — which is
    ///             why it cannot be the only mode.
    ///   `prefix`  row `i` sees `j <= base` and nothing of the window after
    ///             it. Semantically DIFFERENT from the derived bound at every
    ///             row but the first, so a live mask must move the answer and
    ///             an inert one cannot. This is the discriminator.
    ///   `none`    no mask bound at all — the control.
    #[serde(default = "default_mask_mode")]
    mask_mode: String,
    /// **PAD THE WINDOW WITH ROWS THE LOOP IGNORES**, so that a run with no
    /// drafts can be given the same fire SHAPE as a run with `k` of them.
    ///
    /// It exists to separate two claims the campaign's identity gate asks at
    /// once: that the speculative LOGIC is right, and that a `w`-row verify
    /// fire and a one-row decode fire answer the same argmax. The first is
    /// this program's; the second is the engine's, and `k = 0` against
    /// `k = 4` cannot tell them apart because it moves both at once —
    /// `qo_one` is a fact, so a one-row window takes the decode arm and a
    /// five-row one takes prefill.
    #[serde(default)]
    pad: u32,
    /// DIAGNOSTIC: the token the pad rows carry (default: the pending token).
    #[serde(default)]
    pad_token: Option<u32>,
    /// DIAGNOSTIC: run the fire with NO draft head at all (no mtp_logits), so
    /// the lane's `drafts` fact is false and the head's region is empty.
    #[serde(default)]
    no_drafts: bool,
}

fn default_mask_mode() -> String {
    "causal".into()
}

fn default_prompt() -> String {
    "The quick brown fox jumps over".into()
}

fn default_max_tokens() -> usize {
    64
}

fn default_k() -> u32 {
    4
}

/// **WHAT THE GATE READS** (campaign M-4).
///
/// The text is the claim. The counters beside it are the honest half: they say
/// how much of the mechanism ran, and neither number alone is the gate.
#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    /// Verify rounds that committed at least one token.
    rounds: usize,
    /// Draft tokens proposed, `k` per round.
    drafted: usize,
    /// Draft tokens the target's own argmax agreed with.
    accepted: usize,
    /// `accepted / drafted`, or zero when nothing was drafted.
    acceptance_rate: f64,
    /// The draft-window length this run used.
    k: u32,
    /// The FIRST round's proposals and the truths they were judged against —
    /// two short vectors, so a reader can tell "the head proposed nothing
    /// sensible" from "the verify compared the wrong rows".
    draft_sample: Vec<u32>,
    truth_sample: Vec<u32>,
    /// Row 0's argmax in every round — the token the loop actually commits.
    truth0_trace: Vec<u32>,
    peak_trace: Vec<f32>,
}

impl Output {
    /// What a zero-token ask answers: the same fields, every counter zero. A
    /// shape the gate can read without a branch is worth more than a shorter
    /// one it has to special-case.
    fn empty(k: u32) -> Output {
        Output {
            sampler: "mtp-speculative-decoding",
            text: String::new(),
            count: 0,
            rounds: 0,
            drafted: 0,
            accepted: 0,
            acceptance_rate: 0.0,
            k,
            draft_sample: Vec::new(),
            truth_sample: Vec::new(),
            truth0_trace: Vec::new(),
            peak_trace: Vec::new(),
        }
    }
}

/// One fire over `rows` token ids beginning at KV position `base`, answering
/// the target's argmax at every row and the head's `k` drafts.
///
/// **EVERY GEOMETRY CHANNEL HERE IS HOST-KNOWN**, which is the whole point of
/// the rewrite: `base` is an integer this side computed from the last round's
/// accepted count, and every vector below is arithmetic on it.
#[allow(clippy::type_complexity)]
async fn fire(
    ws: &WorkingSet,
    pipeline: &Pipeline,
    tokens: &[u32],
    base: u32,
    k: u32,
    page_size: u32,
    max_pages: u32,
    mask_mode: &str,
    drafts_wanted: bool,
) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    let rows = tokens.len() as u32;
    let total = base + rows;

    // **THE WINDOW'S ROWS ARE CAUSALLY ISOLATED BY A MASK, AND NOTHING ELSE
    // WOULD DO IT** (`.wiki/alto/multimodal.md` §13). A window of `w` rows in
    // one lane is a chunked prefill whose queries all see the lane's whole
    // `kv_len` — so without this, row 0's logits are computed with rows
    // 1..w's DRAFT keys in the context, and the "correction" the loop commits
    // is a token the target never proposed.
    //
    // MEASURED, and this is the experiment the mask has to answer: with no
    // mask, `k = 0` (a one-row window, nothing drafted) answered the greedy
    // non-speculative run BYTE FOR BYTE while `k = 1` and `k = 4` did not —
    // at ZERO accepted drafts in all three, so every round committed one
    // token and all three had to agree. The rows were seeing each other.
    //
    // Row `i` sits at position `base + i` and may see key `j` exactly when
    // `j <= base + i`: its own prefix, itself, and nothing drafted after it.
    // That is the staircase, and a dense `[rows, pool]` mask is how this
    // vocabulary says it (`naive-masked`'s `dense-prefill` builds the same
    // shape for the same reason).
    //
    // **A BOUND MASK REPLACES THE DERIVED CAUSAL BOUND** rather than ANDing
    // with it (`Port::AttnMask`'s own doc, and `trackb-snapkv`'s note on the
    // same door), which is why every row states its whole prefix here and not
    // just the part that is about drafts.
    let pages = total.div_ceil(page_size);
    let pool = max_pages * page_size;

    let ids = Channel::from_iter(tokens.iter().map(|&t| t as i32));
    let embed_indptr = Channel::from([0u32, rows]).named("embed_indptr");
    let positions = Channel::from_iter(base..total).named("positions");
    let page_list = Channel::from_iter(0..pages).named("pages");
    let page_indptr = Channel::from([0u32, pages]).named("page_indptr");
    let w_slot = Channel::from_iter((base..total).map(|p| p / page_size)).named("w_slot");
    let w_off = Channel::from_iter((base..total).map(|p| p % page_size)).named("w_off");
    let kv_len = Channel::from([total]).named("kv_len");
    let bound: Vec<bool> = match mask_mode {
        "prefix" => (0..rows)
            .flat_map(|_| (0..pool).map(move |j| j <= base))
            .collect(),
        // DIAGNOSTIC: row `i` sees `j < base + i` — its own cell EXCLUDED.
        "noself" => (base..total)
            .flat_map(|p| (0..pool).map(move |j| j + 1 <= p))
            .collect(),
        // DIAGNOSTIC: row `i` sees `j <= base + i + 1` — one cell of the
        // future INCLUDED.
        "wide" => (base..total)
            .flat_map(|p| (0..pool).map(move |j| j <= p + 1))
            .collect(),
        // DIAGNOSTIC: causal, MINUS the cell just below the window's base.
        "nolast" => (base..total)
            .flat_map(|p| (0..pool).map(move |j| j <= p && j + 1 != base))
            .collect(),
        // DIAGNOSTIC: causal, MINUS the two cells below the window's base.
        "nolast2" => (base..total)
            .flat_map(|p| (0..pool).map(move |j| j <= p && j + 1 != base && j + 2 != base))
            .collect(),
        // DIAGNOSTIC: causal, minus everything from base-1 up (row 0 sees
        // only cells strictly below base-1).
        "back2" => (base..total)
            .flat_map(|p| (0..pool).map(move |j| j + 2 <= p))
            .collect(),
        _ => (base..total)
            .flat_map(|p| (0..pool).map(move |j| j <= p))
            .collect(),
    };
    let mask = Channel::from_shaped([rows, pool], bound).named("verify_mask");
    let readout = Channel::from_iter(0..rows).named("readout");
    let truth_out = Channel::new([rows], dtype::i32).named("truth");
    let peak_out = Channel::new([rows], dtype::f32).named("peak");
    let drafts_out = Channel::new([k.max(1)], dtype::i32).named("drafts");

    let fwd = ForwardPass::new();
    fwd.embed(&ids, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &page_list,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: (mask_mode != "none").then_some(&mask),
        },
    )?;
    // **THE VERIFY'S HALF THAT STAYS IN THE GRAPH.** The argmax over the whole
    // readout is one reduction on the device; comparing it against the drafts
    // is arithmetic on `k + 1` integers and belongs where the loop is.
    fwd.epilogue(move || {
        truth_out.put(reduce_argmax(intrinsics::logits()));
        peak_out.put(reduce_max(intrinsics::logits()));
        if drafts_wanted {
            drafts_out.put(reduce_argmax(intrinsics::mtp_logits(k.max(1))));
        }
    });
    fwd.submit(pipeline).context("verify-and-extend")?;

    let truth = truth_out
        .take_host::<Vec<i32>>()
        .await?
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let peak = peak_out.take_host::<Vec<f32>>().await?;
    let drafts = if drafts_wanted {
        drafts_out
            .take_host::<Vec<i32>>()
            .await?
            .into_iter()
            .map(|t| t as u32)
            .collect()
    } else {
        Vec::new()
    };
    Ok((truth, drafts, peak))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.max_tokens == 0 {
        return Ok(Output::empty(input.k));
    }
    if input.k > 32 {
        return Err("k must be at most 32".into());
    }

    let k = input.k;
    let mode = input.mask_mode.clone();
    if !matches!(
        mode.as_str(),
        "causal" | "prefix" | "none" | "noself" | "wide" | "nolast" | "nolast2" | "back2"
    ) {
        return Err(format!("unknown mask_mode: {mode}").into());
    }
    let w = (k + 1) as usize;
    let page_size = kv_page_size();

    // **THE RAW ENCODING, AND THAT IS WHAT MAKES THE GATE A GATE.** The claim
    // this program exists to be measured against is that speculation changes
    // how many fires run and nothing about the tokens — so it has to decode
    // the SAME context `naive-baseline` decodes, and that one encodes the
    // prompt and nothing else. A chat wrap here would have made the identity a
    // comparison of two different conversations.
    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let stop_tokens = chat::stop_tokens();

    // One working set for the whole generation: the prompt's KV is written
    // once and every later round appends to it. The lease covers the prompt,
    // every token the host may keep, and the transient overshoot of a window
    // whose drafts are rejected (up to `w` cells above the committed length).
    let ws = WorkingSet::new();
    let max_extent = n + input.max_tokens as u32 + w as u32;
    let max_pages = max_extent.div_ceil(page_size);
    ws.reserve(max_pages).context("reserve KV")?;
    let pipeline = Pipeline::new();

    // ── Prefill: the prompt, one fire, and the first window it seeds ────
    let want_drafts = !input.no_drafts;
    let (truth, drafts, _) = fire(
        &ws, &pipeline, &prompt, 0, k, page_size, max_pages, &mode, want_drafts,
    )
    .await?;
    let mut peak_trace: Vec<f32> = Vec::new();
    let mut x = *truth.last().expect("a prefill answers one row per token");
    let mut pending = drafts;
    let mut base = n;

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let (mut rounds, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    let (mut draft_sample, mut truth_sample) = (Vec::new(), Vec::new());
    let mut truth0_trace: Vec<u32> = Vec::new();

    // The seed token is the first thing the model said and is committed
    // unconditionally: nothing drafted it, so nothing can reject it.
    generated.push(x);
    let mut stopped = stop_tokens.contains(&x) || generated.len() == input.max_tokens;

    while !stopped {
        // ── The window: the pending correct token, then the drafts ──────
        let mut window = Vec::with_capacity(w);
        window.push(x);
        window.extend(pending.iter().copied().take(k as usize));
        // Padding rows carry the pending token again. They are verified like
        // any other row and rejected like any other wrong one — nothing here
        // treats them specially, which is what makes the comparison fair.
        for _ in 0..input.pad {
            window.push(input.pad_token.unwrap_or(x));
        }
        while window.len() < w {
            // A head that answered fewer drafts than asked is a head with a
            // shorter horizon, not an error; the window shrinks and the round
            // still verifies what it has.
            window.pop();
            break;
        }

        let (truth, drafts, peak) = fire(
            &ws, &pipeline, &window, base, k, page_size, max_pages, &mode, want_drafts,
        )
        .await?;
        peak_trace.push(peak[0]);
        rounds += 1;
        if rounds == 1 {
            draft_sample = window[1..].to_vec();
            truth_sample = truth.clone();
        }
        truth0_trace.push(truth[0]);
        let proposed = (window.len() - 1).saturating_sub(input.pad as usize);
        drafted += proposed;

        // ── Verify: the longest matching prefix, and nothing after it ────
        let mut m = 0usize;
        while m < proposed && window[m + 1] == truth[m] {
            m += 1;
        }
        accepted += m;

        // ── Commit `window[1 ..= m]` — `window[0]` was committed last round
        //    as the correction that produced it — then the new correction.
        for &token in &window[1..=m] {
            generated.push(token);
            if stop_tokens.contains(&token) || generated.len() == input.max_tokens {
                stopped = true;
                break;
            }
        }
        if stopped {
            break;
        }
        let correction = truth[m];
        generated.push(correction);
        if stop_tokens.contains(&correction) || generated.len() == input.max_tokens {
            break;
        }

        // Only the accepted run advances the length; the rejected cells stay
        // above it and the next fire writes over them.
        base += (m + 1) as u32;
        x = correction;
        pending = drafts;
    }
    pipeline.close();

    let acceptance_rate = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    // The seed is not a generated-by-speculation token, but it IS text; the
    // count is what the caller asked for and the text is what it reads.
    Ok(Output {
        sampler: "mtp-speculative-decoding",
        text: model::decode(&generated)?,
        count: generated.len(),
        rounds,
        drafted,
        accepted,
        acceptance_rate,
        k,
        draft_sample,
        truth_sample,
        truth0_trace,
        peak_trace,
    })
}
