//! **SnapKV, EXECUTED** — the eviction half, not the observability half.
//!
//! `trackb-snapkv` computes SnapKV's decision and reports it. This program
//! *acts* on it: the prefill epilogue reads the attention-score rectangle,
//! folds it into a per-page mass on the device, and the decode that follows
//! is served a STRICTLY SMALLER KV cache chosen by those numbers — and it
//! still answers a needle question whose answer is buried in the middle of
//! the haystack.
//!
//! It is the gate `.wiki/alto/campaign.md` §2 calls S-4: "one eviction
//! emulation (SnapKV path) demonstrably shrinks the served mask AND still
//! answers a needle prompt correctly".
//!
//! ## What "evict" means here, and the door it goes through
//!
//! `.wiki/alto/attn-score.md` §4 already answered this, and the answer is the
//! first of the two honest deltas it records against the papers: **"evict"
//! executes as CUSTOM MASK updates (the masked arm), with a page drop only
//! when a page's tokens are all dead.** So the door is the mask — a
//! descriptor port the guest writes every fire, served end to end by the
//! `attention.masked` arm the model text has declared since gemma — and the
//! keep-set is a row of bits ANDed into the causal row.
//!
//! **AND THE MASK IS THE DOOR FOR A REASON, NOT FOR CONVENIENCE.** The other
//! route — naming fewer pages in the page list — makes the KV cache shorter
//! without making the SEQUENCE shorter, and from that instant the cache index
//! and the true position are two different numbers. A kept key carries the
//! RoPE it was written with, at its true position, so the query must carry
//! its true position too or every relative distance in the attention is wrong
//! by however much was evicted. This shell derives a lane's positions as
//! `held .. held + rows` and refuses an explicit list by name ("the cuda
//! engine does not serve `explicit lane positions`"), so a page-list cut
//! would have to re-index the query — which is StreamingLLM's technique and
//! NOT SnapKV's, and would be a different paper silently substituted for the
//! one this gate is named after.
//!
//! Under the mask the geometry does not move at all: the same page list, the
//! same `kv_len`, the same positions, and the only thing that changes is
//! which keys the softmax is allowed to see. Quality semantics exact; memory
//! savings quantized to page granularity and taken later, which is precisely
//! the delta §4 wrote down.
//!
//! ## Which pages are kept, and why two of them are not a choice
//!
//! * **Page 0 is always kept.** It holds the attention sink, and a
//!   transformer that loses its sink does not degrade gracefully — it
//!   degenerates. Every eviction paper in this family keeps it, and the score
//!   row says so too: position 0 outweighs the uniform share several times
//!   over on every prompt.
//! * **The last prompt page is always kept**, because it is the page the
//!   decode writes into and because a recency window is in every one of these
//!   policies by construction.
//! * The rest of the budget goes to the highest-mass pages, ranked by the
//!   observation window's own attention.
//!
//! Ranking on the host and not on the device is a departure from §4's "only
//! decisions cross to the host", and it is a deliberate one for a GATE: the
//! kept set is reported so the harness can check it against the needle's
//! actual page. The device-side top-k that a serving program would use is one
//! `rank_le` away and changes nothing about the fire geometry below.

use inferlet::eta::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// The fact hidden in the haystack. Kept short and lexically unlike the
    /// filler, so "did the answer survive" is a substring check and not a
    /// judgement call.
    #[serde(default = "default_needle")]
    needle: String,
    /// The question whose answer is the needle.
    #[serde(default = "default_question")]
    question: String,
    /// How many filler sentences surround the needle.
    ///
    /// **THE DEFAULT IS SIX, AND THAT IS THE MODEL'S LIMIT, NOT THE
    /// POLICY'S.** `Qwen/Qwen3.5-0.8B` recalls the needle from the FULL cache
    /// up to about seven filler sentences and echoes the question past that
    /// — so a longer haystack would make the control arm fail, and a gate
    /// whose control arm fails measures nothing. The eviction is unaffected
    /// by the length; the ability to check it is not.
    #[serde(default = "default_filler")]
    filler: usize,
    /// Where the needle sits, `0.0` = front, `1.0` = back.
    ///
    /// The default lands it wholly inside ONE page in the middle of the
    /// haystack — away from the sink and outside the recency window, the two
    /// regions a keep-set gets for free — which is what makes "the policy
    /// kept the page it needed" a claim about the SCORES and not about the
    /// two pages that were never at risk.
    #[serde(default = "default_depth")]
    depth: f32,
    /// How many PROMPT pages the decode is served. The pages the decode
    /// itself writes are always served; this is the budget for the prompt's.
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// `false` runs the control arm: the same prompt, the same sampling, the
    /// whole prompt served. The gate compares the two.
    #[serde(default = "default_evict")]
    evict: bool,
    /// Exported attention layers — declared, like `intrinsics::hidden`'s
    /// width. qwen35-d0.8b: 24 layers at `attn_every: 4` → 6 attention
    /// mixers, and only those export a score plane.
    #[serde(default = "default_layers")]
    layers: u32,
    /// Query heads per exported layer. qwen35-d0.8b: `q_heads: 8` at `tp == 1`.
    #[serde(default = "default_heads")]
    heads: u32,
    #[serde(default)]
    prefill_chunk: Option<u32>,
}

fn default_needle() -> String {
    "Remember this: the secret code word is banana.".to_string()
}
fn default_question() -> String {
    " What is the secret code word? The secret code word is".to_string()
}
fn default_filler() -> usize {
    6
}
fn default_depth() -> f32 {
    0.6
}
fn default_page_budget() -> u32 {
    5
}
fn default_max_tokens() -> usize {
    1
}
fn default_evict() -> bool {
    true
}
fn default_layers() -> u32 {
    6
}
fn default_heads() -> u32 {
    8
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    /// The continuation, greedily decoded.
    text: String,
    /// Whether eviction ran at all — `false` is the control arm.
    evicted: bool,
    /// Prompt tokens.
    prompt_len: u32,
    /// KV pages the PREFILL was served — the whole prompt.
    prompt_pages: u32,
    /// KV pages the DECODE's mask lets through, prompt pages only. **This is
    /// the shrink S-4 asks to see**, and the control arm reports the same
    /// number as `prompt_pages`.
    served_pages: u32,
    /// Which prompt pages survived, ascending.
    kept_pages: Vec<u32>,
    /// Prompt KV positions the served mask admits, against the number an
    /// unmasked decode attends over. THE MASK IS THE MEASUREMENT: a count of
    /// set bits in the row the fire actually carried, not a page count
    /// multiplied out.
    served_kv: u32,
    full_kv: u32,
    /// Which page the needle's tokens landed in, computed host-side from the
    /// prompt's own token offsets. The gate checks it survived the cut.
    needle_pages: Vec<u32>,
    /// Per-page attention mass from the observation window, device-folded.
    page_mass: Vec<String>,
    /// Attention layers folded into the row (`planes / heads`, declared).
    layers_observed: u32,
    /// Total mass over the live prefix — one distribution per layer, so this
    /// must come out at `layers_observed`.
    score_mass: f32,
    count: usize,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.layers == 0 || input.heads == 0 {
        return Err("layers and heads must both be at least 1".into());
    }
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }
    if !(0.0..=1.0).contains(&input.depth) {
        return Err("depth must be between 0.0 and 1.0".into());
    }
    let layers = input.layers;
    let heads = input.heads;
    let Some(planes) = layers.checked_mul(heads) else {
        return Err("layers * heads overflows the plane count".into());
    };
    let heads_f = heads as f32;
    let max_tokens = input.max_tokens;

    // ── THE HAYSTACK. Built here rather than passed in, so the gate can name
    //    a depth and a filler count instead of pasting a wall of text, and so
    //    the needle's TOKEN OFFSET is known exactly — which is what lets the
    //    output say which page it landed in.
    // VARIED, NOT REPEATED. A haystack of one sentence repeated N times is a
    // degenerate prompt: a small model stops predicting content and starts
    // predicting the loop, and the needle is then lost to the REPETITION
    // rather than to the eviction — which would make this gate measure the
    // wrong thing. Six sentences cycled keeps the filler uninformative
    // without making it pathological.
    const FILLER: [&str; 6] = [
        "The weather in the valley is mild and the roads are clear. ",
        "Deliveries arrive on Tuesdays and the depot closes at six. ",
        "The library keeps its old maps in the east reading room. ",
        "Rain is expected later in the week across the northern hills. ",
        "The bakery on the corner opens early and sells out by noon. ",
        "Two footbridges cross the river between the mill and the park. ",
    ];
    let at = ((input.filler as f32) * input.depth).round() as usize;
    let at = at.min(input.filler);
    let mut haystack = String::new();
    for i in 0..at {
        haystack.push_str(FILLER[i % FILLER.len()]);
    }
    let needle_at_chars = haystack.len();
    haystack.push_str(&input.needle);
    haystack.push(' ');
    for i in at..input.filler {
        haystack.push_str(FILLER[i % FILLER.len()]);
    }

    let before_needle = model::encode(&haystack[..needle_at_chars]);
    let through_needle = model::encode(&haystack[..needle_at_chars + input.needle.len()]);
    // **THE QUESTION IS PART OF THE OBSERVED PROMPT, AND THAT IS SnapKV.**
    // The paper's observation window is the LAST `window` rows of the prompt,
    // and the prompt is the whole instruction — haystack and question
    // together. That is the entire reason the policy works: the last rows are
    // the question, and what the question attends to is what the answer will
    // need. Observing the haystack alone would rank on what the filler
    // attends to, which is the filler, and the needle would lose every time.
    let question = model::encode(&input.question);
    let q = question.len().max(1) as u32;
    let mut prompt = model::encode(&haystack);
    prompt.extend(question.iter().copied());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    // Where the question's rows begin — the rows the answering fire re-reads
    // through the cut.
    let ask_at = n - q;

    let ws = WorkingSet::new();
    let page_size = kv_page_size();
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    let kv_max = max_pages * page_size;
    if kv_max > intrinsics::attn_score_kv_max() {
        return Err(format!(
            "this haystack needs {kv_max} KV slots, past the published attn_score \
             ceiling of {}",
            intrinsics::attn_score_kv_max()
        )
        .into());
    }
    let p_max = max_pages;
    // The last prompt page, which is also the page the decode writes into.
    let last_prompt_page = (n - 1) / page_size;
    let prompt_pages = last_prompt_page + 1;
    // Which pages the needle's tokens landed in — host arithmetic over the
    // token offsets the tokenizer just gave, so the gate can assert that the
    // policy kept the page it needed rather than that it kept SOME page.
    let needle_lo = before_needle.len() as u32;
    let needle_hi = (through_needle.len() as u32).max(needle_lo + 1).min(n);
    let needle_pages: Vec<u32> = (needle_lo / page_size..=(needle_hi - 1) / page_size).collect();

    ws.reserve(max_pages).context("reserve KV")?;

    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_chunks(n, input.prefill_chunk);
    let pipe = Pipeline::new();

    // ── PREFILL, every chunk but the last unobserved. Only the FINAL chunk's
    //    observation window is the prompt's tail, which is the quantity
    //    SnapKV selects on.
    for &(base, end) in &spans[..spans.len() - 1] {
        let chunk = end - base;
        let toks_c = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_c");
        let embed_indptr_c = Channel::from([0u32, chunk]).named("embed_indptr_c");
        let positions_c = Channel::from_iter(base..end).named("positions_c");
        let pages_c = Channel::from_iter(0..max_pages).named("pages_c");
        let page_indptr_c = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_c");
        let w_slot_c = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_c");
        let w_off_c = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_c");
        let kv_len_c = Channel::from([end]).named("kv_len_c");
        let tok_out_c = Channel::new([1], dtype::i32).named("tok_out_c");

        let fwd_c = ForwardPass::new();
        fwd_c.embed(&toks_c, &embed_indptr_c)?;
        fwd_c.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len_c,
                pages: &pages_c,
                page_indptr: &page_indptr_c,
                w_slot: &w_slot_c,
                w_off: &w_off_c,
                positions: &positions_c,
                mask: None,
            },
        )?;
        fwd_c.epilogue(move || {
            let logits = intrinsics::logits();
            tok_out_c.put(&reduce_argmax(logits));
        });
        fwd_c
            .submit(&pipe)
            .with_context(|| format!("prefill chunk submit @{base}"))?;
        tok_out_c
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("prefill chunk take @{base}"))?;
    }

    // ── THE OBSERVED CHUNK. Its epilogue reads the whole score rectangle and
    //    folds it twice on the device: heads and layers into one row of mass
    //    `layers`, then positions into pages. Only the page row and the total
    //    cross to the host, which is the decision §4 allows across.
    let base = spans[spans.len() - 1].0;
    let tail = n - base;
    let toks_p = Channel::from(&prompt_i32[base as usize..n as usize]).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, tail]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(base..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((base..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((base..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let page_mass_out = Channel::new([p_max], dtype::f32).named("evict_page_mass");
    let total_out = Channel::new([1], dtype::f32).named("evict_total");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len_p,
            pages: &pages_p,
            page_indptr: &page_indptr_p,
            w_slot: &w_slot_p,
            w_off: &w_off_p,
            positions: &positions_p,
            mask: None,
        },
    )?;
    fwd_p.epilogue(move || {
        let logits = intrinsics::logits();
        tok_out_p.put(&reduce_argmax(logits));

        // `transpose` puts the planes on the last axis so `reduce_sum` — which
        // reduces the LAST axis — sums down them; `/ heads` turns that
        // plane-sum into (mean over heads, then sum over layers). The result
        // is a row of mass exactly `layers`. `gather` narrows the published
        // width to this program's own page geometry.
        let rect = intrinsics::attn_score(planes);
        let folded = gather(&(&reduce_sum(&transpose(&rect)) / heads_f), iota(kv_max));
        // A page's share of the attention is the sum of its positions' shares.
        // `kv_max = p_max * page_size` exactly, so the reshape reinterprets
        // rather than resizes.
        let per_page = reduce_sum(&reshape(&folded, [p_max, page_size]));
        page_mass_out.put(&per_page);
        total_out.put(&reshape(&reduce_sum(&per_page), [1]));
    });
    fwd_p
        .submit(&pipe)
        .with_context(|| format!("prefill submit @{base}"))?;

    let _g0 = tok_out_p.take_host::<i32>().await?;
    let masses = page_mass_out.take_host::<Vec<f32>>().await?;
    let score_mass = total_out.take_host::<f32>().await?;

    // ── THE CUT. Page 0 and the last prompt page are not a choice (the
    //    header says why); the remaining budget is spent on mass.
    let mut kept: Vec<u32> = if input.evict {
        let budget = input.page_budget.max(2).min(prompt_pages) as usize;
        let mut forced = vec![0u32, last_prompt_page];
        forced.sort_unstable();
        forced.dedup();
        let mut ranked: Vec<u32> = (0..prompt_pages)
            .filter(|page| !forced.contains(page))
            .collect();
        ranked.sort_by(|a, b| {
            let (ma, mb) = (
                masses.get(*a as usize).copied().unwrap_or(0.0),
                masses.get(*b as usize).copied().unwrap_or(0.0),
            );
            mb.partial_cmp(&ma)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        let room = budget.saturating_sub(forced.len());
        let mut kept = forced;
        kept.extend(ranked.into_iter().take(room));
        kept.sort_unstable();
        kept
    } else {
        (0..prompt_pages).collect()
    };
    kept.dedup();
    let served_pages = kept.len() as u32;

    // ── THE KEEP ROW. One bit per readable KV slot: `true` where the softmax
    //    may look. Positions inside a kept prompt page survive; positions
    //    inside a dropped one do not; every position at or past the prompt is
    //    kept unconditionally, because those are the tokens this decode is
    //    about to write and a policy that evicted its own output would be
    //    evicting the future.
    //
    //    ANDed with the causal row in the epilogue rather than substituted
    //    for it: the mask REPLACES the derived causal bound
    //    (`Port::AttnMask`'s own doc), so a row that forgot causality would
    //    let a query attend to slots that do not exist yet. `naive-masked`'s
    //    document-isolation arm is the same shape for the same reason — a
    //    static row of the request, ANDed into a causal row that evolves.
    let pool_len = max_pages * page_size;
    let keep_set: std::collections::BTreeSet<u32> = kept.iter().copied().collect();
    let keep_row: Vec<bool> = (0..pool_len)
        .map(|j| j >= n || keep_set.contains(&(j / page_size)))
        .collect();
    // What the mask actually admits of the PROMPT, counted from the row the
    // fire carries rather than derived from the page count — so the number
    // the gate reads is a measurement and not a restatement of the policy.
    let served_kv = keep_row[..n as usize].iter().filter(|keep| **keep).count() as u32;

    // ── THE ANSWER, READ THROUGH THE CUT. One fire: the question's rows,
    //    embedded after the haystack, attending over the whole page list
    //    behind a mask that admits only the kept pages. Nothing about the
    //    geometry moves — same pages, same `kv_len`, same positions — so
    //    every kept key keeps the RoPE it was written with and the query
    //    keeps its true place in the sequence. The only thing that changed is
    //    which keys the softmax is allowed to see, which is what an eviction
    //    IS at the arithmetic (attn-score §4's first honest delta).
    //
    //    **IT RE-READS ROWS THE PREFILL ALREADY WROTE**, at the same
    //    positions, into the same cells, with the same values — an idempotent
    //    rewrite, and the only way to ask the model a question it has already
    //    been asked while changing what it may look at. The prefill's answer
    //    is the control; this one is the same question through the cut.
    //
    //    **ONE FIRE, AND THE ANSWER IS ONE TOKEN, ON PURPOSE.** A gate wants
    //    the mask row it served to be a value it can report, and a
    //    multi-token continuation would evolve that row on the device where
    //    nobody can read it back. `served_kv` below is a count of set bits in
    //    the row that actually flew.
    let question_i32: Vec<i32> = question.iter().map(|&t| t as i32).collect();
    let keep = &keep_row;
    let rows: Vec<bool> = (0..q)
        .flat_map(|r| {
            let at = ask_at + r;
            (0..pool_len).map(move |j| {
                // Causal, ANDed with the keep set — and the question's own
                // rows are never evicted: a policy that dropped the question
                // would be answering a different one. The mask REPLACES the
                // derived causal bound (`Port::AttnMask`), so forgetting the
                // causal half would let a row attend to slots that do not
                // exist yet.
                j <= at && (j >= ask_at || keep[j as usize])
            })
        })
        .collect();

    let toks_q = Channel::from(&question_i32[..]).named("toks_q");
    let embed_indptr_q = Channel::from([0u32, q]).named("embed_indptr_q");
    let positions_q = Channel::from_iter(ask_at..n).named("positions_q");
    let pages_q = Channel::from_iter(0..max_pages).named("pages_q");
    let page_indptr_q = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_q");
    let w_slot_q = Channel::from_iter((ask_at..n).map(|p| p / page_size)).named("w_slot_q");
    let w_off_q = Channel::from_iter((ask_at..n).map(|p| p % page_size)).named("w_off_q");
    let kv_len_q = Channel::from([n]).named("kv_len_q");
    let mask_q = Channel::from_shaped([q, pool_len], rows).named("mask_q");
    let tok_out_q = Channel::new([1], dtype::i32).named("tok_out_q");

    let fwd_q = ForwardPass::new();
    fwd_q.embed(&toks_q, &embed_indptr_q)?;
    fwd_q.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len_q,
            pages: &pages_q,
            page_indptr: &page_indptr_q,
            w_slot: &w_slot_q,
            w_off: &w_off_q,
            positions: &positions_q,
            mask: Some(&mask_q),
        },
    )?;
    fwd_q.epilogue(move || {
        let logits = intrinsics::logits();
        tok_out_q.put(&reduce_argmax(logits));
    });
    fwd_q.submit(&pipe).context("answer submit")?;
    let answer = tok_out_q.take_host::<i32>().await.context("answer take")?;
    let generated: Vec<u32> = vec![answer as u32];

    pipe.close();

    Ok(Output {
        sampler: "snapkv-eviction",
        text: model::decode(&generated)?,
        evicted: input.evict,
        prompt_len: n,
        prompt_pages,
        served_pages,
        kept_pages: kept,
        served_kv,
        full_kv: n,
        needle_pages,
        page_mass: masses[..prompt_pages as usize]
            .iter()
            .map(|m| format!("{m:.5}"))
            .collect(),
        layers_observed: layers,
        score_mass,
        count: generated.len(),
    })
}
