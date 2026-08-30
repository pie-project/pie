//! **CAPTIONING THROUGH THE MEDIA DOOR** — `.wiki/alto/media-door.md` §1's
//! guest surface, written exactly as it spells it, and the gate M-1(b) closes
//! on.
//!
//! ```text
//! let img = Image::from_bytes(&bytes)?;   // host decodes + preprocesses
//! toks.extend(img.tokens());              // the span IS its token run
//! fwd.embed(&toks)?;
//! fwd.media(&[Span::Image(&img)])?;       // the payload, order-matched
//! ```
//!
//! **ONE LEDGER.** The image enters the sequence as the token run its handle
//! answers — the bound model's own delimiter + placeholder ids — and as
//! nothing else. No anchor list, no offset, no length, no second bookkeeping
//! structure. The handle crosses a second time beside the tokens carrying only
//! the payload, and the HOST scans the submitted tokens for the reserved pad
//! and matches the runs to the spans in order. Nothing in this file names a
//! model, a patch, a grid or a special token: pixels are made here and are
//! never seen again, and every model-specific act happens above the WIT.
//!
//! **THE IMAGE IS SYNTHESIZED, AND THAT IS THE GATE'S OWN ARGUMENT** (see
//! `png`). Determinism is what a caption gate rests on, and a solid square is
//! a picture whose content is a sentence a reader of this file can check.
//!
//! **WHAT `Output` LETS A GATE ASSERT, AND WHAT IT HONESTLY CANNOT.** `text`
//! is a caption; a 0.8-billion-parameter model's caption of a flat colour
//! field is a weak claim about quality and a strong one about plumbing. So the
//! report carries the numbers beside it — how many soft tokens the span
//! occupied, the merged grid it occupied them in, and the span's own digest —
//! and the census asserts three things: that a colour word appears (sensible),
//! that two runs of one colour answer identically (deterministic), and that
//! two DIFFERENT colours answer differently (the tower actually conditioned
//! the trunk, which is the one claim that holds whatever the model's taste).

mod png;

use inferlet::chat;
use inferlet::eta::attention::prelude::*;
use inferlet::media;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// Which colour the synthesized square is filled with.
    #[serde(default = "default_color")]
    color: String,
    /// Its side, in pixels.
    #[serde(default = "default_side")]
    side: u32,
    /// What to ask about it.
    #[serde(default = "default_question")]
    question: String,
    /// System message.
    #[serde(default = "default_system")]
    system: String,
    /// Maximum generated tokens.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_color() -> String {
    "red".into()
}
fn default_side() -> u32 {
    224
}
fn default_question() -> String {
    "What is the dominant colour of the image above? Answer with one word.".into()
}
fn default_system() -> String {
    "You are a helpful assistant that describes images.".into()
}
fn default_max_tokens() -> usize {
    16
}

#[derive(Serialize)]
struct Output {
    /// The caption, decoded.
    text: String,
    /// The colour the square was filled with, echoed so a report is readable
    /// on its own.
    color: String,
    /// How many token rows the span occupied — the length of the placeholder
    /// run the host scanned for.
    soft_tokens: u32,
    /// The merged grid those rows sit in, `[t, h, w]`.
    grid: [u32; 3],
    /// The span's content digest, first eight bytes, hex. media-door §5: two
    /// different images produce IDENTICAL token lists, so this is the only
    /// thing that tells two spans apart — and the only thing a cache key over
    /// a media run may be built on.
    digest: String,
    /// How many tokens the prefill fired, span included.
    prompt_tokens: u32,
}

/// The three unambiguous fills. Named rather than free-form because the gate's
/// claim is "the caption mentions something TRUE about the image", and a colour
/// this list does not carry is a claim nobody wrote down.
fn fill(color: &str) -> Result<[u8; 3]> {
    match color {
        "red" => Ok([255, 0, 0]),
        "green" => Ok([0, 255, 0]),
        "blue" => Ok([0, 0, 255]),
        other => Err(format!(
            "unknown colour {other:?}: this gate fills with red, green or blue, \
             because its assertion is that the caption names the colour and an \
             unnamed colour is an assertion nobody wrote down"
        )),
    }
}

/// Greedy, always: a caption gate that sampled would assert on a distribution.
fn sample_token() -> Tensor {
    reduce_argmax(&intrinsics::logits())
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let rgb = fill(&input.color)?;
    if input.side == 0 {
        return Err("side must be at least one pixel".into());
    }
    let max_tokens = input.max_tokens.max(1);

    // ── THE PICTURE. Bytes in, and the host decides everything about them.
    let bytes = png::solid(input.side, rgb);
    let img = media::Image::from_bytes(&bytes)?;

    // ── THE LEDGER (media-door §0/§1). Text, then the span's own spelling,
    //    then text. This is the whole of the guest's model-specific knowledge:
    //    none. `tokens()` answered the bound checkpoint's delimiter and pad
    //    ids, so nothing here counts a run or names a special.
    let mut prompt_tokens = chat::system(&input.system);
    prompt_tokens.extend(img.tokens());
    prompt_tokens.extend(chat::user(&input.question));
    prompt_tokens.extend(chat::cue());
    let n = prompt_tokens.len() as u32;
    let stop = chat::stop_tokens();

    let soft_tokens = img.token_count();
    let grid = img.grid();
    let digest = img
        .digest()
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    let page_t = kv_page_size();
    let pool_pages = (n + max_tokens as u32 + 2).div_ceil(page_t);
    let ws = WorkingSet::new();
    let slots = ws.reserve(pool_pages).context("ws.reserve")?;
    let pool_ids = slots.ids().to_vec();
    let pipe = Pipeline::new();

    // ───────────────────── 1. THE MEDIA PREFILL (ONE FIRE) ──────────────────
    //
    // **ONE FIRE AND NOT A CHUNKED ONE**, which is a real constraint and not a
    // simplification: the host scans each fire's own token list for maximal
    // placeholder runs, so a prefill split across two fires would hand it half
    // a run twice — `MediaRunLength` on one and `MediaOrphanRuns` on the other.
    // A span's run and its payload arrive together or the door refuses them.
    let g0 = {
        let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let toks_p = Channel::from(prompt_i32).named("toks_p");
        let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
        let positions_p = Channel::from_iter(0..n).named("positions_p");
        let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / page_t) as usize]).collect();
        let w_off_pv: Vec<u32> = (0..n).map(|c| c % page_t).collect();
        let w_slot_p = Channel::from(w_slot_pv).named("w_slot_p");
        let w_off_p = Channel::from(w_off_pv).named("w_off_p");
        let klen_p = Channel::from([n]).named("klen_p");
        let pages_p = Channel::from(pool_ids.clone()).named("pages_p");
        let page_indptr_p = Channel::from([0u32, n.div_ceil(page_t)]).named("pidx_p");
        let g0_ch = Channel::new([1], dtype::i32).named("g0");

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        // **THE PAYLOAD, ORDER-MATCHED TO THE RUNS.** One span, one run, and
        // the host is what checks that — the guest states the correspondence
        // by ordering alone and can state nothing else.
        fwd_p.media(&[media::Span::Image(&img)])?;
        fwd_p.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &klen_p,
                pages: &pages_p,
                page_indptr: &page_indptr_p,
                w_slot: &w_slot_p,
                w_off: &w_off_p,
                positions: &positions_p,
                mask: None,
            },
        )?;
        fwd_p.epilogue(move || {
            g0_ch.put(&sample_token());
        });
        fwd_p.submit(&pipe).context("prefill submit")?;
        g0_ch.take_host::<i32>().await?
    };

    let chat_dec = chat::Decoder::new();
    let mut text = String::new();
    let mut done = stop.contains(&(g0 as u32));
    if !done {
        match chat_dec.feed(&[g0 as u32])? {
            chat::Event::Delta(s) => text.push_str(&s),
            chat::Event::Done(s) => {
                text = s;
                done = true;
            }
            _ => {}
        }
    }

    // ───────────────── 2. THE DECODE LOOP (TEXT ONLY, 1-WIDE) ───────────────
    //
    // Carries no span and never will: the image is in the KV now, and a decode
    // fire that re-attached it would be a second submission of one picture.
    // Its tokens are sampled ids, so the run scan finds nothing in them —
    // which is exactly the right answer for a fire that attached nothing.
    let slot_n = pool_ids[(n / page_t) as usize];
    let tok_in = Channel::from([g0]).named("tok_in");
    let pos = Channel::from([n]).named("pos");
    let fill_ch = Channel::from([n + 1]).named("fill");
    let klen = Channel::from([n + 1]).named("klen");
    let w_slot = Channel::from([slot_n]).named("w_slot");
    let w_off = Channel::from([n % page_t]).named("w_off");
    let pages = Channel::from(pool_ids.clone()).named("pages");
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_t)]).named("page_indptr");
    let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
    let out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("out");
    let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (n / page_t)..,
            kv_len: &klen,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &pos,
            mask: None,
        },
    )?;
    fwd.epilogue(move || {
        let base = fill_ch.take();
        let pids = pool_ids_ch.take();
        let tok = sample_token();

        let logical_slot = &base / page_t;
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = &base % page_t;
        let klen_v = &base + 1u32;
        let next_free = &base + 1u32;
        let pages_v = reshape(&pids, [pool_pages]);
        let page_count = klen_v.div_ceil(page_t);
        let pidx_v = indptr(1, &page_count);

        tok_in.put(&tok);
        out.put(&tok);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill_ch.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
        pool_ids_ch.put(&pids);
    });

    let budget = if done { 0 } else { max_tokens.saturating_sub(1) };
    run_ahead(&pipe, &fwd, budget, async || {
        let t = out.take_host::<Vec<i32>>().await?;
        let token = *t.first().unwrap_or(&0) as u32;
        if stop.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        match chat_dec.feed(&[token])? {
            chat::Event::Delta(s) => text.push_str(&s),
            chat::Event::Done(s) => {
                text = s;
                return Ok(ControlFlow::Break(()));
            }
            _ => {}
        }
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipe.close();

    Ok(Output {
        text: text.trim().to_string(),
        color: input.color,
        soft_tokens,
        grid: [grid.t, grid.h, grid.w],
        digest,
        prompt_tokens: n,
    })
}
