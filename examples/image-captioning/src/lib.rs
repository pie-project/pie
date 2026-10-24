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
use inferlet::eta::prelude::*;
use inferlet::media;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// Which colour the synthesized square is filled with.
    #[serde(default = "default_color")]
    color: String,
    /// Real pictures instead of the synthesized square: PNG or JPEG bytes,
    /// base64 (standard alphabet, padding optional), several separated by
    /// `;` — each becomes its own span, in order, before the question. The
    /// guest has no filesystem, so the pictures ride the arguments. `color`
    /// is then only echoed.
    #[serde(default)]
    image_b64: Option<String>,
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
    /// How many tokens the prefill fired, spans included.
    prompt_tokens: u32,
    /// How many pictures the prompt carried.
    images: usize,
}

/// The three unambiguous fills. Named rather than free-form because the gate's
/// claim is "the caption mentions something TRUE about the image", and a colour
/// this list does not carry is a claim nobody wrote down.
fn fill(color: &str) -> Result<[u8; 3]> {
    match color {
        "red" => Ok([255, 0, 0]),
        "green" => Ok([0, 255, 0]),
        "blue" => Ok([0, 0, 255]),
        // Diagnostics for a tower's numerics: the centre and the two ends of
        // the pixel range. Not a colour claim the gate asserts on.
        "grey" => Ok([128, 128, 128]),
        "black" => Ok([0, 0, 0]),
        "white" => Ok([255, 255, 255]),
        // A hue sweep over a checker (`png::textured`): every patch differs,
        // so the picture's rows route to as many experts as a picture can.
        "textured" => Ok([0, 0, 0]),
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
    let rgb = if input.image_b64.is_some() {
        [0, 0, 0]
    } else {
        fill(&input.color)?
    };
    if input.side == 0 {
        return Err("side must be at least one pixel".into());
    }
    let max_tokens = input.max_tokens.max(1);

    // ── THE PICTURES. Bytes in, and the host decides everything about them.
    let pictures: Vec<Vec<u8>> = match &input.image_b64 {
        Some(b64) => b64
            .split(';')
            .filter(|piece| !piece.trim().is_empty())
            .map(png::base64_decode)
            .collect::<Result<_>>()?,
        None if input.color == "textured" => vec![png::textured(input.side)],
        None => vec![png::solid(input.side, rgb)],
    };
    let images: Vec<media::Image> = pictures
        .iter()
        .map(|bytes| media::Image::from_bytes(bytes))
        .collect::<Result<_>>()?;
    let img = images.first().ok_or("no picture")?;

    // ── THE LEDGER (media-door §0/§1). Text, then each span's own spelling,
    //    then text. This is the whole of the guest's model-specific knowledge:
    //    none. `tokens()` answered the bound checkpoint's delimiter and pad
    //    ids, so nothing here counts a run or names a special.
    let mut prompt_tokens = chat::system(&input.system);
    for image in &images {
        prompt_tokens.extend(image.tokens());
    }
    prompt_tokens.extend(chat::user(&input.question));
    prompt_tokens.extend(chat::cue());
    let n = prompt_tokens.len() as u32;
    let stop = chat::stop_tokens();

    let soft_tokens: u32 = images.iter().map(media::Image::token_count).sum();
    let grid = img.grid();
    let digest = img
        .digest()
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    // ───────────────────────── 1. THE MEDIA PREFILL ─────────────────────────
    //
    // **THE WHOLE PROMPT IN ONE PASS.** The host scans the pass's tokens for
    // the runs and keeps where an M-RoPE span leaves the rotation behind the
    // slots: the rows after it, and the decode, rotate there by default.
    let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(&prompt_i32[..]).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    // **THE PAYLOAD, ORDER-MATCHED TO THE RUNS.** One span, one run, and the
    // host is what checks that — the guest states the correspondence by
    // ordering alone and can state nothing else.
    let spans: Vec<media::Span> = images.iter().map(media::Span::Image).collect();
    fwd_p.media(&spans)?;
    fwd_p.state(&[&ws], StateGeometry::folding())?;
    fwd_p.epilogue(move || {
        g0_ch.put(&sample_token());
    });
    fwd_p.submit(&pipe).context("prefill submit")?;
    let g0 = g0_ch.take_host::<i32>().await?;

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
    // which is exactly the right answer for a fire that attached nothing. It
    // rotates where the working set says: behind its slot, past a span.
    let budget = if done {
        0
    } else {
        max_tokens.saturating_sub(1)
    };
    let tok_in = Channel::from([g0]).named("tok_in");
    let out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("out");
    let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let tok = sample_token();
        tok_in.put(&tok);
        out.put(&tok);
    });

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
        images: images.len(),
    })
}
