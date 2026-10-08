//! One spoken turn of a voice conversation.
//!
//! The app calls this once per utterance, with the whole transcript so far.
//! Everything that makes the turn cheap lives here rather than in the client:
//! the KV pages and recurrent state of earlier turns stay inside the engine,
//! published under the session's prefix index, so turn *n* prefills only the
//! tokens the index does not already hold instead of replaying the whole
//! transcript. That is the difference between a phone assistant that answers
//! in a beat and one that thinks for ten seconds before opening its mouth.
//!
//! Contract with the client:
//!
//! - **stdout** carries speakable text only, streamed as it is generated.
//!   Reasoning blocks never reach it, so the caller can hand chunks straight
//!   to a speech synthesizer without hearing the model reason out loud.
//! - **the return value** is a JSON object with the full reply plus this
//!   turn's prefix accounting, which keeps the numbers off the spoken channel.
//!
//! ```json
//! {"text": "...", "prompt_tokens": 326, "reused": 288, "new_prefill": 38,
//!  "generated": 41, "resumed": true, "note": ""}
//! ```
//!
//! `reused` is the prompt prefix served from earlier turns' published state,
//! `new_prefill` what this turn had to compute, and `resumed` whether any
//! prefix was found at all.
//!
//! The transcript is rendered through the host's chat template, so the one
//! boundary the next turn can share is known: the end of the new user message
//! (floored to a KV page), since the reply generated after it is never the
//! tokens the template replays it as. On a hybrid model the recurrent state
//! exists only where a prefill chunk ends, so the prefill ends a chunk there
//! and publishes that single snapshot, retiring the ones it supersedes: each
//! holds a state slot, a phone's pool has a handful, and which of them the
//! runtime drops once they run out is settled by its own pressure, not by
//! what the next turn needs. The next turn, whose transcript extends this
//! one, adopts it and prefills only what came after.

use std::io::{self, Write};

use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use inferlet::prefix_cache::PrefixCache;
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    /// The transcript so far, ending with the user's new message. Either a
    /// JSON array or that array as a JSON-encoded string, because a shell
    /// `--messages '[...]'` reaches the program as a string.
    #[serde(default, deserialize_with = "messages_or_json_string")]
    messages: Option<Vec<Message>>,

    /// A single user message, for callers without a transcript.
    #[serde(default)]
    text: Option<String>,

    /// Prepended when the transcript does not open with a system message.
    #[serde(default = "default_system")]
    system: String,

    /// Names the prefix index this conversation publishes to and adopts from.
    #[serde(default = "default_session")]
    session: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_temperature")]
    temperature: f32,

    #[serde(default = "default_top_p")]
    top_p: f32,

    /// Let the model reason before answering. Off by default: reasoning is
    /// latency the listener pays for and cannot hear.
    #[serde(default)]
    think: bool,

    /// Debug only: further user messages, each answered in the same process
    /// right after the reply before it, so prefix reuse can be observed from
    /// a shell where every `pie run` boots a fresh engine. One message as a
    /// string, or several as a JSON array (also as a string, for a shell).
    /// The return value then also carries a `turns` array with every turn's
    /// accounting.
    #[serde(default, deserialize_with = "followups")]
    followup: Vec<String>,
}

#[derive(Deserialize)]
struct Message {
    role: Role,
    content: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

fn messages_or_json_string<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<Vec<Message>>, D::Error> {
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => serde_json::from_str(&text)
            .map(Some)
            .map_err(de::Error::custom),
        Some(value) => serde_json::from_value(value)
            .map(Some)
            .map_err(de::Error::custom),
    }
}

fn followups<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<String>, D::Error> {
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        // A string is one message, unless it is a JSON array of them.
        Some(serde_json::Value::String(text)) => {
            Ok(serde_json::from_str(&text).unwrap_or_else(|_| vec![text]))
        }
        Some(value) => serde_json::from_value(value).map_err(de::Error::custom),
    }
}

fn default_session() -> String {
    "voice-session".into()
}
fn default_max_tokens() -> usize {
    120
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.95
}
fn default_system() -> String {
    "You are a voice assistant. Your replies are read aloud, so keep them \
     to one or two short spoken sentences. Never use markdown, lists, code \
     blocks, or emoji. Write numbers and symbols the way a person would say \
     them."
        .into()
}

/// One turn's reply and accounting, in the order the contract lists them.
#[derive(Clone, Serialize)]
struct Turn {
    text: String,
    prompt_tokens: u32,
    reused: u32,
    new_prefill: u32,
    generated: usize,
    resumed: bool,
    note: String,
}

#[derive(Serialize)]
struct Output {
    #[serde(flatten)]
    last: Turn,
    /// Only on the debug `followup` path: every turn this process ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    turns: Option<Vec<Turn>>,
}

#[derive(Clone, Copy)]
struct Sampling {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u32,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !(0.0..=1.0).contains(&input.top_p) || input.top_p == 0.0 {
        return Err("top_p must be greater than 0 and at most 1".into());
    }
    if !input.temperature.is_finite() || input.temperature < 0.0 {
        return Err("temperature must be a finite number >= 0".into());
    }
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }
    match model::pass_kind() {
        model::ForwardKind::Attention | model::ForwardKind::Hybrid => {}
        model::ForwardKind::Recurrent => {
            return Err("this program has no recurrent-only path (it needs a KV cache)".into());
        }
        model::ForwardKind::Diffusion => {
            return Err(
                "this program decodes a token at a time; a diffusion model wants a canvas loop"
                    .into(),
            );
        }
    }

    let mut messages = match (input.messages, input.text) {
        (Some(messages), _) if !messages.is_empty() => messages,
        (_, Some(text)) => vec![Message {
            role: Role::User,
            content: text,
        }],
        _ => return Err("input needs either \"messages\" or \"text\"".into()),
    };
    if messages[0].role != Role::System && !input.system.is_empty() {
        messages.insert(
            0,
            Message {
                role: Role::System,
                content: input.system,
            },
        );
    }

    // Keys are global to the model's store, and the tokens decide the state,
    // so the session name is all that separates one conversation's prefixes
    // from another's.
    let prefixes = PrefixCache::new(input.session.as_bytes());
    let sampling = Sampling {
        max_tokens: input.max_tokens,
        temperature: input.temperature,
        top_p: input.top_p,
        // The same question should not get the same answer in every
        // conversation, so the Gumbel key comes from the clock, not a constant.
        seed: inferlet::monotonic_now_ns() as u32,
    };

    let first = speak(&prefixes, &render(&messages, input.think), sampling).await?;
    if input.followup.is_empty() {
        return Ok(Output {
            last: first,
            turns: None,
        });
    }

    // The debug turns: each extends the transcript the way the app would,
    // with the reply just given and the next user message.
    let mut turns = Vec::with_capacity(input.followup.len() + 1);
    let mut last = first;
    for (i, followup) in input.followup.into_iter().enumerate() {
        messages.push(Message {
            role: Role::Assistant,
            content: last.text.clone(),
        });
        messages.push(Message {
            role: Role::User,
            content: followup,
        });
        println!();
        let seed = sampling.seed.rotate_left(7 * (i as u32 + 1)) ^ 0x9e37_79b9;
        let turn = speak(
            &prefixes,
            &render(&messages, input.think),
            Sampling { seed, ..sampling },
        )
        .await?;
        turns.push(std::mem::replace(&mut last, turn));
    }
    turns.push(last.clone());
    Ok(Output {
        last,
        turns: Some(turns),
    })
}

/// The transcript as tokens, with the offset each message ends at.
struct Prompt {
    tokens: Vec<u32>,
    ends: Vec<u32>,
}

/// Renders the transcript the way the compat server does: role fillers from
/// the host template, then the generation cue, then the closed reasoning
/// block when thinking is off. Message ends are kept because they are where
/// a later turn's transcript can still match this one, so they are where the
/// prefill publishes.
fn render(messages: &[Message], think: bool) -> Prompt {
    let mut tokens = Vec::new();
    let mut ends = Vec::with_capacity(messages.len());
    let mut seen_user = false;
    for m in messages {
        match m.role {
            Role::System => tokens.extend(chat::system(&m.content)),
            Role::User => {
                tokens.extend(if seen_user {
                    chat::user(&m.content)
                } else {
                    chat::first_user(&m.content)
                });
                seen_user = true;
            }
            Role::Assistant => tokens.extend(chat::assistant(&m.content)),
        }
        ends.push(tokens.len() as u32);
    }
    tokens.extend(chat::cue());
    if !think {
        tokens.extend(no_thinking_cue());
    }
    // The prefill needs at least one token.
    if tokens.is_empty() {
        tokens.push(0);
    }
    Prompt { tokens, ends }
}

/// The tokens that tell a thinking model not to think this turn: an empty
/// thinking block right after the cue, which is what the model's own
/// template emits for `enable_thinking: false`. Only when the vocabulary has
/// the block markers as single tokens; on any other model the request to
/// disable thinking is moot and this is nothing.
///
/// Left to itself on a resumed conversation, Qwen3 opens a `<think>` block
/// and never closes it, burying the entire reply inside a region that must
/// not be spoken. Prefilling the closed block puts generation outside any
/// reasoning region from the first token, which makes the turn predictable
/// and saves the tokens the model would have spent thinking.
fn no_thinking_cue() -> Vec<u32> {
    let open = model::encode("<think>");
    let close = model::encode("</think>");
    if open.len() != 1 || close.len() != 1 {
        return Vec::new();
    }
    let mut tokens = open;
    tokens.extend(model::encode("\n\n"));
    tokens.extend(close);
    tokens.extend(model::encode("\n\n"));
    tokens
}

/// The sampled token as a rank-0 i32: exact greedy at temperature zero, else
/// temperature scaling, nucleus truncation and a Gumbel-max draw from `r`,
/// the `[key, ctr]` rng state.
fn sample(vocab: u32, s: Sampling, r: &Tensor) -> Tensor {
    let logits = reshape(intrinsics::logits(), [vocab]);
    if s.temperature <= 0.0 {
        return reduce_argmax(&logits);
    }
    let scaled = &logits / s.temperature.max(1e-4);
    let truncated = if s.top_p < 1.0 {
        let keep = pivot_threshold(softmax(&scaled), cummass_le(s.top_p));
        mask_apply(&scaled, keep)
    } else {
        scaled
    };
    gumbel_max(&truncated, r)
}

/// Generates one assistant turn over `prompt`, streaming speakable text to
/// stdout, and publishes the prompt's prefixes for the next turn.
async fn speak(prefixes: &PrefixCache, prompt: &Prompt, s: Sampling) -> Result<Turn> {
    let tokens = &prompt.tokens;
    let n = tokens.len() as u32;
    let vocab = model::output_vocab_size();
    let page_size = kv_page_size();

    // Pages another turn already computed for the same leading tokens are
    // mapped in, not recomputed; only the tokens after them are prefilled.
    // On a hybrid model the adopted recurrent state already holds those
    // tokens too, so the prefill continues it from `covered` instead of
    // starting a cold one.
    let (ws, cached_rs, covered) = prefixes
        .adopt(tokens, Some(prompt.ends.as_slice()))
        .context("adopt the session's prefix")?;
    let rs_ws: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Hybrid => vec![cached_rs.unwrap_or_default()],
        _ => Vec::new(),
    };

    // An adopted working set already holds its pages (relative indexes
    // 0..page_len()); only the extra pages the rest of the prompt and the
    // decode need are reserved, and the reservation is purely logical.
    let max_pages = (n + s.max_tokens as u32 + 2).div_ceil(page_size);
    let have = ws.page_len();
    if max_pages > have {
        ws.reserve(max_pages - have).context("reserve KV pages")?;
    }

    // The prefill is chunked against the engine's per-launch token capacity,
    // and on a hybrid model also cut once, at the newest of the cache's
    // boundaries, because the recurrent state can be published only where a
    // chunk ends. Only the newest earns a snapshot: the transcript only
    // grows, so the next turn shares everything up to this turn's last
    // message, and an earlier boundary is at best a fallback that pins a
    // state slot.
    let boundary = prefixes
        .boundaries(tokens, covered, Some(prompt.ends.as_slice()))
        .pop();
    let state_keys = prefixes.state_keys(tokens);
    let mut spans: Vec<(u32, u32)> = Vec::new();
    for (base, end) in prefill_chunks(n - covered, None) {
        let (base, end) = (covered + base, covered + end);
        match boundary {
            Some(cut) if base < cut && cut < end => {
                spans.push((base, cut));
                spans.push((cut, end));
            }
            _ => spans.push((base, end)),
        }
    }

    // One pipeline for the whole turn: prefill chunks, then decode, in order.
    let pipe = Pipeline::new();
    let prompt_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let mut notes: Vec<String> = Vec::new();
    let mut states: Vec<u32> = Vec::new();
    let mut g0 = 0i32;
    for (k, &(base, end)) in spans.iter().enumerate() {
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr = Channel::from([0u32, end - base]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let pages = Channel::from_iter(0..max_pages).named("pages_p");
        // The page CSR is the source of truth for the KV length on the wire:
        // the count must track `kv_len`, never the pool size.
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        // A seeded channel attaches to exactly one pass, so each chunk draws
        // from its own rng.
        let rng = Channel::from([s.seed, k as u32]).named("rng_p");
        let g0_ch = Channel::new([1], dtype::i32).named("g0");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (base / page_size)..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            &rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
        .with_context(|| format!("bind prefill chunk @{base}"))?;
        // Every chunk samples, because a pass with no epilogue is refused;
        // only the last chunk's token continues the prompt, the others are
        // drained and dropped.
        fwd.epilogue(move || {
            let r = rng.take();
            g0_ch.put(reshape(sample(vocab, s, &r), [1]));
        });
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        g0 = g0_ch
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill drain @{base}"))?;

        // The chunk has landed: its recurrent state can be published now,
        // its KV once the whole prompt has (see `PrefixCache::publish_state`).
        if let Some(rs) = rs_ws.first().filter(|_| boundary == Some(end)) {
            match prefixes.publish_state(rs, &state_keys, end) {
                Ok(()) => states.push(end),
                Err(why) => notes.push(format!("state at {end} not published: {why}")),
            }
        }
    }

    // The prefill has landed: offer its prefix to the next turn and retire
    // the entries it supersedes. Keys chain the transcript's pages, so every
    // boundary an earlier turn of this session published is a boundary of
    // this transcript too; all but the newest go, because the next turn
    // adopts the newest, and each of the rest holds a recurrent-state slot
    // until the runtime, short of slots, drops unused snapshots on its own
    // schedule. Failing to publish costs the next turn a prefill, not this
    // one its reply, so it is reported rather than fatal.
    let key_at = |b: u32| state_keys[(b / page_size) as usize - 1].as_slice();
    match prefixes.publish(&ws, &states, &pipe, tokens, covered) {
        Ok(()) => {
            let keep = states.last().copied().unwrap_or(covered);
            for b in prefixes.boundaries(tokens, 0, Some(prompt.ends.as_slice())) {
                if b == keep {
                    continue;
                }
                if let Err(why) = retire(key_at(b)) {
                    notes.push(format!("boundary at {b} not retired: {why}"));
                }
            }
        }
        Err(why) => {
            notes.push(format!("prefix not published: {why}"));
            // A state whose KV never made it can never be adopted; it must
            // not keep a slot.
            for &b in &states {
                if let Err(why) = RsWorkingSet::remove_index(key_at(b)) {
                    notes.push(format!("state at {b} not retired: {why}"));
                }
            }
        }
    }

    let mut speech = Speech::new();
    let first_done = speech.feed(g0 as u32)?.is_break();

    // Decode: one pass, device loop-carried. The epilogue feeds the sampled
    // token straight back into the channel `embed` reads and advances the
    // geometry, so the host only drains the mirror in `out`.
    let budget = if first_done { 0 } else { s.max_tokens - 1 };
    if budget > 0 {
        let tok_in = Channel::from([g0]).named("tok_in");
        let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");
        let rng = Channel::from([s.seed ^ 0x9e37, 1u32]).named("rng");
        let out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &embed_indptr)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (n / page_size)..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            &rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
        .context("bind decode state")?;
        fwd.epilogue(move || {
            // Takes and compute first, puts last.
            let length = kv_len.take();
            let r = rng.take();
            let token = reshape(sample(vocab, s, &r), [1]);
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            let r_next = &r + iota(2);
            rng.put(&r_next);
            out.put(&token);
        });

        run_ahead(&pipe, &fwd, budget, async || {
            let t = out.take_host::<Vec<i32>>().await?;
            let token = *t.first().unwrap_or(&0) as u32;
            speech.feed(token)
        })
        .await?;
    }
    // Any fire still in flight after an early stop is left untaken; close
    // releases the scheduler wait-set, reclaims them, and rejects further
    // submissions.
    pipe.close();

    Ok(Turn {
        text: speech.finish(),
        prompt_tokens: n,
        reused: covered,
        new_prefill: n - covered,
        generated: speech.generated,
        resumed: covered > 0,
        note: notes.join("; "),
    })
}

/// Drops both halves of a published boundary from the index. Absent halves
/// are not an error: a turn retires every boundary of its transcript, and
/// most were never published.
fn retire(key: &[u8]) -> std::result::Result<(), String> {
    WorkingSet::remove_index(key)?;
    RsWorkingSet::remove_index(key)?;
    Ok(())
}

/// The spoken side of a turn: detokenizes, strips reasoning, streams.
struct Speech {
    decoder: chat::Decoder,
    stop: Vec<u32>,
    stripper: ReasoningStripper,
    raw: String,
    spoken: String,
    generated: usize,
}

impl Speech {
    fn new() -> Self {
        Self {
            decoder: chat::Decoder::new(),
            stop: chat::stop_tokens(),
            stripper: ReasoningStripper::new(),
            raw: String::new(),
            spoken: String::new(),
            generated: 0,
        }
    }

    /// One sampled token; `Break` ends the turn.
    fn feed(&mut self, token: u32) -> Result<ControlFlow<()>> {
        if self.stop.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        self.generated += 1;
        match self.decoder.feed(&[token])? {
            chat::Event::Delta(delta) => {
                self.raw.push_str(&delta);
                // Only speakable text reaches stdout, and it is flushed per
                // delta so the synthesizer starts on sentence one instead of
                // waiting for the turn to finish.
                let visible = self.stripper.process(&delta);
                self.say(&visible);
                Ok(ControlFlow::Continue(()))
            }
            chat::Event::Done(text) => {
                self.raw = text;
                Ok(ControlFlow::Break(()))
            }
            chat::Event::Interrupt(_) => Ok(ControlFlow::Continue(())),
        }
    }

    fn say(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.spoken.push_str(text);
        print!("{text}");
        let _ = io::stdout().flush();
    }

    /// The whole spoken reply, once the turn is over.
    fn finish(&mut self) -> String {
        // Release the holdback. Without this the last few characters of
        // every reply, held back in case they were the start of a `<think>`
        // tag, are never streamed, so the synthesizer clips the final word.
        let tail = self.stripper.flush();
        self.say(&tail);
        // `Done` hands back the whole turn at once, so re-derive the spoken
        // text from the raw transcript when the stream ended that way.
        let recovered = speakable_text(&self.raw);
        if recovered.trim() != self.spoken.trim() {
            return recovered.trim().to_string();
        }
        self.spoken.trim().to_string()
    }
}

/// The part of a raw assistant turn that should be said out loud.
///
/// With the closed block prefilled there is usually nothing to strip. The
/// two fallbacks matter anyway: a model that emits its own complete block
/// (everything after the last `</think>`), and the case that actually bites,
/// one that opens a block and never closes it, where the reply is the block's
/// contents and dropping it would leave silence.
fn speakable_text(raw: &str) -> String {
    if let Some(index) = raw.rfind("</think>") {
        return raw[index + "</think>".len()..].to_string();
    }
    if let Some(index) = raw.find("<think>") {
        return raw[index + "<think>".len()..].to_string();
    }
    raw.to_string()
}

/// Largest char boundary at or below `idx`. Token deltas are arbitrary
/// UTF-8, so the holdback below cannot slice on raw byte offsets.
fn floor_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Removes `<think>...</think>` from a token stream without ever emitting a
/// partial tag: text is held back until it is certain no tag straddles the
/// chunk boundary.
struct ReasoningStripper {
    in_think: bool,
    pending: String,
    /// Whitespace that opens the reply, or follows a reasoning block, is the
    /// template's padding, not speech; a synthesizer fed it first may pause.
    leading: bool,
}

impl ReasoningStripper {
    fn new() -> Self {
        Self {
            in_think: false,
            pending: String::new(),
            leading: true,
        }
    }

    fn process(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(idx) = self.pending.find("</think>") {
                    self.pending = self.pending.split_off(idx + "</think>".len());
                    self.in_think = false;
                    self.leading = true;
                    continue;
                }
                // Keep the tail that could be a partial "</think>".
                let cut = floor_boundary(&self.pending, self.pending.len().saturating_sub(7));
                if cut > 0 {
                    self.pending = self.pending.split_off(cut);
                }
                break;
            }
            if self.leading {
                let padding = self.pending.len() - self.pending.trim_start().len();
                self.pending.drain(..padding);
                if self.pending.is_empty() {
                    break;
                }
                self.leading = false;
            }
            if let Some(idx) = self.pending.find("<think>") {
                out.push_str(&self.pending[..idx]);
                self.pending = self.pending.split_off(idx + "<think>".len());
                self.in_think = true;
                continue;
            }
            // Same idea for a partial "<think>".
            let safe = floor_boundary(&self.pending, self.pending.len().saturating_sub(6));
            if safe > 0 {
                let head: String = self.pending.drain(..safe).collect();
                out.push_str(&head);
            }
            break;
        }
        out
    }

    /// Everything still held back at the end of the turn.
    ///
    /// Returns nothing while inside an unterminated block: that text is
    /// reasoning as far as this decoder can tell, and `speakable_text`
    /// recovers it from the raw transcript if it turns out to be a reply.
    fn flush(&mut self) -> String {
        if self.in_think {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}
