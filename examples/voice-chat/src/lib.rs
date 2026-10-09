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
//! - **session messages** carry the reasoning, streamed as it is generated,
//!   for a client that shows the model thinking.
//! - **the return value** is a JSON object with the full reply plus this
//!   turn's prefix accounting, which keeps the numbers off the spoken channel.
//!
//! ```json
//! {"text": "...", "reasoning": "", "thought_tokens": 0, "prompt_tokens": 326,
//!  "reused": 288, "new_prefill": 38, "generated": 41, "resumed": true,
//!  "note": ""}
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
//!
//! With `think` on, the prompt opens the reasoning block itself, and the
//! program, not the model, decides when thinking ends: once the reasoning has
//! taken `thinking_budget` tokens without closing, decoding stops, the closing
//! tag is prefilled, and the answer is decoded from there. A small model that
//! would otherwise reason until it runs out of tokens still says something.
//! The one turn that ends without an answer is one the model ends itself,
//! with its stop token, before the block closes: `text` is then empty and
//! `note` says why, because everything it generated was reasoning.
//!
//! A small model can also fall into a loop, writing the same few sentences
//! again and again until `max_tokens` runs out; Qwen3.5-0.8B does it planning
//! a trip with thinking on, in the answer and in the reasoning alike. Two
//! things stand against that. Sampling the answer leans away from every
//! token the answer has already used, dividing a positive logit by
//! `repetition_penalty` and multiplying a negative one, which makes a loop
//! less likely to start. And the host watches every token it drains: once
//! the newest `LOOP_WINDOW` tokens have already appeared, verbatim, earlier
//! in the same stretch (the reasoning block the prompt opened, or the answer
//! after it), the model is repeating itself. In the answer that ends the
//! turn, and `text` is cut back to the last complete sentence before the
//! repeat began. In the reasoning it ends the reasoning, exactly as a spent
//! thinking budget does. Either way `note` says so.
//!
//! The penalty leaves the reasoning alone. Reasoning restates what it has
//! worked out, drafting and redrafting the answer, and penalized it drifts
//! instead: asked for the capital of France with a 150-token budget,
//! Qwen3.5-0.8B named Paris in 1 of 5 turns with its reasoning penalized and
//! in 5 of 6 with only its answer penalized. A reasoning loop is left to the
//! guard, which ends it sooner than the budget would.

use std::collections::HashSet;
use std::io::{self, Write};

use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use inferlet::prefix_cache::PrefixCache;
use inferlet::session;
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};

/// The fewest tokens the answer gets once the program has closed a reasoning
/// block, even past `max_tokens`: room for the one or two spoken sentences the
/// system prompt asks for, so a turn that reasoned never ends in silence.
const MIN_ANSWER: usize = 64;

/// The most a reasoning block gets when the caller names no
/// `thinking_budget`. A small model rarely closes its reasoning on its own
/// (Qwen3.5-0.8B, the model the phone runs, practically never does), so a
/// thinking turn spends whatever budget it gets before it says a word; at
/// the phone's 60 to 70 tokens a second this is about four seconds of
/// silence, where two thirds of a chat-sized `max_tokens` would be fifteen.
/// A caller that wants the model to reason longer passes the budget.
const DEFAULT_THINKING_CAP: usize = 256;

/// How many tokens in a row must recur, verbatim, before the reply counts as
/// a loop. Long enough that ordinary prose never trips it: a stock phrase or
/// a repeated place name runs a handful of tokens, where 24 is most of a
/// sentence. Short enough that a loop is caught a sentence into its first
/// repeat rather than paragraphs later; at the phone's 60 to 70 tokens a
/// second that is under half a second of repeated speech.
const LOOP_WINDOW: usize = 24;

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

    /// Divides the logit of a token the answer has already generated when it
    /// is positive, and multiplies it when it is negative, so the model leans
    /// away from what it has just said. `1.0` turns it off. Neither prompt
    /// nor reasoning tokens count, and the reasoning itself is not penalized:
    /// an answer should be free to use the user's words and to say what the
    /// reasoning settled on.
    #[serde(default = "default_repetition_penalty")]
    repetition_penalty: f32,

    /// Let the model reason before answering. Off by default: reasoning is
    /// latency the listener pays for and cannot hear.
    #[serde(default)]
    think: bool,

    /// With `think` on, the most tokens the reasoning may take before the
    /// program closes it and has the model answer. Two thirds of
    /// `max_tokens`, at most `DEFAULT_THINKING_CAP`, when absent.
    #[serde(default)]
    thinking_budget: Option<usize>,

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
fn default_repetition_penalty() -> f32 {
    1.1
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
    /// What the model reasoned before answering; never part of `text`.
    reasoning: String,
    /// Of `generated`, the tokens spent inside reasoning blocks.
    thought_tokens: usize,
    prompt_tokens: u32,
    reused: u32,
    new_prefill: u32,
    generated: usize,
    resumed: bool,
    /// Why a publish or a retirement was skipped, why `text` is empty after
    /// a turn that only reasoned, or that a repetition loop was stopped.
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
    /// How many tokens a reasoning block the prompt opened may take before
    /// the program closes it.
    thinking_budget: usize,
    temperature: f32,
    top_p: f32,
    /// `1.0` is off; see `Input::repetition_penalty`.
    repetition_penalty: f32,
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
    if input.thinking_budget == Some(0) {
        return Err("thinking_budget must be at least 1".into());
    }
    if !input.repetition_penalty.is_finite() || input.repetition_penalty <= 0.0 {
        return Err("repetition_penalty must be a finite number > 0".into());
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
    let tags = ThinkTags::of_model();
    let sampling = Sampling {
        max_tokens: input.max_tokens,
        thinking_budget: input
            .thinking_budget
            .unwrap_or((input.max_tokens * 2 / 3).clamp(1, DEFAULT_THINKING_CAP)),
        temperature: input.temperature,
        top_p: input.top_p,
        repetition_penalty: input.repetition_penalty,
        // The same question should not get the same answer in every
        // conversation, so the Gumbel key comes from the clock, not a constant.
        seed: inferlet::monotonic_now_ns() as u32,
    };

    let first = speak(
        &prefixes,
        &render(&messages, input.think, tags),
        tags,
        sampling,
    )
    .await?;
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
            &render(&messages, input.think, tags),
            tags,
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
    /// The prompt ends inside a reasoning block it opened, so generation
    /// starts as reasoning.
    reasoning: bool,
}

/// Renders the transcript the way the compat server does: role fillers from
/// the host template, then the generation cue, then the reasoning block's
/// opening when thinking is on, or the closed, empty block when it is off.
/// Message ends are kept because they are where a later turn's transcript
/// can still match this one, so they are where the prefill publishes.
fn render(messages: &[Message], think: bool, tags: Option<ThinkTags>) -> Prompt {
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
    if let Some(tags) = tags {
        tokens.extend(if think {
            tags.open_block()
        } else {
            tags.empty_block()
        });
    }
    // The prefill needs at least one token.
    if tokens.is_empty() {
        tokens.push(0);
    }
    Prompt {
        tokens,
        ends,
        reasoning: think && tags.is_some(),
    }
}

/// The reasoning block's markers, when the vocabulary has each as a single
/// token. Only then can the prompt open or close the block, and only then
/// does the token stream alone say where the block ends, which is what a
/// thinking budget counts in. On any other model a request to think or not
/// to think is moot, and the prompt carries neither.
#[derive(Clone, Copy)]
struct ThinkTags {
    open: u32,
    close: u32,
}

impl ThinkTags {
    fn of_model() -> Option<Self> {
        match (
            model::encode("<think>").as_slice(),
            model::encode("</think>").as_slice(),
        ) {
            (&[open], &[close]) => Some(Self { open, close }),
            _ => None,
        }
    }

    /// The tokens that tell a thinking model not to think this turn: an
    /// empty thinking block right after the cue, which is what the model's
    /// own template emits for `enable_thinking: false`.
    ///
    /// Left to itself on a resumed conversation, Qwen3 opens a `<think>`
    /// block and never closes it, burying the entire reply inside a region
    /// that must not be spoken. Prefilling the closed block puts generation
    /// outside any reasoning region from the first token, which makes the
    /// turn predictable and saves the tokens the model would have spent
    /// thinking.
    fn empty_block(self) -> Vec<u32> {
        let mut tokens = vec![self.open];
        tokens.extend(model::encode("\n\n"));
        tokens.push(self.close);
        tokens.extend(model::encode("\n\n"));
        tokens
    }

    /// The opening the model's own template emits for `enable_thinking:
    /// true`. Opening the block in the prompt, rather than leaving it to the
    /// model, makes every thinking turn reason, and puts generation inside
    /// the block from its first token, which is where the budget counts from.
    fn open_block(self) -> Vec<u32> {
        let mut tokens = vec![self.open];
        tokens.extend(model::encode("\n"));
        tokens
    }

    /// What the program prefills to end a block the budget cut short: the
    /// same shape the model closes one with, so the answer after it starts
    /// the way it would have had the model stopped thinking on its own.
    fn closing(self) -> Vec<u32> {
        let mut tokens = model::encode("\n");
        tokens.push(self.close);
        tokens.extend(model::encode("\n\n"));
        tokens
    }
}

/// The sampled token as a rank-0 i32: exact greedy at temperature zero, else
/// temperature scaling, nucleus truncation and a Gumbel-max draw from `r`,
/// the `[key, ctr]` rng state. `seen`, when the repetition penalty is on,
/// counts how often each token of the vocabulary has been generated in the
/// answer; the penalty goes on before anything else, so greedy decoding
/// takes the penalized maximum and nothing more changes.
fn sample(vocab: u32, s: Sampling, r: &Tensor, seen: Option<&Tensor>) -> Tensor {
    let mut logits = reshape(intrinsics::logits(), [vocab]);
    if let Some(seen) = seen {
        // Dividing a negative logit would raise it, so those are multiplied.
        let zero = broadcast(0.0f32, [vocab]);
        let rp = broadcast(s.repetition_penalty, [vocab]);
        let penalized = select(gt(&logits, &zero), &logits / &rp, &logits * &rp);
        logits = select(gt(seen, &zero), &penalized, &logits);
    }
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
/// stdout and reasoning as session messages, and publishes the prompt's
/// prefixes for the next turn.
async fn speak(
    prefixes: &PrefixCache,
    prompt: &Prompt,
    tags: Option<ThinkTags>,
    s: Sampling,
) -> Result<Turn> {
    let tokens = &prompt.tokens;
    let n = tokens.len() as u32;

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

    // With the reasoning opened by the prompt, the first decode loop stops
    // where the thinking budget runs out, and if the block is still open
    // there the program closes it. The answer after that gets at least
    // `MIN_ANSWER` tokens, so the most this turn can write is more than
    // `max_tokens` by up to that much plus the closing tokens.
    let cut = tags.filter(|_| prompt.reasoning).map(|tags| Cut {
        at: s.thinking_budget.min(s.max_tokens),
        closing: tags.closing(),
    });
    let most = match &cut {
        Some(cut) => s.max_tokens.max(cut.at + MIN_ANSWER) + cut.closing.len(),
        None => s.max_tokens,
    };

    // An adopted working set already holds its pages (relative indexes
    // 0..page_len()); only the extra pages the rest of the prompt and the
    // decode need are reserved, and the reservation is purely logical.
    let page_size = kv_page_size();
    let max_pages = (n + most as u32 + 2).div_ceil(page_size);
    let have = ws.page_len();
    if max_pages > have {
        ws.reserve(max_pages - have).context("reserve KV pages")?;
    }
    let passes = Passes {
        ws: &ws,
        rs: &rs_ws,
        max_pages,
        page_size,
        vocab: model::output_vocab_size(),
        sampling: s,
    };

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
        // A seeded channel attaches to exactly one pass, so each chunk draws
        // from its own rng. Every chunk samples, because a pass with no
        // epilogue is refused; only the last chunk's token continues the
        // prompt, the others are dropped. None of them carries a histogram
        // for the repetition penalty: the answer has no tokens yet, and an
        // empty histogram penalizes nothing.
        g0 = passes
            .prefill(
                &pipe,
                &prompt_i32[base as usize..end as usize],
                base,
                [s.seed, k as u32],
                None,
            )
            .await?;

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

    let mut speech = Speech::new(tags, prompt.reasoning);
    let first_done = matches!(speech.feed(g0 as u32)?, ControlFlow::Break(Stop::Turn));

    // Decode: the whole turn in one device-carried loop, or with the
    // reasoning open, only up to where its budget runs out, or sooner if the
    // reasoning loops. A loop that is stopped early leaves the fires it ran
    // ahead with in flight, and on a hybrid model those have already folded
    // tokens nobody will keep into the recurrent state; a loop that runs its
    // exact count, or settles (see `Stop::Settle`), leaves the cache holding
    // precisely the tokens it returned, which is what lets the program pick
    // up where it left off.
    let budget = if first_done {
        0
    } else {
        cut.as_ref().map_or(s.max_tokens, |cut| cut.at) - 1
    };
    let decode_rng = [s.seed ^ 0x9e37, 1u32];
    let (last, fired) = passes
        .decode(&pipe, n, g0, budget, decode_rng, &mut speech)
        .await?;
    // Any fire still in flight after an early stop is left untaken; close
    // releases the scheduler wait-set, reclaims them, and rejects further
    // submissions.
    pipe.close();

    // The cut, reached with the turn still going: the reasoning used its
    // whole budget or fell into a loop, or the answer had started and goes
    // on past it. The cache holds every token up to the last one sampled,
    // which still needs its forward pass; behind it go the tokens that end
    // the block if it is still open. The rng picks up both streams where the
    // first pipeline left them. The answer's histogram, for the penalty, is
    // built on the host: the answer so far when the model closed the block
    // itself, nothing when the program just did.
    if let Some(cut) = cut.filter(|_| !speech.ended) {
        let mut feed = vec![last];
        if speech.reasoning {
            feed.extend(cut.closing.iter().map(|&t| t as i32));
            speech.close_reasoning(&cut.closing)?;
        }
        let remaining = s
            .max_tokens
            .saturating_sub(speech.generated)
            .max(MIN_ANSWER.saturating_sub(speech.answered));
        if remaining > 0 {
            let pipe = Pipeline::new();
            let start = n + fired as u32;
            let seen = passes.seen(&speech);
            let g = passes
                .prefill(&pipe, &feed, start, [s.seed, spans.len() as u32], seen)
                .await?;
            if !matches!(speech.feed(g as u32)?, ControlFlow::Break(Stop::Turn)) {
                let rng = [decode_rng[0], decode_rng[1] + fired as u32];
                passes
                    .decode(
                        &pipe,
                        start + feed.len() as u32,
                        g,
                        remaining - 1,
                        rng,
                        &mut speech,
                    )
                    .await?;
            }
            pipe.close();
        }
    }

    if speech.thought_looped {
        notes.push("stopped a repetition loop in the reasoning".into());
    }
    if speech.looped_at.is_some() {
        notes.push("stopped a repetition loop".into());
    }
    let text = speech.finish().unwrap_or_else(|| {
        notes.push("the model ended the turn inside its reasoning, before any answer".into());
        String::new()
    });
    Ok(Turn {
        text,
        reasoning: speech.thought.trim().to_string(),
        thought_tokens: speech.thought_tokens,
        prompt_tokens: n,
        reused: covered,
        new_prefill: n - covered,
        generated: speech.generated,
        resumed: covered > 0,
        note: notes.join("; "),
    })
}

/// Where the first decode loop of a turn whose prompt opened the reasoning
/// stops, and the tokens that end the block if it is still open there.
struct Cut {
    at: usize,
    closing: Vec<u32>,
}

/// What every forward pass of one turn binds: its working sets, the page
/// pool they address, and how it samples.
struct Passes<'a> {
    ws: &'a WorkingSet,
    rs: &'a [RsWorkingSet],
    max_pages: u32,
    page_size: u32,
    vocab: u32,
    sampling: Sampling,
}

impl Passes<'_> {
    /// The histogram the repetition penalty reads, for a pass that samples
    /// after the tokens `speech` has taken so far: the answer's, or `None`
    /// when the penalty is off or the pass samples reasoning, which is not
    /// penalized.
    ///
    /// A pass carries it from the moment its first token is an answer
    /// token. The one answer that goes unpenalized for a while is the rare
    /// one the model starts by closing the prompt's block itself, inside the
    /// loop that decodes the reasoning: from there to the cut it relies on
    /// the loop guard alone, and the pass after the cut, which samples
    /// answer from its first token, picks the penalty up with the answer so
    /// far. Carrying an empty histogram through every reasoning fire instead
    /// would cost the penalty's arithmetic on a whole thinking budget to
    /// cover a case the phone's model practically never reaches.
    fn seen(&self, speech: &Speech) -> Option<Vec<f32>> {
        (self.sampling.repetition_penalty != 1.0 && !speech.in_prompt_block)
            .then(|| speech.seen(self.vocab))
    }

    /// One prefill pass: writes `tokens` into the cache at positions
    /// `base..`, advancing the recurrent state over them on a hybrid model,
    /// and returns the token sampled after the last of them. `rng` is the
    /// `[key, ctr]` the pass draws from, and `seen`, with the repetition
    /// penalty on, the histogram it reads (see `Passes::seen`).
    async fn prefill(
        &self,
        pipe: &Pipeline,
        tokens: &[i32],
        base: u32,
        rng: [u32; 2],
        seen: Option<Vec<f32>>,
    ) -> Result<i32> {
        let (page_size, vocab, s) = (self.page_size, self.vocab, self.sampling);
        let end = base + tokens.len() as u32;
        let toks = Channel::from(tokens).named("toks_p");
        let embed_indptr = Channel::from([0u32, end - base]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len = Channel::from([end]).named("kv_len_p");
        let pages = Channel::from_iter(0..self.max_pages).named("pages_p");
        // The page CSR is the source of truth for the KV length on the wire:
        // the count must track `kv_len`, never the pool size.
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let rng = Channel::from(rng).named("rng_p");
        let seen = seen.map(|counts| Channel::from(counts).named("seen_p"));
        let sampled = Channel::new([1], dtype::i32).named("g0");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.attention(
            Some(KvBinding {
                working_set: self.ws,
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
            self.rs,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
        .with_context(|| format!("bind prefill chunk @{base}"))?;
        fwd.epilogue(move || {
            let r = rng.take();
            let counts = seen.map(|seen| seen.take());
            sampled.put(reshape(sample(vocab, s, &r, counts.as_ref()), [1]));
        });
        fwd.submit(pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        sampled
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill drain @{base}"))
    }

    /// Decodes up to `budget` tokens in one pass, device loop-carried, after
    /// `first`, the token sampled for position `start` and not yet in the
    /// cache. The epilogue feeds each sampled token straight back into the
    /// channel `embed` reads and advances the geometry, so the host only
    /// drains the mirror in `out`. When the pass samples answer and the
    /// repetition penalty is on, the histogram it reads starts as the answer
    /// so far, `first` included, and the epilogue carries it the same way,
    /// one more count per token. Every token goes to `speech`, which can end
    /// the loop early.
    /// Returns the last token sampled, which, like `first`, is not in the
    /// cache, and how many fires ran, which is how many tokens the cache
    /// gained.
    async fn decode(
        &self,
        pipe: &Pipeline,
        start: u32,
        first: i32,
        budget: usize,
        rng: [u32; 2],
        speech: &mut Speech,
    ) -> Result<(i32, usize)> {
        if budget == 0 {
            return Ok((first, 0));
        }
        let (page_size, vocab, s) = (self.page_size, self.vocab, self.sampling);
        let tok_in = Channel::from([first]).named("tok_in");
        let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
        let positions = Channel::from([start]).named("positions");
        let pages = Channel::from_iter(0..self.max_pages).named("pages");
        let page_indptr =
            Channel::from([0u32, (start + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([start / page_size]).named("w_slot");
        let w_off = Channel::from([start % page_size]).named("w_off");
        let kv_len = Channel::from([start + 1]).named("kv_len");
        let rng = Channel::from(rng).named("rng");
        let seen = self
            .seen(speech)
            .map(|counts| Channel::from(counts).named("seen"));
        let out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &embed_indptr)?;
        fwd.attention(
            Some(KvBinding {
                working_set: self.ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (start / page_size)..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            self.rs,
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
            let counts = seen.map(|seen| seen.take());
            let sampled = sample(vocab, s, &r, counts.as_ref());
            let next_counts = counts.map(|counts| scatter_add(&counts, &sampled, 1.0f32));
            let token = reshape(&sampled, [1]);
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
            if let (Some(seen), Some(next)) = (seen, next_counts) {
                seen.put(&next);
            }
            out.put(&token);
        });

        let mut last = first;
        let fired = drive(pipe, &fwd, budget, async || {
            let t = out.take_host::<Vec<i32>>().await?;
            last = *t.first().unwrap_or(&0);
            speech.feed(last as u32)
        })
        .await?;
        Ok((last, fired))
    }
}

/// Why the host ends a decode loop before its count.
enum Stop {
    /// The turn is over. Fires still in flight are left untaken; nothing
    /// they compute is used.
    Turn,
    /// The loop must end, but the turn goes on from where it ends: no fire
    /// is submitted after this one, and every fire already in flight is
    /// still taken, so the loop ends with the cache, recurrent state
    /// included, holding exactly the tokens it returned.
    Settle,
}

/// The library's `run_ahead`, with the same pacing, plus `Stop::Settle`.
/// `run_ahead` ends on any `Break` at once, with fires still in flight, and
/// on a hybrid model each of those has already folded its token into the
/// recurrent state, which cannot be rewound. A turn that only ends does not
/// care; a reasoning block that ends early does, because the answer is
/// decoded on top of that state. Settling instead takes the in-flight fires
/// too and hands their tokens to `on_token` like any other, so the caller
/// knows exactly what the cache holds. Returns how many fires were taken.
async fn drive(
    pipe: &Pipeline,
    fwd: &ForwardPass,
    budget: usize,
    mut on_token: impl AsyncFnMut() -> Result<ControlFlow<Stop>>,
) -> Result<usize> {
    // A recurrent state holds one slot per posted frame, so a hybrid model
    // posts one fire at a time; the window is how many fires may be out.
    let per_frame = if model::rs_state_size() > 0 {
        1
    } else {
        frame_size()
    };
    let window = (model::run_ahead_window() as usize / per_frame).max(1);
    let slots = vec![Some(fwd); per_frame];
    let mut submitted = 0usize;
    let submit = |submitted: &mut usize| -> Result<()> {
        let live = per_frame.min(budget - *submitted);
        submit_frame(pipe, &slots[..live])?;
        *submitted += live;
        // End of stream is a pipeline event: the scheduler stops waiting on
        // this one once it is closed, and what is in flight still runs.
        if *submitted == budget {
            pipe.close();
        }
        Ok(())
    };
    for _ in 0..window {
        if submitted == budget {
            break;
        }
        submit(&mut submitted)?;
    }
    let mut taken = 0usize;
    let mut settling = false;
    while taken < submitted {
        let flow = on_token().await?;
        taken += 1;
        match flow {
            ControlFlow::Break(Stop::Turn) => {
                pipe.close();
                return Ok(taken);
            }
            ControlFlow::Break(Stop::Settle) if !settling => {
                settling = true;
                pipe.close();
            }
            _ => {}
        }
        if !settling && submitted < budget && submitted - taken <= (window - 1) * per_frame {
            submit(&mut submitted)?;
        }
    }
    Ok(taken)
}

/// Drops both halves of a published boundary from the index. Absent halves
/// are not an error: a turn retires every boundary of its transcript, and
/// most were never published.
fn retire(key: &[u8]) -> std::result::Result<(), String> {
    WorkingSet::remove_index(key)?;
    RsWorkingSet::remove_index(key)?;
    Ok(())
}

/// The voiced side of a turn: detokenizes, separates reasoning from speech,
/// and streams each to its channel.
struct Speech {
    decoder: chat::Decoder,
    stop: Vec<u32>,
    stripper: ReasoningStripper,
    tags: Option<ThinkTags>,
    raw: String,
    spoken: String,
    /// The reasoning streamed so far.
    thought: String,
    generated: usize,
    /// Of `generated`, the tokens inside reasoning blocks, and the tokens
    /// since the last one closed.
    thought_tokens: usize,
    answered: usize,
    /// The next token lands inside a reasoning block.
    reasoning: bool,
    /// Still inside the block the prompt opened: neither the model nor the
    /// program has closed it, so everything generated so far is reasoning.
    in_prompt_block: bool,
    /// A stop token, the decoder's end of turn, or a repetition loop outside
    /// the block the prompt opened ended the reply.
    ended: bool,
    /// The stretch of the reply the loop guard looks back over, and in the
    /// answer the repetition penalty too.
    stretch: Stretch,
    /// The reasoning fell into a loop, and the program is closing it.
    thought_looped: bool,
    /// Where in `raw` the repeat that ended the reply begins.
    looped_at: Option<usize>,
}

impl Speech {
    /// `reasoning` when the prompt opened a reasoning block, so the stream
    /// starts inside it.
    fn new(tags: Option<ThinkTags>, reasoning: bool) -> Self {
        Self {
            decoder: chat::Decoder::new(),
            stop: chat::stop_tokens(),
            stripper: ReasoningStripper::new(reasoning),
            tags,
            raw: String::new(),
            spoken: String::new(),
            thought: String::new(),
            generated: 0,
            thought_tokens: 0,
            answered: 0,
            reasoning,
            in_prompt_block: reasoning,
            ended: false,
            stretch: Stretch::default(),
            thought_looped: false,
            looped_at: None,
        }
    }

    /// One sampled token; `Break` says how the decode loop should end.
    fn feed(&mut self, token: u32) -> Result<ControlFlow<Stop>> {
        if self.stop.contains(&token) {
            self.ended = true;
            return Ok(ControlFlow::Break(Stop::Turn));
        }
        self.generated += 1;
        let repeat = self.count(token);
        match self.decoder.feed(&[token])? {
            chat::Event::Delta(delta) => {
                self.raw.push_str(&delta);
                self.route(&delta);
            }
            chat::Event::Done(text) => {
                self.raw = text;
                self.ended = true;
                return Ok(ControlFlow::Break(Stop::Turn));
            }
            chat::Event::Interrupt(_) => {}
        }
        Ok(match repeat {
            Some(at) => ControlFlow::Break(self.repeated(at)),
            None => ControlFlow::Continue(()),
        })
    }

    /// Follows the reasoning block token by token. The text-level stripper
    /// decides what is spoken; this decides what the thinking budget has
    /// spent, which only the tokens can say exactly, and which stretch the
    /// token belongs to. Returns where a repeat begins, if this token
    /// completed one.
    fn count(&mut self, token: u32) -> Option<usize> {
        match self.tags {
            // The answer starts here, and with it a stretch of its own.
            Some(tags) if token == tags.close && self.in_prompt_block => {
                self.reasoning = false;
                self.in_prompt_block = false;
                self.stretch = Stretch::default();
                return None;
            }
            // A block the model opened inside the answer is part of it.
            Some(tags) if token == tags.close => self.reasoning = false,
            Some(tags) if token == tags.open => self.reasoning = true,
            _ if self.reasoning => self.thought_tokens += 1,
            _ => self.answered += 1,
        }
        // The token's text, once decoded, starts where `raw` ends now.
        self.stretch.push(token, self.raw.len())
    }

    /// How a repetition loop that begins at byte `at` of `raw` ends the
    /// decode loop.
    fn repeated(&mut self, at: usize) -> Stop {
        if self.in_prompt_block {
            // The program knows how to close this block and have the model
            // answer, as it does when the thinking budget runs out.
            self.thought_looped = true;
            return Stop::Settle;
        }
        self.ended = true;
        self.looped_at.get_or_insert(at);
        Stop::Turn
    }

    /// How often each token of a `vocab`-sized vocabulary occurs in the
    /// current stretch, which is what a device-carried histogram counted
    /// over the same tokens would hold.
    fn seen(&self, vocab: u32) -> Vec<f32> {
        let mut counts = vec![0.0f32; vocab as usize];
        for &token in &self.stretch.tokens {
            if let Some(count) = counts.get_mut(token as usize) {
                *count += 1.0;
            }
        }
        counts
    }

    /// Takes in the tokens the program prefilled to end the reasoning block,
    /// so the transcript and the stripper see the block close exactly as if
    /// the model had closed it. They are not generated and not counted, and
    /// the answer starts a stretch of its own.
    fn close_reasoning(&mut self, tokens: &[u32]) -> Result<()> {
        self.reasoning = false;
        self.in_prompt_block = false;
        self.stretch = Stretch::default();
        if let chat::Event::Delta(delta) = self.decoder.feed(tokens)? {
            self.raw.push_str(&delta);
            self.route(&delta);
        }
        Ok(())
    }

    fn route(&mut self, delta: &str) {
        let split = self.stripper.process(delta);
        self.think(&split.reasoning);
        self.say(&split.speech);
    }

    /// Reasoning goes out as session messages, never on stdout: whatever
    /// reads stdout may be feeding a speech synthesizer.
    fn think(&mut self, text: &str) {
        // The template opens the block with a newline; a client showing the
        // reasoning should not start it with a blank line.
        let text = if self.thought.is_empty() {
            text.trim_start()
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        self.thought.push_str(text);
        session::send(text);
    }

    fn say(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.spoken.push_str(text);
        // Flushed per delta so the synthesizer starts on sentence one
        // instead of waiting for the turn to finish.
        print!("{text}");
        let _ = io::stdout().flush();
    }

    /// The whole spoken reply, once the turn is over, or `None` when the
    /// model ended the turn inside the block the prompt opened, so it never
    /// answered.
    fn finish(&mut self) -> Option<String> {
        // Release the holdback. Without this the last few characters of
        // every reply, held back in case they were the start of a `<think>`
        // tag, are never streamed, so the synthesizer clips the final word.
        // After a loop they are not spoken: the holdback is a few bytes and
        // the repeat that ended the reply is `LOOP_WINDOW` tokens, so they
        // belong to the repeat.
        let tail = self.stripper.flush();
        self.think(&tail.reasoning);
        if self.looped_at.is_none() {
            self.say(&tail.speech);
        }
        // The prompt's opening tag is not in `raw`, so the fallback below
        // would find no block at all and hand the reasoning back as the
        // reply, to be shown, spoken, and replayed as the assistant's turn.
        if self.in_prompt_block {
            return None;
        }
        // The repeat has been streamed already; the reply returned, which a
        // client shows and replays as the assistant's turn, stops before it.
        if let Some(at) = self.looped_at {
            let kept = speakable_text(&self.raw[..at]);
            return Some(through_last_sentence(&kept).to_string());
        }
        // `Done` hands back the whole turn at once, so re-derive the spoken
        // text from the raw transcript when the stream ended that way.
        let recovered = speakable_text(&self.raw);
        if recovered.trim() != self.spoken.trim() {
            return Some(recovered.trim().to_string());
        }
        Some(self.spoken.trim().to_string())
    }
}

/// A stretch of the reply: the reasoning block the prompt opened, or the
/// answer after it. The loop guard looks for a repeat among these tokens,
/// and in the answer the repetition penalty counts them. The answer is a
/// stretch apart from the reasoning because it is expected to restate what
/// the reasoning settled on, place names and numbers included, and that is
/// not a loop.
#[derive(Default)]
struct Stretch {
    tokens: Vec<u32>,
    /// Where in the raw transcript each token's text begins.
    starts: Vec<usize>,
    /// Every run of `LOOP_WINDOW` tokens the stretch has had so far. A set
    /// lookup per token keeps the guard linear in the reply's length.
    windows: HashSet<[u32; LOOP_WINDOW]>,
}

impl Stretch {
    /// Adds `token`, whose text begins at byte `at` of the raw transcript.
    /// When the newest `LOOP_WINDOW` tokens already occurred earlier in the
    /// stretch, the model is repeating itself, and this returns where the
    /// repeat begins. That is the first moment any run of that length
    /// repeats, so nothing before the newest window is a repeat yet.
    /// Overlapping occurrences count, which is what catches a loop shorter
    /// than the window.
    fn push(&mut self, token: u32, at: usize) -> Option<usize> {
        self.tokens.push(token);
        self.starts.push(at);
        let first = self.tokens.len().checked_sub(LOOP_WINDOW)?;
        let window: [u32; LOOP_WINDOW] = self.tokens[first..].try_into().ok()?;
        (!self.windows.insert(window)).then(|| self.starts[first])
    }
}

/// `text` up to the end of its last complete sentence: its final `.`, `!`,
/// `?` or `…`, with any closing quote or bracket after it, followed by
/// whitespace or the end. A decimal point is followed by a digit, so it is
/// never taken for one. Text with no complete sentence is returned whole,
/// trimmed: a fragment says more than silence.
fn through_last_sentence(text: &str) -> &str {
    let text = text.trim();
    let mut end = None;
    for (i, c) in text.char_indices() {
        if !matches!(c, '.' | '!' | '?' | '…') {
            continue;
        }
        let rest = text[i + c.len_utf8()..].trim_start_matches(['"', '\'', ')', ']', '”', '’']);
        if rest.is_empty() || rest.starts_with(char::is_whitespace) {
            end = Some(text.len() - rest.len());
        }
    }
    end.map_or(text, |end| &text[..end])
}

/// The part of a raw assistant turn that should be said out loud.
///
/// With the closed block prefilled there is usually nothing to strip. The
/// two fallbacks matter anyway: a model that emits its own complete block
/// (everything after the last `</think>`), and the case that actually bites,
/// one that opens a block itself and never closes it, where the reply is the
/// block's contents and dropping it would leave silence. A block the prompt
/// opened is never this case: its tag is not in the generated text, and
/// `Speech::finish` settles it before calling this.
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

/// What a stretch of the raw stream splits into.
#[derive(Default)]
struct Split {
    reasoning: String,
    speech: String,
}

/// Separates `<think>...</think>` from the speakable rest of a token stream
/// without ever emitting a partial tag: text is held back until it is certain
/// no tag straddles the chunk boundary.
struct ReasoningStripper {
    in_think: bool,
    pending: String,
    /// Whitespace that opens the reply, or follows a reasoning block, is the
    /// template's padding, not speech; a synthesizer fed it first may pause.
    leading: bool,
}

impl ReasoningStripper {
    /// `in_think` when the prompt opened the block, so the stream starts as
    /// reasoning.
    fn new(in_think: bool) -> Self {
        Self {
            in_think,
            pending: String::new(),
            leading: true,
        }
    }

    fn process(&mut self, delta: &str) -> Split {
        self.pending.push_str(delta);
        let mut out = Split::default();
        loop {
            if self.in_think {
                if let Some(idx) = self.pending.find("</think>") {
                    out.reasoning.push_str(&self.pending[..idx]);
                    self.pending = self.pending.split_off(idx + "</think>".len());
                    self.in_think = false;
                    self.leading = true;
                    continue;
                }
                // Keep the tail that could be a partial "</think>".
                let cut = floor_boundary(&self.pending, self.pending.len().saturating_sub(7));
                if cut > 0 {
                    out.reasoning.extend(self.pending.drain(..cut));
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
                out.speech.push_str(&self.pending[..idx]);
                self.pending = self.pending.split_off(idx + "<think>".len());
                self.in_think = true;
                continue;
            }
            // Same idea for a partial "<think>".
            let safe = floor_boundary(&self.pending, self.pending.len().saturating_sub(6));
            if safe > 0 {
                out.speech.extend(self.pending.drain(..safe));
            }
            break;
        }
        out
    }

    /// Everything still held back at the end of the turn.
    ///
    /// Inside an unterminated block the rest is reasoning as far as this
    /// stripper can tell, so none of it is speech; when the model opened
    /// that block itself, `speakable_text` recovers it from the raw
    /// transcript as the reply.
    fn flush(&mut self) -> Split {
        let pending = std::mem::take(&mut self.pending);
        if self.in_think {
            Split {
                reasoning: pending,
                speech: String::new(),
            }
        } else {
            Split {
                reasoning: String::new(),
                speech: pending,
            }
        }
    }
}
