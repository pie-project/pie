//! EVERY ROW CAN END A TURN, AND ITS DECODER AGREES ABOUT WHEN.
//!
//! `Instruct` is the third backend-shaped axis, and the only one whose
//! failures are invisible to a driver: a plan that binds the wrong tensor
//! refuses at load, but a chat template that ends a turn with a token the
//! decoder reads as ordinary text loads perfectly and generates fluent
//! prose that never stops.
//!
//! # Why an integration test
//!
//! Every family already tests its own chat surface, and the property this
//! file holds is one that per-family tests structurally cannot see.
//!
//! Olmo-3 had two of them. One asserted `seal()` returns the turn end AND
//! eos; the other asserted the chat decoder stops at the turn end. Both
//! passed, for two years, while the two disagreed about `<|endoftext|>` —
//! because each test checked its own half and neither was the place where
//! the halves meet. The meeting place is here.
//!
//! So this walks `catalog()` and asks the questions that only make sense
//! ACROSS a row's answers:
//!
//! - the tokens `seal()` calls terminal are the tokens `chat_decoder()`
//!   ends on,
//! - a row that can be told it has tools can be told a tool's result,
//! - and every row can end a turn at all.
//!
//! The lower bound on the row count is there because a walk of an empty
//! iterator passes every assertion in it.
//!
//! # What is out of reach from here
//!
//! A generation with a chat template and no catalog row is not reachable
//! from a walk of `catalog()` — `llama_2`'s `[INST]` template is the
//! current example, and breaking it on purpose leaves this file green.
//! Those templates are held by their own module tests until a row is
//! transcribed, at which point they arrive here for free.
#![cfg(feature = "chat")]

use model::catalog;
use model::instruct::{ChatEvent, Instruct};
use std::sync::Arc;
use tokenizer::Tokenizer;

/// Every special string any family in the catalog spells, in one vocab.
///
/// A family resolves its stop tokens with
/// `stop_strs.iter().filter_map(|s| tokenizer.token_to_id(s))`, so a vocab
/// missing a family's spelling does not fail — it yields an EMPTY `seal()`,
/// and every assertion about the contents of `seal()` then holds vacuously.
/// That is the failure mode this list exists to prevent, and it is why the
/// list is a union of all families rather than a per-family fixture: a
/// fixture built for one row silently disarms the test for the other 59.
///
/// The three near-identical spellings are deliberate and are not typos.
/// DeepSeek uses FULL-WIDTH brackets (`<｜` U+FF5C, not `<|`) and U+2581 for
/// its word joiner; gemma-4 spells its turn markers `<|turn>` / `<turn|>`
/// where gemma-2 and gemma-3 spell them `<start_of_turn>` /
/// `<end_of_turn>`; kimi ends on `[EOS]` where everyone else ends on
/// something angle-bracketed. Copying a lookalike here would reintroduce
/// exactly the empty-`seal()` vacuity described above.
const SPECIALS: &[&str] = &[
    // Turn framing.
    "<|im_start|>",
    "<|im_end|>",
    "<|im_middle|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|turn>",
    "<turn|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<｜User｜>",
    "<｜Assistant｜>",
    // Sequence framing.
    "<bos>",
    "<eos>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "</s>",
    "<s>",
    // Terminals.
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|EOT|>",
    "<｜end▁of▁sentence｜>",
    "[EOS]",
    "<|audio_eos|>",
    // Reasoning.
    "<think>",
    "</think>",
    "<thinking>",
    "</thinking>",
    // Tools.
    "<tool_response>",
    "</tool_response>",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁output▁begin｜>",
    "<｜tool▁output▁end｜>",
    "<｜tool▁outputs▁begin｜>",
    "<｜tool▁outputs▁end｜>",
    "<function_calls>",
    "</function_calls>",
    // Roles and whitespace, which several families encode as plain text.
    "system",
    "user",
    "assistant",
    "environment",
    "developer",
    "tool",
    "analysis",
    "final",
    "\n",
    "\n\n",
];

fn tokenizer() -> Arc<Tokenizer> {
    let vocab: Vec<String> = SPECIALS.iter().map(|s| (*s).to_string()).collect();
    Arc::new(Tokenizer::from_vocab(&vocab))
}

/// Rows whose chat surface carries no tool vocabulary.
///
/// Named rather than counted so that teaching one of these families to use
/// tools has to delete a line here, which is the moment someone reads the
/// pairing rule below and gives the row an `answer()` to go with its
/// `equip()`.
/// `embeddinggemma-300m` is listed by its whole id rather than by a family
/// prefix because it is not a chat model at all -- it embeds -- and it is
/// the one row here whose emptiness is a statement about the task rather
/// than about the template.
const NO_TOOL_VOCABULARY: &[&str] = &[
    "gemma-2",
    "gemma-3",
    "gemma-3n",
    "gemma-4",
    "kimi-k2",
    "kimi-k3",
    "nemotron-h",
    "olmo-2",
    "phi-3",
    "phi-4",
    "csm",
    "embeddinggemma-300m",
];

fn opts_out(id: &str) -> bool {
    NO_TOOL_VOCABULARY
        .iter()
        .any(|p| id == *p || id.starts_with(&format!("{p}-")))
}

fn rows() -> Vec<(&'static str, Arc<dyn Instruct>)> {
    let tok = tokenizer();
    let rows: Vec<_> = catalog::catalog()
        .iter()
        .map(|row| (row.id(), row.chat(tok.clone())))
        .collect();
    assert!(
        rows.len() >= 50,
        "the catalog reports {} rows, so this file is walking a list that \
         has lost most of itself and every assertion below is passing \
         vacuously",
        rows.len()
    );
    rows
}

/// A row that cannot end a turn generates until it hits the token budget.
///
/// The vocab above contains every spelling any family uses, so an empty
/// `seal()` here is the row's own answer and not a missing fixture.
#[test]
fn every_row_can_end_a_turn() {
    for (id, chat) in rows() {
        assert!(
            !chat.seal().is_empty(),
            "{id} publishes no stop token, so nothing it generates ever \
             terminates -- the sampler runs to the token budget and the \
             decoder never reaches Done"
        );
    }
}

/// The tokens `seal()` calls terminal are the tokens the decoder ends on.
///
/// These are two answers to one question, given by two different methods on
/// one trait, and the guest consults BOTH: it stops sampling on `seal()`
/// (through `chat::stop_tokens`) and reads its text out of `chat_decoder()`.
/// When the two disagree the guest stops at a token the decoder considers
/// ordinary, so `Done` never arrives and the accumulated text is dropped by
/// any inferlet written as a loop until `Done` — which is how the surface
/// reads and how the examples use it.
///
/// Olmo-3 was the row that disagreed: `seal()` published `<|im_end|>` and
/// `<|endoftext|>` while its decoder was constructed with the turn end
/// alone.
///
/// # What this cannot catch
///
/// Most families derive BOTH answers from one `stop_ids` field, and for
/// those two the assertion is unfalsifiable — adding a token to the list
/// adds it to both sides at once, and negative controls that do so are
/// silent here for that reason. What the test sees is precisely the case
/// where a family derives the two separately, which is the shape the bug
/// took and the only shape it can take.
#[test]
fn the_decoder_ends_on_every_token_the_seal_calls_terminal() {
    for (id, chat) in rows() {
        for token in chat.seal() {
            let mut decoder = chat.chat_decoder();
            assert!(
                matches!(decoder.feed(&[token]), ChatEvent::Done(_)),
                "{id} lists {token} in seal() but its chat decoder reads \
                 that token as ordinary text, so generation stops without \
                 ever producing Done"
            );
        }
    }
}

/// A row told it has tools can be told what a tool returned.
///
/// `equip()` opens the conversation by declaring the available tools and
/// `answer()` closes the loop by feeding a result back. A row with the
/// first and not the second can advertise a tool, emit a call for it, and
/// then have nowhere to put the reply — the conversation cannot proceed
/// past the first tool call, which is a dead end reached only at runtime
/// and only by a script that actually calls something.
///
/// Stated as an equivalence in both directions so that the opposite
/// mistake — a row that accepts results for tools it never advertised —
/// is also a failure.
#[test]
fn a_row_that_can_be_given_tools_can_be_given_their_results() {
    let tools = [r#"{"name":"now","description":"the time","parameters":{}}"#.to_string()];
    for (id, chat) in rows() {
        let equips = !chat.equip(&tools).is_empty();
        let answers = !chat.answer("now", "12:00").is_empty();
        assert_eq!(
            equips,
            answers,
            "{id} can{} be told it has tools but can{} be told what one \
             returned, so a conversation that calls a tool has nowhere to \
             put the result",
            if equips { "" } else { "not" },
            if answers { "" } else { "not" }
        );
    }
}

/// The rows with no tool vocabulary are the ones named above.
///
/// Without this the pairing test passes for a row that simply does nothing
/// in either direction, which is how a family that was MEANT to support
/// tools would slip through: both halves empty is a legal pairing.
#[test]
fn only_the_named_rows_have_no_tool_vocabulary() {
    let tools = [r#"{"name":"now","description":"the time","parameters":{}}"#.to_string()];
    for (id, chat) in rows() {
        let silent = chat.equip(&tools).is_empty();
        assert_eq!(
            silent,
            opts_out(id),
            "{id} {} tool vocabulary, which is not what NO_TOOL_VOCABULARY \
             in this file says -- add it to that list, or remove it, and \
             either way say why in the commit",
            if silent { "has no" } else { "has" }
        );
    }
}

/// Every row builds all three decoders.
///
/// `chat`, `reasoning` and `tool` are separate objects the guest may create
/// in any combination, and a family that has nothing to say for one of them
/// is expected to return the noop rather than to be absent. Constructing
/// all three for every row is the cheapest way to catch a `todo!()` or an
/// index panic in a decoder no per-family test happens to build.
#[test]
fn every_row_builds_all_three_decoders() {
    for (id, chat) in rows() {
        let mut c = chat.chat_decoder();
        let mut r = chat.reasoning_decoder();
        let mut t = chat.tool_decoder();
        // Feed each one a token that means nothing to any family, which is
        // the state a decoder is in before the model has said anything.
        let filler = chat.user("hi");
        let _ = c.feed(&filler);
        let _ = r.feed(&filler);
        let _ = t.feed(&filler);
        // Reaching here without a panic is the assertion; `reset` is part
        // of it because a decoder that accumulated state on the way in has
        // to be able to drop it, and the guest resets between turns.
        c.reset();
        r.reset();
        t.reset();
        let _ = id;
    }
}
