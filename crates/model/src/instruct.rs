//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.
//!
//! Both halves are here. They were two files in two crates — the trait below
//! and the [`create`] registry at the bottom — because a generation crate
//! could not depend on the crate that dispatches to it, so the vocabulary had
//! to sit underneath both. One crate, one module.

/// A model-provided tool-call grammar in EBNF form.
pub struct ToolGrammar {
    pub source: String,
}
// The shared decoders, re-exported so `instruct::decoders` stays a valid
// path: it is what every generation's template imports, and the templates
// became crates without their imports needing to know.
pub use crate::shared::decoders;

/// Events emitted by the chat decoder.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Generated text chunk
    Delta(String),
    /// Special token encountered (token ID)
    Interrupt(u32),
    /// Generation complete (full accumulated text)
    Done(String),
}

/// Events emitted by the reasoning decoder.
#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    /// Reasoning block started
    Start,
    /// Reasoning text chunk
    Delta(String),
    /// Reasoning complete (full reasoning text)
    Complete(String),
}

/// Events emitted by the tool decoder.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// NOTHING TO REPORT YET — not "a tool call has started".
    ///
    /// There is no third variant, so this is what every decoder returns
    /// while it is still waiting: before an opening marker, between one
    /// and its close, after a block that named no function, and for
    /// every token of a model that has no tool protocol at all. A
    /// consumer that treats it as a detection sees one on every feed.
    Start,
    /// Complete tool call: (name, arguments-json)
    Call(String, String),
}

/// Classifies generated tokens into text deltas, interrupts, and done.
pub trait ChatDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent;
    fn reset(&mut self);
}

/// Detects reasoning/thinking blocks in the token stream.
pub trait ReasoningDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent;
    fn reset(&mut self);
}

/// Detects tool call blocks in the token stream.
pub trait ToolDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent;
    fn reset(&mut self);
}

/// Model-specific instruct implementation.
///
/// Each architecture provides its own impl with hardcoded tokens & logic.
/// The tokenizer is owned by the implementation to avoid redundant lookups.
pub trait Instruct: Send + Sync {
    fn system(&self, msg: &str) -> Vec<u32>;
    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.user(msg)
    }
    fn user(&self, msg: &str) -> Vec<u32>;
    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }
    fn assistant(&self, msg: &str) -> Vec<u32>;
    fn cue(&self) -> Vec<u32>;
    fn seal(&self) -> Vec<u32>;
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;
    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;
    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;
    /// Returns the parsed tool-call grammar that constrains generation to
    /// the architecture's tool-call format, given a list of tool schemas.
    /// Returns `None` if the architecture doesn't support constrained tool calling.
    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}

// ── The registry ─────────────────────────────────────────────────────

use std::sync::Arc;
use tokenizer::Tokenizer;

/// The chat template that speaks for a model.
///
/// # What this replaced, and why it is worth the churn
///
/// A `match` on `architectures[0]` with thirty arms and a `_ =>` that
/// answered ChatML. The fallback is the whole story: gemma-4's
/// `architectures[0]` is `Gemma4ForConditionalGeneration`, the table
/// knew a different stem, so gemma-4 fell through and got Qwen's
/// template. It then generated *fluently* — ending turns it was not
/// having with an `<|im_end|>` its vocabulary does not contain — because
/// a wrong chat template does not crash, it just makes a model sound
/// slightly deranged.
///
/// There is nowhere to put that now. A row answers
/// [`Variant::chat`](crate::catalog::Variant::chat), the method has no
/// default body, and a model with no row does not resolve to a template
/// at all — it fails to resolve, by name, at the door.
///
/// # Errors
///
/// [`Unmatched::NoSuchId`] when nothing in the catalog carries this id.
/// The error carries the nearest ids, because the overwhelmingly likely
/// cause is a typo in an `--as` override.
///
/// [`Unmatched::NoSuchId`]: crate::catalog::Unmatched::NoSuchId
pub fn create(
    id: &str,
    tokenizer: Arc<Tokenizer>,
) -> Result<Arc<dyn Instruct>, crate::catalog::Unmatched> {
    let row = crate::catalog::find(id).ok_or_else(|| crate::catalog::Unmatched::NoSuchId {
        id: id.to_string(),
        nearest: crate::catalog::nearest_ids(id, 3),
    })?;
    Ok(row.chat(tokenizer))
}

#[cfg(test)]
mod registry_tests {
    /// Every row answers, and no row has to fall through to answer.
    ///
    /// The old registry could not have this test: it took a string
    /// nothing enumerated, so "every input is handled" was true only in
    /// the sense that `_ =>` handles everything.
    ///
    /// It asks every row, through [`create`], because the name is the
    /// claim. It used to assert `!ids().is_empty()` — which is a test
    /// that the catalog exists, and would have passed with ninety rows
    /// answering and one panicking.
    #[test]
    fn every_row_names_a_template() {
        let tok = super::tests::scratch_tokenizer();
        let ids = crate::catalog::ids();
        assert!(!ids.is_empty(), "the catalog is empty");
        for id in ids {
            let instruct = super::create(id, tok.clone())
                .unwrap_or_else(|e| panic!("{id} is in the catalog but create refused it: {e:?}"));
            // A template that returns nothing for a user turn cannot be
            // prompted, and that is the failure the `_ => QwenInstruct`
            // fallback existed to hide.
            assert!(
                !instruct.user("hello").is_empty(),
                "{id}'s template encodes an empty user turn"
            );
            // A cue is what marks where the model starts speaking, so an
            // empty one is normally a template that cannot be prompted.
            // Mistral is the exception and not a defect: its turn is
            // `[INST] … [/INST]`, and the assistant simply continues after
            // the closing tag, so there is no third marker to emit. Named
            // rather than skipped, because the next empty cue is a bug.
            const SPEAKS_WITHOUT_A_CUE: &[&str] = &["mistral-7b-v0.3", "ministral-8b"];
            assert_eq!(
                instruct.cue().is_empty(),
                SPEAKS_WITHOUT_A_CUE.contains(&id),
                "{id}: a template has a generation cue iff its user turn does \
                 not already end where the model speaks"
            );
        }
    }

    /// An id nothing carries is a refusal with a suggestion, not a
    /// silently wrong template.
    ///
    /// Through [`create`], not `find`: the suggestion is built in
    /// `create`'s error arm, so testing `find` leaves the part a user
    /// reads unexecuted.
    #[test]
    fn an_unknown_id_is_refused_rather_than_guessed() {
        let tok = super::tests::scratch_tokenizer();
        let err = super::create("qwen3-0.6", tok)
            .err()
            .expect("a near-miss id must not resolve");
        let crate::catalog::Unmatched::NoSuchId { id, nearest } = err else {
            panic!("a typo'd id is NoSuchId, not {err:?}");
        };
        assert_eq!(id, "qwen3-0.6");
        assert!(
            nearest.contains(&"qwen3-0.6b"),
            "the id it is one character away from should be suggested, got {nearest:?}"
        );
    }

    /// The three defaulted methods on the trait, against the generations
    /// that override them.
    ///
    /// Both defaults are defined in terms of `user` and `system`, and two
    /// gemma generations override both — legitimately: gemma opens the
    /// rendered chat with a single `<bos>`, and has no system role, so its
    /// system message is folded into the first user turn. So the question
    /// cannot be "did you inherit the default" but what the default and
    /// the overrides must agree on.
    #[test]
    fn an_overridden_turn_still_says_what_the_default_says() {
        let tok = super::tests::scratch_tokenizer();
        for id in crate::catalog::ids() {
            let it = super::create(id, tok.clone()).expect("catalogued");

            // An override may only OPEN the turn — with a BOS, in every
            // case that overrides it today. It may not render a different
            // user turn, which is what a `first_user` that forgot the
            // suffix, or double-encoded the message, would do.
            let (first, plain) = (it.first_user("hello"), it.user("hello"));
            assert!(
                first.ends_with(&plain),
                "{id}: first_user must be the user turn, optionally opened by \
                 something; got {first:?} which does not end with {plain:?}"
            );

            // Folded or concatenated, both parts must survive and keep
            // their order. Gemma renders `{system}\n{user}` inside one
            // turn; everyone else renders two turns.
            //
            // Unless the generation has no system role at all, and says
            // so by rendering nothing for one — CSM, whose template
            // `raise_exception`s on any role that is not a speaker
            // number. Asked as a property rather than a list of ids,
            // because dropping the message is only correct for a
            // template that could not have carried it.
            let rendered = it.system_user("be brief", "hello");
            if it.system("be brief").is_empty() {
                assert_eq!(
                    rendered,
                    it.user("hello"),
                    "{id} renders no system turn, so system_user must be the \
                     user turn alone rather than a turn with the system text \
                     silently dropped into it"
                );
            } else {
                let joined = tok.decode(&rendered, false);
                let (sys, usr) = (joined.find("be brief"), joined.find("hello"));
                assert!(
                    sys.is_some() && usr.is_some() && sys < usr,
                    "{id}: system_user must render the system message before \
                     the user message; got {joined:?}"
                );
            }

            assert!(
                it.tool_call_grammar(&[]).is_none(),
                "{id} advertises a tool-call grammar for zero tools"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use tokenizer::Tokenizer;

    /// A vocabulary for asking a template what it emits.
    ///
    /// Every control token any generation in the catalog reaches for,
    /// plus the byte fallback [`Tokenizer::from_vocab`] adds — so
    /// arbitrary message text round-trips and a decoded prompt can be
    /// compared against the published template as a string.
    pub(crate) fn scratch_tokenizer() -> Arc<Tokenizer> {
        let vocab: Vec<String> = [
            "<|start|>",
            "<|message|>",
            "<|channel|>",
            "<|end|>",
            "<|endoftext|>",
            "<|return|>",
            "<|call|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|eom_id|>",
            "<bos>",
            "<eos>",
            "<pad>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "[AVAILABLE_TOOLS]",
            "[/AVAILABLE_TOOLS]",
            "[TOOL_CALLS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<|tool|>",
            "<think>",
            "</think>",
            "<|User|>",
            "<|Assistant|>",
            "<|begin▁of▁sentence|>",
            "<|end▁of▁sentence|>",
            "<|tool▁calls▁begin|>",
            "<|tool▁calls▁end|>",
            "<|tool▁call▁begin|>",
            "<|tool▁call▁end|>",
            "<|tool▁outputs▁begin|>",
            "<|tool▁outputs▁end|>",
            "<|tool▁output▁begin|>",
            "<|tool▁output▁end|>",
            "<|tool▁sep|>",
            "system",
            "developer",
            "user",
            "assistant",
            "analysis",
            "final",
            "commentary",
            "tool",
            "model",
            "\n",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        Arc::new(Tokenizer::from_vocab(&vocab))
    }
}
