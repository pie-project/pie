//! Grammar-guided LLM token generation: constrains model output to a formal
//! grammar (EBNF, JSON Schema, regex) so structured generation is
//! syntactically correct by construction. A Rust rewrite derived in part from
//! the [XGrammar](https://github.com/mlc-ai/xgrammar) project.
//!
//! Flow: build a [`grammar::Grammar`], bind it to a tokenizer through
//! [`compiled_grammar::CompiledGrammar`] — compile once and share it across
//! matchers for batch inference — then loop `fill_next_token_bitmask` → mask
//! logits → sample → `accept_token` on a [`matcher::GrammarMatcher`].

pub mod bitmask;
pub mod brle;
pub mod compiled_grammar;
pub mod compiler;
mod frontend;
pub(crate) mod fsm;
/// The grammar itself: EBNF front end, rule set, and normalisation.
pub mod grammar;
pub mod json_schema;
pub mod matcher;
pub mod regex;
