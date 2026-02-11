//! Grammar-guided LLM token generation.
//!
//! This constrains language model output to follow formal grammars
//! (EBNF, JSON Schema, regex), enabling structured generation with guaranteed
//! syntactic correctness.
//! 
//! The large portion of this code is a port of the xgrammar library to Rust. 
//! (@ingim: add credit to xgrammar in the README)
//! 
//! # Usage
//!
//! The typical flow is:
//!
//! 1. **Create a grammar** from EBNF, JSON Schema, or regex
//! 2. **Build a tokenizer** from your LLM's vocabulary
//! 3. **Create a matcher** (compiles grammar into per-rule DFAs + token masks)
//! 4. **Loop**: `fill_next_token_bitmask` → mask logits → sample → `accept_token`
//!
//! ```rust
//! use std::sync::Arc;
//! use pie::structured::bitmask;
//! use pie::structured::grammar::Grammar;
//! use pie::structured::matcher::GrammarMatcher;
//! use pie::structured::tokenizer::{TokenizerInfo, VocabType};
//!
//! // Create grammar from EBNF
//! let grammar = Arc::new(
//!     Grammar::from_ebnf(r#"root ::= "yes" | "no""#, "root").unwrap()
//! );
//!
//! // Build tokenizer from vocabulary
//! let vocab: Vec<String> = vec!["yes".into(), "no".into(), "maybe".into()];
//! let tokenizer = Arc::new(
//!     TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap()
//! );
//!
//! // Create matcher (compiles grammar for this tokenizer)
//! let mut matcher = GrammarMatcher::new(grammar, tokenizer, vec![], 10);
//!
//! // Generate next-token bitmask
//! let mut bm = vec![0u32; bitmask::bitmask_size(3)];
//! matcher.fill_next_token_bitmask(&mut bm);
//!
//! assert!(bitmask::get_bit(&bm, 0));  // "yes" allowed
//! assert!(bitmask::get_bit(&bm, 1));  // "no" allowed
//! assert!(!bitmask::get_bit(&bm, 2)); // "maybe" blocked
//!
//! // Accept a token
//! assert!(matcher.accept_token(0));
//! assert!(matcher.can_terminate());
//! ```
//!
//! # JSON Schema
//!
//! ```rust
//! use pie::structured::json_schema::{json_schema_to_grammar, JsonSchemaOptions};
//!
//! let grammar = json_schema_to_grammar(r#"{
//!     "type": "object",
//!     "properties": {
//!         "name": {"type": "string"},
//!         "age": {"type": "integer"}
//!     },
//!     "required": ["name", "age"],
//!     "additionalProperties": false
//! }"#, &JsonSchemaOptions::default()).unwrap();
//! ```
//!
//! # Regex
//!
//! ```rust
//! use pie::structured::regex::regex_to_grammar;
//!
//! let grammar = regex_to_grammar(r"[a-z]+@[a-z]+\.[a-z]{2,4}").unwrap();
//! ```
//!
//! # Sharing compiled grammars
//!
//! For batch inference, compile once and share across matchers:
//!
//! ```rust
//! use std::sync::Arc;
//! use pie::structured::compiled_grammar::CompiledGrammar;
//! use pie::structured::grammar::Grammar;
//! use pie::structured::matcher::GrammarMatcher;
//! use pie::structured::tokenizer::{TokenizerInfo, VocabType};
//!
//! let grammar = Arc::new(Grammar::from_ebnf(r#"root ::= [a-z]+"#, "root").unwrap());
//! let vocab: Vec<String> = (0..128u16).map(|b| String::from(b as u8 as char)).collect();
//! let tokenizer = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());
//!
//! // Compile once (builds DFAs + token masks)
//! let compiled = Arc::new(CompiledGrammar::new(&grammar, &tokenizer));
//!
//! // Create multiple matchers cheaply
//! let mut m1 = GrammarMatcher::with_compiled(compiled.clone(), tokenizer.clone(), vec![], 10);
//! let mut m2 = GrammarMatcher::with_compiled(compiled.clone(), tokenizer.clone(), vec![], 10);
//!
//! m1.accept_string("hello");
//! m2.accept_string("world");
//! ```
//!
//! # Modules
//!
//! - [`grammar`] -- Core grammar types and construction (EBNF, builder)
//! - [`json_schema`] -- JSON Schema to grammar conversion
//! - [`regex`] -- Regex to grammar conversion
//! - [`tokenizer`] -- Tokenizer vocabulary handling
//! - [`compiled_grammar`] -- Pre-compiled grammar with DFAs and token masks
//! - [`matcher`] -- Runtime matcher (accept tokens, generate bitmasks, rollback)
//! - [`bitmask`] -- Token bitmask utilities

pub mod bitmask;
pub mod compiled_grammar;
pub(crate) mod fsm;
pub mod grammar;
pub mod json_schema;
pub mod matcher;
pub mod regex;
pub mod tokenizer;
