//! pie:core/inference - ForwardPass + sampler programs; pie:core/tensor -
//! Tensor + Program resources.

//! `pie:inferlet/grammar` — the `Grammar` + `Matcher` resources (grammar-mask
//! compilation and stateful matching).

use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;
use anyhow::Result;
use pie_grammar::compiled_grammar::CompiledGrammar;
use pie_grammar::compiler::GrammarCompiler;
use pie_grammar::json_schema::JsonSchemaOptions;
use pie_grammar::matcher::GrammarMatcher;
use std::sync::{Arc, OnceLock};
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::inferlet::grammar::Host for ProcessCtx {}

/// Aggregate interface-level `Host` for `pie:core/working-set`, required by
/// the generated `HostKvWorkingSet` (charlie) + `HostRsWorkingSet` (delta)
/// resource impls. echo owns this (central bindgen) since it spans both lanes.
impl pie::inferlet::working_set::Host for ProcessCtx {}

pub struct Grammar {
    pub(crate) compiled: Arc<CompiledGrammar>,
}

impl std::fmt::Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grammar").finish()
    }
}

pub(crate) fn grammar_compiler() -> &'static GrammarCompiler {
    static COMPILER: OnceLock<GrammarCompiler> = OnceLock::new();
    COMPILER.get_or_init(|| GrammarCompiler::new(crate::model::model().tokenizer().clone()))
}

impl pie::inferlet::grammar::HostGrammar for ProcessCtx {
    async fn from_json_schema(
        &mut self,
        schema: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        match grammar_compiler().compile_json_schema(&schema, &JsonSchemaOptions::default()) {
            Ok(compiled) => {
                let grammar = Grammar { compiled };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn json(&mut self) -> Result<Resource<Grammar>> {
        let grammar = Grammar {
            compiled: grammar_compiler().compile_builtin_json()?,
        };
        Ok(self.ctx().table.push(grammar)?)
    }

    async fn from_regex(&mut self, pattern: String) -> Result<Result<Resource<Grammar>, String>> {
        match grammar_compiler().compile_regex(&pattern) {
            Ok(compiled) => {
                let grammar = Grammar { compiled };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn from_ebnf(&mut self, ebnf: String) -> Result<Result<Resource<Grammar>, String>> {
        match grammar_compiler().compile_ebnf(&ebnf, "root") {
            Ok(compiled) => {
                let grammar = Grammar { compiled };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn to_string(&mut self, this: Resource<Grammar>) -> Result<String> {
        let g = self.ctx().table.get(&this)?;
        Ok(g.compiled.grammar().to_string())
    }

    async fn drop(&mut self, this: Resource<Grammar>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

// =============================================================================
// Matcher resource
// =============================================================================

/// How many accepted tokens a matcher retains for `rollback`.
const MAX_ROLLBACK_TOKENS: usize = 64;

/// Stateful matcher that walks the grammar automaton, producing token masks.
pub struct Matcher {
    pub(crate) inner: GrammarMatcher,
}

impl std::fmt::Debug for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matcher").finish()
    }
}

impl pie::inferlet::grammar::HostMatcher for ProcessCtx {
    async fn new(&mut self, grammar: Resource<Grammar>) -> Result<Resource<Matcher>> {
        let model = crate::model::model();
        let stop_tokens = model.instruct().seal();
        let compiled = self.ctx().table.get(&grammar)?.compiled.clone();
        // Rollback history depth. Deep enough for the draft lengths that
        // speculative decoding and tree drafting actually use; a token of
        // history costs a `usize` plus a `u16`, so the bound is about
        // memory hygiene, not a real constraint.
        let inner = GrammarMatcher::with_compiled(compiled, stop_tokens, MAX_ROLLBACK_TOKENS);

        let matcher = Matcher { inner };
        Ok(self.ctx().table.push(matcher)?)
    }

    async fn accept_tokens(
        &mut self,
        this: Resource<Matcher>,
        token_ids: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        for &id in &token_ids {
            if !matcher.inner.accept_token(id) {
                return Ok(Err(format!("token {} rejected by grammar", id)));
            }
        }
        Ok(Ok(()))
    }

    async fn mask(&mut self, this: Resource<Matcher>) -> Result<Vec<u32>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        // The packed allowed-token bitmask (`[ceil(vocab/32)]` u32, bit 1 =
        // allowed) — the `mask-apply` (0x65) mask operand. Returned directly,
        // no BRLE round-trip.
        Ok(matcher.inner.fill_next_token_mask())
    }

    async fn is_terminated(&mut self, this: Resource<Matcher>) -> Result<bool> {
        let matcher = self.ctx().table.get(&this)?;
        Ok(matcher.inner.can_terminate())
    }

    async fn reset(&mut self, this: Resource<Matcher>) -> Result<()> {
        let matcher = self.ctx().table.get_mut(&this)?;
        matcher.inner.reset();
        Ok(())
    }

    async fn fork(&mut self, this: Resource<Matcher>) -> Result<Resource<Matcher>> {
        let inner = self.ctx().table.get(&this)?.inner.fork();
        Ok(self.ctx().table.push(Matcher { inner })?)
    }

    async fn rollback(&mut self, this: Resource<Matcher>, num_tokens: u32) -> Result<()> {
        let matcher = self.ctx().table.get_mut(&this)?;
        matcher.inner.rollback(num_tokens as usize);
        Ok(())
    }

    async fn rollback_capacity(&mut self, this: Resource<Matcher>) -> Result<u32> {
        let matcher = self.ctx().table.get(&this)?;
        Ok(matcher.inner.rollback_capacity() as u32)
    }

    async fn drop(&mut self, this: Resource<Matcher>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
