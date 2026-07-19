//! Tokenizer-bound grammar compilation and caching.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use lru::LruCache;
use pie_tokenizer::Tokenizer;

use crate::compiled_grammar::CompiledGrammar;
use crate::grammar::{Expr, Grammar};
use crate::json_schema::{JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar};
use crate::regex::regex_to_grammar;

const DEFAULT_CACHE_CAPACITY: usize = 64;

#[derive(Clone, Debug)]
pub struct GrammarLimits {
    pub max_source_bytes: usize,
    pub max_rules: usize,
    pub max_expressions: usize,
    pub max_repetition: u32,
    pub max_nfa_states_per_rule: usize,
    pub max_dfa_states_per_rule: usize,
    pub max_total_dfa_states: usize,
    pub max_token_mask_bytes: usize,
    pub max_compile_duration: Duration,
}

impl Default for GrammarLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 1024 * 1024,
            max_rules: 4096,
            max_expressions: 65_536,
            max_repetition: 4096,
            max_nfa_states_per_rule: 65_535,
            max_dfa_states_per_rule: 65_535,
            max_total_dfa_states: 262_144,
            max_token_mask_bytes: 256 * 1024 * 1024,
            max_compile_duration: Duration::from_secs(5),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum CacheKey {
    Ebnf {
        source: String,
        root_rule: String,
    },
    Regex(String),
    JsonSchema {
        source: String,
        options: JsonSchemaOptions,
    },
    BuiltinJson,
}

/// Single-tokenizer grammar compiler with a typed LRU cache.
///
/// Cache hits bypass frontend parsing as well as DFA and token-mask compilation.
/// Misses compile outside the cache lock so an unrelated cold grammar does not
/// block cache hits from other requests.
pub struct GrammarCompiler {
    tokenizer: Arc<Tokenizer>,
    limits: GrammarLimits,
    cache: Mutex<LruCache<CacheKey, Arc<CompiledGrammar>>>,
}

impl GrammarCompiler {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self::with_limits(tokenizer, GrammarLimits::default())
    }

    pub fn with_limits(tokenizer: Arc<Tokenizer>, limits: GrammarLimits) -> Self {
        Self {
            tokenizer,
            limits,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_CACHE_CAPACITY).unwrap(),
            )),
        }
    }

    pub fn compile_ebnf(&self, source: &str, root_rule: &str) -> Result<Arc<CompiledGrammar>> {
        self.validate_source(source)?;
        self.get_or_compile(
            CacheKey::Ebnf {
                source: source.to_owned(),
                root_rule: root_rule.to_owned(),
            },
            || Grammar::from_ebnf(source, root_rule),
        )
    }

    pub fn compile_regex(&self, pattern: &str) -> Result<Arc<CompiledGrammar>> {
        self.validate_source(pattern)?;
        self.get_or_compile(CacheKey::Regex(pattern.to_owned()), || {
            regex_to_grammar(pattern)
        })
    }

    pub fn compile_json_schema(
        &self,
        schema: &str,
        options: &JsonSchemaOptions,
    ) -> Result<Arc<CompiledGrammar>> {
        self.validate_source(schema)?;
        self.get_or_compile(
            CacheKey::JsonSchema {
                source: schema.to_owned(),
                options: options.clone(),
            },
            || json_schema_to_grammar(schema, options),
        )
    }

    pub fn compile_builtin_json(&self) -> Result<Arc<CompiledGrammar>> {
        self.get_or_compile(CacheKey::BuiltinJson, builtin_json_grammar)
    }

    fn get_or_compile(
        &self,
        key: CacheKey,
        build_grammar: impl FnOnce() -> Result<Grammar>,
    ) -> Result<Arc<CompiledGrammar>> {
        self.get_or_insert(key, || {
            let started = Instant::now();
            let grammar = build_grammar()?;
            self.validate_grammar(&grammar)?;
            Ok(Arc::new(CompiledGrammar::try_new(
                &grammar,
                &self.tokenizer,
                &self.limits,
                started,
            )?))
        })
    }

    fn get_or_insert(
        &self,
        key: CacheKey,
        build_compiled: impl FnOnce() -> Result<Arc<CompiledGrammar>>,
    ) -> Result<Arc<CompiledGrammar>> {
        if let Some(compiled) = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(compiled);
        }

        let compiled = build_compiled()?;

        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = cache.get(&key) {
            return Ok(existing.clone());
        }
        cache.put(key, compiled.clone());
        Ok(compiled)
    }

    fn validate_source(&self, source: &str) -> Result<()> {
        if source.len() > self.limits.max_source_bytes {
            bail!(
                "grammar source is {} bytes; limit is {}",
                source.len(),
                self.limits.max_source_bytes
            );
        }
        Ok(())
    }

    fn validate_grammar(&self, grammar: &Grammar) -> Result<()> {
        if grammar.rules.len() > self.limits.max_rules {
            bail!(
                "grammar has {} rules; limit is {}",
                grammar.rules.len(),
                self.limits.max_rules
            );
        }
        if grammar.exprs.len() > self.limits.max_expressions {
            bail!(
                "grammar has {} expressions; limit is {}",
                grammar.exprs.len(),
                self.limits.max_expressions
            );
        }
        for expression in &grammar.exprs {
            if let Expr::Repeat { min, max, .. } = expression {
                let largest = max.unwrap_or(*min);
                if *min > self.limits.max_repetition || largest > self.limits.max_repetition {
                    bail!(
                        "grammar repetition exceeds limit {}",
                        self.limits.max_repetition
                    );
                }
            }
        }
        Ok(())
    }
}
