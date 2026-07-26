//! Schema in, artifact out.
//!
//! The stages between the two - lowering, terminal extraction, the lexer, the
//! CFG, the LALR tables - each have a way to fail that the stage before cannot
//! see. Lowering does not know whether the grammar it produced is LALR(1), and
//! finding out costs a table construction. That makes the pipeline a search
//! rather than a function: lower the schema as precisely as it can be
//! expressed, try to build tables, and drop to a coarser lowering only when
//! the precise one has no parser.
//!
//! Every level is sound in the direction that matters for a mask. A coarser
//! lowering accepts more documents than the schema does, never fewer, so a
//! token the grammar allows may turn out to be invalid but a token the schema
//! allows is never masked away.

use std::fmt;

use anyhow::{Result, anyhow};
use gpugrammar_ir::json_schema::{JsonSchemaOptions, Precision, json_schema_to_grammar};
use gpugrammar_lex::lexicon::{DEFAULT_TERMINAL_BUDGET, extract_within, terminal_automata_within};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{build_lexer_within, group_vocabulary};
use gpugrammar_lr::cfg::{DEFAULT_PRODUCTION_BUDGET, flatten_within};
use gpugrammar_lr::tables::build;

use crate::{Artifact, emit};

/// What the compiler is willing to spend on one schema.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Terminals the lexicon may declare.
    pub terminals: u64,
    /// States the lexer DFA may reach.
    pub lexer_states: usize,
    /// Productions the flattened grammar may hold.
    pub productions: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            terminals: DEFAULT_TERMINAL_BUDGET,
            lexer_states: 20_000,
            productions: DEFAULT_PRODUCTION_BUDGET,
        }
    }
}

/// Why a schema did not compile.
///
/// Kept apart because they call for different answers: a budget says the
/// schema is too big for the limits it was given, a conflict says no LALR(1)
/// parser exists for it at any size, and a lowering failure says the schema
/// uses something the front end cannot express at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Failure {
    /// The schema uses a construct the front end does not lower.
    Lowering,
    /// Terminal extraction or the lexer DFA exceeded its budget.
    Lexer,
    /// The flattened grammar exceeded its production budget.
    Productions,
    /// The grammar is not LALR(1).
    Conflict,
    /// Emission failed.
    Emit,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Failure::Lowering => "schema uses a construct the front end cannot lower",
            Failure::Lexer => "terminals do not fit the lexer budget",
            Failure::Productions => "grammar does not fit the production budget",
            Failure::Conflict => "grammar is not LALR(1)",
            Failure::Emit => "artifact emission failed",
        })
    }
}

/// A compiled schema and the lowering that produced it.
pub struct Compiled {
    pub artifact: Artifact,
    /// The most precise level that built tables.
    pub precision: Precision,
}

/// Compile a JSON Schema, trying each lowering from most precise to least.
///
/// Reports the failure of the *most precise* level that got furthest, since
/// that is the one that says what the schema actually needs.
pub fn compile_json_schema(
    schema: &str,
    vocabulary: &[Vec<u8>],
    limits: Limits,
) -> std::result::Result<Compiled, Failure> {
    let mut worst = Failure::Lowering;
    for precision in Precision::LEVELS {
        let options = JsonSchemaOptions {
            precision,
            ..Default::default()
        };
        match compile_at(schema, vocabulary, limits, &options) {
            Ok(artifact) => {
                return Ok(Compiled {
                    artifact,
                    precision,
                });
            }
            // Report the last level's failure: it is the one that had the
            // fewest ways left to go wrong, so it says what the schema
            // actually needs rather than what the most ambitious attempt hit.
            Err(failure) => worst = failure,
        }
    }
    Err(worst)
}

fn compile_at(
    schema: &str,
    vocabulary: &[Vec<u8>],
    limits: Limits,
    options: &JsonSchemaOptions,
) -> std::result::Result<Artifact, Failure> {
    let grammar = json_schema_to_grammar(schema, options).map_err(|_| Failure::Lowering)?;
    let lexicon = extract_within(&grammar, &analyze(&grammar), limits.terminals);
    // Ask from the declared bounds first: a length bound is unrolled into the
    // automaton, so a schema that cannot fit is cheaper to refuse than to build.
    let automata = terminal_automata_within(&grammar, &lexicon, limits.lexer_states as u64)
        .ok_or(Failure::Lexer)?;
    let lexer = build_lexer_within(automata, limits.lexer_states).ok_or(Failure::Lexer)?;
    let groups = group_vocabulary(&lexer, vocabulary);
    let cfg = flatten_within(&lexicon, limits.productions).ok_or(Failure::Productions)?;
    let tables = build(&cfg).map_err(|_| Failure::Conflict)?;
    emit(&lexicon, &lexer, &groups, &cfg, &tables, vocabulary.len()).map_err(|_| Failure::Emit)
}

/// The same search, reporting the failure as an error rather than a code.
pub fn compile_json_schema_or_error(
    schema: &str,
    vocabulary: &[Vec<u8>],
    limits: Limits,
) -> Result<Compiled> {
    compile_json_schema(schema, vocabulary, limits).map_err(|failure| anyhow!("{failure}"))
}
