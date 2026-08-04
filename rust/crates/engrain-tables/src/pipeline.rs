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
use engrain_ir::grammar::Grammar;
use engrain_ir::json_schema::{self, JsonSchemaOptions, Precision, json_schema_to_grammar};
use engrain_lex::lexicon::{DEFAULT_TERMINAL_BUDGET, extract_within, terminal_automata_within};
use engrain_lex::regular::analyze;
use engrain_lex::{build_lexer_within, group_vocabulary};
use engrain_lr::cfg::{DEFAULT_PRODUCTION_BUDGET, flatten_within};
use engrain_lr::tables::build;

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
    /// Start the search at the most precise level rather than below it.
    ///
    /// Buys one thing - a name the schema declares can no longer be read as an
    /// additional property, so a declared type is enforced even while
    /// `additionalProperties` is open - and it is not cheap. See the note on
    /// `compile_json_schema`.
    pub exact: bool,
    /// Digits an unbounded number may run to. See `JsonSchemaOptions`.
    pub max_digits: Option<u32>,
    /// Characters an unbounded string may run to. See `JsonSchemaOptions`.
    pub max_string: Option<u32>,
    /// Whitespace characters allowed at one position. See `JsonSchemaOptions`.
    pub max_whitespace: Option<u32>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            terminals: DEFAULT_TERMINAL_BUDGET,
            lexer_states: 20_000,
            productions: DEFAULT_PRODUCTION_BUDGET,
            exact: false,
            max_digits: None,
            max_string: None,
            max_whitespace: None,
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
    /// What this grammar does not enforce, so the caller knows what a
    /// downstream check still has to do.
    pub relaxations: Vec<String>,
}

/// What this grammar does not enforce, given the level it settled on and the
/// keywords the schema actually uses.
///
/// Gated on both, because a declaration is only useful if it is exact. Saying
/// that `uniqueItems` is unenforced to a caller whose schema never mentions it
/// teaches them to ignore the list, and the list is the only thing standing
/// between a widened mask and a wrong document.
///
/// `oneOf` and `uniqueItems` are here whatever the level: `oneOf` means
/// *exactly one* branch, and a mask that admits a token because some branch
/// allows it cannot also know no other branch does; `uniqueItems` compares an
/// item with every earlier one, which is not a property of the prefix. Both
/// are decidable on the finished document and cheap there.
fn relaxations(schema: &str, precision: Precision) -> Vec<String> {
    let mut found = Vec::new();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(schema) else {
        return found;
    };
    let (mut shadowable, mut counted, mut one_of_objects, mut any_of) = (false, false, false, false);
    let mut asks_counting = false;
    let (mut one_of, mut unique) = (false, false);
    let mut stack = vec![&value];
    while let Some(node) = stack.pop() {
        match node {
            serde_json::Value::Object(map) => {
                let open = !matches!(
                    map.get("additionalProperties"),
                    Some(serde_json::Value::Bool(false))
                );
                let declares = map
                    .get("properties")
                    .and_then(|properties| properties.as_object())
                    .is_some_and(|properties| !properties.is_empty());
                shadowable |= declares && open;
                // `counted` means the schema asks for counting *and does not
                // get it*, which is not the same as the schema asking. An
                // order-free object enumerates subsets of its required set, so
                // it gives up - and silently widens, per object - when that set
                // is past its budget, when `maxProperties` is present at all,
                // or when `minProperties` demands more than `required` names.
                // Reporting only the precision level missed every one of these,
                // which is how 141 of 194 corpus schemas came to accept
                // documents their own `required` forbids without saying so.
                let required = map
                    .get("required")
                    .and_then(|value| value.as_array())
                    .map_or(0, |names| names.len());
                let budget = if open {
                    json_schema::UNORDERED_REQUIRED_BUDGET_OPEN
                } else {
                    json_schema::UNORDERED_REQUIRED_BUDGET_CLOSED
                };
                let minimum = map
                    .get("minProperties")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                asks_counting |= ["required", "minProperties", "maxProperties"]
                    .iter()
                    .any(|keyword| map.contains_key(*keyword));
                counted |= map.contains_key("maxProperties")
                    || required > budget
                    || minimum > required as u64;
                if let Some(branches) = map.get("oneOf").and_then(|value| value.as_array()) {
                    one_of = true;
                    one_of_objects |= branches.len() > 1
                        && branches.iter().all(|branch| {
                            branch.get("properties").is_some()
                                || branch.get("type") == Some(&serde_json::Value::from("object"))
                        });
                }
                any_of |= map.contains_key("anyOf");
                unique |= map.get("uniqueItems") == Some(&serde_json::Value::Bool(true));
                stack.extend(map.values());
            }
            serde_json::Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
    if shadowable && !precision.excludes_declared_names() {
        found.push(json_schema::SHADOWED.to_string());
    }
    if asks_counting && (counted || !precision.enforces_counting()) {
        found.push(json_schema::COUNTING.to_string());
    }
    if one_of_objects && precision.merges_objects() {
        found.push(json_schema::MERGED.to_string());
    }
    if any_of && !precision.merges_branches() {
        found.push(json_schema::SIBLINGS.to_string());
    }
    if one_of {
        found.push(
            "oneOf exclusivity is not enforced: a document may satisfy more than one branch"
                .to_string(),
        );
    }
    if unique {
        found.push("uniqueItems is not enforced".to_string());
    }
    found
}

/// Compile a JSON Schema, trying each lowering from most precise to least.
///
/// Reports the failure of the *most precise* level that got furthest, since
/// that is the one that says what the schema actually needs.
///
/// The search starts below the most precise level, and that is a measured
/// choice rather than an oversight. `Exact` differs from `Shadowed` only in
/// excluding declared names from the generic key, which is exact and regular -
/// but it costs the schema its one shared string lexeme, since every object
/// then carries a key terminal of its own. Over the corpus that is compile p50
/// 27 ms -> 159 ms and a captured step at batch 512 of 72 us -> 155 us, to
/// enforce one keyword interaction a downstream type check settles for
/// nothing. So the default declares it instead, and a caller who would rather
/// pay can start the search at `Exact`.
pub fn compile_json_schema(
    schema: &str,
    vocabulary: &[Vec<u8>],
    limits: Limits,
) -> std::result::Result<Compiled, Failure> {
    let mut worst = Failure::Lowering;
    let entry = if limits.exact { 0 } else { ENTRY };
    for precision in Precision::LEVELS.iter().copied().skip(entry) {
        let options = JsonSchemaOptions {
            precision,
            max_digits: limits.max_digits,
            max_string: limits.max_string,
            max_whitespace: limits.max_whitespace,
            ..Default::default()
        };
        match compile_at(schema, vocabulary, limits, &options) {
            Ok(artifact) => {
                return Ok(Compiled {
                    artifact,
                    precision,
                    relaxations: relaxations(schema, precision),
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

/// Where the search starts in `Precision::LEVELS`.
const ENTRY: usize = 1;

fn compile_at(
    schema: &str,
    vocabulary: &[Vec<u8>],
    limits: Limits,
    options: &JsonSchemaOptions,
) -> std::result::Result<Artifact, Failure> {
    // Which stage a compile spends its time in, since the search tries several
    // lowerings and a slow one is invisible from outside. Behind the same
    // variable as the failure reasons, so the hot path is unchanged.
    let trace = std::env::var_os("ENGRAIN_WHY").is_some();
    let mut mark = std::time::Instant::now();
    let mut lap = |stage: &str| {
        if trace {
            eprintln!("  {:?} {stage} {:?}", options.precision, mark.elapsed());
            mark = std::time::Instant::now();
        }
    };
    let grammar = json_schema_to_grammar(schema, options).map_err(|error| {
        // The reason a schema cannot be lowered is the whole diagnostic - which
        // keyword, in what shape - and collapsing it to a code left the largest
        // remaining coverage gap undiagnosable. Kept behind an environment
        // variable rather than a return type so the hot path stays a code.
        if std::env::var_os("ENGRAIN_WHY").is_some() {
            eprintln!("lowering: {error:#}");
        }
        Failure::Lowering
    })?;
    lap("lower");
    let lexicon = extract_within(&grammar, &analyze(&grammar), limits.terminals);
    // Ask from the declared bounds first: a length bound is unrolled into the
    // automaton, so a schema that cannot fit is cheaper to refuse than to build.
    lap("lexicon");
    let automata = terminal_automata_within(&grammar, &lexicon, limits.lexer_states as u64)
        .ok_or(Failure::Lexer)?;
    lap("automata");
    let lexer = build_lexer_within(automata, limits.lexer_states).ok_or(Failure::Lexer)?;
    lap("lexer");
    if trace {
        eprintln!(
            "  {:?} lexer has {} states, {} terminals",
            options.precision,
            lexer.num_states(),
            lexer.num_terminals()
        );
    }
    // Grouping last of the expensive stages, because it is the expensive one:
    // it scans the whole vocabulary from every lexer state and is 62% of all
    // the time this pipeline spends. The grammar and its tables cost 3%, so
    // finding out whether they build at all is worth doing first - a level
    // that is going to be refused should not pay for a vocabulary it will
    // throw away, and the search tries several levels per schema.
    let cfg = flatten_within(&lexicon, limits.productions).ok_or(Failure::Productions)?;
    lap("cfg");
    let tables = build(&cfg).map_err(|error| {
        if std::env::var_os("ENGRAIN_WHY").is_some() {
            eprintln!("conflict: {error:#}");
        }
        Failure::Conflict
    })?;
    lap("tables");
    let groups = group_vocabulary(&lexer, vocabulary);
    lap("groups");
    let artifact =
        emit(&lexicon, &lexer, &groups, &cfg, &tables, vocabulary.len()).map_err(|_| Failure::Emit);
    lap("emit");
    // Everything the pipeline built is freed on the way out of this function,
    // and it is hundreds of thousands of small vectors - one per group of
    // tokens, one per reading, one per reading's terminals. That was 186 ms of
    // a 2.45 s run over thirty schemas until the extension took mimalloc as
    // its global allocator, and 62 ms after. Named here rather than left as an
    // unattributed gap between the last lap and the return.
    drop(groups);
    drop(cfg);
    drop(tables);
    drop(lexicon);
    drop(lexer);
    lap("drop");
    artifact
}

/// Compile a grammar that is already lowered, under the same budgets the
/// schema search uses.
///
/// A regular expression or an EBNF source arrives lowered, so it skips the
/// search - but it must not skip the budgets. It had been building its lexer
/// and flattening its grammar unbounded, which for a pattern supplied by a
/// request is a way to spend the server's memory rather than to be refused.
pub fn compile_grammar_within(
    grammar: &Grammar,
    vocabulary: &[Vec<u8>],
    limits: Limits,
) -> std::result::Result<Artifact, Failure> {
    let lexicon = extract_within(grammar, &analyze(grammar), limits.terminals);
    let automata = terminal_automata_within(grammar, &lexicon, limits.lexer_states as u64)
        .ok_or(Failure::Lexer)?;
    let lexer = build_lexer_within(automata, limits.lexer_states).ok_or(Failure::Lexer)?;
    let cfg = flatten_within(&lexicon, limits.productions).ok_or(Failure::Productions)?;
    let tables = build(&cfg).map_err(|_| Failure::Conflict)?;
    let groups = group_vocabulary(&lexer, vocabulary);
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
