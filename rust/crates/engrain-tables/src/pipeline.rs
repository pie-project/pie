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
    pub relaxations: Vec<Relaxation>,
}

/// One thing this grammar does not enforce, where it is, and what to change.
///
/// Gated on both the level and the keywords the schema uses, because a
/// declaration is only useful if it is exact. Saying that `uniqueItems` is
/// unenforced to a caller whose schema never mentions it teaches them to
/// ignore the list, and the list is the only thing standing between a widened
/// mask and a wrong document.
///
/// `oneOf` and `uniqueItems` are here whatever the level: `oneOf` means
/// *exactly one* branch, and a mask that admits a token because some branch
/// allows it cannot also know no other branch does; `uniqueItems` compares an
/// item with every earlier one, which is not a property of the prefix. Both
/// are decidable on the finished document and cheap there, which is what the
/// remedy says.
#[derive(Clone, Debug)]
pub struct Relaxation {
    /// The JSON Schema keyword responsible.
    pub keyword: String,
    /// A JSON pointer to where it is, so a schema author can go there.
    pub at: String,
    /// What the mask now admits that the schema does not.
    pub effect: String,
    /// What to change. Empty when there is nothing to change - `uniqueItems`
    /// is not a property of a prefix and no rewrite makes it one.
    pub remedy: String,
}

/// Validation keywords across the drafts that this front end does not lower.
///
/// Naming what is *unenforced* rather than what is enforced, and doing it from
/// the specification rather than from the schema, is what makes the list both
/// closed and quiet. Closed, because a keyword a validator checks and we do
/// not is in this list by construction rather than because somebody
/// remembered it; quiet, because a key that is not a JSON Schema keyword at
/// all - an extension, or a `properties` map somebody forgot to wrap - is
/// ignored by a validator too, so it widens nothing and saying so would be
/// noise. A list that reports noise is a list callers stop reading.
///
/// Measured: before this, 13.8% of over-accepted documents were rejected on a
/// keyword no entry mentioned. That is precisely the caller who reads the
/// list, re-checks what it names, and ships a wrong document.
/// Keywords that constrain the object a choice sits on, and are therefore lost
/// when the choice is lowered branch by branch.
const BESIDE_A_CHOICE: &[&str] = &[
    "required",
    "properties",
    "patternProperties",
    "additionalProperties",
    "minProperties",
    "maxProperties",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "enum",
    "const",
];

const UNLOWERED: &[&str] = &[
    "multipleOf",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "not",
    "contains",
    "minContains",
    "maxContains",
    "propertyNames",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "$recursiveRef",
    "$dynamicRef",
];

fn relaxations(schema: &str, precision: Precision) -> Vec<Relaxation> {
    let mut found: Vec<Relaxation> = Vec::new();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(schema) else {
        return found;
    };
    let mut seen: std::collections::HashSet<(String, String)> = Default::default();
    let mut push =
        |found: &mut Vec<Relaxation>, keyword: &str, at: &str, effect: String, remedy: &str| {
            if seen.insert((keyword.to_string(), at.to_string())) {
                found.push(Relaxation {
                    keyword: keyword.to_string(),
                    at: at.to_string(),
                    effect,
                    remedy: remedy.to_string(),
                });
            }
        };

    // The walk carries the pointer, because "this schema is relaxed" sends an
    // author looking and "this object is" sends them to the object.
    let mut stack = vec![(&value, String::from("#"))];
    while let Some((node, at)) = stack.pop() {
        if let serde_json::Value::Object(map) = node {
            let open = !matches!(
                map.get("additionalProperties"),
                Some(serde_json::Value::Bool(false))
            );
            let declared: Vec<&String> = map
                .get("properties")
                .and_then(|properties| properties.as_object())
                .map(|properties| properties.keys().collect())
                .unwrap_or_default();
            if open && !declared.is_empty() && !precision.excludes_declared_names() {
                let names: Vec<&str> = declared.iter().take(3).map(|name| name.as_str()).collect();
                push(
                    &mut found,
                    "additionalProperties",
                    &at,
                    format!(
                        "the declared types of {}{} are not enforced, because a \
                             key that spells one of them also reads as an additional \
                             property",
                        names.join(", "),
                        if declared.len() > 3 { ", ..." } else { "" }
                    ),
                    "set `additionalProperties: false` here",
                );
            }

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
            let asks = ["required", "minProperties", "maxProperties"]
                .iter()
                .any(|keyword| map.contains_key(*keyword));
            let over =
                map.contains_key("maxProperties") || required > budget || minimum > required as u64;
            if asks && (over || !precision.enforces_counting()) {
                push(
                    &mut found,
                    "required",
                    &at,
                    format!(
                        "an object here may close with properties missing: {required} \
                             are required and the parser can carry {budget} at once"
                    ),
                    if required > budget {
                        "require fewer properties here, or close the object with \
                             `additionalProperties: false` to raise the budget"
                    } else {
                        "drop `maxProperties`, or lower `minProperties` to the \
                             number of required names"
                    },
                );
            }

            if let Some(branches) = map.get("oneOf").and_then(|value| value.as_array()) {
                push(
                    &mut found,
                    "oneOf",
                    &at,
                    "a document may satisfy more than one branch, which `oneOf` \
                         forbids and a mask over a prefix cannot see"
                        .to_string(),
                    "give the branches a discriminator - a `const` on a shared \
                         property - or use `anyOf` if more than one may match",
                );
                let objects = branches.len() > 1
                    && branches.iter().all(|branch| {
                        branch.get("properties").is_some()
                            || branch.get("type") == Some(&serde_json::Value::from("object"))
                    });
                if objects && precision.merges_objects() {
                    push(
                        &mut found,
                        "oneOf",
                        &at,
                        "the branches were merged into one object, so a document \
                             may take properties from several of them"
                            .to_string(),
                        "give the branches a discriminator, which lets them stay \
                             separate",
                    );
                }
            }
            // A choice whose branches cannot be merged is lowered branch by
            // branch, and every keyword sitting beside it is discarded with
            // the object it described. That is the largest relaxation the
            // schema does not show, and it is why schema 25 of the corpus
            // admitted `[{}]` against `required: ["op", "path"]`: the
            // requirement lives beside a `oneOf`, not inside it.
            //
            // Declared per discarded keyword rather than once for the
            // choice, because "your `required` is not enforced, here" is
            // what an author can act on and "this object has a oneOf" is
            // not.
            if map.contains_key("anyOf") || map.contains_key("oneOf") {
                let choice = if map.contains_key("oneOf") {
                    "oneOf"
                } else {
                    "anyOf"
                };
                for key in map.keys() {
                    if !BESIDE_A_CHOICE.contains(&key.as_str()) {
                        continue;
                    }
                    push(
                        &mut found,
                        key,
                        &at,
                        format!(
                            "`{key}` sits beside a `{choice}` whose branches may be \
                                 lowered on their own, and a branch lowered on its own \
                                 does not carry its siblings"
                        ),
                        "move it inside each branch, where it is lowered with them",
                    );
                }
            }

            if map.contains_key("anyOf") {
                push(
                    &mut found,
                    "anyOf",
                    &at,
                    "a document may satisfy no branch: the mask admits any \
                         prefix some branch allows, and whether a branch can \
                         still be completed is not a property of a prefix"
                        .to_string(),
                    "give the branches a discriminator - a `const` on a \
                         shared property - so the first token picks one",
                );
                if !precision.merges_branches() {
                    push(
                        &mut found,
                        "anyOf",
                        &format!("{at}/anyOf"),
                        "the branches do not inherit the keywords beside them".to_string(),
                        "move the sibling keywords inside each branch",
                    );
                }
            }
            // Anything the lowering does not read is unenforced, and the
            // list has to be closed rather than a set of cases somebody
            // remembered. Measured: naming keywords one at a time left
            // 13.8% of over-accepted documents rejected on a keyword no
            // entry mentioned, which is exactly the caller who reads the
            // list, checks what it says, and ships a wrong document.
            for key in map.keys() {
                if !UNLOWERED.contains(&key.as_str()) {
                    continue;
                }
                push(
                    &mut found,
                    key,
                    &at,
                    format!("`{key}` is not lowered, so nothing here enforces it"),
                    "remove it and check it on the finished document instead",
                );
            }

            if map.get("uniqueItems") == Some(&serde_json::Value::Bool(true)) {
                push(
                    &mut found,
                    "uniqueItems",
                    &at,
                    "duplicate items are admitted: uniqueness compares an item \
                         with every earlier one, which a prefix does not know"
                        .to_string(),
                    "",
                );
            }

            // Descend only where a schema can be. `default` and `examples`
            // hold documents, `enum` holds values, and `required` holds
            // names - walking into those would read a property name as a
            // keyword and report it as unenforced, which is how a closed
            // list becomes a list nobody reads.
            for (key, child) in map {
                match key.as_str() {
                    "properties" | "patternProperties" | "$defs" | "definitions"
                    | "dependentSchemas" => {
                        if let Some(entries) = child.as_object() {
                            for (name, value) in entries {
                                stack.push((value, format!("{at}/{key}/{name}")));
                            }
                        }
                    }
                    "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                        if let Some(entries) = child.as_array() {
                            for (index, value) in entries.iter().enumerate() {
                                stack.push((value, format!("{at}/{key}/{index}")));
                            }
                        }
                    }
                    "items"
                    | "additionalItems"
                    | "additionalProperties"
                    | "unevaluatedProperties"
                    | "unevaluatedItems"
                    | "not"
                    | "if"
                    | "then"
                    | "else"
                    | "contains"
                    | "propertyNames" => {
                        if child.is_object() {
                            stack.push((child, format!("{at}/{key}")));
                        } else if let Some(entries) = child.as_array() {
                            for (index, value) in entries.iter().enumerate() {
                                stack.push((value, format!("{at}/{key}/{index}")));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    found.sort_by(|one, other| (&one.at, &one.keyword).cmp(&(&other.at, &other.keyword)));
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
