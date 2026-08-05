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

/// Does this object constrain a shape, rather than only counting properties?
///
/// Mirrors the lowering's own test: `required` alone does not say the document
/// is an object, so a schema built from it admits every JSON value.
fn describes_shape(map: &serde_json::Map<String, serde_json::Value>) -> bool {
    const SHAPE: &[&str] = &[
        "type",
        "properties",
        "patternProperties",
        "items",
        "prefixItems",
        "enum",
        "const",
        "pattern",
        "format",
        "$ref",
        "anyOf",
        "oneOf",
        "allOf",
    ];
    map.keys().any(|key| SHAPE.contains(&key.as_str()))
}

/// Are these branches pinned apart by a shared property fixed to a distinct
/// value in each?
///
/// If so no document satisfies two of them, so `oneOf` asks for nothing a
/// union does not already give. Only `const` and a one-element `enum` count:
/// anything weaker leaves an overlap this cannot rule out, and the whole point
/// of the check is that it may only ever say yes when it is certain.
fn discriminated(branches: &[serde_json::Value]) -> bool {
    if branches.len() < 2 {
        return true;
    }
    let pinned = |branch: &serde_json::Value, name: &str| -> Option<serde_json::Value> {
        let property = branch.get("properties")?.get(name)?;
        if let Some(value) = property.get("const") {
            return Some(value.clone());
        }
        match property.get("enum")?.as_array()?.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        }
    };
    let names: Vec<String> = match branches[0].get("properties").and_then(|p| p.as_object()) {
        Some(properties) => properties.keys().cloned().collect(),
        None => return false,
    };
    names.iter().any(|name| {
        let mut seen: Vec<serde_json::Value> = Vec::new();
        for branch in branches {
            match pinned(branch, name) {
                Some(value) if !seen.contains(&value) => seen.push(value),
                _ => return false,
            }
        }
        true
    })
}

/// The object an `allOf` describes, or `None` where there is no `allOf`.
///
/// A shallow union is enough for what this file asks: which names are
/// declared, how many are required, and whether the object is closed. Local
/// `$ref`s are followed because a branch is usually one, and a conflict is
/// resolved the tighter way, which is the direction the lowering resolves it.
fn merge_all_of(
    map: &serde_json::Map<String, serde_json::Value>,
    root: &serde_json::Value,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let branches = map.get("allOf")?.as_array()?;
    let mut merged = map.clone();
    merged.remove("allOf");
    for branch in branches {
        let Some(branch) = resolve(branch, root).and_then(|node| node.as_object()) else {
            continue;
        };
        for (key, value) in branch {
            match key.as_str() {
                "properties" | "patternProperties" => {
                    let mut names = merged
                        .get(key)
                        .and_then(|existing| existing.as_object())
                        .cloned()
                        .unwrap_or_default();
                    if let Some(added) = value.as_object() {
                        for (name, schema) in added {
                            names.insert(name.clone(), schema.clone());
                        }
                    }
                    merged.insert(key.clone(), serde_json::Value::Object(names));
                }
                "required" => {
                    let mut names = merged
                        .get(key)
                        .and_then(|existing| existing.as_array())
                        .cloned()
                        .unwrap_or_default();
                    if let Some(added) = value.as_array() {
                        for name in added {
                            if !names.contains(name) {
                                names.push(name.clone());
                            }
                        }
                    }
                    merged.insert(key.clone(), serde_json::Value::Array(names));
                }
                "additionalProperties" => {
                    if value == &serde_json::Value::Bool(false)
                        || merged.get(key) == Some(&serde_json::Value::Bool(false))
                    {
                        merged.insert(key.clone(), serde_json::Value::Bool(false));
                    }
                }
                _ => {
                    merged.entry(key.clone()).or_insert_with(|| value.clone());
                }
            }
        }
    }
    Some(merged)
}

/// Follow a local `$ref` once, so a branch that is one can be read.
fn resolve<'a>(
    node: &'a serde_json::Value,
    root: &'a serde_json::Value,
) -> Option<&'a serde_json::Value> {
    let Some(pointer) = node.get("$ref").and_then(|target| target.as_str()) else {
        return Some(node);
    };
    let mut found = root;
    for part in pointer
        .trim_start_matches('#')
        .split('/')
        .filter(|part| !part.is_empty())
    {
        found = found.get(part)?;
    }
    Some(found)
}

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
    // The third element says this node is an `allOf` branch. A branch is never
    // lowered on its own - it is merged into the object above it - so checking
    // it as an object reports a budget it never has to meet. Its children
    // still need walking, and their pointers have to name where they really
    // are, so the branch stays on the stack and only its own checks are off.
    let mut stack = vec![(&value, String::from("#"), false)];
    while let Some((node, at, branch_of_all_of)) = stack.pop() {
        if let serde_json::Value::Object(original) = node
            && !branch_of_all_of
        {
            // `allOf` is merged before anything else is lowered, so the object
            // the parser builds is the merged one and the branches on their
            // own say nothing about it. Reading the branches instead was how a
            // schema could report *nothing* relaxed and still admit a document
            // missing nine required names: each branch was inside the budget
            // and their union was not.
            let merged = merge_all_of(original, &value);
            let map = merged.as_ref().unwrap_or(original);
            let open = !matches!(
                map.get("additionalProperties"),
                Some(serde_json::Value::Bool(false))
            );
            let declared: Vec<&String> = map
                .get("properties")
                .and_then(|properties| properties.as_object())
                .map(|properties| properties.keys().collect())
                .unwrap_or_default();
            // A key matching a `patternProperties` entry is not an additional
            // property, but this lowering reads it as one wherever the object
            // is open - and then the pattern's value schema is not enforced.
            // `Exact` does not help: it excludes the names a schema *spells*,
            // and a pattern spells none of them.
            if open
                && map
                    .get("patternProperties")
                    .and_then(|value| value.as_object())
                    .is_some_and(|patterns| !patterns.is_empty())
            {
                push(
                    &mut found,
                    "patternProperties",
                    &at,
                    "a key matching a pattern here also reads as an additional \
                     property, so the pattern's value schema is not enforced"
                        .to_string(),
                    "set `additionalProperties: false` here",
                );
            }

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

            // What a choice costs depends on which of three shapes the
            // lowering finds, and guessing the worst one for all three is how
            // a list starts crying wolf: `oneOf` and `anyOf` are between them
            // 774 of the 965 notes this file used to emit, and neither fired
            // on a single walk.
            //
            // A choice lowered branch by branch *is* a union, which is what
            // `anyOf` means, so nothing is lost by it. What is lost is the
            // three cases below.
            if let Some(key) = ["oneOf", "anyOf"]
                .into_iter()
                .find(|key| map.contains_key(*key))
            {
                let branches = map
                    .get(key)
                    .and_then(|value| value.as_array())
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);

                // 1. Branches that only say what is required are folded into
                //    the object as the requirements they *share*, because
                //    distributing the siblings one branch at a time gives
                //    alternatives an LALR parser cannot tell apart. Two
                //    branches sharing nothing therefore require nothing.
                let required_only = !branches.is_empty()
                    && branches.iter().all(|branch| {
                        branch
                            .as_object()
                            .is_some_and(|branch| branch.keys().all(|key| key == "required"))
                    })
                    && describes_shape(map);
                if required_only && precision.merges_branches() {
                    push(
                        &mut found,
                        key,
                        &at,
                        format!(
                            "the `{key}` branches only say what is required, so they \
                             are folded into this object as the names every branch \
                             requires - and a document may satisfy none of them"
                        ),
                        "give each branch a shape of its own, so the choice survives \
                         as a choice",
                    );
                }

                // 2. Otherwise the branches lower on their own, and every
                //    keyword beside the choice is discarded with the object it
                //    described. The largest relaxation the schema does not
                //    show: corpus schema 25 admits `[{}]` against
                //    `required: ["op", "path"]` because the requirement lives
                //    beside the choice rather than inside it.
                //
                //    Declared per discarded keyword, since "your `required` is
                //    not enforced, here" is what an author can act on and
                //    "this object has a oneOf" is not.
                if !required_only {
                    for beside in map.keys() {
                        if !BESIDE_A_CHOICE.contains(&beside.as_str()) {
                            continue;
                        }
                        push(
                            &mut found,
                            beside,
                            &at,
                            format!(
                                "`{beside}` sits beside a `{key}` whose branches are \
                                 lowered on their own, and a branch lowered on its own \
                                 does not carry its siblings"
                            ),
                            "move it inside each branch, where it is lowered with them",
                        );
                    }
                }

                // 3. Object branches may collapse into one object, and then a
                //    document can take properties from several of them.
                let objects = branches.len() > 1
                    && branches.iter().all(|branch| {
                        branch.get("properties").is_some()
                            || branch.get("type") == Some(&serde_json::Value::from("object"))
                    });
                if objects && precision.merges_objects() {
                    push(
                        &mut found,
                        key,
                        &at,
                        format!(
                            "the `{key}` branches were merged into one object, so a \
                             document may take properties from several of them"
                        ),
                        "give the branches a discriminator, which lets them stay \
                         separate",
                    );
                }

                // `oneOf` asks for *exactly* one branch, and a union cannot
                // tell that a second one also matched. Unless no second one
                // can: branches pinned apart by a shared discriminator are
                // pairwise disjoint, and there `oneOf` and a union are the
                // same language. That is worth checking rather than assuming,
                // because a discriminated union is the shape structured output
                // is written in.
                if key == "oneOf" && !discriminated(branches) {
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

            walk_children(original, &at, &mut stack);
        } else if let serde_json::Value::Object(original) = node {
            walk_children(original, &at, &mut stack);
        }
    }
    found.sort_by(|one, other| (&one.at, &one.keyword).cmp(&(&other.at, &other.keyword)));
    found
}

/// Push every place under this node where a schema can be.
///
/// `default` and `examples` hold documents, `enum` holds values, and
/// `required` holds names - walking into those would read a property name as a
/// keyword and report it as unenforced, which is how a closed list becomes a
/// list nobody reads.
fn walk_children<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    at: &str,
    stack: &mut Vec<(&'a serde_json::Value, String, bool)>,
) {
    for (key, child) in map {
        match key.as_str() {
            "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas" => {
                if let Some(entries) = child.as_object() {
                    for (name, value) in entries {
                        stack.push((value, format!("{at}/{key}/{name}"), false));
                    }
                }
            }
            "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                if let Some(entries) = child.as_array() {
                    for (index, value) in entries.iter().enumerate() {
                        stack.push((value, format!("{at}/{key}/{index}"), key == "allOf"));
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
                    stack.push((child, format!("{at}/{key}"), false));
                } else if let Some(entries) = child.as_array() {
                    for (index, value) in entries.iter().enumerate() {
                        stack.push((value, format!("{at}/{key}/{index}"), false));
                    }
                }
            }
            _ => {}
        }
    }
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
