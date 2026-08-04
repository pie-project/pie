//! JSON Schema grammar frontend.
//!
//! Supported schema forms:
//! - Primitive JSON types, enums, and constants
//! - String length, pattern, and selected format constraints
//! - Inclusive i64 integer and bounded decimal number ranges
//! - Arrays, fixed-order objects, local `$ref`, `anyOf`, and `oneOf`
//!
//! Bounded `number` schemas emit a sound non-exponent decimal subset.
//! Pattern/format constraints cannot be combined with length constraints.

mod typed;

/// The number of required properties an order-free object will enumerate
/// subsets of, past which it gives up counting and widens. Published because a
/// caller cannot otherwise know which of the two things it was handed: an
/// object that enforces `required`, or one that does not.
pub use typed::{UNORDERED_REQUIRED_BUDGET_CLOSED, UNORDERED_REQUIRED_BUDGET_OPEN};

use anyhow::Result;
use serde_json::Value;

use crate::grammar::Grammar;

/// Options for JSON Schema conversion.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JsonSchemaOptions {
    /// Allow arbitrary whitespace between JSON elements.
    pub any_whitespace: bool,
    /// If true, disallow properties and items the schema does not name.
    ///
    /// JSON Schema says the opposite: an absent `additionalProperties` permits
    /// any additional property, and XGrammar lowers `{"type": "object"}` to a
    /// generic object accordingly. Defaulting this on made every unconstrained
    /// object accept `{}` and nothing else, which silently truncates the
    /// language rather than reporting that it cannot be represented.
    pub strict_mode: bool,
    /// How much of a schema's shape to keep when it cannot be kept exactly.
    ///
    /// Some schemas have no LALR(1) grammar at their most precise lowering, so
    /// the compiler tries the levels in order and keeps the first that builds
    /// tables. Nothing here is unsound in the direction that matters: a lower
    /// level accepts more, never less.
    pub precision: Precision,
    /// Digits an *unbounded* number may run to, when the schema gives no
    /// `minimum` or `maximum` to bound it with.
    ///
    /// `None` is the schema as written: JSON permits an integer of any length,
    /// so the grammar does too. That is correct and, for a generator, ruinous.
    /// A model handed a mask that still admits a digit emits one, and a 0.6B
    /// model at temperature 0.8 keeps emitting them: measured through vLLM on a
    /// schema whose last property is an unbounded integer, **71.9% of requests
    /// ran to the token limit mid-number** - and XGrammar's 70.8% on the same
    /// schema says this is a property of the language, not of either engine.
    ///
    /// Setting it narrows the language deliberately, which is the one direction
    /// this compiler will not take on its own - so it is off unless a caller
    /// asks. It is not in `relaxations`, which is the register for the other
    /// direction: what the grammar admits and the schema does not.
    pub max_digits: Option<u32>,

    /// Characters an *unbounded* string may run to, when the schema gives no
    /// `maxLength` to bound it with.
    ///
    /// Same shape as `max_digits` and found the same way. Of 32 requests that
    /// ran to the token limit on a corpus of real schemas, **19 were inside a
    /// string** - one of them 90 zeroes and counting - because JSON permits a
    /// string of any length and so the grammar does. A model handed a mask
    /// that still admits a character emits one.
    pub max_string: Option<u32>,

    /// Whitespace characters allowed at one position between tokens, when
    /// `any_whitespace` is on.
    ///
    /// The other 13 of those 32. `{"xi":  \n\n\n...` for two hundred tokens is
    /// the shape: the model reaches a position where it has not decided what
    /// to write, whitespace is admitted, and admitting it again after that is
    /// a loop with no pressure to leave. JSON allows it and no document wants
    /// it, so this is the cheapest place the language can be narrowed without
    /// costing a document anyone would write.
    pub max_whitespace: Option<u32>,
}

/// How much the lowering may widen, least first.
///
/// One axis, and every step along it describes a *larger* language than the
/// last. That direction is the contract rather than a preference: a mask that
/// admits too much leaves a document for a checker to reject, while a mask
/// that admits too little makes a valid document ungeneratable and nothing
/// downstream can repair it.
///
/// This used to be a chain that mixed two axes - whether objects accept their
/// properties in any order, and how far branches may be merged - and the first
/// of those runs the wrong way. `Ordered` bought a smaller lexer by refusing
/// permutations of valid documents, which is exactly the failure the contract
/// forbids, so it is gone. Objects are unordered at every level now.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Precision {
    /// The schema as written: objects accept their properties in any order,
    /// which is what JSON Schema means, and branches stay separate. Costs
    /// lexer states, because every property name is live at once.
    Exact,
    /// The name of an additional property may repeat one the schema declares.
    ///
    /// Excluding them is exact and regular - `string_body_excluding` builds
    /// the complement - but it costs the schema its one shared string lexeme:
    /// every object needs a key terminal carrying its own trie, and the lexer
    /// determinises the union of all of them. Measured over the corpus that
    /// is 69 schemas past the lexer budget against 4, so where it does not
    /// fit this level gives the key back and the caller must check that an
    /// additional property is not a declared one wearing the wrong type.
    Shadowed,
    /// `oneOf` branches that all describe objects collapse into one object:
    /// every property any branch names, required only where every branch
    /// requires it. Accepts documents that satisfy no branch exactly, which is
    /// the direction a mask may err in, and removes the k-way choice that a
    /// parser cannot resolve - reduce/reduce conflicts are the largest single
    /// reason a schema has no LALR(1) grammar, and `oneOf` is in 86% of them.
    Merged,
    /// `anyOf` branches lower on their own, without the sibling keywords that
    /// constrain them. Loses the object those siblings described, and is the
    /// last resort before refusing the schema.
    Branches,
}

pub const SHADOWED: &str = "a property whose name the schema declares may also be \
read as an additional one, so its declared type is not enforced while \
additionalProperties is open";
pub const COUNTING: &str = "required, minProperties and maxProperties are not enforced";
pub const MERGED: &str = "branches of a oneOf over objects are merged, so a document \
may satisfy no branch exactly";
pub const SIBLINGS: &str = "anyOf branches do not inherit the keywords beside them";

impl Precision {
    /// Most faithful first.
    pub const LEVELS: [Precision; 4] = [
        Precision::Exact,
        Precision::Shadowed,
        Precision::Merged,
        Precision::Branches,
    ];

    /// Must objects enforce `required`, `minProperties` and `maxProperties`?
    ///
    /// These are the keywords that need a tally in the parser state, and the
    /// tally is what costs productions. A single object too large to count is
    /// handled where it is built, by widening that object alone; this is the
    /// coarser lever, for a schema whose *whole* grammar is over budget, which
    /// is a measure no one object can see.
    pub fn enforces_counting(self) -> bool {
        matches!(self, Precision::Exact | Precision::Shadowed)
    }

    /// May the generic key of `additionalProperties` also spell a name the
    /// schema declares? Only the first level says no, because saying no is
    /// what costs the shared string lexeme.
    pub fn excludes_declared_names(self) -> bool {
        self == Precision::Exact
    }

    /// Do `anyOf` branches inherit their sibling keywords?
    pub fn merges_branches(self) -> bool {
        self != Precision::Branches
    }

    /// May branches that all describe objects collapse into one object?
    pub fn merges_objects(self) -> bool {
        self == Precision::Merged
    }
}

impl Default for JsonSchemaOptions {
    fn default() -> Self {
        Self {
            any_whitespace: true,
            strict_mode: false,
            precision: Precision::Exact,
            max_digits: None,
            max_string: None,
            max_whitespace: None,
        }
    }
}

/// Convert a JSON Schema string directly to a grammar.
pub fn json_schema_to_grammar(schema: &str, options: &JsonSchemaOptions) -> Result<Grammar> {
    let schema: Value = serde_json::from_str(schema)?;
    typed::convert(&schema, options)?.to_grammar()
}

/// Convert a parsed JSON Schema to an EBNF representation.
pub fn json_schema_to_ebnf(schema: &Value, options: &JsonSchemaOptions) -> Result<String> {
    Ok(typed::convert(schema, options)?.to_ebnf())
}

/// Create a grammar for any valid JSON value.
pub fn builtin_json_grammar() -> Result<Grammar> {
    Grammar::from_ebnf(BUILTIN_JSON_EBNF, "root")
}

const BUILTIN_JSON_EBNF: &str = r#"
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= ws string ws ":" ws value
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1f] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
"#;

pub(super) fn parse_i64_keyword(
    schema: &serde_json::Map<String, Value>,
    keyword: &str,
) -> Result<Option<i64>> {
    schema
        .get(keyword)
        .map(|value| {
            // A bound written as a float still bounds an integer: `1.0` and
            // `1e3` are how JSON Schema authors write 1 and 1000, and refusing
            // them costs a schema for nothing. A fractional bound is tightened
            // inward, which is what it means for integers.
            if let Some(exact) = value.as_i64() {
                return Ok(exact);
            }
            let approximate = value
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("{keyword} must be a number"))?;
            if !approximate.is_finite() {
                anyhow::bail!("{keyword} must be finite");
            }
            let tightened = if keyword.contains("inimum") {
                approximate.ceil()
            } else {
                approximate.floor()
            };
            if tightened < i64::MIN as f64 || tightened > i64::MAX as f64 {
                anyhow::bail!("{keyword} is outside the representable range");
            }
            Ok(tightened as i64)
        })
        .transpose()
}

/// The tail of a digit run, bounded or not. One digit is already spent on the
/// leading non-zero, so a budget of `n` leaves `n - 1` here.
pub(super) fn digit_tail(max_digits: Option<u32>) -> String {
    match max_digits {
        Some(budget) if budget > 1 => format!("[0-9]{{0,{}}}", budget - 1),
        Some(_) => String::new(),
        None => "[0-9]*".to_string(),
    }
}

pub(super) fn generate_integer_range_regex(
    min: Option<i64>,
    max: Option<i64>,
    max_digits: Option<u32>,
) -> String {
    match (min, max) {
        (None, None) => format!("-?(?:0|[1-9]{})", digit_tail(max_digits)),
        (Some(min), Some(max)) if min == max => min.to_string(),
        (Some(min), Some(max)) if min >= 0 => positive_range_regex(min as u64, max as u64),
        (Some(min), Some(max)) if max < 0 => {
            format!(
                "-{}",
                positive_range_regex(max.unsigned_abs(), min.unsigned_abs())
            )
        }
        (Some(min), Some(max)) => format!(
            "(?:-{}|{})",
            positive_range_regex(1, min.unsigned_abs()),
            positive_range_regex(0, max as u64)
        ),
        (Some(min), None) if min >= 0 => positive_range_regex_unbounded(min as u64),
        (Some(min), None) => format!(
            "(?:-{}|(?:0|[1-9]{}))",
            positive_range_regex(1, min.unsigned_abs()),
            digit_tail(max_digits)
        ),
        (None, Some(max)) if max < 0 => format!(
            "-(?:{})",
            positive_range_regex_unbounded(max.unsigned_abs())
        ),
        (None, Some(max)) => format!("(?:-[1-9][0-9]*|{})", positive_range_regex(0, max as u64)),
    }
}

pub(super) fn generate_bounded_number_regex(min: Option<i64>, max: Option<i64>) -> String {
    let mut alternatives = Vec::new();
    if min.is_none_or(|value| value <= 0) {
        let magnitude_min = match max {
            Some(value) if value < 0 => value.unsigned_abs(),
            _ => 0,
        };
        let magnitude_max = min.map(i64::unsigned_abs);
        if magnitude_max.is_none_or(|upper| magnitude_min <= upper) {
            alternatives.push(format!(
                "-(?:{})",
                decimal_magnitude_regex(magnitude_min, magnitude_max)
            ));
        }
    }
    if max.is_none_or(|value| value >= 0) {
        let nonnegative_min = min.unwrap_or(0).max(0) as u64;
        let nonnegative_max = max.filter(|&value| value >= 0).map(|value| value as u64);
        if nonnegative_max.is_none_or(|upper| nonnegative_min <= upper) {
            alternatives.push(decimal_magnitude_regex(nonnegative_min, nonnegative_max));
        }
    }
    match alternatives.len() {
        1 => alternatives.pop().unwrap(),
        _ => format!("(?:{})", alternatives.join("|")),
    }
}

fn decimal_magnitude_regex(min: u64, max: Option<u64>) -> String {
    match max {
        None => format!("(?:{})(?:\\.[0-9]+)?", positive_range_regex_unbounded(min)),
        Some(max) if min == max => format!("{}(?:\\.0+)?", max),
        Some(max) => format!(
            "(?:(?:{})(?:\\.[0-9]+)?|{}(?:\\.0+)?)",
            positive_range_regex(min, max - 1),
            max
        ),
    }
}

fn positive_range_regex(min: u64, max: u64) -> String {
    if min == max {
        return min.to_string();
    }
    let min_text = min.to_string();
    let max_text = max.to_string();
    if min_text.len() == max_text.len() {
        return same_length_range(&min_text, &max_text);
    }

    let mut parts = Vec::new();
    let first_ceiling = 10u64.pow(min_text.len() as u32) - 1;
    if min <= first_ceiling {
        parts.push(positive_range_regex(min, first_ceiling));
    }
    for digits in (min_text.len() + 1)..max_text.len() {
        parts.push(format!("[1-9][0-9]{{{}}}", digits - 1));
    }
    let last_floor = 10u64.pow((max_text.len() - 1) as u32);
    if last_floor <= max {
        parts.push(positive_range_regex(last_floor, max));
    }
    match parts.len() {
        1 => parts.pop().unwrap(),
        _ => format!("(?:{})", parts.join("|")),
    }
}

fn positive_range_regex_unbounded(min: u64) -> String {
    match min {
        0 => "(?:0|[1-9][0-9]*)".to_string(),
        1 => "[1-9][0-9]*".to_string(),
        _ => {
            let text = min.to_string();
            let ceiling = 10u64.pow(text.len() as u32) - 1;
            format!(
                "(?:{}|[1-9][0-9]{{{},}})",
                positive_range_regex(min, ceiling),
                text.len()
            )
        }
    }
}

fn same_length_range(min: &str, max: &str) -> String {
    let min: Vec<u8> = min.bytes().map(|byte| byte - b'0').collect();
    let max: Vec<u8> = max.bytes().map(|byte| byte - b'0').collect();
    build_digit_range(&min, &max, 0)
}

fn build_digit_range(min: &[u8], max: &[u8], position: usize) -> String {
    if position >= min.len() {
        return String::new();
    }
    if position == min.len() - 1 {
        return digit_range(min[position], max[position]);
    }
    if min[position] == max[position] {
        return format!(
            "{}{}",
            min[position],
            build_digit_range(min, max, position + 1)
        );
    }

    let mut parts = Vec::new();
    let lower_max = vec![9; min.len() - position - 1];
    parts.push(format!(
        "{}{}",
        min[position],
        build_digit_range(&min[position + 1..], &lower_max, 0)
    ));
    if min[position] + 1 < max[position] {
        parts.push(format!(
            "{}[0-9]{{{}}}",
            digit_range(min[position] + 1, max[position] - 1),
            min.len() - position - 1
        ));
    }
    let upper_min = vec![0; max.len() - position - 1];
    parts.push(format!(
        "{}{}",
        max[position],
        build_digit_range(&upper_min, &max[position + 1..], 0)
    ));
    format!("(?:{})", parts.join("|"))
}

fn digit_range(min: u8, max: u8) -> String {
    match max - min {
        0 => min.to_string(),
        1 => format!("[{}{}]", min, max),
        _ => format!("[{}-{}]", min, max),
    }
}

pub(super) fn format_to_regex(format: &str) -> Option<String> {
    match format {
        "date" => Some(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01])$".to_string()),
        "time" => Some(
            r"^([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$"
                .to_string(),
        ),
        "date-time" => Some(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01])T([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$".to_string()),
        "email" => Some(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$".to_string()),
        "uuid" => Some(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$".to_string()),
        "ipv4" => Some(r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$".to_string()),
        "hostname" => Some(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$".to_string()),
        _ => None,
    }
}

pub(super) fn sanitize_rule_name(name: &str) -> String {
    let mut sanitized: String = name
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        sanitized.push_str("rule");
    }
    sanitized
}
