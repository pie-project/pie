//! Sampling constraints for grammar-guided generation.
//!
//! The common path is declarative: pass a [`Schema`] to a decode loop's
//! constrain step and the SDK compiles the matcher for you. For custom
//! logic (banned tokens, learned constraints, etc.), implement
//! [`Constrain`] and apply its mask each step.
//! Constraints compose — every applied constraint contributes a mask,
//! and the masks are AND-ed before each forward pass.

use crate::Result;
use crate::inference::Grammar;
use crate::pie::core::inference::Matcher;

/// Token sampling constraint.
///
/// On each generation step the decode loop
/// calls [`advance`](Constrain::advance) with the tokens accepted last step
/// (or `&[]` on the first step), then reads [`mask`](Constrain::mask) — the
/// packed allowed-token bitmask for the next position. Multiple constraints
/// compose by word-wise bitwise-AND of their masks.
pub trait Constrain: Send {
    /// Advance internal state with the tokens accepted last step.
    fn advance(&mut self, accepted: &[u32]);

    /// The packed allowed-token bitmask for the next position: a
    /// `[ceil(vocab/32)]` u32 array where bit `j` is 1 iff token `j` is
    /// allowed. An empty `Vec` means "no restriction" (transparent during
    /// composition).
    fn mask(&self) -> Vec<u32>;
}

/// Declarative description of a constraint.
///
/// Implementors are compiled by the SDK
/// into a [`GrammarConstraint`].
///
/// Built-in implementors:
///
/// - [`JsonSchema`] — JSON conforming to a schema string
/// - [`AnyJson`]    — any valid JSON
/// - [`Regex`]      — strings matching a regex pattern
/// - [`Ebnf`]       — custom EBNF grammar
/// - `&Grammar`     — a pre-compiled grammar resource
///
/// User code can implement `Schema` for custom grammar sources (Lark,
/// GBNF, ProtoBuf, …):
///
/// ```ignore
/// struct MyGrammar(String);
/// impl inferlet::Schema for MyGrammar {
///     fn build_constraint(&self) -> Result<GrammarConstraint> {
///         let g = compile_to_pie_grammar(&self.0)?;
///         Ok(GrammarConstraint::from_grammar(&g))
///     }
/// }
///
/// ctx.generate(Sampler::Argmax)
///     .constrain_with(MyGrammar(my_lark_source))?
///     .collect_text().await?;
/// ```
pub trait Schema {
    /// Compile this schema into a [`GrammarConstraint`] for the bound model.
    fn build_constraint(&self) -> Result<GrammarConstraint>;
}

/// JSON conforming to a JSON Schema string.
pub struct JsonSchema<'a>(pub &'a str);

/// Any valid JSON value.
pub struct AnyJson;

/// Strings matching a regular expression pattern.
pub struct Regex<'a>(pub &'a str);

/// A custom EBNF grammar.
pub struct Ebnf<'a>(pub &'a str);

impl Schema for JsonSchema<'_> {
    fn build_constraint(&self) -> Result<GrammarConstraint> {
        GrammarConstraint::from_json_schema(self.0)
    }
}

impl Schema for AnyJson {
    fn build_constraint(&self) -> Result<GrammarConstraint> {
        Ok(GrammarConstraint::json())
    }
}

impl Schema for Regex<'_> {
    fn build_constraint(&self) -> Result<GrammarConstraint> {
        GrammarConstraint::from_regex(self.0)
    }
}

impl Schema for Ebnf<'_> {
    fn build_constraint(&self) -> Result<GrammarConstraint> {
        GrammarConstraint::from_ebnf(self.0)
    }
}

impl Schema for &Grammar {
    fn build_constraint(&self) -> Result<GrammarConstraint> {
        Ok(GrammarConstraint::from_grammar(self))
    }
}

/// Grammar-driven [`Constrain`] backed by a host [`Matcher`].
///
/// Most callers should reach for [`Schema`] instead — `GrammarConstraint`
/// is the lower-level type for callers that want to keep a constraint
/// instance around (e.g., to compose with a decode loop's constrain step).
pub struct GrammarConstraint {
    matcher: Matcher,
}

impl GrammarConstraint {
    /// Wrap an existing [`Matcher`].
    pub fn new(matcher: Matcher) -> Self {
        Self { matcher }
    }

    /// Build from a pre-compiled grammar (compile once, reuse across contexts).
    pub fn from_grammar(grammar: &Grammar) -> Self {
        Self::new(Matcher::new(grammar))
    }

    /// Build a constraint that accepts any valid JSON.
    pub fn json() -> Self {
        Self::from_grammar(&Grammar::json())
    }

    /// Build from a JSON Schema string.
    pub fn from_json_schema(schema: &str) -> Result<Self> {
        let grammar = Grammar::from_json_schema(schema)?;
        Ok(Self::from_grammar(&grammar))
    }

    /// Build from a regular expression pattern.
    pub fn from_regex(pattern: &str) -> Result<Self> {
        let grammar = Grammar::from_regex(pattern)?;
        Ok(Self::from_grammar(&grammar))
    }

    /// Build from an EBNF grammar string.
    pub fn from_ebnf(ebnf: &str) -> Result<Self> {
        let grammar = Grammar::from_ebnf(ebnf)?;
        Ok(Self::from_grammar(&grammar))
    }

    /// Whether the grammar has reached an accepting terminal state (e.g. a
    /// JSON-Schema object fully closed). Thin passthrough over the host
    /// `Matcher::is_terminated`; lets a sequential decode loop end the current
    /// step the moment the structured output completes.
    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }
}

impl Constrain for GrammarConstraint {
    fn advance(&mut self, accepted: &[u32]) {
        if !accepted.is_empty() {
            let _ = self.matcher.accept_tokens(accepted);
        }
    }

    fn mask(&self) -> Vec<u32> {
        self.matcher.mask()
    }
}

