//! Sampling constraints for grammar-guided generation.
//!
//! The common path is declarative: pass a [`Schema`] to
//! [`Generator::constrain_with`](crate::generation::Generator::constrain_with)
//! and the SDK compiles the matcher for you. For custom logic (banned
//! tokens, learned constraints, etc.), implement [`Constrain`] and pass
//! it via [`Generator::constrain`](crate::generation::Generator::constrain).
//! Constraints compose — every applied constraint contributes a mask,
//! and the masks are AND-ed before each forward pass.

use crate::inference::Grammar;
use crate::model::Model;
use crate::Result;
use crate::pie::core::inference::Matcher;

/// Token sampling constraint.
///
/// On each generation step, the [`Generator`](crate::generation::Generator)
/// passes any newly accepted tokens (or `&[]` on the first step) and gets
/// back the BRLE-encoded logit mask for the next position.
///
/// Returning `&[]` (empty mask) means "no restriction" and is treated as
/// transparent during composition.
pub trait Constrain: Send {
    /// Advance internal state with `accepted` tokens, then return the mask
    /// for the next position.
    fn step(&mut self, accepted: &[u32]) -> &[u32];
}

/// Declarative description of a constraint.
///
/// Implementors are passed to
/// [`Generator::constrain_with`](crate::generation::Generator::constrain_with),
/// which compiles them into a [`GrammarConstraint`].
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
///     fn build_constraint(&self, model: &Model) -> Result<GrammarConstraint> {
///         let g = compile_to_pie_grammar(&self.0)?;
///         Ok(GrammarConstraint::from_grammar(&g, model))
///     }
/// }
///
/// ctx.generate(Sampler::Argmax)
///     .constrain_with(MyGrammar(my_lark_source))?
///     .collect_text().await?;
/// ```
pub trait Schema {
    /// Compile this schema into a [`GrammarConstraint`] for `model`.
    fn build_constraint(&self, model: &Model) -> Result<GrammarConstraint>;
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
    fn build_constraint(&self, model: &Model) -> Result<GrammarConstraint> {
        GrammarConstraint::from_json_schema(self.0, model)
    }
}

impl Schema for AnyJson {
    fn build_constraint(&self, model: &Model) -> Result<GrammarConstraint> {
        Ok(GrammarConstraint::json(model))
    }
}

impl Schema for Regex<'_> {
    fn build_constraint(&self, model: &Model) -> Result<GrammarConstraint> {
        GrammarConstraint::from_regex(self.0, model)
    }
}

impl Schema for Ebnf<'_> {
    fn build_constraint(&self, model: &Model) -> Result<GrammarConstraint> {
        GrammarConstraint::from_ebnf(self.0, model)
    }
}

impl Schema for &Grammar {
    fn build_constraint(&self, model: &Model) -> Result<GrammarConstraint> {
        Ok(GrammarConstraint::from_grammar(self, model))
    }
}

/// Grammar-driven [`Constrain`] backed by a host [`Matcher`].
///
/// Most callers should reach for [`Schema`] instead — `GrammarConstraint`
/// is the lower-level type for callers that want to keep a constraint
/// instance around (e.g., to compose with [`Generator::constrain`](crate::generation::Generator::constrain)).
pub struct GrammarConstraint {
    matcher: Matcher,
    /// Cached mask returned from the last `step()`. Reused so `step` can
    /// hand back a `&[u32]` without forcing the caller to clone.
    cached_mask: Vec<u32>,
}

impl GrammarConstraint {
    /// Wrap an existing [`Matcher`].
    pub fn new(matcher: Matcher) -> Self {
        Self { matcher, cached_mask: Vec::new() }
    }

    /// Build from a pre-compiled grammar (compile once, reuse across contexts).
    pub fn from_grammar(grammar: &Grammar, model: &Model) -> Self {
        let tokenizer = model.tokenizer();
        Self::new(Matcher::new(grammar, &tokenizer))
    }

    /// Build a constraint that accepts any valid JSON.
    pub fn json(model: &Model) -> Self {
        Self::from_grammar(&Grammar::json(), model)
    }

    /// Build from a JSON Schema string.
    pub fn from_json_schema(schema: &str, model: &Model) -> Result<Self> {
        let grammar = Grammar::from_json_schema(schema)?;
        Ok(Self::from_grammar(&grammar, model))
    }

    /// Build from a regular expression pattern.
    pub fn from_regex(pattern: &str, model: &Model) -> Result<Self> {
        let grammar = Grammar::from_regex(pattern)?;
        Ok(Self::from_grammar(&grammar, model))
    }

    /// Build from an EBNF grammar string.
    pub fn from_ebnf(ebnf: &str, model: &Model) -> Result<Self> {
        let grammar = Grammar::from_ebnf(ebnf)?;
        Ok(Self::from_grammar(&grammar, model))
    }
}

impl Constrain for GrammarConstraint {
    fn step(&mut self, accepted: &[u32]) -> &[u32] {
        if !accepted.is_empty() {
            let _ = self.matcher.accept_tokens(accepted);
        }
        self.cached_mask = self.matcher.next_token_logit_mask();
        &self.cached_mask
    }
}

// =============================================================================
// BRLE intersection (mask AND)
// =============================================================================

/// AND two BRLE-encoded masks of equal length.
///
/// Both inputs must encode the same total bit count (typically the model's
/// vocabulary size). Returns the BRLE encoding of the bitwise AND.
///
/// BRLE format: `buffer[0]` is a (possibly zero) run of `false`, `buffer[1]`
/// is a run of `true`, and so on.
pub(crate) fn brle_and(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<u32> = Vec::with_capacity(a.len().max(b.len()));
    let mut a_idx = 0usize;
    let mut b_idx = 0usize;
    let mut a_left = a[0];
    let mut b_left = b[0];
    let mut a_value = false;
    let mut b_value = false;

    // Output state. The first emitted run is always a `false` run by
    // convention (may be zero-length).
    let mut want_value = false;
    let mut accum: u32 = 0;

    loop {
        let take = a_left.min(b_left);
        let result = a_value && b_value;

        if result == want_value {
            accum += take;
        } else {
            out.push(accum);
            accum = take;
            want_value = !want_value;
        }

        a_left -= take;
        b_left -= take;

        if a_left == 0 {
            a_idx += 1;
            if a_idx == a.len() {
                break;
            }
            a_left = a[a_idx];
            a_value = !a_value;
        }
        if b_left == 0 {
            b_idx += 1;
            if b_idx == b.len() {
                break;
            }
            b_left = b[b_idx];
            b_value = !b_value;
        }
    }

    out.push(accum);
    out
}

#[cfg(test)]
mod tests {
    use super::brle_and;

    /// Decode a BRLE buffer into a `Vec<bool>` for comparison.
    fn decode(buf: &[u32]) -> Vec<bool> {
        let mut out = Vec::new();
        let mut value = false;
        for &run in buf {
            for _ in 0..run {
                out.push(value);
            }
            value = !value;
        }
        out
    }

    fn encode(bits: &[bool]) -> Vec<u32> {
        let mut buf = Vec::new();
        if bits.is_empty() {
            return buf;
        }
        let mut current = false;
        let mut count = 0u32;
        if bits[0] {
            buf.push(0);
            current = true;
        }
        for &b in bits {
            if b == current {
                count += 1;
            } else {
                buf.push(count);
                current = b;
                count = 1;
            }
        }
        buf.push(count);
        buf
    }

    fn check(a: Vec<bool>, b: Vec<bool>) {
        assert_eq!(a.len(), b.len());
        let expected: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x && *y).collect();
        let result = brle_and(&encode(&a), &encode(&b));
        assert_eq!(decode(&result), expected, "a={:?} b={:?}", a, b);
    }

    #[test]
    fn all_false() {
        check(vec![false; 8], vec![false; 8]);
    }

    #[test]
    fn all_true_and_all_true() {
        check(vec![true; 8], vec![true; 8]);
    }

    #[test]
    fn all_true_and_all_false() {
        check(vec![true; 8], vec![false; 8]);
    }

    #[test]
    fn alternating() {
        let a = vec![true, false, true, false, true, false, true, false];
        let b = vec![false, true, false, true, false, true, false, true];
        check(a, b);
    }

    #[test]
    fn mixed_runs() {
        let a = vec![true, true, false, false, true, true, true, false];
        let b = vec![false, true, true, false, true, false, true, true];
        check(a, b);
    }

    #[test]
    fn leading_true() {
        let a = vec![true, true, true, false, false];
        let b = vec![true, false, true, true, true];
        check(a, b);
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(brle_and(&[], &[]), Vec::<u32>::new());
    }
}
