//! Regex grammar frontend.

mod parser;

use anyhow::Result;

use crate::frontend::FrontendExpr;
use crate::frontend::FrontendGrammar;
use crate::grammar::Grammar;

/// Convert a regex pattern directly to a grammar.
pub fn regex_to_grammar(pattern: &str) -> Result<Grammar> {
    FrontendGrammar::single_root(parser::parse(pattern)?).to_grammar()
}

/// Convert a regex pattern to a compatible EBNF representation.
pub fn regex_to_ebnf(pattern: &str) -> Result<String> {
    Ok(FrontendGrammar::single_root(parser::parse(pattern)?).to_ebnf())
}

pub(crate) fn regex_to_expr(pattern: &str) -> Result<FrontendExpr> {
    parser::parse(pattern)
}

/// A JSON string may not hold these raw, whatever a pattern says.
///
/// RFC 8259: the quotation mark, the reverse solidus and U+0000 through
/// U+001F must be escaped. A pattern's `.` means "any character", and a
/// regex has no idea it is being read inside a string - so `"\\d{2}.\\d{6}"`
/// matched a raw control byte and this engine emitted a document `json.loads`
/// refuses. Found by generating under the mask and parsing what came out,
/// which is the only way a bug of this shape is ever found.
const MUST_ESCAPE: &[(u32, u32)] = &[(0x00, 0x1f), (0x22, 0x22), (0x5c, 0x5c)];

/// Rewrite a pattern's expression so it can only match JSON string content.
///
/// A class loses the characters it may not hold; a literal gains the escape
/// that lets it keep them, so `pattern: "a\"b"` still matches the string it
/// means. Widening is what a lowering may do and narrowing is not, and this
/// narrows - but only over documents that were never JSON, which no caller
/// can want and no downstream check can repair.
pub(crate) fn within_json_string(expr: FrontendExpr) -> Result<FrontendExpr> {
    use anyhow::bail;

    Ok(match expr {
        FrontendExpr::CharacterClass { negated, ranges } => {
            let wanted = if negated {
                complement(&ranges)
            } else {
                ranges.clone()
            };
            let plain = subtract(&wanted, MUST_ESCAPE);
            // A character the class wants and a string may not hold raw is not
            // dropped, it is spelled the way JSON spells it. Dropping would
            // narrow, and narrowing is the one direction a lowering may not go
            // - `pattern: "a.c"` has to keep matching the string `a"c`.
            let mut alternatives = Vec::new();
            if !plain.is_empty() {
                alternatives.push(FrontendExpr::CharacterClass {
                    negated: false,
                    ranges: plain,
                });
            }
            if holds(&wanted, 0x22) {
                alternatives.push(FrontendExpr::literal("\\\""));
            }
            if holds(&wanted, 0x5c) {
                alternatives.push(FrontendExpr::literal("\\\\"));
            }
            for (code, escape) in [
                (0x08u32, "\\b"),
                (0x09, "\\t"),
                (0x0a, "\\n"),
                (0x0c, "\\f"),
                (0x0d, "\\r"),
            ] {
                if holds(&wanted, code) {
                    alternatives.push(FrontendExpr::literal(escape));
                }
            }
            // `\u00XX` for every control character the class wants. One
            // alternative with a class per nibble rather than thirty-two
            // literals, so the common case - a bare `.` - costs the lexer a
            // handful of states instead of a hundred.
            let controls: Vec<u32> = (0x00..=0x1f).filter(|c| holds(&wanted, *c)).collect();
            if !controls.is_empty() {
                let high: Vec<(u32, u32)> = controls
                    .iter()
                    .map(|code| (u32::from(hex(code >> 4)), u32::from(hex(code >> 4))))
                    .collect();
                let low: Vec<(u32, u32)> = controls
                    .iter()
                    .flat_map(|code| {
                        let digit = code & 0xf;
                        let lower = u32::from(hex(digit));
                        let upper = if digit > 9 { lower - 32 } else { lower };
                        [(lower, lower), (upper, upper)]
                    })
                    .collect();
                alternatives.push(FrontendExpr::sequence(vec![
                    FrontendExpr::literal("\\u00"),
                    FrontendExpr::CharacterClass {
                        negated: false,
                        ranges: normalize(high),
                    },
                    FrontendExpr::CharacterClass {
                        negated: false,
                        ranges: normalize(low),
                    },
                ]));
            }
            match alternatives.len() {
                0 => bail!("a pattern matches no character a JSON string can hold"),
                1 => alternatives.pop().expect("checked"),
                _ => FrontendExpr::Choice(alternatives),
            }
        }
        FrontendExpr::Literal(bytes) => {
            let mut escaped = Vec::with_capacity(bytes.len());
            for byte in bytes {
                match byte {
                    b'"' => escaped.extend_from_slice(b"\\\""),
                    b'\\' => escaped.extend_from_slice(b"\\\\"),
                    0x08 => escaped.extend_from_slice(b"\\b"),
                    0x09 => escaped.extend_from_slice(b"\\t"),
                    0x0a => escaped.extend_from_slice(b"\\n"),
                    0x0c => escaped.extend_from_slice(b"\\f"),
                    0x0d => escaped.extend_from_slice(b"\\r"),
                    0x00..=0x1f => {
                        escaped.extend_from_slice(format!("\\u{byte:04x}").as_bytes())
                    }
                    other => escaped.push(other),
                }
            }
            FrontendExpr::Literal(escaped)
        }
        FrontendExpr::Group(inner) => {
            FrontendExpr::Group(Box::new(within_json_string(*inner)?))
        }
        FrontendExpr::Sequence(parts) => FrontendExpr::Sequence(
            parts.into_iter().map(within_json_string).collect::<Result<_>>()?,
        ),
        FrontendExpr::Choice(parts) => FrontendExpr::Choice(
            parts.into_iter().map(within_json_string).collect::<Result<_>>()?,
        ),
        FrontendExpr::Repeat { expr, min, max } => FrontendExpr::Repeat {
            expr: Box::new(within_json_string(*expr)?),
            min,
            max,
        },
        other => other,
    })
}

fn complement(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable();
    let mut out = Vec::new();
    let mut next = 0u32;
    for (start, end) in sorted {
        if start > next {
            out.push((next, start - 1));
        }
        next = next.max(end.saturating_add(1));
    }
    if next <= 0x10ffff {
        out.push((next, 0x10ffff));
    }
    out
}

fn subtract(ranges: &[(u32, u32)], removed: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = ranges.to_vec();
    for &(low, high) in removed {
        let mut next = Vec::new();
        for (start, end) in out {
            if end < low || start > high {
                next.push((start, end));
                continue;
            }
            if start < low {
                next.push((start, low - 1));
            }
            if end > high {
                next.push((high + 1, end));
            }
        }
        out = next;
    }
    out
}

fn holds(ranges: &[(u32, u32)], code: u32) -> bool {
    ranges.iter().any(|(start, end)| (*start..=*end).contains(&code))
}

fn hex(digit: u32) -> u8 {
    b"0123456789abcdef"[digit as usize & 0xf]
}

fn normalize(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.sort_unstable();
    ranges.dedup();
    ranges
}
