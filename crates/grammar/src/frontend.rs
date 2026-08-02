//! Structured frontend IR shared by non-EBNF grammar sources.

use std::collections::HashMap;

use anyhow::{Result, anyhow};

use crate::grammar::builder::GrammarBuilder;
use crate::grammar::{ExprId, Grammar, RuleId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FrontendExpr {
    Empty,
    Literal(Vec<u8>),
    CharacterClass {
        negated: bool,
        ranges: Vec<(u32, u32)>,
    },
    RuleRef(String),
    Group(Box<FrontendExpr>),
    Sequence(Vec<FrontendExpr>),
    Choice(Vec<FrontendExpr>),
    Repeat {
        expr: Box<FrontendExpr>,
        min: u32,
        max: Option<u32>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FrontendRule {
    pub(crate) name: String,
    pub(crate) body: FrontendExpr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FrontendGrammar {
    pub(crate) rules: Vec<FrontendRule>,
    pub(crate) root: String,
}

impl FrontendExpr {
    pub(crate) fn literal(value: impl Into<Vec<u8>>) -> Self {
        Self::Literal(value.into())
    }

    pub(crate) fn sequence(elements: Vec<Self>) -> Self {
        let mut flattened = Vec::new();
        for element in elements {
            match element {
                Self::Empty => {}
                Self::Sequence(inner) => flattened.extend(inner),
                other => flattened.push(other),
            }
        }
        match flattened.len() {
            0 => Self::Empty,
            1 => flattened.pop().unwrap(),
            _ => Self::Sequence(flattened),
        }
    }

    pub(crate) fn choice(alternatives: Vec<Self>) -> Self {
        let mut flattened = Vec::new();
        for alternative in alternatives {
            match alternative {
                Self::Choice(inner) => flattened.extend(inner),
                other => flattened.push(other),
            }
        }
        match flattened.len() {
            0 => Self::Empty,
            1 => flattened.pop().unwrap(),
            _ => Self::Choice(flattened),
        }
    }

    pub(crate) fn repeat(self, min: u32, max: Option<u32>) -> Self {
        Self::Repeat {
            expr: Box::new(self),
            min,
            max,
        }
    }

    pub(crate) fn to_ebnf(&self) -> String {
        match self {
            Self::Empty => "\"\"".to_string(),
            Self::Literal(bytes) => quote_bytes(bytes),
            Self::CharacterClass { negated, ranges } => {
                let mut result = String::from("[");
                if *negated {
                    result.push('^');
                }
                for &(start, end) in ranges {
                    push_class_codepoint(&mut result, start);
                    if start != end {
                        result.push('-');
                        push_class_codepoint(&mut result, end);
                    }
                }
                result.push(']');
                result
            }
            Self::RuleRef(name) => name.clone(),
            Self::Group(expr) => format!("({})", expr.to_ebnf()),
            Self::Sequence(elements) => elements
                .iter()
                .map(Self::to_ebnf)
                .collect::<Vec<_>>()
                .join(" "),
            Self::Choice(alternatives) => format!(
                "({})",
                alternatives
                    .iter()
                    .map(Self::to_ebnf)
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Self::Repeat { expr, min, max } => {
                let inner = match expr.as_ref() {
                    Self::Empty
                    | Self::Literal(_)
                    | Self::CharacterClass { .. }
                    | Self::RuleRef(_)
                    | Self::Group(_) => expr.to_ebnf(),
                    _ => format!("({})", expr.to_ebnf()),
                };
                let suffix = match (*min, *max) {
                    (0, None) => "*".to_string(),
                    (1, None) => "+".to_string(),
                    (0, Some(1)) => "?".to_string(),
                    (min, None) => format!("{{{},}}", min),
                    (min, Some(max)) if min == max => format!("{{{}}}", min),
                    (min, Some(max)) => format!("{{{},{}}}", min, max),
                };
                format!("{}{}", inner, suffix)
            }
        }
    }
}

impl FrontendGrammar {
    pub(crate) fn single_root(body: FrontendExpr) -> Self {
        Self {
            rules: vec![FrontendRule {
                name: "root".to_string(),
                body,
            }],
            root: "root".to_string(),
        }
    }

    pub(crate) fn to_grammar(&self) -> Result<Grammar> {
        Lowerer::new(self).lower()
    }

    pub(crate) fn to_ebnf(&self) -> String {
        let mut result = String::new();
        for rule in &self.rules {
            result.push_str(&rule.name);
            result.push_str(" ::= ");
            result.push_str(&rule.body.to_ebnf());
            result.push('\n');
        }
        result
    }
}

struct Lowerer<'a> {
    grammar: &'a FrontendGrammar,
    builder: GrammarBuilder,
    rules: HashMap<&'a str, RuleId>,
    aux_counter: usize,
}

impl<'a> Lowerer<'a> {
    fn new(grammar: &'a FrontendGrammar) -> Self {
        Self {
            grammar,
            builder: GrammarBuilder::new(),
            rules: HashMap::new(),
            aux_counter: 0,
        }
    }

    fn lower(mut self) -> Result<Grammar> {
        for rule in &self.grammar.rules {
            if self.rules.contains_key(rule.name.as_str()) {
                return Err(anyhow!("duplicate frontend rule '{}'", rule.name));
            }
            let id = self.builder.add_rule(&rule.name);
            self.rules.insert(&rule.name, id);
        }

        for rule in &self.grammar.rules {
            let id = self.rules[rule.name.as_str()];
            let body = self.lower_expr(&rule.body)?;
            self.builder.set_rule_body(id, body);
        }
        self.builder.build(&self.grammar.root)
    }

    fn lower_expr(&mut self, expr: &FrontendExpr) -> Result<ExprId> {
        match expr {
            FrontendExpr::Empty => Ok(self.builder.add_empty_string()),
            FrontendExpr::Literal(bytes) => Ok(self.builder.add_byte_string(bytes)),
            FrontendExpr::CharacterClass { negated, ranges } => {
                Ok(self.builder.add_character_class(*negated, ranges.clone()))
            }
            FrontendExpr::RuleRef(name) => self
                .rules
                .get(name.as_str())
                .copied()
                .map(|rule| self.builder.add_rule_ref(rule))
                .ok_or_else(|| anyhow!("undefined frontend rule '{}'", name)),
            FrontendExpr::Group(expr) => self.lower_expr(expr),
            FrontendExpr::Sequence(elements) => {
                let elements = elements
                    .iter()
                    .map(|element| self.lower_expr(element))
                    .collect::<Result<Vec<_>>>()?;
                Ok(self.builder.add_sequence(elements))
            }
            FrontendExpr::Choice(alternatives) => {
                let alternatives = alternatives
                    .iter()
                    .map(|alternative| self.lower_expr(alternative))
                    .collect::<Result<Vec<_>>>()?;
                Ok(self.builder.add_choices(alternatives))
            }
            FrontendExpr::Repeat { expr, min, max } => {
                if *min == 0 && max.is_none() {
                    if let FrontendExpr::CharacterClass { negated, ranges } = expr.as_ref() {
                        return Ok(self
                            .builder
                            .add_character_class_star(*negated, ranges.clone()));
                    }
                }
                let repeated_rule = match expr.as_ref() {
                    FrontendExpr::RuleRef(name) => self
                        .rules
                        .get(name.as_str())
                        .copied()
                        .ok_or_else(|| anyhow!("undefined frontend rule '{}'", name))?,
                    _ => {
                        let name = self.next_aux_name();
                        let rule = self.builder.add_rule(&name);
                        let body = self.lower_expr(expr)?;
                        self.builder.set_rule_body(rule, body);
                        rule
                    }
                };
                Ok(self.builder.add_repeat(repeated_rule, *min, *max))
            }
        }
    }

    fn next_aux_name(&mut self) -> String {
        loop {
            self.aux_counter += 1;
            let name = format!("__frontend_{}", self.aux_counter);
            if self.builder.find_rule(&name).is_none() {
                return name;
            }
        }
    }
}

fn quote_bytes(bytes: &[u8]) -> String {
    let mut result = String::from("\"");
    for &byte in bytes {
        match byte {
            b'\\' => result.push_str("\\\\"),
            b'"' => result.push_str("\\\""),
            b'\n' => result.push_str("\\n"),
            b'\r' => result.push_str("\\r"),
            b'\t' => result.push_str("\\t"),
            0x20..=0x7e => result.push(byte as char),
            _ => result.push_str(&format!("\\x{:02x}", byte)),
        }
    }
    result.push('"');
    result
}

fn push_class_codepoint(result: &mut String, codepoint: u32) {
    match codepoint {
        0x5c => result.push_str("\\\\"),
        0x5d => result.push_str("\\]"),
        0x5e => result.push_str("\\^"),
        0x2d => result.push_str("\\-"),
        0x0a => result.push_str("\\n"),
        0x0d => result.push_str("\\r"),
        0x09 => result.push_str("\\t"),
        value if (0x20..=0x7e).contains(&value) => {
            result.push(char::from_u32(value).unwrap());
        }
        value if value <= 0xffff => result.push_str(&format!("\\u{:04x}", value)),
        value => result.push_str(&format!("\\U{:08x}", value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_and_renders_shared_structure() {
        let grammar = FrontendGrammar::single_root(FrontendExpr::sequence(vec![
            FrontendExpr::literal(b"a".to_vec()),
            FrontendExpr::CharacterClass {
                negated: false,
                ranges: vec![(b'0' as u32, b'9' as u32)],
            }
            .repeat(1, None),
        ]));

        let lowered = grammar.to_grammar().unwrap();
        assert_eq!(
            lowered.to_string(),
            "root ::= (\"a\" __frontend_1{1,})\n__frontend_1 ::= [0-9]"
        );
        assert_eq!(grammar.to_ebnf(), "root ::= \"a\" [0-9]+\n");
    }
}
