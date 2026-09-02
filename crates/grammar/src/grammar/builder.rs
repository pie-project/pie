use anyhow::{Result, bail};

use super::{Expr, ExprId, Grammar, Rule, RuleId};

/// Programmatic grammar construction.
///
/// # Example
/// ```ignore
/// use ::grammar::grammar::builder::GrammarBuilder;
///
/// let mut b = GrammarBuilder::new();
/// let root = b.add_rule("root");
/// let hello = b.add_byte_string(b"hello");
/// b.set_rule_body(root, hello);
/// let grammar = b.build("root").unwrap();
/// assert_eq!(grammar.num_rules(), 1);
/// ```
pub struct GrammarBuilder {
    pub(crate) rules: Vec<Rule>,
    pub(crate) exprs: Vec<Expr>,
}

impl GrammarBuilder {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            exprs: Vec::new(),
        }
    }

    /// Add a new rule with no body; the body must be set later with
    /// `set_rule_body`.
    pub fn add_rule(&mut self, name: &str) -> RuleId {
        let id = RuleId(self.rules.len() as u32);
        self.rules.push(Rule {
            name: name.to_string(),
            body: ExprId(u32::MAX), // sentinel, must be filled
        });
        id
    }

    pub fn set_rule_body(&mut self, rule: RuleId, body: ExprId) {
        self.rules[rule.0 as usize].body = body;
    }

    pub fn add_expr(&mut self, expr: Expr) -> ExprId {
        let id = ExprId(self.exprs.len() as u32);
        self.exprs.push(expr);
        id
    }

    pub fn add_empty_string(&mut self) -> ExprId {
        self.add_expr(Expr::EmptyString)
    }

    pub fn add_byte_string(&mut self, bytes: &[u8]) -> ExprId {
        self.add_expr(Expr::ByteString(bytes.to_vec()))
    }

    pub fn add_character_class(&mut self, negated: bool, ranges: Vec<(u32, u32)>) -> ExprId {
        self.add_expr(Expr::CharacterClass { negated, ranges })
    }

    /// `[...]*`.
    pub fn add_character_class_star(&mut self, negated: bool, ranges: Vec<(u32, u32)>) -> ExprId {
        self.add_expr(Expr::CharacterClassStar { negated, ranges })
    }

    pub fn add_rule_ref(&mut self, rule: RuleId) -> ExprId {
        self.add_expr(Expr::RuleRef(rule))
    }

    pub fn add_sequence(&mut self, exprs: Vec<ExprId>) -> ExprId {
        self.add_expr(Expr::Sequence(exprs))
    }

    pub fn add_choices(&mut self, exprs: Vec<ExprId>) -> ExprId {
        self.add_expr(Expr::Choices(exprs))
    }

    pub fn add_repeat(&mut self, rule: RuleId, min: u32, max: Option<u32>) -> ExprId {
        self.add_expr(Expr::Repeat { rule, min, max })
    }

    pub fn find_rule(&self, name: &str) -> Option<RuleId> {
        self.rules
            .iter()
            .position(|r| r.name == name)
            .map(|i| RuleId(i as u32))
    }

    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    pub fn build(self, root_rule_name: &str) -> Result<Grammar> {
        let root_rule = self
            .rules
            .iter()
            .position(|r| r.name == root_rule_name)
            .map(|i| RuleId(i as u32));

        let root_rule = match root_rule {
            Some(id) => id,
            None => bail!("root rule '{}' not found", root_rule_name),
        };

        for rule in &self.rules {
            if rule.body == ExprId(u32::MAX) {
                bail!("rule '{}' has no body", rule.name);
            }
        }

        Ok(Grammar {
            rules: self.rules,
            exprs: self.exprs,
            root_rule,
        })
    }
}

impl Default for GrammarBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_grammar() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let hello = b.add_byte_string(b"hello");
        b.set_rule_body(root, hello);

        let grammar = b.build("root").unwrap();
        assert_eq!(grammar.num_rules(), 1);
        assert_eq!(grammar.root_rule(), RuleId(0));
        assert_eq!(grammar.root().name, "root");

        match grammar.get_expr(grammar.root().body) {
            Expr::ByteString(bytes) => assert_eq!(bytes, b"hello"),
            other => panic!("expected ByteString, got {:?}", other),
        }
    }

    #[test]
    fn test_build_missing_root() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("main");
        let e = b.add_empty_string();
        b.set_rule_body(root, e);

        let result = b.build("root");
        assert!(result.is_err());
    }

    #[test]
    fn test_build_missing_body() {
        let mut b = GrammarBuilder::new();
        b.add_rule("root"); // no body set
        let result = b.build("root");
        assert!(result.is_err());
    }

}
