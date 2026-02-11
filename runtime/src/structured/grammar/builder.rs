use anyhow::{Result, bail};

use super::{Expr, ExprId, Grammar, Lookahead, Rule, RuleId};

/// Programmatic grammar construction.
///
/// # Example
/// ```
/// use pie_grammar::grammar::builder::GrammarBuilder;
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

    /// Add a new rule with no body yet. Returns the rule id.
    /// The body must be set later with `set_rule_body`.
    pub fn add_rule(&mut self, name: &str) -> RuleId {
        let id = RuleId(self.rules.len() as u32);
        self.rules.push(Rule {
            name: name.to_string(),
            body: ExprId(u32::MAX), // sentinel, must be filled
            lookahead: None,
        });
        id
    }

    /// Set the body expression of a rule.
    pub fn set_rule_body(&mut self, rule: RuleId, body: ExprId) {
        self.rules[rule.0 as usize].body = body;
    }

    /// Set a lookahead assertion on a rule.
    pub fn set_rule_lookahead(&mut self, rule: RuleId, expr: ExprId, is_exact: bool) {
        self.rules[rule.0 as usize].lookahead = Some(Lookahead { expr, is_exact });
    }

    /// Add an expression to the arena and return its id.
    pub fn add_expr(&mut self, expr: Expr) -> ExprId {
        let id = ExprId(self.exprs.len() as u32);
        self.exprs.push(expr);
        id
    }

    /// Add an empty string expression.
    pub fn add_empty_string(&mut self) -> ExprId {
        self.add_expr(Expr::EmptyString)
    }

    /// Add a byte string expression.
    pub fn add_byte_string(&mut self, bytes: &[u8]) -> ExprId {
        self.add_expr(Expr::ByteString(bytes.to_vec()))
    }

    /// Add a character class expression.
    pub fn add_character_class(&mut self, negated: bool, ranges: Vec<(u32, u32)>) -> ExprId {
        self.add_expr(Expr::CharacterClass { negated, ranges })
    }

    /// Add a character class star expression (`[...]*`).
    pub fn add_character_class_star(&mut self, negated: bool, ranges: Vec<(u32, u32)>) -> ExprId {
        self.add_expr(Expr::CharacterClassStar { negated, ranges })
    }

    /// Add a rule reference expression.
    pub fn add_rule_ref(&mut self, rule: RuleId) -> ExprId {
        self.add_expr(Expr::RuleRef(rule))
    }

    /// Add a sequence expression (concatenation).
    pub fn add_sequence(&mut self, exprs: Vec<ExprId>) -> ExprId {
        self.add_expr(Expr::Sequence(exprs))
    }

    /// Add a choices expression (alternation).
    pub fn add_choices(&mut self, exprs: Vec<ExprId>) -> ExprId {
        self.add_expr(Expr::Choices(exprs))
    }

    /// Add a repeat expression.
    pub fn add_repeat(&mut self, rule: RuleId, min: u32, max: Option<u32>) -> ExprId {
        self.add_expr(Expr::Repeat { rule, min, max })
    }

    /// Look up a rule by name.
    pub fn find_rule(&self, name: &str) -> Option<RuleId> {
        self.rules
            .iter()
            .position(|r| r.name == name)
            .map(|i| RuleId(i as u32))
    }

    /// Number of rules added so far.
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Build the grammar, resolving the root rule by name.
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

        // Validate all rules have bodies set
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
    fn test_build_grammar_with_choices() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let a = b.add_byte_string(b"a");
        let bc = b.add_byte_string(b"bc");
        let choices = b.add_choices(vec![a, bc]);
        b.set_rule_body(root, choices);

        let grammar = b.build("root").unwrap();
        match grammar.get_expr(grammar.root().body) {
            Expr::Choices(exprs) => {
                assert_eq!(exprs.len(), 2);
                match grammar.get_expr(exprs[0]) {
                    Expr::ByteString(bytes) => assert_eq!(bytes, b"a"),
                    other => panic!("expected ByteString, got {:?}", other),
                }
                match grammar.get_expr(exprs[1]) {
                    Expr::ByteString(bytes) => assert_eq!(bytes, b"bc"),
                    other => panic!("expected ByteString, got {:?}", other),
                }
            }
            other => panic!("expected Choices, got {:?}", other),
        }
    }

    #[test]
    fn test_build_grammar_with_sequence_and_rule_ref() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let digit = b.add_rule("digit");

        let digit_class = b.add_character_class(false, vec![(0x30, 0x39)]); // [0-9]
        b.set_rule_body(digit, digit_class);

        let hello = b.add_byte_string(b"num:");
        let digit_ref = b.add_rule_ref(digit);
        let seq = b.add_sequence(vec![hello, digit_ref]);
        b.set_rule_body(root, seq);

        let grammar = b.build("root").unwrap();
        assert_eq!(grammar.num_rules(), 2);

        match grammar.get_expr(grammar.root().body) {
            Expr::Sequence(exprs) => assert_eq!(exprs.len(), 2),
            other => panic!("expected Sequence, got {:?}", other),
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

    #[test]
    fn test_display_grammar() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let digit = b.add_rule("digit");

        let digit_class = b.add_character_class(false, vec![(0x30, 0x39)]);
        b.set_rule_body(digit, digit_class);

        let a = b.add_byte_string(b"a");
        let digit_ref = b.add_rule_ref(digit);
        let seq = b.add_sequence(vec![a, digit_ref]);
        let empty = b.add_empty_string();
        let choices = b.add_choices(vec![seq, empty]);
        b.set_rule_body(root, choices);

        let grammar = b.build("root").unwrap();
        let s = grammar.to_string();
        assert_eq!(s, "root ::= ((\"a\" digit) | \"\")\ndigit ::= [0-9]");
    }

    #[test]
    fn test_character_class_star() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let star = b.add_character_class_star(false, vec![(0x61, 0x7a)]);
        b.set_rule_body(root, star);

        let grammar = b.build("root").unwrap();
        let s = grammar.to_string();
        assert_eq!(s, "root ::= [a-z]*");
    }

    #[test]
    fn test_negated_character_class() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let cc = b.add_character_class(true, vec![(0x61, 0x7a)]);
        b.set_rule_body(root, cc);

        let grammar = b.build("root").unwrap();
        let s = grammar.to_string();
        assert_eq!(s, "root ::= [^a-z]");
    }

    #[test]
    fn test_find_rule() {
        let mut b = GrammarBuilder::new();
        b.add_rule("root");
        b.add_rule("digit");

        assert_eq!(b.find_rule("root"), Some(RuleId(0)));
        assert_eq!(b.find_rule("digit"), Some(RuleId(1)));
        assert_eq!(b.find_rule("missing"), None);
    }
}
