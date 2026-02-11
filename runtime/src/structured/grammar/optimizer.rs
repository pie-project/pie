use std::collections::{HashSet, VecDeque};

use super::{Expr, ExprId, Grammar, Rule, RuleId};

impl Grammar {
    /// Optimize the grammar by inlining single-use rules and eliminating dead code.
    pub fn optimize(&self) -> Grammar {
        let inlined = self.inline_single_use_rules();
        inlined.eliminate_dead_rules()
    }

    /// Remove rules that are not reachable from the root rule.
    pub fn eliminate_dead_rules(&self) -> Grammar {
        let reachable = self.find_reachable_rules();

        if reachable.len() == self.rules.len() {
            return self.clone();
        }

        // Build remapping: old_rule_id -> new_rule_id
        let mut remap: Vec<Option<RuleId>> = vec![None; self.rules.len()];
        let mut new_idx = 0u32;
        for i in 0..self.rules.len() {
            if reachable.contains(&RuleId(i as u32)) {
                remap[i] = Some(RuleId(new_idx));
                new_idx += 1;
            }
        }

        // Clone and update exprs (remap RuleId references)
        let new_exprs: Vec<Expr> = self
            .exprs
            .iter()
            .map(|expr| match expr {
                Expr::RuleRef(rule_id) => match remap[rule_id.0 as usize] {
                    Some(new_id) => Expr::RuleRef(new_id),
                    None => expr.clone(),
                },
                Expr::Repeat { rule, min, max } => match remap[rule.0 as usize] {
                    Some(new_id) => Expr::Repeat {
                        rule: new_id,
                        min: *min,
                        max: *max,
                    },
                    None => expr.clone(),
                },
                _ => expr.clone(),
            })
            .collect();

        // Build new rules (only reachable ones)
        let new_rules: Vec<Rule> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(i, _)| reachable.contains(&RuleId(*i as u32)))
            .map(|(_, rule)| rule.clone())
            .collect();

        let new_root = remap[self.root_rule.0 as usize].expect("root rule must be reachable");

        Grammar {
            rules: new_rules,
            exprs: new_exprs,
            root_rule: new_root,
        }
    }

    /// Inline rules that are referenced exactly once via `RuleRef` (not `Repeat`).
    ///
    /// The inlined rules remain in the grammar but become dead code,
    /// removable by a subsequent `eliminate_dead_rules()` call.
    pub fn inline_single_use_rules(&self) -> Grammar {
        // Count references to each rule and track which are used in Repeat
        let mut ref_counts = vec![0u32; self.rules.len()];
        let mut repeat_rules: HashSet<u32> = HashSet::new();

        for expr in &self.exprs {
            match expr {
                Expr::RuleRef(rule_id) => {
                    ref_counts[rule_id.0 as usize] += 1;
                }
                Expr::Repeat { rule, .. } => {
                    repeat_rules.insert(rule.0);
                    ref_counts[rule.0 as usize] += 1;
                }
                _ => {}
            }
        }

        // Find rules that can be inlined:
        // - Referenced exactly once via RuleRef (not via Repeat)
        // - Not the root rule
        let mut to_inline: HashSet<u32> = HashSet::new();
        for i in 0..self.rules.len() {
            let id = i as u32;
            if id == self.root_rule.0 {
                continue;
            }
            if ref_counts[i] == 1 && !repeat_rules.contains(&id) {
                to_inline.insert(id);
            }
        }

        if to_inline.is_empty() {
            return self.clone();
        }

        // Clone exprs and replace RuleRef with the inlined rule's body
        let mut new_exprs = self.exprs.clone();
        for (expr_idx, expr) in self.exprs.iter().enumerate() {
            if let Expr::RuleRef(rule_id) = expr {
                if to_inline.contains(&rule_id.0) {
                    let body_id = self.rules[rule_id.0 as usize].body;
                    new_exprs[expr_idx] = new_exprs[body_id.0 as usize].clone();
                }
            }
        }

        Grammar {
            rules: self.rules.clone(),
            exprs: new_exprs,
            root_rule: self.root_rule,
        }
    }

    fn find_reachable_rules(&self) -> HashSet<RuleId> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        reachable.insert(self.root_rule);
        queue.push_back(self.root_rule);

        while let Some(rule_id) = queue.pop_front() {
            let rule = &self.rules[rule_id.0 as usize];
            self.collect_rule_refs(rule.body, &mut reachable, &mut queue);
            if let Some(ref la) = rule.lookahead {
                self.collect_rule_refs(la.expr, &mut reachable, &mut queue);
            }
        }

        reachable
    }

    fn collect_rule_refs(
        &self,
        expr_id: ExprId,
        reachable: &mut HashSet<RuleId>,
        queue: &mut VecDeque<RuleId>,
    ) {
        match self.get_expr(expr_id) {
            Expr::RuleRef(rule_id) => {
                if reachable.insert(*rule_id) {
                    queue.push_back(*rule_id);
                }
            }
            Expr::Repeat { rule, .. } => {
                if reachable.insert(*rule) {
                    queue.push_back(*rule);
                }
            }
            Expr::Sequence(exprs) | Expr::Choices(exprs) => {
                for &eid in exprs {
                    self.collect_rule_refs(eid, reachable, queue);
                }
            }
            _ => {} // Leaf nodes: EmptyString, ByteString, CharacterClass, CharacterClassStar
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::structured::grammar::builder::GrammarBuilder;
    use crate::structured::grammar::{Expr, Grammar};
    use std::sync::Arc;

    use crate::structured::matcher::GrammarMatcher;
    use crate::tokenizer::Tokenizer;

    fn accepts(grammar: &Grammar, input: &str) -> bool {
        let vocab: Vec<String> = vec!["dummy".into()];
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));
        let mut m = GrammarMatcher::new(Arc::new(grammar.clone()), tok, vec![], 10);
        if input.is_empty() {
            return m.can_terminate();
        }
        if !m.accept_string(input) {
            return false;
        }
        m.can_terminate()
    }

    #[test]
    fn test_dce_removes_unreachable_rule() {
        // root -> "hello", unreachable rule "dead" -> "world"
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let dead = b.add_rule("dead");

        let hello = b.add_byte_string(b"hello");
        b.set_rule_body(root, hello);
        let world = b.add_byte_string(b"world");
        b.set_rule_body(dead, world);

        let grammar = b.build("root").unwrap();
        assert_eq!(grammar.num_rules(), 2);

        let optimized = grammar.eliminate_dead_rules();
        assert_eq!(optimized.num_rules(), 1);
        assert_eq!(optimized.root().name, "root");

        assert!(accepts(&optimized, "hello"));
    }

    #[test]
    fn test_dce_preserves_transitive_refs() {
        // root -> a -> b -> "hello"; dead -> "world"
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let a = b.add_rule("a");
        let rule_b = b.add_rule("b");
        let dead = b.add_rule("dead");

        let a_ref = b.add_rule_ref(a);
        b.set_rule_body(root, a_ref);
        let b_ref = b.add_rule_ref(rule_b);
        b.set_rule_body(a, b_ref);
        let hello = b.add_byte_string(b"hello");
        b.set_rule_body(rule_b, hello);
        let world = b.add_byte_string(b"world");
        b.set_rule_body(dead, world);

        let grammar = b.build("root").unwrap();
        assert_eq!(grammar.num_rules(), 4);

        let optimized = grammar.eliminate_dead_rules();
        assert_eq!(optimized.num_rules(), 3); // root, a, b (dead removed)

        assert!(accepts(&optimized, "hello"));
    }

    #[test]
    fn test_dce_no_change_when_all_reachable() {
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let digit = b.add_rule("digit");

        let digit_class = b.add_character_class(false, vec![(0x30, 0x39)]);
        b.set_rule_body(digit, digit_class);
        let digit_ref = b.add_rule_ref(digit);
        b.set_rule_body(root, digit_ref);

        let grammar = b.build("root").unwrap();
        let optimized = grammar.eliminate_dead_rules();
        assert_eq!(optimized.num_rules(), 2);
    }

    #[test]
    fn test_inline_single_use_rule() {
        // root -> a, a -> "hello" (a is used once → inlined)
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let a = b.add_rule("a");

        let hello = b.add_byte_string(b"hello");
        b.set_rule_body(a, hello);
        let a_ref = b.add_rule_ref(a);
        b.set_rule_body(root, a_ref);

        let grammar = b.build("root").unwrap();
        let optimized = grammar.inline_single_use_rules();

        // After inlining, root's body should be ByteString("hello"), not RuleRef(a)
        match optimized.get_expr(optimized.root().body) {
            Expr::ByteString(bytes) => assert_eq!(bytes, b"hello"),
            other => panic!("expected ByteString after inlining, got {:?}", other),
        }
    }

    #[test]
    fn test_inline_does_not_inline_multi_use_rule() {
        // root -> seq(a, a), a -> "x" (a is used twice → not inlined)
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let a = b.add_rule("a");

        let x = b.add_byte_string(b"x");
        b.set_rule_body(a, x);
        let a_ref1 = b.add_rule_ref(a);
        let a_ref2 = b.add_rule_ref(a);
        let seq = b.add_sequence(vec![a_ref1, a_ref2]);
        b.set_rule_body(root, seq);

        let grammar = b.build("root").unwrap();
        let optimized = grammar.inline_single_use_rules();

        // a is used twice, so it should NOT be inlined
        assert_eq!(optimized.num_rules(), 2);
        // root body is still a Sequence containing RuleRefs
        match optimized.get_expr(optimized.root().body) {
            Expr::Sequence(exprs) => {
                assert!(matches!(optimized.get_expr(exprs[0]), Expr::RuleRef(_)));
            }
            other => panic!("expected Sequence, got {:?}", other),
        }
    }

    #[test]
    fn test_inline_does_not_inline_repeat_rule() {
        // root -> repeat(a, 1, None), a -> "x" (a is in Repeat → not inlined)
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let a = b.add_rule("a");

        let x = b.add_byte_string(b"x");
        b.set_rule_body(a, x);
        let rep = b.add_repeat(a, 1, None);
        b.set_rule_body(root, rep);

        let grammar = b.build("root").unwrap();
        let optimized = grammar.inline_single_use_rules();

        // a is used in Repeat → should NOT be inlined
        match optimized.get_expr(optimized.root().body) {
            Expr::Repeat { .. } => {} // still a Repeat
            other => panic!("expected Repeat, got {:?}", other),
        }
    }

    #[test]
    fn test_optimize_inline_then_dce() {
        // root -> a -> "hello", dead -> "world"
        // After optimize: a is inlined + dead is removed → 1 rule
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let a = b.add_rule("a");
        let dead = b.add_rule("dead");

        let hello = b.add_byte_string(b"hello");
        b.set_rule_body(a, hello);
        let a_ref = b.add_rule_ref(a);
        b.set_rule_body(root, a_ref);
        let world = b.add_byte_string(b"world");
        b.set_rule_body(dead, world);

        let grammar = b.build("root").unwrap();
        assert_eq!(grammar.num_rules(), 3);

        let optimized = grammar.optimize();
        assert_eq!(optimized.num_rules(), 1);
        assert_eq!(optimized.root().name, "root");

        assert!(accepts(&optimized, "hello"));
        assert!(!accepts(&optimized, "world"));
    }

    #[test]
    fn test_optimize_ebnf_grammar() {
        // Build a grammar with dead rules programmatically
        // (from_ebnf validates all rules are used, so we build manually)
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let value = b.add_rule("value");
        let dead = b.add_rule("dead_rule");

        let t = b.add_byte_string(b"true");
        let f = b.add_byte_string(b"false");
        let n = b.add_byte_string(b"null");
        let choices = b.add_choices(vec![t, f, n]);
        b.set_rule_body(value, choices);
        let value_ref = b.add_rule_ref(value);
        b.set_rule_body(root, value_ref);
        let dead_body = b.add_byte_string(b"dead");
        b.set_rule_body(dead, dead_body);

        let grammar = b.build("root").unwrap();
        assert_eq!(grammar.num_rules(), 3);

        let optimized = grammar.optimize();
        // value is single-use → inlined, then both value and dead are removed by DCE
        assert_eq!(optimized.num_rules(), 1);

        assert!(accepts(&optimized, "true"));
        assert!(accepts(&optimized, "false"));
        assert!(accepts(&optimized, "null"));
        assert!(!accepts(&optimized, "dead"));
    }

    #[test]
    fn test_optimize_preserves_repeat() {
        // root -> repeat(digit, 1, None), digit -> [0-9]
        // digit is used in Repeat → not inlined, but still reachable
        let mut b = GrammarBuilder::new();
        let root = b.add_rule("root");
        let digit = b.add_rule("digit");

        let digit_class = b.add_character_class(false, vec![(0x30, 0x39)]);
        b.set_rule_body(digit, digit_class);
        let rep = b.add_repeat(digit, 1, None);
        b.set_rule_body(root, rep);

        let grammar = b.build("root").unwrap();
        let optimized = grammar.optimize();
        assert_eq!(optimized.num_rules(), 2); // Both kept

        assert!(accepts(&optimized, "5"));
        assert!(accepts(&optimized, "123"));
        assert!(!accepts(&optimized, ""));
    }

    #[test]
    fn test_optimize_from_ebnf() {
        // Test with a real EBNF-parsed grammar
        let g = Grammar::from_ebnf(r#"root ::= "a"+"#, "root").unwrap();
        let before_rules = g.num_rules();
        let optimized = g.optimize();

        // Should still work correctly
        assert!(accepts(&optimized, "a"));
        assert!(accepts(&optimized, "aaa"));
        assert!(!accepts(&optimized, ""));
        assert!(!accepts(&optimized, "b"));

        // May or may not reduce rules depending on normalization
        assert!(optimized.num_rules() <= before_rules);
    }
}
