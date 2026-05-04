//! Grammar normalization for the parser.
//!
//! Transforms a grammar into canonical form where:
//! - Every rule body is `Choices(alts)` or a leaf expression
//! - Each alternative in Choices is `Sequence(elems)` or a leaf
//! - Each element of a Sequence is a leaf: ByteString, CharacterClass,
//!   CharacterClassStar, RuleRef, Repeat, or EmptyString
//! - No nested Choices or Sequences as Sequence elements

use super::builder::GrammarBuilder;
use super::{Expr, ExprId, Grammar, RuleId};

/// Normalize a grammar into canonical form for the parser.
///
/// Nested Choices/Sequences in Sequence elements are extracted into auxiliary rules.
pub fn normalize_grammar(grammar: &Grammar) -> Grammar {
    let mut builder = GrammarBuilder::new();

    // First pass: create all original rules (so RuleIds are preserved)
    for rule in grammar.rules() {
        builder.add_rule(&rule.name);
    }

    // Second pass: normalize each rule body, possibly adding auxiliary rules
    for (i, rule) in grammar.rules().iter().enumerate() {
        let body = normalize_expr(grammar, &mut builder, rule.body);
        builder.set_rule_body(RuleId(i as u32), body);

        if let Some(ref la) = rule.lookahead {
            let la_expr = normalize_expr(grammar, &mut builder, la.expr);
            builder.set_rule_lookahead(RuleId(i as u32), la_expr, la.is_exact);
        }
    }

    builder.build(&grammar.get_rule(grammar.root_rule()).name).unwrap()
}

/// Normalize an expression. If it's a Sequence, ensure all elements are leaves.
fn normalize_expr(grammar: &Grammar, builder: &mut GrammarBuilder, expr_id: ExprId) -> ExprId {
    match grammar.get_expr(expr_id) {
        Expr::EmptyString => builder.add_empty_string(),
        Expr::ByteString(bytes) => builder.add_byte_string(bytes),
        Expr::CharacterClass { negated, ranges } => {
            builder.add_character_class(*negated, ranges.clone())
        }
        Expr::CharacterClassStar { negated, ranges } => {
            builder.add_character_class_star(*negated, ranges.clone())
        }
        Expr::RuleRef(rule_id) => builder.add_rule_ref(*rule_id),
        Expr::Repeat { rule, min, max } => builder.add_repeat(*rule, *min, *max),

        Expr::Sequence(elems) => {
            let new_elems: Vec<ExprId> = elems
                .iter()
                .map(|&eid| normalize_sequence_element(grammar, builder, eid))
                .collect();
            builder.add_sequence(new_elems)
        }

        Expr::Choices(alts) => {
            let mut new_alts: Vec<ExprId> = Vec::new();
            for &eid in alts {
                let normalized = normalize_expr(grammar, builder, eid);
                // Flatten nested Choices: Choices([A, Choices([B, C])]) → Choices([A, B, C])
                if let Expr::Choices(inner) = &builder.exprs[normalized.0 as usize] {
                    new_alts.extend_from_slice(inner);
                } else {
                    new_alts.push(normalized);
                }
            }
            builder.add_choices(new_alts)
        }
    }
}

/// Normalize a sequence element. If it's a Choices or Sequence, extract to auxiliary rule.
fn normalize_sequence_element(
    grammar: &Grammar,
    builder: &mut GrammarBuilder,
    expr_id: ExprId,
) -> ExprId {
    match grammar.get_expr(expr_id) {
        Expr::Choices(_) | Expr::Sequence(_) => {
            // Extract to auxiliary rule
            let aux_name = format!("__aux_{}", builder.num_rules());
            let aux_rule = builder.add_rule(&aux_name);
            let normalized = normalize_expr(grammar, builder, expr_id);
            builder.set_rule_body(aux_rule, normalized);
            builder.add_rule_ref(aux_rule)
        }
        // Leaves are fine as-is
        _ => normalize_expr(grammar, builder, expr_id),
    }
}
