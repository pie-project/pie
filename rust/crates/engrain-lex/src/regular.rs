//! Which parts of a grammar are regular, and therefore belong to the lexer.
//!
//! The lexer/parser split needs a decision for every rule: is its language
//! regular? A regular rule becomes a terminal of the context-free skeleton and
//! is compiled into the scanner; everything else stays for the LR automaton.
//!
//! The interesting case is recursion. `string_body ::= "\"" | [^"\\]
//! string_body` is right-recursive and therefore regular, and it is *the*
//! string rule these front ends emit, so a test that simply rejects all
//! recursion would push JSON strings into the parser and defeat the design.
//! Recursion is accepted when every reference back into the same strongly
//! connected component sits in tail position — the standard right-linear test.

use engrain_ir::grammar::{Expr, ExprId, Grammar, RuleId};
use rustc_hash::{FxHashMap, FxHashSet};

/// The outcome of the analysis, one entry per rule.
#[derive(Debug, Clone)]
pub struct Regularity {
    regular: Vec<bool>,
}

impl Regularity {
    pub fn is_regular(&self, rule: RuleId) -> bool {
        self.regular[rule.0 as usize]
    }

    pub fn regular_rules(&self) -> impl Iterator<Item = RuleId> + '_ {
        self.regular
            .iter()
            .enumerate()
            .filter(|(_, regular)| **regular)
            .map(|(index, _)| RuleId(index as u32))
    }
}

/// Decide regularity for every rule of `grammar`.
pub fn analyze(grammar: &Grammar) -> Regularity {
    let components = strongly_connected_components(grammar);
    let mut component_of = vec![0usize; grammar.rules().len()];
    for (index, component) in components.iter().enumerate() {
        for rule in component {
            component_of[rule.0 as usize] = index;
        }
    }

    // Components come back in reverse topological order, so a component's
    // dependencies are already decided when it is visited.
    let mut regular = vec![false; grammar.rules().len()];
    for (index, component) in components.iter().enumerate() {
        let members: FxHashSet<RuleId> = component.iter().copied().collect();
        let ok = component.iter().all(|rule| {
            let body = grammar.get_rule(*rule).body;
            check(
                grammar,
                body,
                &members,
                &component_of,
                index,
                &regular,
                true,
            )
        });
        if ok {
            for rule in component {
                regular[rule.0 as usize] = true;
            }
        }
    }
    Regularity { regular }
}

/// Walk an expression, requiring that same-component references sit in tail
/// position and that every other reference is to an already-regular rule.
fn check(
    grammar: &Grammar,
    expr: ExprId,
    members: &FxHashSet<RuleId>,
    component_of: &[usize],
    component: usize,
    regular: &[bool],
    tail: bool,
) -> bool {
    match grammar.get_expr(expr) {
        Expr::EmptyString
        | Expr::ByteString(_)
        | Expr::CharacterClass { .. }
        | Expr::CharacterClassStar { .. } => true,
        Expr::RuleRef(target) => {
            if members.contains(target) {
                // Recursion is only regular in tail position.
                tail
            } else if component_of[target.0 as usize] == component {
                tail
            } else {
                regular[target.0 as usize]
            }
        }
        Expr::Sequence(parts) => {
            let last = parts.len().saturating_sub(1);
            parts.iter().enumerate().all(|(index, part)| {
                check(
                    grammar,
                    *part,
                    members,
                    component_of,
                    component,
                    regular,
                    tail && index == last,
                )
            })
        }
        Expr::Choices(alternatives) => alternatives.iter().all(|alternative| {
            check(
                grammar,
                *alternative,
                members,
                component_of,
                component,
                regular,
                tail,
            )
        }),
        Expr::Repeat { rule, .. } => {
            // A repeated rule is entered many times, so a same-component
            // target would recur outside tail position.
            !members.contains(rule)
                && component_of[rule.0 as usize] != component
                && regular[rule.0 as usize]
        }
    }
}

/// Tarjan's algorithm, returning components in reverse topological order.
fn strongly_connected_components(grammar: &Grammar) -> Vec<Vec<RuleId>> {
    struct State {
        index: u32,
        indices: FxHashMap<RuleId, u32>,
        low: FxHashMap<RuleId, u32>,
        stack: Vec<RuleId>,
        on_stack: FxHashSet<RuleId>,
        output: Vec<Vec<RuleId>>,
    }

    fn strongconnect(grammar: &Grammar, rule: RuleId, state: &mut State) {
        state.indices.insert(rule, state.index);
        state.low.insert(rule, state.index);
        state.index += 1;
        state.stack.push(rule);
        state.on_stack.insert(rule);

        for target in references(grammar, grammar.get_rule(rule).body) {
            if !state.indices.contains_key(&target) {
                strongconnect(grammar, target, state);
                let low = state.low[&target];
                let current = state.low[&rule];
                state.low.insert(rule, current.min(low));
            } else if state.on_stack.contains(&target) {
                let index = state.indices[&target];
                let current = state.low[&rule];
                state.low.insert(rule, current.min(index));
            }
        }

        if state.low[&rule] == state.indices[&rule] {
            let mut component = Vec::new();
            while let Some(top) = state.stack.pop() {
                state.on_stack.remove(&top);
                component.push(top);
                if top == rule {
                    break;
                }
            }
            state.output.push(component);
        }
    }

    let mut state = State {
        index: 0,
        indices: FxHashMap::default(),
        low: FxHashMap::default(),
        stack: Vec::new(),
        on_stack: FxHashSet::default(),
        output: Vec::new(),
    };
    for index in 0..grammar.rules().len() {
        let rule = RuleId(index as u32);
        if !state.indices.contains_key(&rule) {
            strongconnect(grammar, rule, &mut state);
        }
    }
    state.output
}

/// Every rule referenced anywhere in an expression tree.
pub fn references(grammar: &Grammar, expr: ExprId) -> Vec<RuleId> {
    let mut found = Vec::new();
    collect(grammar, expr, &mut found);
    found
}

fn collect(grammar: &Grammar, expr: ExprId, found: &mut Vec<RuleId>) {
    match grammar.get_expr(expr) {
        Expr::EmptyString
        | Expr::ByteString(_)
        | Expr::CharacterClass { .. }
        | Expr::CharacterClassStar { .. } => {}
        Expr::RuleRef(target) => found.push(*target),
        Expr::Sequence(parts) | Expr::Choices(parts) => {
            for part in parts {
                collect(grammar, *part, found);
            }
        }
        Expr::Repeat { rule, .. } => found.push(*rule),
    }
}
