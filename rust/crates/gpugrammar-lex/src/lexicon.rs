//! Splitting a grammar into a lexicon and a context-free skeleton.
//!
//! [`regular::analyze`](crate::regular::analyze) decides regularity per rule,
//! but that is not the whole split: a structural rule like
//! `object ::= "{" members "}"` is not regular, yet its `"{"` and `"}"` are.
//! Extraction walks every non-regular rule and cuts out each *maximal* regular
//! subtree, which becomes a terminal; what remains is the skeleton the LR
//! automaton parses.
//!
//! Terminals are interned by structure, so a comma written in five different
//! rules is one terminal with one column in the ACTION table.

use gpugrammar_ir::fsm::{Automaton, FsmEdge, NfaGraph, StateId, build_rule_fsms};
use gpugrammar_ir::grammar::{Expr, ExprId, Grammar, Rule, RuleId};
use rustc_hash::FxHashMap;

use crate::regular::Regularity;
use crate::{Terminal, TerminalId};

/// A terminal of the skeleton: a maximal regular subtree of the grammar.
#[derive(Debug, Clone)]
pub struct TerminalDef {
    pub name: String,
    pub expr: ExprId,
}

/// The skeleton form of an expression, with regular parts already cut out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkeletonExpr {
    Empty,
    Terminal(TerminalId),
    Nonterminal(RuleId),
    Sequence(Vec<SkeletonExpr>),
    Choice(Vec<SkeletonExpr>),
    Repeat {
        inner: Box<SkeletonExpr>,
        min: u32,
        max: Option<u32>,
    },
}

/// One rule of the context-free skeleton.
#[derive(Debug, Clone)]
pub struct SkeletonRule {
    pub rule: RuleId,
    pub name: String,
    pub body: SkeletonExpr,
}

/// The lexicon and skeleton a grammar splits into.
#[derive(Debug, Clone)]
pub struct Lexicon {
    pub terminals: Vec<TerminalDef>,
    pub skeleton: Vec<SkeletonRule>,
}

impl Lexicon {
    pub fn terminal_name(&self, terminal: TerminalId) -> &str {
        &self.terminals[terminal.0 as usize].name
    }
}

/// Cut a grammar into terminals and a skeleton.
///
/// A terminal that is a whole rule takes that rule's name, so the ACTION table
/// reads like the grammar rather than like an arena index.
pub fn extract(grammar: &Grammar, regularity: &Regularity) -> Lexicon {
    let mut interner = Interner::default();
    let mut skeleton = Vec::new();

    for (index, rule) in grammar.rules().iter().enumerate() {
        let id = RuleId(index as u32);
        if regularity.is_regular(id) {
            continue;
        }
        let body = lower(grammar, rule.body, regularity, &mut interner);
        skeleton.push(SkeletonRule {
            rule: id,
            name: rule.name.clone(),
            body,
        });
    }

    if skeleton.is_empty() {
        // Every rule was regular, so the whole document is one lexeme and the
        // parser has nothing to do. That is the best case rather than a
        // failure: the mask comes entirely from the lexer state and no stack
        // is needed. Give the LR construction a trivial skeleton so the rest
        // of the pipeline does not need a special case. Measured on
        // JSONSchemaBench this is 68% of schemas.
        let root = grammar.root_rule();
        let expr = grammar.get_rule(root).body;
        let terminal = if is_terminal_atom(grammar, expr, regularity) {
            interner.intern(grammar, expr)
        } else {
            interner.intern_rule(grammar, root)
        };
        skeleton.push(SkeletonRule {
            rule: root,
            name: grammar.get_rule(root).name.clone(),
            body: SkeletonExpr::Terminal(terminal),
        });
    }

    let mut lexicon = Lexicon {
        terminals: interner.terminals,
        skeleton,
    };
    name_terminals_from_rules(grammar, &mut lexicon);
    lexicon
}

/// Build one byte automaton per terminal, ready for [`crate::build_lexer`].
///
/// Each terminal subtree is appended to a copy of the grammar as its own rule
/// so the existing per-rule construction can be reused, and then every
/// remaining rule-reference edge is resolved.
///
/// Resolution has to happen here because the upstream builder only inlines
/// rules it considers leaves; a recursive one such as `chars ::= "\"" |
/// [^"\\] chars` stays behind a reference edge, and a determiniser that only
/// follows byte edges would silently match nothing. Splicing with memoisation
/// turns that self-reference into a cycle, which is exactly right: regularity
/// analysis only admits recursion in tail position.
pub fn terminal_automata(grammar: &Grammar, lexicon: &Lexicon) -> Vec<Terminal> {
    let mut extended = grammar.clone();
    let base = extended.rules.len();
    for (index, terminal) in lexicon.terminals.iter().enumerate() {
        extended.rules.push(Rule {
            name: format!("__terminal_{index}"),
            body: terminal.expr,
        });
    }
    let automata = build_rule_fsms(&extended);
    lexicon
        .terminals
        .iter()
        .enumerate()
        .map(|(index, terminal)| Terminal {
            name: terminal.name.clone(),
            automaton: resolve_references(&automata, base + index),
        })
        .collect()
}

/// Splice referenced rules into `root`'s automaton until no reference edges
/// remain. Each rule is spliced once, so recursion becomes a cycle.
fn resolve_references(automata: &[Automaton<NfaGraph>], root: usize) -> Automaton<NfaGraph> {
    let mut fsm = NfaGraph::new();
    let mut spliced: FxHashMap<usize, (StateId, StateId)> = FxHashMap::default();
    let (start, end) = splice(automata, root, &mut fsm, &mut spliced);
    let mut ends = vec![false; fsm.num_states()];
    ends[end.0 as usize] = true;
    Automaton { fsm, start, ends }
}

fn splice(
    automata: &[Automaton<NfaGraph>],
    rule: usize,
    fsm: &mut NfaGraph,
    spliced: &mut FxHashMap<usize, (StateId, StateId)>,
) -> (StateId, StateId) {
    if let Some(&existing) = spliced.get(&rule) {
        return existing;
    }
    let source = &automata[rule];
    let offset = fsm.num_states() as u32;
    for _ in 0..source.fsm.num_states() {
        fsm.add_state();
    }
    let entry = StateId(offset + source.start.0);
    let exit = fsm.add_state();
    spliced.insert(rule, (entry, exit));

    for state in 0..source.fsm.num_states() {
        let from = StateId(offset + state as u32);
        for edge in source.fsm.edges(StateId(state as u32)) {
            match edge {
                FsmEdge::CharRange { min, max, target } => fsm.add_edge(
                    from,
                    FsmEdge::CharRange {
                        min: *min,
                        max: *max,
                        target: StateId(offset + target.0),
                    },
                ),
                FsmEdge::Epsilon(target) => {
                    fsm.add_epsilon(from, StateId(offset + target.0));
                }
                FsmEdge::RuleRef { rule: target, .. } => {
                    let (inner_start, inner_end) =
                        splice(automata, target.0 as usize, fsm, spliced);
                    fsm.add_epsilon(from, inner_start);
                    // `edges` borrows `source`, so the continuation is added
                    // from the reference edge's own target below.
                    if let FsmEdge::RuleRef { target: after, .. } = edge {
                        fsm.add_epsilon(inner_end, StateId(offset + after.0));
                    }
                }
            }
        }
        if source.is_end(StateId(state as u32)) {
            fsm.add_epsilon(from, exit);
        }
    }
    (entry, exit)
}

#[derive(Default)]
struct Interner {
    by_key: FxHashMap<String, TerminalId>,
    terminals: Vec<TerminalDef>,
}

impl Interner {
    /// Intern a whole rule as one terminal, by its identity rather than by a
    /// subtree, so a regular root becomes a single lexeme.
    fn intern_rule(&mut self, grammar: &Grammar, rule: RuleId) -> TerminalId {
        let key = format!("r{}", rule.0);
        if let Some(&existing) = self.by_key.get(&key) {
            return existing;
        }
        let id = TerminalId(self.terminals.len() as u32);
        self.terminals.push(TerminalDef {
            name: grammar.get_rule(rule).name.clone(),
            expr: grammar.get_rule(rule).body,
        });
        self.by_key.insert(key, id);
        id
    }

    fn intern(&mut self, grammar: &Grammar, expr: ExprId) -> TerminalId {
        let key = canonical(grammar, expr);
        if let Some(&existing) = self.by_key.get(&key) {
            return existing;
        }
        let id = TerminalId(self.terminals.len() as u32);
        self.terminals.push(TerminalDef {
            name: display_name(&key),
            expr,
        });
        self.by_key.insert(key, id);
        id
    }
}

fn lower(
    grammar: &Grammar,
    expr: ExprId,
    regularity: &Regularity,
    interner: &mut Interner,
) -> SkeletonExpr {
    if matches!(grammar.get_expr(expr), Expr::EmptyString) {
        return SkeletonExpr::Empty;
    }
    if is_terminal_atom(grammar, expr, regularity) {
        return SkeletonExpr::Terminal(interner.intern(grammar, expr));
    }
    match grammar.get_expr(expr) {
        Expr::Sequence(parts) => SkeletonExpr::Sequence(
            parts
                .iter()
                .map(|part| lower(grammar, *part, regularity, interner))
                .collect(),
        ),
        Expr::Choices(alternatives) => SkeletonExpr::Choice(
            alternatives
                .iter()
                .map(|alternative| lower(grammar, *alternative, regularity, interner))
                .collect(),
        ),
        Expr::RuleRef(target) => SkeletonExpr::Nonterminal(*target),
        Expr::Repeat { rule, min, max } => SkeletonExpr::Repeat {
            inner: Box::new(SkeletonExpr::Nonterminal(*rule)),
            min: *min,
            max: *max,
        },
        // Leaves are always regular and were handled above.
        Expr::EmptyString
        | Expr::ByteString(_)
        | Expr::CharacterClass { .. }
        | Expr::CharacterClassStar { .. } => unreachable!("leaf expressions are regular"),
    }
}

/// Is this subtree one terminal?
///
/// Only atoms and references to regular rules qualify. Taking the *maximal*
/// regular subtree instead would merge adjacent punctuation: `object ::= "{"
/// "}"` would yield a single `{}` terminal, which both hides structure from the
/// parser and makes a lone `{` ambiguous, since the scanner could not commit to
/// it without seeing whether a `}` follows. Splitting at atoms is what a
/// hand-written lexer does.
///
/// The cost is that a lexeme spelled inline rather than as its own rule — say
/// `"-"? [0-9]+` written directly inside a structural rule — becomes several
/// terminals. Front ends put lexemes in their own rules, so this has not bitten
/// yet, but it is the case to watch.
fn is_terminal_atom(grammar: &Grammar, expr: ExprId, regularity: &Regularity) -> bool {
    match grammar.get_expr(expr) {
        Expr::ByteString(_) | Expr::CharacterClass { .. } | Expr::CharacterClassStar { .. } => true,
        Expr::RuleRef(target) => regularity.is_regular(*target),
        Expr::Repeat { rule, .. } => regularity.is_regular(*rule),
        Expr::EmptyString | Expr::Sequence(_) | Expr::Choices(_) => false,
    }
}

/// A subtree is regular when every rule it reaches is regular.
pub fn is_regular_expr(grammar: &Grammar, expr: ExprId, regularity: &Regularity) -> bool {
    match grammar.get_expr(expr) {
        Expr::EmptyString
        | Expr::ByteString(_)
        | Expr::CharacterClass { .. }
        | Expr::CharacterClassStar { .. } => true,
        Expr::RuleRef(target) => regularity.is_regular(*target),
        Expr::Sequence(parts) | Expr::Choices(parts) => parts
            .iter()
            .all(|part| is_regular_expr(grammar, *part, regularity)),
        Expr::Repeat { rule, .. } => regularity.is_regular(*rule),
    }
}

/// Structural key, so identical subtrees intern to one terminal.
fn canonical(grammar: &Grammar, expr: ExprId) -> String {
    let mut out = String::new();
    write_canonical(grammar, expr, &mut out);
    out
}

fn write_canonical(grammar: &Grammar, expr: ExprId, out: &mut String) {
    use std::fmt::Write;
    match grammar.get_expr(expr) {
        Expr::EmptyString => out.push_str("e"),
        Expr::ByteString(bytes) => {
            let _ = write!(out, "b{}", String::from_utf8_lossy(bytes));
        }
        Expr::CharacterClass { negated, ranges } => {
            let _ = write!(out, "c{}{:?}", negated, ranges);
        }
        Expr::CharacterClassStar { negated, ranges } => {
            let _ = write!(out, "s{}{:?}", negated, ranges);
        }
        Expr::RuleRef(target) => {
            let _ = write!(out, "r{}", target.0);
        }
        Expr::Sequence(parts) => {
            out.push_str("(.");
            for part in parts {
                out.push(' ');
                write_canonical(grammar, *part, out);
            }
            out.push(')');
        }
        Expr::Choices(alternatives) => {
            out.push_str("(|");
            for alternative in alternatives {
                out.push(' ');
                write_canonical(grammar, *alternative, out);
            }
            out.push(')');
        }
        Expr::Repeat { rule, min, max } => {
            let _ = write!(out, "{{{},{:?}}}r{}", min, max, rule.0);
        }
    }
}

fn display_name(key: &str) -> String {
    if let Some(literal) = key.strip_prefix('b') {
        return format!("'{literal}'");
    }
    key.chars().take(24).collect()
}

/// Name a terminal after the rule it came from, when it is a whole rule.
fn name_terminals_from_rules(grammar: &Grammar, lexicon: &mut Lexicon) {
    for terminal in &mut lexicon.terminals {
        if let Expr::RuleRef(target) = grammar.get_expr(terminal.expr) {
            terminal.name = grammar.get_rule(*target).name.clone();
        }
    }
}
