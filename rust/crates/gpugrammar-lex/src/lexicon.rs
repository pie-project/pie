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
use rustc_hash::{FxHashMap, FxHashSet};

use crate::regular::Regularity;
use crate::{Terminal, TerminalId};

/// A terminal of the skeleton: a maximal regular subtree of the grammar.
#[derive(Debug, Clone)]
pub struct TerminalDef {
    pub name: String,
    /// The concatenation this terminal matches. Usually one expression, but a
    /// run of consecutive regular siblings inside a sequence is merged into
    /// one terminal, which is what keeps a quoted string whole.
    pub parts: Vec<ExprId>,
    /// True when the expression also matches the empty string.
    ///
    /// A scanner cannot emit an empty lexeme, so a nullable terminal would
    /// never reach the parser and every rule needing it would be stuck: with
    /// `__json_ws ::= [ \t\n\r]*` a document without whitespace could not be
    /// parsed at all, and `""` was rejected because its content terminal was
    /// nullable. Nullability therefore moves into the skeleton, as
    /// `ε | terminal`, and the terminal's automaton is stripped of ε.
    pub nullable: bool,
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
    /// The grammar's root. The skeleton follows declaration order, which need
    /// not put the root first, so the LR construction has to be told which
    /// rule to start from rather than guessing at the first one.
    pub root: RuleId,
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
    let root = grammar.root_rule();

    // A regular root means the whole language is regular, since a rule is only
    // regular when everything it reaches is. The document is then one lexeme
    // and the parser has nothing to do, which is the best case rather than a
    // failure: the mask comes entirely from the lexer state and no stack is
    // needed. Give the LR construction a trivial skeleton so the rest of the
    // pipeline needs no special case. Measured on JSONSchemaBench this is 68%
    // of schemas. Testing the root rather than the skeleton matters, because
    // an unreachable non-regular rule would otherwise hide this case.
    if regularity.is_regular(root) {
        let expr = grammar.get_rule(root).body;
        let terminal = if is_terminal_atom(grammar, expr, regularity) {
            interner.intern(grammar, expr)
        } else {
            interner.intern_rule(grammar, root)
        };
        let body = optional_if_nullable(&interner, terminal);
        skeleton.push(SkeletonRule {
            rule: root,
            name: grammar.get_rule(root).name.clone(),
            body,
        });
        let mut lexicon = Lexicon {
            terminals: interner.terminals,
            skeleton,
            root,
        };
        name_terminals_from_rules(grammar, &mut lexicon);
        return lexicon;
    }

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

    let mut lexicon = Lexicon {
        terminals: interner.terminals,
        skeleton,
        root,
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
        let body = if let [single] = terminal.parts[..] {
            single
        } else {
            let id = ExprId(extended.exprs.len() as u32);
            extended.exprs.push(Expr::Sequence(terminal.parts.clone()));
            id
        };
        extended.rules.push(Rule {
            name: format!("__terminal_{index}"),
            body,
        });
    }
    let automata = build_rule_fsms(&extended);
    lexicon
        .terminals
        .iter()
        .enumerate()
        .map(|(index, terminal)| {
            let automaton = resolve_references(&automata, base + index);
            Terminal {
                name: terminal.name.clone(),
                automaton: if terminal.nullable {
                    without_empty(automaton)
                } else {
                    automaton
                },
            }
        })
        .collect()
}

/// Wrap a nullable terminal as `ε | terminal` in the skeleton.
fn optional_if_nullable(interner: &Interner, terminal: TerminalId) -> SkeletonExpr {
    if interner.terminals[terminal.0 as usize].nullable {
        SkeletonExpr::Choice(vec![SkeletonExpr::Empty, SkeletonExpr::Terminal(terminal)])
    } else {
        SkeletonExpr::Terminal(terminal)
    }
}

/// Does this expression match the empty string?
fn is_nullable(grammar: &Grammar, expr: ExprId) -> bool {
    fn walk(grammar: &Grammar, expr: ExprId, visiting: &mut FxHashSet<u32>) -> bool {
        match grammar.get_expr(expr) {
            Expr::EmptyString => true,
            Expr::ByteString(bytes) => bytes.is_empty(),
            Expr::CharacterClass { .. } => false,
            Expr::CharacterClassStar { .. } => true,
            Expr::Sequence(parts) => parts.iter().all(|part| walk(grammar, *part, visiting)),
            Expr::Choices(alternatives) => alternatives
                .iter()
                .any(|alternative| walk(grammar, *alternative, visiting)),
            Expr::RuleRef(rule) => walk_rule(grammar, *rule, visiting),
            Expr::Repeat { rule, min, .. } => *min == 0 || walk_rule(grammar, *rule, visiting),
        }
    }

    // A rule already on the path contributes nothing: reaching it again means
    // at least one more expansion, and regularity only admits tail recursion,
    // so that expansion cannot be empty.
    fn walk_rule(grammar: &Grammar, rule: RuleId, visiting: &mut FxHashSet<u32>) -> bool {
        if !visiting.insert(rule.0) {
            return false;
        }
        let result = walk(grammar, grammar.get_rule(rule).body, visiting);
        visiting.remove(&rule.0);
        result
    }

    walk(grammar, expr, &mut FxHashSet::default())
}

/// The same language without the empty string.
///
/// A fresh start state takes over the byte edges leaving the old start's
/// epsilon closure. Every non-empty word is still accepted, because its first
/// byte was read from that closure; the empty word is not, because the new
/// start is not accepting and has no epsilon edges.
fn without_empty(mut automaton: Automaton<NfaGraph>) -> Automaton<NfaGraph> {
    let mut closure = vec![false; automaton.fsm.num_states()];
    let mut stack = vec![automaton.start];
    closure[automaton.start.0 as usize] = true;
    while let Some(state) = stack.pop() {
        for edge in automaton.fsm.edges(state) {
            if let FsmEdge::Epsilon(target) = edge {
                if !closure[target.0 as usize] {
                    closure[target.0 as usize] = true;
                    stack.push(*target);
                }
            }
        }
    }

    let mut edges = Vec::new();
    for state in 0..automaton.fsm.num_states() {
        if !closure[state] {
            continue;
        }
        for edge in automaton.fsm.edges(StateId(state as u32)) {
            if let FsmEdge::CharRange { min, max, target } = edge {
                edges.push(FsmEdge::CharRange {
                    min: *min,
                    max: *max,
                    target: *target,
                });
            }
        }
    }

    let start = automaton.fsm.add_state();
    automaton.ends.push(false);
    for edge in edges {
        automaton.fsm.add_edge(start, edge);
    }
    automaton.start = start;
    automaton
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
        let body = grammar.get_rule(rule).body;
        self.terminals.push(TerminalDef {
            name: grammar.get_rule(rule).name.clone(),
            parts: vec![body],
            nullable: is_nullable(grammar, body),
        });
        self.by_key.insert(key, id);
        id
    }

    fn intern(&mut self, grammar: &Grammar, expr: ExprId) -> TerminalId {
        self.intern_run(grammar, &[unwrap_singleton(grammar, expr)])
    }

    /// Intern a concatenation of regular expressions as one terminal.
    fn intern_run(&mut self, grammar: &Grammar, parts: &[ExprId]) -> TerminalId {
        let key = parts
            .iter()
            .map(|part| canonical(grammar, *part))
            .collect::<Vec<_>>()
            .join("~");
        if let Some(&existing) = self.by_key.get(&key) {
            return existing;
        }
        let id = TerminalId(self.terminals.len() as u32);
        self.terminals.push(TerminalDef {
            name: display_name(&key),
            parts: parts.to_vec(),
            nullable: parts.iter().all(|part| is_nullable(grammar, *part)),
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
    // Cut out the *maximal* regular subtree. A whole regular rule therefore
    // becomes one terminal, which is how the front end declares what a lexeme
    // is: a quoted string or a number is a rule, so it stays whole instead of
    // splitting into `'"'`, a body and `'"'` whose body class overlaps every
    // punctuation terminal in the grammar. Adjacent literals in a sequence are
    // still separate, so a lone `{` remains committable.
    if is_regular_expr(grammar, expr, regularity) {
        let terminal = interner.intern(grammar, expr);
        return optional_if_nullable(interner, terminal);
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

/// Strip sequences and choices of one element.
///
/// The EBNF front end wraps an alternative in a one-element sequence, so the
/// same rule reference reaches the interner under two shapes and becomes two
/// terminals with two ACTION columns - and only one of them keeps the rule's
/// name.
fn unwrap_singleton(grammar: &Grammar, expr: ExprId) -> ExprId {
    match grammar.get_expr(expr) {
        Expr::Sequence(parts) if parts.len() == 1 => unwrap_singleton(grammar, parts[0]),
        Expr::Choices(alternatives) if alternatives.len() == 1 => {
            unwrap_singleton(grammar, alternatives[0])
        }
        _ => expr,
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
        if let [single] = terminal.parts[..]
            && let Expr::RuleRef(target) = grammar.get_expr(single)
        {
            terminal.name = grammar.get_rule(*target).name.clone();
        }
    }
}
