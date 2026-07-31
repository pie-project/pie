//! The context-free skeleton as a plain production set.
//!
//! [`SkeletonExpr`] is a tree with choice, sequence and repetition; an LR
//! construction wants flat productions. Flattening introduces a fresh
//! nonterminal wherever a nested choice or a repetition appears, which is the
//! usual EBNF-to-BNF expansion.

use engrain_lex::TerminalId;
use engrain_lex::lexicon::{Lexicon, SkeletonExpr};
use rustc_hash::{FxHashMap, FxHashSet};

/// A grammar symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Symbol {
    Terminal(TerminalId),
    Nonterminal(u32),
}

/// One production, `lhs -> rhs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Production {
    pub lhs: u32,
    pub rhs: Vec<Symbol>,
}

/// A flat context-free grammar over the lexicon's terminals.
#[derive(Debug, Clone)]
pub struct Cfg {
    pub productions: Vec<Production>,
    pub nonterminal_names: Vec<String>,
    pub num_terminals: usize,
    pub start: u32,
}

impl Cfg {
    pub fn num_nonterminals(&self) -> usize {
        self.nonterminal_names.len()
    }
}

/// How many productions a grammar may have.
///
/// A bounded repeat costs one nonterminal per remaining count in any machine
/// that keeps the counter in its states, and both the lexer and the parser do.
/// Moving `x{0,65536}` from one to the other only moves the cost: it became
/// 131,115 productions here. The real answer is a counter the runtime holds
/// rather than compiles away; until then the budget says which schemas are out
/// of reach instead of hanging on them.
pub const DEFAULT_PRODUCTION_BUDGET: usize = 20_000;

/// Flatten a skeleton into productions.
pub fn flatten(lexicon: &Lexicon) -> Cfg {
    flatten_within(lexicon, usize::MAX).expect("no production budget was set")
}

/// As [`flatten`], but refused past `budget` productions.
pub fn flatten_within(lexicon: &Lexicon, budget: usize) -> Option<Cfg> {
    let mut builder = Builder {
        productions: Vec::new(),
        names: Vec::new(),
        by_rule: FxHashMap::default(),
        budget,
        over_budget: false,
    };

    // Reserve a nonterminal per skeleton rule first, so references resolve
    // regardless of declaration order.
    for rule in &lexicon.skeleton {
        let id = builder.fresh(rule.name.clone());
        builder.by_rule.insert(rule.rule.0, id);
    }

    for rule in &lexicon.skeleton {
        let lhs = builder.by_rule[&rule.rule.0];
        builder.emit_alternatives(lhs, &rule.body);
    }

    let start = builder
        .by_rule
        .get(&lexicon.root.0)
        .copied()
        .expect("the root rule is always in the skeleton");

    // Identical alternatives are common once a schema has been lowered - two
    // branches of an anyOf can flatten to the same right-hand side - and two
    // productions that differ only by index are a reduce/reduce conflict for
    // no reason.
    let mut seen = FxHashSet::default();
    builder
        .productions
        .retain(|production| seen.insert((production.lhs, production.rhs.clone())));

    if builder.over_budget || builder.productions.len() > budget {
        return None;
    }

    Some(Cfg {
        productions: builder.productions,
        nonterminal_names: builder.names,
        num_terminals: lexicon.terminals.len(),
        start,
    })
}

struct Builder {
    productions: Vec<Production>,
    names: Vec<String>,
    by_rule: FxHashMap<u32, u32>,
    budget: usize,
    over_budget: bool,
}

impl Builder {
    fn fresh(&mut self, name: String) -> u32 {
        let id = self.names.len() as u32;
        self.names.push(name);
        id
    }

    /// Emit one production per alternative of `body` for `lhs`.
    fn emit_alternatives(&mut self, lhs: u32, body: &SkeletonExpr) {
        match body {
            SkeletonExpr::Choice(alternatives) => {
                for alternative in alternatives {
                    let rhs = self.sequence(alternative);
                    self.productions.push(Production { lhs, rhs });
                }
            }
            other => {
                let rhs = self.sequence(other);
                self.productions.push(Production { lhs, rhs });
            }
        }
    }

    /// Lower an expression into a symbol sequence, inventing nonterminals for
    /// anything that cannot be spelled inline.
    fn sequence(&mut self, expr: &SkeletonExpr) -> Vec<Symbol> {
        match expr {
            SkeletonExpr::Empty => Vec::new(),
            SkeletonExpr::Terminal(terminal) => vec![Symbol::Terminal(*terminal)],
            SkeletonExpr::Nonterminal(rule) => {
                vec![Symbol::Nonterminal(self.by_rule[&rule.0])]
            }
            SkeletonExpr::Sequence(parts) => {
                let mut symbols = Vec::new();
                for part in parts {
                    symbols.extend(self.sequence(part));
                }
                symbols
            }
            SkeletonExpr::Choice(_) => {
                let id = self.fresh(format!("__choice{}", self.names.len()));
                self.emit_alternatives(id, expr);
                vec![Symbol::Nonterminal(id)]
            }
            SkeletonExpr::Repeat { inner, min, max } => {
                vec![Symbol::Nonterminal(self.repeat(inner, *min, *max))]
            }
        }
    }

    /// `x{min,max}` becomes a left-recursive nonterminal, which is what keeps
    /// the LR stack shallow.
    ///
    /// This was right-recursive, with a comment claiming the same benefit. That
    /// is the recursive-descent intuition and it is backwards here. An LR parser
    /// shifts every symbol of `tail -> body tail` and can reduce none of them
    /// until the repetition ends, so the stack grows by one per iteration and a
    /// hundred-character run of whitespace is a hundred entries deep. Written
    /// `tail -> tail body` each iteration reduces immediately and the stack does
    /// not move.
    ///
    /// The depth is not a detail here. It sets the stack bound, the length of
    /// the reduction chain that fires when a repetition finally ends, and the
    /// size of the window a device-side replay needs - three limits that had
    /// each been raised by hand after a document overran them.
    fn repeat(&mut self, inner: &SkeletonExpr, min: u32, max: Option<u32>) -> u32 {
        let body = self.sequence(inner);
        match max {
            None => {
                let tail = self.fresh(format!("__star{}", self.names.len()));
                let mut recursive = vec![Symbol::Nonterminal(tail)];
                recursive.extend(body.clone());
                self.productions.push(Production {
                    lhs: tail,
                    rhs: recursive,
                });
                self.productions.push(Production {
                    lhs: tail,
                    rhs: Vec::new(),
                });
                if min == 0 {
                    return tail;
                }
                let head = self.fresh(format!("__plus{}", self.names.len()));
                let mut required = Vec::new();
                for _ in 0..min {
                    required.extend(body.clone());
                }
                required.push(Symbol::Nonterminal(tail));
                self.productions.push(Production {
                    lhs: head,
                    rhs: required,
                });
                head
            }
            // A chain, not one production per count. Enumerating counts is
            // quadratic - `x{0,2048}` is 2049 productions totalling two million
            // symbols - and a bounded repeat is exactly the structure a stack
            // handles for free. One nonterminal per remaining count, each
            // production at most two symbols long, is linear.
            Some(max) => {
                if self.productions.len() + 2 * max as usize > self.budget {
                    self.over_budget = true;
                    return self.fresh(format!("__overbudget{}", self.names.len()));
                }
                let mut next = self.fresh(format!("__repeat{}_{}", self.names.len(), max));
                self.productions.push(Production {
                    lhs: next,
                    rhs: Vec::new(),
                });
                for count in (0..max).rev() {
                    let head = self.fresh(format!("__repeat{}_{}", self.names.len(), count));
                    let mut taken = body.clone();
                    taken.push(Symbol::Nonterminal(next));
                    self.productions.push(Production {
                        lhs: head,
                        rhs: taken,
                    });
                    if count >= min {
                        self.productions.push(Production {
                            lhs: head,
                            rhs: Vec::new(),
                        });
                    }
                    next = head;
                }
                next
            }
        }
    }
}
