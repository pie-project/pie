//! The context-free skeleton as a plain production set.
//!
//! [`SkeletonExpr`] is a tree with choice, sequence and repetition; an LR
//! construction wants flat productions. Flattening introduces a fresh
//! nonterminal wherever a nested choice or a repetition appears, which is the
//! usual EBNF-to-BNF expansion.

use gpugrammar_lex::TerminalId;
use gpugrammar_lex::lexicon::{Lexicon, SkeletonExpr};
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

/// Flatten a skeleton into productions.
pub fn flatten(lexicon: &Lexicon) -> Cfg {
    let mut builder = Builder {
        productions: Vec::new(),
        names: Vec::new(),
        by_rule: FxHashMap::default(),
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

    Cfg {
        productions: builder.productions,
        nonterminal_names: builder.names,
        num_terminals: lexicon.terminals.len(),
        start,
    }
}

struct Builder {
    productions: Vec<Production>,
    names: Vec<String>,
    by_rule: FxHashMap<u32, u32>,
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

    /// `x{min,max}` becomes a right-recursive nonterminal, which keeps the LR
    /// stack shallow in the unbounded case.
    fn repeat(&mut self, inner: &SkeletonExpr, min: u32, max: Option<u32>) -> u32 {
        let body = self.sequence(inner);
        match max {
            None => {
                let tail = self.fresh(format!("__star{}", self.names.len()));
                let mut recursive = body.clone();
                recursive.push(Symbol::Nonterminal(tail));
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
            Some(max) => {
                let head = self.fresh(format!("__repeat{}", self.names.len()));
                for count in min..=max {
                    let mut rhs = Vec::new();
                    for _ in 0..count {
                        rhs.extend(body.clone());
                    }
                    self.productions.push(Production { lhs: head, rhs });
                }
                head
            }
        }
    }
}
