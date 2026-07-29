//! LALR(1) table construction.
//!
//! The GPU runtime needs three things from the parser: an ACTION row per state
//! so that a terminal's admissibility follows from the stack top, a GOTO table
//! so a reduction can be finished on device, and production arities so the
//! stack can be popped. LALR is the target rather than canonical LR(1) because
//! the grammars these front ends emit reach thousands of rules, where canonical
//! state counts explode while the language accepted is the same for every
//! grammar without a mergeable-lookahead conflict.
//!
//! Lookaheads come from the standard propagation algorithm: build the LR(0)
//! automaton, discover which lookaheads each kernel item generates
//! spontaneously and which it passes on, then iterate to a fixpoint.

use std::collections::{BTreeSet, VecDeque};

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{Cfg, Production, Symbol};

/// No action. Zero so a sparse row's default is an error.
pub const ERROR: i32 = 0;
/// Accept. Distinct from every shift and reduce encoding.
pub const ACCEPT: i32 = i32::MIN;

pub fn encode_shift(state: usize) -> i32 {
    state as i32 + 1
}

pub fn decode_shift(action: i32) -> usize {
    debug_assert!(action > 0);
    (action - 1) as usize
}

pub fn encode_reduce(production: usize) -> i32 {
    -(production as i32 + 1)
}

pub fn decode_reduce(action: i32) -> usize {
    debug_assert!(action < 0 && action != ACCEPT);
    (-action - 1) as usize
}

/// An LR(0) item: a position inside a production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Item {
    production: u32,
    dot: u32,
}

/// The compiled parser.
#[derive(Debug, Clone)]
pub struct Tables {
    /// `action[state]` maps a terminal (with EOF last) to its encoded actions.
    ///
    /// Usually one. A reduce/reduce conflict leaves two, because the grammars a
    /// JSON Schema lowers to are genuinely ambiguous where its `oneOf` branches
    /// overlap - measured, by building canonical LR(1) tables and finding that
    /// every conflict survives them - and a mask does not need the ambiguity
    /// resolved. It needs to know whether *some* derivation admits the next
    /// token, so the runtime follows both.
    pub action: Vec<FxHashMap<u32, Vec<i32>>>,
    /// `goto[state]` maps a nonterminal to the next state.
    pub goto: Vec<FxHashMap<u32, u32>>,
    /// `(lhs, rhs_len)` per production, in the augmented numbering.
    pub productions: Vec<(u32, u32)>,
    pub num_terminals: usize,
    pub eof: u32,
    pub start_state: usize,
    /// Shift/reduce conflicts resolved in favour of shift. Reported rather than
    /// hidden: the resolution is standard and, for the adjacent optionals a
    /// lowered schema produces, language-preserving — but it is a choice the
    /// caller is entitled to see, and a grammar with many of them deserves a
    /// second look.
    pub shift_reduce_resolved: usize,
}

impl Tables {
    pub fn num_states(&self) -> usize {
        self.action.len()
    }

    pub fn action_entries(&self) -> usize {
        self.action
            .iter()
            .map(|row| row.values().map(Vec::len).sum::<usize>())
            .sum()
    }

    /// How many `(state, terminal)` cells hold more than one action.
    pub fn conflicts(&self) -> usize {
        self.action
            .iter()
            .map(|row| row.values().filter(|actions| actions.len() > 1).count())
            .sum()
    }

    /// The most actions any one cell holds, which bounds how far a replay forks.
    pub fn widest_cell(&self) -> usize {
        self.action
            .iter()
            .flat_map(|row| row.values().map(Vec::len))
            .max()
            .unwrap_or(1)
    }

    /// Terminals admissible in `state`, which is what the group check needs.
    pub fn admissible(&self, state: usize) -> impl Iterator<Item = u32> + '_ {
        self.action[state].keys().copied()
    }

    /// The encoded action for a terminal, or `None` when the row has no entry.
    ///
    /// The first of them where there are several. Callers that have to follow
    /// every derivation read `action[state]` directly.
    pub fn action(&self, state: usize, terminal: u32) -> Option<i32> {
        self.action[state]
            .get(&terminal)
            .and_then(|actions| actions.first().copied())
    }
}

/// A grammar that is not LALR(1).
#[derive(Debug)]
pub struct Conflict {
    pub state: usize,
    pub terminal: u32,
    pub existing: i32,
    pub incoming: i32,
}

/// Build LALR(1) tables for `cfg`.
pub fn build(cfg: &Cfg) -> Result<Tables> {
    let augmented = augment(cfg);
    let eof = cfg.num_terminals as u32;
    let dummy = eof + 1;

    let automaton = Lr0::build(&augmented, cfg.num_nonterminals());
    let lookaheads = propagate(&augmented, &automaton, eof, dummy);

    let mut resolved = 0usize;
    let mut action: Vec<FxHashMap<u32, Vec<i32>>> =
        vec![FxHashMap::default(); automaton.states.len()];
    let mut goto: Vec<FxHashMap<u32, u32>> = vec![FxHashMap::default(); automaton.states.len()];

    for (index, state) in automaton.states.iter().enumerate() {
        for (symbol, target) in &automaton.transitions[index] {
            match symbol {
                Symbol::Terminal(terminal) => {
                    set_action(
                        &mut action,
                        &mut resolved,
                        index,
                        terminal.0,
                        encode_shift(*target),
                    )?;
                }
                Symbol::Nonterminal(nonterminal) => {
                    goto[index].insert(*nonterminal, *target as u32);
                }
            }
        }

        for item in closure0(&augmented, state) {
            let production = &augmented[item.production as usize];
            if item.dot as usize != production.rhs.len() {
                continue;
            }
            if item.production == 0 {
                set_action(&mut action, &mut resolved, index, eof, ACCEPT)?;
                continue;
            }
            let Some(follow) = lookaheads.get(&(index, item)) else {
                continue;
            };
            for terminal in follow {
                set_action(
                    &mut action,
                    &mut resolved,
                    index,
                    *terminal,
                    encode_reduce(item.production as usize),
                )?;
            }
        }
    }

    Ok(Tables {
        action,
        goto,
        productions: augmented
            .iter()
            .map(|production| (production.lhs, production.rhs.len() as u32))
            .collect(),
        num_terminals: cfg.num_terminals,
        eof,
        start_state: 0,
        shift_reduce_resolved: resolved,
    })
}

/// Build canonical LR(1) tables, as a diagnostic rather than for the runtime.
///
/// Every conflict this corpus produces is reduce/reduce and none is
/// shift/reduce, which is the signature of LALR's one weakness: it merges LR(1)
/// states that share an LR(0) core, and merging their lookahead sets can create
/// a reduce/reduce conflict the grammar does not really have. Canonical LR(1)
/// never merges, so a grammar that builds here and not with `build` had an
/// artefact rather than an ambiguity - and an artefact is what IELR(1) removes
/// while keeping LALR's size.
///
/// Refused past `budget` states, because canonical state counts are what LALR
/// exists to avoid and an unbounded build would hang rather than answer.
pub fn build_canonical(cfg: &Cfg, budget: usize) -> Result<Tables> {
    let augmented = augment(cfg);
    let eof = cfg.num_terminals as u32;
    let nullable = nullable_sets(&augmented);
    let first = first_sets(&augmented, &nullable);

    let closure = |kernel: &BTreeSet<(Item, u32)>| -> BTreeSet<(Item, u32)> {
        let mut items: BTreeSet<(Item, u32)> = BTreeSet::new();
        for (item, ahead) in kernel {
            for entry in closure1(&augmented, &nullable, &first, *item, *ahead) {
                items.insert(entry);
            }
        }
        items
    };

    let start: BTreeSet<(Item, u32)> = BTreeSet::from([(
        Item {
            production: 0,
            dot: 0,
        },
        eof,
    )]);
    let mut states = vec![start.clone()];
    let mut index: FxHashMap<BTreeSet<(Item, u32)>, usize> = FxHashMap::default();
    index.insert(start, 0);
    let mut transitions: Vec<Vec<(Symbol, usize)>> = vec![Vec::new()];
    let mut queue = VecDeque::from([0usize]);

    while let Some(current) = queue.pop_front() {
        let items = closure(&states[current]);
        let mut moves: FxHashMap<Symbol, BTreeSet<(Item, u32)>> = FxHashMap::default();
        for (item, ahead) in &items {
            let production = &augmented[item.production as usize];
            let Some(symbol) = production.rhs.get(item.dot as usize) else {
                continue;
            };
            moves.entry(*symbol).or_default().insert((
                Item {
                    production: item.production,
                    dot: item.dot + 1,
                },
                *ahead,
            ));
        }
        let mut sorted: Vec<_> = moves.into_iter().collect();
        sorted.sort_by_key(|(symbol, _)| *symbol);
        for (symbol, kernel) in sorted {
            let target = match index.get(&kernel) {
                Some(&existing) => existing,
                None => {
                    if states.len() >= budget {
                        bail!("canonical LR(1) exceeds {budget} states");
                    }
                    let fresh = states.len();
                    states.push(kernel.clone());
                    transitions.push(Vec::new());
                    index.insert(kernel, fresh);
                    queue.push_back(fresh);
                    fresh
                }
            };
            transitions[current].push((symbol, target));
        }
    }

    let mut resolved = 0usize;
    let mut action: Vec<FxHashMap<u32, Vec<i32>>> = vec![FxHashMap::default(); states.len()];
    let mut goto: Vec<FxHashMap<u32, u32>> = vec![FxHashMap::default(); states.len()];
    for (state, kernel) in states.iter().enumerate() {
        for (symbol, target) in &transitions[state] {
            match symbol {
                Symbol::Terminal(terminal) => set_action(
                    &mut action,
                    &mut resolved,
                    state,
                    terminal.0,
                    encode_shift(*target),
                )?,
                Symbol::Nonterminal(nonterminal) => {
                    goto[state].insert(*nonterminal, *target as u32);
                }
            }
        }
        for (item, ahead) in closure(kernel) {
            let production = &augmented[item.production as usize];
            if item.dot as usize != production.rhs.len() {
                continue;
            }
            if item.production == 0 {
                set_action(&mut action, &mut resolved, state, eof, ACCEPT)?;
                continue;
            }
            set_action(
                &mut action,
                &mut resolved,
                state,
                ahead,
                encode_reduce(item.production as usize),
            )?;
        }
    }

    Ok(Tables {
        action,
        goto,
        productions: augmented
            .iter()
            .map(|production| (production.lhs, production.rhs.len() as u32))
            .collect(),
        num_terminals: cfg.num_terminals,
        eof,
        start_state: 0,
        shift_reduce_resolved: resolved,
    })
}

fn set_action(
    action: &mut [FxHashMap<u32, Vec<i32>>],
    resolved: &mut usize,
    state: usize,
    terminal: u32,
    incoming: i32,
) -> Result<()> {
    let held = action[state].entry(terminal).or_default();
    if held.contains(&incoming) {
        return Ok(());
    }
    match held.first().copied() {
        Some(existing) => {
            // Shift wins over reduce, as in every parser generator since yacc.
            // The conflicts a lowered schema produces there are adjacent
            // optionals - `ws? ws?` from whitespace allowed between every pair
            // of tokens - where both parses accept the same strings and
            // shifting means "this optional takes it".
            let shift = existing.max(incoming);
            let reduce = existing.min(incoming);
            if shift > 0 && reduce < 0 && reduce != ACCEPT {
                held.clear();
                held.push(shift);
                *resolved += 1;
                return Ok(());
            }
            // Reduce/reduce has no such reading, and on these grammars it is
            // not an artefact of LALR's state merging - canonical LR(1) has it
            // too. So it is kept rather than refused, and the runtime follows
            // both derivations. Ordered so the tables are deterministic.
            held.push(incoming);
            held.sort_unstable();
            Ok(())
        }
        None => {
            held.push(incoming);
            Ok(())
        }
    }
}
/// Prepend `S' -> S`, so accepting is a single distinguished item.
fn augment(cfg: &Cfg) -> Vec<Production> {
    let mut productions = Vec::with_capacity(cfg.productions.len() + 1);
    productions.push(Production {
        lhs: u32::MAX,
        rhs: vec![Symbol::Nonterminal(cfg.start)],
    });
    productions.extend(cfg.productions.iter().cloned());
    productions
}

struct Lr0 {
    states: Vec<BTreeSet<Item>>,
    transitions: Vec<Vec<(Symbol, usize)>>,
}

impl Lr0 {
    fn build(productions: &[Production], _num_nonterminals: usize) -> Self {
        let start = BTreeSet::from([Item {
            production: 0,
            dot: 0,
        }]);
        let mut states = vec![start.clone()];
        let mut index: FxHashMap<BTreeSet<Item>, usize> = FxHashMap::default();
        index.insert(start.clone(), 0);
        let mut transitions: Vec<Vec<(Symbol, usize)>> = vec![Vec::new()];
        let mut queue = VecDeque::from([0usize]);

        while let Some(current) = queue.pop_front() {
            let items = closure0(productions, &states[current]);
            let mut moves: FxHashMap<Symbol, BTreeSet<Item>> = FxHashMap::default();
            for item in items {
                let production = &productions[item.production as usize];
                let Some(symbol) = production.rhs.get(item.dot as usize) else {
                    continue;
                };
                moves.entry(*symbol).or_default().insert(Item {
                    production: item.production,
                    dot: item.dot + 1,
                });
            }
            let mut sorted: Vec<_> = moves.into_iter().collect();
            sorted.sort_by_key(|(symbol, _)| *symbol);
            for (symbol, kernel) in sorted {
                let target = match index.get(&kernel) {
                    Some(&existing) => existing,
                    None => {
                        let fresh = states.len();
                        states.push(kernel.clone());
                        transitions.push(Vec::new());
                        index.insert(kernel, fresh);
                        queue.push_back(fresh);
                        fresh
                    }
                };
                transitions[current].push((symbol, target));
            }
        }

        Lr0 {
            states,
            transitions,
        }
    }
}

/// LR(0) closure of a kernel.
fn closure0(productions: &[Production], kernel: &BTreeSet<Item>) -> BTreeSet<Item> {
    let mut items = kernel.clone();
    let mut pending: Vec<Item> = kernel.iter().copied().collect();
    while let Some(item) = pending.pop() {
        let production = &productions[item.production as usize];
        let Some(Symbol::Nonterminal(nonterminal)) = production.rhs.get(item.dot as usize) else {
            continue;
        };
        for (index, candidate) in productions.iter().enumerate() {
            if candidate.lhs != *nonterminal {
                continue;
            }
            let fresh = Item {
                production: index as u32,
                dot: 0,
            };
            if items.insert(fresh) {
                pending.push(fresh);
            }
        }
    }
    items
}

/// Lookahead sets for every reducible kernel item, by propagation.
fn propagate(
    productions: &[Production],
    automaton: &Lr0,
    eof: u32,
    dummy: u32,
) -> FxHashMap<(usize, Item), FxHashSet<u32>> {
    let nullable = nullable_sets(productions);
    let first = first_sets(productions, &nullable);
    let mut lookaheads: FxHashMap<(usize, Item), FxHashSet<u32>> = FxHashMap::default();
    let mut links: Vec<((usize, Item), (usize, Item))> = Vec::new();

    lookaheads
        .entry((
            0,
            Item {
                production: 0,
                dot: 0,
            },
        ))
        .or_default()
        .insert(eof);

    for (index, state) in automaton.states.iter().enumerate() {
        for kernel in state {
            let seeded = closure1(productions, &nullable, &first, *kernel, dummy);
            for (item, lookahead) in seeded {
                let production = &productions[item.production as usize];
                let Some(symbol) = production.rhs.get(item.dot as usize) else {
                    continue;
                };
                let Some((_, target)) = automaton.transitions[index]
                    .iter()
                    .find(|(candidate, _)| candidate == symbol)
                else {
                    continue;
                };
                let moved = Item {
                    production: item.production,
                    dot: item.dot + 1,
                };
                if lookahead == dummy {
                    links.push(((index, *kernel), (*target, moved)));
                } else {
                    lookaheads
                        .entry((*target, moved))
                        .or_default()
                        .insert(lookahead);
                }
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (from, to) in &links {
            let Some(source) = lookaheads.get(from).cloned() else {
                continue;
            };
            let target = lookaheads.entry(*to).or_default();
            for terminal in source {
                changed |= target.insert(terminal);
            }
        }
    }

    // Reductions read the lookahead of the closure item, not just the kernel,
    // so push kernel lookaheads through the closure once more.
    let mut result: FxHashMap<(usize, Item), FxHashSet<u32>> = FxHashMap::default();
    for (index, state) in automaton.states.iter().enumerate() {
        for kernel in state {
            let inherited = lookaheads
                .get(&(index, *kernel))
                .cloned()
                .unwrap_or_default();
            for (item, lookahead) in closure1(productions, &nullable, &first, *kernel, dummy) {
                let production = &productions[item.production as usize];
                if item.dot as usize != production.rhs.len() {
                    continue;
                }
                let entry = result.entry((index, item)).or_default();
                if lookahead == dummy {
                    entry.extend(inherited.iter().copied());
                } else {
                    entry.insert(lookahead);
                }
            }
        }
    }
    result
}

/// LR(1) closure of a single seeded item.
fn closure1(
    productions: &[Production],
    nullable: &[bool],
    first: &[FxHashSet<u32>],
    seed: Item,
    lookahead: u32,
) -> Vec<(Item, u32)> {
    let mut items: FxHashSet<(Item, u32)> = FxHashSet::default();
    items.insert((seed, lookahead));
    let mut pending = vec![(seed, lookahead)];

    while let Some((item, ahead)) = pending.pop() {
        let production = &productions[item.production as usize];
        let Some(Symbol::Nonterminal(nonterminal)) = production.rhs.get(item.dot as usize) else {
            continue;
        };
        let rest = &production.rhs[item.dot as usize + 1..];
        let mut heads = first_of(productions, nullable, first, rest);
        if nullable_sequence(nullable, rest) {
            heads.insert(ahead);
        }
        for (index, candidate) in productions.iter().enumerate() {
            if candidate.lhs != *nonterminal {
                continue;
            }
            for head in &heads {
                let fresh = (
                    Item {
                        production: index as u32,
                        dot: 0,
                    },
                    *head,
                );
                if items.insert(fresh) {
                    pending.push(fresh);
                }
            }
        }
    }
    items.into_iter().collect()
}

/// FIRST for every nonterminal, indexed by nonterminal id.
fn first_sets(productions: &[Production], nullable: &[bool]) -> Vec<FxHashSet<u32>> {
    let count = productions
        .iter()
        .filter_map(|production| {
            (production.lhs != u32::MAX).then_some(production.lhs as usize + 1)
        })
        .max()
        .unwrap_or(0);
    let mut first = vec![FxHashSet::default(); count];
    let mut changed = true;
    while changed {
        changed = false;
        for production in productions {
            if production.lhs == u32::MAX {
                continue;
            }
            let lhs = production.lhs as usize;
            for symbol in &production.rhs {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        changed |= first[lhs].insert(terminal.0);
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        let inherited: Vec<u32> =
                            first[*nonterminal as usize].iter().copied().collect();
                        for head in inherited {
                            changed |= first[lhs].insert(head);
                        }
                        if !nullable[*nonterminal as usize] {
                            break;
                        }
                    }
                }
            }
        }
    }
    first
}

/// The terminals a symbol sequence can begin with.
///
/// A nullable symbol has to be looked past, not stopped at. Optional parts are
/// everywhere in a lowered schema — whitespace, a sign, an absent property — and
/// two of them in a row is the common case: `"id" ws? ":" ws? "-"? digits`.
/// Stopping at the first symbol makes the digits invisible as a lookahead, so
/// the parser has no action for them and rejects a number it should accept.
fn first_of(
    productions: &[Production],
    nullable: &[bool],
    first: &[FxHashSet<u32>],
    symbols: &[Symbol],
) -> FxHashSet<u32> {
    let _ = productions;
    let mut heads = FxHashSet::default();
    for symbol in symbols {
        match symbol {
            Symbol::Terminal(terminal) => {
                heads.insert(terminal.0);
                return heads;
            }
            Symbol::Nonterminal(nonterminal) => {
                heads.extend(first[*nonterminal as usize].iter().copied());
                if !nullable[*nonterminal as usize] {
                    return heads;
                }
            }
        }
    }
    heads
}

/// Which nonterminals derive the empty string, to a fixpoint.
///
/// Not a one-level test: flattening produces chains such as
/// `properties -> choice` with `choice -> ε`, so a nonterminal can be nullable
/// without having an empty production of its own.
fn nullable_sets(productions: &[Production]) -> Vec<bool> {
    let count = productions
        .iter()
        .filter_map(|production| {
            (production.lhs != u32::MAX).then_some(production.lhs as usize + 1)
        })
        .max()
        .unwrap_or(0);
    let mut nullable = vec![false; count];
    let mut changed = true;
    while changed {
        changed = false;
        for production in productions {
            if production.lhs == u32::MAX || nullable[production.lhs as usize] {
                continue;
            }
            let empty = production.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => false,
                Symbol::Nonterminal(nonterminal) => nullable[*nonterminal as usize],
            });
            if empty {
                nullable[production.lhs as usize] = true;
                changed = true;
            }
        }
    }
    nullable
}

fn nullable_sequence(nullable: &[bool], symbols: &[Symbol]) -> bool {
    symbols.iter().all(|symbol| match symbol {
        Symbol::Terminal(_) => false,
        Symbol::Nonterminal(nonterminal) => nullable[*nonterminal as usize],
    })
}
