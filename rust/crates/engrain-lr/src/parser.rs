//! A reference shift/reduce parser over the tables.
//!
//! The GPU kernel executes this same loop against a device-resident stack. The
//! CPU version exists so the tables can be tested for what they accept, rather
//! than only for their shape.

use crate::cfg::Symbol;
use crate::tables::{ACCEPT, ERROR, Tables, decode_reduce, decode_shift};

/// Outcome of feeding one terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Shifted,
    Accepted,
    Rejected,
}

/// A parse in progress.
#[derive(Debug, Clone)]
pub struct Parse<'a> {
    tables: &'a Tables,
    stack: Vec<usize>,
}

impl<'a> Parse<'a> {
    pub fn new(tables: &'a Tables) -> Self {
        Self {
            tables,
            stack: vec![tables.start_state],
        }
    }

    pub fn top(&self) -> usize {
        *self.stack.last().expect("the stack is never empty")
    }

    /// Terminals the grammar admits right now.
    ///
    /// This is the property the whole design rests on: it depends only on the
    /// stack top, so one lookup per token group decides a whole batch.
    pub fn admissible(&self) -> Vec<u32> {
        let mut terminals: Vec<u32> = self.tables.admissible(self.top()).collect();
        terminals.sort_unstable();
        terminals
    }

    pub fn feed(&mut self, terminal: u32) -> Step {
        loop {
            let action = self.tables.action(self.top(), terminal).unwrap_or(ERROR);
            if action == ERROR {
                return Step::Rejected;
            }
            if action == ACCEPT {
                return Step::Accepted;
            }
            if action > 0 {
                self.stack.push(decode_shift(action));
                return Step::Shifted;
            }
            let production = decode_reduce(action);
            let (lhs, arity) = self.tables.productions[production];
            for _ in 0..arity {
                self.stack.pop();
            }
            let Some(next) = self.tables.goto[self.top()].get(&lhs).copied() else {
                return Step::Rejected;
            };
            self.stack.push(next as usize);
        }
    }

    /// Feed a whole sentence, then end of input.
    pub fn accepts(tables: &'a Tables, sentence: &[Symbol]) -> bool {
        let mut parse = Parse::new(tables);
        for symbol in sentence {
            let Symbol::Terminal(terminal) = symbol else {
                return false;
            };
            if parse.feed(terminal.0) != Step::Shifted {
                return false;
            }
        }
        parse.feed(tables.eof) == Step::Accepted
    }
}
