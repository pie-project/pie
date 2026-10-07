use serde::{Deserialize, Serialize};

use crate::{Guard, Request};

/// A fact the runtime derives from what a lane carries, so an inferlet
/// cannot state it apart from the buffers a kernel reads under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Builtin {
    /// The lane carries a custom attention mask.
    Masked,
    /// The lane carries adapter routes.
    Adapted,
    /// The lane carries media patches.
    Media,
    /// The lane queries one token.
    SingleToken,
}

impl Builtin {
    #[must_use]
    pub fn holds(self, request: &Request) -> bool {
        match self {
            Builtin::Masked => request.has_custom_mask(),
            Builtin::Adapted => request.has_adapter(),
            Builtin::Media => request.has_media(),
            Builtin::SingleToken => request.query_len() == 1,
        }
    }
}

/// One fact a trace branches on, holding one bit of a request's word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fact {
    Builtin(Builtin),
    /// A custom fact an inferlet turns on or leaves off.
    Flag(String),
    /// A custom choice an inferlet sets, holding where it is set to `value`.
    /// A choice's values are a bit each, so the rows a value does not hold
    /// for are one literal too.
    Choice {
        name: String,
        value: String,
    },
}

/// The facts a trace branches on, in the order it first read them; the
/// fact at `k` is bit `k` of a request's word.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Facts {
    pub table: Vec<Fact>,
}

impl Facts {
    /// The guard over the rows `builtin` holds for, giving it a bit if the
    /// trace has not read it yet.
    pub fn builtin(&mut self, builtin: Builtin) -> Guard {
        self.bit_of(Fact::Builtin(builtin))
    }

    /// The guard over the rows whose inferlet turned the flag `name` on.
    pub fn flag(&mut self, name: &str) -> Guard {
        assert!(
            Request::FLAGS.contains(&name),
            "no inferlet can set the flag `{name}` yet; the runtime carries {:?}",
            Request::FLAGS,
        );
        self.bit_of(Fact::Flag(name.to_string()))
    }

    /// The guard over the rows whose inferlet set the choice `name` to
    /// `value`.
    pub fn choice(&mut self, name: &str, value: &str) -> Guard {
        assert!(
            Request::CHOICES.contains(&name),
            "no inferlet can set the choice `{name}` yet; the runtime carries {:?}",
            Request::CHOICES,
        );
        self.bit_of(Fact::Choice {
            name: name.to_string(),
            value: value.to_string(),
        })
    }

    fn bit_of(&mut self, fact: Fact) -> Guard {
        let bit = match self.table.iter().position(|f| *f == fact) {
            Some(bit) => bit,
            None => {
                self.table.push(fact);
                self.table.len() - 1
            }
        };
        Guard::Fact(u8::try_from(bit).expect("a trace reads fewer than 256 facts"))
    }

    /// The values the trace reads of the choice `name`, in the order it
    /// first read them.
    pub fn values<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.table.iter().filter_map(move |fact| match fact {
            Fact::Choice { name: n, value } if n == name => Some(value.as_str()),
            _ => None,
        })
    }

    /// Whether every row `guard` holds for is one `builtin` holds for.
    #[must_use]
    pub fn implies(&self, guard: &Guard, builtin: Builtin) -> bool {
        let Some(bit) = self
            .table
            .iter()
            .position(|fact| *fact == Fact::Builtin(builtin))
        else {
            return false;
        };
        let mut bits = guard.referenced_bits();
        bits.push(bit as u8);
        bits.sort_unstable();
        bits.dedup();
        (0..1u64 << bits.len()).all(|pick| {
            let word = bits
                .iter()
                .enumerate()
                .filter(|(k, _)| pick & (1 << k) != 0)
                .fold(0u64, |word, (_, b)| word | (1 << b));
            !guard.holds(word) || word & (1 << bit) != 0
        })
    }

    /// The word a request's rows are classified by.
    #[must_use]
    pub fn word(&self, request: &Request) -> u64 {
        self.table.iter().enumerate().fold(0, |word, (bit, fact)| {
            let holds = match fact {
                Fact::Builtin(builtin) => builtin.holds(request),
                Fact::Flag(name) => request.flag(name),
                Fact::Choice { name, value } => request.choice(name) == Some(value.as_str()),
            };
            word | (u64::from(holds) << bit)
        })
    }
}
