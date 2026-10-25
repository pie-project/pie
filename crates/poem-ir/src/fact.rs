use serde::{Deserialize, Serialize};

use crate::{Guard, Request};

/// A runtime switch: a fact the runtime derives from what a lane carries and
/// what its program reads, so an inferlet cannot state it apart from the
/// buffers a kernel reads under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Builtin {
    /// The lane carries a custom attention mask (`has_mask`).
    Masked,
    /// The lane carries adapter routes (`has_adapter`).
    Adapted,
    /// The lane carries media patches (`has_media`).
    Media,
    /// The lane queries one token (`single_token`).
    SingleToken,
    /// The lane's program reads its drafts (`drafts`).
    Drafts,
    /// The lane's program reads its attention scores (`scores`).
    Scores,
    /// The lane's rows are a prompt only a causal encoder-decoder's encoder
    /// reads (`encode`): a backend sets it on the leading rows it replays.
    Encode,
    /// The lane's recurrent state holds nothing yet: its cursor is 0, and
    /// this fire starts it from zeros (`fresh`).
    Fresh,
}

impl Builtin {
    pub const ALL: [Builtin; 8] = [
        Builtin::Masked,
        Builtin::Adapted,
        Builtin::Media,
        Builtin::SingleToken,
        Builtin::Drafts,
        Builtin::Scores,
        Builtin::Encode,
        Builtin::Fresh,
    ];

    #[must_use]
    pub fn holds(self, request: &Request) -> bool {
        match self {
            Builtin::Masked => request.has_custom_mask(),
            Builtin::Adapted => request.has_adapter(),
            Builtin::Media => request.has_media(),
            Builtin::SingleToken => request.query_len() == 1,
            Builtin::Drafts => request.drafts(),
            Builtin::Scores => request.captures_scores(),
            Builtin::Encode => request.encodes(),
            Builtin::Fresh => request.fresh(),
        }
    }
}

/// One fact a trace branches on, holding one bit of a request's word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fact {
    Builtin(Builtin),
    /// The model's switch `name` is set to `value`. A switch's values are a
    /// bit each, so the rows a value does not hold for are one literal too.
    Switch {
        name: String,
        value: String,
    },
}

/// A switch a model declares: its name and the values a lane may set it to,
/// the first its default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwitchDecl {
    pub name: String,
    pub values: Vec<String>,
}

impl SwitchDecl {
    #[must_use]
    pub fn default_value(&self) -> &str {
        &self.values[0]
    }
}

/// The facts a trace branches on, in the order it first read them (the fact
/// at `k` is bit `k` of a request's word), and the switches its model
/// declares.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Facts {
    pub table: Vec<Fact>,
    #[serde(default)]
    pub switches: Vec<SwitchDecl>,
}

impl Facts {
    /// The guard over the rows `builtin` holds for, giving it a bit if the
    /// trace has not read it yet.
    pub fn builtin(&mut self, builtin: Builtin) -> Guard {
        self.bit_of(Fact::Builtin(builtin))
    }

    /// Declares the switch `name` over `values`, the first its default.
    /// Declaring it again over the same values is declaring it once.
    pub fn declare(&mut self, name: &str, values: &[&str]) -> Result<(), String> {
        if name.is_empty() {
            return Err("a switch is named".to_string());
        }
        if values.is_empty() {
            return Err(format!("the switch `{name}` declares no value"));
        }
        if let Some((at, value)) = values
            .iter()
            .enumerate()
            .find(|(at, value)| value.is_empty() || values[..*at].contains(value))
        {
            return Err(format!(
                "the switch `{name}` declares {} at {at}",
                if value.is_empty() {
                    "an empty value".to_string()
                } else {
                    format!("`{value}` twice")
                }
            ));
        }
        let decl = SwitchDecl {
            name: name.to_string(),
            values: values.iter().map(|v| (*v).to_string()).collect(),
        };
        match self.declared(name) {
            Some(prior) if *prior == decl => Ok(()),
            Some(prior) => Err(format!(
                "the switch `{name}` is declared over {:?} and then over {:?}",
                prior.values, decl.values
            )),
            None => {
                self.switches.push(decl);
                Ok(())
            }
        }
    }

    /// The switch `name` the model declares.
    #[must_use]
    pub fn declared(&self, name: &str) -> Option<&SwitchDecl> {
        self.switches.iter().find(|decl| decl.name == name)
    }

    /// The guard over the rows whose switch `name` is set to `value`.
    pub fn switch(&mut self, name: &str, value: &str) -> Guard {
        let Some(decl) = self.declared(name) else {
            panic!("the switch `{name}` is read but not declared");
        };
        assert!(
            decl.values.iter().any(|v| v == value),
            "the switch `{name}` declares no value `{value}`; it declares {:?}",
            decl.values
        );
        self.bit_of(Fact::Switch {
            name: name.to_string(),
            value: value.to_string(),
        })
    }

    /// Whether `value` is one the switch `name` declares.
    pub fn check(&self, name: &str, value: &str) -> Result<(), String> {
        let decl = self
            .declared(name)
            .ok_or_else(|| format!("this model declares no switch `{name}`"))?;
        if decl.values.iter().any(|v| v == value) {
            Ok(())
        } else {
            Err(format!(
                "the switch `{name}` takes {}, not `{value}`",
                decl.values
                    .iter()
                    .map(|v| format!("`{v}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
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

    /// The value the switch `name` takes on `request`: the one it sets, else
    /// the declared default.
    fn value_of<'a>(&'a self, request: &'a Request, name: &str) -> Option<&'a str> {
        request
            .switch(name)
            .or_else(|| self.declared(name).map(SwitchDecl::default_value))
    }

    /// The word a request's rows are classified by.
    #[must_use]
    pub fn word(&self, request: &Request) -> u64 {
        self.table.iter().enumerate().fold(0, |word, (bit, fact)| {
            let holds = match fact {
                Fact::Builtin(builtin) => builtin.holds(request),
                Fact::Switch { name, value } => self.value_of(request, name) == Some(value),
            };
            word | (u64::from(holds) << bit)
        })
    }

    /// Whether `word` is a plain lane's: no runtime switch on but a single
    /// token's, every switch at its default.
    #[must_use]
    pub fn plain(&self, word: u64) -> bool {
        self.defaults(word)
            && Builtin::ALL
                .into_iter()
                .all(|builtin| builtin == Builtin::SingleToken || !self.holds(word, builtin))
    }

    /// Whether `word` leaves every switch of the model at its default.
    #[must_use]
    pub fn defaults(&self, word: u64) -> bool {
        self.table.iter().enumerate().all(|(bit, fact)| match fact {
            Fact::Builtin(_) => true,
            Fact::Switch { name, value } => {
                (word & (1 << bit) != 0)
                    == (self.declared(name).map(SwitchDecl::default_value) == Some(value))
            }
        })
    }

    /// Whether the runtime switch `builtin` is on in `word`; off where the
    /// trace never reads it.
    #[must_use]
    pub fn holds(&self, word: u64, builtin: Builtin) -> bool {
        self.table
            .iter()
            .position(|fact| *fact == Fact::Builtin(builtin))
            .is_some_and(|bit| word & (1 << bit) != 0)
    }

    /// A request of every word a lane can be stamped with, the plainest
    /// first: each runtime switch on and off, a query of one token and of
    /// more, and each switch at its default, at each value the trace reads
    /// and at one value it does not.
    #[must_use]
    pub fn requests(&self) -> Vec<Request> {
        let mut guests: Vec<Vec<(&str, &str)>> = vec![Vec::new()];
        for decl in &self.switches {
            let read = |value: &str| {
                self.table.iter().any(|fact| {
                    matches!(fact, Fact::Switch { name, value: v } if *name == decl.name && v == value)
                })
            };
            let others = &decl.values[1..];
            let picks: Vec<&str> = others
                .iter()
                .filter(|v| read(v))
                .chain(others.iter().find(|v| !read(v)))
                .map(String::as_str)
                .collect();
            guests = guests
                .into_iter()
                .flat_map(|set| {
                    std::iter::once(set.clone()).chain(picks.iter().map(move |&pick| {
                        let mut set = set.clone();
                        set.push((decl.name.as_str(), pick));
                        set
                    }))
                })
                .collect();
        }
        let mut requests = Vec::with_capacity(guests.len() << 8);
        for guest in &guests {
            for bits in 0..1u32 << 8 {
                let request = Request::new(if bits & 1 == 0 { 1 } else { 2 }, bits & 2 != 0)
                    .adapted(bits & 4 != 0)
                    .drafting(bits & 8 != 0)
                    .capturing_scores(bits & 16 != 0)
                    .with_media(bits & 32 != 0)
                    .encoding(bits & 64 != 0)
                    .starting(bits & 128 != 0);
                requests.push(
                    guest
                        .iter()
                        .fold(request, |r, (name, value)| r.switched(name, value)),
                );
            }
        }
        requests.sort_by_key(Request::departures);
        requests
    }
}

#[cfg(test)]
mod tests {
    use super::{Builtin, Facts};
    use crate::Request;

    fn facts() -> Facts {
        let mut facts = Facts::default();
        facts
            .declare("stream", &["text", "image", "video"])
            .unwrap();
        facts
    }

    #[test]
    fn fact_every_case() {
        a_switch_reads_its_default_where_a_lane_leaves_it_unset();
        a_switch_is_declared_once_over_its_values();
        every_word_a_lane_can_take_is_requested();
    }

    fn a_switch_reads_its_default_where_a_lane_leaves_it_unset() {
        let mut facts = facts();
        let _ = facts.switch("stream", "text");
        let _ = facts.builtin(Builtin::SingleToken);
        assert_eq!(facts.word(&Request::new(1, false)), 0b11);
        assert_eq!(
            facts.word(&Request::new(2, false).switched("stream", "image")),
            0
        );
        assert!(facts.plain(0b11));
        assert!(facts.plain(0b01));
        assert!(!facts.plain(0b10), "an image lane is no plain one");
        assert!(facts.check("stream", "audio").is_err());
        assert!(facts.check("reading", "text").is_err());
    }

    fn a_switch_is_declared_once_over_its_values() {
        let mut facts = facts();
        facts
            .declare("stream", &["text", "image", "video"])
            .unwrap();
        assert!(facts.declare("stream", &["text", "image"]).is_err());
        assert!(facts.declare("mode", &[]).is_err());
        assert!(facts.declare("mode", &["a", "a"]).is_err());
        assert_eq!(facts.switches.len(), 1);
    }

    fn every_word_a_lane_can_take_is_requested() {
        let mut facts = facts();
        let _ = facts.switch("stream", "image");
        let words: std::collections::BTreeSet<u64> =
            facts.requests().iter().map(|r| facts.word(r)).collect();
        assert_eq!(words.into_iter().collect::<Vec<_>>(), vec![0, 1]);
        let requests = facts.requests();
        assert!(requests[0].plain());
        assert_eq!(
            requests.len(),
            3 << 8,
            "default, `image`, and `video` unread"
        );
    }
}
