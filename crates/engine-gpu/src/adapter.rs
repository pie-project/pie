use std::collections::BTreeMap;

use crate::error::{Fault, Result};
use crate::weights::BankSeat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    Q,

    K,

    V,

    O,

    GateUp,

    Down,
}

impl Site {
    pub const ALL: [Site; 6] = [Site::Q, Site::K, Site::V, Site::O, Site::GateUp, Site::Down];

    #[must_use]
    pub const fn spelled(self) -> &'static str {
        match self {
            Site::Q => "q",
            Site::K => "k",
            Site::V => "v",
            Site::O => "o",
            Site::GateUp => "gate_up",
            Site::Down => "down",
        }
    }

    #[must_use]
    pub fn parse(word: &str) -> Option<Site> {
        Site::ALL.into_iter().find(|site| site.spelled() == word)
    }

    #[must_use]
    pub fn vocabulary() -> String {
        Site::ALL
            .iter()
            .map(|site| format!("`{}`", site.spelled()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    #[must_use]
    pub fn stated(site: Option<Site>) -> String {
        match site {
            Some(site) => format!("at site `{}`", site.spelled()),
            None => "at no stated site".to_string(),
        }
    }
}

#[must_use]
pub fn role_of(bank: &str) -> &str {
    parsed(bank).map_or(bank, |(_, _, role)| role)
}

#[must_use]
pub fn layer_of(bank: &str) -> u64 {
    parsed(bank).map_or(0, |(layer, _, _)| layer)
}

#[must_use]
pub fn site_of(bank: &str) -> Option<Site> {
    parsed(bank).and_then(|(_, site, _)| site)
}

fn parsed(bank: &str) -> Option<(u64, Option<Site>, &str)> {
    let (head, role) = bank.rsplit_once('.')?;
    let last = head.rsplit('.').next()?;

    if let Some(layer) = numbered(last) {
        return Some((layer, None, role));
    }

    let site = Site::parse(last)?;
    let (rest, _) = head.rsplit_once('.')?;
    let layer = numbered(rest.rsplit('.').next()?)?;
    Some((layer, Some(site), role))
}

fn numbered(part: &str) -> Option<u64> {
    match !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()) {
        true => part.parse().ok(),
        false => None,
    }
}

/// This load's banks as `engine::bundle` fills them.
#[must_use]
pub fn banks(seats: &[BankSeat]) -> Vec<engine::bundle::Bank<'_>> {
    seats
        .iter()
        .map(|seat| engine::bundle::Bank {
            name: &seat.name,
            rows: seat.rows,
            cols: seat.cols,
            slot: seat.slot,
            dtype: seat.dtype,
            shard: &seat.shard,
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    Shared {
        name: &'a str,
    },

    /// A bundle; every bind of the same bytes shares one slot.
    Bundle {
        bundle: &'a engine::bundle::Bundle<'a>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub slot: u32,

    pub shared: bool,

    pub landed: bool,

    pub key: Key,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    Shared(crate::blob::Stamp),

    /// A bundle, by the digest of its bytes (`engine::bundle::Bundle::key`).
    Content([u8; 32]),
}

#[derive(Debug, Default)]
pub struct Slots {
    seats: u32,

    held: BTreeMap<Key, (u32, u32)>,

    /// The slots no bind holds that still keep a bundle, the oldest release
    /// first: a bind of the same bytes takes its slot back unwritten, and a
    /// new adapter takes a free slot before it takes one of these.
    kept: Vec<(Key, u32)>,
}

#[derive(Debug, Clone, Copy)]
pub struct Grant {
    pub slot: u32,

    pub fresh: bool,
}

impl Slots {
    #[must_use]
    pub fn new(seats: u32) -> Slots {
        Slots {
            seats,
            held: BTreeMap::new(),
            kept: Vec::new(),
        }
    }

    #[must_use]
    pub fn seats(&self) -> u32 {
        self.seats
    }

    #[must_use]
    pub fn live(&self) -> usize {
        self.held.len()
    }

    pub fn acquire(&mut self, key: Key) -> Result<Grant> {
        if let Some((slot, count)) = self.held.get_mut(&key) {
            *count += 1;
            return Ok(Grant {
                slot: *slot,
                fresh: false,
            });
        }
        if let Some(at) = self.kept.iter().position(|(kept, _)| *kept == key) {
            let (_, slot) = self.kept.remove(at);
            self.held.insert(key, (slot, 1));
            return Ok(Grant { slot, fresh: false });
        }
        let taken: Vec<u32> = self
            .held
            .values()
            .map(|(slot, _)| *slot)
            .chain(self.kept.iter().map(|(_, slot)| *slot))
            .collect();
        let slot = match (0..self.seats).find(|slot| !taken.contains(slot)) {
            Some(slot) => slot,
            None if !self.kept.is_empty() => self.kept.remove(0).1,
            None => return Err(Fault::AdapterSlots { seats: self.seats }),
        };
        self.held.insert(key, (slot, 1));
        Ok(Grant { slot, fresh: true })
    }

    /// Gives back a hold whose write failed: its slot keeps nothing.
    pub fn abandon(&mut self, key: &Key) {
        self.release_key(key);
    }

    /// Gives back a hold. A bundle's slot is kept once the last hold goes,
    /// until another adapter needs it.
    pub fn release(&mut self, key: &Key) {
        if let Some(slot) = self.release_key(key)
            && matches!(key, Key::Content(_))
        {
            self.kept.push((key.clone(), slot));
        }
    }

    /// Drops one hold of `key`, and answers its slot if that was the last.
    fn release_key(&mut self, key: &Key) -> Option<u32> {
        let (slot, count) = self.held.get_mut(key)?;
        *count = count.saturating_sub(1);
        let (slot, last) = (*slot, *count == 0);
        if last {
            self.held.remove(key);
        }
        last.then_some(slot)
    }
}
