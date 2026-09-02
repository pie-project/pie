//! The resolved index: names to addresses.
//!
//! A [`Catalog`] is what a consumer queries, and it is deliberately not a
//! [`Manifest`](crate::format::Manifest). A manifest is one file's own claim,
//! addressed through that file's shard table. A catalog is process-local: its
//! addresses are [`StoreId`]s, so it can span files that never heard of each
//! other: a sharded snapshot, a mixed set, or a single foreign file, without
//! anyone having to claim an identity nobody wrote down.
//!
//! Every projection in the compat crate produces one of these. None of them
//! produces a manifest, because none of them ever had one.

use std::collections::BTreeMap;

use crate::error::Result;
use crate::format::cbor::Value;
use crate::format::{canonical_term, check_shape, Blocks, Digest, Leaf, Plane, Term};
use crate::provide::store::StoreId;

/// Where decoded bytes are: a range of one store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    pub store: StoreId,
    pub offset: u64,
    pub len: u64,
}

impl Location {
    /// Largest power of two dividing the offset. The pointer alignment of a
    /// whole-file mapping is `min(alignment, page_size)`.
    pub fn alignment(&self) -> u64 {
        if self.offset == 0 {
            return crate::provide::store::page_size();
        }
        1u64 << self.offset.trailing_zeros().min(63)
    }
}

/// How a tensor's bytes can be reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    /// Raw decoded bytes, exactly at this range. Addressable and mappable.
    At(Location),
    /// Stored at this range under an encoding profile. The range is *not* the
    /// tensor, so it is not an address a consumer can read directly.
    Encoded {
        at: Location,
        encoding: String,
        decoded_len: u64,
    },
    /// Only the projection that opened the file can produce these bytes, as
    /// with a deflated archive entry or a chunked dataset.
    Opaque {
        store: StoreId,
        key: u64,
        decoded_len: u64,
    },
}

impl Payload {
    pub fn location(&self) -> Option<Location> {
        match self {
            Payload::At(at) => Some(*at),
            _ => None,
        }
    }

    pub fn store(&self) -> StoreId {
        match self {
            Payload::At(at) | Payload::Encoded { at, .. } => at.store,
            Payload::Opaque { store, .. } => *store,
        }
    }

    pub fn decoded_len(&self) -> u64 {
        match self {
            Payload::At(at) => at.len,
            Payload::Encoded { decoded_len, .. } | Payload::Opaque { decoded_len, .. } => {
                *decoded_len
            }
        }
    }
}

/// One named tensor: what it is, and where its bytes are.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub shape: Vec<u64>,
    /// Absent only under a layout that defines the values itself.
    pub term: Option<Term>,
    /// Absent ⇒ canonical layout.
    pub layout: Option<String>,
    pub attributes: Option<Value>,
    pub payload: Payload,
    /// Over decoded bytes, when the format carries one. Most foreign formats
    /// do not.
    pub digest: Option<Digest>,
    pub blocks: Option<Blocks>,
}

impl Entry {
    /// A tensor of one leaf at a raw range, which is what most formats have.
    pub fn leaf(shape: Vec<u64>, leaf: Leaf, at: Location) -> Self {
        Entry::at(shape, Term::Leaf(leaf), at)
    }

    /// A tensor of any term in canonical layout at a raw range.
    pub fn at(shape: Vec<u64>, term: Term, at: Location) -> Self {
        Entry {
            shape,
            term: Some(term),
            layout: None,
            attributes: None,
            payload: Payload::At(at),
            digest: None,
            blocks: None,
        }
    }

    pub fn num_elements(&self) -> Result<u64> {
        check_shape(&self.shape)
    }

    /// The planes under the canonical layout; `Unsupported` under a named
    /// layout or without a term.
    pub fn planes(&self) -> Result<Vec<Plane>> {
        canonical_term(self.term.as_ref(), self.layout.as_deref())?.planes(&self.shape)
    }

    pub(crate) fn store(&self) -> StoreId {
        self.payload.store()
    }
}

/// Names to entries, sorted, with the file-level attributes of whatever was
/// opened.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: BTreeMap<String, Entry>,
    attributes: Option<Value>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a tensor, returning the entry it displaced (as `BTreeMap` does).
    pub fn insert(&mut self, name: impl Into<String>, entry: Entry) -> Option<Entry> {
        self.entries.insert(name.into(), entry)
    }

    pub fn set_attributes(&mut self, attributes: Option<Value>) {
        self.attributes = attributes;
    }

    pub fn attributes(&self) -> Option<&Value> {
        self.attributes.as_ref()
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.get(name)
    }

    /// The entry and the name as this catalog stores it, so a handle can
    /// borrow the stored `&str`.
    pub(crate) fn get_key_value(&self, name: &str) -> Option<(&str, &Entry)> {
        self.entries
            .get_key_value(name)
            .map(|(k, v)| (k.as_str(), v))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Entry)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub(crate) fn into_iter_sorted(self) -> impl Iterator<Item = (String, Entry)> {
        self.entries.into_iter()
    }

    pub(crate) fn rebase(&mut self, f: impl Fn(StoreId) -> StoreId) {
        for entry in self.entries.values_mut() {
            match &mut entry.payload {
                Payload::At(at) => at.store = f(at.store),
                Payload::Encoded { at, .. } => at.store = f(at.store),
                Payload::Opaque { store, .. } => *store = f(*store),
            }
        }
    }
}
