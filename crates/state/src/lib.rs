//! What a pie cluster's state is.
//!
//! Three facts about a piece of state are kept apart. *What* it is: a content
//! key ([`page_hash`] over [`chain_slots`]), from the model's
//! [`ModelIdentity`] and the tokens it covers.
//! *How it was computed*: a [`NumericsClass`], since CUDA and Metal produce
//! different bits for one key. *How it is laid out*: a [`StateFormat`]. A
//! directory finds by key; class and format ride along, and whether one class
//! may use another's state is a policy's grant.
//!
//! Everything here is immutable once sealed. Mutable state has one owner and
//! is not described by this crate.

pub mod format;
pub mod identity;
pub mod key;

pub use format::{PageLayout, Plane, Planes, StateFormat, reshard};
pub use identity::{ModelIdentity, NumericsClass};
pub use key::{Hash256, cache_domain, chain_slots, page_hash};
