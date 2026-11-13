//! Config keys that are gone, and the refusal each one earns.
//!
//! `deny_unknown_fields` alone answers a removed key with "unknown field",
//! which reads as a typo. A removed key instead keeps its field, typed
//! [`Removed`], so a file that still states it fails to parse with why it
//! went and what to do:
//!
//! ```text
//! `max_state_slots` was removed: state now shares the device budget with kv
//! pages; delete the key
//! ```
//!
//! Removing a key is one entry in [`removed_keys!`] and its field's type:
//!
//! ```ignore
//! removed_keys! {
//!     MaxStateSlots = "max_state_slots":
//!         "state now shares the device budget with kv pages";
//! }
//!
//! #[serde(default, skip_serializing)]
//! pub max_state_slots: Removed<MaxStateSlots>,
//! ```
//!
//! [`Removed`] has no `Serialize`, so a struct that derives it fails to
//! compile until the field says `skip_serializing`: nothing that writes a
//! config (`pie config`, the embedded settings) can emit a removed key.
//! The schema `pie config list` walks skips the field, and `pie config set`
//! of a removed key answers with the same refusal.

use std::fmt;
use std::marker::PhantomData;

use serde::de::{Deserialize, Deserializer, Error};

/// A removed key: its name as a file spells it, and why it went.
pub trait Gone {
    const KEY: &'static str;
    const WHY: &'static str;
}

/// A field that no longer configures anything. Absent, it is its default;
/// stated, it refuses the whole file with [`Gone::WHY`].
pub struct Removed<G>(PhantomData<G>);

impl<G> Default for Removed<G> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<G> Clone for Removed<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G> Copy for Removed<G> {}

impl<G> PartialEq for Removed<G> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<G> Eq for Removed<G> {}

impl<G: Gone> fmt::Debug for Removed<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Removed({})", G::KEY)
    }
}

impl<'de, G: Gone> Deserialize<'de> for Removed<G> {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(refusal(G::KEY, G::WHY)))
    }
}

fn refusal(key: &str, why: &str) -> String {
    format!("`{key}` was removed: {why}; delete the key")
}

/// Declares each removed key's marker type and lists it in [`REMOVED`].
macro_rules! removed_keys {
    ($($(#[$meta:meta])* $marker:ident = $key:literal: $why:literal;)*) => {
        $(
            $(#[$meta])*
            pub enum $marker {}

            impl Gone for $marker {
                const KEY: &'static str = $key;
                const WHY: &'static str = $why;
            }
        )*

        /// Every removed key, as a file spells it, with why it went.
        pub const REMOVED: &[(&str, &str)] = &[$(($key, $why)),*];
    };
}

removed_keys! {
    DeviceWeightBudget = "device_weight_budget":
        "dense weights always live in device memory; only routed experts and token-id tables leave it, through the weight cache. A model whose dense weights do not fit the device needs a quantized deployment or tensor parallelism";
    HostWeightBudget = "host_weight_budget":
        "dense weights always live in device memory; only routed experts and token-id tables leave it, through the weight cache, which sizes its host copy itself";
    HostWeightDeferred = "host_weight_deferred":
        "the engine's own weight tier it deferred is gone: dense weights always live in device memory, and the weight cache fills its own host copy";
    MaxStateSlots = "max_state_slots":
        "state now shares the device budget with kv pages";
    WgpuDeviceMemory = "device_memory":
        "the engine measures what the device has; state `[engine.memory] budget_device` to cap what a load takes";
}

/// The refusal for `key` (dotted or bare) if it names a removed key.
pub fn refusal_for(key: &str) -> Option<String> {
    refusal_in(REMOVED, key)
}

fn refusal_in(removed: &[(&str, &str)], key: &str) -> Option<String> {
    let leaf = key.rsplit('.').next().unwrap_or(key);
    removed
        .iter()
        .find(|(gone, _)| *gone == leaf)
        .map(|(gone, why)| refusal(gone, why))
}
