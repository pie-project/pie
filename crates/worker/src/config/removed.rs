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
    MaxStateSlots = "max_state_slots":
        "state now shares the device budget with kv pages";
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

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;

    removed_keys! {
        OldKnob = "old_knob": "the planner sizes it now";
    }

    #[derive(Debug, Clone, Default, Deserialize, Serialize)]
    #[serde(default, deny_unknown_fields)]
    struct Options {
        kept: Option<u32>,
        #[serde(skip_serializing)]
        old_knob: Removed<OldKnob>,
    }

    #[test]
    fn removed_every_case() {
        a_file_without_the_key_parses();
        a_file_stating_the_key_is_refused_with_why();
        json_states_the_same_refusal();
        the_key_is_never_written();
        an_unknown_key_is_still_only_unknown();
        the_registry_answers_dotted_and_bare_keys();
    }

    fn a_file_without_the_key_parses() {
        let options: Options = toml::from_str("kept = 3\n").unwrap();
        assert_eq!(options.kept, Some(3));
    }

    fn a_file_stating_the_key_is_refused_with_why() {
        let err = toml::from_str::<Options>("kept = 3\nold_knob = 4\n")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`old_knob` was removed: the planner sizes it now; delete the key"),
            "got: {err}"
        );
        // Any value is refused, not only the type the key once took.
        assert!(toml::from_str::<Options>("old_knob = \"x\"\n").is_err());
        assert!(toml::from_str::<Options>("[old_knob]\na = 1\n").is_err());
    }

    fn json_states_the_same_refusal() {
        let err = serde_json::from_str::<Options>(r#"{"old_knob": null}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("`old_knob` was removed"), "got: {err}");
    }

    fn the_key_is_never_written() {
        let written = toml::to_string(&Options::default()).unwrap();
        assert!(!written.contains("old_knob"), "got: {written}");
        let json = serde_json::to_string(&Options::default()).unwrap();
        assert!(!json.contains("old_knob"), "got: {json}");
    }

    fn an_unknown_key_is_still_only_unknown() {
        let err = toml::from_str::<Options>("new_knob = 1\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown field"), "got: {err}");
        assert!(!err.contains("was removed"), "got: {err}");
    }

    fn the_registry_answers_dotted_and_bare_keys() {
        let expected = "`old_knob` was removed: the planner sizes it now; delete the key";
        assert_eq!(REMOVED, &[("old_knob", "the planner sizes it now")]);
        assert_eq!(refusal_in(REMOVED, "old_knob").as_deref(), Some(expected));
        assert_eq!(
            refusal_in(REMOVED, "engine.old_knob").as_deref(),
            Some(expected)
        );
        assert_eq!(refusal_in(REMOVED, "engine.kept"), None);
        assert_eq!(refusal_for("engine.kept"), None);
    }
}
