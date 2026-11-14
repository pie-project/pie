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
    DeviceMemoryShare = "device_memory_share":
        "the load measures the device and takes all of it; to share the device with another process, state `[engine.memory] budget_device`";
    DeviceKvBudget = "device_kv_budget":
        "the kv pages and state slots take what the device holds past the weights and the fires; to cap what the load takes, state `[engine.memory] budget_device`";
    HostKvBudget = "host_kv_budget":
        "the host tier's cap is `[engine.memory] budget_host` now: move the value there";
    DiskKvBudget = "disk_kv_budget":
        "the disk tier's cap is `[engine.memory] budget_disk` now: move the value there";
    DiskKvDir = "disk_kv_dir":
        "the disk tier's directory is `[engine.memory] disk_path` now: move the value there";
    DiskKvPersistent = "disk_kv_persistent":
        "keeping the disk tier is `[engine.memory] disk_persistent` now: move the value there";
    MemoryTolerance = "memory_tolerance":
        "state `[engine.memory] governor_backoff` (the share of memory kept each congested stretch: dedicated 1.0, aggressive 0.9, balanced 0.8, polite 0.7) and `governor_release_idle = true` to give back memory demand stopped asking for";
    WeightStreaming = "weight_streaming":
        "the weight cache has no mode: every streamable weight stays resident while the device has room, and live demand takes chunks from it before any kv or state gives way";
    MaxPatches = "max_patches":
        "the patch rungs are derived from the device like the token rungs; a request past the largest is refused";
    MaxImages = "max_images":
        "the patch rungs are derived from the device like the token rungs; a request past the largest is refused";
    MaxVoxels = "max_voxels":
        "the voxel rungs are derived from the device like the token rungs; a request past the largest is refused";
    MaxClips = "max_clips":
        "the voxel rungs are derived from the device like the token rungs; a request past the largest is refused";
    MaxForwardTokens = "max_forward_tokens":
        "a fire's rows are the largest rung the device holds, derived at load; a request past it is refused";
    MaxForwardRequests = "max_forward_requests":
        "a fire's lanes are the largest rung the device holds, derived at load; a request past it is refused";
    MaxModelLen = "max_model_len":
        "the context cap is `[engine] max_context` now: move the value there";
    KvPageSize = "kv_page_size":
        "a kv page is the compiler's chunk plan";
    DeviceMemory = "device_memory":
        "the load measures the device; where the adapter says nothing, state `[engine.memory] budget_device`";
    AdapterSeats = "seats":
        "an adapter's room is not configured: its planes lie in the pool's `Adapter` slots, which the runtime takes when an instance binds a program that carries one and gives back when it closes (the CUDA engine; the others keep the banks the model text sizes), so the pool's room is the only bound; `[[model.adapters.registered]]` ids number the roster alone";
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
