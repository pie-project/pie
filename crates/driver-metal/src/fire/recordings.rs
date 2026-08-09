//! Recordings, kept by what they are valid for.
//!
//! The recorded unit itself is [`crate::device::recording`] — an ICB and
//! the count of commands in it, which is device vocabulary. What lives here
//! is the CACHE: the map from a fire's shape key to the recording that serves
//! it, which is fire state and has the same field structure as
//! `driver-cuda`'s `SupergraphCache`. `.wiki/driver/real-metal-north-star.md`
//! §5 names them one concept under two names and asks for this one.

use crate::bind::encode::{Params, Pipelines, commands};
use crate::device::context::Context;
use crate::device::recording::{Recording, record};
use crate::device::regions::Regions;
use crate::error::{Error, Result};
use crate::lowering::dispatch::Dispatch;

/// Recordings, kept by what they are valid for.
///
/// Bounded by the number of distinct `(plan, row shape, address set)` a
/// deployment fires. With [`Scratch`](super::Scratch) pooling the three regions that
/// vary, that is about two per shape -- `ALLOCATOR_COUNT = 2` means two fires
/// are in flight at once and they hold different arenas.
///
/// **Nothing is ever re-recorded in place.** A fire in flight is executing
/// out of its ICB, and rewriting the commands under it is a use-after-free
/// that a green run does not show. A new fingerprint gets a new buffer.
#[derive(Default)]
pub struct Recordings {
    by_fingerprint: std::collections::HashMap<u64, Recording>,
    /// How many recordings have been made, for the test that asks whether
    /// the cache is a cache.
    recorded: usize,
    /// The fingerprints that cannot be recorded, and why.
    refused: std::collections::HashMap<u64, String>,
}

impl Recordings {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The recording for this fire, made if there is not one.
    ///
    /// # Errors
    ///
    /// As [`record`].
    pub fn get_or_record(
        &mut self,
        context: &Context,
        pipelines: &Pipelines,
        params: &Params,
        regions: &Regions,
        dispatches: &[Dispatch<'_>],
    ) -> Result<&Recording> {
        let key = fingerprint(dispatches, params);
        // REFUSED ONCE, not once a step. A fire that cannot be recorded --
        // an operand in no registered allocation, or a weight more than
        // 4 GiB into its buffer -- is refused for a reason that does not
        // change while the addresses do not, and `commands` allocates one
        // vector per dispatch before it can be asked. Remembering the refusal
        // is what keeps a large checkpoint from paying 425 allocations every
        // step to be told the same thing.
        if let Some(why) = self.refused.get(&key) {
            return Err(Error::Unrecordable {
                what: "fire",
                message: why.clone(),
            });
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = self.by_fingerprint.entry(key) {
            // Lowered on a MISS only. `commands` allocates one vector per
            // dispatch, and a hit is the path that matters -- it exists to
            // skip the encode, so putting 425 allocations in front of it
            // would spend part of what recording buys.
            let commands = commands(pipelines, params, dispatches)?;
            match record(context, regions, &commands) {
                Ok(recording) => {
                    slot.insert(recording);
                    self.recorded += 1;
                }
                Err(err @ Error::Unrecordable { .. }) => {
                    self.refused.insert(key, err.to_string());
                    return Err(err);
                }
                Err(other) => return Err(other),
            }
        }
        Ok(&self.by_fingerprint[&key])
    }

    /// How many recordings have been made.
    #[must_use]
    pub fn recorded(&self) -> usize {
        self.recorded
    }

    /// Forget every recording.
    ///
    /// For a model reload, which moves every weight address and invalidates
    /// all of them at once. Cheaper to state than to detect: the fingerprint
    /// would catch it, and this makes the intent visible.
    pub fn clear(&mut self) {
        self.by_fingerprint.clear();
        // The refusals too: a refusal is a statement about where the operands
        // are, and a reload is exactly the event that moves them.
        self.refused.clear();
    }
}

impl std::fmt::Debug for Recordings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recordings")
            .field("live", &self.by_fingerprint.len())
            .field("recorded", &self.recorded)
            .finish()
    }
}

/// What a recording is only valid for.
///
/// A recording bakes an operand's **buffer and offset**, its grid and its
/// pipeline. Replaying one against a fire that differs in any of those runs
/// the wrong program and says nothing — the failure class this crate spends
/// most of its tests on. So validity is *checked*, not assumed: this is a
/// digest of everything a command carries, and a fire whose digest differs
/// gets its own recording.
///
/// Cheap enough to be worth it: hashing 424 dispatches walks the same list
/// `encode` walks, without 5 000 Objective-C messages at the end of it — and
/// without the `Command` vectors, which is why it keys off the dispatches a
/// caller already holds rather than off what a recording is finally made of.
#[must_use]
pub fn fingerprint(dispatches: &[Dispatch<'_>], params: &Params) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    dispatches.len().hash(&mut hasher);
    for (index, d) in dispatches.iter().enumerate() {
        d.symbol.hash(&mut hasher);
        d.grid.hash(&mut hasher);
        d.threadgroup.hash(&mut hasher);
        for arg in d.args.iter() {
            arg.slice.address.hash(&mut hasher);
        }
        // The scalars' ADDRESS, not their values: a recording binds where the
        // run is, and `Params` rewrites the bytes in place every fire. That
        // is the whole reason a recording can be replayed at all -- the
        // CONTENTS of a bound buffer are free to change.
        params.address_of(index).unwrap_or(0).hash(&mut hasher);
        for p in &d.param_slots {
            (p.slot, p.at).hash(&mut hasher);
        }
    }
    hasher.finish()
}
