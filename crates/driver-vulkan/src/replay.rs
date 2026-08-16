//! A decode step re-deriving, every token, a fire it already recorded.
//!
//! # What a step really does twice
//!
//! Two consecutive decode steps of one conversation differ in four things:
//! the token id, the position, the page table, and how many rows of the KV
//! cache are live. **None of the four reaches a dispatch.** They reach the
//! GPU as the CONTENTS of the fire tables [`crate::resources::Pool::stage`]
//! writes, and a table is written into the buffer it already had whenever the
//! length has not changed -- which for a decode is every step but the one
//! where a page is added.
//!
//! Everything else a fire is made of is a function of the LOWERING, and
//! `turns::Lowerings` already established that the lowering itself does not
//! change: `Row` carries no position, no history length, no page and no
//! token. So the plan -- 452 rectangles of symbol, descriptor ranges, write
//! masks, push constants, scalar blocks and workgroup counts -- is the same
//! plan every token, and so is the command buffer recorded from it.
//!
//! That was measured before it was relied on, and the measurement is
//! `tests/replay.rs`: with `PIE_VULKAN_REPLAY_VERIFY=1` every fire is planned
//! in full and DIFFED against the plan the step before it produced, field by
//! field. Over fifty consecutive decode steps of the real 4-bit qwen3-0.6B,
//! across a page boundary and a change of context length, the count of
//! rectangles differing in symbol, in any bound range, in any write mask, in
//! any push constant, in any scalar block byte and in any grid is **zero**.
//!
//! # What that buys
//!
//! Measured, release, `tests/hostprof.rs`, per decode step on a 4090, at 24
//! tokens of history:
//!
//! | phase | fresh | replayed |
//! |---|---|---|
//! | `fire/plan` | 0.654 ms | -- |
//! | `fire/pipelines` | 0.054 ms | -- |
//! | `fire/recorded` | 0.016 ms | -- |
//! | `fire/run_all/checks` | 0.006 ms | -- |
//! | `fire/run_all/descriptors` | 0.108 ms | -- |
//! | `fire/run_all/recording` | 0.342 ms | -- |
//!
//! -- against a 4.46 ms step of which 2.83 ms is the submit. The host outside
//! `run_all` was 1.17 ms and the recording inside it another 0.45; a replay
//! pays neither.
//!
//! # Why this is sound
//!
//! A recorded command buffer names three kinds of thing Vulkan will not
//! re-check at submit time: pipelines, descriptor sets, and -- through those
//! sets -- buffers at offsets. The argument that a replay is safe has to
//! cover all three, and it is made out of counters rather than out of rules a
//! caller has to keep:
//!
//! * **Pipelines.** [`crate::device::Pipelines::clear`] is the only thing
//!   that destroys one, and it tells the device to forget the recording.
//! * **Buffers.** [`crate::device::Device::free`] tells the device to forget
//!   the recording BEFORE it destroys anything, so a recording can never
//!   name a destroyed handle -- not even one Vulkan recycled into a new
//!   buffer of the same size.
//! * **Which buffers, and where.** [`Key`] carries the device's allocation
//!   and free counts. A plan is reused only when NOTHING has been allocated
//!   or freed on this device since the fire that produced it, so the set of
//!   live buffers is the set the descriptors were written against, and the
//!   arena's own handle is compared as well.
//! * **The recording itself.** [`crate::device::Device::run_all_reusable`]
//!   hands back a token, and any other fire recording over the top clears
//!   it. A replay checks the token under the same lock the recording is
//!   made under.
//!
//! The rest of the key is the plan's own inputs: which lowering, the fire's
//! geometry, the tier, the alignment the offsets were computed against, and
//! one number the caller states for everything its resolver answers FROM --
//! see [`Key::state`].
//!
//! # What is deliberately not attempted
//!
//! Nothing here writes into a command buffer after it is recorded, and
//! nothing rewrites a descriptor set between submits. Both are legal and both
//! would widen what can be reused -- a push constant that varied per token
//! would need the first, a KV extent that grew would need the second -- and
//! neither is needed, because the measurement says nothing varies. A
//! mechanism for a variation that does not happen is a mechanism nothing
//! tests.

use crate::device::Buffer;
use crate::dispatch::Geometry;
use crate::serve::Fired;
use kernels_vulkan::Capability;

/// Everything that has to hold for a recorded fire to be the right answer
/// again.
///
/// Compared for EQUALITY, whole. There is no partial match and no "close
/// enough": a field this does not carry is a field the cache is claiming
/// cannot matter, so the list is the claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    /// Which lowering, as the caller numbers them.
    ///
    /// A `&Lowered` cannot be its own key: `turns::Lowerings` holds them in a
    /// `Vec` that reallocates and evicts, so an address can name one lowering
    /// and later another. The caller hands out serial numbers instead, and
    /// [`crate::turns::Lowerings`] never reuses one.
    pub plan: u64,
    /// Everything the fire's [`Resolve`](crate::binding::Resolve) answers
    /// from, as the caller numbers it.
    ///
    /// The resolver is asked for weights, KV buffers, driver tables and five
    /// fire-wide NUMBERS -- the page size, two strides, the mask pitch and
    /// the bucketed history -- and the numbers are not buffers, so no
    /// allocation counter can see them change. The caller states them because
    /// the caller owns the pool.
    ///
    /// # The bucketed history is in here because a GRID depends on it
    ///
    /// The other four are read into push blocks, and a push block is recorded
    /// with the command buffer, so a stale one would be wrong everywhere at
    /// once and loudly. `FireNumber::KvHistoryBucket` is different: the arm
    /// turns it into `attn::decode_splits`, which is the flash decode's
    /// `vkCmdDispatch` z extent. A key that did not carry it would replay the
    /// grid from before the bucket doubled -- half the history unattended --
    /// and would do so at exactly ONE token, the one where the history
    /// crosses a power of two. Every short test would pass.
    ///
    /// The bucket is a power of two rather than the exact length for the same
    /// reason: an exact history changes every token, so every token would
    /// miss here and re-plan 452 rectangles. Rounded up, a decode re-plans
    /// about ten times over a 1024-token conversation -- which is what the
    /// `0.06` re-plans a step in `tests/hostprof.rs` is.
    pub state: u64,
    /// The arena buffer's handle.
    pub arena: u64,
    /// How many bytes of it the plan was bound against.
    pub arena_bytes: u64,
    /// [`Device::allocations`](crate::device::Device::allocations) when the
    /// plan was made.
    pub allocations: u32,
    /// [`Device::frees`](crate::device::Device::frees) when the plan was
    /// made.
    pub frees: u32,
    /// The fire-wide shape every launch rule read.
    pub geometry: Geometry,
    /// The tier every pipeline was built at.
    pub tier: Capability,
    /// The offset alignment every descriptor range was checked against.
    pub align: u64,
}

/// One recorded rectangle, owned.
///
/// The same six things [`crate::device::Recorded`] borrows, in a form that
/// outlives the fire that produced them. `buffers` is a range into
/// [`Held::operands`] rather than a `Vec` of its own: a decode's 452
/// rectangles bind about 1,800 descriptors between them, and 452 heap
/// allocations to hold two words each is the cost this module exists to
/// avoid.
#[derive(Debug)]
struct One {
    symbol: String,
    from: usize,
    to: usize,
    writes: Vec<bool>,
    push: Vec<u8>,
    groups: [u32; 3],
}

/// A fire, planned and recorded, kept for the next one.
#[derive(Debug)]
pub struct Held {
    /// Every rectangle's descriptor ranges, concatenated.
    operands: Vec<(Buffer, u64, u64)>,
    /// The rectangles, in the order they were recorded.
    dispatches: Vec<One>,
    /// Every scalar block of the fire, gathered as the block buffer holds
    /// them.
    scalars: Vec<u8>,
    /// The handle of the block buffer those descriptors name, or zero when
    /// the fire states no blocks.
    block: u64,
    /// The device recording, or zero when there is none to submit again.
    token: u64,
    /// What the fire that produced this reported.
    fired: Fired,
}

impl Held {
    /// How many rectangles.
    #[must_use]
    pub fn dispatches(&self) -> usize {
        self.dispatches.len()
    }
}

/// A fire being gathered into a [`Held`], one rectangle at a time.
///
/// Separate from `Held` so that the recording path can push as it goes
/// without deciding, per rectangle, whether anyone wants the answer.
#[derive(Debug, Default)]
pub struct Gathering {
    operands: Vec<(Buffer, u64, u64)>,
    dispatches: Vec<One>,
}

impl Gathering {
    /// Room for a fire of `n` rectangles.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            operands: Vec::with_capacity(n * 4),
            dispatches: Vec::with_capacity(n),
        }
    }

    /// Keep one recorded rectangle.
    pub fn push(
        &mut self,
        symbol: &str,
        buffers: &[crate::device::Bound<'_>],
        writes: &[bool],
        push: &[u8],
        groups: [u32; 3],
    ) {
        let from = self.operands.len();
        self.operands
            .extend(buffers.iter().map(crate::device::Bound::parts));
        self.dispatches.push(One {
            symbol: symbol.to_owned(),
            from,
            to: self.operands.len(),
            writes: writes.to_vec(),
            push: push.to_vec(),
            groups,
        });
    }

    /// Seal it with what the fire reported and what its blocks were.
    #[must_use]
    pub fn sealed(self, scalars: Vec<u8>, block: u64, token: u64, fired: Fired) -> Held {
        Held {
            operands: self.operands,
            dispatches: self.dispatches,
            scalars,
            block,
            token,
            fired,
        }
    }
}

/// Where two plans disagreed, and how often.
///
/// Counted per RECTANGLE and per field, not summed into one number, because
/// the question this file exists to answer is not "does anything vary" -- it
/// is which of the four places a varying thing could live it lives in. A
/// scalar in a block can be rewritten between submits; the same scalar in a
/// push constant is baked into the command buffer and forbids reuse outright.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diff {
    /// How many fires were compared against the one before them.
    pub compared: u64,
    /// Comparisons where the KEY differed, so the two fires were never
    /// claimed to be the same one.
    ///
    /// Not a disagreement. A step that crossed a KV page boundary
    /// reallocated a driver table, which the key sees as a changed
    /// allocation count, and the fire that follows binds the new buffer --
    /// exactly the case the key exists to catch. Counted and skipped, so that
    /// every other counter in here is about fires the key called
    /// INTERCHANGEABLE.
    pub rekeyed: u64,
    /// Fires whose rectangle COUNT differed, which stops the walk.
    pub shape: u64,
    /// Rectangles naming a different entrypoint.
    pub symbol: u64,
    /// Rectangles binding a different buffer, offset or length anywhere.
    pub buffers: u64,
    /// Individual descriptor ranges that differed.
    pub bindings: u64,
    /// Rectangles whose write masks differed.
    pub writes: u64,
    /// Rectangles whose PUSH CONSTANTS differed.
    ///
    /// The field that decides everything. A push constant is recorded by
    /// `vkCmdPushConstants` INTO the command buffer, so one that varies per
    /// token cannot be reused without re-recording.
    pub push: u64,
    /// Rectangles whose scalar BLOCK bytes differed.
    ///
    /// Harmless in principle -- the block is a device buffer the host
    /// rewrites before each submit -- and counted separately for exactly that
    /// reason.
    pub block: u64,
    /// Rectangles whose workgroup counts differed.
    pub groups: u64,
    /// The first few rectangles that differed, as `symbol: field`.
    pub witnesses: Vec<String>,
}

impl Diff {
    /// How many witnesses to keep. Enough to see whether a difference is one
    /// rectangle or one per layer.
    const WITNESSES: usize = 16;

    /// Did anything at all differ?
    #[must_use]
    pub fn quiet(&self) -> bool {
        self.shape == 0
            && self.symbol == 0
            && self.buffers == 0
            && self.writes == 0
            && self.push == 0
            && self.block == 0
            && self.groups == 0
    }

    fn note(&mut self, symbol: &str, field: &str) {
        if self.witnesses.len() < Self::WITNESSES {
            self.witnesses.push(format!("{symbol}: {field}"));
        }
    }
}

/// The one fire a driver keeps between steps, and what it did with it.
///
/// One, not a map. A server decodes at one shape for as long as a
/// conversation runs, so a second slot would hold a fire that is not the next
/// one; and a prefill in between invalidates by allocating, which no
/// capacity would help with. The same argument `turns::Arenas` makes for
/// holding one buffer.
#[derive(Debug, Default)]
pub struct Plans {
    held: Option<(Key, Held)>,
    replays: u64,
    records: u64,
    planned: u64,
    diff: Diff,
    /// Whether reuse is off for this cache.
    off: bool,
    /// Whether every fire is planned in full and diffed against the last.
    verify: bool,
}

impl Plans {
    /// Nothing held, with both switches read from the environment.
    ///
    /// `PIE_VULKAN_NO_REPLAY` turns reuse off and `PIE_VULKAN_REPLAY_VERIFY`
    /// turns the diff on. Read per CACHE rather than once per process, so
    /// `tests/replay.rs` can run a reusing shell and a non-reusing one in one
    /// binary and compare their logits -- an A/B against a separately built
    /// baseline measures the compiler as well as the change.
    #[must_use]
    pub fn new() -> Self {
        Self {
            off: std::env::var_os("PIE_VULKAN_NO_REPLAY").is_some(),
            verify: std::env::var_os("PIE_VULKAN_REPLAY_VERIFY").is_some(),
            ..Self::default()
        }
    }

    /// Never reuse a fire. What `PIE_VULKAN_NO_REPLAY` does, as a call.
    pub fn disable(&mut self) {
        self.off = true;
        self.held = None;
    }

    /// Plan every fire in full and diff it against the one before.
    ///
    /// Nothing is reused while this is on -- the point is to measure what the
    /// reuse would have ASSUMED, and a run that reused would have nothing
    /// fresh to compare against. So this is the slow path by construction.
    pub fn verify(&mut self, on: bool) {
        self.verify = on;
    }

    /// Whether reuse is off for this cache.
    #[must_use]
    pub fn off(&self) -> bool {
        self.off
    }

    /// Whether every fire is planned in full and diffed.
    #[must_use]
    pub fn verifying(&self) -> bool {
        self.verify
    }

    /// How many fires were submitted from a recording they did not make.
    #[must_use]
    pub fn replays(&self) -> u64 {
        self.replays
    }

    /// How many fires re-recorded a plan they did not have to re-plan.
    ///
    /// The middle outcome: the plan was still good and the recording was not,
    /// which is what a step after a staged read or a table reallocation gets.
    #[must_use]
    pub fn records(&self) -> u64 {
        self.records
    }

    /// How many fires planned every rectangle from the lowering.
    #[must_use]
    pub fn planned(&self) -> u64 {
        self.planned
    }

    /// What [`Self::verifying`] found.
    #[must_use]
    pub fn diff(&self) -> &Diff {
        &self.diff
    }

    /// The held fire, if it is the one `key` describes, taken out.
    ///
    /// Taken rather than lent for the reason
    /// [`Pipelines::block`](crate::device::Pipelines::block) is: the caller
    /// holds it while it asks the same cache other questions.
    pub fn take(&mut self, key: &Key) -> Option<Held> {
        match &self.held {
            Some((held, _)) if held == key => self.held.take().map(|(_, h)| h),
            _ => None,
        }
    }

    /// Hold a fire against a key.
    pub fn keep(&mut self, key: Key, held: Held) {
        self.held = Some((key, held));
    }

    /// Count a fire that planned everything.
    pub fn planned_one(&mut self) {
        self.planned += 1;
    }

    /// Count a fire that recorded a plan it did not re-derive.
    pub fn recorded_one(&mut self) {
        self.records += 1;
    }

    /// Count a fire that submitted a recording it did not make.
    pub fn replayed_one(&mut self) {
        self.replays += 1;
    }

    /// Compare a freshly planned fire against the one held, field by field.
    ///
    /// Only under [`Self::verifying`]. The claim being checked is precisely
    /// the one the cache makes and no more: when the key says two fires are
    /// the same fire, everything the command buffer bakes -- the entrypoint,
    /// the descriptor ranges, the write masks, the push constants and the
    /// workgroup counts -- is identical. A comparison across a key change is
    /// counted as [`Diff::rekeyed`] and no further, because there the cache
    /// claims nothing.
    pub fn compare(&mut self, key: &Key, fresh: &Held) {
        let Some((was, old)) = &self.held else {
            return;
        };
        self.diff.compared += 1;
        if was != key {
            self.diff.rekeyed += 1;
            return;
        }
        if old.dispatches.len() != fresh.dispatches.len() {
            self.diff.shape += 1;
            self.diff.note("<fire>", "a different number of rectangles");
            return;
        }
        if old.scalars.len() != fresh.scalars.len() {
            self.diff.note("<fire>", "a different scalar run length");
        }
        for (a, b) in old.dispatches.iter().zip(&fresh.dispatches) {
            if a.symbol != b.symbol {
                self.diff.symbol += 1;
                self.diff.note(&b.symbol, "symbol");
            }
            let (one, two) = (&old.operands[a.from..a.to], &fresh.operands[b.from..b.to]);
            if one.len() != two.len() || one.iter().zip(two).any(|(x, y)| !same(x, y)) {
                self.diff.buffers += 1;
                self.diff.bindings += one.iter().zip(two).filter(|(x, y)| !same(x, y)).count()
                    as u64
                    + one.len().abs_diff(two.len()) as u64;
                self.diff.note(&b.symbol, "a bound range");
            }
            if a.writes != b.writes {
                self.diff.writes += 1;
                self.diff.note(&b.symbol, "a write mask");
            }
            if a.push != b.push {
                self.diff.push += 1;
                self.diff.note(&b.symbol, "a push constant");
            }
            if a.groups != b.groups {
                self.diff.groups += 1;
                self.diff.note(&b.symbol, "a grid");
            }
        }
        // The blocks, by the bytes each rectangle's span holds rather than by
        // the run as a whole: a run that differs somewhere says nothing about
        // WHICH rectangle's scalars moved, and that is the answer this
        // counter exists to give.
        if old.scalars != fresh.scalars {
            self.diff.block += 1;
            self.diff.note("<fire>", "the gathered scalar blocks");
        }
    }
}

/// Do two descriptor ranges name the same memory?
fn same(a: &(Buffer, u64, u64), b: &(Buffer, u64, u64)) -> bool {
    a.0.identity() == b.0.identity() && a.1 == b.1 && a.2 == b.2
}

/// One rectangle of a held fire, as the recorder needs it.
///
/// The borrowed twin of [`One`], and the reason the recorder does not simply
/// get a `Vec<Recorded>`: a [`Recorded`](crate::device::Recorded) carries a
/// `&Pipeline`, and the pipeline cache is the one thing this module has no
/// business holding a reference into.
pub struct Rect<'a> {
    /// The entrypoint.
    pub symbol: &'a str,
    /// Where this rectangle's descriptor ranges start in [`Held::bounds`].
    pub from: usize,
    /// Where they end.
    pub to: usize,
    /// Which of them the shader may write through.
    pub writes: &'a [bool],
    /// The push block, empty when the module has none.
    pub push: &'a [u8],
    /// Workgroups in each dimension.
    pub groups: [u32; 3],
}

impl Held {
    /// What this fire reported when it ran.
    #[must_use]
    pub fn fired(&self) -> Fired {
        self.fired
    }

    /// The scalar bytes the block buffer must hold.
    #[must_use]
    pub fn scalars(&self) -> &[u8] {
        &self.scalars
    }

    /// The handle of the block buffer the recorded descriptors name.
    #[must_use]
    pub fn block(&self) -> u64 {
        self.block
    }

    /// The device recording, or zero.
    #[must_use]
    pub fn token(&self) -> u64 {
        self.token
    }

    /// Say the recording is gone and the plan must be recorded again.
    pub fn forget(&mut self) {
        self.token = 0;
    }

    /// Take the new recording's token.
    pub fn recorded(&mut self, token: u64) {
        self.token = token;
    }

    /// Every rectangle, in the order it was recorded.
    pub fn rectangles(&self) -> impl Iterator<Item = Rect<'_>> {
        self.dispatches.iter().map(|d| Rect {
            symbol: &d.symbol,
            from: d.from,
            to: d.to,
            writes: &d.writes,
            push: &d.push,
            groups: d.groups,
        })
    }

    /// Every descriptor range of the fire, rebuilt against the buffers it was
    /// planned over.
    ///
    /// Checked again rather than trusted, at `align`: the ranges were legal
    /// when they were computed and the buffers are the same buffers, so this
    /// cannot fail -- and a version that assumed so would be assuming exactly
    /// the thing that makes a stale plan dangerous.
    ///
    /// # Errors
    ///
    /// [`crate::device::Failed`] if a range no longer fits its buffer.
    pub fn bounds(
        &self,
        align: u64,
    ) -> Result<Vec<crate::device::Bound<'_>>, crate::device::Failed> {
        self.operands
            .iter()
            .map(|(b, at, len)| crate::device::Bound::within(b, *at, *len, align))
            .collect()
    }
}
