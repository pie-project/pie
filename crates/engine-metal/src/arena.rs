//! The activation arena: one allocation for the model's whole load, and the
//! per-fire table that turns the compiler's static offsets into handles.
//!
//! **THE OFFSETS ARE NOT THIS CRATE'S.** P7 carved them once
//! (`model_compiler::arena`), giving values whose lives do not overlap the
//! same bytes on purpose, and what is left for a shell is one addition:
//! `base + offset`. Only the LENGTH moves with the fire, and it moves by the
//! row expression the carve already wrote down — which is the whole reason a
//! composition never triggers a re-anything.
//!
//! # What "base + offset" means here, and why it costs a table
//!
//! **A METAL VIEW IS A HANDLE, NOT AN ADDRESS.** The CUDA sibling's carve is
//! pure arithmetic — it holds one `cudaMalloc`'d base and hands out
//! `base + offset`, a `u64` a kernel dereferences. Metal has no address to
//! hand out: a compute encoder binds a BUFFER and an OFFSET
//! (`setBuffer:offset:atIndex:`), so a `kernels_metal::Tensor` carries a
//! `u32` row into [`Handles`] and the resolution is a table lookup at encode
//! time. So every function here that builds a `Tensor` takes a
//! `handles: &Handles` and MINTS, where its CUDA twin only added — and every
//! one of them is fallible for the same reason, because minting is where the
//! rectangle is bounds-checked against the reservation it claims to live in
//! (`Handles::bind` calls `Buffer::span`).
//!
//! That is the whole structural divergence. The offsets, the row expressions,
//! the aliasing through `root`, the rule that a rectangle spans the whole
//! fire — none of it moves.
//!
//! # Why the table is rebuilt every fire, and why that is cheap
//!
//! [`SlotTable`] is `ValueId`-indexed and every row is a `Copy` handle, so
//! rebuilding it is one pass of arithmetic over the plan's values — 855 of
//! them for the smoke's SKU — plus one `Vec` push per bound row into
//! [`Handles`], which the fire drops in one `truncate` at
//! [`Handles::rewind`]. The alternative, mutating a resident table in place,
//! would save that pass and cost the property that makes it easy to reason
//! about: a `Run` borrows a table that describes exactly one fire. On this
//! plane it would cost more than that — a resident handle row whose offset is
//! rewritten between fires is exactly the stale-resolution bug the seal/rewind
//! watermark exists to make impossible.
//!
//! # A row count here is THE FIRE'S, and that is what makes windows work
//!
//! Every rectangle in this table spans the whole fire: a `Dim::Tokens` value
//! gets one column of `rows` rows, indexed by ABSOLUTE fire row. That is not
//! a simplification waiting to be narrowed — it is the substrate design §0's
//! window-split stands on. A merge lowers to "the arms write disjoint row
//! ranges of one buffer", which is a sentence about one column with two
//! writers, and the compiler's carve aliases the arms onto exactly that
//! column ([`arena`](model_compiler::arena)'s union-find through `root`).
//!
//! So the window is applied by the READER, not by the carve:
//! [`Run::tensor`] cuts each operand to the window of the node asking for it
//! ([`window`](crate::window)), which is what lets a decode node take rows
//! `[10,13)` of the same `q` a prefill node takes `[0,10)` of. A per-value
//! span would not answer that, because `Value::split` refines a cond and
//! hands back the SAME `ValueId`. On this plane the cut is
//! [`Handles::cut`] — a second row naming the same buffer further in —
//! rather than a pointer addition, which is the one place the reader pays for
//! the handle too.
//!
//! [`Run::tensor`]: crate::run::Run
//! [`SlotTable`]: crate::run::SlotTable

use engine::store::arena::rect;
use kernels_metal::Tensor;
use model_compiler::ArenaMap;
use model_ir::ValueId;

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::run::SlotTable;

/// The device bytes every arena rectangle is an offset into.
#[derive(Debug)]
pub struct Arena {
    store: Buffer,
}

impl Arena {
    /// Reserve what the carve says the largest admissible fire needs.
    ///
    /// Takes the [`Context`] because a Metal reservation is a call ON a
    /// device — `newBufferWithLength:options:` — where `cudaMalloc` reads the
    /// thread's current one out of ambient state. The parameter is the
    /// platform's difference showing through, not a preference.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] for a non-Apple build, [`Fault::Ceiling`] for an
    /// arena past `maxBufferLength`, [`Fault::Device`] when the device
    /// declined the length.
    pub fn reserve(device: &Context, map: &ArenaMap) -> Result<Arena> {
        Ok(Arena {
            store: Buffer::zeroed(device, map.bytes)?,
        })
    }

    /// The reservation itself, for whoever needs to mint a view into it that
    /// is not one of the carve's.
    ///
    /// There is no `base()` twin of the CUDA sibling's here, and that is the
    /// point: an address is the thing this platform does not have, so the
    /// buffer is what a caller is handed and [`Handles::bind`] is how a view
    /// of it is named.
    #[must_use]
    pub fn store(&self) -> &Buffer {
        &self.store
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    /// The table a fire of `tokens` rows over `lanes` requests resolves
    /// through.
    ///
    /// `handles` is the fire's minting table: one row per bound rectangle,
    /// dropped wholesale by [`Handles::rewind`] when the fire ends.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a rectangle that leaves the reservation — which
    /// is a carve that disagrees with its own `ArenaMap::bytes` — or for a
    /// handle table already full.
    pub fn slots(
        &self,
        handles: &Handles,
        map: &ArenaMap,
        tokens: u64,
        lanes: u64,
    ) -> Result<SlotTable> {
        carve(handles, &self.store, map, tokens, lanes)
    }

    /// Copy `into.len()` bytes back from `offset` bytes into this arena.
    ///
    /// The one read a fire ends with: the `"out"` seam's rows, whose slot the
    /// carve deliberately holds open past the last node so that nothing
    /// shares its bytes with the reader that has not run yet.
    ///
    /// **AN OFFSET, WHERE THE CUDA TWIN TAKES AN ADDRESS.** That shell had to
    /// subtract its own base back off a pointer a caller read out of a
    /// `Tensor`; here the `Tensor` never carried an address in the first
    /// place, so the subtraction — and the `Fault::Ceiling` it raised on a
    /// pointer from another allocation — has nothing to guard. The bound is
    /// checked once, in `Buffer::span`. A caller holding the seam's handle
    /// rather than its offset wants [`Arena::read_view`].
    ///
    /// On Apple silicon this is a `memcpy` out of a shared mapping and not a
    /// device-to-host transfer; what makes it correct is that the fire's
    /// command buffer was committed and waited on first, which is the call
    /// order in the shell.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span that leaves the arena,
    /// [`Fault::Deviceless`] for a non-Apple build.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.store.read(offset, into)
    }

    /// Copy `into.len()` bytes back from `skip` bytes into whatever `handle`
    /// names.
    ///
    /// The call the CUDA sibling spells `arena.read(tensor.ptr + skip, ..)`.
    /// A `Tensor` on this plane carries a handle, so the row is resolved
    /// first and the offset comes out of it — and the handle had better name
    /// a view of THIS arena, which is the caller's business: the table is one
    /// table across every reservation the load made, exactly so that a
    /// `Tensor` is one `u32` wide.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a handle no row answers, [`Fault::Ceiling`] for
    /// a span that leaves the arena, [`Fault::Deviceless`] for a non-Apple
    /// build.
    pub fn read_view(
        &self,
        handles: &Handles,
        handle: u32,
        skip: u64,
        into: &mut [u8],
    ) -> Result<()> {
        let at = {
            let row = handles.get(handle).ok_or_else(|| Fault::Unbound {
                what: format!("handle {handle}, which this load minted no row for"),
            })?;
            row.offset()
        };
        let at = at.checked_add(skip).ok_or(Fault::Ceiling {
            what: "bytes into the arena",
            need: u64::MAX,
            have: self.store.bytes(),
        })?;
        self.store.read(at, into)
    }
}

/// One fire's slot table: the carve's arithmetic, and one minted handle per
/// rectangle it names.
///
/// Separated from [`Arena`] so it can be driven over a buffer the caller
/// chose. The CUDA original's stronger claim — that this is exercisable with
/// no device in the room — does not survive the handle: a view must name a
/// buffer, and a buffer is a device call. What survives is the property
/// underneath it, that every value resolves to the rectangle its root names
/// at the row count this fire has, and that is
/// [`engine::store::arena::rect`] — neutral, and held to account host-side
/// in the tests beside it. What is left here is the one line a Metal view
/// is: an `(offset, bytes)` pair minted into a bounds-checked handle row,
/// where the CUDA plane adds `base + offset` and hands out a `u64`.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a rectangle that leaves `store` — a carve
/// disagreeing with its own `ArenaMap::bytes` — or for a full handle table.
pub fn carve(
    handles: &Handles,
    store: &Buffer,
    map: &ArenaMap,
    tokens: u64,
    lanes: u64,
) -> Result<SlotTable> {
    let mut rows: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        rows.push(match rect(map, value, tokens, lanes) {
            Some(rect) => Some(Tensor::new(
                handles.bind(store, rect.offset, rect.bytes)?,
                rect.rows,
                rect.width,
                rect.dtype,
            )),
            None => None,
        });
    }
    Ok(SlotTable(rows))
}
