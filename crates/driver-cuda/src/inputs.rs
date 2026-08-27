//! The resident fire inputs: one allocation, carved once, overwritten every
//! fire and never moved.
//!
//! **POINTER-STABLE IS THE WHOLE POINT.** Step 5 records these addresses into
//! a graph that is never re-captured, so a buffer that were reallocated when a
//! fire got bigger would leave the graph reading the old one — which does not
//! fault, because the old allocation is still mapped. So every vector is
//! reserved at the budget's ceiling at load and a smaller fire writes its
//! prefix; the LENGTH rides on the handle and on the geometry, and the
//! address never changes. Eager mode does not need this yet, which is exactly
//! why it is written now: the eager shell is the golden the recorded one is
//! diffed against, and a difference in where the bytes live would be a
//! difference the diff cannot see.
//!
//! # What is here, and what the plan names
//!
//! ```text
//! tokens, positions        RuntimeInput::Tokens / ::Positions
//! per space: indptr,       RuntimeInput::Geometry { space, kind }
//!   indices, last_page_len,
//!   kv_len, write_page,
//!   write_offset
//! window boundaries        ambient — no op names it (design §5); one
//!                          rebased `[lanes + 1]` run per WINDOW, not one
//!                          per fire
//! per space: mask bits     RuntimeInput::Mask { space }
//! mask spans               ambient — `attention.masked`'s op-named bits have
//!                          no seat for their per-request byte offsets, so
//!                          the plan-prefill arm binds one onto the schedule
//! row_valid                the padding mask the kv writers read past the IR
//! slot_ids                 which recurrent bank each lane owns
//! plan workspace           the prepare phase's staging, granted per plan kind
//! ```
//!
//! The unseated ones are not oversights: the qo boundaries were deliberately
//! unnamed, and `row_valid`/`slot_ids`/the mask spans/the workspace are
//! driver facts the entries take beside the ops' operands (the `MENLO-SEAM`
//! markers `run.rs` catalogues).
//!
//! # The mask slab is reserved against the CONTEXT, not measured
//!
//! A lane's mask expands to `rows x (held + rows)` bits ([`crate::mask`]), so
//! the fire-wide worst case is every row of the ceiling against a full
//! context — `max_tokens * pages_per_slot * page_size / 8`, plus one byte per
//! lane because each lane's region starts on a byte boundary. Reserved like
//! everything else here, for the same reason: the address is recorded into a
//! graph that is never re-captured. A fire past it is `Fault::Ceiling`
//! naming the mask bits, never a reallocation.
//!
//! # Grants are disjoint carvings, one per plan kind
//!
//! `CachePlanning` wants a separate [`Workspace`] for the decode and prefill
//! builders because their staged int images coexist within a fire — the
//! prepare phase builds both before either is consumed. One pool, cut in two,
//! is what that sentence means in bytes.

use kernels_cuda::Tensor;
use kernels_cuda::attn::plan::Workspace;
use model_compiler::Budgets;
use model_ir::Dtype;

use crate::device::Buffer;
use crate::error::Result;
use crate::store::SpaceSeat;
use crate::store::kv::{Geometry, Paging, SpaceFacts};

/// The int side of one plan grant: where a built schedule's offset table is
/// staged.
///
/// Sized rather than measured. The builders refuse at build time when a
/// schedule does not fit its grant — the refusal names the bytes asked and
/// the bytes left — so an over-grant costs address space and an under-grant
/// costs a typed refusal, never a wrong schedule. These are the numbers a
/// deployment would tune; they are stated here because `kernels-cuda`
/// recommends none.
const GRANT_INT_BYTES: u64 = 8 << 20;

/// The float side's FLOOR: split-kv's partial outputs and their
/// log-sum-exps, for a schedule whose padding is the fire's own.
///
/// A graph-shaped prefill schedule wants more, and how much more is a
/// function of the model's attention rather than of a deployment's taste —
/// so [`graph_float_bytes`] computes it and this is only the floor beneath
/// that answer.
const GRANT_FLOAT_BYTES: u64 = 64 << 20;

/// The largest float workspace a GRAPH-SHAPED prefill schedule can ask for,
/// over every kv space this plan declares.
///
/// **A SHORT GRANT HERE DOES NOT FAIL — IT DECLINES**, which is why it is
/// computed rather than guessed. `plan_prefill` asked to be capturable pads
/// its work items to `2·SMs / kv_heads` regardless of how few rows the fire
/// carries (that padding is the whole point: the schedule's shape must be a
/// function of the KEY, not of this fire's kv lengths), and its partial
/// output buffer is `q_heads × padded × cta_tile_q × head_dim` floats. When
/// that does not fit, the builder quietly falls back to a schedule that does
/// and reports `graph_capturable = false` — and the shell, reading that
/// honestly, never captures a prefill again. Measured on the smoke's SKU: 8
/// query heads over 2 kv heads at width 256 on 142 SMs wants 71 MiB, and the
/// old flat 64 MiB grant is the reason a mixed fire declined every time.
///
/// `cta_tile_q` is bounded rather than predicted: 128 is the widest tile the
/// schedule picks, except at `head_dim >= 256` where `plan_prefill` refuses
/// it outright (no `KernelTraits` exist), so 64 is the bound there.
fn graph_float_bytes(spaces: &[Option<SpaceFacts>], sms: u32) -> u64 {
    spaces
        .iter()
        .flatten()
        .map(|facts| {
            let padded = u64::from(2 * sms.max(1)) / u64::from(facts.kv_heads.max(1)).max(1);
            let tile = if facts.head_dim >= 256 { 64 } else { 128 };
            let heads = u64::from(facts.q_heads);
            // `tmp_v` is the partials, `tmp_s` their log-sum-exps; both are
            // f32, and each starts on a 16-byte boundary.
            let v = heads * padded * tile * u64::from(facts.head_dim) * 4;
            let s = heads * padded * tile * 4;
            (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
        })
        .max()
        .unwrap_or(0)
}

/// The alignment every carved region starts on.
const ALIGN: u64 = 256;

/// One kv space's six vectors, as offsets into the store.
#[derive(Debug, Clone, Copy)]
struct SpaceAt {
    indptr: u64,
    indices: u64,
    last_page_len: u64,
    kv_len: u64,
    write_page: u64,
    write_offset: u64,
}

/// The handles one fire's inputs resolve to.
#[derive(Debug, Clone)]
pub struct Handles {
    /// `RuntimeInput::Tokens`.
    pub tokens: Tensor,
    /// `RuntimeInput::Positions`.
    pub positions: Tensor,
    /// Where the packed per-window boundary vectors landed —
    /// [`Windows::bind`](crate::window::Windows::bind) cuts them apart.
    pub windows: u64,
    /// One entry per kv geometry space, in space order.
    pub spaces: Vec<SpaceHandles>,
    /// `i32`, `[lanes]`: which recurrent bank each lane owns.
    pub slot_ids: Tensor,
    /// The padding mask the kv writers read.
    pub row_valid: Tensor,
    /// `RuntimeInput::Mask`: the packed `u8` (query, key) bits, fire-wide.
    /// `None` when no lane of this fire carried a mask — the shell then binds
    /// no mask seat at all, so a masked consumer answers `attn::masked`'s own
    /// refusal instead of reading a rectangle of zeros, which is every
    /// position masked OUT.
    pub mask: Option<Tensor>,
    /// `i32`, `[lanes + 1]`: each lane's ABSOLUTE byte offset into
    /// [`mask`](Handles::mask). Absolute, so a windowed consumer takes a
    /// slice of this table and the whole slab.
    pub mask_indptr: Option<Tensor>,
}

/// One kv space's device seats.
#[derive(Debug, Clone, Copy)]
pub struct SpaceHandles {
    /// `GeomKind::Indptr`.
    pub indptr: Tensor,
    /// `GeomKind::Indices`.
    pub indices: Tensor,
    /// `GeomKind::LastPageLen`.
    pub last_page_len: Tensor,
    /// `GeomKind::KvLen`.
    pub kv_len: Tensor,
    /// `GeomKind::WritePage`.
    pub write_page: Tensor,
    /// `GeomKind::WriteOffset`.
    pub write_offset: Tensor,
}

/// What one fire wants written, host side.
#[derive(Debug, Clone)]
pub struct Fire<'a> {
    /// Token ids, in fire row order.
    pub tokens: &'a [i32],
    /// Absolute positions, in fire row order.
    pub positions: &'a [i32],
    /// Every window's rebased boundaries, end to end
    /// ([`Windows::packed`](crate::window::Windows::packed)).
    pub windows: &'a [i32],
    /// Which recurrent bank each lane owns, in fire lane order.
    pub slot_ids: &'a [i32],
    /// One geometry per kv space, in space order.
    pub spaces: &'a [Geometry],
    /// This fire's expanded lane masks, or `None` when no lane carried one.
    pub mask: Option<&'a crate::mask::Staged>,
}

/// The resident inputs, carved once.
#[derive(Debug)]
pub struct Inputs {
    store: Buffer,
    tokens: u64,
    positions: u64,
    windows: u64,
    window_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    mask_bits: u64,
    mask_bytes: u64,
    mask_indptr: u64,
    spaces: Vec<SpaceAt>,
    decode: Workspace,
    prefill: Workspace,
}

impl Inputs {
    /// Reserve the vectors a deployment's ceilings admit.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::Fault::Device) for the allocation.
    pub fn reserve(
        budgets: &Budgets,
        paging: Paging,
        spaces: &[Option<SpaceFacts>],
        classes: usize,
        sms: u32,
    ) -> Result<Inputs> {
        let rows = u64::from(budgets.max_tokens);
        let lanes = u64::from(budgets.max_lanes);
        let pages = u64::from(budgets.max_lanes) * u64::from(paging.pages_per_slot);
        // A window is one contiguous run of the fire's class order, so a plan
        // of `k` classes has at most `k(k+1)/2` of them — plus one for the
        // zero window every empty region shares. Reserved rather than
        // measured, because these addresses are recorded into a graph that is
        // never re-captured (the note at the top of this file).
        let window_ints = (classes * (classes + 1) / 2 + 1) as u64 * (lanes + 1);

        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(ALIGN);
            here
        };
        let tokens = take(rows * 4);
        let positions = take(rows * 4);
        let windows = take(window_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * 4);
        // The masked axis's two vectors. `context` is what a slot can hold,
        // so `rows * context` bounds every (query, key) cell a fire can
        // present, and the per-lane byte alignment costs one byte a lane.
        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_bytes = (rows * context).div_ceil(8) + lanes;
        let mask_bits = take(mask_bytes);
        let mask_indptr = take((lanes + 1) * 4);
        // The prefill grant is the graph-shaped requirement or the flat
        // floor, whichever is larger; the decode grant keeps the floor,
        // because a decode schedule's padding carries no tile factor.
        let prefill_float_bytes = graph_float_bytes(spaces, sms).max(GRANT_FLOAT_BYTES);
        let spaces: Vec<SpaceAt> = (0..spaces.len())
            .map(|_| SpaceAt {
                indptr: take((lanes + 1) * 4),
                indices: take(pages * 4),
                last_page_len: take(lanes * 4),
                kv_len: take(lanes * 4),
                write_page: take(rows * 4),
                write_offset: take(rows * 4),
            })
            .collect();
        let decode_int = take(GRANT_INT_BYTES);
        let decode_float = take(GRANT_FLOAT_BYTES);
        let prefill_int = take(GRANT_INT_BYTES);
        let prefill_float = take(prefill_float_bytes);
        let total = at;

        let store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;
        let base = store.ptr();
        Ok(Inputs {
            tokens,
            positions,
            windows,
            window_ints,
            row_valid,
            slot_ids,
            mask_bits,
            mask_bytes,
            mask_indptr,
            spaces,
            decode: Workspace {
                int_ptr: base + decode_int,
                int_bytes: GRANT_INT_BYTES as usize,
                float_ptr: base + decode_float,
                float_bytes: GRANT_FLOAT_BYTES as usize,
            },
            prefill: Workspace {
                int_ptr: base + prefill_int,
                int_bytes: GRANT_INT_BYTES as usize,
                float_ptr: base + prefill_float,
                float_bytes: usize::try_from(prefill_float_bytes).unwrap_or(usize::MAX),
            },
            store,
        })
    }

    /// The decode builder's grant.
    #[must_use]
    pub fn decode_grant(&self) -> Workspace {
        self.decode
    }

    /// The prefill builder's grant.
    #[must_use]
    pub fn prefill_grant(&self) -> Workspace {
        self.prefill
    }

    /// Every byte the inputs hold.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }

    /// Write one fire's vectors on `stream` and hand back their handles.
    ///
    /// **THE COPIES ARE ON THE STREAM.** A synchronous copy would be ordered
    /// against every stream in the process, which is both slower and a lie
    /// about what this fire depends on; an asynchronous one on the fire's own
    /// stream is exactly the dependency the launches behind it have.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved
    /// ceilings, [`Fault::Device`](crate::Fault::Device) for a copy.
    pub fn write(&mut self, stream: *mut core::ffi::c_void, fire: &Fire<'_>) -> Result<Handles> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32;

        // The padding mask is all-valid in an eager fire: every row a fire
        // carries is a row it means. Under capture it is what tells the
        // writers which of a bucket's padded rows are real, which is why the
        // buffer exists now rather than at step 5.
        let valid = vec![1u8; rows as usize];

        self.store
            .stage(stream, self.tokens, bytes_of(fire.tokens))?;
        self.store
            .stage(stream, self.positions, bytes_of(fire.positions))?;
        if fire.windows.len() as u64 > self.window_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: self.window_ints,
            });
        }
        self.store
            .stage(stream, self.windows, bytes_of(fire.windows))?;
        self.store.stage(stream, self.row_valid, &valid)?;
        self.store
            .stage(stream, self.slot_ids, bytes_of(fire.slot_ids))?;

        // THE MASKED AXIS, STAGED OR NOT STAGED. A fire no lane masked writes
        // nothing here and binds no seat, which is what makes
        // `attn::masked`'s "no mask span table rides this plan" refusal
        // reachable — the alternative, a zeroed slab, is every position
        // masked out and a row of `-inf`.
        let mask = match fire.mask {
            None => None,
            Some(staged) => {
                if staged.bits.len() as u64 > self.mask_bytes {
                    return Err(crate::error::Fault::Ceiling {
                        what: "mask bits",
                        need: staged.bits.len() as u64,
                        have: self.mask_bytes,
                    });
                }
                self.store.stage(stream, self.mask_bits, &staged.bits)?;
                self.store
                    .stage(stream, self.mask_indptr, bytes_of(&staged.indptr))?;
                Some(u32::try_from(staged.bits.len()).unwrap_or(u32::MAX))
            }
        };

        let mut spaces = Vec::with_capacity(self.spaces.len());
        for (at, geometry) in self.spaces.iter().zip(fire.spaces) {
            self.store
                .stage(stream, at.indptr, bytes_of(&geometry.indptr))?;
            self.store
                .stage(stream, at.indices, bytes_of(&geometry.indices))?;
            self.store
                .stage(stream, at.last_page_len, bytes_of(&geometry.last_page_len))?;
            self.store
                .stage(stream, at.kv_len, bytes_of(&geometry.kv_len))?;
            self.store
                .stage(stream, at.write_page, bytes_of(&geometry.write_page))?;
            self.store
                .stage(stream, at.write_offset, bytes_of(&geometry.write_offset))?;
            let base = self.store.ptr();
            spaces.push(SpaceHandles {
                indptr: i32s(base + at.indptr, lanes + 1),
                indices: i32s(base + at.indices, geometry.indices.len() as u32),
                last_page_len: i32s(base + at.last_page_len, lanes),
                kv_len: i32s(base + at.kv_len, lanes),
                write_page: i32s(base + at.write_page, rows),
                write_offset: i32s(base + at.write_offset, rows),
            });
        }

        let base = self.store.ptr();
        Ok(Handles {
            tokens: i32s(base + self.tokens, rows),
            positions: i32s(base + self.positions, rows),
            windows: base + self.windows,
            spaces,
            slot_ids: i32s(base + self.slot_ids, lanes),
            row_valid: Tensor::new(base + self.row_valid, rows, 1, Dtype::U8),
            // The slab is handed over WHOLE: its entries are bits, not fire
            // rows, so `Run::cut` excludes it for the same reason it excludes
            // the page-id list, and the span table beside it is what carries
            // a windowed consumer to the right lane.
            mask: mask.map(|bytes| Tensor::new(base + self.mask_bits, bytes, 1, Dtype::U8)),
            mask_indptr: mask.map(|_| i32s(base + self.mask_indptr, lanes + 1)),
        })
    }

    /// The pool seats one fire lends its cache table.
    #[must_use]
    pub fn seats(&self, handles: &Handles, pages: u32, rows: u32, lanes: u32) -> crate::store::Seats {
        crate::store::Seats {
            lanes,
            rows,
            pages,
            spaces: handles
                .spaces
                .iter()
                .map(|space| SpaceSeat {
                    page_indptr: space.indptr,
                    page_indices: space.indices,
                    last_page_lens: space.last_page_len,
                    row_valid: handles.row_valid,
                })
                .collect(),
            slot_ids: handles.slot_ids,
        }
    }
}

/// One `i32` column, `n` rows tall.
fn i32s(ptr: u64, rows: u32) -> Tensor {
    Tensor::new(ptr, rows, 1, Dtype::I32)
}

/// A vector of `i32` as the bytes a copy takes.
///
/// Little-endian, stated rather than derived: every device this ships on is,
/// and the fire descriptor's own layout says the same thing for the same
/// reason. The one reinterpretation in the shell, and it is the operation
/// `bytemuck::cast_slice` exists to name — pulling a crate in for three lines
/// would add a dependency and say nothing this comment does not.
fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is `Copy`, has no padding and no niche, so all `4 * len`
    // of its bytes are initialized and readable as `u8`. The result borrows
    // the input and is read, never written, for the length of one enqueue.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}
