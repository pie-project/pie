//! The resident fire inputs: one allocation, carved once, overwritten every
//! fire and never moved.
//!
//! **RESERVED AT THE CEILING, AND ON THIS PLANE THE REASON IS NOT CAPTURE.**
//! The CUDA sibling reserves because its step 5 records device ADDRESSES into
//! a graph that is never re-captured, so a buffer that were reallocated when a
//! fire got bigger would leave the graph reading the old one — which does not
//! even fault, because the old allocation is still mapped. This shell records
//! nothing: design §6 gives it no `record.rs`, dispatch is encode-only, and
//! eager IS the mode. That argument is therefore not copied here; two honest
//! ones survive it, and they are enough.
//!
//! The first is arithmetic: ONE allocation at load, rather than one per fire.
//! `newBufferWithLength:options:` is a kernel-side reservation and a
//! zero-fill, and paying for one every fire to hold vectors whose ceiling a
//! deployment already stated is work with nothing to show for it.
//!
//! The second is ordering: **a fire's staged vectors must not move while a
//! command buffer that binds them is in flight.** The failure that would
//! follow is shaped differently here than on the CUDA plane, and worse for
//! it. A handle row RETAINS its `MTLBuffer` ([`crate::device::Handles`]), so a
//! reallocation would not dangle — the old bytes stay mapped and stay alive —
//! it would simply mean the encoded work reads the previous reservation while
//! the host writes the next one, with nothing anywhere to fault on. So every
//! vector is reserved at the budget's ceiling at load and a smaller fire
//! writes its prefix; the LENGTH rides on the `Tensor` the handle is wrapped
//! in and on the geometry, and the carve never moves.
//!
//! # There is no stream here, and nothing to be asynchronous about
//!
//! The CUDA `Inputs::write` takes a `stream` and pushes every vector across
//! the bus with an async H2D memcpy, because on that plane host memory and
//! device memory are two places and a copy between them is real work that can
//! overlap. Apple silicon has one physical pool behind the CPU and the GPU,
//! and this shell's reservations are `StorageModeShared`
//! ([`crate::device::Buffer`]), so a write IS a `memcpy` into the very bytes a
//! shader will read. There is no transfer to put on a queue, so the parameter
//! is DROPPED rather than answered with a null: a stream argument nobody could
//! pass anything meaningful to is a lie about what this plane does.
//!
//! **What survives the drop is ORDER, not asynchrony.** A host write must
//! happen before the command buffer that reads those bytes is committed. That
//! is a fact about the call sequence in `serve` — write the fire's vectors,
//! then encode, then commit — and not a flag on a buffer, an event on a queue,
//! or anything this module can enforce. It is stated here because it is the
//! one piece of the CUDA stream discipline that did not evaporate.
//!
//! # A carved view is a handle, not an address
//!
//! A `kernels_cuda::Tensor` carries a device address, so its shell answers
//! "where did the tokens land" with `base + offset` and no state. A
//! `kernels_metal::Tensor` carries a `u32` row of [`crate::device::Handles`],
//! because an encoder binds a BUFFER and an OFFSET rather than a pointer. So
//! every vector this module stages is followed by a MINT: one row naming the
//! store and the offset the vector was written at. The rows are per-fire and
//! `Handles::rewind` drops them at the end of one, which is why minting here
//! costs a `Vec` push and no bookkeeping.
//!
//! Two tables in this crate are called `Handles`, and the collision is worth
//! naming once: [`Handles`] in this file is the fire's RESOLVED VIEWS, one
//! `Tensor` per input the plan or the driver reads, and
//! [`crate::device::Handles`] is the shell's handle table those `Tensor`s are
//! rows of. This file spells the second by path everywhere, and never imports
//! it.
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
//! adapter routes           RuntimeInput::AdapterRoutes
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
//! everything else here, for the reason the top of this file states: the carve
//! happens once, at load, and a fire past it is `Fault::Ceiling` naming the
//! mask bits, never a reallocation under an encoder that has already bound the
//! old one.
//!
//! # Grants are disjoint carvings, one per plan kind
//!
//! `CachePlanning` wants a separate [`Workspace`] for the decode and prefill
//! builders because their staged int images coexist within a fire — the
//! prepare phase builds both before either is consumed. One pool, cut in two,
//! is what that sentence means in bytes.

use kernels_metal::Tensor;
use model_compiler::Budget;
use model_ir::Dtype;

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};
use crate::store::SpaceSeat;
use crate::store::kv::{Geometry, Paging};

/// **THERE IS NO SCHEDULE GRANT ON THIS PLANE, AND THE ABSENCE IS THE POINT.**
/// The CUDA sibling carves one workspace per PLAN VALUE at this point in the
/// file — `graph_float_bytes` of split-kv partial outputs beside a flat
/// integer slab — because `kernels-cuda`'s `plan_prefill` builds a schedule
/// whose partials it owns, and because a captured schedule's footprint has to
/// be a function of the KEY (build log 13, where a 10 MiB shortfall silently
/// declined every capture and cost only speed).
///
/// `kernels-metal` builds no schedule at all. `attn::plan_decode` and
/// `attn::plan_prefill` are pure carriers — they check the fire tables agree
/// and hand them back — and the sdpa shaders split no kv, so there are no
/// partials to hold. A grant here would be a reservation nothing reads: at
/// qwen35-d0.8b's four plan values it measured 216 MiB of one, which was most
/// of this shell's whole `inputs` footprint. The arithmetic is not kept "for
/// the day"; it lives in the CUDA shell, where it has a consumer, and the day
/// a metal builder wants a workspace it arrives with one.

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
    /// The handle the packed per-window boundary vectors were minted at —
    /// [`Windows::bind`](crate::window::Windows::bind) cuts them apart, one
    /// row per window, through
    /// [`Handles::cut`](crate::device::Handles::cut). A `u32` where the CUDA
    /// shell carries a `u64` base address, for the reason the top of this
    /// file states: on this plane the cut IS a handle.
    pub windows: u32,
    /// One entry per kv geometry space, in space order.
    pub spaces: Vec<SpaceHandles>,
    /// `i32`, `[lanes]`: which recurrent bank each lane owns.
    pub slot_ids: Tensor,
    /// `i32`, one per token ROW: which recurrent bank the row's lane owns —
    /// the vector the ssm shaders actually index. See the carve.
    pub slot_of_row: Tensor,
    /// The padding mask the kv writers read.
    pub row_valid: Tensor,
    /// `RuntimeInput::AdapterRoutes`: `i32`, one adapter id per token row.
    /// `None` when no lane of this fire carried one — the shell then binds no
    /// seat, exactly as it does for the mask, and the correction's window is
    /// empty so nothing reads it.
    pub adapter_routes: Option<Tensor>,
    /// `i32`, one per token row: which lane owns it. Read directly by every
    /// sdpa entry, which is why this plane stages it and the CUDA one does
    /// not.
    pub request_of_token: Tensor,
    /// `u8`, `[rows * mask_stride]`: 1 keeps the (query, key) pair. Always
    /// bound — see [`mask_enabled`](Handles::mask_enabled).
    pub mask: Tensor,
    /// `u8`, one per token row: whether that row's mask plane is consulted.
    ///
    /// **ALWAYS BOUND, NEVER OPTIONAL.** Every shipped sdpa entry
    /// instantiates `FAST_FULL = false` and therefore reads
    /// `attention_mask_enabled[row]` on every launch, masked fire or not. An
    /// unbound seat here is a null dereference on the first decode; a zeroed
    /// one is the unmasked reading, which is what a fire no lane masked
    /// wants.
    pub mask_enabled: Tensor,
    /// Key positions from one row's plane to the next, as the shaders read
    /// it.
    pub mask_stride: u32,
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
    /// The same fact per token ROW, in fire row order — what the ssm
    /// shaders index.
    pub slot_of_row: &'a [i32],
    /// Which adapter each token ROW routes to, in fire row order, or `None`
    /// when no lane carried one. Per ROW and not per lane, because that is
    /// what the correction kernel indexes with: `routes[row]` beside
    /// `x[row]`, the same shape `tokens` and `positions` have.
    pub adapter_routes: Option<&'a [i32]>,
    /// Which lane owns each token row, in fire row order.
    pub request_of_token: &'a [i32],
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
    slot_of_row: u64,
    adapter_routes: u64,
    request_of_token: u64,
    mask_planes: u64,
    mask_plane_bytes: u64,
    mask_enabled: u64,
    /// Key positions from one masked row's plane to the next, at the ceiling
    /// — what a lane can hold, so every fire's own stride fits inside it.
    mask_stride: u32,
    spaces: Vec<SpaceAt>,
}

impl Inputs {
    /// Reserve the vectors a deployment's ceilings admit.
    ///
    /// `device` is taken where the CUDA sibling takes nothing: a
    /// `cudaMalloc` addresses the thread's bound context implicitly, and an
    /// `MTLBuffer` is made BY a device object, so the reservation has to be
    /// handed the one this shell bound.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::Fault::Device) when the device declined the
    /// reservation, [`Fault::Ceiling`](crate::Fault::Ceiling) when the carve
    /// is longer than one `MTLBuffer` may be, and
    /// [`Fault::Deviceless`](crate::Fault::Deviceless) off Apple.
    pub fn reserve(
        device: &Context,
        budget: &Budget,
        paging: Paging,
        spaces: usize,
        classes: usize,
    ) -> Result<Inputs> {
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(budget.max_lanes);
        let pages = u64::from(budget.max_lanes) * u64::from(paging.pages_per_slot);
        // A window is one contiguous run of the fire's class order, so a plan
        // of `k` classes has at most `k(k+1)/2` of them — plus one for the
        // zero window every empty region shares. Reserved rather than
        // measured, because this carve is made once at load and a fire that
        // grew it would move bytes an encoder has already bound (the note at
        // the top of this file).
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
        // **THE RECURRENT SLOT MAP IS PER ROW ON THIS PLANE, NOT PER LANE.**
        // The CUDA sibling's `RecurrentPool::slot_ids` is one entry per lane
        // and its scans read `slot_ids[lane]`; every metal ssm shader reads
        // `slots[r]` where `r` is a TOKEN ROW — the decode arms index the
        // grid row directly (`ssm_causal_conv1d.metal`'s `slots[r]`,
        // `ssm_gated_delta.metal`'s `slots[n]`) and the chunked arms index
        // the request's FIRST row (`slots[indptr[r]]`). For a fire of one
        // lane the two readings coincide, which is exactly why the
        // difference survived every solo gate and only showed up when a
        // decode lane stood beside a prefill one.
        let slot_of_row = take(rows * 4);
        // The adapter axis's one vector, reserved at the row ceiling like
        // every other row-shaped table here — 32 KiB at `max_tokens = 8192`,
        // paid by every load whether or not the plan declares a correction,
        // because a conditional carve would make the STORE's layout depend on
        // the plan and this layout is fixed at load.
        let adapter_routes = take(rows * 4);
        // WHICH LANE EACH TOKEN ROW BELONGS TO, and it is reserved here
        // rather than derived at the shader because the metal sdpa entries
        // read it directly: `req_of_token[row]` is what indexes the page
        // table. The CUDA sibling has no such seat — its plan builders walk
        // the boundaries host-side and bake the answer into a schedule — so
        // this vector is one of the two places the two planes' fire inputs
        // genuinely differ.
        let request_of_token = take(rows * 4);
        // **THE MASKED AXIS, IN THIS PLANE'S OWN ABI.** The CUDA shell packs
        // one BIT per (query, key) pair with a per-lane indptr; the metal
        // shaders read one BYTE per pair out of a row-major plane with a
        // stated stride (`attention_mask[row * stride + kp]`), gated per row
        // by `attention_mask_enabled[row]`. So the carve is a rectangle:
        // `max_tokens` rows of `context` bytes, plus the enable column. It is
        // eight times the CUDA reservation for the same fire and it is the
        // ABI the shipped shaders read; `crate::mask` argues the expansion.
        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_stride = u32::try_from(context).unwrap_or(u32::MAX);
        let mask_plane_bytes = rows * context;
        let mask_planes = take(mask_plane_bytes);
        // Always bound, never optional: `attention_mask_enabled[row]` is read
        // on EVERY sdpa launch (the shipped entries instantiate
        // `FAST_FULL = false`), so an unbound seat is a null dereference on
        // the first decode rather than a mask nobody asked for. Zeroed at
        // reservation and rewritten per fire, which is what makes "no lane
        // masked" cost one `memset` of `rows` bytes.
        let mask_enabled = take(rows);
        let spaces: Vec<SpaceAt> = (0..spaces)
            .map(|_| SpaceAt {
                indptr: take((lanes + 1) * 4),
                indices: take(pages * 4),
                last_page_len: take(lanes * 4),
                kv_len: take(lanes * 4),
                write_page: take(rows * 4),
                write_offset: take(rows * 4),
            })
            .collect();
        let total = at;

        let store = Buffer::zeroed(device, total)?;
        Ok(Inputs {
            store,
            tokens,
            positions,
            windows,
            window_ints,
            row_valid,
            slot_ids,
            slot_of_row,
            adapter_routes,
            request_of_token,
            mask_planes,
            mask_plane_bytes,
            mask_enabled,
            mask_stride,
            spaces,
        })
    }

    /// Every byte the inputs hold.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    /// Write one fire's vectors into the store and hand back their handles.
    ///
    /// **THE WRITES ARE `memcpy`, AND THE ORDER IS THE CALLER'S.** The CUDA
    /// twin of this method takes a stream and stages every vector
    /// asynchronously on it, because there the copy is a transfer that can
    /// overlap. Here the store is `StorageModeShared` — one physical pool,
    /// mapped — so each write lands in the bytes a shader will read and
    /// there is nothing to overlap and nothing to order against. What the
    /// caller still owes is the sequence: every call to this method must
    /// precede the commit of the command buffer that reads what it wrote.
    ///
    /// Each staged vector is followed by a MINT: one row of `handles` naming
    /// the store and the offset, which is what a `kernels_metal::Tensor`
    /// carries in place of an address. The rows die with the fire.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved
    /// ceilings or a handle table that is full,
    /// [`Fault::Deviceless`](crate::Fault::Deviceless) off Apple.
    pub fn write(
        &mut self,
        handles: &crate::device::Handles,
        fire: &Fire<'_>,
    ) -> Result<Handles> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32;

        // The padding mask is all-valid in an eager fire: every row a fire
        // carries is a row it means. It exists as a vector rather than as an
        // absent argument because the kv writers read it unconditionally, and
        // because a plane that ever pads a bucket's rows would have exactly
        // this table to say which of them are real.
        let valid = vec![1u8; rows as usize];

        self.store.write(self.tokens, bytes_of(fire.tokens))?;
        self.store.write(self.positions, bytes_of(fire.positions))?;
        if fire.windows.len() as u64 > self.window_ints {
            return Err(Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: self.window_ints,
            });
        }
        self.store.write(self.windows, bytes_of(fire.windows))?;
        self.store.write(self.row_valid, &valid)?;
        self.store.write(self.slot_ids, bytes_of(fire.slot_ids))?;
        self.store
            .write(self.request_of_token, bytes_of(fire.request_of_token))?;
        self.store
            .write(self.slot_of_row, bytes_of(fire.slot_of_row))?;

        // THE ADAPTER AXIS, WRITTEN OR NOT WRITTEN — the mask's rule, for the
        // mask's reason. A fire no lane routed writes nothing here and binds
        // no seat, so a correction that somehow reached a launch would hit
        // `Run::tensor`'s named panic rather than read a slab of zeros, which
        // is every row routed to adapter 0 of a bank nobody registered.
        let adapter_routes = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                self.store.write(self.adapter_routes, bytes_of(routes))?;
                Some(routes.len() as u32)
            }
        };

        // **THE MASKED AXIS, IN THIS PLANE'S ROW-MAJOR ABI.** The metal sdpa
        // entries read `attention_mask[row * stride + kp]` gated by
        // `attention_mask_enabled[row]`, and both seats are read on EVERY
        // launch — the shipped instantiations are `FAST_FULL = false`. So the
        // choice the CUDA sibling has (bind no seat at all, and let a masked
        // consumer refuse) does not exist here: an unbound seat is a null
        // dereference on the first decode of every fire. What replaces it is
        // the enable column, written to zeros for a fire no lane masked,
        // which is exactly the unmasked reading.
        let stride = fire.mask.map_or(0, |staged| staged.stride);
        if u64::from(stride) > u64::from(self.mask_stride) {
            return Err(Fault::Ceiling {
                what: "key positions in one mask row",
                need: u64::from(stride),
                have: u64::from(self.mask_stride),
            });
        }
        match fire.mask {
            None => {
                // One `memset` of `rows` bytes, and nothing else: the plane
                // itself is never read when no row enables it.
                self.store
                    .zero_span(self.mask_enabled, u64::from(rows))?;
            }
            Some(staged) => {
                if staged.bytes.len() as u64 > self.mask_plane_bytes {
                    return Err(Fault::Ceiling {
                        what: "mask plane bytes",
                        need: staged.bytes.len() as u64,
                        have: self.mask_plane_bytes,
                    });
                }
                self.store.write(self.mask_planes, &staged.bytes)?;
                self.store.write(self.mask_enabled, &staged.enabled)?;
            }
        }

        let mut spaces = Vec::with_capacity(self.spaces.len());
        for (at, geometry) in self.spaces.iter().zip(fire.spaces) {
            self.store.write(at.indptr, bytes_of(&geometry.indptr))?;
            self.store.write(at.indices, bytes_of(&geometry.indices))?;
            self.store
                .write(at.last_page_len, bytes_of(&geometry.last_page_len))?;
            self.store.write(at.kv_len, bytes_of(&geometry.kv_len))?;
            self.store
                .write(at.write_page, bytes_of(&geometry.write_page))?;
            self.store
                .write(at.write_offset, bytes_of(&geometry.write_offset))?;
            spaces.push(SpaceHandles {
                indptr: i32s(handles, &self.store, at.indptr, lanes + 1)?,
                indices: i32s(
                    handles,
                    &self.store,
                    at.indices,
                    geometry.indices.len() as u32,
                )?,
                last_page_len: i32s(handles, &self.store, at.last_page_len, lanes)?,
                kv_len: i32s(handles, &self.store, at.kv_len, lanes)?,
                write_page: i32s(handles, &self.store, at.write_page, rows)?,
                write_offset: i32s(handles, &self.store, at.write_offset, rows)?,
            });
        }

        Ok(Handles {
            tokens: i32s(handles, &self.store, self.tokens, rows)?,
            positions: i32s(handles, &self.store, self.positions, rows)?,
            // The packed run is minted WHOLE, at the bytes this fire wrote:
            // `Windows::bind` cuts one row per window out of it, and a cut
            // past what was written is refused there rather than read here.
            windows: handles.bind(
                &self.store,
                self.windows,
                fire.windows.len() as u64 * 4,
            )?,
            spaces,
            slot_ids: i32s(handles, &self.store, self.slot_ids, lanes)?,
            slot_of_row: i32s(handles, &self.store, self.slot_of_row, rows)?,
            adapter_routes: match adapter_routes {
                None => None,
                Some(rows) => Some(i32s(handles, &self.store, self.adapter_routes, rows)?),
            },
            row_valid: Tensor::new(
                handles.bind(&self.store, self.row_valid, u64::from(rows))?,
                rows,
                1,
                Dtype::U8,
            ),
            request_of_token: i32s(handles, &self.store, self.request_of_token, rows)?,
            // The plane is minted at the fire's OWN rectangle — `rows` of
            // `stride` bytes — so a windowed launch's `Run::cut_rows` steps
            // by the stride the shader was told, and a cut past what this
            // fire wrote is refused by the handle table rather than read.
            mask: Tensor::new(
                handles.bind(
                    &self.store,
                    self.mask_planes,
                    u64::from(rows) * u64::from(stride),
                )?,
                rows,
                stride.max(1),
                Dtype::U8,
            ),
            mask_enabled: Tensor::new(
                handles.bind(&self.store, self.mask_enabled, u64::from(rows))?,
                rows,
                1,
                Dtype::U8,
            ),
            mask_stride: stride,
        })
    }

    /// The pool seats one fire lends its cache table.
    ///
    /// **NOTHING IS MINTED HERE, AND THE TABLE IS TAKEN ANYWAY.** Every seat
    /// below is a view [`write`](Inputs::write) already minted a row for, and
    /// a handle is minted once and shared rather than re-minted per reader —
    /// so this entry stays infallible, as its CUDA twin is. `_handles` is
    /// taken so the call reads like every other fire-time entry in this
    /// module and so the caller must have the table these `Tensor`s resolve
    /// through in hand at the moment it lends them.
    #[must_use]
    pub fn seats(
        &self,
        _handles: &crate::device::Handles,
        views: &Handles,
        pages: u32,
        rows: u32,
        lanes: u32,
    ) -> crate::store::Seats {
        crate::store::Seats {
            lanes,
            rows,
            pages,
            spaces: views
                .spaces
                .iter()
                .map(|space| SpaceSeat {
                    page_indptr: space.indptr,
                    page_indices: space.indices,
                    last_page_lens: space.last_page_len,
                    row_valid: views.row_valid,
                })
                .collect(),
            slot_ids: views.slot_ids,
            slot_of_row: views.slot_of_row,
        }
    }
}

/// One `i32` column, `rows` tall, as a freshly minted handle into `store`.
///
/// The CUDA shell spells this `Tensor::new(base + offset, rows, 1, I32)` and
/// needs no fallibility, because address arithmetic cannot fail. Here the
/// same sentence is a row in a table that is bounds-checked and finite, so it
/// answers a `Result`.
fn i32s(
    handles: &crate::device::Handles,
    store: &Buffer,
    at: u64,
    rows: u32,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::I32))
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
    // the input and is read, never written, for the length of one `memcpy`.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}
