//! Resident fire inputs: one allocation per in-flight step, carved once at
//! load and never moved. Writes are direct memcpys (`StorageModeShared`);
//! ordering is write, then encode, then commit. Each staged vector is
//! followed by a mint into [`crate::device::Handles`].

use kernels_metal::Tensor;
use model_compiler::Budget;
use model_ir::Dtype;

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};
use crate::store::SpaceSeat;
use crate::store::kv::{Geometry, Paging};

// No schedule-grant workspace on this plane: `kernels-metal`'s sdpa shaders
// split no kv, so there are no partials to hold.

/// The alignment every carved region starts on.
const ALIGN: u64 = 256;

/// The axes a multimodal position carries: time, row, column. Matches
/// `kernels_metal::elemwise::rope_mrope::AXES`.
const AXES: u64 = 3;

/// What a plan's patch axis asks of the store, read off the ladder at load;
/// `None` for a plan with no patch row, so a text-only load carves nothing.
#[derive(Debug, Clone, Copy)]
pub struct PatchSeat {
    /// The most patch rows one fire may carry — the ladder's `max_patches`.
    pub rows: u64,
    /// One patch row's bytes: the plan's declared `[Dim::Patches, C·T·P²]`
    /// width, times its element.
    pub row_bytes: u64,
    /// The most images one fire may carry — the ladder's `max_images`, which
    /// is what `[Dim::ImagesPlus(1)]` is sized at.
    pub images: u64,
    /// The element the plan computes patches in, which the marshal converts
    /// the submission's `f32` into.
    pub dtype: Dtype,
    /// `RuntimeInput::PatchEmbedRows`' declared tap count — 1 on the native
    /// grid, 2 for a separable table, 4 bilinear, 16 bicubic. `0` for a plan
    /// that declares no position-table read at all.
    pub embed_taps: u64,
    /// Whether the plan also declares `RuntimeInput::PatchEmbedWeights`. A
    /// native-grid read has ids and no weights, and then this region is not
    /// carved.
    pub embed_weights: bool,
}

/// The patch axis's six regions, as offsets into the store.
#[derive(Debug, Clone, Copy)]
struct PatchAt {
    payload: u64,
    segments: u64,
    routes: u64,
    positions: u64,
    embed_rows: u64,
    embed_weights: u64,
    seat: PatchSeat,
}

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
    /// The handle the packed per-window boundary vectors were minted at;
    /// [`Windows::bind`](crate::window::Windows::bind) cuts one row per
    /// window out of it.
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
    /// `None` when no lane of this fire carried one, and no seat is bound.
    pub adapter_routes: Option<Tensor>,
    /// `i32`, one per token row: which lane owns it. Read directly by every
    /// sdpa entry.
    pub request_of_token: Tensor,
    /// `u8`, `[rows * mask_stride]`: 1 keeps the (query, key) pair. Always
    /// bound — see [`mask_enabled`](Handles::mask_enabled).
    pub mask: Tensor,
    /// `u8`, one per token row: whether that row's mask plane is consulted.
    /// Always bound, never optional — every sdpa entry reads it on every
    /// launch; zeroed means unmasked.
    pub mask_enabled: Tensor,
    /// Key positions from one row's plane to the next, as the shaders read
    /// it.
    pub mask_stride: u32,
    /// The second row axis's seats, or `None` for a fire whose lanes carried
    /// no image.
    pub patches: Option<PatchHandles>,
    /// `RuntimeInput::MropePositions`: `i32`, `[rows, 3]`, the trunk's
    /// triple-wide position stream. `None` for a plan with no `rope_mrope`.
    pub mrope_positions: Option<Tensor>,
    /// `i32`, `[lanes]`: buffered tokens each lane replays ahead of its rows,
    /// and the rows of its extended run whose recurrent state persists
    /// (`crate::rs`). `None` for a fire every lane of which folds.
    pub rs_replay: Option<Tensor>,
    pub rs_commit: Option<Tensor>,
}

/// The patch axis's device seats, as one fire resolved them.
#[derive(Debug, Clone, Copy)]
pub struct PatchHandles {
    /// `RuntimeInput::Patches`: `[patch rows, C·T·P²]` in the plan's element.
    pub patches: Tensor,
    /// `RuntimeInput::PatchSegments`: `i32`, `[images + 1]`.
    pub segments: Tensor,
    /// `RuntimeInput::PatchRoutes`: `i32`, `[patch rows]`, `-1` for a row
    /// with no destination.
    pub routes: Tensor,
    /// `RuntimeInput::PatchPositions`: `i32`, `[patch rows, 3]`.
    pub positions: Tensor,
    /// `RuntimeInput::PatchEmbedRows`: `i32`, `[patch rows, taps]`, or `None`
    /// for a plan that reads the table on its native grid.
    pub embed_rows: Option<Tensor>,
    /// `RuntimeInput::PatchEmbedWeights`: `f32`, `[patch rows, taps]`.
    pub embed_weights: Option<Tensor>,
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
    /// Which adapter each token row routes to, in fire row order (the
    /// correction kernel indexes `routes[row]` beside `x[row]`), or `None`
    /// when no lane carried one.
    pub adapter_routes: Option<&'a [i32]>,
    /// Which lane owns each token row, in fire row order.
    pub request_of_token: &'a [i32],
    /// One geometry per kv space, in space order.
    pub spaces: &'a [Geometry],
    /// This fire's expanded lane masks, or `None` when no lane carried one.
    pub mask: Option<&'a crate::mask::Staged>,
    /// The patch rectangle, already seriated into fire patch order, or `None`
    /// for a fire with no image in it.
    pub patches: Option<PatchFire<'a>>,
    /// The trunk's `(t, h, w)` stream, three per TOKEN row in fire row order,
    /// or `None` for a plan that declares no `rope_mrope`.
    pub mrope_positions: Option<&'a [i32]>,
    /// The recurrent seat's two per-lane tables (`crate::rs`), or `None` for
    /// a fire every lane of which folds in the forward.
    pub rs_replay: Option<&'a [i32]>,
    pub rs_commit: Option<&'a [i32]>,
}

/// The patch axis's six vectors, host side, in fire patch order. Already
/// placed at each lane's own `patch_offset`; the seriation happened upstream
/// in `serve`.
#[derive(Debug, Clone, Copy)]
pub struct PatchFire<'a> {
    /// `[patch rows, C·T·P²]` in the plan's element, little-endian.
    pub payload: &'a [u8],
    /// `[images + 1]` `i32`: the patch axis's own indptr.
    pub segments: &'a [i32],
    /// `[patch rows]` `i32`, already rebased onto absolute fire token rows —
    /// except the `-1`s, which are a sentinel and not an address.
    pub routes: &'a [i32],
    /// `[patch rows * 3]` `i32`.
    pub positions: &'a [i32],
    /// `[patch rows * taps]` `i32`, or empty on the native grid.
    pub embed_rows: &'a [i32],
    /// `[patch rows * taps]` `f32`, or empty beside an empty `embed_rows`.
    pub embed_weights: &'a [f32],
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
    /// The patch axis's six regions, or `None` for a load whose plan states
    /// no patch row — where the axis costs the reservation nothing.
    patch: Option<PatchAt>,
    /// `RuntimeInput::MropePositions`' region, or `None` for a plan that
    /// declares no multimodal rotation. `rows * 3` `i32` at the ceiling.
    mrope: Option<u64>,
    /// The recurrent seat's per-lane tables, `lanes` `i32` each. Reserved
    /// unconditionally: two words a lane.
    rs_replay: u64,
    rs_commit: u64,
}

impl Inputs {
    /// Reserve the vectors a deployment's ceilings admit. `device` is taken
    /// because an `MTLBuffer` is made by a device object.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::Fault::Device) when the device declined the
    /// reservation, [`Fault::Ceiling`](crate::Fault::Ceiling) when the carve
    /// is longer than one `MTLBuffer` may be, and
    /// [`Fault::Deviceless`](crate::Fault::Deviceless) off Apple.
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        device: &Context,
        budget: &Budget,
        paging: Paging,
        spaces: usize,
        classes: usize,
        gathered: usize,
        patch: Option<PatchSeat>,
        mrope: bool,
    ) -> Result<Inputs> {
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(budget.max_lanes);
        let pages = u64::from(budget.max_lanes) * u64::from(paging.pages_per_slot);
        // At most `k(k+1)/2` windows for `k` classes, plus one shared zero
        // window; `gathered` bounds how many need the larger `Fallback::Copy`
        // layout.
        let per_gathered =
            3 * rows + spaces as u64 * (2 * lanes + (lanes + 1) + pages);
        let window_ints =
            (classes * (classes + 1) / 2 + 1) as u64 * (lanes + 1) + gathered as u64 * per_gathered;

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
        // Per row, not per lane: every metal ssm shader indexes `slots[r]` by
        // token row.
        let slot_of_row = take(rows * 4);
        // Reserved unconditionally so the store's layout does not depend on
        // whether the plan declares a correction.
        let adapter_routes = take(rows * 4);
        let request_of_token = take(rows * 4);
        // One byte per (query, key) pair, row-major with a stated stride
        // (`attention_mask[row * stride + kp]`).
        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_stride = u32::try_from(context).unwrap_or(u32::MAX);
        let mask_plane_bytes = rows * context;
        let mask_planes = take(mask_plane_bytes);
        // Always bound: read on every sdpa launch, so an unbound seat would
        // be a null dereference.
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
        // `None` for a text-only load: no region taken, `at` does not move.
        let patch = patch.map(|seat| PatchAt {
            payload: take(seat.rows * seat.row_bytes),
            segments: take((seat.images + 1) * 4),
            routes: take(seat.rows * 4),
            positions: take(seat.rows * AXES * 4),
            embed_rows: take(seat.rows * seat.embed_taps * 4),
            embed_weights: if seat.embed_weights {
                take(seat.rows * seat.embed_taps * 4)
            } else {
                0
            },
            seat,
        });
        let mrope = mrope.then(|| take(rows * AXES * 4));
        let rs_replay = take(lanes * 4);
        let rs_commit = take(lanes * 4);
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
            patch,
            mrope,
            rs_replay,
            rs_commit,
        })
    }

    /// The patch element this load computes in, or `None` for a plan that
    /// states no patch row.
    #[must_use]
    pub fn patch_element(&self) -> Option<Dtype> {
        self.patch.map(|at| at.seat.dtype)
    }

    /// Every byte the inputs hold.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    /// Write one fire's vectors into the store and hand back their handles.
    /// The caller must call this before committing the command buffer that
    /// reads what it wrote.
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

        // All-valid in an eager fire; the kv writers read this unconditionally.
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
        let rs_tables = match (fire.rs_replay, fire.rs_commit) {
            (Some(replay), Some(commit)) => {
                if replay.len() != lanes as usize || commit.len() != lanes as usize {
                    return Err(Fault::Ceiling {
                        what: "recurrent seat tables, one entry per lane",
                        need: replay.len().max(commit.len()) as u64,
                        have: u64::from(lanes),
                    });
                }
                self.store.write(self.rs_replay, bytes_of(replay))?;
                self.store.write(self.rs_commit, bytes_of(commit))?;
                true
            }
            _ => false,
        };

        // A fire no lane routed writes nothing here and binds no seat.
        let adapter_routes = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                self.store.write(self.adapter_routes, bytes_of(routes))?;
                Some(routes.len() as u32)
            }
        };

        // Both are read on every launch, so always bound; no lane masked
        // means the enable column is zeroed, not that the seat is absent.
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

        // A fire with no image writes nothing here and binds no seat.
        let patches = match (fire.patches, self.patch) {
            (None, _) => None,
            (Some(_), None) => {
                return Err(Fault::Ceiling {
                    what: "patch rows against a load that reserved none",
                    need: 1,
                    have: 0,
                });
            }
            (Some(staged), Some(at)) => {
                let seat = at.seat;
                // Each of the six is checked against its own reservation,
                // never against the region behind it.
                let rows = (staged.payload.len() as u64)
                    .checked_div(seat.row_bytes)
                    .unwrap_or(0);
                for (what, have, ceiling) in [
                    ("patch payload bytes", staged.payload.len() as u64, seat.rows * seat.row_bytes),
                    ("patch segments", staged.segments.len() as u64, seat.images + 1),
                    ("patch routes", staged.routes.len() as u64, seat.rows),
                    ("patch positions", staged.positions.len() as u64, seat.rows * AXES),
                    ("patch table rows", staged.embed_rows.len() as u64, seat.rows * seat.embed_taps),
                    (
                        "patch table weights",
                        staged.embed_weights.len() as u64,
                        if seat.embed_weights { seat.rows * seat.embed_taps } else { 0 },
                    ),
                ] {
                    if have > ceiling {
                        return Err(Fault::Ceiling { what, need: have, have: ceiling });
                    }
                }
                self.store.write(at.payload, staged.payload)?;
                self.store.write(at.segments, bytes_of(staged.segments))?;
                self.store.write(at.routes, bytes_of(staged.routes))?;
                self.store.write(at.positions, bytes_of(staged.positions))?;
                if !staged.embed_rows.is_empty() {
                    self.store.write(at.embed_rows, bytes_of(staged.embed_rows))?;
                }
                if !staged.embed_weights.is_empty() {
                    self.store
                        .write(at.embed_weights, f32_bytes_of(staged.embed_weights))?;
                }
                let taps = u32::try_from(seat.embed_taps).unwrap_or(u32::MAX).max(1);
                let rows32 = u32::try_from(rows).unwrap_or(u32::MAX);
                let element = model_compiler::arena::elem_bytes(seat.dtype).unwrap_or(1);
                let width = u32::try_from(seat.row_bytes.checked_div(element).unwrap_or(0))
                    .unwrap_or(u32::MAX);
                Some(PatchHandles {
                    patches: Tensor::new(
                        handles.bind(&self.store, at.payload, staged.payload.len() as u64)?,
                        rows32,
                        width,
                        seat.dtype,
                    ),
                    segments: i32s(
                        handles,
                        &self.store,
                        at.segments,
                        staged.segments.len() as u32,
                    )?,
                    routes: i32s(handles, &self.store, at.routes, staged.routes.len() as u32)?,
                    // `[patch rows, 3]`: one triple per row, not a column.
                    positions: Tensor::new(
                        handles.bind(
                            &self.store,
                            at.positions,
                            staged.positions.len() as u64 * 4,
                        )?,
                        rows32,
                        AXES as u32,
                        Dtype::I32,
                    ),
                    embed_rows: if staged.embed_rows.is_empty() {
                        None
                    } else {
                        Some(Tensor::new(
                            handles.bind(
                                &self.store,
                                at.embed_rows,
                                staged.embed_rows.len() as u64 * 4,
                            )?,
                            staged.embed_rows.len() as u32 / taps,
                            taps,
                            Dtype::I32,
                        ))
                    },
                    embed_weights: if staged.embed_weights.is_empty() {
                        None
                    } else {
                        Some(Tensor::new(
                            handles.bind(
                                &self.store,
                                at.embed_weights,
                                staged.embed_weights.len() as u64 * 4,
                            )?,
                            staged.embed_weights.len() as u32 / taps,
                            taps,
                            Dtype::F32,
                        ))
                    },
                })
            }
        };

        // `[rows, 3]` `i32`. A plan with no multimodal rotation binds nothing.
        let mrope_positions = match (fire.mrope_positions, self.mrope) {
            (None, _) | (_, None) => None,
            (Some(triples), Some(at)) => {
                if triples.len() as u64 > u64::from(rows) * AXES {
                    return Err(Fault::Ceiling {
                        what: "trunk rotation triples",
                        need: triples.len() as u64,
                        have: u64::from(rows) * AXES,
                    });
                }
                self.store.write(at, bytes_of(triples))?;
                Some(Tensor::new(
                    handles.bind(&self.store, at, triples.len() as u64 * 4)?,
                    triples.len() as u32 / AXES as u32,
                    AXES as u32,
                    Dtype::I32,
                ))
            }
        };

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
                write_page: u32s(handles, &self.store, at.write_page, rows)?,
                write_offset: u32s(handles, &self.store, at.write_offset, rows)?,
            });
        }

        Ok(Handles {
            tokens: i32s(handles, &self.store, self.tokens, rows)?,
            positions: i32s(handles, &self.store, self.positions, rows)?,
            // Minted whole, at the bytes this fire wrote; `Windows::bind`
            // cuts one row per window out of it.
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
            // Minted at the fire's own rectangle: `rows` of `stride` bytes.
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
            patches,
            mrope_positions,
            rs_replay: if rs_tables {
                Some(i32s(handles, &self.store, self.rs_replay, lanes)?)
            } else {
                None
            },
            rs_commit: if rs_tables {
                Some(i32s(handles, &self.store, self.rs_commit, lanes)?)
            } else {
                None
            },
        })
    }

    /// The pool seats one fire lends its cache table. Nothing is minted
    /// here — every seat is a view [`write`](Inputs::write) already minted,
    /// so this stays infallible.
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
/// Fallible: the handle table is bounds-checked and finite.
fn i32s(
    handles: &crate::device::Handles,
    store: &Buffer,
    at: u64,
    rows: u32,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::I32))
}

/// The same column, wearing `u32` — for `write_page`/`write_offset`, which
/// `attn/kv_write.metal` declares `const device uint*` (same bytes).
fn u32s(
    handles: &crate::device::Handles,
    store: &Buffer,
    at: u64,
    rows: u32,
) -> Result<Tensor> {
    let buf = handles.bind(store, at, u64::from(rows) * 4)?;
    Ok(Tensor::new(buf, rows, 1, Dtype::U32))
}

/// A vector of `f32` as the bytes a copy takes — [`bytes_of`]'s twin for the
/// one staged vector that is not an integer (interpolation weights).
fn f32_bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy` with no padding/niche; all bytes are init and
    // readable as `u8`.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

/// A vector of `i32` as the bytes a copy takes. Little-endian, which every
/// device this ships on is.
fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is `Copy` with no padding/niche; all bytes are init and
    // readable as `u8`.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}
