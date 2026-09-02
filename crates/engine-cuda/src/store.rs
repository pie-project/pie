//! Cudamalloc-backed kv page and recurrent-state pools; page/cell geometry (which lane's pages are, which cell a token lands in) lives in [`kv`].

pub mod kv;
pub mod rs;

use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use model_ir::{CacheRow, Dtype, Trace};

use crate::device::elastic::{self, Arena, Commit, PhysicalPool};
use crate::error::{Fault, Result};
use crate::settle::Airborne;
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Facts, Paging};

/// Maps `model_exec::store` faults into this shell's `Fault` type.
impl From<model_exec::store::Fault> for Fault {
    fn from(fault: model_exec::store::Fault) -> Fault {
        match fault {
            model_exec::store::Fault::Ceiling { what, need, have } => {
                Fault::Ceiling { what, need, have }
            }
            model_exec::store::Fault::Unbound { what } => Fault::Unbound { what },
            model_exec::store::Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            } => Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            },
        }
    }
}

/// Page layout this shell writes and reads: NHD, `[page][token][head][dim]`. `head_stride`/`seq_stride` must be set consistent with NHD — `kv::head_split` reads them assuming this layout.
const NHD: i32 = 0;

/// Whether a deployment fits on the card, computed once before any allocation.
///
/// ```text
/// card         what the device has, total
/// ceiling      card x utilization        the operator's whole allowance
/// weights      the T0 weight tier        what `Plan::device_demand` will hold
/// floor        min(128 MiB, card/10)     the driver's landing room
/// pool         ceiling - weights - floor what the elastic supply may hold
/// minimum      one slot at the declared context, every cache row
/// ```
/// Admits when `pool >= minimum`; failing it is [`Fault::Residency`], while a peer process holding memory instead fails later as `Fault::OutOfMemory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    /// What the device has, total.
    pub card: u64,
    /// `card x utilization` — the operator's whole allowance for pie.
    pub ceiling: u64,
    /// The T0 weight tier's bytes.
    pub weights: u64,
    /// `min(128 MiB, card/10)`, held back for the driver.
    pub floor: u64,
    /// `ceiling - weights - floor`: what is left for the elastic supply.
    pub pool: u64,
    /// One slot at the declared context, across every cache row.
    pub minimum: u64,
}

impl Accounting {
    /// Computes the accounting from the card and the three demands. Pure arithmetic; needs no device.
    #[must_use]
    pub fn of(card: u64, utilization: f64, weights: u64, minimum: u64) -> Accounting {
        let fraction = if utilization.is_finite() {
            utilization.clamp(0.0, 1.0)
        } else {
            1.0
        };
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a byte count of a GPU card is far inside f64's exact integer range, \
                      and the product is floored back into u64 deliberately"
        )]
        let ceiling = (card as f64 * fraction) as u64;
        let floor = elastic::safety_floor_bytes(card);
        Accounting {
            card,
            ceiling,
            weights,
            floor,
            pool: ceiling.saturating_sub(weights).saturating_sub(floor),
            minimum,
        }
    }

    /// Whether the card holds this deployment. # Errors: [`Fault::Residency`] when the elastic pool's share is under one slot at the declared context.
    pub fn admit(&self) -> Result<()> {
        if self.pool >= self.minimum {
            return Ok(());
        }
        Err(Fault::Residency(format!(
            "the card does not hold this deployment: {card} bytes on the device, of which \
             `[engine] gpu_mem_utilization` allows pie {ceiling}; this load's weight tier \
             takes {weights} and the driver's safety floor holds back {floor}, leaving the \
             elastic pool {pool} bytes — and one sequence at the declared context needs \
             {minimum} across this model's cache rows. weight tier + elastic pool + safety \
             floor must fit inside the fraction of the card, and here they do not. Lower \
             `[model] max_context` or `[model] slots`, raise `[engine] \
             gpu_mem_utilization`, or state a `[model] device_weight_budget` that streams \
             the weight tier down.",
            card = self.card,
            ceiling = self.ceiling,
            weights = self.weights,
            floor = self.floor,
            pool = self.pool,
            minimum = self.minimum,
        )))
    }
}

/// Bytes one slot of every cache row occupies: `pages_per_slot` pages of every kv plane at its own width, plus one slot of every recurrent slab. # Errors: [`Fault::Unbound`] for a cache row whose element has no size.
pub fn one_slot_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut bytes: u64 = 0;
    for row in &trace.caches {
        match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                ..
            } => {
                let element = elem_bytes(name, *dtype)?;
                let cells = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
                for width in planes {
                    bytes = bytes.saturating_add(cells * width * element);
                }
            }
            CacheRow::State { name, slab, dtype } => {
                let stride: u64 = slab.iter().product();
                bytes = bytes.saturating_add(stride * elem_bytes(name, *dtype)?);
            }
        }
    }
    Ok(bytes)
}

/// Queries the card and checks the accounting before `Shell::load` allocates anything. `weights == 0` means full residency (`Plan::default()`), not no weights. # Errors: [`Fault::Runtimeless`] with no runtime, [`Fault::Device`] for the memory query, [`Fault::Unbound`] for a sizeless cache element, [`Fault::Residency`] for a deployment the card does not hold.
pub fn admit_the_card(
    utilization: f64,
    weights: u64,
    trace: &Trace,
    paging: Paging,
) -> Result<Accounting> {
    let full: u64 = crate::weights::plane_bytes(trace)?
        .iter()
        .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
        .sum();
    let weights = match weights {
        0 => full,
        stated => stated.min(full),
    };
    let accounting = Accounting::of(
        card_bytes()?,
        utilization,
        weights,
        one_slot_bytes(trace, paging)?,
    );
    accounting.admit()?;
    Ok(accounting)
}

/// What this device has, total — the one number [`Accounting`] cannot derive. # Errors: [`Fault::Runtimeless`] with no runtime, [`Fault::Device`] for the query.
fn card_bytes() -> Result<u64> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        crate::device::ctx::check("cudaMemGetInfo", asked)?;
        Ok(total as u64)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// How one cache row is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A paged kv space: which geometry space it belongs to, and how the row's planes are handed out as key/value.
    Kv {
        space: u32,
        dtype: Dtype,
        /// Elements one token writes into the key plane (plane 0).
        keys_width: u64,
        /// Elements one token writes into the value plane: plane 1's width if declared, else plane 0's (same plane as keys).
        values_width: u64,
        /// Index of the arena the value handle names: 0 for a one-plane row (same bytes as keys), 1 otherwise.
        values_plane: usize,
        /// One head's width under NHD, read by kernels that recover head count from the stride pair (`kv::head_split`).
        head_stride: u64,
    },
    /// A recurrent slab: elements per slot.
    State { stride: u64, dtype: Dtype },
}

/// Per-fire geometry a pool row borrows: this fire's page-list vectors and its padding mask. [`CacheTable`] is rebuilt each fire from this short-lived `Copy` data plus long-lived storage.
#[derive(Debug, Clone, Copy)]
pub struct SpaceSeat {
    /// `i32`, `[lanes + 1]`: this space's page-list bounds.
    pub page_indptr: Tensor,
    /// `i32`: the flat page-id list.
    pub page_indices: Tensor,
    /// `i32`, `[lanes]`: each lane's last-page fill.
    pub last_page_lens: Tensor,
    /// `u8`, `[rows]`: the padding mask the writers read.
    pub row_valid: Tensor,
}

/// What a fire lends the pools.
#[derive(Debug, Clone)]
pub struct Seats {
    /// This fire's lanes.
    pub lanes: u32,
    /// This fire's token rows.
    pub rows: u32,
    /// How many pages its geometry named.
    pub pages: u32,
    /// One seat per kv geometry space.
    pub spaces: Vec<SpaceSeat>,
    /// `i32`, `[lanes]`: which recurrent slot each lane owns.
    pub slot_ids: Tensor,
    /// Whether any lane of this fire folds at all. `false` means every recurrent launch computes rows without writing banks back.
    pub write_state: bool,
    /// `u8`, `[lanes]`: per-request fold predicate, or [`Tensor::ABSENT`]. Indexed per lane (per request), not per token row — matches what `attn/ssm.cuh`'s `row_persists` reads.
    pub write_state_mask: Tensor,
    /// `i32`, `[lanes]`: where each request's accepted prefix ends, or [`Tensor::ABSENT`] for a fire that truncates nothing.
    pub commit_len: Tensor,
    /// `i32`, `[lanes]`: where each request's tail segment begins, or [`Tensor::ABSENT`] for a fire no row splits — same vector as [`Seats::commit_len`], read from the other end; never both on one launch.
    pub begin_at: Tensor,
}

impl Seats {
    /// Sets this fire's recurrent write/fold fields.
    #[must_use]
    pub fn rs(mut self, write_state: bool, mask: Tensor, commit_len: Tensor) -> Seats {
        self.write_state = write_state;
        self.write_state_mask = mask;
        self.commit_len = commit_len;
        self
    }

    /// Sets the interior fold boundary for a fire whose row splits head/tail.
    #[must_use]
    pub fn splitting(mut self, boundary: Tensor) -> Seats {
        self.begin_at = boundary;
        self
    }
}

/// One span of kv cells moved inside this device's pools — a whole page and a single token are both just a run of `tokens` cells at `(page, token)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    /// The page read.
    pub src_page: u32,
    /// The first token slot read in it.
    pub src_token: u32,
    /// The page written.
    pub dst_page: u32,
    /// The first token slot written in it.
    pub dst_token: u32,
    /// How many consecutive token slots move. `page_size` is a whole page.
    pub tokens: u32,
}

/// Every cache space's bytes: one elastic arena per row plane, reserved at the budget's ceiling with physical pages mapped under its front as admission commits them. Arena base addresses never move once reserved.
#[derive(Debug)]
pub struct Pools {
    /// The budgeted supply every arena below draws physical pages from.
    pool: PhysicalPool,
    /// One entry per cache row: declared planes for a kv row, one arena for a state row.
    rows: Vec<Vec<Arena>>,
    shapes: Vec<Shape>,
    paging: Paging,
    /// Whether the device is idle — the run-ahead counter. `trim` must not unmap while an unsettled step may still be reading the tail.
    airborne: Option<Airborne>,
    /// The kv-page watermark the last admitted frame committed to.
    committed_kv_pages: u32,
    /// The state-slot watermark, likewise.
    committed_state_slots: u32,
}

impl Pools {
    /// Reserves the pools one plan needs at one deployment's budget. A kv row is sized by its declaration, plane by plane; `facts` is indexed by cache row (not geometry space) and checked where a paged launch restated a width. # Errors: [`Fault::Unbound`] for a kv row declaring no planes or more than two, a paged-reader width mismatch, or a sizeless element; [`Fault::Device`] for the allocations.
    pub fn reserve(
        device: i32,
        utilization: f64,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
    ) -> Result<Pools> {
        // Arenas reserve address space at the budget ceiling; nothing is mapped until admission asks for it.
        let pool = PhysicalPool::open(device, utilization)?;
        let mut rows: Vec<Vec<Arena>> = Vec::with_capacity(trace.caches.len());
        let mut shapes = Vec::with_capacity(trace.caches.len());

        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    space,
                } => {
                    let element = elem_bytes(name, *dtype)?;
                    let cells = paging.pages() * u64::from(paging.page_size);
                    let (keys_width, values_width, values_plane) = match planes.as_slice() {
                        // One plane addressed as both k and v: both handles name the same bytes.
                        [plane] => (*plane, *plane, 0),
                        [keys, values] => (*keys, *values, 1),
                        [] => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares no planes at all — one \
                                     token's entry is written as at least one plane"
                                ),
                            });
                        }
                        many => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares {} planes — this shell \
                                     binds a key plane and a value plane, and knows no third",
                                    many.len()
                                ),
                            });
                        }
                    };
                    let restated = facts.row(index);
                    // A restatement only exists where a paged launch named a head count (`kv_heads != 0`); where present it must match the declaration.
                    if let Some(seat) = restated.filter(|seat| seat.kv_heads != 0) {
                        let heads = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                        if heads != keys_width || heads != values_width {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares the planes {planes:?} while \
                                     the paged launches that read it restate {} heads of {} — a \
                                     {heads}-wide row",
                                    seat.kv_heads, seat.head_dim
                                ),
                            });
                        }
                    }
                    // One arena per declared plane; page `p` of plane `i` is at `p * page_size * width_i * element` from that plane's base, so a page-count commit is a prefix.
                    let mut planes_of_row = Vec::with_capacity(planes.len());
                    for width in planes {
                        planes_of_row.push(Arena::reserve(
                            &pool,
                            cells * width * element,
                            "bytes of a kv plane",
                        )?);
                    }
                    rows.push(planes_of_row);
                    shapes.push(Shape::Kv {
                        space: *space,
                        dtype: *dtype,
                        keys_width,
                        values_width,
                        values_plane,
                        // Falls back to keys_width when no paged launch stated a head width.
                        head_stride: restated.map_or(keys_width, |seat| u64::from(seat.head_dim)),
                    });
                }
                CacheRow::State { name, slab, dtype } => {
                    let stride: u64 = slab.iter().product();
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, *dtype)?;
                    // Slot `s` at `s * stride * element`: the slot watermark is a prefix here too.
                    rows.push(vec![Arena::reserve(
                        &pool,
                        bytes,
                        "bytes of a recurrent slab",
                    )?]);
                    shapes.push(Shape::State { stride, dtype: *dtype });
                }
            }
            debug_assert_eq!(rows.len(), index + 1, "one arena set per cache row");
        }
        Ok(Pools {
            pool,
            rows,
            shapes,
            paging,
            airborne: None,
            committed_kv_pages: 0,
            committed_state_slots: 0,
        })
    }

    /// Watches the run-ahead counter so `trim` can tell an idle device from a busy one. Set once at load, since `Pools` is built before the counters exist.
    pub fn watch(&mut self, airborne: Airborne) {
        self.airborne = Some(airborne);
    }

    /// How the pages are handed out.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// Every byte these pools may ever hold — the ceiling the address space was reserved at.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::max_bytes)
            .sum()
    }

    /// Bytes actually under a mapping right now.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::committed_bytes)
            .sum()
    }

    /// The most that has ever been mapped, summed across arenas — the high water a trim is measured against.
    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::high_water_bytes)
            .sum()
    }

    /// Bytes one logical page of the elastic supply holds — what `PoolFacts::elastic_page_bytes` publishes.
    #[must_use]
    pub fn elastic_page_bytes(&self) -> u64 {
        self.pool.page_bytes()
    }

    /// The most logical pages this load may ever map.
    #[must_use]
    pub fn elastic_budget_pages(&self) -> u64 {
        self.pool.hard_pages()
    }

    /// The kv-page and state-slot watermarks the last admitted frame committed to.
    #[must_use]
    pub fn committed_watermarks(&self) -> (u32, u32) {
        (self.committed_kv_pages, self.committed_state_slots)
    }

    /// Every arena's base address, in row-then-plane order. These addresses never move once reserved.
    #[must_use]
    pub fn bases(&self) -> Vec<u64> {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::base)
            .collect()
    }

    /// The cache table one fire resolves its cache ids through. # Errors: [`Fault::Unbound`] for a kv row whose space this fire seated no geometry for.
    pub fn table(&self, seats: &Seats) -> Result<CacheTable> {
        let mut table = Vec::with_capacity(self.shapes.len());
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            table.push(match *shape {
                Shape::Kv {
                    space,
                    dtype,
                    keys_width,
                    values_width,
                    values_plane,
                    head_stride,
                } => {
                    let seat = seats
                        .spaces
                        .get(space as usize)
                        .ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "cache space {space}, for which this fire wrote no \
                                     geometry"
                            ),
                        })?;
                    let cells = self.paging.pages() * u64::from(self.paging.page_size);
                    // The base, not the committed edge — a pool handle names the whole plane at its ceiling; how much is backed is checked at admission, not here.
                    let plane = |at: usize, width: u64| {
                        Tensor::new(
                            planes.get(at).map_or(0, elastic::Arena::base),
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    CachePool::Kv { space, pool: KvPool {
                        keys: plane(0, keys_width),
                        // A one-plane row seats values_plane == 0, so both handles name the same plane.
                        values: plane(values_plane, values_width),
                        // Shadow/scale/envelope planes belong to quantized schemes; a native pool binds none.
                        bf16_keys: Tensor::new(0, 0, 0, dtype),
                        bf16_values: Tensor::new(0, 0, 0, dtype),
                        key_scales: Tensor::new(0, 0, 0, Dtype::U8),
                        value_scales: Tensor::new(0, 0, 0, Dtype::U8),
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        last_page_lens: seat.last_page_lens,
                        row_valid: seat.row_valid,
                        env_min: Tensor::new(0, 0, 0, dtype),
                        env_max: Tensor::new(0, 0, 0, dtype),
                        has_envelopes: false,
                        page_size: narrow(u64::from(self.paging.page_size)),
                        // NHD: seq_stride is the key plane's whole width, head_stride one head's share — the pair `kv::head_split` reads the head width from.
                        seq_stride: wide(keys_width),
                        head_stride: wide(head_stride),
                        layout: NHD,
                        scheme_byte: 0,
                        block_size: 0,
                        max_pages_per_request: narrow(u64::from(self.paging.pages_per_slot)),
                        pages_in_batch: narrow(u64::from(seats.pages)),
                    },
                    }
                }
                Shape::State { stride, dtype } => CachePool::Recurrent(RecurrentPool {
                    write_state: seats.write_state,
                    write_state_mask: seats.write_state_mask,
                    commit_len: seats.commit_len,
                    begin_at: seats.begin_at,
                    // `false` is the plain fold-per-token forward's policy, bound everywhere so a replay matches to the byte.
                    fused_decay: false,
                    // One row serves both seats: `slab` and `conv_slab` point at the same bytes for a `CacheRow::State` row.
                    slab: Tensor::new(
                        planes.first().map_or(0, elastic::Arena::base),
                        self.paging.slots,
                        narrow(stride) as u32,
                        dtype,
                    ),
                    slot_ids: seats.slot_ids,
                    slot_stride_elems: stride as i64,
                    conv_slab: Tensor::new(
                        planes.first().map_or(0, elastic::Arena::base),
                        self.paging.slots,
                        narrow(stride) as u32,
                        dtype,
                    ),
                    conv_stride: stride as i64,
                }),
            });
        }
        Ok(CacheTable(table))
    }

    /// Clears one slot's recurrent state. A slot is its history, so opening a sequence in a reused slot must zero what the previous one left — the scan reads the whole bank on its first step. # Errors: [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the fill.
    pub fn clear(&mut self, slot: u32) -> Result<()> {
        self.zero_slot(None, slot)
    }

    /// [`Pools::clear`], on the fire's stream. A synchronous `cudaMemset` here would drain everything airborne, so the fire path issues it on the stream instead, ahead of the launches that read the bank. # Errors: as [`Pools::clear`].
    pub fn clear_on(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<()> {
        self.zero_slot(Some(stream), slot)
    }

    /// Copies one recurrent slot's banks onto another, on `stream` — the device half of a copy-on-write fork. Whole slots only: a recurrent bank is a folded summary, not per-token entries, so partial moves are refused before reaching here. # Errors: [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the copy.
    pub fn copy_slot(
        &mut self,
        stream: *mut core::ffi::c_void,
        src: u32,
        dst: u32,
    ) -> Result<()> {
        for slot in [src, dst] {
            if slot >= self.paging.slots {
                return Err(Fault::Ceiling {
                    what: "recurrent slots",
                    need: u64::from(slot) + 1,
                    have: u64::from(self.paging.slots),
                });
            }
        }
        if src == dst {
            return Ok(());
        }
        // Both slots must be backed: a fork may name a slot no frame has admitted yet, so commit the watermark before reading/writing.
        self.ensure_state(src.max(dst) + 1)?;
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            crate::device::copy_d2d(
                stream,
                arena.span(u64::from(dst) * bytes, bytes)?,
                arena.span(u64::from(src) * bytes, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
        }
        Ok(())
    }

    /// Copies kv cells between pages of these pools, on `stream` — the device half of a prefix-tree fork. A page id names one cell run per plane of every `CacheRow::Kv` row, so the loop is over `rows x planes`. # Errors: [`Fault::Ceiling`] for a page past the pool or a run past a page's tokens, [`Fault::Device`] for the copies; overlapping src/dst is refused as `Invalid`.
    pub fn copy_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        moves: &[Move],
    ) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        let page_size = u64::from(self.paging.page_size);
        let mut highest = 0u64;
        for span in moves {
            for (page, token) in [
                (span.src_page, span.src_token),
                (span.dst_page, span.dst_token),
            ] {
                let end = u64::from(token) + u64::from(span.tokens);
                if end > page_size {
                    return Err(Fault::Ceiling {
                        what: "token slots in one kv page",
                        need: end,
                        have: page_size,
                    });
                }
                highest = highest.max(u64::from(page) + 1);
            }
        }
        // Both ends must be backed: a fork may name a destination page before any frame has admitted it.
        self.ensure_kv(u32::try_from(highest).unwrap_or(u32::MAX))?;
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::Kv {
                dtype,
                keys_width,
                values_width,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            for (at, arena) in planes.iter().enumerate() {
                // Plane 0 is keys, plane 1 values, each at its own width. A one-plane row has one arena, so copying it copies both.
                let width = if at == 0 { keys_width } else { values_width };
                let cell = width * element;
                for span in moves {
                    if span.tokens == 0 {
                        continue;
                    }
                    let bytes = u64::from(span.tokens) * cell;
                    let src = (u64::from(span.src_page) * page_size
                        + u64::from(span.src_token))
                        * cell;
                    let dst = (u64::from(span.dst_page) * page_size
                        + u64::from(span.dst_token))
                        * cell;
                    if src == dst {
                        continue;
                    }
                    crate::device::copy_d2d(
                        stream,
                        arena.span(dst, bytes)?,
                        arena.span(src, bytes)?,
                        usize::try_from(bytes).unwrap_or(0),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Reads back one slot's recurrent banks, every `CacheRow::State` row end to end in plan order. Synchronous D2H, not on any fire path. # Errors: [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the read.
    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        self.ensure_state(slot + 1)?;
        let mut out = Vec::new();
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let at = out.len();
            out.resize(at + usize::try_from(bytes).unwrap_or(0), 0);
            crate::device::copy_d2h(
                arena.span(u64::from(slot) * bytes, bytes)?,
                &mut out[at..],
            )?;
        }
        Ok(out)
    }

    fn zero_slot(&mut self, stream: Option<*mut core::ffi::c_void>, slot: u32) -> Result<()> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        // The bank must exist before it can be zeroed: `Shell::open` may reach a slot no frame has admitted yet.
        self.ensure_state(slot + 1)?;
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let at = arena.span(u64::from(slot) * bytes, bytes)?;
            let len = usize::try_from(bytes).unwrap_or(0);
            match stream {
                Some(stream) => crate::device::zero_span_on(stream, at, len)?,
                None => crate::device::zero_span(at, len)?,
            }
        }
        Ok(())
    }

    /// Commits the kv arenas up to a page watermark, for control-plane verbs that reach a page no frame admitted. Refuses with the same ceiling [`Supply::commit`](engine::frame::Supply::commit) would.
    fn ensure_kv(&mut self, pages: u32) -> Result<()> {
        let capacity = self.paging.pages();
        if u64::from(pages) > capacity {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(pages),
                have: capacity,
            });
        }
        match self.commit_to(pages, 0)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// Commits the recurrent arenas up to a slot watermark, for control-plane verbs that reach a slot no frame admitted.
    fn ensure_state(&mut self, slots: u32) -> Result<()> {
        match self.commit_to(0, slots)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// The atomic multi-arena commit: every arena is asked for the prefix its watermark names, and the whole set moves or none does. Only [`Pools::release_to`] ever lowers a watermark.
    fn commit_to(&mut self, kv_pages: u32, state_slots: u32) -> Result<Commit> {
        let kv_pages = kv_pages.max(self.committed_kv_pages);
        let state_slots = state_slots.max(self.committed_state_slots);
        let page_size = self.paging.page_size;
        let Pools {
            pool,
            rows,
            shapes,
            ..
        } = self;
        let mut targets = Vec::new();
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            for (at, arena) in planes.iter_mut().enumerate() {
                let bytes = watermark_bytes(shape, at, kv_pages, state_slots, page_size);
                targets.push(elastic::Target { arena, bytes });
            }
        }
        let outcome = elastic::commit_atomically(pool, &mut targets)?;
        if outcome == Commit::Committed {
            self.committed_kv_pages = kv_pages;
            self.committed_state_slots = state_slots;
        }
        Ok(outcome)
    }

    /// Unmaps every arena's tail down to the watermarks `hint` names — the inverse of [`Pools::commit_to`], and the only thing that lowers a watermark. Best-effort: releases whole map units only.
    fn release_to(&mut self, kv_pages: u32, state_slots: u32) {
        let page_size = self.paging.page_size;
        let Pools {
            pool,
            rows,
            shapes,
            ..
        } = self;
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            for (at, arena) in planes.iter_mut().enumerate() {
                let bytes = watermark_bytes(shape, at, kv_pages, state_slots, page_size);
                arena.release_tail(pool, bytes);
            }
        }
        self.committed_kv_pages = kv_pages;
        self.committed_state_slots = state_slots;
    }
}

/// How many bytes of one arena a pair of watermarks makes hot. A kv plane is `pages * page_size` cells of its own width; a recurrent slab is `slots` banks of its own stride.
fn watermark_bytes(
    shape: &Shape,
    plane: usize,
    kv_pages: u32,
    state_slots: u32,
    page_size: u32,
) -> u64 {
    match *shape {
        Shape::Kv {
            dtype,
            keys_width,
            values_width,
            ..
        } => {
            let width = if plane == 0 { keys_width } else { values_width };
            u64::from(kv_pages) * u64::from(page_size) * width * u64::from(elem_size(dtype))
        }
        Shape::State { stride, dtype } => {
            u64::from(state_slots) * stride * u64::from(elem_size(dtype))
        }
    }
}

/// One refused commit, as the fault this shell already speaks: `Fault::OutOfMemory` crosses as `Error::Exhausted` (worth retrying), `Fault::Ceiling` as `Error::Impossible`.
fn refuse(pool: &PhysicalPool, outcome: Commit) -> Fault {
    let page = pool.page_bytes();
    match outcome {
        Commit::Committed => Fault::program(
            "store::commit",
            "a committed outcome reached the refusal path".to_string(),
        ),
        Commit::Exhausted { required, budget } => Fault::OutOfMemory {
            need: required.saturating_mul(page),
            have: budget.saturating_mul(page),
        },
        Commit::Impossible { required, ceiling } => Fault::Ceiling {
            what: "bytes of elastic device memory",
            need: required.saturating_mul(page),
            have: ceiling.saturating_mul(page),
        },
    }
}

fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    model_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, stored as {dtype:?}, which has no element size"),
    })
}

fn elem_size(dtype: Dtype) -> u32 {
    model_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u32
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// A plane width, as the `i64` a pool's strides are spelled in.
fn wide(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// The engine's half of memory, elastic. [`Pools`] reserves address space at the budget's ceiling and maps physical pages as frames are admitted, one atomic commit across every arena ([`Pools::commit_to`]).
impl engine::frame::Supply for Pools {
    type Error = Fault;

    fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        // Only slots this shell pages are its supply — a lane with its own page table has its own addressing, so its slot number is not an index here.
        if demand.state_slots > self.paging.slots {
            return Err(Fault::Ceiling {
                what: "kv slots",
                need: u64::from(demand.state_slots),
                have: u64::from(self.paging.slots),
            });
        }
        let capacity =
            u64::from(self.paging.slots).saturating_mul(u64::from(self.paging.pages_per_slot));
        if u64::from(demand.kv_pages) > capacity {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(demand.kv_pages),
                have: capacity,
            });
        }
        match self.commit_to(demand.kv_pages, demand.state_slots)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// Gives tails back when the device is idle and the hint is below what is mapped; never maps a page. Hysteretic: a drop smaller than [`TRIM_HYSTERESIS_SHIFT`] is deferred, since `cuMemUnmap` isn't free.
    fn trim(&mut self, hint: engine::frame::Demand) {
        let idle = self.airborne.as_ref().is_none_or(|counts| counts.count() == 0);
        if !idle {
            return;
        }
        if u64::from(hint.kv_pages) >= u64::from(self.committed_kv_pages)
            && hint.state_slots >= self.committed_state_slots
        {
            return;
        }
        if !self.trim_is_worth_the_unmap(hint) {
            return;
        }
        self.release_to(
            hint.kv_pages.min(self.committed_kv_pages),
            hint.state_slots.min(self.committed_state_slots),
        );
    }
}

/// How far below the watermark a hint must fall before an unmap is paid for, as a right shift (an eighth). Also the pressure line: within an eighth of budget, any hint triggers a trim.
const TRIM_HYSTERESIS_SHIFT: u32 = 3;

impl Pools {
    /// Is this drop large enough — or the pool tight enough — to be worth a `cuMemUnmap`? See [`Supply::trim`](engine::frame::Supply::trim)'s note on the band.
    fn trim_is_worth_the_unmap(&self, hint: engine::frame::Demand) -> bool {
        let budget = self.pool.budget_pages();
        let free = budget.saturating_sub(self.pool.committed_pages());
        if free <= budget >> TRIM_HYSTERESIS_SHIFT {
            // Under pressure: the band is off and the hint is obeyed.
            return true;
        }
        let kv_drop = u64::from(self.committed_kv_pages)
            .saturating_sub(u64::from(hint.kv_pages.min(self.committed_kv_pages)));
        let state_drop = u64::from(
            self.committed_state_slots
                .saturating_sub(hint.state_slots.min(self.committed_state_slots)),
        );
        kv_drop > u64::from(self.committed_kv_pages) >> TRIM_HYSTERESIS_SHIFT
            || state_drop > u64::from(self.committed_state_slots) >> TRIM_HYSTERESIS_SHIFT
    }
}
