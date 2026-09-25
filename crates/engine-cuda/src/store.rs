pub mod kv;
pub mod rs;

use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use model_ir::{Attention, CacheRow, Def, Dtype, Operation, Trace};

use crate::device::elastic::{self, Arena, Commit, PhysicalPool};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable, PoolSlabs};
use crate::settle::Airborne;
use crate::store::kv::{Facts, Paging};

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

const NHD: i32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    pub card: u64,
    pub ceiling: u64,
    pub weights: u64,
    pub floor: u64,
    pub pool: u64,
    pub minimum: u64,
}

impl Accounting {
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

/// What `Pools::fit_the_card` did when the declared pool was larger than
/// the card: the page count asked, the count it was sized to, the bytes the
/// card had for the cache rows, and what was held back from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fitted {
    pub asked: u64,
    pub fit: u64,
    pub room: u64,
    pub held: u64,
    pub bodies: u64,
    pub spare: u64,
    pub slots: u32,
}

/// The least `[engine] bodies_mem` is held down to on a card with little
/// room: enough for the decode bodies, which are what a serve replays every
/// step and cost tens of megabytes each.
pub const BODIES_FLOOR_BYTES: u64 = 512 << 20;

/// What the graph bodies may take on this card: the configured allowance,
/// or a quarter of `room` — what is left for them and the cache rows once
/// one sequence and the guests' programs are seated — whichever is less,
/// but not under the floor while the room affords it. A fixed 4 GiB on a
/// 24 GB card is a sixth of the device before a second sequence seats.
#[must_use]
pub fn bodies_allowance(configured: u64, room: u64) -> u64 {
    configured.min(room).min((room / 4).max(BODIES_FLOOR_BYTES))
}

/// What is held for the bodies and the guests' programs together on a
/// card with `spare` bytes past one sequence, in the order they yield: the
/// bodies' floor first, since a load with no armed body walks every step
/// eagerly and that is no serve, then the programs, then the bodies' share
/// of what is left. Returns `(bodies, programs)`.
#[must_use]
pub fn holds_within(configured: u64, programs: u64, spare: u64) -> (u64, u64) {
    let floor = configured.min(BODIES_FLOOR_BYTES).min(spare);
    let programs = programs.min(spare.saturating_sub(floor));
    let bodies = bodies_allowance(configured, spare.saturating_sub(programs)).max(floor);
    (bodies, programs)
}

/// The largest token budget at or under `max_tokens`, halving toward
/// `floor`, whose working set (`working`, the activation arena and the
/// attention workspaces at that budget) fits `room`. Halving keeps the
/// bucket lattice a prefix of what the operator asked for.
#[must_use]
pub fn tokens_within(max_tokens: u32, floor: u32, room: u64, working: impl Fn(u32) -> u64) -> u32 {
    let mut tokens = max_tokens;
    while tokens > floor && working(tokens) > room {
        tokens = (tokens / 2).max(floor);
    }
    tokens
}

/// The rows a guest program's epilogue holds per lane, each as wide as the
/// out seam: the logits it reads, and the mask, noise and probability rows
/// a sampling program derives from them.
pub const PROGRAM_ROWS_A_LANE: u64 = 4;

/// What one guest program's scratch takes at the admitted lane count. A
/// program is registered after the load, so its stride is not known when
/// the cache rows are declared; what is known is that its per-lane scratch
/// is laid out over rows of the out seam, that the shell sizes it for the
/// lane count rounded up to a power of two, and that the program contract
/// caps the whole at `eta_exec::SCRATCH_MAX_BYTES`. The pool leaves this
/// much on the card so the first program to arrive is not the frame that
/// finds the pool has taken everything.
#[must_use]
pub fn program_scratch_reserve(max_lanes: u32, out_row_bytes: u64) -> u64 {
    let lanes = u64::from(
        max_lanes
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(u32::MAX),
    );
    let row = out_row_bytes
        .checked_next_multiple_of(eta_exec::SCRATCH_ALIGN)
        .unwrap_or(u64::MAX);
    lanes
        .saturating_mul(row)
        .saturating_mul(PROGRAM_ROWS_A_LANE)
        .min(eta_exec::SCRATCH_MAX_BYTES)
}

/// What the decoded-weight tiles take on this load: a plane stored in codes
/// and scales is decoded to bf16 before a prefill's dense gemm reads it,
/// into one tile a stream the size of the widest plane fired on it, in the
/// 8 MiB grain the scratch grows by. The pool holds this out for them as it
/// does for the guests' programs, since the tiles are taken at fire time.
#[must_use]
pub fn decoded_weight_reserve(widest_plane_bytes: u64, streams: u32) -> u64 {
    const GRAIN: u64 = 8 << 20;
    widest_plane_bytes
        .checked_next_multiple_of(GRAIN)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(streams.max(1)))
}

/// The largest page count at or under `asked` whose watermark fits `room`,
/// or zero when not even the first page does. `declared_at` is the watermark
/// at a page count and only ever grows with it, so this is a bisection over
/// a step function, not a division: each plane rounds up to a map unit, and
/// a model has dozens of planes.
#[must_use]
pub fn pages_within(asked: u64, room: u64, declared_at: impl Fn(u64) -> u64) -> u64 {
    if declared_at(asked) <= room {
        return asked;
    }
    let (mut fits, mut over) = (0u64, asked);
    while over - fits > 1 {
        let probe = fits + (over - fits) / 2;
        if declared_at(probe) <= room {
            fits = probe;
        } else {
            over = probe;
        }
    }
    fits
}

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

/// One sequence's cache rows: its kv pages at the declared context and the
/// recurrent slabs, the least `Pools::fit_the_card` seats before refusing.
pub fn least_sequence_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut state: u64 = 0;
    for row in &trace.caches {
        if let CacheRow::State { name, slab, dtype } = row {
            let stride: u64 = slab.iter().product();
            state = state.saturating_add(stride * elem_bytes(name, *dtype)?);
        }
    }
    Ok(one_slot_bytes(trace, paging)?.saturating_add(state))
}

pub fn admit_the_card(
    utilization: f64,
    weights: u64,
    extra_resident: u64,
    trace: &Trace,
    paging: Paging,
) -> Result<Accounting> {
    let accounting = Accounting::of(
        device_memory()?.1,
        utilization,
        weight_tier_bytes(trace, weights, extra_resident)?,
        one_slot_bytes(trace, paging)?,
    );
    accounting.admit()?;
    Ok(accounting)
}

/// The bytes this load's weight tier puts on the device: every plane the
/// trace declares, or the `[model] device_weight_budget` stated under it,
/// plus what else rides along resident.
pub fn weight_tier_bytes(trace: &Trace, stated: u64, extra_resident: u64) -> Result<u64> {
    let full: u64 = crate::weights::plane_bytes(trace)?
        .iter()
        .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
        .sum();
    Ok(match stated {
        0 => full,
        stated => stated.min(full),
    }
    .saturating_add(extra_resident))
}

/// The device's free and total bytes, as the driver reports them now.
pub fn device_memory() -> Result<(u64, u64)> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        crate::device::ctx::check("cudaMemGetInfo", asked)?;
        Ok((free as u64, total as u64))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Kv {
        space: u32,
        dtype: Dtype,
        keys_width: u64,
        values_width: u64,
        values_plane: usize,
        head_stride: u64,
    },
    State {
        stride: u64,
        dtype: Dtype,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct SpaceSeat {
    pub page_indptr: Tensor,
    pub page_indices: Tensor,
    pub last_page_lens: Tensor,
    pub row_valid: Tensor,
}

#[derive(Debug, Clone)]
pub struct Seats {
    pub lanes: u32,
    pub rows: u32,
    pub pages: u32,
    pub spaces: Vec<SpaceSeat>,
    pub slot_ids: Tensor,
    pub write_state: bool,
    pub write_state_mask: Tensor,
    pub commit_len: Tensor,
    pub begin_at: Tensor,
}

impl Seats {
    #[must_use]
    pub fn rs(mut self, write_state: bool, mask: Tensor, commit_len: Tensor) -> Seats {
        self.write_state = write_state;
        self.write_state_mask = mask;
        self.commit_len = commit_len;
        self
    }

    #[must_use]
    pub fn splitting(mut self, boundary: Tensor) -> Seats {
        self.begin_at = boundary;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub src_page: u32,
    pub src_token: u32,
    pub dst_page: u32,
    pub dst_token: u32,
    pub tokens: u32,
}

#[derive(Debug)]
pub struct Pools {
    pool: PhysicalPool,
    rows: Vec<Vec<Arena>>,
    shapes: Vec<Shape>,
    pooled: Vec<PoolRow>,
    paging: Paging,
    asked: u64,
    ceiling: u64,
    airborne: Option<Airborne>,
    committed_kv_pages: u32,
    committed_state_slots: u32,
    backed_pages: Vec<u64>,
}

impl Pools {
    pub fn reserve(
        device: i32,
        utilization: f64,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
    ) -> Result<Pools> {
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
                        head_stride: restated.map_or(keys_width, |seat| u64::from(seat.head_dim)),
                    });
                }
                CacheRow::State { name, slab, dtype } => {
                    let stride: u64 = slab.iter().product();
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, *dtype)?;
                    rows.push(vec![Arena::reserve(
                        &pool,
                        bytes,
                        "bytes of a recurrent slab",
                    )?]);
                    shapes.push(Shape::State {
                        stride,
                        dtype: *dtype,
                    });
                }
            }
            debug_assert_eq!(rows.len(), index + 1, "one arena set per cache row");
        }
        let mut pooled = Vec::new();
        for (space, width) in pooled_spaces(trace) {
            let cells = paging.pages() * u64::from(paging.page_size);
            let bytes = cells * width * u64::from(elem_size(POOL_STATE));
            let planes = (0..2)
                .map(|_| Arena::reserve(&pool, bytes, "bytes of a compressor state slab"))
                .collect::<Result<Vec<Arena>>>()?;
            pooled.push(PoolRow {
                space,
                width,
                planes,
            });
        }
        Ok(Pools {
            pool,
            rows,
            shapes,
            pooled,
            paging,
            asked: paging.pages(),
            ceiling: paging.pages(),
            airborne: None,
            committed_kv_pages: 0,
            committed_state_slots: 0,
            backed_pages: Vec::new(),
        })
    }

    pub fn watch(&mut self, airborne: Airborne) {
        self.airborne = Some(airborne);
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    fn arenas(&self) -> impl Iterator<Item = &Arena> {
        self.rows
            .iter()
            .flatten()
            .chain(self.pooled.iter().flat_map(|row| row.planes.iter()))
    }

    #[must_use]
    pub fn pool_slabs(&self) -> Vec<(u32, PoolSlabs)> {
        let cells = self.paging.pages() * u64::from(self.paging.page_size);
        self.pooled
            .iter()
            .map(|row| {
                let plane = |at: usize| {
                    Tensor::new(
                        row.planes.get(at).map_or(0, elastic::Arena::base),
                        narrow(cells) as u32,
                        narrow(row.width) as u32,
                        POOL_STATE,
                    )
                };
                (
                    row.space,
                    PoolSlabs {
                        state_kv: plane(0),
                        state_score: plane(1),
                        ape: Tensor::ABSENT,
                    },
                )
            })
            .collect()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.arenas().map(elastic::Arena::max_bytes).sum()
    }

    #[must_use]
    pub fn declared_bytes(&self) -> u64 {
        self.declared_at(self.paging.pages())
    }

    /// The watermark the cache rows reach with `pages` kv pages declared and
    /// the paging's state slots, as the map units their arenas back.
    #[must_use]
    pub fn declared_at(&self, pages: u64) -> u64 {
        self.declared_by(pages, self.paging.slots, |arena, _| arena.map_unit())
    }

    /// The same watermark were the planes re-reserved for `pages`: each in
    /// the unit an arena of that size earns, which for a small pool is the
    /// driver's granularity rather than the handle a huge one maps in.
    fn declared_reshaped(&self, pages: u64, slots: u32) -> u64 {
        let (granularity, handle) = (self.pool.granularity(), self.pool.map_unit_bytes());
        self.declared_by(pages, slots, |_, bytes| {
            elastic::map_unit_for(bytes, granularity, handle)
        })
    }

    fn declared_by(&self, pages: u64, slots: u32, unit_of: impl Fn(&Arena, u64) -> u64) -> u64 {
        let pages = u32::try_from(pages).unwrap_or(u32::MAX);
        let page_size = self.paging.page_size;
        let mapped = |bytes: u64, arena: &Arena| {
            let unit = unit_of(arena, bytes).max(1);
            bytes.div_ceil(unit).saturating_mul(unit)
        };
        let rows: u64 = self
            .rows
            .iter()
            .zip(self.shapes.iter())
            .map(|(planes, shape)| {
                planes
                    .iter()
                    .enumerate()
                    .map(|(at, arena)| {
                        mapped(watermark_bytes(shape, at, pages, slots, page_size), arena)
                    })
                    .sum::<u64>()
            })
            .sum();
        let pooled: u64 = self
            .pooled
            .iter()
            .map(|row| {
                let bytes = row.watermark_bytes(pages, page_size);
                row.planes
                    .iter()
                    .map(|arena| mapped(bytes, arena))
                    .sum::<u64>()
            })
            .sum();
        rows.saturating_add(pooled)
    }

    /// Re-reserves the kv planes and compressor slabs for a pool of `pages`,
    /// so each maps in the unit its size earns rather than the one the asked
    /// count gave it. Only before anything is committed to them: a mapped
    /// unit has a base a table may already hold.
    fn reshape(&mut self, pages: u64, slots: u32) -> Result<()> {
        let narrow = u32::try_from(pages).unwrap_or(u32::MAX);
        let page_size = self.paging.page_size;
        let fresh = |arena: &Arena| {
            if arena.committed_bytes() == 0 {
                Ok(())
            } else {
                Err(Fault::program(
                    "store::reshape",
                    "a cache plane reshaped under committed pages",
                ))
            }
        };
        for (planes, shape) in self.rows.iter_mut().zip(self.shapes.iter()) {
            let (label, changed) = match shape {
                Shape::Kv { .. } => ("bytes of a kv plane", true),
                Shape::State { .. } => ("bytes of a recurrent slab", slots != self.paging.slots),
            };
            if !changed {
                continue;
            }
            for (at, arena) in planes.iter_mut().enumerate() {
                fresh(arena)?;
                *arena = Arena::reserve(
                    &self.pool,
                    watermark_bytes(shape, at, narrow, slots, page_size),
                    label,
                )?;
            }
        }
        for row in &mut self.pooled {
            let bytes = row.watermark_bytes(narrow, page_size);
            for arena in &mut row.planes {
                fresh(arena)?;
                *arena = Arena::reserve(&self.pool, bytes, "bytes of a compressor state slab")?;
            }
        }
        self.ceiling = pages;
        self.paging.pages = pages;
        self.paging.slots = slots;
        Ok(())
    }

    #[must_use]
    pub fn map_unit_bytes(&self) -> u64 {
        self.pool.map_unit_bytes()
    }

    pub fn spare_bytes(&self) -> Result<u64> {
        let reserve = self
            .declared_bytes()
            .max(self.high_water_bytes())
            .saturating_sub(self.committed_bytes());
        self.pool.spare_bytes(reserve)
    }

    pub fn spare_bytes_for_bodies(&self, room: u64) -> Result<u64> {
        let live = self.pool.spare_bytes(0)?;
        let reserve = self
            .declared_bytes()
            .max(self.high_water_bytes())
            .saturating_sub(self.committed_bytes());
        Ok(live.saturating_sub(reserve).max(live.min(room)))
    }

    /// Sizes the declared pool to the card. The page count arrives from the
    /// operator's config (or its 256-slot default) knowing nothing of the
    /// device, and the watermark it declares is what `spare_bytes` holds back
    /// and what the runtime admits frames against, so a pool declared past
    /// what the card can back turns every growth into a refusal the runtime's
    /// static admission has already forbidden. The fit is against the elastic
    /// budget as it stands now, so it is called once everything else this
    /// load puts on the device is resident, and again after arming, when the
    /// bodies have taken what they take and the rest is the cache rows' to
    /// declare. `programs` is held for the guests' programs throughout;
    /// `bodies` is the configured allowance, which `bodies_allowance` holds
    /// down to what the room affords, and both yield before one sequence at
    /// the declared context does, since a pool under one sequence is no
    /// deployment. While nothing is committed to the planes, a pool the
    /// asked count's map unit would leave under a sequence is re-reserved
    /// in the finer unit its own size earns. `resident` spells what is on
    /// the device already, for the refusal when not even that sequence fits.
    pub fn fit_the_card(
        &mut self,
        bodies: u64,
        programs: u64,
        resident: &dyn core::fmt::Display,
    ) -> Result<Fitted> {
        let live = self
            .pool
            .spare_bytes(0)?
            .saturating_add(self.committed_bytes());
        let fresh = self
            .rows
            .iter()
            .flatten()
            .chain(self.pooled.iter().flat_map(|row| row.planes.iter()))
            .all(|arena| arena.committed_bytes() == 0);
        let one_slot = u64::from(self.paging.pages_per_slot).min(self.ceiling);
        let mut need = if fresh {
            self.declared_at(one_slot)
                .min(self.declared_reshaped(one_slot, self.paging.slots))
        } else {
            self.declared_at(one_slot)
        };
        let configured = bodies;
        let asked_programs = programs;
        let (mut bodies, mut programs) =
            holds_within(configured, asked_programs, live.saturating_sub(need));
        let mut held = programs.saturating_add(bodies);
        let mut room = live.saturating_sub(held);
        let mut fit = pages_within(self.ceiling, room, |pages| self.declared_at(pages));
        if fresh && fit < self.ceiling {
            let slots = self.paging.slots;
            let fine = pages_within(self.ceiling, room, |pages| {
                self.declared_reshaped(pages, slots)
            });
            if fine > fit {
                self.reshape(fine, slots)?;
                fit = fine;
            }
        }
        // A hybrid declares a recurrent slab for every state slot, whatever
        // the pool seats, and on a short card that slab alone can be the
        // refusal. The runtime seats a lane on two slots, so two is the
        // least; between that and the count asked, the slots and the kv
        // seats are brought level, each giving the other its room.
        if fit < one_slot && fresh && self.has_state() {
            let mut slots = self.paging.slots;
            let mut pages = fit;
            for _ in 0..8 {
                let fewer = pages_within(u64::from(slots), room, |slots| {
                    self.declared_reshaped(one_slot, u32::try_from(slots).unwrap_or(u32::MAX))
                });
                if fewer < 2 {
                    break;
                }
                slots = u32::try_from(fewer).unwrap_or(u32::MAX);
                pages = pages_within(self.ceiling, room, |pages| {
                    self.declared_reshaped(pages, slots)
                });
                let seated = u32::try_from(pages / one_slot.max(1)).unwrap_or(u32::MAX);
                if slots <= seated.saturating_mul(2).max(2) {
                    break;
                }
                slots = seated.saturating_mul(2).max(2);
            }
            if pages >= one_slot && slots < self.paging.slots {
                self.reshape(pages, slots)?;
                need = self.declared_at(one_slot);
                // The slab the slots gave back is room the bodies were
                // refused when the allowance was struck; strike it again.
                (bodies, programs) =
                    holds_within(configured, asked_programs, live.saturating_sub(need));
                held = programs.saturating_add(bodies);
                room = live.saturating_sub(held);
                fit = pages_within(self.ceiling, room, |pages| self.declared_at(pages));
            }
        }
        if fit < one_slot {
            return Err(Fault::Residency(format!(
                "the card does not hold this deployment: {resident}; under `[engine] \
                 gpu_mem_utilization` that leaves {room} bytes for the cache rows, and one \
                 sequence at the declared context, beside the declared state slots, needs \
                 {need} across them. Lower `[engine] max_model_len` or `[engine] \
                 max_state_slots`, raise `[engine] gpu_mem_utilization`, state a `[model] \
                 device_weight_budget` that streams the weight tier down, or free what \
                 else holds the device."
            )));
        }
        self.paging.pages = fit;
        // The runtime admits frames against this watermark, so the pool is
        // pledged it beside what its other arenas hold now: a frame inside
        // it is a frame the fit already sized the card for, and what a fire
        // allocates beside the pool later is not its refusal to answer.
        let others = self
            .pool
            .committed_pages()
            .saturating_sub(elastic::pages_for_bytes(self.committed_bytes()));
        self.pool
            .pledge(elastic::pages_for_bytes(self.declared_at(fit)).saturating_add(others));
        Ok(Fitted {
            asked: self.asked,
            fit,
            room,
            held,
            bodies,
            spare: live.saturating_sub(need).saturating_sub(programs),
            slots: self.paging.slots,
        })
    }

    pub(crate) fn reserve_arena(&self, max_bytes: u64, label: &'static str) -> Result<Arena> {
        Arena::reserve(&self.pool, max_bytes, label)
    }

    pub(crate) fn commit_arena(&mut self, arena: &mut Arena, bytes: u64) -> Result<()> {
        let mut targets = [elastic::Target {
            arena,
            want: elastic::Want::Prefix(bytes),
        }];
        match elastic::commit_atomically(&mut self.pool, &mut targets)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.arenas().map(elastic::Arena::committed_bytes).sum()
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.arenas().map(elastic::Arena::high_water_bytes).sum()
    }

    #[must_use]
    pub fn elastic_page_bytes(&self) -> u64 {
        self.pool.page_bytes()
    }

    #[must_use]
    pub fn elastic_budget_pages(&self) -> u64 {
        self.pool.hard_pages()
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match *shape {
                Shape::State { stride, dtype } => stride * u64::from(elem_size(dtype)),
                Shape::Kv { .. } => 0,
            })
            .sum()
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::State { .. }))
    }

    #[must_use]
    pub fn committed_watermarks(&self) -> (u32, u32) {
        (self.committed_kv_pages, self.committed_state_slots)
    }

    #[must_use]
    pub fn bases(&self) -> Vec<u64> {
        self.arenas().map(elastic::Arena::base).collect()
    }

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
                    let plane = |at: usize, width: u64| {
                        Tensor::new(
                            planes.get(at).map_or(0, elastic::Arena::base),
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    CachePool::Kv {
                        space,
                        pool: KvPool {
                            keys: plane(0, keys_width),
                            values: plane(values_plane, values_width),
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
                    fused_decay: false,
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

    pub fn clear(&mut self, slot: u32) -> Result<()> {
        self.zero_slot(None, slot)
    }

    pub fn clear_on(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<()> {
        self.zero_slot(Some(stream), slot)
    }

    pub fn copy_slot(&mut self, stream: *mut core::ffi::c_void, src: u32, dst: u32) -> Result<()> {
        for slot in [src, dst] {
            if !slab_row(self.has_state(), self.paging.slots, slot)? {
                return Ok(());
            }
        }
        if src == dst {
            return Ok(());
        }
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

    pub fn copy_kv(&mut self, stream: *mut core::ffi::c_void, moves: &[Move]) -> Result<()> {
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
                let width = if at == 0 { keys_width } else { values_width };
                let cell = width * element;
                for span in moves {
                    if span.tokens == 0 {
                        continue;
                    }
                    let bytes = u64::from(span.tokens) * cell;
                    let src =
                        (u64::from(span.src_page) * page_size + u64::from(span.src_token)) * cell;
                    let dst =
                        (u64::from(span.dst_page) * page_size + u64::from(span.dst_token)) * cell;
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

    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        if !slab_row(self.has_state(), self.paging.slots, slot)? {
            return Ok(Vec::new());
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
            crate::device::copy_d2h(arena.span(u64::from(slot) * bytes, bytes)?, &mut out[at..])?;
        }
        Ok(out)
    }

    fn zero_slot(&mut self, stream: Option<*mut core::ffi::c_void>, slot: u32) -> Result<()> {
        if !slab_row(self.has_state(), self.paging.slots, slot)? {
            return Ok(());
        }
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

    fn ensure_state(&mut self, slots: u32) -> Result<()> {
        match self.commit_to(0, slots)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    fn commit_to(&mut self, kv_pages: u32, state_slots: u32) -> Result<Commit> {
        self.commit_ranges(kv_pages, state_slots, None)
    }

    pub fn commit_frame(
        &mut self,
        demand: engine::frame::Demand,
        kv_ranges: &[(u64, u64)],
    ) -> Result<()> {
        if demand.kv_pages <= self.committed_kv_pages
            && demand.state_slots <= self.committed_state_slots
            && kv_ranges
                .iter()
                .all(|&(first, count)| self.pages_backed(first, count))
        {
            return Ok(());
        }
        match self.commit_ranges(demand.kv_pages, demand.state_slots, Some(kv_ranges))? {
            Commit::Committed => {
                self.note_backed(kv_ranges);
                Ok(())
            }
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    fn pages_backed(&self, first: u64, count: u64) -> bool {
        (first..first.saturating_add(count)).all(|page| {
            self.backed_pages
                .get((page / 64) as usize)
                .is_some_and(|word| word & (1u64 << (page % 64)) != 0)
        })
    }

    fn note_backed(&mut self, ranges: &[(u64, u64)]) {
        for &(first, count) in ranges {
            for page in first..first.saturating_add(count) {
                let (word, bit) = ((page / 64) as usize, page % 64);
                if self.backed_pages.len() <= word {
                    self.backed_pages.resize(word + 1, 0);
                }
                self.backed_pages[word] |= 1u64 << bit;
            }
        }
    }

    fn commit_ranges(
        &mut self,
        kv_pages: u32,
        state_slots: u32,
        kv_ranges: Option<&[(u64, u64)]>,
    ) -> Result<Commit> {
        let kv_pages = kv_pages.max(self.committed_kv_pages);
        let state_slots = state_slots.max(self.committed_state_slots);
        let page_size = self.paging.page_size;
        let Pools {
            pool,
            rows,
            shapes,
            pooled,
            ..
        } = self;
        let mut targets = Vec::new();
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            for (at, arena) in planes.iter_mut().enumerate() {
                let want = match (shape, kv_ranges) {
                    (Shape::Kv { .. }, Some(ranges)) => {
                        let page_bytes = watermark_bytes(shape, at, 1, 0, page_size);
                        elastic::Want::Ranges(
                            ranges
                                .iter()
                                .map(|&(first, count)| (first * page_bytes, count * page_bytes))
                                .collect(),
                        )
                    }
                    _ => elastic::Want::Prefix(watermark_bytes(
                        shape,
                        at,
                        kv_pages,
                        state_slots,
                        page_size,
                    )),
                };
                targets.push(elastic::Target { arena, want });
            }
        }
        for row in pooled.iter_mut() {
            let bytes = row.watermark_bytes(kv_pages, page_size);
            for arena in row.planes.iter_mut() {
                targets.push(elastic::Target {
                    arena,
                    want: elastic::Want::Prefix(bytes),
                });
            }
        }
        let outcome = elastic::commit_atomically(pool, &mut targets)?;
        if outcome == Commit::Committed {
            self.committed_kv_pages = kv_pages;
            self.committed_state_slots = state_slots;
        }
        Ok(outcome)
    }

    fn release_to(&mut self, kv_pages: u32, state_slots: u32) {
        let page_size = self.paging.page_size;
        let Pools {
            pool,
            rows,
            shapes,
            pooled,
            ..
        } = self;
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            for (at, arena) in planes.iter_mut().enumerate() {
                let bytes = watermark_bytes(shape, at, kv_pages, state_slots, page_size);
                arena.release_tail(pool, bytes);
            }
        }
        for row in pooled.iter_mut() {
            let bytes = row.watermark_bytes(kv_pages, page_size);
            for arena in row.planes.iter_mut() {
                arena.release_tail(pool, bytes);
            }
        }
        self.committed_kv_pages = kv_pages;
        self.committed_state_slots = state_slots;
        self.backed_pages.clear();
    }
}

const POOL_STATE: Dtype = Dtype::Bf16;

#[derive(Debug)]
struct PoolRow {
    space: u32,
    width: u64,
    planes: Vec<Arena>,
}

impl PoolRow {
    fn watermark_bytes(&self, kv_pages: u32, page_size: u32) -> u64 {
        u64::from(kv_pages) * u64::from(page_size) * self.width * u64::from(elem_size(POOL_STATE))
    }
}

const fn compressor_coff(ratio: u32) -> u64 {
    if ratio == 4 { 2 } else { 1 }
}

fn pooled_spaces(trace: &Trace) -> Vec<(u32, u64)> {
    let mut spaces: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        let Operation::Attention(Attention::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        }) = &node.op
        else {
            continue;
        };
        let Some(Def::Cache(space)) = trace.values.get(pages.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let width = compressor_coff(*ratio) * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match spaces.iter_mut().find(|(held, _)| *held == *space) {
            Some((_, held)) => *held = (*held).max(width),
            None => spaces.push((*space, width)),
        }
    }
    spaces
}

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

/// Whether a slab holds a row for the seat. The runtime seats a kv-only
/// model with no ceiling, so the slot count only binds a model with a slab.
fn slab_row(has_state: bool, slots: u32, slot: u32) -> Result<bool> {
    if !has_state {
        return Ok(false);
    }
    if slot >= slots {
        return Err(Fault::Ceiling {
            what: "recurrent slots",
            need: u64::from(slot) + 1,
            have: u64::from(slots),
        });
    }
    Ok(true)
}

fn refuse(pool: &PhysicalPool, outcome: Commit) -> Fault {
    let page = pool.page_bytes();
    match outcome {
        Commit::Committed => Fault::program(
            "store::commit",
            "a committed outcome reached the refusal path".to_string(),
        ),
        Commit::Exhausted { growth, spare } => Fault::OutOfMemory {
            need: growth.saturating_mul(page),
            have: spare.saturating_mul(page),
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

fn wide(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

impl engine::frame::Supply for Pools {
    type Error = Fault;

    fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        if let Some(highest) = demand.state_slots.checked_sub(1) {
            slab_row(self.has_state(), self.paging.slots, highest)?;
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

    fn trim(&mut self, hint: engine::frame::Demand) {
        let idle = self
            .airborne
            .as_ref()
            .is_none_or(|counts| counts.count() == 0);
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

const TRIM_HYSTERESIS_SHIFT: u32 = 3;

impl Pools {
    fn trim_is_worth_the_unmap(&self, hint: engine::frame::Demand) -> bool {
        let budget = self.pool.budget_pages();
        let free = budget.saturating_sub(self.pool.committed_pages());
        if free <= budget >> TRIM_HYSTERESIS_SHIFT {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_every_case() {
        a_seat_past_the_slot_count_is_a_row_only_a_slab_can_refuse();
    }

    // gemma-4 (no slab) with 256 slots reserved seated a lane at 312.
    fn a_seat_past_the_slot_count_is_a_row_only_a_slab_can_refuse() {
        assert!(!slab_row(false, 256, 312).expect("no slab, no ceiling"));
        assert!(slab_row(true, 256, 255).expect("the last row"));
        assert!(matches!(
            slab_row(true, 256, 312),
            Err(Fault::Ceiling {
                what: "recurrent slots",
                need: 313,
                have: 256
            })
        ));
    }
}
