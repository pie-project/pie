//! The shell's scratch plane: working rectangles (`precast`, `routed`,
//! `index`, `copy`, `pool`, `ple`) reserved once at load, aliased where
//! roles are never live together and added where they must persist across a
//! region or across fires.

use kernels_metal::Tensor;
use kernels_metal::attn::ple;
use kernels_metal::linear::moe::{self, RoutedScratch};
use kernels_metal::linear::quant;
use model_compiler::{Budget, Budgets, CompiledModel, Fallback, FireRows};
use model_exec::store::arena::rect;
use model_ir::{Attention, Def, Dim, Dtype, Linear, Operation, Trace, Ty, ValueId};

use crate::store::kv::Paging;

use crate::device::{Buffer, Context, Handles};
use crate::error::Result;
use crate::run::{PoolSlabs, WeightRow, WeightTable};

/// The alignment every role and every sub-rectangle starts on — `inputs`'
/// number, for `inputs`' reason.
const ALIGN: u64 = 256;

/// Rows the split-K tile may carry: the 8 rung's, since the split is that rung's alone.
const SPLITK_ROWS: u64 = 8;
/// The most K partitions the split takes (`linear::quant::splitk`'s cap).
const SPLITK_MAX: u64 = 8;

/// One working rectangle inside the plane: where it starts, and the shape a
/// [`Tensor`] states for it.
#[derive(Clone, Copy, Debug)]
struct Room {
    at: u64,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

impl Room {
    /// A rectangle of `rows x width`, laid down at the next aligned byte of a
    /// running offset and advancing it.
    fn lay(at: &mut u64, rows: u64, width: u64, dtype: Dtype) -> Room {
        let start = at.next_multiple_of(ALIGN);
        let room = Room {
            at: start,
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype,
        };
        *at = start + room.bytes();
        room
    }

    fn bytes(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.width) * self.dtype.bytes_ceil()
    }

    /// Mint the handle this rectangle resolves through. Fire-lived, like
    /// every arena row: `Handles::rewind` drops it at the end of the fire and
    /// the reservation underneath does not move.
    fn bind(&self, handles: &Handles, plane: &Buffer) -> Result<Tensor> {
        Ok(Tensor::new(
            handles.bind(plane, self.at, self.bytes())?,
            self.rows,
            self.width,
            self.dtype,
        ))
    }
}

/// The copy role's whole slab: where it starts, and how many bytes it holds.
///
/// A byte extent rather than a [`Room`]: `crate::dispatch::copy` sub-divides
/// it per region/operand; this only states how far it may go.
#[derive(Clone, Copy, Debug)]
struct CopyRoom {
    at: u64,
    bytes: u64,
}

/// One pooled space's two state slabs, in the order they are laid down.
/// `space` is the source cache's index; two pooled layers hold different
/// state at the same paged cell and cannot share a plane.
#[derive(Clone, Copy, Debug)]
struct Pool {
    space: u32,
    state_kv: Room,
    state_score: Room,
}

/// One hashing's constants: the plane they were written into, and the numbers
/// themselves as the key a dispatch arm finds it by.
#[derive(Clone, Debug)]
struct PleHash {
    /// `[mults][primes][offsets]`, as [`kernels_metal::attn::ple::hash_constants`]
    /// lays them down; also the plane's actual on-device contents.
    key: Box<[u64]>,
    room: Room,
}

/// The sorted arm's six rectangles, in the order they are laid down.
#[derive(Clone, Copy, Debug)]
struct Routed {
    perm: Room,
    row_expert: Room,
    tile_expert: Room,
    inv: Room,
    x: Room,
    y: Room,
}

/// The reservation, its roles, and the two load-time tables a dispatch arm
/// reads beside them. A role no node asks for is `None`, not a zero-sized
/// rectangle: the caller's answer is to take the arm that needs no plane.
#[derive(Debug)]
pub struct Scratch {
    /// The one reservation, or an empty buffer for an artifact with no role
    /// at all. Zero bytes mints no handle, the state every accessor below
    /// reports as `None`.
    plane: Buffer,

    precast: Option<Room>,
    /// Split-K partials for the sparse `bm = 8` tile: `SPLITK_ROWS x SPLITK_MAX`
    /// f32 rows at the widest dense N `linear::quant::splitk` would split.
    /// `None` when no dense projection is narrow enough to split. Aliased
    /// with the roles beside it — consumed by the reduce in the same chain.
    partials: Option<Room>,
    routed: Option<Routed>,
    /// The NSA indexer's per-row score slab: `rows x max_kv` floats, `None`
    /// for an artifact with no `attention.index_topk` node. Aliased with the
    /// two above since it's dead before the next dispatch.
    index: Option<Room>,
    /// Not aliased: a copy's bytes are live across a whole region's dispatch
    /// chains, not just one. `None` for an artifact with no copyable region.
    copy: Option<CopyRoom>,

    /// The dsv4 compressor's rolling state, one entry per pooled space,
    /// empty for an artifact with no `attention.pool_gather` node.
    ///
    /// Not aliased: addressed by the source pool's paged slot rather than
    /// written fresh each chain, so it must survive across fires.
    pool: Vec<Pool>,

    /// qwen4's PLE hash constants: one `u64` plane per distinct hashing,
    /// empty for an artifact with no `attention.ple_ngram_ids` node. Written
    /// once at load, before any command buffer exists. A list because the
    /// decode and chunked arms of one PLE share a plane, looked up by the
    /// numbers themselves rather than by node.
    ple: Vec<PleHash>,

    /// Per `ValueId`: how many rows the arena slot behind it can hold — the
    /// slot's whole reservation at the budget's ceiling, not this fire's
    /// extent. [`crate::arena::capacities`] is the arithmetic.
    capacity: Vec<u32>,

    /// Per `ValueId`: the expert count of the router that produced it, or 0
    /// for a value no router named. Carried off `MoeTopk*`'s `experts` field.
    routers: Vec<u32>,
}

impl Scratch {
    /// Carve the plane this artifact can ask for, at the budget's ceiling.
    ///
    /// Ceilings are read off the same carve the arena is driven by, so this
    /// plane and the arena cannot disagree about fire size. The patch/image
    /// ceilings are stated, not zeroed: a zero-row patch ceiling would
    /// undersize gemma's vision-embedding pre-cast room.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`](crate::error::Fault::Deviceless) for a non-Apple
    /// build, [`Fault::Device`](crate::error::Fault::Device) when the device
    /// declined the length.
    pub fn reserve(
        device: &Context,
        trace: &Trace,
        weights: &WeightTable,
        compiled: &CompiledModel,
        budgets: &Budgets,
        paging: Paging,
    ) -> Result<Scratch> {
        let budget = &budgets.tokens;
        let map = &compiled.arena;
        let ceiling = FireRows {
            tokens: u64::from(budget.max_tokens),
            lanes: u64::from(budget.max_lanes),
            patches: u64::from(budgets.max_patches()),
            images: u64::from(budgets.max_images()),
        };
        let of = |id: ValueId| rect(map, id, ceiling);
        let banked = |id: ValueId| match trace.values.get(id.0 as usize).map(|v| &v.def) {
            Some(Def::Weight(w)) => matches!(
                weights.0.get(*w as usize).copied().flatten(),
                Some(WeightRow::Planes(_))
            ),
            _ => false,
        };

        let routers = routers(trace);
        let tuning = kernels_metal::tuning::current();

        // Dense and routed quantized projections, in one pass. `dense`
        // collects the (N, K) pairs the split sweep needs; the rest is
        // running maxima.
        let mut dense: Vec<(u32, u32)> = Vec::new();
        let (mut act_rows, mut act_k) = (0u64, 0u64);
        let (mut sorted, mut pairs, mut routed_k, mut routed_n) = (0u64, 0u64, 0u64, 0u64);
        for node in &trace.nodes {
            let Operation::Linear(op) = &node.op else {
                continue;
            };
            match op {
                Linear::Matmul { act, w, y } | Linear::LmHead { act, w, y } if banked(*w) => {
                    let (Some(act), Some(y)) = (of(*act), of(*y)) else {
                        continue;
                    };
                    act_rows = act_rows.max(u64::from(act.rows));
                    act_k = act_k.max(u64::from(act.width));
                    let pair = (y.width, act.width);
                    if !dense.contains(&pair) {
                        dense.push(pair);
                    }
                }
                Linear::MoeMatmulSelectBias { x, routes, y, .. }
                | Linear::MoeMatmulSelectQuant { x, routes, y, .. } => {
                    let (Some(x), Some(y)) = (of(*x), of(*y)) else {
                        continue;
                    };
                    let experts = routers.get(routes.0 as usize).copied().unwrap_or(0);
                    if experts == 0 {
                        continue;
                    }
                    // A node the sorted arm declines at the ceiling asks for
                    // nothing: a ceiling fire that won't batch means no fire
                    // will.
                    if moe::tile_rows(y.rows, experts, &tuning) <= 1 {
                        continue;
                    }
                    let rows = moe::sorted_rows(y.rows, experts, &tuning);
                    sorted = sorted.max(u64::from(rows));
                    pairs = pairs.max(u64::from(y.rows));
                    routed_k = routed_k.max(u64::from(x.width));
                    routed_n = routed_n.max(u64::from(y.width));
                }
                _ => {}
            }
        }

        // Product of the maxima, not max over nodes of `rows x width`: they
        // differ only when the widest projection is not the tallest.
        let precast = (act_rows > 0 && act_k > 0).then(|| {
            let mut at = 0u64;
            Room::lay(&mut at, act_rows, act_k, Dtype::F16)
        });
        // The split is the 8 rung's alone, so the plane is eight rows by the
        // widest split, at the widest N that splits. Group 32 is the least
        // restrictive divisibility, so the width is a ceiling.
        let partials = dense
            .iter()
            .filter(|&&(n, k)| {
                let (Ok(n), Ok(k)) = (i32::try_from(n), i32::try_from(k)) else {
                    return false;
                };
                quant::splitk(n, 8, 8, k, 32, 4) > 1
            })
            .map(|&(n, _)| u64::from(n))
            .max()
            .map(|n| {
                let mut at = 0u64;
                Room::lay(&mut at, SPLITK_ROWS * SPLITK_MAX, n, Dtype::F32)
            });
        let routed = (sorted > 0).then(|| {
            let mut at = 0u64;
            Routed {
                perm: Room::lay(&mut at, 1, sorted, Dtype::I32),
                row_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),
                // One entry per tile of the sorted stack; sized per row
                // instead, since the tile width is a fire-time choice and
                // its narrowest value is the deepest case.
                tile_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),
                inv: Room::lay(&mut at, 1, pairs, Dtype::I32),
                x: Room::lay(&mut at, sorted, routed_k, Dtype::Bf16),
                y: Room::lay(&mut at, sorted, routed_n, Dtype::Bf16),
            }
        });

        // Indexer's score slab, at the paging's own ceiling: one row per
        // query row, width `pages_per_slot * page_size`.
        let index = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Attention(Attention::IndexTopk { selection, .. }) => of(*selection),
                _ => None,
            })
            .map(|rect| u64::from(rect.rows))
            .max()
            .filter(|rows| *rows > 0)
            .and_then(|rows| {
                let keys = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
                (keys > 0).then(|| {
                    let mut at = 0u64;
                    Room::lay(&mut at, rows, keys, Dtype::F32)
                })
            });

        let union = precast
            .map_or(0, |r| r.at + r.bytes())
            .max(partials.map_or(0, |r| r.at + r.bytes()))
            .max(routed.map_or(0, |r| r.y.at + r.y.bytes()))
            .max(index.map_or(0, |r| r.at + r.bytes()));
        // Added, not unioned: starts where the three aliased roles end, so
        // a copied region holding a routed matmul is two disjoint spans.
        let copy = copy_ceiling(trace, compiled, budget).map(|bytes| CopyRoom {
            at: union.next_multiple_of(ALIGN),
            bytes,
        });

        let mut at = copy.map_or(union, |room| room.at + room.bytes);
        // Compressor state, added after copy, one plane per pooled space
        // (keyed by the gather's own `pages` operand; see the field).
        let pool: Vec<Pool> = pool_state(trace, paging)
            .into_iter()
            .map(|(space, cells, width)| Pool {
                space,
                state_kv: Room::lay(&mut at, cells, width, Dtype::Bf16),
                state_score: Room::lay(&mut at, cells, width, Dtype::Bf16),
            })
            .collect();

        // PLE constants, added last and written once; one plane per
        // distinct hashing (see the field).
        let mut ple: Vec<PleHash> = Vec::new();
        for node in &trace.nodes {
            let Operation::Attention(
                Attention::PleNgramIds {
                    mults,
                    primes,
                    offsets,
                    ..
                }
                | Attention::PleNgramIdsChunked {
                    mults,
                    primes,
                    offsets,
                    ..
                },
            ) = &node.op
            else {
                continue;
            };
            let key: Box<[u64]> = ple::hash_constants(mults, primes, offsets).into();
            if key.is_empty() || ple.iter().any(|held| held.key == key) {
                continue;
            }
            let room = Room::lay(&mut at, 1, key.len() as u64, Dtype::U64);
            ple.push(PleHash { key, room });
        }

        let mut plane = Buffer::zeroed(device, at)?;
        for hashing in &ple {
            let bytes: Vec<u8> = hashing
                .key
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            plane.write(hashing.room.at, &bytes)?;
        }

        Ok(Scratch {
            plane,
            precast,
            partials,
            routed,
            index,
            copy,
            pool,
            ple,
            capacity: crate::arena::capacities(map),
            routers,
        })
    }

    /// Every byte the plane holds — the number a footprint line prints.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.plane.bytes()
    }

    /// How many rows the arena slot behind `id` can hold.
    ///
    /// `0` for a value the arena binds no rectangle for, which is the answer
    /// that makes every guard reading it fall back rather than pad.
    #[must_use]
    pub fn capacity(&self, id: ValueId) -> u32 {
        self.capacity.get(id.0 as usize).copied().unwrap_or(0)
    }

    /// The expert count of the router that wrote `routes`, or `0` for a value
    /// no router named — see the field.
    #[must_use]
    pub fn experts(&self, routes: ValueId) -> u32 {
        self.routers.get(routes.0 as usize).copied().unwrap_or(0)
    }

    /// The FP16 staging plane, cut to `rows x contraction`. `None` when this
    /// artifact reserved none, or (unreachable in practice) the rectangle
    /// exceeds the reserved ceiling.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn precast(&self, handles: &Handles, rows: u32, contraction: u32) -> Option<Result<Tensor>> {
        let room = self.precast?;
        if u64::from(rows) * u64::from(contraction) > room.bytes() / 2 {
            return None;
        }
        Some(Room {
            rows,
            width: contraction,
            ..room
        }
        .bind(handles, &self.plane))
    }

    /// The split-K partials plane, cut to `rows x width` f32. `None` when
    /// this artifact reserved none or the rectangle exceeds it.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn partials(&self, handles: &Handles, rows: u32, width: u32) -> Option<Result<Tensor>> {
        let room = self.partials?;
        if u64::from(rows) * u64::from(width) > room.bytes() / 4 {
            return None;
        }
        Some(Room { rows, width, ..room }.bind(handles, &self.plane))
    }

    /// One rectangle of the copy role's slab, at `offset` bytes into it. The
    /// offset is the caller's; this checks the place is inside what was
    /// reserved. `None` (not a panic) for an out-of-range offset or an
    /// artifact with no copy role.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn copy(&self, handles: &Handles, offset: u64, bytes: u64) -> Option<Result<u32>> {
        let room = self.copy?;
        if offset.saturating_add(bytes) > room.bytes {
            return None;
        }
        Some(handles.bind(&self.plane, room.at + offset, bytes))
    }

    /// The sorted arm's six rectangles, minted for the node asking.
    ///
    /// These are the ceiling's rectangles, not the node's: the only shape
    /// [`moe::matmul_select_batched`] reads off them is `x.rows >=
    /// sorted_rows`, which the ceiling always satisfies. `None` when the
    /// load reserved no routed room; the caller's answer is the matvec arm.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn routed(&self, handles: &Handles) -> Option<Result<RoutedScratch>> {
        let r = self.routed?;
        let mint = || {
            Ok(RoutedScratch {
                perm: r.perm.bind(handles, &self.plane)?,
                row_expert: r.row_expert.bind(handles, &self.plane)?,
                tile_expert: r.tile_expert.bind(handles, &self.plane)?,
                inv: r.inv.bind(handles, &self.plane)?,
                x: r.x.bind(handles, &self.plane)?,
                y: r.y.bind(handles, &self.plane)?,
            })
        };
        Some(mint())
    }

    /// The NSA indexer's score slab, whole — `rows x max_kv` f32 at
    /// `score_stride` width. Handed over at its reserved extent rather than
    /// cut to the fire, since narrowing per fire would move every row.
    ///
    /// `None` for an artifact whose trace names no `attention.index_topk`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn index_scores(&self, handles: &Handles) -> Option<Result<Tensor>> {
        Some(self.index?.bind(handles, &self.plane))
    }

    /// The dsv4 compressor's two state slabs for one pooled space, whole.
    /// `space` is the source cache's index (the gather's `pages` operand);
    /// two pooled layers never share a plane.
    ///
    /// `None` for an artifact whose trace names no `attention.pool_gather`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn pool_state(&self, handles: &Handles, space: u32) -> Option<Result<PoolSlabs>> {
        let rooms = *self.pool.iter().find(|held| held.space == space)?;
        let mint = || {
            Ok(PoolSlabs {
                state_kv: rooms.state_kv.bind(handles, &self.plane)?,
                state_score: rooms.state_score.bind(handles, &self.plane)?,
            })
        };
        Some(mint())
    }

    /// The `u64` plane holding this hashing's constants, minted into the
    /// fire. Found by the numbers, not the node, since the shader reads the
    /// numbers.
    ///
    /// `None` for an artifact whose trace states no hashing with these constants.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn ple_hash(
        &self,
        handles: &Handles,
        mults: &[u64],
        primes: &[u64],
        offsets: &[u64],
    ) -> Option<Result<Tensor>> {
        let want = ple::hash_constants(mults, primes, offsets);
        let held = self.ple.iter().find(|h| *h.key == *want)?;
        Some(held.room.bind(handles, &self.plane))
    }
}

/// `2` for the overlapping window of the ratio-4 compressor, `1` otherwise —
/// mirrors `kernels_metal::attn::pool`'s private `compressor_coff`.
const fn compressor_coff(ratio: u32) -> u64 {
    if ratio == 4 { 2 } else { 1 }
}

/// The compressor state planes this artifact asks for: one
/// `(space, cells, width)` per pooled cache space, empty for a trace that
/// names no `attention.pool_gather`. Keyed by the space the gather addresses
/// (`pages` operand via `Def::Cache`); width is a max within a space since
/// two ratios can share one space.
///
/// Rows are the pool's cells, not the fire's rows, since `pool_gather_paged`
/// addresses state by a global paged slot.
fn pool_state(trace: &Trace, paging: Paging) -> Vec<(u32, u64, u64)> {
    let cells = paging.pages() * u64::from(paging.page_size);
    if cells == 0 {
        return Vec::new();
    }
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
        .into_iter()
        .map(|(space, width)| (space, cells, width))
        .collect()
}

/// The most bytes any one copied region's staging rectangles come to.
///
/// Dedup is by arena root: two values the carve folded onto one column take
/// one rectangle. `None` when nothing in this artifact can be copied.
fn copy_ceiling(trace: &Trace, compiled: &CompiledModel, budget: &Budget) -> Option<u64> {
    let lanes = u64::from(budget.max_lanes);
    let mut most = 0u64;
    for region in compiled.template() {
        let rows = compiled
            .fallback
            .rows
            .iter()
            .filter(|row| region.nodes.contains(&row.node) && row.fallback == Fallback::Copy)
            .flat_map(|row| row.buckets.clone())
            .filter_map(|bucket| budget.buckets.get(bucket as usize).copied())
            .max()
            .or_else(|| {
                // A row with no lattice behind it: one implicit bucket at the
                // budget's ceiling.
                compiled
                    .fallback
                    .rows
                    .iter()
                    .any(|row| region.nodes.contains(&row.node) && row.fallback == Fallback::Copy)
                    .then_some(budget.max_tokens)
            });
        let Some(rows) = rows else { continue };
        if !crate::window::copyable_mask(trace, compiled, &region.mask) {
            continue;
        }
        let Some((ins, outs)) = crate::window::operands(&trace.nodes, region) else {
            continue;
        };
        let mut seen: Vec<u64> = Vec::new();
        let mut at = 0u64;
        for id in ins.iter().chain(outs.iter()) {
            // Row-shaped only: everything else is handed over whole or
            // re-cut on the host, and takes no byte here.
            let Some(decl) = trace.values.get(id.0 as usize) else {
                continue;
            };
            if !matches!(
                &decl.ty,
                Ty::Tensor { shape, .. } if matches!(shape.first(), Some(Dim::Tokens))
            ) {
                continue;
            }
            let Some(rect) = rect(&compiled.arena, *id, FireRows::text_only(u64::from(rows), lanes))
            else {
                continue;
            };
            if seen.contains(&rect.offset) {
                continue;
            }
            seen.push(rect.offset);
            let row = u64::from(rect.width) * rect.dtype.bytes_ceil();
            at = at.next_multiple_of(COPY_ALIGN) + u64::from(rows) * row;
        }
        most = most.max(at);
    }
    (most > 0).then_some(most)
}

/// Every staging rectangle starts 16-byte aligned (matches
/// `engine_cuda::dispatch::copy`'s `align`). Smaller than [`ALIGN`] since
/// this divides within one role, not the start of one.
const COPY_ALIGN: u64 = 16;

/// Per `ValueId`: the expert count of the router that wrote it. One pass,
/// since the router variants share output shape and only one writes any
/// given vector.
fn routers(trace: &Trace) -> Vec<u32> {
    let mut out = vec![0u32; trace.values.len()];
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        let (routes, experts) = match op {
            Linear::MoeTopkSoftmax {
                routes, experts, ..
            }
            | Linear::MoeTopkSoftmaxScaled {
                routes, experts, ..
            }
            | Linear::MoeTopkSigmoid {
                routes, experts, ..
            }
            | Linear::MoeTopkSqrtSoftplus {
                routes, experts, ..
            }
            // The lookup router: omitted, hash-routed layers would reserve
            // nothing for the sorted arm (measured ~28% slower prefill on
            // dsv4-flash-u2g64).
            | Linear::MoeHashRoute {
                routes, experts, ..
            } => (*routes, *experts),
            _ => continue,
        };
        if let Some(seat) = out.get_mut(routes.0 as usize) {
            *seat = (*seat).max(experts);
        }
    }
    out
}
