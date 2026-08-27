//! The model-state bytes: paged kv pages and recurrent slabs, one allocation
//! per cache row, alive for the model's whole load.
//!
//! **THE GEOMETRY IS NOT HERE.** Where a lane's pages are and which cell a
//! token lands in is [`kv`]'s arithmetic, backend-neutral and host-tested;
//! this module owns only what that arithmetic cannot: `cudaMalloc`, and the
//! [`KvPool`]/[`RecurrentPool`] rows the dispatch arms resolve a cache id to.
//! The split is design §6's `driver::store` / shell `store/` line, drawn here
//! ahead of the module that will hold the first half.
//!
//! # Sizes come off the plan, not off a config
//!
//! `CacheRow::Kv { row, dtype }` states a per-token row and its element, and
//! `CacheRow::State { slab }` states a per-lane bank — so a pool's bytes are
//! the plan's declaration times the deployment's budget, and there is no
//! second place where a head count could disagree with the model text. The
//! one fact the IR genuinely does not carry is the recurrent element: the ssm
//! entries instantiate their state at `state_bf16`, so the slabs are bf16 and
//! this file is where that is written down.
//!
//! # k and v share a row and not a buffer
//!
//! `row[0]` is the k|v plane count and the rest is the row proper. The append
//! kernel writes two independent page arrays, so one allocation per cache row
//! is cut in half: keys in the front, values behind them. Cutting it here
//! rather than allocating twice keeps a layer's kv contiguous, which is what
//! the page addressing assumes.

pub mod kv;

use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use model_ir::{CacheRow, Dtype, Plan};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Paging, SpaceFacts};

/// The element the ssm entries instantiate their recurrent state at.
///
/// Stated, not declared: `CacheRow::State` carries a slab shape and no dtype,
/// and `kernels/attn/ssm.cuh` fixes the type at `__nv_bfloat16` in every
/// state-taking instantiation. A shell that guessed wider would allocate
/// twice the bytes and read every scan's history at half stride.
const STATE_DTYPE: Dtype = Dtype::Bf16;

/// The page layout enumerator this shell writes and the entries read: NHD,
/// `[page][token][head][dim]`.
///
/// One layout rather than a choice, because the strides that spell it are
/// cross-checked: `kv::head_split` reads `head_stride` as the head width
/// under NHD and `seq_stride` under HND, so a shell that set the pair for one
/// layout and the enumerator for the other would have every append refused —
/// or, worse, accepted at the wrong head count.
const NHD: i32 = 0;

/// How one cache row is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A paged kv space: which geometry space it belongs to, and the row the
    /// pages are cut at.
    Kv {
        space: u32,
        head_dim: u32,
        kv_heads: u32,
        dtype: Dtype,
        /// Bytes from the front of the allocation to the value pages.
        values_at: u64,
    },
    /// A recurrent slab: elements per slot.
    State { stride: u64 },
}

/// The per-fire handles a pool row borrows: the geometry vectors this fire
/// wrote, and the graph-padding mask beside them.
///
/// A `KvPool` is a storage pointer plus the tables that address it, and the
/// tables are fire data — so the [`CacheTable`] is rebuilt each fire out of
/// long-lived storage and short-lived geometry. Rebuilding 42 rows of `Copy`
/// structs is arithmetic, not allocation.
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
}

/// Every cache space's bytes, one allocation per row.
#[derive(Debug)]
pub struct Pools {
    slabs: Vec<Buffer>,
    shapes: Vec<Shape>,
    paging: Paging,
}

impl Pools {
    /// Reserve the pools one plan needs at one deployment's budget.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a cache row this shell cannot size — a kv row
    /// whose space no attention op reads, or one whose declared row is not
    /// `k|v` by a head-multiple — and [`Fault::Device`] for the allocations.
    pub fn reserve(plan: &Plan, paging: Paging, facts: &[Option<SpaceFacts>]) -> Result<Pools> {
        let mut slabs = Vec::with_capacity(plan.caches.len());
        let mut shapes = Vec::with_capacity(plan.caches.len());

        for (index, row) in plan.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    row,
                    dtype,
                    space,
                } => {
                    let seat = facts
                        .get(*space as usize)
                        .copied()
                        .flatten()
                        .ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "cache `{name}` in geometry space {space}, which no \
                                 attention op reads — so nothing states its heads"
                            ),
                        })?;
                    let (planes, width) = split(name, row)?;
                    let heads = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                    if heads != width {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache `{name}`, whose row is {width} wide while its \
                                 consumers state {} heads of {}",
                                seat.kv_heads, seat.head_dim
                            ),
                        });
                    }
                    let element = elem_bytes(name, *dtype)?;
                    let plane = paging.pages() * u64::from(paging.page_size) * width * element;
                    slabs.push(Buffer::zeroed(usize::try_from(plane * planes).unwrap_or(usize::MAX))?);
                    shapes.push(Shape::Kv {
                        space: *space,
                        head_dim: seat.head_dim,
                        kv_heads: seat.kv_heads,
                        dtype: *dtype,
                        values_at: plane,
                    });
                }
                CacheRow::State { name, slab } => {
                    let stride: u64 = slab.iter().product();
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, STATE_DTYPE)?;
                    slabs.push(Buffer::zeroed(usize::try_from(bytes).unwrap_or(usize::MAX))?);
                    shapes.push(Shape::State { stride });
                }
            }
            debug_assert_eq!(slabs.len(), index + 1, "one allocation per cache row");
        }
        Ok(Pools {
            slabs,
            shapes,
            paging,
        })
    }

    /// How the pages are handed out.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// Every byte these pools hold.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slabs.iter().map(|s| s.bytes() as u64).sum()
    }

    /// The cache table one fire resolves its cache ids through.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a kv row whose space this fire seated no
    /// geometry for.
    pub fn table(&self, seats: &Seats) -> Result<CacheTable> {
        let mut rows = Vec::with_capacity(self.shapes.len());
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            rows.push(match *shape {
                Shape::Kv {
                    space,
                    head_dim,
                    kv_heads,
                    dtype,
                    values_at,
                } => {
                    let seat =
                        seats
                            .spaces
                            .get(space as usize)
                            .ok_or_else(|| Fault::Unbound {
                                what: format!(
                                    "cache space {space}, for which this fire wrote no \
                                     geometry"
                                ),
                            })?;
                    let cells = self.paging.pages() * u64::from(self.paging.page_size);
                    let plane = |at: u64| {
                        Tensor::new(
                            slab.ptr() + at,
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            kv_heads * head_dim,
                            dtype,
                        )
                    };
                    CachePool::Kv(KvPool {
                        keys: plane(0),
                        values: plane(values_at),
                        // The shadow, scale and envelope planes belong to the
                        // quantized schemes; a native pool binds none and the
                        // entries never reach for them (`kv::native_bf16`).
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
                        // NHD: one token row is `kv_heads * head_dim`
                        // elements and one head plane is `head_dim`. The pair
                        // is what `kv::head_split` reads the head width back
                        // out of.
                        seq_stride: i64::from(kv_heads) * i64::from(head_dim),
                        head_stride: i64::from(head_dim),
                        layout: NHD,
                        scheme_byte: 0,
                        block_size: 0,
                        max_pages_per_request: narrow(u64::from(self.paging.pages_per_slot)),
                        pages_in_batch: narrow(u64::from(seats.pages)),
                    })
                }
                Shape::State { stride } => CachePool::Recurrent(RecurrentPool {
                    // One row serves both seats. A `CacheRow::State` is one
                    // slab and the ops that read it name it once — the
                    // gated-delta scan through `slab`, the causal convolution
                    // through `conv_slab` — so pointing both at this row's
                    // bytes is what makes `conv.L` and `delta.L` two
                    // independent spaces rather than two halves of one.
                    slab: Tensor::new(
                        slab.ptr(),
                        self.paging.slots,
                        narrow(stride) as u32,
                        STATE_DTYPE,
                    ),
                    slot_ids: seats.slot_ids,
                    slot_stride_elems: stride as i64,
                    conv_slab: Tensor::new(
                        slab.ptr(),
                        self.paging.slots,
                        narrow(stride) as u32,
                        STATE_DTYPE,
                    ),
                    conv_stride: stride as i64,
                }),
            });
        }
        Ok(CacheTable(rows))
    }

    /// Clear one slot's recurrent state.
    ///
    /// A RECURRENT SLOT IS ITS HISTORY, so opening a sequence in a slot
    /// another sequence used means zeroing what that one left — unlike a kv
    /// page, which is overwritten before it is read because `kv_len` says
    /// nothing lives past the append. There is no cheaper truth available:
    /// the scan reads the whole bank on its first step.
    ///
    /// Called from [`Shell::open`](crate::serve::Shell::open) by a caller
    /// whose page table is the shell's, and from the fire path for one whose
    /// page table is its own — there, a lane stating `held == 0` IS the
    /// sequence beginning. One `cudaMemset` per sequence either way.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the
    /// fill.
    pub fn clear(&mut self, slot: u32) -> Result<()> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        let element = u64::from(elem_size(STATE_DTYPE));
        for (slab, shape) in self.slabs.iter_mut().zip(&self.shapes) {
            let Shape::State { stride } = *shape else {
                continue;
            };
            let bytes = stride * element;
            slab.zero_span(u64::from(slot) * bytes, usize::try_from(bytes).unwrap_or(0))?;
        }
        Ok(())
    }
}

/// A kv row's `(planes, width)`: the leading dim is the k|v plane count and
/// the rest is one plane's row.
fn split(name: &str, row: &[u64]) -> Result<(u64, u64)> {
    let (planes, rest) = row.split_first().ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, which declares a row of no dims at all"),
    })?;
    if *planes != 2 {
        return Err(Fault::Unbound {
            what: format!(
                "cache `{name}`, whose row leads with {planes} planes — this shell \
                 cuts kv pages into a key half and a value half, and knows no third"
            ),
        });
    }
    Ok((*planes, rest.iter().product()))
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
