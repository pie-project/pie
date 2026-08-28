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
//! `CacheRow::Kv { planes, dtype }` states the planes one token's entry is
//! written as, each by its own per-token width, and `CacheRow::State { slab }`
//! states a per-lane bank — so a pool's bytes are the plan's declaration times
//! the deployment's budget, and there is no second place where a head count
//! could disagree with the model text. The one fact the IR genuinely does not
//! carry is the recurrent element: the ssm entries instantiate their state at
//! `state_bf16`, so the slabs are bf16 and this file is where that is written
//! down.
//!
//! # A row's planes share one allocation, and which is k and which is v
//!
//! One allocation per cache row, cut into the planes that row declares: plane
//! `i` begins `Σ_{j<i} pages · page_size · planes[j] · element` bytes from the
//! front. Cutting one allocation rather than allocating per plane keeps a
//! layer's kv contiguous, which is what the page addressing assumes.
//!
//! Which plane an entry reaches for is the plane COUNT, because the pool hands
//! out exactly two names:
//!
//! ```text
//! [w]        keys = values = plane 0      one plane, addressed as both
//! [w0, w1]   keys = plane 0, values = plane 1, and the widths may differ
//! ```
//!
//! `[w, w]` is the ordinary key|value pair. `[w]` is a row whose single plane
//! every reader walks through `pool.keys`: `attention.kv_append_shared` writes
//! the one rectangle to `keys` and to `values` alike
//! (`kernels-cuda/src/attn.rs`, `kv_append_shared`), and an indexer's keys and
//! a pooled cache's entries are written and read through `pool.keys` alone
//! (`attn/index.rs`, `attn/pool.rs`, both `kv_append`) — so pointing the two
//! handles at the same bytes is what makes one declared plane serve both
//! names. Two planes of DIFFERENT widths is the latent page: the mla kernels
//! take `pool.keys` as the compressed pages at `kv_lora_rank` and
//! `pool.values` as the rope pages at `rope_dim` (`attn/mla.rs`, `Layer::of`
//! and `kv_append`). Three planes or more is a refusal — this shell binds a
//! key plane and a value plane, and knows no third.

pub mod kv;

use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use model_ir::{CacheRow, Dtype, Plan};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Facts, Paging};

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
    /// A paged kv space: which geometry space it belongs to, and how the row's
    /// declared planes are handed out as the pool's key plane and value plane.
    Kv {
        space: u32,
        dtype: Dtype,
        /// Elements one token writes into the key plane — plane 0, whatever
        /// the row declared.
        keys_width: u64,
        /// Elements one token writes into the value plane: plane 1's width
        /// when the row declares two, and plane 0's again when it declares one
        /// (the same plane under both names).
        values_width: u64,
        /// Bytes from the front of the allocation to the value plane. Zero for
        /// a one-plane row, which is how the two handles come to name the same
        /// bytes.
        values_at: u64,
        /// One head plane under NHD, for the paged kernels that read a head
        /// count back out of the stride pair (`kv::head_split`). The token
        /// pitch beside it IS `keys_width` and is not written down twice.
        head_stride: u64,
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
    /// **A KV ROW IS SIZED BY ITS DECLARATION**, plane by plane at each
    /// plane's own width, and `facts` is the RESTATEMENT checked against it
    /// where a paged launch made one. A row no paged launch reads — a latent
    /// page, an indexer's keys, a pooled cache's entries — is allocated as
    /// declared and checked against nothing, because nothing else states it.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a kv row this shell cannot cut into a key plane
    /// and a value plane — one that declares no planes, or three and more —
    /// for one whose paged readers restate a width its declaration does not
    /// spell, and for an element with no size; [`Fault::Device`] for the
    /// allocations.
    ///
    /// `facts` is indexed by CACHE ROW, not by geometry space: a page id says
    /// which page, never how wide the row it addresses is, and gemma's
    /// sliding and global layers share one page-id space at two widths
    /// ([`SpaceFacts`](crate::store::kv::SpaceFacts)).
    pub fn reserve(plan: &Plan, paging: Paging, facts: &Facts) -> Result<Pools> {
        let mut slabs = Vec::with_capacity(plan.caches.len());
        let mut shapes = Vec::with_capacity(plan.caches.len());

        for (index, row) in plan.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    space,
                } => {
                    let element = elem_bytes(name, *dtype)?;
                    let cells = paging.pages() * u64::from(paging.page_size);
                    let (keys_width, values_width, values_at) = match planes.as_slice() {
                        // One plane, addressed as both k and v: the two
                        // handles name the same bytes, which is what
                        // `kv_append_shared` writes and what an index or pool
                        // reader's `keys`-only walk needs.
                        [plane] => (*plane, *plane, 0),
                        [keys, values] => (*keys, *values, cells * keys * element),
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
                    // A restatement only exists where a PAGED launch read the
                    // row and named a head count: the prefill arms state one,
                    // the decode and masked arms state a head width alone
                    // (`kv_heads` 0), and the latent, index and pool launches
                    // do not feed the row pass at all. Where it exists it must
                    // be the declaration.
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
                    let bytes = cells * planes.iter().sum::<u64>() * element;
                    slabs.push(Buffer::zeroed(
                        usize::try_from(bytes).unwrap_or(usize::MAX),
                    )?);
                    shapes.push(Shape::Kv {
                        space: *space,
                        dtype: *dtype,
                        keys_width,
                        values_width,
                        values_at,
                        // One head of the whole plane where no paged launch
                        // stated a head width: the latent, index and pool
                        // kernels take their widths from their op operands and
                        // never consult the strides.
                        head_stride: restated.map_or(keys_width, |seat| u64::from(seat.head_dim)),
                    });
                }
                CacheRow::State { name, slab } => {
                    let stride: u64 = slab.iter().product();
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, STATE_DTYPE)?;
                    slabs.push(Buffer::zeroed(
                        usize::try_from(bytes).unwrap_or(usize::MAX),
                    )?);
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
                    dtype,
                    keys_width,
                    values_width,
                    values_at,
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
                    let plane = |at: u64, width: u64| {
                        Tensor::new(
                            slab.ptr() + at,
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    CachePool::Kv { space, pool: KvPool {
                        keys: plane(0, keys_width),
                        // A one-plane row seats `values_at == 0` and the key
                        // plane's own width, so both handles are the one
                        // plane.
                        values: plane(values_at, values_width),
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
                        // NHD: one token's step through the key plane is that
                        // plane's whole width, and one head plane is a share
                        // of it. The pair is what `kv::head_split` reads the
                        // head width back out of and what `index::pool_pitch`
                        // reads the whole-row pitch out of.
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
