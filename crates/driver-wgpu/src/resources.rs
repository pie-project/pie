//! The memory a plan never mentions.
//!
//! Everything else in this crate reads a statement and works out what it means.
//! This module holds what no statement can: the paged KV cache, and the tables
//! a fire assembles for the rows it is about to run.
//!
//! The distinction is not a matter of taste. A plan is compiled once and run
//! against many deployments, so any number that belongs to a DEPLOYMENT — how
//! many pages the cache has, how many rows a page holds, which pages a request
//! happens to own — cannot be in it. A text that stated its page size would be
//! right for one server and silently wrong for the next, which is why the
//! kernel rows name these as [`Source`](kernels::Source)s and ask the driver
//! rather than reading them off the statement.
//!
//! # Half of this module needs no device, and it is not gated
//!
//! [`Shape`], [`Request`], [`Frame`] and [`Unstageable`] are integer arithmetic
//! over page numbers. [`Pool`], [`Weights`] and [`Model`] hold buffers. Only
//! the second three are behind `native`, where `driver-vulkan` gates the whole
//! file — and the difference is the point rather than tidiness: the checks that
//! matter most here are the ones nothing downstream can make (a row reaching
//! into the next request's pages, two requests owning one page), and those are
//! exactly the ones that should run on a machine with no adapter in it.
//!
//! It is also what lets [`crate::pages`] be ungated, since the book speaks in
//! `Shape` and `Request` and nothing else.
//!
//! # The cache layout is the shaders', not this module's
//!
//! `attn/kv_write.wgsl` writes
//!
//! ```text
//! slot       = w_page[i] * page_size + w_off[i]
//! row_stride = n_kv_heads * head_dim
//! dst        = slot * row_stride + h * head_dim + d
//! ```
//!
//! and `attn/sdpa_paged.wgsl` reads `(slot * n_kv_heads + kv_head) * head_dim +
//! d`, which is the same expression with the product written out. Two modules
//! compiled separately from separate sources agree on it, so this file
//! transcribes a fact rather than choosing a convention, and [`Shape::slot`] is
//! where a driver can ask for the arithmetic instead of repeating it.
//!
//! **Checked against `driver-vulkan`'s copy, which was transcribed from
//! `attn/kv_write.comp`: the two agree exactly.** The WGSL is the authority
//! here and there is nothing to report — same slot, same row stride, same
//! ordering of head and channel. The one difference is in the shader and not
//! in the layout: the WGSL body owns a PAIR of channels per invocation because
//! a bf16 pair shares a `u32` and WGSL has no sub-word atomic, which changes
//! the GRID and not where an element lives.

use crate::binding::FireNumber;
use std::collections::BTreeMap;

#[cfg(feature = "native")]
use crate::binding::{FireTable, Resolve};
#[cfg(feature = "native")]
use crate::device::{Buffer, Device, Failed, Move};
#[cfg(feature = "native")]
use model_compiler::trace::ValueId;

/// What a deployment decided about its cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// How many layers the model has. One key and one value buffer each.
    pub layers: u16,
    /// Key/value heads, which is what the cache is wide in.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Rows per page.
    ///
    /// The number `Source::KvPageSize` asks for, and the one a statement cannot
    /// carry.
    pub page_size: u32,
    /// How many pages the pool holds, across all requests.
    pub pages: u32,
    /// Bytes per element. Two for the `bfloat16` cache every current entrypoint
    /// is built for.
    pub bytes: u32,
}

impl Shape {
    /// Elements in one row of the cache, across every head.
    ///
    /// `n_kv_heads * head_dim`, which is what `attn/kv_write.wgsl` calls
    /// `row_stride`.
    #[must_use]
    pub const fn row(&self) -> u64 {
        self.kv_heads as u64 * self.head_dim as u64
    }

    /// Bytes in one layer's key cache, which is also one layer's value cache.
    #[must_use]
    pub const fn layer_bytes(&self) -> u64 {
        self.elements() * self.bytes as u64
    }

    /// Where the element `(page, offset, head, at)` lives, in ELEMENTS.
    ///
    /// Transcribed from the two shaders rather than chosen. A driver that needs
    /// to read a row out of the cache — to check it, to evict it, to copy it —
    /// should ask here instead of writing the expression again, because writing
    /// it again is how the two copies come to disagree.
    #[must_use]
    pub const fn slot(&self, page: u32, offset: u32, head: u32, at: u32) -> u64 {
        let slot = page as u64 * self.page_size as u64 + offset as u64;
        slot * self.row() + head as u64 * self.head_dim as u64 + at as u64
    }

    /// Elements the pool holds in one layer, which is one past the largest
    /// index [`Shape::slot`] can return.
    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.pages as u64 * self.page_size as u64 * self.row()
    }

    /// The number a row asks the driver for, or `None` if it does not fit the
    /// channel the uniform block carries it in.
    ///
    /// Here rather than on [`Pool`] because none of it needs a device, and a
    /// claim about the cache's arithmetic that can only be made against a card
    /// gets checked on the handful of positions a card test has time for.
    ///
    /// The two strides are in ELEMENTS: `attn/kv_write.wgsl` adds them to an
    /// index, not to a byte offset. Which of them is which is fixed by
    /// [`Shape::slot`] and not free — see
    /// `the_two_stride_numbers_are_the_only_pair_that_agrees_with_slot`.
    #[must_use]
    pub fn number(&self, which: FireNumber) -> Option<u32> {
        match which {
            FireNumber::KvPageSize => Some(self.page_size),
            FireNumber::KvHeadStride => Some(self.head_dim),
            FireNumber::KvSeqStride => u32::try_from(self.row()).ok(),
        }
    }

    /// The most pages a cache of this shape could have if one of its buffers
    /// may hold at most `bytes`.
    ///
    /// # Why a per-BUFFER budget and not a total
    ///
    /// A pool of this shape is `layers * 2` separate allocations, each holding
    /// one layer's keys or one layer's values, and each spending
    /// `page_size * row * bytes` on a page. So the layers do not multiply in
    /// here: they are not competing for one allocation's size, they are
    /// competing for memory, and memory is not what
    /// [`crate::device::Device::budget`] can report. See its docs, and
    /// [`Pool::ceiling`] for the one caller.
    ///
    /// Zero when a page costs more than the whole budget, which is a cache this
    /// adapter cannot open at any size — the caller owes a refusal there and
    /// not a clamp, since a pool of no pages cannot answer.
    ///
    /// Here rather than on [`Pool`] for the same reason [`Shape::number`] is: a
    /// division that decides whether a scheduler waits forever should be
    /// checkable without an adapter.
    #[must_use]
    pub fn pages_within(&self, bytes: u64) -> u32 {
        let per_page = (self.page_size as u64)
            .saturating_mul(self.row())
            .saturating_mul(self.bytes as u64);
        if per_page == 0 {
            // A cache with no width holds every page in nothing. Degenerate
            // rather than impossible, and answering zero here would report a
            // model with no KV heads as unservable at the pool's expense
            // instead of at the geometry's, which is where it is caught.
            return u32::MAX;
        }
        u32::try_from(bytes / per_page).unwrap_or(u32::MAX)
    }
}

/// One request in a fire: the rows it contributes and the pages it owns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// The position in this request's own sequence of each row it contributes,
    /// in the order the rows appear in the fire.
    ///
    /// A decode states one; a prefill states its whole prompt. Positions are
    /// per-REQUEST, which is what makes the page arithmetic below a division.
    pub positions: Vec<u32>,
    /// The physical pages this request owns, in sequence order.
    ///
    /// Position `p` of this request lives in `pages[p / page_size]`. Nothing
    /// requires them to be contiguous or ascending — that they need not be is
    /// the entire point of a paged cache.
    pub pages: Vec<u32>,
    /// Which of THIS request's rows are read out, as indices into
    /// [`Self::positions`].
    ///
    /// Per-request and not fire-global because a request owns its rows and
    /// nothing else does: a scheduler that dropped a request between building
    /// two fires would have to renumber every later index, and the renumbering
    /// is exactly the kind of arithmetic that produces a valid index into
    /// somebody else's logits.
    ///
    /// Empty means the LAST row, which is the decode case and the overwhelming
    /// majority of fires.
    ///
    /// So a request that reads NOTHING cannot be said here, and that is a
    /// limitation rather than a decision: such a request would have to be left
    /// out of the fire. It has not come up — a request contributes rows because
    /// something wants its answer — and a third meaning for a vector that
    /// already has two would be worse than the gap.
    pub samples: Vec<u32>,
}

impl Request {
    /// A request whose last row is read out.
    #[must_use]
    pub fn of(positions: Vec<u32>, pages: Vec<u32>) -> Self {
        Self {
            positions,
            pages,
            samples: Vec::new(),
        }
    }

    /// Which of this request's rows are read out, with the default resolved.
    ///
    /// The empty case is the last row and not "no rows", so this is where the
    /// two meanings of an empty vector are separated — a caller that means no
    /// rows says so by putting a row index out of range, which [`Frame::of`]
    /// refuses.
    fn read(&self) -> Vec<u32> {
        if self.samples.is_empty() {
            self.positions
                .len()
                .checked_sub(1)
                .map(|last| vec![u32::try_from(last).unwrap_or(0)])
                .unwrap_or_default()
        } else {
            self.samples.clone()
        }
    }
}

/// What a fire's tables cannot be built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unstageable {
    /// A row's position needs a page past the end of its own request's list.
    ///
    /// The dangerous one. The page lists of every request live end to end in
    /// one table, so reading one entry past a request's span reads the NEXT
    /// request's page — a real, resident, correctly-aligned page belonging to
    /// somebody else. The append writes it, the other request's attention reads
    /// it back, and nothing faults. Refused here because it cannot be detected
    /// anywhere later.
    PastItsPages {
        /// Which request.
        request: usize,
        /// The position that asked.
        position: u32,
        /// How many pages that request has.
        pages: usize,
    },
    /// A request names a page the pool does not have.
    NoSuchPage {
        /// Which request.
        request: usize,
        /// The page it named.
        page: u32,
        /// How many the pool holds.
        pages: u32,
    },
    /// A request reads out a row it does not have.
    ///
    /// The readout gathers `logits[sampling_indices[i]]` out of a buffer whose
    /// rows belong to every request in the fire, so an index past this request's
    /// own rows is not out of bounds — it is another request's hidden state,
    /// gathered into this request's distribution and sampled from. Nothing
    /// downstream can tell, which is why it is refused here.
    NotItsRow {
        /// Which request.
        request: usize,
        /// The row within that request it asked to read.
        row: u32,
        /// How many rows it contributes.
        rows: usize,
    },
    /// Two requests in one fire own the same page.
    ///
    /// Not an error the shaders could survive: both would append to it and each
    /// would read the other's rows as its own history.
    SharedPage {
        /// The page both named.
        page: u32,
        /// The request that named it first.
        first: usize,
        /// And second.
        second: usize,
    },
    /// The shape says a page holds no slots.
    ///
    /// `driver-vulkan` found this by deleting a `.max(1)` and watching nothing
    /// fail. The clamp was there to stop `Frame::of` dividing by zero, and it
    /// did — but it built the tables for a page size of ONE while
    /// `FireNumber::KvPageSize` still handed the shader a ZERO, so every append
    /// would have addressed slot `page * 0 + off` and every request in the fire
    /// would have written the same handful of slots at the front of the cache.
    /// Silently disagreeing with the shader is worse than dividing by zero, so
    /// it is refused.
    NoSlots,
    /// The fire has no rows.
    ///
    /// Also found by deleting a `.max(1)`. Without it a rowless fire asks for a
    /// zero-byte mask table, which `wgpu` refuses outright and Vulkan refuses
    /// through a VUID; with it, the fire is accepted, allocates a word,
    /// dispatches nothing on every axis and reports success. Neither is an
    /// answer, so the emptiness is refused where it can still be named instead
    /// of at the first table that trips over it.
    NoRows,
}

impl std::fmt::Display for Unstageable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PastItsPages {
                request,
                position,
                pages,
            } => write!(
                f,
                "request {request} puts position {position} past the {pages} pages it owns"
            ),
            Self::NoSuchPage {
                request,
                page,
                pages,
            } => write!(
                f,
                "request {request} names page {page} and the pool holds {pages}"
            ),
            Self::SharedPage {
                page,
                first,
                second,
            } => write!(f, "requests {first} and {second} both own page {page}"),
            Self::NotItsRow { request, row, rows } => write!(
                f,
                "request {request} reads out its row {row} and contributes {rows}"
            ),
            Self::NoSlots => write!(f, "the shape says a page holds no slots"),
            Self::NoRows => write!(f, "the fire has no rows"),
        }
    }
}

impl std::error::Error for Unstageable {}

/// Every table a fire states, computed from what the fire is.
///
/// The tables the rows name are not independent: `kv_write_page` is
/// `kv_page_indices` indexed through `kv_page_indptr` at a position's own page,
/// and `request_of_token` is the run lengths of the requests. A caller that
/// filled them by hand can fill them inconsistently, and the shaders have no
/// way to notice.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Frame {
    /// One position per row, in fire order.
    pub positions: Vec<u32>,
    /// One request index per row.
    pub request_of_token: Vec<u32>,
    /// Every request's pages, end to end.
    pub kv_page_indices: Vec<u32>,
    /// Where each request's pages start in [`Self::kv_page_indices`], with a
    /// final entry equal to its length. `requests + 1` long.
    pub kv_page_indptr: Vec<u32>,
    /// The physical page each row's KV is appended to.
    pub kv_write_page: Vec<u32>,
    /// The row within that page.
    pub kv_write_offset: Vec<u32>,
    /// The rows the readout gathers, as indices into the FIRE, ascending.
    ///
    /// `row_gather` compacts these out of a buffer of one row per token into a
    /// buffer of one row per readout, and the lm head runs on what it wrote. So
    /// this is the only table whose length is not the fire's row count, and the
    /// count is what the row calls `RequestCount`.
    pub sampling_indices: Vec<u32>,
}

impl Frame {
    /// Work out every table from the requests.
    ///
    /// # Errors
    ///
    /// [`Unstageable`] — and the first two are checks nothing downstream can
    /// make. By the time these tables reach a shader they are just numbers, and
    /// a wrong one addresses memory that is every bit as valid as the right one.
    pub fn of(shape: Shape, requests: &[Request]) -> Result<Self, Unstageable> {
        if shape.page_size == 0 {
            return Err(Unstageable::NoSlots);
        }
        if requests.iter().all(|r| r.positions.is_empty()) {
            return Err(Unstageable::NoRows);
        }
        let page_size = shape.page_size;
        let mut frame = Self {
            kv_page_indptr: Vec::with_capacity(requests.len() + 1),
            ..Self::default()
        };
        let mut owner: BTreeMap<u32, usize> = BTreeMap::new();
        for (r, request) in requests.iter().enumerate() {
            frame
                .kv_page_indptr
                .push(u32::try_from(frame.kv_page_indices.len()).unwrap_or(u32::MAX));
            for &page in &request.pages {
                if page >= shape.pages {
                    return Err(Unstageable::NoSuchPage {
                        request: r,
                        page,
                        pages: shape.pages,
                    });
                }
                if let Some(&first) = owner.get(&page) {
                    return Err(Unstageable::SharedPage {
                        page,
                        first,
                        second: r,
                    });
                }
                owner.insert(page, r);
                frame.kv_page_indices.push(page);
            }
            // Before the rows are pushed, so `base` is where this request's
            // rows start in the fire.
            let base = u32::try_from(frame.positions.len()).unwrap_or(u32::MAX);
            for row in request.read() {
                if row as usize >= request.positions.len() {
                    return Err(Unstageable::NotItsRow {
                        request: r,
                        row,
                        rows: request.positions.len(),
                    });
                }
                frame.sampling_indices.push(base + row);
            }
            for &position in &request.positions {
                let virt = (position / page_size) as usize;
                let Some(&page) = request.pages.get(virt) else {
                    return Err(Unstageable::PastItsPages {
                        request: r,
                        position,
                        pages: request.pages.len(),
                    });
                };
                frame.positions.push(position);
                frame
                    .request_of_token
                    .push(u32::try_from(r).unwrap_or(u32::MAX));
                frame.kv_write_page.push(page);
                frame.kv_write_offset.push(position % page_size);
            }
        }
        frame
            .kv_page_indptr
            .push(u32::try_from(frame.kv_page_indices.len()).unwrap_or(u32::MAX));
        Ok(frame)
    }

    /// How many rows the fire has.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.positions.len()
    }

    /// How many readouts the fire produces.
    ///
    /// What a kernel row calls `RequestCount`, and it is not the number of
    /// requests: a request may read out more than one of its rows, and the
    /// gather's output is one row per READOUT.
    ///
    /// **A whole fire may not use this**, and [`crate::turns::Serving::step`]
    /// does not. The texts spell their epilogue as plain launches, so the head
    /// runs over the token window whatever the sampling says; a step therefore
    /// tells the lowering every row samples and stages the identity, and the
    /// number that matters there is the ROW count. This stays because it is what
    /// a frame means — the rows a caller asked to read — and because the day a
    /// text names its epilogue it becomes the number again.
    #[must_use]
    pub const fn readouts(&self) -> usize {
        self.sampling_indices.len()
    }

    /// The rows to lower this fire against.
    ///
    /// The lowering takes a per-row `samples` flag and the tables take a list of
    /// indices, and the two are the same claim said twice. Producing both from
    /// one place is what keeps a fire whose gather reads row 2 from being
    /// lowered as a fire whose row 2 does not sample — which lowers to a plan
    /// with no gather in it at all, and passes.
    #[must_use]
    pub fn seriation(&self) -> Vec<model_compiler::lower::Row> {
        let mut rows = vec![model_compiler::lower::Row::default(); self.rows()];
        // A request contributing more than one row is a prefill, and
        // `multi_token` is how the lowering is told. It matters here and not
        // only in the attention: `n_requests` is the count of rows that are NOT
        // multi-token, maxed with the count that sample, and a prefill whose
        // rows all claimed to be single-token would size the epilogue for one
        // readout per TOKEN.
        for &of in &self.request_of_token {
            let many = self.request_of_token.iter().filter(|&&r| r == of).count() > 1;
            for (row, &owner) in rows.iter_mut().zip(&self.request_of_token) {
                if owner == of {
                    row.multi_token = many;
                }
            }
        }
        for &at in &self.sampling_indices {
            if let Some(row) = rows.get_mut(at as usize) {
                row.samples = true;
            }
        }
        rows
    }
}

/// The driver's own memory for one fire.
///
/// Holds the cache, which outlives a fire, and the tables, which do not. Kept
/// together because a [`Resolve`] has to answer for both and a caller holding
/// two objects would eventually hand a kernel one fire's tables and another
/// fire's cache.
///
/// # No `release`, no `close`, and no `Device` in either
///
/// `driver-vulkan`'s pool has both, and its docs explain at length that a
/// `Drop` cannot free a Vulkan buffer because freeing one needs the device that
/// made it. A `wgpu::Buffer` is `Arc`-backed and releases its allocation when
/// the last handle drops, so the whole pair of methods — and the class of
/// defect they exist to catch, which that crate found the hard way when the
/// validation layer said `vkDestroyDevice(): VkBuffer 0x97 has not been
/// destroyed` — is deleted rather than ported.
#[cfg(feature = "native")]
pub struct Pool {
    shape: Shape,
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    tables: BTreeMap<FireTable, Buffer>,
    /// The stand-in for a weight or a seam value, if the caller gave one.
    named: Option<Buffer>,
}

#[cfg(feature = "native")]
impl Pool {
    /// Allocate one key and one value buffer per layer.
    ///
    /// Zeroed — and nothing is uploaded to make it so. WebGPU requires a new
    /// buffer's contents to be zero, so this is `layers * 2` allocations and no
    /// transfer, where the Vulkan sibling builds a host-side
    /// `vec![0u8; layer_bytes]` and pushes it across for each. For a
    /// twenty-eight layer cache that is the difference between a gigabyte of
    /// memset-and-upload and none.
    ///
    /// The zeroing is load-bearing whichever way it is achieved: a cache that
    /// came up holding the previous fire's rows would produce attention over
    /// sequences nobody asked about, and the attention would look plausible.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the first allocation that does not fit. The layers taken
    /// so far are dropped on the way out, so a partial failure leaves nothing
    /// held — which here is the ordinary Rust unwinding of a `Vec` rather than
    /// the explicit walk its sibling needs.
    pub fn open(device: &Device, shape: Shape) -> Result<Self, Failed> {
        let bytes = shape.layer_bytes();
        let mut keys = Vec::with_capacity(shape.layers as usize);
        let mut values = Vec::with_capacity(shape.layers as usize);
        for _ in 0..shape.layers {
            keys.push(device.zeroed(bytes)?);
            values.push(device.zeroed(bytes)?);
        }
        Ok(Self {
            shape,
            keys,
            values,
            tables: BTreeMap::new(),
            named: None,
        })
    }

    /// The largest page count this pool could ever be grown to.
    ///
    /// # What it is for, and the only thing it is for
    ///
    /// Telling a demand that can never be met apart from one that cannot be
    /// met now. A scheduler that waits on the first waits forever; one that
    /// drops the second drops work it had correctly admitted.
    /// [`crate::shell::Shell::launch`] is the one caller, and it answers
    /// `Impossible` above this number and grows below it.
    ///
    /// # What bounds a pool here, and it is not the heap
    ///
    /// One page costs `page_size * row * bytes` in EACH of the `layers * 2`
    /// cache buffers, and [`Self::open`] and [`Self::resize`] take those as
    /// `layers * 2` separate allocations, each bound whole as a storage
    /// buffer. So the ceiling divides [`Device::budget`] — a per-ALLOCATION
    /// cap on this adapter, see its docs for why `wgpu` will state nothing
    /// else — by what ONE buffer spends on a page, not by what the whole pool
    /// spends. Dividing by the across-layers cost instead would answer
    /// `Impossible` at a fifty-sixth of a pool this adapter would accept, and
    /// the frames it refused were ones the engine had correctly admitted.
    ///
    /// # Why there is no halving, where `driver-vulkan` has one
    ///
    /// That crate halves because its budget is a HEAP and [`Self::resize`]
    /// holds both sizes at once — deliberately, so a failed growth leaves the
    /// pool intact — so a growth's peak is twice its result. This resize holds
    /// both sizes too, and it does not move the number being divided: a new
    /// buffer of `n` bytes is legal or not on its own, and how many others are
    /// live at that moment does not enter into `max_buffer_size`. Halving here
    /// would refuse half the pools this adapter states it can bind, as a
    /// gesture at a heap size neither this method nor any other in `wgpu` can
    /// see.
    ///
    /// # What it is not
    ///
    /// A promise, and less of one than its sibling's. That number at least
    /// came from a heap; this one is a per-buffer cap, which on a discrete
    /// card is reached long after the memory is — 2 GiB a buffer across
    /// fifty-six buffers is past every consumer part. A growth well under this
    /// number can and will fail, with [`Device::zeroed`]'s own refusal.
    /// Generous is the safe direction: it turns a permanent refusal into a
    /// retried one rather than the reverse.
    #[must_use]
    pub fn ceiling(&self, device: &Device) -> u32 {
        // The division is [`Shape::pages_within`] so that it can be checked
        // without an adapter; the only thing this adds is which number to
        // divide.
        self.shape.pages_within(device.budget())
    }

    /// Grow or shrink the cache to `pages`, keeping what the pages that survive
    /// hold.
    ///
    /// # Why this reallocates instead of remapping
    ///
    /// `driver-metal`'s pool is sparse: it commits and releases pages without
    /// moving a single address a fire has bound. **WebGPU has no sparse binding
    /// at all** — there is no `map_ranges` to serve and no optional feature to
    /// ask for — so this is not a choice this backend gets to make, where
    /// `driver-vulkan` at least declines `sparseBinding` on purpose.
    ///
    /// It does not need one. **Every bind group in this driver is built during
    /// the dispatch that uses it** — [`crate::device::Device::run_all`] creates
    /// them from the pool's current buffers each fire — so no binding survives
    /// a step for a resize to invalidate. That is a property of the recording
    /// path and not an accident, so a change that cached bind groups across
    /// steps must change this too.
    ///
    /// The pages that survive keep their contents, at the same page numbers. A
    /// shrink drops the tail; the caller owes the check that nobody holds a page
    /// in it, which [`crate::shell::Shell::resize_pool`] makes.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the allocation or the copy, in which case the pool is
    /// UNCHANGED and still usable — the new buffers are all taken before any old
    /// one is released, which is why the peak is both sizes at once. A pool that
    /// half-resized would have some layers at the new page count and some at the
    /// old, and [`Shape::slot`] would index every one of them wrongly.
    ///
    /// Refuses a target of zero: a cache with no page is not a smaller cache, it
    /// is one that cannot answer.
    pub fn resize(&mut self, device: &Device, pages: u32) -> Result<(), Failed> {
        if pages == 0 {
            return Err(Failed::Wgpu(
                "a cache of zero pages cannot hold a conversation".to_string(),
            ));
        }
        if pages == self.shape.pages {
            return Ok(());
        }
        let mut grown = self.shape;
        grown.pages = pages;
        let kept = self.shape.pages.min(pages) as u64
            * self.shape.page_size as u64
            * self.shape.row()
            * self.shape.bytes as u64;

        let mut fresh = Vec::with_capacity(self.keys.len() + self.values.len());
        for _ in 0..self.keys.len() + self.values.len() {
            fresh.push(device.zeroed(grown.layer_bytes())?);
        }
        // The copies second, and all in one command buffer: nothing above this
        // line touched the pool, so an allocation failure leaves it whole.
        // Device to device -- a resize never crosses the bus, which is what the
        // Vulkan sibling's read-then-write round trip does.
        if kept > 0 {
            device.transfer(
                self.keys
                    .iter()
                    .chain(&self.values)
                    .zip(&fresh)
                    .map(|(old, new)| (old, 0u64, new, 0u64))
                    .collect::<Vec<_>>()
                    .as_slice(),
                kept,
            )?;
        }
        let values = fresh.split_off(self.keys.len());
        self.keys = fresh;
        self.values = values;
        self.shape = grown;
        Ok(())
    }

    /// What the cache was built to.
    #[must_use]
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    /// Give the pool one of the fire's tables, replacing any it held.
    ///
    /// Takes the words rather than a buffer so that the pool owns every
    /// allocation it hands out. A caller that kept the buffer could drop it
    /// while a fire still named it — which on this backend is not a
    /// use-after-free, since `wgpu` keeps the allocation alive behind the bind
    /// group, but is still a table nobody can find to restage.
    ///
    /// # Errors
    ///
    /// [`Failed`] if the table does not allocate.
    pub fn state(
        &mut self,
        device: &Device,
        which: FireTable,
        words: &[u32],
    ) -> Result<(), Failed> {
        let buffer = device.words(words)?;
        self.tables.insert(which, buffer);
        Ok(())
    }

    /// Give the pool every table of one fire at once.
    ///
    /// The reason to prefer this over six calls to [`Pool::state`] is that the
    /// tables are not independent — `kv_write_page` is the page list indexed
    /// through the CSR — so filling them separately is filling them from six
    /// chances to be inconsistent. [`Frame::of`] derives all six from one
    /// description.
    ///
    /// The attention mask goes in too, as zeros: `attn/sdpa_paged.wgsl` reads
    /// `attention_mask_enabled[row]` unconditionally, and a slot nobody filled
    /// is a slot bound to something else. Zero is the true answer for causal
    /// attention, which is the only kind this pool can describe.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the first table that does not allocate.
    pub fn stage(&mut self, device: &Device, frame: &Frame) -> Result<(), Failed> {
        for (which, words) in [
            (FireTable::Positions, &frame.positions),
            (FireTable::RequestOfToken, &frame.request_of_token),
            (FireTable::KvPageIndices, &frame.kv_page_indices),
            (FireTable::KvPageIndptr, &frame.kv_page_indptr),
            (FireTable::KvWritePage, &frame.kv_write_page),
            (FireTable::KvWriteOffset, &frame.kv_write_offset),
            (FireTable::SamplingIndices, &frame.sampling_indices),
        ] {
            self.state(device, which, words)?;
        }
        // One byte per row, and a `u32` of zeros is four zero bytes -- the
        // narrowing is the shader's. Rounded up so a fire of one row still gets
        // a word. No `.max(1)`: `Frame::of` refuses a rowless fire by name
        // (`Unstageable::NoRows`), so the round-up cannot reach zero.
        let bytes = frame.rows().div_ceil(4);
        self.state(device, FireTable::AttentionMask, &vec![0; bytes])?;
        self.state(device, FireTable::AttentionMaskEnabled, &vec![0; bytes])?;
        Ok(())
    }

    /// The rotary ladder, staged as the table the rope rows name.
    ///
    /// Not part of [`Self::stage`], and not derived from [`Shape`], because it
    /// is neither a function of the fire nor of the cache: it is a function of
    /// the MODEL's rotary width and theta and of the DEPLOYMENT's rescaling, and
    /// it does not change between fires. Staged separately so a server builds it
    /// once.
    ///
    /// `rotary_dims` rather than [`Shape::head_dim`] because they are not always
    /// the same number — a partial-rotary model rotates a prefix of each head —
    /// and the table is `rotary_dims / 2` long.
    ///
    /// Zeros are the trap this exists to avoid. An unset table is an angle of
    /// zero, which is the identity, which agrees with every reference and every
    /// other ladder; a rope that silently did nothing is a failure this family
    /// of drivers has already made once.
    ///
    /// # Errors
    ///
    /// [`Failed`] if the table does not allocate.
    pub fn ladder(
        &mut self,
        device: &Device,
        rotary_dims: u32,
        theta: f32,
        rescale: Option<crate::rope::Rescale>,
    ) -> Result<(), Failed> {
        let words = crate::rope::words(rotary_dims, theta, rescale);
        self.state(device, FireTable::RopeFrequencies, &words)
    }

    /// A single buffer standing in for every weight and seam value.
    ///
    /// A driver that has loaded a model answers those from its own tables; this
    /// exists so that a caller exercising the cache does not have to.
    ///
    /// # Errors
    ///
    /// [`Failed`] if it does not allocate.
    pub fn stand_in(&mut self, device: &Device, bytes: u64) -> Result<(), Failed> {
        self.named = Some(device.zeroed(bytes)?);
        Ok(())
    }

    /// One layer's cache, for a caller that wants to read it back.
    #[must_use]
    pub fn cache(&self, layer: u16, values: bool) -> Option<&Buffer> {
        let side = if values { &self.values } else { &self.keys };
        side.get(layer as usize)
    }

    /// Copy one page's rows onto another page, in every layer.
    ///
    /// The unit is a PAGE and not a row range, because a page is the unit the
    /// book hands out and the only one whose bytes are contiguous:
    /// [`Shape::slot`] puts a page's rows next to each other, so one page in one
    /// layer is one `memmove` of `page_size * row()` elements.
    ///
    /// Both the key and the value cache, for every layer. Copying one side, or
    /// all but the last layer, produces a conversation that attends over its own
    /// history for part of the model and over somebody else's for the rest —
    /// which is finite, plausible, and wrong.
    ///
    /// # Errors
    ///
    /// [`Failed`] if either page is past the pool, or the copy leaves a layer's
    /// buffer.
    pub fn copy_page(&self, device: &Device, from: u32, to: u32) -> Result<(), Failed> {
        if from >= self.shape.pages || to >= self.shape.pages {
            return Err(Failed::Wgpu(format!(
                "page {from} to page {to} in a pool of {}",
                self.shape.pages
            )));
        }
        self.copy_rows(device, (from, 0), (to, 0), self.shape.page_size)
    }

    /// Copy `tokens` rows from one place in the cache to another.
    ///
    /// Each side is `(page, offset in tokens within that page)`. This is what
    /// [`Pool::copy_page`] is written in terms of, because a whole page and a
    /// run of rows differ only in the length: [`Shape::slot`] lays a page's rows
    /// out contiguously, so both are one move per layer per side. One
    /// implementation, so a fork and a partial prefix share cannot disagree
    /// about where a row is.
    ///
    /// # Why this is one call and not `2 * layers` of them
    ///
    /// `wgpu` refuses `copy_buffer_to_buffer` when source and destination are
    /// the same buffer, which every one of these is — a page moves WITHIN a
    /// layer's cache. So each move is really `src -> scratch -> dst`, and
    /// [`crate::device::Device::shuffle`] does the whole list against ONE
    /// scratch in ONE command buffer. Done per layer it would be `2 * layers`
    /// scratch allocations and `2 * layers` submissions for one page.
    ///
    /// # Errors
    ///
    /// [`Failed`] if a page is past the pool, if a run leaves its page, or if a
    /// layer has no cache. A run that left its page would land in the NEXT page
    /// rather than out of bounds, which nothing would report.
    pub fn copy_rows(
        &self,
        device: &Device,
        from: (u32, u32),
        to: (u32, u32),
        tokens: u32,
    ) -> Result<(), Failed> {
        for (page, offset) in [from, to] {
            if page >= self.shape.pages {
                return Err(Failed::Wgpu(format!(
                    "page {page} in a pool of {}",
                    self.shape.pages
                )));
            }
            if offset
                .checked_add(tokens)
                .is_none_or(|e| e > self.shape.page_size)
            {
                return Err(Failed::Wgpu(format!(
                    "{tokens} rows at offset {offset} in a {}-row page",
                    self.shape.page_size
                )));
            }
        }
        // In BYTES, from the same expression the shaders index with. Written as
        // a slot difference rather than as `page * page_size * row * bytes` so
        // that a change to the layout reaches this too.
        let at = |(page, offset): (u32, u32)| {
            self.shape.slot(page, offset, 0, 0) * self.shape.bytes as u64
        };
        let bytes = tokens as u64 * self.shape.row() * self.shape.bytes as u64;
        let mut moves = Vec::with_capacity(self.shape.layers as usize * 2);
        for layer in 0..self.shape.layers {
            for values in [false, true] {
                let Some(buffer) = self.cache(layer, values) else {
                    return Err(Failed::Wgpu(format!("layer {layer} has no cache")));
                };
                moves.push(Move {
                    buffer,
                    from: at(from),
                    to: at(to),
                });
            }
        }
        device.shuffle(&moves, bytes)
    }

    /// Apply the engine's `copy_kv` plan: a list of whole-page moves and a list
    /// of single-row cells.
    ///
    /// This is the SHAPE the engine speaks, and [`crate::shell::Shell::fork`] is
    /// the shape a conversation has; they are different verbs on purpose. The
    /// engine's prefix cache knows which physical page it wants where and does
    /// not have a conversation id to name; a fork knows the conversation and not
    /// the pages. Both end at [`Pool::copy_rows`].
    ///
    /// Returns how many copies were made — pages plus cells.
    ///
    /// # What is checked before anything moves
    ///
    /// Every page and every cell, against the pool. The C++ this replaces
    /// applies the pages first and notices a bad cell afterwards, which leaves
    /// the cache half-moved with no way back. So the plan is walked once for
    /// refusals and once for work.
    ///
    /// # Errors
    ///
    /// [`Failed::Wgpu`] naming which page, which cell, or which domain.
    ///
    /// # The domain, which this now checks as strictly as its siblings
    ///
    /// `driver-vulkan` refuses any domain that is not
    /// `PIE_MEMORY_DOMAIN_VULKAN_DEVICE` and `driver-metal` any that is not
    /// `PIE_MEMORY_DOMAIN_METAL_SHARED`. This file used to record that there
    /// was no `PIE_MEMORY_DOMAIN_WEBGPU_DEVICE` to be that strict about, and
    /// checked only that both ends agreed and that neither was host memory.
    ///
    /// The constant exists now — it arrived with the engine seam that selects
    /// this backend, `crates/engine/src/driver/backend/wgpu.rs`, because
    /// `DriverBackend::device_domain` has to answer something the OTHER
    /// backends do not — and it is its own tag rather than the tag of whichever
    /// API `wgpu` opened. A `wgpu` pool and a `driver-vulkan` pool on one
    /// machine are two allocations neither driver can address in the other, so
    /// answering `VULKAN_DEVICE` here would accept a plan naming somebody
    /// else's pages, which is a prefix-cache hit reading another pool's
    /// memory.
    pub fn copy_plan(
        &self,
        device: &Device,
        plan: &driver_api::KvCopyPlan,
    ) -> Result<usize, Failed> {
        let ours = driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE;
        if plan.src_domain != ours || plan.dst_domain != ours {
            return Err(Failed::Wgpu(format!(
                "a copy from memory domain {} to domain {} is not a move within \
                 this pool, whose pages are domain {ours}",
                plan.src_domain, plan.dst_domain
            )));
        }
        if plan.src_page_ids.len() != plan.dst_page_ids.len() {
            return Err(Failed::Wgpu(format!(
                "{} source pages and {} destination pages",
                plan.src_page_ids.len(),
                plan.dst_page_ids.len()
            )));
        }
        let shape = self.shape;
        let check = |page: u32, offset: u32, what: &str| -> Result<(), Failed> {
            if page >= shape.pages || offset >= shape.page_size {
                return Err(Failed::Wgpu(format!(
                    "{what} names page {page} row {offset}, and the pool has {} pages \
                     of {} rows",
                    shape.pages, shape.page_size
                )));
            }
            Ok(())
        };
        for (i, (&src, &dst)) in plan.src_page_ids.iter().zip(&plan.dst_page_ids).enumerate() {
            check(src, 0, &format!("page move {i}'s source"))?;
            check(dst, 0, &format!("page move {i}'s destination"))?;
        }
        for (i, cell) in plan.cells.iter().enumerate() {
            check(
                cell.src_page_id,
                cell.src_token_offset,
                &format!("cell {i}'s source"),
            )?;
            check(
                cell.dst_page_id,
                cell.dst_token_offset,
                &format!("cell {i}'s destination"),
            )?;
        }

        // Nothing above this line moved a byte.
        for (&src, &dst) in plan.src_page_ids.iter().zip(&plan.dst_page_ids) {
            self.copy_page(device, src, dst)?;
        }
        for cell in &plan.cells {
            self.copy_rows(
                device,
                (cell.src_page_id, cell.src_token_offset),
                (cell.dst_page_id, cell.dst_token_offset),
                1,
            )?;
        }
        Ok(plan.src_page_ids.len() + plan.cells.len())
    }
}

#[cfg(feature = "native")]
impl Resolve for Pool {
    type Buffer = Buffer;

    fn weight(&self, _name: &str) -> Option<&Buffer> {
        self.named.as_ref()
    }

    fn named(&self, _value: ValueId) -> Option<&Buffer> {
        self.named.as_ref()
    }

    fn kv(&self, layer: u16, values: bool) -> Option<&Buffer> {
        self.cache(layer, values)
    }

    fn table(&self, which: FireTable) -> Option<&Buffer> {
        self.tables.get(&which)
    }

    fn number(&self, which: FireNumber) -> Option<u32> {
        self.shape.number(which)
    }
}

/// One buffer per weight, under the name a PLAN uses for it.
///
/// By NAME and not by index, because that is what a plan states: `Arg::Weight`
/// carries a trace name and nothing else. Two layers that happen to hold the
/// same bytes are two entries.
///
/// Separate from [`Pool`] because the two have opposite lifetimes. A pool
/// belongs to a deployment and its tables belong to a fire; weights belong to a
/// MODEL and outlive both.
#[cfg(feature = "native")]
#[derive(Default)]
pub struct Weights {
    held: BTreeMap<String, Buffer>,
    seam: Option<Buffer>,
}

#[cfg(feature = "native")]
impl Weights {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold `bytes` under `name`, replacing whatever was there.
    ///
    /// # Errors
    ///
    /// [`Failed`] if it does not allocate.
    pub fn hold(&mut self, device: &Device, name: &str, bytes: &[u8]) -> Result<(), Failed> {
        let buffer = device.buffer(bytes)?;
        self.held.insert(name.to_owned(), buffer);
        Ok(())
    }

    /// The buffer held under `name`.
    ///
    /// Public so that a caller can assert WHICH buffer a binder chose rather
    /// than only that it chose one. A test that reads the slot the binder filled
    /// and compares it against itself passes for every name.
    #[must_use]
    pub fn at(&self, name: &str) -> Option<&Buffer> {
        self.held.get(name)
    }

    /// One buffer standing in for every value the seam binds by name.
    ///
    /// Unlike the weights these are not distinguished here, because a seam value
    /// is an ACTIVATION — the observed query, the logits — and a driver that has
    /// a frame has somewhere real to put them. This exists so that a caller
    /// exercising the weights does not also have to build one.
    ///
    /// # Errors
    ///
    /// [`Failed`] if it does not allocate.
    pub fn seam(&mut self, device: &Device, bytes: u64) -> Result<(), Failed> {
        self.seam = Some(device.zeroed(bytes)?);
        Ok(())
    }

    /// How many names are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.held.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}

#[cfg(feature = "native")]
impl Resolve for Weights {
    type Buffer = Buffer;

    fn weight(&self, name: &str) -> Option<&Buffer> {
        self.held.get(name)
    }

    fn named(&self, _value: ValueId) -> Option<&Buffer> {
        self.seam.as_ref()
    }
}

/// A model's weights and a deployment's cache, as one resolver.
///
/// Neither half can answer a fire on its own. [`Pool`] answers the cache, the
/// tables and the fire's numbers, and answers every weight with a single
/// stand-in buffer; [`Weights`] answers weights by name and knows nothing about
/// a cache. A real plan states both — 704 weight names and 28 layers of KV for
/// qwen3-0.6B — so a resolver that was only one of them is deliberately wrong
/// about half of every fire.
///
/// A borrow of each rather than ownership, and that is the point of the type
/// rather than an implementation detail: they have different LIFETIMES. A
/// model's weights are loaded once and outlive every deployment of it; a pool is
/// sized for one deployment's context and outlives every fire in it. A struct
/// that owned both would tie the weights to the pool, and reopening a pool —
/// which is what changing the context length is — would drop them.
///
/// # Where each question goes
///
/// Weights and seam values to the weights; cache, tables and numbers to the
/// pool. There is no overlap and no precedence to get wrong, which is why this
/// is a pair and not a chain of fallbacks: a chain would answer an unknown
/// weight name with the pool's stand-in, and a stand-in of zeros computes an
/// answer rather than refusing.
#[cfg(feature = "native")]
pub struct Model<'a> {
    /// One buffer per tensor name.
    pub weights: &'a Weights,
    /// The cache, the fire's tables and the fire's numbers.
    pub pool: &'a Pool,
}

#[cfg(feature = "native")]
impl Resolve for Model<'_> {
    type Buffer = Buffer;

    fn weight(&self, name: &str) -> Option<&Buffer> {
        self.weights.weight(name)
    }

    fn named(&self, value: ValueId) -> Option<&Buffer> {
        self.weights.named(value)
    }

    fn kv(&self, layer: u16, values: bool) -> Option<&Buffer> {
        self.pool.cache(layer, values)
    }

    fn table(&self, which: FireTable) -> Option<&Buffer> {
        Resolve::table(self.pool, which)
    }

    fn number(&self, which: FireNumber) -> Option<u32> {
        Resolve::number(self.pool, which)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache of a shape no test size divides evenly.
    ///
    /// 13 pages of 16 rows over 3 heads of 10 elements: `.wiki/new-driver`'s
    /// §12 finding is that a round size makes `div_ceil` and plain division the
    /// same expression, and a page count of 16 over a page size of 16 would
    /// make several of the checks below pass for a shape that is not the one
    /// they mean.
    fn shape() -> Shape {
        Shape {
            layers: 2,
            kv_heads: 3,
            head_dim: 10,
            page_size: 16,
            pages: 13,
            bytes: 2,
        }
    }

    /// Every element of the cache has exactly one address, and the highest one
    /// is one below what the pool holds.
    ///
    /// The claim `Shape::slot` exists to make: a driver reading a row out of the
    /// cache must land where `attn/kv_write.wgsl` put it. A bijection is what
    /// says the four coordinates are laid out with no overlap and no gap, and
    /// overlap is the failure that matters — two rows sharing an address is one
    /// conversation reading another's history.
    #[test]
    fn every_element_of_the_cache_has_exactly_one_address() {
        let shape = shape();
        let mut seen = vec![false; usize::try_from(shape.elements()).expect("a small pool")];
        for page in 0..shape.pages {
            for off in 0..shape.page_size {
                for head in 0..shape.kv_heads {
                    for at in 0..shape.head_dim {
                        let slot = usize::try_from(shape.slot(page, off, head, at))
                            .expect("inside the pool");
                        assert!(
                            !seen[slot],
                            "({page}, {off}, {head}, {at}) collides at element {slot}"
                        );
                        seen[slot] = true;
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|s| *s),
            "the pool holds {} elements and the addresses do not cover them",
            shape.elements()
        );
    }

    /// The two stride numbers a row asks for are the only pair that agrees with
    /// [`Shape::slot`].
    ///
    /// `KvHeadStride` and `KvSeqStride` are handed to the shader as scalars and
    /// the shader adds them to an index. Swapping them produces a cache that is
    /// self-consistent — every append and every read uses the same wrong
    /// strides — for exactly the shapes where `head_dim == row()`, which is a
    /// one-head model. This asserts against the layout instead.
    #[test]
    fn the_two_stride_numbers_are_the_only_pair_that_agrees_with_slot() {
        let shape = shape();
        let head = shape
            .number(FireNumber::KvHeadStride)
            .expect("a head stride");
        let seq = shape.number(FireNumber::KvSeqStride).expect("a seq stride");
        // One head along is `head_dim` elements; one row along is the whole
        // width of every head.
        assert_eq!(
            u64::from(head),
            shape.slot(0, 0, 1, 0) - shape.slot(0, 0, 0, 0)
        );
        assert_eq!(
            u64::from(seq),
            shape.slot(0, 1, 0, 0) - shape.slot(0, 0, 0, 0)
        );
        assert_ne!(head, seq, "this shape cannot tell the two apart");
        assert_eq!(
            shape.number(FireNumber::KvPageSize),
            Some(shape.page_size),
            "the page size is the pool's and not the fire's"
        );
    }

    /// The pool's ceiling is a per-BUFFER division, and the layers do not
    /// multiply into it.
    ///
    /// # Why this is the assertion and not a round number
    ///
    /// [`Shape::pages_within`] is what turns `Launched::Impossible` on and off,
    /// and both directions of getting it wrong are silent. Dividing by the
    /// across-layers cost — `layers * 2` times larger — would answer
    /// `Impossible` for a pool this adapter states it can bind, and the engine
    /// permanently drops frames it had correctly admitted. Multiplying instead
    /// of dividing would admit a pool whose per-layer buffer `Device::zeroed`
    /// then refuses, which turns a scheduling answer into a device error.
    ///
    /// So the claim is the exact page count, and it is asserted to be
    /// independent of `layers` — the one term that must not appear.
    #[test]
    fn the_pools_ceiling_counts_one_buffers_pages_and_not_every_layers() {
        let shape = shape();
        // 16 rows a page, 3 heads of 10 elements, 2 bytes each: 960 bytes.
        let per_page = 16 * 3 * 10 * 2;
        assert_eq!(shape.pages_within(per_page * 100), 100);
        // Not a multiple: the tail page does not fit, and a `div_ceil` here
        // would admit a pool one page past what the adapter will bind.
        assert_eq!(shape.pages_within(per_page * 100 + per_page - 1), 100);

        let deeper = Shape {
            layers: 64,
            ..shape
        };
        assert_eq!(
            deeper.pages_within(per_page * 100),
            100,
            "the ceiling fell with the layer count, so a deep model is refused \
             at a pool a shallow one is admitted at -- and the layers are \
             separate allocations, which is the whole reason they do not \
             multiply in"
        );

        // A budget under one page is a cache that cannot be opened at all.
        // Zero rather than one, because `Pool::resize` refuses zero by name and
        // a clamp to one would hand it a size it cannot allocate.
        assert_eq!(shape.pages_within(per_page - 1), 0);

        // Past `u32::MAX` pages is clamped rather than wrapped: a wrap would
        // report a huge budget as a tiny pool and refuse everything.
        assert_eq!(shape.pages_within(u64::MAX), u32::MAX);
    }

    /// A page holding no slots is refused rather than rounded up to one.
    #[test]
    fn a_page_that_holds_no_slots_is_refused_and_not_rounded_up_to_one() {
        let mut shape = shape();
        shape.page_size = 0;
        assert_eq!(
            Frame::of(shape, &[Request::of(vec![0], vec![0])]).err(),
            Some(Unstageable::NoSlots)
        );
    }

    /// A fire with no rows is refused rather than staged as an empty one.
    #[test]
    fn a_fire_with_no_rows_is_refused_and_not_staged_as_an_empty_one() {
        assert_eq!(
            Frame::of(shape(), &[Request::of(Vec::new(), vec![0])]).err(),
            Some(Unstageable::NoRows)
        );
    }

    /// A frame states each row once and puts it in its own request's page.
    #[test]
    fn a_frame_states_each_row_once_and_puts_it_in_its_own_requests_page() {
        // 19 positions over a 16-row page: the second page is PARTIAL, which is
        // what makes the division a division.
        let shape = shape();
        let first = Request::of((0..19).collect(), vec![7, 2]);
        let second = Request::of(vec![40], vec![11, 5, 3]);
        let frame = Frame::of(shape, &[first, second]).expect("a stageable fire");
        assert_eq!(frame.rows(), 20);
        assert_eq!(frame.kv_page_indptr, vec![0, 2, 5]);
        assert_eq!(frame.kv_page_indices, vec![7, 2, 11, 5, 3]);
        // Rows 0..15 are page 7 and rows 16..18 are page 2, which is the
        // division the page size makes.
        assert_eq!(frame.kv_write_page[15], 7);
        assert_eq!(frame.kv_write_page[16], 2);
        assert_eq!(frame.kv_write_offset[16], 0);
        // Position 40 of the second request is its page index 2, which is 3.
        assert_eq!(frame.kv_write_page[19], 3);
        assert_eq!(frame.kv_write_offset[19], 8);
        // The last row of each, by default.
        assert_eq!(frame.sampling_indices, vec![18, 19]);
        assert_eq!(frame.readouts(), 2);
    }

    /// A row that would reach into the next request's pages is refused.
    #[test]
    fn a_row_that_would_reach_into_the_next_requests_pages_is_refused() {
        let shape = shape();
        // Position 16 needs page index 1 and the request owns one page.
        let err = Frame::of(shape, &[Request::of(vec![16], vec![4])]).expect_err("past its pages");
        assert_eq!(
            err,
            Unstageable::PastItsPages {
                request: 0,
                position: 16,
                pages: 1
            }
        );
    }

    /// A frame refuses pages that are not the pool's or not its own.
    #[test]
    fn a_frame_refuses_pages_that_are_not_the_pools_or_not_its_own() {
        let shape = shape();
        assert_eq!(
            Frame::of(shape, &[Request::of(vec![0], vec![13])]).err(),
            Some(Unstageable::NoSuchPage {
                request: 0,
                page: 13,
                pages: 13
            })
        );
        assert_eq!(
            Frame::of(
                shape,
                &[Request::of(vec![0], vec![4]), Request::of(vec![0], vec![4])]
            )
            .err(),
            Some(Unstageable::SharedPage {
                page: 4,
                first: 0,
                second: 1
            })
        );
    }

    /// A request cannot read out a row that is not its own.
    #[test]
    fn a_request_cannot_read_out_a_row_that_is_not_its_own() {
        let shape = shape();
        let mut greedy = Request::of(vec![0, 1], vec![6]);
        greedy.samples = vec![2];
        assert_eq!(
            Frame::of(shape, &[greedy]).err(),
            Some(Unstageable::NotItsRow {
                request: 0,
                row: 2,
                rows: 2
            })
        );
    }

    /// A request's own row becomes the fire's row.
    ///
    /// The offset that makes a second request's `samples: [0]` mean ITS first
    /// row and not the fire's.
    #[test]
    fn a_requests_own_row_becomes_the_fires_row() {
        let shape = shape();
        let mut first = Request::of(vec![0, 1, 2], vec![1]);
        first.samples = vec![0, 2];
        let mut second = Request::of(vec![0, 1], vec![9]);
        second.samples = vec![0];
        let frame = Frame::of(shape, &[first, second]).expect("stageable");
        assert_eq!(frame.sampling_indices, vec![0, 2, 3]);
    }

    /// The seriation says the same thing the sampling table does.
    #[test]
    fn the_seriation_and_the_sampling_table_are_one_claim() {
        let shape = shape();
        let frame = Frame::of(
            shape,
            &[
                Request::of(vec![0, 1, 2], vec![1]),
                Request::of(vec![0], vec![9]),
            ],
        )
        .expect("stageable");
        let rows = frame.seriation();
        assert_eq!(rows.len(), frame.rows());
        for (at, row) in rows.iter().enumerate() {
            assert_eq!(
                row.samples,
                frame.sampling_indices.contains(&(at as u32)),
                "row {at} disagrees about sampling"
            );
        }
        // The first request contributes three rows, so it is a prefill and its
        // rows say so; the second contributes one and does not.
        assert!(rows[0].multi_token && rows[1].multi_token && rows[2].multi_token);
        assert!(!rows[3].multi_token);
    }
}
