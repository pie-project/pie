//! The memory a plan never mentions.
//!
//! Everything else in this crate reads a statement and works out what it
//! means. This module holds what no statement can: the paged KV cache, and
//! the tables a fire assembles for the rows it is about to run.
//!
//! The distinction is not a matter of taste. A plan is compiled once and run
//! against many deployments, so any number that belongs to a DEPLOYMENT --
//! how many pages the cache has, how many rows a page holds, which pages a
//! request happens to own -- cannot be in it. A text that stated its page
//! size would be right for one server and silently wrong for the next, which
//! is why the kernel rows name these as [`Source`](kernels::Source)s and ask
//! the driver rather than reading them off the statement.
//!
//! # The cache layout is the shaders', not this module's
//!
//! `attn/kv_write.comp` writes
//!
//! ```text
//! slot = page[i] * page_size + off[i]
//! at   = slot * (kv_heads * head_dim) + h * head_dim + d
//! ```
//!
//! and `attn/sdpa_paged.comp` reads
//! `(slot * n_kv_heads + kv_head) * head_dim + d_out`, the same expression. Two modules compiled separately
//! from separate sources agree on it, so this file transcribes a fact rather
//! than choosing a convention, and [`Shape::slot`] is where a driver can ask
//! for the arithmetic instead of repeating it.

use crate::binding::{FireNumber, FireTable, Resolve};
use crate::device::{Buffer, Device, Failed};
use model_compiler::trace::ValueId;
use std::collections::BTreeMap;

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
    /// The number `Source::KvPageSize` asks for, and the one a statement
    /// cannot carry.
    pub page_size: u32,
    /// How many pages the pool holds, across all requests.
    pub pages: u32,
    /// Bytes per element. Two for the `bfloat16` cache every current
    /// entrypoint is built for.
    pub bytes: u32,
}

impl Shape {
    /// Elements in one row of the cache, across every head.
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
    /// Transcribed from the two shaders rather than chosen. A driver that
    /// needs to read a row out of the cache -- to check it, to evict it, to
    /// copy it -- should ask here instead of writing the expression again,
    /// because writing it again is how the two copies come to disagree.
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
    /// channel the push block carries it in.
    ///
    /// Here rather than on [`Pool`] because none of it needs a device, and a
    /// claim about the cache's arithmetic that can only be made against a card
    /// gets checked on the handful of positions a card test has time for.
    ///
    /// The two strides are in ELEMENTS: `attn/kv_write.comp` adds them to an
    /// index, not to a byte offset. Which of them is which is fixed by
    /// [`Shape::slot`] and not free -- see
    /// `the_two_stride_numbers_are_the_only_pair_that_agrees_with_slot`.
    #[must_use]
    pub fn number(&self, which: FireNumber) -> Option<u32> {
        match which {
            FireNumber::KvPageSize => Some(self.page_size),
            FireNumber::KvHeadStride => Some(self.head_dim),
            FireNumber::KvSeqStride => u32::try_from(self.row()).ok(),
        }
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
    /// requires them to be contiguous or ascending -- that they need not be is
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
    /// majority of fires -- `driver-metal`'s `rows_of` reaches the same default
    /// by the same reasoning.
    ///
    /// So a request that reads NOTHING cannot be said here, and that is a
    /// limitation rather than a decision: such a request would have to be left
    /// out of the fire. It has not come up -- a request contributes rows
    /// because something wants its answer -- and a third meaning for a vector
    /// that already has two would be worse than the gap.
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
    /// two meanings of an empty vector are separated -- a caller that means no
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
    /// request's page -- a real, resident, correctly-aligned page belonging to
    /// somebody else. The append writes it, the other request's attention
    /// reads it back, and nothing faults. Refused here because it cannot be
    /// detected anywhere later.
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
    /// rows belong to every request in the fire, so an index past this
    /// request's own rows is not out of bounds -- it is another request's
    /// hidden state, gathered into this request's distribution and sampled
    /// from. Nothing downstream can tell, which is why it is refused here.
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
    /// Not an error the shaders could survive: both would append to it and
    /// each would read the other's rows as its own history.
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
    /// Found by deleting a `.max(1)` and watching nothing fail. The clamp was
    /// there to stop `Frame::of` dividing by zero, and it did -- but it made
    /// the tables for a page size of ONE while `FireNumber::KvPageSize` still
    /// handed the shader a ZERO, so every append would have addressed slot
    /// `page * 0 + off` and every request in the fire would have written the
    /// same handful of slots at the front of the cache. Silently disagreeing
    /// with the shader is worse than dividing by zero, so it is refused.
    NoSlots,
    /// The fire has no rows.
    ///
    /// Also found by deleting a `.max(1)` -- the one rounding the mask tables
    /// up to a word -- and watching nothing fail. Without it a rowless fire
    /// asks Vulkan for a zero-byte buffer, which is a VUID; with it, the fire
    /// is accepted, allocates a word, dispatches nothing on every axis and
    /// reports success. Neither is an answer, so the emptiness is refused
    /// where it can still be named instead of at the first table that trips
    /// over it.
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
/// `kv_page_indices` indexed through `kv_page_indptr` at a position's own
/// page, and `request_of_token` is the run lengths of the requests. A caller
/// that filled them by hand -- which is what every test in this crate did
/// before this existed -- can fill them inconsistently, and the shaders have
/// no way to notice.
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
    /// buffer of one row per readout, and the lm head runs on what it wrote.
    /// So this is the only table whose length is not the fire's row count, and
    /// the count is what the row calls `RequestCount`.
    pub sampling_indices: Vec<u32>,
}

impl Frame {
    /// Work out every table from the requests.
    ///
    /// # Errors
    ///
    /// [`Unstageable`] -- and the first two are checks nothing downstream can
    /// make. By the time these tables reach a shader they are just numbers,
    /// and a wrong one addresses memory that is every bit as valid as the
    /// right one.
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
    /// gather's output is one row per READOUT. `driver-metal` reaches the same
    /// number by the same route, from the same table.
    ///
    /// **A whole fire may not use this**, and `turns::Serving::step` does not.
    /// qwen3's text spells its epilogue as plain launches, so the head runs
    /// over the token window whatever the sampling says; a step therefore
    /// tells the lowering every row samples and stages the identity, and the
    /// number that matters there is the ROW count. This stays because it is
    /// what a frame means -- the rows a caller asked to read -- and because
    /// the day a text names its epilogue it becomes the number again.
    #[must_use]
    pub const fn readouts(&self) -> usize {
        self.sampling_indices.len()
    }

    /// The rows to lower this fire against.
    ///
    /// The lowering takes a per-row `samples` flag and the tables take a list
    /// of indices, and the two are the same claim said twice. Producing both
    /// from one place is what keeps a fire whose gather reads row 2 from being
    /// lowered as a fire whose row 2 does not sample -- which lowers to a plan
    /// with no gather in it at all, and passes.
    #[must_use]
    pub fn seriation(&self) -> Vec<model_compiler::lower::Row> {
        let mut rows = vec![model_compiler::lower::Row::default(); self.rows()];
        // A request contributing more than one row is a prefill, and
        // `multi_token` is how the lowering is told. It matters here and not
        // only in the attention: `n_requests` is the count of rows that are
        // NOT multi-token, maxed with the count that sample, and a prefill
        // whose rows all claimed to be single-token would size the epilogue
        // for one readout per TOKEN.
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
pub struct Pool {
    shape: Shape,
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    tables: BTreeMap<FireTable, Buffer>,
    /// The stand-in for a weight or a seam value, if the caller gave one.
    named: Option<Buffer>,
}

impl Pool {
    /// Allocate one key and one value buffer per layer.
    ///
    /// Zeroed. A cache that came up holding the previous fire's rows would
    /// produce attention over sequences nobody asked about, and the attention
    /// would look plausible.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the first allocation that does not fit.
    pub fn open(device: &Device, shape: Shape) -> Result<Self, Failed> {
        let zeros = vec![0u8; usize::try_from(shape.layer_bytes()).unwrap_or(usize::MAX)];
        let mut keys = Vec::with_capacity(shape.layers as usize);
        let mut values = Vec::with_capacity(shape.layers as usize);
        // Freed on the way out of a partial failure: an allocator that leaks
        // the layers it did get is an allocator whose second call fails for a
        // reason that has nothing to do with the second call.
        for _ in 0..shape.layers {
            match device.buffer(&zeros) {
                Ok(b) => keys.push(b),
                Err(e) => {
                    for b in keys.into_iter().chain(values) {
                        device.free(b);
                    }
                    return Err(e);
                }
            }
            match device.buffer(&zeros) {
                Ok(b) => values.push(b),
                Err(e) => {
                    for b in keys.into_iter().chain(values) {
                        device.free(b);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self {
            shape,
            keys,
            values,
            tables: BTreeMap::new(),
            named: None,
        })
    }

    /// Grow or shrink the cache to `pages`, keeping what the pages that
    /// survive hold.
    ///
    /// # Why this reallocates instead of mapping
    ///
    /// `driver-metal`'s pool is sparse: it commits and releases pages without
    /// moving a single address a fire has bound, and it has to be, because
    /// Metal binds its heap once. Vulkan has sparse binding too, and this
    /// does not use it -- `sparseBinding` is an optional feature, and the
    /// whole point of [`crate::device::Tier`] is that this driver runs where
    /// the optional features are absent.
    ///
    /// It does not need to. **Every descriptor in this driver is written
    /// during the step that uses it** -- `turns::Serving::step` records a
    /// fresh command buffer and binds the pool's buffers by handle each time
    /// -- so no address survives a step for a resize to invalidate. That is a
    /// property of the recording path and not an accident, so a change that
    /// cached descriptor sets across steps must change this too.
    ///
    /// The pages that survive keep their contents, at the same page numbers.
    /// A shrink drops the tail; the caller owes the check that nobody holds a
    /// page in it, which [`crate::shell::Shell::resize_pool`] makes.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the allocation, in which case the pool is UNCHANGED
    /// and still usable -- the new buffers are all taken before any old one
    /// is freed, which is why the peak is both sizes at once. A pool that
    /// half-resized would have some layers at the new page count and some at
    /// the old, and `Shape::slot` would index every one of them wrongly.
    ///
    /// Refuses a target of zero: a cache with no page is not a smaller cache,
    /// it is one that cannot answer.
    pub fn resize(&mut self, device: &Device, pages: u32) -> Result<(), Failed> {
        if pages == 0 {
            return Err(Failed::Vulkan(
                "a cache of zero pages cannot hold a conversation".to_string(),
            ));
        }
        if pages == self.shape.pages {
            return Ok(());
        }
        let mut grown = self.shape;
        grown.pages = pages;
        let kept = usize::try_from(
            self.shape.pages.min(pages) as u64
                * self.shape.page_size as u64
                * self.shape.row()
                * self.shape.bytes as u64,
        )
        .unwrap_or(usize::MAX);
        let bytes = usize::try_from(grown.layer_bytes()).unwrap_or(usize::MAX);

        let mut fresh = Vec::with_capacity(self.keys.len() + self.values.len());
        for old in self.keys.iter().chain(&self.values) {
            // Read before allocate, so a failure has nothing half-written.
            let held = device.read(old)?;
            let mut filled = vec![0u8; bytes];
            filled[..kept].copy_from_slice(&held[..kept]);
            match device.buffer(&filled) {
                Ok(b) => fresh.push(b),
                Err(e) => {
                    // Nothing of the pool has changed yet.
                    for b in fresh {
                        device.free(b);
                    }
                    return Err(e);
                }
            }
        }
        let values = fresh.split_off(self.keys.len());
        for b in std::mem::replace(&mut self.keys, fresh)
            .into_iter()
            .chain(std::mem::replace(&mut self.values, values))
        {
            device.free(b);
        }
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
    /// allocation it hands out. A caller that kept the buffer could free it
    /// while a command buffer still named it, which is a use-after-free the
    /// layer reports and the caller does not.
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
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let buffer = device.buffer(&bytes)?;
        if let Some(old) = self.tables.insert(which, buffer) {
            device.free(old);
        }
        Ok(())
    }

    /// Give the pool every table of one fire at once.
    ///
    /// The reason to prefer this over six calls to [`Pool::state`] is that
    /// the tables are not independent -- `kv_write_page` is the page list
    /// indexed through the CSR -- so filling them separately is filling them
    /// from six chances to be inconsistent. [`Frame::of`] derives all six
    /// from one description.
    ///
    /// The attention mask goes in too, as zeros: `attn/sdpa_paged.comp` reads
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
        // narrowing is the shader's. Rounded up so a fire of one row still
        // gets a word.
        // No `.max(1)`: `Frame::of` refuses a rowless fire by name
        // (`Unstageable::NoRows`), so the round-up below cannot reach zero.
        // The clamp that used to be here would have let one through to
        // allocate a word and dispatch nothing.
        let bytes = frame.rows().div_ceil(4);
        self.state(device, FireTable::AttentionMask, &vec![0; bytes])?;
        self.state(device, FireTable::AttentionMaskEnabled, &vec![0; bytes])?;
        Ok(())
    }

    /// The rotary ladder, staged as the table the rope rows name.
    ///
    /// Not part of [`Self::stage`], and not derived from [`Shape`], because it
    /// is neither a function of the fire nor of the cache: it is a function of
    /// the MODEL's rotary width and theta and of the DEPLOYMENT's rescaling,
    /// and it does not change between fires. Staged separately so a server
    /// builds it once.
    ///
    /// `rotary_dims` rather than [`Shape::head_dim`] because they are not
    /// always the same number -- a partial-rotary model rotates a prefix of
    /// each head -- and the table is `rotary_dims / 2` long.
    ///
    /// Zeros are the trap this exists to avoid. An unset table is an angle of
    /// zero, which is the identity, which agrees with every reference and
    /// every other ladder; a rope that silently did nothing is the failure
    /// this crate has already made once.
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

    /// Give every buffer back to the device.
    ///
    /// Explicit, for the same reason [`Device::free`] is: a `Drop` would need
    /// the device, and a pool that carried one could not be stored beside it.
    ///
    /// A caller that OWNS its device has to call this. The validation layer
    /// reports the alternative as
    /// `vkDestroyDevice(): VkBuffer ... has not been destroyed`, and this
    /// crate treats a layer error as fatal -- which is how the omission was
    /// found, the first time anything here owned a device rather than
    /// borrowing a shared one.
    pub fn release(&mut self, device: &Device) {
        for buffer in self
            .keys
            .drain(..)
            .chain(self.values.drain(..))
            .chain(std::mem::take(&mut self.tables).into_values())
            .chain(self.named.take())
        {
            device.free(buffer);
        }
    }

    /// A single buffer standing in for every weight and seam value.
    ///
    /// A driver that has loaded a model answers those from its own tables;
    /// this exists so that a caller exercising the cache does not have to.
    ///
    /// # Errors
    ///
    /// [`Failed`] if it does not allocate.
    pub fn stand_in(&mut self, device: &Device, bytes: u64) -> Result<(), Failed> {
        let buffer = device.buffer(&vec![0u8; usize::try_from(bytes).unwrap_or(0)])?;
        if let Some(old) = self.named.replace(buffer) {
            device.free(old);
        }
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
    /// [`Shape::slot`] puts a page's rows next to each other, so one page in
    /// one layer is one `memmove` of `page_size * row()` elements. A row range
    /// inside a page is contiguous too and would be a second entry point with
    /// a second off-by-one; a caller that wants rows can copy the page and
    /// grow the destination to fewer tokens.
    ///
    /// Both the key and the value cache, for every layer. Copying one side, or
    /// all but the last layer, produces a conversation that attends over its
    /// own history for part of the model and over somebody else's for the
    /// rest -- which is finite, plausible, and wrong.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if either page is past the pool, or the copy leaves
    /// a layer's buffer.
    pub fn copy_page(&self, device: &Device, from: u32, to: u32) -> Result<(), Failed> {
        if from >= self.shape.pages || to >= self.shape.pages {
            return Err(Failed::Vulkan(format!(
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
    /// run of rows differ only in the length: [`Shape::slot`] lays a page's
    /// rows out contiguously, so both are one `memmove` per layer per side.
    /// One implementation, so a fork and a partial prefix share cannot
    /// disagree about where a row is.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if a page is past the pool, if a run leaves its
    /// page, or if a layer has no cache. A run that left its page would land
    /// in the NEXT page rather than out of bounds, which nothing would
    /// report.
    pub fn copy_rows(
        &self,
        device: &Device,
        from: (u32, u32),
        to: (u32, u32),
        tokens: u32,
    ) -> Result<(), Failed> {
        for (page, offset) in [from, to] {
            if page >= self.shape.pages {
                return Err(Failed::Vulkan(format!(
                    "page {page} in a pool of {}",
                    self.shape.pages
                )));
            }
            if offset
                .checked_add(tokens)
                .is_none_or(|e| e > self.shape.page_size)
            {
                return Err(Failed::Vulkan(format!(
                    "{tokens} rows at offset {offset} in a {}-row page",
                    self.shape.page_size
                )));
            }
        }
        // In BYTES, from the same expression the shaders index with. Written
        // as a slot difference rather than as `page * page_size * row * bytes`
        // so that a change to the layout reaches this too.
        let at = |(page, offset): (u32, u32)| {
            self.shape.slot(page, offset, 0, 0) * self.shape.bytes as u64
        };
        let bytes = tokens as u64 * self.shape.row() * self.shape.bytes as u64;
        for layer in 0..self.shape.layers {
            for values in [false, true] {
                let Some(buffer) = self.cache(layer, values) else {
                    return Err(Failed::Vulkan(format!("layer {layer} has no cache")));
                };
                device.copy_within(buffer, at(from), at(to), bytes)?;
            }
        }
        Ok(())
    }

    /// Apply the engine's `copy_kv` plan: a list of whole-page moves and a
    /// list of single-row cells.
    ///
    /// This is the SHAPE the engine speaks, and [`crate::shell::Shell::fork`] is the shape a
    /// conversation has; they are different verbs on purpose. The engine's
    /// prefix cache knows which physical page it wants where and does not have
    /// a conversation id to name; a fork knows the conversation and not the
    /// pages. Both end at [`Pool::copy_rows`].
    ///
    /// Returns how many copies were made -- pages plus cells.
    ///
    /// # What is checked before anything moves
    ///
    /// Every page and every cell, against the pool. The C++ this replaces
    /// applies the pages first and notices a bad cell afterwards, which
    /// leaves the cache half-moved with no way back; `driver-metal`'s port
    /// records the same finding. So the plan is walked once for refusals and
    /// once for work.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] naming which page, which cell, or which domain. A
    /// domain that is not [`driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE`] is
    /// refused rather than assumed: a plan addressed to another backend's
    /// memory that this one served would copy the right bytes to the wrong
    /// device's pages.
    pub fn copy_plan(
        &self,
        device: &Device,
        plan: &driver_api::KvCopyPlan,
    ) -> Result<usize, Failed> {
        let vulkan = driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE;
        if plan.src_domain != vulkan || plan.dst_domain != vulkan {
            return Err(Failed::Vulkan(format!(
                "a copy from domain {} to domain {} was given to the Vulkan driver, \
                 which serves domain {vulkan}",
                plan.src_domain, plan.dst_domain
            )));
        }
        if plan.src_page_ids.len() != plan.dst_page_ids.len() {
            return Err(Failed::Vulkan(format!(
                "{} source pages and {} destination pages",
                plan.src_page_ids.len(),
                plan.dst_page_ids.len()
            )));
        }
        let shape = self.shape;
        let check = |page: u32, offset: u32, what: &str| -> Result<(), Failed> {
            if page >= shape.pages || offset >= shape.page_size {
                return Err(Failed::Vulkan(format!(
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

    /// Give every allocation back.
    ///
    /// Not [`Drop`]: freeing a Vulkan buffer needs the device that made it,
    /// and a `Drop` that cannot reach one either stores a handle it must not
    /// outlive or leaks. Stated as a call so the leak is the caller's to
    /// avoid rather than this module's to hide.
    pub fn close(mut self, device: &Device) {
        // One implementation, because two ways to free the same buffers is
        // one way to free half of them. `release` exists for the owner that
        // frees during `Drop` and cannot consume itself.
        self.release(device);
    }
}

impl Resolve for Pool {
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

#[cfg(test)]
mod tests {
    use super::{FireNumber, Frame, Request, Shape, Unstageable};

    /// Small enough to walk exhaustively, and no two dimensions equal, so a
    /// transposition of any two of them shows.
    const SMALL: Shape = Shape {
        layers: 2,
        kv_heads: 3,
        head_dim: 5,
        page_size: 4,
        pages: 7,
        bytes: 2,
    };

    /// A shape with a zero page size is refused rather than treated as one.
    ///
    /// The refusal replaced a `.max(1)`, which had been unwitnessed: deleting
    /// it left the whole suite green because nothing had ever staged a zero
    /// page size. What made the clamp wrong rather than merely untested is
    /// that it was one-sided. `Frame::of` would have divided by one while
    /// `FireNumber::KvPageSize` kept answering zero, so the tables and the
    /// shader would have disagreed about where every row lives, and the
    /// append would have piled every request onto the same few slots.
    #[test]
    fn a_page_that_holds_no_slots_is_refused_and_not_rounded_up_to_one() {
        let request = Request {
            positions: vec![0],
            pages: vec![0],
            samples: vec![0],
        };
        let zero = Shape {
            page_size: 0,
            ..SMALL
        };
        assert!(
            matches!(
                Frame::of(zero, std::slice::from_ref(&request)),
                Err(Unstageable::NoSlots)
            ),
            "a zero page size was accepted"
        );
        // The control: the same fire against the same shape with its page
        // size restored has to stage, or the refusal above is being made by
        // something other than the page size.
        assert!(
            Frame::of(SMALL, &[request]).is_ok(),
            "the control fire did not stage"
        );
        // And the number the shader is handed agrees with the number the
        // tables were built from, which is the disagreement the refusal is
        // about.
        assert_eq!(
            SMALL.number(FireNumber::KvPageSize),
            Some(SMALL.page_size),
            "the shader is handed a page size the tables were not built from"
        );
    }

    /// A fire with no rows is refused rather than staged as an empty one.
    ///
    /// Same shape of finding as the page size above and the same origin: a
    /// `.max(1)`, this one rounding the attention-mask tables up to a word,
    /// could be deleted with the suite staying green. A rowless fire is the
    /// failure this crate refuses hardest -- every grid comes out zero, every
    /// dispatch runs nothing, and the step returns `Ok`. It is refused here,
    /// which is the last place it still has a name.
    #[test]
    fn a_fire_with_no_rows_is_refused_and_not_staged_as_an_empty_one() {
        assert!(
            matches!(Frame::of(SMALL, &[]), Err(Unstageable::NoRows)),
            "a fire with no requests was accepted"
        );
        // A request present but contributing nothing is the same emptiness
        // wearing a request, and the one a scheduler actually produces.
        let hollow = Request {
            positions: Vec::new(),
            pages: vec![0],
            samples: Vec::new(),
        };
        assert!(
            matches!(
                Frame::of(SMALL, std::slice::from_ref(&hollow)),
                Err(Unstageable::NoRows)
            ),
            "a fire whose only request contributes no rows was accepted"
        );
        // The control: one row anywhere in the fire is enough, so the refusal
        // is about rows and not about the hollow request being present.
        let real = Request {
            positions: vec![0],
            pages: vec![1],
            samples: vec![0],
        };
        let frame = Frame::of(SMALL, &[hollow, real]).expect("one row is a fire");
        assert_eq!(frame.rows(), 1, "the surviving row was lost");
    }

    /// Every element of the pool is one element of the pool.
    ///
    /// A scatter is only safe if this holds: `kv_write` computes a
    /// destination per invocation and never checks for a collision, so two
    /// distinct positions sharing an index would have one of them silently
    /// overwrite the other. Walked exhaustively over the whole small pool,
    /// which is what a card test cannot afford.
    #[test]
    fn every_element_of_the_cache_has_exactly_one_address() {
        let n = usize::try_from(SMALL.elements()).expect("small");
        let mut seen = vec![false; n];
        for page in 0..SMALL.pages {
            for off in 0..SMALL.page_size {
                for head in 0..SMALL.kv_heads {
                    for at in 0..SMALL.head_dim {
                        let ix = SMALL.slot(page, off, head, at);
                        let ix = usize::try_from(ix).expect("in range");
                        assert!(
                            ix < n,
                            "({page}, {off}, {head}, {at}) is at {ix}, past the {n} the pool holds"
                        );
                        assert!(!seen[ix], "({page}, {off}, {head}, {at}) collides at {ix}");
                        seen[ix] = true;
                    }
                }
            }
        }
        // Onto, not merely into: a layout that left gaps would allocate memory
        // the cache can never use, and one that packed too tightly would have
        // collided above.
        assert!(seen.iter().all(|&s| s), "the layout leaves holes");
    }

    /// The two strides are the only pair that describes the same memory as
    /// [`Shape::slot`].
    ///
    /// `attn/kv_write.comp`'s contiguous half writes
    /// `h * k_head_stride + pos * k_seq_stride + d`; its paged half writes
    /// what `slot` says. Both are
    /// checked on a card over six positions and two heads. Here the same
    /// identity is walked over every position and head the small pool has,
    /// and every other assignment of the two numbers is checked to break it --
    /// the card test can only afford to try the swap.
    #[test]
    fn the_two_stride_numbers_are_the_only_pair_that_agrees_with_slot() {
        let head = SMALL
            .number(FireNumber::KvHeadStride)
            .expect("a head stride") as u64;
        let seq = SMALL.number(FireNumber::KvSeqStride).expect("a seq stride") as u64;
        let contiguous = |h: u64, pos: u64, d: u64, head: u64, seq: u64| h * head + pos * seq + d;
        let slots = SMALL.pages as u64 * SMALL.page_size as u64;
        for pos in 0..slots {
            for h in 0..u64::from(SMALL.kv_heads) {
                for d in 0..u64::from(SMALL.head_dim) {
                    let page = u32::try_from(pos).expect("small") / SMALL.page_size;
                    let off = u32::try_from(pos).expect("small") % SMALL.page_size;
                    let want = SMALL.slot(
                        page,
                        off,
                        u32::try_from(h).expect("small"),
                        u32::try_from(d).expect("small"),
                    );
                    assert_eq!(
                        contiguous(h, pos, d, head, seq),
                        want,
                        "position {pos}, head {h}, channel {d}"
                    );
                }
            }
        }
        // And no other pair does. Anything drawn from the shape's own numbers
        // is a plausible mistake -- the row's comment names one of them -- so
        // each is tried and each has to fail somewhere.
        let candidates = [
            u64::from(SMALL.head_dim),
            SMALL.row(),
            u64::from(SMALL.page_size),
            slots * u64::from(SMALL.head_dim),
            1,
        ];
        for &a in &candidates {
            for &b in &candidates {
                if (a, b) == (head, seq) {
                    continue;
                }
                let agrees = (0..slots).all(|pos| {
                    (0..u64::from(SMALL.kv_heads)).all(|h| {
                        (0..u64::from(SMALL.head_dim)).all(|d| {
                            let page = u32::try_from(pos).expect("small") / SMALL.page_size;
                            let off = u32::try_from(pos).expect("small") % SMALL.page_size;
                            contiguous(h, pos, d, a, b)
                                == SMALL.slot(
                                    page,
                                    off,
                                    u32::try_from(h).expect("small"),
                                    u32::try_from(d).expect("small"),
                                )
                        })
                    })
                });
                assert!(
                    !agrees,
                    "a head stride of {a} and a sequence stride of {b} also describe the cache, \
                     so the pair this driver states is not forced"
                );
            }
        }
    }

    /// Two requests, one prefill and one decode, and every table follows.
    #[test]
    fn a_frame_states_each_row_once_and_puts_it_in_its_own_requests_page() {
        // Non-contiguous, descending, and not starting at zero on either
        // request: a builder that ignored the page list and used the position
        // directly would agree with none of this.
        let requests = [
            Request {
                positions: (0..6).collect(),
                pages: vec![5, 2],
                samples: Vec::new(),
            },
            Request {
                positions: vec![4],
                pages: vec![6, 1],
                samples: Vec::new(),
            },
        ];
        let frame = Frame::of(SMALL, &requests).expect("a stageable fire");

        assert_eq!(frame.rows(), 7, "six prompt rows and one decode row");
        assert_eq!(frame.positions, [0, 1, 2, 3, 4, 5, 4]);
        assert_eq!(frame.request_of_token, [0, 0, 0, 0, 0, 0, 1]);
        // The CSR: two requests, so three entries, and the last is the length.
        assert_eq!(frame.kv_page_indices, [5, 2, 6, 1]);
        assert_eq!(frame.kv_page_indptr, [0, 2, 4]);
        // `page_size` is 4, so request 0's positions 0..4 are in its first
        // page and 4..6 in its second, and request 1's position 4 is in ITS
        // second -- page 1, not page 2, which is where a builder that indexed
        // the shared list without the CSR base would have put it.
        assert_eq!(frame.kv_write_page, [5, 5, 5, 5, 2, 2, 1]);
        assert_eq!(frame.kv_write_offset, [0, 1, 2, 3, 0, 1, 0]);

        // No two rows of one fire land in one slot. The append does not check.
        let mut slots: Vec<(u32, u32)> = frame
            .kv_write_page
            .iter()
            .zip(&frame.kv_write_offset)
            .map(|(&p, &o)| (p, o))
            .collect();
        slots.sort_unstable();
        let before = slots.len();
        slots.dedup();
        assert_eq!(before, slots.len(), "two rows of one fire share a slot");
    }

    /// A position past its own request's pages is refused rather than sent.
    ///
    /// This is the check that has nowhere else to happen. The page lists sit
    /// end to end, so one entry past request 0's span IS request 1's first
    /// page -- resident, aligned, and owned by somebody else. The append
    /// writes it, request 1 reads it back as its own history, and no layer
    /// and no fault says anything.
    ///
    /// Measured, not assumed: a table one entry short was handed to a whole
    /// real plan under the validation layer with GPU-AV on, and nothing was
    /// reported. An overrun inside a bound storage buffer is not something
    /// Vulkan checks.
    #[test]
    fn a_row_that_would_reach_into_the_next_requests_pages_is_refused() {
        let requests = [
            Request {
                // 8 needs a third page and this request has two.
                positions: vec![8],
                pages: vec![5, 2],
                samples: Vec::new(),
            },
            Request {
                positions: vec![0],
                pages: vec![6],
                samples: Vec::new(),
            },
        ];
        assert_eq!(
            Frame::of(SMALL, &requests),
            Err(Unstageable::PastItsPages {
                request: 0,
                position: 8,
                pages: 2,
            })
        );
        // And the last position that DOES fit is accepted, so the refusal is
        // about the boundary and not about the request.
        let ok = [Request {
            positions: vec![7],
            pages: vec![5, 2],
            samples: Vec::new(),
        }];
        assert_eq!(
            Frame::of(SMALL, &ok)
                .expect("the last row of the last page")
                .kv_write_page,
            [2]
        );
    }

    /// A page the pool does not have, and a page two requests both claim.
    #[test]
    fn a_frame_refuses_pages_that_are_not_the_pools_or_not_its_own() {
        let past = [Request {
            positions: vec![0],
            pages: vec![SMALL.pages],
            samples: Vec::new(),
        }];
        assert_eq!(
            Frame::of(SMALL, &past),
            Err(Unstageable::NoSuchPage {
                request: 0,
                page: SMALL.pages,
                pages: SMALL.pages,
            })
        );
        let shared = [
            Request {
                positions: vec![0],
                pages: vec![3],
                samples: Vec::new(),
            },
            Request {
                positions: vec![0],
                pages: vec![3],
                samples: Vec::new(),
            },
        ];
        assert_eq!(
            Frame::of(SMALL, &shared),
            Err(Unstageable::SharedPage {
                page: 3,
                first: 0,
                second: 1,
            })
        );
        // The same two requests on different pages are fine, so the refusal
        // is about the sharing.
        let apart = [
            Request {
                positions: vec![0],
                pages: vec![3],
                samples: Vec::new(),
            },
            Request {
                positions: vec![0],
                pages: vec![4],
                samples: Vec::new(),
            },
        ];
        assert!(Frame::of(SMALL, &apart).is_ok());
    }

    /// The CSR a frame builds is the one `kv_write_page` was read through.
    ///
    /// Stated separately because the shaders use both: the append takes
    /// `kv_write_page` directly, and `attn/sdpa_paged.comp` walks
    /// `kv_page_indices[indptr[r] .. indptr[r+1]]`. If those two disagreed,
    /// every fire would append somewhere its own attention does not look.
    #[test]
    fn what_the_append_is_told_is_inside_what_the_attention_will_walk() {
        let requests = [
            Request {
                positions: (0..9).collect(),
                pages: vec![5, 2, 6],
                samples: Vec::new(),
            },
            Request {
                positions: (0..5).collect(),
                pages: vec![1, 4],
                samples: Vec::new(),
            },
        ];
        let frame = Frame::of(SMALL, &requests).expect("a stageable fire");
        for (t, &page) in frame.kv_write_page.iter().enumerate() {
            let r = frame.request_of_token[t] as usize;
            let span = frame.kv_page_indptr[r] as usize..frame.kv_page_indptr[r + 1] as usize;
            let walked = &frame.kv_page_indices[span];
            assert!(
                walked.contains(&page),
                "row {t} appends to page {page} and request {r} walks {walked:?}"
            );
            // And at the entry the position's own division names, not merely
            // somewhere in the span.
            let virt = (frame.positions[t] / SMALL.page_size) as usize;
            assert_eq!(walked[virt], page, "row {t} is at the wrong entry");
        }
    }

    /// A page size the cache is not a multiple of would put a row across the
    /// end of the buffer.
    #[test]
    fn the_buffer_is_exactly_as_big_as_the_addresses_it_has_to_hold() {
        let last = SMALL.slot(
            SMALL.pages - 1,
            SMALL.page_size - 1,
            SMALL.kv_heads - 1,
            SMALL.head_dim - 1,
        );
        assert_eq!(
            last + 1,
            SMALL.elements(),
            "the highest address and the allocation disagree"
        );
        assert_eq!(
            SMALL.layer_bytes(),
            SMALL.elements() * u64::from(SMALL.bytes),
            "bytes and elements disagree about the same buffer"
        );
    }

    /// A pool wider than four billion elements per layer cannot state its own
    /// sequence stride, and says so rather than wrapping.
    ///
    /// `Source::KvSeqStride` reaches the shader through a 32-bit channel --
    /// `PIE_STRIDE` is a `uvec2` whose low half is all the shaders read -- so
    /// a `row()` past `u32::MAX` has nowhere to go. Truncating it would put
    /// every position after the first at a wrong address, on a card, with no
    /// error anywhere.
    #[test]
    fn a_cache_too_wide_to_state_refuses_rather_than_wraps() {
        let wide = Shape {
            kv_heads: 1 << 20,
            head_dim: 1 << 13,
            ..SMALL
        };
        assert!(wide.row() > u64::from(u32::MAX), "the premise");
        assert_eq!(
            wide.number(FireNumber::KvSeqStride),
            None,
            "a stride that does not fit was handed over anyway"
        );
        // The narrow one still answers, so the refusal is about the width and
        // not about the method.
        assert_eq!(
            SMALL.number(FireNumber::KvSeqStride),
            Some(SMALL.kv_heads * SMALL.head_dim)
        );
    }

    /// The readout indices are fire-global and the request states its own.
    ///
    /// The renumbering is the whole reason `samples` is per-request, so it is
    /// asked of a fire whose requests contribute different numbers of rows --
    /// on a fire of equal requests, offsetting by the wrong base is a multiple
    /// of the same number and several wrong answers coincide with the right
    /// one.
    #[test]
    fn a_requests_own_row_becomes_the_fires_row() {
        let requests = [
            // Three rows, reads its middle one. Deliberately not the last, so
            // the default below is not what produces it.
            Request {
                positions: vec![0, 1, 2],
                pages: vec![0],
                samples: vec![1],
            },
            // Two rows, reads both -- so `readouts` is not the request count.
            Request {
                positions: vec![0, 1],
                pages: vec![1],
                samples: vec![0, 1],
            },
            // One row and says nothing, which is the decode default.
            Request {
                positions: vec![7],
                pages: vec![2, 3],
                samples: Vec::new(),
            },
            // FOUR rows and says nothing, which is the prefill default. The
            // decode above cannot tell "the last row" from "row zero" -- it
            // has one row and they are the same index -- so without this the
            // default is only half checked.
            Request {
                positions: vec![0, 1, 2, 3],
                pages: vec![4],
                samples: Vec::new(),
            },
        ];
        let frame = Frame::of(SMALL, &requests).expect("a fire");
        assert_eq!(frame.rows(), 10);
        assert_eq!(frame.sampling_indices, vec![1, 3, 4, 5, 9]);
        assert_eq!(frame.readouts(), 5);
    }

    /// A request cannot read a row it did not contribute.
    ///
    /// The dangerous case, and the reason this is checked at all: index 3 of
    /// the first request is a perfectly valid row of the FIRE -- it is the
    /// second request's first token -- so nothing downstream faults, and the
    /// gather would put another request's hidden state into this one's
    /// distribution.
    #[test]
    fn a_request_cannot_read_out_a_row_that_is_not_its_own() {
        let requests = [
            Request {
                positions: vec![0, 1, 2],
                pages: vec![0],
                samples: vec![3],
            },
            Request::of(vec![0], vec![1]),
        ];
        assert_eq!(
            Frame::of(SMALL, &requests),
            Err(Unstageable::NotItsRow {
                request: 0,
                row: 3,
                rows: 3
            })
        );
        // And the same fire with the row it does own is fine, so the refusal
        // is about the index rather than about the shape.
        let ok = [
            Request {
                positions: vec![0, 1, 2],
                pages: vec![0],
                samples: vec![2],
            },
            Request::of(vec![0], vec![1]),
        ];
        assert_eq!(
            Frame::of(SMALL, &ok).expect("a fire").sampling_indices,
            vec![2, 3]
        );
    }

    /// The rows this frame lowers against produce the count it staged.
    ///
    /// Two ways of saying the same thing meet at the gather: the driver stages
    /// `sampling_indices` and the lowering computes `n_requests`, and the
    /// shader reads the first at every index below the second. They are
    /// computed by different crates from different inputs and nothing has ever
    /// compared them.
    ///
    /// `n_requests` is `max(rows that are not multi-token, rows that sample,
    /// 1)`, so the fires below are chosen to make those three quantities
    /// disagree with each other: a pure decode, a lone prefill whose readout
    /// count is far below its row count, and a mix where the prefill's
    /// readouts exceed the decodes.
    #[test]
    fn the_count_the_lowering_computes_is_the_length_of_the_table_staged() {
        use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
        use model::shared::llama_like::forward::llama_like_metal;
        use model_compiler::lower::{Fire, lower};
        use model_compiler::trace::FireClass;

        let fires: [(&str, Vec<Request>); 3] = [
            (
                "three decodes",
                vec![
                    Request::of(vec![3], vec![0]),
                    Request::of(vec![9], vec![1, 2, 5]),
                    Request::of(vec![0], vec![3]),
                ],
            ),
            (
                "one prefill of five, reading its last",
                vec![Request::of(vec![0, 1, 2, 3, 4], vec![0, 4])],
            ),
            (
                "two decodes and a prefill reading three of its rows",
                vec![
                    Request::of(vec![1], vec![0]),
                    Request {
                        positions: vec![0, 1, 2, 3],
                        pages: vec![1],
                        samples: vec![0, 2, 3],
                    },
                    Request::of(vec![2], vec![2]),
                ],
            ),
        ];
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = llama_like_metal(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeMetalFacts::synthetic(),
                class,
            );
            for (what, requests) in &fires {
                let frame = Frame::of(SMALL, requests).expect("a fire");
                let low = lower(
                    &plan,
                    &frame.seriation(),
                    Fire {
                        captures_across_splits: false,
                    },
                )
                .expect("the text lowers");
                assert_eq!(
                    low.n_requests as usize,
                    frame.readouts(),
                    "{class:?}, {what}: the gather would read {} entries of a table of {}",
                    low.n_requests,
                    frame.readouts()
                );
            }
        }
    }
}

/// The tensors a plan names, one buffer each.
///
/// A driver that answered every weight from one buffer -- which is what
/// [`Pool::stand_in`] does, and what every test in this crate did before this
/// existed -- cannot tell a binder that resolved `layer.3.q_proj` from one
/// that resolved `layer.27.k_proj`. Both get the same memory and compute the
/// same answer. That is not a hypothetical: a plan states 704 weight names for
/// a 0.6B model, and the name is the only thing distinguishing them.
///
/// Separate from [`Pool`] because the two have opposite lifetimes. A pool
/// belongs to a deployment and its tables belong to a fire; weights belong to
/// a MODEL and outlive both.
#[derive(Default)]
pub struct Weights {
    held: BTreeMap<String, Buffer>,
    seam: Option<Buffer>,
}

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
        if let Some(old) = self.held.insert(name.to_owned(), buffer) {
            device.free(old);
        }
        Ok(())
    }

    /// The buffer held under `name`.
    ///
    /// Public so that a caller can assert WHICH buffer a binder chose rather
    /// than only that it chose one. A test that reads the slot the binder
    /// filled and compares it against itself passes for every name.
    #[must_use]
    pub fn at(&self, name: &str) -> Option<&Buffer> {
        self.held.get(name)
    }

    /// Give every held weight and the seam back to the device.
    ///
    /// See [`Pool::release`] for why this is not a `Drop`.
    pub fn release(&mut self, device: &Device) {
        for buffer in std::mem::take(&mut self.held)
            .into_values()
            .chain(self.seam.take())
        {
            device.free(buffer);
        }
    }

    /// One buffer standing in for every value the seam binds by name.
    ///
    /// Unlike the weights these are not distinguished here, because a seam
    /// value is an ACTIVATION -- the observed query, the logits -- and a
    /// driver that has a frame has somewhere real to put them. This exists so
    /// that a caller exercising the weights does not also have to build one.
    ///
    /// # Errors
    ///
    /// [`Failed`] if it does not allocate.
    pub fn seam(&mut self, device: &Device, bytes: u64) -> Result<(), Failed> {
        let buffer = device.buffer(&vec![0u8; usize::try_from(bytes).unwrap_or(0)])?;
        if let Some(old) = self.seam.replace(buffer) {
            device.free(old);
        }
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

    /// Give every buffer back.
    pub fn close(mut self, device: &Device) {
        // See `Pool::close`: one implementation.
        self.release(device);
    }
}

impl Resolve for Weights {
    fn weight(&self, name: &str) -> Option<&Buffer> {
        self.held.get(name)
    }

    fn named(&self, _value: ValueId) -> Option<&Buffer> {
        self.seam.as_ref()
    }
}

/// A model's weights and a deployment's cache, as one resolver.
///
/// Neither half can answer a fire on its own, and until this existed nothing
/// in this crate could. [`Pool`] answers the cache, the tables and the fire's
/// numbers, and answers every weight with a single stand-in buffer;
/// [`Weights`] answers weights by name and knows nothing about a cache. A real
/// plan states both -- 704 weight names and 28 layers of KV for qwen3-0.6B --
/// so every test that fired a whole plan did so against a resolver that was
/// deliberately wrong about one half.
///
/// A borrow of each rather than ownership, and that is the point of the type
/// rather than an implementation detail: they have different LIFETIMES. A
/// model's weights are loaded once and outlive every deployment of it; a pool
/// is sized for one deployment's context and outlives every fire in it. A
/// struct that owned both would tie the weights to the pool, and reopening a
/// pool -- which is what changing the context length is -- would drop them.
///
/// # Where each question goes
///
/// Weights and seam values to the weights; cache, tables and numbers to the
/// pool. There is no overlap and no precedence to get wrong, which is why this
/// is a pair and not a chain of fallbacks: a chain would answer an unknown
/// weight name with the pool's stand-in, and a stand-in of zeros computes an
/// answer rather than refusing.
pub struct Model<'a> {
    /// One buffer per tensor name.
    pub weights: &'a Weights,
    /// The cache, the fire's tables and the fire's numbers.
    pub pool: &'a Pool,
}

impl Resolve for Model<'_> {
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
