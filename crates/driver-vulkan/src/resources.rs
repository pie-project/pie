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
//! `attn/kv_write.slang` writes
//!
//! ```text
//! slot = page[i] * page_size + off[i]
//! at   = slot * (kv_heads * head_dim) + h * head_dim + d
//! ```
//!
//! and `attn/sdpa_paged.slang` reads
//! `(slot * n_kv_heads + kv_head) * head_dim + d_out`, the same expression. Two modules compiled separately
//! from separate sources agree on it, so this file transcribes a fact rather
//! than choosing a convention, and [`Shape::slot`] is where a driver can ask
//! for the arithmetic instead of repeating it.

use crate::binding::{FireNumber, FireTable, Resolve};
use crate::device::{Buffer, Device, Failed};
use model_ir::trace::ValueId;
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
    /// The number `Source::Named(<keys::KvPageSize as keys::Fact>::KEY)` asks for, and the one a statement
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

    /// Bytes ONE page costs across the whole pool.
    ///
    /// Every layer holds the page twice, once in keys and once in values, so
    /// this is what a caller gets back per page it hands over -- and it is
    /// what the Vulkan seam publishes as `elastic_page_bytes`. Not
    /// `layer_bytes` divided by pages, which is one layer's half of it and
    /// would understate the saving by `2 * layers`.
    #[must_use]
    pub const fn page_bytes(&self) -> u64 {
        self.page_size as u64 * self.row() * self.bytes as u64 * self.layers as u64 * 2
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
    /// The two strides are in ELEMENTS: `attn/kv_write.slang` adds them to an
    /// index, not to a byte offset. Which of them is which is fixed by
    /// [`Shape::slot`] and not free -- see
    /// `the_two_stride_numbers_are_the_only_pair_that_agrees_with_slot`.
    #[must_use]
    pub fn number(&self, which: FireNumber) -> Option<u32> {
        match which {
            FireNumber::KvPageSize => Some(self.page_size),
            FireNumber::KvHeadStride => Some(self.head_dim),
            FireNumber::KvSeqStride => u32::try_from(self.row()).ok(),
            // Not a fact about the cache. A shape cannot know the pitch of a
            // rectangle built from one fire's requests, and answering zero
            // here would be a shape claiming a fire states no mask.
            FireNumber::AttentionMaskStride | FireNumber::KvHistoryBucket => None,
        }
    }
}

/// Four allow-bytes to the word, little-endian, the tail zero-filled.
///
/// The narrowing the shader does in reverse: `attn/sdpa_paged.slang` indexes
/// these buffers as bytes and Vulkan hands them over as words, so the packing
/// has to be the same one on both sides. Zero-filling the tail is what makes a
/// rectangle whose byte count is not a multiple of four end in forbidden keys
/// rather than in whatever the last word held.
fn pack_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut word = [0u8; 4];
            word[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(word)
        })
        .collect()
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
    /// One allow-byte row per row this request contributes, or empty.
    ///
    /// A byte per KEY: non-zero admits that key, zero forbids it. Empty is not
    /// a mask of zeros -- it leaves this request's rows with their enable byte
    /// clear, and `attn/sdpa_paged.slang` then applies the causal rule alone. A
    /// mask of zeros would forbid every key and produce a softmax over
    /// nothing.
    ///
    /// Rows may be RAGGED. A guest builds its mask as `[queries, pool_len]`
    /// and the pool it names need not be the widest in the fire; the missing
    /// tail is forbidden either way, by the shader's own `kp >= stride` bound.
    pub mask: Vec<Vec<u8>>,
    /// This request's pages were TRACED by the program, not derived here.
    ///
    /// A device-geometry pass resolves its own paging on the device and states
    /// it; `envelope::fill` reads the statement, translates it, and checks the
    /// stated write target against the same arithmetic `Frame::of` would have
    /// used. What it does NOT get is `Frame::of`'s [`Unstageable::SharedPage`]
    /// refusal, which asks a scheduler question -- "did two requests get placed
    /// on one page" -- of a request the scheduler did not place.
    pub traced: bool,
    /// Where each row WRITES its key and value: `(physical page, offset)`, one
    /// per row of [`Self::positions`], or empty.
    ///
    /// Empty is the derivation: page `position / page_size` of [`Self::pages`],
    /// offset `position % page_size`, which is the placement for every fire the
    /// scheduler pages itself.
    ///
    /// It is NOT the placement when a device-geometry program states its own,
    /// and beam search is where the two part: two lanes of one instance share
    /// the page they forked from and take separate SLOTS inside it, so the
    /// second lane writes offset 2 of a page the division puts it at offset 1
    /// of. The derivation cannot express that -- it reads the offset off the
    /// position, and both lanes are at the same position. Stated, it is exact.
    pub writes: Vec<(u32, u32)>,
}

impl Request {
    /// A request whose last row is read out.
    #[must_use]
    pub fn of(positions: Vec<u32>, pages: Vec<u32>) -> Self {
        Self {
            positions,
            pages,
            samples: Vec::new(),
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
        }
    }

    /// Which of this request's rows are read out, with the default resolved.
    ///
    /// The empty case is the last row and not "no rows", so this is where the
    /// two meanings of an empty vector are separated -- a caller that means no
    /// rows says so by putting a row index out of range, which [`Frame::of`]
    /// refuses.
    pub(crate) fn read(&self) -> Vec<u32> {
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
    /// A request states a write target for a different number of rows than it
    /// contributes.
    ///
    /// One short would leave its last rows on the derivation, which is the
    /// placement the statement exists to correct.
    WriteRows {
        /// Which request.
        request: usize,
        /// How many targets it stated.
        stated: usize,
        /// How many rows it contributes.
        rows: usize,
    },
    /// A stated write target is outside the pool.
    NoSuchSlot {
        /// Which request.
        request: usize,
        /// The page it named.
        page: u32,
        /// The offset within it.
        offset: u32,
        /// How many pages the pool holds.
        pages: u32,
        /// How many slots a page holds.
        slots: u32,
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
    /// A request states a mask for some of its rows and not all of them.
    ///
    /// The rectangle is indexed by the FIRE's row number, so a request that
    /// states three rows of mask for four rows of tokens does not leave the
    /// fourth unmasked -- it shifts every later request's rows one place up
    /// the rectangle and masks them against another request's keys. There is
    /// no partial reading of this that is safe, so it is refused.
    MaskRows {
        /// Which request.
        request: usize,
        /// How many mask rows it stated.
        stated: usize,
        /// How many rows it contributes.
        rows: usize,
    },
    /// A mask row is wider than a `u32` can express as a stride.
    ///
    /// The shader takes the pitch as a `u32` and multiplies it by a row index,
    /// so a pitch that does not fit is a rectangle nobody can address.
    MaskTooWide {
        /// The widest row.
        widest: usize,
    },
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
            Self::WriteRows {
                request,
                stated,
                rows,
            } => write!(
                f,
                "request {request} states {stated} write target(s) for {rows} row(s)"
            ),
            Self::NoSuchSlot {
                request,
                page,
                offset,
                pages,
                slots,
            } => write!(
                f,
                "request {request} writes page {page} offset {offset}, and the pool holds \
                 {pages} page(s) of {slots} slot(s)"
            ),
            Self::NoSlots => write!(f, "the shape says a page holds no slots"),
            Self::NoRows => write!(f, "the fire has no rows"),
            Self::MaskRows {
                request,
                stated,
                rows,
            } => write!(
                f,
                "request {request} states {stated} mask row(s) and contributes {rows}"
            ),
            Self::MaskTooWide { widest } => {
                write!(f, "a mask row of {widest} key(s) has no u32 stride")
            }
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
    /// The dense mask rectangle: one allow-byte per key, `rows * stride` long.
    ///
    /// Empty when no request in the fire states a mask, which is the ordinary
    /// case and the reason [`Self::attention_mask_stride`] is zero there.
    pub attention_mask: Vec<u8>,
    /// One byte per ROW saying whether its rectangle row is to be read.
    ///
    /// Separate from the rectangle because "no mask" and "a mask that forbids
    /// everything" are different fires and the bytes cannot tell them apart:
    /// `attn/sdpa_paged.slang` reads this unconditionally and falls back to the
    /// causal rule where it is clear.
    pub attention_mask_enabled: Vec<u8>,
    /// The rectangle's pitch in keys: the widest mask row in the WHOLE fire.
    ///
    /// One pitch and not one per request, because the rectangle is indexed by
    /// the fire's row number. A request whose own mask is narrower is padded
    /// with forbidding zeros, which is what the shader would do anyway past
    /// its own `kp >= stride` bound.
    pub attention_mask_stride: u32,
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
                frame.kv_page_indices.push(page);
            }
            // The pages this request WRITES: one per row, the page its
            // position lands in. Two requests may NAME one page -- a grafted
            // prefix is read by both and written by neither -- but two that
            // write it would each append over the other, which is what this
            // refusal is for and all it is for.
            //
            // `vulkan_shared_prefix`'s own doc predicted this: "named twice"
            // stops implying "written twice" the day
            // `pipeline::fire::kv::match_prefix` is wired in, and "this
            // driver will refuse a correct plan, in a fault rather than in
            // silence". Narrowed before that day rather than after.
            //
            // And narrowed once more, for the same reason and by measurement:
            // a request whose pages the PROGRAM traced (`Request::traced`) did
            // not have its write target placed by the scheduler -- it stated
            // it, the engine bounded it in `LaunchPlan::validate_kv_writes`,
            // and `envelope::fill` checks the statement against this very
            // arithmetic. Beam search fire 0 is the case: one instance, two
            // lanes, both still the conversation they forked from, both
            // tracing their first page. Both write the same slot because at
            // that moment they ARE the same rows, and refusing it refused
            // beam search and consensus decoding outright.
            for &position in &request.positions {
                if request.traced {
                    break;
                }
                let virt = (position / page_size) as usize;
                let Some(&page) = request.pages.get(virt) else {
                    continue;
                };
                if let Some(&first) = owner.get(&page)
                    && first != r
                {
                    return Err(Unstageable::SharedPage {
                        page,
                        first,
                        second: r,
                    });
                }
                owner.insert(page, r);
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
            if !request.writes.is_empty() && request.writes.len() != request.positions.len() {
                return Err(Unstageable::WriteRows {
                    request: r,
                    stated: request.writes.len(),
                    rows: request.positions.len(),
                });
            }
            for (row, &position) in request.positions.iter().enumerate() {
                // Stated first, derived otherwise. Both are checked against the
                // pool's own shape here rather than trusted, because a page
                // past the pool is another model's memory and an offset past a
                // page is the next page's first rows.
                let (page, offset) = match request.writes.get(row) {
                    Some(&(page, offset)) => {
                        if page >= shape.pages || offset >= page_size {
                            return Err(Unstageable::NoSuchSlot {
                                request: r,
                                page,
                                offset,
                                pages: shape.pages,
                                slots: page_size,
                            });
                        }
                        (page, offset)
                    }
                    None => {
                        let virt = (position / page_size) as usize;
                        let Some(&page) = request.pages.get(virt) else {
                            return Err(Unstageable::PastItsPages {
                                request: r,
                                position,
                                pages: request.pages.len(),
                            });
                        };
                        (page, position % page_size)
                    }
                };
                frame.positions.push(position);
                frame
                    .request_of_token
                    .push(u32::try_from(r).unwrap_or(u32::MAX));
                frame.kv_write_page.push(page);
                frame.kv_write_offset.push(offset);
            }
        }
        frame
            .kv_page_indptr
            .push(u32::try_from(frame.kv_page_indices.len()).unwrap_or(u32::MAX));
        frame.mask_from(requests)?;
        Ok(frame)
    }

    /// Build the mask rectangle, after every row is placed.
    ///
    /// After, and not inside the loop above, because the pitch is the widest
    /// row of the WHOLE fire and no request knows it while its own rows are
    /// being pushed. Two passes over the requests is the price of one pitch.
    ///
    /// # Errors
    ///
    /// [`Unstageable::MaskRows`] for a request that masks some of its rows and
    /// not all of them, [`Unstageable::MaskTooWide`] for a pitch with no
    /// `u32`.
    fn mask_from(&mut self, requests: &[Request]) -> Result<(), Unstageable> {
        let mut widest = 0usize;
        for (r, request) in requests.iter().enumerate() {
            if request.mask.is_empty() {
                continue;
            }
            if request.mask.len() != request.positions.len() {
                return Err(Unstageable::MaskRows {
                    request: r,
                    stated: request.mask.len(),
                    rows: request.positions.len(),
                });
            }
            widest = widest.max(request.mask.iter().map(Vec::len).max().unwrap_or(0));
        }
        if widest == 0 {
            return Ok(());
        }
        let stride = u32::try_from(widest).map_err(|_| Unstageable::MaskTooWide { widest })?;
        let rows = self.positions.len();
        let mut bytes = vec![0u8; rows * widest];
        let mut enabled = vec![0u8; rows];
        let mut at = 0usize;
        for request in requests {
            for (j, row) in request.mask.iter().enumerate() {
                enabled[at + j] = 1;
                let into = (at + j) * widest;
                // Shorter than the pitch is the ragged case and the zeros are
                // correct: a key this row's mask does not mention is a key it
                // does not see, and the shader forbids it either way -- a zero
                // byte within the rectangle, `kp >= stride` past it.
                bytes[into..into + row.len()].copy_from_slice(row);
            }
            at += request.positions.len();
        }
        self.attention_mask = bytes;
        self.attention_mask_enabled = enabled;
        self.attention_mask_stride = stride;
        Ok(())
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
    /// How many table buffers [`Pool::state`] has allocated, ever.
    ///
    /// The saving in `state` stated as a number a test can read. A pool that
    /// reallocated every table every step would answer identically and only
    /// be slower, and on a shared box a duration measures the neighbours --
    /// so the claim is counted. It is also not only a matter of speed:
    /// `maxMemoryAllocationCount` is a hard device ceiling, which is the
    /// argument `Device::allocations` makes at greater length.
    restaged: u32,
    /// The pitch of the mask rectangle [`Pool::stage`] last staged.
    ///
    /// On the POOL and not on [`Shape`], because it is a fact about one fire
    /// rather than about the cache: two fires of the same pool mask against
    /// different pool lengths. Zero means the last fire stated no mask, which
    /// is what a row reading `Source::Named(<keys::AttentionMaskStride as keys::Fact>::KEY)` needs to see so the
    /// shader takes its causal path.
    mask_stride: u32,
    /// One past the largest position the last staged fire attends from,
    /// rounded up to a power of two.
    ///
    /// On the pool for the same reason `mask_stride` is: a fact about one
    /// fire, not about the cache. Zero until a fire is staged, and zero reads
    /// as "one split", which is the single-pass decode this backend has
    /// always fired.
    history_bucket: u32,
}

impl Pool {
    /// How many table buffers this pool has allocated, ever.
    ///
    /// See the field. A steady decode should not move this.
    #[must_use]
    pub fn restaged(&self) -> u32 {
        self.restaged
    }

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
        let bytes = shape.layer_bytes();
        // Filled BY THE DEVICE, one `vkCmdFillBuffer` a layer-half, rather
        // than by uploading a zeroed `Vec` of `layer_bytes` to each of them.
        //
        // The upload version was correct and cost the whole cache in bus
        // traffic: measured on a 28-layer, 512-page pool, opening it wrote
        // **939 MB and took 162 ms** -- and a serving pool is sized to fill
        // the card, so on this 24 GB 4090 that is seconds of startup spent
        // sending zeros to memory that can write them itself. `Pool::resize`
        // three hundred lines below already zeroes its new tail with
        // `Device::zero`; this is the same call at the same job, and the two
        // being different was the whole defect.
        let zeroed = |device: &Device| -> Result<crate::device::Buffer, Failed> {
            let b = device.empty(bytes)?;
            device.zero(&b, 0, bytes).inspect_err(|_| device.free(b))?;
            Ok(b)
        };
        let mut keys = Vec::with_capacity(shape.layers as usize);
        let mut values = Vec::with_capacity(shape.layers as usize);
        // Freed on the way out of a partial failure: an allocator that leaks
        // the layers it did get is an allocator whose second call fails for a
        // reason that has nothing to do with the second call.
        for _ in 0..shape.layers {
            match zeroed(device) {
                Ok(b) => keys.push(b),
                Err(e) => {
                    for b in keys.into_iter().chain(values) {
                        device.free(b);
                    }
                    return Err(e);
                }
            }
            match zeroed(device) {
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
            restaged: 0,
            mask_stride: 0,
            history_bucket: 0,
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
    /// # Why the halving
    ///
    /// [`Self::resize`] takes every new buffer BEFORE it frees an old one --
    /// deliberately, so a failed growth leaves the pool intact -- so the peak
    /// a growth needs is both sizes at once. A ceiling that ignored that
    /// would admit a frame the resize then refuses, with nothing changed and
    /// the scheduler none the wiser about why.
    ///
    /// # What it is not
    ///
    /// A promise. The heap is shared with this model's weights, with every
    /// other process on the device, and with whatever the allocator has
    /// fragmented, so a growth under this number can still fail. Too generous
    /// is the safe direction: it turns a permanent refusal into a retried one
    /// rather than the reverse.
    #[must_use]
    pub fn ceiling(&self, device: &Device) -> u32 {
        // Both KV halves, every layer.
        let per_page = (self.shape.page_size as u64)
            .saturating_mul(self.shape.row())
            .saturating_mul(self.shape.bytes as u64)
            .saturating_mul(self.shape.layers as u64)
            .saturating_mul(2);
        if per_page == 0 {
            return u32::MAX;
        }
        let held = per_page.saturating_mul(u64::from(self.shape.pages));
        u32::try_from(device.budget().saturating_add(held) / per_page / 2).unwrap_or(u32::MAX)
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
    /// # What it costs
    ///
    /// The bytes that SURVIVE, moved on the device, plus a zero-fill of the
    /// tail a grow adds. Nothing crosses the bus and no host memory is held.
    ///
    /// It did not start there. The first version read every layer's whole old
    /// buffer down to host memory and wrote a fresh one back up, which is the
    /// pool's size TWICE across a bus, through a mapping that reads at ten
    /// megabytes a second because mappable VRAM is write-combined. Measured
    /// at 256 pages of qwen3-0.6b, dropping ONE page took 2.77 s and dropping
    /// a hundred and twenty-six took 0.74 s -- the DEEPER cut was cheaper,
    /// because the destination it filled was smaller. The cheapest trim that
    /// pool offered was the largest one, which is the opposite of what a trim
    /// task is for.
    ///
    /// `vkCmdCopyBuffer` inverts that into the shape a caller expects: the
    /// charge follows what is kept, so a shallow trim is cheap and a deep one
    /// is cheaper still, and neither pays for the delta.
    /// `a_deep_trim_is_not_cheaper_than_a_shallow_one` pins the inversion
    /// itself rather than any number, since the numbers are a card's.
    ///
    /// # Peak memory, and why a shrink is not a grow
    ///
    /// A GROW takes every new buffer before freeing any old one, so it peaks
    /// at both sizes at once and either wholly happens or wholly does not. A
    /// pool that half-resized would have some layers at the new page count
    /// and some at the old, and `Shape::slot` would index every one of them
    /// wrongly.
    ///
    /// A SHRINK cannot afford that. The reason to shrink is that memory is
    /// short, and a trim that first needs the whole pool again is a trim that
    /// fails exactly when it is wanted. So a shrink goes layer by layer --
    /// take the smaller buffer, move what survives, free the larger one --
    /// and peaks at the old pool plus ONE layer of the new. It is monotonic
    /// after the first step, because each step frees more than the next one
    /// takes, so the allocation that could fail is the first, before anything
    /// has moved.
    ///
    pub fn resize(&mut self, device: &Device, pages: u32) -> Result<(), Failed> {
        if pages == 0 {
            return Err(Failed::Vulkan(
                "a cache of zero pages cannot hold a conversation".to_string(),
            ));
        }
        if pages == self.shape.pages {
            return Ok(());
        }
        let mut resized = self.shape;
        resized.pages = pages;
        let kept = self.shape.pages.min(pages) as u64
            * self.shape.page_size as u64
            * self.shape.row()
            * self.shape.bytes as u64;
        let bytes = resized.layer_bytes();

        // A shrink migrates in place, one layer at a time, so the pool never
        // needs the whole of itself again to give part of it back. Safe to do
        // in place BECAUSE it is a shrink: the first allocation is the only
        // one that can meaningfully fail, and it fails before anything moves.
        if pages < self.shape.pages {
            let layers = self.keys.len();
            for index in 0..layers + self.values.len() {
                let old = if index < layers {
                    &self.keys[index]
                } else {
                    &self.values[index - layers]
                };
                let fresh = device.empty(bytes)?;
                if let Err(e) = device.copy_between(old, 0, &fresh, 0, kept) {
                    device.free(fresh);
                    // Layers before this one are already at the new size and
                    // this one is not, so the pool's shape no longer
                    // describes it. Say so rather than report a failure a
                    // caller would read as "nothing happened".
                    return Err(Failed::Vulkan(format!(
                        "a shrink to {pages} pages stopped at layer buffer {index} of \
                         {}: {e}. The pool is no longer one shape and must be rebuilt",
                        layers + self.values.len()
                    )));
                }
                let old = if index < layers {
                    std::mem::replace(&mut self.keys[index], fresh)
                } else {
                    std::mem::replace(&mut self.values[index - layers], fresh)
                };
                device.free(old);
            }
            self.shape = resized;
            return Ok(());
        }

        // A grow takes everything before it gives anything back, so a failure
        // leaves the pool exactly as it was.
        let mut fresh = Vec::with_capacity(self.keys.len() + self.values.len());
        for old in self.keys.iter().chain(&self.values) {
            let made = (|| {
                let new = device.empty(bytes)?;
                // The tail a grow adds is zeroed, because the pages in it are
                // read before they are written -- `sdpa_paged` reads a whole
                // page and lets `kv_len` decide what counts -- and a fresh
                // Vulkan allocation holds whatever was there.
                if let Err(e) = device
                    .copy_between(old, 0, &new, 0, kept)
                    .and_then(|()| device.zero(&new, kept, bytes - kept))
                {
                    device.free(new);
                    return Err(e);
                }
                Ok(new)
            })();
            match made {
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
        self.shape = resized;
        Ok(())
    }

    /// One buffer per layer, holding that layer's KEYS.
    ///
    /// Read-only, and for a caller that needs to CHECK the cache rather than
    /// dispatch against it: [`Shape::slot`] says where a row lives and this
    /// is what it lives in. Nothing in the fire path takes them this way --
    /// `turns::Serving::step` binds them by handle from inside the pool -- so
    /// handing them out cannot make a descriptor outlive its step.
    #[must_use]
    pub fn keys(&self) -> &[Buffer] {
        &self.keys
    }

    /// One buffer per layer, holding that layer's VALUES. See [`Pool::keys`].
    #[must_use]
    pub fn values(&self) -> &[Buffer] {
        &self.values
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
        // OVER THE OLD BUFFER when it is exactly the right size, which for a
        // server it almost always is: a conversation decoding states the same
        // one row and the same page count for eight tokens at a stretch, and
        // then one table grows by one word.
        //
        // Worth caring about because the nine tables were nine
        // `vkAllocateMemory` calls and nine frees EVERY STEP -- measured as
        // 1.30 ms of an 8.1 ms decode, second only to the dispatches
        // themselves. Writing into the mapping instead is a memcpy of a few
        // hundred bytes into ReBAR, which is the direction that mapping is
        // fast in (see `Device::read_at` for the direction it is not).
        //
        // EXACTLY, and not "big enough". A larger buffer would be bound
        // `whole` with the previous fire's numbers in its tail, and while no
        // shader here reads past the extent it was pushed, `Device::read` of
        // a table would then answer with that tail -- so a test reading a
        // table back would be reading a different object than the one the
        // fire was given. Growing by reallocating keeps the buffer's size and
        // the table's length the same fact.
        //
        // Safe against the GPU because `Serving::once` waits on the fire's
        // fence before it returns, so nothing is reading these tables when
        // the next step writes them. That was already true of the free below,
        // which would otherwise have been unmapping memory in flight.
        if let Some(old) = self.tables.get(&which)
            && old.size() == bytes.len() as u64
        {
            return device.write(old, &bytes);
        }
        self.restaged += 1;
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
    /// The attention mask goes in too, as zeros: `attn/sdpa_paged.slang` reads
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
        //
        // Zeros only while the fire states no mask. A fire that states one
        // ships its own rectangle here, and `Pool::number` hands the shader
        // the pitch that indexes it -- the two travel together because a
        // rectangle read at another fire's pitch is a row masked against
        // another row's keys.
        let bytes = frame.rows().div_ceil(4);
        if frame.attention_mask_stride == 0 {
            self.state(device, FireTable::AttentionMask, &vec![0; bytes])?;
            self.state(device, FireTable::AttentionMaskEnabled, &vec![0; bytes])?;
        } else {
            self.state(
                device,
                FireTable::AttentionMask,
                &pack_bytes(&frame.attention_mask),
            )?;
            self.state(
                device,
                FireTable::AttentionMaskEnabled,
                &pack_bytes(&frame.attention_mask_enabled),
            )?;
        }
        self.mask_stride = frame.attention_mask_stride;
        // Rounded UP to a power of two, and that is not a detail: this number
        // decides a grid, the grid is recorded, and `crate::replay`
        // re-submits the recording across decode steps. An exact history
        // would change every token and re-plan every token. See
        // `FireNumber::KvHistoryBucket`.
        let longest = frame.positions.iter().copied().max().unwrap_or(0);
        self.history_bucket = longest.saturating_add(1).next_power_of_two();
        Ok(())
    }

    /// Give the pool the flash decode's scratch, sized for `floats`.
    ///
    /// Reallocated only when the size CHANGES, which for a steady decode is
    /// never: the split count moves at a power-of-two history boundary and
    /// the row and head counts do not move at all. That matters beyond the
    /// allocation cost, because `crate::replay::Key` carries the device's
    /// allocation and free counts -- a pool that reallocated this every step
    /// would invalidate the recorded command buffer every step.
    ///
    /// Not zeroed. Every workgroup of the split pass writes its own whole
    /// entry before the fold reads any of them, so there is nothing in here
    /// a fire can observe from the fire before it.
    ///
    /// # Errors
    ///
    /// [`Failed`] if the allocation does not fit.
    pub fn partials(&mut self, device: &Device, floats: u64) -> Result<(), Failed> {
        let bytes = floats.max(1) * 4;
        if let Some(old) = self.tables.get(&FireTable::AttnPartials)
            && old.size() >= bytes
        {
            return Ok(());
        }
        self.restaged += 1;
        let buffer = device.empty(bytes)?;
        if let Some(old) = self.tables.insert(FireTable::AttnPartials, buffer) {
            device.free(old);
        }
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
    /// one layer is one copy of `page_size * row()` elements. A row range
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
    /// rows out contiguously, so both are one copy per layer per side.
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
        match which {
            // The one number that is a fact about the FIRE and not about the
            // cache, so it is answered here rather than delegated.
            FireNumber::AttentionMaskStride => Some(self.mask_stride),
            FireNumber::KvHistoryBucket => Some(self.history_bucket),
            which => self.shape.number(which),
        }
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

    /// One page costs what the whole pool costs, divided by its pages.
    ///
    /// `Shape::page_bytes` is what the Vulkan seam publishes as
    /// `elastic_page_bytes`, so it is a NUMBER THE ENGINE ACTS ON: the trim
    /// task converts a recurrent-state high water into pages with it, and an
    /// answer that forgot a factor would understate every saving this pool
    /// reports. It is stated against `layer_bytes`, which the pool itself
    /// allocates with, rather than against a second copy of the arithmetic --
    /// the two agreeing is the whole claim.
    ///
    /// The factor a hand-written version drops is the TWO: every layer holds
    /// each page once in keys and once in values.
    #[test]
    fn a_page_is_the_pool_divided_by_its_pages() {
        for shape in [
            SMALL,
            Shape { pages: 1, ..SMALL },
            Shape {
                layers: 28,
                kv_heads: 8,
                head_dim: 128,
                page_size: 16,
                pages: 256,
                bytes: 2,
            },
        ] {
            let whole = shape.layer_bytes() * u64::from(shape.layers) * 2;
            assert_eq!(
                shape.page_bytes() * u64::from(shape.pages),
                whole,
                "a page of {shape:?} does not tile the pool"
            );
        }
    }

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
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
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
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
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
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
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
    /// `attn/kv_write.slang`'s contiguous half writes
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

    /// A window reaches the rectangle as the window, at the fire's own pitch.
    ///
    /// The rows here are `contrastive-decoding`'s own shape with its window at
    /// 1: each query sees one key, its own. That mask was measured all the way
    /// to `Pool::stage` on a 4090 and reached it intact, which is what says the
    /// staging is not the reason the inferlet answers a window of 1 and a
    /// window of 100000 with the same eight tokens -- see
    /// `tests/gpu/tests/vulkan_padded_causal_mask.rs`.
    ///
    /// The pitch is the FIRE's widest row and not each request's own, because
    /// one rectangle is bound for the whole fire and `sdpa_paged.slang` reads
    /// `attention_mask[row * stride + key]`. A per-request pitch would read
    /// every later row against the wrong keys.
    #[test]
    fn a_windowed_row_is_staged_as_that_window_and_padded_to_the_fires_pitch() {
        let window = |query: usize, keys: usize| {
            (0..keys)
                .map(|key| u8::from(key == query))
                .collect::<Vec<u8>>()
        };
        let requests = [
            // The WIDEST request first, so that "the last request's pitch" and
            // "the fire's widest" are different answers. They were the same
            // when this fixture had them the other way round, which a mutation
            // of `mask_from` walked straight through.
            Request {
                positions: vec![0],
                pages: vec![0],
                samples: Vec::new(),
                // One row of SEVEN, which is the fire's pitch.
                mask: vec![window(0, 7)],
                traced: false,
                writes: Vec::new(),
            },
            Request {
                positions: vec![1, 2],
                pages: vec![1],
                samples: Vec::new(),
                // Two rows of FIVE keys: the narrower request, and the one
                // that is padded rather than read past.
                mask: vec![window(1, 5), window(2, 5)],
                traced: false,
                writes: Vec::new(),
            },
        ];
        let frame = Frame::of(SMALL, &requests).expect("a stageable fire");

        assert_eq!(frame.attention_mask_stride, 7, "the fire's widest row");
        assert_eq!(
            frame.attention_mask_enabled,
            [1, 1, 1],
            "every row states a mask, so every row's rule is the mask's"
        );
        assert_eq!(
            frame.attention_mask,
            [
                1, 0, 0, 0, 0, 0, 0, //
                0, 1, 0, 0, 0, 0, 0, //
                0, 0, 1, 0, 0, 0, 0,
            ],
            "each row's own window, padded with the forbidding byte"
        );
    }

    /// A row count that is not the request's is refused, not padded.
    #[test]
    fn a_mask_is_stated_for_every_row_of_a_request_or_none() {
        let short = [Request {
            positions: vec![0, 1],
            pages: vec![0],
            samples: Vec::new(),
            mask: vec![vec![1, 0]],
            traced: false,
            writes: Vec::new(),
        }];
        assert_eq!(
            Frame::of(SMALL, &short).err(),
            Some(Unstageable::MaskRows {
                request: 0,
                stated: 1,
                rows: 2
            })
        );
    }

    /// No mask is not a mask of zeros: the enable byte stays clear.
    ///
    /// The distinction is the whole difference between "the causal rule alone"
    /// and "a softmax over nothing", and it lives in one byte per row.
    #[test]
    fn a_fire_that_states_no_mask_stages_no_rectangle_and_no_pitch() {
        let plain = [Request::of(vec![0, 1], vec![0])];
        let frame = Frame::of(SMALL, &plain).expect("a stageable fire");
        assert_eq!(frame.attention_mask_stride, 0);
        assert!(frame.attention_mask.is_empty());
        assert!(frame.attention_mask_enabled.is_empty());
    }

    /// A stated write target is used, and it is one the derivation cannot say.
    ///
    /// Both requests are at position 1 of the same page, which is beam search's
    /// own shape: two lanes forked from one prefix, sharing the page and taking
    /// separate slots inside it. The derivation reads the offset off the
    /// position, so it can only ever name slot 1 for both -- and `Frame::of`
    /// refused the pair for sharing a page, which is how this was found.
    #[test]
    fn a_stated_write_target_places_two_lanes_of_one_page_in_separate_slots() {
        let requests = [
            Request {
                positions: vec![1],
                pages: vec![3],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: true,
                writes: vec![(3, 1)],
            },
            Request {
                positions: vec![1],
                pages: vec![3],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: true,
                writes: vec![(3, 2)],
            },
        ];
        let frame = Frame::of(SMALL, &requests).expect("two lanes of one page");
        assert_eq!(frame.kv_write_page, [3, 3], "both lanes share the page");
        assert_eq!(
            frame.kv_write_offset,
            [1, 2],
            "and the second lane takes the slot it stated, not the one its \
             position divides to"
        );

        // The same pair with the statement dropped is the refusal that was
        // there before: without it, nothing distinguishes the two lanes.
        let derived: Vec<Request> = requests
            .iter()
            .cloned()
            .map(|mut r| {
                r.writes = Vec::new();
                r.traced = false;
                r
            })
            .collect();
        assert_eq!(
            Frame::of(SMALL, &derived).err(),
            Some(Unstageable::SharedPage {
                page: 3,
                first: 0,
                second: 1
            })
        );
    }

    /// The page-sharing refusal asks a question a traced request has answered.
    #[test]
    fn a_traced_request_is_not_refused_for_sharing_a_page() {
        let shared = |traced| {
            [
                Request {
                    positions: vec![0],
                    pages: vec![3],
                    samples: Vec::new(),
                    mask: Vec::new(),
                    traced,
                    writes: Vec::new(),
                },
                Request {
                    positions: vec![0],
                    pages: vec![3],
                    samples: Vec::new(),
                    mask: Vec::new(),
                    traced,
                    writes: Vec::new(),
                },
            ]
        };
        assert!(
            Frame::of(SMALL, &shared(true)).is_ok(),
            "the program placed these, and the engine bounded them"
        );
        assert!(
            Frame::of(SMALL, &shared(false)).is_err(),
            "and a scheduler-placed pair on one page is still two appends over \
             each other"
        );
    }

    /// A statement for the wrong number of rows is refused, not padded.
    #[test]
    fn a_write_target_is_stated_for_every_row_or_none() {
        let short = [Request {
            positions: vec![0, 1],
            pages: vec![3],
            samples: Vec::new(),
            mask: Vec::new(),
            traced: true,
            writes: vec![(3, 0)],
        }];
        assert_eq!(
            Frame::of(SMALL, &short).err(),
            Some(Unstageable::WriteRows {
                request: 0,
                stated: 1,
                rows: 2
            })
        );
    }

    /// A stated target outside the pool is refused by both of its numbers.
    #[test]
    fn a_stated_write_target_is_checked_against_the_pools_own_shape() {
        let far = |page, offset| {
            [Request {
                positions: vec![0],
                pages: vec![3],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: true,
                writes: vec![(page, offset)],
            }]
        };
        assert_eq!(
            Frame::of(SMALL, &far(SMALL.pages, 0)).err(),
            Some(Unstageable::NoSuchSlot {
                request: 0,
                page: SMALL.pages,
                offset: 0,
                pages: SMALL.pages,
                slots: SMALL.page_size
            }),
            "a page past the pool is another model's memory"
        );
        assert_eq!(
            Frame::of(SMALL, &far(3, SMALL.page_size)).err(),
            Some(Unstageable::NoSuchSlot {
                request: 0,
                page: 3,
                offset: SMALL.page_size,
                pages: SMALL.pages,
                slots: SMALL.page_size
            }),
            "and an offset past a page is the next page's first rows"
        );
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
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            Request {
                positions: vec![4],
                pages: vec![6, 1],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
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
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            Request {
                positions: vec![0],
                pages: vec![6],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
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
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
        }];
        assert_eq!(
            Frame::of(SMALL, &ok)
                .expect("the last row of the last page")
                .kv_write_page,
            [2]
        );
    }

    /// Two requests may share a page they only READ, and not one they write.
    ///
    /// A grafted prefix is read by both and written by neither, which is the
    /// case `vulkan_shared_prefix`'s doc says this refusal gets wrong the day
    /// the prefix probe is wired in. Narrowed before that day.
    #[test]
    fn two_requests_share_a_page_they_read_and_not_one_they_write() {
        // `SMALL` pages four positions, so position 5 lands in virtual page 1
        // and page 0 is read-only for both.
        let lane = |last: u32, pages: Vec<u32>| Request {
            positions: vec![last],
            pages,
            samples: Vec::new(),
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
        };
        assert!(
            Frame::of(SMALL, &[lane(5, vec![1, 2]), lane(5, vec![1, 2])]).is_err(),
            "both write virtual page 1, which is physical page 2"
        );
        let frame = Frame::of(SMALL, &[lane(5, vec![1, 2]), lane(5, vec![1, 3])])
            .expect("physical page 1 is read by both and written by neither");
        assert_eq!(frame.kv_page_indices, vec![1, 2, 1, 3]);
    }

    /// A page the pool does not have, and a page two requests both claim.
    #[test]
    fn a_frame_refuses_pages_that_are_not_the_pools_or_not_its_own() {
        let past = [Request {
            positions: vec![0],
            pages: vec![SMALL.pages],
            samples: Vec::new(),
            mask: Vec::new(),
            traced: false,
            writes: Vec::new(),
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
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            Request {
                positions: vec![0],
                pages: vec![3],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
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
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            Request {
                positions: vec![0],
                pages: vec![4],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
        ];
        assert!(Frame::of(SMALL, &apart).is_ok());
    }

    /// The CSR a frame builds is the one `kv_write_page` was read through.
    ///
    /// Stated separately because the shaders use both: the append takes
    /// `kv_write_page` directly, and `attn/sdpa_paged.slang` walks
    /// `kv_page_indices[indptr[r] .. indptr[r+1]]`. If those two disagreed,
    /// every fire would append somewhere its own attention does not look.
    #[test]
    fn what_the_append_is_told_is_inside_what_the_attention_will_walk() {
        let requests = [
            Request {
                positions: (0..9).collect(),
                pages: vec![5, 2, 6],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            Request {
                positions: (0..5).collect(),
                pages: vec![1, 4],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
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
    /// `Source::Named(<keys::KvSeqStride as keys::Fact>::KEY)` reaches the shader through a 32-bit channel --
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
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            // Two rows, reads both -- so `readouts` is not the request count.
            Request {
                positions: vec![0, 1],
                pages: vec![1],
                samples: vec![0, 1],
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            // One row and says nothing, which is the decode default.
            Request {
                positions: vec![7],
                pages: vec![2, 3],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
            },
            // FOUR rows and says nothing, which is the prefill default. The
            // decode above cannot tell "the last row" from "row zero" -- it
            // has one row and they are the same index -- so without this the
            // default is only half checked.
            Request {
                positions: vec![0, 1, 2, 3],
                pages: vec![4],
                samples: Vec::new(),
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
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
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
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
                mask: Vec::new(),
                traced: false,
                writes: Vec::new(),
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
        use model_ir::trace::FireClass;

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
                        mask: Vec::new(),
                        traced: false,
                        writes: Vec::new(),
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
/// # Why a `HashMap` and not a `BTreeMap`
///
/// It was a `BTreeMap`, and nothing about this store is ordered: the only
/// question ever asked of it is [`Weights::weight`], by exact name, once per
/// weight operand of every rectangle a fire plans.
///
/// A `BTreeMap` answers that in about ten string comparisons, and these are
/// the worst strings to compare: `model.layers.13.self_attn.q_proj.weight`
/// and `model.layers.13.self_attn.k_proj.weight` agree for their first
/// twenty-eight bytes, over nodes the allocator scattered. Measured GPU-free
/// by `tests/planbench.rs`, planning a qwen3-0.6b decode's 452 rectangles
/// against a store of the same 704 names: `bind` is **68 ns a rectangle
/// ordered and 35 ns hashed**, against 15 ns with a store that answers
/// without looking anything up. So more than half of what binding an operand
/// cost was the tree.
#[derive(Default)]
pub struct Weights {
    held: std::collections::HashMap<String, Buffer>,
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
    /// The plan's runtime streams, value id → fire table.
    ///
    /// The no-ask channel's staged half: a text's `positions` is a NAMED
    /// value like a seam's, and this is what tells the two apart at
    /// [`Resolve::named`]. Per plan, because value ids are the plan's own
    /// numbering — the step that builds this `Model` builds it from the plan
    /// it is about to fire.
    pub runtime: &'a crate::runtime::Streams,
}

impl Resolve for Model<'_> {
    fn weight(&self, name: &str) -> Option<&Buffer> {
        self.weights.weight(name)
    }

    fn named(&self, value: ValueId) -> Option<&Buffer> {
        // A runtime STREAM binds the fire's own staged table; everything
        // else named is a seam value and keeps the stand-in the seam sized.
        // The two id populations are disjoint by construction — the trace
        // mints runtime values, the seam publishes its own — so there is no
        // precedence to get wrong, only a lookup that misses.
        if let Some(which) = self.runtime.table_of(value) {
            return Resolve::table(self.pool, which);
        }
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
