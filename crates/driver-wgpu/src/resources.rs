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
use model_ir::trace::ValueId;

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
    /// The number `Source::Named(<keys::KvPageSize as keys::Fact>::KEY)` asks for, and the one a statement cannot
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
            // NOT the pool's to answer. A mask rectangle is as wide as the
            // widest row of the FIRE that supplied it, and a `Shape` outlives
            // every fire it serves. `Pool::number` answers this one from what
            // it staged; answering a stale width here would index a row's mask
            // at the previous fire's pitch.
            FireNumber::AttentionMaskStride => None,
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
    /// Where each of this request's rows writes its KV, when the PROGRAM said
    /// so: `(physical page, offset within it)`, one per row.
    ///
    /// Empty is the ordinary case and means "derive": a row appends at
    /// `pages[position / page_size]`, offset `position % page_size`, which is
    /// what every host and envelope fire wants and what this driver did for
    /// everything until a beam asked for something else. A device-geometry
    /// program binds `W_SLOT`/`W_OFF` and traces its own placement -- forked
    /// lanes take different copies of a page at the SAME position -- and
    /// `envelope::fill` translates those into this.
    pub writes: Vec<Option<(u32, u32)>>,
    /// One row of allow-bytes per query row, or empty for no mask.
    ///
    /// Decoded HERE rather than carried as runs, because this half of the
    /// crate is the portable one and `driver_api::plan::EncodedMask` is a wire
    /// type. `frames::requests_of` does the decoding; this holds the
    /// rectangle.
    ///
    /// Each inner vector is `1` where the key is visible to that row and `0`
    /// where it is not, indexed from key ZERO of the request's history. Rows
    /// may be ragged -- a later row sees more keys -- and [`Frame::of`] pads
    /// to the widest.
    ///
    /// Empty means the request states no mask, which is not the same as a mask
    /// of all zeros: the first leaves `attention_mask_enabled` clear and the
    /// causal rule alone applies, the second forbids everything.
    pub mask: Vec<Vec<u8>>,
    /// Which recurrent slot holds this request's gated-DeltaNet carry.
    ///
    /// [`Frame::of`] writes it into [`Frame::recurrent_slots`] once per ROW,
    /// because that is how the `*_slotted` kernels index it: their grid is
    /// `rows * v_heads` on z and they recover `slot_ids[z / v_heads]`, so the
    /// table is per row and every row of one request names one slot.
    ///
    /// Zero for a model with no recurrent state, where nothing reads it. That
    /// default is safe only because the table is not staged at all when the
    /// deployment holds no pool -- see [`Pool::stage`] -- and it is the reason
    /// a `Request` built by hand does not have to know about a family it does
    /// not use.
    pub slot: u32,
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
            writes: Vec::new(),
            slot: 0,
        }
    }

    /// Which of this request's OWN rows are read out, with the default
    /// resolved.
    ///
    /// Numbered from the request: index 0 is its first row. That is what the
    /// scheduler writes -- measured twice, most recently on the merged tree
    /// (`qo=[0, 92, 93] sidx=[91, 0]`: request 1 spans rows 92..93 and names
    /// `0`, its own) -- and `driver::resolve` pushes `span - 1` from the
    /// request's own row count when no read-out port is bound.
    ///
    /// The empty case is this request's last row and not "no rows", so this is
    /// where the two meanings of an empty vector are separated — a caller that
    /// means no rows says so by putting a row index out of range, which
    /// [`Frame::of`] refuses.
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

/// Little-endian bytes into the `u32` words a storage buffer is written in.
///
/// The shader reads one byte out of a word with `(word >> ((at & 3) * 8)) &
/// 0xff`, which is little-endian, and `from_le_bytes` is the same statement
/// said once here instead of open-coded at each caller.
fn pack_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
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
    /// A request states a write descriptor that does not cover its rows.
    ///
    /// One slot per row or none at all: a table one short would give a row
    /// another row's slot, which is the single failure the descriptor exists
    /// to prevent.
    WriteRows {
        /// Which request.
        request: usize,
        /// How many slots it states.
        writes: usize,
        /// How many rows it has.
        rows: usize,
    },
    /// Two requests in one fire WRITE the same page.
    ///
    /// Not an error the shaders could survive: both would append to it and each
    /// would read the other's rows as its own history.
    ///
    /// Sharing a page to READ is allowed and load-bearing: a beam's lanes are
    /// separate requests of one conversation and they share every page of the
    /// prefix they forked from. Refusing that refused beam search before it
    /// could state what it wanted, which is how the narrower rule was found.
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
    /// A request states a mask for some of its rows but not all of them.
    ///
    /// The rectangle is indexed by the FIRE's row number, so a short mask list
    /// would leave later rows enabled against entries belonging to earlier
    /// ones. The shader's own stride bound cannot see that -- every index is
    /// inside the rectangle -- so it is refused here.
    MaskRows {
        /// How many mask rows the request states.
        stated: usize,
        /// How many rows it contributes.
        rows: usize,
    },
    /// A mask row longer than a `u32` can address.
    MaskTooWide {
        /// The widest row's length.
        widest: usize,
    },
}

impl std::fmt::Display for Unstageable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WriteRows {
                request,
                writes,
                rows,
            } => write!(
                f,
                "request {request} states {writes} write slot(s) for its {rows} row(s)"
            ),
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
            Self::MaskRows { stated, rows } => write!(
                f,
                "a request states {stated} mask rows and contributes {rows}"
            ),
            Self::MaskTooWide { widest } => {
                write!(f, "a mask row of {widest} keys does not fit a u32 stride")
            }
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
    /// Row offsets splitting [`Self::sampling_indices`] per request; one
    /// longer than the request count.
    ///
    /// Recorded rather than re-derived because a request may name SEVERAL
    /// readout rows -- a speculative verifier names one per drafted token --
    /// and without the boundaries there is no telling whose rows are whose.
    /// `sampling_indices` alone was enough only while every request named
    /// exactly one.
    pub sampling_indptr: Vec<u32>,
    /// Which recurrent SLOT each ROW's carry lives in.
    ///
    /// The gated DeltaNet's page table: its `*_slotted` kernels read it to
    /// find their own state, the way a paged attention reads `kv_page_indices`.
    /// EMPTY for every model whose deployment opens no recurrent pool, and an
    /// empty table stages nothing — see [`Pool::stage`].
    ///
    /// # Per ROW, and it was per nothing at all
    ///
    /// One entry per row, parallel to [`Self::request_of_token`], holding the
    /// slot of the request that owns that row. That is the subscript the
    /// kernels use: their grid is `rows * v_heads` on z and each workgroup
    /// recovers `slot_ids[z / v_heads]`.
    ///
    /// **This vector was declared and staged and never written**, which is a
    /// defect with a particular shape: an unwritten table stages as empty, an
    /// empty storage buffer answers every subscript with a clamp instead of a
    /// trap, and so every request in every fire read slot zero. Nothing
    /// refused, every dispatch succeeded, and each fire inherited the carry
    /// the last one left — so the same prompt answered differently every time
    /// it was asked. `driver-wgpu/tests/hybrid_probe.rs` is where that was
    /// caught, by a control that was there for something else.
    ///
    /// The slot is the REQUEST's ([`Request::slot`]), assigned by
    /// [`crate::pages::Book`] beside the pages, because where a conversation's
    /// history lives is one fact and not two.
    pub recurrent_slots: Vec<u32>,
    /// The custom mask, one BYTE per `(row, key)`, packed four to a word.
    ///
    /// `attn/sdpa_paged.wgsl` reads `attention_mask[row * stride + kp]` as a
    /// byte and treats non-zero as "allowed". Empty when no request in the
    /// fire states a mask.
    pub attention_mask: Vec<u32>,
    /// One BYTE per row: non-zero where that row's mask applies.
    ///
    /// Always staged, even with no mask anywhere, because the shader reads it
    /// unconditionally and a slot nobody filled is a slot bound to something
    /// else.
    pub attention_mask_enabled: Vec<u32>,
    /// Bytes between one row's mask entry and the next.
    ///
    /// Zero when the fire carries no mask, which is also what makes the
    /// shader's own bound (`kp >= stride` forbids) refuse every key for an
    /// enabled row with no rectangle behind it -- the safe direction.
    pub attention_mask_stride: u32,
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
                frame.kv_page_indices.push(page);
            }
            // The pages this request WRITES: one per row, the page its
            // position lands in. Two requests may READ one page -- a beam's
            // lanes share the prefix they forked from, and that sharing is
            // the whole economy of a beam -- but two that write it would each
            // append over the other, which is what this refusal is for and
            // all it is for.
            for (i, &position) in request.positions.iter().enumerate() {
                // A STATED slot is not this rule's business.
                //
                // The rule catches a DERIVED collision: this driver's own
                // arithmetic putting two requests in one page, where "both
                // would append to it" is a thing neither asked for. A program
                // that traces its own placement has asked -- a beam's lanes
                // name the same cell before they fork, on purpose, because
                // they are the same conversation writing the same token --
                // and the driver refusing it was refusing a plan, not a bug.
                //
                // What still holds them apart is the descriptor itself: after
                // the fork the lanes state DIFFERENT slots, and `Frame::of`
                // writes where each one said.
                if request.writes.get(i).copied().flatten().is_some() {
                    continue;
                }
                let page = {
                    let virt = (position / page_size) as usize;
                    let Some(&page) = request.pages.get(virt) else {
                        continue;
                    };
                    page
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
            if frame.sampling_indptr.is_empty() {
                frame.sampling_indptr.push(0);
            }
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
            frame
                .sampling_indptr
                .push(u32::try_from(frame.sampling_indices.len()).unwrap_or(u32::MAX));
            if !request.writes.is_empty() && request.writes.len() != request.positions.len() {
                return Err(Unstageable::WriteRows {
                    request: r,
                    writes: request.writes.len(),
                    rows: request.positions.len(),
                });
            }
            for (i, &position) in request.positions.iter().enumerate() {
                // STATED first. A program that traces its own placement is
                // answered with it; everything else derives, which is the
                // same arithmetic this driver has always used.
                let (page, offset) = match request.writes.get(i).copied().flatten() {
                    Some((page, offset)) => (page, offset),
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
                if page >= shape.pages {
                    return Err(Unstageable::NoSuchPage {
                        request: r,
                        page,
                        pages: shape.pages,
                    });
                }
                frame.positions.push(position);
                frame
                    .request_of_token
                    .push(u32::try_from(r).unwrap_or(u32::MAX));
                frame.kv_write_page.push(page);
                frame.kv_write_offset.push(offset);
                // PER ROW, beside `request_of_token`, because that is how the
                // `*_slotted` gated-DeltaNet kernels index it: their grid is
                // `rows * v_heads` on z and each workgroup recovers
                // `slot_ids[z / v_heads]`, so the subscript is a ROW and not a
                // request. A table with one entry per request would be read
                // past its end by every row after the first — and `wgpu`
                // clamps an out-of-bounds storage read rather than trapping,
                // so every token of one prompt would resolve the LAST slot and
                // the fire would look like it worked.
                frame.recurrent_slots.push(request.slot);
            }
        }
        frame
            .kv_page_indptr
            .push(u32::try_from(frame.kv_page_indices.len()).unwrap_or(u32::MAX));

        // The mask rectangle, after every row is placed, because its pitch is
        // the widest row of the WHOLE fire and its row index is the fire's.
        //
        // A request that states a mask must state one per row it contributes:
        // a partial rectangle would leave some rows enabled against entries
        // that belong to another row, which is the failure the shader's own
        // stride bound cannot see.
        let mut widest = 0usize;
        for request in requests {
            if request.mask.is_empty() {
                continue;
            }
            if request.mask.len() != request.positions.len() {
                return Err(Unstageable::MaskRows {
                    stated: request.mask.len(),
                    rows: request.positions.len(),
                });
            }
            widest = widest.max(request.mask.iter().map(Vec::len).max().unwrap_or(0));
        }
        if widest > 0 {
            let stride = u32::try_from(widest).map_err(|_| Unstageable::MaskTooWide { widest })?;
            let rows = frame.positions.len();
            let mut bytes = vec![0u8; rows * widest];
            let mut enabled = vec![0u8; rows];
            let mut at = 0usize;
            for request in requests {
                for (j, _) in request.positions.iter().enumerate() {
                    if let Some(row) = request.mask.get(j) {
                        enabled[at + j] = 1;
                        let into = (at + j) * widest;
                        // Shorter than the pitch is the ragged case and the
                        // zeros are correct: a key this row's mask does not
                        // mention is a key it does not see, and the shader
                        // forbids it either way (`kp >= stride` for the tail
                        // past the rectangle, a zero byte within it).
                        bytes[into..into + row.len()].copy_from_slice(row);
                    }
                }
                at += request.positions.len();
            }
            frame.attention_mask = pack_bytes(&bytes);
            frame.attention_mask_enabled = pack_bytes(&enabled);
            frame.attention_mask_stride = stride;
        }
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
    /// How many rows the fire this pool last staged has.
    ///
    /// Beside `mask_stride` and for its reasons: a per-FIRE number living on
    /// the pool because the pool is what stages a fire's tables, and zero
    /// until one has been.
    rows: u32,
    /// The mask pitch of the fire this pool last staged, in bytes.
    ///
    /// A per-FIRE number living on the pool because the pool is what stages a
    /// fire's tables and what a row's `Source::Named(<keys::AttentionMaskStride as keys::Fact>::KEY)` is
    /// resolved against. Zero until a fire with a mask is staged, and back to
    /// zero for one without -- see `Pool::stage`.
    mask_stride: u32,
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
            rows: 0,
            mask_stride: 0,
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
            (FireTable::RecurrentSlots, &frame.recurrent_slots),
        ] {
            self.state(device, which, words)?;
        }
        // The mask, when the fire carries one, and zeros when it does not.
        //
        // `attn/sdpa_paged.wgsl` reads `attention_mask_enabled[row]`
        // unconditionally, so the enable table is always staged: a slot nobody
        // filled is a slot bound to something else. Zero there is the true
        // answer for causal attention, which is what a fire with no mask is.
        //
        // One byte per row, and a `u32` of zeros is four zero bytes -- the
        // narrowing is the shader's. Rounded up so a fire of one row still gets
        // a word. No `.max(1)`: `Frame::of` refuses a rowless fire by name
        // (`Unstageable::NoRows`), so the round-up cannot reach zero.
        let words = frame.rows().div_ceil(4);
        if frame.attention_mask_stride > 0 {
            self.state(device, FireTable::AttentionMask, &frame.attention_mask)?;
            self.state(
                device,
                FireTable::AttentionMaskEnabled,
                &frame.attention_mask_enabled,
            )?;
        } else {
            self.state(device, FireTable::AttentionMask, &vec![0; words])?;
            self.state(device, FireTable::AttentionMaskEnabled, &vec![0; words])?;
        }
        // Recorded because the SCALAR is the fire's, not the pool's: the row
        // names `Source::Named(<keys::AttentionMaskStride as keys::Fact>::KEY)` and `Pool::number` answers it
        // from here. Set on every stage, including to zero, so a fire with no
        // mask cannot read the previous fire's pitch.
        self.mask_stride = frame.attention_mask_stride;
        self.rows = u32::try_from(frame.rows()).unwrap_or(u32::MAX);
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
        match which {
            // The fire's, from what this pool last staged. `Shape` answers
            // `None` for it and says why.
            FireNumber::AttentionMaskStride => Some(self.mask_stride),
            other => self.shape.number(other),
        }
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

/// What a driver must allocate to run a gated DeltaNet's recurrent stack.
///
/// Every field is a COUNT, not a byte total — the byte totals are the methods,
/// so there is one place that multiplies and one place that can be wrong.
/// Transcribed from `driver-metal`'s `layout::recurrent::Shape`, whose
/// arithmetic is integers and holds on any backend; what differs between them
/// is the memory next door, not this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recurrent {
    /// How many layers of the stack are LINEAR-attention layers.
    ///
    /// Not the stack's depth: a hybrid interleaves full-attention layers that
    /// carry KV pages and no state at all, and those allocate nothing here.
    /// qwen3.5-0.8b is 24 layers of which 18 are linear.
    pub linear_layers: u32,
    /// Conv channel count — the width of the mixed q|k|v bank.
    pub conv_dim: u32,
    /// Conv kernel width, the window's depth in rows.
    pub conv_k: u32,
    /// Value heads.
    pub v_heads: u32,
    /// Value head dim.
    pub v_dim: u32,
    /// Key head dim — the recurrent state's inner extent.
    pub k_dim: u32,
    /// How many requests can hold a seat at once.
    pub slots: u32,
}

/// Both recurrent planes are `f32`, and that is the KERNEL's property.
///
/// `ssm/gdn_prep.wgsl` and `ssm/gdn_core.wgsl` declare `conv_state` and
/// `rstate` as `array<f32>`, so a driver sizing these from a model's
/// `state_elem` — which some texts state as 2, meaning the bf16 a CUDA build
/// uses — would allocate half a slab and index off the end of it on the first
/// slot past zero. `driver-metal` says the same thing about its own `device
/// float*`, which is why this constant is stated rather than derived.
pub const RECURRENT_ELEM_BYTES: u64 = 4;

/// Two conv planes per layer: the one a fire READS and the one it WRITES.
///
/// A count rather than a `+ conv` folded into one expression, because it is
/// the thing a reader doubts. The planes are the same size, the kernel takes
/// both, and the second is not scratch that could be shared between layers —
/// the carry back happens after the whole fire, so every layer's second plane
/// is still live when the next layer runs.
pub const CONV_PLANES: u64 = 2;

impl Recurrent {
    /// Bytes of one slot's conv window — the stride a conv plane is indexed by.
    #[must_use]
    pub const fn conv_bytes_per_slot(&self) -> u64 {
        self.conv_k as u64 * self.conv_dim as u64 * RECURRENT_ELEM_BYTES
    }

    /// Bytes one slot's recurrent state occupies in ONE layer.
    #[must_use]
    pub const fn state_bytes_per_slot(&self) -> u64 {
        self.v_heads as u64 * self.v_dim as u64 * self.k_dim as u64 * RECURRENT_ELEM_BYTES
    }

    /// Bytes of ONE of a layer's two conv planes.
    #[must_use]
    pub const fn conv_bytes_per_layer(&self) -> u64 {
        self.conv_bytes_per_slot() * self.slots as u64
    }

    /// Bytes of one layer's whole recurrent-state plane.
    #[must_use]
    pub const fn state_bytes_per_layer(&self) -> u64 {
        self.state_bytes_per_slot() * self.slots as u64
    }

    /// Bytes one slot costs across the WHOLE stack.
    ///
    /// What a scheduler divides its budget by.
    #[must_use]
    pub const fn bytes_per_slot(&self) -> u64 {
        self.linear_layers as u64
            * (CONV_PLANES * self.conv_bytes_per_slot() + self.state_bytes_per_slot())
    }

    /// Bytes of the entire pool.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.slots as u64 * self.bytes_per_slot()
    }

    /// The same shape with as many slots as `budget` bytes will hold.
    ///
    /// `None` when it will not hold one, which is a refusal rather than a
    /// zero-slot pool: a stack with nowhere to keep its carry cannot serve a
    /// single request, and a pool that reported zero seats would be discovered
    /// at the first fire instead of at open.
    #[must_use]
    pub const fn slots_within(&self, budget: u64) -> Option<Self> {
        let per = self.bytes_per_slot();
        if per == 0 || budget < per {
            return None;
        }
        let slots = budget / per;
        Some(Self {
            slots: slots as u32,
            ..*self
        })
    }
}

/// The memory a [`Recurrent`] shape describes.
///
/// Same cut as [`Shape`] and [`Pool`]: the arithmetic is integers and holds
/// with no adapter, the allocation is here behind `native`.
///
/// THREE planes per linear layer, not two. `conv_state` and `new_conv_state`
/// are separate buffers because the kernel is still reading the old taps while
/// it writes the new ones, and `recurrent_state` is updated in place and is
/// both read and written. An arm that handed the same buffer twice would make
/// a scan read what it had just written.
#[cfg(feature = "native")]
pub struct RecurrentPool {
    shape: Recurrent,
    /// One per linear layer, indexed by the layer's position in the stack.
    ///
    /// Sparse by layer NUMBER: a hybrid's linear layers are interleaved with
    /// full-attention ones, and a kernel asks by the layer it is planning. So
    /// the map is keyed on that number rather than packed, which costs a
    /// lookup and cannot be indexed off by one.
    conv: BTreeMap<u16, Buffer>,
    fresh: BTreeMap<u16, Buffer>,
    state: BTreeMap<u16, Buffer>,
}

#[cfg(feature = "native")]
impl RecurrentPool {
    /// Allocate three planes for each of `layers`.
    ///
    /// ZEROED, and nothing is uploaded to make it so — WebGPU requires a new
    /// buffer's contents to be zero, which [`Pool::open`] relies on for the
    /// same reason. Here the zero is not merely tidy: a carry that came up
    /// holding the previous deployment's state would make the first token of
    /// every request continue a sequence nobody asked about, fluently.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the first allocation that does not fit.
    pub fn open(
        device: &Device,
        shape: Recurrent,
        layers: impl IntoIterator<Item = u16>,
    ) -> Result<Self, Failed> {
        let conv_bytes = shape.conv_bytes_per_layer();
        let state_bytes = shape.state_bytes_per_layer();
        let mut conv = BTreeMap::new();
        let mut fresh = BTreeMap::new();
        let mut state = BTreeMap::new();
        for layer in layers {
            conv.insert(layer, device.zeroed(conv_bytes)?);
            fresh.insert(layer, device.zeroed(conv_bytes)?);
            state.insert(layer, device.zeroed(state_bytes)?);
        }
        Ok(Self {
            shape,
            conv,
            fresh,
            state,
        })
    }

    /// The shape this pool was opened at.
    #[must_use]
    pub const fn shape(&self) -> Recurrent {
        self.shape
    }

    /// Carry the freshly rolled convolution windows back to the plane the
    /// kernels READ, once per layer per fire, over the whole plane.
    ///
    /// `gdn_core` cannot shift its window in place -- `convsilu` reads the taps
    /// while the writeback shifts them, from different workgroups -- so the
    /// shifted window lands in a second plane. The kernels always READ
    /// `conv_state` and always WRITE `new_conv_state`; they never alternate. So
    /// after a fire the live windows are in the wrong plane and somebody has to
    /// bring them back. Nothing did, and the consequence was not a crash: every
    /// fire convolved over the window as it stood one fire ago, which for a
    /// prompt into a fresh seat is genuinely zeros and therefore RIGHT, and for
    /// every continuation after it is one step stale.
    ///
    /// `whether_the_fused_decode_computes_the_step_its_own_operands_imply`
    /// caught it by walking the step twice: the same CPU reference reproduced
    /// the decode to 1.2e-7 when fed `conv_state` and reproduced the prefill to
    /// 1.0e-7 when fed `new_conv_state`. The kernel was faithful to what it was
    /// handed and was handed the stale plane.
    ///
    /// # Why the whole plane, and why not a swap
    ///
    /// Swapping the two binds is the obvious way to avoid the copy and it is
    /// wrong, in a way worth stating because it looks right. A bind is one
    /// address for every row of a batch, while which plane holds a slot's live
    /// window is per SLOT: a request that sat out this fire had nothing written
    /// to the other plane, so after one swap it reads a window one step stale
    /// and after the next, two. Copying the whole plane keeps both planes
    /// identical for every slot the fire did not touch, which is the invariant
    /// a swap breaks. `driver-metal`'s `layout::recurrent` states the same
    /// reasoning and this backend now matches it.
    ///
    /// The whole plane rather than the touched slots because a fire's slots are
    /// scattered, and scattered blits cost more setup than one contiguous copy
    /// of a plane this size.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the copy.
    pub fn carry_back(&self, device: &Device) -> Result<(), Failed> {
        let moves: Vec<(&Buffer, u64, &Buffer, u64)> = self
            .conv
            .iter()
            .filter_map(|(layer, read)| self.fresh.get(layer).map(|wrote| (wrote, 0, read, 0)))
            .collect();
        device.transfer(&moves, self.shape.conv_bytes_per_layer())
    }

    /// The plane a kernel names, or `None` for a layer this pool does not hold.
    ///
    /// A full-attention layer of a hybrid holds none, and answering `None` for
    /// it is correct: nothing should be asking, and a driver that handed back
    /// another layer's carry would be worse than refusing.
    #[must_use]
    pub fn slab(&self, layer: u16, which: &str) -> Option<&Buffer> {
        match which {
            "conv_state" => self.conv.get(&layer),
            "new_conv_state" => self.fresh.get(&layer),
            "recurrent_state" => self.state.get(&layer),
            _ => None,
        }
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
    /// The gated DeltaNet's carry, for a deployment that has one.
    ///
    /// `None` for every model this backend serves today, and a hybrid's arms
    /// then decline by name with `Unplanned::NoSlab` rather than being handed
    /// a null carry — see [`crate::binding::Resolve::slab`] for why that
    /// distinction is not fussiness.
    pub recurrent: Option<&'a RecurrentPool>,
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

    fn slab(&self, layer: u16, which: &'static str) -> Option<&Buffer> {
        self.recurrent?.slab(layer, which)
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

    /// A ragged mask is staged at the FIRE's pitch, and a row nobody masked
    /// is left disabled beside it.
    ///
    /// The one control on the rectangle this driver actually ships. It is the
    /// counterpart of `driver-vulkan`'s
    /// `a_windowed_row_is_staged_as_that_window_and_padded_to_the_fires_pitch`,
    /// whose GPU gate names it as the reason an end-to-end run is not the
    /// control: a driver that accepted masks and dropped them would answer the
    /// same words. Only the bytes say whether the mask arrived.
    ///
    /// Three claims, and each is a way to be wrong:
    ///
    /// * the pitch is the widest row of the WHOLE fire, not of a request. Two
    ///   requests staged at their own pitches would put row 2 of the second
    ///   where the shader looks for row 1, since `row * stride + kp` is one
    ///   arithmetic over one buffer;
    /// * a short row is padded with zeros, because a key its mask does not
    ///   mention is a key it does not see -- the same answer the shader's
    ///   `kp >= stride` bound gives past the rectangle;
    /// * a request that states no mask leaves its rows DISABLED, and a step
    ///   mixes. `attention_mask_enabled[row]` is read per row for exactly
    ///   that, and a fire-wide flag would make one request's window into
    ///   everybody's.
    #[test]
    fn a_ragged_mask_is_staged_at_the_fires_pitch_and_an_unmasked_row_is_disabled() {
        let masked = Request {
            mask: vec![vec![1, 0, 1], vec![1, 1, 0, 0, 1]],
            ..Request::of(vec![0, 1], vec![7])
        };
        let plain = Request::of(vec![0], vec![2]);
        let frame = Frame::of(shape(), &[masked, plain]).expect("a stageable fire");

        assert_eq!(
            frame.attention_mask_stride, 5,
            "the widest row of the fire is five keys, and it is five for every row"
        );
        // Three rows at a pitch of five is fifteen bytes, packed four to a
        // word little-endian, the last word short and zero-filled.
        assert_eq!(
            frame.attention_mask,
            vec![
                u32::from_le_bytes([1, 0, 1, 0]),
                u32::from_le_bytes([0, 1, 1, 0]),
                u32::from_le_bytes([0, 1, 0, 0]),
                u32::from_le_bytes([0, 0, 0, 0]),
            ],
            "row 0 is its three keys then two zeros, row 1 is its five, and \
             row 2 is the unmasked request's"
        );
        assert_eq!(
            frame.attention_mask_enabled,
            vec![u32::from_le_bytes([1, 1, 0, 0])],
            "the two rows that stated a mask are enabled and the third is not"
        );
    }

    /// A request that masks some of its rows and not others is refused.
    ///
    /// A partial rectangle is the one shape that cannot be padded into
    /// correctness. The missing rows would be staged as all-zero AND enabled
    /// -- a row that attends nothing -- or disabled, which is a row that
    /// attends everything; neither is what a guest that skipped them meant,
    /// and the shader cannot tell either from a mask that was meant.
    #[test]
    fn a_request_that_masks_some_of_its_rows_is_refused() {
        let partial = Request {
            mask: vec![vec![1, 1]],
            ..Request::of(vec![0, 1], vec![7])
        };
        assert_eq!(
            Frame::of(shape(), &[partial]).err(),
            Some(Unstageable::MaskRows { stated: 1, rows: 2 })
        );
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

    /// A STATED write slot is where the row is written, and the derivation is
    /// not consulted.
    ///
    /// The whole point of `W_SLOT`/`W_OFF`: a beam's lanes fork and take
    /// different copies of a page at the SAME position, so a target worked
    /// out from the position can only ever name one cell for both.
    #[test]
    fn a_stated_write_slot_is_the_one_written() {
        let shape = shape();
        let mut lane = Request::of(vec![20], vec![4, 5]);
        // Position 20 derives page 5 (virtual 1) offset 4. The program says
        // page 9, offset 11, and that is where it goes.
        lane.writes = vec![Some((9, 11))];
        let frame = Frame::of(shape, &[lane]).expect("stageable");
        assert_eq!(frame.kv_write_page, vec![9]);
        assert_eq!(frame.kv_write_offset, vec![11]);
    }

    /// And a stated slot outside the pool is refused, exactly like a derived
    /// one: the program states LOGICAL pages and `envelope::fill` translates
    /// them, so a number past the pool means the translation went wrong.
    #[test]
    fn a_stated_write_page_past_the_pool_is_refused() {
        let shape = shape();
        let mut lane = Request::of(vec![0], vec![4]);
        lane.writes = vec![Some((shape.pages, 0))];
        assert_eq!(
            Frame::of(shape, &[lane]).err(),
            Some(Unstageable::NoSuchPage {
                request: 0,
                page: shape.pages,
                pages: shape.pages,
            })
        );
    }

    /// A write descriptor that does not cover the request's rows is refused,
    /// because a table one short gives a row another row's slot.
    #[test]
    fn a_write_descriptor_short_of_its_rows_is_refused() {
        let shape = shape();
        let mut lane = Request::of(vec![0, 1, 2], vec![4]);
        lane.writes = vec![Some((4, 0)), Some((4, 1))];
        assert_eq!(
            Frame::of(shape, &[lane]).err(),
            Some(Unstageable::WriteRows {
                request: 0,
                writes: 2,
                rows: 3,
            })
        );
    }

    /// Two requests may STATE the same slot, which the ownership rule leaves
    /// alone: a beam's lanes name the same cell before they fork, on purpose.
    #[test]
    fn two_requests_may_state_the_same_slot() {
        let shape = shape();
        let lane = || {
            let mut r = Request::of(vec![0], vec![7]);
            r.writes = vec![Some((7, 0))];
            r
        };
        let frame = Frame::of(shape, &[lane(), lane()]).expect("the program said so");
        assert_eq!(frame.kv_write_page, vec![7, 7]);
        assert_eq!(frame.kv_write_offset, vec![0, 0]);
    }

    /// A multi-request prefill states its seats ROW BY ROW, in run order.
    ///
    /// This is the table `ssm/gdn_prep.wgsl`'s prompt scan walks: it reads
    /// `slot_ids[t]` at every token and stores its state back and reloads when
    /// the seat changes, so what the rectangle means is "one sequence per
    /// seat, laid end to end". A table with one entry per REQUEST rather than
    /// per row would make the scan re-seat at the wrong tokens; a table that
    /// repeated one seat would make two conversations into one.
    ///
    /// Two requests of unequal length on purpose. Equal lengths would pass
    /// against a builder that pushed a fixed number of entries per request,
    /// which is the mistake worth being able to fail.
    #[test]
    fn a_fire_of_two_conversations_states_a_seat_for_every_row_in_run_order() {
        let shape = shape();
        let mut first = Request::of(vec![0, 1, 2], vec![3]);
        first.writes = vec![Some((3, 0)), Some((3, 1)), Some((3, 2))];
        first.slot = 5;
        let mut second = Request::of(vec![0, 1], vec![4]);
        second.writes = vec![Some((4, 0)), Some((4, 1))];
        second.slot = 2;
        let frame = Frame::of(shape, &[first, second]).expect("two stageable requests");
        assert_eq!(
            frame.recurrent_slots,
            vec![5, 5, 5, 2, 2],
            "the seat table is per ROW and in run order, so a three-token \
             conversation contributes three entries and the next one starts \
             where it ends"
        );
        assert_eq!(
            frame.recurrent_slots.len(),
            frame.kv_write_page.len(),
            "the seat table is the same height as every other per-row table, \
             and a shorter one is read past its end by the rows after it"
        );
    }

    /// Two requests may share a page they only READ, and not one they write.
    ///
    /// A beam's lanes are separate requests of one conversation: they share
    /// every page of the prefix they forked from, and each writes only the
    /// page its own position lands in. Refusing the shared prefix refused beam
    /// search itself.
    #[test]
    fn two_requests_share_a_page_they_read_and_not_one_they_write() {
        let shape = shape();
        // Page size is 16, so position 20 lands in virtual page 1 and page 0
        // is read-only for both lanes.
        let lane = |last: u32| Request::of(vec![last], vec![4, 5]);
        Frame::of(shape, &[lane(20), lane(20)])
            .expect_err("both lanes write virtual page 1, which is physical page 5");

        let shared_prefix_only = [
            Request::of(vec![20], vec![4, 5]),
            Request::of(vec![20], vec![4, 6]),
        ];
        let frame = Frame::of(shape, &shared_prefix_only)
            .expect("physical page 4 is read by both and written by neither");
        assert_eq!(
            frame.kv_page_indices,
            vec![4, 5, 4, 6],
            "the shared prefix page appears once per lane"
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
    /// row and not the fire's -- the numbering the scheduler writes, measured
    /// on its own output.
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

    /// An unstated read-out is still this request's own last row, resolved
    /// against where its rows begin.
    #[test]
    fn an_unstated_readout_is_this_requests_last_row() {
        let shape = shape();
        let frame = Frame::of(
            shape,
            &[
                Request::of(vec![0, 1, 2], vec![1]),
                Request::of(vec![0, 1], vec![9]),
            ],
        )
        .expect("stageable");
        assert_eq!(
            frame.sampling_indices,
            vec![2, 4],
            "the last row of each, in the fire's numbering"
        );
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
