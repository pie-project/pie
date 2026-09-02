//! **THE RECURRENT-STATE DEVICE HALF** — what makes `RsVerb::Buffer` and
//! `RsVerb::FoldBuffered` true on this plane (alto design §6; the CUDA
//! sibling is `engine_cuda::store::rs`).
//!
//! A recurrence is a fold: once a token's activations are folded into the
//! bank there is no cell to take back when a verifier rejects the token. So a
//! speculative window is not folded. Its PRE-RECURRENCE activations — the
//! conv's input rows, the gate projection's rows, the hasher's token ids —
//! are scattered into a page slab this module owns, the recurrence runs over
//! them from a WORK copy of the bank, and the bank itself is written only as
//! of the row the verb names (`commit`). The rows past it are computed, so
//! the window's logits are right, and leave nothing behind.
//!
//! **THE READ PATH.** A round's accepted prefix survives in the slab
//! unfolded; the NEXT round's fire names it as `replay` and this module
//! gathers those rows AHEAD of the fire's own, so the recurrence starts from
//! `folded (+) replay(buffer)` and lands its state exactly after the prefix.
//! That is one fire per speculative round — the shape `discard-buffered`'s
//! doc promises and the CUDA plane refuses by name.
//!
//! **WHAT IS BUFFERED** is discovered from the trace, not declared per
//! family: every op that reads a recurrent state cache has one input operand
//! whose rows are the recurrence's input (`SsmCausalConv1d*::x`,
//! `PleNgramIds*::ids`), and `SsmGdnPrep::ba` is paired to the delta state
//! its gates feed. Both arms of a mixer — the one-row decode op and the
//! chunked prefill op — resolve to ONE plane, keyed by the state cache they
//! feed, because a lane's rows land on one arm this fire and may land on the
//! other next fire and still have to find the same bytes.
//!
//! The extended rows live in a scratch the shell grows on demand
//! ([`Seat::scratch_bytes`]); the page slab is reserved at load, one page
//! slot per recurrent slot, since the runtime's RS store hands out page ids
//! from the same pool the banks are counted in.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine::fire::{FoldLen, RsVerb};
use kernels_metal::Tensor;
use model_ir::{Attention, Def, Dim, Dtype, Operation, Trace, Ty, ValueId};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::store::kv::Paging;

/// Which recurrence input a plane holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// A causal conv's input rows (the GDN qkv, or the PLE's normed rows).
    ConvX,
    /// The `[b | a]` gate projection the delta scan's decay is prepared from.
    Ba,
    /// The token ids the n-gram hasher walks.
    Ids,
    /// KDA's per-channel forget-gate projection (`f`, one plane wide).
    KdaF,
    /// KDA's per-head write-strength projection (`b`, one lane per head).
    KdaB,
}

/// One buffered plane: what it holds and how wide one token's row is.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    /// The state cache (row of `Trace::caches`) this plane's rows feed.
    pub cache: u32,
    pub role: Role,
    pub width: u32,
    pub dtype: Dtype,
    /// Bytes one token writes.
    pub row_bytes: u64,
}

/// One extended-output rectangle a committed launch lands: the conv's
/// output, the prepared gates, the scan's output, the hasher's rows.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub width: u32,
    pub dtype: Dtype,
    pub row_bytes: u64,
}

/// Every plane and region the plan needs, keyed by the operands that read
/// and write them.
#[derive(Debug, Default)]
pub struct Layout {
    pub planes: Vec<Plane>,
    pub regions: Vec<Region>,
    /// Input operand value -> plane.
    pub in_of: HashMap<u32, usize>,
    /// Output operand value -> region.
    pub out_of: HashMap<u32, usize>,
    /// Bytes one token occupies across every plane.
    pub token_bytes: u64,
    /// The widest delta bank one lane's work copy needs, in f32 elements.
    pub work_elems: u64,
}

fn row_of(trace: &Trace, id: ValueId) -> Result<(u32, Dtype, u64)> {
    let decl = trace.values.get(id.0 as usize).ok_or_else(|| Fault::Unbound {
        what: format!("value {}, which the trace does not declare", id.0),
    })?;
    let Ty::Tensor { shape, dtype } = &decl.ty else {
        return Err(Fault::Unbound {
            what: format!("value {}, a struct where the recurrence reads token rows", id.0),
        });
    };
    let width: u64 = shape
        .iter()
        .skip(1)
        .map(|dim| match dim {
            Dim::Const(n) => *n,
            _ => 1,
        })
        .product();
    let element = model_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
        what: format!("value {}, whose element {dtype:?} has no size", id.0),
    })?;
    Ok((
        u32::try_from(width).unwrap_or(u32::MAX),
        *dtype,
        width * element,
    ))
}

fn cache_of(trace: &Trace, state: ValueId) -> Result<u32> {
    match trace.values.get(state.0 as usize).map(|decl| &decl.def) {
        Some(Def::Cache(row)) => Ok(*row),
        _ => Err(Fault::Unbound {
            what: format!("value {}, read as a recurrent state and declared as no cache", state.0),
        }),
    }
}

impl Layout {
    /// The plan's buffered planes, or `None` for a plan with no recurrent
    /// state to buffer.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a gate projection whose scan cannot be found,
    /// or an operand the trace does not type.
    pub fn read(trace: &Trace) -> Result<Option<Layout>> {
        let mut layout = Layout::default();
        let mut plane_keys: HashMap<(u32, Role), usize> = HashMap::new();
        let mut region_keys: HashMap<(u32, &'static str), usize> = HashMap::new();
        let mut gates_cache: HashMap<u32, u32> = HashMap::new();

        let mut plane = |layout: &mut Layout, cache: u32, role: Role, value: ValueId| -> Result<()> {
            let (width, dtype, row_bytes) = row_of(trace, value)?;
            let at = match plane_keys.get(&(cache, role)) {
                Some(&at) => {
                    let have = layout.planes[at];
                    if have.width != width || have.dtype != dtype {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache {cache}'s {role:?} plane, read {width} wide as {dtype:?} by value {} \
                                 and {} wide as {:?} elsewhere",
                                value.0, have.width, have.dtype
                            ),
                        });
                    }
                    at
                }
                None => {
                    layout.planes.push(Plane {
                        cache,
                        role,
                        width,
                        dtype,
                        row_bytes,
                    });
                    plane_keys.insert((cache, role), layout.planes.len() - 1);
                    layout.planes.len() - 1
                }
            };
            layout.in_of.insert(value.0, at);
            Ok(())
        };
        let mut region = |layout: &mut Layout, cache: u32, what: &'static str, value: ValueId| -> Result<()> {
            let (width, dtype, row_bytes) = row_of(trace, value)?;
            let at = match region_keys.get(&(cache, what)) {
                Some(&at) => at,
                None => {
                    layout.regions.push(Region {
                        width,
                        dtype,
                        row_bytes,
                    });
                    region_keys.insert((cache, what), layout.regions.len() - 1);
                    layout.regions.len() - 1
                }
            };
            layout.out_of.insert(value.0, at);
            Ok(())
        };

        for node in &trace.nodes {
            let Operation::Attention(op) = &node.op else {
                continue;
            };
            match op {
                Attention::SsmCausalConv1d { x, state, y, .. }
                | Attention::SsmCausalConv1dChunked { x, state, y, .. } => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::ConvX, *x)?;
                    region(&mut layout, cache, "conv", *y)?;
                }
                Attention::SsmGatedDelta {
                    gates,
                    state,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                }
                | Attention::SsmGatedDeltaChunked {
                    gates,
                    state,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    y,
                    ..
                } => {
                    let cache = cache_of(trace, *state)?;
                    let _ = k_heads;
                    gates_cache.insert(gates.0, cache);
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*v_heads) * u64::from(*v_dim) * u64::from(*k_dim));
                }
                Attention::SsmKdaStep {
                    f,
                    b,
                    state,
                    heads,
                    head_dim,
                    y,
                    ..
                }
                | Attention::SsmKdaChunked {
                    f,
                    b,
                    state,
                    heads,
                    head_dim,
                    y,
                    ..
                } => {
                    // The scan reads the conv's extended output (`mixed`, a
                    // region above) and its own two projections, buffered.
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::KdaF, *f)?;
                    plane(&mut layout, cache, Role::KdaB, *b)?;
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*heads) * u64::from(*head_dim) * u64::from(*head_dim));
                }
                Attention::PleNgramIds {
                    ids,
                    state,
                    ngram_ids,
                    ..
                }
                | Attention::PleNgramIdsChunked {
                    ids,
                    state,
                    ngram_ids,
                    ..
                } => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::Ids, *ids)?;
                    region(&mut layout, cache, "ngram", *ngram_ids)?;
                }
                _ => {}
            }
        }
        for node in &trace.nodes {
            let Operation::Attention(Attention::SsmGdnPrep { ba, gates, .. }) = &node.op else {
                continue;
            };
            let cache = *gates_cache.get(&gates.0).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "the gates value {} `attention.ssm_gdn_prep` lands, which no delta scan reads",
                    gates.0
                ),
            })?;
            plane(&mut layout, cache, Role::Ba, *ba)?;
            region(&mut layout, cache, "gates", *gates)?;
        }
        if layout.planes.is_empty() {
            return Ok(None);
        }
        layout.token_bytes = layout.planes.iter().map(|plane| plane.row_bytes).sum();
        Ok(Some(layout))
    }
}

/// The page slab: `[page slot][plane][token]`, one page slot per recurrent
/// slot.
#[derive(Debug)]
pub struct Buffers {
    slab: Buffer,
    page_tokens: u32,
    slots: u32,
    page_bytes: u64,
    /// Byte offset of each plane's first token inside a page slot.
    plane_at: Vec<u64>,
}

impl Buffers {
    /// Reserve the slab one layout needs at one deployment's paging.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the allocation, [`Fault::Deviceless`] off Apple.
    pub fn reserve(device: &Context, layout: &Layout, paging: Paging) -> Result<Buffers> {
        let page_tokens = paging.page_size.max(1);
        let mut plane_at = Vec::with_capacity(layout.planes.len());
        let mut at = 0u64;
        for plane in &layout.planes {
            plane_at.push(at);
            at += u64::from(page_tokens) * plane.row_bytes;
        }
        let page_bytes = at.next_multiple_of(256);
        let slots = paging.slots.max(1);
        let bytes = page_bytes.saturating_mul(u64::from(slots));
        Ok(Buffers {
            slab: Buffer::zeroed(device, bytes.max(256))?,
            page_tokens,
            slots,
            page_bytes,
            plane_at,
        })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slab.bytes()
    }

    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    fn handle(&self, handles: &Handles) -> Result<u32> {
        handles.bind(&self.slab, 0, self.slab.bytes())
    }
}

/// A run of buffer tokens `[from, from + count)` over a lane's page list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub pages: Vec<u32>,
    pub from: u32,
    pub count: u32,
}

/// What one lane's verb asks of this fire, host-resolved.
#[derive(Clone, Debug, Default)]
pub struct LanePlan {
    /// Buffered tokens replayed ahead of the lane's own rows.
    pub replay: u32,
    /// Rows of the extended run whose state persists.
    pub commit: u32,
    /// The lane's own rows this fire.
    pub rows: u32,
    /// Buffer tokens gathered into the extended run — ahead of the rows, or
    /// in their place when [`override_rows`](LanePlan::override_rows).
    pub gather: Option<Run>,
    /// Where the lane's own rows are scattered to.
    pub scatter: Option<Run>,
    /// `FoldBuffered`: the gathered tokens ARE the rows (the lane's own
    /// tokens are placeholders the replay overwrites).
    pub override_rows: bool,
}

impl LanePlan {
    /// Resolve one lane's verb against the rows it carries.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a device-resident fold length (this plane has
    /// no descriptor port to read it from), a `FoldBuffered` whose bound is
    /// not the lane's row count, or a fold past the extended run.
    pub fn of(verb: &RsVerb, rows: u32, lane: u32, device_fold: Option<u32>) -> Result<LanePlan> {
        let host = |len: FoldLen| -> Result<u32> {
            match len {
                FoldLen::Host(n) => Ok(n),
                FoldLen::Device(port) => device_fold.ok_or_else(|| {
                    Fault::program(
                        "serve::rs",
                        format!(
                            "lane {lane} states a device-resident fold length on port {}, and the \
                             program attached to it resolved no such port this fire",
                            port.name()
                        ),
                    )
                }),
            }
        };
        Ok(match verb {
            RsVerb::Fold => LanePlan {
                replay: 0,
                commit: rows,
                rows,
                ..LanePlan::default()
            },
            RsVerb::Buffer {
                pages,
                at,
                fold,
                replay,
            } => {
                let commit = host(*fold)?;
                if commit > replay.saturating_add(rows) {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {lane} folds {commit} of the {rows} rows it carries plus the \
                             {replay} it replays"
                        ),
                    ));
                }
                if *replay > *at {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {lane} replays {replay} buffered token(s) below buffer position \
                             {at}, which has only {at}"
                        ),
                    ));
                }
                LanePlan {
                    replay: *replay,
                    commit,
                    rows,
                    gather: (*replay > 0).then(|| Run {
                        pages: pages.clone(),
                        from: at - replay,
                        count: *replay,
                    }),
                    scatter: (rows > 0).then(|| Run {
                        pages: pages.clone(),
                        from: *at,
                        count: rows,
                    }),
                    override_rows: false,
                }
            }
            RsVerb::Window { read, write, fold } => {
                let fold = host(*fold)?;
                LanePlan {
                    replay: fold,
                    commit: fold,
                    rows,
                    gather: (fold > 0).then(|| Run {
                        pages: read.clone(),
                        from: 0,
                        count: fold,
                    }),
                    scatter: (rows > 0).then(|| Run {
                        pages: write.clone(),
                        from: 0,
                        count: rows,
                    }),
                    override_rows: false,
                }
            }
            RsVerb::FoldBuffered {
                pages,
                at,
                bound,
                len,
            } => {
                if *bound != rows {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {lane} replays a buffer bounded at {bound} tokens in a fire that \
                             gave it {rows} rows — the bound IS what sizes the launch, so the two \
                             are one number"
                        ),
                    ));
                }
                let commit = host(*len)?.min(*bound);
                LanePlan {
                    replay: 0,
                    commit,
                    rows,
                    gather: Some(Run {
                        pages: pages.clone(),
                        from: *at,
                        count: *bound,
                    }),
                    scatter: None,
                    override_rows: true,
                }
            }
        })
    }
}

/// One fire's recurrent-state seat: the per-lane plans, the tables the
/// committed launches read, the slab and the extended scratch, all minted
/// for this fire.
#[derive(Debug)]
pub struct Seat {
    /// In fire lane order.
    pub lanes: Vec<LanePlan>,
    /// `i32`, `[fire lanes]`.
    pub replay: Tensor,
    /// `i32`, `[fire lanes]`.
    pub commit: Tensor,
    /// `i32`, `[fire lanes]`: the bank each lane owns.
    pub slots: Tensor,
    /// The whole page slab.
    pub pool: u32,
    pub layout: Arc<Layout>,
    pub plane_at: Vec<u64>,
    pub page_tokens: u32,
    pub page_bytes: u64,
    pub pool_slots: u32,
    /// Per plane: the extended input rectangle, `rows_ext` tall.
    pub ext_in: Vec<Tensor>,
    /// Per region: the extended output rectangle, `rows_ext` tall.
    pub ext_out: Vec<Tensor>,
    /// `[fire lanes][work_elems]` f32.
    pub work: Tensor,
    pub rows_ext: u32,
    /// Extended outputs by the value they stand in for, filled as the walk
    /// lands them and read by the ops downstream.
    pub ext: RefCell<HashMap<u32, Tensor>>,
}

impl Seat {
    /// Bytes of scratch a fire of `rows_ext` extended rows over `lanes`
    /// lanes needs.
    #[must_use]
    pub fn scratch_bytes(layout: &Layout, rows_ext: u32, lanes: u32) -> u64 {
        let rows = u64::from(rows_ext.max(1));
        let mut at = 0u64;
        // Each take below binds at least 4 bytes (a zero-row plane, or a
        // layout with no delta work copy), so the sum must hold those too.
        for plane in &layout.planes {
            at += (rows * plane.row_bytes).max(4).next_multiple_of(256);
        }
        for region in &layout.regions {
            at += (rows * region.row_bytes).max(4).next_multiple_of(256);
        }
        at += (u64::from(lanes.max(1)) * layout.work_elems * 4)
            .max(4)
            .next_multiple_of(256);
        at.max(256)
    }

    /// Mint the seat for one fire.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a handle table already full or a scratch too
    /// small for the extended rows.
    #[allow(clippy::too_many_arguments)]
    pub fn mint(
        handles: &Handles,
        layout: &Arc<Layout>,
        buffers: &Buffers,
        scratch: &Buffer,
        lanes: Vec<LanePlan>,
        replay: Tensor,
        commit: Tensor,
        slots: Tensor,
        rows_ext: u32,
    ) -> Result<Seat> {
        let lane_count = u32::try_from(lanes.len()).unwrap_or(u32::MAX);
        let need = Seat::scratch_bytes(layout, rows_ext, lane_count);
        if scratch.bytes() < need {
            return Err(Fault::Ceiling {
                what: "recurrent buffer scratch bytes",
                need,
                have: scratch.bytes(),
            });
        }
        let rows = u64::from(rows_ext.max(1));
        let mut at = 0u64;
        let mut take = |row_bytes: u64, width: u32, dtype: Dtype| -> Result<Tensor> {
            let bytes = rows * row_bytes;
            let here = at;
            at += bytes.next_multiple_of(256);
            Ok(Tensor::new(
                handles.bind(scratch, here, bytes.max(4))?,
                rows_ext.max(1),
                width,
                dtype,
            ))
        };
        let mut ext_in = Vec::with_capacity(layout.planes.len());
        for plane in &layout.planes {
            ext_in.push(take(plane.row_bytes, plane.width, plane.dtype)?);
        }
        let mut ext_out = Vec::with_capacity(layout.regions.len());
        for region in &layout.regions {
            ext_out.push(take(region.row_bytes, region.width, region.dtype)?);
        }
        let work_bytes = u64::from(lane_count.max(1)) * layout.work_elems * 4;
        let work = Tensor::new(
            handles.bind(scratch, at, work_bytes.max(4))?,
            lane_count.max(1),
            u32::try_from(layout.work_elems).unwrap_or(u32::MAX),
            Dtype::F32,
        );
        Ok(Seat {
            lanes,
            replay,
            commit,
            slots,
            pool: buffers.handle(handles)?,
            layout: Arc::clone(layout),
            plane_at: buffers.plane_at.clone(),
            page_tokens: buffers.page_tokens,
            page_bytes: buffers.page_bytes,
            pool_slots: buffers.slots,
            ext_in,
            ext_out,
            work,
            rows_ext,
            ext: RefCell::new(HashMap::new()),
        })
    }

    /// Byte offset of buffer token `token` of plane `plane` inside the slab,
    /// and how many tokens from it stay inside the same page.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a token past the lane's page list or a page
    /// past the pool.
    pub fn locate(&self, run: &Run, plane: usize, token: u32) -> Result<(u64, u32)> {
        let page = token / self.page_tokens;
        let in_page = token % self.page_tokens;
        let slot = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
            what: "recurrent buffer pages",
            need: u64::from(page) + 1,
            have: run.pages.len() as u64,
        })?;
        if slot >= self.pool_slots {
            return Err(Fault::Ceiling {
                what: "recurrent buffer page slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.pool_slots),
            });
        }
        let row_bytes = self.layout.planes[plane].row_bytes;
        Ok((
            u64::from(slot) * self.page_bytes + self.plane_at[plane] + u64::from(in_page) * row_bytes,
            self.page_tokens - in_page,
        ))
    }
}
