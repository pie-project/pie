use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine::rs::{LanePlan, Run};
use kernels_slang::{Segments, Tensor};
use poem_ir::{Def, Dim, Dtype, Gather, Operation, Ssm, Trace, Ty, ValueId};

use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};
use crate::handles::Handles;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    ConvX,

    Ba,

    Ids,

    KdaF,

    KdaB,
}

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub cache: u32,
    pub role: Role,
    pub width: u32,
    pub dtype: Dtype,

    pub row_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub width: u32,
    pub dtype: Dtype,
    pub row_bytes: u64,
}

#[derive(Debug, Default)]
pub struct Layout {
    pub planes: Vec<Plane>,
    pub regions: Vec<Region>,

    pub in_of: HashMap<u32, usize>,

    pub out_of: HashMap<u32, usize>,

    pub token_bytes: u64,

    pub work_elems: u64,
}

fn row_of(trace: &Trace, id: ValueId) -> Result<(u32, Dtype, u64)> {
    let decl = trace
        .values
        .get(id.0 as usize)
        .ok_or_else(|| Fault::Unbound {
            what: format!("value {}, which the trace does not declare", id.0),
        })?;
    let Ty::Tensor { shape, dtype } = &decl.ty else {
        return Err(Fault::Unbound {
            what: format!(
                "value {}, a struct where the recurrence reads token rows",
                id.0
            ),
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
    let element = poem_compiler::arena::elem_bytes(*dtype).ok_or_else(|| Fault::Unbound {
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
            what: format!(
                "value {}, read as a recurrent state and declared as no cache",
                state.0
            ),
        }),
    }
}

impl Layout {
    pub fn read(trace: &Trace) -> Result<Option<Layout>> {
        let mut layout = Layout::default();
        let mut plane_keys: HashMap<(u32, Role), usize> = HashMap::new();
        let mut region_keys: HashMap<(u32, &'static str), usize> = HashMap::new();
        let mut gates_cache: HashMap<u32, u32> = HashMap::new();

        let mut plane = |layout: &mut Layout,
                         cache: u32,
                         role: Role,
                         value: ValueId|
         -> Result<()> {
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
        let mut region =
            |layout: &mut Layout, cache: u32, what: &'static str, value: ValueId| -> Result<()> {
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

        // The convs the rs buffer covers: the chunk plan's rule
        // (`poem_compiler::rs::buffered_convs`), the one the recipe's rs rows
        // read, not a walk of this shell's own. A conv feeding a recurrence
        // and a conv no recurrence reads (Qwen4's PLE conv, `Feeds::Alone`)
        // alike buffer their input and replay it at a fold. This shell
        // replays no Mamba-2 layer, so a Mamba-2 conv commits with it and
        // keeps no plane.
        for conv in poem_compiler::rs::buffered_convs(trace, false) {
            let cache = cache_of(trace, conv.state)?;
            plane(&mut layout, cache, Role::ConvX, conv.x)?;
            region(&mut layout, cache, "conv", conv.y)?;
        }
        for node in &trace.nodes {
            match &node.op {
                Operation::Ssm(
                    Ssm::GatedDelta {
                        gates,
                        state,
                        k_heads,
                        v_heads,
                        k_dim,
                        v_dim,
                        y,
                        ..
                    }
                    | Ssm::GatedDeltaChunked {
                        gates,
                        state,
                        k_heads,
                        v_heads,
                        k_dim,
                        v_dim,
                        y,
                        ..
                    },
                ) => {
                    let cache = cache_of(trace, *state)?;
                    let _ = k_heads;
                    gates_cache.insert(gates.0, cache);
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*v_heads) * u64::from(*v_dim) * u64::from(*k_dim));
                }
                Operation::Ssm(
                    Ssm::Kda {
                        f,
                        b,
                        state,
                        heads,
                        head_dim,
                        y,
                        ..
                    }
                    | Ssm::KdaChunked {
                        f,
                        b,
                        state,
                        heads,
                        head_dim,
                        y,
                        ..
                    },
                ) => {
                    let cache = cache_of(trace, *state)?;
                    plane(&mut layout, cache, Role::KdaF, *f)?;
                    plane(&mut layout, cache, Role::KdaB, *b)?;
                    region(&mut layout, cache, "scan", *y)?;
                    layout.work_elems = layout
                        .work_elems
                        .max(u64::from(*heads) * u64::from(*head_dim) * u64::from(*head_dim));
                }
                Operation::Gather(
                    Gather::NgramIds { ids, state, .. } | Gather::NgramIdsChunked { ids, state, .. },
                ) => {
                    // Its own rows' table ids land where the fire's land
                    // (`gather.ngram_ids_committed`): no extended region.
                    plane(&mut layout, cache_of(trace, *state)?, Role::Ids, *ids)?;
                }
                _ => {}
            }
        }
        for node in &trace.nodes {
            let Operation::Ssm(Ssm::GdnGates { ba, gates, .. }) = &node.op else {
                continue;
            };
            let cache = *gates_cache.get(&gates.0).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "the gates value {} `ssm.gdn_gates` lands, which no delta scan reads",
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

/// The rs list in the chunk pool (CONTRACT-BC 18, 20): an rs page is an
/// rs-list slot, `page_tokens` unfolded rows of every plane, plane `p` at
/// `plane_at[p]` in it; a page id counts `stride` bytes of the pool.
#[derive(Debug)]
pub struct Buffers {
    page_tokens: u32,
    stride: u64,
    plane_at: Vec<u64>,
}

impl Buffers {
    #[must_use]
    pub fn in_pool(plane_at: Vec<u64>, page_tokens: u32, stride: u64) -> Buffers {
        Buffers {
            page_tokens: page_tokens.max(1),
            stride,
            plane_at,
        }
    }

    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }
}

#[derive(Debug)]
pub struct Seat {
    pub lanes: Vec<LanePlan>,

    pub replay: Tensor,

    pub commit: Tensor,

    pub slots: Tensor,

    /// The chunk pool the rs pages lie in.
    pub pool: Segments,
    pub layout: Arc<Layout>,
    pub plane_at: Vec<u64>,
    pub page_tokens: u32,
    /// The bytes an rs page id counts.
    pub stride: u64,

    pub ext_in: Vec<Tensor>,

    pub ext_out: Vec<Tensor>,

    pub work: Tensor,
    pub rows_ext: u32,

    pub ext: RefCell<HashMap<u32, Tensor>>,
}

impl Seat {
    #[must_use]
    pub fn scratch_bytes(layout: &Layout, rows_ext: u32, lanes: u32) -> u64 {
        let rows = u64::from(rows_ext.max(1));
        let mut at = 0u64;

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

    #[allow(clippy::too_many_arguments)]
    pub fn mint<B: Backend>(
        handles: &Handles<B>,
        layout: &Arc<Layout>,
        buffers: &Buffers,
        pool: Segments,
        scratch: &B::Buffer,
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
            pool,
            layout: Arc::clone(layout),
            plane_at: buffers.plane_at.clone(),
            page_tokens: buffers.page_tokens,
            stride: buffers.stride,
            ext_in,
            ext_out,
            work,
            rows_ext,
            ext: RefCell::new(HashMap::new()),
        })
    }

    /// Where token `token` of `run`'s plane `plane` lies: the segment's
    /// handle, the byte there, and the rows left in its page.
    pub fn locate(&self, run: &Run, plane: usize, token: u32) -> Result<(u32, u64, u32)> {
        let page = token / self.page_tokens;
        let in_page = token % self.page_tokens;
        let id = *run.pages.get(page as usize).ok_or(Fault::Ceiling {
            what: "recurrent buffer pages",
            need: u64::from(page) + 1,
            have: run.pages.len() as u64,
        })?;
        let row_bytes = self.layout.planes[plane].row_bytes;
        let at =
            u64::from(id) * self.stride + self.plane_at[plane] + u64::from(in_page) * row_bytes;
        let span = u64::from(self.pool.words) * 4;
        let segment = usize::try_from(at / span).unwrap_or(usize::MAX);
        let handle = self
            .pool
            .bufs
            .get(segment)
            .copied()
            .filter(|&handle| handle != kernels_slang::ABSENT)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment as u64 + 1,
                have: 0,
            })?;
        Ok((handle, at % span, self.page_tokens - in_page))
    }
}

/// The snapshot rows one lane lands in a fire (`Lane::rs_snapshots`):
/// [`kernels_slang::ssm::RS_SNAPSHOTS`] where the trace holds state rows and
/// every op that reads one is a causal conv, a gated delta or a KDA scan,
/// whose chunked and committed arms land them; else none, and the runtime
/// cuts a pass at each.
#[must_use]
pub fn snapshots(trace: &poem_ir::Trace) -> u32 {
    use poem_ir::{Def, Operands, Operation, Ssm};
    let state = |id: &poem_ir::ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|value| &value.def),
            Some(Def::Cache(row))
                if matches!(trace.caches.get(*row as usize), Some(poem_ir::CacheRow::State { .. }))
        )
    };
    let mut reads = Vec::new();
    let mut any = false;
    let lands = trace.nodes.iter().all(|node| {
        reads.clear();
        node.op.inputs(&mut reads);
        if !reads.iter().any(state) {
            return true;
        }
        any = true;
        matches!(
            &node.op,
            Operation::Ssm(
                Ssm::CausalConv1d { .. }
                    | Ssm::CausalConv1dChunked { .. }
                    | Ssm::GatedDelta { .. }
                    | Ssm::GatedDeltaChunked { .. }
                    | Ssm::Kda { .. }
                    | Ssm::KdaChunked { .. },
            )
        )
    });
    if lands && any {
        kernels_slang::ssm::RS_SNAPSHOTS
    } else {
        0
    }
}

/// The lagged fold's decode replay (`new-memory/designs/lagged-fold.md`):
/// in a decode window, each lane's carried rows (the buffered rows past its
/// committed state, at most [`Lag::carry`]) are replayed by the decode
/// step's own recurrent kernels, one pass a layer, from the rs buffer they
/// lie in (`kernels_slang::ssm::Lag`). A fire's lanes are told by a table
/// in its inputs; the conv's carried outputs (the scan's `[q | k | v]`) and
/// the carried gates land in one scratch laid here, at load, for the widest
/// fire, and reused layer after layer. One decode body a rung replays any
/// lane's rows, a lane carrying none replaying none: no fire of a decode
/// class takes the extended path ([`Seat`]) for rows within the bound.
///
/// Laid where every rs plane the load buffers is a conv's input or a delta
/// scan's gate projection (no KDA, no n-gram window: those replay through
/// the extended path).
#[derive(Debug)]
pub struct Lag {
    /// The most rows a lane carries: the run-ahead window's bound, the
    /// binding's submit depth times the most steps a frame holds
    /// (`engine::runahead::Runahead`), which bounds the runtime's
    /// `run_ahead_window()` at every frame size it admits.
    pub carry: u32,
    /// The classes whose regions step a buffered conv or a delta scan one
    /// row a lane: where the replay kernels run.
    pub classes: poem_ir::ClassSet,
    /// The gate rows a decode scan reads (`ssm.gdn_gates` lands them).
    pub decode_gates: std::collections::HashSet<u32>,
    pub layout: Arc<Layout>,
    /// Each lane's carried rows' conv outputs, `carry` rows a lane.
    pub carried_qkv: Tensor,
    /// Each lane's carried rows' `[g | beta]`, `carry` rows a lane.
    pub carried_gates: Tensor,
    /// A lane's copy of a delta state walked in memory (a head too wide for
    /// a lane's registers), where the load has one.
    pub work: Option<Tensor>,
}

/// A fire's view of the [`Lag`]: its lanes' table and where the rs pages
/// lie as it fires.
#[derive(Clone)]
pub struct LagFire {
    pub lag: Arc<Lag>,
    pub table: Tensor,
    pub pool: Segments,
    pub page_tokens: u32,
    /// The words an rs page id counts.
    pub page_words: u32,
    /// Each plane's offset in an rs page, in words.
    pub plane_words: Arc<[u32]>,
}

impl std::fmt::Debug for LagFire {
    // What a recording binds: the table, the pool and the scratch, not the
    // layout (the load's, the same every fire).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LagFire")
            .field("table", &self.table)
            .field("pool", &self.pool)
            .field("page_tokens", &self.page_tokens)
            .field("page_words", &self.page_words)
            .field("carried_qkv", &self.lag.carried_qkv)
            .field("carried_gates", &self.lag.carried_gates)
            .field("work", &self.lag.work)
            .finish()
    }
}

impl LagFire {
    /// Where value `value`'s rs plane lies, as a replay kernel reads it.
    pub fn plane(&self, op: &'static str, value: ValueId) -> std::result::Result<kernels_slang::ssm::RsPlane, kernels_slang::Error> {
        let at = *self.lag.layout.in_of.get(&value.0).ok_or_else(|| kernels_slang::Error::Backend {
            op,
            detail: format!("value {} is no recurrence input this load buffers", value.0),
        })?;
        let row_bytes = self.lag.layout.planes[at].row_bytes;
        Ok(kernels_slang::ssm::RsPlane {
            pool: self.pool,
            page_tokens: self.page_tokens,
            page_words: self.page_words,
            plane_words: self.plane_words[at],
            row_words: u32::try_from(row_bytes / 4).unwrap_or(u32::MAX),
        })
    }

    /// The table as the kernels of a window from lane `lane0` read it.
    #[must_use]
    pub fn kernel(&self, lane0: u32) -> kernels_slang::ssm::Lag {
        kernels_slang::ssm::Lag {
            table: self.table,
            lane0,
            carry: self.lag.carry,
        }
    }
}

impl Lag {
    /// The carried rows a lane holds at most, under `runahead`.
    #[must_use]
    pub fn carry_of(runahead: engine::runahead::Runahead) -> u32 {
        let rows = runahead.submit_depth() * usize::from(engine::runahead::Runahead::STEPS_MAX);
        u32::try_from(rows).unwrap_or(u32::MAX)
    }

    /// Whether `layout`'s planes are all the replay kernels read.
    #[must_use]
    pub fn replays(layout: Option<&Layout>) -> bool {
        layout.is_some_and(|layout| {
            layout
                .planes
                .iter()
                .all(|plane| matches!(plane.role, Role::ConvX | Role::Ba))
        })
    }

    /// The load's lag, its scratch laid for `max_lanes` lanes, where the
    /// trace has a decode class stepping a buffered recurrence and
    /// [`Lag::replays`] its planes.
    #[allow(clippy::too_many_arguments)]
    pub fn lay<B: Backend>(
        device: &B::Context,
        handles: &Handles<B>,
        trace: &Trace,
        compiled: &poem_compiler::CompiledModel,
        layout: Option<&Arc<Layout>>,
        max_lanes: u32,
        runahead: engine::runahead::Runahead,
    ) -> Result<Option<Lag>> {
        let Some(layout) = layout.filter(|layout| Lag::replays(Some(&***layout))) else {
            return Ok(None);
        };
        let mut classes = poem_ir::ClassSet::default();
        let mut decode_gates = std::collections::HashSet::new();
        let mut in_memory = false;
        for region in compiled.template() {
            let mut steps = false;
            for node in region.nodes.clone() {
                match trace.nodes.get(node as usize).map(|node| &node.op) {
                    Some(Operation::Ssm(Ssm::CausalConv1d { y, .. }))
                        if layout.out_of.contains_key(&y.0) =>
                    {
                        steps = true;
                    }
                    Some(Operation::Ssm(Ssm::GatedDelta {
                        gates, k_dim, v_dim, ..
                    })) => {
                        steps = true;
                        decode_gates.insert(gates.0);
                        in_memory |=
                            !kernels_slang::ssm::gdn_recurrent::holds_in_registers(*k_dim, *v_dim);
                    }
                    _ => {}
                }
            }
            if steps {
                for class in region.mask.iter() {
                    classes.insert(class);
                }
            }
        }
        if classes.is_empty() {
            return Ok(None);
        }
        let carry = Lag::carry_of(runahead);
        let widest = |role: Role| {
            layout
                .planes
                .iter()
                .filter(|plane| plane.role == role)
                .map(|plane| plane.width)
                .max()
                .unwrap_or(1)
        };
        let rows = u64::from(max_lanes.max(1)) * u64::from(carry);
        let rows32 = u32::try_from(rows).map_err(|_| Fault::Ceiling {
            what: "carried rows a fire's lanes hold",
            need: rows,
            have: u64::from(u32::MAX),
        })?;
        let (qkv_width, gates_width) = (widest(Role::ConvX), widest(Role::Ba));
        let qkv_bytes = (rows * u64::from(qkv_width) * 2).max(4).next_multiple_of(256);
        let gates_bytes = (rows * u64::from(gates_width) * 4).max(4).next_multiple_of(256);
        let work_elems = if in_memory {
            u64::from(max_lanes.max(1)) * layout.work_elems
        } else {
            0
        };
        let work_bytes = (work_elems * 4).next_multiple_of(256);
        let scratch = B::Buffer::zeroed(
            device,
            qkv_bytes + gates_bytes + work_bytes,
            crate::books::owners::RS_CARRIED,
        )?;
        let carried_qkv = Tensor::new(
            handles.bind(&scratch, 0, qkv_bytes)?,
            rows32,
            qkv_width,
            Dtype::Bf16,
        );
        let carried_gates = Tensor::new(
            handles.bind(&scratch, qkv_bytes, gates_bytes)?,
            rows32,
            gates_width,
            Dtype::F32,
        );
        let work = if in_memory {
            Some(Tensor::new(
                handles.bind(&scratch, qkv_bytes + gates_bytes, work_bytes)?,
                max_lanes.max(1),
                u32::try_from(layout.work_elems).unwrap_or(u32::MAX),
                Dtype::F32,
            ))
        } else {
            None
        };
        tracing::info!(
            carry,
            lanes = max_lanes,
            bytes = qkv_bytes + gates_bytes + work_bytes,
            "the lagged fold's decode replay: carried rows' outputs laid at load"
        );
        Ok(Some(Lag {
            carry,
            classes,
            decode_gates,
            layout: Arc::clone(layout),
            carried_qkv,
            carried_gates,
            work,
        }))
    }

    /// A fire's view, over the rs pages `buffers` lays in `pool`.
    pub fn fire(self: &Arc<Self>, buffers: &Buffers, pool: Segments, table: Tensor) -> Result<LagFire> {
        let words = |bytes: u64, what: &'static str| {
            if bytes % 4 != 0 {
                return Err(Fault::Unbound {
                    what: format!("{what} at {bytes} bytes, which a replay kernel reads in words"),
                });
            }
            u32::try_from(bytes / 4).map_err(|_| Fault::Ceiling {
                what,
                need: bytes / 4,
                have: u64::from(u32::MAX),
            })
        };
        let plane_words: Vec<u32> = buffers
            .plane_at
            .iter()
            .map(|&at| words(at, "an rs plane's offset in its page"))
            .collect::<Result<_>>()?;
        Ok(LagFire {
            lag: Arc::clone(self),
            table,
            pool,
            page_tokens: buffers.page_tokens,
            page_words: words(buffers.stride, "an rs page's stride")?,
            plane_words: plane_words.into(),
        })
    }
}

impl Lag {
    /// Appends a lane's table entry ([`kernels_slang::ssm::Lag`]) for its
    /// plan over `rows` rows of class `class`, its rs pages `page_tokens`
    /// tokens each; whether the replay kernels serve it: a lane of a class
    /// they run in, one row, carrying at most [`Lag::carry`] rows. One they
    /// do not serve carries nothing in the table (its fire takes the
    /// extended path, [`Seat`], whose ops come first).
    pub fn entry(
        &self,
        plan: &LanePlan,
        rows: u32,
        class: usize,
        page_tokens: u32,
        out: &mut Vec<i32>,
    ) -> Result<bool> {
        let start = out.len();
        let ints = kernels_slang::ssm::Lag::entry_ints(self.carry) as usize;
        out.resize(start + ints, 0);
        let entry = &mut out[start..];
        let serves = rows == 1 && self.classes.contains(class) && plan.replay <= self.carry;
        let unit = |run: &Run, token: u32| -> Result<i32> {
            let page = *run
                .pages
                .get((token / page_tokens.max(1)) as usize)
                .ok_or(Fault::Ceiling {
                    what: "recurrent buffer pages",
                    need: u64::from(token / page_tokens.max(1)) + 1,
                    have: run.pages.len() as u64,
                })?;
            let at = u64::from(page) * u64::from(page_tokens) + u64::from(token % page_tokens.max(1));
            i32::try_from(at).map_err(|_| Fault::Ceiling {
                what: "an rs-buffer row as a replay kernel's int",
                need: at,
                have: i32::MAX as u64,
            })
        };
        if !serves {
            entry[1] = i32::try_from(plan.commit).unwrap_or(i32::MAX);
            entry[2] = -1;
            return Ok(false);
        }
        entry[0] = i32::try_from(plan.replay).unwrap_or(i32::MAX);
        entry[1] = i32::try_from(plan.commit).unwrap_or(i32::MAX);
        entry[2] = match &plan.scatter {
            Some(run) => unit(run, run.from)?,
            None => -1,
        };
        let mut at = Lag::HEAD_INTS;
        for run in &plan.gathers {
            for k in 0..run.count {
                entry[at] = unit(run, run.from + k)?;
                at += 1;
            }
        }
        debug_assert_eq!(at - Lag::HEAD_INTS, plan.replay as usize, "a plan's gathers are its replay");
        Ok(true)
    }

    const HEAD_INTS: usize = kernels_slang::ssm::Lag::HEAD as usize;
}
