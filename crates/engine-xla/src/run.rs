//! A fire's walk: plan values resolved to handles, every node dispatched to
//! a kernels-xla entry that emits into the fire's tracer.
//!
//! This is engine-wgpu's `Run` with the device-memory machinery taken out:
//! no expert seats, pumps or scratch rooms (every weight is resident and XLA
//! plans its own temporaries), and no copy compaction (a traced slice is
//! free).

use kernels_xla::{Bank, Ctx, DecodePlan, KvPool, PrefillPlan, RaggedTensor, RecurrentPool, Tensor};
use model_ir::{Def, Dim, Dtype, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::store::{CachePool, CacheTable};
use crate::trace::Handles;
use crate::weights::{WeightRow, WeightTable};
use crate::window::{At, Window, Windows};

#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheGeometry {
    pub indptr: Option<Tensor>,

    pub indices: Option<Tensor>,

    pub seq_lens: Option<Tensor>,

    pub last_page_len: Option<Tensor>,

    pub kv_len: Option<Tensor>,

    pub row_valid: Option<Tensor>,

    pub request_of_token: Option<Tensor>,

    pub write_page: Option<Tensor>,

    pub write_offset: Option<Tensor>,
}

#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    pub request_of_token: Tensor,

    pub mask: Tensor,

    pub mask_enabled: Tensor,

    pub mask_stride: u32,
}

/// A fire's image-tower inputs.
#[derive(Clone, Copy, Debug)]
pub struct PatchBindings {
    pub patches: Tensor,
    pub segments: Tensor,
    pub routes: Tensor,
    pub positions: Tensor,
    pub embed_rows: Option<Tensor>,
    pub embed_weights: Option<Tensor>,
}

/// One packing of a fire's rows by attention group (`model_exec::fire::pack`),
/// for the selection of lanes a plan's grouped attention reads.
#[derive(Clone, Copy, Debug)]
pub struct PackingBindings {
    pub group_indptr: Tensor,
    pub lane_indptr: Tensor,
    pub reference_tag: Tensor,
    pub permutation: Tensor,
    /// One attention class per packed row (-1 where the lane states none).
    pub attn_class: Tensor,
}

/// A float port's rectangle for this fire.
#[derive(Clone, Copy, Debug)]
pub struct PortBinding {
    pub kind: engine::fire::PortKind,
    pub port: u8,
    pub tensor: Tensor,
}

/// The inputs of the diffusion and video families: the voxel axis (clip
/// grids, the voxel port), float ports fed from channels, the attention
/// group packings, and self-conditioning taps. Empty for a text fire.
#[derive(Clone, Debug, Default)]
pub struct DitBindings {
    pub grid: Option<Tensor>,
    pub token_grid: Option<Tensor>,
    pub voxels: Option<Tensor>,
    pub self_cond_rows: Option<Tensor>,
    pub self_cond_weights: Option<Tensor>,
    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<(model_ir::Selection, PackingBindings)>,
    pub ports: Vec<PortBinding>,
    /// The fire's attention class table (`count x count` u8) and its count,
    /// when a lane states classes.
    pub class_table: Option<(Tensor, u32)>,
}

#[derive(Clone, Debug)]
pub struct FireBindings {
    pub tokens: Tensor,

    /// Each row's stated position: what the rotary ops read.
    pub positions: Tensor,

    /// Each row's index in its lane's cache (`kv_len - qo_len + i`): what
    /// every attention bound reads, as engine-cuda counts it.
    pub cache_rows: Tensor,

    pub readout_rows: Tensor,

    pub adapter_routes: Option<Tensor>,

    pub mrope_positions: Option<Tensor>,

    /// The slot of each clip (voxel lane), i32, for the frame caches of
    /// causal video convolutions; `None` when the fire stages no voxels.
    pub clip_slots: Option<Tensor>,

    pub patches: Option<PatchBindings>,

    pub geometry: Vec<CacheGeometry>,

    pub tables: FireTables,

    pub dit: DitBindings,
}

#[derive(Clone, Copy, Debug)]
pub enum StructSlot {
    Decode(DecodePlan),

    Prefill(PrefillPlan),

    Mla(kernels_xla::attn::mla::MlaPlan),
}

pub struct Run<'c> {
    ctx: &'c Ctx<'c>,

    handles: &'c Handles,

    values: &'c [ValueDecl],

    nodes: &'c [Node],

    weights: &'c WeightTable,

    arena: &'c SlotTable,

    caches: &'c CacheTable,

    structs: Vec<Option<StructSlot>>,

    values_wide: usize,

    fire: FireBindings,

    windows: &'c Windows,

    place: &'c At,

    compressor: Vec<(u32, [Tensor; 2])>,

    rs: Option<std::sync::Arc<crate::rs::Seat>>,

    /// Values whose planes a debugging reader asked for, and what the walk
    /// read of them as their writers ran (see `DispatchProbe`).
    probes: Vec<ValueId>,
    probed: std::cell::RefCell<Vec<(ValueId, kernels_xla::hlo::Val)>>,

    /// Token-declared values on a readout-rowed path, cut to the readout
    /// rows (`crate::rows`), and a readout-rowed value telling them.
    readout_rowed: (std::sync::Arc<Vec<bool>>, Option<ValueId>),
}

impl<'c> Run<'c> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &'c Ctx<'c>,
        handles: &'c Handles,
        values: &'c [ValueDecl],
        nodes: &'c [Node],
        weights: &'c WeightTable,
        arena: &'c SlotTable,
        caches: &'c CacheTable,
        fire: FireBindings,
        windows: &'c Windows,
        place: &'c At,
    ) -> Self {
        Self {
            ctx,
            handles,
            values,
            nodes,
            weights,
            arena,
            caches,
            structs: vec![None; values.len() * windows.max_runs() as usize],
            values_wide: values.len(),
            fire,
            windows,
            place,
            compressor: Vec::new(),
            rs: None,
            probes: Vec::new(),
            probed: std::cell::RefCell::new(Vec::new()),
            readout_rowed: (std::sync::Arc::new(Vec::new()), None),
        }
    }

    /// Cuts the token-declared values of a readout-rowed path to the
    /// fire's readout rows (see `crate::rows`).
    #[must_use]
    pub fn with_readout_rowed(
        mut self,
        rowed: &(std::sync::Arc<Vec<bool>>, Option<ValueId>),
    ) -> Self {
        self.readout_rowed = (std::sync::Arc::clone(&rowed.0), rowed.1);
        self
    }

    /// Hands the walk this fire's recurrent seat, when a lane buffers.
    #[must_use]
    pub fn with_rs(mut self, rs: Option<std::sync::Arc<crate::rs::Seat>>) -> Self {
        self.rs = rs;
        self
    }

    /// Reads each of `probes` as its writer runs, for a debugging reader.
    #[must_use]
    pub fn with_probes(mut self, probes: Vec<ValueId>) -> Self {
        self.probes = probes;
        self
    }

    pub(crate) fn probes(&self) -> &[ValueId] {
        &self.probes
    }

    pub(crate) fn record_probe(&self, value: ValueId, read: kernels_xla::hlo::Val) {
        self.probed.borrow_mut().push((value, read));
    }

    /// What the walk read of the probed values, in the order it wrote them.
    pub fn take_probed(&self) -> Vec<(ValueId, kernels_xla::hlo::Val)> {
        std::mem::take(&mut self.probed.borrow_mut())
    }

    pub(crate) fn rs_seat(&self) -> Option<std::sync::Arc<crate::rs::Seat>> {
        self.rs.clone()
    }

    /// Hands the walk the compressor planes of each space that has them.
    #[must_use]
    pub fn with_compressor(mut self, compressor: Vec<(u32, [Tensor; 2])>) -> Self {
        self.compressor = compressor;
        self
    }

    pub(crate) fn window(&self) -> &'c Window {
        self.windows
            .at(self.place.region.get(), self.place.run.get())
    }

    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span.rows
    }

    pub(crate) fn cut_rows(&self, handle: Tensor) -> Tensor {
        let span = self.window().span;
        self.slice(handle, span.row_offset, span.rows)
    }

    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }

    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    pub(crate) fn handles(&self) -> &'c Handles {
        self.handles
    }

    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// The root and row a handle starts at: two handles are the same memory
    /// exactly when these agree.
    pub(crate) fn address(&self, handle: u32) -> Option<(u32, u32)> {
        self.handles.locate(handle)
    }

    pub(crate) fn ctx(&self) -> &'c Ctx<'c> {
        self.ctx
    }

    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    /// This replica's rank in the tensor-parallel group. kernels-xla compiles
    /// for one replica (`collective`'s world of one), so it is 0 until the
    /// engine runs SPMD with `num_replicas > 1`.
    pub(crate) fn rank(&self) -> u32 {
        0
    }

    /// The fire's attention class table and its count, when a lane states
    /// classes.
    pub(crate) fn class_table(&self) -> Option<(Tensor, u32)> {
        self.fire.dit.class_table
    }

    /// The packed attention classes of the rows `indptr` (a group-packed
    /// `GroupIndptr` input) bounds.
    pub(crate) fn packed_classes(&self, indptr: ValueId) -> Option<Tensor> {
        match &self.values[indptr.0 as usize].def {
            Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::GroupIndptr { select },
                ..
            }) => self
                .fire
                .dit
                .packings
                .iter()
                .find(|(have, _)| have == select)
                .map(|(_, tables)| tables.attn_class),
            _ => None,
        }
    }

    /// The fire's clip slot table (one i32 slot per voxel lane).
    pub(crate) fn clip_slots(&self) -> Option<Tensor> {
        self.fire.clip_slots
    }

    /// A fresh per-fire array of `rows × width` `dtype`, zeros until
    /// written: the scratch a GPU dispatch would carve from its arena.
    pub(crate) fn temp(&self, rows: u32, width: u32, dtype: Dtype) -> Tensor {
        self.handles.root(crate::trace::Root {
            source: crate::trace::Source::Temp,
            dtype,
            rows,
            width,
        })
    }

    fn slice(&self, handle: Tensor, skip: u32, keep: u32) -> Tensor {
        self.handles.cut(handle, skip, keep)
    }

    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        if matches!(
            self.values[at].def,
            Def::Input(RuntimeInput::Mask { .. })
                | Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::Indices,
                    ..
                })
        ) {
            return handle;
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        if self.readout_rowed.0.get(at).copied().unwrap_or(false)
            && let Some(probe) = self.readout_rowed.1
        {
            let readouts = self.whole(probe).rows;
            match shape.first() {
                Some(Dim::Tokens) => return self.slice(handle, 0, readouts),
                Some(Dim::TokensTimes(k)) => return self.slice(handle, 0, readouts * k),
                _ => {}
            }
        }
        let seated = self.window();
        let window = seated.span;
        let patch = seated.patch;
        let voxel = seated.voxel;
        let (skip, keep) = match shape.first() {
            Some(Dim::Tokens) => (window.row_offset, window.rows),
            Some(Dim::TokensTimes(k)) => (window.row_offset * k, window.rows * k),
            Some(Dim::Lanes) => (window.lane_offset, window.lanes),
            Some(Dim::LanesPlus(k)) => (window.lane_offset, window.lanes + k),
            Some(Dim::Readouts) => return handle,
            Some(Dim::Const(_)) | None => return handle,
            Some(Dim::Patches) => (patch.row_offset, patch.rows),
            Some(Dim::Images) => (patch.lane_offset, patch.lanes),
            Some(Dim::ImagesPlus(k)) => (patch.lane_offset, patch.lanes + k),
            Some(Dim::Voxels) => (voxel.row_offset, voxel.rows),
            Some(Dim::VoxelsTimes(k)) => (voxel.row_offset * k, voxel.rows * k),
            Some(Dim::Clips) => (voxel.lane_offset, voxel.lanes),
            Some(Dim::ClipsPlus(k)) => (voxel.lane_offset, voxel.lanes + k),
        };
        self.slice(handle, skip, keep)
    }

    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::ReadoutRows) => self.fire.readout_rows,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,
            Def::Input(RuntimeInput::Mask { space: _ }) => self.fire.tables.mask,
            Def::Input(RuntimeInput::AdapterRoutes) => {
                self.fire.adapter_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's adapter ids, which no lane of it \
                         carried"
                    )
                })
            }
            Def::Input(RuntimeInput::MropePositions) => {
                self.fire.mrope_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the fire's (t, h, w) token positions, and this \
                         load reserved no triple"
                    )
                })
            }
            Def::Input(
                which @ (RuntimeInput::Grid
                | RuntimeInput::TokenGrid { .. }
                | RuntimeInput::Voxels { .. }
                | RuntimeInput::RowPermutation { .. }
                | RuntimeInput::Latents { .. }
                | RuntimeInput::LaneVector { .. }
                | RuntimeInput::Context { .. }
                | RuntimeInput::AxisPositions { .. }
                | RuntimeInput::SelfCondRows
                | RuntimeInput::SelfCondWeights
                | RuntimeInput::Geometry {
                    kind:
                        GeomKind::GroupOfLane
                        | GeomKind::GroupIndptr { .. }
                        | GeomKind::LaneIndptr { .. }
                        | GeomKind::ReferenceTag { .. },
                    ..
                }),
            ) => self.dit_input(at, which),
            Def::Input(RuntimeInput::Geometry { space, kind }) => {
                let space = *space as usize;
                let seat = self.fire.geometry.get(space).unwrap_or_else(|| {
                    panic!(
                        "value {at} names cache space {space}, and this fire binds \
                         {} geometry spaces",
                        self.fire.geometry.len()
                    )
                });
                let bound = match kind {
                    GeomKind::Indptr => seat.indptr,
                    GeomKind::Indices => seat.indices,
                    GeomKind::SeqLens => seat.seq_lens,
                    GeomKind::LastPageLen => seat.last_page_len,
                    GeomKind::KvLen => seat.kv_len,
                    GeomKind::RowValid => seat.row_valid,
                    GeomKind::RequestOfToken => seat.request_of_token,
                    GeomKind::WritePage => seat.write_page,
                    GeomKind::WriteOffset => seat.write_offset,
                    GeomKind::GroupOfLane
                    | GeomKind::GroupIndptr { .. }
                    | GeomKind::LaneIndptr { .. }
                    | GeomKind::ReferenceTag { .. } => None,
                };
                bound.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads {kind:?} of cache space {space}, which this \
                         fire left unbound"
                    )
                })
            }
            Def::Input(
                which @ (RuntimeInput::Patches
                | RuntimeInput::PatchSegments
                | RuntimeInput::PatchRoutes
                | RuntimeInput::PatchPositions
                | RuntimeInput::PatchEmbedRows
                | RuntimeInput::PatchEmbedWeights),
            ) => {
                let seat = self.fire.patches.unwrap_or_else(|| {
                    panic!("value {at} reads {which:?}, which no lane of this fire submitted")
                });
                let bound = match which {
                    RuntimeInput::Patches => Some(seat.patches),
                    RuntimeInput::PatchSegments => Some(seat.segments),
                    RuntimeInput::PatchRoutes => Some(seat.routes),
                    RuntimeInput::PatchPositions => Some(seat.positions),
                    RuntimeInput::PatchEmbedRows => seat.embed_rows,
                    _ => seat.embed_weights,
                };
                bound.unwrap_or_else(|| {
                    panic!("value {at} reads {which:?}, which this load stages none of")
                })
            }
            Def::Weight(w) => {
                let row = *w as usize;
                match self.weights.0.get(row).copied().flatten() {
                    Some(WeightRow::Dense(handle)) => handle,
                    Some(WeightRow::Planes(_)) => panic!(
                        "value {at} is weight {row}, a split-plane bank; it resolves \
                         through `Run::planes`, never as one dense handle"
                    ),
                    None => panic!("value {at} is weight {row}, which the shell has not bound"),
                }
            }
            Def::Op(_) | Def::Merge(_) => {
                self.arena.0.get(at).copied().flatten().unwrap_or_else(|| {
                    panic!("value {at} has no arena slot, which the compiler should have cut")
                })
            }
            Def::Cache(_) => panic!(
                "value {at} is a cache space; it resolves to a pool through `Run::pool`, \
                 never to a tensor"
            ),
        }
    }

    /// The diffusion/video inputs (`DitBindings`), by what the plan reads.
    fn dit_input(&self, at: usize, which: &RuntimeInput) -> Tensor {
        let dit = &self.fire.dit;
        let missing = |what: &str| -> ! {
            panic!("value {at} reads {which:?} ({what}), which this fire staged none of")
        };
        let packing = |select: model_ir::Selection| -> PackingBindings {
            dit.packings
                .iter()
                .find(|(have, _)| *have == select)
                .map(|(_, tables)| *tables)
                .unwrap_or_else(|| missing("the packing of its selection"))
        };
        let port = |kind: engine::fire::PortKind, port: u8| -> Tensor {
            dit.ports
                .iter()
                .find(|bound| bound.kind == kind && bound.port == port)
                .map(|bound| bound.tensor)
                .unwrap_or_else(|| missing("a float port"))
        };
        match *which {
            RuntimeInput::Grid => dit.grid.unwrap_or_else(|| missing("the clip grid")),
            RuntimeInput::TokenGrid { .. } => dit
                .token_grid
                .unwrap_or_else(|| missing("the token-side clip grid")),
            RuntimeInput::Voxels { channels, .. } => {
                let port = dit.voxels.unwrap_or_else(|| missing("the voxel port"));
                assert_eq!(
                    port.width, channels,
                    "value {at} reads a {channels}-wide voxel port and this fire fed a {}-wide \
                     one: the clips ran in another reading's class",
                    port.width
                );
                port
            }
            RuntimeInput::RowPermutation { select } => packing(select).permutation,
            RuntimeInput::Latents { port: p, .. } => port(engine::fire::PortKind::Latents, p),
            RuntimeInput::LaneVector { port: p, .. } => {
                port(engine::fire::PortKind::LaneVector, p)
            }
            RuntimeInput::Context { port: p, .. } => port(engine::fire::PortKind::Context, p),
            RuntimeInput::AxisPositions { port: p, .. } => {
                port(engine::fire::PortKind::AxisPositions, p)
            }
            RuntimeInput::SelfCondRows => dit
                .self_cond_rows
                .unwrap_or_else(|| missing("self-conditioning taps")),
            RuntimeInput::SelfCondWeights => dit
                .self_cond_weights
                .unwrap_or_else(|| missing("self-conditioning weights")),
            RuntimeInput::Geometry { kind, .. } => match kind {
                GeomKind::GroupOfLane => dit
                    .group_of_lane
                    .unwrap_or_else(|| missing("the group table")),
                GeomKind::GroupIndptr { select } => packing(select).group_indptr,
                GeomKind::LaneIndptr { select } => packing(select).lane_indptr,
                GeomKind::ReferenceTag { select } => packing(select).reference_tag,
                _ => missing("a geometry table"),
            },
            _ => missing("an input"),
        }
    }

    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    pub(crate) fn planes(&self, id: ValueId) -> Bank {
        self.banked(id).unwrap_or_else(|| {
            panic!(
                "value {} is bound as one dense handle, and this op reads a split-plane \
                 bank",
                id.0
            )
        })
    }

    /// A weight stored in a K-quant block format, as its raw bytes.
    pub(crate) fn maybe_stored(&self, id: ValueId) -> Option<Tensor> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            return None;
        };
        let Some(WeightRow::Dense(handle)) = self.weights.0.get(*w as usize).copied().flatten()
        else {
            return None;
        };
        if !matches!(
            handle.dtype,
            Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k
        ) {
            return None;
        }
        Some(self.cut(id, handle))
    }

    pub(crate) fn banked(&self, id: ValueId) -> Option<Bank> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and split-plane banks live in the weight table");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            // A bank's codes reach the kernels as the handle of their stored
            // form (`kernels_xla::pack`): one code per element under the
            // checkpoint's packed dtype, or `E5m2` pre-scaled mxfp4 weights.
            Some(WeightRow::Planes(bank)) => Some(bank),
            Some(WeightRow::Dense(_)) => None,
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    pub(crate) fn pool(&self, id: ValueId) -> &KvPool {
        match self.cache(id) {
            CachePool::Kv(pool) => pool,
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    pub(crate) fn recurrent(&self, id: ValueId) -> RecurrentPool {
        match self.cache(id) {
            CachePool::Recurrent(pool) => RecurrentPool {
                slots: self.cut_rows(pool.slots),
                ..*pool
            },
            CachePool::Kv(_) => panic!(
                "value {} is a paged kv space, and this op scans a recurrent state pool",
                id.0
            ),
        }
    }

    fn cache(&self, id: ValueId) -> &CachePool {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Cache(c) => {
                let row = *c as usize;
                self.caches.0.get(row).unwrap_or_else(|| {
                    panic!(
                        "value {at} is cache space {row}, and the shell binds {} pools",
                        self.caches.0.len()
                    )
                })
            }
            _ => panic!("value {at} is not a cache space; tensors resolve through `Run::tensor`"),
        }
    }

    /// The compressor state planes `[kv, score]` of the kv space `pages`
    /// names.
    pub(crate) fn pool_state(&self, pages: ValueId) -> Option<[Tensor; 2]> {
        let Def::Cache(space) = self.values[pages.0 as usize].def else {
            return None;
        };
        self.compressor.iter().find(|(s, _)| *s == space).map(|(_, p)| *p)
    }

    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        self.structs[at] = Some(built);
    }

    pub(crate) fn decode_plan(&self, id: ValueId) -> &DecodePlan {
        match &self.structs[self.struct_at(id)] {
            Some(StructSlot::Decode(plan)) => plan,
            Some(_) => panic!(
                "value {} holds another plan kind, and this op consumes a decode plan",
                id.0
            ),
            None => panic!(
                "value {} holds no plan payload; its plan op has not fired, and the \
                 prepare phase runs first",
                id.0
            ),
        }
    }

    pub(crate) fn prefill_plan(&self, id: ValueId) -> &PrefillPlan {
        match &self.structs[self.struct_at(id)] {
            Some(StructSlot::Prefill(plan)) => plan,
            Some(_) => panic!(
                "value {} holds another plan kind, and this op consumes a prefill plan",
                id.0
            ),
            None => panic!(
                "value {} holds no plan payload; its plan op has not fired, and the \
                 prepare phase runs first",
                id.0
            ),
        }
    }

    pub(crate) fn mla_plan(&self, id: ValueId) -> &kernels_xla::attn::mla::MlaPlan {
        match &self.structs[self.struct_at(id)] {
            Some(StructSlot::Mla(plan)) => plan,
            Some(_) => panic!(
                "value {} holds another plan kind, and this op consumes an MLA plan",
                id.0
            ),
            None => panic!(
                "value {} holds no plan payload; its plan op has not fired",
                id.0
            ),
        }
    }
}

impl model_exec::fire::fallback::Serve for Run<'_> {}

