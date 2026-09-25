use kernels_metal::attn::mla::MlaPlan;
use kernels_metal::linear::moe::RoutedScratch;
use kernels_metal::{
    Bank, Ctx, DecodePlan, KvPool, PrefillPlan, RaggedTensor, RecurrentPool, Tensor,
};
use model_ir::{Def, Dim, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::device::Handles;
use crate::dispatch::copy::CopyPlan;
use crate::scratch::Scratch;
use crate::window::{At, Window, Windows};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    Dense(Tensor),

    Planes(Bank),
}

#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    Kv(KvPool),
    Recurrent(RecurrentPool),
}

#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

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

#[derive(Clone, Copy, Debug)]
pub struct PoolSlabs {
    pub state_kv: Tensor,

    pub state_score: Tensor,
}

#[derive(Clone, Debug)]
pub struct FireBindings {
    pub tokens: Tensor,

    pub positions: Tensor,

    pub adapter_routes: Option<Tensor>,

    pub readout_rows: Tensor,

    pub nan_flags: Option<Tensor>,

    pub patches: Option<Tensor>,

    pub patch_segments: Option<Tensor>,

    pub patch_routes: Option<Tensor>,

    pub patch_positions: Option<Tensor>,

    pub patch_embed_rows: Option<Tensor>,

    pub patch_embed_weights: Option<Tensor>,

    pub mrope_positions: Option<Tensor>,

    pub self_cond_rows: Option<Tensor>,
    pub self_cond_weights: Option<Tensor>,

    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<(model_ir::Selection, crate::inputs::PackingHandles)>,

    pub ports: Vec<(engine::fire::PortKind, u8, Tensor)>,

    pub voxels: Option<crate::inputs::VoxelHandles>,

    pub geometry: Vec<CacheGeometry>,

    pub tables: FireTables,

    pub scores: Option<crate::scores::ScoreSeat>,

    pub rs: Option<std::sync::Arc<crate::rs::Seat>>,
}

#[derive(Clone, Copy, Debug)]
pub enum StructSlot {
    Decode(DecodePlan),

    Prefill(PrefillPlan),

    Mla(MlaPlan),
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

    copy: CopyPlan,

    scratch: &'c Scratch,
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
        scratch: &'c Scratch,
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
            copy: CopyPlan::default(),
            scratch,
        }
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

    #[allow(dead_code)]
    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    #[allow(dead_code)]
    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span.rows
    }

    #[allow(dead_code)]
    pub(crate) fn multi_token(&self) -> bool {
        self.qo_indptr_host()
            .windows(2)
            .any(|pair| pair[1] - pair[0] > 1)
    }

    pub(crate) fn cut_rows(&self, handle: Tensor) -> Tensor {
        if let Some(gathered) = &self.window().gathered {
            if handle.buf == self.fire.positions.buf {
                return gathered.positions;
            }
            if handle.buf == self.fire.tables.request_of_token.buf {
                return gathered.request_of_token;
            }
        }
        let span = self.window().span;
        self.slice(handle, span.row_offset, span.rows)
    }

    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }

    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    pub(crate) fn run_index(&self) -> u32 {
        self.place.run.get()
    }

    fn voxel_seat(&self, at: usize) -> crate::inputs::VoxelHandles {
        self.fire.voxels.unwrap_or_else(|| {
            panic!("value {at} reads the voxel axis, and no lane of this fire submitted a clip")
        })
    }

    fn port(&self, at: usize, kind: engine::fire::PortKind, port: u8) -> Tensor {
        self.fire
            .ports
            .iter()
            .find(|(have, index, _)| *have == kind && *index == port)
            .map(|(_, _, plane)| *plane)
            .unwrap_or_else(|| {
                panic!(
                    "value {at} reads the {kind:?} port {port}, and this load carved \
                     {} float port(s)",
                    self.fire.ports.len()
                )
            })
    }

    fn packing(&self, at: usize, select: model_ir::Selection) -> crate::inputs::PackingHandles {
        self.fire
            .packings
            .iter()
            .find(|(have, _)| *have == select)
            .map(|(_, tables)| *tables)
            .unwrap_or_else(|| {
                panic!(
                    "value {at} reads a packing table keyed by {select:?}, and this load \
                     carved {} selection(s)",
                    self.fire.packings.len()
                )
            })
    }

    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    pub(crate) fn address(&self, handle: u32) -> Option<(u64, u64)> {
        let row = self.handles.get(handle)?;
        Some((crate::device::alloc::slab_id(row.slab()), row.offset()))
    }

    pub(crate) fn set_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    pub(crate) fn staged_copy(&self) -> &CopyPlan {
        &self.copy
    }

    pub(crate) fn copy_room(&self, offset: u64, bytes: u64) -> Option<u32> {
        Some(
            self.scratch
                .copy(self.handles, offset, bytes)?
                .unwrap_or_else(|fault| {
                    panic!("the copy rectangle this load reserved does not mint: {fault}")
                }),
        )
    }

    pub(crate) fn handles(&self) -> &'c Handles {
        self.handles
    }

    pub(crate) fn rs_seat(&self) -> Option<std::sync::Arc<crate::rs::Seat>> {
        self.fire.rs.clone()
    }

    pub(crate) fn ctx(&self) -> &'c Ctx<'c> {
        self.ctx
    }

    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    fn slice(&self, handle: Tensor, skip: u32, keep: u32) -> Tensor {
        if skip == 0 && keep >= handle.rows {
            return handle;
        }
        let stride = u64::from(handle.width)
            * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
                panic!(
                    "a {:?} rectangle has no element size and so no row to step by",
                    handle.dtype
                )
            });
        let rows = keep.min(handle.rows.saturating_sub(skip));
        let cut = self
            .handles
            .cut(
                handle.buf,
                u64::from(skip) * stride,
                u64::from(rows) * stride,
            )
            .unwrap_or_else(|fault| {
                panic!(
                    "the window's cut of handle {} at row {skip} for {rows} rows does \
                     not land: {fault}",
                    handle.buf
                )
            });
        Tensor::new(cut, rows, handle.width, handle.dtype)
    }

    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        if self.window().gathered.is_some() {
            return self.compacted(id, handle);
        }
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
        let seated = self.window();
        let window = seated.span;
        let patch = seated.patch;
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
            Some(Dim::Voxels) => (seated.voxel.row_offset, seated.voxel.rows),
            Some(Dim::VoxelsTimes(k)) => (seated.voxel.row_offset * k, seated.voxel.rows * k),
            Some(Dim::Clips) => (seated.voxel.lane_offset, seated.voxel.lanes),
            Some(Dim::ClipsPlus(k)) => (seated.voxel.lane_offset, seated.voxel.lanes + k),
        };
        self.slice(handle, skip, keep)
    }

    fn compacted(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        let gathered = self
            .window()
            .gathered
            .as_ref()
            .expect("`compacted` is reached only through a gathered window");
        if let Def::Input(RuntimeInput::Geometry { space, kind }) = &self.values[at].def {
            let Some(space) = gathered.spaces.get(*space as usize) else {
                return handle;
            };
            return match kind {
                GeomKind::Indptr => space.page_indptr,
                GeomKind::Indices => space.page_indices,
                GeomKind::LastPageLen => space.last_page_lens,
                GeomKind::KvLen => space.kv_len,
                _ => handle,
            };
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        match shape.first() {
            Some(Dim::Tokens) => {
                assert_eq!(
                    self.copy.region,
                    self.place.region.get(),
                    "value {at} is being resolved inside a copied region whose gather \
                     has not run; `model_exec::fire::walk` brackets a copied region's \
                     nodes and this is what says the bracket was lost",
                );
                let Some(key) = self.address(handle.buf) else {
                    panic!(
                        "value {at} is row-shaped and its handle {} resolves to no \
                         binding; every operand of a copied region was minted by this \
                         same fire",
                        handle.buf
                    )
                };
                self.copy.tight(key).unwrap_or_else(|| {
                    panic!(
                        "value {at} is row-shaped and its column was not compacted; the \
                         copy plan is built from the same region's operands the walk is \
                         dispatching, so a miss is a plan and a template built apart"
                    )
                })
            }
            _ => handle,
        }
    }

    pub(crate) fn resolvable(&self, id: ValueId) -> bool {
        matches!(
            self.values.get(id.0 as usize).map(|decl| &decl.def),
            Some(Def::Op(_) | Def::Merge(_))
        ) && self.arena.0.get(id.0 as usize).copied().flatten().is_some()
    }

    pub(crate) fn clip_slots(&self) -> Option<Tensor> {
        self.fire.voxels.map(|voxels| voxels.slots)
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
            Def::Input(RuntimeInput::Patches) => self.fire.patches.unwrap_or_else(|| {
                panic!("value {at} reads this fire's patch rows, which no lane of it submitted")
            }),
            Def::Input(RuntimeInput::PatchSegments) => {
                self.fire.patch_segments.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's image boundaries, which no lane of \
                         it submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchRoutes) => self.fire.patch_routes.unwrap_or_else(|| {
                panic!(
                    "value {at} reads where this fire's tower rows land, which no lane \
                     of it submitted"
                )
            }),
            Def::Input(RuntimeInput::PatchPositions) => {
                self.fire.patch_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's patch grid positions, which no lane \
                         of it submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchEmbedRows) => {
                self.fire.patch_embed_rows.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's position-table taps, and this load \
                         stages none — the plan reads the table on its native grid"
                    )
                })
            }
            Def::Input(RuntimeInput::SelfCondRows) => {
                self.fire.self_cond_rows.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's self-conditioning taps, which this load \
                         reserved no seat for"
                    )
                })
            }
            Def::Input(RuntimeInput::SelfCondWeights) => {
                self.fire.self_cond_weights.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's self-conditioning weights, which this \
                         load reserved no seat for"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchEmbedWeights) => {
                self.fire.patch_embed_weights.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's interpolation weights, and this load \
                         stages none — the plan reads the table on its native grid"
                    )
                })
            }
            Def::Input(RuntimeInput::MropePositions) => {
                self.fire.mrope_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the fire's (t, h, w) token positions, and this \
                         load reserved no triple — the plan declares no multimodal rotation"
                    )
                })
            }
            Def::Input(RuntimeInput::Latents { port, .. }) => {
                self.port(at, engine::fire::PortKind::Latents, *port)
            }
            Def::Input(RuntimeInput::LaneVector { port, .. }) => {
                self.port(at, engine::fire::PortKind::LaneVector, *port)
            }
            Def::Input(RuntimeInput::Context { port, .. }) => {
                self.port(at, engine::fire::PortKind::Context, *port)
            }
            Def::Input(RuntimeInput::AxisPositions { port, .. }) => {
                self.port(at, engine::fire::PortKind::AxisPositions, *port)
            }
            Def::Input(RuntimeInput::RowPermutation { select }) => {
                self.packing(at, *select).permutation
            }
            Def::Input(RuntimeInput::Grid) => self.voxel_seat(at).grid,
            Def::Input(RuntimeInput::TokenGrid { .. }) => {
                self.voxel_seat(at).token_grid.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the token side of the patchify pair, which this \
                         load carved none of"
                    )
                })
            }
            Def::Input(RuntimeInput::Voxels { .. }) => self.voxel_seat(at).payload,
            Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::GroupOfLane,
                ..
            }) => self.fire.group_of_lane.unwrap_or_else(|| {
                panic!("value {at} reads the group-of-lane table, which this load carved none of")
            }),
            Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::GroupIndptr { select },
                ..
            }) => self.packing(at, *select).group_indptr,
            Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::LaneIndptr { select },
                ..
            }) => self.packing(at, *select).lane_indptr,
            Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::ReferenceTag { select },
                ..
            }) => self.packing(at, *select).reference_tag,
            Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::RequestOfToken,
                ..
            }) => self.fire.tables.request_of_token,
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
                    GeomKind::WritePage => seat.write_page,
                    GeomKind::WriteOffset => seat.write_offset,
                    GeomKind::RequestOfToken
                    | GeomKind::GroupOfLane
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

    pub(crate) fn banked(&self, id: ValueId) -> Option<Bank> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and split-plane banks live in the weight table");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Planes(bank)) => Some(bank),
            Some(WeightRow::Dense(_)) => None,
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    pub(crate) fn experts(&self, routes: ValueId) -> u32 {
        self.scratch.experts(routes)
    }

    pub(crate) fn routed_scratch(&self) -> Option<RoutedScratch> {
        Some(self.scratch.routed(self.handles)?.unwrap_or_else(|fault| {
            panic!("the routed scratch this load reserved does not mint: {fault}")
        }))
    }

    pub(crate) fn index_scores(&self) -> Option<Tensor> {
        Some(
            self.scratch
                .index_scores(self.handles)?
                .unwrap_or_else(|fault| {
                    panic!("the index score slab this load reserved does not mint: {fault}")
                }),
        )
    }

    pub(crate) fn pool_state(&self, pages: ValueId) -> Option<PoolSlabs> {
        let at = pages.0 as usize;
        let Some(Def::Cache(space)) = self.values.get(at).map(|v| &v.def) else {
            panic!("value {at} is not a cache space; the pooled state is keyed by one")
        };
        Some(
            self.scratch
                .pool_state(self.handles, *space)?
                .unwrap_or_else(|fault| {
                    panic!("the compressor state this load reserved does not mint: {fault}")
                }),
        )
    }

    pub(crate) fn ple_hash(
        &self,
        mults: &[u64],
        primes: &[u64],
        offsets: &[u64],
    ) -> Option<Tensor> {
        Some(
            self.scratch
                .ple_hash(self.handles, mults, primes, offsets)?
                .unwrap_or_else(|fault| {
                    panic!("the PLE hash plane this load wrote does not mint: {fault}")
                }),
        )
    }

    pub(crate) fn capacity(&self, id: ValueId) -> u32 {
        self.scratch.capacity(id)
    }

    pub(crate) fn precast(&self, rows: u32, contraction: u32) -> Option<Tensor> {
        Some(
            self.scratch
                .precast(self.handles, rows, contraction)?
                .unwrap_or_else(|fault| {
                    panic!("the precast plane this load reserved does not mint: {fault}")
                }),
        )
    }

    pub(crate) fn spatial_moments(&self, clips: u32, groups: u32) -> Option<(Tensor, Tensor)> {
        Some(
            self.scratch
                .spatial_moments(self.handles, clips, groups)?
                .unwrap_or_else(|fault| {
                    panic!("the moment planes this load reserved do not mint: {fault}")
                }),
        )
    }

    pub(crate) fn partials(&self, rows: u32, width: u32) -> Option<Tensor> {
        Some(
            self.scratch
                .partials(self.handles, rows, width)?
                .unwrap_or_else(|fault| {
                    panic!("the partials plane this load reserved does not mint: {fault}")
                }),
        )
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
}
