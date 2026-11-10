use kernels_slang::attn::mla::MlaPlan;
use kernels_slang::attn::sdpa::Tables;
use kernels_slang::linear::moe::select::RoutedScratch;
use kernels_slang::{Bank, Ctx, KvPool, RaggedTensor, Tensor};

use crate::store::RecurrentPool;
use poem_exec::fire::At;
use poem_ir::{Def, Dim, Dtype, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::device::Handles;
use crate::dispatch::copy::CopyPlan;
use crate::scratch::Scratch;
use crate::window::{Window, Windows};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    Dense(Tensor),

    Planes(Bank),
}

#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

#[derive(Clone, Debug, Default, Hash)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

#[derive(Clone, Copy, Debug, Hash)]
pub enum CachePool {
    Kv(KvPool),

    Recurrent(RecurrentPool),
}

#[derive(Clone, Debug, Default, Hash)]
pub struct CacheTable(pub Vec<CachePool>);

#[derive(Clone, Copy, Debug, Default, Hash)]
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

    pub readout_rows: Tensor,

    pub adapter_routes: Option<Tensor>,

    pub patches: Option<Tensor>,

    pub patch_segments: Option<Tensor>,

    pub patch_routes: Option<Tensor>,

    pub patch_positions: Option<Tensor>,

    pub patch_grids: Option<Tensor>,

    pub mrope_positions: Option<Tensor>,

    pub geometry: Vec<CacheGeometry>,

    /// The page table each window's write descriptors read, by the window
    /// a geometry input states (CONTRACT-BC 21); any other reads table 0.
    pub window_tables: Vec<(u32, usize)>,

    pub tables: FireTables,

    pub rs: Option<std::sync::Arc<crate::rs::Seat>>,

    /// Where the prompt scans land the state as of rows inside a lane, when
    /// a lane of the fire snapshots (CONTRACT-BC 16).
    pub rs_snaps: Option<Tensor>,

    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<(poem_ir::Selection, crate::inputs::PackingHandles)>,
    pub group_masks: Vec<crate::inputs::GroupMaskHandles>,
    pub inputs: Vec<(u32, Tensor)>,
    pub voxels: Option<crate::inputs::VoxelHandles>,
}

#[derive(Clone, Copy, Debug)]
pub enum StructSlot {
    Decode(Tables),

    Prefill(Tables),

    Mla(MlaPlan),
}

pub struct Run<'c> {
    pub(crate) ctx: &'c Ctx<'c>,

    handles: &'c Handles,

    values: &'c [ValueDecl],

    nodes: &'c [Node],

    weights: &'c WeightTable,

    rows: Option<&'c crate::experts::Gathered>,

    host: Option<&'c crate::experts::HostTier>,

    pump: Option<&'c crate::experts::Pump>,

    /// The weight cache's row tables and routed experts, their host copies
    /// by param, the pool their slots lie in, the ids a fire's gathers read
    /// back, and the layer whose routed experts were staged last.
    row_caches: Option<&'c crate::store::rows::RowCaches>,
    expert_caches: Option<&'c crate::store::experts::ExpertCaches>,
    cached_planes: std::collections::BTreeMap<usize, &'c crate::experts::HostPlane>,
    cache_pool: Option<kernels_slang::Segments>,
    row_ids: std::cell::RefCell<Option<(u32, Vec<i32>)>>,
    staged_layer: std::cell::RefCell<Option<crate::store::experts::StageKey>>,

    arena: &'c SlotTable,

    caches: &'c CacheTable,

    structs: Vec<Option<StructSlot>>,

    values_wide: usize,

    fire: FireBindings,

    windows: &'c Windows,

    place: &'c At,

    copy: CopyPlan,

    scratch: &'c Scratch,

    tuning: &'c crate::tuning::Tuning,

    /// The layer whose work the frame holds unsubmitted, when the step is
    /// submitted a layer at a time.
    split: Option<Option<u32>>,
}

impl<'c> Run<'c> {
    pub(crate) fn tuning(&self) -> &'c crate::tuning::Tuning {
        self.tuning
    }

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
        tuning: &'c crate::tuning::Tuning,
        rows: Option<&'c crate::experts::Gathered>,
        host: Option<&'c crate::experts::HostTier>,
        pump: Option<&'c crate::experts::Pump>,
    ) -> Self {
        Self {
            ctx,
            handles,
            values,
            nodes,
            weights,
            rows,
            host,
            pump,
            row_caches: None,
            expert_caches: None,
            cached_planes: std::collections::BTreeMap::new(),
            cache_pool: None,
            row_ids: std::cell::RefCell::new(None),
            staged_layer: std::cell::RefCell::new(None),
            arena,
            caches,
            structs: vec![None; values.len() * windows.max_runs() as usize],
            values_wide: values.len(),
            fire,
            windows,
            place,
            copy: CopyPlan::default(),
            scratch,
            tuning,
            split: None,
        }
    }

    /// Submits the frame each time a node of a new layer has run.
    pub fn flush_between_layers(&mut self) {
        self.split = Some(None);
    }

    /// After `node` runs: submits what the frame holds once a new layer
    /// starts. A failed submit faults the frame, which its commit reports.
    pub(crate) fn split_at(&mut self, node: &Node) {
        let Some(held) = self.split.as_mut() else {
            return;
        };
        let Some(layer) = node.layer else {
            if matches!(
                node.op,
                poem_ir::Operation::Conv(poem_ir::Conv::Conv3d { .. })
            ) && let Err(fault) = crate::probe::flush()
            {
                tracing::warn!(%fault, "submitting a convolution of a split step");
            }
            return;
        };
        if held.is_some_and(|held| held != layer)
            && let Err(fault) = crate::probe::flush()
        {
            tracing::warn!(%fault, "submitting a layer of a split step");
        }
        *held = Some(layer);
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

    pub(crate) fn run_index(&self) -> u32 {
        self.place.run.get()
    }

    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    pub(crate) fn rs_seat(&self) -> Option<std::sync::Arc<crate::rs::Seat>> {
        self.fire.rs.clone()
    }

    /// The snapshot table this window's scans read, when a lane of the fire
    /// snapshots: its requests from the window's first lane.
    pub(crate) fn rs_snapshots(&self) -> Option<kernels_slang::ssm::Snapshots> {
        self.fire
            .rs_snaps
            .map(|table| kernels_slang::ssm::Snapshots {
                table,
                lane0: self.window().span.lane_offset,
            })
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

    pub(crate) fn group_mask(
        &self,
        q_indptr: ValueId,
        kv_indptr: ValueId,
    ) -> Option<crate::inputs::GroupMaskHandles> {
        let packing = |id: ValueId| match self.values.get(id.0 as usize).map(|decl| &decl.def) {
            Some(Def::Input(RuntimeInput::Geometry {
                kind: GeomKind::GroupIndptr { select },
                ..
            })) => Some(*select),
            _ => None,
        };
        let (q, kv) = (packing(q_indptr)?, packing(kv_indptr)?);
        self.fire
            .group_masks
            .iter()
            .find(|mask| mask.q == q && mask.kv == kv)
            .copied()
    }

    pub(crate) fn address(&self, handle: u32) -> Option<(u64, u64)> {
        let row = self.handles.get(handle)?;
        Some((row.slab_id(), row.offset()))
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

    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    fn slice(&self, handle: Tensor, skip: u32, keep: u32) -> Tensor {
        if skip == 0 && keep >= handle.rows {
            return handle;
        }
        let stride = u64::from(handle.width)
            * poem_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
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
            Def::Input(RuntimeInput::Mask)
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
            Some(&Dim::VoxelsScaled(n, d)) => {
                poem_ir::value::scaled_span(n, d, seated.voxel.row_offset, seated.voxel.rows)
            }
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
        if let Def::Input(RuntimeInput::Geometry { kind }) = &self.values[at].def {
            let Some(table) = gathered.page_tables.get(self.table_of(*kind)) else {
                return handle;
            };
            return match kind {
                GeomKind::Indptr => table.page_indptr,
                GeomKind::Indices => table.page_indices,
                GeomKind::LastPageLen => table.last_page_lens,
                GeomKind::KvLen => table.kv_len,

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
                     has not run; `poem_exec::fire::walk` brackets a copied region's \
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

    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    /// The row `grid` addresses `rows`' row 0 at: the fire's grids hold fire-wide rows, so the
    /// window's first row; a grid an op derived counts from the window's first clip, so 0.
    pub(crate) fn grid_base(&self, grid: ValueId, rows: ValueId) -> u32 {
        let seated = self.window();
        if seated.gathered.is_some()
            || !matches!(
                self.values[grid.0 as usize].def,
                Def::Input(RuntimeInput::Grid | RuntimeInput::TokenGrid { .. })
            )
        {
            return 0;
        }
        let Ty::Tensor { shape, .. } = &self.values[rows.0 as usize].ty else {
            return 0;
        };
        match shape.first() {
            Some(Dim::Tokens) => seated.span.row_offset,
            Some(Dim::Voxels) => seated.voxel.row_offset,
            _ => 0,
        }
    }

    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::ReadoutRows) => self.fire.readout_rows,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,

            Def::Input(RuntimeInput::Mask) => self.fire.tables.mask,

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
            Def::Input(RuntimeInput::ImageGrids) => self.fire.patch_grids.unwrap_or_else(|| {
                panic!("value {at} reads this fire's image grids, which no lane of it submitted")
            }),
            Def::Input(RuntimeInput::MropePositions) => {
                self.fire.mrope_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the fire's (t, h, w) token positions, and this \
                         load reserved no triple — the plan declares no multimodal rotation"
                    )
                })
            }
            Def::Input(RuntimeInput::RowPermutation { select }) => {
                self.packing(at, *select).permutation
            }
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
                kind: GeomKind::RequestOfToken,
                ..
            }) if self.fire.geometry.is_empty() => self.fire.tables.request_of_token,
            Def::Input(RuntimeInput::Grid) => self.voxel_seat(at).grid,
            // The seat is as wide as the widest voxel input; each is fed at
            // its own width.
            Def::Input(RuntimeInput::Guest {
                axis: poem_ir::InputAxis::Voxel,
                width,
                ..
            }) => Tensor {
                width: *width,
                ..self.voxel_seat(at).payload
            },
            Def::Input(RuntimeInput::Guest { id, .. }) => self.guest(at, *id),
            Def::Input(RuntimeInput::TokenGrid { .. }) => {
                self.voxel_seat(at).token_grid.unwrap_or_else(|| {
                    panic!("value {at} reads the token side of the patchify pair, which this load carved none of")
                })
            }
            Def::Input(RuntimeInput::Geometry { kind }) => {
                // The All axis's table, or a write descriptor's row's list's.
                let table = self.table_of(*kind);
                let seat = self.fire.geometry.get(table).unwrap_or_else(|| {
                    panic!(
                        "value {at} reads page table {table}, and this fire binds {}",
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
                    GeomKind::WritePage { .. } => seat.write_page,
                    GeomKind::WriteOffset { .. } => seat.write_offset,
                    GeomKind::GroupOfLane
                    | GeomKind::GroupIndptr { .. }
                    | GeomKind::LaneIndptr { .. } => None,
                };
                bound.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads {kind:?} of the page table, which this \
                         fire left unbound"
                    )
                })
            }
            Def::Weight(w) => {
                let row = *w as usize;
                match self.weights.0.get(row).copied().flatten() {
                    Some(WeightRow::Dense(handle)) => self.pumped(handle),
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
                "value {at} is a cache row; it resolves to a pool through `Run::pool`, \
                 never to a tensor"
            ),
        }
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

    fn voxel_seat(&self, at: usize) -> crate::inputs::VoxelHandles {
        self.fire.voxels.unwrap_or_else(|| {
            panic!("value {at} reads the voxel axis, and no lane of this fire submitted a clip")
        })
    }

    fn guest(&self, at: usize, id: u32) -> Tensor {
        self.fire
            .inputs
            .iter()
            .find(|(seat, _)| *seat == id)
            .map(|(_, tensor)| *tensor)
            .unwrap_or_else(|| {
                panic!("value {at} reads guest input {id}, which this load carved no rectangle for")
            })
    }

    fn packing(&self, at: usize, select: poem_ir::Selection) -> crate::inputs::PackingHandles {
        self.fire
            .packings
            .iter()
            .find(|(held, _)| *held == select)
            .map(|(_, packing)| *packing)
            .unwrap_or_else(|| {
                panic!("value {at} reads a packing of {select:?}, which this fire staged none of")
            })
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
        let seated = self.cut(id, handle);
        Some(Tensor::new(
            seated.buf,
            seated.rows,
            seated.width,
            Dtype::U8,
        ))
    }

    pub(crate) fn banked(&self, id: ValueId) -> Option<Bank> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and split-plane banks live in the weight table");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Planes(bank)) => Some(Bank {
                codes: self.pumped(bank.codes),
                scales: self.pumped(bank.scales),
                biases: bank.biases.map(|b| self.pumped(b)),
                ..bank
            }),
            Some(WeightRow::Dense(_)) => None,
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    /// Weight value `id`'s split-plane bank as the load bound it, its
    /// planes not pumped: the scheme a tabled kernel reads it at.
    pub(crate) fn scheme(&self, id: ValueId) -> Bank {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Weight(w) => match self.weights.0.get(*w as usize).copied().flatten() {
                Some(WeightRow::Planes(bank)) => bank,
                _ => {
                    panic!("value {at} is not a split-plane bank, and a quantized select reads one")
                }
            },
            _ => {
                panic!("value {at} is not a weight, and split-plane banks live in the weight table")
            }
        }
    }

    fn pumped(&self, plane: Tensor) -> Tensor {
        let (Some(pump), Some(host)) = (self.pump, self.host) else {
            return plane;
        };
        let Some(spilled) = host.plane(plane.buf) else {
            return plane;
        };
        crate::probe::with_frame(|frame| pump.stage(frame, host.map(), spilled, plane))
            .unwrap_or_else(|| {
                panic!(
                    "weight plane {} is pumped from the artifact and no frame is recording \
                     on this thread",
                    plane.buf
                )
            })
            .unwrap_or_else(|fault| panic!("weight plane {} does not pump: {fault}", plane.buf))
    }

    /// This run with the weight cache's row tables and routed experts:
    /// `planes` their host copies by param, `pool` the pool their slots
    /// lie in.
    #[must_use]
    pub fn with_cache(
        mut self,
        rows: &'c crate::store::rows::RowCaches,
        experts: &'c crate::store::experts::ExpertCaches,
        planes: std::collections::BTreeMap<usize, &'c crate::experts::HostPlane>,
        pool: kernels_slang::Segments,
    ) -> Self {
        if !rows.is_empty() {
            self.row_caches = Some(rows);
        }
        if !experts.is_empty() {
            self.expert_caches = Some(experts);
        }
        if !rows.is_empty() || !experts.is_empty() {
            self.cached_planes = planes;
            self.cache_pool = Some(pool);
        }
        self
    }

    /// The expert table a routed op reading weight values `bank` and
    /// `bias` through `routes` reads, where the weight cache streams them
    /// (`op` names it in a fault).
    pub(crate) fn expert_table(
        &self,
        op: &'static str,
        bank: Option<ValueId>,
        bias: Option<ValueId>,
        routes: Tensor,
    ) -> Result<Option<kernels_slang::linear::moe::select::ExpertTable>, kernels_slang::Error> {
        let (Some(experts), Some(pool), Some(host)) =
            (self.expert_caches, self.cache_pool, self.host)
        else {
            return Ok(None);
        };
        let param = |v: Option<ValueId>| {
            v.and_then(|v| match &self.values[v.0 as usize].def {
                Def::Weight(w) => Some(*w as usize),
                _ => None,
            })
        };
        let Some(planes) = experts.planes_of(param(bank), param(bias)) else {
            return Ok(None);
        };
        experts
            .table(
                self.handles,
                host,
                &self.cached_planes,
                planes,
                routes,
                (self.place.region.get(), self.place.run.get()),
                &self.staged_layer,
                pool,
            )
            .map_err(|why| kernels_slang::Error::Backend {
                op,
                detail: why.to_string(),
            })
    }

    /// Weight value `table`'s row table where the weight cache streams it,
    /// for a gather of `ids`.
    pub(crate) fn row_table(
        &self,
        table: ValueId,
        ids: Tensor,
    ) -> Result<Option<kernels_slang::gather::embed::RowTable>, kernels_slang::Error> {
        let (Some(rows), Some(pool), Some(host)) = (self.row_caches, self.cache_pool, self.host)
        else {
            return Ok(None);
        };
        let Def::Weight(w) = &self.values[table.0 as usize].def else {
            return Ok(None);
        };
        let param = *w as usize;
        let Some(plane) = self.cached_planes.get(&param) else {
            return Ok(None);
        };
        rows.table(self.handles, host, plane, param, ids, &self.row_ids, pool)
            .map_err(|why| kernels_slang::Error::Backend {
                op: "gather.embed",
                detail: why.to_string(),
            })
    }

    pub(crate) fn gathered_table(
        &self,
        bank: Bank,
        ids: Tensor,
    ) -> Result<Option<(Bank, Tensor)>, kernels_slang::Error> {
        let (Some(rows), Some(host)) = (self.rows, self.host) else {
            return Ok(None);
        };
        rows.stage(self.handles, host, bank, ids)
            .map_err(|why| kernels_slang::Error::Backend {
                op: "gather.embed",
                detail: why.to_string(),
            })
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

    pub(crate) fn split_scratch(&self) -> Option<kernels_slang::attn::sdpa::Split> {
        Some(self.scratch.split(self.handles)?.unwrap_or_else(|fault| {
            panic!("the split partials this load reserved do not mint: {fault}")
        }))
    }

    pub(crate) fn pool_state(&self, pages: ValueId) -> Option<PoolSlabs> {
        let at = pages.0 as usize;
        let Some(Def::Cache(row)) = self.values.get(at).map(|v| &v.def) else {
            panic!("value {at} is not a cache row; the pooled state is keyed by one")
        };
        Some(
            self.scratch
                .pool_state(self.handles, *row)?
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

    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The page table geometry input `kind` reads: the All axis's (the
    /// first), or a write descriptor's row's list's.
    fn table_of(&self, kind: GeomKind) -> usize {
        poem_exec::store::kv::geometry_window(kind)
            .and_then(|window| {
                self.fire
                    .window_tables
                    .iter()
                    .find(|(held, _)| *held == window)
                    .map(|&(_, table)| table)
            })
            .unwrap_or(0)
    }

    pub(crate) fn pool(&self, id: ValueId) -> &KvPool {
        match self.cache(id) {
            CachePool::Kv(pool) => pool,
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state row, and this op walks a paged kv pool",
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
                "value {} is a paged kv row, and this op scans a recurrent state pool",
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
                        "value {at} is cache row {row}, and the shell binds {} pools",
                        self.caches.0.len()
                    )
                })
            }
            _ => panic!("value {at} is not a cache row; tensors resolve through `Run::tensor`"),
        }
    }

    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        self.structs[at] = Some(built);
    }

    pub(crate) fn decode_plan(&self, id: ValueId) -> &Tables {
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

    pub(crate) fn prefill_plan(&self, id: ValueId) -> &Tables {
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
