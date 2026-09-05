use kernels_vulkan::attn::mla::MlaPlan;
use kernels_vulkan::linear::moe::RoutedScratch;
use kernels_vulkan::{
    Bank, Ctx, DecodePlan, KvPool, PrefillPlan, RaggedTensor, RecurrentPool, Tensor,
};
use model_ir::{Def, Dim, Dtype, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::device::Handles;
use crate::device::ctx::Frame;
use crate::dispatch::copy::CopyPlan;
use crate::scratch::Scratch;
use crate::window::{At, Window, Windows};

type Seated = (Option<Bank>, Option<Tensor>, Tensor);

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

    pub patches: Option<Tensor>,

    pub patch_segments: Option<Tensor>,

    pub patch_routes: Option<Tensor>,

    pub patch_positions: Option<Tensor>,

    pub patch_embed_rows: Option<Tensor>,

    pub patch_embed_weights: Option<Tensor>,

    pub mrope_positions: Option<Tensor>,

    pub geometry: Vec<CacheGeometry>,

    pub tables: FireTables,

    pub rs: Option<std::sync::Arc<crate::rs::Seat>>,

    pub scores: Option<crate::scores::ScoreSeat>,
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

    seats: Option<&'c crate::experts::Seats>,

    rows: Option<&'c crate::experts::Gathered>,

    host: Option<&'c crate::experts::HostTier>,

    pump: Option<&'c crate::experts::Pump>,

    stage: std::cell::RefCell<StageCache>,

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

#[derive(Default)]
pub(crate) struct StageCache {
    key: Option<(u32, u32, u32)>,
    unique: Vec<i32>,
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
        seats: Option<&'c crate::experts::Seats>,
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
            seats,
            rows,
            host,
            pump,
            stage: std::cell::RefCell::new(StageCache::default()),
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

    pub(crate) fn rs_seat(&self) -> Option<std::sync::Arc<crate::rs::Seat>> {
        self.fire.rs.clone()
    }

    pub(crate) fn score_seat(&self) -> Option<crate::scores::ScoreSeat> {
        self.fire.scores.clone()
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
            Some(Dim::Const(_)) | None => return handle,
            Some(Dim::Patches) => (patch.row_offset, patch.rows),
            Some(Dim::Images) => (patch.lane_offset, patch.lanes),
            Some(Dim::ImagesPlus(k)) => (patch.lane_offset, patch.lanes + k),
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

    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
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
            Def::Input(RuntimeInput::SelfCondRows | RuntimeInput::SelfCondWeights) => {
                panic!(
                    "value {at} reads a self-conditioning input, which this shell stages none \
                     of; the load refuses such a plan"
                )
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

    pub(crate) fn staged(
        &self,
        bank: Option<Bank>,
        bias: Option<Tensor>,
        routes: Tensor,
    ) -> Result<Option<Seated>, kernels_vulkan::Error> {
        use crate::experts::Kind;
        const OP: &str = "linear.expert_stage";
        let (Some(seats), Some(host)) = (self.seats, self.host) else {
            return Ok(None);
        };
        let codes = bank.and_then(|bank| host.plane(bank.codes.buf));
        let bias_plane = bias.and_then(|b| host.plane(b.buf).map(|p| (b, p)));
        if codes.is_none() && bias_plane.is_none() {
            return Ok(None);
        }
        let fail = |detail: String| kernels_vulkan::Error::Backend { op: OP, detail };
        let key = (routes.buf, routes.rows, routes.width);
        if self.stage.borrow().key != Some(key) {
            let n = u64::from(routes.rows) * u64::from(routes.width);

            crate::probe::with_frame(Frame::flush)
                .ok_or_else(|| fail("no frame is recording on this thread".into()))?
                .map_err(|f| fail(f.to_string()))?;
            let raw = self
                .handles
                .read(routes.buf, n * 4)
                .map_err(|f| fail(f.to_string()))?;
            let mut unique: Vec<i32> = Vec::new();
            let mut seat_of: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
            let mut seated = Vec::with_capacity(raw.len() / 4);
            for word in raw.chunks_exact(4) {
                let e = i32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                if e < 0 {
                    seated.push(-1);
                    continue;
                }
                let seat = *seat_of.entry(e).or_insert_with(|| {
                    unique.push(e);
                    unique.len() as i32 - 1
                });
                seated.push(seat);
            }
            seats
                .write_routes(&seated)
                .map_err(|f| fail(f.to_string()))?;
            *self.stage.borrow_mut() = StageCache {
                key: Some(key),
                unique,
            };
        }
        let cache = self.stage.borrow();
        let mut wanted: Vec<(Kind, Tensor, &crate::experts::HostPlane)> = Vec::with_capacity(4);
        if let (Some(plane), Some(bank)) = (codes, bank) {
            wanted.push((Kind::Codes, bank.codes, plane));
            let scales = host
                .plane(bank.scales.buf)
                .ok_or_else(|| fail("a host-tier bank's scales are not on the host tier".into()))?;
            wanted.push((Kind::Scales, bank.scales, scales));
            if let Some(b) = bank.biases {
                let biases = host.plane(b.buf).ok_or_else(|| {
                    fail("a host-tier bank's zero points are not on the host tier".into())
                })?;
                wanted.push((Kind::Biases, b, biases));
            }
        }
        if let Some((b, plane)) = bias_plane {
            wanted.push((Kind::Dense, b, plane));
        }
        let seated = crate::probe::with_frame(|frame| {
            seats.gather(frame, host.map(), &wanted, &cache.unique)
        })
        .ok_or_else(|| fail("no frame is recording on this thread".into()))?
        .map_err(|f| fail(f.to_string()))?;
        let mut seated = seated.into_iter();
        let mut out_bank = bank;
        if let (Some(out_bank), Some(bank)) = (out_bank.as_mut(), bank)
            && codes.is_some()
        {
            out_bank.codes = seated.next().expect("the codes seat");
            out_bank.scales = seated.next().expect("the scales seat");
            if bank.biases.is_some() {
                out_bank.biases = seated.next();
            }
        }
        let out_bias = match bias_plane {
            Some(_) => seated.next(),
            None => bias,
        };
        Ok(Some((out_bank, out_bias, seats.routes(routes))))
    }

    pub(crate) fn gathered_table(
        &self,
        bank: Bank,
        ids: Tensor,
    ) -> Result<Option<(Bank, Tensor)>, kernels_vulkan::Error> {
        let (Some(rows), Some(host)) = (self.rows, self.host) else {
            return Ok(None);
        };
        rows.stage(self.handles, host, bank, ids)
            .map_err(|why| kernels_vulkan::Error::Backend {
                op: "layout.embed_concat",
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

    pub(crate) fn split_scratch(&self) -> Option<kernels_vulkan::Split> {
        Some(self.scratch.split(self.handles)?.unwrap_or_else(|fault| {
            panic!("the split partials this load reserved do not mint: {fault}")
        }))
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
