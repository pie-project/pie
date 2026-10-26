use kernels_metal::attn::mla::MlaPlan;
use kernels_metal::linear::moe::RoutedScratch;
use kernels_metal::{
    Bank, Ctx, DecodePlan, KvPool, PrefillPlan, RaggedTensor, RecurrentPool, Tensor,
};
use poem_exec::fire::At;
use poem_ir::{Def, Dim, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};
use poem_ir::{NormAxis, NormWeight};

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

#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    Kv(KvPool),
    Recurrent(RecurrentPool),
}

#[derive(Clone, Debug, Default)]
pub struct CacheTable {
    /// One pool per cache space, in space order.
    pub pools: Vec<CachePool>,
    /// The compressor state beside each kv space a pool gather reads.
    pub compressed: Vec<(u32, PoolSlabs)>,
}

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

    /// Each readout row's position, lane and mask flag (`inputs::Handles`).
    pub readout_positions: Tensor,

    pub readout_request_of_token: Tensor,

    pub readout_mask_enabled: Tensor,

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

    pub readout_indptr: Vec<u32>,

    pub nan_flags: Option<Tensor>,

    pub patches: Option<Tensor>,

    pub patch_segments: Option<Tensor>,

    pub patch_routes: Option<Tensor>,

    pub patch_positions: Option<Tensor>,

    pub patch_grids: Option<Tensor>,

    pub mrope_positions: Option<Tensor>,

    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<(poem_ir::Selection, crate::inputs::PackingHandles)>,
    pub group_masks: Vec<crate::inputs::GroupMaskHandles>,

    pub inputs: Vec<(u32, Tensor)>,

    pub voxels: Option<crate::inputs::VoxelHandles>,

    pub geometry: Vec<CacheGeometry>,

    pub tables: FireTables,

    pub rs: Option<std::sync::Arc<crate::rs::Seat>>,

    /// The fire's recurrent snapshot table, when a lane snapshots
    /// (`kernels_metal::ssm::Snapshots`).
    pub rs_snaps: Option<Tensor>,
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

    /// The Neural Engine's share of this load, when a live encode may hand
    /// it work.
    ane: Option<&'c crate::ane::Ane>,

    /// The chip profile and model facts this load's kernels run under.
    tuning: &'c kernels_metal::Tuning,

    /// A routed MLP whose rows stay sorted from gate-up to the fold.
    chain: std::cell::Cell<Option<Chain>>,

    /// A routed MLP on the register expert tiles, from gate-up to the fold
    /// (or to the gated add, where its shared expert rides along).
    tiles: std::cell::Cell<Option<Tiled>>,

    /// Per value, the nodes that read it, made on first ask.
    readers: std::cell::OnceCell<Vec<Vec<u32>>>,

    /// The register tile's input table the precast room holds, if one does.
    table: std::cell::Cell<Option<Laid>>,

    /// The outputs a segment launch landed ahead of the nodes that write them.
    landed: std::cell::Cell<Option<Landed>>,

    /// The compiled regions the walk's place indexes, when the encode names
    /// them: a segment launch takes only nodes of the region it runs in.
    regions: Option<&'c [poem_compiler::Region]>,
}

/// The values a segment launch landed at `rows` of region `region` ahead of
/// the nodes that write them (a later matmul's output, a split's left half):
/// each of those nodes, reached at the same place, has nothing left to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Landed {
    ys: [Option<ValueId>; 3],
    region: u32,
    rows: Rows,
}

/// A register-tile input table standing in the precast room: `act`'s rows as
/// the tile's lanes take them, laid over the window `rows` names, with
/// `counters` arrival counters zeroed. Another matmul of `act` over the same
/// window reads it rather than laying it again, in this region or a later one
/// (a layer's MoE norm and its shared expert sit in different regions);
/// anything else that takes a scratch room the precast room overlaps knocks
/// it down, and `act` is written once, so it cannot change under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Laid {
    act: ValueId,
    rows: Rows,
    contraction: u32,
    counters: u32,
}

/// The rows of a value a walk's place reads: its window's spans (rows,
/// patches, voxels), pass and tail; or, where the window is gathered into the
/// copy room or cut into passes the encoder may skip, the region, run and
/// tail themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rows {
    Span([poem_exec::fire::MaskSpan; 3], u32, bool),
    Region(u32, u32, bool),
}

/// Where a routed MLP that keeps its rows sorted, as Splash's does, has got
/// to: its gate-up left its rows in the routed stack's output, the swiglu
/// moved them to its input, the down left its rows in the output for the
/// fold. Each step runs only on the values the chain named when it began.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Chain {
    step: ChainStep,
    routes: ValueId,
    packed: ValueId,
    hidden: ValueId,
    routed: ValueId,
    padded: u32,
    /// The row tile the down's MPP kernel reads its input sums at, when it takes it.
    sums_tile: Option<u32>,
    /// The activation landed the down's input sums at `sums_tile`.
    sums: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChainStep {
    GateUp,
    Swiglu,
    Down,
}

/// Where a routed MLP on the register expert tiles has got to: its gate-up
/// grouped the routes into expert tiles and landed its SwiGLU as the down's
/// input tables (the swiglu node then runs nothing), the down left each tile's
/// rows in the register rooms for the fold. Where the shared expert rides
/// along as tile 0, the fold, the shared expert and its gate's matmul run
/// nothing, and the gated add lands the whole sum from the rooms.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tiled {
    step: TiledStep,
    routes: ValueId,
    packed: ValueId,
    hidden: ValueId,
    routed: ValueId,
    /// The tiles a fire of these tokens may fill.
    tiles: u32,
    shared: Option<SharedTile>,
    /// The region and run the gate-up ran in.
    region: (u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TiledStep {
    GateUp,
    Swiglu,
    Down,
    /// The fold, the shared expert and its gate, in walk order, ran nothing.
    Folded,
    Shared,
    Gated,
}

/// The shared expert riding along as tile 0: its fused MLP's banks and
/// output, its gate's bank and output, and the gated add's output, which the
/// tiles' combine lands.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedTile {
    gate_up: kernels_metal::Bank,
    down: kernels_metal::Bank,
    y: ValueId,
    gate: ValueId,
    gate_bank: kernels_metal::Bank,
    out: ValueId,
}

/// What the fold of a register chain's rows does.
pub(crate) enum TiledFold {
    /// The gated add lands the sum: the fold runs nothing.
    Deferred,
    /// The sorted fold of the tiles' rows, the route-to-row map and the fan-out.
    Sorted(Tensor, Tensor, u32),
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
        tuning: &'c kernels_metal::Tuning,
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
            ane: None,
            tuning,
            chain: std::cell::Cell::new(None),
            tiles: std::cell::Cell::new(None),
            readers: std::cell::OnceCell::new(),
            table: std::cell::Cell::new(None),
            landed: std::cell::Cell::new(None),
            regions: None,
        }
    }

    /// The single node that reads `value`, if exactly one does.
    pub(crate) fn sole_reader(&self, value: ValueId) -> Option<&'c Node> {
        match self.readers_of(value) {
            &[only] => self.nodes.get(only as usize),
            _ => None,
        }
    }

    /// The nodes that read `value`, in trace order.
    pub(crate) fn readers_of(&self, value: ValueId) -> &[u32] {
        use poem_ir::Operands as _;
        let readers = self.readers.get_or_init(|| {
            let mut readers = vec![Vec::new(); self.values.len()];
            let mut inputs = Vec::new();
            for (at, node) in self.nodes.iter().enumerate() {
                inputs.clear();
                node.op.inputs(&mut inputs);
                inputs.sort_unstable_by_key(|v| v.0);
                inputs.dedup();
                for input in &inputs {
                    if let Some(list) = readers.get_mut(input.0 as usize) {
                        list.push(at as u32);
                    }
                }
            }
            readers
        });
        readers.get(value.0 as usize).map_or(&[], Vec::as_slice)
    }

    /// When a projection of `value` may run on the register tile (a 4-bit,
    /// 64-group bank over at most eight rows, a whole number of the tile's
    /// 64 columns), the arrival counters the widest split one takes: the
    /// producer of `value` then lays the tile's input table with that many
    /// zeroed, and the projections read it. A fused MLP's gate-up is one.
    pub(crate) fn register_readers(&self, value: ValueId) -> Option<u32> {
        use poem_ir::{Fused, Linear, Operation};
        self.readers_of(value)
            .iter()
            .filter_map(|&at| match &self.nodes.get(at as usize)?.op {
                Operation::Linear(
                    Linear::Matmul {
                        act,
                        w,
                        y,
                        quant: None,
                        logits: false,
                        ..
                    }
                    | Linear::Matmul {
                        act,
                        w,
                        y,
                        quant: None,
                        logits: true,
                        ..
                    },
                ) if *act == value => self.register_counters(value, *w, self.tensor(*y).width),
                Operation::Fused(Fused::MatmulSwigluMatmul {
                    act,
                    gate_up,
                    packed,
                    ..
                }) if *act == value => {
                    self.register_counters(value, *gate_up, self.tensor(*packed).width)
                }
                _ => None,
            })
            .reduce(|a, b| a + b)
    }

    /// The per-head norm the trace places right after the one of `x` into `y`,
    /// in the region the walk is in, over the same head width and epsilon and
    /// bf16 rows of its own (a layer's key heads after its query heads): its
    /// input, weight and output, when its rows stand apart from the first's,
    /// so one launch lands both.
    pub(crate) fn norm_pair(
        &self,
        x: ValueId,
        y: ValueId,
        head_dim: u32,
        eps: f32,
    ) -> Option<(ValueId, ValueId, ValueId)> {
        use poem_ir::{Elemwise, Operation};
        let (nodes, _) = self.region_nodes()?;
        let at = self.readers_of(x).iter().copied().find(|&at| {
            matches!(&self.nodes.get(at as usize).map(|n| &n.op),
                Some(Operation::Elemwise(Elemwise::Rmsnorm { x: a, y: b, weight: NormWeight::PlusOne(_), axis: NormAxis::Head(_), .. }))
                    if *a == x && *b == y)
        })?;
        let Some(Operation::Elemwise(Elemwise::Rmsnorm {
            x: x2,
            weight: NormWeight::PlusOne(w2),
            axis: NormAxis::Head(d2),
            eps: e2,
            y: y2,
        })) = self.nodes.get(at as usize + 1).map(|n| &n.op)
        else {
            return None;
        };
        if !nodes.contains(&at)
            || !nodes.contains(&(at + 1))
            || *d2 != head_dim
            || *e2 != eps
            || *x2 == y
        {
            return None;
        }
        let span = |v: ValueId| {
            let t = self.tensor(v);
            (t.dtype == poem_ir::Dtype::Bf16)
                .then(|| self.address(t.buf))
                .flatten()
                .map(|(slab, at)| (slab, at, at + u64::from(t.rows) * u64::from(t.width) * 2))
        };
        let spans = [span(x)?, span(y)?, span(*x2)?, span(*y2)?];
        let apart = |a: (u64, u64, u64), b: (u64, u64, u64)| a.0 != b.0 || a.2 <= b.1 || b.2 <= a.1;
        // The second's rows land apart from everything the first reads and writes, and the
        // first's apart from what the second reads.
        (apart(spans[3], spans[0]) && apart(spans[3], spans[1]) && apart(spans[1], spans[2]))
            .then_some((*x2, *w2, *y2))
    }

    /// The merge of the arms `value` is one of, when the merge's rows here are
    /// the very bytes `value` lands in: the merge's readers then read what the
    /// arm writes, in place.
    pub(crate) fn merged_into(&self, value: ValueId) -> Option<ValueId> {
        let at = self.values.iter().position(|decl| {
            matches!(&decl.def, poem_ir::Def::Merge(arms) if arms.iter().any(|(v, _)| *v == value))
        })?;
        let merged = ValueId(u32::try_from(at).ok()?);
        let (arm, whole) = (self.tensor(value), self.tensor(merged));
        (arm.rows == whole.rows
            && arm.width == whole.width
            && arm.dtype == whole.dtype
            && self.address(arm.buf) == self.address(whole.buf))
        .then_some(merged)
    }

    /// When the projection of `act` by bank `w` into `columns` columns runs on
    /// the register tile, the arrival counters its split takes.
    pub(crate) fn register_counters(&self, act: ValueId, w: ValueId, columns: u32) -> Option<u32> {
        let x = self.tensor(act);
        if !self.tuning.qmm_register()
            || x.dtype != poem_ir::Dtype::Bf16
            || x.rows == 0
            || x.rows > 8
            || !x.width.is_multiple_of(256)
            || !columns.is_multiple_of(64)
        {
            return None;
        }
        let bank = self.banked(w)?;
        if bank.bits != 4 || bank.group != 64 || bank.biases.is_none() {
            return None;
        }
        // Only where the profile runs the register tile first (not one row's
        // decode, which runs the qmv): a table nothing reads is work for none.
        let at = self
            .tuning
            .qmm_at(x.rows, columns, x.width, bank.bits, bank.group);
        if self.tuning.qmm(&at).first() != Some(&kernels_metal::tuning::Qmm::Register) {
            return None;
        }
        let (cols, k) = (i32::try_from(columns).ok()?, i32::try_from(x.width).ok()?);
        let split = kernels_metal::linear::qmm::sg::register_split(cols, k, self.tuning.cores());
        Some(if split > 1 { columns / 64 } else { 0 })
    }

    pub(crate) fn chain_clear(&self) {
        self.chain.set(None);
    }

    /// How a batched routed matmul of `x` over `routes` into `y` stands in a
    /// sorted chain: a gate-up whose output only a gated activation reads, whose output
    /// only a down over the same routes reads, whose output only a fold
    /// reads, starts one; that down continues it. Anything else runs as it
    /// always has.
    pub(crate) fn chain_select(
        &self,
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    ) -> kernels_metal::linear::moe::Sorted {
        use kernels_metal::linear::moe::Sorted;
        use poem_ir::{Linear, Operation};
        if let Some(chain) = self.chain.get()
            && chain.step == ChainStep::Swiglu
            && chain.hidden == x
            && chain.routes == routes
            && chain.routed == y
        {
            self.chain.set(Some(Chain {
                step: ChainStep::Down,
                ..chain
            }));
            return Sorted {
                input: true,
                output: true,
                sums: chain.sums,
            };
        }
        self.chain.set(None);
        let (packed, hidden) = match self.sole_reader(y).map(|n| &n.op) {
            Some(Operation::Linear(
                Linear::Glu {
                    packed: Some(packed),
                    y: hidden,
                    gate: None,
                    weights: None,
                    limit: None,
                    act: poem_ir::GluAct::Swiglu,
                    ..
                }
                | Linear::Glu {
                    packed: Some(packed),
                    y: hidden,
                    gate: None,
                    weights: None,
                    limit: Some(_),
                    act: poem_ir::GluAct::SwigluAlpha { .. },
                    ..
                }
                | Linear::Glu {
                    packed: Some(packed),
                    y: hidden,
                    gate: None,
                    weights: None,
                    limit: None,
                    act: poem_ir::GluAct::GegluTanh,
                    ..
                },
            )) => (packed, hidden),
            _ => return Sorted::default(),
        };
        let Some(Operation::Linear(Linear::MoeSelect {
            x: down_x,
            bank: down_bank,
            routes: down_routes,
            y: routed,
            bias: None,
            quantized: true,
            quant: None,
            ..
        })) = self.sole_reader(*hidden).map(|n| &n.op)
        else {
            return Sorted::default();
        };
        let Some(Operation::Linear(Linear::MoeCombine {
            routed: folded,
            weights: Some(_),
            shared: None,
            ..
        })) = self.sole_reader(*routed).map(|n| &n.op)
        else {
            return Sorted::default();
        };
        if *packed != y || down_x != hidden || *down_routes != routes || folded != routed {
            return Sorted::default();
        }
        let stack = |x, bank, y| {
            kernels_metal::linear::moe::sorted_stack(
                self.tensor(x),
                self.planes(bank),
                self.tensor(y),
                self.experts(routes),
                self.tuning,
            )
        };
        // The down reads the gate-up's sorted rows, so both passes sort alike.
        let Some((padded, tile)) = stack(x, bank, y) else {
            return Sorted::default();
        };
        if stack(*hidden, *down_bank, *routed) != Some((padded, tile)) {
            return Sorted::default();
        }
        let sums_tile = self.routed_scratch().and_then(|scratch| {
            kernels_metal::linear::moe::mpp_sums_tile(
                self.tensor(*hidden),
                self.planes(*down_bank),
                self.tensor(*routed),
                self.experts(routes),
                scratch.sums,
                self.tuning,
            )
        });
        self.chain.set(Some(Chain {
            step: ChainStep::GateUp,
            routes,
            packed: y,
            hidden: *hidden,
            routed: *routed,
            padded,
            sums_tile,
            sums: false,
        }));
        Sorted {
            input: false,
            output: true,
            sums: false,
        }
    }

    /// The sorted gate-up rows, where their swiglu lands and which rows are
    /// padding, when `packed` is the chain's gate-up output.
    pub(crate) fn chain_swiglu(
        &self,
        packed: ValueId,
        hidden: ValueId,
    ) -> Option<(Tensor, Tensor, Tensor)> {
        let chain = self.chain.get()?;
        if chain.step != ChainStep::GateUp || chain.packed != packed || chain.hidden != hidden {
            self.chain.set(None);
            return None;
        }
        let scratch = self.routed_scratch()?;
        let (up, inter) = (self.tensor(packed).width, self.tensor(hidden).width);
        self.chain.set(Some(Chain {
            step: ChainStep::Swiglu,
            ..chain
        }));
        Some((
            Tensor::new(scratch.y.buf, chain.padded, up, poem_ir::Dtype::Bf16),
            Tensor::new(scratch.x.buf, chain.padded, inter, poem_ir::Dtype::Bf16),
            scratch.perm,
        ))
    }

    /// Where the chain's activation lands the down's input sums and at which
    /// row tile, when the down's MPP kernel reads them; the down then skips its
    /// sums pass. Only an activation that writes them asks.
    pub(crate) fn chain_sums(&self) -> Option<(Tensor, u32)> {
        let chain = self.chain.get()?;
        if chain.step != ChainStep::Swiglu {
            return None;
        }
        let tile = chain.sums_tile?;
        let scratch = self.routed_scratch()?;
        self.chain.set(Some(Chain {
            sums: true,
            ..chain
        }));
        Some((scratch.sums, tile))
    }

    /// The sorted down rows, the route-to-row map and the fan-out, when
    /// `routed` is the chain's down output; the chain ends here.
    pub(crate) fn chain_combine(&self, routed: ValueId) -> Option<(Tensor, Tensor, u32)> {
        let chain = self.chain.take()?;
        if chain.step != ChainStep::Down || chain.routed != routed {
            return None;
        }
        let scratch = self.routed_scratch()?;
        let routes = self.tensor(chain.routes);
        let width = self.tensor(routed).width;
        Some((
            Tensor::new(scratch.y.buf, chain.padded, width, poem_ir::Dtype::Bf16),
            scratch.inv,
            routes.width,
        ))
    }

    /// The register expert tiles' rooms, which overlap the precast room.
    pub(crate) fn register_rooms(&self) -> Option<kernels_metal::linear::moe::RegisterRooms> {
        self.table.set(None);
        Some(
            self.scratch
                .register(self.handles)?
                .unwrap_or_else(|fault| {
                    panic!("the register expert rooms this load reserved do not mint: {fault}")
                }),
        )
    }

    /// Whether the routed gate-up of `x` by `bank` over `routes` into `y`
    /// starts a routed MLP on the register expert tiles: 2 to 8 tokens (the
    /// rows of one tile an expert), a gate-up whose output only a SwiGLU
    /// reads, whose output only a down over the same routes reads, whose
    /// output only a fold reads, each bank one the tile takes. Where the fold,
    /// the shared expert of `x` (a fused MLP), its gate's matmul and their
    /// gated add follow in turn in this region, that expert rides along. The
    /// chain then runs at each of its nodes; the gate-up's runs here.
    pub(crate) fn tiles_start(
        &self,
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    ) -> Result<bool, kernels_metal::Error> {
        use kernels_metal::linear::moe;
        use poem_ir::{Linear, Operation};
        self.tiles.set(None);
        if !self.tuning.moe_register() {
            return Ok(false);
        }
        let (xt, rt) = (self.tensor(x), self.tensor(routes));
        if rt.rows < 2
            || rt.rows > moe::REGISTER_TOKENS
            || rt.width > moe::REGISTER_FAN_OUT
            || xt.rows != rt.rows
            || self.windows.runs(self.place.region.get()) > 1
        {
            return Ok(false);
        }
        let Some(Operation::Linear(Linear::Glu {
            packed: Some(packed),
            intermediate,
            y: hidden,
            gate: None,
            weights: None,
            limit: None,
            act: poem_ir::GluAct::Swiglu,
            ..
        })) = self.sole_reader(y).map(|n| &n.op)
        else {
            return Ok(false);
        };
        let Some(Operation::Linear(Linear::MoeSelect {
            x: down_x,
            bank: down_bank,
            routes: down_routes,
            y: routed,
            bias: None,
            quantized: true,
            quant: None,
            ..
        })) = self.sole_reader(*hidden).map(|n| &n.op)
        else {
            return Ok(false);
        };
        let &[fold_at] = self.readers_of(*routed) else {
            return Ok(false);
        };
        let Some(fold) = self.nodes.get(fold_at as usize) else {
            return Ok(false);
        };
        let Operation::Linear(Linear::MoeCombine {
            routed: summed,
            weights: Some(weights),
            y: folded,
            shared: None,
        }) = &fold.op
        else {
            return Ok(false);
        };
        if *packed != y || down_x != hidden || *down_routes != routes || summed != routed {
            return Ok(false);
        }
        let inter = *intermediate;
        let (gate_up, down) = (self.planes(bank), self.planes(*down_bank));
        let columns = self.tensor(*routed).width;
        if self.tensor(y).width != 2 * inter
            || !moe::register_cuts(inter, columns, moe::REGISTER_CUT)
            || !moe::register_bank(gate_up, xt.width, 2 * inter)
            || !moe::register_bank(down, inter, columns)
        {
            return Ok(false);
        }
        let experts = self.experts(routes);
        if experts == 0 || experts > moe::REGISTER_EXPERTS {
            return Ok(false);
        }
        let shared = self.shared_tile(x, fold_at, fold, *folded, inter, columns);
        let tiles = moe::register_tiles(rt.rows, rt.width, experts, shared.is_some());
        let Some(rooms) = self.register_rooms() else {
            return Ok(false);
        };
        if rooms.tiles.width <= tiles
            || u64::from(rooms.table.width) < u64::from(tiles) * moe::register_table_width(xt.width)
            || u64::from(rooms.hidden.width) < u64::from(tiles) * moe::register_table_width(inter)
            || u64::from(rooms.y.rows) < 8 * u64::from(tiles)
            || rooms.y.width < columns
            || rooms.inv.width < rt.rows * rt.width
        {
            return Ok(false);
        }
        let gate = shared.map(|s| moe::RegisterGate {
            weights: self.tensor(*weights),
            gate: s.gate_bank,
        });
        moe::register_group(
            self.ctx(),
            xt,
            rt,
            experts,
            shared.is_some(),
            gate,
            moe::REGISTER_CUT.groups(inter, columns),
            rooms,
        )?;
        moe::register_glu(
            self.ctx(),
            gate_up,
            shared.map(|s| s.gate_up),
            xt.width,
            inter,
            tiles,
            experts,
            rooms,
            moe::REGISTER_CUT,
        )?;
        self.tiles.set(Some(Tiled {
            step: TiledStep::GateUp,
            routes,
            packed: y,
            hidden: *hidden,
            routed: *routed,
            tiles,
            shared,
            region: (self.place.region.get(), self.place.run.get()),
        }));
        Ok(true)
    }

    /// The shared expert of `x` that rides along as tile 0: the nodes after
    /// the fold (`fold_at`, landing `folded`) are, in turn, the shared expert
    /// (a fused MLP of `x`), its gate's matmul of `x` (a bank of one column)
    /// and their sigmoid-gated add, which alone reads the fold, the expert and
    /// the gate; the walk runs them in turn with nothing between
    /// ([`Run::walks_in_turn`]).
    fn shared_tile(
        &self,
        x: ValueId,
        fold_at: u32,
        fold: &Node,
        folded: ValueId,
        inter: u32,
        columns: u32,
    ) -> Option<SharedTile> {
        use kernels_metal::linear::moe;
        use poem_ir::{Fused, Linear, Operation};
        if !self.walks_in_turn(fold_at, fold_at + 3) {
            return None;
        }
        let at = |n: u32| {
            self.nodes
                .get((fold_at + n) as usize)
                .filter(|node| node.guard == fold.guard)
        };
        let Operation::Fused(Fused::MatmulSwigluMatmul {
            act,
            gate_up,
            down,
            intermediate,
            y,
            ..
        }) = &at(1)?.op
        else {
            return None;
        };
        let Operation::Linear(Linear::Matmul {
            act: gate_act,
            w: gate_w,
            y: gate,
            input_scales: None,
            scales: None,
            quant: None,
            logits: false,
        }) = &at(2)?.op
        else {
            return None;
        };
        let Operation::Linear(Linear::MoeSigmoidGateAdd {
            routed,
            shared,
            gate: gated,
            y: out,
        }) = &at(3)?.op
        else {
            return None;
        };
        let add = fold_at + 3;
        if *act != x
            || *gate_act != x
            || *intermediate != inter
            || *routed != folded
            || shared != y
            || gated != gate
            || [folded, *y, *gate]
                .iter()
                .any(|v| self.readers_of(*v) != [add])
        {
            return None;
        }
        let (gate_up, down) = (self.banked(*gate_up)?, self.banked(*down)?);
        let gate_bank = self.banked(*gate_w)?;
        let width = self.tensor(x).width;
        let rows = self.tensor(*out);
        (self.tensor(*y).width == columns
            && rows.width == columns
            && rows.rows == self.tensor(x).rows
            && moe::register_bank(gate_up, width, 2 * inter)
            && moe::register_bank(down, inter, columns)
            && moe::register_gate_bank(gate_bank, width))
        .then_some(SharedTile {
            gate_up,
            down,
            y: *y,
            gate: *gate,
            gate_bank,
            out: *out,
        })
    }

    /// Whether the walk runs the nodes `first..=last` in turn, each once over
    /// the same rows, with nothing between: their regions (a layer's fold
    /// sits in the region of its routed passes, its shared expert in the
    /// next) are no arms of a switch (the walk's sink runs an `If` region's
    /// nodes as any other), each in one run over one window of the same rows,
    /// in one pass.
    fn walks_in_turn(&self, first: u32, last: u32) -> bool {
        let Some(regions) = self.regions else {
            return false;
        };
        let Some(from) = regions.iter().position(|r| r.nodes.contains(&first)) else {
            return false;
        };
        let mut rows = None;
        for (at, region) in regions.iter().enumerate().skip(from) {
            let Ok(index) = u32::try_from(at) else {
                return false;
            };
            if matches!(region.lowering, poem_compiler::Lowering::Switch { .. })
                || region.phase != regions[from].phase
                || self.windows.runs(index) != 1
            {
                return false;
            }
            let window = self.windows.at(index, 0);
            if window.gathered.is_some()
                || window.passes > 1
                || window.span.rows == 0
                || *rows.get_or_insert(window.span.rows) != window.span.rows
            {
                return false;
            }
            if region.nodes.contains(&last) {
                return true;
            }
        }
        false
    }

    /// Whether the SwiGLU of `packed` into `hidden` is the register chain's,
    /// which its gate-up already landed: it then runs nothing.
    pub(crate) fn tiles_swiglu(&self, packed: ValueId, hidden: ValueId) -> bool {
        let here = self.tiles_at(TiledStep::GateUp, |t| {
            t.packed == packed && t.hidden == hidden
        });
        if here {
            self.tiles_step(TiledStep::Swiglu);
        }
        here
    }

    /// Runs the routed down of `x` over `routes` into `y` on the register
    /// tiles, when it is the register chain's: true where it ran.
    pub(crate) fn tiles_down(
        &self,
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    ) -> Result<bool, kernels_metal::Error> {
        if !self.tiles_at(TiledStep::Swiglu, |t| {
            t.hidden == x && t.routes == routes && t.routed == y
        }) {
            return Ok(false);
        }
        let tiled = self.tiles.get().expect("the chain is at its down");
        let rooms = self
            .register_rooms()
            .expect("the register chain began over the rooms the load reserved");
        kernels_metal::linear::moe::register_down(
            self.ctx(),
            self.planes(bank),
            tiled.shared.map(|s| s.down),
            self.tensor(x).width,
            self.tensor(y).width,
            tiled.tiles,
            self.experts(routes),
            rooms,
            kernels_metal::linear::moe::REGISTER_CUT,
        )?;
        self.tiles_step(TiledStep::Down);
        Ok(true)
    }

    /// The fold of `routed`, when it is the register chain's down's output:
    /// deferred to the gated add where the shared expert rides along, else
    /// the sorted fold of the tiles' rows (eight a tile), which ends the chain.
    pub(crate) fn tiles_fold(&self, routed: ValueId) -> Option<TiledFold> {
        if !self.tiles_at(TiledStep::Down, |t| t.routed == routed) {
            return None;
        }
        let tiled = self.tiles.get()?;
        if tiled.shared.is_some() {
            self.tiles_step(TiledStep::Folded);
            return Some(TiledFold::Deferred);
        }
        self.tiles.set(None);
        let rooms = self
            .register_rooms()
            .expect("the register chain began over the rooms the load reserved");
        Some(TiledFold::Sorted(
            Tensor::new(
                rooms.y.buf,
                tiled.tiles * 8,
                self.tensor(routed).width,
                poem_ir::Dtype::Bf16,
            ),
            rooms.inv,
            self.tensor(tiled.routes).width,
        ))
    }

    /// Whether the fused MLP landing `y` is the shared expert that rides
    /// along: it then runs nothing, its rows tile 0 of the chain's down.
    pub(crate) fn tiles_shared(&self, y: ValueId) -> bool {
        let here = self.tiles_at(TiledStep::Folded, |t| t.shared.is_some_and(|s| s.y == y));
        if here {
            self.tiles_step(TiledStep::Shared);
        }
        here
    }

    /// Whether the matmul landing `y` is the riding shared expert's gate,
    /// which the grouping landed: it then runs nothing.
    pub(crate) fn tiles_gate(&self, y: ValueId) -> bool {
        let here = self.tiles_at(TiledStep::Shared, |t| t.shared.is_some_and(|s| s.gate == y));
        if here {
            self.tiles_step(TiledStep::Gated);
        }
        here
    }

    /// Lands the gated add's `y` from the register chain's rooms, when it is
    /// the chain's: true where it ran, which ends the chain.
    pub(crate) fn tiles_gate_add(&self, y: ValueId) -> Result<bool, kernels_metal::Error> {
        if !self.tiles_at(TiledStep::Gated, |t| t.shared.is_some_and(|s| s.out == y)) {
            return Ok(false);
        }
        let tiled = self.tiles.get().expect("the chain is at its gated add");
        self.tiles.set(None);
        let rooms = self
            .register_rooms()
            .expect("the register chain began over the rooms the load reserved");
        kernels_metal::linear::moe::register_combine(
            self.ctx(),
            rooms,
            tiled.tiles,
            self.tensor(tiled.routes).width,
            self.tensor(y),
        )?;
        Ok(true)
    }

    /// Whether the register chain stands at `step` and `here` names its next
    /// node. A chain that deferred its fold must reach its gated add with
    /// nothing between: its start checked that the walk runs them in turn.
    fn tiles_at(&self, step: TiledStep, here: impl Fn(&Tiled) -> bool) -> bool {
        let Some(tiled) = self.tiles.get() else {
            return false;
        };
        let at = tiled.step == step && here(&tiled);
        assert!(
            at || !matches!(
                tiled.step,
                TiledStep::Folded | TiledStep::Shared | TiledStep::Gated
            ),
            "a routed MLP whose shared expert rides on the register tiles walks its fold, \
             shared expert, gate and gated add in turn"
        );
        assert!(
            !at || self.tiles_reach(&tiled),
            "a routed MLP on the register tiles walks its nodes over the rows its gate-up laid"
        );
        at
    }

    fn tiles_step(&self, step: TiledStep) {
        if let Some(tiled) = self.tiles.get() {
            self.tiles.set(Some(Tiled { step, ..tiled }));
        }
    }

    /// Whether a node of the register chain walked here reads the rows its
    /// gate-up laid: in the region and run the gate-up ran in, or in a region
    /// of one run after a gate-up whose region was too, each over every row.
    fn tiles_reach(&self, tiled: &Tiled) -> bool {
        let here = (self.place.region.get(), self.place.run.get());
        let whole = |region: u32| self.windows.runs(region) <= 1;
        tiled.region == here || (whole(tiled.region.0) && whole(here.0))
    }

    pub(crate) fn tuning(&self) -> &'c kernels_metal::Tuning {
        self.tuning
    }

    #[must_use]
    pub(crate) fn with_ane(mut self, ane: Option<&'c crate::ane::Ane>) -> Self {
        self.ane = ane;
        self
    }

    /// The compiled regions the walk's place indexes.
    #[must_use]
    pub(crate) fn with_regions(mut self, regions: &'c [poem_compiler::Region]) -> Self {
        self.regions = Some(regions);
        self
    }

    /// The nodes of the region the walk is in, and whether its rows run in one
    /// pass over one window (no gathered copy, no tail cut), so every node of
    /// it reads the same rows: `None` where the encode names no regions.
    pub(crate) fn region_nodes(&self) -> Option<(std::ops::Range<u32>, bool)> {
        let region = self.regions?.get(self.place.region.get() as usize)?;
        let window = self.window();
        Some((
            region.nodes.clone(),
            window.gathered.is_none() && window.passes <= 1,
        ))
    }

    pub(crate) fn ane(&self) -> Option<&'c crate::ane::Ane> {
        self.ane
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

    fn guest(&self, at: usize, id: u32) -> Tensor {
        self.fire
            .inputs
            .iter()
            .find(|(have, _)| *have == id)
            .map(|(_, plane)| *plane)
            .unwrap_or_else(|| {
                panic!("value {at} reads guest input {id}, which this load carved no rectangle for")
            })
    }

    fn packing(&self, at: usize, select: poem_ir::Selection) -> crate::inputs::PackingHandles {
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
        Some((crate::device::alloc::slab_id(row.slab()), row.offset()))
    }

    pub(crate) fn set_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    pub(crate) fn staged_copy(&self) -> &CopyPlan {
        &self.copy
    }

    pub(crate) fn copy_room(&self, offset: u64, bytes: u64) -> Option<u32> {
        self.table.set(None);
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

    /// Where this window's prompt scans land the state as of rows inside a
    /// lane, when a lane of the fire snapshots.
    pub(crate) fn rs_snapshots(&self) -> Option<kernels_metal::ssm::Snapshots> {
        self.fire
            .rs_snaps
            .map(|table| kernels_metal::ssm::Snapshots {
                table,
                lane0: self.window().span.lane_offset,
            })
    }

    pub(crate) fn ctx(&self) -> &'c Ctx<'c> {
        self.ctx
    }

    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    pub(crate) fn slice(&self, handle: Tensor, skip: u32, keep: u32) -> Tensor {
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
            Some(Dim::Readouts) => {
                readout_span(&self.fire.readout_indptr, window.lane_offset, window.lanes)
            }
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

    pub(crate) fn resolvable(&self, id: ValueId) -> bool {
        matches!(
            self.values.get(id.0 as usize).map(|decl| &decl.def),
            Some(Def::Op(_) | Def::Merge(_))
        ) && self.arena.0.get(id.0 as usize).copied().flatten().is_some()
    }

    /// The fire's clip slots cut to this window's clips, indexed as its grids index lanes.
    pub(crate) fn clip_slots(&self) -> Option<Tensor> {
        let slots = self.fire.voxels.map(|voxels| voxels.slots)?;
        let seated = self.window();
        if seated.gathered.is_some() {
            return Some(slots);
        }
        Some(self.slice(slots, seated.voxel.lane_offset, seated.voxel.lanes))
    }

    /// `mask`'s bits, else the fire's for a non-causal prefill (its classes stage every key).
    pub(crate) fn mask_bits(&self, mask: Option<ValueId>) -> Tensor {
        mask.map_or(self.fire.tables.mask, |mask| self.tensor(mask))
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
            Def::Input(RuntimeInput::Grid) => self.voxel_seat(at).grid,
            Def::Input(RuntimeInput::TokenGrid { .. }) => {
                self.voxel_seat(at).token_grid.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the token side of the patchify pair, which this \
                         load carved none of"
                    )
                })
            }
            Def::Input(RuntimeInput::Guest {
                axis: poem_ir::InputAxis::Voxel,
                ..
            }) => self.voxel_seat(at).payload,
            Def::Input(RuntimeInput::Guest { id, .. }) => self.guest(at, *id),
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
                    | GeomKind::LaneIndptr { .. } => None,
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
        self.table.set(None);
        self.routed_rooms()
    }

    /// The routed scratch, for a pass that may yet decline it: one that runs
    /// over it knocks the register table down itself ([`Run::knock_table`]).
    pub(crate) fn routed_rooms(&self) -> Option<RoutedScratch> {
        Some(self.scratch.routed(self.handles)?.unwrap_or_else(|fault| {
            panic!("the routed scratch this load reserved does not mint: {fault}")
        }))
    }

    /// Forgets the register table: a pass wrote a scratch room the precast
    /// room overlaps.
    pub(crate) fn knock_table(&self) {
        self.table.set(None);
    }

    pub(crate) fn index_scores(&self) -> Option<Tensor> {
        self.table.set(None);
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
        self.caches
            .compressed
            .iter()
            .find(|(held, _)| *held == *space)
            .map(|&(_, slabs)| slabs)
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
        self.table.set(None);
        Some(
            self.scratch
                .precast(self.handles, rows, contraction)?
                .unwrap_or_else(|fault| {
                    panic!("the precast plane this load reserved does not mint: {fault}")
                }),
        )
    }

    pub(crate) fn spatial_moments(&self, clips: u32, groups: u32) -> Option<(Tensor, Tensor)> {
        self.table.set(None);
        Some(
            self.scratch
                .spatial_moments(self.handles, clips, groups)?
                .unwrap_or_else(|fault| {
                    panic!("the moment planes this load reserved do not mint: {fault}")
                }),
        )
    }

    /// The register tile's input table for `act` in the precast room, and
    /// whether it already stands there, laid by another matmul or the
    /// producer of `act` over the same window with at least `counters` arrival counters
    /// zeroed (each split matmul's last partition puts its counters back to
    /// zero, so the next reader finds them so). A table laid now is
    /// remembered as standing: the caller lays it before anything else runs.
    pub(crate) fn register_table(
        &self,
        act: ValueId,
        rows: u32,
        contraction: u32,
        counters: u32,
    ) -> Option<(Tensor, bool)> {
        let room = self
            .scratch
            .precast(self.handles, rows, contraction)?
            .unwrap_or_else(|fault| {
                panic!("the precast plane this load reserved does not mint: {fault}")
            });
        let window = self.window();
        let rows = if window.gathered.is_some() || window.passes > 1 {
            Rows::Region(
                self.place.region.get(),
                self.place.run.get(),
                self.place.tail.get(),
            )
        } else {
            Rows::Span(
                [window.span, window.patch, window.voxel],
                window.pass,
                self.place.tail.get(),
            )
        };
        let standing = self.table.get().is_some_and(|laid| {
            laid.act == act
                && laid.rows == rows
                && laid.contraction == contraction
                && laid.counters >= counters
        });
        if !standing {
            self.table.set(Some(Laid {
                act,
                rows,
                contraction,
                counters,
            }));
        }
        Some((room, standing))
    }

    /// The rows of a value the walk's place reads ([`Rows`]).
    fn rows_here(&self) -> Rows {
        let window = self.window();
        if window.gathered.is_some() || window.passes > 1 {
            Rows::Region(
                self.place.region.get(),
                self.place.run.get(),
                self.place.tail.get(),
            )
        } else {
            Rows::Span(
                [window.span, window.patch, window.voxel],
                window.pass,
                self.place.tail.get(),
            )
        }
    }

    /// Records that a segment launch here landed `ys` ahead of the nodes that
    /// write them.
    pub(crate) fn land_ahead(&self, ys: [Option<ValueId>; 3]) {
        self.landed.set(Some(Landed {
            ys,
            region: self.place.region.get(),
            rows: self.rows_here(),
        }));
    }

    /// Whether a segment launch at this place already landed `y`; it is then
    /// struck off, and the node writing it has nothing to do. Anywhere else
    /// a matmul runs again, which lands the same values.
    pub(crate) fn landed_ahead(&self, y: ValueId) -> bool {
        let Some(mut landed) = self.landed.get() else {
            return false;
        };
        if landed.region != self.place.region.get() || landed.rows != self.rows_here() {
            self.landed.set(None);
            return false;
        }
        let Some(at) = landed.ys.iter().position(|&v| v == Some(y)) else {
            return false;
        };
        landed.ys[at] = None;
        self.landed
            .set(landed.ys.iter().any(Option::is_some).then_some(landed));
        true
    }

    /// The register tile's input table room for `value`, laid by the op that
    /// writes `value` (with the counters its widest register projection may
    /// take zeroed), when a projection of it runs on the register tile:
    /// recorded as standing, so the projections read it.
    pub(crate) fn producer_table(&self, value: ValueId) -> Option<(Tensor, u32)> {
        self.laid_table(value, self.register_readers(value)?)
    }

    /// [`Run::producer_table`] for a producer that knows its projection's
    /// counters (a fused MLP's swiglu, whose down the same node runs).
    pub(crate) fn laid_table(&self, value: ValueId, counters: u32) -> Option<(Tensor, u32)> {
        let width = self.tensor(value).width;
        // The swiglu, gate and convolution producers stage at most 8192 of a row's inputs.
        if width > 8192 {
            return None;
        }
        let rows = (17 * width + 4 * counters).div_ceil(2 * width);
        let (table, _) = self.register_table(value, rows, width, counters)?;
        Some((table, counters))
    }

    pub(crate) fn partials(&self, rows: u32, width: u32) -> Option<Tensor> {
        if !self.scratch.partials_clear_of_precast() {
            self.table.set(None);
        }
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
                self.caches.pools.get(row).unwrap_or_else(|| {
                    panic!(
                        "value {at} is cache space {row}, and the shell binds {} pools",
                        self.caches.pools.len()
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

    /// The decode plan `id` as a query of readout rows reads it: each row
    /// attends as the token it reads out, its position, lane and mask flag
    /// those of that token, over this window's readouts.
    pub(crate) fn readout_plan(&self, id: ValueId) -> DecodePlan {
        let plan = *self.decode_plan(id);
        let window = self.window();
        assert!(
            window.gathered.is_none(),
            "a decode over readout rows runs in a window of whole lanes, and this one \
             was gathered"
        );
        let (skip, keep) = readout_span(
            &self.fire.readout_indptr,
            window.span.lane_offset,
            window.span.lanes,
        );
        let t = &self.fire.tables;
        DecodePlan {
            positions: self.slice(t.readout_positions, skip, keep),
            request_of_token: self.slice(t.readout_request_of_token, skip, keep),
            mask_enabled: self.slice(t.readout_mask_enabled, skip, keep),
            ..plan
        }
    }

    /// Whether `id`'s rows are the fire's readouts rather than its tokens.
    pub(crate) fn reads_out(&self, id: ValueId) -> bool {
        matches!(
            &self.values[id.0 as usize].ty,
            Ty::Tensor { shape, .. } if shape.first() == Some(&Dim::Readouts)
        )
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

fn readout_span(indptr: &[u32], lane_offset: u32, lanes: u32) -> (u32, u32) {
    let first = indptr[lane_offset as usize];
    let end = indptr[(lane_offset + lanes) as usize];
    (first, end - first)
}

#[cfg(test)]
mod readout_tests {
    use super::readout_span;
    #[test]
    fn lane_windows_preserve_readout_order_and_duplicates() {
        let rows = [0, 512, 1, 1, 520, 515];
        let indptr = [0, 4, 6];
        let select = |lane, count| {
            let (first, rows_count) = readout_span(&indptr, lane, count);
            &rows[first as usize..(first + rows_count) as usize]
        };
        assert_eq!(select(0, 1), &[0, 512, 1, 1]);
        assert_eq!(select(1, 1), &[520, 515]);
        assert_eq!(select(0, 2), &rows);
        assert!(select(2, 0).is_empty());
    }
}
