//! How the cache rows sit in the state pool's chunks: the compiler's chunk
//! recipe ([`poem_compiler::chunk::plan`]) read for this engine's paging.
//!
//! The pool is `total` chunks of `chunk_bytes`, chunk `p` at
//! `base + p * chunk_bytes`. A chunk holds slots of one shape. A **list**
//! slot is one kv page of every full-retention row, layer after layer, and a
//! **window list** slot one page of every windowed row; a row's key plane and then
//! its value plane run where the recipe puts the row.
//!
//! A paged kernel reads page `q` of a plane at `plane base + q * grain`
//! ([`kernels_cuda::KvPool::page_stride`]). `grain` is the largest step both
//! a chunk and the shape's slot are whole numbers of, so every slot of every
//! chunk has a `q`: slot `i` of physical chunk `p` is
//! `(p * chunk_bytes + i * slot_bytes) / grain`. The recipe makes a chunk a
//! whole number of list slots, so a list slot's `q` is its slot ref.

use engine::chunk::ShapeId;
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};
use poem_ir::Trace;

use crate::error::{Fault, Result};

/// The list shape's id in the facts this engine states.
pub const LIST: ShapeId = ShapeId(0);
/// The first (narrowest) window list's; the lists after it are
/// [`window_id`]'s.
pub const WINDOW: ShapeId = ShapeId(1);
/// The window lists a load may state (CONTRACT-BC 21).
pub const MOST_WINDOWS: usize = 4;
/// The ids of the window lists past the first start here, clear of the
/// weight shapes' ([`WEIGHTS`] on).
const MORE_WINDOWS: u32 = 1 << 16;

/// Window list `at`'s id in the facts this engine states, narrowest first.
#[must_use]
pub const fn window_id(at: usize) -> ShapeId {
    if at == 0 {
        WINDOW
    } else {
        ShapeId(MORE_WINDOWS + at as u32)
    }
}

/// The window list shape id `id` names, if it names one.
#[must_use]
pub fn window_of(id: ShapeId) -> Option<usize> {
    (0..MOST_WINDOWS).find(|&at| window_id(at) == id)
}
/// The fixed shape's.
pub const FIXED: ShapeId = ShapeId(2);
/// The rs list's.
pub const RS: ShapeId = ShapeId(3);

/// The guest shapes' ids start here, clear of the weight shapes' and the
/// window lists'.
const GUEST: u32 = 1 << 17;
/// A guest instance's own tables (`SlotRole::Instance`): its rings'
/// registry and the native cells of the rings the host reads or writes.
pub const INSTANCE: ShapeId = ShapeId(GUEST);
/// A ring only the device reads and writes (`SlotRole::Channel`).
pub const CHANNEL: ShapeId = ShapeId(GUEST + 1);
/// An `Instance` slot's bytes (`new-memory/STATE-ROLES.md` 3).
pub const INSTANCE_SLOT: u64 = 4 << 10;
/// A `Channel` slot's bytes.
pub const CHANNEL_SLOT: u64 = 1 << 10;
/// An adapter's planes (`SlotRole::Adapter`): one slot a bank.
pub const ADAPTER: ShapeId = ShapeId(GUEST + 2);
/// The adapter ids a fire's tables keep naming zero planes: what a load's
/// own synthetic fires route their adapted rows to.
pub const ZERO_IDS: u32 = 2;

/// One bank an adapter has a plane of: the trace's param (a `registered`
/// weight `[rows, ..]`, an adapter's plane its `[..]`), its name, and the
/// bytes of one adapter's plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterBank {
    pub param: u32,
    pub name: String,
    pub plane: u64,
}

/// The load's adapters (`SlotRole::Adapter`): an adapter is one slot a
/// bank, bank `k`'s plane in slot `k` of its refs, and is read through one
/// table a bank of `ids` entries, an address an id (the reserved run's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterShape {
    pub slot_bytes: u64,
    pub per_chunk: u32,
    /// In name order: the order an adapter's slots hold their planes.
    pub banks: Vec<AdapterBank>,
    /// [`ZERO_IDS`] ids that name zero planes, then one a lane a fire
    /// carries.
    pub ids: u32,
}

impl AdapterShape {
    /// The banks of `trace` (its `registered` weights), a slot the largest
    /// plane's bytes; `None` where it has none.
    pub fn of(trace: &Trace, chunk_bytes: u64, lanes: u32) -> Result<Option<AdapterShape>> {
        let mut banks: Vec<AdapterBank> = trace
            .params
            .iter()
            .enumerate()
            .filter(|(_, param)| param.source == poem_ir::ParamSource::Registered)
            .map(|(at, param)| {
                let elems: u64 = param.shape.get(1..).unwrap_or(&[]).iter().product();
                let elem = poem_compiler::arena::elem_bytes(param.dtype).unwrap_or(0);
                AdapterBank {
                    param: at as u32,
                    name: param.name.clone(),
                    plane: elems * elem,
                }
            })
            .collect();
        if banks.is_empty() {
            return Ok(None);
        }
        banks.sort_by(|a, b| a.name.cmp(&b.name));
        let slot_bytes = banks
            .iter()
            .map(|bank| bank.plane)
            .max()
            .unwrap_or(0)
            .max(1)
            .next_multiple_of(TABLE_ALIGN);
        if slot_bytes > chunk_bytes {
            return Err(Fault::Unbound {
                what: format!(
                    "an adapter plane of {slot_bytes} bytes, past a chunk of {chunk_bytes}: \
                     a plane lies in one `Adapter` slot"
                ),
            });
        }
        Ok(Some(AdapterShape {
            slot_bytes,
            per_chunk: u32::try_from(chunk_bytes / slot_bytes).unwrap_or(u32::MAX),
            banks,
            ids: ZERO_IDS + lanes.max(1),
        }))
    }

    /// What an adapter takes: a slot a bank, anywhere.
    #[must_use]
    pub fn footprint(&self) -> engine::program::Footprint {
        engine::program::Footprint {
            bytes: self.banks.iter().map(|bank| bank.plane).sum(),
            slots: self.banks.len() as u32,
            contiguous: false,
        }
    }

    /// The bytes of one bank's table.
    #[must_use]
    pub fn table_bytes(&self) -> u64 {
        table_span(u64::from(self.ids) * 8)
    }

    /// The adapters' region of the reserved run: a zero plane, then each
    /// bank's table.
    #[must_use]
    pub fn region_bytes(&self) -> u64 {
        table_span(self.slot_bytes) + self.banks.len() as u64 * self.table_bytes()
    }
}

/// One of the two guest shapes: fixed slots an item takes a run of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestShape {
    pub id: ShapeId,
    pub role: engine::chunk::SlotRole,
    pub slot_bytes: u64,
    pub per_chunk: u32,
}

impl GuestShape {
    /// The slots an item of `bytes` takes: one, or one run of adjacent
    /// slots it is addressed through as a span.
    #[must_use]
    pub fn footprint(&self, bytes: u64) -> engine::program::Footprint {
        let slots = u32::try_from(bytes.div_ceil(self.slot_bytes)).unwrap_or(u32::MAX);
        engine::program::Footprint {
            bytes,
            slots,
            contiguous: slots > 1,
        }
    }
}

/// Which of a load's slot shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A page of every full-retention kv row.
    List,
    /// A page of every kv row read through one window: window list `.0`,
    /// narrowest first.
    Window(u8),
    /// A working set's record of every recurrent row.
    Fixed,
    /// A page of the rs buffer: the rows a lane lands past its state before
    /// they fold, every recurrent layer's.
    Rs,
}

impl Kind {
    /// Every kind, the window lists a load may state among them.
    pub const ALL: [Kind; Kind::COUNT] = [
        Kind::List,
        Kind::Fixed,
        Kind::Rs,
        Kind::Window(0),
        Kind::Window(1),
        Kind::Window(2),
        Kind::Window(3),
    ];

    /// The shape the pages of a kv row of list `list` are slots of: 0 the
    /// kv list, `n` window list `n - 1`.
    #[must_use]
    pub const fn paged(list: u8) -> Kind {
        if list == 0 {
            Kind::List
        } else {
            Kind::Window(list - 1)
        }
    }

    /// A place of its own among the kinds, for a table a kind.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Kind::List => 0,
            Kind::Fixed => 1,
            Kind::Rs => 2,
            Kind::Window(at) => 3 + at as usize,
        }
    }

    /// The kinds [`Kind::index`] numbers.
    pub const COUNT: usize = 3 + MOST_WINDOWS;

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Kind::List => "kv page",
            Kind::Window(_) => "windowed page",
            Kind::Fixed => "state slot",
            Kind::Rs => "rs buffer page",
        }
    }
}

/// One slot shape: the pages of the kv rows that share a page table, or the
/// record of the recurrent rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotShape {
    pub id: ShapeId,
    /// Tokens a slot holds: the page size (0 for the fixed shape).
    pub tokens: u32,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    /// The step page ids count in.
    pub grain: u64,
    /// The window its rows are read through; `None` for the list.
    pub window: Option<u32>,
    /// Each cache row laid in the slot, and where its run starts.
    pub rows: Vec<(u32, u64)>,
    /// Each compressor state laid in the slot, by the cache row its gather
    /// keys it by, and where its two planes start.
    pub pools: Vec<(u32, u64)>,
}

impl SlotShape {
    fn of(layout: &SlotLayout, chunk_bytes: u64, id: ShapeId) -> Result<SlotShape> {
        let (tokens, bytes, window, runs) = match layout {
            SlotLayout::List {
                tokens,
                bytes,
                rows,
                window,
            } => (*tokens, *bytes, *window, rows),
            SlotLayout::RsList {
                tokens,
                bytes,
                rows,
            } => (*tokens, *bytes, None, rows),
            SlotLayout::Fixed { .. } => {
                return Err(Fault::program(
                    "store::layout",
                    "a fixed slot read as a token slot",
                ));
            }
        };
        let per_chunk = u32::try_from(chunk_bytes / bytes.max(1)).unwrap_or(u32::MAX);
        if bytes == 0 || per_chunk == 0 {
            return Err(Fault::program(
                "store::layout",
                format!("a token slot of {bytes} bytes fits no chunk of {chunk_bytes}"),
            ));
        }
        Ok(SlotShape {
            id,
            tokens,
            slot_bytes: bytes,
            per_chunk,
            grain: gcd(chunk_bytes, bytes),
            window,
            rows: runs
                .iter()
                .filter_map(|run| match run.row {
                    RowId::Cache(row) => Some((row, run.offset)),
                    RowId::Rs(_) | RowId::Pool(_) => None,
                })
                .collect(),
            pools: runs
                .iter()
                .filter_map(|run| match run.row {
                    RowId::Pool(row) => Some((row, run.offset)),
                    RowId::Cache(_) | RowId::Rs(_) => None,
                })
                .collect(),
        })
    }

    /// Where the compressor state keyed by cache row `row` starts in a slot.
    #[must_use]
    pub fn pool_of(&self, row: u32) -> Option<u64> {
        self.pools
            .iter()
            .find(|(held, _)| *held == row)
            .map(|&(_, offset)| offset)
    }

    /// Where cache row `row`'s run starts in a slot.
    #[must_use]
    pub fn offset_of(&self, row: u32) -> Option<u64> {
        self.rows
            .iter()
            .find(|(held, _)| *held == row)
            .map(|&(_, offset)| offset)
    }

    /// Bytes from the pool's base to slot `index` of physical chunk `chunk`.
    #[must_use]
    pub fn slot_at(&self, chunk_bytes: u64, chunk: u32, index: u32) -> u64 {
        u64::from(chunk) * chunk_bytes + u64::from(index) * self.slot_bytes
    }

    /// The page id a kernel reads that slot by.
    #[must_use]
    pub fn page_id(&self, chunk_bytes: u64, chunk: u32, index: u32) -> u64 {
        self.slot_at(chunk_bytes, chunk, index) / self.grain
    }

    #[must_use]
    pub fn slot_ref(&self, chunk: u32, index: u32) -> u32 {
        chunk.saturating_mul(self.per_chunk).saturating_add(index)
    }
}

/// The slot shapes of a load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub chunk_bytes: u64,
    /// Chunks a segment, by the recipe's device profile.
    pub segment_chunks: u32,
    pub list: Option<SlotShape>,
    /// The window lists, narrowest first: the kv rows read through one
    /// window page by one table.
    pub windows: Vec<SlotShape>,
    pub fixed: Option<FixedShape>,
    pub rs: Option<RsShape>,
    /// The tables the weight cache streams by their rows, a shape each.
    pub rows: Vec<RowsShape>,
    /// The routed experts the weight cache streams, a shape a layer size.
    pub experts: Vec<ExpertShape>,
    /// The adapters' planes, where the trace has banks.
    pub adapter: Option<AdapterShape>,
}

/// The first weight shape's id; the shapes after it follow.
pub const WEIGHTS: u32 = 4;

/// A table a token-id gather reads, streamed by its rows: a slot holds a
/// block of `block_rows` rows of `row_bytes`, of the table's `rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowsShape {
    pub id: ShapeId,
    /// The trace's param the table is.
    pub param: u32,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    pub block_rows: u32,
    pub row_bytes: u64,
    /// The table's rows.
    pub rows: u64,
    /// The blocks the table makes: its atoms.
    pub blocks: u32,
    /// The blocks one fire may read: what the cache never drops below.
    pub floor: u32,
}

/// The boundary each plane of an expert's slot lies on.
pub const PLANE_ALIGN: u64 = 16;

/// One plane of a routed expert: the bank `param`, an expert's `bytes` of
/// it `offset` into the expert's slot, and the row a TMA map of it indexes
/// (`row`, 0 where none is known): a code, bias or zeros row, or a 128-row
/// block of scales (`poem_compiler::chunk::weights::tma_rows`), stated where
/// this device's kernels map windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPlane {
    pub param: u32,
    pub offset: u64,
    pub bytes: u64,
    pub row: u64,
}

/// The routed experts of the MoE layers of one size: a slot holds one
/// expert of one layer, its plane of each bank the layer's selects read.
/// Atom `layer * experts + expert` is expert `expert` of `banks[layer]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertShape {
    pub id: ShapeId,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    /// Experts a layer.
    pub experts: u32,
    /// Each layer's planes.
    pub banks: Vec<Vec<ExpertPlane>>,
    /// The slots a fire's routes may read: what the cache never drops below.
    pub floor: u32,
}

impl ExpertShape {
    /// The atoms there are.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.banks.len() as u32 * self.experts
    }
}

/// Whether ROCm lays MXFP4 bank `codes` in the matrix core's lane order
/// (`weights::hip::order_mxfp4_banks`): an `[experts, n, ..]` bank of n in
/// whole 16-row strips and K in whole 128-code units, under e8m0 exponents
/// alone, that only routed selects of bf16 rows read (the plain quantized
/// select, or its e4m3-quantized twin). Never on another backend.
#[must_use]
pub fn lane_ordered(trace: &poem_ir::Trace, codes: usize) -> bool {
    use poem_ir::{ActQuant, Def, Dtype, Fused, Linear, Operands, Operation, Ty};
    if !cfg!(feature = "rocm") {
        return false;
    }
    let Some(param) = trace.params.get(codes) else {
        return false;
    };
    if param.dtype != Dtype::Mxfp4
        || param.shape.len() < 3
        || param.shape[1] % 16 != 0
        || (2 * param.shape[2..].iter().product::<u64>()) % 128 != 0
        || select_bias(trace, codes).is_some_and(|bias| bias.is_some())
    {
        return false;
    }
    let reads_it = |id: poem_ir::ValueId| {
        matches!(trace.values.get(id.0 as usize).map(|decl| &decl.def),
            Some(Def::Weight(w)) if *w as usize == codes)
    };
    let bf16 = |id: poem_ir::ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|decl| &decl.ty),
            Some(Ty::Tensor {
                dtype: Dtype::Bf16,
                ..
            })
        )
    };
    let (mut reads, mut selects) = (0u32, 0u32);
    let mut ins = Vec::new();
    for node in &trace.nodes {
        ins.clear();
        node.op.inputs(&mut ins);
        reads += ins.iter().filter(|&&id| reads_it(id)).count() as u32;
        let legs: Vec<&Linear> = match &node.op {
            Operation::Linear(op) => vec![op],
            Operation::Fused(
                Fused::MoeExperts { select, down, .. } | Fused::MoeMega { select, down, .. },
            ) => vec![&**select, &**down],
            Operation::Fused(Fused::MoeSelectGlu { select, .. }) => vec![&**select],
            _ => Vec::new(),
        };
        for leg in legs {
            let x = match leg {
                Linear::MoeSelect {
                    x,
                    bank,
                    bias: None,
                    quantized: true,
                    quant: None | Some(ActQuant::E4m3 { .. }),
                    ..
                } if reads_it(*bank) => *x,
                _ => continue,
            };
            if bf16(x) {
                selects += 1;
            }
        }
    }
    selects > 0 && selects == reads
}

/// The planes an expert's entry of a packed bank's table lists
/// (`kernels_cuda::linear::moe::Planes`), by the bank's codes: a block-FP8
/// bank's `[codes, atoms]` (its scales laid at load), an MXFP4
/// bank's `[codes, scales, bias]`, an NVFP4 bank's `[codes, scales,
/// tensor_scales]`, an mlx affine bank's (2, 4 or 8 bits) `[codes, scales,
/// zeros]`; `None` a plane the bank lacks. Empty for a param that is no
/// packed bank this engine streams.
#[must_use]
pub fn packed_planes(trace: &poem_ir::Trace, codes: usize) -> Vec<Option<u32>> {
    let Some(param) = trace.params.get(codes) else {
        return Vec::new();
    };
    use poem_ir::Dtype;
    let leads = poem_ir::residency::leads(trace);
    let named = |name: String| {
        (0..trace.params.len())
            .find(|at| *at != codes && leads[*at] == codes && trace.params[*at].name == name)
            .map(|at| at as u32)
    };
    let scales = named(poem::declare::scales_name(&param.name));
    // A block-FP8 (or MXFP8) bank whose scales the load lays in atoms, an
    // expert's whole 128-row blocks: `[codes, atoms]`.
    if matches!(
        param.dtype,
        Dtype::E4m3 | Dtype::E4m3tile128 | Dtype::E4m3g128
    ) {
        let laid = crate::weights::laid_planes(trace).into_iter().any(|laid| {
            laid.codes == codes && laid.leads > 1 && Some(laid.scales as u32) == scales
        });
        return match (laid, scales) {
            (true, Some(scales)) => vec![Some(codes as u32), Some(scales)],
            _ => Vec::new(),
        };
    }
    let third = match param.dtype {
        // Its expert bias where its select adds one (gpt-oss).
        Dtype::Mxfp4 => match select_bias(trace, codes) {
            Some(bias) => bias,
            None => return Vec::new(),
        },
        Dtype::Nvfp4 => match named(poem::declare::global_scales_name(&param.name)) {
            Some(tensor_scales) => Some(tensor_scales),
            None => return Vec::new(),
        },
        // mlx affine codes of 2, 4 or 8 bits (`matmul_select_mlxu4`).
        Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128
        | Dtype::U4g32
        | Dtype::U4g64
        | Dtype::U8g64 => match named(poem::declare::biases_name(&param.name)) {
            Some(zeros) => Some(zeros),
            None => return Vec::new(),
        },
        _ => return Vec::new(),
    };
    // An MXFP4 bank ROCm lays in the matrix core's lane order: its codes
    // and exponents, as the lane-order selects read them (`[codes, scales]`).
    if param.dtype == Dtype::Mxfp4 && third.is_none() && lane_ordered(trace, codes) {
        return match scales {
            Some(scales) => vec![Some(codes as u32), Some(scales)],
            None => Vec::new(),
        };
    }
    match scales {
        Some(scales) => vec![Some(codes as u32), Some(scales), third],
        None => Vec::new(),
    }
}

/// The expert bias the routed selects of bank `codes` add: `Some(None)`
/// where none adds one, `None` where they add different ones (no one table
/// lists them all).
fn select_bias(trace: &poem_ir::Trace, codes: usize) -> Option<Option<u32>> {
    use poem_ir::{Def, Fused, Linear, Operation};
    let weight = |v: poem_ir::ValueId| match trace.values.get(v.0 as usize).map(|d| &d.def) {
        Some(Def::Weight(w)) => Some(*w),
        _ => None,
    };
    // Every routed select, plain or inside the fused op a load folds it
    // into (the trace a load seats its banks by is the fused one).
    let selects = trace.nodes.iter().flat_map(|node| match &node.op {
        Operation::Linear(select) => vec![select],
        Operation::Fused(Fused::MoeSelectGlu { select, .. }) => vec![&**select],
        Operation::Fused(
            Fused::MoeExperts { select, down, .. } | Fused::MoeMega { select, down, .. },
        ) => vec![&**select, &**down],
        _ => Vec::new(),
    });
    let mut found: Option<Option<u32>> = None;
    for select in selects {
        let Linear::MoeSelect { bank, bias, .. } = select else {
            continue;
        };
        let (bank, bias) = (*bank, *bias);
        if weight(bank) != Some(codes as u32) {
            continue;
        }
        let this = match bias {
            Some(bias) => Some(weight(bias)?),
            None => None,
        };
        match found {
            Some(was) if was != this => return None,
            _ => found = Some(this),
        }
    }
    Some(found.flatten())
}

/// The planes of a packed bank that hold one value an expert (an NVFP4
/// bank's input scales, its activations' calibration): no expert slot
/// carries them and its table lists none of them; they stay on the device
/// beside the bank, read where they lie.
#[must_use]
pub fn beside_planes(trace: &poem_ir::Trace, codes: usize) -> Vec<u32> {
    use poem_ir::residency::{Residency, residency};
    let listed = packed_planes(trace, codes);
    if listed.is_empty() {
        return Vec::new();
    }
    let classes = residency(trace);
    let leads = poem_ir::residency::leads(trace);
    (0..trace.params.len())
        .filter(|&plane| {
            plane != codes
                && leads[plane] == codes
                && matches!(classes[plane], Residency::Expert(_))
                && !listed.contains(&Some(plane as u32))
                && trace.params[plane].shape.len() == 1
        })
        .map(|plane| plane as u32)
        .collect()
}

/// The routed banks of `trace` this engine streams an expert at a time:
/// planes out of the checkpoint read only at the `bank` port of a plain
/// routed select with no bias, the select whose kernels read an expert
/// table: a dense bank, or (off tcgen05, whose grouped GEMM reads no table)
/// an MXFP4, NVFP4 or affine u4 bank against bf16 activations or (W4A8,
/// W4A4) against the rows its select quantizes, with the planes its table
/// lists (a bank ROCm lays in lane order, its lane-ordered planes).
#[must_use]
pub fn streamed_experts(trace: &poem_ir::Trace) -> Vec<usize> {
    use poem_ir::residency::{Residency, residency};
    use poem_ir::{Def, Linear, Operands, Operation};
    let classes = residency(trace);
    let leads = poem_ir::residency::leads(trace);
    let packs = !packed_unread();
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    'params: for (at, param) in trace.params.iter().enumerate() {
        if !matches!(classes[at], Residency::Expert(_))
            || param.shape.len() < 2
            || param.source != poem_ir::ParamSource::Checkpoint
        {
            continue;
        }
        let planes = packed_planes(trace, at);
        let mut read = false;
        for node in &trace.nodes {
            inputs.clear();
            node.op.inputs(&mut inputs);
            let reads = inputs.iter().any(|v| {
                matches!(trace.values.get(v.0 as usize).map(|d| &d.def),
                    Some(Def::Weight(w)) if *w as usize == at)
            });
            if !reads {
                continue;
            }
            match &node.op {
                Operation::Linear(Linear::MoeSelect {
                    bank,
                    bias: None,
                    quantized: false,
                    quant: None,
                    ..
                }) if matches!(trace.values.get(bank.0 as usize).map(|d| &d.def),
                    Some(Def::Weight(w)) if *w as usize == at) =>
                {
                    read = true;
                }
                // An expert bias summed into the routed rows (gpt-oss's
                // down leg): a one-plane bank the sum reads through its
                // table.
                Operation::Linear(Linear::MoeBiasSum { bias, .. })
                    if packs
                        && planes.is_empty()
                        && matches!(trace.values.get(bias.0 as usize).map(|d| &d.def),
                            Some(Def::Weight(w)) if *w as usize == at) =>
                {
                    read = true;
                }
                // An MXFP4 bank's biased select: its bias is the third plane
                // its table lists.
                Operation::Linear(Linear::MoeSelect {
                    bank,
                    bias: Some(_),
                    quantized: false,
                    quant: None,
                    ..
                }) if packs
                    && param.dtype == poem_ir::Dtype::Mxfp4
                    && planes.get(2).is_some_and(Option::is_some)
                    && matches!(trace.values.get(bank.0 as usize).map(|d| &d.def),
                        Some(Def::Weight(w)) if *w as usize == at) =>
                {
                    read = true;
                }
                Operation::Linear(Linear::MoeSelect {
                    bank,
                    bias: None,
                    quantized: true,
                    quant: None,
                    ..
                }) if packs
                    && !planes.is_empty()
                    && matches!(trace.values.get(bank.0 as usize).map(|d| &d.def),
                        Some(Def::Weight(w)) if *w as usize == at) =>
                {
                    read = true;
                }
                // A select quantizing its rows (W4A8, W4A4): where there are no
                // tcgen05 GEMMs its block-scaled mma.sync reads a held table and
                // its bf16-activation arm a streamed one, both in the bank's
                // row-major planes.
                Operation::Linear(Linear::MoeSelect {
                    bank,
                    bias: None,
                    quantized: true,
                    quant: Some(_),
                    ..
                }) if packs
                    && !planes.is_empty()
                    && matches!(trace.values.get(bank.0 as usize).map(|d| &d.def),
                        Some(Def::Weight(w)) if *w as usize == at) =>
                {
                    read = true;
                }
                _ => continue 'params,
            }
        }
        if !read {
            continue;
        }
        // A packed bank streams with every plane of it, each an expert's:
        // the planes its table lists, and no other that lies an expert at a
        // time (it would have no entry).
        let companions: Vec<usize> = planes
            .iter()
            .flatten()
            .map(|plane| *plane as usize)
            .filter(|plane| *plane != at)
            .collect();
        let beside = beside_planes(trace, at);
        let unlisted = (0..trace.params.len()).any(|plane| {
            plane != at
                && leads[plane] == at
                && matches!(classes[plane], Residency::Expert(_))
                && !companions.contains(&plane)
                && !beside.contains(&(plane as u32))
        });
        if !unlisted
            && companions.iter().all(|plane| {
                matches!(classes[*plane], Residency::Expert(_))
                    && trace.params[*plane].source == poem_ir::ParamSource::Checkpoint
            })
        {
            out.push(at);
            out.extend(companions);
        }
    }
    // A layer's banks stream together or not at all, and so do the layers
    // the recipe lays in one shape: an atom is an expert of every bank of
    // its layer. The grouping is the recipe's, whatever its chunk.
    let profile = poem_compiler::DeviceProfile {
        chunk: poem_compiler::chunk::ChunkProfile::SERVER_CUDA,
        ..poem_compiler::DeviceProfile::default()
    };
    let budgets = poem_compiler::Budgets::of(poem_compiler::Budget::default());
    let view = crate::weights::laid_view(trace);
    let Ok(recipe) = poem_compiler::chunk::plan(&view, &budgets, &profile) else {
        return Vec::new();
    };
    // A plane held beside its bank stays out, and keeps no layer out.
    let beside: Vec<usize> = out
        .iter()
        .flat_map(|&bank| beside_planes(trace, bank))
        .map(|plane| plane as usize)
        .collect();
    let mut whole = Vec::new();
    for slot in &recipe.weights {
        if slot.kind != poem_compiler::chunk::WeightKind::Expert {
            continue;
        }
        let params = || {
            slot.banks
                .iter()
                .flat_map(|bank| bank.params.iter())
                .map(|param| *param as usize)
                .filter(|param| !beside.contains(param))
        };
        if params().all(|param| out.contains(&param)) {
            whole.extend(params());
        }
    }
    whole.sort_unstable();
    whole.dedup();
    whole
}

/// Whether this device's selects of a packed bank read no expert table: the
/// grouped tcgen05 GEMM.
fn packed_unread() -> bool {
    #[cfg(feature = "cuda")]
    {
        kernels_cuda::linear::gemm_sm100::supported()
    }
    #[cfg(feature = "rocm")]
    {
        false
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        true
    }
}

/// The tables of `trace` this engine streams by their rows: read only by the
/// plain embedding gather (the one whose kernel reads a row table), a bf16
/// plane of its own with no companion, out of the checkpoint.
#[must_use]
pub fn streamed_rows(trace: &poem_ir::Trace) -> Vec<usize> {
    use poem_ir::residency::{Residency, residency};
    use poem_ir::{Def, Gather, Operands, Operation};
    let classes = residency(trace);
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    'params: for (at, param) in trace.params.iter().enumerate() {
        if !matches!(classes[at], Residency::Rows(_))
            || param.dtype != poem_ir::Dtype::Bf16
            || param.shape.len() != 2
            || param.source != poem_ir::ParamSource::Checkpoint
        {
            continue;
        }
        let mut read = false;
        for node in &trace.nodes {
            inputs.clear();
            node.op.inputs(&mut inputs);
            let reads = inputs.iter().any(|v| {
                matches!(trace.values.get(v.0 as usize).map(|d| &d.def),
                    Some(Def::Weight(w)) if *w as usize == at)
            });
            if !reads {
                continue;
            }
            match &node.op {
                Operation::Gather(Gather::Embed {
                    weights: None,
                    scales: None,
                    concat: false,
                    vocab,
                    ..
                }) if u64::from(*vocab) == param.shape[0] => read = true,
                _ => continue 'params,
            }
        }
        if read {
            out.push(at);
        }
    }
    out
}

/// The rs list: a slot holds `slot.tokens` buffered rows of every recurrent
/// layer's rs planes, a layer's rows `layers[k]` bytes into it and a token
/// `per_token[k]` bytes of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsShape {
    pub slot: SlotShape,
    /// Where each rs row's run starts in a slot and its bytes a token, in
    /// the recipe's order (a recurrent layer each, then what else buffers).
    pub layers: Vec<(u64, u64)>,
}

/// One recurrent row of a working set's record: `pieces` equal pieces of
/// `atoms_per_piece` atoms (the last padded past the row's `atoms`), each
/// `offset` bytes into a slot of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedRow {
    /// The cache row.
    pub row: u32,
    pub pieces: u32,
    pub atoms_per_piece: u64,
    pub atoms: u64,
    pub atom_bytes: u64,
    pub offset: u64,
    /// The record's slots its pieces lie in, one a piece.
    pub slots: Vec<u32>,
    /// Where its ids start among a lane's staged ids ([`FixedShape::tables`]).
    pub column: u32,
}

impl FixedRow {
    /// The bytes piece `piece` holds of the row.
    #[must_use]
    pub fn piece_bytes(&self, piece: u32) -> u64 {
        let start = (u64::from(piece) * self.atoms_per_piece).min(self.atoms);
        ((start + self.atoms_per_piece).min(self.atoms) - start) * self.atom_bytes
    }
}

/// The fixed shape: a working set's record of every recurrent row, `pieces`
/// slots of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedShape {
    pub slot: SlotShape,
    /// The slots a record takes.
    pub pieces: u32,
    pub rows: Vec<FixedRow>,
    /// The id tables a fire stages a lane: each a run of the record's slots
    /// some row's pieces lie in, the rows that share a run sharing a table.
    pub tables: Vec<Vec<u32>>,
}

impl FixedShape {
    fn of(recipe: &ChunkRecipe, layout: &SlotLayout) -> Result<FixedShape> {
        let chunk_bytes = recipe.chunk_bytes;
        let SlotLayout::Fixed {
            bytes,
            pieces_per_ws,
            rows: runs,
            ..
        } = layout
        else {
            return Err(Fault::program(
                "store::layout",
                "a token slot read as the fixed slot",
            ));
        };
        let per_chunk = u32::try_from(chunk_bytes / (*bytes).max(1)).unwrap_or(u32::MAX);
        let mut tables: Vec<Vec<u32>> = Vec::new();
        let mut columns: Vec<u32> = Vec::new();
        let mut rows = Vec::new();
        for run in runs {
            let atom = recipe.atoms.get(&run.row).ok_or_else(|| {
                Fault::program(
                    "store::layout",
                    format!("the chunk recipe states no atom for {:?}", run.row),
                )
            })?;
            match run.row {
                RowId::Cache(row) => {
                    let table = match tables.iter().position(|held| *held == run.slots) {
                        Some(at) => at,
                        None => {
                            columns.push(tables.iter().map(|t| t.len() as u32).sum());
                            tables.push(run.slots.clone());
                            tables.len() - 1
                        }
                    };
                    rows.push(FixedRow {
                        row,
                        pieces: run.pieces,
                        atoms_per_piece: run.atoms_per_piece,
                        atoms: atom.extent,
                        atom_bytes: atom.bytes,
                        offset: run.offset,
                        slots: run.slots.clone(),
                        column: columns[table],
                    });
                }
                RowId::Rs(_) => {
                    return Err(Fault::program(
                        "store::layout",
                        "an rs row in the fixed record: the rs buffer is a list shape",
                    ));
                }
                RowId::Pool(_) => {}
            }
        }
        Ok(FixedShape {
            slot: SlotShape {
                id: FIXED,
                tokens: 0,
                slot_bytes: *bytes,
                per_chunk: per_chunk.max(1),
                grain: gcd(chunk_bytes, *bytes),
                window: None,
                rows: Vec::new(),
                pools: Vec::new(),
            },
            pieces: (*pieces_per_ws).max(1),
            rows,
            tables,
        })
    }

    /// Cache row `row`'s place in the record.
    #[must_use]
    pub fn row(&self, row: u32) -> Option<&FixedRow> {
        self.rows.iter().find(|held| held.row == row)
    }

    /// The ids a fire stages a lane: every table's.
    #[must_use]
    pub fn ids_a_lane(&self) -> u32 {
        self.tables.iter().map(|table| table.len() as u32).sum()
    }
}

impl Layout {
    /// The recipe's token shapes. This engine pages every list by its kv
    /// page, so a window list of another page size is refused.
    ///
    /// `streamed` are the tables streamed by their rows and `banks` the
    /// routed banks streamed an expert at a time, each with its plane's
    /// bytes; a weight slot the recipe lays with a plane outside them stays
    /// as the load seats it, no atom of it cached.
    pub fn of(
        recipe: &ChunkRecipe,
        streamed: &[usize],
        banks: &[(usize, u64, u64)],
    ) -> Result<Layout> {
        let chunk_bytes = recipe.chunk_bytes;
        let list = recipe
            .list()
            .map(|layout| SlotShape::of(layout, chunk_bytes, LIST))
            .transpose()?;
        let windows = recipe
            .windows()
            .enumerate()
            .map(|(at, layout)| SlotShape::of(layout, chunk_bytes, window_id(at)))
            .collect::<Result<Vec<_>>>()?;
        if windows.len() > MOST_WINDOWS {
            return Err(Fault::Unbound {
                what: format!(
                    "kv rows read through {} windows; a load states at most {MOST_WINDOWS} \
                     window lists",
                    windows.len()
                ),
            });
        }
        // Every list pages by the engine's one page size.
        let page = list.as_ref().or(windows.first()).map(|shape| shape.tokens);
        if let Some(window) = windows.iter().find(|shape| Some(shape.tokens) != page) {
            return Err(Fault::Unbound {
                what: format!(
                    "a window list of {} tokens a slot beside pages of {}, which this \
                     engine's kernels page alike",
                    window.tokens,
                    page.unwrap_or(0)
                ),
            });
        }
        let fixed = recipe
            .fixed()
            .map(|layout| FixedShape::of(recipe, layout))
            .transpose()?;
        let rs = recipe
            .rs_list()
            .map(|layout| -> Result<RsShape> {
                let mut slot = SlotShape::of(layout, chunk_bytes, RS)?;
                slot.rows.clear();
                Ok(RsShape {
                    slot,
                    layers: recipe
                        .rs
                        .runs
                        .iter()
                        .map(|run| (run.offset, run.per_token))
                        .collect(),
                })
            })
            .transpose()?;
        let (_, segment_chunks) = recipe.profile.segment(chunk_bytes);
        let mut rows = Vec::new();
        for &param in streamed {
            let slot = recipe.weights.iter().find(|slot| {
                slot.kind == poem_compiler::chunk::WeightKind::Rows
                    && slot.pieces == 1
                    && matches!(slot.banks.as_slice(), [bank] if bank.params == [param as u32])
            });
            // A table the recipe lays with others is served from its host
            // copy, no row of it cached.
            let Some(slot) = slot else { continue };
            let per_chunk = u32::try_from(chunk_bytes / slot.bytes.max(1)).unwrap_or(u32::MAX);
            let block_rows = u32::try_from(slot.atoms_per_slot)
                .unwrap_or(u32::MAX)
                .max(1);
            let table_rows = slot.banks[0].atoms;
            rows.push(RowsShape {
                id: ShapeId(WEIGHTS + rows.len() as u32),
                param: param as u32,
                slot_bytes: slot.bytes,
                per_chunk: per_chunk.max(1),
                block_rows,
                row_bytes: slot.atom_bytes,
                rows: table_rows,
                blocks: u32::try_from(table_rows.div_ceil(u64::from(block_rows)))
                    .unwrap_or(u32::MAX),
                floor: u32::try_from(slot.floor).unwrap_or(u32::MAX),
            });
        }
        let mut experts = Vec::new();
        for slot in &recipe.weights {
            if slot.kind != poem_compiler::chunk::WeightKind::Expert
                || slot.pieces != 1
                || slot.atoms_per_slot != 1
            {
                continue;
            }
            let Some(count) = slot.banks.first().map(|bank| bank.atoms) else {
                continue;
            };
            let mut layers = Vec::with_capacity(slot.banks.len());
            for bank in &slot.banks {
                let mut at = 0u64;
                let mut planes = Vec::with_capacity(bank.params.len());
                let mut laid = 0;
                for &param in &bank.params {
                    let Some(&(_, bytes, rows)) =
                        banks.iter().find(|(held, ..)| *held == param as usize)
                    else {
                        break;
                    };
                    if bank.atoms != count || count == 0 || !bytes.is_multiple_of(count) {
                        break;
                    }
                    // A plane with TMA rows on its boundary, as the recipe
                    // laid the slot; every plane on a 16-byte boundary of
                    // the slot, as its kernels' vector loads read it.
                    if rows > 0 && bytes > 0 {
                        at = at.next_multiple_of(poem_compiler::chunk::weights::plane_boundary(
                            bytes / count,
                            rows,
                        ));
                    }
                    if !at.is_multiple_of(PLANE_ALIGN) {
                        break;
                    }
                    laid += 1;
                    // A plane held beside the bank (no bytes): no part of the
                    // slot.
                    if bytes == 0 {
                        continue;
                    }
                    let per = bytes / count;
                    planes.push(ExpertPlane {
                        param,
                        offset: at,
                        bytes: per,
                        row: if rows > 0 && per.is_multiple_of(rows) {
                            per / rows
                        } else {
                            0
                        },
                    });
                    at += bytes / count;
                }
                if laid == bank.params.len() && at <= slot.bytes {
                    layers.push(planes);
                }
            }
            // Every layer of the shape or none: an atom's number is its
            // layer's place among them.
            if layers.len() != slot.banks.len() {
                continue;
            }
            experts.push(ExpertShape {
                id: ShapeId(WEIGHTS + (rows.len() + experts.len()) as u32),
                slot_bytes: slot.bytes,
                per_chunk: u32::try_from(chunk_bytes / slot.bytes.max(1))
                    .unwrap_or(u32::MAX)
                    .max(1),
                experts: u32::try_from(count).unwrap_or(u32::MAX),
                banks: layers,
                floor: u32::try_from(slot.floor).unwrap_or(u32::MAX),
            });
        }
        Ok(Layout {
            chunk_bytes,
            segment_chunks: u32::try_from(segment_chunks).unwrap_or(u32::MAX).max(1),
            list,
            windows,
            fixed,
            rs,
            rows,
            experts,
            adapter: None,
        })
    }

    #[must_use]
    pub fn shape(&self, kind: Kind) -> Option<&SlotShape> {
        match kind {
            Kind::List => self.list.as_ref(),
            Kind::Window(at) => self.windows.get(at as usize),
            Kind::Fixed => self.fixed.as_ref().map(|fixed| &fixed.slot),
            Kind::Rs => self.rs.as_ref().map(|rs| &rs.slot),
        }
    }

    /// The guest shape of `role`; `None` for a role that is not a guest's.
    #[must_use]
    pub fn guest(&self, role: engine::chunk::SlotRole) -> Option<GuestShape> {
        let (id, slot_bytes) = match role {
            engine::chunk::SlotRole::Instance => (INSTANCE, INSTANCE_SLOT),
            engine::chunk::SlotRole::Channel => (CHANNEL, CHANNEL_SLOT),
            engine::chunk::SlotRole::Adapter => (ADAPTER, self.adapter.as_ref()?.slot_bytes),
            _ => return None,
        };
        Some(GuestShape {
            id,
            role,
            slot_bytes,
            per_chunk: u32::try_from(self.chunk_bytes / slot_bytes).unwrap_or(u32::MAX),
        })
    }

    /// The bytes of the engine's own tables in the pool's reserved run
    /// (`Reserved::Tables`), each region whole: the row caches' tables and
    /// counters, then the expert caches', then the adapters'.
    #[must_use]
    pub fn table_bytes(&self) -> u64 {
        self.row_table_bytes()
            + self.expert_table_bytes()
            + self
                .adapter
                .as_ref()
                .map_or(0, AdapterShape::region_bytes)
    }

    /// The row caches' region: a u64 address and a u32 counter a block.
    #[must_use]
    pub fn row_table_bytes(&self) -> u64 {
        self.rows
            .iter()
            .map(|rows| {
                let blocks = u64::from(rows.blocks);
                table_span(blocks * 8) + table_span(blocks * 4)
            })
            .sum()
    }

    /// The expert caches' region: each bank an address an expert of each
    /// plane its select lists (at most three: codes, scales, a bias), and a
    /// u32 counter an expert a layer.
    #[must_use]
    pub fn expert_table_bytes(&self) -> u64 {
        self.experts
            .iter()
            .map(|shape| {
                let experts = u64::from(shape.experts);
                shape
                    .banks
                    .iter()
                    .map(|layer| {
                        layer.len() as u64 * table_span(experts * TABLE_PLANES * 8)
                            + table_span(experts * 4)
                    })
                    .sum::<u64>()
            })
            .sum()
    }

    /// The chunks the reserved run of tables takes.
    #[must_use]
    pub fn table_chunks(&self) -> u32 {
        u32::try_from(self.table_bytes().div_ceil(self.chunk_bytes.max(1))).unwrap_or(u32::MAX)
    }

    /// The chunks `slots` slots of a shape take.
    #[must_use]
    pub fn chunks_for(&self, kind: Kind, slots: u64) -> u64 {
        self.shape(kind)
            .map_or(0, |shape| slots.div_ceil(u64::from(shape.per_chunk.max(1))))
    }
}

/// The boundary each table of the reserved run starts on.
pub const TABLE_ALIGN: u64 = 256;

/// The planes an expert table's entry lists at most.
const TABLE_PLANES: u64 = 3;

/// The bytes a table of `bytes` takes in the reserved run.
#[must_use]
pub fn table_span(bytes: u64) -> u64 {
    bytes.next_multiple_of(TABLE_ALIGN)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}
