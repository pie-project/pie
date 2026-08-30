//! The §8 `Run`, in metal's encode world: one fire's dispatch state, and the
//! one function that turns a plan id into a device handle.
//!
//! Long-lived state — the weight and arena tables, the cache pools — arrives
//! pre-built and borrowed: building those tables is the shell's binding
//! business (the successor of the old `bind/`), not this layer's. Fire-lived
//! state — the input bindings, the plan payloads — is owned: a `Run` is
//! constructed per fire and dropped with it.
//!
//! Everything here answers to one rule: a [`KernelError`] is about the
//! backend, never about the plan (`model_exec::error`). A hole in a table,
//! a cache id in a tensor seat, a plan consumed before its plan op — those
//! are integrity failures of the shell or the compiler, and they panic with
//! a sentence instead of dressing up as a backend refusal.
//!
//! [`KernelError`]: model_exec::KernelError

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

/// One loader-resolved weight. Most rows are one dense handle; a quantized
/// weight is two or three device planes under one `Def::Weight` id — the form
/// this shell's one-handle rows once refused.
///
/// **THE ROW IS WHERE THE FORMAT IS SETTLED, AND IT IS SETTLED ONCE.** The
/// two four-bit formats this shell serves differ in what a scale entry means
/// (mxfp4's e8m0 byte is the whole dequantization; MLX affine's bf16 factor
/// is half of it, the other half being the bank's zero points) and in how
/// many codes share one — and a checkpoint is not uniform in either
/// (`mlx_lm` publishes a 4-bit stack whose router gate is 8-bit). So the
/// group size, the bit width and the presence of the third plane travel with
/// the planes, inside [`Bank`], and every dispatch arm downstream picks its
/// point off that one value rather than off a model-wide setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    /// One dense handle, resolved by [`Run::tensor`].
    Dense(Tensor),

    /// A split-plane quantized bank, resolved by [`Run::planes`] or asked
    /// after by [`Run::banked`] — never as one tensor.
    Planes(Bank),
}

/// Loader-resolved weights, one row per `Trace::params` entry —
/// `Def::Weight(i)` resolves to row `i`. `None` marks a param the shell has
/// not bound; resolving such a row is a binding bug and panics.
#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

/// Arena slots at the compiler's offsets, `ValueId`-indexed. Op outputs and
/// merges alike land here: the compiler aliased every merge arm onto one
/// slot and wrote that slot at the merged id's row too, so a φ resolves like
/// any op output. Rows for ids that own no arena slot (inputs, weights,
/// caches, structs) stay `None`.
#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

/// One resolved cache space — the storage pointer and nothing else (design
/// §7); its geometry rides in [`FireBindings`] as declared inputs.
#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    /// A paged kv space (`CacheRow::Kv`).
    Kv(KvPool),
    /// A recurrent state space (`CacheRow::State`).
    Recurrent(RecurrentPool),
}

/// Cache-index-indexed pools, aligned with `Trace::caches`.
#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

/// The geometry vectors one cache space declared. Only what the plan names
/// gets bound, so every seat is optional; resolving an unbound seat is a
/// binding bug and panics.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheGeometry {
    /// `RuntimeInput::Geometry { kind: Indptr }`.
    pub indptr: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: Indices }`.
    pub indices: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: SeqLens }`.
    pub seq_lens: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: LastPageLen }`.
    pub last_page_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: KvLen }`.
    pub kv_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RowValid }`.
    pub row_valid: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RequestOfToken }`.
    pub request_of_token: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WritePage }`.
    pub write_page: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WriteOffset }`.
    pub write_offset: Option<Tensor>,
}

/// The per-fire tables the sdpa shaders read beside the pool. Positions ride
/// in [`FireBindings::positions`]; these are the rest. `attention.plan_*`
/// names kv geometry instead of these, so the plan-building arms bind them
/// from here; `mask` alone is op-named too now (`attention.masked`), and
/// [`Run::tensor`] resolves `RuntimeInput::Mask` onto the same seat. This is
/// the engine side of the seam the `MENLO-SEAM` markers in
/// `kernels_metal::attn` describe.
#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    /// `i32`, one per token: the owning request.
    pub request_of_token: Tensor,

    /// `u8` packed mask planes, one row per request — the plan builders'
    /// table and `RuntimeInput::Mask`'s resolution, one seat wearing both
    /// names.
    pub mask: Tensor,

    /// `u8`, one per request: whether its mask row is live.
    pub mask_enabled: Tensor,

    /// Elements from one request's mask row to the next.
    pub mask_stride: u32,
}

/// What the engine binds each fire, owned by the [`Run`] for its lifetime.
///
/// `tokens`, `positions`, and `geometry` are the op-visible inputs —
/// `RuntimeInput` routes onto them in [`Run::tensor`]. The `tables` are
/// ambient: the plan builders read them and no op names them.
///
/// **TWO FIELDS THE CUDA SIBLING'S TWIN CARRIED ARE GONE, AND FOR THE
/// REASONS THAT SIBLING RECORDED.** `facts` — the fire-wide fact word — was
/// deleted from the CUDA shell in palo build log 8 with zero readers, and
/// deleted here for the same reason: which classes run a node is
/// `Region::mask` resolved per region against the window table, which is
/// exactly what a fire-wide word cannot say (design §0's collapse). And the
/// shared `indptr` is gone because a ragged view is assembled from the
/// WINDOW's own rebased boundaries ([`Run::qo_indptr`]) — a fire-wide vector
/// handed to a windowed launch sends every entry past the end of the
/// rectangle it was given, by exactly the rows the classes before it hold.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per
    /// token — also the plan builders' causal-bound table.
    pub positions: Tensor,

    // adapter — lane J's one field here, kept contiguous.
    /// `RuntimeInput::AdapterRoutes`: `i32`, one adapter id per token row,
    /// `-1` for a row whose lane routes nowhere.
    ///
    /// **HERE AND NOT IN [`FireTables`], BECAUSE AN OP NAMES IT.** Everything
    /// on `tables` is a seat no `Operands` impl mentions; this one is a
    /// declared `RuntimeInput` that `linear.lora_correct` lists among its
    /// inputs, so it stands beside `tokens` and `positions`.
    ///
    /// `None` for a fire no lane carried an adapter into, and that absence is
    /// load-bearing: nothing staged, no seat bound, and the axis costs the
    /// fire zero bytes and zero launches, because its window is empty and the
    /// walk skips it.
    pub adapter_routes: Option<Tensor>,

    /// Per cache space, aligned with `Trace::caches`:
    /// `RuntimeInput::Geometry { space, kind }` routes to that space.
    pub geometry: Vec<CacheGeometry>,

    /// The fire tables the attention plan builders consume.
    pub tables: FireTables,

    /// **THE OBSERVABILITY SEAT** (`.wiki/alto/attn-score.md` §4), `None` for
    /// a load whose plan declares no `attn.scores` export and for every fire
    /// of a load whose lanes all asked for nothing.
    ///
    /// A `MENLO-SEAM` in the strict sense — no `Operands` impl mentions the
    /// slab, and no `Operands` impl should: the score write is not a value the
    /// graph computes for another node, it is an OBSERVATION the graph makes
    /// on its way past. What the IR names is the capture arm, and the capture
    /// arm is `attention.prefill_lse`, which the plan already carried.
    ///
    /// It stands here rather than on [`FireTables`] for `adapter_routes`'s
    /// reason one field up: the seat carries a list (which value is which
    /// plane) and `FireTables` is the `Copy` half.
    pub scores: Option<crate::scores::ScoreSeat>,
}

/// One built plan payload. An enum over the three kinds this plane can be
/// asked to build, not `Box<dyn Any>`: the IR's `StructKind` is closed, and
/// this crate names every payload type at compile time — erasure would buy
/// no generality, only a silent-downcast failure mode. Here a wrong kind is
/// a named panic.
#[derive(Clone, Copy, Debug)]
pub enum StructSlot {
    /// `StructKind::AttnDecodePlan`.
    Decode(DecodePlan),

    /// `StructKind::AttnPrefillPlan`.
    Prefill(PrefillPlan),

    /// `StructKind::MlaPlan` — declared for shape; the metal builder refuses
    /// before one exists, so this variant is never stored.
    Mla(MlaPlan),
}

/// One fire's dispatch state: the encode sink, the resolution tables, the
/// fire bindings, and the plan payloads this fire builds. The shell
/// constructs one per fire and drives the substrate's walk
/// (`model_exec::fire::walk`) over it — prepare phase first, so every plan
/// payload exists before its consumers encode.
pub struct Run<'c> {
    /// The encode sink — the real shell behind `dyn Encode`. Everything this
    /// crate does to the device goes through it; nothing here names Metal.
    ctx: &'c Ctx<'c>,

    /// The handle table every carve is minted into and every argument is
    /// resolved through. A windowed cut IS a new row here — Metal binds a
    /// buffer and an offset, so there is no address to add a row stride to.
    handles: &'c Handles,

    /// The routing: `Trace::values`, read by [`Run::tensor`] to send each id
    /// to its table.
    values: &'c [ValueDecl],

    /// `Trace::nodes`, read by ONE caller and named here rather than passed
    /// to it: `crate::dispatch::copy` walks a copied region's node range to
    /// find which rectangles it moves, and the walk's `Serve` signature hands
    /// it a `&Region` and nothing else. Same borrow, same lifetime and same
    /// reason as [`values`](Run::values) beside it.
    nodes: &'c [Node],

    /// `Def::Weight` rows, loader-resolved.
    weights: &'c WeightTable,

    /// `Def::Op` / `Def::Merge` rows, carved at the compiler's offsets.
    arena: &'c SlotTable,

    /// `Def::Cache` rows — pool pointers, resolved through [`Run::pool`] and
    /// [`Run::recurrent`], never through [`Run::tensor`].
    caches: &'c CacheTable,

    /// Plan payloads: filled by the plan-building arms in the prepare phase,
    /// read by the consuming arms afterwards.
    ///
    /// **KEYED BY `(RUN, VALUE)`, NOT BY VALUE**, and the extra key is P4's
    /// split (design §3). A schedule is carved for ONE window, so a region the
    /// layout could not seat carves one per interval of it — all built in the
    /// prepare phase, all read in the capture phase. One slot per value would
    /// let run 1's builder overwrite run 0's, and run 0's encode would then
    /// read a schedule describing run 1's requests: not a fault, just wrong
    /// logits for the lanes in the first interval.
    ///
    /// Flat rather than nested, at `run * values + value`: a plan has
    /// thousands of values and a `Vec` per value would be thousands of
    /// allocations per fire, where this is one. The width is
    /// [`Windows::max_runs`] — `1` for every artifact P4 seated whole.
    structs: Vec<Option<StructSlot>>,
    /// How many values one run's slice of [`structs`](Run::structs) holds.
    values_wide: usize,

    /// This fire's bindings.
    fire: FireBindings,

    /// Every region's windows, resolved once per fire from the composition's
    /// class table. Indexed by [`place`](Run::place).
    windows: &'c Windows,

    /// Which region the walk is inside and which run of its window, written
    /// by [`Cursor`](crate::window::Cursor) on `region_begin` and on `run` —
    /// before that region's nodes are dispatched, and before each encode of
    /// them. **THIS IS THE WHOLE MIXED-FIRE MECHANISM**: the walk's `Dispatch`
    /// signature is fixed and carries no region, so the sink and the resolver
    /// share one cell instead.
    place: &'c At,

    // fallback.copy — the one field a copied region adds, seated by the
    // gather and read until the scatter.
    /// **The copied region the walk is inside**, or the default no cursor
    /// ever names.
    ///
    /// `model_exec::fire::walk` brackets a copied region's nodes with
    /// `Serve::gather` and `Serve::scatter`, and the gather is what decides
    /// where in the scratch role each of the region's rectangles was laid
    /// down. Every operand resolved between the two brackets has to answer
    /// THAT layout, so the plan is seated here on the way past and read back
    /// by [`Run::compacted`]. It carries the region index it was built for,
    /// so a stale plan is a panic with a sentence rather than a silent read
    /// of another region's offsets.
    copy: CopyPlan,

    // scratch — the load-time working plane and the two tables that ride
    // with it, kept contiguous.
    /// The shell's scratch reservation (`crate::scratch`): the working
    /// rectangles a dispatch arm needs and no op names, plus the arena's slot
    /// capacities and the routers' expert counts.
    ///
    /// **BORROWED, NOT OWNED, AND MINTED PER FIRE.** The reservation is one
    /// allocation made at load and never moved; what a fire has is a handle
    /// row into it, minted on the way past like every arena row and dropped
    /// by the same `Handles::rewind`. So this field is the same kind of thing
    /// [`weights`](Run::weights) is — a table the shell built once — and the
    /// accessors below are where a rectangle of it becomes a `Tensor`.
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

    /// The window of the region the walk is inside, cut at the run it is on.
    pub(crate) fn window(&self) -> &'c Window {
        self.windows.at(self.place.region.get(), self.place.run.get())
    }

    /// Where this run's payload for `id` sits in [`structs`](Run::structs).
    ///
    /// The run comes off the same cell the window does, so a schedule is
    /// stored and read at the same key by construction — a builder cannot
    /// carve for one interval and an encode read another.
    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    /// This window's own qo boundaries, rebased to start at zero.
    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    /// The same vector, host-side — what an entry that needs to READ a
    /// boundary (rather than bind it) takes.
    ///
    /// Unread today: every metal entry that is boundary-aware takes the
    /// pair (`RaggedTensor`) and walks it on the device. Kept beside
    /// [`Run::multi_token`] and [`Run::total_tokens`], which are the two
    /// questions a host-side arm-picking builder would ask of it, because
    /// the answer is one line and re-deriving it in a dispatch arm is how
    /// two readings of one window come to disagree.
    #[allow(dead_code)]
    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    /// How many token rows this window covers.
    #[allow(dead_code)]
    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span.rows
    }

    /// Whether any lane of this window carries more than one row — the
    /// question a chunked linear-attention entry asks to choose its arm.
    #[allow(dead_code)]
    pub(crate) fn multi_token(&self) -> bool {
        self.qo_indptr_host()
            .windows(2)
            .any(|pair| pair[1] - pair[0] > 1)
    }

    /// A raw ambient table, cut to this window's ROWS.
    ///
    /// The plan builders' tables (`positions`, `request_of_token`, the mask
    /// planes) are staged once per fire at absolute fire rows, and the
    /// shaders index them by the LOCAL row of the launch — `req_of_token[row]`
    /// where `row` runs `0..q.rows`. So a windowed launch needs them cut the
    /// way its `q` is. They are not `ValueId`s, so [`Run::tensor`]'s
    /// declaration-driven cut cannot reach them and this is the same
    /// arithmetic, stated for a bare handle.
    ///
    /// What stays ABSOLUTE is what the shaders index by lane through a
    /// value the table itself holds: `request_of_token`'s entries are lane
    /// ids into the fire-wide `page_indptr`, and cutting the vector does not
    /// renumber its contents — which is exactly the property that makes one
    /// page table serve every window.
    ///
    /// **AND A GATHERED WINDOW'S ROWS ARE NOT A SLICE, SO TWO OF THESE
    /// TABLES ARE ANSWERED BY NAME.** A `Fallback::Copy` window covers rows
    /// the fire keeps in several intervals, laid down contiguously by the
    /// gather (`crate::window::Gathered`); the shaders still index them by the
    /// launch's own row, so `positions` and `request_of_token` have to be
    /// PERMUTED and not cut. They are `i32`, which
    /// `kernels_metal::layout::gather_rows` does not stamp, so the permutation
    /// is done on the host at `Windows::of` and staged
    /// ([`Gathered::positions_host`](crate::window::Gathered::positions_host))
    /// — and this is where the twin is handed back.
    ///
    /// **WHICH TABLE IS ASKED BY THE HANDLE, AND THAT IS THE HONEST KEY.**
    /// This method takes a bare `Tensor` because the plan builders' tables are
    /// not `ValueId`s, so there is no declaration to route on; what identifies
    /// the vector is the row [`crate::inputs::Inputs::write`] minted for it,
    /// one per table per fire. Answering here rather than at each call site is
    /// deliberate: a gather is a permutation, and a call site that forgot to
    /// ask for the twin would read the first `n` fire rows in fire order —
    /// plausible numbers, wrong requests, no fault.
    ///
    /// The mask plane and its enable column are NOT permuted, and
    /// [`Copies::enabled`](crate::window::Copies::enabled) is why: a fire any
    /// lane masked never copies, so the enable column is all zeros and any
    /// `rows` of it are the same `rows` of zeros.
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

    /// Which region of the template the walk is inside — the cursor's own
    /// index, read by `crate::dispatch::copy` so that a copy plan built for
    /// one region cannot be read inside another.
    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }

    /// `Trace::nodes`, for the one caller that walks a region's node range.
    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    /// `Trace::values`, for the same caller: what a node's operand ids are
    /// declared as.
    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    /// One value's FIRE-WIDE rectangle, uncut — what a copy plan compacts
    /// FROM, and what a scatter puts back.
    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// Where a handle actually points: which reservation, and how far into
    /// it.
    ///
    /// **THIS IS THE COPY PLAN'S KEY, AND ON THIS PLANE IT CANNOT BE THE
    /// HANDLE.** The CUDA sibling keys its plan by `Tensor::ptr`, an address,
    /// which two values the carve aliased onto one column answer identically —
    /// so an in-place op reads and writes one compacted rectangle. Here
    /// `crate::arena::carve` mints a row per VALUE, following the arena root
    /// for the offset but never for the row, so two aliased values answer two
    /// different `u32`s at one offset. Keying by the handle would give an
    /// in-place op two staging rectangles and stop it being in place; keying
    /// by the resolved `(reservation, offset)` is the same fact the address
    /// was.
    ///
    /// `None` for [`NIL`](crate::device::handles::NIL) and for a row this
    /// fire did not mint — neither of which a plan operand can be, and both
    /// of which are answered rather than panicked because a copy plan that
    /// cannot key a rectangle simply does not move it.
    pub(crate) fn address(&self, handle: u32) -> Option<(u64, u64)> {
        let row = self.handles.get(handle)?;
        Some((crate::device::alloc::slab_id(row.slab()), row.offset()))
    }

    /// Seat the plan a copied region's gather just built — read back by
    /// [`Run::compacted`] for every operand until the region's scatter.
    pub(crate) fn seat_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    /// The plan the current region's gather seated, for the scatter that
    /// closes the bracket.
    pub(crate) fn staged_copy(&self) -> &CopyPlan {
        &self.copy
    }

    /// One rectangle of the copy role, minted for this fire.
    ///
    /// A mint that fails is an INTEGRITY failure and not a refusal, for
    /// [`Run::slice`]'s reason; a role that does not HOLD the span is the
    /// caller's refusal to make, so the two are separated: `None` is
    /// "reserved smaller than this", and the panic is "the handle table is
    /// full".
    pub(crate) fn copy_room(&self, offset: u64, bytes: u64) -> Option<u32> {
        Some(
            self.scratch
                .copy(self.handles, offset, bytes)?
                .unwrap_or_else(|fault| {
                    panic!("the copy rectangle this load reserved does not mint: {fault}")
                }),
        )
    }

    /// The encode sink, for the arms.
    pub(crate) fn ctx(&self) -> &'c Ctx<'c> {
        self.ctx
    }

    /// The fire bindings, for the plan-building arms' seam.
    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    /// One rectangle, sliced to `keep` rows starting at `skip`.
    ///
    /// The one place a windowed cut becomes a handle. A CUDA shell answers
    /// this with `ptr + skip * stride` and no state; here the row is minted
    /// into [`Handles`], which is also where the bounds check lives — a cut
    /// past the end of the reservation is caught before a shader
    /// dereferences it.
    ///
    /// A cut that fails is an INTEGRITY failure, not a refusal: the offsets
    /// come from the compiler's carve and the composition's window table,
    /// and a disagreement between those two is a bug in this crate or in the
    /// bake. It panics with a sentence, per the file's rule.
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
            .cut(handle.buf, u64::from(skip) * stride, u64::from(rows) * stride)
            .unwrap_or_else(|fault| {
                panic!(
                    "the window's cut of handle {} at row {skip} for {rows} rows does \
                     not land: {fault}",
                    handle.buf
                )
            });
        Tensor::new(cut, rows, handle.width, handle.dtype)
    }

    /// One value's rectangle, cut to the window of the node asking for it.
    ///
    /// **EVERY ROW-SHAPED TABLE IN THIS SHELL IS INDEXED BY ABSOLUTE FIRE
    /// ROW** — the arena carve gives a `Dim::Tokens` value one column at the
    /// fire's row count, the geometry vectors one entry per fire lane — so a
    /// window is a slice, and which slice is read off the value's own leading
    /// `Dim`. A `Dim::Const` column (a weight plane, a bias) is not
    /// fire-aligned and is handed over whole.
    ///
    /// `GeomKind::Indices` is the one declared shape that is not what it
    /// says: the IR spells the flat page-id list `Dim::Lanes` because it has
    /// no page symbol, and its entries are pages rather than lanes. Slicing
    /// it by a lane offset would hand a windowed consumer somebody else's
    /// pages, so it is excluded — and its bounds vector stays absolute,
    /// which is what makes a sliced `Indptr` still address the whole list.
    ///
    /// `RuntimeInput::Mask` is the second of those, for the same reason: the
    /// IR spells the mask slab `Dim::Tokens` and its entries are (query,
    /// key) BITS, so a row offset is not a byte offset. This plane binds the
    /// mask through [`FireTables`] rather than through a declared input and
    /// cuts it with [`Run::cut_rows`], which knows the stride.
    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        // **THE OTHER ANSWER, ASKED FIRST** (design §3). A gathered window's
        // rows do not lie in the arena at all — they were compacted into the
        // copy role of `crate::scratch` before the region's first node — so a
        // slice of the fire-wide column is not a narrower reading of the same
        // bytes, it is the wrong bytes. [`Run::compacted`] is the whole of
        // what a copy changes about resolution.
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
        let window = self.window().span;
        let (skip, keep) = match shape.first() {
            Some(Dim::Tokens) => (window.row_offset, window.rows),
            Some(Dim::TokensTimes(k)) => (window.row_offset * k, window.rows * k),
            Some(Dim::Lanes) => (window.lane_offset, window.lanes),
            Some(Dim::LanesPlus(k)) => (window.lane_offset, window.lanes + k),
            Some(Dim::Const(_)) | None => return handle,
            // THE SECOND ROW AXIS, WHICH THIS PLANE HAS NO WINDOW FOR AND NO
            // KERNEL BEHIND. `window.span` is the token rectangle's pair, and
            // patches do not break where tokens do (multimodal §5.1), so
            // cutting a patch column at a token row offset would hand a launch
            // the wrong rows — and handing it over whole is only right when
            // every class is present, which is a composition and not an
            // invariant. Refused by name rather than left partial, and the
            // same sentence stands one method down where the axis's own
            // `RuntimeInput` resolves: this mirror serves no tower op, so a
            // patch-carrying plan is a plan this plane cannot run.
            Some(Dim::Patches | Dim::Images | Dim::ImagesPlus(_)) => panic!(
                "value {at} is a patch-axis rectangle and this plane binds no patch \
                 window; the metal mirror carries no tower kernel, and no model text \
                 states a patch row before wave M3"
            ),
        };
        self.slice(handle, skip, keep)
    }

    /// [`Run::cut`]'s other half: what a `Fallback::Copy` resolves to.
    ///
    /// **THREE ANSWERS, AND THEY ARE THE THREE `window::copyable` ADMITS.**
    /// A row-shaped value is the staging rectangle the region's gather laid
    /// it in; the four kv geometry vectors are the twins re-cut for the
    /// gathered lanes
    /// ([`GatheredSpace`](crate::window::GatheredSpace)); everything
    /// window-free is handed over whole, exactly as a split hands it over.
    /// Nothing else can arrive — `Windows::of` declines to gather a region
    /// naming anything else, and the region then takes the split, which is
    /// always correct.
    ///
    /// The POOL is not among them, and that is this plane's own line. A
    /// gathered lane's page tables are re-cut here for the ops that NAME a
    /// geometry vector and index it by the launch's own lane; the sdpa
    /// entries do not — they read `kv_page_indptr[req_of_token[row]]` with
    /// absolute lane ids, and the gather permutes `request_of_token` without
    /// renumbering it, so the fire-wide pool is still the right table
    /// ([`Run::pool`] argues the same thing for a split window).
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
                // `window::copyable` admits no other kind into a copied
                // region, so this is the arm nothing reaches — and it
                // answers the fire-wide vector rather than a wrong window,
                // which is the conservative direction.
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

    /// The crate's heart: one plan id in, one device handle out, routed on
    /// the id's `Def`. Every dispatch arm resolves through here, so
    /// provenance handling exists exactly once.
    ///
    /// Cache ids never resolve to a tensor — a cache is a pool pointer and
    /// resolves through [`Run::pool`] or [`Run::recurrent`]; a cache id
    /// arriving here is a dispatch-arm bug, answered with a panic. So is a
    /// split-plane weight: two planes resolve through [`Run::planes`].
    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    /// The same resolution, uncut — the fire-wide rectangle a value names.
    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,
            // The op-named mask (`attention.masked`) resolves onto the fire
            // table the plan builders already carry: this plane binds one
            // mask per fire — every sdpa launch reads its seats — so a
            // second seat would only exist to drift, and the space
            // collapses onto it.
            Def::Input(RuntimeInput::Mask { space: _ }) => self.fire.tables.mask,
            // THE ADAPTER AXIS'S ONE RUNTIME INPUT (design §8). Bound only
            // when a lane of this fire carried an adapter — a fire none did
            // stages nothing, and nothing can reach this arm either, because
            // the correction's window is empty and the walk skips a zero-row
            // region before it dispatches a node. So the panic is not a hole:
            // it is the same "unbound seat" statement the mask makes, and
            // reaching it would mean a word said `has_adapter` where the
            // submission said no adapter — which `Fault::AdapterWord` refuses
            // before anything launches.
            Def::Input(RuntimeInput::AdapterRoutes) => {
                self.fire.adapter_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's adapter ids, which no lane of it \
                         carried"
                    )
                })
            }
            // THE SECOND ROW AXIS IS NOT SERVED HERE EITHER, AND FOR THE
            // ADAPTER AXIS'S REASON. This plane binds no patch seat and
            // carries no tower kernel, so nothing can reach this id: no model
            // text states a patch row yet (wave M3), and a plan that stated
            // one is refused at the bake against budgets that size no patch
            // ceiling. Resolving a bare zero here would point every tower
            // launch at whatever the arena's base holds.
            Def::Input(
                RuntimeInput::Patches
                | RuntimeInput::PatchSegments
                | RuntimeInput::PatchRoutes
                | RuntimeInput::PatchPositions
                | RuntimeInput::PatchEmbedRows
                | RuntimeInput::PatchEmbedWeights,
            ) => panic!(
                "value {at} reads the fire's patch rows, which this plane binds no seat                  for; the metal mirror serves no vision tower"
            ),
            // **THE TRUNK'S TRIPLE-WIDE POSITION STREAM** (multimodal §6.3),
            // on the FIRST axis and refused for the same reason the four
            // above it are: this plane stages one scalar per token row and
            // reserves no triple, so resolving a bare zero here would point
            // the rotation at whatever the arena's base holds. Its consumer
            // — `elementwise.rope_mrope` under
            // [`MropeForm::Blocked`](model_ir::MropeForm::Blocked) — is
            // already a named refusal one file over, and the interleaved arm
            // that does forward can only be reached by a plan that declares
            // this input.
            Def::Input(RuntimeInput::MropePositions) => panic!(
                "value {at} reads the fire's (t, h, w) token positions, and this plane                  stages one scalar per row"
            ),
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
                    Some(WeightRow::Dense(handle)) => handle,
                    Some(WeightRow::Planes(_)) => panic!(
                        "value {at} is weight {row}, a split-plane bank; it resolves \
                         through `Run::planes`, never as one dense handle"
                    ),
                    None => panic!("value {at} is weight {row}, which the shell has not bound"),
                }
            }
            // A φ resolves like the op output it merges: the compiler
            // aliased every arm onto one arena slot, written at this id's
            // row — so `Merge` is the same read as `Op`.
            Def::Op(_) | Def::Merge(_) => self
                .arena
                .0
                .get(at)
                .copied()
                .flatten()
                .unwrap_or_else(|| {
                    panic!("value {at} has no arena slot, which the compiler should have cut")
                }),
            Def::Cache(_) => panic!(
                "value {at} is a cache space; it resolves to a pool through `Run::pool`, \
                 never to a tensor"
            ),
        }
    }

    /// A fire-aligned value viewed through THIS WINDOW's boundaries. The
    /// indptr is ambient (design §5): no op names it, and this pairing is
    /// where it re-enters.
    ///
    /// The boundaries are the window's own, rebased: `data` already points
    /// at the window's first row, so a fire-wide vector would send every
    /// ragged entry past the end of the rectangle it was handed, by exactly
    /// the number of rows the classes before it occupy.
    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    /// The planes of a split-plane bank — the resolution
    /// `linear.moe_matmul_select_bias` needs where [`Run::tensor`] would have
    /// to lie with one handle.
    ///
    /// For the ops whose IR variant names a bank UNCONDITIONALLY. An arm that
    /// serves both forms of weight asks [`Run::banked`] instead, because for
    /// it a dense row is a selection and not a bug.
    pub(crate) fn planes(&self, id: ValueId) -> Bank {
        self.banked(id).unwrap_or_else(|| {
            panic!(
                "value {} is bound as one dense handle, and this op reads a split-plane \
                 bank",
                id.0
            )
        })
    }

    /// The bank behind a weight id, or `None` when the row is one dense
    /// handle — the question the arms that serve BOTH forms ask.
    ///
    /// `None` is an answer and not a refusal: `linear.matmul` and
    /// `layout.embed` name a weight, not a format, and which point they fire
    /// is exactly this. An unbound row is still a binding bug and still
    /// panics, because a weight nothing seated is not a dense weight.
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

    // scratch — the four accessors the working plane is read through, kept
    // contiguous with the field they resolve against.

    /// How many experts the router that wrote `routes` declared.
    ///
    /// **THE ROUTER OP NAMES IT AND NO OPERAND OF THE SELECT OPS DOES.**
    /// `moe::tile_rows` prices its tile off rows per expert, and
    /// `MoeMatmulSelect*` states `x`, `bank`, `routes` and `y` — none of
    /// which carries a count. What does carry it is the `MoeTopk*` node that
    /// wrote this very `routes` vector, so the fact travels the edge the plan
    /// already drew: resolved once at load into a `ValueId`-indexed table
    /// (`crate::scratch`), read here per node. `0` for a routing vector no
    /// router in this artifact wrote, which is a plan the sorted arm declines
    /// rather than guesses at.
    pub(crate) fn experts(&self, routes: ValueId) -> u32 {
        self.scratch.experts(routes)
    }

    /// The sorted arm's working rectangles, minted into this fire.
    ///
    /// `None` when the load reserved none — no mixture, or a mixture whose
    /// expert count no router stated — and the arm answering `None` takes the
    /// matvec, which needs no plane.
    ///
    /// A mint that fails is an INTEGRITY failure and not a refusal, for
    /// [`Run::slice`]'s reason: the rectangles were sized against the budget's
    /// ceiling at load, so a span that does not land is this crate
    /// disagreeing with its own reservation.
    pub(crate) fn routed_scratch(&self) -> Option<RoutedScratch> {
        Some(
            self.scratch
                .routed(self.handles)?
                .unwrap_or_else(|fault| {
                    panic!("the routed scratch this load reserved does not mint: {fault}")
                }),
        )
    }

    /// How many rows the arena slot behind `id` can hold — its whole
    /// reservation at the budget's ceiling, not this fire's extent.
    ///
    /// **READ AT THE DENSE QUANTIZED ARMS, AND THE READER IS NAMED.**
    /// `kernels_metal::linear::quant::mb_rows` takes exactly this as its
    /// `capacity`: the rows a launch may write into before it runs into the
    /// next value's slot, which is what makes padding a fire up to its row
    /// rung free of consequence. `dispatch::linear` hands it the MINIMUM over
    /// the two slots a padded launch touches — the activation it reads and
    /// the result it writes — because a rung either rectangle cannot hold is
    /// a rung neither takes.
    pub(crate) fn capacity(&self, id: ValueId) -> u32 {
        self.scratch.capacity(id)
    }

    /// The FP16 staging plane at `rows x contraction`, and the split-K
    /// partials plane at `split * rows x width`.
    ///
    /// **BOTH ARE SEATED AT `quant::Scratch`, AND AS MINTS RATHER THAN AS
    /// RECTANGLES.** `quant::precast_stage`/`quant::precast_point` and
    /// `quant::splitk_point`/`quant::splitk_reduce_point` are the consumers,
    /// and the shape each one wants is `quant::mb_rows`' and
    /// `quant::split_k`' answer — decided inside `quant::act_x_wt`, several
    /// guards past the call. So `dispatch::linear` hands the entry a closure
    /// over each of these and the entry asks with the numbers it selected;
    /// `None` is this shell saying the load-time reservation does not hold
    /// that shape, and the ladder answers it by taking the rung that needs no
    /// plane.
    ///
    /// The reservation costs nothing besides: the three roles alias, so on
    /// any artifact with a mixture these two are inside the routed plane's
    /// bytes — which is also why a chain may be inside ONE of them and never
    /// both.
    pub(crate) fn precast(&self, rows: u32, contraction: u32) -> Option<Tensor> {
        Some(
            self.scratch
                .precast(self.handles, rows, contraction)?
                .unwrap_or_else(|fault| {
                    panic!("the precast plane this load reserved does not mint: {fault}")
                }),
        )
    }

    /// See [`Run::precast`].
    pub(crate) fn partials(&self, split: u32, rows: u32, width: u32) -> Option<Tensor> {
        Some(
            self.scratch
                .partials(self.handles, split, rows, width)?
                .unwrap_or_else(|fault| {
                    panic!("the partials plane this load reserved does not mint: {fault}")
                }),
        )
    }

    /// The `StructKind` a plan op's output value declares — how the
    /// plan-building arms check the trace against what this plane can build:
    /// the trace wrote the choice into `Trace::values`, the arm only follows
    /// it.
    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The paged kv pool a cache id names.
    ///
    /// **NOTHING IS CUT HERE, AND THAT IS A STATEMENT ABOUT THIS PLANE'S
    /// ABI.** The CUDA sibling slices its pool's `page_indptr`,
    /// `last_page_lens` and `row_valid` to the window, because its schedules
    /// number requests from the launch's own zero. The metal sdpa entries
    /// read the page table through `kv_page_indptr[req_of_token[row]]`, and
    /// `request_of_token` is staged with ABSOLUTE lane ids — so the table
    /// stays fire-wide and the two agree. Slicing it here would send every
    /// windowed launch to somebody else's pages. (`page_indices` is absolute
    /// on both planes, for the reason `Run::cut` gives.)
    ///
    /// **AND A GATHERED WINDOW CHANGES NOTHING HERE EITHER**, which is the
    /// one place this plane's copy is simpler than the CUDA one. There the
    /// pool's lane tables are SLICED per window, so a copy has to re-cut them
    /// over the gathered lanes; here they are never sliced, and the vector
    /// that does the addressing — `request_of_token` — is PERMUTED by the
    /// gather with its absolute lane ids intact
    /// (`crate::window::Gathered::request_of_token_host`). So the fire-wide
    /// page tables answer a gathered launch exactly as they answer every
    /// other one. The re-cut `GatheredSpace` twins exist for the other
    /// reading: an op that NAMES a geometry vector and indexes it by the
    /// launch's own lane, which `Run::compacted` answers.
    pub(crate) fn pool(&self, id: ValueId) -> &KvPool {
        match self.cache(id) {
            CachePool::Kv(pool) => pool,
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    /// The recurrent state pool a cache id names, with its slot map cut to
    /// the asking node's window.
    ///
    /// A recurrent bank is addressed by SLOT, and every metal ssm shader
    /// reads its slot out of `slots[r]` where `r` counts from the LAUNCH's
    /// own zero — so a windowed scan gets the window's rows of that vector
    /// and nothing else. The banks themselves are the model's state, whole.
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

    /// Store a plan payload a prepare-phase arm just built.
    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        self.structs[at] = Some(built);
    }

    /// The decode plan a consuming arm names.
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

    /// The prefill plan a consuming arm names.
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
