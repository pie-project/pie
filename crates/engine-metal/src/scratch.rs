//! The shell's scratch plane: the working rectangles a dispatch arm needs
//! and no op names.
//!
//! **THE ARENA ANSWERS ONLY WHAT THE COMPILER CARVED.** `model_compiler::
//! arena` places one rectangle per plan VALUE, and [`Run::tensor`] hands a
//! dispatch arm exactly the rectangles its op named. Three measured wins want
//! something no op names and no value is: a staging plane the device writes
//! and reads inside one dispatch chain, dead before the chain ends. This
//! module is where that room comes from.
//!
//! ```text
//! precast    rows x K halves      the FP16 pre-cast GEMM stages its
//!                                 activation here once per projection
//! routed     the sorted stack     the MoE sort-batch's permutation, its
//!                                 inverse, the per-tile expert, and the
//!                                 gathered x/y in expert-major order
//! copy       the dense rectangle  a `Fallback::Copy` region's operands,
//!                                 gathered contiguous for one encode and
//!                                 scattered back after it
//! pool       the pool's cells     the dsv4 compressor's rolling state, which
//!                                 `attention.pool_gather` reads beside the
//!                                 cache and no op names
//! ple        one u64 row per      qwen4's PLE hash constants, written once
//!            hashing              at load because this plane's `ArgValue`
//!                                 has no by-value blob seat
//! ```
//!
//! (`index`, the NSA selection's score slab, is the sixth and is documented at
//! its field rather than here, because it aliases and the four above do not
//! all do the same thing about that.)
//!
//! # Reserved at load, at the budget's ceiling — article 7
//!
//! [`Inputs`](crate::inputs) is the precedent and the argument is the same
//! one, minus the half that does not apply. A fire path allocates nothing, so
//! the plane is one `newBufferWithLength:` at load sized against what the
//! COMPILED artifact can ask for at `max_tokens x max_lanes`; a fire past it
//! would be `Fault::Ceiling`, and there is no fire past it because every
//! rectangle here is derived from the same carve the arena is.
//!
//! What does NOT carry over from `inputs` is the second copy. That plane is
//! duplicated per in-flight arm because the HOST writes it — a `memcpy` into
//! shared storage lands in the bytes a committed command buffer is already
//! reading. **No FIRE PATH here is ever touched by the host** (the `ple`
//! role is written once, inside [`Scratch::reserve`], before a command buffer
//! exists — see its field). Every other byte is
//! written by a shader and read by a shader later in the same command
//! buffer, which puts this plane in the same class as the arena and the
//! pools: ONE copy, resting on the property `serve`'s header states and gates
//! by measurement — command buffers committed to one `MTLCommandQueue`
//! execute in commit order and do not overlap. Two arms cannot be inside this
//! plane at once for the same reason two arms cannot be inside the arena at
//! once, and if that property were false the arena would be wrong first.
//!
//! # THREE roles alias and the FOURTH is summed, and both are one argument
//!
//! One reservation sized at the largest single role, not the sum. The
//! reference driver calls it scratch coloring; here it is three sentences,
//! because the lifetimes are short enough to check by hand:
//!
//!   * **No two roles are live in one dispatch chain.** A dense projection
//!     against a bank is not a routed matmul, which is a different op. So a
//!     role's bytes are dead before the next role's first dispatch is
//!     encoded.
//!
//!     **THIS BULLET USED TO HAVE A THIRD ROLE TO ARGUE ABOUT.** It read the
//!     pre-cast arm against the split-K arm and said the two cannot compose;
//!     the split arm is gone (`linear::quant::act_x_wt`, for having different
//!     bits from the rest of its family) and so is the plane it wrote, so
//!     there are two aliased roles here and the question does not arise.
//!   * **Within a role, order is the encoder's.** A compute pass opened by
//!     `Context::frame` is `MTLDispatchTypeSerial`: every dispatch observes
//!     the writes of every dispatch before it, which is what makes
//!     `cast_qmm_input -> qmm_precast` and `route_sort -> gather -> gemm ->
//!     scatter` mean what they read as. The indirect plane is the concurrent
//!     one, and `crate::icb` answers it by putting `setBarrier()` on every
//!     slot — measured, not assumed.
//!   * **Nothing reads a role's bytes after its chain.** Every field is
//!     working storage by construction; `moe::RoutedScratch`'s own doc says
//!     it — "NOTHING HERE IS AN OUTPUT". A stale byte from the previous role
//!     is never read, because each role's first dispatch writes every byte it
//!     later reads.
//!
//! So the footprint of those two is `max(precast, routed)` and the dense role
//! is free on any artifact whose mixture is bigger than it is, which is every
//! artifact that has one.
//!
//! **AND THE COPY ROLE FAILS THE FIRST SENTENCE, SO IT IS ADDED RATHER THAN
//! UNIONED.** A `Fallback::Copy` brackets a whole REGION
//! (`model_exec::fire::walk`): the gather is encoded before the region's first
//! node and the scatter after its last, and every byte the gather laid down
//! has to still be there when the scatter reads it. So the copy rectangle is
//! live ACROSS the region's dispatch chains, not inside one — and a region is
//! a coalesced run of nodes with one class mask, which is a set the compiler
//! chooses and this file does not get to constrain. Nothing stops such a
//! region from holding a quantized projection's pre-cast staging or a routed
//! matmul; P4's withdrawn regions on today's catalog hold neither,
//! but that is a fact about five model texts and not an invariant, and a
//! reservation resting on it would come apart the first time a mixture landed
//! in a withdrawn window. The three sentences above are checkable by hand
//! precisely because those roles live inside ONE dispatch chain; this one does
//! not, so it is charged honestly:
//!
//! ```text
//! bytes = max(precast, routed, index) + copy + pool
//! ```
//!
//! **AND THE POOL ROLE IS ADDED FOR A HARDER REASON THAN THE COPY'S.** A copy
//! is live across a region; the compressor state is live across FIRES. It is
//! indexed by the source pool's paged slot, so a cell holds the state of a
//! token some earlier fire wrote and the whole point of the plane is that it
//! survives — it fails the third sentence above, not merely the first. See
//! the field.
//!
//! What that costs is bounded and small, and the bound is the bucket lattice's
//! rather than the budget's: P4 writes `Fallback::Copy` only BELOW the
//! copy/split crossover (`model_compiler::layout`'s `CROSSOVER_ROWS`), so the
//! rectangle is sized at the largest bucket a `Copy` row actually covers and
//! not at `max_tokens`. A fire above the crossover takes the split and asks
//! this role for nothing. An artifact P4 withdrew no region from — every SKU
//! outside the qwen family — reserves zero here, because the ceiling walk
//! finds no region to size.
//!
//! **AND ON AN ARTIFACT WITH NO MIXTURE THE FOOTPRINT IS ONE ACTIVATION.**
//! `inputs` records the rule this could otherwise break — a grant nothing
//! reads is a reservation to delete, and it names the 216 MiB of workspace
//! that taught it. This plane is not that, and the reason is arithmetic
//! rather than intent: the precast rectangle is BY CONSTRUCTION one arena
//! activation, and the arena holds dozens of those live at the same instant.
//! So the whole plane is a low single-digit percentage of the arena beside
//! it, on every shape it is sized at.
//!
//! # What each ceiling is, in arithmetic
//!
//! **precast.** The staged copy is element-for-element the activation, and
//! `half` is the same two bytes the `bf16` it stages from is — so the plane
//! is `max_tokens x max_K x 2B`, where `max_K` is the widest activation any
//! quantized dense projection in this artifact contracts over. Both numbers
//! are read off the carve at the ceiling rather than off the weight rows: a
//! weight's declared rectangle is `[N, K]` for a dense bank and `[E, N*K]`
//! for an expert bank, so a table walk that did not know which was which
//! would size the plane at an expert bank's whole slab.
//!
//! **routed.** Whatever [`RoutedScratch`] declares, at `sorted_rows` of the
//! ceiling fire: four `i32` vectors and the gathered `x`/`y` at
//! `sorted_rows x K` and `sorted_rows x N`. `moe::sorted_rows` is
//! deliberately pessimistic — every touched expert can waste `tile - 1` rows
//! — and that pessimism is the reservation's, which is the point of asking it
//! here rather than measuring a fire.
//!
//! **copy.** The widest copied region's own operands, and the walk is the
//! same one the fire makes: every region P4 wrote a `Fallback::Copy` row for
//! and `crate::window::copyable` admits, its row-shaped operands deduplicated
//! by ARENA ROOT (two values the carve folded onto one column share one
//! staging rectangle, or an in-place op would stop being in place), each at
//! `rows x width x element` and 16-byte aligned, summed; the largest such sum
//! wins. `rows` is the largest bucket that row's range covers — see the
//! aliasing note above for why that is the honest ceiling and `max_tokens` is
//! not.
//!
//! **pool.** TWO slabs at `pool cells x coff·head_dim x 2B`, where the cells
//! are the whole paging's (`slots x context`) and not a fire's rows — the
//! gather addresses the state by the source pool's paged slot. `coff` is 2 at
//! ratio 4 and 1 elsewhere, and the widest gather in the artifact sets the
//! pitch. This is the same size class as one layer's kv pool, twice over, and
//! it is charged rather than trimmed: a slab short of the paging's ceiling is
//! an out-of-bounds device read on the first long sequence. Zero for every
//! artifact whose trace names no `attention.pool_gather`, which is every SKU
//! outside the dsv4-flash family.
//!
//! [`Run::tensor`]: crate::run::Run::tensor
//! [`RoutedScratch`]: kernels_metal::linear::moe::RoutedScratch

use kernels_metal::Tensor;
use kernels_metal::attn::ple;
use kernels_metal::linear::moe::{self, RoutedScratch};
use model_compiler::{Budget, Budgets, CompiledModel, Fallback, FireRows};
use model_exec::store::arena::rect;
use model_ir::{Attention, Def, Dim, Dtype, Linear, Operation, Trace, Ty, ValueId};

use crate::store::kv::Paging;

use crate::device::{Buffer, Context, Handles};
use crate::error::Result;
use crate::run::{PoolSlabs, WeightRow, WeightTable};

/// The alignment every role and every sub-rectangle starts on — `inputs`'
/// number, for `inputs`' reason.
const ALIGN: u64 = 256;

/// One working rectangle inside the plane: where it starts, and the shape a
/// [`Tensor`] states for it.
#[derive(Clone, Copy, Debug)]
struct Room {
    at: u64,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

impl Room {
    /// A rectangle of `rows x width`, laid down at the next aligned byte of a
    /// running offset and advancing it.
    fn lay(at: &mut u64, rows: u64, width: u64, dtype: Dtype) -> Room {
        let start = at.next_multiple_of(ALIGN);
        let room = Room {
            at: start,
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype,
        };
        *at = start + room.bytes();
        room
    }

    fn bytes(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.width) * self.dtype.bytes_ceil()
    }

    /// Mint the handle this rectangle resolves through. Fire-lived, like
    /// every arena row: `Handles::rewind` drops it at the end of the fire and
    /// the reservation underneath does not move.
    fn bind(&self, handles: &Handles, plane: &Buffer) -> Result<Tensor> {
        Ok(Tensor::new(
            handles.bind(plane, self.at, self.bytes())?,
            self.rows,
            self.width,
            self.dtype,
        ))
    }
}

/// The copy role's whole slab: where it starts, and how many bytes it holds.
///
/// A byte extent rather than a [`Room`] because it has no shape of its own —
/// `crate::dispatch::copy` sub-divides it per region, one rectangle per
/// operand at that operand's own width and element, and the only thing this
/// reservation states is how far it may go.
#[derive(Clone, Copy, Debug)]
struct CopyRoom {
    at: u64,
    bytes: u64,
}

/// **ONE POOLED SPACE'S two state slabs**, in the order they are laid down.
///
/// `space` is the SOURCE cache's index — the `pages` operand of the
/// `attention.pool_gather` that reads these slabs, and of the
/// `attention.pool_state_write` that fills them. A layer's compressor state
/// is a function of that layer's own kv rows, so two pooled layers hold
/// DIFFERENT state at the same paged cell and cannot share a plane.
#[derive(Clone, Copy, Debug)]
struct Pool {
    space: u32,
    state_kv: Room,
    state_score: Room,
}

/// One hashing's constants: the plane they were written into, and the numbers
/// themselves as the key a dispatch arm finds it by.
#[derive(Clone, Debug)]
struct PleHash {
    /// `[mults][primes][offsets]`, exactly as
    /// [`kernels_metal::attn::ple::hash_constants`] lays them down — which is
    /// also what was written to the device, so this field IS the plane's
    /// contents and not a summary of them.
    key: Box<[u64]>,
    room: Room,
}

/// The sorted arm's six rectangles, in the order they are laid down.
#[derive(Clone, Copy, Debug)]
struct Routed {
    perm: Room,
    row_expert: Room,
    tile_expert: Room,
    inv: Room,
    x: Room,
    y: Room,
}

/// The reservation, its three roles, and the two load-time tables a dispatch
/// arm reads beside them.
///
/// A role no node in this artifact asks for is `None` and not a zero-sized
/// rectangle: an absence a caller answers by taking the arm that needs no
/// plane, rather than an extent every caller would have to check.
#[derive(Debug)]
pub struct Scratch {
    /// The one reservation, or an empty buffer for an artifact that asks for
    /// no role at all. Zero bytes is a legal `Buffer` and mints no handle,
    /// which is exactly the state every accessor below reports as `None`.
    plane: Buffer,

    precast: Option<Room>,
    routed: Option<Routed>,
    /// The NSA indexer's per-row score slab: `rows x max_kv` floats,
    /// `None` for an artifact with no `attention.index_topk` node — which is
    /// every SKU outside the dsv4-flash family.
    ///
    /// **ALIASED WITH THE TWO ABOVE, AND IT PASSES ALL THREE SENTENCES.** The
    /// selection writes this rectangle and bisects over it INSIDE ONE
    /// KERNEL — `index_topk_paged_bfloat16` scores every visible key into its
    /// row, folds a min/max over it, halves forty times counting against it,
    /// and collects; nothing outside that dispatch ever reads a byte of it.
    /// So it is not merely dead before the next chain, it is dead before the
    /// next DISPATCH, which is the strongest form of the first bullet
    /// available. It cannot collide with a copied region either, because
    /// `copy` is added rather than unioned.
    index: Option<Room>,
    /// **NOT ALIASED ONTO THE THREE ABOVE**, and the header's fourth bullet
    /// is why: a copy's bytes are live across a whole region's dispatch
    /// chains, where every other role's are dead before the next chain
    /// starts. `None` for an artifact P4 withdrew no copyable region from,
    /// which is every SKU outside the qwen family.
    copy: Option<CopyRoom>,

    /// The dsv4 compressor's rolling state, ONE ENTRY PER POOLED SPACE and
    /// empty for an artifact with no `attention.pool_gather` node — which is
    /// every SKU outside the dsv4 family.
    ///
    /// **ALSO NOT ALIASED, AND FOR A STRONGER REASON THAN THE COPY'S.** Every
    /// other role here is working storage whose first dispatch writes every
    /// byte it later reads. This one is not written by its reader at all: it
    /// is addressed by the SOURCE POOL'S PAGED SLOT, so a cell holds the
    /// compressor state of a token from some earlier fire, and the whole
    /// point of the plane is that it survives. Unioning it with a pre-cast
    /// staging rectangle would let a projection three layers up overwrite the
    /// window this fire is about to pool. So it is added, like the copy, and
    /// the honest cost is stated rather than hidden.
    ///
    /// **AND IT IS ONE PLANE PER SPACE NOW, NOT ONE PLANE FOR ALL OF THEM.**
    /// The reservation used to lay one, on the argument that nothing wrote
    /// either slab so no two pooled layers could disagree about a cell — a
    /// statement about the seam and explicitly not a licence.
    /// `attention.pool_state_write` is that writer, so the licence expired:
    /// layer 2's `wkv · x` and layer 3's are different numbers at the same
    /// paged cell, and one plane would hand the earlier layer's gather the
    /// later layer's projections. The cost is honest and large — a pooled
    /// layer's state is two slabs the size class of its own kv pool — and it
    /// is charged per pooled layer because that is what the addressing is.
    pool: Vec<Pool>,

    /// qwen4's PLE hash constants: one `u64` plane per DISTINCT hashing the
    /// trace states, empty for an artifact with no `attention.ple_ngram_ids`
    /// node — which is every SKU outside the qwen38-flash family.
    ///
    /// **THE ONE ROLE THE HOST WRITES, AND IT WRITES IT ONCE.** The header
    /// says nothing here is ever touched by the host; this is the exception
    /// and it is a narrow one. The bytes are written inside
    /// [`Scratch::reserve`] — before the reservation is handed to a `Shell`,
    /// so before any command buffer exists that could be reading them — and
    /// no fire path ever writes a byte of this plane again. That is why it
    /// needs no second copy where [`crate::inputs`] needs one: the hazard
    /// `inputs` duplicates against is a host `memcpy` landing in bytes a
    /// committed buffer is already reading, and a write that happens before
    /// the first command buffer cannot be that.
    ///
    /// **ALSO NOT ALIASED**, for a stronger reason than the compressor's: it
    /// is not working storage at all. It is a trace CONSTANT — the seed-derived
    /// multipliers, primes and offsets `models::qwen_4::hash_constants` derives
    /// — living on the device because `kernels_metal::ArgValue` has no
    /// by-value blob seat to carry them through, and the day a pre-cast
    /// overwrote it the hasher would still answer a number.
    ///
    /// **AND IT IS A LIST BECAUSE A HASHING IS NOT A LAYER.** The two arms of
    /// one PLE — the decode form and the chunked form — state the SAME
    /// constants, so today's catalog lays exactly one plane; a second PLE
    /// layer would state its own and get its own, and the lookup is by the
    /// numbers themselves rather than by node, because the numbers are what
    /// the shader reads.
    ple: Vec<PleHash>,

    /// Per `ValueId`: how many ROWS the arena slot behind it can hold — the
    /// slot's whole reservation at the budget's ceiling, not this fire's
    /// extent. [`crate::arena::capacities`] is the arithmetic; the reason it
    /// is here is that "rows past the fire's own to write discardable output
    /// into" is the same noun as the rest of this file.
    capacity: Vec<u32>,

    /// Per `ValueId`: the expert count of the router that produced it, or 0
    /// for a value no router named.
    ///
    /// **THE ROUTER OP NAMES IT AND NOTHING ELSE DOES.** `moe::tile_rows`
    /// prices its tile off rows per expert; no operand of `MoeMatmulSelect*`
    /// states how many experts there are, and the bank does not carry it —
    /// its declared rectangle is `[E, N*K]` only by the convention of the
    /// text that emitted it. What DOES state it is `MoeTopk*`'s `experts`
    /// field, and the routing vector it writes is the very operand the select
    /// op reads. So the fact is carried along the edge that already exists,
    /// resolved once at load, and read per node — never remembered between
    /// two dispatches of one fire, which is the version of this that would
    /// break the day a plan interleaved two mixtures.
    routers: Vec<u32>,
}

impl Scratch {
    /// Carve the plane this artifact can ask for, at the budget's ceiling.
    ///
    /// Every ceiling is read off the same carve the arena is driven by, at
    /// ALL FOUR of the budget's row counts — so a rectangle this plane is
    /// sized for is a rectangle the arena also reserved room for, and the two
    /// cannot disagree about how big a fire may be.
    ///
    /// **AND THE PATCH PAIR IS STATED RATHER THAN ZEROED**, which is the same
    /// sentence [`crate::arena::carve`] carries: `FireRows::text_only` sizes a
    /// patch rectangle at no rows, and a tower node whose bank the loader
    /// seated as an affine triplet would then ask this plane for a
    /// zero-row pre-cast room and be handed one. Every tower in the catalog
    /// is dense bf16 and asks for none — the towers are never quantized
    /// (multimodal §21.1) — but gemma's `embed_vision.embedding_projection`
    /// IS a quantized bank reading a patch-axis activation, so the honest
    /// ceiling is the one that counts its rows.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`](crate::error::Fault::Deviceless) for a non-Apple
    /// build, [`Fault::Device`](crate::error::Fault::Device) when the device
    /// declined the length.
    pub fn reserve(
        device: &Context,
        trace: &Trace,
        weights: &WeightTable,
        compiled: &CompiledModel,
        budgets: &Budgets,
        paging: Paging,
    ) -> Result<Scratch> {
        let budget = &budgets.tokens;
        let map = &compiled.arena;
        let ceiling = FireRows {
            tokens: u64::from(budget.max_tokens),
            lanes: u64::from(budget.max_lanes),
            patches: u64::from(budgets.max_patches()),
            images: u64::from(budgets.max_images()),
        };
        let of = |id: ValueId| rect(map, id, ceiling);
        let banked = |id: ValueId| match trace.values.get(id.0 as usize).map(|v| &v.def) {
            Some(Def::Weight(w)) => matches!(
                weights.0.get(*w as usize).copied().flatten(),
                Some(WeightRow::Planes(_))
            ),
            _ => false,
        };

        let routers = routers(trace);
        let tuning = kernels_metal::tuning::current();

        // The dense quantized projections, and the routed ones, in one pass.
        // `dense` collects the (N, K) pairs the split sweep needs; the rest
        // is running maxima.
        let mut dense: Vec<(u32, u32)> = Vec::new();
        let (mut act_rows, mut act_k) = (0u64, 0u64);
        let (mut sorted, mut pairs, mut routed_k, mut routed_n) = (0u64, 0u64, 0u64, 0u64);
        for node in &trace.nodes {
            let Operation::Linear(op) = &node.op else {
                continue;
            };
            match op {
                Linear::Matmul { act, w, y } | Linear::LmHead { act, w, y } if banked(*w) => {
                    let (Some(act), Some(y)) = (of(*act), of(*y)) else {
                        continue;
                    };
                    act_rows = act_rows.max(u64::from(act.rows));
                    act_k = act_k.max(u64::from(act.width));
                    let pair = (y.width, act.width);
                    if !dense.contains(&pair) {
                        dense.push(pair);
                    }
                }
                Linear::MoeMatmulSelectBias { x, routes, y, .. }
                | Linear::MoeMatmulSelectQuant { x, routes, y, .. } => {
                    let (Some(x), Some(y)) = (of(*x), of(*y)) else {
                        continue;
                    };
                    let experts = routers.get(routes.0 as usize).copied().unwrap_or(0);
                    if experts == 0 {
                        continue;
                    }
                    // **A NODE THE SORTED ARM DECLINES AT THE CEILING ASKS
                    // FOR NOTHING.** `moe::tile_rows` answers 1 below
                    // `should_batch`'s threshold and `matmul_select_batched`
                    // returns without encoding when it does — and both
                    // `should_batch` and the tile widen with the pair count,
                    // so a ceiling fire that will not batch means no fire
                    // will. Reserving for it would be bytes for a chain that
                    // cannot run.
                    if moe::tile_rows(y.rows, experts, &tuning) <= 1 {
                        continue;
                    }
                    let rows = moe::sorted_rows(y.rows, experts, &tuning);
                    sorted = sorted.max(u64::from(rows));
                    pairs = pairs.max(u64::from(y.rows));
                    routed_k = routed_k.max(u64::from(x.width));
                    routed_n = routed_n.max(u64::from(y.width));
                }
                _ => {}
            }
        }

        // **THE PRODUCT OF THE MAXIMA, AND THE PESSIMISM IS ONE SENTENCE.**
        // A shape-exact plane would be the max over nodes of `rows x width`;
        // this is the max of each factor multiplied. They differ only for an
        // artifact whose widest projection is not its tallest, and on every
        // stack measured the routed shapes are one family — so the difference
        // is zero in practice and the arithmetic stays one line.
        let precast = (act_rows > 0 && act_k > 0).then(|| {
            let mut at = 0u64;
            Room::lay(&mut at, act_rows, act_k, Dtype::F16)
        });
        let routed = (sorted > 0).then(|| {
            let mut at = 0u64;
            Routed {
                perm: Room::lay(&mut at, 1, sorted, Dtype::I32),
                row_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),
                // One entry per TILE of the sorted stack. Sized at one per
                // ROW instead: the tile is a fire-time choice off
                // `moe::tile_rows` and its narrowest value is what makes the
                // stack deepest in tiles, so a per-row vector covers every
                // tile count the selection can land on, at four bytes a row.
                tile_expert: Room::lay(&mut at, 1, sorted, Dtype::I32),
                inv: Room::lay(&mut at, 1, pairs, Dtype::I32),
                x: Room::lay(&mut at, sorted, routed_k, Dtype::Bf16),
                y: Room::lay(&mut at, sorted, routed_n, Dtype::Bf16),
            }
        });

        // **THE INDEXER'S SCORE SLAB, AT THE PAGING'S OWN CEILING.** One row
        // per query row the selection launches over, one float per cached key
        // that row can see. The width is `pages_per_slot * page_size` — the
        // per-request page budget the pool was reserved at, which is exactly
        // the `max_pages_per_request * page_size` the CUDA entry sizes its
        // process-global scratch by. It is a ceiling and not a measurement,
        // so the shader's `nkeys > score_stride` clamp is the same clamp on
        // both planes.
        //
        // A trace with no `attention.index_topk` node asks for nothing, so
        // the bytes are charged to the one model family that selects.
        let index = trace
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Attention(Attention::IndexTopk { selection, .. }) => of(*selection),
                _ => None,
            })
            .map(|rect| u64::from(rect.rows))
            .max()
            .filter(|rows| *rows > 0)
            .and_then(|rows| {
                let keys = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
                (keys > 0).then(|| {
                    let mut at = 0u64;
                    Room::lay(&mut at, rows, keys, Dtype::F32)
                })
            });

        let union = precast
            .map_or(0, |r| r.at + r.bytes())
            .max(routed.map_or(0, |r| r.y.at + r.y.bytes()))
            .max(index.map_or(0, |r| r.at + r.bytes()));
        // **ADDED, NOT UNIONED** — the header's fourth bullet. It starts
        // where the three aliased roles end, so a copied region holding a
        // routed matmul is two disjoint spans and not one span read twice.
        let copy = copy_ceiling(trace, compiled, budget).map(|bytes| CopyRoom {
            at: union.next_multiple_of(ALIGN),
            bytes,
        });

        let mut at = copy.map_or(union, |room| room.at + room.bytes);
        // **THE COMPRESSOR'S STATE, ADDED AFTER THE COPY, ONE PLANE PER
        // POOLED SPACE** — see the field. A gather reads the cell its own
        // `pages` operand addresses, so the plane is keyed by that space:
        // `attention.pool_state_write` now fills the slabs from the layer's
        // own `wkv`/`wgate` projections, and two pooled layers hold different
        // state at the same paged cell.
        let pool: Vec<Pool> = pool_state(trace, paging)
            .into_iter()
            .map(|(space, cells, width)| Pool {
                space,
                state_kv: Room::lay(&mut at, cells, width, Dtype::Bf16),
                state_score: Room::lay(&mut at, cells, width, Dtype::Bf16),
            })
            .collect();

        // **THE PLE'S CONSTANTS, ADDED LAST AND WRITTEN ONCE** — see the
        // field. One plane per distinct hashing, so the two arms of one PLE
        // layer share the plane whose numbers they share.
        let mut ple: Vec<PleHash> = Vec::new();
        for node in &trace.nodes {
            let Operation::Attention(
                Attention::PleNgramIds {
                    mults,
                    primes,
                    offsets,
                    ..
                }
                | Attention::PleNgramIdsChunked {
                    mults,
                    primes,
                    offsets,
                    ..
                },
            ) = &node.op
            else {
                continue;
            };
            let key: Box<[u64]> = ple::hash_constants(mults, primes, offsets).into();
            if key.is_empty() || ple.iter().any(|held| held.key == key) {
                continue;
            }
            let room = Room::lay(&mut at, 1, key.len() as u64, Dtype::U64);
            ple.push(PleHash { key, room });
        }

        let mut plane = Buffer::zeroed(device, at)?;
        for hashing in &ple {
            let bytes: Vec<u8> = hashing
                .key
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            plane.write(hashing.room.at, &bytes)?;
        }

        Ok(Scratch {
            plane,
            precast,
            routed,
            index,
            copy,
            pool,
            ple,
            capacity: crate::arena::capacities(map),
            routers,
        })
    }

    /// Every byte the plane holds — the number a footprint line prints.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.plane.bytes()
    }

    /// How many rows the arena slot behind `id` can hold.
    ///
    /// `0` for a value the arena binds no rectangle for, which is the answer
    /// that makes every guard reading it fall back rather than pad.
    #[must_use]
    pub fn capacity(&self, id: ValueId) -> u32 {
        self.capacity.get(id.0 as usize).copied().unwrap_or(0)
    }

    /// The expert count of the router that wrote `routes`, or `0` for a value
    /// no router named — see the field.
    #[must_use]
    pub fn experts(&self, routes: ValueId) -> u32 {
        self.routers.get(routes.0 as usize).copied().unwrap_or(0)
    }

    /// The FP16 staging plane, cut to `rows x contraction`.
    ///
    /// `None` when this artifact reserved none, and when the fire's rectangle
    /// is larger than the ceiling reserved — which cannot happen, and is
    /// answered rather than asserted because the caller's answer to `None` is
    /// already "take the arm that needs no plane".
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn precast(&self, handles: &Handles, rows: u32, contraction: u32) -> Option<Result<Tensor>> {
        let room = self.precast?;
        if u64::from(rows) * u64::from(contraction) > room.bytes() / 2 {
            return None;
        }
        Some(Room {
            rows,
            width: contraction,
            ..room
        }
        .bind(handles, &self.plane))
    }

    /// One rectangle of the copy role's slab, at `offset` bytes into it.
    ///
    /// **THE OFFSETS ARE THE CALLER'S AND THE BOUND IS THIS FILE'S.**
    /// `crate::dispatch::copy` lays a copied region's operands out end to
    /// end and hands each one's place back here; what this checks is that the
    /// place is inside what the load reserved. A miss is `None` and not a
    /// panic, because the reservation is sized at the largest bucket a
    /// `Fallback::Copy` row covers and the caller is the code that knows a
    /// fire cannot be bigger — the honest answer to "this does not fit" is a
    /// refusal that names the copy, which is what the caller writes.
    ///
    /// `None` also for an artifact that reserved no copy role at all, which
    /// no gathered window can arise on: [`copy_ceiling`] and
    /// `crate::window::Windows::of` ask the same two questions of the same
    /// table.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn copy(&self, handles: &Handles, offset: u64, bytes: u64) -> Option<Result<u32>> {
        let room = self.copy?;
        if offset.saturating_add(bytes) > room.bytes {
            return None;
        }
        Some(handles.bind(&self.plane, room.at + offset, bytes))
    }

    /// The sorted arm's six rectangles, minted for the node asking.
    ///
    /// The rectangles are the CEILING's, not the node's: every extent the
    /// shaders index by is stated to them as a scalar argument
    /// (`padded`, `x.width`, `y.width`), and the only shape
    /// [`moe::matmul_select_batched`] reads off these tensors is
    /// `x.rows >= sorted_rows` — which the ceiling satisfies for every fire,
    /// because `sorted_rows` rises with the pair count and the tile, and a
    /// fire has no more pairs than the budget admits. So the same six
    /// rectangles answer every routed node, and what a call costs is six rows
    /// of the fire's handle table.
    ///
    /// `None` when the load reserved no routed room — an artifact with no
    /// mixture, or one whose router this shell could not read an expert count
    /// off — and the caller's answer to `None` is the matvec arm.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn routed(&self, handles: &Handles) -> Option<Result<RoutedScratch>> {
        let r = self.routed?;
        let mint = || {
            Ok(RoutedScratch {
                perm: r.perm.bind(handles, &self.plane)?,
                row_expert: r.row_expert.bind(handles, &self.plane)?,
                tile_expert: r.tile_expert.bind(handles, &self.plane)?,
                inv: r.inv.bind(handles, &self.plane)?,
                x: r.x.bind(handles, &self.plane)?,
                y: r.y.bind(handles, &self.plane)?,
            })
        };
        Some(mint())
    }

    /// The NSA indexer's score slab, whole — `rows x max_kv` f32, where the
    /// WIDTH is the `score_stride` the selection shader clamps each row's
    /// visible key count against.
    ///
    /// Handed over at its reserved extent rather than cut to the fire, for
    /// the reason the shader reads it: `score_stride` is the row pitch a
    /// launch addresses by, so narrowing it per fire would move every row of
    /// the slab and mean a different thing by the same argument.
    ///
    /// `None` for an artifact whose trace names no `attention.index_topk` —
    /// and the caller's answer to `None` is the refusal that says the load
    /// reserved no slab, not a fire against a rectangle that is not there.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn index_scores(&self, handles: &Handles) -> Option<Result<Tensor>> {
        Some(self.index?.bind(handles, &self.plane))
    }

    /// The dsv4 compressor's two state slabs for ONE POOLED SPACE, whole.
    ///
    /// `space` is the source cache's index — the `pages` operand of the
    /// gather asking, and of the state write that filled it. Two pooled
    /// layers hold different state at the same paged cell, so the plane they
    /// are handed is the one reserved for their own space.
    ///
    /// Handed over at their reserved extent for [`Scratch::index_scores`]'s
    /// reason: the width is the ROW PITCH the shader addresses cells by, so
    /// narrowing it per fire would move every row of the plane and mean a
    /// different thing by the same argument. The rows are the pool's cells,
    /// which no fire is a fraction of.
    ///
    /// `None` for an artifact whose trace names no `attention.pool_gather` —
    /// and the caller's answer to `None` is the refusal that says the load
    /// reserved no state, not a launch against a plane that is not there.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn pool_state(&self, handles: &Handles, space: u32) -> Option<Result<PoolSlabs>> {
        let rooms = *self.pool.iter().find(|held| held.space == space)?;
        let mint = || {
            Ok(PoolSlabs {
                state_kv: rooms.state_kv.bind(handles, &self.plane)?,
                state_score: rooms.state_score.bind(handles, &self.plane)?,
            })
        };
        Some(mint())
    }

    /// The `u64` plane holding THIS hashing's constants, minted into the fire.
    ///
    /// **FOUND BY THE NUMBERS AND NOT BY THE NODE**, which is what makes the
    /// two arms of one PLE share one plane without anything having to say they
    /// are the same layer: the shader reads the numbers, so the numbers are
    /// the identity. The list is one entry long on every SKU in the catalog,
    /// so the scan is a comparison and not a search.
    ///
    /// `None` for an artifact whose trace states no hashing with these
    /// constants — and the caller's answer to `None` is the refusal that says
    /// the load wrote no such plane, not a fire against constants that are not
    /// there.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a handle table
    /// already full.
    pub fn ple_hash(
        &self,
        handles: &Handles,
        mults: &[u64],
        primes: &[u64],
        offsets: &[u64],
    ) -> Option<Result<Tensor>> {
        let want = ple::hash_constants(mults, primes, offsets);
        let held = self.ple.iter().find(|h| *h.key == *want)?;
        Some(held.room.bind(handles, &self.plane))
    }
}

/// `2` for the overlapping `2*ratio` window of the ratio-4 compressor, `1`
/// otherwise — the reservation's copy of `kernels_metal::attn::pool`'s own
/// `compressor_coff`, which is private to that module and derived from the
/// ratio on both sides of the seam.
const fn compressor_coff(ratio: u32) -> u64 {
    if ratio == 4 { 2 } else { 1 }
}

/// The compressor state planes this artifact asks for: one
/// `(space, cells, width)` per POOLED CACHE SPACE, empty for a trace that
/// names no `attention.pool_gather`.
///
/// **KEYED BY THE SPACE THE GATHER ADDRESSES**, which is the `pages` operand
/// resolved through `Def::Cache`. Every gather over one space states the same
/// ratio in this catalog, but the width is taken as a maximum inside a space
/// all the same — a space read at two ratios strides the wider row, which is
/// exactly what `pool_gather_paged`'s `state_pitch` is for.
///
/// **THE ROWS ARE THE POOL'S CELLS AND NOT THE FIRE'S ROWS**, which is the
/// one thing that makes this reservation a different animal from every other
/// role in this file. `pool_gather_paged` addresses the state at
/// `paged_slot(...) * width + col`, and `paged_slot` is
/// `page_indices[...] * page_size + pos % page_size` — a GLOBAL cell of the
/// source pool. So the plane has to cover every cell the paging can hand out
/// (`pages() * page_size`, which is `slots x context`), at the widest pitch
/// any gather in this artifact states.
///
/// That is a large number — the same size class as one layer's kv pool, twice
/// over — and it is charged honestly rather than trimmed to something that
/// fits: a slab short of the paging's ceiling is an out-of-bounds device read
/// on the first long sequence, which is the failure mode this crate reserves
/// at ceilings to avoid.
fn pool_state(trace: &Trace, paging: Paging) -> Vec<(u32, u64, u64)> {
    let cells = paging.pages() * u64::from(paging.page_size);
    if cells == 0 {
        return Vec::new();
    }
    let mut spaces: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        let Operation::Attention(Attention::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        }) = &node.op
        else {
            continue;
        };
        let Some(Def::Cache(space)) = trace.values.get(pages.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let width = compressor_coff(*ratio) * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match spaces.iter_mut().find(|(held, _)| *held == *space) {
            Some((_, held)) => *held = (*held).max(width),
            None => spaces.push((*space, width)),
        }
    }
    spaces
        .into_iter()
        .map(|(space, width)| (space, cells, width))
        .collect()
}

/// The most bytes any one copied region's staging rectangles come to.
///
/// **THE SAME TWO QUESTIONS THE FIRE ASKS, ASKED AT THE CEILING.**
/// `crate::window::Windows::of` gathers a region iff P4 wrote it a
/// `Fallback::Copy` row at this fire's bucket AND
/// [`copyable_mask`](crate::window::copyable_mask) admits its operands; this walks
/// every region P4 wrote such a row for at ANY bucket and asks the second
/// question of it, so a region that can be gathered by some fire has room
/// here and one that cannot costs nothing.
///
/// The row ceiling is the largest bucket the `Copy` row's own range covers,
/// which is the honest one: above the crossover the table says `Split` and
/// this role is never asked for. A deployment that declared no lattice has
/// one implicit bucket, and the answer is then `Budget::max_tokens`.
///
/// Dedup is by ARENA ROOT — `rect` follows `ArenaMap::root`, so two values the
/// carve folded onto one column answer one `offset` and take one rectangle,
/// which is what keeps an in-place op in place when its operand and its result
/// are both compacted.
///
/// `None` when nothing in this artifact can be copied at all.
fn copy_ceiling(trace: &Trace, compiled: &CompiledModel, budget: &Budget) -> Option<u64> {
    let lanes = u64::from(budget.max_lanes);
    let mut most = 0u64;
    for region in compiled.template() {
        let rows = compiled
            .fallback
            .rows
            .iter()
            .filter(|row| region.nodes.contains(&row.node) && row.fallback == Fallback::Copy)
            .flat_map(|row| row.buckets.clone())
            .filter_map(|bucket| budget.buckets.get(bucket as usize).copied())
            .max()
            .or_else(|| {
                // A row with no lattice behind it: one implicit bucket at the
                // budget's ceiling (`FallbackRow::buckets`), which is the row
                // count a fire in it carries.
                compiled
                    .fallback
                    .rows
                    .iter()
                    .any(|row| region.nodes.contains(&row.node) && row.fallback == Fallback::Copy)
                    .then_some(budget.max_tokens)
            });
        let Some(rows) = rows else { continue };
        if !crate::window::copyable_mask(trace, compiled, &region.mask) {
            continue;
        }
        let Some((ins, outs)) = crate::window::operands(&trace.nodes, region) else {
            continue;
        };
        let mut seen: Vec<u64> = Vec::new();
        let mut at = 0u64;
        for id in ins.iter().chain(outs.iter()) {
            // Row-shaped, and only row-shaped: everything else a copied
            // region names is handed over whole or re-cut on the host
            // (`crate::window::Gathered`), and neither takes a byte here.
            let Some(decl) = trace.values.get(id.0 as usize) else {
                continue;
            };
            if !matches!(
                &decl.ty,
                Ty::Tensor { shape, .. } if matches!(shape.first(), Some(Dim::Tokens))
            ) {
                continue;
            }
            let Some(rect) = rect(&compiled.arena, *id, FireRows::text_only(u64::from(rows), lanes))
            else {
                continue;
            };
            if seen.contains(&rect.offset) {
                continue;
            }
            seen.push(rect.offset);
            let row = u64::from(rect.width) * rect.dtype.bytes_ceil();
            at = at.next_multiple_of(COPY_ALIGN) + u64::from(rows) * row;
        }
        most = most.max(at);
    }
    (most > 0).then_some(most)
}

/// Every staging rectangle starts 16-byte aligned, which is what lets the row
/// move pick its widest copy unit — the CUDA sibling's number, for its reason
/// (`engine_cuda::dispatch::copy`'s `align`). Smaller than [`ALIGN`] because
/// this is a division WITHIN one role and not the start of one.
const COPY_ALIGN: u64 = 16;

/// Per `ValueId`: the expert count of the router that wrote it.
///
/// One pass, because the three router variants all end in the same pair of
/// outputs and only one of them can have written any given vector.
fn routers(trace: &Trace) -> Vec<u32> {
    let mut out = vec![0u32; trace.values.len()];
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        let (routes, experts) = match op {
            Linear::MoeTopkSoftmax {
                routes, experts, ..
            }
            | Linear::MoeTopkSoftmaxScaled {
                routes, experts, ..
            }
            | Linear::MoeTopkSigmoid {
                routes, experts, ..
            }
            | Linear::MoeTopkSqrtSoftplus {
                routes, experts, ..
            }
            // **AND THE LOOKUP ROUTER**, which lands the same vector off a
            // table instead of off logits. Omitted, `experts` reads 0 for
            // every hash-routed layer, the sweep above `continue`s on it, and
            // the sorted arm reserves nothing for a mixture that is otherwise
            // exactly the ranked routers' — a silent narrowing, not a fault,
            // and so one nothing went red over. Measured on
            // `what_a_two_bit_prefill_costs`: dsv4-flash-mlxu2's 512-token
            // prefill runs 555 tok/s with the arm below missing and 768 with
            // it, because three of that model's five mixtures route by table
            // and were taking the matvec arm at every rectangle.
            | Linear::MoeHashRoute {
                routes, experts, ..
            } => (*routes, *experts),
            _ => continue,
        };
        if let Some(seat) = out.get_mut(routes.0 as usize) {
            *seat = (*seat).max(experts);
        }
    }
    out
}
