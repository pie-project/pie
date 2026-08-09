//! ② KERNEL SIGNATURES — the vocabulary. The rows live with the kernels
//! (`.wiki/tart/dsl.md` ②).
//!
//! `dsl::cuda` has ten wrappers over five attention kernels because
//! `_region` / `_planned` / `_capture` / `_dequant` encode the DISPATCH
//! CONTEXT in the wrapper name. The context is a property of the call site;
//! what belongs to the kernel is its symbol and its contract. A [`KernelSig`]
//! is that contract, once per symbol.
//!
//! Four declarations, each replacing something that is a hand-written runtime
//! rule today:
//!
//! | declaration | replaces |
//! |---|---|
//! | `whole`   | `if c.head_dim_padded \|\| (window_one && c.xqa_decode)` in the model body |
//! | `lacks`   | "a score-wanting program under XQA fails loudly PTIR-side" (a C++ throw) |
//! | `needs`   | the prepare a stated kernel obligates, named nowhere |
//! | `sink`    | `emit_cuda::emit_masked_pages_bracket`'s hardcoded page substitution |
//!
//! `whole` is CHECKED at trace time — which is load time, since a declaration
//! is traced when the model loads. The other three are declared but not yet
//! consumed: `needs`/`sink` are the emitter's knowledge until the launch ABI
//! flattens (migration step 6), and `lacks` needs the deployment's
//! servable-seam set, which is the support-matrix work. Declaring them first
//! is the point — the table is where they land, and it exists.
//!
//! ## Why this is its own crate
//!
//! The rows are in [`kernels-cuda`](../kernels_cuda/index.html) and
//! [`kernels-metal`](../kernels_metal/index.html), one crate per backend,
//! each beside the `.cu`/`.metal` it describes — so a new kernel is one
//! source file and one table row in the same directory and the same diff
//! hunk. Both tables have to be written in the same words, and neither
//! backend owns those words, so they are here.
//!
//! Bare-named for the same reason [`driver`](../driver/index.html) is: it is
//! the shared floor under a `-`-prefixed pair, holding what both members
//! speak rather than anything either one does. Nothing depends on it but the
//! two tables and the compiler that reads them, and it depends on nothing at
//! all — a row must be writable next to its kernel without dragging a
//! dependency graph along.

/// A capability a seam may ask of the kernel covering its rows. Named after
/// the seam vocabulary (`.wiki/tart/dsl.md` ①), because that is what a
/// `lacks` line refuses to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    /// The attention scores, published for an `attn.out` observer.
    Scores,
    /// The page-mask sink an `attn.q` tap writes.
    PageMaskSink,
}

/// How a kernel turns a rectangle into a thread grid.
///
/// A variant is a **shape of launch, not a kernel**: `Elementwise` serves every
/// 256-wide pointwise pass and `PerHead` both the q/k/v split and the KV
/// append, so a new kernel that launches like an existing one costs nothing.
///
/// This is data. The arithmetic each variant names stays in the driver, beside
/// the doc comment that explains it — the same split [`Prepare`] and
/// [`Source`] already make between what a contract STATES and what a backend
/// DOES about it.
///
/// The alternative was a `const` expression grammar on the row. Every rule in
/// use is uniformly `source -> max -> min -> divide-rounding-up -> multiply`,
/// so it fits; it was rejected because spelling a rule as
/// `Term { floor: 1, cap: 1024, div_ceil: 32, mul: 32 }` loses the sentence
/// that says why, and those sentences carry findings — one of them records
/// that a round-up is the difference between computing every output and
/// silently dropping the last few.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LaunchRule {
    /// The row has not said. A backend must REFUSE rather than guess: a
    /// guessed grid runs a kernel over the wrong extent, which no hardware
    /// reports. Same meaning [`Source::Unbound`] has for an operand.
    #[default]
    Unstated,
    /// Affine GEMV: four outputs per simdgroup, two simdgroups per
    /// threadgroup, rounded up.
    Qmv,
    /// Row-wise norm: one threadgroup per row, four elements per thread,
    /// capped at the widest threadgroup the backend allows.
    Rms,
    /// Rope: half the rotary channels, per head.
    Rope,
    /// Pointwise over one row, 256-wide — residual adds, embeddings, silu-mul.
    /// Rows stack flat: `width * rows` threads on one axis.
    Elementwise,
    /// Pointwise with the row on its own grid axis rather than stacked flat —
    /// what a gather whose rows are not contiguous needs
    /// (`embed_gather_mb`). One rule apart from [`LaunchRule::Elementwise`]
    /// because the two agree at one row and disagree above it, which is
    /// exactly the distinction a row has to be able to state.
    ElementwiseRows,
    /// One threadgroup per head, `head_dim` wide — the q/k/v split, the KV
    /// append. The row is the third grid axis.
    PerHead,
    /// Single-pass decode attention: one 1024-thread threadgroup per query
    /// head, rows on the second axis.
    SdpaVector,
    /// Pointwise over every head's channels, 256-wide.
    PerHeadElementwise,
    /// Gated norm over the value heads.
    GatedRms,
    /// One threadgroup as wide as the expert count PER ROW — the router's
    /// top-k, which `route.metal` indexes with `tgid.y`.
    ///
    /// The row axis is load-bearing and was missing: with `grid.y = 1` a
    /// mixture prefill routed row 0 only, and every other row's expert ids
    /// were whatever the last layer left there.
    RouterLane,
    /// ONE threadgroup as wide as the expert count, whatever the row count —
    /// the counting sort, which reduces across all `(row, slot)` pairs
    /// through threadgroup atomics and stripes them over its own lanes.
    ///
    /// Split from [`LaunchRule::RouterLane`] because they are two different
    /// rules that shared one name. Giving this one the row axis launches N
    /// copies of the same sort, each clearing and rewriting the permutation
    /// the others are reading — the grid is the contract, so two contracts
    /// need two rows.
    RouterSort,
    /// One threadgroup per row, as wide as the row, capped at 256.
    RouteRows,
    /// Routed GEMV: [`LaunchRule::Qmv`] per row, per expert slot.
    RoutedQmv,
    /// Pointwise over the launch's INPUT width with the row on its own axis —
    /// a statement that reads one packed buffer and writes several, where the
    /// output widths are each a fraction of the work. The QKV split is the
    /// case: three outputs, and the grid has to cover their sum.
    SplitPacked,
    /// Affine GEMM: the batched projection, tiled over rows and columns.
    /// Distinct from [`LaunchRule::Qmv`] because it is a different kernel with
    /// a different name, not the same one launched wider — which is what makes
    /// the M>1 lane a ROW's statement rather than a mode the driver picks.
    Qmm,
    /// One block per (row, head), 128 wide, with **two head-wide float arrays
    /// staged in shared memory** — the gated-delta recurrence and its chunked
    /// prefill, which read `q` and `k` out of `2 · head_dim` floats the launch
    /// hands them.
    ///
    /// [`LaunchRule::GatedRms`] is this grid to the digit and cannot serve it,
    /// which is the split [`LaunchRule::RouterSort`] made first: the norm
    /// launches 256 threads and no shared memory, and the scan needs 128 and a
    /// staging slab. Too LITTLE dynamic shared memory is not a launch failure.
    /// The kernels take their second array as `smem + head_dim`, so half the
    /// allocation writes `k` past the end of the block's slab and into what
    /// the next block is reading — a recurrence that answers, finitely, from
    /// another block's keys.
    ///
    /// The head axis is the VALUE head count and the shared size is the KEY
    /// head width. A model whose two differ is a model this rule cannot state
    /// twice over, and the backend refuses rather than picking one.
    RecurrentScan,
    /// One block per row, a fixed 256 wide, no shared memory — a scatter whose
    /// body strides its own row.
    ///
    /// The paged KV writes are the case: one block per destination token, a
    /// stride loop over `kv_heads · head_dim`, nothing reduced and nothing
    /// shared. Distinct from [`LaunchRule::Rms`], which is the same grid and
    /// block and means a REDUCTION — its width is the fold's contract and its
    /// shared scratch is the fold's — and from [`LaunchRule::RouteRows`],
    /// which sizes the block from the row and would launch a different
    /// geometry. Stating either would put a scatter under a reduction's rule,
    /// and the day a backend sizes that reduction's block on its width the
    /// scatter follows it silently.
    PerRow,
    /// One block per COLUMN, 64 wide, the row axis walked inside the block —
    /// the short causal convolution, which carries a per-channel state across
    /// the tokens it convolves.
    ///
    /// The grid is the transpose of every other rule here and it is the
    /// kernel's own: a channel's output at token `t` reads its state at `t-1`,
    /// so the token axis cannot be a grid axis and the channel must be one.
    /// Spread over rows instead and each block recomputes a state its
    /// neighbours are advancing — no fault, a convolution answered from the
    /// wrong history.
    PerChannel,
    /// Flat pointwise over the launch's INPUT extent — `rows · in_width`
    /// elements, the row folded into the index.
    ///
    /// [`LaunchRule::Elementwise`]'s arithmetic read at the other width, and
    /// the same distinction [`LaunchRule::SplitPacked`] makes against
    /// [`LaunchRule::ElementwiseRows`]: a statement that reads one buffer and
    /// writes a WIDER one is sized by what it reads, because the output extent
    /// is a multiple nothing on the launch states. Sized on the output it
    /// launches a multiple of the blocks it needs and relies on a bounds guard
    /// to throw them away; sized on the output of a NARROWING pass it launches
    /// a fraction and leaves the tail holding the previous layer's residual.
    ElementwiseIn,
    /// One block per row, 256 wide, with **one float of shared scratch per row
    /// of the rectangle** — a causal score buffer over the fire's own tokens.
    ///
    /// The sparse-attention index network is the case: every key is a token,
    /// so the scratch is `rows` floats and not a function of the block. That
    /// is the whole distance from [`LaunchRule::Rms`], whose smem is the warp
    /// count — and it is not a distance a backend may close by rounding up,
    /// because a top-k that reads its logits out of an allocation shorter than
    /// the row count masks the wrong keys and reports success.
    RowScores,
    /// One block per row PER HEAD — `rows · (width / head_dim)` blocks of the
    /// reduction's own width, falling back to `rows` when the statement named
    /// no head width.
    ///
    /// [`LaunchRule::Rms`]' grid with the per-head reading of the same symbol
    /// folded in. Two ops lower to one launcher — a norm over whole rows and a
    /// norm over each head's channels — and the C++ told them apart in the
    /// ARGUMENT it passed as `num_rows`, which is the row a backend that makes
    /// its own grid has to make instead. Under [`LaunchRule::Rms`] the per-head
    /// reading norms a whole q projection as a single row.
    RowsPerHead,
    /// `ceil(rows / 256)` blocks of 256 — ONE THREAD per row, not one block.
    ///
    /// The shape a per-row body with no reduction in it wants: a table read and
    /// a short gather cost a thread, and a block each would launch 256 times
    /// the blocks and idle 255 lanes of every one. Every other flat rule here
    /// multiplies the rows by a width before dividing; this one does not, and
    /// stating one of those instead covers `width` times too much ground.
    RowsFlat,
    /// A grid-stride slab: `min(ceil(units / 256), 1024)` blocks of 256, where
    /// `units` is the vectorised element count.
    ///
    /// The grid is CAPPED, so it is a launch shape rather than a cover: the
    /// kernel walks `i += gridDim.x * blockDim.x` and a grid short of the
    /// extent is correct, while under any rule that covers the extent exactly
    /// the same kernel runs a grid the device has to serialise anyway. The cap
    /// is the contract — a kernel without the stride loop launched this way
    /// computes a prefix and reports success.
    Slab,
    /// A 16x16 block over a rectangle — `ceil(width / 16)` by `ceil(rows / 16)`
    /// blocks, the only rule here whose block is not one-dimensional.
    ///
    /// What a kernel that indexes a matrix with both `threadIdx.x` and
    /// `threadIdx.y` needs. Flattening it to a 1-D block of 256 is the same
    /// thread count and a different addressing: every such kernel reads
    /// `threadIdx.y` for its row and would find zero.
    Tile16,
    /// One warp per (head, row), heads on `grid.y` and rows on `grid.z` —
    /// `dim3(1, heads, rows)` at 32 threads.
    ///
    /// The first rule in this vocabulary with a third grid axis. `grid.x` is
    /// LITERALLY one: the head's channels fit a warp, so the axis a channel
    /// tiling would use is spent and the two counts move up. Stated as a 2-D
    /// grid with the row on `grid.y` it addresses one head's worth of a
    /// buffer with `heads` of them.
    AxialRope,
    /// The recurrence tiled by warps over the VALUE width — `dim3(rows, heads,
    /// ceil(value_width / 4))` at 128 threads, nothing shared.
    ///
    /// [`LaunchRule::RecurrentScan`]'s two axes with the value channels split
    /// four ways across the block's warps, which is what lets the block hold a
    /// tile of the state in registers instead of a slab in shared memory. The
    /// missing shared allocation is the tell that these are two rules: the
    /// scan's block reads `2 · K_d` floats it must be given, and this one
    /// reads none.
    WarpTiledScan,
    /// One block per row, **128** wide, no shared memory — the same grid
    /// [`LaunchRule::PerRow`] states at half its block.
    ///
    /// A separate rule and not a parameter because the block width is a
    /// NUMERICS contract wherever a kernel folds warp partials serially: the
    /// audio tower's layernorm sums `(blockDim.x + 31) / 32` of them in thread
    /// zero, so 128 threads and 256 threads add the same values in a different
    /// order and answer with a different last bit. Stating the 256-wide rule
    /// on a 128-wide launcher is therefore a model change wearing a spelling
    /// change's clothes.
    PerRowNarrow,
    /// The reference paged attention's PREFILL grid —
    /// `dim3(requests, rows, q_heads)` at 128 threads, with
    /// `(head_dim + 128) * sizeof(float)` of DYNAMIC shared memory.
    ///
    /// Two things separate it from every rule above. The first is the shared
    /// allocation: it is never a literal, it is a head width plus a BLOCK
    /// width, and the block is the ADDEND rather than a factor —
    /// [`LaunchRule::SdpaVector`]'s `(rows + 256) * 4` adds a block to the
    /// wrong extent and would size the scratch on the token count. The
    /// second is `grid.x`: a REQUEST count, which is not the row count on a
    /// prefill and which no other rule reads.
    PagedScores,
    /// The same kernel family's DECODE grid — `dim3(rows, q_heads)` at 128
    /// threads with the same `(head_dim + 128) * sizeof(float)`.
    ///
    /// A separate rule and not a degenerate case of
    /// [`LaunchRule::PagedScores`], because it is a different `__global__`
    /// with a different `blockIdx` reading: decode's `blockIdx.y` is the
    /// QUERY HEAD and prefill's is the token offset within a request. One
    /// token per request makes the two grids numerically equal on a decode
    /// shape and semantically different everywhere else.
    PagedScoresDecode,
    /// MLA's fused prepare — `dim3(rows, 1 + ceil(q_heads / heads_per_block))`
    /// at 256 threads, nothing shared.
    ///
    /// The leading `1` is a LANE and not padding: `grid.y == 0` owns the
    /// latent norm, the `k_pe` rotation and the paged write, and
    /// `grid.y >= 1` splits the query heads. It cannot fold into the head
    /// axis because the page write consumes the rotated `k_pe`, and a
    /// cross-block dependency inside one kernel would need a grid sync.
    ///
    /// `heads_per_block` is [`LaunchRule::Rope`]'s head packing computed on
    /// the ROTARY width rather than the head width — `half = rotary / 2`,
    /// then `half >= 256 ? 1 : 256 / half` — because the query lane's work
    /// per block is one rotation and not one whole head.
    MlaPrepare,
    /// One block per (row, packed head) — `dim3(rows, q_heads + kv_heads)` at
    /// 256 threads, nothing shared.
    ///
    /// The second axis is the SUM of two head counts and is undivided: a
    /// fused QKV epilogue unpacks `[q | k | v]` and gives one block to every
    /// q head and every kv head of a row. [`LaunchRule::GatedRms`] is the
    /// nearest ported shape and its `grid.y` is `kv_heads` alone, so it is
    /// short by every query head — 32 of 40 blocks missing on a
    /// grouped-query shape, with the blocks it does launch correct.
    RowsPackedHeads,
    /// [`LaunchRule::RowsPackedHeads`] at **128** threads.
    ///
    /// A separate rule for [`LaunchRule::PerRowNarrow`]'s reason and one
    /// more. The block width is a NUMERICS contract — the kernel reduces a
    /// head's norm through `__shared__ float buf[BLOCK]` by halving, so a
    /// different width sums the same values in a different order — and here
    /// it is a CORRECTNESS one as well, because `BLOCK` is the template
    /// argument that SIZES that array: a 256-wide launch of the `<128>`
    /// instantiation reads 128 slots that were never written.
    RowsPackedHeadsNarrow,
    /// One WARP per (row, packed head), flattened —
    /// `ceil(rows * (q_heads + kv_heads) / (256 / 32))` blocks of 256.
    ///
    /// The unit of work is a warp and the grid is one-dimensional, which is
    /// what a kernel that reduces with `__shfl_xor_sync` and no
    /// `__syncthreads` wants: the (row, head) pair is recovered inside the
    /// kernel from `blockIdx.x * warps_per_block + warp_id`. Stated as a 2-D
    /// grid it would launch eight times the blocks; stated at
    /// [`LaunchRule::PerRow`]'s one block per row it would cover one head.
    WarpPackedHeads,
    /// [`LaunchRule::RoutedQmv`]'s two axes TRANSPOSED —
    /// `dim3(ceil(width / 8), rows * experts_per_token)` at 256 threads.
    ///
    /// Two launchers of one family put the routed slot count on different
    /// axes, and one rule cannot carry both: `wna16_gate_up_decode` reads
    /// `blockIdx.x` as its route and `blockIdx.y` as its warp slab, and
    /// `wna16_down_decode` reads them the other way round. Firing either
    /// under the other's rule covers a rectangle of the same AREA with the
    /// axes swapped, which for a non-square shape leaves most of the output
    /// untouched and faults on nothing.
    RoutedQmvTransposed,
    /// A third grid axis over an ALTUP STREAM count — `dim3(rows, streams,
    /// ceil(width / streams / 128))` at 128 threads.
    ///
    /// The second axis is a residual-stream index and the third tiles ONE
    /// stream's hidden width. [`LaunchRule::WarpTiledScan`] is the only other
    /// rule with a `grid.z` at this block width and is wrong twice over: its
    /// `z` is `ceil(V_d / 4)` where this is `ceil(H / 128)`, and its `grid.y`
    /// is filled from an attention head count. A stream count is neither.
    AltUpStreams,
    /// [`LaunchRule::RoutedQmv`]'s two axes at a **quad** tile, over a
    /// **stacked** output — `dim3(rows * experts_per_token,
    /// ceil((width / experts_per_token) / 16))` at **128** threads.
    ///
    /// The same axis ORDER as [`LaunchRule::RoutedQmv`] — routes on `x`, the
    /// output width slabbed on `y` — and neither of that rule's two numbers,
    /// nor its reading of the width. `quant/dequant_fp4.cu:67-70` and
    /// `:152-156` spell the geometry twice:
    ///
    /// ```text
    /// const int warps = kMxfp4DecodeBlock / 32;              // 128 / 32 = 4
    /// const int pairs_per_block = warps * kMxfp4GateUpPairs; //   4 *   4 = 16
    /// dim3 grid(num_tokens * top_k,
    ///           (intermediate + pairs_per_block - 1) / pairs_per_block);
    /// ```
    ///
    /// **Two constants differ from [`LaunchRule::RoutedQmv`] and they push in
    /// opposite directions**, which is why the near miss does not announce
    /// itself. `RoutedQmv` is `dim3(routes, ceil(width / 8))` at 256; this is
    /// `dim3(routes, ceil(width / 16))` at 128. The block is HALF as wide and
    /// the slab is TWICE as tall, so the two grids have the same `grid.x`, and
    /// on gpt-oss's 2 880-wide intermediate `ceil(2880/8) = 360` against
    /// `ceil(2880/16) = 180`: firing the `<4>` instantiation under `RoutedQmv`
    /// launches 360 blocks of 256 where 180 of 128 are wanted.
    ///
    /// # And that near miss is INVISIBLE in the output. Measured.
    ///
    /// `tests/launch_rules.rs::the_routed_qmv_near_miss_is_absorbed_and_the_wrong_divide_truncates`
    /// fires `mxfp4_moe_down_decode<4>` at `hidden = 128`, `k = 2`, 8 routes.
    /// The wrong grid is `[8, 32]` of 256 against the right `[8, 8]` of 128 —
    /// **four times the blocks** — and **0 of 2 048 bytes differ**, both
    /// writing all 1 024 values.
    ///
    /// The reason is `row0 = (blockIdx.y * (blockDim.x >> 5) + warp) * kRows`
    /// followed by `if (row0 >= hidden) return`. Both factors are read from
    /// `blockDim.x` at run time, so a wider block does not overrun a slab —
    /// it RENUMBERS the warps, and the renumbering is exactly the identity on
    /// the warps that fall inside the tensor. The extra three quarters take
    /// the guard. So this pair fails hazard 1's test the way `AltUpStreams`
    /// did: **the output cannot distinguish them, and only the block count
    /// can.**
    ///
    /// What CAN be seen is the divide going the wrong way, which is why the
    /// section below exists and why it is checked over the rows rather than
    /// inside `eval`: applying the fanout divide to a width that was already
    /// per-route makes `grid.y` a factor of `k` too SMALL, and a short grid
    /// is a truncation the bytes report. The same test fires `[8, 4]` — the
    /// divide applied twice — and measures **512 of 1 024 values written and
    /// 1 022 of 2 048 bytes differing.**
    ///
    /// The tile is `warps * rows_per_warp` and the `rows_per_warp` is the
    /// TEMPLATE argument — `kMxfp4GateUpPairs` for the gate/up leg,
    /// `kMxfp4DownRows` for the down leg, both `4` — so this rule states the
    /// product it was swept at and a sweep that changed either constant is a
    /// row that states a different rule rather than a rule that reads a
    /// number off nothing.
    ///
    /// # The width is STACKED, and the rule checks that it is
    ///
    /// The launcher divides `intermediate`, a PER-ROUTE width, and the two
    /// statements that reach it declare their outputs as `[Tokens, k,
    /// intermediate]` — the routed extent as a third dim, deliberately: the
    /// collapsed shape *"made the two `bf16_to_fp16` sites indistinguishable
    /// to anything reading the trace"* and was a live defect. So the first
    /// output's ROW width is `k * intermediate` and this rule divides by the
    /// fanout before it slabs, where [`LaunchRule::RoutedQmv`]'s statements
    /// declare `[Tokens, intermediate]` and it does not.
    ///
    /// That is a difference a rule NAME cannot carry, so the rule does not
    /// rely on one. Two things carry it instead. The rule refuses a `width`
    /// that does not divide by the fanout — a rectangle whose third dim is
    /// not `k` has no per-route width to slab. And
    /// `tests/launch_rules.rs::every_stacked_rule_reads_a_route_index_row`
    /// holds every row that states this variant to declaring its first input
    /// as the route-index row, `[Tokens, k]` of `i32`, which is the operand
    /// whose width IS the fanout. A collapsed-shape row fails that test at
    /// build rather than slabbing `intermediate / k` columns at a fire and
    /// leaving the rest of every row unwritten.
    RoutedQmvQuad,
    /// **Exactly one block** of 256 threads, whatever the rectangle — the
    /// grid is a literal `1` the host wrote and not a quotient.
    ///
    /// Three launchers in two families are this shape, and all three are
    /// kernels whose ONE block owns a whole serial structure: a prefix over a
    /// CSR (`attn/kv_paged.cu:516`'s `build_window_page_view`), a
    /// single-slot byte copy (`layout/slot_ops.cu:61`'s
    /// `copy_if_valid_slot`). The block strides its extent internally, so the
    /// rectangle reaches the kernel as an OPERAND and never as a grid.
    ///
    /// # Why the `1` may not be derived
    ///
    /// [`LaunchRule::RowsFlat`] answers `ceil(rows / 256)`, which equals `1`
    /// for every rectangle of 256 rows or fewer and grows past it — so a row
    /// that reached for it would be right on the fixtures and wrong in
    /// production, which is the shape §22.7 measured twice. [`LaunchRule::RouteRows`]
    /// and [`LaunchRule::PerRow`] open one block PER ROW: against
    /// `copy_if_valid_slot` that is the same copy repeated `rows` times
    /// (idempotent, so correct by accident), and against
    /// `build_window_page_view` it is `rows` blocks racing to write one
    /// output CSR.
    ///
    /// The alternative rejected here was a `Dims` field carrying the block
    /// width, which fails §21.14's test outright: a block width is a property
    /// of the LAUNCHER, so `Dims { block: 512 }` would be a well-formed
    /// statement that no fire can make true or false. The two widths this
    /// tree launches single blocks at are two variants, exactly as
    /// [`LaunchRule::PerRow`] and [`LaunchRule::PerRowNarrow`] are.
    Single,
    /// [`LaunchRule::Single`] at ONE WARP — `<<<1, 32>>>`.
    ///
    /// `attn/kv_paged.cu:533`'s `build_full_split_view` is the case and it is
    /// the only one: the kernel's whole body is a serial walk over `splits`
    /// on **thread zero** with the rest of the warp idle, and the launcher
    /// picked 32 because a warp is the smallest block that does not waste a
    /// scheduling slot. Stating [`LaunchRule::Single`] instead launches 256
    /// threads where 32 are wanted — not a wrong answer for this kernel, and
    /// still a rule that does not reproduce its launcher, which is the only
    /// property a rule has.
    SingleWarp,
    /// One block per REQUEST, 256 wide, nothing shared — [`LaunchRule::PerRow`]'s
    /// launch over [`crate::LaunchRule`]'s other row-shaped axis.
    ///
    /// `attn/attention_naive.cu:174`, `attn/page_compact.cu:45` and `:48` are
    /// the three, and the distinction from [`LaunchRule::PerRow`] is the one
    /// `Dims::requests`' own doc is about: **a request count is not a row
    /// count.** A prefill of 4 requests and 512 tokens has `rows == 512` and
    /// `requests == 4`, so `PerRow` opens 128 times the blocks, and every
    /// extra one indexes `qo_indptr[r]` and `slot_ids[r]` past their ends —
    /// off a buffer with one slot per REQUEST. On a pure decode the two
    /// numbers coincide, which is why the substitution survives every
    /// single-token fixture.
    PerRequest,
}

impl LaunchRule {
    /// Every variant, so a caller can enumerate the vocabulary rather than
    /// remember it.
    ///
    /// The `Metal` driver's `a_rule_that_ignores_its_rows_has_to_say_so` is
    /// what this exists for: the `RouterLane` row-axis defect survived
    /// because the rule was *absent* from the list that would have caught
    /// it, and a list you must remember to extend does not catch what you
    /// forgot. Adding a variant below without adding it here fails to
    /// compile, because the array's length is checked against the match.
    pub const ALL: &'static [Self] = &[
        Self::Unstated,
        Self::Qmv,
        Self::Rms,
        Self::Rope,
        Self::Elementwise,
        Self::ElementwiseRows,
        Self::PerHead,
        Self::SdpaVector,
        Self::PerHeadElementwise,
        Self::GatedRms,
        Self::RouterLane,
        Self::RouterSort,
        Self::RouteRows,
        Self::RoutedQmv,
        Self::SplitPacked,
        Self::Qmm,
        Self::RecurrentScan,
        Self::PerRow,
        Self::PerChannel,
        Self::ElementwiseIn,
        Self::RowScores,
        Self::RowsPerHead,
        Self::RowsFlat,
        Self::Slab,
        Self::Tile16,
        Self::AxialRope,
        Self::WarpTiledScan,
        Self::PerRowNarrow,
        Self::PagedScores,
        Self::PagedScoresDecode,
        Self::MlaPrepare,
        Self::RowsPackedHeads,
        Self::RowsPackedHeadsNarrow,
        Self::WarpPackedHeads,
        Self::RoutedQmvTransposed,
        Self::AltUpStreams,
        Self::RoutedQmvQuad,
        Self::Single,
        Self::SingleWarp,
        Self::PerRequest,
    ];

    /// A discriminant, used only to prove [`Self::ALL`] is complete.
    const fn index(self) -> usize {
        match self {
            Self::Unstated => 0,
            Self::Qmv => 1,
            Self::Rms => 2,
            Self::Rope => 3,
            Self::Elementwise => 4,
            Self::ElementwiseRows => 5,
            Self::PerHead => 6,
            Self::SdpaVector => 7,
            Self::PerHeadElementwise => 8,
            Self::GatedRms => 9,
            Self::RouterLane => 10,
            Self::RouterSort => 11,
            Self::RouteRows => 12,
            Self::RoutedQmv => 13,
            Self::SplitPacked => 14,
            Self::Qmm => 15,
            Self::RecurrentScan => 16,
            Self::PerRow => 17,
            Self::PerChannel => 18,
            Self::ElementwiseIn => 19,
            Self::RowScores => 20,
            Self::RowsPerHead => 21,
            Self::RowsFlat => 22,
            Self::Slab => 23,
            Self::Tile16 => 24,
            Self::AxialRope => 25,
            Self::WarpTiledScan => 26,
            Self::PerRowNarrow => 27,
            Self::PagedScores => 28,
            Self::PagedScoresDecode => 29,
            Self::MlaPrepare => 30,
            Self::RowsPackedHeads => 31,
            Self::RowsPackedHeadsNarrow => 32,
            Self::WarpPackedHeads => 33,
            Self::RoutedQmvTransposed => 34,
            Self::AltUpStreams => 35,
            Self::RoutedQmvQuad => 36,
            Self::Single => 37,
            Self::SingleWarp => 38,
            Self::PerRequest => 39,
        }
    }
}

// `ALL` is complete and in discriminant order, checked at COMPILE time: a new
// variant makes `index` non-exhaustive, and forgetting to list it here makes
// this assertion fail. Neither can be missed by a reviewer.
const _: () = {
    assert!(LaunchRule::ALL.len() == 40);
    let mut i = 0;
    while i < LaunchRule::ALL.len() {
        assert!(LaunchRule::ALL[i].index() == i);
        i += 1;
    }
};

/// The host-side plan a kernel's contract obligates: stated so a reader of
/// the model text can see which prepare a launch drags in, rather than
/// reading the driver to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prepare {
    /// No host plan.
    None,
    /// The FlashInfer decode plan (per fire, per layer group).
    DecodePlan,
    /// The FlashInfer ragged prefill plan.
    PrefillPlan,
    /// The custom-mask plan (`attn_page_mask`'s consumer).
    CustomPlan,
    /// XQA's fire-wide prepare — R-shaped, so it cannot be built per row
    /// window. This is why `xqa_decode` is also `whole`.
    FireWide,
    /// MLA's plan (`kernels::attn::plan_attention_mla_bf16`), which is its own kind
    /// rather than a FlashInfer plan under another name: it is built from
    /// `kv_lora_rank` and `qk_rope_head_dim` — a latent KV geometry no other
    /// prepare here has a field for — and it is cached in an `MlaPlanCache`
    /// the dispatch borrows, not in the shared attention workspace.
    MlaPlan,
}

/// One point of one instantiation axis, and the text it contributes to a name.
///
/// See [`KernelSig::axes`] for why a row has these at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    /// What varies. Prose, for a reader of the table; the matcher never reads
    /// it.
    pub what: &'static str,
    /// The suffixes this axis can contribute, in the order a name spells them.
    /// Exactly one is present in any entrypoint the axis reaches.
    ///
    /// A point MAY be `""`, for an axis whose default specialisation adds no
    /// text — `sdpa_paged_decode<…, 0, false, 32>` is spelled
    /// `sdpa_paged_decode_bfloat16_d_128` and the two others are `…_p32` and
    /// `…_p32_sg8`, off ONE template. Two rules follow and both are checked by
    /// [`KernelSig::covers`]'s ordering rather than asserted:
    ///
    /// * the empty point goes LAST, because matching is first-wins and an
    ///   empty suffix matches everything;
    /// * a longer point goes before a shorter one it ends with (`_p32_sg8`
    ///   before `_p32`), for the same reason.
    pub points: &'static [&'static str],
}

/// What one operand of a launcher is, in words neither backend owns.
///
/// This is deliberately a SMALL vocabulary. It is not a type system and it is
/// not trying to describe what a buffer contains — `q`, `k` and `k_pages` are
/// all [`Ty::BufMut`], because how a kernel reads its own tensor is the
/// kernel's business. What it has to describe is exactly what a CALLER must
/// know to place an argument: how wide the word is, whether the callee may
/// write through it, and whether it may be absent.
///
/// The element-typed array kinds exist because the C++ spells them
/// (`const std::uint32_t*` for a CSR array, `const std::int32_t*` for
/// positions) and losing that would make the generated declaration a `void*`
/// that no longer proves anything — see [`crate`]'s note on why the shim is
/// generated rather than written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    /// An opaque device buffer the launcher may WRITE through (`void*`).
    BufMut,
    /// An opaque device buffer the launcher only reads (`const void*`).
    Buf,
    /// A read-only device array of `i32` — positions, and the like.
    I32s,
    /// A read-only device array of `i64`. One row needs it — kimi_k3's
    /// hash routing reads a `[vocab, K]` token-to-expert table — and it
    /// is its own kind rather than a `Buf` because the pilot caught
    /// exactly that substitution: `const void*` and `const int64_t*` are
    /// both pointers, so only the DECLARED width makes the mismatch a
    /// compile error instead of a stride bug.
    I64s,
    /// A read-only device array of `u32` — the CSR/indptr family.
    U32s,
    /// A read-only device array of `u8` — the per-row validity masks.
    U8s,
    /// A device array of `f32` the launcher WRITES — the built tables.
    F32sMut,
    /// A read-only device array of `f32` — softmax scales, sinks, biases.
    F32s,
    /// A device array of `i32` the launcher WRITES.
    I32sMut,
    /// A device array of `u32` the launcher WRITES — the indptrs a plan
    /// builds, as opposed to the ones a dispatch reads.
    U32sMut,
    /// A device array of `u8` the launcher WRITES — the packed masks.
    U8sMut,
    /// A host scalar.
    I32,
    /// A host scalar. Distinct from [`Ty::I32`] because the headers spell it
    /// `std::uint32_t`, and a mirror that guessed `i32` would be a silent
    /// sign bug on any value above 2^31 rather than a compile error.
    U32,
    /// ── POINTER ARRAYS ────────────────────────────────────────────
    ///
    /// A batched or grouped launch is handed an ARRAY of buffers rather
    /// than one, and two independent things can be const about it: the
    /// array itself, and what its entries point at. Four kinds, because
    /// all four combinations are live and C++ will silently accept
    /// three of them where a fourth was meant.
    ///
    /// Naming reads outside-in — `BufArray` is the array the launcher
    /// READS of buffers it READS; `Mut` on the end means the BUFFERS are
    /// writable; `Out` in the middle means the ARRAY is.
    ///
    /// The distinction is not pedantry. `gemm::batched_act_x_wt_bf16`
    /// takes its destination array as `void* const*` — it writes the
    /// buffers and reads the array — while the nemotron pointer
    /// BUILDERS take `void**`, because building the array is what they
    /// do. A row that swapped them would compile as `-fpermissive` and
    /// hand a builder an array it must not write.

    /// `const void* const*` — an array the launcher reads, of buffers it
    /// reads.
    BufArray,
    /// `void* const*` — an array the launcher reads, of buffers it
    /// writes. Every batched GEMM's destination list.
    BufArrayMut,
    /// `const void**` — an array the launcher WRITES, of buffers to be
    /// read. What a pointer builder fills for a later batched call.
    BufArrayOut,
    /// `void**` — an array the launcher writes, of buffers to be
    /// written. The destination half of the same builder.
    BufArrayOutMut,
    /// `const std::uint8_t* const*` — the same shape as [`Ty::BufArray`]
    /// with the element type spelled, which the MXFP4 banks need: their
    /// entries are packed nibbles and block scales, and a `void*` row
    /// would let a bf16 bank through.
    U8Array,
    /// A read-only device array of `u16`. Two launchers spell their
    /// bf16 buffers this way rather than as `void*`, and the pilot will
    /// not let a `Buf` row stand in: the pointer types differ, so the
    /// forward is a conversion C++ refuses.
    U16s,
    /// A device array of `u16` the launcher WRITES.
    U16sMut,
    /// A read-only device array of `i8` — an int8 LM head's weights.
    I8s,
    /// A device array of `i8` the launcher WRITES — an INT8 quantiser's
    /// destination.
    ///
    /// [`Ty::I8s`]' missing twin, and its absence was written down before it
    /// was closed: `quant::quantize_bf16_to_int8_per_channel` and
    /// `quant::cast_bf16_to_int8_per_channel` narrow to
    /// `int8_sym::store = i8` and had to say `U8sMut`, because unsigned
    /// bytes were the only byte-wide destination the vocabulary had. Same
    /// width and the same addresses, so nothing miscomputed — but the row
    /// told a reader to allocate unsigned where the kernel stores signed,
    /// and a function-pointer initialisation refuses `u8*` for `i8*`, so
    /// those two rows sat outside the offline check that would have caught
    /// the next drift.
    I8sMut,
    /// ── THE TWO SIXTEEN-BIT FORMATS, NAMED APART ──────────────────
    ///
    /// `Bf16s` and `F16s` are distinct kinds for exactly the reason
    /// `csrc/src/pie_device.cuh` makes `device::bf16` and `device::f16`
    /// distinct STRUCTS rather than two spellings of `unsigned short`:
    /// *"as typedefs they would be ONE type … the generated typecheck would
    /// accept a row that swapped them because there would be nothing to
    /// swap."* [`Ty::U16s`] is that collapsed spelling — it says the WIDTH
    /// and nothing about the format — so it cannot stand in for either.
    ///
    /// They exist because a kernel may FIX one end of a conversion and
    /// template the other. `bf16_to_narrow<T>` takes `const bf16*` whatever
    /// `T` is, and `cast_f16_to<T>` takes `const f16*`; a row's `elem` names
    /// the templated end, so [`Ty::Buf`] — which the device typecheck reads
    /// as `const {elem}*` — spells the fixed end as the wrong format. The
    /// neighbours with a fixed `float` or `u8` end escape through
    /// [`Ty::F32s`] and [`Ty::U8s`]; half precision had no such kind.
    ///
    /// `Bf16s` itself: a read-only device array of `device::bf16` — a fixed
    /// bfloat16 operand beside a templated one.
    ///
    /// The negative control is `kernels-cuda-new`'s
    /// `tests/device_typecheck_types.rs`, which compiles
    /// `quant::bf16_to_fp16`'s checker as stated and again with `in_bf16`
    /// spelled `F16s`, and reports nvcc accepting the first and answering the
    /// second with *"no instance of function template … bf16_to_narrow
    /// matches the required type"*.
    Bf16s,
    /// A read-only device array of `device::f16`. See [`Ty::Bf16s`].
    F16s,
    /// `const std::int32_t* const*` — an array of int32 buffers the
    /// launcher reads. The WNA16 expert banks are packed as `int32`
    /// words rather than opaque bytes, and a `BufArray` row would hand
    /// their kernel a void array it dereferences as int32.
    I32Array,
    /// `MoeActivation` — which activation a fused MoE leg runs. A
    /// caller-stated enum like [`Ty::Dtype`], and for the same reason:
    /// the declaration knows which, and a driver that inferred it from
    /// a config would be choosing.
    MoeActivation,
    /// `Mxfp4RowSelect` — which half of an interleaved MXFP4 bank a
    /// repack reads.
    Mxfp4RowSelect,
    /// The NVLink P2P all-reduce instance a fused collective is issued
    /// through — `kernels::comm::CustomAllReduce*`.
    ///
    /// A HANDLE, like [`Ty::Stream`] and [`Ty::CublasHandle`]: the arm is
    /// given it and never asks for it. It exists because the fused
    /// landing was a METHOD, and a method has no address the generated
    /// ABI can forward to — the free form takes the instance first, and
    /// the row can then describe the call.
    CustomAllReduce,
    /// The element type a buffer is stored in — `DType`, a
    /// `std::uint8_t`-backed enum class.
    ///
    /// It is a scalar the CALLER states, not a property the launcher
    /// discovers, and that is the whole reason the scaled GEMM entry
    /// points take it: the storage a weight is in used to reach them
    /// inside a `WeightView`, and a driver built that descriptor by
    /// looking at a per-layer struct no statement mentioned. The
    /// descriptor is assembled INSIDE the launcher now and the caller
    /// passes what the declaration said.
    ///
    /// Its own kind rather than [`Ty::U32`]: an enum class does not
    /// convert from an integer, so a row that widened it would not
    /// compile -- which is the answer wanted, but for the wrong reason.
    /// Spelling it means the shim forwards the enum the header declares.
    Dtype,
    /// `attn::device::KvScheme` — which quantisation a paged KV bank is
    /// stored under (`enum class … : ::std::uint8_t`,
    /// `attn/attention_naive_paged.cuh:141`).
    ///
    /// `naive_paged_attn` and `naive_paged_decode` take it BY VALUE, so a
    /// row that could not spell it could not be a row at all. Its own kind
    /// rather than [`Ty::U8s`]-of-nothing or a widened [`Ty::U32`] for
    /// [`Ty::Dtype`]'s reason: an `enum class` does not convert from an
    /// integer, so the shim forwards the enum the header declares and a row
    /// that widened it would not compile.
    ///
    /// # Why it is not the same kind as [`Ty::KvDType`]
    ///
    /// The two are `enum class … : ::std::uint8_t` in the same namespace and
    /// the two kernels take them ADJACENTLY, in that order. One shared
    /// `U8Enum` kind would make the swap type-check on every side — same
    /// width, same crossing, same `ArgValue` — and the swap is the classic
    /// same-width hazard [`Ty::Bf16s`] and [`Ty::F16s`] are two kinds for.
    /// Distinct C++ types are what make
    /// `abi::emit_device_typecheck`'s function-pointer initialisation catch
    /// it: an initialisation admits no conversions, and `enum class` admits
    /// none to begin with.
    KvScheme,
    /// `attn::device::KvDType` — which element type a paged KV bank stores
    /// (`enum class … : ::std::uint8_t`,
    /// `attn/attention_naive_paged.cuh:152`). See [`Ty::KvScheme`], whose
    /// parameter it follows in both kernels.
    ///
    /// Not [`Ty::Dtype`], which is `::pie_cuda_driver::DType` — a different
    /// enumeration with a different member list, declared in a different
    /// header. `naive_paged_attn` takes this one and the two do not convert.
    KvDType,
    /// A host scalar spelled `long long` — the recurrent state's slot
    /// stride, which is an ELEMENT count into a multi-gigabyte arena and
    /// so was widened deliberately. Its own kind for `Ty::U32`'s reason:
    /// a mirror that guessed `int` is a silent truncation, not an error.
    I64,
    /// A host byte count, spelled `std::size_t`. Its width is the platform's,
    /// which is why it is not [`Ty::U32`] widened by hand.
    Usize,
    /// A scalar that rides the PRECEDING packed struct rather than a buffer of
    /// its own.
    ///
    /// `RowGatherParams` packs width and count into one buffer and there is no
    /// second: the count is the struct's second FIELD. A row lists it so the
    /// driver knows to supply the value, and this says "append it to the
    /// scalars, bind nothing" — a packed slot's run already covers every
    /// scalar after it, so the field lands where the struct expects it.
    InPacked,
    /// A host scalar.
    F32,
    /// A host flag. Spelled `bool` in C++, so it is ONE byte and not `i32`;
    /// a binding that gets this wrong is a silent stack-layout bug rather
    /// than a compile error, which is why it is its own kind.
    Bool,
    /// The stream the launch is ordered on.
    Stream,
    /// The cuBLAS handle a library-issued launch is ordered through.
    ///
    /// A launcher is "anything that issues device work", so the MLA absorb
    /// pair are launchers even though the work is `cublasGemmStridedBatchedEx`
    /// and not a kernel of ours. This is what they take instead of a stream —
    /// the stream is set on the handle.
    CublasHandle,

    // ---- The struct-shaped operands. ----
    //
    // Four passing modes, and the mode is the launcher's choice rather than
    // the row author's, so it is recorded here in `cpp()` rather than spelled
    // at each use. What separates them is what the RUST side can say:
    //
    //   by value, POD     a `#[repr(C)]` mirror, its layout proven by
    //                     `emit_layout_assertions`
    //   by const ref/ptr, POD    the same mirror, behind a `*const`
    //   by const ref, INCOMPLETE the C++ never defines the type, so Rust
    //                     has nothing to mirror and gets `*const c_void`
    //
    // A row cannot pick the wrong one by accident: the shim initialises a
    // function pointer, and a function pointer takes no conversions.
    /// The attention scratch, by value — [`Ty::Usize`]-sized buffers the
    /// driver owns and the kernels only read out of. Five words, so it is
    /// cheaper to copy than to chase.
    AttentionWorkspaceView,
    /// One layer's paged KV, by value.
    KvCacheLayerView,
    /// One layer's paged MLA cache, by value.
    MlaCacheLayerView,
    /// FlashInfer's decode plan, by `const&`. **Incomplete** in the header —
    /// `struct DecodePlanCache;` and nothing more — so this is a handle the
    /// driver holds and hands back, never a layout.
    DecodePlanCache,
    /// FlashInfer's prefill plan, by `const&`. Incomplete, as above.
    PrefillPlanCache,
    /// The MLA plan, by `const&`. Incomplete, as above.
    MlaPlanCache,
    /// The sm90 prefill schedule, by `const&`. Unlike the plan caches this
    /// one IS defined — a POD of offsets and extents — so Rust can mirror it.
    HopperPrefillPlan,
    /// Original-YaRN scaling, by `const*`. POD, and a pointer rather than a
    /// reference because it is optional: see `nullable`.
    YarnOriginalParams,
    /// A read-only device array of `attn::StructuredMaskParams` — the
    /// per-lane structured-mask descriptors `attn::pack_structured_mask`
    /// reads. POD (three `u32`s), so Rust mirrors it and the array crosses
    /// as `*const StructuredMaskParams`.
    ///
    /// Unlike the descriptor kinds this file lost, nothing about it is a
    /// ROUTE: the packer reads every lane's kind and window the same way,
    /// and no caller is choosing a kernel by what it finds inside. It is
    /// an operand shaped like a struct, which is why it survives where
    /// `WeightView` did not.
    StructuredMasks,
}

impl Ty {
    /// How C++ spells this, for a generated declaration.
    ///
    /// The pointer kinds are spelled with the fixed-width `std::` names the
    /// headers use, not `int`/`unsigned`, so the generated text is the same
    /// text a reader finds in the header it is checked against.
    pub const fn cpp(self) -> &'static str {
        match self {
            // No C spelling: it is a FIELD of the preceding struct, so the
            // header already names it there.
            Ty::InPacked => "::std::uint32_t",
            Ty::BufMut => "void*",
            Ty::Buf => "const void*",
            Ty::I32s => "const ::std::int32_t*",
            Ty::I64s => "const ::std::int64_t*",
            Ty::BufArray => "const void* const*",
            Ty::BufArrayMut => "void* const*",
            Ty::BufArrayOut => "const void**",
            Ty::BufArrayOutMut => "void**",
            Ty::U8Array => "const ::std::uint8_t* const*",
            Ty::CustomAllReduce => "::pie_cuda_driver::kernels::comm::CustomAllReduce*",
            Ty::I8s => "const ::std::int8_t*",
            Ty::I8sMut => "::std::int8_t*",
            // NOT `::std::uint16_t`, and the whole point of these two kinds
            // is that they are not interchangeable with it or with each
            // other. The prelude's `bf16` and `f16` are one-member structs,
            // so `const bf16*` where `const f16*` is meant is a pointer
            // conversion C++ refuses -- which is what makes
            // `abi::emit_device_typecheck`'s function-pointer initialisation
            // catch a swapped numeric format instead of reinterpreting it.
            //
            // Fully qualified, like `Ty::MoeActivation` and
            // `Ty::Mxfp4RowSelect`: the generated file states the whole path
            // rather than relying on where it happens to be included.
            Ty::Bf16s => "const ::pie_cuda_driver::kernels::device::bf16*",
            Ty::F16s => "const ::pie_cuda_driver::kernels::device::f16*",
            Ty::I32Array => "const ::std::int32_t* const*",
            Ty::MoeActivation => "::pie_cuda_driver::kernels::moe::MoeActivation",
            Ty::Mxfp4RowSelect => "::pie_cuda_driver::kernels::quant::Mxfp4RowSelect",
            Ty::U16s => "const ::std::uint16_t*",
            Ty::U16sMut => "::std::uint16_t*",
            Ty::Dtype => "::pie_cuda_driver::DType",
            Ty::KvScheme => "::pie_cuda_driver::kernels::attn::device::KvScheme",
            Ty::KvDType => "::pie_cuda_driver::kernels::attn::device::KvDType",
            Ty::I64 => "long long",
            Ty::U32s => "const ::std::uint32_t*",
            Ty::U8s => "const ::std::uint8_t*",
            Ty::F32sMut => "float*",
            Ty::F32s => "const float*",
            Ty::I32sMut => "::std::int32_t*",
            Ty::U32sMut => "::std::uint32_t*",
            Ty::U8sMut => "::std::uint8_t*",
            Ty::I32 => "int",
            Ty::U32 => "::std::uint32_t",
            Ty::Usize => "::std::size_t",
            Ty::F32 => "float",
            Ty::Bool => "bool",
            Ty::Stream => "cudaStream_t",
            Ty::CublasHandle => "cublasHandle_t",
            Ty::AttentionWorkspaceView => "::pie_cuda_driver::AttentionWorkspaceView",
            Ty::KvCacheLayerView => "::pie_cuda_driver::KvCacheLayerView",
            Ty::MlaCacheLayerView => "::pie_cuda_driver::MlaCacheLayerView",
            Ty::DecodePlanCache => "const ::pie_cuda_driver::kernels::attn::DecodePlanCache&",
            Ty::PrefillPlanCache => "const ::pie_cuda_driver::kernels::attn::PrefillPlanCache&",
            Ty::MlaPlanCache => "const ::pie_cuda_driver::kernels::attn::MlaPlanCache&",
            Ty::HopperPrefillPlan => "const ::pie_cuda_driver::kernels::attn::HopperPrefillPlan&",
            Ty::YarnOriginalParams => "const ::pie_cuda_driver::kernels::attn::YarnOriginalParams*",
            Ty::StructuredMasks => "const ::pie_cuda_driver::kernels::attn::StructuredMaskParams*",
        }
    }

    /// How Rust spells this on an `extern "C"` declaration.
    ///
    /// `Stream` lands as a plain opaque pointer rather than any driver type:
    /// `cudaStream_t` is `CUstream_st*` and `CUstream` is the same pointer,
    /// so a driver-side binding can pass its own handle without a conversion,
    /// and this crate does not have to name either API to say so.
    pub const fn rust(self) -> &'static str {
        match self {
            Ty::InPacked => "u32",
            Ty::BufMut => "*mut ::core::ffi::c_void",
            Ty::Buf => "*const ::core::ffi::c_void",
            Ty::I32s => "*const i32",
            Ty::I64s => "*const i64",
            Ty::BufArray => "*const *const ::core::ffi::c_void",
            Ty::BufArrayMut => "*const *mut ::core::ffi::c_void",
            Ty::BufArrayOut => "*mut *const ::core::ffi::c_void",
            Ty::BufArrayOutMut => "*mut *mut ::core::ffi::c_void",
            Ty::U8Array => "*const *const u8",
            Ty::CustomAllReduce => "*mut ::core::ffi::c_void",
            Ty::I8s => "*const i8",
            Ty::I8sMut => "*mut i8",
            // THE WIDTH, AND ONLY THE WIDTH -- deliberately the same
            // spelling `U16s` gets, because Rust has no bf16 and no f16 and
            // inventing a newtype here would be a mirror this crate would
            // then have to own (`needs_mirror`). The format's identity is
            // checked where it is checkable: in the C++, against the
            // instantiation the row names. A raw pointer that claimed
            // otherwise would advertise a check no `extern "C"` performs.
            Ty::Bf16s | Ty::F16s => "*const u16",
            Ty::I32Array => "*const *const i32",
            Ty::MoeActivation => "u32",
            Ty::Mxfp4RowSelect => "i32",
            Ty::U16s => "*const u16",
            Ty::U16sMut => "*mut u16",
            Ty::Dtype => "u8",
            // A `u8`, like `Ty::Dtype`, and for its reason: the enum's
            // underlying type is stated in the C++ (`: ::std::uint8_t`), so
            // the crossing is a byte and no mirror is owed. `needs_mirror`
            // reads that off this spelling rather than off a list.
            Ty::KvScheme | Ty::KvDType => "u8",
            Ty::I64 => "::core::ffi::c_longlong",
            Ty::U32s => "*const u32",
            Ty::U8s => "*const u8",
            Ty::F32sMut => "*mut f32",
            Ty::F32s => "*const f32",
            Ty::I32sMut => "*mut i32",
            Ty::U32sMut => "*mut u32",
            Ty::U8sMut => "*mut u8",
            Ty::I32 => "::core::ffi::c_int",
            Ty::U32 => "u32",
            Ty::Usize => "usize",
            Ty::F32 => "f32",
            Ty::Bool => "bool",
            Ty::Stream | Ty::CublasHandle => "*mut ::core::ffi::c_void",
            // Unqualified on purpose: a generated binding is placed in a
            // module that has the mirrors in scope, and this crate does not
            // know — and must not have to know — which module that is.
            Ty::AttentionWorkspaceView => "AttentionWorkspaceView",
            Ty::KvCacheLayerView => "KvCacheLayerView",
            Ty::MlaCacheLayerView => "MlaCacheLayerView",
            // Incomplete in the C++, so there is no layout to mirror and the
            // honest Rust type is the pointer a `const&` already is.
            Ty::DecodePlanCache | Ty::PrefillPlanCache | Ty::MlaPlanCache => {
                "*const ::core::ffi::c_void"
            }
            Ty::HopperPrefillPlan => "*const HopperPrefillPlan",
            Ty::YarnOriginalParams => "*const YarnOriginalParams",
            Ty::StructuredMasks => "*const StructuredMaskParams",
        }
    }

    /// Whether [`rust`](Self::rust) names a type the generated declaration
    /// does not itself define.
    ///
    /// The six below spell an UNQUALIFIED `#[repr(C)]` mirror, so a binding
    /// using one only compiles inside a module that has that mirror in scope.
    /// Everything else lands as a primitive or a raw pointer, which any crate
    /// can state without owning a layout.
    ///
    /// The distinction is what lets a row be callable by more than the crate
    /// that owns the mirrors: `Mxfp4RowSelect`, `MoeActivation` and `Dtype`
    /// look like struct kinds and are not — they cross as `i32`/`u32`/`u8` —
    /// so a row using them stays portable. Reading the answer off `rust()`'s
    /// own spelling rather than off a hand-kept list is what keeps the two
    /// from drifting when a kind changes how it crosses.
    #[must_use]
    pub const fn needs_mirror(self) -> bool {
        matches!(
            self,
            Ty::AttentionWorkspaceView
                | Ty::KvCacheLayerView
                | Ty::MlaCacheLayerView
                | Ty::HopperPrefillPlan
                | Ty::YarnOriginalParams
                | Ty::StructuredMasks
        )
    }
}

/// One operand of a launcher, in the position the launcher takes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operand {
    /// The C++ parameter's name. Prose — nothing matches on it — but it is
    /// what makes a row diffable against the header by eye, and what a
    /// generated binding names its argument.
    pub name: &'static str,
    /// What has to be placed here.
    pub ty: Ty,
    /// The launcher accepts a null here and means something by it.
    ///
    /// Not checkable by the C++ compiler — every pointer accepts null — which
    /// is exactly why it belongs in the table. `rope_write_kv_bf16`'s
    /// `row_valid` says "may be null" in a comment today, and a comment is
    /// not something a binding can be generated from.
    pub nullable: bool,
    /// WHERE the value comes from when a driver binds this slot.
    ///
    /// The signature says what TYPE goes here and the row's arity says
    /// how many; neither says that `q` is the statement's first OUTPUT
    /// and `positions` is the fire's. That correspondence is what every
    /// hand-written arm encodes, one arm at a time, and it is the last
    /// thing standing between a table that describes a call and a table
    /// a call can be GENERATED from.
    ///
    /// [`Source::Unbound`] — the default — means the row has not said,
    /// and a generator skips it exactly as it skips a row with no
    /// operands. Filling it is per-row work like the signature was.
    pub source: Source,
}

/// Where a bound argument comes from.
///
/// The vocabulary is deliberately small and describes the STATEMENT and
/// the FIRE, never a family: an operand that could only be sourced from
/// a workspace field is an operand whose arm is not shareable, which is
/// the same boundary `ExecCtx` draws.
/// `PartialEq` but not `Eq`: a [`Lit::F32`] is a float, and a float is
/// not a total order. Nothing here is a map key — the comparisons are
/// all "is this operand sourced from X", which partial equality answers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// The row has not stated it; nothing may be generated from this.
    Unbound,
    /// The statement's `i`-th operand, as a device pointer.
    In(u8),
    /// The statement's `i`-th result, as a device pointer.
    Out(u8),
    /// The `i`-th weight the statement NAMES, resolved through the
    /// binder.
    Weight(u8),
    /// The weight the statement names on `spec.weight` — a NAME the
    /// binder resolves, not a slot in the argument run.
    ///
    /// The other spelling, and both are live. A weight reaches a launch
    /// two ways: as an operand the trace placed in the run, which is
    /// [`Source::Weight`], or as the statement's own named weight, which
    /// is this. `layout::embed_bf16` only ever has the second — a vocab
    /// table is not something a trace produces — and it is the first
    /// launch of every fire.
    ///
    /// The cost this admits is that a generated branch now needs the
    /// RESOLVER, which is the one thing it had been able to do without.
    /// The resolve happens once before the match rather than inside a
    /// branch, so the guard can test it: a name the store lacks is
    /// DRIFT, and declining to a hand arm that will say
    /// `UnknownWeight` is a better answer than binding null.
    WeightNamed,
    /// The `i`-th scalar the statement carries (`Launch`'s params).
    Param(u8),
    /// The KEY pages of the layer this statement runs in.
    ///
    /// A statement writing or reading the KV cache names it as STATE
    /// (`StateRef { store: KvCache, layer }`) and not as an operand, because
    /// the cache outlives the fire and the trace has no value for it. So the
    /// pointer cannot come from the statement's args, and every backend has
    /// had to know which of its own buffers to bind — one hand-written arm at
    /// a time, which is the thing this table exists to end.
    ///
    /// The layer is the statement's own; a rolled trace states a span and an
    /// unrolled one states a layer, and both reach the same lookup.
    KvKeys,
    /// The VALUE pages of the layer this statement runs in. See
    /// [`Source::KvKeys`].
    KvValues,
    /// The token ids this fire gathers embeddings for.
    TokenIds,
    /// Which request owns each token — the paged attention's causal bound.
    RequestOfToken,
    /// The fire's KV page translation, and the CSR that partitions it by
    /// request.
    KvPageIndices,
    /// See [`Source::KvPageIndices`].
    KvPageIndptr,
    /// The custom attention mask, and the per-lane byte saying whether it
    /// applies.
    AttentionMask,
    /// See [`Source::AttentionMask`].
    AttentionMaskEnabled,
    /// Rows between one KV head's pages and the next, in ELEMENTS.
    ///
    /// The pool's own geometry, and the reason it is a `Source` rather than a
    /// scalar the text states: a stride is `max_ctx * head_dim` for the pool
    /// the DRIVER allocated. A model text cannot know it and should not guess
    /// — a text that guessed would be right for one deployment and silently
    /// wrong for the next, which is the failure this table exists to prevent.
    KvHeadStride,
    /// Rows between one token and the next within a head. See
    /// [`Source::KvHeadStride`].
    KvSeqStride,
    /// Token rows per page.
    KvPageSize,
    /// Per token: the physical page its KV row is written into.
    ///
    /// The paged append's destination, normalized from the ring position the
    /// frame states: `position / page_size`. Driver arithmetic over a driver
    /// allocation, which is why it is a table the resolver answers rather than
    /// anything a text could name.
    KvWritePage,
    /// Per token: the row within [`Source::KvWritePage`]'s page.
    KvWriteOffset,
    /// HOW MANY rows the fire samples, one per request.
    ///
    /// A number and not an address, so it rides the scalar channel beside the
    /// statement's own — the same shape as the KV pool's strides, and for the
    /// same reason: the driver knows it and no text can state it. `Lowered`
    /// publishes it because it is `rows` filtered two ways and maxed, not
    /// `rows.len()`.
    RequestCount,
    /// Which ROWS the fire samples, one per request.
    ///
    /// A prefill's readout is one distribution per request and its stream is
    /// one row per token, so something has to pick. `Step::sampling_indices`
    /// is the frame's answer and the row is where it is named.
    SamplingIndices,
    /// The rotary INVERSE FREQUENCIES, `[rotary_dims/2]` f32.
    ///
    /// A table rather than a base, because a base cannot express what a
    /// deployment does to it. llama-3 rescales its frequencies piecewise
    /// (`rope_type: llama3`), YaRN rescales them differently, and a text that
    /// stated `theta` would be right only for the deployments that leave the
    /// ladder alone.
    ///
    /// Derived at LOAD from the checkpoint's config, so it is the driver's to
    /// answer -- the same argument the KV pool's strides make for themselves.
    RopeFrequencies,
    /// The same slot read as a FLOAT.
    ///
    /// The param channel is untyped `u32` — "what each slot means is the
    /// SYMBOL's contract" — and a scale is a float. gemma-4 fires one
    /// kernel four times per fire with four different constants, all
    /// derived from dims the HOST knows, and the driver held a
    /// name-to-arithmetic table to get them back. The statement carries
    /// the number instead, in the bits the slot already has room for.
    ///
    /// WHAT IS STILL BLOCKED ON THIS, and it is one thing wearing three
    /// hats. Every row below is fully derivable EXCEPT for a load-time
    /// number the model-side FACTS do not carry:
    ///
    /// * gemma-4's per-layer rope theta (`rope_partial_bf16`,
    ///   `qk_rmsnorm_rope_bf16_rounded`) — the driver reads it off the
    ///   weights struct, which is loaded, not traced.
    /// * gemma-4's per-layer PLE scalar
    ///   (`rmsnorm_residual_add_scale_rmsnorm_bf16`) — the same.
    /// * gpt-oss's five yarn constants (`rope_yarn_original_bf16`) — in
    ///   the forward config, not in `GptOssFacts`.
    ///
    /// Each is one fixture field away, and the field is the work: a
    /// declaration states what the deployment IS, so the number has to
    /// be the deployment's real one. `dsl::cuda::rope_partial`'s note
    /// says why inventing them is worse than a driver reading a config
    /// value, and that judgment is what these rows are waiting on rather
    /// than any missing vocabulary.
    ParamF32(u8),
    /// The fire's POSITIONS, advanced to the rectangle's first row.
    ///
    /// `Ctx("positions")` would bind the fire's base, and a rectangle
    /// that starts partway in would then rotate its own rows against
    /// another rectangle's positions — the same defect `ArmCtx::row`
    /// exists for, on the one fire input that is token-rowed. So it gets
    /// the same treatment and its own source rather than a `Ctx` a
    /// reader has to notice is special.
    ///
    /// The DEVICE-WINDOW call forms are the exception and they do not
    /// use this: their kernel reads the split off a device word and
    /// wants the fire's positions unadvanced. Those are hand-written
    /// arms, which is where a per-call-form fact belongs.
    Positions,
    /// The rectangle's row count.
    ///
    /// The fire's, and only right for a statement whose rows ARE the
    /// fire's — which is most of them and not all. Where a value's own
    /// leading extent is the answer, [`Source::OutRows`] says so and
    /// covers this case too.
    Rows,
    /// The `i`-th result if the statement declares one, and the
    /// enclosing value-producing guard's value otherwise.
    ///
    /// A REGION LAUNCH declares no result: the guard owns the value and
    /// its arms bind it, so which value that is depends on where the
    /// statement sits rather than on what it says. qwen3.5's recurrence
    /// three-way and gpt-oss's attention chain are the same shape, and
    /// a row whose spellings appear both ways — the decode step states
    /// its result, the prefill spellings do not — cannot say `Out` and
    /// cannot say `Ctx`.
    ResultOrRegion(u8),
    /// The `i`-th FOREIGN value the join collected for this statement.
    ///
    /// nemotron's mamba block wires values ACROSS statements: the dt/dA
    /// prep and the scan consume the SPLIT's raw `dt` and the PARAMS
    /// prep's fp32 tables, none of which their own statements carry. The
    /// C++ hand pass routed them through its workspace; the Rust join
    /// collects them per layer into [`LaunchSpec::aux`], and this is how a
    /// row reaches one.
    ///
    /// Its own source rather than a use of [`Source::In`] because these
    /// are not the statement's operands -- the statement does not mention
    /// them, and their INDEX is the join's convention rather than the
    /// trace's.
    Aux(u8),
    /// The `i`-th result's LEADING extent, resolved for this fire.
    ///
    /// `Rows` for a token-shaped value, a constant for a fixed one, and
    /// for the MoE aligned path the padded block-major count — which is
    /// `Dim::MoeAlignedRoutes`, the one extent in the tree that is
    /// neither the fire's rows nor a load-time number. Five hand-written
    /// forwards restate the formula for it; a row that says `OutRows`
    /// gets it from the one place that computes it.
    OutRows(u8),
    /// The same for the `i`-th operand.
    InRows(u8),
    /// The trailing-dims product of the `i`-th result — what a row of
    /// it is worth in elements.
    OutWidth(u8),
    /// The same for the `i`-th operand.
    InWidth(u8),
    /// Rows times the `i`-th result's row width — the ELEMENT count a
    /// flat launcher takes where a row-shaped one takes both.
    OutElements(u8),
    /// The same for the `i`-th OPERAND.
    ///
    /// The MoE routing kernels want the ROUTE count, which is the fire's
    /// tokens times `top_k` — and `topk_idx` is `[Tokens, top_k]`, so it
    /// is exactly that operand's element count. A product the table has
    /// no arithmetic for, read off a value that already is it.
    InElements(u8),
    /// A field of the fire's ATTENTION context: the device-resident page
    /// CSRs, the write descriptors, the request count, the planned
    /// caches.
    ///
    /// The same shape as [`Source::Gdn`] and for the same reason — a
    /// different struct, and an OPTIONAL one, since a fire with no
    /// attention carries none. Thirteen hand arms open with
    /// `let a = attn.ok_or(NoAttnCtx)?;`, and the four that need ONLY
    /// this and [`Source::KvLayerView`] are the ones a row can reach.
    ///
    /// The other nine WERE blocked on vocabulary after all, and the
    /// three things they do have three names now: the arities are `Or`,
    /// the guard-owned output is `Or`, and the plan selection is
    /// [`Source::AttnPlan`]. What is left of "per-call form" is a
    /// statement of what the call needs, which is a row.
    Attn(&'static str),
    /// The layer's attention WINDOW: how far back a query may look.
    ///
    /// Not `CtxByLayer` because the fall-through is three deep — the
    /// statement's own first param, then the per-layer vector, then the
    /// fire's default — and a row has no business spelling that. `-1` is
    /// unbounded.
    AttnWindow,
    /// The planned cache the driver would use for THIS statement's layer.
    ///
    /// A ROW STATES WHAT IT NEEDS, NEVER HOW THE DRIVER FINDS IT, and
    /// this is the line between them. Two-kind families keep a second
    /// plan for their full-attention layers and the C++ picks with
    /// `cur_full ? decode_plan_full : decode_plan_sliding`; the rule is
    /// the driver's, because the driver owns `window_left_by_layer` and
    /// built both plans. The row says "the decode plan for my layer" and
    /// gets the one that is right for it.
    ///
    /// The argument is the plan FAMILY — `"decode"`, `"prefill"` — not a
    /// field name, so a row cannot name the sliding one specifically and
    /// then be wrong on a full layer.
    AttnPlan(&'static str),
    /// An attention-context field a fire leaves NULL to say "not
    /// published" — the write descriptors, which only a fire that
    /// computed them carries.
    ///
    /// [`Source::CtxNonZero`]'s test on the other struct, and the hand
    /// arm it replaces made exactly this check by hand and returned
    /// `NoAttnCtx` with a message saying so. A generated branch declines
    /// instead and the fallthrough reports, which is the same answer one
    /// layer earlier.
    AttnNonZero(&'static str),
    /// The KV cache view for the layer this statement runs in, BY VALUE.
    ///
    /// The reason `attn` is the largest hand-written block in the table.
    /// [`Source::KvKeys`] and [`Source::KvValues`] spell a cache as two
    /// device pointers, which is Metal's shape; CUDA's launchers take a
    /// `KvCacheLayerView` whole, so a CUDA row stating the pointer pair
    /// is one the emitter refuses rather than mis-bind. This is the
    /// spelling CUDA can answer.
    ///
    /// Indexed by the statement's own layer, like [`Source::CtxByLayer`],
    /// and guarded on the fire holding one there: a rolled trace states a
    /// span and an unrolled one states a layer, and both reach the same
    /// lookup.
    KvLayerView,
    /// A FIELD of that view — `head_dim`, `page_size`.
    ///
    /// The planless prefills divide the query width by the head dim to
    /// get a head count, and the head dim is a property of the CACHE
    /// rather than of the fire: two layer kinds may disagree on it,
    /// which is the whole reason gemma-4 keeps two decode plans.
    KvLayerField(&'static str),
    /// A field of the fire's GDN context, which is the hybrids' recurrent
    /// geometry: head counts, conv width, group count, slab strides.
    ///
    /// Its own source and not a `Ctx` because it comes from a DIFFERENT
    /// struct and an OPTIONAL one — a fire with no recurrent layers
    /// carries no GDN context at all, and a row reading one has to
    /// decline rather than read a default. Ten hand arms open with
    /// `let g = gdn_ctx()?;` for exactly that reason, which made this the
    /// largest single blocker in the table once the resolver landed.
    ///
    /// Like [`Source::CtxByLayer`], the name is the DRIVER's field and
    /// the generator's claim is only where to look.
    Gdn(&'static str),
    /// The STATEMENT'S OWN LAYER's entry in a per-layer GDN slab vector —
    /// the conv window or the recurrent state, as a device address.
    ///
    /// Its own source and not a use of [`Source::Gdn`] because the field
    /// is a `Vec<u64>` and what a kernel wants is ONE of its entries,
    /// chosen by the layer the statement is tagged with. Nine arms open
    /// with `slab(&g.conv_state, state_layer()?, "conv")?` for exactly
    /// that, and the three-way it guards is real: a fire may carry a GDN
    /// context, and that context may still hold no slab at this layer,
    /// and an op may state no layer at all. All three decline.
    GdnSlab(&'static str),
    /// The statement's SECOND named weight (`LaunchSpec::weight2`).
    ///
    /// The mirror of [`Source::WeightNamed`]. A statement that names two
    /// tensors by name — the GDN prep's `a_log` and `dt_bias` — has
    /// nowhere to put the second without this, which is the whole of why
    /// that row stayed hand-written.
    WeightNamed2,
    /// The statement's weight name plus a SUFFIX, or null.
    ///
    /// A conv or a norm whose checkpoint may or may not ship a bias: the
    /// tensor is `<weight>_bias`, and its absence is a property of the
    /// CHECKPOINT rather than drift, so null is the answer and not a
    /// refusal. That is the one thing distinguishing it from
    /// [`Source::WeightNamed`], whose absence IS drift and which declines
    /// the branch.
    WeightSuffix(&'static str),
    /// A context field the fire holds PER LAYER, read at the statement's
    /// own layer.
    ///
    /// Rope theta is the example that forces it: gemma-4 splits theta by
    /// layer kind, so `ctx.rope_theta` is right for a uniform family and
    /// wrong for that one, and six hand arms call a `theta_of(layer)`
    /// helper to say so. The field is the DRIVER's — the table has no
    /// business knowing whether a family's per-layer vector has a
    /// fallback, a filter or a refusal behind it — so this names an
    /// ACCESSOR the driver implements, exactly as [`Source::CtxNonZero`]
    /// names an `is_set` the driver implements. The generator's whole
    /// claim is "this value is indexed by the statement's layer", which
    /// is the part it can know.
    CtxByLayer(&'static str),
    /// Dimension `d` of the `i`-th operand. The routed combine reads
    /// `[Tokens, top_k, H]` and both extents come off it.
    InDim(u8, u8),
    /// Dimension `d` of the `i`-th result, which is how a head count
    /// reaches a launcher: the shape says `[Tokens, heads, dim]`.
    OutDim(u8, u8),
    /// The fire's ROUTE COUNT: rows times the `i`-th param.
    ///
    /// The MoE aligned path's `num_routes`, and the one product the
    /// table could not otherwise reach. `InElements` covers the case
    /// where a value already IS the routes (`topk_idx` is
    /// `[Tokens, top_k]`, so its element count is the answer), but the
    /// gather and the reorder take the PERMUTATION as their integer
    /// operand — `[max_blocks * block_size]`, the padded aligned extent —
    /// and neither their other operand nor their result carries the
    /// router's width. So the statement states `top_k` on the param
    /// channel and this reads rows times it.
    ///
    /// Deliberately narrow: it is a row's way of saying "per-token
    /// fan-out", not a general arithmetic escape hatch. A row wanting a
    /// different product should say what the product IS.
    RoutesOfParam(u8),
    /// A named field of the executing context — the stream, the handle,
    /// `eps`, the head geometry.
    ///
    /// The name is the FIELD's, and nothing else. It used to carry the
    /// C++ context object's struct nesting (`"arm.stream"`), which made
    /// this vocabulary speak one consumer's shape: every other consumer
    /// then had to strip a prefix it never had. A row names a fact; where
    /// that fact sits inside a driver's context is the driver's business.
    Ctx(&'static str),
    /// [`Source::Ctx`], plus a GUARD: the generated branch fires only
    /// when the field is non-zero, and a family that leaves it zero
    /// keeps its own arm.
    ///
    /// This exists because of one number. gemma-4 alternates its rope
    /// theta per layer, so the single `rope_theta` a context can carry
    /// is the wrong one for half that model, and the family says so by
    /// leaving the field zero — a convention the hand-written shared
    /// rope arm already had (`if (c.rope_theta == 0.f) return false;`).
    /// When the rope rows started generating, the generated branch ran
    /// FIRST and had no such refusal: it would have rotated half of
    /// gemma-4 by nothing, silently, past an arm written to prevent
    /// exactly that.
    ///
    /// So the refusal belongs to the ROW rather than to whichever arm
    /// happens to be reading the field. Zero means "not this family's",
    /// which is a claim about the context field and not about any one
    /// call site.
    CtxNonZero(&'static str),
    /// A literal VALUE — for the arguments a launcher takes that no
    /// statement and no context carries: an `interleaved` flag a family
    /// never sets, a `beta` of zero, an optional buffer left absent.
    ///
    /// A value, not a string. It was `&'static str` holding C++ source
    /// text (`"1.702f"`, `"nullptr"`, `"false"`), which meant every
    /// consumer that was not C++ had to PARSE another language's syntax
    /// to find out what number the row meant. `1.702f` is not a float in
    /// any language that reads this table; it was a float in the one
    /// language that wrote it.
    Lit(Lit),

    // ── The grammar ──────────────────────────────────────────────────
    //
    // Everything above is a LEAF: a fact about the statement, the fire
    // or the driver. Everything below COMBINES leaves, and the four of
    // them together replace thirteen variants that were the same three
    // operations over different leaves — `InWidthOver`, `OutWidthOver`,
    // `OutWidthOverIn`, `RowsTimesParam`, `GdnProduct`, `RowsTimesGdn`,
    // `InWidthOverGdn`, `InWidthIsqrt`, `InWidthOverOut`,
    // `OutWidthOverInWidth`, `InWidthOverInWidth`, `RowsPerHead`,
    // `WidthPerHead`.
    //
    // THE POINT IS NOT THE COUNT. It is that a flat enum of
    // combinations needs a new variant per row, so every kernel added
    // was a chance to edit the emitter — and the emitter's arity
    // computation is a hand-maintained `match` over those variants,
    // where a forgotten arm is a branch that declines silently. That is
    // exactly how `Source::Aux` came to emit code nothing could reach.
    // A grammar closes the table: a new row composes, and the arity
    // falls out of walking the tree.
    //
    // `&'static Source` rather than `Box`, because a kernel table is a
    // `const` and a const can hold a reference to another const.
    /// An operand's row width.
    Width(&'static Source),
    /// Product.
    Mul(&'static Source, &'static Source),
    /// Difference, floored at zero.
    ///
    /// FLOORED for `Div`'s reason: a bind expression runs after the
    /// guard and has nowhere to refuse from, and a negative head count
    /// is a shape the launcher rejects one layer lower anyway.
    Sub(&'static Source, &'static Source),
    /// Quotient, with the divisor floored at one.
    ///
    /// FLOORED, not checked, and the reason is the same one
    /// `InWidthOver` gave: a family that states no groups means one, and
    /// a division that refused would decline the family rather than
    /// serve it.
    Div(&'static Source, &'static Source),
    /// Exact integer square root, or `0` when the value is not a perfect
    /// square.
    ///
    /// Zero rather than a refusal, because bind expressions run after
    /// the guard and have nowhere to refuse from. The launcher rejects a
    /// zero, which is the same outcome one layer lower.
    Isqrt(&'static Source),
    /// Inequality, as a bool operand.
    Ne(&'static Source, &'static Source),

    /// The first source if it is PRESENT, the second otherwise.
    ///
    /// A source that DEGRADES rather than DEMANDS, which is how one row
    /// serves an arity family without the emitter knowing arities
    /// exist. `[x, y]` and `[x, y, w]` are two live spellings of
    /// rmsnorm; `Or(&Weight(0), &WeightNamed)` serves both.
    Or(&'static Source, &'static Source),
    /// `IfPresent(probe, then, else)` — `then` when `probe` resolves,
    /// `else` when it does not.
    ///
    /// The per-head reading is this: a statement carrying `PerHeadDim`
    /// norms `rows * (width / head_dim)` rows of `head_dim`, and one
    /// without it norms `rows` of `width`.
    IfPresent(&'static Source, &'static Source, &'static Source),
    /// The statement's per-head dim, when it states one.
    PerHeadDim,
    /// A scalar constant the statement names in its `scale.<name>` slot.
    ///
    /// A scale is a CONSTANT, not a tensor (the dsl's own words), so it
    /// resolves out of a table the driver built from the config rather
    /// than out of the weight store. Paired with `ParamF32` under an
    /// `Or`: a statement carrying the number rides the params, and one
    /// that does not names it.
    NamedScale,
    /// The ROTARY WIDTH for this statement — how many channels rotate.
    ///
    /// Three places it can come from, and the order prefers what a
    /// STATEMENT said: the launch's own param, then the semantic
    /// `Rope { partial }`, then the fire's per-layer table. The first two
    /// are one fact under two spellings and both are live — qwen3_5's
    /// prefill states the launch and its decode records the semantic op.
    ///
    /// A row may not spell that, for [`Source::AttnWindow`]'s reason: a
    /// fall-through is the driver's rule, and a row states what it needs.
    RotaryWidth,
    /// The layer's own scalar, or `1.0` where the layer states none.
    ///
    /// [`Source::NamedScale`]'s sibling and NOT the same thing: that one
    /// resolves a `scale.<name>` and refuses a miss, because the whole
    /// launch is the multiply. This one is one term of a fused norm, and
    /// a family whose landing carries no scalar means one — the C++
    /// reads `layer_scalar_value` the same way. A refusal here would
    /// decline every family but the one with a PLE.
    LayerScale,
    /// The accumulation coefficient the STATEMENT implies: `1.0` when it
    /// accumulates into its destination, `0.0` when it overwrites.
    ///
    /// One symbol serves both because the launcher takes beta as an
    /// argument — what differs is the statement, not the kernel, and
    /// `spec.beta_one` is where the lowering already wrote it down.
    Beta,
}

/// What a [`Source::Lit`] holds.
///
/// Deliberately small. A row states a constant the launcher needs and no
/// statement supplies; anything that wants arithmetic or a name is a
/// different `Source`, not a richer literal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lit {
    /// An absent optional pointer. Typed by the operand it binds — the
    /// row already says which pointer is missing.
    Null,
    /// A flag.
    Bool(bool),
    /// A float. `1.702` is gpt-oss's GLU alpha; the row carries the
    /// number, and each consumer spells it its own way.
    F32(f32),
    /// An integer.
    I32(i32),
}

/// One kernel's contract.
pub struct KernelSig {
    /// The dsl-side name (what a model text spells).
    pub name: &'static str,
    /// The C++ launcher symbol the trace records.
    pub symbol: &'static str,
    /// The source file that defines the entry point, relative to the backend's
    /// shader directory.
    ///
    /// Metal compiles at run time from `(path, entry name)`, so a symbol alone
    /// cannot be built — and the only things that knew the file were the
    /// hand-written per-family plans, which is one more fact a driver knew and
    /// a statement did not. `None` means the row has not said; a backend that
    /// links its kernels (CUDA) never needs it, because the linker knows.
    pub file: Option<&'static str>,
    /// How a rectangle becomes a thread grid. See [`LaunchRule`].
    pub launch: LaunchRule,
    /// The kernel REFUSES a row split: it may not be stated inside a peel's
    /// regions, because its addressing (a fire-wide prepare, a padded staging
    /// buffer) is not row-offsettable. `model-compiler`'s `OpKind::Peel` is
    /// the op this refuses, and its `check_plan` is what enforces the refusal.
    pub whole: bool,
    /// The host plan its contract obligates.
    pub needs: Prepare,
    /// Capabilities this kernel cannot serve — a seam asking for one of these
    /// over rows this kernel covers is unservable.
    pub lacks: &'static [Cap],
    /// Where a sink-writing seam's output lands, if this kernel accepts one
    /// (`sink pages -> kv.pages`).
    pub sink: Option<&'static str>,
    /// Which OUTPUTS this kernel writes over which INPUTS, as
    /// `(output index, input index)` pairs.
    ///
    /// `launch_residual_add_bf16(y, x, n)` writes its result over `y`, so
    /// its row says `in_place = &[(0, 0)]`. That is a fact about the
    /// KERNEL and not about any statement using it — every call of it is
    /// in-place — which is why it lives here rather than at the call
    /// site.
    ///
    /// A LIST because one pair cannot say it. `launch_rope_bf16(q, k,
    /// ...)` rotates both tensors where they lie, which is two
    /// independent aliases, and the single-index form this field used to
    /// have could express neither of them together nor either of them
    /// honestly.
    ///
    /// `lower::Buffers` is what reads it: an in-place op's output takes
    /// its operand's OFFSET instead of an allocation of its own. Without
    /// that, a text accumulating into a WINDOW (gemma3n's per-layer
    /// embedding, added back into K-1 corrected AltUp streams) would
    /// produce fresh values nothing downstream reads, and the streams
    /// would silently stay pre-update — and an executor that binds
    /// operands from the trace hands one pointer to a kernel that means
    /// to read and write it, having written nothing there.
    pub in_place: &'static [(u32, u32)],
    /// On a union tail layer this dispatch pairs the DEPTH PREFIX plan (and
    /// its dedicated workspace) instead of the fire's own plan.
    ///
    /// This was the `PrefixPlanSwap` half of the retired per-op `DepthRole` —
    /// a word the IR carried on one launch per layer of every depth-declaring
    /// trace, restating a fact about the KERNEL. Migration step 5 moved it
    /// here.
    pub depth_prefix_plan: bool,
    /// Which of this kernel's OUTPUTS fill which of the layer's AUX slots,
    /// as `(aux slot, output index)` pairs.
    ///
    /// The PUBLISHER half of [`Source::Aux`], which is its consumer half.
    /// A block that wires values across statements has two ends, and only
    /// one of them was ever stated: a row could say "my third operand is
    /// the join's `Aux(3)`", but nothing could say WHICH kernel's output
    /// put it there. So the join hard-coded four kernel names —
    /// `ssm::nemotron_mamba_split_bf16` and its three neighbours — inside
    /// `driver-cuda`, which made a backend the only place that knew how one
    /// architecture's block is wired.
    ///
    /// It is a fact about the KERNEL, for the same reason [`Self::in_place`]
    /// is: every call of the split publishes its raw `dt`, at the same slot,
    /// whatever statement made the call. Stating it here lets the join be
    /// arithmetic over rows — collect what publishers publish, hand it to
    /// consumers that ask — with no kernel named on the driver side at all.
    ///
    /// The slot numbering is the block's own convention and is shared with
    /// the [`Source::Aux`] indices that read it; for the mamba block that
    /// order is `[dt_raw, a, d, dt_bias, dt_pre, da_pre]`.
    ///
    /// An out-of-range output index publishes nothing rather than panicking:
    /// a statement with fewer results than the row expects is a trace that
    /// does not fit this kernel, and the arity guard is what reports it.
    pub publishes_aux: &'static [(u8, u8)],
    /// The operands `symbol` takes, in order — the launch ABI, as data.
    ///
    /// Empty means UNSTATED, not "takes nothing": a launcher that genuinely
    /// took no operands would still take a stream. Rows are being filled in a
    /// family at a time (`rope` is the pilot), so an empty list is how a row
    /// says it has not been done yet, and nothing may infer a nullary call
    /// from one.
    ///
    /// Stating this is what lets the DECLARATION be generated from the row
    /// instead of written beside it. That turns the crate's own invariant —
    /// every symbol resolves to exactly one declaration, every declaration
    /// has exactly one row — from a check into a tautology, the same way
    /// deriving `symbol` from the module path did for names. It is also the
    /// only way this contract can be PROVEN: a generated shim that calls the
    /// real function makes a wrong row a C++ compile error, so the compiler
    /// is the oracle and no golden can drift.
    ///
    /// Default arguments are not representable and that is deliberate. A row
    /// lists every operand the callee has, defaulted or not, because a
    /// caller that is not C++ cannot omit one — and because a default is a
    /// choice the table should be able to see.
    /// The name a LOWERING gives this kernel, where it differs.
    ///
    /// A portable lowering names an operation and a backend names a
    /// symbol, and they are not always the same word: the lowering says
    /// `gemm::act_x_w` and CUDA's is `gemm::act_x_wt_bf16`, which is
    /// what `gemm.hpp` defines as `act_x_w` with `WeightView::raw(W,
    /// BF16)` — the one view the dense path ever built. Metal binds the
    /// same lowering to its own.
    ///
    /// So the row keeps the SYMBOL as its identity (the shim, the
    /// audit and the ABI are all built from it) and says here which
    /// lowering it answers to. The alternative was renaming in the
    /// lowering, which would have told Metal a CUDA word.
    pub lowered_as: Option<&'static str>,
    pub operands: &'static [Operand],
    /// What the launcher RETURNS, spelled as C++ spells it.
    ///
    /// `""` — the default — means `void`, which is what a launcher is
    /// nearly always. Three were not: `gemv3_bf16`, `rmsnorm_bf16_tuned`
    /// and `lm_head_argmax_chunked` returned `bool`, and the bool meant
    /// "did the fused/tuned form run" rather than "did it succeed".
    /// `new-horizon.md` §43 deleted `rmsnorm_bf16_tuned` — nothing in any
    /// language reached it — so the tree carries two of the three today.
    /// The field stays: the reason below is about the shim's types, not
    /// about how many rows currently exercise them.
    ///
    /// It is on the ROW rather than inferred because the shim has to
    /// declare the forwarding pointer's full type: a `void` forward to a
    /// `bool` launcher is a conversion C++ refuses, which is how these
    /// three were found. Stating it is also the honest reading — a
    /// launcher that answers something is a different contract from one
    /// that only acts, and a table that could not tell them apart was
    /// describing the second and meaning the first.
    ///
    /// The generated entry point returns the same type; a caller that
    /// ignores it is doing what the C++ call sites already do.
    pub returns: &'static str,
    /// The axes `symbol` is instantiated over, if it names a FAMILY of
    /// entrypoints rather than one.
    ///
    /// Empty is the CUDA case and the default: a launcher there is an authored
    /// C++ function, so one row is one symbol and [`sig_in`] matches it whole.
    ///
    /// Metal's are generated. `quantized_qmm_t.metal` holds one template body
    /// and a macro that stamps it over `(group, bits) × (bm, bn)`, so `54` of
    /// its entrypoints are one kernel evaluated at 54 points. Enumerating them
    /// as 54 rows would state the macro's job a second time, by hand, and
    /// `.wiki/kernel-refactor.md` §5's own rule — *would the two share one C++
    /// definition?* — says they are not distinct kernels. So the row is the
    /// base and the axes are declared beside it.
    ///
    /// This is not a Metal-only idea. CUDA writes the same product into
    /// FILENAMES (`attn/flashinfer_hd{64,128,256,512}.cu`,
    /// `attn/xqa_gqa{2,4,8}.cu`) and cannot state it, because each of those is
    /// separately authored. When that changes, the axis is already spelled
    /// here.
    pub axes: &'static [Axis],
    /// Which stated scalar carries this row's GRID extent, when the fire's
    /// geometry cannot answer it.
    ///
    /// A rule evaluates against `Dims`, which a driver fills from the FIRE —
    /// one `rotary_dims`, one `head_dim`, one of everything. That is right
    /// until a deployment states the number PER LAYER, and gemma-4 does:
    /// `partial_rotary_factor: 0.25` on its full-attention layers means they
    /// rotate 128 of their 512 channels while its sliding layers rotate all
    /// 256 of theirs. One fire-wide `rotary_dims` cannot be both, and
    /// rotating the wrong count returns fluent text rather than failing.
    ///
    /// So the row says where to look instead: `Some(i)` means *"my rule's
    /// extent is the statement's param `i`, not the fire's geometry"*. The
    /// scalar rides the channel that already exists — `Launch::params` — and
    /// the only new thing is that the driver may READ one for the grid rather
    /// than only forwarding it to the kernel.
    ///
    /// Stated on the row rather than matched on in the driver deliberately.
    /// A `if rule == Rope { ... }` arm beside the dispatcher is the shape
    /// this family's north star calls a bug report against a table row; this
    /// is the row answering.
    pub grid_param: Option<u8>,
    /// Where this row's HEAD WIDTH is stated, when the fire's is not it.
    ///
    /// [`Self::grid_param`]'s sibling, and it exists because the same
    /// deployment breaks the same assumption twice. gemma-4's full-attention
    /// layers have 512-wide heads and its sliding layers 256-wide ones, so a
    /// fire-wide `head_dim` is wrong for one of them whichever it holds.
    ///
    /// A rope's row already STATES its head width -- `head_dim` is one of its
    /// params, and the kernel reads the statement's number, not the fire's.
    /// The grid did not: it divided the tensor's width by the FIRE's
    /// `head_dim` to count heads. Two numbers for one quantity, and the
    /// kernel multiplies them back together to find a row:
    /// `row_base = m * n_head * head_dim`. When they disagree by two, every
    /// row after the first is written two rows along and most of them land
    /// past the tensor -- a rotation that silently applies to almost nothing.
    ///
    /// `Some(i)` means *"my head width is the statement's param `i`"*.
    pub head_param: Option<u8>,
    /// Where this row's HEAD COUNT is stated, when the fire's is not it.
    ///
    /// The third of the same family, and the one that shows a deployment
    /// states a head SHAPE rather than a head width: gemma-4's full-attention
    /// layers carry four KV heads of 512 channels where its sliding layers
    /// carry sixteen of 256. A rule that spans heads needs both numbers, and
    /// taking either from the fire is taking half of one layer's shape and
    /// half of another's.
    ///
    /// Measured on gemma-4-31b. `kv_append_paged` is told `head_dim` and
    /// `n_kv_heads` by its own params -- the kernel addresses the pool with
    /// the statement's numbers -- while [`LaunchRule::PerHead`] built the grid
    /// from the fire's. On a full-attention layer the kernel wrote at
    /// `(slot * 4 + head) * 512 + channel` under a grid of `[256, 16]`, so
    /// **channels 256..511 of every KV head were never written** and heads
    /// 4..15 landed in the next token's rows. The gap does not fail: those
    /// channels read back as the zeros the pool was born with, attention
    /// returns a value whose second half is zero, and the fire completes.
    ///
    /// `Some(i)` means *"count my heads by the statement's param `i`"*.
    pub heads_param: Option<u8>,
}

impl KernelSig {
    /// Does `symbol` name this row at one point of its axes?
    ///
    /// Order matters and is the whole implementation: the axes are declared in
    /// the order a name spells them, so this peels suffixes from the END, one
    /// axis at a time, and what must remain is the base. That refuses
    /// `qmm_t_bfloat16_gs_64_b_4` (a `bm`/`bn` short of a real entrypoint) and
    /// refuses a permuted spelling, both of which a "contains all the points"
    /// test would wave through.
    /// Does `symbol` name this row — as the kernel itself, or at one point of
    /// its axes?
    ///
    /// Both are legitimate and they come from different places. **A model text
    /// states the KERNEL**, because the axis point is a deployment fact: which
    /// affine format a checkpoint is, how wide its heads are. `dsl::metal`
    /// records `affine_qmv_fast` and the driver resolves
    /// `affine_qmv_fast_bfloat16_gs_64_b_4` at load, from `AffineFormat`. **The
    /// driver and the audit name the POINT**, because that is what a pipeline
    /// is built from.
    ///
    /// So the base resolves, and so does every point. What does not resolve is
    /// anything between them: [`Self::covers_point`] peels the axes from the
    /// END in declaration order, so a half-spelled name is refused rather than
    /// rounded to the nearest row.
    pub fn covers(&self, symbol: &str) -> bool {
        self.symbol == symbol || self.covers_point(symbol)
    }

    /// `symbol` is this row at one point of its axes — not the bare base.
    ///
    /// Order is the whole implementation: the axes are declared in the order a
    /// name spells them, so this peels suffixes from the end, one axis at a
    /// time, and what must remain is the base. That refuses
    /// `qmm_t_bfloat16_gs_64_b_4` (a tile short of a real entrypoint) and
    /// refuses a permuted spelling, both of which a "contains all the points"
    /// test would wave through.
    pub fn covers_point(&self, symbol: &str) -> bool {
        if self.axes.is_empty() {
            return false;
        }
        let mut rest = symbol;
        for axis in self.axes.iter().rev() {
            match axis
                .points
                .iter()
                .find(|point| rest.len() > point.len() && rest.ends_with(**point))
            {
                Some(point) => rest = &rest[..rest.len() - point.len()],
                None => return false,
            }
        }
        rest == self.symbol
    }

    /// Every entrypoint this row names: the product of its axes, appended in
    /// declaration order. One element (the symbol itself) when there are none.
    ///
    /// This is the other half of [`KernelSig::covers`], and the reason both
    /// exist: `covers` answers "is this name mine", `entrypoints` answers
    /// "what are all of mine", and `scripts/metal-kernel-audit.py` compares
    /// the second against the shader tree. A row that generates a name no
    /// shader instantiates, or misses one that exists, fails there — which is
    /// the invariant `.wiki/kernel-metal-refactor.md` §6 (1) states.
    pub fn entrypoints(&self) -> Vec<String> {
        let mut out = vec![self.symbol.to_string()];
        for axis in self.axes {
            out = out
                .iter()
                .flat_map(|stem| {
                    axis.points
                        .iter()
                        .map(move |point| format!("{stem}{point}"))
                })
                .collect();
        }
        out
    }
}

/// Declare one kernel. The syntax is `.wiki/tart/dsl.md` ②'s.
///
/// Operand shapes used to be excluded from this on the grounds that they
/// stayed with the emitter until the launch ABI flattened, and that stating
/// them here would duplicate it. They are admitted now, through
/// [`KernelSig::operands`] and the [`operands!`] macro, because flattening
/// the ABI is precisely what a stated operand list DOES: once the row carries
/// the signature, the C++ declaration and every non-C++ binding are generated
/// from it, and the emitter reads the row rather than knowing a second copy.
///
/// Exported so the two backend tables can declare rows in the same words. It
/// names [`KernelSig`], [`Prepare`] and [`Cap`] through `$crate`, so a table
/// crate needs no `use` beyond the macro itself.
#[macro_export]
macro_rules! kernel {
    ($name:ident $symbol:literal $(, $key:ident = $value:expr)* $(,)?) => {
        $crate::KernelSig {
            name: stringify!($name),
            symbol: $symbol,
            $($key: $value,)*
            ..$crate::KernelSig {
                name: "",
                symbol: "",
                file: None,
                launch: $crate::LaunchRule::Unstated,
                whole: false,
                needs: $crate::Prepare::None,
                lacks: &[],
                sink: None,
                in_place: &[],
                depth_prefix_plan: false,
                publishes_aux: &[],
                operands: &[],
                returns: "",
                axes: &[],
                grid_param: None,
                head_param: None,
                heads_param: None,
                lowered_as: None,
            }
        }
    };
}

/// An operand list, spelled the way the C++ declaration reads.
///
/// `name: Ty`, in the callee's parameter order, with `| null` on the ones
/// that accept an absent pointer:
///
/// ```ignore
/// operands![
///     q: BufMut, k: BufMut,
///     positions: I32s,
///     row_valid: U8s | null,
///     num_tokens: I32, theta: F32,
///     stream: Stream,
/// ]
/// ```
///
/// `| null` rather than a `?` suffix on purpose: `?` is a token a `tt` would
/// swallow ahead of the `,` separating two operands, and the arm would then
/// depend on macro lookahead rather than on anything a reader can see.
#[macro_export]
macro_rules! operands {
    ($($name:ident : $ty:ident $(| $null:ident)? $(<- $src:expr)?),* $(,)?) => {
        &[$($crate::Operand {
            name: stringify!($name),
            ty: $crate::Ty::$ty,
            nullable: $crate::operands!(@nullable $($null)?),
            source: $crate::operands!(@source $($src)?),
        }),*]
    };
    (@source) => { $crate::Source::Unbound };
    (@source $src:expr) => { $src };
    (@nullable) => { false };
    (@nullable null) => { true };
}

/// The contract for one symbol, in `table`.
///
/// A linear scan: the tables are ~100 and ~90 rows, and the call sites are
/// load-time (a declaration is traced when the model loads), not per-fire.
///
/// Exact matches on the symbol are tried first and across the WHOLE table,
/// before any row is allowed to claim `symbol` as a point of its axes. Without
/// that two-pass order a row could swallow a sibling whose base happens to end
/// in one of its points, and which row won would depend on declaration order.
///
/// CUDA's rows carry no axes, so for them the second pass never fires and this
/// is the same linear scan it always was.
pub fn sig_in(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    table
        .iter()
        .find(|k| k.symbol == symbol)
        .or_else(|| table.iter().find(|k| k.covers_point(symbol)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const AFFINE: Axis = Axis {
        what: "affine group and width",
        points: &[
            "_gs_32_b_4",
            "_gs_64_b_4",
            "_gs_128_b_4",
            "_gs_32_b_8",
            "_gs_64_b_8",
            "_gs_128_b_8",
        ],
    };
    const TILE: Axis = Axis {
        what: "routed GEMM tile",
        points: &["_bm_16_bn_16", "_bm_32_bn_32", "_bm_64_bn_64"],
    };
    const DTYPE: Axis = Axis {
        what: "activation dtype",
        points: &["_bfloat16"],
    };

    static TABLE: &[KernelSig] = &[
        kernel!(qmv "affine_qmv_fast", axes = &[DTYPE, AFFINE]),
        kernel!(qmm_t "affine_qmm_t", axes = &[DTYPE, AFFINE, TILE]),
        // A base that is ALSO a legal entrypoint, next to its dtyped form.
        kernel!(route_sort "moe_route_sort"),
        kernel!(router "router_topk", axes = &[DTYPE]),
    ];

    fn named(symbol: &str) -> Option<&'static str> {
        sig_in(TABLE, symbol).map(|k| k.name)
    }

    #[test]
    fn a_row_covers_every_point_of_its_axes() {
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_64_b_4"), Some("qmv"));
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_128_b_8"), Some("qmv"));
        assert_eq!(
            named("affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_64"),
            Some("qmm_t")
        );
    }

    /// The axes are peeled from the END in declaration order, so a name that
    /// stops short of a full point set is NOT covered. This is the case a
    /// "contains all the points" test would wave through, and it is exactly
    /// the shape of the bug the table exists to catch: `decode_psos.cpp`
    /// building `"affine_qmm_t" + q` and forgetting the tile.
    #[test]
    fn a_partial_or_permuted_spelling_is_refused() {
        assert_eq!(named("affine_qmm_t_bfloat16_gs_64_b_4"), None); // no tile
        assert_eq!(named("affine_qmm_t_bm_16_bn_16"), None); // no dtype/affine
        // Right points, wrong order.
        assert_eq!(named("affine_qmm_t_bfloat16_bm_16_bn_16_gs_64_b_4"), None);
        // A point that is not on the axis.
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_16_b_4"), None);
    }

    /// A row whose base is itself an entrypoint keeps it, and does not get
    /// eaten by a sibling that could peel to the same text.
    #[test]
    fn a_row_resolves_by_its_base_and_by_every_point() {
        assert_eq!(named("moe_route_sort"), Some("route_sort"));
        assert_eq!(named("router_topk_bfloat16"), Some("router"));
        // The BASE resolves too, and this is not a convenience: a model text
        // states the kernel, not the instantiation, because the affine format
        // is a checkpoint fact the lowering does not have. `dsl::metal` records
        // `affine_qmv_fast`; the driver resolves the point at load.
        assert_eq!(named("router_topk"), Some("router"));
        assert_eq!(named("affine_qmm_t"), Some("qmm_t"));
    }

    /// CUDA's rows carry no axes, and this is the assertion that the addition
    /// changed nothing for them: an axisless row matches its symbol and
    /// nothing else, prefix or suffix.
    #[test]
    fn an_axisless_row_is_unchanged_by_the_axis_machinery() {
        assert_eq!(named("moe_route_sort_bfloat16"), None);
        assert_eq!(named("moe_route_sor"), None);
        assert_eq!(named("xmoe_route_sort"), None);
    }

    /// The `sdpa_paged_decode` case, and the reason `points` may hold `""`.
    ///
    /// Three macros in `sdpa_paged.metal` stamp ONE template —
    /// `sdpa_paged_decode<itype, d, v, sink, PAGES, FIXED, SG>` — at
    /// `<…, 0, false, 32>`, `<…, 32, true, 32>` and `<…, 32, true, 8>`. Same
    /// body, three points, and the first contributes no text.
    #[test]
    fn an_axis_may_have_a_point_that_adds_no_text() {
        const DIM: Axis = Axis {
            what: "head dim",
            points: &["_d_64", "_d_128"],
        };
        // Longest first, empty last: both orderings are load-bearing.
        const PAGE: Axis = Axis {
            what: "page table width and simdgroup count",
            points: &["_p32_sg8", "_p32", ""],
        };
        static T: &[KernelSig] =
            &[kernel!(sdpa_paged "sdpa_paged_decode", axes = &[DTYPE, DIM, PAGE])];

        for name in [
            "sdpa_paged_decode_bfloat16_d_128",
            "sdpa_paged_decode_bfloat16_d_128_p32",
            "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
        ] {
            assert!(sig_in(T, name).is_some(), "{name}");
        }
        // Still not a licence to match anything.
        assert!(sig_in(T, "sdpa_paged_decode_bfloat16_d_256").is_none());
        assert!(sig_in(T, "sdpa_paged_decode_bfloat16").is_none());
        assert_eq!(T[0].entrypoints().len(), 2 * 3);
    }

    /// `covers` and `entrypoints` are two directions on one relation, and the
    /// audit script trusts both. Round-trip them.
    #[test]
    fn everything_a_row_generates_is_something_it_covers() {
        for row in TABLE {
            for name in row.entrypoints() {
                assert_eq!(
                    sig_in(TABLE, &name).map(|k| k.name),
                    Some(row.name),
                    "{name}"
                );
            }
        }
    }
}
