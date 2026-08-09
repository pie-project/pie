//! `moe`'s JIT units — thirteen rows over four units, and the twenty-six
//! kernels a rule cannot launch.
//!
//! # What this module holds
//!
//! One [`Unit`] per migrated `.cuh`, the [`DeviceKernel`] rows those units
//! instantiate, and the [`KernelSig`]s behind them. Each sig is its
//! ahead-of-time twin minus the stream — `cuLaunchKernel`'s sixth PARAMETER,
//! outside the `void**`, so it was never an operand — and minus whatever
//! extent the launch rule recovers.
//!
//! # Five files, thirty-nine entry points, thirteen rows
//!
//! The family's five `.cu` files held 31 `__global__`s and fired 39 launches.
//! Every one of them was SPLIT — the device text moved into a `.cuh` and the
//! `.cu` kept only its launchers — so the tree holds exactly one definition
//! of each kernel. That property is the point: `norm/altup_aux` shipped two
//! copies of a kernel for a release, each right for whichever half its tests
//! exercised.
//!
//! The 31 became 39 entry points, because two of them were run-time ladders
//! that a row cannot climb. The refactor's stated reason has since been
//! measured false — [`crate::device::args`] records that `elem` is a string
//! pasted between angle brackets, so `PER_LANE` and `FUSED_GEMV` were always
//! nameable — but the refactor stands on the reason that was always the real
//! one: **the host CHOSE between the rungs at run time.** A row names one
//! instantiation, and `topk_softmax_bf16` picks its rung from `num_experts`
//! and `K` while `moe_decode_gemv` picks its warp count from a device
//! property. Splitting each ladder into a `__device__` body plus one thin
//! `__global__` per form — five rungs for `topk_softmax_warp`, two for
//! `moe_decode_wmma`, two for `moe_decode_gemv` — turns a choice a row
//! cannot state into a set of symbols a trace can, which is what the JIT
//! needed and what naming `<T, 4>` would not have given it.
//!
//! One `__global__` in this family is still multi-argument —
//! `moe_decode_gemv<T, ActByToken, kWarps, kUnroll>` — and arity is not what
//! blocks it; see the 2-D block bullet below.
//!
//! Thirteen of the 39 are rowed. The other twenty-six are migrated as TEXT
//! and unmigrated as ROWS, and the reason is always the same shape: the
//! launcher states a geometry that [`crate::runtime::launch::eval`] does not
//! produce. It evaluates twelve of the vocabulary's sixteen rules and refuses
//! the rest, and its own test walks [`crate::unit::rows`] and fails on a row
//! stating anything else. A rule that computes the wrong extent is worse than
//! a launcher that still exists.
//!
//! # What the launch port unblocked, and what it did not
//!
//! Four rows landed the day [`kernels::LaunchRule::RouterLane`] and
//! [`kernels::LaunchRule::RouterSort`] were ported, and they are four of the
//! six kernels this module's previous revision listed as blocked on a fixed
//! block width or on shared memory sized from an operand. Both blockers were
//! real and both are gone: `RouterLane` fires exactly
//! `moe/topk_softmax.cuh`'s `kSoftmaxBlock`, and `RouterSort` computes
//! `(3 · n_experts + 34) · sizeof(int)` from [`crate::runtime::launch::Dims`]
//! rather than from a constant.
//!
//! **The two are two rules on purpose, and this family is where getting that
//! wrong would not show.** `RouterLane` opens its grid over rows and
//! `RouterSort` opens ONE BLOCK whatever the routing, because the counting
//! sort's scan is block-wide and its counters are in shared memory —
//! `moe_dispatch.cu`'s own words, *"a grid over rows would run N copies of
//! the sort, each clearing what the others are reading."* N copies of a sort
//! do not fail. They produce a permutation, the GEMMs consume it, and the
//! mixture answers with tokens routed to experts the router did not pick.
//!
//! # The twenty-six, by what blocks them
//!
//! **Re-audited at `LaunchRule` 21 → 28,** and exactly one entry moved:
//! `hash_route_lookup`, which [`kernels::LaunchRule::RowsFlat`] states digit
//! for digit and which is now [`DSV4_ROUTING_SIGS`]`[1]`. The other seven new
//! rules were each checked against every launcher here. `RoutedQmv` is
//! `quant`'s and reaches no `moe` launcher. `Tile16` is a 16×16 block and
//! this family's 2-D blocks are `dim3(32, kWarps)` and `dim3(32, 8)`.
//! `WarpTiledScan`'s three axes are `[rows, kv_heads, ceil(v / 4)]` at 128
//! and `transpose_expert_scales`' three are
//! `[ceil(k_groups / 32), ceil(n / 8), num_experts]` at `dim3(32, 8)`. `Slab` divides by 8 and caps at 1024, which no launcher
//! here does. `PerRowNarrow`, `RowsPerHead` and `AxialRope` have no launcher
//! of their shape in this family at all. Everything below still stands.
//!
//! **None of them is blocked on template arity, and that is a measurement.**
//! Every `__global__` in this family is already `template <class T>` with
//! exactly one type parameter — a sweep of all five files found no concrete
//! `nv_bfloat16` kernel to retype, unlike `attn` and `quant` where the split
//! left several behind. `moe` was converted with the element type already
//! abstracted, so the arity blocker that costs `attn` a row here costs
//! nothing. Every entry below is GEOMETRY, and templating cannot move one of
//! them.
//!
//! The same holds for the multi-argument finding — `elem` carrying a template
//! argument LIST rather than a single type — re-checked kernel by kernel
//! after it landed. `moe_decode_gemv<class T, bool ActByToken, int kWarps,
//! int kUnroll = 1>` is the family's ONE multi-argument template, and the
//! finding makes its name spellable, which changes nothing: the two thin
//! wrappers a statement would reach, `moe_decode_gemv_by_token` and
//! `_by_route`, are single-type templates that were always nameable, and both
//! launch `dim3 block(32, kWarps)` over `dim3 grid(ceil(N / kWarps), routes)`.
//! No ported rule states a 2-D BLOCK at all. The parked reason was the right
//! one and it was never arity.
//!
//!
//! * **A fixed block width that is part of the algorithm.** The five
//!   `topk_softmax_warp_x*` rungs fire ONE warp and reduce with
//!   `__shfl_xor_sync`; no rule states 32. `topk_softmax` and
//!   `topk_sigmoid_bias` are the same objection at 64 and are now ROWED,
//!   because `RouterLane` is that width by construction —
//!   `moe/topk_softmax.cuh`'s `static_assert(kSoftmaxBlock == 64,
//!   "block_argmax folds exactly one upper warp")` is the rule's constant
//!   rather than a number a rule happened to agree with.
//! * **A grid over routes or padded blocks.** `scatter_add_weighted`,
//!   `build_moe_ptrs_decode_batched`, `build_moe_ptrs_aligned`, and the four
//!   decode projections `moe_decode_{wmma,gemv}_by_{token,route}`. The route
//!   count is `tokens · top_k` and the padded block count is a host bound;
//!   [`crate::runtime::launch::Dims`] carries a rectangle's rows and widths
//!   and neither number is one. `scatter_add_weighted` is the one worth
//!   naming twice, because it looks like [`kernels::LaunchRule::Rms`] and is
//!   not: `<<<num_routed, kDispatchBlock>>>` is one block per ROUTE while the
//!   value it accumulates into is `[tokens, hidden]`, so a rows-shaped grid
//!   launches `top_k` times too few blocks and scatters a prefix of the
//!   routes. `add_moe_route_bias` is the same `<<<>>>` and IS `Rms`, for the
//!   one reason that decides it — the value THAT one writes is the
//!   route-major staging, so its rectangle's rows are the launcher's routes.
//! * **A grid of one that is not the sort's.** `build_dual_gemm_ptrs` is
//!   `<<<1, 1>>>` and `build_moe_ptrs_decode` is `<<<1, top_k>>>`.
//!   [`kernels::LaunchRule::RouterSort`] is `<<<1, 1024>>>` and its block is
//!   the scan's, not a free parameter: `build_moe_ptrs_decode` handed 1024
//!   lanes would run `1024 - top_k` threads through a pointer build indexed
//!   by `threadIdx.x`.
//! * **A 2-D or 3-D block.** `moe_decode_gemv_*` wants `dim3(32, kWarps)`,
//!   `transpose_expert_scales` wants `dim3(32, 8)` on a 3-D grid, and
//!   `moe_grouped_gemm` wants `dim3(N / 64, max_blocks)`. Every ported rule
//!   produces a 1-D block. This is what blocks
//!   `moe_decode_gemv<T, ActByToken, kWarps, kUnroll>`, the family's one
//!   multi-argument `__global__` — its three non-type arguments are all
//!   spellable in an `elem` string, and a rule that answered `[256, 1, 1]`
//!   would still give a kernel indexing `threadIdx.y` a single row of warps.
//! * **A choice the host makes at run time.** The four `_vec` forms are
//!   selected by `hidden % 8` and by the 16-byte ALIGNMENT of three
//!   pointers. A [`Source`] states where a value comes from, never a
//!   predicate over one, and firing a vectorised form on an odd hidden size
//!   puts every second row on a 2-byte boundary and faults. The scalar twins
//!   are rowed and are always correct.
//! * **A single-row reduction.** `batched_weighted_sum` collapses `batch`
//!   rows into one and launches `ceil(hidden / 256)`. `Elementwise` would
//!   multiply that by the fire's rows and lean on a guard to discard the
//!   surplus — the shape `norm/dsv4_hc` refused for `hc_post`.
//! * **A grid whose rows are the INPUT's.** `reorder_moe_aligned_output`
//!   reads the padded rectangle and writes route rows, so its grid is
//!   `aligned_rows` deep while its output is `routes` deep.
//!   `ElementwiseRows` opens over the output's rows and would drop the tail.
//!   Its mirror `gather_moe_aligned_inputs` WRITES the padded rectangle, and
//!   is rowed for exactly that reason.
//! * **A thread per token — RETRACTED, and the row is
//!   [`DSV4_ROUTING_SIGS`]`[1]`.** `hash_route_lookup` launches
//!   `ceil(tokens / 256)`, and the refusal was that `Elementwise` sizes on
//!   `rows · width` where this statement's width is `top_k` — the surplus
//!   blocks would return on the kernel's own `t >= tokens`, so the OUTPUT
//!   would be right, which is exactly the shape refused for `hc_post`.
//!   [`kernels::LaunchRule::RowsFlat`] was ported FROM this launcher and is
//!   the first flat rule here that does not multiply the rows by a width:
//!   `runtime::launch::rows_flat` is `grid [ceil(rows / 256), 1, 1]`,
//!   `block [256, 1, 1]`, `smem 0`, cited against `moe/dsv4_routing.cu:56-60`
//!   with the launcher's own comment — *"One thread per token, not one
//!   block"* — quoted as the rule. Nothing about the `hc_post` refusal
//!   changes: `Elementwise` is still `top_k` times too many blocks there and
//!   here, and what closed this one is a rule that divides the rows and
//!   stops.
//!
//! # Two rows exist because a stride constant became `blockDim.x`
//!
//! `topk_sigmoid` and `topk_sqrtsoftplus` staged their experts with loops
//! stepping by a `constexpr` block width. That pinned each launch to the
//! width its `.cu` happened to pass — 128 and 256 — and 128 is not a rule.
//! Both now step by `blockDim.x`, which is the same arithmetic per element at
//! the width the ahead-of-time path still fires and correct at any other, so
//! `LaunchRule::Rms` states both exactly. Their static `__shared__` slabs are
//! sized by an expert bound the launcher REFUSES to exceed, which is why
//! widening the block is safe and widening the router is not.
//!
//! # Four rows exist because an axis moved
//!
//! `token_batched_weighted_sum`, its `_add`, the aligned combine and the
//! aligned gather launched `dim3(ceil(width / 256), rows)` — the row on `y`.
//! `LaunchRule::ElementwiseRows` is the same rectangle with the row on `x`.
//! The two index lines in each kernel moved with the rule and the `dim3` in
//! `moe_dispatch.cu` moved with them, so both compilers launch the
//! transposed grid and every thread computes the element it computed before;
//! the guard is `h >= hidden` either way. `mlp`'s `gpt_oss_glu_strided`
//! kernel made the same move for the same reason (its row has since gone as
//! a §28.4 duplicate; the device text and its index lines have not).
//!
//! # The wmma call sites, which is why this family waited
//!
//! `moe_dispatch.cuh` and `moe_grouped_gemm.cuh` are the two `wmma` users in
//! the tree. NVRTC 13.0 refuses `mma.h` outright — *"could not open source
//! file 'mma.h'"* — so until `pie_mma.cuh` existed and was proved
//! bit-identical to `nvcuda::wmma` on an L40S, no unit carrying either file
//! could compile at all. Both call sites want the one shape it implements:
//! 16×16×16, bf16 × bf16 → f32, A `row_major`, B `col_major`, store
//! `mem_row_major`. Neither is rowed — `moe_grouped_gemm` has a 2-D grid over
//! a host bound and the decode wmma pair a 2-D grid over routes — so
//! `examples/unit_probe_moe.rs` instantiates all three by hand, which is the
//! only way the shim's coverage of this family is a measurement rather than a
//! reading.
//!
//! `moe/moe_grouped_gemm` therefore gets NO unit: a unit is a list of
//! instantiations, and that file has none a rule can state.
//!
//! # And `moe/moe_grouped_gemm_tile` gets none either, for a different reason
//!
//! That file is the same GEMM written in `cuda::tiles` — one
//! `__tile_global__`, no launcher, no `wmma`, no CUTLASS. It is correct
//! (worst relative error **0** against an fp64 reference at every shape and
//! tiling swept, with the padding blocks' poison bytes untouched, so the
//! early exit is measured rather than claimed) and it is **faster than
//! either kernel pie fires for MoE decode**. It is text and not a unit for
//! the compiler-floor reason below, and for that reason only.
//!
//! At `kTileM = 16`, which is what `moe_align_decode` emits and the only
//! height `moe_grouped_gemm_bf16_supported` accepts (`M == kFrag`), on the
//! decode census of 318 aligned blocks with 106 live:
//!
//! ```text
//!                       gate_up            down
//!                       N=512 K=2048       N=2048 K=256
//!    cutile (best)      0.324 ms           0.149 ms
//!    wmma twin          0.858              0.214
//!    cuBLAS, captured   0.972              0.449
//!    cuBLAS, ideal      0.327              0.177
//! ```
//!
//! `cuBLAS, ideal` batches the 106 LIVE blocks and is unattainable: the
//! batch count is a host argument baked into a captured graph and must be
//! the worst case, which is the 318-block row. So against what pie can
//! actually run the tile kernel wins by 3.0x at both shapes — and it now
//! beats the unattainable ideal too, which says the early exit is not
//! merely cheap but free.
//!
//! Those figures moved once already and by a lot. Until §23.20 they read
//! 0.349 and 0.185, measured on a kernel whose extents and loop bounds were
//! run-time values; making them compile-time constants, as NVIDIA's own
//! `matmul.cuh` does, is the whole difference. N and K are model constants
//! and a JIT instantiates per shape, so this costs nothing but saying so.
//!
//! **An earlier version of this comment said the opposite**, reporting the
//! tile kernel 1.93x slow at gate_up and 1.21-1.31x slow at down, and
//! locating the limit at 214 registers against the twin's 40. Those numbers
//! were real but they measured a workaround, not the kernel: bf16 was being
//! carried through as `unsigned short` and widened to fp32 inside the tile,
//! which doubled the operand register footprint and bypassed the bf16
//! tensor-core path. The widening existed because 16-bit float tiles
//! appeared not to compile, and they appeared not to compile because the
//! runtime headers were CUDA 13.0 under a 13.3 tile compiler.
//! `.wiki/driver/cutile-16bit-header-trap.cu` is that story; the kernel's
//! own header carries the full before-and-after.
//!
//! Re-measured since the rewrite, and it does not all go one way. The
//! CUTLASS island (`flashinfer_cutlass_moe_bf16`, which fuses permute + fc1
//! + activation + fc2 + finalise) was timed directly rather than quoted, at
//! 318 tokens / hidden 2048 / inter 256 / 256 experts / top-k 8:
//!
//! ```text
//!                                    island       two tile GEMMs
//!                                    (5 stages)   (2 stages)
//!   256 experts, 16 rows each        1.241 ms     1.328 ms   kTileM=16
//!   106 experts, 24 rows each        0.581        0.654      kTileM=32
//! ```
//!
//! The GEMM gap closed — a previously recorded 1.9x is now 7% at the first
//! census and **12%** at the second, where the island does five stages
//! against two. The FUSION gap did not close: fusing the three stages into
//! one kernel measures 0.984 ms, worse than not fusing, because
//! fewer experts over the same rows means more rows per expert, and this
//! kernel reads `W[e]` once per block and round-trips the intermediate
//! through HBM where the island reads each expert once and keeps fc1's
//! output resident.
//!
//! So the CUTLASS dependency is not removable on these numbers, which was
//! the original question. What changed is that it is now a question about
//! one fused kernel rather than a class of them.
//!
//!
//! # And `moe/moe_fused_tile` gets none either, being a negative result
//!
//! That file answers the question the row above leaves open. It writes
//! fc1 + swiglu + fc2 as one `__tile_global__`, one expert block per CUDA
//! block owning the whole fc2 output panel, intermediate never stored —
//! the island's access pattern stated directly. It compiles, and it is
//! correct to 0.42% worst relative error on positive data, which is 2^-8
//! and therefore the bf16 rounding floor.
//!
//! ```text
//!   island   permute+fc1+act+fc2+finalise   0.581 ms   573 GB/s
//!   two unfused tile GEMMs                  0.933      ~358
//!   the fused kernel, best of the sweep     1.778      187
//! ```
//!
//! Fusing made it twice as slow as not fusing. The cause is shared memory
//! rather than registers: the tile compiler stages `partition_view` loads
//! through it, and the fused working set takes 92-99 KB of a 100 KB budget
//! against the unfused grouped GEMM's 16 KB. That is one block per SM, and
//! 106 to 318 blocks each alone on an SM cannot hide HBM latency.
//!
//! It is pinned there — `FNK`, `FN2` and `FM` sweeps never drop below 92 KB
//! — and `cuda::tiles` exposes no shared scratch, no `insert` and no
//! occupancy control, so there is no third version to reach for. The file is
//! carried, with a banner that says all of this, because it looks exactly
//! like a kernel someone should finish and it is not.
//! `tests/vendor_manifest.rs` gates the banner.
//!
//!
//! # And the compiler floor, which is the actual refusal
//!
//! A unit is a claim that **this crate's compiler** can compile it, and
//! that claim is false today:
//!
//! * the crate loads the system `libnvrtc`, 13.0.88 here, and CUDA 13.0
//!   ships the `__tile_global__` macro with no tile API behind it
//!   (`cuda_tile.h` is 60 lines and declares `print`). `tests/units.rs`
//!   compiles every declared unit with that compiler, so a unit added today
//!   would fail the gate for the whole crate;
//! * the runtime headers must be 13.3 or newer for any 16-bit float tile to
//!   compile at all, per the trap above;
//! * `tileiras` must be on `PATH` with `CUDA_ROOT` set, to assemble what
//!   NVRTC returns.
//!
//! All three are pip wheels. None is a wait on NVIDIA or on a driver.
//!
//! **The JIT path itself works on this box's 13.0 driver**, which two
//! earlier versions of this comment denied. Measured end to end with a bf16
//! tile `mma`: nvrtc compiles, `nvrtcGetTileIR` yields 6,314 bytes,
//! `tileiras` assembles it, `cuModuleLoadData` and `cuModuleGetFunction`
//! and `cuLaunchKernel` all succeed, and the result is exact. NVRTC returns
//! pure Tile IR — `.note.nv.tkinfo`, no `.text` — and the driver loads that
//! image without assembling it, so stopping there gives `NOT_FOUND`;
//! running it through `tileiras` yourself closes the gap. Cold latency is
//! 0.62-0.71 s, of which `tileiras` is 0.18 s, which is what
//! `PIE_HOME/cache` is for.
//!
//! `tileiras` requires `CUDA_ROOT` and does not say so — without it every
//! input fails with a bare `failed to compile Tile IR program`, including
//! nvcc's own `.tilebc`. See `.wiki/driver/new-horizon.md` §23.18.
//!
//! When those wheels are packaged the unit is three lines and its
//! `Unit::opts` are already known and each load-bearing:
//! `-std=c++20 -enable-tile -default-device`. `kTileN`/`kTileK` want to be
//! set per row — 32x128 at gate_up, 32x32 at down — which is what `opts` is
//! for.
//!
//! The file's own header carries the numbers, the sweeps, the header trap
//! and the wrong conclusions drawn on the way.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Lit;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The sigmoid router — one block per token, and the family's simplest unit.
pub const TOPK_SIGMOID: Unit = Unit {
    name: "moe/topk_sigmoid",
    root: include_str!("../../csrc/src/moe/topk_sigmoid.cuh"),
    rows: TOPK_SIGMOID_ROWS,
    options: &[],
};

/// DeepSeek-V4's two routers: the sqrt-softplus top-k, and the hash lookup
/// that has no rule.
pub const DSV4_ROUTING: Unit = Unit {
    name: "moe/dsv4_routing",
    root: include_str!("../../csrc/src/moe/dsv4_routing.cuh"),
    rows: DSV4_ROUTING_ROWS,
    options: &[],
};

/// The softmax routers and the per-expert rescale — nine templates, three
/// rows.
pub const TOPK_SOFTMAX: Unit = Unit {
    name: "moe/topk_softmax",
    root: include_str!("../../csrc/src/moe/topk_softmax.cuh"),
    rows: TOPK_SOFTMAX_ROWS,
    options: &[],
};

/// The dispatch path proper: the sorts, the permutes, the pointer builds, the
/// decode projections and the combines.
pub const MOE_DISPATCH: Unit = Unit {
    name: "moe/moe_dispatch",
    root: include_str!("../../csrc/src/moe/moe_dispatch.cuh"),
    rows: MOE_DISPATCH_ROWS,
    options: &[],
};

/// The units `moe` compiles.
pub static UNITS: &[Unit] = &[TOPK_SIGMOID, DSV4_ROUTING, TOPK_SOFTMAX, MOE_DISPATCH];

/// [`TOPK_SIGMOID`]'s instantiation.
static TOPK_SIGMOID_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &TOPK_SIGMOID_SIGS[0],
    template_path: "moe::device::topk_sigmoid",
    elem: "device::bf16",
}];

/// The contract, in [`TOPK_SIGMOID_ROWS`]' order.
#[rustfmt::skip]
static TOPK_SIGMOID_SIGS: [KernelSig; 1] = [
    // `Rms` -- one block per token, 256 threads, the block striding the
    // expert axis. The ahead-of-time launcher fired 128; the staging loops
    // step by `blockDim.x`, so the rule's wider block reaches the same
    // experts in half the iterations and the arithmetic per element is
    // unchanged. `tokens` leaves the row because the grid IS the tokens.
    //
    // The exception in a family of refusals, and it is the router: a token's
    // top-k reads only its own logits row, so this statement splits like any
    // elementwise one and `whole` is not set.
    kernel!(topk_sigmoid "moe::topk_sigmoid_bf16",
        file = Some("moe/topk_sigmoid.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf <- Source::In(0),
            topk_idx: I32sMut <- Source::Out(0),
            topk_w: F32sMut <- Source::Out(1),
            // A FAMILY WITHOUT A CORRECTION BIAS STATES NO FOURTH OPERAND,
            // and the kernel reads a null as "there is none" -- the same
            // reading its ahead-of-time twin took.
            correction_bias: F32s <- Source::Or(
                &Source::Weight(0),
                &Source::Lit(Lit::Null),
            ),
            num_experts: I32 <- Source::Width(&Source::In(0)),
            top_k: I32 <- Source::Width(&Source::Out(0)),
            // The deployment's, both of them -- `norm_topk_prob` and
            // `routed_scaling_factor` are config values the driver reads at
            // load, which is why they are context and not params.
            renormalize: Bool <- Source::Ctx("moe_norm_topk"),
            routed_scaling_factor: F32 <- Source::Ctx("moe_routed_scaling"),
        ]),
];

/// [`DSV4_ROUTING`]'s instantiations — both of the file's two templates.
static DSV4_ROUTING_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DSV4_ROUTING_SIGS[0],
        template_path: "moe::device::topk_sqrtsoftplus",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_ROUTING_SIGS[1],
        template_path: "moe::device::hash_route_lookup",
        elem: "device::bf16",
    },
];

/// The contracts, in [`DSV4_ROUTING_ROWS`]' order.
#[rustfmt::skip]
static DSV4_ROUTING_SIGS: [KernelSig; 2] = [
    // `Rms` again, and here it is the launcher unchanged: `<<<tokens, 256>>>`
    // is the rule's grid and the rule's block. The 32 bytes it hands the
    // launch as dynamic shared memory are never read -- this kernel reduces
    // on thread 0 and never calls `device::block_sum`.
    kernel!(topk_sqrtsoftplus "moe::topk_sqrtsoftplus_bf16",
        file = Some("moe/dsv4_routing.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf <- Source::In(0),
            topk_idx: I32sMut <- Source::Out(0),
            topk_w: F32sMut <- Source::Out(1),
            correction_bias: F32s <- Source::Or(
                &Source::Weight(0),
                &Source::Lit(Lit::Null),
            ),
            num_experts: I32 <- Source::Width(&Source::In(0)),
            top_k: I32 <- Source::Width(&Source::Out(0)),
            renormalize: Bool <- Source::Ctx("moe_norm_topk"),
            routed_scaling_factor: F32 <- Source::Ctx("moe_routed_scaling"),
        ]),
    // ONE THREAD PER TOKEN, and the launcher says so in its own words:
    //
    // ```text
    // // One thread per token, not one block: the kernel's whole body is a
    // // table read and a K-long gather.
    // const int grid = (tokens + kDsv4Block - 1) / kDsv4Block;
    // device::hash_route_lookup<device::bf16><<<grid, kDsv4Block, 0, stream>>>(
    // ```
    //
    // at `moe/dsv4_routing.cu:56-66`, with `kDsv4Block = 256` at `19`.
    // `runtime::launch::rows_flat` answers `grid [ceil(rows / 256), 1, 1]`,
    // `block [256, 1, 1]`, `smem 0` — the same three numbers, and the rule
    // was ported from this `<<<>>>`.
    //
    // **It is the sibling above's grid divided by 256, and that is the whole
    // difference between the two rows.** `topk_sqrtsoftplus` gives a token a
    // BLOCK because it stages `num_experts` logits and reduces them; this
    // kernel reads one `tid2eid` row and gathers `K` weights, so it gives a
    // token a THREAD. Stating `Rms` here would launch 256 times the blocks
    // and idle 255 lanes of each; stating `Elementwise` would launch
    // `top_k` times too many and hand every surplus block a `blockIdx.x *
    // 256 + threadIdx.x` past the token count. Both would produce the right
    // OUTPUT, because `if (n >= tokens) return;` is the kernel's first line
    // — which is why neither was written, and why this row waited for a rule
    // that divides the rows and stops.
    //
    // `tokens` STAYS an operand at the width the `__global__` declares.
    // The launcher spends it on the grid AND passes it, because the guard is
    // the kernel's own and the last block is partial.
    //
    // UNSOURCED, exactly as the ahead-of-time twin in `crate::table::moe`
    // leaves them -- `table/moe.rs:152` states eleven operands and a stream
    // with no `Source` on any of them. Two of the eleven have no spelling
    // here at all: `tid2eid` is a `[vocab, K]` table keyed by TOKEN ID rather
    // than by anything the fire's rectangle carries, and `vocab_size` is its
    // first extent. The sibling above sources all eight of its operands
    // because every one is the statement's; a guessed `Weight(0)` for a table
    // no declaration has named yet would bind the wrong buffer with nothing
    // to report it, which is the standard `causal_conv1d_update_batched`
    // sets in `crate::families::ssm`. The geometry, the device text and the
    // offline typecheck all land; the sources land when a statement does.
    kernel!(hash_route_lookup "moe::hash_route_lookup",
        file = Some("moe/dsv4_routing.cuh"),
        launch = LaunchRule::RowsFlat,
        operands = operands![
            token_ids: I32s,
            tid2eid: I64s,
            logits: Buf,
            topk_idx: I32sMut,
            topk_w: F32sMut,
            tokens: I32,
            vocab_size: I32,
            num_experts: I32,
            top_k: I32,
            renormalize: Bool,
            routed_scaling_factor: F32,
        ]),
];

/// [`TOPK_SOFTMAX`]'s instantiations — three of the file's nine templates.
///
/// The six routers still carried as device text and unrowed are the five
/// `topk_softmax_warp_x*` rungs, whose block is ONE WARP where
/// [`kernels::LaunchRule::RouterLane`] is 64 and which the host picks between
/// at run time, and `router_topk_softmax`, which `RouterLane` states exactly
/// and which has no
/// ahead-of-time row to mirror its operands from — see
/// [`TOPK_SOFTMAX_SIGS`]'s last note for why that is a refusal rather than an
/// omission. The file's own header says which and why.
static TOPK_SOFTMAX_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &TOPK_SOFTMAX_SIGS[0],
        template_path: "moe::device::apply_per_expert_scale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOPK_SOFTMAX_SIGS[1],
        template_path: "moe::device::topk_softmax",
        elem: "device::bf16",
    },
    // The fp32 instantiation, because the fp32 launcher is the one with an
    // ahead-of-time row: `topk_sigmoid_bias_bf16` exists in the `.cu` and in
    // no table, so rowing it would be naming a symbol no model text can
    // state. One template, two element types, and the row picks the one a
    // trace can reach.
    //
    // `moe::device::f32` and NOT `device::f32` — the prelude names no fp32
    // alias, and `topk_softmax.cuh` declares its own beside the `Load`
    // specialisation that makes an fp32 router one kernel with the bf16 one.
    // The unqualified spelling compiles to `namespace ... has no member
    // "f32"` at the name-map pragma, before any launch.
    DeviceKernel {
        sig: &TOPK_SOFTMAX_SIGS[2],
        template_path: "moe::device::topk_sigmoid_bias",
        elem: "moe::device::f32",
    },
];

/// The contracts, in [`TOPK_SOFTMAX_ROWS`]' order.
#[rustfmt::skip]
static TOPK_SOFTMAX_SIGS: [KernelSig; 3] = [
    // `Elementwise` IS the launcher: `ceil(N * K / 256)` blocks of 256 over
    // the `[tokens, top_k]` rectangle of routes, same block, same rounding.
    // `n` and `k` were two operands of the twin and are one here, because the
    // kernel only ever used their product -- and the product is the weight
    // vector's own element count, which is why it can be sourced at all.
    kernel!(apply_per_expert_scale "moe::apply_per_expert_scale_bf16",
        file = Some("moe/topk_softmax.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            topk_idx: I32s <- Source::In(0),
            topk_w: F32sMut <- Source::Out(0),
            // A per-expert gain from the checkpoint, not a value the trace
            // computes -- the same slot the routers' correction bias takes.
            per_expert_scale: Buf <- Source::Weight(0),
            total: I32 <- Source::OutElements(0),
        ]),
    // `RouterLane` — `<<<N, kSoftmaxBlock>>>`, the launcher unchanged, and
    // the 64 is the rule's own constant rather than a width it happens to
    // agree with: `topk_softmax.cuh` carries
    // `static_assert(kSoftmaxBlock == 64, "block_argmax folds exactly one
    // upper warp")`, and a rule that widened it would compile, launch, and
    // fold a warp that was never written. `n` leaves the row because the grid
    // IS the tokens.
    //
    // **This row is the BLOCK form, and the ahead-of-time symbol is a
    // ladder.** `topk_softmax_bf16` picks between five warp rungs and this
    // kernel on `num_experts` and `K` — a RUN-TIME choice, which is what a
    // row cannot state, and not an arity problem: the rungs are separate
    // `__global__`s and would each be nameable if a rule launched 32. The
    // same shape the `_vec` forms are refused for. The block form is what
    // `PIE_TOPK_WARP=0` fires and is the arm every A/B in this family was
    // measured against, so the row is always correct and, on the one
    // configuration the launcher's comment measured, slower: Qwen3.6-35B-A3B
    // routes through more than 128 experts at 7.56 us/call, 4.9% of its step.
    // That number is the cost of rowing this symbol today and the argument
    // for a rule that states a warp-wide router later; it is not a reason to
    // leave the kernel unfireable.
    //
    // There is now a CuTile ALTERNATIVE to this arm --
    // `moe/topk_softmax_tile.cuh` -- measured against THIS kernel at 256
    // experts: 4.52 us against 6.23 at one row and 4.65 against 6.52 at 128,
    // with identical expert indices, crossing over at about a thousand rows.
    // A tile does not care how wide the router is, which is the whole reason
    // the warp ladder exists. It does not retire the ladder: the alternative
    // needs a toolchain this crate does not load, so this row and its five
    // rungs stay.
    //
    // `act`, `bias` and `hidden` are the FUSED form's operands and the
    // launcher passes two nulls and a zero. They stay in the row because the
    // `__global__` declares them: the row's operand list is the kernel's
    // parameter list, and `cuLaunchKernel` reads `sizeof(param)` per cell off
    // an array whose length nothing else checks. `router_topk_softmax` is the
    // same body with `FusedGemv` true and is what reads them.
    kernel!(topk_softmax "moe::topk_softmax_bf16",
        file = Some("moe/topk_softmax.cuh"),
        launch = LaunchRule::RouterLane,
        operands = operands![
            logits: Buf <- Source::In(0),
            act: Buf <- Source::Lit(Lit::Null),
            bias: Buf <- Source::Lit(Lit::Null),
            topk_idx: I32sMut <- Source::Out(0),
            topk_w: F32sMut <- Source::Out(1),
            num_experts: I32 <- Source::InWidth(0),
            k: I32 <- Source::OutWidth(0),
            hidden: I32 <- Source::Lit(Lit::I32(0)),
        ]),
    // `RouterLane` again, and the same `<<<N, kSoftmaxBlock>>>` — DeepSeek's
    // sigmoid routing with the correction bias entering the ranking and not
    // the published weight.
    //
    // `normalize` is `I32` here and `Bool` in the ahead-of-time row, and the
    // difference is not drift: the C++ HOST function takes a `bool` and
    // narrows it with `normalize ? 1 : 0`, and the `__global__` this row
    // names declares `int normalize`. A row describes the cubin's parameter
    // list, so the ternary that lived in the launcher becomes the row's type.
    // The cell would carry the right value either way — `u64::from(bool)` is
    // 0 or 1 and the driver copies four bytes of it — which is exactly why
    // the wrong spelling here would never be caught by a fire.
    //
    // `correction_bias` is read UNCONDITIONALLY by this kernel, unlike
    // `topk_sigmoid`'s optional one, so there is no `Source::Or` and no null:
    // this entry point is the one a checkpoint WITH a bias uses, and a null
    // here is a fault rather than an absence.
    kernel!(topk_sigmoid_bias "moe::topk_sigmoid_bias_fp32",
        file = Some("moe/topk_softmax.cuh"),
        launch = LaunchRule::RouterLane,
        operands = operands![
            logits: F32s <- Source::In(0),
            correction_bias: F32s <- Source::Weight(0),
            topk_idx: I32sMut <- Source::Out(0),
            topk_w: F32sMut <- Source::Out(1),
            num_experts: I32 <- Source::InWidth(0),
            k: I32 <- Source::OutWidth(0),
            normalize: I32 <- Source::Ctx("moe_norm_topk"),
            routed_scaling_factor: F32 <- Source::Ctx("moe_routed_scaling"),
        ]),
];

/// [`MOE_DISPATCH`]'s instantiations — nine of the file's twenty-four.
///
/// Six are bf16 and two are `i32`, and the pair is the counting sort: its
/// element type is the routing INDEX rather than an activation, and
/// `moe_dispatch.cuh` says so in a `static_assert(is_same<T, i32>::value,
/// "the routing indices are i32")` that a row naming any other type would
/// trip at compile rather than at fire. The file's other sixteen templates
/// are carried as device text; this module's header lists what blocks each of
/// them, and the `.cuh`'s own header repeats the list beside the kernels it
/// is about.
static MOE_DISPATCH_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[0],
        template_path: "moe::device::scalar_weighted_add",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[1],
        template_path: "moe::device::token_batched_weighted_sum",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[2],
        template_path: "moe::device::token_batched_weighted_sum_add",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[3],
        template_path: "moe::device::gather_moe_aligned_inputs",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[4],
        template_path: "moe::device::add_moe_route_bias",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[5],
        template_path: "moe::device::moe_align_decode",
        elem: "device::i32",
    },
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[6],
        template_path: "moe::device::moe_bucket_exact",
        elem: "device::i32",
    },
    DeviceKernel {
        sig: &MOE_DISPATCH_SIGS[7],
        template_path: "moe::device::scatter_add_weighted",
        elem: "device::bf16",
    },
];

/// The contracts, in [`MOE_DISPATCH_ROWS`]' order.
#[rustfmt::skip]
static MOE_DISPATCH_SIGS: [KernelSig; 8] = [
    // `out += weight * src` over a flat buffer -- the shared expert's
    // contribution folded onto the routed one. `Elementwise` is the launcher
    // exactly: `ceil(n / 256)` blocks of 256, `n` rounded UP so the last
    // block runs threads the buffer does not have and the kernel's `i >= n`
    // guard is what stops them.
    //
    // In place over the value it accumulates onto -- one result, the same
    // bytes -- so `out` binds from `Out(0)` and `src` is the statement's
    // SECOND operand. That is `norm::add_bias_bf16`'s reading, and the
    // aliasing is what makes `In(1)` right where `In(0)` would look right.
    kernel!(scalar_weighted_add "moe::scalar_weighted_add_bf16",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            out: BufMut <- Source::Out(0),
            src: Buf <- Source::In(1),
            // The shared expert's gate, a scalar the arm computed per fire.
            // Nothing in the vocabulary names it yet, so it stays unsourced
            // and the row is declared, compiled and instantiated while the
            // binder still cannot fill it.
            weight: F32,
            n: I32 <- Source::OutElements(0),
        ]),
    // The combine: `out[n, h] = Σ_k weights[n, k] · src[n, k, h]`.
    //
    // `ElementwiseRows` -- one block ROW per token, `ceil(hidden / 256)`
    // tiles wide. The C++ launched the transpose of this; the kernel's two
    // index lines and its `dim3` moved together, so the coverage is identical
    // and the guard is `h >= hidden` either way. `num_tokens` leaves the row
    // because the grid IS the tokens.
    //
    // Neither extent needs the plan: `weights` IS `[tokens, top_k]`, so its
    // row width is the route count, and the result IS `[tokens, hidden]`.
    // Asking the plan for a number two operands already state is an inference
    // pass replacing a one-line answer.
    kernel!(moe_weighted_sum "moe::token_batched_weighted_sum_bf16",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            out: BufMut <- Source::Out(0),
            src: Buf <- Source::In(0),
            weights: F32s <- Source::In(1),
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::OutWidth(0),
        ]),
    // The `_add` spelling accumulates into the residual, which the statement
    // carries as its THIRD operand; the plain spelling above writes a fresh
    // value and aliases nothing. A separate kernel and not a flag, because a
    // read-modify-write that a branch skips still costs the read.
    kernel!(moe_weighted_sum_add "moe::token_batched_weighted_sum_add_bf16",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::ElementwiseRows,
        in_place = &[(0, 2)],
        operands = operands![
            out: BufMut <- Source::Out(0),
            src: Buf <- Source::In(0),
            weights: F32s <- Source::In(1),
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::OutWidth(0),
        ]),
    // The gather that fills the padded block-major rectangle the routed GEMMs
    // read. `ElementwiseRows` opens its grid over the OUTPUT's rows, and the
    // output here IS that rectangle -- which is the whole of why this one is
    // rowed and `reorder_moe_aligned_output` is not.
    //
    // `aligned_rows` stays an operand even though the rule now produces
    // exactly that many blocks: the kernel guards on it, and a row that drops
    // an operand because a rule happens to make its guard unreachable is a
    // row that breaks when the rule changes.
    //
    // `shared_row_begin` is `-1` at EVERY call site in the C++ tree, so the
    // row states it once instead of each arm restating it. `num_tokens` is
    // the FIRE's rows and not the grid's -- the two differ here, which is
    // exactly why `Source::Rows` and `Dims::rows` are two things.
    kernel!(gather_moe_aligned_inputs "moe::gather_moe_aligned_inputs_bf16",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::ElementwiseRows,
        whole = true,
        operands = operands![
            norm_x: Buf <- Source::In(0),
            sorted_route_ids: I32s <- Source::In(1),
            aligned_in: BufMut <- Source::Out(0),
            num_routes: I32 <- Source::RoutesOfParam(0),
            aligned_rows: I32 <- Source::OutRows(0),
            top_k: I32 <- Source::Param(0),
            hidden: I32 <- Source::OutWidth(0),
            shared_row_begin: I32 <- Source::Lit(Lit::I32(-1)),
            num_tokens: I32 <- Source::Rows,
        ]),
    // Each route's expert bias, added onto the route's row in place.
    //
    // `Rms` -- one block per row, 256 threads, the block striding the row --
    // and the grid is the launcher's unchanged, because the value this writes
    // IS the route-major staging: one row per route, so the rectangle's rows
    // and the launcher's routes are the same number.
    //
    // `whole` all the same: `topk_idx` is route-global, so a window over rows
    // would read another window's experts. `cols` and `out_stride` stay
    // unsourced, as the twin left them -- `cols` is the BIAS's width and
    // `out_stride` the staging's pitch, and a fire that splits a fused bias
    // holds neither as an extent of a value.
    kernel!(add_moe_route_bias "moe::add_moe_route_bias_bf16",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::Rms,
        whole = true,
        operands = operands![
            out: BufMut <- Source::Out(0),
            bias: Buf <- Source::Weight(0),
            topk_idx: I32s <- Source::In(0),
            num_routes: I32 <- Source::OutRows(0),
            cols: I32,
            out_stride: I32,
        ]),
    // The counting sort that buckets routes by expert and pads each bucket to
    // a block boundary, so one batched GEMM covers every expert.
    //
    // `RouterSort` — `<<<1, 1024, (3·num_experts + 34)·4>>>`, the launcher's
    // `<<<1, kSortBlock, (3 * num_experts + 1 + 33) * sizeof(int32_t)>>>`
    // rearranged: `3E + 34` and `3E + 1 + 33` are the same number, and the
    // slab's five regions — `counts`, `offsets` (+1), `fill`, 32 warp
    // partials, one running base — add to exactly it.
    //
    // **The grid is ONE BLOCK and that is the rule, not a coincidence of this
    // fire's routing.** The exclusive scan over per-expert padded counts is
    // block-wide and the counters live in that shared slab, so a rule with a
    // row axis would launch N copies of the sort, each zeroing the counters
    // the others were accumulating into. Nothing about that fails: it
    // returns, the permutation is a permutation, the batched GEMM consumes
    // it, and the mixture answers with tokens delivered to experts the router
    // did not choose. `RouterLane` next door DOES open over rows, and the two
    // rules exist separately so that this distinction is stated in the
    // vocabulary rather than remembered.
    //
    // `whole` because the permutation is computed over ALL routes in the
    // fire — a row window would sort a different set of routes than the
    // pointer builds and the reorder downstream address through.
    //
    // `T` is `i32` and the kernel `static_assert`s it: this sort has no
    // element type, the indices ARE its data, and a signature format that
    // supplies exactly one template argument still has to be handed something
    // true. A row naming bf16 here fails NVRTC rather than a fire.
    //
    // `num_tokens_past_padded` is `Lit::Null` as the ahead-of-time row has
    // it: the Marlin/Triton grouped GEMMs read it and cuBLAS's does not, and
    // the kernel guards the store. `block_size` and `max_blocks` ride the
    // param channel for the reason the twin records at length — the aligned
    // rectangle's rows are their product, and no `Source` divides.
    kernel!(moe_align "moe::moe_align_decode",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::RouterSort,
        whole = true,
        operands = operands![
            topk_idx: I32s <- Source::In(0),
            sorted_route_ids: I32sMut <- Source::Out(0),
            expert_ids: I32sMut <- Source::Out(1),
            route_to_aligned_row: I32sMut <- Source::Out(2),
            num_routes: I32 <- Source::InElements(0),
            num_experts: I32 <- Source::Param(0),
            block_size: I32 <- Source::Param(1),
            max_blocks: I32 <- Source::Param(2),
            num_tokens_past_padded: I32sMut <- Source::Lit(Lit::Null),
        ]),
    // The UNPADDED sort: exact per-expert counts, for the host to build
    // cuBLAS grouped shapes from.
    //
    // `RouterSort` again, and the launcher is `<<<1, kSortBlock,
    // (3 * num_experts + 1) * sizeof(int32_t)>>>` — thirty-three words FEWER
    // than the rule computes, because this scan is serial on thread 0 and
    // allocates no warp partials and no running base. The rule OVER-allocates
    // 132 bytes for this kernel and `launch.rs` blesses that in the same
    // words this row will: one slab size for both sorts is a rule, two would
    // be two rules that differ by a constant, and over-allocating dynamic
    // shared memory is legal while under-allocating is a launch failure or,
    // worse, a silent overlap. 132 bytes against an L40S's 100 KB per block
    // is not a number that changes an occupancy.
    //
    // The operands mirror the ahead-of-time row exactly, minus the stream:
    // this one publishes `counts_out` and `route_to_sorted_row` rather than
    // `expert_ids` and an aligned map, because its consumer is the host's
    // shape builder rather than a padded batched GEMM.
    //
    // `num_routes` and `num_experts` stay unsourced as the twin left them:
    // this entry point is reached from a path that holds neither as an extent
    // of a value it named, and inventing an edge to `topk_idx` that the
    // route count happens to equal would be a claim about the trace rather
    // than a reading of it.
    kernel!(moe_bucket_exact "moe::moe_bucket_exact",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::RouterSort,
        whole = true,
        operands = operands![
            topk_idx: I32s,
            sorted_route_ids: I32sMut,
            route_to_sorted_row: I32sMut,
            counts_out: I32sMut,
            num_routes: I32,
            num_experts: I32,
        ]),
    // THE INDEXED SCATTER, one block per ROUTED ROW.
    //
    // `moe/moe_dispatch.cu`:
    //
    // ```text
    // if (num_routed <= 0 || hidden <= 0) return;
    // device::scatter_add_weighted<device::bf16><<<
    //     num_routed, device::kDispatchBlock, 0, stream>>>(
    //     out, src, dst_idx, row_weights, hidden);
    // ```
    //
    // `kDispatchBlock` is 256 and `Rule::PerRow` launches `grid(rows)` at 256
    // with no shared memory — the launcher's three fields, unchanged. There
    // is no non-type template argument to cite: `scatter_add_weighted` is
    // `template <class T>`.
    //
    // **The 256 is contract here even though nothing is reduced.** The stride
    // loop is `for (h = threadIdx.x; h < hidden; h += kDispatchBlock)` — the
    // file-scope CONSTANT, not `blockDim.x` — so a launch at any other width
    // leaves a slice of every row uncomputed while the guard-free loop
    // double-adds the rest, on a read-modify-write. `moe_dispatch.cuh` says
    // so beside the kernel. That the rule and the constant are both 256 is
    // the agreement this row rests on, and the two do not name each other.
    //
    // **`Dims::rows` must be the ROUTED SLOTS, not the tokens.** The grid is
    // `num_routed = tokens * experts_per_token`, and `Dims::rows` is defined
    // as *"tokens, requests or routed slots, whichever the statement's
    // lowering counted"*. A fire whose rectangle counted tokens would launch
    // one block per token against a `dst_idx` with one entry per route: the
    // first `tokens` routes land and the rest are dropped, so the experts a
    // token was sent to past the first silently contribute nothing. Nothing
    // faults — the reason this row is unsourced and `whole`, exactly as its
    // twin is, and the reason that precondition is written here rather than
    // assumed.
    //
    // `hidden` stays: it is the pitch the kernel addresses `src` and `out`
    // with, not an extent the grid recovers.
    kernel!(scatter_add_weighted "moe::scatter_add_weighted_bf16",
        file = Some("moe/moe_dispatch.cuh"),
        launch = LaunchRule::PerRow,
        whole = true,
        operands = operands![
            out: BufMut,
            src: Buf,
            dst_idx: I32s,
            row_weights: F32s,
            hidden: I32,
        ]),
];
