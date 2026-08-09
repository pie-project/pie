//! `gemm`'s JIT unit — `gemv.cuh`'s two `__global__` templates, four rows.
//!
//! `gemm/gemv.cu` held two `__global__` templates and four launches inside one
//! host launcher, `gemv_bf16`. Three findings were recorded here as reasons no
//! row could exist. **All three are retired, none of them was ever a reason
//! for a `.cu` to survive, and the file is deleted.** The principle that
//! retires them:
//!
//! > Every CUDA kernel is compiled by NVRTC. Where host code is needed to
//! > compose several kernels — because kernels produce intermediate results,
//! > or because device-specific tuning is involved — that host code is all
//! > Rust.
//!
//! Each finding was a correct answer to *can a ROW state this?* and was then
//! used to answer *must this stay in C++?*, which is a different question with
//! a different answer. They are struck through rather than deleted because the
//! reasoning is sound and the misuse is the instructive part.
//!
//! * ~~**The geometry.**~~ **RETIRED — by a path that was already in the
//!   tree.** Every launch is `dim3(32, kWarps)`: a warp per row, `kWarps` rows
//!   per block. No [`kernels::LaunchRule`] states a 2-D BLOCK, and the rules
//!   `runtime::launch` evaluates all fix `blockDim.x` at 256. A rule invented
//!   for these would be a geometry only these mean, which is the vocabulary
//!   growth `new-horizon.md` §10.5 forbids.
//!
//!   **The census was run and it answers the question against a rule.** Over
//!   every launch in all three CUDA source trees — `kernels-cuda/csrc/src`,
//!   `driver-cuda/csrc`, `kernels-cuda-new/csrc/src` — parsed by balanced
//!   parentheses rather than by a comma regex:
//!
//!   ```text
//!   launches            240
//!     1-D block         236
//!     2-D block           4     <- all four in gemm/gemv.cu
//!   ```
//!
//!   **And that census UNDER-COUNTS, which is worth writing down beside it
//!   rather than quietly correcting.** It parses the argument list of the
//!   `<<<>>>` itself, so it sees a block only where the block is spelled
//!   INLINE — which is `gemv.cu`'s idiom and not the tree's. A launcher that
//!   declares `const dim3 block(32, kWarps);` and then writes
//!   `<<<grid, block, 0, stream>>>` presents the census with an identifier,
//!   and the census counts it 1-D. Re-derived by looking for the
//!   DECLARATIONS as well: `moe/moe_dispatch.cu:103`, `:130` and `:192`
//!   (`dim3(32, kWarps)` twice and `dim3(32, 8)` once),
//!   `quant/quant_bf16_to_fp8.cu:108` (`dim3(BX, BY)` = 32×8), and the three
//!   multimodal towers, then under `driver-cuda/csrc/vision/`, which shared
//!   one file-scope `dim3 B2(16,16)` across twenty-one launches. (Those three
//!   are Rust now and that directory is deleted; the twenty-one launches are
//!   `tower::gemma4_vision`, `tower::gemma4_audio` and `tower::qwen3_vl`
//!   fires of `LaunchRule::Tile16`, each quoting the `B2`/`G2` line it came
//!   from. The census stands as taken.)
//!
//!   **The conclusion does not move, and the correction is what makes it
//!   solid rather than lucky.** §10.5's bar is *a rule must serve more
//!   kernels than the one that wants it*, and the corrected census is worse
//!   for a rule than the original was: the 2-D blocks that exist are 32×4,
//!   32×8, 32×`kWarps` and 16×16 — four different geometries from four
//!   authors' idioms, with no shared expression of the fire's rectangle
//!   between them. `families::moe`'s header already refuses its own rows over
//!   this exact block AND over a host predicate no [`kernels::Source`] can
//!   spell, so a rule written for `dim3(32, kWarps)` would not even unblock
//!   the family that shares the shape. **So `LaunchRule` does not grow — and
//!   it does not need to.**
//!
//!   [`kernels::LaunchRule`] is not the only way to open a grid. `driver-cuda`'s
//!   `fire/attn_score.rs` builds a `kernels_cuda_new::runtime::Launch`
//!   directly — `grid: [x, y, 1]`, `block: [w, 1, 1]`, `smem` — and fires it
//!   through `KernelModule::fire`, stating a geometry no rule states, for
//!   exactly the reason recorded at its `FOLD_GRID_Y`: *"`gridDim.y` is an
//!   OCCUPANCY FANOUT … that is why no `LaunchRule` states it."* A `Launch`
//!   carries three block axes, so `block: [32, kWarps, 1]` was expressible the
//!   hour that path landed. **A row cannot state this geometry, and a
//!   driver-owned launch is not a row.**
//!
//!   Worth recording how the census was nearly wrong in the OTHER direction
//!   too, because it is the session's recurring shape: the first attempt
//!   matched `<<<[^,>]+, [^,>]+?` and reported **0 two-D blocks out of 235** —
//!   a clean, wrong answer, because `dim3(static_cast<unsigned>(N))` contains
//!   a `<` that ends the character class early. A regex that finds *some*
//!   results does not announce that it is parsing the wrong thing.
//! * ~~**The template arguments come from a device query.**~~ **RETIRED.**
//!   `gemv_unroll_depth()` read `cudaDevAttrComputeCapabilityMajor` to choose
//!   between unroll 2 and unroll 4; the split-K leg was picked by comparing
//!   the row count against `kSplitKMaxRows`. A row names ONE instantiation,
//!   and the launcher picked one per call.
//!
//!   Every clause of that is true and none of it is a wall. A device query is
//!   **device-specific tuning**, which the principle sends to Rust by name;
//!   each instantiation is its own row; and what the sentence "a row names one
//!   instantiation" describes is not a limit, it is the **unit of work**. Four
//!   launches over two templates is four rows and a Rust `match` — `GEMV_ROWS`
//!   below, and `driver-cuda`'s `fire::gemv`.
//! * ~~**The launcher returns `bool`.**~~ **RETIRED.** `K % 8 != 0`, or a
//!   pointer not aligned to 16, and it returned `false` meaning "I did not
//!   launch — use cuBLAS". A row cannot decline, and that is still true.
//!
//!   But nothing asks a row to. The eligibility test happens on the host
//!   BEFORE any launch, so under the principle it is Rust — and it is a
//!   **refusal**, which the design already requires of every failure. The
//!   danger the sentence names is real and is why the check had to move rather
//!   than be dropped: dispatching a row unconditionally would launch the
//!   kernel the C++ refused and read past the buffer it refused over.
//!   `fire::gemv::Gemv::Declined` is that refusal, and it is an enum rather
//!   than a `bool` so that a caller cannot spell "it declined" the same way it
//!   spells "it ran".
//!
//! # Four rows, all `Unstated`, all UNSOURCED, none in `JIT_DISPATCHED`
//!
//! ```text
//! row                            instantiation                       rule       sourced  routed
//! [0] gemv_splitk_bf16_w4_u2     gemv_splitk_bf16_kernel<4, 2>       Unstated   no       no
//! [1] gemv_splitk_bf16_w8_u1     gemv_splitk_bf16_kernel<8, 1>       Unstated   no       no
//! [2] gemv_bf16_w4_u2            gemv_bf16_kernel<4, 2>              Unstated   no       no
//! [3] gemv_bf16_w4_u4            gemv_bf16_kernel<4, 4>              Unstated   no       no
//! ```
//!
//! `Unstated` is the refusal `families::attn`'s `ATTN_SCORE_FOLD_SIGS[0]`
//! already makes and for the same reason: the kernel is named, compiled and
//! resolved, and the vocabulary declines to state a grid it would state
//! WRONGLY. Writing `LaunchRule::Elementwise` here with a caveat comment would
//! be worse than saying nothing, because four readers take `KernelSig::launch`
//! as a contract.
//!
//! **Unsourced to the last operand, and not one operand is a near miss.**
//! These kernels are not reached from a lowered statement at all. `gemv_bf16`
//! was called from `gemm/gemm.cpp`'s dense autotuner — `:544`, `:962`, `:2356`
//! in the numbering `crate::execution::SERVED` cites — where `weight`, `act`,
//! `bias`, `out`, `N`, `K` and `beta` are the arguments of a `gemm::act_x_wt_*`
//! call that has ALREADY been bound, several frames up. There is no
//! `Source::In`, `Source::Out` or `Source::Ctx` that names them from here, and
//! `families::rope` states the rule that follows: a half-bound row is worse
//! than an unbound one, so the WHOLE row is left unsourced and this paragraph
//! is the sentence saying why. None of the four is in
//! [`crate::device::JIT_DISPATCHED`] — naming one there would delete a shim
//! entry that does not exist, skip a dispatch arm that was never emitted, and
//! fail at LINK (§22.1).
//!
//! What a row buys these kernels that a comment would not: `tests/units.rs`
//! compiles [`GEMV`] and resolves all four instantiations through
//! `nvrtcAddNameExpression`/`nvrtcGetLoweredName` on every run against a
//! device. Device text named only in prose is device text nothing compiles.
//!
//! # `abi::emit_device_typecheck` refuses these four, by name
//!
//! It spells every buffer operand as a pointer to the HEAD of `elem`, and
//! these kernels are `template <int kWarps, int kUnrollP>` — the head is
//! `device::i32(4)`, a value. The emitter says so with the row's name attached
//! rather than emitting `const ::pie_cuda_driver::kernels::device::i32(4)*`
//! into a generated file and letting nvcc complain about a line nobody wrote.
//! `families::rope` and `families::layout` are already in that state for the
//! same reason. It is not a gap that can be closed from this file: closing it
//! means templating the kernels over their element type, and these kernels are
//! CARRIED, not authored — `gemv.cuh` is `gemv.cu`'s text and improving it
//! here would make the migration unmeasurable.
//!
//! # THE SPECIFICATION OF THE HOST PROGRAM THESE ROWS SERVE
//!
//! The host half of `gemv_bf16` is Rust in `driver-cuda/src/fire/gemv.rs` —
//! that is where the program LIVES, and this section is its specification
//! stated from the table's side, in the four terms the owner asked for. It is
//! written here because the four rows below are meaningless without it: a row
//! says *this instantiation exists*, and nothing in a row says WHICH of the
//! four a given call must fire, or when none of them may. Read this with
//! `fire/gemv.rs` open; neither is complete alone, and the same four terms are
//! restated there against the code that implements them.
//!
//! (`kernels-cuda`, the AOT archive `gemv.cu` came out of, is reference
//! material for this crate and is being deleted entirely. `driver-cuda` is
//! not — it is where all host code ends up, in Rust.)
//!
//! ## 1. Which JIT units it fires, and in what order
//!
//! **One unit, one fire, and there is no order.** `gemm/gemv` is the only unit
//! involved and exactly ONE of its four rows runs per call. That is worth
//! stating positively rather than skipping, because it is what makes this
//! program small: it is not a composition, it has no sequence, and nothing in
//! [`crate::execution::Step`] is needed to describe it. What it has instead is
//! a 2×2 CHOICE, over two independent host decisions:
//!
//! ```text
//!                          cc major >= 10        cc major < 10
//!   N <= 4096              [0] splitk<4, 2>      [1] splitk<8, 1>
//!   N >  4096              [2] gemv<4, 2>        [3] gemv<4, 4>
//! ```
//!
//! The two axes are not the same kind of thing and must not be collapsed. The
//! ROW axis picks between two different `__global__`s at two different grids —
//! `N` blocks of `32 x kWarps` against `ceil(N/4)` blocks of `32 x 4`. The
//! COLUMN axis picks a template argument on whichever kernel the row axis
//! chose, and on the bottom row it changes ONLY `kUnrollP`, which is why rows
//! [2] and [3] are bit-identical to each other and rows [0] and [1] are not
//! (see `GEMV_ROWS`).
//!
//! ## 2. What intermediate buffers sit between them
//!
//! **None. There is no scratch, no staging and no temporary of any kind**, and
//! this is a claim about all four rows:
//!
//! * The split-K rows reduce across warps through `__shared__ float
//!   partial[kWarps]` — STATIC shared memory, sized by the instantiation and
//!   living inside the block, so the launch's dynamic `smem` is 0 and no
//!   allocation reaches the driver.
//! * The row-per-warp rows reduce through `__shfl_down_sync` alone and touch
//!   no shared memory at all.
//! * `beta != 0` is served IN PLACE: the epilogue reads `out[row]`, scales it
//!   and writes it back, so an accumulating projection needs no second buffer
//!   and no second launch. This is why `beta` is an operand rather than a
//!   caller-side add — refusing it used to push those shapes onto cuBLAS, at
//!   17.3 us against 11.2 for gpt-oss's o_proj.
//! * `bias` is folded into the epilogue rather than run as a second kernel.
//!   The fold is BIT-IDENTICAL to a following `norm::add_bias_bf16`, by
//!   construction — the epilogue computes `bf16(bf16(dot) + bias[n])`, and the
//!   redundant-looking double rounding is what makes it so. Measured over
//!   14,497 values on hostile input by
//!   `driver-cuda/tests/gemm_service_parity.rs`. So the driver has a real
//!   choice here and both answers are correct: fold (one launch) or compose
//!   with `norm::add_bias_bf16` (two). `fire/gemv.rs` folds, because it costs
//!   ~120 launches per decode step on gpt-oss-20b, each ~3.6 us against a
//!   2.2 us empty-launch floor.
//!
//! ## 3. What it decides on the host
//!
//! Four decisions and a refusal set, in the order the C++ made them. Every one
//! of them is host code by principle (1): a device query and a shape threshold
//! are *device-specific tuning*, and an eligibility test made before any
//! launch is a host decision.
//!
//! ```text
//!   #  what                              from                       picks
//!   1  K % 8 != 0, or N <= 0, or K <= 0  operands                   REFUSE
//!   2  weight / act / out is null        operands                   REFUSE
//!   3  weight or act not 16-B aligned    operands                   REFUSE
//!   4  N <= 4096                         operand N, const 4096      row axis
//!   5  cc major >= 10                    cudaDevAttrComputeCapa-    column
//!                                        bilityMajor, cached        axis
//!   6  blocks = ceil(N / 4) > i32::MAX   arithmetic on N            REFUSE
//! ```
//!
//! **The refusal is the part a re-implementation is most likely to get wrong,
//! so it is stated twice.** A refusal means *"I did not launch — use cuBLAS"*.
//! It is not a fallback, not a no-op and not an error: nothing is enqueued,
//! `out` is left exactly as it was found, and the CALLER must still be able to
//! see that it must run the shape some other way. `gemm.cpp` reads it as the
//! last term of a short-circuiting `&&` (see the next section). A driver that
//! turned a decline into a silent success would produce an untouched output
//! buffer and report that it had been written.
//!
//! Decisions 4 and 5 are what pick among instantiations; 1, 2, 3 and 6 pick
//! nothing and only ever refuse. Decision 5 is cached process-wide, which is
//! what the C++'s function-local `static` did and is wrong the same way on a
//! multi-device process. Grid arithmetic: `grid.x` is `N` on the top row and
//! `ceil(N / kWarps)` with `kWarps = 4` on the bottom; `block` is
//! `(32, kWarps, 1)` for all four. **`kWarps` is load-bearing twice** — it is
//! the block's second axis AND the divisor in the grid — and a grid computed
//! with a different number than the block uses leaves the tail rows unwritten
//! silently, because `if (row >= N) return;` cannot tell a short grid from a
//! rounded one.
//!
//! ## 4. What is missing to STATE this, in each vocabulary
//!
//! Named per type, because three of the four are near misses and saying "the
//! vocabulary cannot express it" would hide which half already can.
//!
//! * **[`kernels::LaunchRule`] — a 2-D BLOCK.** Every rule it evaluates fixes
//!   `blockDim.x` at 256 and states `blockDim.y = blockDim.z = 1`. All four
//!   launches are `dim3(32, kWarps)`. The census above says a rule must not
//!   grow for this, and the finding at the top of this module says why a
//!   `Launch` built by the driver is the answer instead. **This gap is
//!   accepted, not open.**
//! * **[`kernels::Source`] — every operand.** These kernels are not reached
//!   from a lowered statement at all; their arguments are bound frames up in a
//!   dense autotuner. Not one of the seven is a near miss. Recorded above.
//! * **[`crate::device::Specialisation`] — HALF of the refusal is already
//!   expressible, and three things are not.** `K % 8 == 0` is
//!   [`crate::device::Term::Multiple`]`{ of: 8 }` and the two pointer tests
//!   are [`crate::device::Term::Aligned`]`{ bytes: 16 }`, exactly as written.
//!   What is missing:
//!   1. **A THRESHOLD term.** `N <= 4096` is a comparison, and the three
//!      terms are `Aligned`, `Multiple` and `Is` — there is no `AtMost`. This
//!      is the same unary-vocabulary wall `execution::Control`'s header
//!      records for `max_global_tokens + history_steps > 8192`, one operand
//!      narrower.
//!   2. **A DEVICE fact.** [`crate::device::Fact`] is derived from the bound
//!      OPERANDS — `Address`, `Int`, `Bool`, `Opaque` — so no `Term` can read
//!      a compute capability. Decision 5 is unspellable, and it is the one
//!      decision principle (1) names by title.
//!   3. **A REFUSING arm.** `Specialisation::choose` answers
//!      `Option<&Arm>`, and `None` means *the base row*, i.e. FIRE ANYWAY.
//!      There is no arm that means *do not fire; the caller must go
//!      elsewhere*. A specialisation cannot decline, and declining is half of
//!      what `gemv_bf16` does.
//! * **[`crate::execution::Execution`] — [`crate::execution::Walk`] is the
//!   near miss and misses in three places.** `Jit` is one row and this is
//!   four; `Composed` is a fixed `&[Step]` and this is a choice, not a
//!   sequence; `Service` is somebody else's. `Walk` fits best — it is a host
//!   program whose shape comes from the input, and
//!   [`crate::execution::Walk::refuses`] exists precisely because *a walk that
//!   cannot say no is a walk that will guess*. What stops an entry being
//!   written today:
//!   1. **`Walk::symbol` must be a row of [`crate::table`], and there is
//!      none.** `gemm::gemv_bf16` was never a stated symbol — it was reached
//!      from inside `gemm.cpp`'s tactic runner, never from a lowered
//!      statement — and `table::gemm::KERNELS` has no row for it. A `Walk`
//!      needs a name to hang on and this program has none.
//!   2. **[`crate::execution::Control`] has no PRODUCT shape.**
//!      `Switch { on }` carries ONE discriminant; this program switches on a
//!      shape AND on a device attribute, independently, giving four arms that
//!      are a 2×2 product and not a chain. Writing `Switch { on: "N" }` would
//!      be true and would lose the axis that principle (1) exists to name.
//!   3. **`Control` cannot read the DEVICE either** — same gap as
//!      `Specialisation`'s (2), in a second vocabulary, which is the evidence
//!      that it is one gap and not two.
//!
//!   Per that type's own bar, none of these is a variant to add today: the
//!   shape is real and the member is one program. Whoever lands the second
//!   device-selected walk adds `Control`'s device arm with both members in the
//!   same edit.
//!
//! # `gemm/gemm.cpp` is not this module's, and what it now needs
//!
//! `gemm.cpp` is host C++ compiled by `g++` — cuBLASLt plumbing and dispatch,
//! no `__global__`, no `<<<>>>` of its own — and it is separately owned by the
//! cuBLAS rewrite. It `#include`s `gemm/gemv.hpp` at `:23` and calls
//! `kernels::gemm::gemv_bf16` twice:
//!
//! ```text
//!   gemm.cpp:560  run_dense_tactic   case GemmKind::Gemv, the BIASED arm --
//!                                    `(beta == 0.f || beta == 1.f) && M == 1
//!                                     && cublas_stream(handle, stream)
//!                                     && gemv_bf16(W, act, bias, y, N, K,
//!                                                  stream, beta)`
//!   gemm.cpp:978  gemm_bf16_impl     the unbiased fast path --
//!                                    `M == 1 && beta == 0.f && ...
//!                                     && gemv_bf16(W, act, nullptr, y, N, K,
//!                                                  gemv_stream)`
//! ```
//!
//! **The declaration is deleted with the header, so those two call sites must
//! go to `driver_cuda::fire::gemv::gemv_bf16`.** What that costs, precisely:
//!
//! * The argument list is unchanged and in the same order —
//!   `(weight, act, bias, out, n, k, stream, beta)` — except that `beta` had a
//!   `= 0.f` default in C++ and Rust has none, so `:978` must pass `0.0`
//!   explicitly, which is the value its own `beta == 0.f` guard already
//!   proves.
//! * The return is `fire::gemv::Gemv`, not `bool`. Both sites use the answer
//!   as the last term of a short-circuiting `&&`, so each needs
//!   `matches!(.., Gemv::Launched)` — or, from C++, an `extern "C"` wrapper
//!   that maps `Launched` to `true` and `Declined(_)` to `false`. **A DECLINE
//!   MUST STAY VISIBLE.** Both sites read `false` as *"I did not launch, fall
//!   through to cuBLAS"*, and both are correct only because a declined call
//!   enqueues nothing; turning it into a no-op would leave `y` untouched and
//!   report success.
//! * `cublas_stream(handle, stream)` still has to run FIRST, at both sites.
//!   The Rust launcher takes the stream it is given and does not ask cuBLAS
//!   for one.
//!
//! Nothing here edits them; this paragraph is the report.
//!
//! It was nine launches over three templates until §45. `gemv3_bf16` (the
//! fused Q/K/V triple, whose row §27 had already deleted) and the three sweep
//! entry points `gemv_bf16_tuned`, `gemv3_bf16_tuned` and `gemv_splitk_tuned`
//! were reachable from NO root — the harness the sweeps name,
//! `driver/cuda/bench/gemv_bench.cu`, is in no source directory of this
//! repository — and `gemv3_bf16_kernel` went with its two callers. That is why
//! this unit hosts two templates and not three.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// `gemm`'s device text: the row-per-warp GEMV and its split-K twin.
pub const GEMV: Unit = Unit {
    name: "gemm/gemv",
    root: include_str!("../../csrc/src/gemm/gemv.cuh"),
    rows: GEMV_ROWS,
    options: &[],
};

/// The units `gemm` compiles.
pub static UNITS: &[Unit] = &[GEMV];

/// [`GEMV`]'s instantiations — one per `<<<>>>` the deleted launcher held.
///
/// `elem` is an ARGUMENT LIST and its head is a NON-TYPE, because both
/// templates are `template <int kWarps, int kUnrollP>` and neither takes a
/// type parameter at all. `DeviceKernel::instantiation` pastes the string
/// between the angle brackets and glues `::pie_cuda_driver::kernels::` to its
/// FRONT — to the first token only — so the first argument has to be a
/// QUALIFIED constant expression and cannot be a bare literal. `device::i32(4)`
/// is `pie_device.cuh:463`'s alias applied as a functional cast, which is the
/// spelling `families::rope` measured through `nvrtcAddNameExpression` and
/// `nvrtcGetLoweredName`; a bare `4` would expand to
/// `::pie_cuda_driver::kernels::4` and fail with *expected an identifier*.
/// The SECOND argument needs no such care, which is why it is a plain `2`.
///
/// # THE MEASUREMENTS THAT CHOSE THESE FOUR
///
/// Carried here, on the rows, and not only in the driver that fires them.
/// `driver-cuda` is reference material and is going to be deleted; a
/// measurement that lives only in its `fire/gemv.rs` dies with it, and a port
/// that consumes a measurement is worse than a port that fails, because the
/// number cannot be recovered by reading the code it justified. Everything
/// below came from `gemm/gemv.cu`'s comments, which are also gone.
///
/// ## Why the kernel exists at all
///
/// M=1 is the decode shape: one activation row against the whole weight, so
/// there is no reuse for a tiled GEMM to exploit and the call is a pure
/// streaming read. cuBLAS picks kernels sized for an M worth filling and
/// reaches roughly half of HBM bandwidth on these. Against `cublasGemmEx`,
/// bf16, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, A100-SXM4-80GB:
///
/// ```text
///   N=2048  K=4096   17.97 -> 9.47 us   (934 -> 1771 GB/s)
///   N=4096  K=2048   15.42 -> 9.24 us   (1088 -> 1815 GB/s)
///   N=8192  K=2048   24.91 -> 20.71 us  (1347 -> 1620 GB/s)
///   N=32    K=2048    8.19 -> 4.45 us   (launch-floor bound either way)
/// ```
///
/// ## The COLUMN axis: `kUnrollP`, and why it is read from the device
///
/// The unroll exists to keep several loads in flight. Written flat, each lane
/// has exactly ONE load in flight — the FMA on `w4[i]` is the next
/// instruction after it — and a warp waiting on a single HBM round trip
/// cannot cover the latency however many warps the SM holds; measured on an
/// H100 at gpt-oss's o_proj (N=2880, K=4096, 23.6 MB/layer) that version
/// sustained ~963 GB/s.
///
/// **Blackwell wants a SHALLOWER unroll than Hopper, which is the opposite of
/// what the kernel's own comment predicts**, so the depth is a device fact and
/// not a constant. On B200 four is past diminishing returns and costs more
/// than it buys; two covers the latency and leaves registers for occupancy.
/// Measured cold, 8 rotating buffers, bf16, M=1:
///
/// ```text
///   shape                     unroll=4          unroll=2
///   qwen27 gate/up      34.2us 5.22TB/s   30.7us 5.80TB/s   -10.2%
///   qwen27 down         35.3us 5.06TB/s   32.2us 5.54TB/s    -8.8%
///   gemma31 down        43.3us 5.34TB/s   39.1us 5.91TB/s    -9.7%
///   gptoss lm_head     227.4us 5.09TB/s  194.5us 5.95TB/s   -14.5%
/// ```
///
/// This kernel is 78% of Qwen3.6-27B's decode step and 77% of gemma-4-31B's.
/// **Hopper and below keep 4**: that is where the depth was tuned and nothing
/// has re-measured it on that part.
///
/// ## The split-K column: `w=4,u=2` against `w=8,u=1`, under GRAPH REPLAY
///
/// On the top row the same device fact changes MORE than the unroll — rows
/// [0] and [1] differ in warp count too. Two earlier attempts got this wrong
/// the same way: both swept EAGER, where the launch floor on that box is
/// ~4.1 us and most of these shapes run under it, so the sweep compared launch
/// overhead rather than kernels. The first shipped a blanket `warps = 2` and
/// cost gemma-4-26B 3.4% and Qwen3.6-27B 1.9%; the second papered over it with
/// a size threshold. Timed the way pie actually decodes — inside a captured
/// graph — `w=8,u=1` is the WORST config on 11 of 12 shapes:
///
/// ```text
///   shape                MB   w8u1   w4u2   per-shape best
///   qwen35 q_proj      16.8   5.45   4.00   3.79 (w2u1)
///   qwen35 o_proj      16.8   4.32   3.45   3.45
///   qwen35 lin qk       8.4   3.52   2.65   2.65
///   qwen27 gdn qk      21.0   5.22   4.17   3.91 (w2u2)
///   gptoss o_proj      23.6   6.25   4.86   4.80 (w2u1)
///   gemma31 kv_proj    44.0  11.35   9.25   9.25
///   gemma26 q_proj     23.1   7.34   5.48   4.85 (w2u2)
/// ```
///
/// Summed over all twelve, `w=4,u=2` is **20% faster** than `w=8,u=1` and
/// within 3% of picking the best config per shape — one config, no threshold.
/// Its only loss is gpt-oss's 0.2 MB router, 1.6% of 1.9 us. **Hopper keeps
/// `w=8,u=1`**, which is where it was tuned, which is why the arm hangs off
/// the compute capability and not off a shape.
///
/// ## Which pairs are BIT-IDENTICAL, and which are not
///
/// This decides what a re-implementation is allowed to collapse. Measured on
/// an L40S, nine shapes, both arms fired against byte-identical inputs with a
/// poisoned output buffer, a permutation control moving ~89% of the weight
/// bytes and a truncation control leaving half the output poison, all firing
/// at every shape (`new-horizon.md` §36):
///
/// ```text
///   arms                         benign data   wide-exponent data
///   [2] gemv<4,4> vs [3] <4,2>      0 bytes       0 bytes
///   [0] splitk<8,1> vs [1] <4,2>    0 bytes       5 bytes / 9 shapes
///   (deleted gemv3<8,1> vs <2,2>)   0 bytes       3 bytes / 9 shapes
/// ```
///
/// The unroll depth alone is safe BY CONSTRUCTION and measured so: at
/// `kUnroll` 4 and 2 a lane visits the same vectors in the same order
/// (`i`, `i+32`, `i+64`, …), so the fp32 accumulation is the same additions in
/// the same sequence. **The warp count is not**: eight warps partition K at
/// stride 256 and four at stride 128, and the shared-memory tree then sums
/// different partials. That is below bf16's last bit on model-shaped weights,
/// which is why the benign column is zero, and it is not below it once the
/// exponents spread. So rows [2] and [3] MAY be collapsed on a machine whose
/// capability is known; rows [0] and [1] may not, and neither pair may be
/// selected by anything a replay cannot reproduce — this is why
/// `PIE_GEMV_B200_TUNING` was deleted rather than kept as an override.
///
/// ## The ROW axis: `N <= 4096`, and why 4096 was not moved
///
/// `kSplitKMaxRows` read `getenv("PIE_GEMV_SPLITK_MAX_ROWS")` until §36 folded
/// it at its unchanged default. It is a threshold, not a toggle: the two sides
/// are two different `__global__`s at different grids, so the variable chose
/// which kernel ran at all. Under graph replay the two disagree in TIME in
/// both directions:
///
/// ```text
///   N        [1] splitk<8,1>   [3] gemv<4,4>          (L40S, 142 SM)
///   32          1.48 us           2.47 us     split-K 1.67x
///   512         2.39              3.09        split-K 1.29x
///   2048        3.47              3.32        row-per-warp 1.05x
///   4096        5.32              4.73        row-per-warp 1.12x
///   8192        9.10              7.59        row-per-warp 1.20x
/// ```
///
/// So on that part the crossover is near 2048 — 2048 rows is 512 blocks, about
/// four per SM, which is where the row-per-warp form stops being the
/// constraint — and the shipping 4096 takes the slower kernel over
/// `2048 < N <= 4096`. **4096 was NOT changed by the migration**: it is what
/// every deployment runs today, the tables it came from were taken on a
/// 132-SM B200, and moving it is a separate claim that wants a B200 to make.
/// A port that changes a tuning constant makes the parity run that would have
/// checked it meaningless.
static GEMV_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GEMV_SIGS[0],
        template_path: "gemm::device::gemv_splitk_bf16_kernel",
        elem: "device::i32(4), 2",
    },
    DeviceKernel {
        sig: &GEMV_SIGS[1],
        template_path: "gemm::device::gemv_splitk_bf16_kernel",
        // `<8>` at the launcher, taking `kUnrollP`'s default. Stated in full
        // here: a default argument is the TEMPLATE's fact and a row is a
        // contract, and `<8>` and `<8, 1>` are one instantiation only for as
        // long as nobody edits the default.
        elem: "device::i32(8), 1",
    },
    DeviceKernel {
        sig: &GEMV_SIGS[2],
        template_path: "gemm::device::gemv_bf16_kernel",
        elem: "device::i32(4), 2",
    },
    DeviceKernel {
        sig: &GEMV_SIGS[3],
        template_path: "gemm::device::gemv_bf16_kernel",
        elem: "device::i32(4), 4",
    },
];

/// The contracts, in [`GEMV_ROWS`]' order.
///
/// All four state the same seven operands, because they are two templates over
/// one parameter list and the template arguments are in the INSTANTIATION, not
/// in the signature. The stream is absent: a stream is `cuLaunchKernel`'s sixth
/// parameter and not a member of the `void**`. `N` is present, unlike
/// `attn_score_fold_heads`' `num_requests`, because both kernels READ it —
/// `if (row >= N) return;` is the guard that makes a rounded-up grid safe, and
/// a row that dropped it would bind six values into a seven-parameter kernel.
#[rustfmt::skip]
static GEMV_SIGS: [KernelSig; 4] = [
    // `gemm/gemv.cu:344-346`:
    //
    //     gemv_splitk_bf16_kernel<kSplitWarpsB, /*kUnrollP=*/2>
    //         <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarpsB), 0,
    //            stream>>>(
    //
    // `kSplitWarpsB = 4`. One block per output row, four warps splitting K,
    // reducing through `__shared__ float partial[4]` — STATIC shared memory,
    // so the launch's dynamic `smem` is 0 and not 16.
    kernel!(gemv_splitk_bf16_w4_u2 "gemm::gemv_splitk_bf16_w4_u2",
        file = Some("gemm/gemv.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            weight: Buf,
            act: Buf,
            bias: Buf | null,
            out: BufMut,
            n: I32,
            k: I32,
            beta: F32,
        ]),
    // `gemm/gemv.cu:355-357`:
    //
    //     gemv_splitk_bf16_kernel<kSplitWarps>
    //         <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarps), 0,
    //            stream>>>(
    //
    // `kSplitWarps = 8`, `kUnrollP` defaulted to 1. The pre-Blackwell arm of
    // the same leg, and the one `fire::gemv` fires on this box.
    kernel!(gemv_splitk_bf16_w8_u1 "gemm::gemv_splitk_bf16_w8_u1",
        file = Some("gemm/gemv.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            weight: Buf,
            act: Buf,
            bias: Buf | null,
            out: BufMut,
            n: I32,
            k: I32,
            beta: F32,
        ]),
    // `gemm/gemv.cu:372-374`:
    //
    //     gemv_bf16_kernel<kWarps, 2>
    //         <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0,
    //            stream>>>(
    //
    // `kWarps = 4`, `blocks = (N + kWarps - 1) / kWarps`. One warp per output
    // row, so the grid is a ROUNDED-UP row count and `if (row >= N) return;`
    // is what makes the last block safe.
    kernel!(gemv_bf16_w4_u2 "gemm::gemv_bf16_w4_u2",
        file = Some("gemm/gemv.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            weight: Buf,
            act: Buf,
            bias: Buf | null,
            out: BufMut,
            n: I32,
            k: I32,
            beta: F32,
        ]),
    // `gemm/gemv.cu:382-383`:
    //
    //     gemv_bf16_kernel<kWarps, 4>
    //         <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0, stream>>>(
    //
    // The same grid and block as `[2]` at a deeper unroll — the two differ in
    // INSTANTIATION and in nothing else, which is exactly the claim
    // `gemv_unroll_depth()` rested on: at `kUnroll` 4 and 2 a lane visits the
    // same vectors in the same order, so the fp32 accumulation is the same
    // additions in the same sequence and the two arms are bit-identical.
    kernel!(gemv_bf16_w4_u4 "gemm::gemv_bf16_w4_u4",
        file = Some("gemm/gemv.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            weight: Buf,
            act: Buf,
            bias: Buf | null,
            out: BufMut,
            n: I32,
            k: I32,
            beta: F32,
        ]),
];
