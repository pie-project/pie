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
//!
//! ---------------------------------------------------------------------
//!
//! # `gemm` IN FN-WORLD — north star §5 step 5
//!
//! The header above is what this file said while it was `families/gemm.rs`
//! and it is unchanged, because every word of it is still true: the unit is
//! still `gemv.cuh`'s two templates and four instantiations, and the census
//! that retired the three "no row can state this" findings is still the
//! census. What changed is everything AROUND it.
//!
//! **This family's point is the host program.** Eleven of its twelve symbols
//! never had a device row in this crate at all — five are collectives that
//! call NCCL or a peer-to-peer all-reduce, three are cuBLASLt matmuls over
//! quantized weights, three are `cublasGemmEx` — and the twelfth, the GEMV,
//! is the only `__global__` here. §5 step 5's sentence for this family is
//! therefore *"fns move in from `fire/` and `bind/service.rs`"*, and the
//! fns that moved are the big ones:
//!
//! * [`dense`] — `driver-cuda/src/fire/gemm.rs`, whole. The runtime
//!   autotuner, the cuBLASLt plan cache, the on-disk tactic cache and
//!   [`dense::GemmKind`]. Moved by `git mv`; the only edits are the two
//!   `use` paths and the doc links that named driver modules.
//! * [`gemv`] — `driver-cuda/src/fire/gemv.rs`, whole, same treatment, plus
//!   one substantive change recorded at [`gemv`]'s `unroll_depth`: it asked
//!   `driver_cuda::device::Device` for the compute capability and cannot
//!   from here, so it asks the driver API directly, the way
//!   `runtime::cache`'s `arch()` already does.
//! * [`act_x_wt_bf16`], [`act_x_wt_bf16_out_fp32`],
//!   [`grouped_act_x_wt_bf16`] and [`act_x_wt_bias_bf16`] —
//!   `driver-cuda/src/bind/service.rs`, which held them because a
//!   `RUST_SERVED` row's generated arm had to find them by name in that
//!   file. Nothing generated finds them now; a bind calls them.
//!
//! ## What did NOT move, and why that is not a half-port
//!
//! `driver-cuda/src/bind/quant_gemm.rs` — the three `WeightView` routers.
//! It reaches `fire::quant_int8` for its dequant kernels, which is
//! `quant`'s family and another agent's port, and moving it would take
//! `quant`'s host programs with it. It stays, and **nothing regresses by
//! that**: none of the three states a `Source` on any operand, so
//! `emit_dispatch` never wrote them an arm and no trace has ever been able
//! to fire one.
//!
//! The five collectives are the same case for a different reason: their
//! first operand is an `NcclComm` or a `CustomAllReduce`, both of which are
//! the driver's own objects with a lifecycle no `Cx` describes. Their host
//! programs stay in `driver-cuda/src/fire/all_reduce.rs` and their
//! contracts here say so.
//!
//! ## THIS FAMILY IS THE THIRD REGISTRATION SHAPE — `SIGS` and not `FAMILIES`
//!
//! `x::SIGS`' own table names three shapes, and `gemm` is entirely the
//! third: **a driver op contributes to `SIGS` and not to `FAMILIES`.** There
//! is no `bind!` in this file and no `ENTRIES` static, deliberately, and the
//! long comment where a `bind!` would have gone records the twelve reasons.
//!
//! The forcing fact is the cuBLAS handle. The dense symbols' rows bound it
//! as `handle: CublasHandle <- Source::Ctx("cublas")`, which reads
//! `DispatchCtx::cublas` — the engine's handle, created once at boot by
//! `driver_cuda::device::cublas`, whose own doc records why it is created
//! once (`cublasDestroy` costs **3.2 ms**, three quarters of what a warm
//! decode spent being issued) and that the stream is rebound per fire. A
//! bind is handed a [`Cx`](crate::x::Cx), which has no method for it, and
//! **widening `Cx` to reach a device API would undo the one safety property
//! the whole floor rests on** — §3.3's forbidden surface, a handle with a
//! settable stream, math mode and workspace.
//!
//! So the driver keeps firing them, exactly as it does today, and the
//! contracts here exist so that `model-compiler` can read a row and *must
//! not be able to tell what serves it*. Every public `fn` below takes
//! `handle: *mut c_void` first, unchanged from `bind/service.rs`, because it
//! always did; the driver passes `ctx.cublas`, the way
//! `pie_lora_qkv_correction` passes it at `bind/mod.rs:1895`.
//!
//! ## `Fired` and `Gemv`, which are NOT collapsed
//!
//! §5.1 asks for these two to be reconciled or for the reason not to be
//! stated. The reason not to: [`gemv::Decline`] has four arms —
//! `Shape`, `Null`, `Misaligned`, `Grid` — and
//! [`Refusal`](crate::x::Refusal) has no spelling for the last two.
//! `Misaligned` is *"weight or act was not 16-byte aligned"*
//! (`gemv.cu:313`) and `Grid` is *"the row-per-warp grid would not fit"*
//! (`gemv.cu:381`); mapping either onto `Unstated` would turn a fact about
//! POINTERS into a claim about what a statement carries, which is a
//! different sentence. They are also not the same audience: a `Refusal`
//! is printed to whoever loaded the model, and a `Decline` is read by
//! [`dense::act_x_wt_bf16`]'s fallback ladder, which does not print
//! anything and does need to know which. So `Gemv` stays the ladder's
//! private answer, [`Fired`](crate::x::Fired) is what a bind's `fn`
//! returns, and the conversion happens exactly once, in
//! [`act_x_wt_bf16`], where the ladder ends.
//!
//! ## Two kernels in one body — §2.3's `Composed`, exercised
//!
//! [`act_x_wt_bias_bf16`] is a `gemm::act_x_wt_bf16` and then a
//! `norm::add_bias_bf16` over the result, which is exactly what
//! `execution::COMPOSED` stated for it, citing `gemm.cpp:2395-2398`. In
//! fn-world it is a two-call body and needs no `Composed` type, no `Take`,
//! and no `Composition::agrees` — see that function's own note for what
//! that costs and what it retires.

#![allow(clippy::too_many_arguments)]

use crate::x::abi::{MaybeConst, bf16};
use crate::x::contract::{Fired, Refusal};
use crate::x::launch::Launch;
use crate::{contract, unit};

use core::ffi::c_void;

/// The dense matmul's host program: the autotuner, the plan cache and the
/// on-disk tactic cache.
#[cfg(feature = "_cuda")]
pub mod dense;
/// The GEMV's host program: the four instantiations' selection and launch.
#[cfg(feature = "_cuda")]
pub mod gemv;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
// ---------------------------------------------------------------------------

unit! {
    /// `gemm`'s device text: the row-per-warp GEMV and its split-K twin.
    ///
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
    unit GEMV = "gemm/gemv",
        text = include_str!("../../csrc/src/gemm/gemv.cuh"),
        file = "gemm/gemv.cuh";

    /// `gemv.cuh` — the SPLIT-K form: one block per output row, `kWarps`
    /// warps splitting K between them, reducing through
    /// `__shared__ float partial[kWarps]`.
    ///
    /// The seven operands are the `__global__`'s parameter list in its own
    /// order, and they are the same seven at either instantiation, because
    /// the two rows differ in TEMPLATE ARGUMENTS and in nothing else. The
    /// stream is absent: a stream is `cuLaunchKernel`'s sixth parameter and
    /// not a member of the `void**`. `n` is present, unlike
    /// `attn_score_fold_heads`' `num_requests`, because both kernels READ it
    /// — `if (row >= N) return;` is the guard that makes a rounded-up grid
    /// safe, and a row that dropped it would bind six values into a
    /// seven-parameter kernel.
    fn gemv_splitk = "gemm::device::gemv_splitk_bf16_kernel" (
        weight: *const bf16,
        act: *const bf16,
        bias: MaybeConst<bf16>,
        out: *mut bf16,
        n: i32,
        k: i32,
        beta: f32,
    ) {
        /// `gemm/gemv.cu:344-346`:
        ///
        /// ```text
        /// gemv_splitk_bf16_kernel<kSplitWarpsB, /*kUnrollP=*/2>
        ///     <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarpsB), 0,
        ///        stream>>>(
        /// ```
        ///
        /// `kSplitWarpsB = 4`. One block per output row, four warps splitting
        /// K, reducing through `__shared__ float partial[4]` — STATIC shared
        /// memory, so the launch's dynamic `smem` is 0 and not 16.
        "gemm::gemv_splitk_bf16_w4_u2" => "device::i32(4), 2",
        /// `gemm/gemv.cu:355-357`:
        ///
        /// ```text
        /// gemv_splitk_bf16_kernel<kSplitWarps>
        ///     <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarps), 0,
        ///        stream>>>(
        /// ```
        ///
        /// `kSplitWarps = 8`, `kUnrollP` defaulted to 1. The pre-Blackwell arm
        /// of the same leg, and the one [`gemv`] fires on this box.
        ///
        /// `<8>` at the launcher, taking `kUnrollP`'s default. Stated in full
        /// here: a default argument is the TEMPLATE's fact and a row is a
        /// contract, and `<8>` and `<8, 1>` are one instantiation only for as
        /// long as nobody edits the default.
        "gemm::gemv_splitk_bf16_w8_u1" => "device::i32(8), 1",
    }

    /// `gemv.cuh` — the ROW-PER-WARP form: one warp per output row, `kWarps`
    /// rows per block, the grid a rounded-up row count.
    ///
    /// The same seven operands as [`gemv_splitk`](raw::gemv_splitk), for the
    /// same reason: two templates over one parameter list.
    fn gemv = "gemm::device::gemv_bf16_kernel" (
        weight: *const bf16,
        act: *const bf16,
        bias: MaybeConst<bf16>,
        out: *mut bf16,
        n: i32,
        k: i32,
        beta: f32,
    ) {
        /// `gemm/gemv.cu:372-374`:
        ///
        /// ```text
        /// gemv_bf16_kernel<kWarps, 2>
        ///     <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0,
        ///        stream>>>(
        /// ```
        ///
        /// `kWarps = 4`, `blocks = (N + kWarps - 1) / kWarps`. One warp per
        /// output row, so the grid is a ROUNDED-UP row count and
        /// `if (row >= N) return;` is what makes the last block safe.
        "gemm::gemv_bf16_w4_u2" => "device::i32(4), 2",
        /// `gemm/gemv.cu:382-383`:
        ///
        /// ```text
        /// gemv_bf16_kernel<kWarps, 4>
        ///     <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0, stream>>>(
        /// ```
        ///
        /// The same grid and block as the row above at a deeper unroll — the
        /// two differ in INSTANTIATION and in nothing else, which is exactly
        /// the claim `gemv_unroll_depth()` rested on: at `kUnroll` 4 and 2 a
        /// lane visits the same vectors in the same order, so the fp32
        /// accumulation is the same additions in the same sequence and the two
        /// arms are bit-identical.
        "gemm::gemv_bf16_w4_u4" => "device::i32(4), 4",
    }
}

// ---------------------------------------------------------------------------
// WHERE THE cuBLAS HANDLE COMES FROM, and why this file does not hold one.
//
// An earlier draft of this port published the engine's `cublasHandle_t`
// into a `static AtomicPtr` here so that a `bind!` body could read it. That
// was wrong, and the reason is `x::route`'s: a bind is handed a `Cx`, and
// `Cx` is deliberately narrow. Reaching around it with a static does not
// widen `Cx` — it routes around the one safety property the floor rests on,
// which is worse than widening it, because a widening is reviewable and a
// static is not.
//
// The floor's answer is the third registration shape: a symbol whose host
// program needs a driver-owned resource is a DRIVER OP. It contributes a
// `contract!` to `x::SIGS` and NO `Entry` to `x::FAMILIES`, `x::entry()`
// answers `None`, and `x::route` reaches its `Service::DriverOp` arm. The
// driver calls the `fn` with `ctx.cublas` in hand. That is what every
// public `fn` below expects: `handle: *mut c_void` is the FIRST parameter
// of all four, unchanged from `bind/service.rs`, because it always was.
//
// The measurement that made a second handle unacceptable is worth keeping
// where a reader of this file will find it. `driver_cuda::device::cublas`
// records that `cublasDestroy` costs **3.2 ms** and was three quarters of
// what a warm decode spent being issued, and that creating a handle per
// fire also meant a fresh workspace allocation each time, which is the part
// that actually takes the time. The driver's handle is created once per
// shell and carries `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)`. A handle of
// this module's own would have been a SECOND one and would have had to
// re-state the math mode.
//
// The stream is not published either, and must not be: the handle's stream
// is rebound per fire by the driver (`serve::state`: *"the stream is
// rebound per fire instead, which is what `cublasSetStream` is for"*), and
// every host program below either takes the stream as an argument or reads
// it back off the handle exactly as it did in `driver-cuda`.

// ---------------------------------------------------------------------------
// Truth two: the host programs. The two big ones are `dense` and `gemv`,
// moved whole; these four are the entry points a bind calls, moved from
// `driver-cuda/src/bind/service.rs`, each returning `Fired` so that "it
// declined" cannot be spelled like "it ran".
// ---------------------------------------------------------------------------

/// `gemm::act_x_wt_bf16` — the dense matmul, tactic-selected.
///
/// Moved from `driver-cuda/src/bind/service.rs:458`, which was a one-line
/// forwarder into [`dense::act_x_wt_bf16`] and is now a two-line one: the
/// forward, and the conversion of the ladder's answer into [`Fired`].
///
/// # Why the handle is a parameter and not a fact
///
/// The row stated `handle: CublasHandle <- Source::Ctx("cublas")` and the
/// emitted arm passed both `ctx` and the bound handle — the same redundancy
/// [`act_x_wt_bias_bf16`] documented, and for the same reason: the
/// composition took this row as its first step and `Composition::agrees`
/// type-checked `Take::From(i)` against the operands as stated. **Neither
/// the composition type nor the type-check exists any more** — the
/// composition is a two-call `fn` body — so the parameter survives for the
/// plainer reason that a host program takes what it needs.
///
/// # Safety
///
/// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
/// and outlive the launch — asynchronous on the handle's stream, so
/// "outlive" ends at the next synchronisation and not at this call's return.
#[cfg(feature = "_cuda")]
pub unsafe fn act_x_wt_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Fired {
    // SAFETY: the caller's obligation, above.
    unsafe { dense::act_x_wt_bf16(handle, act, w, y, m, n, k, beta) }
    Fired::Launched
}

/// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
///
/// Moved from `driver-cuda/src/bind/service.rs:120`. The body went to
/// [`dense::act_x_wt_bf16_out_fp32`], where `COMPUTE`, `ALGO_TENSOR_OP` and
/// `check` already live; what is here is the `Fired` conversion and the
/// handle, which the service read off its `DispatchCtx` and this takes as an
/// argument because there is no `DispatchCtx` in this crate.
///
/// # Safety
///
/// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
/// address `M*N` live floats, and all three must outlive the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn act_x_wt_bf16_out_fp32(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Fired {
    // SAFETY: the caller's obligation, above.
    unsafe { dense::act_x_wt_bf16_out_fp32(handle, act, w, y, m, n, k) }
    Fired::Launched
}

/// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
///
/// Moved from `driver-cuda/src/bind/service.rs:181`, body and all, into
/// [`dense::grouped_act_x_wt_bf16`]. It already took a handle rather than a
/// `DispatchCtx` and the service's own note said why: *"Its row states
/// `Source::Unbound` for every operand — a group boundary is fire-global and
/// no `Source` names one — so `emit_dispatch` writes no arm for it and its
/// only consumer is `fire::lora`'s hand-written staged apply, which holds a
/// `cublasHandle_t` and no context."* That consumer is unchanged and now
/// calls this.
///
/// `group_count <= 0` is a [`Refusal::Empty`], where the service returned
/// silently. Same behaviour, named: the C++ `if (group_count <= 0) return;`
/// could not be told from a launch by its caller.
///
/// # Safety
///
/// The three pointer arrays must be HOST arrays of `group_count` device
/// addresses (cuBLAS reads them on the host for the grouped form), and
/// `m_array_host` a host array of `group_count` row counts.
#[cfg(feature = "_cuda")]
pub unsafe fn grouped_act_x_wt_bf16(
    handle: *mut c_void,
    act_ptrs_host: *const *const c_void,
    w_ptrs_host: *const *const c_void,
    y_ptrs_host: *const *mut c_void,
    m_array_host: *const i32,
    group_count: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Fired {
    if group_count <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_count" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dense::grouped_act_x_wt_bf16(
            handle,
            act_ptrs_host,
            w_ptrs_host,
            y_ptrs_host,
            m_array_host,
            group_count,
            n,
            k,
            beta,
        );
    }
    Fired::Launched
}

/// `gemm::act_x_wt_bias_bf16` — TWO KERNELS IN ONE BODY.
///
/// Moved from `driver-cuda/src/bind/service.rs:527`. `execution::COMPOSED`
/// stated this row step for step and cited `gemm.cpp:2395-2398` for it: a
/// `gemm::act_x_wt_bf16` and then a `norm::add_bias_bf16` over the result.
///
/// # This is §2.3's `Composed`, and it needed none of `Composed`
///
/// North star §2.3 asks whether a body that fires two DIFFERENT kernels
/// needs a type. This is that body, and the answer here is that it does not:
/// two `unsafe` blocks in sequence, the second reading what the first wrote,
/// with the ordinary Rust control flow between them (`if bias.is_null() {
/// return … }`). What the row world needed — `Composition`, `Take::From(i)`,
/// `Composition::agrees` type-checking the second step's operands against
/// the first's — existed to express in DATA what a `fn` expresses in CODE,
/// and every one of those three is deleted with the row.
///
/// The one thing it costs, stated rather than measured away: the second
/// kernel is fired **by symbol**. It goes through `norm`'s own `raw::` stub,
/// so it is fully typed at this end, but the stub in turn calls
/// [`crate::x::fire::fire`], which resolves `norm::add_bias_bf16` through
/// `unit_of` GLOBALLY — so this body depends on `norm`'s device text being
/// in some unit, but NOT on `norm` having been ported, routed, or
/// declared. That was deliberate in the driver version
/// (*"routing it is someone else's change and this one must not depend on
/// the order"*) and is deliberate here: `norm` is another agent's family.
///
/// # What is lost, exactly
///
/// The archive had a second arm: at `M == 1` with a bias, it asked
/// `dense_tactic_for` whether the tuner's chosen tactic could absorb the
/// bias into its epilogue, and `run_dense_tactic` declines every tactic
/// except the warp-per-row GEMV. So the fused arm fired **only** on the
/// GEMV, and its kernels state what they compute:
/// `out[n] = bf16(bf16(dot) + bias[n])`, the double rounding deliberate,
/// *"bit-identical to running `add_bias_bf16` afterwards"*. (That was
/// `gemv.hpp`'s wording; the header is deleted and the sentence is now at
/// both epilogues of `csrc/src/gemm/gemv.cuh`, which is the text NVRTC
/// compiles.) The composition therefore produces THE SAME BYTES and costs
/// one extra launch per biased `M == 1` projection.
///
/// That is the whole cost and it is stated rather than measured away: the
/// fusion was worth 11.9% of gpt-oss-20b's decode time when it was added
/// (`gemm.hpp`), and what buys it back is a bias epilogue on a JIT'd GEMV.
/// **That kernel now exists** — the `gemm/gemv` unit's four rows all take
/// `bias` and fold it, and [`gemv::gemv_bf16`] passes it through — so what
/// is missing is no longer a kernel but a caller that reaches it at `M == 1`
/// instead of reaching the dense ladder. **That enumeration now exists** —
/// [`dense`] — so the remaining work is a [`dense`] entry that takes a
/// `bias` and, when the tuned tactic for the shape is `GemmKind::Gemv`,
/// passes it down instead of adding it afterwards.
/// [`dense::dense_tactic_is_gemv`] is the side-effect-free peek that arm
/// needs, ported and waiting.
///
/// # Safety
///
/// `act`, `w`, `bias` and `y` must address live device memory of the extents
/// `M`, `N` and `K` describe, and `y` must be writable.
#[cfg(feature = "_cuda")]
pub unsafe fn act_x_wt_bias_bf16(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    bias: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
    beta: f32,
) -> Fired {
    // Step one: `gemm::act_x_wt_bf16`'s own body — the runtime autotuner and
    // the fallback ladder.
    // SAFETY: the caller's obligation, above.
    unsafe {
        dense::act_x_wt_bf16(handle, act, w, y, m, n, k, beta);
    }
    if bias.is_null() {
        return Fired::Launched;
    }
    // Step two: `norm::add_bias_bf16(y, bias, N)` over `m` rows.
    //
    // GEOMETRY, CITED AND NOT INVENTED. The driver version passed
    // `Dims { rows: m, width: n }` to a fire that evaluated the device row's
    // `LaunchRule::RouteRows`. That rule is `runtime::launch::route_rows`:
    //
    //     grid  = [rows, 1, 1]
    //     block = [min(1024, max(32, ceil(width / 32) * 32)), 1, 1]
    //     smem  = 0
    //
    // which is `Launch::per_row` with the block width written out. `x::fire`
    // takes a geometry rather than a rule, so the rule is evaluated here —
    // the same arithmetic, at the one call site that needs it.
    let block = n.max(0).unsigned_abs().div_ceil(32).max(1).saturating_mul(32).min(1024);
    // fires: norm::add_bias_bf16
    //
    // THE SECOND LAUNCH IS ANOTHER FAMILY'S, AND IT IS STILL FULLY TYPED.
    //
    // A `raw::` stub is not bound to the unit it was declared beside: the
    // expansion takes `symbol`, `launch`, its typed parameters and `stream`
    // and calls `x::fire::fire`, which resolves `unit_of(symbol)` GLOBALLY.
    // `$UNIT` appears nowhere in a stub body. So the module path here is
    // Rust namespacing and only that, and this call gets `norm/add_bias.cuh`'s
    // real `Abi::CPP` spellings — a `*mut f16` where the kernel wants
    // `*mut bf16` does not compile — while declaring the kernel exactly once,
    // in `x::norm::add_bias`.
    //
    // An earlier draft hand-built a `&[ArgValue]` and called `x::fire::fire`
    // by symbol string, on the belief that `gemm::raw::` could not spell
    // another family's symbol. The premise was right and the conclusion was
    // wrong; this was the only place in this file where the floor's type
    // checking stopped.
    //
    // THE ONE REAL CONSEQUENCE: this makes `x::norm::add_bias`'s unit a
    // dependency of this host program and nothing in the type system says
    // so, because `symbol` is a `&'static str`. A missing unit panics at the
    // fire naming the symbol — right behaviour, wrong time. The `fires:`
    // line above is the remedy: it is grep bait, so this caller is
    // discoverable from the callee.
    //
    // SAFETY: `y` was just written by the GEMM above and is `m * n` bf16
    // elements; `bias` is `n`; the stream is the fire's.
    unsafe {
        crate::x::norm::add_bias::raw::add_bias(
            "norm::add_bias_bf16",
            Launch::per_row(m.max(0).unsigned_abs(), block),
            // The casts are the point rather than a nuisance. This fn's
            // signature is the DRIVER's — opaque `*mut c_void`, because the
            // driver hands over device pointers it has no element type for —
            // and the stub's is the KERNEL's. Naming `bf16` here is what
            // makes `T` resolve to the instantiation the symbol belongs to;
            // a `*mut f16` would not compile.
            y.cast::<bf16>(),
            bias.cast::<bf16>(),
            n,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// The declaration the readers that cannot call read.
//
// Twelve contracts, one per statement, carrying `table/gemm.rs`'s twelve rows
// minus everything that described a launcher. `whole` and `in_place` survive
// because `model-compiler` reads them; `operands`, `launch` and `file` do
// not, because they are this file's `fn`s.
//
// `lowered_as` survives on `GEMM_XWT` and is the only one in the census. See
// the note on it: `x::entry` does not consult it yet, and until it does the
// portable spelling reaches `Route::Unknown`.
// ---------------------------------------------------------------------------

contract! {
    /// The in-place NCCL all-reduce.
    ///
    /// `whole` for a reason stronger than XQA's: every rank must enter the
    /// same collective the same number of times, so a row window that split
    /// one rank's launch and not another's would DEADLOCK rather than
    /// compute the wrong answer. The refusal is not an optimisation. They
    /// are also SYNCHRONISATION points, which the graph-capture rules have
    /// to know.
    ALL_REDUCE = "dist::all_reduce_bf16" as all_reduce {
        whole: true,
        in_place: &[(0, 0)],
    }

    /// The OUT-OF-PLACE sum. Same collective, a separate destination —
    /// which the two-step landing needs, because its residual add reads the
    /// summed partial and writes somewhere else again. No alias pair, and
    /// that absence is the whole difference from the contract above.
    ALL_REDUCE_OUT = "dist::all_reduce_bf16_out" as all_reduce_out {
        whole: true,
    }

    /// The all-gather.
    ALL_GATHER = "dist::all_gather_bf16" as all_gather {
        whole: true,
    }

    /// The peer-to-peer all-reduce — `kernels::comm::CustomAllReduce`'s, not
    /// NCCL's.
    ALL_REDUCE_P2P = "comm::all_reduce_bf16" as all_reduce_p2p {
        whole: true,
    }

    /// The FUSED landing: sum, add the residual, norm. Two results — the
    /// residual stream updated in place (operand 1) and the normed
    /// activation — which is why it needs a pair list and not a sink.
    ALL_REDUCE_RESIDUAL_RMSNORM = "comm::all_reduce_residual_rmsnorm_bf16"
        as all_reduce_residual_rmsnorm {
        whole: true,
        in_place: &[(0, 1)],
    }

    /// The plain x·Wᵀ, which every family fires.
    ///
    /// `lowered_as` is an ALIAS and not a rename: the symbol is this
    /// contract's identity everywhere else, and both spellings reach a
    /// lowering — `gemm::act_x_wt_bf16` is what a text naming the CUDA
    /// symbol directly produces, `gemm::act_x_w` what the portable operation
    /// does. **`x::entry` matches `contract.symbol` and not this field**, so
    /// today the portable spelling resolves to `Route::Unknown`; that is the
    /// state of the floor as this family was ported and it is in the report.
    GEMM_XWT = "gemm::act_x_wt_bf16" as gemm_xwt {
        lowered_as: Some("gemm::act_x_w"),
    }

    /// `y[M, N] = act[M, K] x W[N, K]^T` with `W` quantized per output
    /// channel: one scale per row of `W`. Serves both FP8 E4M3 and INT8
    /// weights, and the two take completely different routes inside — FP8
    /// per-channel always dequants to bf16 (cuBLASLt has no per-channel FP8
    /// scale mode this tree targets), INT8 per-channel runs the native
    /// `CUBLAS_COMPUTE_32I` path.
    GEMM_XWT_CHANNEL_SCALED = "gemm::act_x_wt_channel_scaled" as gemm_xwt_channel_scaled

    /// The same, with `W` quantized per GROUP along K.
    GEMM_XWT_GROUPED_SCALED = "gemm::act_x_wt_grouped_scaled" as gemm_xwt_grouped_scaled

    /// MXFP4 through the Marlin kernels.
    GEMM_XWT_MXFP4_MARLIN = "gemm::act_x_wt_mxfp4_marlin" as gemm_xwt_mxfp4_marlin

    /// bf16 in, fp32 out — the one whose destination is not bf16.
    GEMM_OUT_FP32 = "gemm::act_x_wt_bf16_out_fp32" as gemm_out_fp32

    /// The grouped form. `whole` because the group boundaries (`M_array`)
    /// are fire-global, so a row window would cut a group in half.
    GEMM_GROUPED = "gemm::grouped_act_x_wt_bf16" as gemm_grouped {
        whole: true,
    }

    /// A projection with its bias in the EPILOGUE — one statement where a
    /// matmul plus an AddBias is two, and a different accumulation order.
    /// TWO weights, and the order is the statement's: the projection first,
    /// then the bias it lands with.
    GEMM_BIAS = "gemm::act_x_wt_bias_bf16" as gemm_bias
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Two of twelve bind. That ratio is this family's shape and not a shortfall:
// of the other ten, NINE never had a generated dispatch arm at all — their
// rows state `Source::Unbound` for every operand, so `emit_dispatch` skipped
// them and no trace has ever been able to fire one. The tenth is `GEMM_XWT`
// and it is the one that hurts; its `none:` says exactly why.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// NO `bind!`, AND THAT IS THE DECLARATION.
//
// `x::SIGS`' "three registration shapes" names this one: **a driver op
// contributes to `SIGS` and not to `FAMILIES`.** Every one of this family's
// twelve symbols is fired by something the driver owns, so this module
// declares twelve contracts and no `Entry`. `x::entry()` answers `None` for
// all twelve, `x::route` falls to its `Service::DriverOp` arm, and
// `bind/mod.rs`'s driver-op match — the one its own comment calls "the
// driver-op table `Route::Driver` names" — fires them with the resources a
// `Cx` cannot carry.
//
// Writing `none:` arms here instead was the first draft and it was the
// exact mistake `x::route`'s "THE ONE OVERLAP" section warns about: an
// `Entry` shadows the DriverOp arm, `Route::Unbound` refuses the model at
// load, and the sentence printed says "cannot fire" about a symbol that
// fires on every dense matmul. `gemm::act_x_w` is on the first launch of
// every forward pass, so the blast radius of that one line would have been
// every deployment.
//
// The reasons those twelve arms would have carried are measurements about
// who fires what, and they are kept here in full because deleting a row is
// what took them out of the tree.
//
// ── the three NCCL collectives ────────────────────────────────────────────
//
// `dist::all_reduce_bf16`, `dist::all_reduce_bf16_out`, `dist::all_gather_bf16`
// (`execution::SERVED`: `Service::Nccl`). The collective is a METHOD on the
// driver's `NcclComm`, and this crate neither includes `nccl.h` nor links
// NCCL. A free wrapper here would have to either take a driver type this
// crate cannot see, or reimplement the dispatch each method does — the
// custom-all-reduce fast path, the watchdog count, the async NCCL error
// check — which is a second implementation, not a wrapper. No lowering
// emits any of the three (`model-compiler/src/lower.rs` names exactly one
// `gemm::` symbol and no `dist::` or `comm::` symbol at all), so no trace
// ever routes them: the driver issues them from its tensor-parallel
// plumbing, outside the fire.
//
// ── the two peer-to-peer all-reduces ──────────────────────────────────────
//
// `comm::all_reduce_bf16`, `comm::all_reduce_residual_rmsnorm_bf16`
// (`execution::SERVED`: `Service::CustomAllReduce`, and both are still on
// `RUST_SERVED` — the pairing `SERVED`'s doc calls exact). These two ARE
// kernels-side — `kernels::comm::CustomAllReduce` — but the INSTANCE is the
// driver's: it is registered at engine boot with every rank's IPC handles
// and its lifecycle is the shell's. No `Cx` describes one, and the row
// never bound one either: every operand was `Source::Unbound`, so no arm
// was ever emitted for either.
//
// ── the dense four ────────────────────────────────────────────────────────
//
// `gemm::act_x_wt_bf16` (lowered as `gemm::act_x_w`),
// `gemm::act_x_wt_bf16_out_fp32`, `gemm::act_x_wt_bias_bf16`,
// `gemm::grouped_act_x_wt_bf16`. All four take the cuBLAS handle as their
// first parameter and all four are below. Two further facts kept them out
// of a bind even before the handle did:
//
//   * `gemm::act_x_wt_bf16`'s `beta` was `Source::Beta`, which `abi.rs`
//     emitted as `if spec.beta_one { 1.0f32 } else { 0.0f32 }` — the
//     residual fold. `Cx` has no `beta_one` and guessing is not available:
//     `beta = 0` OVERWRITES a destination a fold meant to accumulate into,
//     which is a silently wrong answer rather than a refusal. A driver op
//     never has to ask, because `LaunchSpec::beta_one` is right there.
//   * `gemm::grouped_act_x_wt_bf16`'s group boundaries are fire-global and
//     no `Source` names one. Its consumer is `fire::lora`'s hand-written
//     staged apply, which calls it directly and holds a handle of its own —
//     the same shape `pie_lora_qkv_correction` took to `Route::Driver`.
//
// ── the three quantized routers ───────────────────────────────────────────
//
// `gemm::act_x_wt_channel_scaled`, `gemm::act_x_wt_grouped_scaled`,
// `gemm::act_x_wt_mxfp4_marlin`. Their bodies are `driver-cuda`'s
// `bind::quant_gemm`, which reaches `fire::quant_int8` for its dequant
// kernels — another family, another agent. They did not move, and nothing
// regresses by that: all three state `Source::Unbound` for every operand,
// so `emit_dispatch` never wrote them an arm and no trace has ever fired
// one. `quant_gemm::act_x_w` is the router all three inline into, and it
// takes the handle first as well.
