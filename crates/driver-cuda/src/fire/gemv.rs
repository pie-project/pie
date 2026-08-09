//! The M=1 bf16 GEMV: the host half of `gemm/gemv.cu`, in Rust.
//!
//! # The specification of this program, in four terms
//!
//! Written as a design note against the code below, so that a reader can check
//! the code against a claim rather than infer the claim from the code. The
//! same four terms are stated from the table's side in
//! `kernels_cuda_new::families::gemm`'s module header — that one is about what
//! the ROWS must offer, this one is about what this FUNCTION does. Neither is
//! complete alone.
//!
//! **1. Which JIT units it fires, in what order.** One unit, `gemm/gemv`, and
//! there is no order: exactly one of its four rows runs per call, and the
//! other three do not. This is not a composition and needs no
//! `execution::Step`. What it is instead is a 2×2 choice over two independent
//! host decisions — see `Which of the four` below.
//!
//! **2. What intermediate buffers sit between them.** None, for any of the
//! four. The split-K rows reduce through `__shared__ float partial[kWarps]`,
//! which is STATIC shared memory sized by the instantiation, so every
//! [`Launch`] below carries `smem: 0` and no allocation reaches the driver.
//! The row-per-warp rows use `__shfl_down_sync` only. `beta != 0` is served in
//! place — the epilogue reads `out[row]`, scales and writes back — and `bias`
//! is folded into the epilogue instead of running a second kernel. So this
//! function allocates nothing, frees nothing, and has no lifecycle: the four
//! pointers in and the four pointers out are the caller's.
//!
//! **3. What it decides on the host, and what picks among instantiations.**
//! Six decisions, in the order the C++ made them and the order they appear in
//! [`gemv_bf16`]:
//!
//! ```text
//!   #  what                              from                      picks
//!   1  K % 8 != 0, or N <= 0, or K <= 0  operands                  REFUSE
//!   2  weight / act / out is null        operands                  REFUSE
//!   3  weight or act not 16-B aligned    operands                  REFUSE
//!   4  N <= SPLIT_K_MAX_ROWS (4096)      operand N, const          row axis
//!   5  unroll_depth(): cc major >= 10    device attribute, cached  col axis
//!   6  blocks = ceil(N / 4) > i32::MAX   arithmetic on N           REFUSE
//! ```
//!
//! Only 4 and 5 pick anything; 1, 2, 3 and 6 only ever refuse. Decision 5 is
//! the one the owner's principle names by title — *device-specific tuning* —
//! and it asks [`crate::device::Device::compute_capability`] rather than
//! opening a second way to read a device attribute.
//!
//! **4. What in `Source` / `LaunchRule` / `Specialisation` / `Execution` is
//! missing to state it.** Four gaps, and the reason this is a hand-built
//! [`Launch`] rather than a row that fires itself. In one line each:
//! `LaunchRule` fixes `blockDim.y = 1` and cannot state a 2-D block (accepted,
//! not open — the census in `families::gemm` refuses to grow it for four
//! launches); `Source` has no honest binding for any of the seven operands,
//! which are bound frames up in a dense autotuner, so the rows are deliberately
//! left unsourced whole; `Specialisation` can already say `K % 8` and both
//! 16-byte tests as `Term::Multiple` and `Term::Aligned`, but has no threshold
//! term for `N <= 4096`, no `Fact` derived from a DEVICE rather than from an
//! operand, and no arm that can REFUSE (`choose` answering `None` means *fire
//! the base row*); `Execution::Walk` is the near miss and needs a
//! `crate::table` symbol this program never had, plus a `Control` that can
//! carry a 2×2 product of a shape and a device fact rather than one
//! discriminant. All four are argued at length, per type, in
//! `kernels_cuda_new::families::gemm`'s header. **Until they close, this
//! function is how the program is stated, and that is a decision rather than a
//! stopgap** — see the next section.
//!
//! # The M=1 bf16 GEMV, and what it replaced
//!
//! Ports `kernels::gemm::gemv_bf16` — one host launcher over four `<<<>>>`,
//! plus the four decisions above them. The launcher, its header
//! (`gemm/gemv.hpp`) and its `CMakeLists.txt` entry are DELETED; the two
//! `__global__` templates are `kernels-cuda-new`'s `gemm/gemv` unit, which
//! NVRTC compiles, and everything else is here.
//!
//! The owner's principle is what decides the split, and it decides all of it:
//!
//! > Every CUDA kernel is compiled by NVRTC. Where host code is needed to
//! > compose several kernels — because kernels produce intermediate results,
//! > or because device-specific tuning is involved — that host code is all
//! > Rust.
//!
//! `gemv_bf16` is the second clause twice over: it reads a device attribute to
//! pick a template argument, and it compares a row count against a threshold
//! to pick between two different `__global__`s.
//!
//! # Why this is fired by hand, like `super::attn_score`
//!
//! Every one of the four launches is `dim3(32, kWarps)` — a warp per row,
//! `kWarps` rows per block, a **two-dimensional block**. No
//! [`kernels::LaunchRule`] states one, and
//! `kernels_cuda_new::families::gemm`'s header carries the census that refuses
//! to add one: the 2-D blocks in the tree are 32×4, 32×8, 32×`kWarps` and
//! 16×16, four geometries from four idioms with nothing shared to extract, and
//! `new-horizon.md` §10.5's bar is that a rule must serve more kernels than
//! the one that wants it.
//!
//! A [`Launch`] is not a rule. It carries three block axes, so
//! `block: [32, kWarps, 1]` is simply expressible, and
//! `KernelModule::fire`'s own doc anticipates the caller that builds one —
//! *"reaching here with one means a caller built a `Launch` by hand"*. That is
//! exactly what [`super::attn_score`] does for its literal `gridDim.y`, and
//! the discipline it comes with is the same: **every number below is a
//! citation, not a derivation**, and the `<<<>>>` it was copied from is quoted
//! beside it. The four launcher lines are also recorded verbatim at the top of
//! `kernels-cuda-new/csrc/src/gemm/gemv.cuh`, because the file they stood in
//! is gone and a citation has to resolve to text that still exists.
//!
//! # The refusal is a refusal, and it is not a fallback
//!
//! The C++ returned `bool`: `false` meant *"I did not launch — use cuBLAS"*,
//! and `gemm.cpp` read it as exactly that. [`Gemv`] is that answer with the
//! ambiguity removed. It is an enum and not a `bool` because a caller must not
//! be able to spell "it declined" the same way it spells "it ran", and it
//! carries a [`Decline`] because *which* gate refused is the difference
//! between a shape cuBLAS should have (an odd `K`) and a pointer somebody
//! staged wrong (a misaligned weight).
//!
//! **Nothing here turns a decline into a silent no-op.** A declined call
//! enqueues nothing at all — no partial launch, no memset, no zeroed output —
//! so a caller that ignores the answer gets an untouched `out`, which is the
//! same thing the C++ handed it.
//!
//! **And a broken JIT is not a decline.** If the unit will not compile, or the
//! row and the kernel table disagree, this panics with the symbol named — the
//! same choice `attn_score.rs` makes and for the same reason. Answering
//! [`Gemv::Declined`] there would send the shape to cuBLAS and produce correct
//! numbers forever while the JIT path silently never ran, which is the one
//! failure a parity measurement cannot see.

use std::sync::OnceLock;

use kernels_cuda_new::runtime::{ArgValue, Args, Launch, Stream, cache};

/// Warps per block in the row-per-warp form — `gemv.cu:329`'s
/// `constexpr int kWarps = 4`.
///
/// Load-bearing in two places at once, which is why it is one constant: it is
/// the block's second axis AND the divisor in `blocks = ceil(N / kWarps)`. A
/// grid that used one number and a block that used another would leave the
/// tail rows unwritten, silently, because the kernel's `if (row >= N) return;`
/// cannot tell a short grid from a rounded one.
const WARPS: u32 = 4;

/// Warps per block in the split-K form on Blackwell — `gemv.cu:342`'s
/// `constexpr int kSplitWarpsB = 4`.
const SPLIT_WARPS_B: u32 = 4;

/// Warps per block in the split-K form everywhere else — `gemv.cu:352`'s
/// `constexpr int kSplitWarps = 8`.
const SPLIT_WARPS: u32 = 8;

/// A warp, which is the first block axis of all four launches.
///
/// The kernels index lanes with `threadIdx.x` and warps with `threadIdx.y`,
/// and finish with `__shfl_down_sync(0xffffffffu, acc, off)` over `off` 16 →
/// 1. A block narrower than 32 on the first axis would shuffle in lanes the
/// mask claims are live and nothing wrote.
const WARP_LANES: u32 = 32;

/// The row count below which K is split INSIDE the block — `gemv.cu:317`'s
/// `constexpr int kSplitKMaxRows = 4096`.
///
/// # Why a constant, and why it is still 4096
///
/// This read `getenv("PIE_GEMV_SPLITK_MAX_ROWS")`, defaulting to 4096, until
/// `new-horizon.md` §36 folded it away at its unchanged default. It is a
/// threshold, not a toggle, and the two sides of it are two DIFFERENT
/// `__global__`s at different grids — `N` blocks of `32 x 8` against `N/4`
/// blocks of `32 x 4` — so the variable chose which kernel ran at all.
///
/// **The L40S crossover table, and why 4096 was not moved, are on
/// `kernels_cuda_new::families::gemm`'s `GEMV_ROWS`** — under *The ROW axis*.
/// It lives there because that is where the two rows it chooses between are
/// stated side by side. The one-line version: the crossover is near 2048 on a
/// 142-SM part, the shipping constant came from a 132-SM B200, and a port that
/// changes a tuning constant makes the parity run that would have checked it
/// meaningless.
const SPLIT_K_MAX_ROWS: i32 = 4096;

/// The largest grid the row-per-warp form will open — `gemv.cu:381`'s
/// `if (blocks > 2147483647LL) return false;`.
///
/// Carried rather than dropped, and it is worth saying why since it is
/// unreachable from this signature: `N` is an `i32`, so `ceil(N / 4)` cannot
/// exceed 2^29 and the guard cannot fire. It is the LAUNCHER's refusal, and a
/// refusal deleted because today's argument type happens to exclude it is a
/// refusal that comes back wrong the day the width changes.
const MAX_BLOCKS: i64 = 2_147_483_647;

/// The split-K row, four warps, unroll 2 — `families::gemm`'s `GEMV_SIGS[0]`.
///
/// Resolved through `unit_of` rather than declared as a path here, so a rename
/// in that crate is a refusal at this call and not a silent miss.
const SPLITK_W4_U2: &str = "gemm::gemv_splitk_bf16_w4_u2";

/// The split-K row, eight warps, unroll 1 — `GEMV_SIGS[1]`.
const SPLITK_W8_U1: &str = "gemm::gemv_splitk_bf16_w8_u1";

/// The row-per-warp row, four warps, unroll 2 — `GEMV_SIGS[2]`.
const ROW_W4_U2: &str = "gemm::gemv_bf16_w4_u2";

/// The row-per-warp row, four warps, unroll 4 — `GEMV_SIGS[3]`.
const ROW_W4_U4: &str = "gemm::gemv_bf16_w4_u4";

/// Why [`gemv_bf16`] did not launch.
///
/// Every arm enqueues NOTHING, which is the whole contract: the caller runs
/// the shape through cuBLAS and `out` is exactly as it found it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    /// `N <= 0`, `K <= 0`, or `K % 8 != 0` — `gemv.cu:311`.
    ///
    /// The reduction extent has to be a multiple of 8 bf16 because the walk is
    /// in `float4` strides: 8 bf16 is 16 bytes, and a `K` that is not a
    /// multiple of 8 leaves a tail no lane reads and puts every row after the
    /// first on an unaligned boundary.
    Shape,
    /// `weight`, `act` or `out` was null — `gemv.cu:312`.
    ///
    /// `bias` is NOT in this list: a null `bias` is the "no bias" signal the
    /// kernel's own `if (bias != nullptr)` reads.
    Null,
    /// `weight` or `act` was not 16-byte aligned — `gemv.cu:313`.
    ///
    /// `out` is deliberately absent, matching the C++: the epilogue writes
    /// `out[row]` as a scalar bf16 and never through a `float4`, so it has no
    /// alignment requirement beyond its own element.
    Misaligned,
    /// The row-per-warp grid would not fit — `gemv.cu:381`. See `MAX_BLOCKS`
    /// for why this is unreachable today and carried anyway.
    Grid,
}

/// What [`gemv_bf16`] did.
///
/// The C++'s `bool`, with the ambiguity removed. `#[must_use]` because
/// ignoring this answer is the one way to get a wrong result out of this
/// function: a declined call leaves `out` untouched, and a caller that reads
/// it anyway reads whatever was there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Gemv {
    /// The launch is on the stream. Exactly one kernel was enqueued.
    Launched,
    /// Nothing was enqueued. **Use cuBLAS for this shape.**
    Declined(Decline),
}

/// How deep to unroll the row walk: 2 on Blackwell and later, 4 below.
///
/// Ports `gemv_unroll_depth()` (`gemv.cu:132-143`), including its
/// process-wide caching — the C++ held the answer in a function-local
/// `static`, so it was read once per process and every replay on a machine saw
/// the same number.
///
/// # The measurement, and where the tables live
///
/// Blackwell wants a SHALLOWER unroll than Hopper, which is the opposite of
/// what the kernel's own comment predicts, so it is selected from the device
/// rather than fixed. **The four-shape unroll table (~-10% on B200), the
/// twelve-shape graph-replay table for the split-K leg's `w=4,u=2` against
/// `w=8,u=1`, and the §36 byte-difference table saying which pairs are
/// bit-identical are all on `kernels_cuda_new::families::gemm`'s
/// `GEMV_ROWS`** — under *The COLUMN axis*, *The split-K column* and *Which
/// pairs are BIT-IDENTICAL*. They are beside the four rows they chose, which
/// is the only place all four can be compared at once; a second copy here
/// would only give the numbers somewhere to drift. Their conclusions, which
/// this function is the implementation of: `w=4,u=2` is 20% faster than
/// `w=8,u=1` summed over twelve graph-replayed decode shapes and is the worst
/// config on 11 of them, Hopper keeps `w=8,u=1` because that is where it was
/// tuned, and the two earlier attempts that got this wrong both swept EAGER,
/// under a ~4.1 us launch floor most of these shapes run beneath.
///
/// The one thing this function must carry, because it is about THIS code: the
/// split-K leg reads the same answer and uses it for MORE than the unroll —
/// `warps 4, unroll 2` against `warps 8, unroll 1` — and that pair is NOT
/// bit-identical under wide exponents, while the row-per-warp pair is.
///
/// # This answered to `getenv("PIE_GEMV_B200_TUNING")`, and it must not
///
/// The variable set the return to 4 — "revert to the Hopper constants without
/// a rebuild" — and it reached THREE launchers. It is deleted, and unlike
/// `PIE_QWEN35_GDN_SMEM_STEP` (§30) it is NOT deleted because the arms agree.
/// They do not, and the table on `GEMV_ROWS` is the run that says so. An
/// env-var selector makes the same trace on the same weights on the same GPU
/// emit different bits, with nothing in the plan, the replay, or another
/// backend able to say which arm ran. What is left is a DEVICE FACT, which is
/// a different thing: the compute capability is a property of the machine, the
/// same on every replay on that machine, and discoverable by any backend that
/// asks.
///
/// # What it asks, and the one way this differs from the C++
///
/// [`crate::device::Device::compute_capability`] — the home this driver
/// already has for a device attribute, rather than a second
/// `cudaDeviceGetAttribute` written here. Reaching it needs a [`Device`]
/// token, and `Device::bind` is how one is made, so this call forces the
/// primary context where the C++ only read an attribute. On any thread that
/// reaches a GEMV the context is long since live and the bind is a
/// `cudaSetDevice` to the ordinal already current.
///
/// [`Device`]: crate::device::Device
///
/// **Every failure answers 4**, exactly as the C++ did, and the consequence is
/// worth stating because it is not symmetric: 4 is the pre-Blackwell arm, so a
/// failed query on a B200 costs ~10% on the row-per-warp leg and nothing else
/// (the two unrolls are bit-identical), but on the split-K leg it also changes
/// the warp count, and that arm is NOT bit-identical. A query that fails is a
/// machine where nothing else is going to work either; it is a slower, still
/// correct answer, cached once so it cannot differ between two fires of one
/// process.
fn unroll_depth() -> i32 {
    static DEPTH: OnceLock<i32> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        use cudarc::runtime::sys as rt;

        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live, writable out-parameter for the call.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return 4;
        }
        let Ok(device) = crate::device::Device::bind(ordinal) else {
            return 4;
        };
        let Ok((major, _minor)) = device.compute_capability() else {
            return 4;
        };
        if major >= 10 { 2 } else { 4 }
    })
}

/// Single-row bf16 GEMV: `out[n] = sum_k W[n][k] * x[k] + bias[n] + beta * out[n]`.
///
/// `weight` is bf16 `[N, K]` with row stride `K`, `act` is bf16 `[K]`, `bias`
/// is bf16 `[N]` or null, `out` is bf16 `[N]`. `beta` had a default of `0.f`
/// in C++ and has none here: `beta = 1` is what a projection accumulating into
/// a residual asks for, and refusing it used to push those shapes onto cuBLAS
/// — gpt-oss's o_proj measured 17.3 us there against 11.2 for the same bytes
/// through this kernel.
///
/// # Why the kernel exists at all
///
/// This is the M=1 decode shape. There is no weight reuse to exploit, so the
/// kernel is a pure streaming read and the only thing that matters is HBM
/// bandwidth. cuBLAS tiles these for an M worth filling and leaves half the
/// bandwidth unused — the A100 table against `cublasGemmEx` is on
/// `kernels_cuda_new::families::gemm`'s `GEMV_ROWS`.
///
/// When `bias` is present the epilogue computes `out[n] = bf16(bf16(dot) +
/// bias[n])`, which is bit-identical to running `kernels::norm::add_bias_bf16`
/// afterwards — the double rounding is intentional and the kernel says so at
/// both of its epilogues. This removes a whole kernel launch per biased
/// projection; on gpt-oss-20b that is 120 launches per decode step, each
/// ~3.6 us against a 2.2 us empty-launch floor, for a few KB of arithmetic.
///
/// # The four launches, and which one this fires
///
/// ```text
///   N <= 4096   and cc >= 10   splitk<4, 2>   grid [N, 1, 1]       block [32, 4, 1]
///   N <= 4096   and cc <  10   splitk<8, 1>   grid [N, 1, 1]       block [32, 8, 1]
///   N >  4096   and cc >= 10   gemv<4, 2>     grid [ceil(N/4),..]  block [32, 4, 1]
///   N >  4096   and cc <  10   gemv<4, 4>     grid [ceil(N/4),..]  block [32, 4, 1]
/// ```
///
/// Why those four and not others — the twelve-shape graph-replay table, the
/// two earlier eager-swept mistakes it corrects, and the 20% figure — is on
/// `GEMV_ROWS`, beside the rows it chose, under *The split-K column*.
///
/// # Panics
///
/// If `gemm/gemv` is in no JIT unit, if the unit will not compile or load, or
/// if the row's operands and the kernel's parameters have drifted. None of
/// those is a shape this function may decline over — see the module header.
///
/// # What the caller is still asserting
///
/// Not `unsafe`, and the pointers are never dereferenced here: they are
/// compared against null, tested for alignment, and handed to the launch as
/// addresses. The caller asserts, exactly as it did when it handed the same
/// four pointers and a `cudaStream_t` to a C++ launcher, that they are device
/// addresses of the stated extents and that `stream` is live across the
/// launch.
#[allow(clippy::too_many_arguments)] // the C++ launcher's parameter list, unchanged
#[allow(clippy::not_unsafe_ptr_arg_deref)] // nothing here dereferences one
pub fn gemv_bf16(
    weight: *const std::ffi::c_void,
    act: *const std::ffi::c_void,
    bias: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    n: i32,
    k: i32,
    stream: *mut std::ffi::c_void,
    beta: f32,
) -> Gemv {
    // `gemv.cu:311` — "The float4 loads need each row to start 16-byte
    // aligned: that holds iff the base is aligned and the row stride is a
    // multiple of 8 bf16."
    //
    //     if (N <= 0 || K <= 0 || (K % 8) != 0) return false;
    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Gemv::Declined(Decline::Shape);
    }
    // `gemv.cu:312` — `if (weight == nullptr || act == nullptr || out == nullptr)`
    if weight.is_null() || act.is_null() || out.is_null() {
        return Gemv::Declined(Decline::Null);
    }
    // `gemv.cu:313` — `if (!aligned16(weight) || !aligned16(act)) return false;`
    if !aligned16(weight) || !aligned16(act) {
        return Gemv::Declined(Decline::Misaligned);
    }

    let values = [
        ArgValue::Ptr(weight.cast_mut()),
        ArgValue::Ptr(act.cast_mut()),
        ArgValue::Ptr(bias.cast_mut()),
        ArgValue::Ptr(out),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(beta),
    ];

    // `gemv.cu:318` — `if (N <= kSplitKMaxRows) {`
    if n <= SPLIT_K_MAX_ROWS {
        let (symbol, warps) = if unroll_depth() == 2 {
            (SPLITK_W4_U2, SPLIT_WARPS_B)
        } else {
            (SPLITK_W8_U1, SPLIT_WARPS)
        };
        // `gemv.cu:345` and `:356`, which are the same two lines at two warp
        // counts:
        //
        //     <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarpsB), 0,
        //        stream>>>(
        //     <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarps), 0,
        //        stream>>>(
        //
        // One block per output row. `smem` is 0 because the block's reduction
        // buffer is `__shared__ float partial[kWarps]` — STATIC shared memory,
        // sized by the instantiation, and passing its size as the dynamic
        // extent would allocate it twice.
        fire(
            symbol,
            Launch {
                grid: [n.unsigned_abs(), 1, 1],
                block: [WARP_LANES, warps, 1],
                smem: 0,
            },
            &values,
            stream,
        );
        return Gemv::Launched;
    }

    // `gemv.cu:380` — `const long long blocks = (N + kWarps - 1) / kWarps;`
    let warps = i64::from(WARPS);
    let blocks = (i64::from(n) + warps - 1) / warps;
    // `gemv.cu:381` — `if (blocks > 2147483647LL) return false;`
    if blocks > MAX_BLOCKS {
        return Gemv::Declined(Decline::Grid);
    }
    let Ok(grid_x) = u32::try_from(blocks) else {
        return Gemv::Declined(Decline::Grid);
    };

    // `gemv.cu:382-384` — "Everything below is unconditional, so the caller
    // never has to reason about a half-enqueued launch. In particular this
    // must not poll `cudaGetLastError`: that would consume an unrelated
    // pending error the driver's own checks are waiting to report." The Rust
    // keeps that promise the same way, by not asking.
    let symbol = if unroll_depth() == 2 { ROW_W4_U2 } else { ROW_W4_U4 };
    // `gemv.cu:373` and `:383`, the same grid and block at two unrolls:
    //
    //     <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0,
    //        stream>>>(
    //     <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0, stream>>>(
    //
    // One warp per output row, `kWarps` rows per block, so the grid is a
    // ROUNDED-UP row count and the kernel's `if (row >= N) return;` is what
    // makes the last block safe. `smem` is 0: this form reduces through
    // `__shfl_down_sync` alone and allocates no shared memory at all.
    fire(
        symbol,
        Launch {
            grid: [grid_x, 1, 1],
            block: [WARP_LANES, WARPS, 1],
            smem: 0,
        },
        &values,
        stream,
    );
    Gemv::Launched
}

/// `gemv.cu:299` — `(reinterpret_cast<std::uintptr_t>(p) & 15u) == 0`.
///
/// A HOST test made before any launch, which is why it is here and why
/// `gemv.cuh` no longer includes `<cstdint>`: `std::uintptr_t` was the only
/// thing that header was for, and NVRTC has no `<cstdint>` to offer.
///
/// Duplicated by `fire/hand.rs::aligned16`, which is in flight beside this
/// file and names it. When `pub mod hand;` lands in `fire/mod.rs` this
/// function and [`fire`] below both delete, and their call sites take
/// `hand::aligned16` and `hand::fire` unchanged — the bodies are already
/// identical. They are kept private here only so that this module compiles
/// against a `fire/mod.rs` that has not declared `hand` yet.
fn aligned16(p: *const std::ffi::c_void) -> bool {
    p.addr() & 15 == 0
}

/// Resolve one row through the JIT table, bind the operands, launch.
///
/// One function for all four arms, because the four differ in exactly two
/// values — the symbol and the [`Launch`] — and a copy per arm is four places
/// for an operand order to drift. `Args::bind` checks `values` against the
/// row's signature, so a drift between the list built in [`gemv_bf16`] and
/// `families::gemm`'s `GEMV_SIGS` is a refusal here rather than a shifted
/// argument at the kernel.
///
/// # Panics
///
/// Every failure on this path is drift between this driver and its kernel
/// table, or a unit that will not compile. See the module header for why none
/// of them may be answered with [`Gemv::Declined`].
#[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
fn fire(symbol: &'static str, launch: Launch, values: &[ArgValue], stream: *mut std::ffi::c_void) {
    let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
    };
    let module = match cache::module(index, unit) {
        Ok(module) => module,
        Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
    };
    let mut args = match Args::bind(sig, values) {
        Ok(args) => args,
        Err(why) => panic!("{symbol}: {why}"),
    };
    // SAFETY: the caller holds the fire's stream live across the launch — the
    // same assertion it made when it handed the stream to a C++ launcher that
    // put it in a `<<<>>>`.
    let stream = unsafe { Stream::from_runtime(stream) };
    if let Err(why) = module.fire(sig, launch, &mut args, stream) {
        panic!("{symbol}: {why}");
    }
}
