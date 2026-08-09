//! `gemm`'s JIT units — none, and `gemv.cu` is why.
//!
//! `gemm/gemv.cu` holds two `__global__` templates and four launches, and
//! not one of them can be stated as a row. It was READ for this migration
//! rather than skipped, and the three findings are worth recording so the
//! next reader does not repeat the reading:
//!
//! * **The geometry.** Every launch is `dim3(32, kWarps)` — a warp per row,
//!   `kWarps` rows per block. No `LaunchRule` states a 2-D BLOCK, and the
//!   rules `runtime::launch` evaluates all fix `blockDim.x` at 256. A rule
//!   invented for these three would be a geometry only these three mean,
//!   which is the vocabulary growth `new-horizon.md` §10.5 forbids.
//! * **The template arguments come from a device query.** `gemv_unroll_depth()`
//!   reads `cudaDevAttrComputeCapabilityMajor` to choose between `<4, 2>` and
//!   `<8, 1>`; the split-K leg is picked by comparing the row count against
//!   `kSplitKMaxRows`. A row names ONE instantiation. These launchers pick one
//!   per call, on a fact a name expression is fixed before it can see.
//!
//!   Both selectors were `getenv` until §36 measured them and deleted the
//!   knobs — `PIE_GEMV_B200_TUNING` outright (its arms differ by 5 bytes only
//!   under wide exponents, it was unreachable on sm_89, and the arm it could
//!   reach was slower), `PIE_GEMV_SPLITK_MAX_ROWS` into the `constexpr` above
//!   at its unchanged default. **That did not lower this wall**, and the
//!   distinction is the point: what a row cannot state is a choice made per
//!   call, and a `cudaDeviceGetAttribute` is exactly as unnameable as a
//!   `getenv` was. What changed is that the choice is now reproducible — the
//!   same machine gives the same answer, so a parity run means something.
//! * **The launchers return `bool`.** `K % 8 != 0`, or a pointer not aligned
//!   to 16, and they return `false` meaning "I did not launch — use cuBLAS".
//!   A row cannot decline. Dispatching one through the JIT would launch the
//!   kernel the C++ refused and read past the buffer it refused over.
//!
//! The four launches over two templates are all in `gemv_bf16`
//! (`splitk<4,2>`, `splitk<8,1>`, `gemv<4,2>`, `gemv<4,4>`). No launch in
//! the file names a kernel defined elsewhere, and no kernel in the file is
//! launched from elsewhere.
//!
//! It was nine launches over three templates until §45. `gemv3_bf16` (the
//! fused Q/K/V triple, whose row §27 had already deleted) and the three
//! sweep entry points `gemv_bf16_tuned`, `gemv3_bf16_tuned` and
//! `gemv_splitk_tuned` were reachable from NO root — the harness the sweeps
//! name, `driver/cuda/bench/gemv_bench.cu`, is in no source directory of
//! this repository. Each of the four was checked on its own against the
//! whole worktree before it went (§10.10: a launcher goes when its WHOLE
//! consumer set has), and `gemv3_bf16_kernel` went with its two callers.
//! **None of that lowered the wall above** — the three reasons are
//! properties of `gemv_bf16` itself, which is live and stays.
//!
//! `gemm/gemm.cpp` is host C++ compiled by `g++` — cuBLASLt plumbing and
//! dispatch, no `__global__`, no `<<<>>>` of its own. §45 moved the part of
//! it that is pure cuBLAS ARGUMENT ASSEMBLY into Rust (`driver-cuda`'s
//! `bind::service`, reached through [`crate::execution::Execution::Service`]),
//! and left the part that is a runtime autotuner — the one that calls
//! `gemv_bf16` and therefore cannot be stated either. See §45.
//!
//! The module stays so that [`super::ALL`] can name it, and so that the first
//! `gemm` unit — when a rule exists that can state a warp-per-row grid —
//! touches one file rather than three.

use crate::unit::Unit;

/// The units `gemm` compiles. Empty: see this module's header for the three
/// reasons `gemv.cu` has no row.
pub static UNITS: &[Unit] = &[];
