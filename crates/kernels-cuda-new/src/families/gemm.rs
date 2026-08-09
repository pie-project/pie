//! `gemm`'s JIT units — none, and `gemv.cu` is why.
//!
//! `gemm/gemv.cu` holds three `__global__` templates and nine launches, and
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
//!   reads `cudaDevAttrComputeCapabilityMajor` and `getenv("PIE_GEMV_B200_TUNING")`
//!   to choose between `<4, 2>`, `<8, 1>` and `<4, 4>`; the split-K leg is
//!   picked by comparing the row count against `getenv("PIE_GEMV_SPLITK_MAX_ROWS")`.
//!   A row names ONE instantiation. These launchers pick one of four per
//!   call, on facts a name expression is fixed before it can see.
//! * **The launchers return `bool`.** `K % 8 != 0`, or a pointer not aligned
//!   to 16, and they return `false` meaning "I did not launch — use cuBLAS".
//!   A row cannot decline. Dispatching one through the JIT would launch the
//!   kernel the C++ refused and read past the buffer it refused over.
//!
//! The nine launches over three templates are four in `gemv_bf16`
//! (`splitk<4,2>`, `splitk<8,1>`, `gemv<4,2>`, `gemv<4,4>`), two in
//! `gemv3_bf16`, and three in the tuning entry points `gemv_bf16_tuned`,
//! `gemv3_bf16_tuned` and `gemv_splitk_tuned`, whose `kWarps`/`kUnroll` come
//! from their ARGUMENTS — a sweep harness parameterises them. No launch in
//! the file names a kernel defined elsewhere, and no kernel in the file is
//! launched from elsewhere.
//!
//! `gemv.cu` is therefore left whole: no `.cuh`, no split, no rows. Splitting
//! device text out of a file that will never be JIT-compiled buys nothing and
//! costs a second place for the kernel to live.
//!
//! `gemm/gemm.cpp` is host C++ compiled by `g++` — cuBLASLt plumbing and
//! dispatch, no `__global__`, no `<<<>>>` of its own. It is out of scope by
//! construction, not by choice.
//!
//! The module stays so that [`super::ALL`] can name it, and so that the first
//! `gemm` unit — when a rule exists that can state a warp-per-row grid —
//! touches one file rather than three.

use crate::unit::Unit;

/// The units `gemm` compiles. Empty: see this module's header for the three
/// reasons `gemv.cu` has no row.
pub static UNITS: &[Unit] = &[];
