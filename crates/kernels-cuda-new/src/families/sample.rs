//! `sample`'s JIT unit — the argmax, and the ten kernels a rule cannot launch.
//!
//! One unit, twelve `__global__` templates, three rows. The device text is
//! `kernels-cuda-new/csrc/src/sample/argmax.cuh`, which the ahead-of-time
//! `argmax.cu` now includes,
//! so the ahead-of-time archive holds exactly ONE definition of each kernel —
//! the property `tests/sources.rs::no_global_is_defined_twice` exists to keep
//! after `norm/altup_aux` shipped two copies of a kernel for a release.
//!
//! # Three rows over two templates
//!
//! `argmax` is rowed twice, at `device::bf16` and at `sample::device::f32`.
//! Those were TWO kernels in the ahead-of-time build — `argmax_bf16_kernel`
//! and `argmax_fp32_kernel`, identical but for a load — because instantiating
//! a template twice cost a translation unit. Under a JIT the second format
//! costs a row, which is `norm/elementwise`'s measurement restated in a
//! family that had already paid for it by hand.
//!
//! # The ten with no row
//!
//! This family is where the recipe's warning lands: `sample/argmax.cu` is the
//! file whose launchers ask the DEVICE how big their grid should be. Ten of
//! the twelve templates are carried as device text and left unmigrated, and
//! the reasons are five distinct ones rather than ten:
//!
//! * `lm_head_gemv_argmax_int8` and `lm_head_gemv_argmax` — grid.x is
//!   `min(num_sms * blocks_per_sm, ceil(vocab / 8))`, straight off
//!   `cudaDevAttrMultiProcessorCount`; the grid is 2-D over (blocks, rows);
//!   the dynamic shared memory is `hidden * sizeof(float)`, not a constant;
//!   the launcher owns a `static cudaMalloc`'d scratch buffer that grows with
//!   the batch; and it fires TWO kernels per call. Five separate things no
//!   row says.
//! * `select_lm_head_argmax_pairs` — an `Elementwise` grid, but its
//!   `num_tiles` OPERAND is the block count that same SM query produced. The
//!   grid a rule states; a device query it does not.
//! * `argmax_vec2` and `argmax_compact_scatter_vec2` — selected by
//!   `argmax_vec2_usable(logits, vocab)`, a run-time test on an operand's
//!   ADDRESS and on the parity of the vocab. A `Source` states where a value
//!   comes from, never a predicate over one, and firing the vec2 form on an
//!   odd vocab puts every second row on a 2-byte boundary and faults.
//! * `masked_embedding_argmax`, `topk_centroids` and
//!   `masked_embedding_tile_argmax_pairs` — their launchers CLAMP
//!   `centroid_top_k` to 64 before passing it, and the kernels index
//!   `__shared__` arrays of exactly 64 with it. The clamp is load-bearing and
//!   no `Source` expresses `min(k, 64)`, so a row would turn a truncation
//!   into a shared-memory overrun on the first config that asked for more.
//!   The tile form is doubly blocked: `dim3(rows, tiles)`.
//! * `argmax_accumulate` and `argmax_finalize` — 1024 threads and 32 threads.
//!   No rule states either width, `argmax_accumulate`'s per-warp accumulator
//!   count is `static_assert`ed against the block it is launched with, and it
//!   takes two `bool` operands the binder has no type for.
//!
//! An honest "blocked because its grid comes from an occupancy query" is
//! worth more than a migration that launches the wrong extent.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The greedy decode: one block per row, 256 threads striding the vocab.
pub const ARGMAX: Unit = Unit {
    name: "sample/argmax",
    root: include_str!("../../csrc/src/sample/argmax.cuh"),
    rows: ARGMAX_ROWS,
    options: &[],
};

/// The units `sample` compiles.
pub static UNITS: &[Unit] = &[ARGMAX];

/// [`ARGMAX`]'s instantiations.
///
/// `sample::device::f32` is a `using` alias for `float` inside the device
/// namespace, not a prelude type: `Elem<T>` has no `float` specialisation and
/// should not grow one — there fp32 is what a kernel COMPUTES in, and a
/// specialisation would make `Elem<float>::from_f32` an identity that reads
/// like a conversion. The header's `Logit<T>` carries that one widening.
static ARGMAX_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ARGMAX_SIGS[0],
        template_path: "sample::device::argmax",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ARGMAX_SIGS[1],
        template_path: "sample::device::argmax",
        elem: "sample::device::f32",
    },
    DeviceKernel {
        sig: &ARGMAX_SIGS[2],
        template_path: "sample::device::argmax_compact_scatter",
        elem: "device::bf16",
    },
];

/// The contracts, in [`ARGMAX_ROWS`]' order.
///
/// All three launched `<<<rows, 256>>>`, which is `LaunchRule::Rms`: one
/// block per row, 256 threads, the block striding the row. So `num_rows` and
/// the stream leave every row and nothing else does. The rule also asks for
/// 32 bytes of dynamic shared memory that these kernels never declare — an
/// unread allocation is not a behaviour, and the alternative was a fourth
/// rule that differs from `Rms` in a number nobody reads.
///
/// The 256 is not merely what the launcher passed: the kernels stride the
/// vocab by a compile-time `BLOCK = 256` and size their `__shared__`
/// reduction with it, so a rule that launched any other width would fold over
/// a buffer it had not filled. `Rms` gives exactly 256, which is why these
/// three fit and the vectorised pair — 128 threads doing two elements each —
/// does not.
#[rustfmt::skip]
static ARGMAX_SIGS: [KernelSig; 3] = [
    // Not in `crate::table::sample`'s table, and deliberately: that table and
    // `dsl::cuda` are the same set, a DSL statement is something a TRACE
    // records, and CSM's backbone -- the one caller -- is a hand-written
    // forward. A DEVICE row has no such constraint: it names an
    // instantiation to compile, not a statement to lower, and the hand-written
    // forward can dispatch it by symbol like any other.
    kernel!(argmax "sample::argmax_bf16",
        file = Some("sample/argmax.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf <- Source::In(0),
            out: I32sMut <- Source::Out(0),
            vocab: I32 <- Source::InWidth(0),
        ]),
    // The fp32 twin. Same template, same rule, same operand order -- the row
    // differs from the one above in its `elem` and in nothing else, which is
    // the whole claim the JIT makes.
    kernel!(argmax_f32 "sample::argmax_f32",
        file = Some("sample/argmax.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf <- Source::In(0),
            out: I32sMut <- Source::Out(0),
            vocab: I32 <- Source::InWidth(0),
        ]),
    // The compact form: logits indexed by the COMPACT row, output by
    // `row_indices[compact_row]`, so a fire that dropped rows writes its
    // answers where the un-dropped batch expects them. `Rms` reads the
    // fire's rectangle, which is the compact one -- the same count the
    // launcher passed.
    kernel!(argmax_compact_scatter "sample::argmax_compact_scatter_bf16",
        file = Some("sample/argmax.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf <- Source::In(0),
            row_indices: I32s <- Source::In(1),
            out: I32sMut <- Source::Out(0),
            vocab: I32 <- Source::InWidth(0),
        ]),
];
