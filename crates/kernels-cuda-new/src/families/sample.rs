//! `sample`'s JIT unit — the argmax, and the ten kernels a rule cannot launch.
//!
//! One unit, twelve `__global__` templates, five rows. The device text is
//! `kernels-cuda-new/csrc/src/sample/argmax.cuh`, and the ahead-of-time
//! `argmax.cu` that used to include it is DELETED — its one surviving
//! launcher, `lm_head_gemv_argmax_int8`, is
//! `driver-cuda/src/fire/lm_head_argmax.rs`. The property
//! `tests/sources.rs::no_global_is_defined_twice` exists to keep exactly one
//! definition of each kernel, after `norm/altup_aux` shipped two copies of a
//! kernel for a release; with the `.cu` gone this unit is the only copy.
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
//! # The ten with no row, of which TWO now have one
//!
//! This family is where the recipe's warning lands: `sample/argmax.cu` is the
//! file whose launchers ask the DEVICE how big their grid should be. Ten of
//! the twelve templates were carried as device text and left unmigrated, and
//! the reasons are five distinct ones rather than ten:
//!
//! * `lm_head_gemv_argmax_int8` and `lm_head_gemv_argmax` — grid.x is
//!   `min(num_sms * blocks_per_sm, ceil(vocab / 8))`, straight off
//!   `cudaDevAttrMultiProcessorCount`; the grid is 2-D over (blocks, rows);
//!   the dynamic shared memory is `hidden * sizeof(float)`, not a constant;
//!   the launcher owns a `static cudaMalloc`'d scratch buffer that grows with
//!   the batch; and it fires TWO kernels per call. Five separate things no
//!   row says.
//!
//!   **The int8 one is now rowed, and none of those five sentences is
//!   retracted.** Every one of them is a reason no `LaunchRule` can state the
//!   launch, and no `LaunchRule` states it: [`ARGMAX_SIGS`]`[3]` says
//!   `LaunchRule::Unstated`, which is `families::gemm`'s answer for its four
//!   `gemv_*` rows and means *"a caller builds the `Launch` by hand"*. What
//!   changed is that the caller exists: `driver-cuda/src/fire/lm_head_argmax.rs`
//!   reads the SM count through `device::Device::sm_count`, owns the scratch
//!   as a Rust value instead of a function-local `static`, and fires both
//!   kernels through `KernelModule::fire`. A row is a name and an operand
//!   list; it was never the thing that was missing.
//!
//!   `lm_head_gemv_argmax` — the bf16-weight twin — stays unrowed, because
//!   nothing calls it: `sample/argmax.cu` launched only the int8 form and
//!   `table::sample` holds only the int8 symbol.
//! * `select_lm_head_argmax_pairs` — an `Elementwise` grid, but its
//!   `num_tiles` OPERAND is the block count that same SM query produced. The
//!   grid a rule states; a device query it does not. **Rowed for the same
//!   reason as its partner and with the same `Unstated`**: it is the second
//!   half of one call and could not be left behind by it.
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
//! worth more than a migration that launches the wrong extent — and the two
//! rows above are what that sentence looks like when the host program that
//! owns the query is Rust the driver runs, rather than C++ nobody can
//! intercept.
//!
//! # The `static_assert` `argmax.cu` carried
//!
//! `sample/argmax.cu:38-40` held
//! `static_assert(device::kAccumWarps == kArgmaxAccumSlots)` — the
//! accumulator carries one slot per warp, `sample/argmax.hpp` published that
//! count to its callers as `kArgmaxAccumSlots`, and the two files agreed by
//! assertion rather than by assumption. Both files are deleted and the two
//! constants are now one: `sample/argmax.cuh:150-151` defines
//! `kAccumThreads = 1024` and `kAccumWarps = kAccumThreads / 32`, and nothing
//! outside that header names a slot count. The assertion is not lost, it is
//! unnecessary — but a future caller of `argmax_accumulate` that sizes a
//! scratch buffer must read `kAccumWarps` from the header rather than writing
//! 32 by hand, which is what the deleted `.hpp` constant existed to prevent.

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
    // The fused LM-head GEMV. Rowed at bf16 alone, because that is the only
    // instantiation `sample/argmax.cu` ever launched and the only one
    // `driver_cuda::fire::lm_head_argmax` names.
    DeviceKernel {
        sig: &ARGMAX_SIGS[3],
        template_path: "sample::device::lm_head_gemv_argmax_int8",
        elem: "device::bf16",
    },
    // Its second half. `select_lm_head_argmax_pairs` has NO template
    // parameter list -- the header says so in as many words, *"not a template
    // -- there is no element type in it, only packed pairs"* -- so its `elem`
    // is [`DeviceKernel::PLAIN`] and its instantiation is its bare path.
    DeviceKernel {
        sig: &ARGMAX_SIGS[4],
        template_path: "sample::device::select_lm_head_argmax_pairs",
        elem: DeviceKernel::PLAIN,
    },
];

/// The contracts, in [`ARGMAX_ROWS`]' order.
///
/// The first three launched `<<<rows, 256>>>`, which is `LaunchRule::Rms`: one
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
///
/// The last two state [`LaunchRule::Unstated`], and that is the whole of what
/// the module header above refused. It refused a RULE, and it was right to:
/// the fused GEMV's `grid.x` is `min(num_sms * 2, ceil(vocab / GEMV_WARPS))`
/// off `cudaDevAttrMultiProcessorCount`, its `grid.y` is `num_rows`, its
/// dynamic shared memory is `hidden * sizeof(float)`, and its partner's
/// `num_tiles` operand is that same block count. No `LaunchRule` states any of
/// those and `new-horizon.md` §10.5 forbids growing the vocabulary for one
/// kernel. What a row costs is a NAME and an operand list, which is all
/// `Args::bind` and `nvrtcAddNameExpression` need; the geometry is built by
/// hand in `driver-cuda/src/fire/lm_head_argmax.rs` and fired through
/// `KernelModule::fire`, exactly as `fire/gemv.rs` and `fire/attn_score.rs`
/// do for their own unstatable grids. `families::gemm`'s four `gemv_*` rows
/// are the precedent, character for character.
///
/// [`LaunchRule::Unstated`]: kernels::LaunchRule::Unstated
///
/// # One operand these two rows cannot spell, said out loud
///
/// `partial_pairs` is a `u64*` — one packed `(value, token)` pair per tile per
/// row — and [`kernels::Ty`] has no word for that. It is stated `BufMut` on
/// the producer and `Buf` on the consumer, which is right for
/// `runtime::args::Args::bind` (both are `ArgValue::Ptr`) and WRONG for
/// `abi::emit_device_typecheck`, which spells a buffer kind as a pointer to
/// the head of the row's `elem` and would therefore write `device::bf16*`.
/// That is exactly the shape this family's `quant` sibling documents as the
/// root cause of its own seven-row gap — *"a kernel with a FIXED-element
/// buffer beside a templated one"* — and closing it is one variant,
/// `Ty::U64sMut`, plus its `cpp()`/`rust()`/`ArgValue` arms. It is not added
/// here because nothing automated compiles this unit's typecheck TU today and
/// a variant added for one operand is the bar §10.5 sets in the other
/// direction. The consumer row is PLAIN, so `emit_device_typecheck` refuses it
/// by name before it reaches an operand at all — `families::layout`'s
/// `copy_if_valid_slot` has the same standing refusal.
#[rustfmt::skip]
static ARGMAX_SIGS: [KernelSig; 5] = [
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
    // `sample/argmax.cuh:670-677`, the eight parameters in order. NOT the
    // table row's operand list and the difference is the point: the TABLE's
    // `sample::lm_head_gemv_argmax_int8` states `(hidden_states,
    // lm_head_weight, scale_inv, token_ids, num_rows, hidden, vocab, stream)`
    // — a launcher's list, with the caller's OUTPUT and a stream. The kernel
    // writes `partial_pairs`, never sees `token_ids`, takes `num_blocks_x`
    // (its own grid extent, which it needs for the grid-stride bound) and
    // takes no stream, because a stream is `cuLaunchKernel`'s sixth
    // parameter. The two rows are different contracts over the same job and
    // the symbols differ so that `unit::unit_of` keeps answering `None` for
    // the stated one — which `tests/layers.rs` requires of every row
    // `execution::WALKED` names.
    //
    // No `Source` on any operand, like `families::gemm`'s four `gemv_*` rows:
    // a `Source` is what an emitted dispatch arm binds from, and nothing
    // emits an arm for this row. `driver_cuda::fire::lm_head_argmax` binds it
    // by hand, in this order, and `Args::bind` refuses a drift.
    kernel!(lm_head_gemv_argmax_int8_bf16 "sample::lm_head_gemv_argmax_int8_bf16",
        file = Some("sample/argmax.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            hidden_states: Buf,
            lm_head_weight: I8s,
            scale_inv: F32s,
            partial_pairs: BufMut,
            num_rows: I32,
            hidden: I32,
            vocab: I32,
            num_blocks_x: I32,
        ]),
    // `sample/argmax.cuh:546-551`. `num_tiles` is `num_blocks_x` under
    // another name — the producer's grid.x, which is how many pairs per row
    // the scratch holds — and it is the operand the module header above cites
    // as the reason an `Elementwise` rule cannot state this launch: the grid
    // a rule computes, over an extent a device query produced.
    kernel!(select_lm_head_argmax_pairs "sample::select_lm_head_argmax_pairs",
        file = Some("sample/argmax.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            partial_pairs: Buf,
            out_tokens: I32sMut,
            num_rows: I32,
            num_tiles: I32,
        ]),
];
