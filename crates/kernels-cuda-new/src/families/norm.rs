//! `norm`'s JIT units — the pilot's two, and the three the family's launchers
//! became.
//!
//! `norm/altup_aux` and `norm/elementwise` were the pilot: the first
//! `__global__` templates this tree ever compiled at run time, and the first
//! two `.cu` launchers it deleted. They live here rather than in
//! [`crate::unit`] because a family owning its own units is what lets the rest
//! of the migration proceed one file at a time — see [`super`].
//!
//! `norm/elementwise` carries the measurement that made the whole design
//! worth doing: `residual_add_f16` is a kernel the ahead-of-time build never
//! had, not because it was hard but because instantiating it cost a
//! translation unit's worth of `cicc` for something nobody had asked for yet.
//! Under a JIT it cost one row.
//!
//! # The three that followed, and the twelve kernels that did not
//!
//! `norm/add_bias`, `norm/dsv4_hc` and `norm/rmsnorm` are the device halves of
//! `add_bias.cu`, `dsv4_hc.cu` and `rmsnorm.cu`, split out so that exactly one
//! definition of each `__global__` exists in the tree. Their launchers stay
//! where they are for now — the AOT path must keep running — but the kernels
//! are no longer inside them, so a fix lands in one place.
//!
//! Eleven rows name eleven of the twenty-one templates those three headers
//! hold. The rest are not rowed, and the reasons are worth stating here
//! because every one of them is a missing RULE and not a missing kernel:
//!
//! **Re-audited at `LaunchRule` 21 → 28.** Every refusal below was re-decided
//! against the eight rules that landed with §21.13, one launcher at a time,
//! and every one of them stands. `PerRowNarrow` is `<<<rows, 128>>>` and no
//! launcher in `rmsnorm.cu`, `dsv4_hc.cu`, `add_bias.cu` or `residual_add.cu`
//! is that shape — they are 256 or 512 wide, and `altup.cu`'s 128 is on three
//! axes. `RowsFlat` is `ceil(rows / 256)` over a row count and
//! `residual_add.cu:25`'s `ceil(n / 256)` is over ELEMENTS, which is
//! `Elementwise` and is already a row. `Slab`, `Tile16`, `AxialRope`,
//! `RoutedQmv` and `WarpTiledScan` have no launcher of their shape in this
//! family. `RowsPerHead` does — four of them, digit for digit — and it is the
//! one entry below where the reason CHANGED without the answer changing.

//! * `altup.cu` was split too, into `norm/altup.cuh`, and it is [`ALTUP`]
//!   now — two rows, both symbols matched, and the refusal below is kept
//!   because a blocker retracted without its reasoning is a blocker that
//!   comes back. It read: both its launchers build a `dim3(T, K, ceil(H/128))`
//!   grid at 128 threads, a third grid axis is no longer the vocabulary's gap
//!   — [`LaunchRule::WarpTiledScan`] produces one, and at this exact block
//!   width — and it is still not this launch. Two things separate them and
//!   either is enough. `warp_tiled_scan`'s third axis is `ceil(V_d / 4)`,
//!   four value channels per warp; this one is `ceil(H / 128)`, one channel
//!   per THREAD, so the rule would open a thirty-second of the blocks and
//!   leave `31/32` of the hidden size untouched. And `grid.y` here is `K`,
//!   the ALTUP STREAM count, which every headed rule fills from
//!   `Dims::kv_heads` — an attention head count, so a rule that took this
//!   grid would not refuse, it would launch a residual-stream axis sized by
//!   the attention configuration.
//!
//!   *"The gap is a `Dims` that carries a stream count and a rule that tiles
//!   by one, and both are the vocabulary's."* Both landed:
//!   [`kernels_cuda_new::Dims::altup_streams`] is the field and
//!   [`LaunchRule::AltUpStreams`] is the rule, cited at `norm/altup.cu:18-19`
//!   and `:32-33`. The field is a DISTINCT QUANTITY and not a sentinel in
//!   `kv_heads`, for the reason §22 gives for `stated_head_dim`: seven rules
//!   read `kv_heads` and would have started answering a stream count.
//! * `hc_post` and `hc_expand` **were** the entry above, and are now
//!   [`DSV4_HC_SIGS`]`[5]` and `[6]`. The refusal read: they size their grid
//!   on the INPUT width and are FLAT — `<<<(N * hidden_size + 255) / 256,
//!   256>>>`, the row folded into the index — while `LaunchRule::Elementwise`
//!   reads `rows * width` at the OUTPUT width, which is `hc_mult` times
//!   wider, and `LaunchRule::SplitPacked` — the only rule that then read
//!   `in_width` — puts the row on `grid.y` and covers `1 / N` of the
//!   elements. Every word of that is still true and the conclusion is not:
//!   **[`LaunchRule::ElementwiseIn`] landed, and it was derived from THESE
//!   TWO LAUNCHERS.** `runtime::launch::elementwise_in` quotes
//!   `norm/dsv4_hc.cu`'s three lines verbatim and reproduces them digit for
//!   digit. The refusal was correct on the day it was written and stale the
//!   day the rule it asked for arrived, which is the failure mode a refusal
//!   citing its launcher and its line is meant to make cheap to overturn.
//! * `attn_sink_correction` and `per_head_rmsnorm` now have rows, and they
//!   state `LaunchRule::GatedRms` rather than the `PerHead` their `.cuh`
//!   prose predicted. See [`DSV4_HC_SIGS`] for the transpose that reading
//!   would have launched.
//! * `rmsnorm_bf16`, `rmsnorm_gemma_bf16`, `rmsnorm_no_scale_bf16` and
//!   `rmsnorm_gated_bf16` are each named by a symbol with TWO readings:
//!   `OpKind::RmsnormPerHead` lowers to the same symbol and needs
//!   `rows · (width / head_dim)` blocks. `LaunchRule::Rms` reads `dims.rows`
//!   alone, so a row would norm gemma-4's whole q projection as one row.
//!
//!   **[`LaunchRule::RowsPerHead`] states exactly that conditional, and the
//!   four are now rowed.** `runtime::launch::rows_per_head` is written to
//!   these launchers digit for digit — `norm/rmsnorm.cu:85-98`, `:259-271`,
//!   `:283-285` and `:311-313` — and answers `rows` blocks when the
//!   statement named no per-head width and `rows · (width / head_dim)` when
//!   it named one, which is `table/norm.rs:36`'s `IfPresent(PerHeadDim, …)`
//!   in the only two terms a launch has.
//!
//!   **What it was missing was the ABSENT case, and the fix was a second
//!   field rather than a different value in the first.** `driver-cuda`'s
//!   `jit_dims` fills `head_dim: spec.per_head_dim.unwrap_or_else(|| extent(ctx.head_dim))`
//!   at `driver-cuda/src/bind/mod.rs:1320` — deliberately, so that a
//!   statement naming a head width beats the fire's, which is the defect
//!   `driver-metal`'s `stated_head.unwrap_or(geometry.head_dim)` records
//!   having had. Under that filler alone the fire's ATTENTION head width
//!   stood in wherever the statement named none and the zero never arrived.
//!   [`crate::runtime::launch::Dims::stated_head_dim`] is the second, distinct
//!   quantity: the width THE STATEMENT named, zero when it named none, filled
//!   from `spec.per_head_dim` with no fallback at all. `head_dim` still means
//!   what it meant and still wins over the fire's; `stated_head_dim` is the
//!   only field on `Dims` whose zero is a value rather than a refusal, and
//!   [`LaunchRule::RowsPerHead`] is the only rule that reads it.
//!
//!   **Zeroing `head_dim` itself would have been the cheaper edit and it is
//!   wrong**: `PerHead`, `PerHeadElementwise`, `GatedRms`, `Rope`,
//!   `RecurrentScan`, `AxialRope` and `WarpTiledScan` all pass it to
//!   `headed(…)`, and none of their statements set `per_head_dim` either, so
//!   all seven would begin refusing every fire. `tests/launch_rules.rs`'s
//!   `the_two_head_widths_are_independent` holds that line.
//!
//!   **What a row written before the field would have done: launch sixteen
//!   times the blocks.** A plain `Rmsnorm` of 2048 channels with 128-wide
//!   heads arrived as `head_dim = 128`, took the second arm, and opened
//!   `rows · 16` blocks — each of them norming a whole row's `hidden`
//!   channels from a sixteenth of a row's offset. `Ungeometric` never fires,
//!   because `width % head_dim == 0` holds; the launch runs, the tower
//!   answers, and nothing compares it to anything. That defect is now a test
//!   that reproduces it ON DEMAND rather than a paragraph:
//!   `tests/launch_rules.rs::the_absent_arm_is_not_the_fires_head_width`
//!   computes the 16 vs 256 blocks, and
//!   `tests/rows_per_head.rs::the_sixteen_times_grid_is_a_wrong_answer_and_not_a_crash`
//!   FIRES it on the device over an oversized buffer and counts the 491520
//!   values it writes past the fire's rectangle while the rectangle itself
//!   stays byte-identical — a launch that "works" and is wrong.
//!
//!   The AOT operand side could always see the absence and the geometry side
//!   could not: `Source::IfPresent` reads `spec.per_head_dim` directly,
//!   before the filler, so the ahead-of-time rows have been right about this
//!   the whole time. `stated_head_dim` is that same read, reaching the
//!   geometry side. Five rows waited on it — these four and
//!   `rmsnorm_gated_f32_in` below — and all five are stated at the end of
//!   [`RMSNORM_ROWS`], each citing its launcher by file and line.
//! * The three VECTORISED RMSNorm kernels — `rmsnorm_vec8<BLOCK,
//!   WEIGHT_PLUS_ONE, EMIT_FP16>`, `residual_add_rmsnorm_vec8<BLOCK>` and
//!   `rmsnorm_rasr_vec8<BLOCK>` — are reached only through
//!   `rmsnorm_vec8_ok(...)`, a RUN-TIME test of three pointer alignments and
//!   three strides. **One of the three is now rowed, and the entry above is
//!   the reason the other two are not yet.** `rmsnorm_vec8` is named by
//!   [`RMSNORM_ROWS`]`[4]` and chosen at fire time by
//!   [`RMSNORM_STRIDED_VEC8`]: the predicate is six clauses over three
//!   addresses and three strides, every one of them an operand the fire has
//!   already bound, so a `Select` over [`crate::device::Fact`]s answers it
//!   with no device read and no synchronisation. What was true and is still
//!   true is that a row cannot FREEZE an arm — a row that always took the
//!   vector path would read past an unaligned row's end. What was wrong is
//!   the conclusion that the decision therefore belongs to a host `if`: nvcc
//!   had to choose its instantiations months before a pointer existed, and
//!   NVRTC does not.
//!
//!   The other two are unblocked by the same shape and are not taken here,
//!   for reasons that are theirs rather than the design's:
//!   `residual_add_rmsnorm_vec8` needs a SEVENTH clause the launcher spells
//!   inline — `residual % 16 == 0`, which is one more `Term::Aligned` and no
//!   new vocabulary — and `rmsnorm_rasr_vec8`'s scalar twin is not a row at
//!   all, because its launcher is 512 threads wide against `Rule::Rms`'s 256,
//!   so there is no base row to specialise. A specialisation is a second
//!   instantiation of an existing contract; it cannot conjure the contract.
//!
//!   The rows that stay scalar are still what the vectorised forms were
//!   measured against — bit-identical at hidden 2048/2816/5376 for the `rasr`
//!   pair, 0 of those values differing. A row is slower there, never wrong.
//! * `rmsnorm_residual_add_scale_rmsnorm` is the SCALAR fallback of that same
//!   decision, and it is the one kernel here the multi-argument finding
//!   changes the story for without changing the answer. Its `BLOCK` has no
//!   default, so it was unspellable until `elem` was shown to carry an
//!   argument list; it is spellable now and still unrowable, because
//!   `rmsnorm.cu` instantiates it at `constexpr int BLOCK = 512` and
//!   `Rule::Rms` launches 256. Compiled at 512 and launched at 256 it reduces
//!   through `buf[256..512]` that no thread wrote — a finite, wrong norm.
//!   Compiled at 256 to match the rule it is correct and it is no longer the
//!   kernel the launcher shipped: the sweep in `rmsnorm.cu` put scalar/512 at
//!   3.68/4.83/6.55 us against scalar/256's 4.38/6.17/8.48 at hidden
//!   2048/2816/5376. Neither arm is a row worth writing.
//! * `rmsnorm_gated_f32_in` launches `<<<num_rows, 256>>>`, which is `Rms` to
//!   the digit, and is blocked one level up: its ahead-of-time row states
//!   `num_rows <- Mul(Rows, Gdn("v_h"))` — one row per (token, VALUE HEAD) —
//!   and `hidden <- Gdn("v_d")`. `Rule::Rms` reads `dims.rows`, so the row
//!   would launch the right kernel over `1 / v_h` of its rectangle.
//!   `driver_internal.rs` records that the hybrid's prefill caught exactly
//!   this, and only because the walk asserts every launch ran.
//!
//!   [`LaunchRule::RowsPerHead`] computes `rows · (width / head_dim)`, which
//!   IS `rows · v_h` — but only if the head width it reads carries `v_d`.
//!   It now reads [`crate::runtime::launch::Dims::stated_head_dim`], which carries
//!   what the STATEMENT named and nothing else, so the coincidence that used
//!   to threaten this row — a GDN fire whose value head is 128 wide and whose
//!   attention head is also 128 wide, right by accident and wrong the first
//!   time a config separates them — cannot occur: the attention width does
//!   not reach the rule at all.
//!
//!   The row is stated. Its remaining gap is the binder's and is named in the
//!   row's own doc: `OpKind::RmsnormGated` does not set `spec.per_head_dim`,
//!   and `v_d` lives in `GdnCtx` which `jit_dims` is not handed, so a GDN
//!   fire arrives with `stated_head_dim = 0` and takes the ABSENT arm —
//!   `rows` blocks over a rectangle whose `rows` is already `tokens · v_h`,
//!   which is what `Rule::Rms` would have done and is right whenever the
//!   binder counts the rectangle in value-head rows. The rule is
//!   correct-by-construction the day `jit_dims` can see `v_d`, and
//!   `tests/rows_per_head.rs` fires this symbol BOTH ways to show it.
//!
//! # The ceiling that was not one, and what it moved here
//!
//! `DeviceKernel::elem` was read across the migration as naming a TYPE, which
//! made every `__global__` with a second template argument unrowable. It was
//! measured and it is false — `elem` is pasted between angle brackets and
//! NVRTC parses C++ there, so an argument list works and
//! [`crate::device::args`] carries the measurement. Thirty-seven kernels came
//! off that list tree-wide and **not one of them is in this family**: the
//! eight unrowed kernels in `rmsnorm.cuh` were each checked against it
//! individually and every one has a second blocker that stands on its own —
//! a per-head second reading, a GDN head rectangle, a run-time alignment
//! branch, or a `BLOCK` the launcher fixes at 512 against a rule that fixes
//! 256. The finding is recorded here anyway, because the tree-wide count
//! names `rmsnorm.cuh` first and a reader arriving with that number should
//! find out why it bought nothing rather than conclude the work was missed.
//!
//! What it did buy is honesty in the rows that already existed: the seven
//! `<class T, int BLOCK>` templates this file states now spell their width
//! instead of inheriting a default that happened to match. See
//! [`DSV4_HC_ROWS`].
//!
//! Each `.cuh` repeats its own share of that list beside the kernel it is
//! about. Nothing here was left unrowed because it was awkward: a rule that
//! computes the wrong extent is worse than a launcher that still exists.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::device::{Arm, Specialisation, Take, Term};
use crate::source::roots;
use crate::unit::Unit;

/// `norm`'s six AltUp auxiliary templates, and the fp16 `tanh` the
/// ahead-of-time build never had.
pub const ALTUP_AUX: Unit = Unit {
    name: "norm/altup_aux",
    root: roots::NORM_ALTUP_AUX,
    rows: crate::device::ALTUP_AUX,
    options: &[],
};

/// `norm`'s pointwise pair: `residual_add` and `scalar_mul`.
pub const ELEMENTWISE: Unit = Unit {
    name: "norm/elementwise",
    root: roots::NORM_ELEMENTWISE,
    rows: crate::device::ELEMENTWISE,
    options: &[],
};

/// A bias added onto rows already in place, contiguous and strided.
pub const ADD_BIAS: Unit = Unit {
    name: "norm/add_bias",
    root: include_str!("../../csrc/src/norm/add_bias.cuh"),
    rows: ADD_BIAS_ROWS,
    options: &[],
};

/// DeepSeek-V4's hyper-connection kernels — the five of the seven a ported
/// rule states, three of them one block per row and two of them one block per
/// (row, head).
pub const DSV4_HC: Unit = Unit {
    name: "norm/dsv4_hc",
    root: include_str!("../../csrc/src/norm/dsv4_hc.cuh"),
    rows: DSV4_HC_ROWS,
    options: &[],
};

/// The RMSNorm family proper — the four scalar kernels whose launcher was
/// `<<<rows, 256>>>` and nothing else.
pub const RMSNORM: Unit = Unit {
    name: "norm/rmsnorm",
    root: include_str!("../../csrc/src/norm/rmsnorm.cuh"),
    rows: RMSNORM_ROWS,
    options: &[],
};

/// gemma-3n's AltUp predict and correct — the pair `altup.cuh` was written
/// for and left rowless.
///
/// **The rowless state was a vocabulary gap and it is closed.**
/// [`LaunchRule::AltUpStreams`] states `dim3(T, K, ceil(H / 128))` at 128
/// threads against `norm/altup.cu:18-19` and `:32-33`, and
/// [`kernels_cuda_new::Dims::altup_streams`] is the field `K` lives in — its
/// own doc says why it could not be `kv_heads` (an attention head count, and
/// [`LaunchRule::WarpTiledScan`] already reads it) or `n_experts` (the
/// router's vocabulary, and both numbers coexist in a gemma-3n fire).
///
/// **`H` is not the rectangle's width and neither row pretends it is.** The
/// value is `[K, tokens, H]`, so `Dims::width` is `K * H` and the rule
/// divides — exactly as [`crate::table::norm`]'s ahead-of-time rows already
/// do with `Source::Div(Width(In(0)), CtxNonZero("altup_streams"))`. A width
/// that does not divide by the stream count is refused rather than floored.
///
/// Both symbols match [`crate::table::KERNELS`], so both rows move
/// `examples/migration_status`.
pub const ALTUP: Unit = Unit {
    name: "norm/altup",
    root: include_str!("../../csrc/src/norm/altup.cuh"),
    rows: ALTUP_ROWS,
    options: &[],
};

/// The units `norm` compiles.
pub static UNITS: &[Unit] = &[ALTUP_AUX, ALTUP, ELEMENTWISE, ADD_BIAS, DSV4_HC, RMSNORM];

/// [`ALTUP`]'s instantiations — one template argument each, and it is a TYPE,
/// so these are the ordinary shape and not [`DeviceKernel::PLAIN`]'s.
static ALTUP_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ALTUP_SIGS[0],
        template_path: "norm::device::altup_predict",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ALTUP_SIGS[1],
        template_path: "norm::device::altup_correct",
        elem: "device::bf16",
    },
];

/// The contracts, in [`ALTUP_ROWS`]' order.
///
/// **Each row's operand list is the KERNEL's, which is the launcher's minus
/// `T` and the stream.** `altup_predict_bf16` takes `(streams, coefs,
/// predictions, K, T, H, stream)` and hands the kernel `(streams, coefs,
/// predictions, K, T, H)`; the rule recovers `T` from [`Source::Rows`] and
/// the binder supplies the stream, so seven operands become six. `K` and `H`
/// STAY, because the kernels read them — `if (t >= T_len || k >= K || h >= H)
/// return;` is the first line of both — and a grid is not readable from
/// inside a kernel. `T_len` stays for the same reason even though the rule
/// computes `grid.x` from it.
///
/// `coefs` is `F32s` and not `Buf` in both: the coefficients are a small
/// dense matrix the host computes and `altup.cuh` says rounding them to `T`
/// "would change the sum this kernel exists to make exact". A row that spelled
/// them `Buf` would follow the element type the day a second format lands.
#[rustfmt::skip]
static ALTUP_SIGS: [KernelSig; 2] = [
    // `norm/altup.cu:18-19` --
    // `device::altup_predict<device::bf16><<<dim3(T, K, ceil(H/128)), 128, 0, stream>>>`.
    kernel!(altup_predict "norm::altup_predict_bf16",
        file = Some("norm/altup.cuh"),
        launch = LaunchRule::AltUpStreams,
        operands = operands![
            streams: Buf <- Source::In(0),
            coefs: F32s <- Source::In(1),
            predictions: BufMut <- Source::Out(0),
            k: I32 <- Source::Ctx("altup_streams"),
            t: I32 <- Source::Rows,
            h: I32 <- Source::Div(&Source::Width(&Source::In(0)), &Source::CtxNonZero("altup_streams")),
        ]),
    // `norm/altup.cu:32-33` --
    // `device::altup_correct<device::bf16><<<dim3(T, K, ceil(H/128)), 128, 0, stream>>>`.
    //
    // `k` comes off the COEFFICIENT width and `h` off the ACTIVATED width
    // here, not off the context and a quotient -- the ahead-of-time row makes
    // the same two readings, because this launch is handed both tensors and
    // `correction_coefs_plus_one` is `[tokens, K]` while `activated` is
    // `[tokens, H]`. Two operands that name the extents outright are better
    // than a context lookup and a division, and the rule still reads
    // `Dims::altup_streams`: the operands bound the kernel's guard and the
    // rule sizes the grid, which is the split `Rms` keeps for `numel`.
    kernel!(altup_correct "norm::altup_correct_bf16",
        file = Some("norm/altup.cuh"),
        launch = LaunchRule::AltUpStreams,
        operands = operands![
            predictions: Buf <- Source::In(0),
            activated: Buf <- Source::In(1),
            correction_coefs_plus_one: F32s <- Source::In(2),
            corrected: BufMut <- Source::Out(0),
            k: I32 <- Source::InWidth(2),
            t: I32 <- Source::Rows,
            h: I32 <- Source::InWidth(1),
            active_idx: I32 <- Source::Ctx("altup_active"),
        ]),
];

/// [`ADD_BIAS`]'s instantiation.
///
/// One symbol now. `add_bias_strided` was the second, and it was the second
/// because `add_bias.hpp` declares both — `new-horizon.md` §28.4 measured the
/// row as a header overload nothing states, and it went with its row. The
/// `__global__` stays in `norm/add_bias.cuh`; only the claim that some fire
/// wants it has gone.
static ADD_BIAS_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ADD_BIAS_SIGS[0],
        template_path: "norm::device::add_bias",
        elem: "device::bf16",
    },
];

/// The contracts, in [`ADD_BIAS_ROWS`]' order.
///
/// It is its ahead-of-time twin minus the stream — `cuLaunchKernel`'s sixth
/// PARAMETER, outside the `void**` — and minus `num_rows`, which
/// `LaunchRule::RouteRows` recovers from the fire's rectangle. Two operands
/// out of five.
#[rustfmt::skip]
static ADD_BIAS_SIGS: [KernelSig; 1] = [
    // `RouteRows` -- one block per row, the block as wide as the row rounded
    // to a warp. The launcher was `<<<num_rows, 256>>>` with a stride loop
    // over `dim`, so the rule's wider block reaches the same elements in
    // fewer iterations and the arithmetic per element is unchanged.
    //
    // In place over the value it biases -- one operand, one result, the same
    // bytes -- so `out` binds from `Out(0)` and the staging comes off the
    // pair. The bias is the statement's named weight, like the embedding's
    // table.
    kernel!(add_bias "norm::add_bias_bf16",
        file = Some("norm/add_bias.cuh"),
        launch = LaunchRule::RouteRows,
        in_place = &[(0, 0)],
        operands = operands![
            out: BufMut <- Source::Out(0),
            bias: Buf <- Source::WeightNamed,
            dim: I32 <- Source::OutWidth(0),
        ]),
];

/// [`DSV4_HC`]'s instantiations — the three `Rms` states and the two
/// [`LaunchRule::GatedRms`] does.
///
/// The first three spell TWO template arguments. `hc_pre_postprocess`,
/// `hc_head_postprocess` and `hc_rmsnorm_to_f32` are
/// `template <class T, int BLOCK = 256>`, and `elem` carries an argument list
/// as readily as a type — see [`crate::device::args`], which measured
/// `nvrtcAddNameExpression` accepting `probe::scaled<float, 128>` on an L40S.
/// The default would have produced the same instantiation silently. It is
/// written out because a non-type argument is a value the kernel is compiled
/// AGAINST: `block_reduce_sum_exact<BLOCK>` unrolls its tree to `BLOCK` and
/// reduces through a static `__shared__ float[BLOCK]`, so the width is not a
/// tuning knob but the size of an array the launch must match. THREE numbers
/// have to agree for these rows to be correct — `dsv4_hc.cu`'s
/// `constexpr int BLOCK = 256`, the template's own default, and
/// `runtime::launch`'s `const BLOCK: u32 = 256` that `Rule::Rms` puts in
/// `block.x` — and until this line they agreed by coincidence of three
/// independent 256s, none of which named the others. Retune the rule to 512
/// and the launch brings 512 threads to a 256-entry reduction buffer: half
/// of them write past it, the sum is whatever those neighbours held, and the
/// norm is finite and wrong. Spelling it here does not make the compiler
/// check it, but it puts the coupling where a reader changing one of the
/// three will see it.
static DSV4_HC_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DSV4_HC_SIGS[0],
        template_path: "norm::device::hc_pre_postprocess",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &DSV4_HC_SIGS[1],
        template_path: "norm::device::hc_head_postprocess",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &DSV4_HC_SIGS[2],
        template_path: "norm::device::hc_rmsnorm_to_f32",
        elem: "device::bf16, 256",
    },
    // `attn_sink_correction` and `per_head_rmsnorm` are `template <class T>`
    // and take no width at all — they read `blockDim.x`. One argument here is
    // the whole list, not an elision of one.
    DeviceKernel {
        sig: &DSV4_HC_SIGS[3],
        template_path: "norm::device::attn_sink_correction",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_HC_SIGS[4],
        template_path: "norm::device::per_head_rmsnorm",
        elem: "device::bf16",
    },
    // The two flat scatters. `template <class T>` and nothing else: their
    // grid is `ceil(N * H / 256)` and the block width is never a template
    // argument, so one argument here is the whole list.
    DeviceKernel {
        sig: &DSV4_HC_SIGS[5],
        template_path: "norm::device::hc_post",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_HC_SIGS[6],
        template_path: "norm::device::hc_expand",
        elem: "device::bf16",
    },
];

/// The contracts, in [`DSV4_HC_ROWS`]' order.
///
/// The first three launched `<<<N, 256>>>`, which is `LaunchRule::Rms`: one
/// block per token, 256 threads, the block striding the row. So `n` and the
/// stream leave those rows, and nothing else does.
///
/// The last two launched `dim3 grid(N, num_heads); dim3 block(256)`, which is
/// [`LaunchRule::GatedRms`] — the row on `grid.x`, the head on `grid.y`, 256
/// threads. **Not `PerHead`**, though both `.cuh` doc comments predicted it
/// before the port existed and the name reads like the shape. `PerHead` was
/// derived from `attn/head_dim_pad.cu` and computes
/// `grid(heads, rows)` at 128 threads: the TRANSPOSE, at half the width.
/// Handing it either kernel here swaps `n` and `h` in every block — the
/// launch is legal, the tensor comes out fully written, and each cell holds
/// another head's answer. `runtime::launch::gated_rms` names both of these
/// launchers as the ones it was checked against, which is why the rule is
/// right by construction and the prose it contradicts is not.
///
/// The last two launched `<<<ceil(N · hidden_size / 256), 256>>>` — flat over
/// what they READ, which is [`LaunchRule::ElementwiseIn`] and the pair
/// `runtime::launch::elementwise_in` was ported from. See the rows themselves
/// for why `Elementwise` is not merely a coarser answer there.
#[rustfmt::skip]
static DSV4_HC_SIGS: [KernelSig; 7] = [
    // The mix split, the Sinkhorn, and the collapse of `hc_mult` residual
    // streams into the layer's input -- one block per token because `pre`,
    // `post` and `comb` live in shared memory across all of it.
    kernel!(hc_pre "norm::hc_pre_postprocess_bf16",
        file = Some("norm/dsv4_hc.cuh"),
        launch = LaunchRule::PerRow,
        // `PerRow`, not `Rms`. The launcher is `<<<N, BLOCK, 0, stream>>>`
        // in `norm/dsv4_hc.cu`, and every reduction buffer in
        // `dsv4_hc.cuh` is STATIC -- `__shared__ float pre[MAX_HC_MULT]`
        // and its two siblings, sized by the template rather than by the
        // launch. So `Rms`'s thirty-two dynamic bytes are memory no
        // launcher passes and no kernel reads. Harmless in effect and
        // wrong as a contract: a rule is meant to REPRODUCE its launcher,
        // and one that asks for memory the launcher did not is a rule
        // nobody can check against the `<<<>>>` it came from.
        operands = operands![
            mixes: F32s,
            scale: F32s,
            base: F32s,
            residual: Buf,
            post_mix: F32sMut,
            comb_mix: F32sMut,
            layer_input: BufMut,
            hc_mult: I32,
            hidden_size: I32,
            hc_eps: F32,
            hc_post_alpha: F32,
            sinkhorn_iters: I32,
        ]),
    // The head variant: a gated sum with no Sinkhorn. `hc_eps` sits LAST
    // because the twin's C++ signature put the stream before it, and dropping
    // the stream is the only change a row may make to that order.
    kernel!(hc_head "norm::hc_head_postprocess_bf16",
        file = Some("norm/dsv4_hc.cuh"),
        launch = LaunchRule::PerRow,
        // `PerRow`, not `Rms`. The launcher is `<<<N, BLOCK, 0, stream>>>`
        // in `norm/dsv4_hc.cu`, and every reduction buffer in
        // `dsv4_hc.cuh` is STATIC -- `__shared__ float pre[MAX_HC_MULT]`
        // and its two siblings, sized by the template rather than by the
        // launch. So `Rms`'s thirty-two dynamic bytes are memory no
        // launcher passes and no kernel reads. Harmless in effect and
        // wrong as a contract: a rule is meant to REPRODUCE its launcher,
        // and one that asks for memory the launcher did not is a rule
        // nobody can check against the `<<<>>>` it came from.
        operands = operands![
            mixes: F32s,
            scale: F32s,
            base: F32s,
            residual: Buf,
            out: BufMut,
            hc_mult: I32,
            hidden_size: I32,
            hc_eps: F32,
        ]),
    // RMSNorm that widens to fp32 on the way out, for the mix GEMM that runs
    // in fp32. The reduction is a fixed-order tree on STATIC shared memory,
    // so the 32 bytes `Rms` hands the launch are never read -- the rule sizes
    // `device::block_sum`'s scratch, and this kernel does not use it.
    kernel!(hc_rmsnorm_to_f32 "norm::hc_rmsnorm_to_f32",
        file = Some("norm/dsv4_hc.cuh"),
        launch = LaunchRule::PerRow,
        // `PerRow`, not `Rms`. The launcher is `<<<N, BLOCK, 0, stream>>>`
        // in `norm/dsv4_hc.cu`, and every reduction buffer in
        // `dsv4_hc.cuh` is STATIC -- `__shared__ float pre[MAX_HC_MULT]`
        // and its two siblings, sized by the template rather than by the
        // launch. So `Rms`'s thirty-two dynamic bytes are memory no
        // launcher passes and no kernel reads. Harmless in effect and
        // wrong as a contract: a rule is meant to REPRODUCE its launcher,
        // and one that asks for memory the launcher did not is a rule
        // nobody can check against the `<<<>>>` it came from.
        operands = operands![
            input: Buf,
            output: F32sMut,
            dim: I32,
            eps: F32,
        ]),
    // The sink correction, IN PLACE on the attention output it rescales --
    // one operand, one result, the same bytes, so `attn_out` binds from
    // `Out(0)` and the staging comes off the pair.
    //
    // `N` leaves and `num_heads` stays, and the split is the KERNEL's
    // parameter list rather than a preference: the rule recovers the row
    // count as `grid.x` and this kernel never reads it, while `num_heads` is
    // the stride it addresses `lse[n * num_heads + h]` and its own row with.
    // Dropping a parameter a `__global__` declares is not a shorter row, it
    // is a `void**` one entry short — and `cuLaunchKernel` reads the missing
    // argument from whatever follows the array.
    //
    // The head count comes off the RESULT's width over the context's
    // `head_dim`, exactly as the ahead-of-time row computes it, because the
    // join has never carried a rank-3 value's second extent. `head_dim`
    // itself stays an operand because the kernel strides over it.
    kernel!(attn_sink_correction "norm::attn_sink_correction_bf16",
        file = Some("norm/dsv4_hc.cuh"),
        launch = LaunchRule::GatedRms,
        in_place = &[(0, 0)],
        operands = operands![
            attn_out: BufMut <- Source::Out(0),
            lse: F32s <- Source::In(1),
            sink: F32s <- Source::Weight(0),
            num_heads: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::CtxNonZero("head_dim")),
            head_dim: I32 <- Source::Ctx("head_dim"),
        ]),
    // The same grid, and the kernel that proves the head axis is load-bearing
    // rather than convenient: `num_heads` is not a parameter here at all --
    // `const int num_heads = gridDim.y;` -- so the count the rule puts on
    // `grid.y` IS the count the kernel strides by. A rule that folded the
    // head axis away would not waste blocks; it would tell the kernel there
    // is one head and walk every row one head's width apart.
    //
    // Three operands where the ahead-of-time twin takes six: `n` and
    // `num_heads` are the rule's grid and the stream never was an operand.
    // The norm is in place on the q it normalises.
    kernel!(per_head_rmsnorm "norm::per_head_rmsnorm_bf16",
        file = Some("norm/dsv4_hc.cuh"),
        launch = LaunchRule::GatedRms,
        in_place = &[(0, 0)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            head_dim: I32 <- Source::Ctx("head_dim"),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // THE SCATTER BACK ACROSS THE RESIDUAL STREAMS, and the row that
    // overturns this family's longest-standing geometric refusal.
    //
    // The launcher is `norm/dsv4_hc.cu`:
    //
    // ```text
    // constexpr int BLOCK = 256;
    // const long long total = static_cast<long long>(N) * hidden_size;
    // if (total <= 0 || hc_mult > device::MAX_HC_MULT) return;
    // const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    // device::hc_post<device::bf16><<<grid, BLOCK, 0, stream>>>(...);
    // ```
    //
    // `Rule::ElementwiseIn` evaluates `ceil(rows * in_width / 256)` blocks of
    // 256 with no shared memory, and `in_width` is `x`'s row — `[N, H]` — so
    // the two agree on grid, block and smem for every rectangle. There is no
    // template argument to cite: `hc_post` is `template <class T>` and reads
    // its width from `blockDim.x` through a flat index, so the 256 that the
    // rule launches is not baked into any instantiation.
    //
    // **`Elementwise` is the reading this row exists to decline.** Sized on
    // `Dims::width` — the OUTPUT's `M * H` — it issues `M` times the blocks
    // and throws `M - 1` of every `M` away on the kernel's own
    // `if (idx >= N * H) return;`. Correct, up to eightfold, and unfalsifiable
    // against the `<<<>>>` it claims to reproduce, because the two agree on
    // the answer and differ on the launch.
    //
    // `n` stays an operand although the rule recovers the grid from it: the
    // grid rounds UP, and the kernel's guard is the only thing that stops the
    // last block's tail threads from indexing `x` past its end.
    //
    // **`MAX_HC_MULT` moved from the launcher into the kernel.** `hc_post`
    // holds its `M` residual values in a `float r[MAX_HC_MULT]` register
    // array because it runs in place, and the ahead-of-time launcher refused
    // `hc_mult > MAX_HC_MULT` on the host rather than let a thread write past
    // it. No `LaunchRule` carries a precondition on an operand's VALUE and no
    // `Source` computes one, so a row could not have inherited that check —
    // and a fire is a `void**` and a grid, with no host in front of it. The
    // guard is now `dsv4_hc.cuh`'s first statement. It changes nothing the
    // archive does, because the archive never launched the case it catches.
    kernel!(hc_post "norm::hc_post_bf16",
        file = Some("norm/dsv4_hc.cuh"),
        launch = LaunchRule::ElementwiseIn,
        operands = operands![
            x: Buf,
            residual: Buf,
            post_mix: F32s,
            comb_mix: F32s,
            out_residual: BufMut,
            n: I32,
            hc_mult: I32,
            hidden_size: I32,
        ]),
    // The degenerate mixer: one stream broadcast into `M`, the launch
    // `hc_post`'s to the digit and for the same reason — a thread owns one
    // INPUT element and loops `M` writes over it, so the extent the rule
    // states is the extent the kernel bounds itself by.
    //
    // Its `Source`s are its ahead-of-time twin's unchanged. `hc_mult` is the
    // output's width over the input's, which is the one place the `M` this
    // kernel loops over is recoverable from the rectangle: the rule reads
    // `in_width` and the operand reads the ratio, and they are derived from
    // the same two extents rather than from each other.
    kernel!(hc_expand "norm::hc_expand_bf16",
        file = Some("norm/dsv4_hc.cuh"),
        launch = LaunchRule::ElementwiseIn,
        operands = operands![
            input: Buf <- Source::In(0),
            output: BufMut <- Source::Out(0),
            n: I32 <- Source::Rows,
            hc_mult: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::Width(&Source::In(0))),
            hidden_size: I32 <- Source::InWidth(0),
        ]),
];

/// [`RMSNORM`]'s instantiations — the four scalar kernels a `Rule::Rms` row
/// states, the one vectorised kernel a fire can choose, and the five
/// [`LaunchRule::RowsPerHead`] rows that were held on a `Dims` field.
///
/// The first four are `template <class T, int BLOCK = 256>` and all four
/// spell the width, for the reason [`DSV4_HC_ROWS`] gives at length: `BLOCK`
/// sizes the `__shared__ float[BLOCK]` these kernels reduce through and fixes
/// the unroll of `block_reduce_sum_exact<BLOCK>`, so it is part of the launch
/// contract and not a default worth inheriting quietly. `rmsnorm.cu` writes
/// `constexpr int BLOCK = 256` above each of those four `<<<>>>`s; the rows
/// say the same number in the same place a reader looks.
///
/// The fifth is `rmsnorm_vec8`, which this file's own header spent a
/// paragraph explaining why no row could name. That paragraph is now half
/// wrong and the half that survives is in [`RMSNORM_STRIDED_VEC8`].
///
/// **Rows six to ten are the five this file held**, and they are at the end
/// rather than beside their kin because [`RMSNORM_STRIDED_VEC8`] names
/// [`RMSNORM_ROWS`]`[4]` by index: a row inserted above it would move the
/// specialisation's base silently, and `Specialisation::agrees` would then be
/// checking an arm against a contract that is not its own. Appending is the
/// operation this list is safe under.
///
/// **Two of them instantiate a template another row already asked for**, and
/// that is not a mistake. `rmsnorm_bf16` IS `rmsnorm_strided_bf16` —
/// `rmsnorm.cu:38-44`'s whole body is `rmsnorm_strided_bf16(x, weight, y,
/// num_rows, hidden, hidden, hidden, eps, stream)`, the strides being the
/// width — so both rows name `norm::device::rmsnorm<device::bf16, 256>` and
/// resolve to one `CUfunction`. `nvrtcAddNameExpression` takes the same
/// string twice, `nvrtcGetLoweredName` answers the same mangled name twice,
/// and `KernelModule::load_mangled` pairs by the ROW's symbol, so the two
/// rows differ where they must: in their operands and in their RULE.
static RMSNORM_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &RMSNORM_SIGS[0],
        template_path: "norm::device::rmsnorm",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &RMSNORM_SIGS[1],
        template_path: "norm::device::residual_add_rmsnorm",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &RMSNORM_SIGS[2],
        template_path: "norm::device::rmsnorm_residual_add",
        elem: "device::bf16, 256",
    },
    // The fourth row is not a fourth kernel: it is the SAME contract as the
    // first, compiled out of `rmsnorm_vec8` instead of `rmsnorm`, and it
    // exists because a fire can see what an ahead-of-time build could not.
    // See [`RMSNORM_STRIDED_VEC8`] for the predicate and
    // [`RMSNORM_SIGS`]`[3]` for why the template arguments are what they are.
    //
    // `device::i32(256)` and not `256`: `DeviceKernel::instantiation`
    // prefixes the FIRST argument with `::pie_cuda_driver::kernels::`, and
    // `rmsnorm_vec8`'s first template parameter is `int BLOCK` — so a bare
    // literal lands as `::pie_cuda_driver::kernels::256` and NVRTC answers
    // *expected an identifier*. The prelude's `i32` alias
    // (`pie_device.cuh:463`) makes it a qualified constant expression that
    // resolves where the prefix puts it. `crate::device::args` records the
    // eight forms this was measured over.
    DeviceKernel {
        sig: &RMSNORM_SIGS[3],
        template_path: "norm::device::rmsnorm_vec8",
        elem: "device::i32(256), false, false",
    },
    // ── The five that were waiting on a `Dims` ───────────────────────
    //
    // `LaunchRule::RowsPerHead` reads `Dims::stated_head_dim`, which
    // `driver-cuda/src/bind/mod.rs` now fills from `spec.per_head_dim` with
    // no fallback. The rule was written to these five launchers before the
    // field existed, deliberately, so that the binder had something to be
    // fixed AGAINST; the field landed, and these are the rows.
    DeviceKernel {
        sig: &RMSNORM_SIGS[4],
        template_path: "norm::device::rmsnorm",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &RMSNORM_SIGS[5],
        template_path: "norm::device::rmsnorm_gemma",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &RMSNORM_SIGS[6],
        template_path: "norm::device::rmsnorm_no_scale",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &RMSNORM_SIGS[7],
        template_path: "norm::device::rmsnorm_gated",
        elem: "device::bf16, 256",
    },
    DeviceKernel {
        sig: &RMSNORM_SIGS[8],
        template_path: "norm::device::rmsnorm_gated_f32_in",
        elem: "device::bf16, 256",
    },
    // The tenth row: `rmsnorm_vec8` again, with `EMIT_FP16` flipped. See
    // [`RMSNORM_SIGS`]`[9]` for the launcher it cites and for why its block
    // is 256 where the launcher's is 512.
    //
    // `device::i32(256)` for [`RMSNORM_SIGS`]`[3]`'s reason: the first
    // template argument is prefixed with `::pie_cuda_driver::kernels::` by
    // `DeviceKernel::instantiation`, and a bare `256` lands as
    // `::pie_cuda_driver::kernels::256`, which NVRTC answers *expected an
    // identifier* to.
    DeviceKernel {
        sig: &RMSNORM_SIGS[9],
        template_path: "norm::device::rmsnorm_vec8",
        elem: "device::i32(256), false, true",
    },
];

/// The contracts, in [`RMSNORM_ROWS`]' order.
///
/// Every one is `<<<num_rows, 256>>>` in its twin, so `num_rows` and the
/// stream leave and nothing else does. What differs between the first four
/// and the last six is WHERE `num_rows` comes from, which is the whole
/// content of this family's migration:
///
/// * the first four state `LaunchRule::Rms`, whose `num_rows` is
///   `dims.rows` and nothing conditional, because the symbols they name have
///   a SINGLE reading — `OpKind::RmsnormPerHead` does not lower to any of
///   them;
/// * the last six state [`LaunchRule::RowsPerHead`], because their symbols
///   have TWO, and the launcher took the choice as an argument.
///
/// It was five and five until `new-horizon.md` §28.4 measured
/// `residual_add_scale_rmsnorm_bf16` — the second of the `Rms` five — as a
/// row nothing states, and the row went.
///
/// `Rule::Rms` also hands each launch 32 bytes of dynamic shared memory that
/// these kernels never read: they reduce through `block_reduce_sum_exact` on
/// a static `__shared__ float[BLOCK]` — the `BLOCK` each row now spells, and
/// the reason it is worth spelling — which is what makes the sum's ORDER
/// fixed, and the unread bytes cost nothing. `RowsPerHead` requests none, and
/// that is the same claim read the other way.
///
/// `bf16` and only `bf16`, though every template here is written over `T`.
/// A second format costs one row and no C++ — but no fire asks for one, and
/// a row for a kernel nothing states is a claim about a caller that does not
/// exist.
#[rustfmt::skip]
static RMSNORM_SIGS: [KernelSig; 10] = [
    // The strided norm, and the only one of the four plain RMSNorm symbols
    // with a SINGLE reading: `num_rows <- Source::Rows` and nothing
    // conditional, because `OpKind::RmsnormPerHead` does not lower here.
    //
    // The strides are the two values' OWN widths, which is the whole of what
    // "strided" means: a row of `x` is `x_row_stride` wide and only `hidden`
    // of it is read. So `hidden` comes off the RESULT and the strides off
    // each side.
    kernel!(rmsnorm_strided "norm::rmsnorm_strided_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            y: BufMut <- Source::Out(0),
            hidden: I32 <- Source::OutWidth(0),
            x_row_stride: I32 <- Source::InWidth(0),
            y_row_stride: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // Residual add + the next block's pre-norm, fused. Numerically the
    // two-kernel sequence -- the add rounds to bf16 exactly where
    // `norm/elementwise.cuh`'s `residual_add` rounds it, and only then is the
    // sum squared -- which is what makes it a binding a declaration may state
    // rather than a different computation.
    kernel!(residual_add_rmsnorm "norm::residual_add_rmsnorm_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            hidden: BufMut <- Source::In(0),
            residual: Buf <- Source::In(1),
            weight: Buf <- Source::Weight(0),
            norm_out: BufMut <- Source::Out(0),
            hidden_size: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // Norm, then land it on the residual stream. `(x, hidden)` over one
    // result: the stream operand is the one it lands on, so the staging comes
    // off `in_place`.
    kernel!(norm_residual_add "norm::rmsnorm_residual_add_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::Rms,
        in_place = &[(0, 1)],
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            hidden: BufMut <- Source::Out(0),
            hidden_size: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // The vectorised twin of the FIRST row -- the same contract, the same
    // rule, the same values, out of a different template. Every operand is
    // `rmsnorm_vec8`'s own parameter in `rmsnorm_vec8`'s own order, which is
    // `rmsnorm`'s with `y_fp16` inserted third; `RMSNORM_STRIDED_VEC8`'s
    // `take` is what puts the base's values in these slots and the null in
    // that one.
    //
    // # The symbol is unspellable on purpose
    //
    // `#vec8` is not a suffix any C++ symbol carries or any trace can name,
    // and that is the point: `model-compiler` writes `norm::rmsnorm_strided\
    // _bf16` and the dispatcher matches it, so the only two things that ever
    // reach this row are the specialisation and a test that means to. It is
    // a row and not a hidden table entry because a row is what
    // `unit::UNITS` compiles, what `tests/units.rs` proves instantiable on
    // this architecture, what `Unit::cache_key` folds, and what
    // `KernelModule` resolves an entry for -- five mechanisms this arm gets
    // for nothing by being one.
    //
    // # No `Source` on any operand, deliberately
    //
    // The others carry the source their ahead-of-time twin binds from. This
    // row has none because nothing binds it from a fire's rectangle: its
    // values are the BASE row's, already bound and already checked, moved
    // across by `Take`. A `Source` here would be a claim about a binder that
    // never sees this row -- and `Source::In(0)` on an operand nobody sources
    // is exactly the kind of statement nothing checks.
    //
    // # `y_fp16` is `| null` and it is the arm that nulls it
    //
    // `EMIT_FP16` is `false` in this instantiation, so the parameter is read
    // only inside an `if constexpr` that is not compiled -- but it is still a
    // parameter, so the `void**` still needs an eighth cell and
    // `cuLaunchKernel` reads it from whatever follows the array otherwise.
    // The nullability is what lets `Specialisation::agrees` accept
    // `Take::Null` there and refuse it everywhere else.
    //
    // # Every buffer here is unspellable, and not only `y_fp16`
    //
    // `rmsnorm_vec8` templates a block width and two flags and hard-codes
    // `bf16` in its own parameters — the element type is not a template
    // argument at all — so this row's `elem` is `device::i32(256), false,
    // false` and its HEAD is a value. `Buf` and `BufMut` derive their
    // element from that head, which here denotes no type at all, and the
    // fourth parameter is `f16*` besides. Measured with a function-pointer
    // initialisation under nvcc 13.0 `-arch=sm_89`, which admits no
    // conversions: the kernel is `(const bf16*, const bf16*, bf16*, f16*,
    // int, int, int, float)`, and the elem-derived reading matches none of
    // it.
    //
    // Nothing consults that derivation at fire time — a `Buf` crosses as
    // `*const c_void` and a `BufMut` as `*mut c_void`, so an address is what
    // goes either way — and the specialisation nulls the fp16 slot, which is
    // why the row is inert today and wrong on paper. It stays wrong until a
    // `Ty` states an element of its own.
    // `kernels_cuda::abi::emit_device_typecheck` builds its C++ parameter
    // out of `elem`'s head, and this row is the counterexample to the
    // comment there claiming the head is always the storage type.
    kernel!(rmsnorm_strided_vec8 "norm::rmsnorm_strided_bf16#vec8",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            x: Buf,
            weight: Buf,
            y: BufMut,
            y_fp16: BufMut | null,
            hidden: I32,
            x_row_stride: I32,
            y_row_stride: I32,
            eps: F32,
        ]),
    //===------------------------------------------------------------===//
    //
    // THE FIVE ROWS THAT WERE HELD, and what each one cites.
    //
    //===------------------------------------------------------------===//
    //
    // This file's header carried the refusal for these five in its own
    // words, and the reason was never the rule: `runtime::launch::
    // rows_per_head` has reproduced all five launchers digit for digit
    // since it landed. What was missing was a `Dims` that could say "the
    // statement named NO per-head width", because `driver-cuda`'s
    // `jit_dims` filled the only head-width field it had from the fire's
    // attention configuration when the statement named none — so the
    // rule's absent arm was unreachable and its present arm fired on a
    // number nobody stated. `Dims::stated_head_dim` is that field; it is
    // filled with `spec.per_head_dim.unwrap_or(0)` and no fallback, and
    // these are the rows it unblocks.
    //
    // # The law this section is written under
    //
    // A rule with no cited launcher is a guess, so every row below names
    // the `<<<>>>` it was written from by file and line. All five are in
    // `crates/kernels-cuda/csrc/src/norm/rmsnorm.cu` and all five are the
    // same three lines:
    //
    //     constexpr int BLOCK = 256;
    //     dim3 grid(num_rows);
    //     dim3 block(BLOCK);
    //
    // with `num_rows` an ARGUMENT. `table/norm.rs:36` says what the
    // caller passes and it is a conditional on `spec.per_head_dim`, which
    // is why one rule serves all five and why that rule needed a field.
    //
    // # Why `num_rows` is not in any of these operand lists
    //
    // It is the grid. A JIT row states its geometry as a `LaunchRule` and
    // its operands as what crosses in the `void**`; `num_rows` was in the
    // ahead-of-time lists because a HOST launcher takes its grid as an
    // argument, and the `__global__`s below read `blockIdx.x` instead.
    // The `stream` leaves for the same reason — `runtime::fire` takes it.
    //
    // # Where `hidden` comes from, and why it is not `OutWidth(0)`
    //
    // `Source::IfPresent(&PerHeadDim, &PerHeadDim, &Width(&In(0)))`, which
    // is the ahead-of-time row's own expression, unchanged. It is the
    // other half of the same conditional the rule computes: the grid gets
    // `rows · (width / head_dim)` blocks and each block norms `head_dim`
    // channels, or the grid gets `rows` blocks and each norms the row.
    // `OutWidth(0)` would be right on the absent arm and wrong on the
    // present one — every block would norm a whole row's width from a
    // per-head offset, which is exactly the defect the field removed,
    // reintroduced one layer down.

    // 1/5. `rmsnorm.cu:38-44` — which is a FORWARD, and the grid it
    // forwards to is `rmsnorm_strided_bf16`'s at `:85-98`:
    //
    //     void rmsnorm_bf16(const void* x, const void* weight, void* y,
    //         int num_rows, int hidden, float eps, cudaStream_t stream)
    //     {
    //         rmsnorm_strided_bf16(
    //             x, weight, y, num_rows, hidden, hidden, hidden, eps, stream);
    //     }
    //
    //     constexpr int BLOCK = 256;
    //     dim3 grid(num_rows);
    //     dim3 block(BLOCK);
    //     device::rmsnorm<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
    //         x, weight, y, hidden, x_row_stride, y_row_stride, eps);
    //
    // The forward is why both strides are `hidden` here and not the
    // operands' own widths: this symbol's contract is that a row of `x`
    // is exactly `hidden` wide. `RMSNORM_SIGS[0]` is the same kernel
    // under the other symbol, where the strides are stated and the
    // reading is single.
    //
    // The vec8 arm at `:87-96` is NOT taken by this row, and that is the
    // same decision `RMSNORM_SIGS[0]` records: the fast path is a
    // `Specialisation`, and one for the per-head reading would need its
    // own — `rmsnorm_vec8` reads `hidden` from a `hidden`-strided row, so
    // the predicate's three strides are the STATED head width here and
    // not the row's. Left unstated rather than guessed.
    //
    // `weight` is `Source::Or(&Weight(0), &WeightNamed)` and not
    // `Weight(0)`: two spellings of this kernel are live, `OpKind::
    // Rmsnorm` lowering to `[x, y]` with the weight by name and
    // `dsl::cuda::rmsnorm` stating it as an operand. The ahead-of-time
    // row at `table/norm.rs:102` records that a row demanding the slot
    // declines every fire built the first way.
    kernel!(rmsnorm "norm::rmsnorm_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::RowsPerHead,
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Or(&Source::Weight(0), &Source::WeightNamed),
            y: BufMut <- Source::Out(0),
            hidden: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            x_row_stride: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            y_row_stride: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // 2/5. `rmsnorm.cu:254-276`, grid at `:259`, the scalar launch at
    // `:271-276`:
    //
    //     constexpr int BLOCK = 256;
    //     dim3 grid(num_rows);
    //     dim3 block(BLOCK);
    //     device::rmsnorm_gemma<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
    //         x, weight, y, hidden, hidden, hidden, eps);
    //
    // Gemma folds `(1 + w)` instead of `w` — different arithmetic, same
    // signature, same row space, same grid. The launcher passes `hidden`
    // for both strides inline rather than through a forward, which is the
    // only structural difference from the row above.
    kernel!(rmsnorm_gemma "norm::rmsnorm_gemma_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::RowsPerHead,
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Or(&Source::Weight(0), &Source::WeightNamed),
            y: BufMut <- Source::Out(0),
            hidden: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            x_row_stride: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            y_row_stride: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // 3/5. `rmsnorm.cu:278-289`, grid at `:283-285`:
    //
    //     constexpr int BLOCK = 256;
    //     dim3 grid(num_rows);
    //     dim3 block(BLOCK);
    //     device::rmsnorm_no_scale<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
    //         x, y, hidden, eps);
    //
    // The weightless per-head norm — the V-norm — so no gamma, no
    // variant, no strides: the template's own parameters are `(x, y,
    // hidden, eps)` and it reads its row at `row * hidden`.
    //
    // `in_place = &[(0, 0)]` is the ahead-of-time row's, at
    // `table/norm.rs:352`: this norm lands on its own input, which is
    // what makes it the V-norm rather than a copy.
    kernel!(rmsnorm_no_scale "norm::rmsnorm_no_scale_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::RowsPerHead,
        in_place = &[(0, 0)],
        operands = operands![
            x: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            hidden: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // 4/5. `rmsnorm.cu:306-319`, grid at `:311-313`:
    //
    //     constexpr int BLOCK = 256;
    //     dim3 grid(num_rows);
    //     dim3 block(BLOCK);
    //     device::rmsnorm_gated<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
    //         x, gate, weight, y, hidden, eps);
    //
    // This is the launcher `runtime::launch::rows_per_head`'s doc quotes,
    // and it is quoted there because it is the shortest of the five.
    //
    // **`weight` is `F32s` and not `Buf`.** The `__global__` takes
    // `const float* __restrict__ weight` in every instantiation —
    // `rmsnorm.cuh:668` — because qwen3.5 ships RMSNormGated weights in
    // fp32 alongside bf16 activations, and the launcher's
    // `static_cast<const float*>(weight)` at `:317` is where the AOT ABI's
    // `const void*` stops being opaque. A row saying `Buf` would be
    // stating the LAUNCHER's parameter where the kernel's is meant; both
    // marshal as one address, so nothing would fail, and the row would be
    // wrong on paper for as long as anybody read it. `RMSNORM_SIGS[3]`
    // records the same distinction from the other side.
    kernel!(rmsnorm_gated "norm::rmsnorm_gated_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::RowsPerHead,
        operands = operands![
            x: Buf <- Source::In(0),
            gate: Buf <- Source::In(1),
            weight: F32s <- Source::Weight(0),
            y: BufMut <- Source::Out(0),
            hidden: I32 <- Source::IfPresent(&Source::PerHeadDim, &Source::PerHeadDim, &Source::Width(&Source::In(0))),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // 5/5. `rmsnorm.cu:291-304`, grid at `:296-298`:
    //
    //     constexpr int BLOCK = 256;
    //     dim3 grid(num_rows);
    //     dim3 block(BLOCK);
    //     device::rmsnorm_gated_f32_in<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
    //         x, gate, weight, y, hidden, eps);
    //
    // The gated norm that reads its input in fp32 — qwen3.5's GDN landing
    // norm, which saves one full pass over the intermediate buffer. Both
    // `x` and `weight` are `const float*` in the template
    // (`rmsnorm.cuh:713`), so both are `F32s`; `gate` and `y` are `T`.
    //
    // # The one row here whose BINDER is not finished, stated plainly
    //
    // Its rectangle is `rows · v_h` rows of `v_d` and both numbers come
    // off `GdnCtx` — the ahead-of-time row at `driver_internal.rs:213`
    // says `num_rows <- Mul(Rows, Gdn("v_h"))` and `hidden <- Gdn("v_d")`.
    // `RowsPerHead` computes `rows · (width / stated_head_dim)`, which IS
    // `rows · v_h` exactly when `stated_head_dim` carries `v_d`, and
    // `hidden` then resolves to `v_d` on the same arm. The rule is
    // therefore correct by construction for this row — `tests/
    // rows_per_head.rs` fires it at a stated `v_d` and gets the launcher's
    // bytes — and the binder is not there yet: `OpKind::RmsnormGated`
    // never sets `spec.per_head_dim`, so a GDN fire reaches `jit_dims`
    // with nothing to state and takes the ABSENT arm, which is `rows`
    // blocks of `width` where `rows · v_h` blocks of `v_d` were meant.
    //
    // That is a fix in `driver-cuda`'s binder — `spec.per_head_dim` set
    // from `gdn.v_d` where the statement is a gated norm — and it is
    // named here rather than worked around, in the same way this row's
    // rule was named before the field existed. **Nothing fires it
    // wrongly today**: `norm::rmsnorm_gated_fp32_in_bf16` is not in
    // `device::JIT_DISPATCHED`, so the generated dispatch still routes it
    // through the C shim, and the row is compiled, resolved and fireable
    // by a caller that states the rectangle it means.
    kernel!(rmsnorm_gated_fp32_in "norm::rmsnorm_gated_fp32_in_bf16",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::RowsPerHead,
        operands = operands![
            x: F32s <- Source::In(0),
            gate: Buf <- Source::In(1),
            weight: F32s <- Source::WeightNamed,
            y: BufMut <- Source::Out(0),
            hidden: I32 <- Source::Gdn("v_d"),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // The EMIT_FP16 arm. `rmsnorm.cu:54-79`, the vectorised branch at
    // `:68-79`:
    //
    //     constexpr int VBLOCK = 512;
    //     dim3 grid(num_rows);
    //     device::rmsnorm_vec8<VBLOCK, /*WEIGHT_PLUS_ONE=*/false, /*EMIT_FP16=*/true>
    //         <<<grid, VBLOCK, 0, stream>>>(
    //             x, weight, y, y_fp16, hidden, hidden, hidden, eps);
    //
    // The third arm of `rmsnorm_bf16_with_fp16`, and the row whose absence
    // `execution.rs` measured: the op's three predicates are all already
    // spellable — `Term::Present` on `y_fp16`, `Term::Multiple { of: 8 }` on
    // `hidden`, `Term::Aligned { bytes: 16 }` on the pointers — and the
    // composition was refused anyway because the instantiation those
    // predicates SELECT was not carried here. This is that instantiation.
    // The predicate vocabulary was sufficient; the row set was not.
    //
    // # DO NOT WIRE THIS INTO A `Choose` YET, and the reason is measured
    //
    // Firing this row at two shapes, as the bar requires, did not certify
    // it: it found that the KERNEL is wrong for `num_rows > 1`.
    // `rmsnorm.cuh:277-279` offsets both row pointers by
    // `row * {x,y}_row_stride`, and `rmsnorm.cuh:318` writes
    // `y_fp16[i * 8 + j]` with the WITHIN-ROW vector index and no offset at
    // all — so every block writes its fp16 copy into row 0's slice.
    // Measured at `rows = 3`, `hidden = 2048`: row 0 held 1 947 of 2 048
    // live fp16 values, rows 1 and 2 held **0 of 4 096**, and two
    // byte-identical hand launches disagreed on 249 fp16 bytes because three
    // blocks race for one row's slots. The bf16 output was byte-identical on
    // every run, which is exactly why this has never been noticed.
    //
    // **This is not a JIT defect and this row does not introduce it.** The
    // ahead-of-time launcher at `rmsnorm.cu:68-79` fires the same kernel
    // with the same arguments and gets the same wrong buffer on every
    // prefill whose rows are 16-byte aligned. The row reproduces its
    // launcher; the launcher is the thing that is wrong. It is recorded here
    // rather than fixed because a fix is a change to device text, and
    // `new-horizon.md` §10.10 fixes the order — extract, add rows, measure,
    // and only then change what was measured.
    //
    // `tests/launch_rules.rs`'s
    // `the_emit_fp16_kernel_is_wrong_above_one_row` asserts the defect's
    // exact signature, so the day `rmsnorm.cuh:318` grows its row offset
    // that test fails and says so. Until then this row is certified at ONE
    // row and at two widths, and a composition that fires it at a prefill
    // rectangle would ship a race.
    //
    // # `EMIT_FP16` is the only template argument that changes
    //
    // [`RMSNORM_SIGS`]`[4]` is `rmsnorm_vec8<256, false, false>` and this is
    // `rmsnorm_vec8<256, false, true>` — the SAME template, the same block,
    // the same `WEIGHT_PLUS_ONE`. Nothing else about the two rows differs
    // except the eighth operand's nullability, and that difference is the
    // whole content of the flag: `false` compiles the `y_fp16` write out
    // inside `if constexpr` and the arm nulls the slot, `true` compiles it in
    // and the slot must be a buffer.
    //
    // # `BLOCK` is 256 here and 512 in the launcher, for [`RMSNORM_SIGS`]`[4]`'s reason
    //
    // Restated rather than cross-referenced, because it is the one thing
    // about this row that reads as a mismatch. `BLOCK` sizes the
    // `__shared__ float[BLOCK]` that `block_reduce_sum_exact` folds through,
    // so an instantiation at 512 launched at `Rule::Rms`'s 256 folds 256
    // floats no thread wrote — finite, plausible and wrong. The row is the
    // launcher's DECISION at the width the row's rule states, which is the
    // trade `RMSNORM_STRIDED_VEC8` documents, measured, and `specialise.rs`
    // timed. A row may not buy a rule's fit with the kernel's arithmetic;
    // here it does not have to, because the reduction at 256 is the
    // reduction the sibling row already ships.
    //
    // # No `Source` on any operand, for [`RMSNORM_SIGS`]`[4]`'s reason too
    //
    // Nothing binds this row from a fire's rectangle. `model-compiler` never
    // writes `norm::rmsnorm_bf16_with_fp16#vec8` — `#vec8` is unspellable by
    // any trace — so the only two things that reach it are a
    // `execution::Step::Fire` that names it and a test that means to, and
    // both supply operands themselves. `abi.rs:737` skips a row with an
    // unbound operand, so this row generates no dispatch, which is the
    // correct answer and not an omission.
    //
    // # `y_fp16` is NOT `| null` here, and that is the contract
    //
    // The sibling row spells it `BufMut | null` because `EMIT_FP16=false`
    // reads it only inside a dead `if constexpr`. This arm writes through
    // it on every row, so a null is a null store and not a spare cell. The
    // non-nullability is what makes `Take::Null` in this slot a refusal
    // where it is an acceptance next door — the two rows are distinguished
    // by exactly the thing the flag means.
    kernel!(rmsnorm_with_fp16_vec8 "norm::rmsnorm_bf16_with_fp16#vec8",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            x: Buf,
            weight: Buf,
            y: BufMut,
            y_fp16: BufMut,
            hidden: I32,
            x_row_stride: I32,
            y_row_stride: I32,
            eps: F32,
        ]),
];

/// `norm::rmsnorm_strided_bf16`'s two instantiations, and the fire-time test
/// that picks between them.
///
/// # What this is, and why it could not be a row before
///
/// `rmsnorm.cu`'s `rmsnorm_strided_bf16` is not one kernel. It is an `if`:
///
/// ```text
/// if (rmsnorm_vec8_ok(x, y, weight, hidden, x_row_stride, y_row_stride)) {
///     device::rmsnorm_vec8<512, false><<<rows, 512>>>(x, w, y, nullptr, ...);
///     return;
/// }
/// device::rmsnorm<device::bf16, 256><<<rows, 256>>>(x, w, y, ...);
/// ```
///
/// and `rmsnorm_vec8_ok` is six clauses over three ADDRESSES and three
/// strides. This file's header used to say a row could not state that, and
/// gave the right reason: a row that froze either arm would be wrong on the
/// other, because whether the arm applies is a fact about the buffers a fire
/// was handed rather than about its shape. **The reason was right and the
/// conclusion was for an ahead-of-time build.** nvcc had to choose its
/// instantiations months before any pointer existed; NVRTC compiles both out
/// of one unit and the fire reads the addresses it is about to launch over.
///
/// # The predicate, clause for clause
///
/// `rmsnorm.cu`'s six, in its order, against this arm's six [`Term`]s:
///
/// | `rmsnorm_vec8_ok` | operand | term |
/// |---|---|---|
/// | `hidden % 8 == 0` | 3 `hidden` | `Multiple { of: 8 }` |
/// | `x_row_stride % 8 == 0` | 4 `x_row_stride` | `Multiple { of: 8 }` |
/// | `y_row_stride % 8 == 0` | 5 `y_row_stride` | `Multiple { of: 8 }` |
/// | `aligned(x)` | 0 `x` | `Aligned { bytes: 16 }` |
/// | `aligned(y)` | 2 `y` | `Aligned { bytes: 16 }` |
/// | `aligned(weight)` | 1 `weight` | `Aligned { bytes: 16 }` |
///
/// `aligned` is `(uintptr_t(p) & 15u) == 0`; a mask and `% 16` agree on every
/// value, and [`Specialisation::agrees`] refuses a non-power-of-two `bytes`
/// so the two spellings cannot part company on a case nobody swept.
///
/// **`tests/specialise.rs` swept 98 304 cases across those six boundaries —
/// eight byte offsets on each of three pointers (on 16, one byte off, one
/// bf16 element off, half a chunk, two short, on the next chunk, and past
/// it), twelve widths covering every residue of `hidden % 8`, and four
/// offsets on each of two strides — and the two agreed on all 98 304. 128 of
/// those cases took the vectorised arm, which is exactly the
/// 8 x 4 x 4 the six clauses predict, so the sweep is not passing by
/// refusing everything.** Deleting the `weight` clause — the realistic
/// mistake, because `weight` is the one pointer of the three that is not an
/// activation — put 96 of 6 144 cases on the wrong arm, and the sweep caught
/// it. It also pins the C++ predicate's text, so the day `rmsnorm_vec8_ok`
/// gains a seventh clause is a failing test naming this table rather than a
/// silent divergence.
///
/// # `BLOCK` is 256 here and 512 in the launcher, and that is a decision
///
/// The launcher's vectorised arm is `rmsnorm_vec8<512, false>` at
/// `<<<rows, 512>>>`. This row instantiates the same template at **256**,
/// because `LaunchRule::Rms` launches 256 threads and `BLOCK` is the size of
/// the `__shared__ float[BLOCK]` the reduction folds through: compiled at 512
/// and launched at 256, `block_reduce_sum_exact` folds through 256 floats no
/// thread wrote and the norm is finite and wrong. The alternative is a rule
/// that launches 512, which would be a SECOND decision stacked on the
/// alignment one and a change to `runtime::launch`, which this work does not
/// own.
///
/// So this arm is the launcher's DECISION reproduced exactly and the
/// launcher's WIDTH deliberately not: it is `rmsnorm_vec8` at the width the
/// row's rule already states. `rmsnorm.cu`'s own sweep put vec256 at 2.72 us
/// against vec512's 2.93 at hidden 2048 and 3.46 against 3.12 at 2816, so
/// the narrower block is not a concession at decode's shapes — and it was
/// measured here rather than inherited. `tests/specialise.rs` timed 300
/// launches of each arm through the same symbol on an L40S, release, the two
/// argument lists differing only in two bytes on the output pointer:
///
/// ```text
///   rows  hidden   scalar us  vector us   ratio
///      1    2048        2.52       2.23    1.13x
///      1    4096        2.97       2.64    1.13x
///      1    8192        4.03       3.14    1.28x
///      8    4096        3.09       2.76    1.12x
///     64    4096        3.38       2.96    1.14x
///    512    4096        6.40       4.54    1.41x
///   1024    2048        6.45       4.75    1.36x
/// ```
///
/// The arm wins at every shape measured — 1.12x at decode, 1.41x at prefill,
/// 0.29 to 1.86 us saved per fire — and the choice that picks it cost 21 ns
/// over 100 000 evaluations, about one per cent of the cheapest launch in
/// the table. Those are minima of five batches of 300 and not means of one,
/// because at decode's shapes the two arms are a few hundred nanoseconds
/// apart and a single batch put the ratio either side of 1.0 on consecutive
/// runs. Repeat runs reproduced every ratio in the table to within 0.02.
///
/// In a DEBUG build the same choice cost 304 ns and the arm lost below 512
/// rows, which is recorded rather than dropped: at those shapes the harness
/// is timing `fire`'s own host work and not the kernel, and a specialisation
/// that adds a `choose`, a reshape and a second `Args::bind` to the launch
/// path is not free in a build that does not inline them. The win above is a
/// release-build claim and should be read as one.
///
/// # Both arms compute the same bf16, and that was measured rather than
/// assumed
///
/// The two kernels reduce in different orders. Scalar thread `t` sums
/// `x[t], x[t+256], ...`; vectorised thread `t` sums four `float2` pairs out
/// of each 16-byte chunk it owns. That is a reassociation, and the header of
/// `rmsnorm.cuh` is careful to call it one — so bit-identity was the bar and
/// it was a real question, not a formality.
///
/// **It held. `tests/specialise.rs` fired both arms on identical bytes at
/// five shapes — hidden 2048/2816/4096 at one row, 5376 at three, 2048 at
/// seven — and 0 of 39 424 bf16 values differed, worst case 0 ulp.** The
/// reassociation is real in fp32 and dies in the round to bf16: eight bits of
/// mantissa, reached through one `rsqrtf` of a sum whose last fp32 bits
/// moved.
///
/// So the tolerance is zero, and it is zero because it was measured to be,
/// not because a reassociated fp32 sum was expected to agree. A shape at
/// which it stops holding is a finding about this table and should fail that
/// test rather than widen it.
///
/// # What a wrong choice actually looks like, measured
///
/// The negative control fires the vectorised arm at `hidden = 4095`, where
/// `rmsnorm_vec8_ok` says it must not go — the exact damage a `Select` that
/// had dropped the `hidden % 8` clause would do. **7 of 4 095 values moved,
/// and 0 of the 4 088 the kernel actually wrote.** `rmsnorm_vec8` computes
/// `nvec = hidden / 8`, sums 4 088 of the 4 095 squares, still divides by
/// 4 095, and the resulting norm is wrong by under a tenth of a per cent —
/// which bf16's eight mantissa bits cannot see. The only trace is a
/// seven-element tail the kernel never touched.
///
/// That is the failure mode this design exists to prevent, and it is worse
/// than it sounds: a wrong choice here is not a crash and not a NaN, it is
/// 99.83 per cent of the right answer, feeding sixty more layers. **No
/// relative-error tolerance loose enough to admit a reassociated reduction
/// would flag it** — which is why the parity bar is zero differing values and
/// why the agreement between these terms and `rmsnorm_vec8_ok` is swept
/// rather than argued.
/// This family's specialised rows, which is how [`crate::device::SPECIALISED`]
/// finds them.
///
/// The family owns the list so that specialising a second norm row is an edit
/// here and nowhere else — `device.rs` names this slice once and never again.
/// Without it every new arm needed a line in a runtime file its author may not
/// own, which is how a written-but-unregistered `Specialisation` becomes dead
/// code that reads like a live decision.
pub static SPECIALISATIONS: &[&Specialisation] = &[&RMSNORM_STRIDED_VEC8];

pub static RMSNORM_STRIDED_VEC8: Specialisation = Specialisation {
    base: "norm::rmsnorm_strided_bf16",
    arms: &[Arm {
        name: "vec8",
        // The order is `rmsnorm_vec8_ok`'s own, so the two read as one list.
        // Order is not semantic -- every term is evaluated on every fire,
        // deliberately, so that a term naming a bad operand faults instead of
        // hiding behind an earlier `false`.
        when: &[
            Term::Multiple { operand: 3, of: 8 },
            Term::Multiple { operand: 4, of: 8 },
            Term::Multiple { operand: 5, of: 8 },
            Term::Aligned { operand: 0, bytes: 16 },
            Term::Aligned { operand: 2, bytes: 16 },
            Term::Aligned { operand: 1, bytes: 16 },
        ],
        row: &RMSNORM_ROWS[4],
        // `rmsnorm`'s seven arguments into `rmsnorm_vec8`'s eight. The only
        // difference between the two parameter lists is `y_fp16` third, which
        // `rmsnorm.cu` passes as a literal `nullptr` in exactly this slot.
        //
        // `Take::Null` here is LOAD-BEARING and not a convenience. `y_fp16` is
        // the one parameter of this kernel whose element type differs from
        // every other buffer's -- `f16*`, where `x`, `weight` and `y` are
        // `bf16*` -- and `Ty::BufMut` is documented as an OPAQUE `void*` that
        // deliberately does not describe what a buffer contains, so the row
        // cannot distinguish the two and `Args::bind` will not either. What
        // stops that from mattering is this slot and the `false` in `elem`:
        // the value is always null and `EMIT_FP16` compiles the store away.
        // `tests/specialise.rs` pins BOTH, because a later edit that sourced
        // this operand from `y` would type-check in Rust, bind in the driver,
        // and write half-width data into a bf16 buffer at legal addresses.
        take: &[
            Take::From(0),
            Take::From(1),
            Take::From(2),
            Take::Null,
            Take::From(3),
            Take::From(4),
            Take::From(5),
            Take::From(6),
        ],
        because: "csrc/src/norm/rmsnorm.cu, `rmsnorm_vec8_ok` and the `if` in \
                  `rmsnorm_strided_bf16` that consults it",
    }],
};
