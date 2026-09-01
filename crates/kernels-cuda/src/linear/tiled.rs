//! `tiled`: the post-affine W4A16 projection as a TILED tensor-core GEMM —
//! §J4's hybrid, phases A (correctness) and B (tuning).
//!
//! `linear::quant` already serves this stored form twice. [`quant::matmul`]
//! carves one block column per activation row and re-reads the whole weight
//! inside each of them: parity with cuBLAS at one token, 98–189× slower over
//! a prefill. [`quant::matmul_via_dense`] answers that by decoding the
//! weight once into an `n·k` bf16 scratch slab and firing the dense point —
//! 0.209 ms at 512×2048×10240, which is 1.8× cuBLAS and pays a transient
//! twice the size of the resident plane.
//!
//! This module is the third reading, and the one the other two are a
//! staircase towards: the weight is read ONCE, as nibbles, straight into
//! `mma.sync` B fragments, and the affine fold happens in registers on the
//! way. No scratch, no second launch, one pass over the codes.
//!
//! # The arm IS wired now (§J4b), and this is what wires it
//!
//! **THE SPEED GATE PASSED FIRST.** `tests/tiled_matmul.rs`'s sweep, on an
//! L40S, against the arm this replaces (ms, [`tuple_for`]'s pick):
//!
//! ```text
//!                    n >= k (2048 -> 10240)      k >  n (10240 -> 2048)
//!   rows      tiled    via_dense    cublas     tiled  via_dense    cublas
//!    128      0.045        0.123     0.034     0.087      0.159     0.070
//!    512      0.118        0.208     0.117     0.136      0.210     0.109
//!   2048      0.484        0.650     0.593     0.481      0.637     0.485
//! ```
//!
//! **AND THEN THE DECODE GATE**, which is what phase B did not have. The
//! tile point is arranged for arithmetic and a decode step is arranged for
//! bytes in flight, so [`matmul_gemv`] is a second reading of the same
//! layout — and it had to be at least as fast as the fused GEMV it replaces
//! or the repack could not be taken at all. Measured on the same box against
//! `quant::matmul`, at [`carve_for`]'s pick (ms):
//!
//! ```text
//!                    n >= k (2048 -> 10240)      k >  n (10240 -> 2048)
//!   rows      gemv    fused GEMV               gemv   fused GEMV
//!      1     0.0089      0.0224              0.0087      0.0250
//!      4     0.0160      0.0751              0.0157      0.0794
//!     16     0.0522      0.2962              0.0468      0.2866
//! ```
//!
//! 1.4x at one row in the tall direction and 6.1x at sixteen in the wide
//! one: the fused point re-reads the whole weight per activation row and
//! this one reads it once, so the gap opens with the batch and never closes.
//!
//! **WHAT MOVED SO THAT DISPATCH COULD ASK.** [`matmul`] reads [`repack`]'s
//! output, which is the same bytes in a different order — and firing it on a
//! landed plane is NOT refusable, because the rectangles are identical. It
//! answers nonsense. So the flip needed three things and has them:
//!
//! 1. **A DECODE READER**, which is [`matmul_gemv`]. Without one, a repacked
//!    plane could only be served above sixteen rows and a decode step would
//!    have nowhere to go.
//! 2. **THE REPACK AT IMPORT.** `checkpoint`'s `Expr::Repack` carries the
//!    two `RepackLayout::TiledAffine*` rows [`repack`] implements, and only
//!    `CONVERT_TILE_MAP_MASK` admits one: `pie model import` runs the
//!    permutation on the host, once per weight, and the artifact holds the
//!    relaid plane. A serving load that meets one is refused with the layout
//!    and the command named.
//! 3. **A WITNESS ON THE ROW.** `dtype::Dtype::U4g64tiled` is the model
//!    text's word for "these three rectangles are in fragment order";
//!    `engine_cuda`'s `WeightRow::Planes` carries it off the plane's own
//!    declaration, and `Run::maybe_tiled_planes` is a resolution DISJOINT
//!    from the row-major one — so a row-major reader handed a repacked plane
//!    gets `None` and a named panic rather than a wrong number.
//!
//! `engine_cuda::dispatch::linear` then reads exactly like the row-major
//! ladder beside it: [`matmul`] above `PREFILL_ROWS`, [`matmul_gemv`] below,
//! and a plane that was not repacked takes the roads it always took.
//!
//! **WHAT THE ARM GATES ON** is this module's own ladder and nothing added:
//! four-bit codes with a bf16 (scale, bias) pair and the post-offset fold;
//! `k % 64 == 0`; a group that is a whole number of 16-wide mma k tiles; a
//! resident (non-streaming) seat; and, for [`matmul_gemv`], sixteen rows or
//! fewer. Each is a typed refusal here rather than a second copy of the
//! ladder at the call site.
//!
//! # The stored form does not change
//!
//! [`repack`] is a relabelling — the
//! same codes and the same factors, in the order a lane's mma fragment
//! wants them — run once at load time, on the ruling `matmul_via_dense`
//! already stands on: what is SERVED is the row the checkpoint stated, and
//! the order a kernel reads it in is implementation.
//!
//! A module's own items need their paths spelled out here: this file's
//! `//!` prose merges with the `///` line above `pub mod tiled` in
//! `linear.rs`, and the merged block resolves its links in THAT scope.
//!
//! [`quant::matmul`]: crate::linear::quant::matmul
//! [`quant::matmul_via_dense`]: crate::linear::quant::matmul_via_dense
//! [`matmul`]: crate::linear::tiled::matmul
//! [`repack`]: crate::linear::tiled::repack
//! [`repacked_rows`]: crate::linear::tiled::repacked_rows
//! [`tuple_for`]: crate::linear::tiled::tuple_for

use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/tiled.cuh";

/// The relabelling passes' launch width. Nothing about them is shaped; they
/// are one thread per output word.
const BLOCK: u32 = 256;

/// The contraction step, and the one config axis that is NOT a tuple axis —
/// four 16-wide mma k tiles is exactly the `uint4` superword a lane pulls in
/// one instruction. `linear/tiled.cuh`'s `kTiledK` is this number and the
/// repacked layout is grouped by it.
const TILE_K: u32 = 64;

/// The staged activation tile's row stride in bf16 — `TILE_K` plus the
/// eight elements of padding that spread one `ldmatrix`'s eight addresses
/// across eight shared-memory segments.
const LD_A: u32 = TILE_K + 8;

/// The code band a repacked plane is padded up to, which is the mma tile's
/// n extent and the column span one warp owns. A weight whose `n` is not a
/// whole band gets zero codes and zero factors in the tail, and those
/// columns decode to a zero weight.
const BAND: u32 = 16;

/// **A CONFIG TUPLE, MIRRORED FROM `linear/tiled.cuh`'s TEMPLATE
/// PARAMETERS.** A change here that is not the same change there is a launch
/// whose grid or shared memory does not match the tile the kernel walks —
/// nothing refuses it and the answer is wrong, so the two lists are kept
/// adjacent and short.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tuple {
    /// The rows one block lands.
    pub m: u32,
    /// The columns one block lands. It is `warps * 16` by construction:
    /// one warp owns one 16-column band, because one repacked word is a
    /// whole B fragment for two columns eight apart.
    pub n: u32,
    /// The block's threads.
    pub threads: u32,
    /// The cp.async depth over the activation tile.
    pub stages: u32,
}

impl Tuple {
    /// The dynamic shared memory this tuple asks for: the deepest of the
    /// staged activation tile and the epilogue tile it is reused as. The
    /// weight never lands in shared memory at either.
    const fn smem(self) -> u32 {
        let staging = self.stages * self.m * LD_A;
        let epilogue = self.m * (self.n + 8);
        if staging > epilogue { staging * 2 } else { epilogue * 2 }
    }
}

/// **THE LONG-PREFILL TUPLE** — a 64-row by 128-column tile over eight
/// warps, two cp.async stages deep. Eight warps is what phase B's sweep
/// found and phase A's four did not: at 512 tokens by 2048 by 10240 it is
/// 0.118 ms against the four-warp 64x64's 0.140, and it is the faster tuple
/// on BOTH projection directions from 512 rows up.
///
/// **TWO STAGES AND NOT FOUR.** Four was on phase B's list and four LOSES,
/// which is worth stating because it is the one item that did: a stage is
/// 9KB, so four of them is 36KB and an L40S's 100KB of shared memory then
/// holds two blocks an SM instead of five. The prefetch depth bought less
/// than the occupancy cost paid — 0.160 ms against 0.140 at the phase-A
/// tile, and the same sign at every other tuple measured. Four stages
/// survives here only as the [`SHORT`] tuple's depth, where the row tile is
/// half as tall and the arithmetic behind each fetch is half as deep.
pub const LONG: Tuple = Tuple {
    m: 64,
    n: 128,
    threads: 256,
    stages: 2,
};

/// **THE SHORT-PREFILL TUPLE** — the same 128-column tile over the same
/// eight warps, with the row tile halved and the pipeline twice as deep.
///
/// The row tile is the whole difference, and what it buys is BLOCKS. A
/// down projection is 2048 columns, which is sixteen column tiles; at 128
/// rows [`LONG`] carves two row tiles, so its grid is 32 blocks over an
/// L40S's 142 SMs and four fifths of the machine is idle. Halving the row
/// tile doubles the grid, and at that shape it is 0.096 ms against
/// [`LONG`]'s 0.147 — a 1.5x that no amount of prefetch depth reaches,
/// because the problem is not latency but empty SMs. Four stages is then
/// affordable (a 32-row stage is 4.5KB, so four is the 18KB two of
/// [`LONG`]'s are) and slightly ahead.
pub const SHORT: Tuple = Tuple {
    m: 32,
    n: 128,
    threads: 256,
    stages: 4,
};

/// **THE ROW COUNT AT WHICH THE TALLER TILE STARTS WINNING**, measured on
/// both projection directions at 16, 64, 128, 256, 384, 512, 768, 1024 and
/// 2048 rows.
///
/// Below it the grid is the binding constraint and [`SHORT`] doubles it;
/// above it there are blocks to spare and [`LONG`]'s taller tile reads each
/// weight word for twice as many activations. The two directions cross in
/// the same place, which is why this is one number and not two: at 384 rows
/// they are level (0.100 against 0.111 one way, 0.148 against 0.149 the
/// other) and at 512 [`LONG`] is ahead on both (0.124 against 0.141, and
/// 0.151 against 0.163).
pub const LONG_PREFILL_ROWS: u32 = 512;

/// **THE PICK, AND IT IS THE WHOLE SELECTION.** No search, no cache, no
/// autotune: a row count against [`LONG_PREFILL_ROWS`].
///
/// **AND IT IS NOT AN ASPECT TEST**, which is what phase B set out to write.
/// The `k >> n` direction really is the hard one — at 128 rows it runs at
/// half the wide direction's efficiency — but the axis that fixes it turned
/// out to be the ROW tile and not the contraction step, and the row tile's
/// crossover sits at the same place in both directions. A tuple keyed on
/// `n` against `k` would have named the right second config for the wrong
/// reason and then mispicked it at 2048 rows, where the down projection
/// wants the tall tile exactly as the up projection does.
#[must_use]
pub const fn tuple_for(rows: u32) -> Tuple {
    if rows >= LONG_PREFILL_ROWS { LONG } else { SHORT }
}

/// The planes this point was handed, measured and checked.
#[derive(Clone, Copy)]
struct Tiled {
    /// The codes under one factor — off the factor row against `k`, the same
    /// reading `linear::quant`'s ladder takes.
    group: u32,
    /// `n` rounded up to a whole 16-column band: the row count the repacked
    /// planes carry.
    n_pad: u32,
}

/// `n` rounded up to a whole band.
const fn padded(n: u32) -> u32 {
    n.div_ceil(BAND) * BAND
}

/// The plane ladder, read once for both entries.
///
/// It is deliberately narrower than [`crate::linear::quant`]'s: that one
/// serves four offset arms, two code widths and two factor dtypes, and this
/// point is stamped for exactly one row of that matrix — four-bit codes, a
/// bf16 factor pair, the post-offset fold. Everything else is refused here
/// rather than silently folded as something it is not.
fn tiled(
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    n: u32,
    k: u32,
) -> Result<Tiled, Error> {
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(biases.dtype, Dtype::U8, "a packed plane binds as bytes");
    // **THE CONTRACTION IS A WHOLE NUMBER OF STEPS.** The mainloop stages
    // `TILE_K` activations at a time with no tail stage, and the repacked
    // plane groups its words by the same sixty-four, so a `k` that does not
    // divide is refused rather than half-walked. Every projection width a
    // serving checkpoint brings is a multiple of sixty-four.
    if !k.is_multiple_of(TILE_K) {
        return Err(refuse(
            op,
            format!("a {k}-wide row is not a whole number of {TILE_K}-wide contraction steps"),
        ));
    }
    if codes.width * 2 != k {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at something other than four bits",
                codes.width
            ),
        ));
    }
    if scales.width == 0 || scales.width % 2 != 0 {
        return Err(refuse(
            op,
            format!(
                "a {}-byte factor row is not a whole number of two-byte factors",
                scales.width
            ),
        ));
    }
    if biases.width != scales.width {
        return Err(refuse(
            op,
            format!(
                "the post-offset arm reads a {}-byte bias row beside a {}-byte scale row",
                biases.width, scales.width
            ),
        ));
    }
    let groups = scales.width / 2;
    if !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    // A 16-wide k tile is one lane's whole B fragment, so it must sit inside
    // ONE group or the lane would fold two of its four contraction
    // positions against the wrong factor.
    if !group.is_multiple_of(BAND) {
        return Err(refuse(
            op,
            format!("a {group}-code group is not a whole number of {BAND}-wide mma k tiles"),
        ));
    }
    Ok(Tiled {
        group,
        n_pad: padded(n),
    })
}

/// **THE TILED POST-AFFINE PROJECTION** — `y = act · W^T` over the repacked
/// planes, with `W[n][k] = s·code + b` folded in registers at the mma. The
/// config tuple is [`tuple_for`]'s.
///
/// The planes are [`repack`]'s output and not the checkpoint's rectangles:
/// same bytes, same count, a different order. Firing this on an un-repacked
/// plane is not refusable — the rectangles are identical — and answers
/// nonsense, which is why the two entries live in one module and the
/// module doc says so twice.
///
/// **A STREAMED SEAT IS REFUSED**, on `matmul_via_dense`'s ground: the
/// repacked plane is a load-time artefact of a plane that stays put, and a
/// tier that moves its bytes between fires has not repacked anything.
pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    let tuple = tuple_for(y.rows);
    tiled_launch(ctx, "linear.matmul", act, codes, scales, biases, y, seat, tuple)
}

/// [`matmul`] under the head's own op name, `linear::gemm`'s pairing kept —
/// [`quant::lm_head`]'s relation to [`quant::matmul`], on this point.
///
/// # Errors
///
/// [`matmul`]'s.
///
/// [`quant::lm_head`]: crate::linear::quant::lm_head
/// [`quant::matmul`]: crate::linear::quant::matmul
pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    let tuple = tuple_for(y.rows);
    tiled_launch(ctx, "linear.lm_head", act, codes, scales, biases, y, seat, tuple)
}

/// [`matmul`] with the config tuple named rather than derived.
///
/// This exists for the golden's timing sweep, which has to hold BOTH tuples
/// against BOTH projection directions to justify [`tuple_for`]'s one line —
/// a pick nobody measured the other side of is not a pick. It is not a
/// tuning hook: there is no search here and no place to put the answer if
/// there were.
///
/// # Errors
///
/// The plane ladder [`matmul`] runs, plus a tuple whose row tile does not
/// divide the block's thread count.
#[allow(clippy::too_many_arguments)]
pub fn matmul_with(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    tuple: Tuple,
) -> Result<(), Error> {
    tiled_launch(ctx, "linear.matmul", act, codes, scales, biases, y, seat, tuple)
}

/// The one launch behind [`matmul`], [`lm_head`] and [`matmul_with`]. The op
/// name is an argument for [`crate::linear::quant`]'s reason: the two entries
/// differ in the word a refusal carries and in nothing else, and a second
/// copy of the ladder would be a second chance to disagree about what one
/// stored form means.
#[allow(clippy::too_many_arguments)]
fn tiled_launch(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    tuple: Tuple,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "the tiled arm serves resident dense projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    let Tiled { group, n_pad } = tiled(op, codes, scales, biases, n, k)?;
    // The repacked planes carry the padded band count, not `n`: a warp whose
    // band starts inside `n` reads all sixteen of its columns.
    if codes.rows != n_pad || scales.rows != n_pad || biases.rows != n_pad {
        return Err(refuse(
            op,
            format!(
                "a {n}-column projection repacks into {n_pad} rows, and the planes hold \
                 ({}, {}, {})",
                codes.rows, scales.rows, biases.rows
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let Tuple {
        m,
        n: tile_n,
        threads,
        stages,
    } = tuple;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_affine_tiled<{t}, ::pie::i32(4), ::pie::i32({group}), \
                 ::pie::i32({m}), ::pie::i32({tile_n}), ::pie::i32({threads}), \
                 ::pie::i32({stages})>"
            )),
        )
        .apply(
            Launch::grid([y.rows.div_ceil(m), n.div_ceil(tile_n), 1], [threads, 1, 1])
                .smem(tuple.smem()),
        ),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// **HOW A DECODE BLOCK CARVES THE WEIGHT**, and it is a tuple for the same
/// reason [`Tuple`] is: the two projection directions want opposite answers
/// and neither is derivable from the other.
///
/// A warp owns one 16-column band and a slice of the contraction. `bands`
/// warps cover adjacent bands, `split` warps share one band and reduce
/// through shared memory at the end, and `bands * split` is the block. The
/// axis that matters is BLOCKS: a decode step is bound by the weight's own
/// bytes, so what the launch has to produce is enough concurrent loads to
/// cover the memory latency, and a projection with few columns has to reach
/// that by splitting `k` because it cannot reach it by carving `n`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Carve {
    /// The 16-column bands one block covers, one to a warp.
    pub bands: u32,
    /// The contraction slices one band is split across.
    pub split: u32,
}

impl Carve {
    /// The block, which is one warp per (band, slice).
    #[must_use]
    pub const fn threads(self) -> u32 {
        32 * self.bands * self.split
    }

    /// The cross-slice reduction buffer: one band's sixteen columns by
    /// `rows` rows for every warp. Nothing at all when there is one slice,
    /// because then a warp owns its band's whole contraction and the whole
    /// stage compiles out.
    const fn smem(self, rows: u32) -> u32 {
        if self.split > 1 {
            self.bands * self.split * rows * BAND * 4
        } else {
            0
        }
    }
}

/// **THE WARPS A DECODE LAUNCH AIMS TO PUT ON THE MACHINE**, and it is the
/// whole of [`carve_for`]'s arithmetic.
///
/// A decode step is bound by the weight's bytes, so what a launch has to
/// produce is enough concurrent loads to cover the memory latency — which is
/// warps, and an L40S holds 6816 of them. Measured, at eight decode shapes
/// (two directions by five row counts by six carves): the fastest carve is
/// the one that lands between one and two waves of warps, and it is the same
/// number in both directions. Below it the machine is idle; above it a warp
/// walks one contraction step and the launch is all prologue.
pub const TARGET_WARPS: u32 = 8 * 1024;

/// The shallowest split. A band's contraction is split at least eight ways
/// whatever the column count says, because the warps of one band are also
/// this point's only prefetch: a warp holds ONE superword in flight, so the
/// bytes outstanding are the warp count times sixteen.
pub const MIN_SPLIT: u32 = 8;

/// The deepest split a thin batch can afford, and the deepest there is: one
/// block is `32 * split` threads, so this is 1024 of them.
pub const THIN_SPLIT: u32 = 32;

/// The deepest split a FULL mma tile of rows can afford.
///
/// **THE ONE CLIFF IN THE SWEEP.** The accumulators are two floats a row a
/// lane, so a 16-row launch holds 32 of them, and at 1024 threads a block
/// that is where the occupancy falls off: 0.186 ms against [`THIN_SPLIT`]'s
/// own 0.033 at eight rows on the same shape. Sixteen rows is the row count
/// this point is capped at, so the cap is stated as the split it forces
/// rather than left to be rediscovered.
pub const WIDE_SPLIT: u32 = 16;

/// The row count above which [`THIN_SPLIT`] stops paying — measured level at
/// eight (0.0242 against 0.0254 on the tall direction, 0.0326 against 0.0237
/// on the wide one) and a cliff at sixteen.
pub const THIN_ROWS: u32 = 8;

/// **THE PICK, AND IT IS THE WHOLE SELECTION.** No search, no cache, no
/// autotune: a band count and a row count against the two constants above.
///
/// **ONE BAND TO A BLOCK, IN BOTH DIRECTIONS.** Phase B's tile point picks
/// its column tile because eight warps sharing a 128-column tile share the
/// activation they multiply; nothing is shared here — a decode step's
/// activation is one row that every warp reads from L1 anyway — so a block
/// wider than a band buys nothing and costs blocks. The sweep says so
/// plainly: at eight bands a block the tall direction runs 0.088 ms where at
/// one it runs 0.012.
///
/// **AND THE CONTRACTION SPLIT IS WHAT REPLACES THE COLUMN TILE.** A down
/// projection lands 2048 columns, which is 128 bands and 128 blocks; the
/// only other axis a decode step has is `k`, and splitting it is what fills
/// the machine. The reduction it costs is one barrier and sixteen floats a
/// warp.
#[must_use]
pub const fn carve_for(n: u32, rows: u32) -> Carve {
    let bands = n.div_ceil(BAND);
    let deepest = if rows <= THIN_ROWS { THIN_SPLIT } else { WIDE_SPLIT };
    let want = if bands == 0 {
        deepest
    } else {
        (TARGET_WARPS / bands).next_power_of_two()
    };
    let split = if want < MIN_SPLIT {
        MIN_SPLIT
    } else if want > deepest {
        deepest
    } else {
        want
    };
    Carve {
        bands: 1,
        split,
    }
}

/// The row count a launch is INSTANTIATED at: the next power of two, so a
/// decode step's varying batch reaches at most five entries of the jit's
/// name-expression cache instead of one per row count. The kernel takes the
/// live count as an argument and masks against it, so a bucket costs the
/// activation half of a few lanes' arithmetic and nothing of the weight's —
/// which is the half a decode step is bound by.
const fn bucket(rows: u32) -> u32 {
    if rows <= 1 {
        1
    } else if rows <= 2 {
        2
    } else if rows <= 4 {
        4
    } else if rows <= 8 {
        8
    } else {
        16
    }
}

/// **THE TILED LAYOUT AT A DECODE STEP** — the same repacked planes
/// [`matmul`] reads, gathered through the same fragment map, at the row
/// counts where a tensor-core tile has nothing to multiply.
///
/// This is the point that makes the repack safe to take at import: with it,
/// BOTH sides of `dispatch::linear`'s `PREFILL_ROWS` gate can read a
/// repacked plane, so a weight can be relaid once and served at every shape.
/// Without it a repacked plane has no decode reader and the flip cannot
/// happen — see the module doc.
///
/// The ladder is [`matmul`]'s, plus a row cap: above one mma tile of rows
/// the tiled GEMM is the point, and this entry refuses rather than running a
/// shape it is slower at.
///
/// # Errors
///
/// [`matmul`]'s plane ladder, a streamed seat, and a row count above 16.
pub fn matmul_gemv(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    gemv_with(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        biases,
        y,
        seat,
        carve_for(y.width, y.rows),
    )
}

/// [`matmul_gemv`] under the head's own op name, [`matmul`]'s pairing kept.
///
/// # Errors
///
/// [`matmul_gemv`]'s.
pub fn lm_head_gemv(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    gemv_with(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        biases,
        y,
        seat,
        carve_for(y.width, y.rows),
    )
}

/// [`matmul_gemv`] with the carve named rather than derived — the golden's
/// timing sweep holds both carves against both projection directions, for
/// the reason [`matmul_with`] exists: a pick nobody measured the other side
/// of is not a pick.
///
/// # Errors
///
/// [`matmul_gemv`]'s.
#[allow(clippy::too_many_arguments)]
pub fn gemv_with(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    y: &mut Tensor,
    seat: GroupSeat,
    carve: Carve,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "the tiled arm serves resident dense projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    let Tiled { group, n_pad } = tiled(op, codes, scales, biases, n, k)?;
    if codes.rows != n_pad || scales.rows != n_pad || biases.rows != n_pad {
        return Err(refuse(
            op,
            format!(
                "a {n}-column projection repacks into {n_pad} rows, and the planes hold \
                 ({}, {}, {})",
                codes.rows, scales.rows, biases.rows
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    // **THE ROW CAP IS THE POINT'S OWN SHAPE AND NOT A DISPATCH RULE.** The
    // accumulators are registers, one pair per row, and above an mma tile of
    // rows the tiled GEMM reads each weight word for more activations than
    // this point ever will. A caller past this row count wanted [`matmul`].
    if y.rows > BAND {
        return Err(refuse(
            op,
            format!(
                "the decode point holds {} rows in registers, and a {}-row fire is the \
                 tiled GEMM's shape",
                BAND, y.rows
            ),
        ));
    }
    let rows = bucket(y.rows);
    let Carve { bands, split } = carve;
    let threads = carve.threads();
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::gemv_affine_tiled<{t}, ::pie::i32(4), ::pie::i32({group}), \
                 ::pie::i32({rows}), ::pie::i32({bands}), ::pie::i32({split})>"
            )),
        )
        .apply(
            Launch::grid([(n_pad / BAND).div_ceil(bands), 1, 1], [threads, 1, 1])
                .smem(carve.smem(rows)),
        ),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// **THE LOAD-TIME RELABELLING** — the dense `[n, k]` four-bit plane and its
/// two factor planes, put into the order [`matmul`] reads.
///
/// Shifts, masks and ors on the code side; a pure permutation on the factor
/// side. No arithmetic touches a value, which is what makes this legal under
/// serve-as-stored rather than a quantization decision: the served row is
/// the stored row, and `tests/tiled_matmul.rs` holds a host un-repack
/// against the original plane bit for bit.
///
/// The destinations are the same rectangles as the sources, except that
/// their rows are `n` rounded up to a whole 16-column band — the tail is
/// written as zero codes and zero factors, which decodes to a zero weight.
/// [`repacked_rows`] is the row count a caller allocates for.
///
/// **THE CODE LAYOUT IS PHASE B'S**, which is phase A's B-fragment
/// ownership with four k tiles grouped as one lane's `uint4`. Nothing about
/// the plane's SIZE changed; a reader written against phase A's word order
/// gets the right bytes in the wrong places, which is what the roundtrip
/// gate is for.
pub fn repack(
    ctx: &Ctx,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    out_codes: &mut Tensor,
    out_scales: &mut Tensor,
    out_biases: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.repack";

    let n = nonzero(OP, "N, the rows of the weight", codes.rows)?;
    let k = nonzero(OP, "K, the width of the weight", codes.width)? * 2;
    let Tiled { group, n_pad } = tiled(OP, codes, scales, biases, n, k)?;
    if scales.rows != n || biases.rows != n {
        return Err(refuse(
            OP,
            format!(
                "an {n}-row code plane sits beside a {}-row scale plane and a {}-row bias \
                 plane",
                scales.rows, biases.rows
            ),
        ));
    }
    if out_codes.rows != n_pad
        || out_codes.width != codes.width
        || out_scales.rows != n_pad
        || out_scales.width != scales.width
        || out_biases.rows != n_pad
        || out_biases.width != biases.width
    {
        return Err(refuse(
            OP,
            format!(
                "the repack of a [{n}, {k}] weight is a [{n_pad}, {}] code plane and two \
                 [{n_pad}, {}] factor planes",
                codes.width, scales.width
            ),
        ));
    }
    let groups = k / group;
    let words = n_pad / BAND * (k / BAND) * 32;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            "::pie::linear::repack_affine_tiled<::pie::i32(4)>",
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            out_codes.arg(),
            stated(OP, n)?.arg(),
            stated(OP, k)?.arg(),
        ],
    )?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::linear::repack_factors_tiled")
            .apply(Launch::flat(n_pad * groups, BLOCK)),
        &[
            scales.arg(),
            biases.arg(),
            out_scales.arg(),
            out_biases.arg(),
            stated(OP, n)?.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

/// The row count [`repack`]'s destinations carry for an `n`-row weight —
/// `n` rounded up to a whole 16-column mma band.
#[must_use]
pub const fn repacked_rows(n: u32) -> u32 {
    padded(n)
}
