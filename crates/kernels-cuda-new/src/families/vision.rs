//! `vision`'s JIT units — the three towers' device text, and the twenty-eight
//! rows of the thirty-nine kernels a ported rule states.
//!
//! # Why this family is different, and why it was last
//!
//! Every other family here migrated a `.cu` whose launchers are rows of
//! `KERNELS` — symbols `model-compiler` can state, reached from a dispatcher
//! arm generated out of a trace. The vision towers are not that. Their entry
//! points are `driver_internal`, the second table, whose invariant is the
//! opposite one: *`model-compiler` cannot state them, so nothing in a trace
//! ever names them*, and the per-family exhaustiveness tests classify them as
//! `DriverInternal` for exactly that reason. `examples/migration_status` never
//! counted them and no migration agent touched them.
//!
//! It did not make them un-migratable, and the reading that said so was
//! confusing an entry point with a kernel. A JIT row needs a symbol, a
//! template and a launch rule; it does not need a DSL statement. What being
//! `driver_internal` changes is the CONSUMER — these fire from `driver-cuda`'s
//! own code rather than from a generated arm — and a consumer is not a
//! property of a row.
//!
//! # What `driver_internal` actually holds, which is the finding
//!
//! Three rows, not thirty-two. `vision::qwen3vl_scatter`,
//! `vision::gemma4_vision_encode` and `vision::gemma4_audio_encode`, every one
//! of them `whole = true` — a WHOLE TOWER behind one symbol, taking host
//! pointers and a stream and running a hundred launches inside. There is no
//! per-kernel symbol in that table for any of the thirty-nine `__global__`s
//! these files hold, so Rule 1's *use the symbol verbatim* had nothing to bind
//! to and the symbols below are new. They are spelled `vision::<kernel>_bf16`
//! against a `vision::device::<kernel>` template, leaf name unchanged from the
//! `k_*` the towers wrote, so that the split is auditable as the move it is.
//!
//! Those three whole-tower rows are UNCHANGED by this migration and stay on
//! the ahead-of-time path. A `whole = true` row is a C++ function with a
//! mutex-guarded interpolation cache, a `std::map`, a cuBLAS handle and a
//! flashinfer plan behind it; none of that is device text and none of it can
//! be carried into an NVRTC header set. What moved is the device text those
//! three functions launch.
//!
//! What has since ALSO moved is the three functions themselves, though not
//! into a row. They were compiled by `kernels-cuda` and are now compiled by
//! `driver-cuda`, from `crates/driver-cuda/csrc/vision/`; the rows did not
//! change and neither did the shim entries. The reason is the sentence above
//! read the other way round: every `.cuh` those three functions include is
//! in THIS crate's `csrc/src` — the five in `vision/` that the table below
//! rows, plus `norm/rmsnorm.cuh`, `norm/elementwise.cuh`, `mlp/swiglu.cuh`
//! and `ssm/causal_conv1d.cuh` — so the archive was never holding a tower's
//! device code. It was holding a host walk over device code that had already
//! arrived here. `new-horizon.md` §42 has the measurement, and the argument
//! for why `Execution::Composed` is the near miss that does not fit: it
//! carries a `&'static [Step]`, and a tower's trip count is data.
//!
//! For anyone routing `norm::rmsnorm_bf16`: the Gemma-4 vision tower used to
//! call it six times and no longer calls it at all, having taken the two
//! `<<<>>>` arms verbatim. That launcher's C++ consumer set is now
//! `norm/rmsnorm.cu` alone.
//!
//! # The accounting, measured
//!
//! | file | `__global__` | `<<<` lines | of another file's kernels |
//! |---|---|---|---|
//! | `gemma4_audio.cu` | 12 | 34 | 17 |
//! | `qwen3_vl_tower.cu` | 11 | 16 | 7 |
//! | `gemma4_vision.cu` | 9 | 10 | 4 |
//!
//! The launch counts exceed the kernel counts because twenty-eight of the
//! sixty launches fire kernels the two SHARED headers hold —
//! `tower_naive_kernels.cuh`'s six and `gemma4_naive_kernels.cuh`'s one, which
//! had already been extracted in an earlier pass. That is header-shared device
//! text and not the C++-calling-C++ shape: no host function calls another
//! host function there, the `#include` is the whole mechanism, and both
//! headers are inside this family. Converting them is what unlocks seven of
//! the twenty-eight rows.
//!
//! The genuine C++-calling-C++ calls in these three files are
//! `gemm::act_x_wt_bf16`, `norm::rmsnorm_bf16`, `norm::rmsnorm_no_scale_bf16`,
//! `norm::residual_add_bf16`, `mlp::geglu_tanh_bf16`,
//! `ssm::causal_conv1d_prefill_noact_bf16`, and `qwen3vl_vis_gemm_bf16` /
//! `qwen3vl_vis_attn` in the adapter `.cpp`. None of them is touched here.
//!
//! # Twenty-eight rows of thirty-nine kernels, and why the other eleven are
//! not
//!
//! Fifteen landed first, on the vocabulary as it then stood: nine
//! `LaunchRule::Elementwise` and four `LaunchRule::PerRow` covered every flat
//! and per-row shape the towers reach, and twenty-four kernels were left with
//! their `<<<>>>` recorded and no rule to state it. Three of the four causes
//! have since been answered by rules asked for BY THESE LAUNCHERS and cited
//! against them — [`kernels::LaunchRule::Tile16`],
//! [`kernels::LaunchRule::AxialRope`] and
//! [`kernels::LaunchRule::PerRowNarrow`], `new-horizon.md` §21.13 — and
//! thirteen more rows land here on them:
//!
//! * **`Tile16` — eleven.** All three towers index a rectangle with
//!   `dim3 B2(16,16)` over `G2(X,Y) = dim3((X+15)/16, (Y+15)/16)`, declared on
//!   one line in each of the three `.cu`s (`gemma4_vision.cu:117`,
//!   `gemma4_audio.cu:131`, `qwen3_vl_tower.cu:139`). The kernels read
//!   `threadIdx.y`, so this was never a 1-D block wearing a 2-D spelling, and
//!   `tile16`'s block is the first in [`crate::runtime::launch`] that is not
//!   `[n,1,1]`. `k_matmul`, `k_matmul_bias`, `k_addpos_grid2d`, `k_qk`,
//!   `k_av`, `k_pool`, `k_glu`, `k_sscp_flatten`, `k_qkv_scale`,
//!   `k_rel_pos_enc` and `k_merge_gather`.
//! * **`AxialRope` — one.** `k_rope_axial2d` launches `dim3 rg(1,NH,N)` at 32
//!   threads (`gemma4_vision.cu:150`), and the rule that reproduces it is the
//!   first in the crate with a `grid.z` above one.
//! * **`PerRowNarrow` — one.** `k_layernorm_relu` launches `<<<rows, 128>>>`
//!   (`gemma4_audio.cu:189` and `:196`) where [`kernels::LaunchRule::PerRow`]
//!   fixes 256. It is a separate rule and not a widened `PerRow` because the
//!   fold sums `(blockDim.x + 31) / 32` warp partials SERIALLY in thread zero:
//!   four addends at 128 and eight at 256, which is a different last bit and
//!   not a tuning knob.
//!
//! **A rule existing is not a row landing**, so each of the thirteen was
//! checked by evaluating its rule at the rectangle its launcher runs on and
//! comparing with the launcher's own arithmetic — `G2`/`B2`, `rg`, `128` —
//! transcribed from the `.cu`. Three of the thirteen do not take the
//! statement's output rectangle and say so in their own comment: `k_av`'s
//! width is ONE HEAD's 64, `k_rel_pos_enc`'s rows are POSITIONS, and
//! `k_glu`'s width is its output's where its input is twice that.
//!
//! The eleven that remain refused fall into three kinds, and every one is
//! recorded on the kernel in its `.cuh` with the `<<<>>>` it was checked
//! against:
//!
//! * **Three independent extents — three kernels.** `k_conv2d_s2`, `k_chlast`
//!   and `k_chfirst` launch `dim3((F+15)/16, (T+15)/16, C)` at
//!   `gemma4_audio.cu:186-197` — `Tile16`'s grid with a convolution's channel
//!   count on `grid.z`. The axis is no longer the blocker; the VALUE is.
//!   [`crate::runtime::launch::Dims`] carries two extents plus head counts,
//!   these rectangles are `[C, T, F]`, and the tiled pair are transposes of
//!   each other — so a rule taking `rows` and `width` off a statement would
//!   have to be told which of three axes each one is. Spelling `C` as a head
//!   count is the failure `Dims::kv_heads` is filled from an attention head
//!   count to guarantee.
//! * **A block width a rule fixes — one kernel.** `k_split_rope_qkv` launches
//!   `<<<dim3(NH,N), HEAD/2>>>` (`qwen3_vl_tower.cu:249`) where
//!   [`kernels::LaunchRule::PerHead`]'s grid matches to the digit and its
//!   `PAD_BLOCK = 128` does not. `PerRowNarrow` is the precedent for what a
//!   fix looks like — a rule per block width, one function each — and the
//!   reason this one has not been asked for is that widening the launch
//!   QUADRUPLES it at qwen3-vl's 32-wide half-head, where `PerRowNarrow`'s
//!   sibling would only have halved a fold.
//! * **A tile count on `grid.x` — one kernel.** `k_local_attn` launches
//!   `dim3((N+127)/128, NH)` at 128 (`gemma4_audio.cu:243`). Every ported
//!   rule's leading axis is a count of THINGS; this one counts tiles, and the
//!   128 is an occupancy decision about a 1 KiB per-thread local array rather
//!   than a fold order a rule could name.
//! * **Dead — six kernels.** `k_gelu_mul`, `k_add_inplace`, `k_split_qkv`,
//!   `k_split_qkv_bias`, `k_rope_vis`, `k_rope_qk`. Nothing launches them, so
//!   there is no `<<<>>>` to check a rule against and Rule 3's *a row with no
//!   cited launcher is a guess* applies with nothing left over. Five of the
//!   six are in `qwen3_vl_tower.cuh` and were superseded in place by fused
//!   successors; the sixth lost its call site to `mlp::geglu_tanh_bf16`. They
//!   are not deleted — removing device text is its own change — and
//!   `examples/unit_probe_vision` instantiates every one so NVRTC compiles
//!   them and none can rot unnoticed.
//!
//! One of the dead is worth its own line. `k_add_inplace` is byte-for-byte
//! `k_add` in `tower_naive_kernels.cuh`, byte-for-byte `k_add_pe` below it,
//! and byte-for-byte `norm::device::residual_add` in another family. Four
//! copies of `a[i] += b[i]`, and `tests/sources.rs::no_global_is_defined_twice`
//! sees none of them, because it compares namespace-qualified NAMES and all
//! four are spelled differently. That is the limit of a name-based duplicate
//! check, and it is recorded because the next person to widen that test should
//! know what it will find.
//!
//! # One refusal that was two reasons, and only one of them was a reason
//!
//! `gemma4_vision.cuh` refuses `k_qk` twice over: for the 2-D block, and
//! because *"`head` is a launch-time scalar the host varies across twelve
//! fires, and no rule's `Dims` carries a head INDEX."* The second half is not
//! a blocker and never was. A head index is an OPERAND — the host loop passes
//! `hh` as the kernel's sixth argument, exactly as it passes `k_pool`'s `k2`
//! and `k_axpy`'s `scale` — and no rule has to recover it, because a rule
//! produces a GRID. Twelve fires of one row at twelve operand values is the
//! same thing `k_silu`'s one row covering two rectangles already is. The
//! block was the whole refusal, and `Tile16` answers it.
//!
//! # bf16 only, and what a second format would now cost
//!
//! Every row states `device::bf16`. The towers are bf16 end to end — bf16
//! storage, fp32 compute, checked against HF-bf16 dumps — so an fp16 row would
//! name a format no caller holds. The kernels are templated over `T` with
//! `device::Elem<T>` all the same, per the migration's fourth rule, so the
//! second format is a row rather than a build: the ahead-of-time archive had
//! to choose its instantiations before a pointer existed and NVRTC does not.
//! That is the measurement `norm/elementwise` made and it holds here unchanged.

use kernels::{KernelSig, LaunchRule, kernel, operands};

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The six naive kernels more than one tower launches.
///
/// A unit of its own rather than folded into the towers because it IS shared:
/// three translation units include it, the split gave it one definition, and a
/// second unit carrying a second copy of the same root would compile the same
/// kernels to a second cubin under a second cache key.
pub const TOWER_NAIVE_KERNELS: Unit = Unit {
    name: "vision/tower_naive_kernels",
    root: include_str!("../../csrc/src/vision/tower_naive_kernels.cuh"),
    rows: TOWER_NAIVE_ROWS,
    options: &[],
};

/// The one kernel the two gemma-4 towers share.
pub const GEMMA4_NAIVE_KERNELS: Unit = Unit {
    name: "vision/gemma4_naive_kernels",
    root: include_str!("../../csrc/src/vision/gemma4_naive_kernels.cuh"),
    rows: GEMMA4_NAIVE_ROWS,
    options: &[],
};

/// The Gemma-4 vision encoder's nine.
pub const GEMMA4_VISION: Unit = Unit {
    name: "vision/gemma4_vision",
    root: include_str!("../../csrc/src/vision/gemma4_vision.cuh"),
    rows: GEMMA4_VISION_ROWS,
    options: &[],
};

/// The Gemma-4 audio encoder's twelve.
pub const GEMMA4_AUDIO: Unit = Unit {
    name: "vision/gemma4_audio",
    root: include_str!("../../csrc/src/vision/gemma4_audio.cuh"),
    rows: GEMMA4_AUDIO_ROWS,
    options: &[],
};

/// The Qwen3-VL vision encoder's eleven.
pub const QWEN3_VL_TOWER: Unit = Unit {
    name: "vision/qwen3_vl_tower",
    root: include_str!("../../csrc/src/vision/qwen3_vl_tower.cuh"),
    rows: QWEN3_VL_ROWS,
    options: &[],
};

/// The units `vision` compiles.
///
/// The two shared headers come first because the tower headers `#include`
/// them, and a reader following the `#include`s reads them in this order. The
/// ORDER carries nothing else — `unit::UNITS` concatenates and a unit's slot
/// is its position in a per-process cache, which nothing keys on.
pub static UNITS: &[Unit] = &[
    TOWER_NAIVE_KERNELS,
    GEMMA4_NAIVE_KERNELS,
    GEMMA4_VISION,
    GEMMA4_AUDIO,
    QWEN3_VL_TOWER,
];

/// [`TOWER_NAIVE_KERNELS`]' instantiations — all six.
///
/// `k_matmul` was the sixth and had no row: `<<<G2(O,N), B2, 0, S>>>` is a
/// 16x16 block, which no rule stated. [`kernels::LaunchRule::Tile16`] does,
/// and it is cited against this launcher among the eleven.
///
/// The row does not make the kernel a good idea. This whole header calls
/// itself a way station and `gemm::act_x_wt_bf16` is what this should be — a
/// row preserves a scalar loop the tree is trying to delete. What it does not
/// do is preserve it SILENTLY: the swap changes cuBLAS's accumulation for a
/// serial one and needs the tower parity harnesses, and a rowed kernel is
/// visible to `migration_status` where an unrowed one is not.
static TOWER_NAIVE_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[0],
        template_path: "vision::device::k_rms",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[1],
        template_path: "vision::device::k_add",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[2],
        template_path: "vision::device::k_f32_to_bf16",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[3],
        template_path: "vision::device::k_gelu_erf",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[4],
        template_path: "vision::device::k_layernorm",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &TOWER_NAIVE_SIGS[5],
        template_path: "vision::device::k_matmul",
        elem: "device::bf16",
    },
];

/// The contracts, in [`TOWER_NAIVE_ROWS`]' order.
///
/// Each is the `__global__`'s parameter list and nothing else. There is no
/// ahead-of-time twin to subtract a stream from — these kernels never had a
/// `pie_k_*` entry point, because they were anonymous-namespace text inside
/// three translation units and the only thing that could reach them was a
/// `<<<>>>` in the same file. Every operand below is a parameter; every
/// operand a rule recovers is absent, and for these five that is the flat
/// element count `k_add`, `k_f32_to_bf16` and `k_gelu_erf` DO still take,
/// because `Rule::Elementwise` recovers the GRID from it and the kernel
/// recovers its bound from the argument. Both readings are the same number
/// and neither is the other's source — a launch cannot pass its grid to a
/// kernel, and a kernel cannot see one.
///
/// `Source::Unbound` throughout. These fire from `driver-cuda`'s own tower
/// code, not from a lowered statement, so there is no `Out(0)` or
/// `WeightNamed` for a binder to read: the caller holds every pointer already.
/// Nothing about how an operand is SOURCED changes what compiles, which is
/// what lets the rows land ahead of a binder that could fill them.
#[rustfmt::skip]
static TOWER_NAIVE_SIGS: [KernelSig; 6] = [
    // `PerRow` -- `k_rms<<<R, 256, 0, S>>>` at all ten call sites (eight in
    // the audio tower's conformer loop, two in its embedder), and `per_row`
    // evaluates `grid[rows,1,1] block[256,1,1] smem 0`. `R` is the row count
    // and `D` the width; the width reaches the kernel as an OPERAND because
    // the stride loop walks it, and `per_row` deliberately leaves
    // `Dims::width` unread.
    //
    // Not `Rule::Rms`, which is the same grid and block with `(256/32)*4`
    // bytes of DYNAMIC shared memory. This kernel's scratch is
    // `__shared__ float warp[32], ss;` -- static, allocated by the compile --
    // so those 32 bytes would be an allocation nothing reads. `per_row`'s own
    // doc names this as the case it exists for.
    //
    // `w` is nullable: the embedder norms with gamma=1 and passes `nullptr`.
    kernel!(k_rms "vision::k_rms_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            x: Buf,
            w: Buf | null,
            o: BufMut,
            rows: I32,
            width: I32,
            eps: F32,
        ]),
    // `Elementwise` -- `k_add<<<((long)N*Hd+255)/256, 256, 0, S>>>` at both
    // audio call sites, and `elementwise` evaluates `rows * width` to the same
    // `ceil(n/256)` blocks of 256.
    //
    // In place over its first operand, and the row says so: `a` is read and
    // written, `b` is read. The AOT tree spells this same arithmetic three
    // more times under three more names -- see the module header.
    kernel!(k_add "vision::k_add_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            a: BufMut,
            b: Buf,
            n: Usize,
        ]),
    // `Elementwise` -- `k_f32_to_bf16<<<(n+255)/256, 256, 0, S>>>` at all
    // three call sites (gemma-4 vision's scatter and encode, gemma-4 audio's
    // feature upload, qwen3-vl's two pixel uploads).
    //
    // The INPUT is `F32s` and not `Buf`: `Buf` is spelled in the row's
    // element type, and this kernel's input is float whatever `T` is. That is
    // the narrowing, and a row that made both operands `device::bf16` would
    // read the source at half its stride.
    kernel!(k_f32_to_bf16 "vision::k_f32_to_bf16_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            a: F32s,
            o: BufMut,
            n: Usize,
        ]),
    // `Elementwise` -- `k_gelu_erf<<<((long)N*Dmid+255)/256, 256, 0, S>>>` in
    // qwen3-vl's `mlp()` when `erf_gelu` is set, which is both patch mergers.
    //
    // The tanh form is `qwen3_vl_tower.cuh`'s `k_gelu_tanh` and has its own
    // row. Two kernels and two rows because they are two FUNCTIONS: HF's
    // `ACT2FN["gelu"]` against `gelu_pytorch_tanh`, and the tree already
    // recorded that merging them by name changed numerics silently.
    kernel!(k_gelu_erf "vision::k_gelu_erf_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            n: Usize,
        ]),
    // `PerRow` -- `k_layernorm<<<R, 256, 0, S>>>` at all four qwen3-vl call
    // sites (two per merger, two per block), and `per_row` evaluates
    // `grid[rows,1,1] block[256,1,1] smem 0`. Static shared memory again, so
    // zero dynamic bytes is the whole contract.
    //
    // Both gamma and beta are nullable. The general form is here because
    // mimi's copy dereferenced them and qwen3-vl's guards them, and mimi
    // always passes non-null -- so one kernel is bit-identical for both.
    kernel!(k_layernorm "vision::k_layernorm_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            x: Buf,
            g: Buf | null,
            beta: Buf | null,
            o: BufMut,
            rows: I32,
            width: I32,
            eps: F32,
        ]),
    // `Tile16` -- `k_matmul<<<G2(Out,N), B2, 0, S>>>` at all four audio call
    // sites, and `tile16` evaluates
    // `grid[ceil(width/16), ceil(rows/16), 1] block[16,16,1] smem 0` at every
    // one of them:
    //
    // | `<<<>>>` | line | rectangle | grid |
    // |---|---|---|---|
    // | `G2(Out,N)` | `gemma4_audio.cu:165` | `[47, 4096]` | `[256, 3, 1]` |
    // | `G2(Hd,N)` | `gemma4_audio.cu:203` | `[47, 1024]` | `[64, 3, 1]` |
    // | `G2(Hd,P)` | `gemma4_audio.cu:242` | `[13, 1024]` | `[64, 1, 1]` |
    // | `G2(TXT,N)` | `gemma4_audio.cu:289` | `[47, 2560]` | `[160, 3, 1]` |
    //
    // at 188 mel frames, and `B2`/`G2` are declared together on
    // `gemma4_audio.cu:131`. The `:242` fire is the one to read twice: its
    // rows are the POSITION TABLE's `P = context_left`, not the token count,
    // and a row that assumed the rectangle was the tower's activation would
    // launch 3 tile-rows over a 13-row matrix.
    //
    // **`x` is the ROW COUNT and `o` the width, and the grid takes them the
    // other way round.** `G2(X, Y)` is `(width, rows)` -- the kernel reads
    // `blockIdx.x` as a column of `O` and `blockIdx.y` as a row of `N` -- so
    // the operand order and the grid order are transposes and neither may be
    // read off the other.
    //
    // This is the kernel the header calls itself a way station for. See
    // [`TOWER_NAIVE_ROWS`] for why a row is still the right thing.
    kernel!(k_matmul "vision::k_matmul_bf16",
        file = Some("vision/tower_naive_kernels.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            x: Buf,
            w: Buf,
            y: BufMut,
            n: I32,
            k: I32,
            o: I32,
        ]),
];

/// [`GEMMA4_NAIVE_KERNELS`]' one instantiation.
static GEMMA4_NAIVE_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &GEMMA4_NAIVE_SIGS[0],
    template_path: "vision::device::k_clamp",
    elem: "device::bf16",
}];

/// The contract, and the one thing about it worth stating twice.
#[rustfmt::skip]
static GEMMA4_NAIVE_SIGS: [KernelSig; 1] = [
    // `Elementwise` -- `k_clamp<<<((long)N*Kin+255)/256, 256, 0, S>>>` and
    // `<<<((long)N*Out+255)/256, ...>>>`, twice per clipped linear in each
    // gemma-4 tower, and `elementwise` evaluates `rows * width` to the same
    // `ceil(t/256)` blocks of 256.
    //
    // `lo` and `hi` are DEVICE pointers to single elements, not floats, and
    // both are nullable. They are per-layer weights sitting in the checkpoint
    // next to the matrix they clip; reading them on the host would be a
    // synchronising copy per linear per layer. `Buf | null` is the whole of
    // what a row can say about that, and it is enough -- the kernel's
    // `lo ? F(*lo) : neg_inf()` is what a null means.
    kernel!(k_clamp "vision::k_clamp_bf16",
        file = Some("vision/gemma4_naive_kernels.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            lo: Buf | null,
            hi: Buf | null,
            t: Usize,
        ]),
];

/// [`GEMMA4_VISION`]'s instantiations — eight of its nine.
///
/// The ninth is `k_gelu_mul`, which is dead. The five that landed after the
/// first three are four `Tile16` tiles and the crate's one `AxialRope`; see
/// the module header for the taxonomy and `gemma4_vision.cuh` for the `<<<>>>`
/// each was checked against.
static GEMMA4_VISION_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[0],
        template_path: "vision::device::k_scale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[1],
        template_path: "vision::device::k_softmax",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[2],
        template_path: "vision::device::k_pool_finish",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[3],
        template_path: "vision::device::k_addpos_grid2d",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[4],
        template_path: "vision::device::k_rope_axial2d",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[5],
        template_path: "vision::device::k_qk",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[6],
        template_path: "vision::device::k_av",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_VISION_SIGS[7],
        template_path: "vision::device::k_pool",
        elem: "device::bf16",
    },
];

/// The contracts, in [`GEMMA4_VISION_ROWS`]' order.
#[rustfmt::skip]
static GEMMA4_VISION_SIGS: [KernelSig; 8] = [
    // `Elementwise` -- `k_scale<<<((long)N*Hd+255)/256, 256, 0, S>>>` with
    // `t = (long)N*Hd`, and `elementwise` evaluates `rows * width` from a
    // rectangle of `rows = N`, `width = Hd` to the same `ceil(N*Hd/256)`
    // blocks of 256.
    kernel!(k_scale "vision::k_scale_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            p: Buf,
            o: BufMut,
            t: Usize,
        ]),
    // `PerRow` -- `k_softmax<<<N, 256, 0, S>>>`, once per head per layer, and
    // `per_row` evaluates `grid[rows,1,1] block[256,1,1] smem 0` with
    // `rows = N`. Static shared memory, so zero dynamic bytes is the contract.
    //
    // The one row here whose element type appears in NO operand. The score
    // matrix is fp32 and stays fp32 -- this kernel neither reads nor writes
    // the storage format -- so `elem` names the instantiation and nothing
    // else. It is a template for the reason every kernel in these headers is
    // one: a non-template `__global__` in a header included by three
    // translation units emits three strong definitions and does not link.
    kernel!(k_softmax "vision::k_softmax_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::PerRow,
        in_place = &[(0, 0)],
        operands = operands![
            s: F32sMut,
            n: I32,
        ]),
    // `Elementwise` -- `k_pool_finish<<<((long)OUTL*Hd+255)/256, 256, 0, S>>>`
    // with `t = (long)OUTL*Hd`, and `elementwise` evaluates `rows * width`
    // from `rows = OUTL`, `width = Hd` to the same `ceil(OUTL*Hd/256)` blocks
    // of 256.
    //
    // `s` is `sqrtf((float)Hd)`, computed on the host. It is an OPERAND and
    // not an extent the rule recovers: nothing about the rectangle determines
    // it, and a row that left it out would bind one argument short.
    kernel!(k_pool_finish "vision::k_pool_finish_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            input: F32s,
            o: BufMut,
            s: F32,
            t: Usize,
        ]),
    // `Tile16` -- `k_addpos_grid2d<<<G2(Hd,N), B2, 0, S>>>` at
    // `gemma4_vision.cu:144`, once per forward, with `B2`/`G2` declared on
    // `:117`. At the tower's only legal shape (`Hd = 768`, checked by the
    // launcher) and 4 096 patches the launcher builds
    // `dim3((768+15)/16, (4096+15)/16) = [48, 256, 1]` over `dim3(16,16)`, and
    // `tile16` at `rows = 4096, width = 768` evaluates the same.
    //
    // In place over its first operand: the kernel reads `y[n*O+o]` and writes
    // it back, and the launcher passes `h` for both.
    //
    // `pos` is `F32s` and not `I32s` although every value it holds is a grid
    // INDEX. It is the same buffer `k_rope_axial2d` consumes, where the values
    // are trigonometric arguments; the `llrintf` and the two clamps in the
    // kernel are what an index costs for sharing one allocation. A row that
    // narrowed it to `I32s` would describe a second buffer that does not
    // exist.
    kernel!(k_addpos_grid2d "vision::k_addpos_grid2d_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut,
            tb: Buf,
            pos: F32s,
            n: I32,
            o: I32,
            p: I32,
        ]),
    // `AxialRope` -- `dim3 rg(1,NH,N); k_rope_axial2d<<<rg, 32, 0, S>>>` at
    // `gemma4_vision.cu:150`, fired TWICE on that line, once over `q` and once
    // over `k` with the same `NH`. `axial_rope` at `rows = 4096,
    // kv_heads = 12` evaluates `grid[1,12,4096] block[32,1,1] smem 0`, which
    // is `rg` and the literal 32 digit for digit.
    //
    // **The crate's first `grid.z` above one, and its `grid.x` is a literal
    // one.** The kernel reads `blockIdx.z` as its token, `blockIdx.y` as its
    // head and `threadIdx.x` as a channel of the 16 pairs a 64-wide head
    // holds -- one warp covers them, so the axis a channel tiling would have
    // used is spent and the two counts move UP rather than across. Packing
    // them into `[NH, N, 1]` out of habit is a launch this kernel indexes as
    // one token.
    //
    // `kv_heads` and not `q_heads`, per [`kernels::LaunchRule::AxialRope`]'s
    // own doc: the head count is the ADDRESSED tensor's, and both fires pass
    // the same `NH`.
    //
    // The launcher throws unless `Hd == 768 && NH == 12`, because the kernel
    // hard-codes a 64-wide head and a 16-wide half. `axial_rope` checks
    // `head_dim` and does not read it -- a warp is a warp -- so a `Dims` with
    // a 128-wide head would produce this grid and launch half a rotation. That
    // precondition lives on the tower, and it is recorded here because nothing
    // in the row states it.
    //
    // One operand list covers both fires. `q` is `BufMut` and rotated in
    // place; there is no `k` parameter, because the second fire is the same
    // kernel over the other tensor rather than a second half of one launch.
    kernel!(k_rope_axial2d "vision::k_rope_axial2d_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::AxialRope,
        in_place = &[(0, 0)],
        operands = operands![
            q: BufMut,
            pos: F32s,
            n: I32,
            h: I32,
            theta: F32,
        ]),
    // `Tile16` -- `k_qk<<<G2(N,N), B2, 0, S>>>` at `gemma4_vision.cu:151`,
    // inside a host loop over the twelve heads. The rectangle is SQUARE and it
    // is the score matrix's, not the activation's: `rows = width = N`, so at
    // 4 096 patches `tile16` evaluates `[256, 256, 1]` over `dim3(16,16)` and
    // so does `G2(N,N)`.
    //
    // **`head` is an operand and this row is why the second half of that
    // refusal was wrong.** The host varies `hh` across twelve fires; a rule
    // produces a GRID, and twelve fires of one row at twelve operand values is
    // what `k_silu`'s single row covering two rectangles already is. See the
    // module header.
    //
    // `s` is `F32sMut` and the element type appears in no operand of it: the
    // scores are fp32 and stay fp32, which is `k_softmax`'s situation one
    // kernel later. `scale` is the host's `1.0f` -- an operand, because
    // nothing about the rectangle determines it.
    kernel!(k_qk "vision::k_qk_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            q: Buf,
            k: Buf,
            s: F32sMut,
            n: I32,
            h: I32,
            head: I32,
            scale: F32,
        ]),
    // `Tile16` -- `k_av<<<G2(64,N), B2, 0, S>>>` at `gemma4_vision.cu:151`,
    // the same host loop over heads as `k_qk`.
    //
    // **The width is ONE HEAD's 64 and not the tower's 768.** The kernel
    // writes `o[(n*H + head)*64 + d]` -- a slice of the `[N, H, 64]`
    // activation -- so the rectangle a caller hands this row is
    // `[rows = N, width = 64]` and the head axis is walked by the host, not by
    // the grid. At 4 096 patches that is `[4, 256, 1]` blocks; the statement's
    // own output rectangle `[4096, 768]` would give `[48, 256, 1]`, twelve
    // times the work, eleven twelfths of it addressing another head's cells.
    // [`kernels::LaunchRule::Tile16`]'s doc names this and `k_rel_pos_enc` as
    // the two of the eleven that do not walk the statement's output.
    kernel!(k_av "vision::k_av_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            s: F32s,
            v: Buf,
            o: BufMut,
            n: I32,
            h: I32,
            head: I32,
        ]),
    // `Tile16` -- `k_pool<<<G2(Hd,N), B2, 0, S>>>` at
    // `gemma4_vision.cu:165`, once per forward. `rows = N`, `width = Hd`, so
    // at 4 096 patches and 768 hidden the launcher and `tile16` both build
    // `[48, 256, 1]` over `dim3(16,16)`.
    //
    // The INPUT rectangle, not the output's. The grid covers the patches being
    // scattered; the destination has `OUTL` rows and `atomicAdd` puts several
    // patches on one of them, which is why `k_pool_finish` is a second kernel
    // and why the accumulator is `F32sMut` rather than the storage format.
    // `k2` is the host's `9.f` -- the pooling group size squared, an operand.
    //
    // NOT `Rule::Elementwise` over `rows * width`. The flat product is right
    // and the addressing is not: this kernel reads `threadIdx.y` for its
    // patch, so a 256-wide 1-D block gives every thread patch zero.
    kernel!(k_pool "vision::k_pool_bf16",
        file = Some("vision/gemma4_vision.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            h: Buf,
            grp: I32s,
            o: F32sMut,
            n: I32,
            d: I32,
            k2: F32,
        ]),
];

/// [`GEMMA4_AUDIO`]'s instantiations — eight of its twelve.
///
/// Two landed on the first vocabulary and six on the second: five `Tile16`
/// tiles and the crate's one `PerRowNarrow`. The four that remain refused are
/// the three-extent conv trio (`k_conv2d_s2`, `k_chlast`, `k_chfirst`, whose
/// `[C, T, F]` rectangles `Dims` cannot carry and whose tiled pair are
/// transposes of each other) and `k_local_attn`, whose leading grid axis
/// counts TILES and whose 128-wide block is an occupancy decision about a
/// 1 KiB per-thread local array. See `gemma4_audio.cuh`.
static GEMMA4_AUDIO_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[0],
        template_path: "vision::device::k_silu",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[1],
        template_path: "vision::device::k_axpy",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[2],
        template_path: "vision::device::k_matmul_bias",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[3],
        template_path: "vision::device::k_glu",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[4],
        template_path: "vision::device::k_layernorm_relu",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[5],
        template_path: "vision::device::k_sscp_flatten",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[6],
        template_path: "vision::device::k_qkv_scale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GEMMA4_AUDIO_SIGS[7],
        template_path: "vision::device::k_rel_pos_enc",
        elem: "device::bf16",
    },
];

/// The contracts, in [`GEMMA4_AUDIO_ROWS`]' order.
#[rustfmt::skip]
static GEMMA4_AUDIO_SIGS: [KernelSig; 8] = [
    // `Elementwise` -- fired twice with two different rectangles,
    // `k_silu<<<((long)N*IM+255)/256, 256, 0, S>>>` in the feed-forward and
    // `k_silu<<<((long)N*Hd+255)/256, 256, 0, S>>>` after the conv module.
    // `elementwise` evaluates `rows * width` to the same `ceil(n/256)` blocks
    // of 256 for both, which is what ONE row covering two call sites means:
    // they differ only in the rectangle handed to the rule.
    kernel!(k_silu "vision::k_silu_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            t: Usize,
        ]),
    // `Elementwise` -- `k_axpy<<<((long)N*Hd+255)/256, 256, 0, S>>>` in the
    // macaron half-step, and `elementwise` evaluates `rows * width` to the
    // same `ceil(N*Hd/256)` blocks of 256.
    //
    // `scale` is `w.residual_weight` off the checkpoint. An operand, for the
    // same reason `k_pool_finish`'s `s` is one.
    kernel!(k_axpy "vision::k_axpy_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            a: BufMut,
            b: Buf,
            scale: F32,
            t: Usize,
        ]),
    // `Tile16` -- `k_matmul_bias<<<G2(OPD,N), B2, 0, S>>>` at
    // `gemma4_audio.cu:283`, the output projection's one fire, with `B2`/`G2`
    // on `:131`. `rows = N`, `width = OPD`: at 47 subsampled frames and
    // `output_proj_dim = 1536` the launcher builds
    // `dim3((1536+15)/16, (47+15)/16) = [96, 3, 1]` over `dim3(16,16)`, and
    // `tile16` evaluates the same.
    //
    // Two rows and not one shared with `k_matmul`, because they are two
    // KERNELS: `a = b ? F(b[o]) : 0.f` against `a = 0` is the whole
    // difference, and the header declines to merge them rather than put a
    // branch in the inner loop of the tower's hottest naive kernel. `b` is
    // nullable because the kernel guards it, even though this call site
    // always passes the projection's bias.
    kernel!(k_matmul_bias "vision::k_matmul_bias_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            x: Buf,
            w: Buf,
            b: Buf | null,
            y: BufMut,
            n: I32,
            k: I32,
            o: I32,
        ]),
    // `Tile16` -- `k_glu<<<G2(Hd,N), B2, 0, S>>>` at `gemma4_audio.cu:250`,
    // once per conformer layer. `rows = N`, `width = Hd`: 47 frames and 1 024
    // hidden give `[64, 3, 1]` blocks of `dim3(16,16)` from both the launcher
    // and `tile16`.
    //
    // **The width is the OUTPUT's and the input row is twice it.** The kernel
    // reads `x[n*2*D + d]` and `x[n*2*D + D + d]` off a `[N, 2*Hd]` buffer and
    // writes `[N, Hd]`; the launcher passes `Hd`. A row that took the input's
    // width would launch two tile-columns for every one the output has.
    //
    // NOT `Rule::SplitPacked`, which is the rule for a packed-in split-out
    // pointwise and is the one a reader reaches for next. It states `in_width`
    // over a flat grid with a 1-D block, and this kernel reads `threadIdx.y`.
    kernel!(k_glu "vision::k_glu_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            x: Buf,
            o: BufMut,
            n: I32,
            d: I32,
        ]),
    // `PerRowNarrow` -- `k_layernorm_relu<<<T1*F1, 128, 0, S>>>` at
    // `gemma4_audio.cu:189` and the same line over `T2*F2` and `C1` at `:196`.
    // `per_row_narrow` evaluates `grid[rows,1,1] block[128,1,1] smem 0`, and
    // at 188 mel frames the two fires are `[6016,1,1]` over 128 channels and
    // `[1504,1,1]` over 32.
    //
    // **The block width is the whole rule and it is numerics, not tuning.**
    // `Rule::PerRow` is this grid to the digit at 256 threads, and the fold
    // below sums `(blockDim.x + 31) / 32` per-warp partials SERIALLY in thread
    // zero -- four addends at 128, eight at 256. Same values, different order,
    // different last bit, and the tower is parity-checked against
    // `gemma4_audio_parity_ref.py` at cosine 0.99997 on THIS order. Stating
    // `PerRow` here is a launch that runs, a tower that answers and an encoder
    // that is no longer the checkpoint's.
    //
    // The rectangle is `[T*F, C]`: a row is one `(t, f)` cell's channel
    // vector, the grid counts cells and `width` is the channel count the block
    // folds. `per_row_narrow` deliberately leaves `Dims::width` unread -- `C`
    // reaches the kernel as an operand because the stride loop walks it.
    //
    // The 32 floats of `__shared__ float wm[32], wv[32]` are STATIC, so zero
    // dynamic bytes is the whole contract.
    //
    // In place over its first operand: both call sites pass `c0cl`/`c1cl` for
    // input and output. `w` is nullable because the kernel guards it, though
    // both fires pass an SSCP norm weight.
    kernel!(k_layernorm_relu "vision::k_layernorm_relu_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::PerRowNarrow,
        in_place = &[(0, 0)],
        operands = operands![
            x: Buf,
            w: Buf | null,
            o: BufMut,
            r: I32,
            c: I32,
            eps: F32,
        ]),
    // `Tile16` -- `dim3 g((FLAT+15)/16,(N+15)/16); k_sscp_flatten<<<g, B2, 0,
    // S>>>` at `gemma4_audio.cu:201`. `G2` spelled longhand, and the same
    // numbers: `rows = N = T2`, `width = FLAT = F2*C1`, so at 47 frames and
    // `32*32` flattened channels the launcher and `tile16` both build
    // `[64, 3, 1]` over `dim3(16,16)`.
    //
    // The width is a PRODUCT of two operands and neither of them is it. The
    // kernel takes `OC`, `To` and `Fo` and computes `FoOC = Fo * OC` itself;
    // the rule takes the width from the rectangle. They agree and neither is
    // the other's source -- a `__global__` cannot see its grid -- which is the
    // general shape of every extent in this family.
    kernel!(k_sscp_flatten "vision::k_sscp_flatten_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            input: Buf,
            out: BufMut,
            oc: I32,
            t_out: I32,
            f_out: I32,
        ]),
    // `Tile16` -- `k_qkv_scale<<<G2(Hd,N), B2, 0, S>>>` at
    // `gemma4_audio.cu:240`, once per conformer layer. `rows = N`,
    // `width = Hd = H * hd`: 47 frames and 1 024 hidden give `[64, 3, 1]`
    // blocks of `dim3(16,16)` from both.
    //
    // The width is the FLATTENED head axis, not a head count: the kernel reads
    // `e` as a column of `H*hd` and recovers `d = e % hd` inside. So the
    // rectangle is the activation's and no head field of `Dims` is involved,
    // which is what keeps this a tile rather than a head-shaped rule.
    //
    // In place over BOTH its buffers, which one pair could not say: `q` and
    // `k` are each read, scaled and written where they lie. `q_scale` and
    // `k_scale` are host constants -- `hd^-0.5 / ln2` and `ln(1+e) / ln2`,
    // computed once per encode -- and look like extents without being any.
    kernel!(k_qkv_scale "vision::k_qkv_scale_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut,
            k: BufMut,
            pds: Buf,
            n: I32,
            h: I32,
            hd: I32,
            q_scale: F32,
            k_scale: F32,
        ]),
    // `Tile16` -- `dim3 g((Hd+15)/16,(P+15)/16); k_rel_pos_enc<<<g, B2, 0,
    // S>>>` at `gemma4_audio.cu:220`, once per encode and shared across the
    // layers.
    //
    // **The rows are POSITIONS.** The rectangle is the relative-position table
    // `[P, hidden]` where `P = context_left`, not the tower's `[N, hidden]`
    // activation: at gemma-4's 13-deep window and 1 024 hidden the launcher
    // builds `[64, 1, 1]` blocks, where the activation's 47 frames would give
    // `[64, 3, 1]` and fill three tile-rows of a one-tile-row table.
    // [`kernels::LaunchRule::Tile16`]'s doc names this and `k_av` as the two
    // of the eleven whose rectangle is not the statement's output.
    //
    // The kernel takes no input buffer. It BUILDS its table from `P` and
    // `hidden` alone -- `pe` is write-only -- so there is no `in_place` pair
    // and no read operand to state.
    kernel!(k_rel_pos_enc "vision::k_rel_pos_enc_bf16",
        file = Some("vision/gemma4_audio.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            pe: BufMut,
            p: I32,
            hidden: I32,
        ]),
];

/// [`QWEN3_VL_TOWER`]'s instantiations — five of its eleven.
///
/// Five of the six refusals are DEAD kernels, which is the largest single
/// finding of this migration: `k_split_rope_qkv` fused `k_split_qkv_bias` and
/// `k_rope_qk` into one pass and both survivors were left behind, those two
/// have un-fused ancestors that were left behind before them, and
/// `k_add_inplace` is `k_add` under another name. The sixth is
/// `k_split_rope_qkv` itself — the closest call in the family, and the one
/// refusal here that is still about geometry. See `qwen3_vl_tower.cuh`.
static QWEN3_VL_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[0],
        template_path: "vision::device::k_bias",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[1],
        template_path: "vision::device::k_add_pe",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[2],
        template_path: "vision::device::k_gelu_tanh",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[3],
        template_path: "vision::device::k_gelu_bias",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QWEN3_VL_SIGS[4],
        template_path: "vision::device::k_merge_gather",
        elem: "device::bf16",
    },
];

/// The contracts, in [`QWEN3_VL_ROWS`]' order.
#[rustfmt::skip]
static QWEN3_VL_SIGS: [KernelSig; 5] = [
    // `Elementwise` -- `k_bias<<<((long)M*O+255)/256, 256, 0, S>>>` at all
    // three call sites (`gemm_bias`'s epilogue, o_proj's, fc2's), and
    // `elementwise` evaluates `rows * width` to the same `ceil(M*O/256)`
    // blocks of 256.
    //
    // `m` is `Usize` and `n` is `I32`, which is the kernel's own asymmetry and
    // not a row's carelessness: the bias index is `i % n` on a 64-bit `i`
    // against a 32-bit column count, and narrowing `i` first would wrap on a
    // tower whose token count times hidden crosses 2^31. `Ty::Usize` is
    // `std::size_t` and `device::usize` exactly, where `Ty::I64` is
    // `long long` and would be a third spelling of the same eight bytes.
    kernel!(k_bias "vision::k_bias_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut,
            b: Buf,
            m: Usize,
            n: I32,
        ]),
    // `Elementwise` -- `k_add_pe<<<((long)N*Hd+255)/256, 256, 0, S>>>` once
    // per forward, and `elementwise` evaluates `rows * width` to the same
    // `ceil(N*Hd/256)` blocks of 256.
    //
    // `pe` is the BILINEARLY INTERPOLATED absolute position embedding, and
    // the interpolation is not in this crate: it runs on the host under a
    // mutex, cached on `(grid_h, grid_w)`, because the table is per image
    // shape. This row is the four lines that are left of it.
    kernel!(k_add_pe "vision::k_add_pe_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            h: BufMut,
            pe: Buf,
            t: Usize,
        ]),
    // `Elementwise` -- `k_gelu_tanh<<<((long)N*Dmid+255)/256, 256, 0, S>>>` in
    // `mlp()` when `erf_gelu` is clear, which is every ViT block. Same
    // `ceil(n/256)` blocks of 256.
    kernel!(k_gelu_tanh "vision::k_gelu_tanh_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf,
            o: BufMut,
            t: Usize,
        ]),
    // `Elementwise` -- `k_gelu_bias<<<((long)N*IM+255)/256, 256, 0, S>>>`,
    // fc1's bias folded into its activation. Same `ceil(n/256)` blocks of 256.
    //
    // It takes `n` and `d` where the flat kernels take one count, and
    // computes `t = (long)N*D` itself. The rule recovers the same product
    // from `rows * width`; the kernel recovers it from its arguments. They
    // agree and neither is the other's source, which is the general shape of
    // every extent in this family: a `__global__` cannot see its grid.
    kernel!(k_gelu_bias "vision::k_gelu_bias_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut,
            b: Buf | null,
            n: I32,
            d: I32,
        ]),
    // `Tile16` -- `k_merge_gather<<<G2(W,n_token), B2, 0, S>>>` at
    // `qwen3_vl_tower.cu:165` and `:168`, the main merger's arm and the
    // deepstack one, with `B2`/`G2` on `:139`. `rows = n_token`,
    // `width = W = merge_unit * hidden`: at 256 merged tokens and a 4 608-wide
    // group both the launcher and `tile16` build
    // `dim3((4608+15)/16, (256+15)/16) = [288, 16, 1]` over `dim3(16,16)`.
    //
    // One row covers both fires. They differ in WHICH buffer is gathered --
    // the main merger norms first and gathers the normed patches, the
    // deepstack one gathers first and norms the group -- which is an operand,
    // not a rectangle, and the two `G2(W, n_token)` are the same expression.
    //
    // The width is a product the kernel recomputes: it takes `U` and `C` and
    // forms `W = U * C` itself, where the host formed the same product for the
    // grid. The input is already in spatial-merge order because the HOST
    // reordered it (`merge_reorder` in the `.cu`), which is why a plain
    // concatenation suffices here where HF needs a five-way reshape -- and it
    // is also why `n_patch` never appears: the row's rows are the OUTPUT's
    // tokens, and the input's `n_token * U` patches are addressed inside.
    kernel!(k_merge_gather "vision::k_merge_gather_bf16",
        file = Some("vision/qwen3_vl_tower.cuh"),
        launch = LaunchRule::Tile16,
        operands = operands![
            h: Buf,
            g: BufMut,
            n_token: I32,
            u: I32,
            c: I32,
        ]),
];
