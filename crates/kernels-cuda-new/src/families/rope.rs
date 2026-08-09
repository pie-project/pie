//! `rope`'s JIT unit — one header, ten kernels, six rows.
//!
//! `rope/rope.cu` held 1058 lines in which ten `__global__`s and twelve host
//! launchers were interleaved. The device text is now `rope/rope.cuh`, which
//! the `.cu` includes, so exactly ONE definition of each kernel exists in the
//! tree. That is a SPLIT and not a copy on purpose: `norm/altup_aux` shipped
//! a release with two definitions of six kernels, they agreed the day they
//! were written, each stayed right for whichever half of the tests exercised
//! it, and nobody could see the drift until one half was edited.
//!
//! # Six rows out of ten kernels, and why the other four cannot have one
//!
//! **Re-audited at `LaunchRule` 21 → 28, and again at 36.** The first
//! re-audit moved nothing and said so; the second moves TWO, and it moves
//! them by retiring a claim this file made and no longer holds. The claim
//! was that a PACKED `q + kv` head axis is a shape no rule states. It is
//! `LaunchRule::RowsPackedHeadsNarrow` now, and the two kernels that launch
//! it with `rows` on the first axis — `qk_rmsnorm_rotate` and
//! `qk_rmsnorm_rotate_mrope` — are rows. The other two forms of the same
//! kernel are still refused and NOT for that reason; see below, because the
//! reason changed and a refusal that keeps its old sentence is a refusal
//! nobody can check.
//!
//! What is left is structural rather than arithmetic: two are host-computed
//! OPERANDS (`rotate`'s `kWriteKv`/`kHnd` at `rope/rope.cu:85` and `:88`,
//! and the YaRN ramp) and three are plain `__global__`s with nothing to
//! instantiate. `AxialRope` is the one rule whose NAME suggests this family,
//! and its launch is `grid [1, heads, rows]` at a 32-wide block — a vision
//! tower's axial table, not a token-per-block rotation — which none of them
//! launch.
//!
//! Two templates carry the four rows. `standard_table<P>` builds the cos/sin
//! table one block per token, and `rotate_partial<T>` rotates the first
//! `rotary_dim` channels of q and k in place, also one block per token — both
//! are `LaunchRule::RouteRows` exactly. `rotate_partial` gets three rows: the
//! q-only rotation, the position-delta form, and an fp16 instantiation the
//! ahead-of-time build never had. That last one is the measurement
//! `norm/elementwise` made first — a second numeric format cost a
//! translation unit's worth of `cicc` under nvcc and costs a ROW here.
//!
//! # What the launch-rule port changed here: the rule arrived, the name did not
//!
//! `Rule::Rope` landed with the rest of the head-shaped rules, and it was
//! ported FROM this file's `rope_bf16` — `runtime::launch::rope` quotes the
//! `<<<>>>` it reproduces, and it reproduces it digit for digit: `BLOCK` 256,
//! `half = head_dim / 2`, `heads_per_block = half >= 256 ? 1 : 256 / half`,
//! `grid(num_tokens, ceil(total_heads / heads_per_block))`, and
//! `smem = (half <= 4096 ? half : 0) * 2 * sizeof(float)`. Checked against
//! the launcher for the same extents, the rule and the `<<<>>>` agree on
//! every field. So the six kernels below are no longer blocked on GEOMETRY.
//! They are blocked on being NAMEABLE, which is a different fact about a
//! different file:
//!
//! * `rotate<bool kWriteKv, bool kHnd>` is the kernel `Rule::Rope` was ported
//!   from, it is templated on two NON-TYPE parameters and no type parameter
//!   at all, and it is NAMEABLE. That was reported here as the blocker and
//!   the report was wrong; the correction is measured, on this L40S under
//!   NVRTC 13.0, and it is recorded here rather than in the earlier text
//!   because a refusal that cites the wrong reason is a refusal nobody can
//!   overturn.
//!
//!   [`DeviceKernel::instantiation`](crate::device::DeviceKernel) pastes
//!   `elem` whole between the angle brackets and glues
//!   `::pie_cuda_driver::kernels::` to its FRONT — to the string, which means
//!   to its first token only. So the arity was never the constraint and the
//!   SPELLING of the first argument always was:
//!
//!   ```text
//!   elem = "device::false_type::value, false"
//!     -> ::pie_cuda_driver::kernels::rope::device::rotate<
//!            ::pie_cuda_driver::kernels::device::false_type::value, false>
//!     -> _ZN15pie_cuda_driver7kernels4rope6device6rotateILb0ELb0EEE...
//!   elem = "false, false"
//!     -> ::pie_cuda_driver::kernels::rope::device::rotate<
//!            ::pie_cuda_driver::kernels::false, false>
//!     -> error: expected an identifier
//!   ```
//!
//!   A bare literal cannot survive the prefix; a QUALIFIED constant
//!   expression can, and `pie_device.cuh` already spells both kinds —
//!   `device::false_type::value` and `device::true_type::value` at
//!   `pie_device.cuh:485`, `device::i32(...)` at `:463`. `nvrtcAddNameExpression`
//!   parses C++, not a name, and both lowered names above came back from it.
//!   The earlier measurement it contradicts — `recurrent_step_batched_gqa_smem`
//!   failing at `<::pie_cuda_driver::kernels::64>` — is real and is the bare
//!   spelling of the same thing.
//!
//!   `rotate` is therefore blocked one level further in than this file
//!   claimed, on OPERANDS, and there it is blocked hard. Two of its twenty
//!   parameters are host conditionals (`rope.cu:85` and `:88`):
//!
//!   ```text
//!   const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
//!   const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
//!   ```
//!
//!   `runtime::launch`'s own `rope` doc names this exact hazard — *"the rule
//!   and the binder must derive it from the same `head_dim` or the grid
//!   covers a head count the kernel does not agree it has"* — and `Source`
//!   cannot. There is no `Max`, no `Min` and no comparison in the grammar;
//!   `Ne` yields a bool operand and `Or` tests whether a binding is PRESENT,
//!   not whether a computed value is zero. The nearest expression,
//!   `Div(Lit(256), Div(head_dim, Lit(2)))`, agrees with the launcher at
//!   every `head_dim` up to 512 and returns **0** past it — MLA's 576 gives
//!   `256 / 288` — and the kernel reads that value twice, as
//!   `head_base = blockIdx.y * heads_per_block` and
//!   `heads_here = min(heads_per_block, total_heads - head_base)`. Zero makes
//!   both zero: every block in a full grid falls out of the loop on its first
//!   test and rotates NOTHING, silently, on the tensor attention reads next.
//!   That is the failure mode this design exists to prevent, arriving through
//!   an operand instead of an extent. `cache_pairs` is the same shape with
//!   the shared table's size attached: `Lit(0)` is safe and bit-identical —
//!   the kernel recomputes each pair when the cache is empty — but `half`
//!   unconditionally overruns the zero `smem` the rule allocates past
//!   `head_dim` 8192, and neither spelling is what the launcher computes.
//!
//!   `rope_write_kv_bf16` blocks the `kWriteKv = true` arm a second time:
//!   it instantiates `rotate<true, decltype(hnd)::value>` through a lambda
//!   over the RUN-TIME `hnd_layout`, so two instantiations sit behind the one
//!   ahead-of-time symbol `rope::rope_write_kv_bf16` and a row would state
//!   one arm of a host decision. That is the case the arity note calls out
//!   by name and leaves blocked.
//!
//!   The kernel stays as it is. Templating it over `T` would ADD a third
//!   parameter and name nothing new — `rotate_pair` in `rope_device.cuh`
//!   takes `bf16*`, and that shared header is what stops all six of these
//!   being widened to fp16.
//! * `rotate_yarn` is `dim3(num_tokens, ceil(total_heads / heads_per_block))`
//!   at 256 with the same `heads_per_block` arithmetic — `Rule::Rope`'s grid
//!   and block exactly, with `smem` the one field that differs: the rule
//!   would hand it up to 32 KB of dynamic shared memory it never reads, an
//!   over-allocation in the direction that is legal and visible rather than
//!   silent. `rotate_yarn_original` matches on `smem` too —
//!   `cache_pairs * sizeof(float2)` is the rule's `cache_pairs * 2 * 4` — so
//!   the rule reproduces that launcher completely. Both are plain
//!   `__global__`s. `rotate_yarn_original` is blocked a second time: its
//!   `low_dim` and `high_dim` come from `yarn_original_ramp_bounds` running
//!   ON THE HOST before the launch.
//! * `qk_rmsnorm_rotate` and `qk_rmsnorm_rotate_mrope` **are rows.** This
//!   file refused all four forms of this kernel as
//!   `dim3(num_tokens, num_q_heads + num_kv_heads)` at 128, a packed head
//!   axis no rule stated, and gave three independent sub-reasons. All three
//!   are stale, and they are stale in the same stroke:
//!   `LaunchRule::RowsPackedHeadsNarrow` is `[rows, q_heads + kv_heads, 1]`
//!   at `[128, 1, 1]` with no dynamic shared memory. Checked against
//!   `rope/rope.cu:189-191`
//!
//!   ```text
//!   constexpr int BLOCK = 128;
//!   dim3 grid(num_tokens, num_q_heads + num_kv_heads);
//!   device::qk_rmsnorm_rotate<BLOCK><<<grid, BLOCK, 0, stream>>>(
//!   ```
//!
//!   the rule and the `<<<>>>` agree on every field, and the sub-reasons
//!   fall with it: `GatedRms`' 256 is not what the rule opens, so the static
//!   `__shared__ float buf[BLOCK]` is exactly the block the rule opens; and
//!   the reduction's `i += BLOCK` strides that same 128. The mrope form is
//!   `rope/rope.cu:45-47`, the same three numbers over
//!   `qk_rmsnorm_rotate_mrope`. Note the rule was ported from a DIFFERENT
//!   launcher — `attn/qkv_fused.cu:98-102` — which is why it could arrive
//!   without anybody here noticing, and why both citations are pinned in
//!   `tests/launch_rules.rs::mod transcribed` and both rows are fired
//!   against a raw `cuLaunchKernel` at the launcher's own geometry.
//!
//!   `RowsPackedHeadsNarrow` is STRICTER than these two launchers in one
//!   place, and which way it is strict is the whole of why that is
//!   tolerable: `packed_heads` refuses `kv_heads == 0`, where the kernel
//!   would have run (`total_heads` would be `num_q_heads` and the k branch
//!   never taken). The strictness is a REFUSAL — `Ungeometric::Empty`, so
//!   `Error::Geometry` at the fire, loud and before any launch — and not a
//!   wrong grid. No fire of either symbol reaches it: both state two results
//!   at every call site (`model-compiler/src/dsl.rs:7342` and `:4233`), so
//!   the second bank is a width and not an absence. The form where a zero
//!   bank IS reached is `_rounded`, and there the zero never gets to the
//!   rule at all — which is that refusal, below.
//! * `qk_rmsnorm_rotate_rounded` launches the same three numbers at
//!   `rope/rope.cu:213-215` and is **refused**, for a reason this file did
//!   not previously state. The rule's arrival did not remove it; it made it
//!   visible. For this symbol the rule's head counts and the launcher's
//!   arguments are not the same numbers.
//!
//!   `rope::qk_rmsnorm_rope_bf16_rounded` is stated at TWO call sites.
//!   `model-compiler/src/dsl.rs:6588` records it from
//!   `dsl::cuda::qk_rmsnorm_rope_rounded_q_only`, which gemma-4's SHARED
//!   sliding layer calls (`model/src/gemma_4/forward/mod.rs:217`) with one
//!   result and one weight; the driver reaches the launcher by passing
//!   `k_norm = nullptr` and `num_kv_heads = 0`. The ahead-of-time row says
//!   exactly that in its `Source`s (`table/rope.rs:191-218`):
//!   `k <- Or(Out(1), Lit::Null)` and
//!   `num_kv_heads <- Or(Div(Width(Out(1)), head_dim), Lit::I32(0))`.
//!
//!   The LAUNCHER takes its grid from those arguments, so a q-only fire
//!   opens `q + 0` columns. The RULE takes its grid from `Dims`, and
//!   `Dims::kv_heads` is `extent(ctx.num_kv_heads)`
//!   (`driver-cuda/src/bind/mod.rs:1379`) — a FIRE-WIDE geometry fact the
//!   context carries for the model, documented as such where the field is
//!   declared (`bind/mod.rs:767`: *"a fire-wide geometry fact, not something
//!   an arm derives from a width"*), and non-zero on a gemma-4 shared layer.
//!   So the rule opens `q + kv` columns where the launcher opens `q`, and
//!   every excess block takes `is_q == false` and addresses
//!   `k + (n * num_kv_heads + local) * head_dim` with `num_kv_heads == 0`
//!   and `k == nullptr`. `packed_heads`' zero-refusal never saves it,
//!   because the zero is in the OPERAND and the rule never reads an operand.
//!
//!   This is the hazard `runtime::launch`'s own `rope` doc names — *"the
//!   rule and the binder must derive it from the same `head_dim` or the grid
//!   covers a head count the kernel does not agree it has"* — arriving
//!   through a head COUNT rather than a head dim. It cannot be fixed from
//!   this file. Sourcing `num_kv_heads <- Ctx("num_kv_heads")` would make
//!   rule and operand agree and would then rotate `k` through a null pointer
//!   at the q-only site; `k <- Or(Out(1), Out(0))`, which
//!   `rope_partial_q_only` uses legitimately, would be worse here, because
//!   the excess blocks would land on token 0's q heads and clobber them
//!   silently rather than fault. And a row is ONE contract per symbol, so it
//!   cannot be one thing at each site. **What it needs, reported and not
//!   built:** either the q-only form given its own symbol, as `_devwin` has
//!   one, or a `LaunchRule` whose head axis is read from the same expression
//!   the operand is — reproducing `rope/rope.cu:214`,
//!   `dim3 grid(num_tokens, num_q_heads + num_kv_heads)`, where
//!   `num_kv_heads` is the STATEMENT's second width and not the fire's.
//!
//!   The refusal is MEASURED rather than argued.
//!   `tests/launch_rules.rs::mod transcribed` transcribes `rope.cu:214` and
//!   asserts `RowsPackedHeadsNarrow` agrees with it at a both-banks fire and
//!   DISAGREES at the q-only one; `mod fires` fires both grids on the device
//!   and counts the bytes.
//! * `qk_rmsnorm_rotate_devwin` is `dim3(n_max, num_q_heads + num_kv_heads)`
//!   at 128 (`rope/rope.cu:161-164`) — the same packed axis. **Landed**, as
//!   `ROPE_SIGS[6]`, on the chain the hold recorded: `n_max` is the fire's
//!   FULL lane count (`DispatchCtx::rows_total`, `bind/mod.rs:884-887`), the
//!   twin is `whole = true` (`table/rope.rs:86`), and `lower.rs:1064`
//!   refuses a `whole` statement any window but `[0, rows)` — so
//!   `Dims::rows` IS `n_max` and not by coincidence. What held it was never
//!   geometry but a grant boundary, and the pass that owned class B found
//!   the boundary was not there: class B is the RESIDUE in
//!   `examples/migration_status`, computed rather than listed, so it carries
//!   no per-row text and this row rewrote none.
//! * `rotate_partial_last` takes `low_dim` and `high_dim` as arguments
//!   because the YaRN ramp is computed ON THE HOST, in the launcher, over
//!   `rotary_dim` rather than `head_dim`. No `Source` in
//!   `kernels/src/lib.rs` derives a ramp bound from `beta_fast`, `beta_slow`
//!   and `rotary_dim`, and `new-horizon.md` §10.5 refuses an invented one.
//!   It is a plain `__global__` besides. The same host function is the one
//!   NVRTC rejected outright when it was unannotated — nvcc accepts a bare
//!   `inline` inside a `.cu`, NVRTC does not, and it carries
//!   `__host__ __device__` now because the host still calls it.
//!
//! So the port moved this family's blocker from "no rule states this launch"
//! to three separate places, and the arity finding — which arrived after this
//! file was first written and cost it the paragraph above — moved NONE of the
//! six. The `LaunchRule` 36 re-audit then moved two, by retiring the packed
//! head axis as a refusal: `rope` stands at **5 of 12**, and what stays is
//! `rotate` (nameable, blocked on an operand no `Source` computes),
//! `rotate_yarn`, `rotate_yarn_original` and `rotate_partial_last` (plain
//! `__global__`s, two of them behind a host-computed YaRN ramp),
//! `qk_rmsnorm_rotate_rounded` (whose head COUNT is the fire's and not the
//! statement's), and `qk_rmsnorm_rotate_devwin` (held for classification,
//! not for geometry). Every one of them was re-derived against its `<<<>>>`
//! and its parameter list, twice now, because a blocker that was stated
//! wrong once has to be restated from the source and not from the report —
//! and the second re-derivation is what caught that three of the four
//! sub-reasons against the packed axis had gone stale together.
//!
//! Every one of the twelve launchers stays where it is. This migration
//! extracts device text and adds rows; it deletes nothing, and
//! `new-horizon.md` §10.10 fixes that order so the two paths can be measured
//! against each other before either is retired.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Lit;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// `rope`'s device text: the table builder, the five fused QK-norm rotations,
/// the two YaRN forms, and the two partial rotations.
pub const ROPE: Unit = Unit {
    name: "rope/rope",
    root: include_str!("../../csrc/src/rope/rope.cuh"),
    rows: ROPE_ROWS,
    options: &[],
};

/// The units `rope` compiles.
pub static UNITS: &[Unit] = &[ROPE];

/// [`ROPE`]'s instantiations.
///
/// `standard_table` is templated over its POSITION type and `rotate_partial`
/// over its ELEMENT type, which is why the `elem` column reads `i32` on the
/// first row and a float format on the rest: `DeviceKernel::instantiation`
/// writes one type argument, and what that type MEANS is the template's
/// business. `graph_pad` writes its pad lanes' positions as `u32`, so the day
/// a caller wants the table built straight off those it costs a row here and
/// no C++ anywhere.
static ROPE_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ROPE_SIGS[0],
        template_path: "rope::device::standard_table",
        elem: "device::i32",
    },
    DeviceKernel {
        sig: &ROPE_SIGS[1],
        template_path: "rope::device::rotate_partial",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ROPE_SIGS[2],
        template_path: "rope::device::rotate_partial",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ROPE_SIGS[3],
        template_path: "rope::device::rotate_partial",
        elem: "device::f16",
    },
    // THE TWO THE PACKED HEAD AXIS COST. `elem` is a non-type argument here
    // and a TYPE on every row above, which is `instantiation`'s business and
    // not this table's: it pastes the string between the brackets. It has to
    // be `device::i32(128)` and not `128`, because the prefix is glued to the
    // FIRST TOKEN — see the arity note in this module's header.
    DeviceKernel {
        sig: &ROPE_SIGS[4],
        template_path: "rope::device::qk_rmsnorm_rotate",
        elem: "device::i32(128)",
    },
    DeviceKernel {
        sig: &ROPE_SIGS[5],
        template_path: "rope::device::qk_rmsnorm_rotate_mrope",
        elem: "device::i32(128)",
    },
    // THE THIRD, and the one the header held rather than refused.
    DeviceKernel {
        sig: &ROPE_SIGS[6],
        template_path: "rope::device::qk_rmsnorm_rotate_devwin",
        elem: "device::i32(128)",
    },
];

/// The contracts, in [`ROPE_ROWS`]' order.
///
/// Each is its ahead-of-time twin in `kernels-cuda-new/src/table/rope.rs`
/// minus two
/// things. The stream goes because a stream is `cuLaunchKernel`'s SIXTH
/// PARAMETER and not a member of the `void**` — the pilot's rows carried it
/// as an operand and it was wrong in a way nothing caught until a launch
/// argument list was counted. `num_tokens` goes because
/// `LaunchRule::RouteRows` IS one block per row, so the extent is the fire's
/// rectangle and an operand restating it is an operand that can disagree with
/// the grid. Four operands out of six for the table, nine out of eleven for
/// the rotations.
#[rustfmt::skip]
static ROPE_SIGS: [KernelSig; 7] = [
    // The cos/sin table `attn`'s fused prepare reads. One block per token,
    // the block striding `head_dim/2` pairs -- `RouteRows` sizes it
    // `min(1024, ceil(width/32)*32)` where the launcher fixed 256, so the
    // wider block reaches the same pairs in fewer iterations and the
    // arithmetic per pair is unchanged. `powf`/`__sincosf` are the same
    // instructions either way.
    kernel!(rope_standard_table "rope::rope_standard_table",
        file = Some("rope/rope.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            positions: I32s <- Source::Positions,
            table: F32sMut <- Source::Out(0),
            head_dim: I32 <- Source::Ctx("head_dim"),
            theta: F32 <- Source::CtxNonZero("rope_theta"),
        ]),
    // Q-only rotation: a KV-shared layer's K was rotated at its source layer,
    // so one operand is the whole statement. Rotates q and k WHERE THEY LIE
    // -- two aliases, which is what the pair list exists for, and a q-only
    // site states one result so the second pair falls outside its arity.
    //
    // `position_delta` is a LITERAL zero here where the twin had no such
    // parameter at all. The ahead-of-time build shipped `rope_partial_bf16`
    // and `rope_partial_bf16_position_delta` as two `__global__`s that
    // differed by `+ 0`; one template with a delta is the same instruction
    // count, and the delta sits between `positions` and the extents because
    // that is where the kernel's signature puts it -- exactly the kind of
    // fact a hand-written binding gets wrong and a generated one cannot.
    kernel!(rope_partial_q_only "rope::rope_partial_bf16",
        file = Some("rope/rope.cuh"),
        launch = LaunchRule::RouteRows,
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            // A Q-ONLY SITE STATES ONE RESULT and the launcher takes q for k
            // with `num_kv_heads = 0`. An `Or`, and what decides is whether
            // the second result is there.
            k: BufMut <- Source::Or(&Source::Out(1), &Source::Out(0)),
            positions: I32s <- Source::Positions,
            position_delta: I32 <- Source::Lit(Lit::I32(0)),
            // The head COUNTS off the cache's head dim rather than the ctx's,
            // because a KV-shared layer's q and k disagree.
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &Source::KvLayerField("head_dim"),
            ),
            // ZERO when there is no second result, which is the q-only form's
            // whole signal to the kernel: `total_heads` is then `num_q_heads`
            // and the k branch is never taken.
            num_kv_heads: I32 <- Source::Or(
                &Source::Div(
                    &Source::Width(&Source::Out(1)),
                    &Source::KvLayerField("head_dim"),
                ),
                &Source::Lit(Lit::I32(0)),
            ),
            head_dim: I32 <- Source::KvLayerField("head_dim"),
            rotary_dim: I32 <- Source::RotaryWidth,
            theta: F32 <- Source::CtxByLayer("theta"),
        ]),
    // The shifted form -- `positions` plus a host constant, for a speculative
    // window whose absolute positions are the verify pass's. Sourced by
    // nothing, exactly as its twin sources nothing: the delta is a fact about
    // a draft/verify pairing that no statement and no context carries yet.
    kernel!(rope_partial_position_delta "rope::rope_partial_bf16_position_delta",
        file = Some("rope/rope.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            q: BufMut,
            k: BufMut,
            positions: I32s,
            position_delta: I32,
            num_q_heads: I32,
            num_kv_heads: I32,
            head_dim: I32,
            rotary_dim: I32,
            theta: F32,
        ]),
    // THE ROW THE AHEAD-OF-TIME BUILD COULD NOT AFFORD. Identical to
    // `rope_partial_q_only` but for the type argument, and it exists because
    // under a JIT that is all a second numeric format costs. Under nvcc it
    // cost a translation unit's worth of `cicc` for something no caller had
    // asked for yet, which is why `rope.cu` named every kernel `_bf16` and
    // meant "the one instantiation we could pay for".
    //
    // `rotate_partial` converts through `Elem<T>` and never touches a
    // `__nv_bfloat16` intrinsic, so fp16 is the same rounding at a different
    // exponent width -- which is why THIS template could be widened and the
    // six that call `rope_device.cuh`'s `rotate_pair` could not: that header
    // takes `bf16*`, and it is shared and read-only.
    kernel!(rope_partial_f16 "rope::rope_partial_f16",
        file = Some("rope/rope.cuh"),
        launch = LaunchRule::RouteRows,
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            k: BufMut <- Source::Or(&Source::Out(1), &Source::Out(0)),
            positions: I32s <- Source::Positions,
            position_delta: I32 <- Source::Lit(Lit::I32(0)),
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &Source::KvLayerField("head_dim"),
            ),
            num_kv_heads: I32 <- Source::Or(
                &Source::Div(
                    &Source::Width(&Source::Out(1)),
                    &Source::KvLayerField("head_dim"),
                ),
                &Source::Lit(Lit::I32(0)),
            ),
            head_dim: I32 <- Source::KvLayerField("head_dim"),
            rotary_dim: I32 <- Source::RotaryWidth,
            theta: F32 <- Source::CtxByLayer("theta"),
        ]),
    // The fused QK RMS-norm + rotation, `rope/rope.cu:189-191`:
    //
    //     constexpr int BLOCK = 128;
    //     dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    //     device::qk_rmsnorm_rotate<BLOCK><<<grid, BLOCK, 0, stream>>>(
    //
    // which is `LaunchRule::RowsPackedHeadsNarrow` on every field. The
    // header records what this row cost and why it took a second re-audit
    // to notice: the rule was ported from `attn/qkv_fused.cu:98-102`, a
    // launcher in a different family that computes the same three numbers.
    //
    // Two operands of the twin go, and the reasons are the two this file
    // already gives. `stream` is `cuLaunchKernel`'s sixth parameter and not
    // a member of the `void**`. `num_tokens` goes because the kernel reads
    // `blockIdx.x` and never the argument -- it was never in the KERNEL's
    // parameter list, only the launcher's, and an operand restating the
    // grid is an operand that can disagree with it. Ten out of twelve.
    //
    // The `Source`s are the twin's (`table/rope.rs:66`) verbatim otherwise,
    // and both head counts come off a RESULT width over `ctx.head_dim` --
    // the same divisor the rule's `Dims::head_dim` is filled from, which is
    // what makes the grid the rule opens and the rows the kernel addresses
    // the same rectangle. `Dims::q_heads`/`kv_heads` come from the ctx, and
    // for this symbol they agree with the widths because llama_like states
    // both banks at every call site. The form where they DO NOT agree is
    // `_rounded`, and that is why `_rounded` has no row.
    kernel!(qk_rmsnorm_rope "rope::qk_rmsnorm_rope_bf16",
        file = Some("rope/rope.cuh"),
        launch = LaunchRule::RowsPackedHeadsNarrow,
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            k: BufMut <- Source::Out(1),
            q_weight: Buf <- Source::Weight(0),
            k_weight: Buf <- Source::Weight(1),
            positions: I32s <- Source::Positions,
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &Source::CtxNonZero("head_dim"),
            ),
            num_kv_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(1)),
                &Source::CtxNonZero("head_dim"),
            ),
            head_dim: I32 <- Source::Ctx("head_dim"),
            theta: F32 <- Source::CtxByLayer("theta"),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // MROPE: the same rotation over `[num_tokens, 3]` (t, h, w) positions,
    // because a vision model's tokens sit in a grid. `rope/rope.cu:45-47`
    // is the same three numbers over `qk_rmsnorm_rotate_mrope`, so the same
    // rule, and the three `mrope_section_*` extents ride at the end where
    // the kernel's signature puts them.
    //
    // SOURCED BY NOTHING, exactly as its twin (`table/rope.rs:112`) sources
    // nothing. The section split is a property of a vision checkpoint that
    // no statement and no context carries yet, and §10.5 refuses an invented
    // one; the head counts could be sourced as above but a half-bound row is
    // a row whose unbound cells look like an oversight rather than a fact.
    kernel!(qk_rmsnorm_mrope "rope::qk_rmsnorm_mrope_bf16",
        file = Some("rope/rope.cuh"),
        launch = LaunchRule::RowsPackedHeadsNarrow,
        operands = operands![
            q: BufMut,
            k: BufMut,
            q_weight: Buf,
            k_weight: Buf,
            positions: I32s,
            num_q_heads: I32,
            num_kv_heads: I32,
            head_dim: I32,
            theta: F32,
            eps: F32,
            mrope_section_t: I32,
            mrope_section_h: I32,
            mrope_section_w: I32,
        ]),
    // The device-window form, `rope/rope.cu:161-164`:
    //
    //     if (n_max <= 0) return;                                     // :160
    //     constexpr int BLOCK = 128;                                  // :161
    //     dim3 grid(n_max, num_q_heads + num_kv_heads);               // :162
    //     device::qk_rmsnorm_rotate_devwin<BLOCK><<<grid, BLOCK, 0, stream>>>(
    //
    // the SAME three numbers as `[4]` above with `num_tokens` renamed
    // `n_max`, so the same `LaunchRule::RowsPackedHeadsNarrow`, and the
    // module header held it rather than refusing it for exactly this reason.
    //
    // **Why `Dims::rows` IS `n_max`, which is the whole of the claim.**
    // `n_max` is the fire's FULL lane count, not the window's: the kernel
    // reads `win[0]`/`win[1]` out of DEVICE memory at `rope.cuh:485-487` and
    // early-outs the lanes outside, so the grid must span every lane whatever
    // the window turns out to be. The chain that makes `Dims::rows` that
    // number is `attn`'s `write_kv_explicit_devwin` block's, verbatim: the
    // ahead-of-time twin is `whole = true` (`table/rope.rs:86`), and
    // `model-compiler`'s `lower.rs:1064` refuses a `whole` statement any
    // window but `[0, rows)`. A `_devwin` row is `whole` BECAUSE its window is
    // a device word the lowering cannot see — `table/rope.rs:82-85` says so —
    // which is the same fact read from the other end.
    //
    // Two operands of the twin go, and they are `[4]`'s two: `stream`, which
    // is `cuLaunchKernel`'s sixth parameter, and `n_max`, which the KERNEL
    // never had — its parameter list at `rope.cuh:470-481` runs `win` then
    // straight to `num_q_heads`, and `n` is `blockIdx.x`. Eleven out of
    // thirteen.
    //
    // UNSOURCED, and the twin is too. A hooked pure-decode fire is graph
    // CAPTURED and `win_d` is a device word the driver writes between
    // replays; no `Source` reads device memory, and one invented to make this
    // row look bound would be a guess in the one place the design has none.
    // The three `write_kv_explicit_devwin` rows in `families/attn.rs` are
    // unsourced for the same sentence.
    kernel!(qk_rmsnorm_rope_devwin "rope::qk_rmsnorm_rope_bf16_devwin",
        file = Some("rope/rope.cuh"),
        launch = LaunchRule::RowsPackedHeadsNarrow,
        whole = true,
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut,
            k: BufMut,
            q_weight: Buf,
            k_weight: Buf,
            positions: I32s,
            win_d: U32s,
            num_q_heads: I32,
            num_kv_heads: I32,
            head_dim: I32,
            theta: F32,
            eps: F32,
        ]),
];
