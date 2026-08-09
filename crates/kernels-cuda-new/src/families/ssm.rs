//! `ssm`'s JIT units — all five of the family's headers, and eighteen rows.
//!
//! The family is four `.cu` files and 40 `__global__`s, and the split that
//! made it JIT-able produced five `.cuh` headers, each a unit here. The
//! hardest — `ssm/gated_delta_net.cuh` — holds fourteen kernels and five
//! rows over three of them, and its header says at length what blocks the
//! other eleven.
//!
//! # Twenty-one rows became eighteen, and the three that went were duplicates
//!
//! `new-horizon.md` §28.4: six of this family's rows were a second name for
//! a job a `_batched` row already does — `ssm::recurrent_gated_delta_step`
//! and its `_state_bf16`, `ssm::chunk_gated_delta_prefill` and its
//! `_state_bf16`, `ssm::causal_conv1d_update_bf16` and
//! `ssm::causal_conv1d_prefill_bf16` — and four of the six were hosted here.
//! Every golden that runs a gated-delta model names the `_batched` spelling:
//! `_batched` at 2 goldens against 0, `causal_conv1d_update_batched_bf16` at
//! 6 against 0. Nothing named the singles: no model text, no `lower.rs`
//! arm, no driver fire, no test, and `dsl.rs`'s five wrappers had zero call
//! sites between them. The `.cu` launchers stay — `gated_delta_net.cu`'s
//! host `for` calls the single step T times from inside C++, which is the
//! consumer §10.10 requires to be gone first.
//!
//! # Two blockers this file named, and neither survived
//!
//! An earlier revision said one sentence: *a recurrent state family puts the
//! HEAD on a grid axis, and CUDA's `LaunchRule` cannot name one.*
//! [`crate::runtime::launch::Dims`] now carries nine extents —
//! `q_heads`, `kv_heads`, `head_dim`, `rotary_dims`, `n_experts`,
//! `experts_per_token` beside the three that were there — and `eval`
//! evaluates twelve of the vocabulary's sixteen rules, including
//! [`kernels::LaunchRule::PerHeadElementwise`] on `dim3(rows, heads)`,
//! [`kernels::LaunchRule::GatedRms`] on the same grid at 256, and
//! [`kernels::LaunchRule::SplitPacked`] on `dim3(ceil(width / 256), rows)`.
//! `Args::bind` now marshals [`kernels::Ty::I64`], so `slot_stride_elems` —
//! a `long long` on purpose, an element count into a multi-gigabyte state
//! arena — is passable, and [`crate::runtime::launch::Launch`] carries an
//! `smem` four rules COMPUTE rather than one rule setting a constant.
//!
//! The next revision named a second, and it was wrong: *a row spells ONE
//! template argument, so `<StateT, bool KLast>` and `<T, int BLOCK>` are
//! unnameable.* [`crate::device::args`] records the measurement that killed
//! it — `elem` is a string pasted between angle brackets and
//! `nvrtcAddNameExpression` parses C++, so `"device::bf16, 256"` names
//! `rmsnorm<bf16, 256>` and nothing about the table or the compile path
//! changes. **This family gained no row from that finding directly**, and
//! two from the `csrc` edits it made worthwhile: `zamba_rmsnorm_gated` and
//! `repeat_interleave_heads_fp32` were plain `__global__`s parked behind the
//! ceiling, are `template <class T>` now for no reason but to be nameable,
//! and both are [`kernels::LaunchRule::GatedRms`].
//!
//! # What blocks the rest, and none of it is arity
//!
//! **Re-audited at `LaunchRule` 21 → 28,** and three entries moved:
//! `l2norm_scale` under `PerRowNarrow` and the two warp-tiled prefills under
//! `WarpTiledScan`, all three below with the launcher each reproduces. Of the
//! other five new rules, none reaches this family: `Slab`, `RowsFlat`,
//! `RowsPerHead`, `Tile16`, `AxialRope` and `RoutedQmv` have no launcher of
//! their shape in `ssm/*.cu`. Every remaining entry below stands, and one of
//! them deserves its name said twice because it is the easiest thing here to
//! get wrong: `recurrent_gated_delta_step_batched_gqa_state_bf16` picks a
//! DIFFERENT KERNEL and not a different grid.
//!
//! It used to pick it on `std::getenv("PIE_QWEN35_GDN_SMEM_STEP")`. **An
//! environment variable is not geometry**, and no rule — of 21, of 28, or of
//! any number — can make it one; but the fix was not to find the rule that
//! could. §30 measured the two arms first, and they are **byte-identical**
//! on the state slab and on `out` at eight shapes, including the two where
//! the gate is off. A knob whose arms never differ chose only speed, and
//! only downward, so it was DELETED rather than relocated. What is left is
//! `V_d == 128 && K_d == 128` — a shape the fire already carries, which
//! costs this vocabulary nothing and which §26.10(b)'s `Term::IntIs` reads
//! directly. The residue below is now honestly geometric.
//!
//! * **A grid no ported rule computes — HALF RETIRED.** `mamba_split` is a
//!   row now. That bullet said *"`Elementwise` sizes on the OUTPUT width
//!   where the launcher used the input's"*, and it was a report on the
//!   vocabulary of the day: [`kernels::LaunchRule::ElementwiseIn`] sizes on
//!   `rows · in_width`, which is `N · projection_dim`, which is the
//!   launcher's `total` to the digit. See [`NEMOTRON_H_SIGS`]`[3]`.
//!
//!   **`mamba_split_conv_dt` stays refused, and its extent is now named.**
//!   Its grid is `ceil(N · (conv_dim + num_heads) / 256)` — `rows` times the
//!   SUM OF TWO RESULTS' WIDTHS — while it reads the same
//!   `[N, projection_dim]` input, striding `projected` by `projection_dim`
//!   and skipping `intermediate` (`nemotron_h.cuh:153-154`). So
//!   `ElementwiseIn` over-launches it by `N · intermediate` elements' worth
//!   of blocks, every one of which returns on `i >= total` leaving the
//!   output BYTE-IDENTICAL. That is the near miss that a single small
//!   fixture certifies and production does not. [`crate::runtime::Dims`]
//!   carries one `width`; this needs two. A `Dims` field, named and not
//!   built.
//!
//!   **`SplitPacked`'s 2-D grid** was the other half of the original bullet
//!   and remains the right refusal for both: it would leave every row past
//!   the first unvisited.
//!
//!   **`causal_conv1d_prefill<T, SILU>` was the third part of that bullet
//!   and is a row now.** *"No rule opens a grid over a width one block to the
//!   column"* was a report on the vocabulary, and
//!   [`kernels::LaunchRule::PerChannel`] opens exactly that grid —
//!   `runtime::launch::per_channel` cites `ssm/causal_conv1d.cu`'s
//!   `prefill_dispatch` as the launcher it reproduces and names this file's
//!   two `SILU` symbols as the pair it serves. Both halves of the old
//!   refusal have now been answered from opposite directions: the `SILU`
//!   arm by `elem` carrying an argument list, the grid by a rule ported from
//!   this launcher. See [`CAUSAL_CONV1D_SIGS`]`[2]`.
//! * **`nemotron_mamba_ssm_batched_bf16` — TWO live arms, not three, and
//!   both blocked on the same thing.** Re-derived from
//!   `nemotron_h.cu:97-190` rather than from a report. The launcher reads as
//!   four kernel forms and two of them are unreachable: the second is inside
//!   `if constexpr (false)` (`:143`), and the fourth follows an
//!   unconditional `return` inside a bare block (`:181`). What ships is
//!   `mamba_ssm_batched_prefill_reg<<<dim3(R, num_heads,
//!   ceil(head_dim/16)), 512, 2·state_size·4>>>` when `sequence_prefill`,
//!   and `mamba_ssm_batched_warp<<<dim3(R, num_heads), 256,
//!   2·state_size·4>>>` when not.
//!
//!   `Term::Is` on the `sequence_prefill` bool operand does select between
//!   them, as reported. Neither arm has a row for a reason that is not the
//!   predicate: **the dynamic shared memory is `2 · state_size ·
//!   sizeof(float)` and no rule computes it**, because
//!   [`crate::runtime::Dims`] carries no state width. It is FILLABLE —
//!   `DispatchCtx::k_d` is `state_size` on a mamba fire, and
//!   `bind/mod.rs:1088-1092` says so in its own words — so this is a `Dims`
//!   field plus a binder line plus a rule, not a wall. It is named here and
//!   not built because `jit_dims` also fills `q_heads` from
//!   `ctx.num_q_heads`, the ATTENTION head count, where these two grids want
//!   `ctx.v_h`; so the honest cost is TWO `Dims` fields and one rule for two
//!   rows, and the prefill arm needs a 3-D grid and a 512 block besides.
//!   §10.5 says a rule invented for one kernel is a geometry only that
//!   kernel means; this is two kernels and three growths, and it should be
//!   decided deliberately rather than by momentum.
//! * **A block width the kernel is compiled against, disagreeing with the
//!   rule's — RETRACTED for `l2norm_scale`, and the disagreement is what
//!   went away.** `l2norm_scale<T, BLOCK>` is the trap this family is known
//!   for and the arity finding sharpened it rather than removing it: `BLOCK`
//!   is statable, its `__shared__ float buf[BLOCK]` and its
//!   `for (off = BLOCK / 2; ...)` fold both follow the parameter, so
//!   `<bf16, 256>` under [`kernels::LaunchRule::Rms`] would launch, stay in
//!   bounds and return a plausible number. It would also fold 256 partials
//!   where the launcher's `<<<N, 128>>>` folds 128, and this file's own
//!   `g_beta` note records the standard that decides it — a wider block over
//!   a REDUCTION sums to a different last bit, and a row may not buy a rule's
//!   fit with the kernel's arithmetic. The refusal then ended in a sentence
//!   about the vocabulary rather than about the kernel: *"Its ahead-of-time
//!   value is 128 and no rule launches 128 over `dim3(N)`."*
//!   [`kernels::LaunchRule::PerRowNarrow`] launches exactly that —
//!   `runtime::launch::per_row_narrow` is `grid [rows, 1, 1]`,
//!   `block [128, 1, 1]`, `smem 0` — so the row states 128 in the rule AND
//!   128 in `elem`, the two agree by construction, and the fold adds the same
//!   128 partials in the same order the launcher's does. The numerics
//!   objection was never that a block width appears in the algebra; it was
//!   that a rule fixing a DIFFERENT one silently changes the answer. See
//!   [`GATED_DELTA_NET_PREP_SIGS`]`[5]`, which cites
//!   `ssm/gated_delta_net.cu:152-164`.
//! * **A block width derived from an operand that only coincidentally equals
//!   the template's.** `qwen_gdn_qk_norm<T, BLOCK>` is named in
//!   `runtime::launch`'s own refusal: it reduces through
//!   `__shared__ float[BLOCK]` while `PerHeadElementwise` answers
//!   `clamp(K_d, 32, 128)`, which is 128 for every Qwen3.5 config measured
//!   and is not a property of the rule — at `K_d = 64` the fold reads
//!   sixty-four entries nothing wrote. `qwen_gdn_v_g_beta<T, BLOCK>` strides
//!   by the template constant rather than by `blockDim.x`, so the same
//!   disagreement skips channels instead of reading garbage. Both are
//!   additionally unrowable for a reason arity never touched: neither has an
//!   ahead-of-time symbol.
//! * **A host choice at run time.** `causal_conv1d_prefill_batched` and its
//!   `_channel_tile` twin sit behind one symbol, chosen on `requests >= 8`,
//!   and the three `mamba_ssm_batched*` forms behind another, chosen on
//!   `sequence_prefill`. A row states one arm of a decision the host makes;
//!   the `.cuh` headers record both refusals.
//! * **Dynamic shared memory no rule computes.** This is what holds most of
//!   `gated_delta_net`'s fourteen recurrence kernels, and it survived both
//!   the arity finding and the `<cstdint>` fix. Their launchers read
//!   `dim3 grid(B, V_h); dim3 block(128); shmem = 2 * K_d *
//!   sizeof(float)` — `gated_delta_net.cu:227-230`, `:248-251`, `:328-332`
//!   and the `_fla` forms' `dim3 grid_fla(NV, R, V_h)` at `:383-385` —
//!   which [`kernels::LaunchRule::RecurrentScan`] states and five rows now
//!   take. Three OTHER ported rules set a non-zero `smem` and not one of them
//!   computes that expression: `SdpaVector` answers `(rows + 256) · 4`,
//!   `RouterSort` `(3 · E + 34) · 4`, and `Rope` `(head_dim / 2) · 2 · 4`,
//!   which is `K_d · 4` — half of what the scan needs, and the nearest miss
//!   in the file. A row on any of them would hand an
//!   `extern __shared__ float[]` half its staging area and stage `k` over
//!   `v`. The `mamba_ssm_batched*` trio fails identically at
//!   `2 · state_size · sizeof(float)`.
//!
//!   `chunk_gated_delta_prefill_batched_cached` is the one still refused for
//!   a shared allocation, and it is refused for a sharper reason than the
//!   size: it wants `K_d * V_d * sizeof(float)` plus a
//!   `cudaFuncSetAttribute` raising the per-block maximum. `Dims` carries one
//!   head width, the rule that reads it reads the KEY width, and this kernel
//!   needs BOTH widths multiplied. The two warp-tiled prefills escape by
//!   needing neither: they allocate nothing and want only the value width,
//!   which [`kernels::LaunchRule::WarpTiledScan`] takes as
//!   `Dims::width / Dims::kv_heads` — see [`GATED_DELTA_NET_SIGS`]`[5]` and
//!   `[6]`.
//!
//!   Their `KLast` and their fused/legacy arm carry a third refusal worth
//!   naming separately, because it is the one the arity note warns about and
//!   it does NOT look like a host choice from the call site.
//!   `qwen_gdn_k_last_state_enabled()` and `qwen_gdn_fused_step_enabled()`
//!   are `constexpr bool ... { return false; }` at `gated_delta_net.cu:62`
//!   and `:69` — build-time switches, so nvcc folds the four-way dispatch at
//!   `:282-314` down to one `recurrent_step_batched<float, false>` and folds
//!   `+ (fused ? 1 : 0)` out of the `smem` expression. A row could therefore
//!   spell `"device::bf16, device::false_type::value"` and be right TODAY.
//!   It would also be a copy of a constant living in a file the row cannot
//!   see: flip either switch for the benchmark each comment invites and the
//!   archive takes the other arm while the row keeps this one — two launches
//!   under one symbol computing a different recurrence, with nothing
//!   comparing them. A non-type argument is a value the kernel is compiled
//!   AGAINST, and that is what it costs when the value lives on the far side
//!   of the split.
//! * **An extent no shape on the fire produces.**
//!   `build_nemotron_moe_ptrs_decode_batched` is sized `rows · top_k` where
//!   `Elementwise` reads output elements; `build_nemotron_moe_ptrs_aligned`
//!   is sized by a host scalar off a padded expert histogram.
//! * **A `Source` that names a kernel no model text states.**
//!   `qwen_gdn_qk_norm<T, BLOCK>` and `qwen_gdn_v_g_beta<T, BLOCK>` are fired
//!   only from inside `qwen_gdn_post_conv_prep_bf16`, so a row for either
//!   would name a kernel no trace can ask for. `gated_delta_g_beta` was the
//!   same defect with a row on it, and the row is GONE: the launcher is
//!   declared at `gated_delta_net.hpp:345`, defined in the `.cu`, and called
//!   from nowhere, superseded by the `v_g_beta` fusion that computes the same
//!   `g_log` and `beta` in one pass over the post-conv activation.
//! * **A header NVRTC would not open, and now does.**
//!   `ssm/gated_delta_net.cuh` failed at preprocessing —
//!   `gated_delta_net.cuh(73): catastrophic error: could not open source file
//!   "cstdint"` — while the `<cuda_bf16.h>` seven lines above it resolved
//!   against the carried shims. NVRTC ships no standard library at all (§13's
//!   probe: 0 of 31 headers answered), so the include is gone and the file
//!   takes `u8`, `u32`, `i32` and `usize` from the prelude, which are the
//!   compiler's own types and therefore the same types `<cstdint>` was
//!   handing out. **Measured after the change**: the header compiles at
//!   sm_89, and three of its recurrence kernels —
//!   `recurrent_step<bf16, false>`,
//!   `recurrent_step_batched_gqa_fla<bf16, 64, 128>` and
//!   `chunk_gated_delta_prefill_batched_gqa_fla<bf16, 64, 128>` —
//!   instantiated and returned lowered names, for 133 808 bytes of cubin in
//!   1 564 ms. It carried no row and no unit then — not for arity and
//!   not for the include, but for the dynamic `smem` bullet above.
//!   It is a unit now with five rows over three of its fourteen kernels, and
//!   the eleven that remain are held by that same bullet, by a host
//!   `constexpr`, or by an environment variable. What the fix bought is that
//!   the text is KNOWN-GOOD under both compilers
//!   instead of only under the one anybody had run.
//!
//!   That probe also corrected a claim this file used to make, and the
//!   correction has since been corrected in turn — the second version is the
//!   one to trust, because it was measured against every argument form
//!   rather than the one that failed. The multi-argument templates ARE
//!   nameable: `elem` is pasted between angle brackets, so
//!   `"device::bf16, 64, 128"` names a three-parameter template, and slots
//!   2+ take those bare literals because
//!   `DeviceKernel::instantiation` prefixes the string ONCE, at its front. A
//!   NAME in slot 2 would have to be written out from `::` — probed as
//!   `"device::bf16, device::kBlock256"`, refused with *name followed by
//!   "::" must be a class or namespace name*, and accepted the moment the
//!   same constant is spelled `::pie_cuda_driver::kernels::device::\
//!   kBlock256`.
//!
//!   `recurrent_step_batched_gqa_smem<int BV>` was reported unnameable off
//!   that same prefix and it is not. `64` under it is
//!   `expected an identifier` — true, and a fact about BARE tokens rather
//!   than about non-type arguments, because slot 1 has only to RESOLVE under
//!   the kernels root and a value resolves there as readily as a type:
//!   `examples/argform_probe.rs` instantiates `sized<256>` from
//!   `"device::kBlock256"`, and `"device::i32(64)"` is the same qualified
//!   constant expression with the prelude's `i32` alias (`pie_device.cuh:463`)
//!   doing the work. This kernel is spellable and is refused by the dynamic
//!   `smem` bullet above like the other thirteen. The prefix is also why the
//!   probe instantiated these at the prelude's `bf16` rather than at
//!   `__nv_bfloat16`, which lives at global scope and is unreachable from
//!   inside `kernels::`.
//!
//! # The payoff this family did collect
//!
//! Two rows the ahead-of-time build never had: `widen` and `narrow` at
//! `f16`. They are the same two templates the bf16 rows instantiate, and the
//! reason the archive holds only bf16 is that a second instantiation cost a
//! translation unit's worth of `cicc` for a type nobody had asked for yet.
//! Here each costs a row — the same measurement `norm/elementwise` made with
//! `residual_add_f16`, repeated on a family that was chosen for being hard.

use crate::unit::Unit;
use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Lit;
use kernels::Source;
use kernels::kernel;
use kernels::operands;
use crate::device::DeviceKernel;

/// Gated Delta Net's pre-recurrence kernels: the two casts, their fp16 twins,
/// the gate/beta activation, and the GQA head fan-out.
pub const GATED_DELTA_NET_PREP: Unit = Unit {
    name: "ssm/gated_delta_net_prep",
    root: include_str!("../../csrc/src/ssm/gated_delta_net_prep.cuh"),
    rows: GATED_DELTA_NET_PREP_ROWS,
    options: &[],
};

/// Gated Delta Net's recurrence itself — five of the header's fourteen.
///
/// The unit `ssm/gated_delta_net.cuh` said it could never be. Its header
/// carried three refusals — a `(requests, heads)` grid `Dims` could not
/// state, two- and three-argument templates `instantiation()` could not
/// spell, and a `long long` slot stride `Args::bind` answered
/// `ArgError::Unsupported` for — and closed *"ready for the day a rule can
/// say `(rows, heads)` and the binder can marshal an `I64`"*. All three
/// landed together, and this unit is that day's diff.
pub const GATED_DELTA_NET: Unit = Unit {
    name: "ssm/gated_delta_net",
    root: include_str!("../../csrc/src/ssm/gated_delta_net.cuh"),
    rows: GATED_DELTA_NET_ROWS,
    options: &[],
};

/// Nemotron-H's two Mamba parameter preparations, and Zamba's gated norm.
pub const NEMOTRON_H: Unit = Unit {
    name: "ssm/nemotron_h",
    root: include_str!("../../csrc/src/ssm/nemotron_h.cuh"),
    rows: NEMOTRON_H_ROWS,
    options: &[],
};

/// The single-request causal convolution update, and the batched one.
pub const CAUSAL_CONV1D: Unit = Unit {
    name: "ssm/causal_conv1d",
    root: include_str!("../../csrc/src/ssm/causal_conv1d.cuh"),
    rows: CAUSAL_CONV1D_ROWS,
    options: &[],
};

/// Kimi Delta Attention's two elementwise preparations.
///
/// The unit exists because two of the header's four kernels are
/// single-argument templates on a `dim3(tokens, heads)` grid, which is
/// exactly [`kernels::LaunchRule::PerHeadElementwise`]. The other two —
/// `kda_recurrent_step_batched` and `kda_prefill_batched` — are plain
/// `__global__`s carrying the recurrence itself, and compiling the header
/// pulls them in as text either way. That is the unit's real cost and it was
/// measured: 41 392 bytes of cubin in 140 ms on an L40S, for two rows.
pub const KDA: Unit = Unit {
    name: "ssm/kda",
    root: include_str!("../../csrc/src/ssm/kda.cuh"),
    rows: KDA_ROWS,
    options: &[],
};

/// The units `ssm` compiles.
///
/// `ssm/gated_delta_net` is absent, and the reason changed twice. It was a
/// caution, then a MEASUREMENT — a unit for it was declared and run, and
/// NVRTC answered `gated_delta_net.cuh(73): catastrophic error: could not
/// open source file "cstdint"`, with the `<cuda_bf16.h>` at line 66 having
/// resolved, so the worry this file recorded about `__nv_bfloat162` and
/// `__floats2bfloat162_rn` not shimming was wrong. The include is gone now:
/// the header takes its integer names from the prelude, and the same probe
/// answered `OK` — three recurrence kernels instantiated, 133 808 bytes of
/// cubin in 1 564 ms at sm_89.
///
/// What keeps the unit out is what was behind the include all along. None of
/// the fourteen recurrence kernels is stated by any model text — they are
/// reached from launchers the driver calls directly — so none can carry a
/// row, and [`crate::unit::Unit::compile_with`] refuses a unit with no
/// instantiations. The header is still carried in
/// [`crate::source::DEVICE_HEADERS`], so the day a statement for one of them
/// exists the unit costs a line here and no C++ at all. Being able to say
/// that is the whole return on fixing the include: the text is known-good
/// under both compilers rather than under only the one anybody had run.
pub static UNITS: &[Unit] =
    &[GATED_DELTA_NET_PREP, GATED_DELTA_NET, NEMOTRON_H, CAUSAL_CONV1D, KDA];

/// The templates `csrc/src/ssm/gated_delta_net_prep.cuh` holds that a row can
/// state, and the instantiations of them these rows ask for.
static GATED_DELTA_NET_PREP_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GATED_DELTA_NET_PREP_SIGS[0],
        template_path: "ssm::device::widen",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GATED_DELTA_NET_PREP_SIGS[1],
        template_path: "ssm::device::narrow",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &GATED_DELTA_NET_PREP_SIGS[2],
        template_path: "ssm::device::widen",
        elem: "device::f16",
    },
    DeviceKernel {
        sig: &GATED_DELTA_NET_PREP_SIGS[3],
        template_path: "ssm::device::narrow",
        elem: "device::f16",
    },
    // `ssm::device::f32` and NOT `device::f32` -- the prelude names no fp32
    // alias, and this header declares its own beside a comment explaining
    // why a leaf may typedef but must not specialise `Elem`. The unqualified
    // spelling fails at the name-map pragma with `namespace ... has no member
    // "f32"`, before any launch.
    DeviceKernel {
        sig: &GATED_DELTA_NET_PREP_SIGS[4],
        template_path: "ssm::device::repeat_interleave_heads_fp32",
        elem: "ssm::device::f32",
    },
    // TWO template arguments, and the second is the block width the launcher
    // compiles this kernel against: `l2norm_scale<device::bf16, BLOCK>` with
    // `constexpr int BLOCK = 128`. The bare `128` sits in slot 2 and is not
    // prefixed -- `DeviceKernel::instantiation` pastes
    // `::pie_cuda_driver::kernels::` ONCE, at the front of the string -- so
    // this names `l2norm_scale<::pie_cuda_driver::kernels::device::bf16, 128>`
    // and the same `128` the rule launches is the one `buf[BLOCK]` is sized
    // by.
    DeviceKernel {
        sig: &GATED_DELTA_NET_PREP_SIGS[5],
        template_path: "ssm::device::l2norm_scale",
        elem: "device::bf16, 128",
    },
];

/// The contracts, in [`GATED_DELTA_NET_PREP_ROWS`]'s order.
///
/// Each of the first two is its AOT twin in [`crate::table::ssm`] minus the
/// stream: a stream is `cuLaunchKernel`'s sixth PARAMETER, outside the
/// `void**`, so it is not an operand and stating it as one was the shim's
/// requirement rather than the kernel's (§4.2).
#[rustfmt::skip]
static GATED_DELTA_NET_PREP_SIGS: [KernelSig; 6] = [
    // `Elementwise` IS the launcher this replaces, line for line: the `if
    // (n == 0) return;` became `eval`'s refusal of a zero extent, and
    // `(n + 255) / 256` blocks of 256 became the rule's arithmetic. Nothing
    // about the geometry is stated twice.
    //
    // `n` STAYS an operand even though the rule recovers the same number.
    // The rule sizes the GRID; the kernel still needs the bound for its
    // `i < n` guard, because the last block is partial. What changed is
    // where `n` comes from — `Source::OutElements(0)`, the result's own
    // extent, which is exactly what `Elementwise` divides.
    kernel!(gdn_bf16_to_fp32 "ssm::bf16_to_fp32",
        file = Some("ssm/gated_delta_net_prep.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf <- Source::In(0),
            y: F32sMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(gdn_fp32_to_bf16 "ssm::fp32_to_bf16",
        file = Some("ssm/gated_delta_net_prep.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: F32s <- Source::In(0),
            y: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    // The fp16 twins, which have no AOT counterpart and therefore no symbol
    // to share -- named for what they are. They are the same two templates
    // one line above, at a different `Elem`, and the archive has neither:
    // instantiating a second element type cost a translation unit's worth of
    // `cicc` for something nobody had asked for. Here it costs a row.
    kernel!(gdn_f16_to_fp32 "ssm::f16_to_fp32",
        file = Some("ssm/gated_delta_net_prep.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf <- Source::In(0),
            y: F32sMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(gdn_fp32_to_f16 "ssm::fp32_to_f16",
        file = Some("ssm/gated_delta_net_prep.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: F32s <- Source::In(0),
            y: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    // **`gdn_g_beta` was here, and it is gone because the kernel it named
    // is dead.** The row was `ElementwiseRows` over `device::g_beta` and it
    // documented its own doubt: no ahead-of-time twin to mirror, its sources
    // read off the kernel's parameter list rather than off a second statement
    // of the same contract. The doubt was checked and the answer is that
    // nothing calls it. `device::g_beta` is reached by exactly one launcher
    // in the tree, `gated_delta_g_beta`, declared at `gated_delta_net.hpp`
    // line 345 and defined in `gated_delta_net.cu` -- and that launcher is
    // called from nowhere: no `extern` in the Rust FFI, no table row, no
    // hand-written arm, no sibling `.cu`. What SUPERSEDED it is in the same
    // file. `qwen_gdn_post_conv_prep_bf16` computes the same `g_log` and
    // `beta` by firing `qwen_gdn_v_g_beta<bf16, BLOCK>`, which fuses them
    // with the v-projection read so the post-conv activation is walked once
    // instead of twice; the standalone kernel is the unfused predecessor
    // that fusion left behind.
    //
    // A row for an uncallable symbol is worse than no row: `model-compiler`
    // writes a symbol into a trace and `runtime::fire` looks a row up by it,
    // so a row nothing can state is a contract with nothing on the other end
    // -- and it costs a template instantiation in every compile of this unit
    // to say so. The C++ is untouched: deleting the kernel, its launcher and
    // its declaration is a separate change with its own blast radius, and
    // this one only stops claiming the symbol is reachable.
    // The GQA fan-out: `[N, K_h, D]` read, `[N, V_h, D]` written, value head
    // `h_v` taking key head `h_v / repeat`.
    //
    // `GatedRms` -- `grid [rows, kv_heads, 1]`, `block [256, 1, 1]` -- against
    // a launcher of
    //
    // ```text
    // const int block = (D < 128) ? 64 : 128;
    // dim3 grid(N, V_h);
    // device::repeat_interleave_heads_fp32<device::f32><<<grid, block, 0, stream>>>(...);
    // ```
    //
    // The grid is identical, `Dims::kv_heads` being the field whose own
    // documentation names the GDN recurrence's VALUE heads as the axis it
    // carries. The block is 256 against 64 or 128, and the header states the
    // licence for that in terms of this body rather than of the rule: a pure
    // copy behind `h_v >= V_h || d >= D`, no shared memory, no fold, no
    // `__syncthreads`. A surplus lane returns before it addresses anything,
    // and the tail loop `for (dd = d + blockDim.x; dd < D; dd += blockDim.x)`
    // covers every remaining channel exactly once at any width. The same
    // substitution one kernel down, under `l2norm_scale`, writes past a
    // `__shared__ float[BLOCK]` -- which is why the check is of the body and
    // never of the grid alone.
    //
    // **This row exists because the one-template-argument ceiling was not
    // one.** The kernel was a plain `__global__` and was reported unrowable
    // on that ground; it is `template <class T>` now for no reason but to be
    // nameable, `T = f32` is its only instantiation, and its body contains no
    // `Elem<T>` at all -- a copy has nothing to widen. The ahead-of-time
    // build emits what it emitted before.
    //
    // `n` leaves the operand list: the launcher took it to build the grid and
    // the `__global__` declares `(in, out, K_h, V_h, D, repeat)` without it.
    // `repeat` arrives the other way -- the launcher computed `V_h / K_h` on
    // the host, so the row states that division where the host performed it.
    kernel!(repeat_interleave_heads "ssm::repeat_interleave_heads_fp32",
        file = Some("ssm/gated_delta_net_prep.cuh"),
        launch = LaunchRule::GatedRms,
        operands = operands![
            in_: F32s <- Source::In(0),
            out: F32sMut <- Source::ResultOrRegion(0),
            k_h: I32 <- Source::Gdn("k_h"),
            // The repeated head count and the head width -- `OutDim(0, 1)`
            // and `OutDim(0, 2)` on Metal, where a value's dims are the
            // binder's to read. Here they are the GDN context's, which is
            // the same two numbers from the place that computes them.
            v_h: I32 <- Source::Gdn("v_h"),
            d: I32 <- Source::Gdn("v_d"),
            repeat: I32 <- Source::Div(&Source::Gdn("v_h"), &Source::Gdn("k_h")),
        ]),
    // THE ROW THE BLOCK WIDTH USED TO REFUSE.
    //
    // `ssm/gated_delta_net.cu:152-164`:
    //
    // ```text
    // if (N <= 0 || hidden <= 0) return;
    // constexpr int BLOCK = 128;
    // dim3 grid(N);
    // dim3 block(BLOCK);
    // device::l2norm_scale<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
    //     static_cast<const device::bf16*>(x), y, hidden, scale, eps);
    // ```
    //
    // `runtime::launch::per_row_narrow(rows)` answers `grid [rows, 1, 1]`,
    // `block [128, 1, 1]`, `smem 0`. Three numbers, three matches, and `N` is
    // `Dims::rows` because the ahead-of-time twin already states it as
    // `Source::Rows`.
    //
    // **The block width IS the algebra here, and that is why the row can
    // only be written on a rule whose block width is 128.** `BLOCK` is the
    // template argument, `__shared__ float buf[BLOCK]` is sized by it, the
    // strided load is `for (i = tid; i < hidden; i += BLOCK)`, and the fold
    // is `for (off = BLOCK / 2; off > 0; off >>= 1)`. Under `Rms` this row
    // would launch 256 threads into a 128-float array and fold 256 partials
    // where the launcher folds 128 -- in bounds for the strided loops,
    // out of bounds for `buf`, and a different last bit even where it is not.
    // This file's header carried that refusal for as long as no rule launched
    // 128 over `dim3(N)`; `PerRowNarrow` does, its 128 is the same literal
    // the `elem` string states, and the two can no longer drift apart within
    // one row. `repeat_interleave_heads_fp32` one entry above takes a WIDER
    // block than its launcher and is safe because it is a pure copy behind a
    // bounds check with no shared memory and no fold; the check is of the
    // body, and this body fails it at any width but its own.
    //
    // `N` leaves the operand list -- the launcher spent it on `dim3 grid(N)`
    // and the `__global__` declares `(x, y, hidden, scale, eps)` without it.
    // `scale` is `Lit::F32(1.0)`, the twin's spelling: the KDA convention is
    // an unscaled L2 norm, and the parameter exists because the same kernel
    // serves a scaled caller that does not exist in this tree yet.
    kernel!(l2norm_scale_to_f32 "ssm::l2norm_scale_bf16_to_fp32",
        file = Some("ssm/gated_delta_net_prep.cuh"),
        launch = LaunchRule::PerRowNarrow,
        operands = operands![
            x: Buf <- Source::In(0),
            y: F32sMut <- Source::Out(0),
            hidden: I32 <- Source::OutWidth(0),
            scale: F32 <- Source::Lit(Lit::F32(1.0)),
            eps: F32 <- Source::Ctx("eps"),
        ]),
];

/// The three Nemotron-H / Zamba preparations `csrc/src/ssm/nemotron_h.cuh`
/// holds that a row can state.
static NEMOTRON_H_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &NEMOTRON_H_SIGS[0],
        template_path: "ssm::device::prepare_mamba_params",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &NEMOTRON_H_SIGS[1],
        template_path: "ssm::device::prepare_mamba_dt_da",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &NEMOTRON_H_SIGS[2],
        template_path: "ssm::device::zamba_rmsnorm_gated",
        elem: "device::bf16",
    },
    // The fourth row is one ARM of a launcher that is not one kernel. See
    // [`NEMOTRON_H_SIGS`]`[3]` for the launcher, the arm, and the sibling
    // arm that is still refused.
    //
    // `mamba_split` takes no template parameter list at all — it is a plain
    // `__global__` (`nemotron_h.cuh:111`) — so `elem` is
    // [`DeviceKernel::PLAIN`].
    DeviceKernel {
        sig: &NEMOTRON_H_SIGS[3],
        template_path: "ssm::device::mamba_split",
        elem: DeviceKernel::PLAIN,
    },
];

/// The contracts, in [`NEMOTRON_H_ROWS`]'s order.
#[rustfmt::skip]
static NEMOTRON_H_SIGS: [KernelSig; 4] = [
    // Three bf16 tables widened to fp32 once per layer. `ceil(num_heads /
    // 256)` blocks of 256 behind `h >= num_heads` is `Elementwise` exactly.
    //
    // `num_heads` changed SOURCE and not meaning: the AOT row reads it from
    // `Source::Gdn("v_h")`, the GDN context's head count, and the row here
    // reads it from `A`'s own extent -- `A` is `[num_heads]` fp32, so
    // `OutElements(0)` IS `num_heads`. Stating it that way makes the rule
    // and the guard the same number by construction; the `Gdn` spelling made
    // them two numbers that happened to agree.
    kernel!(nemotron_prepare_mamba_params_jit "ssm::nemotron_prepare_mamba_params",
        file = Some("ssm/nemotron_h.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            a_log: Buf <- Source::Weight(0),
            d: Buf <- Source::Weight(1),
            dt_bias: Buf <- Source::Weight(2),
            a: F32sMut <- Source::Out(0),
            d_f32: F32sMut <- Source::Out(1),
            dt_bias_f32: F32sMut <- Source::Out(2),
            num_heads: I32 <- Source::OutElements(0),
        ]),
    // The kernel's first extent is `total`, not `n`. The C++ launcher took
    // `N` and `num_heads` and multiplied them on the host -- so the operand
    // list here is NOT the AOT row's: `total` replaces `n`, and its source is
    // `Source::OutElements(0)` because `dt_out` is `[N, num_heads]` and that
    // product is what the launcher computed. The rule divides the same
    // product by 256, so the grid and the guard cannot disagree.
    //
    // `dt_bias` stays `Source::Aux(3)`. It is a FOREIGN value -- the split's
    // raw table, which the statement does not carry as an arg -- at index 3
    // of the join order `[dt_raw, a, d, dt_bias, dt_pre, da_pre]`. An aux
    // operand does not raise the arity guard, which is right: a row
    // demanding it as an argument would decline every site.
    kernel!(nemotron_prepare_mamba_dt_da_jit "ssm::nemotron_prepare_mamba_dt_da",
        file = Some("ssm/nemotron_h.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            dt: Buf <- Source::In(0),
            a: F32s <- Source::In(1),
            dt_bias: F32s <- Source::Aux(3),
            dt_out: F32sMut <- Source::Out(0),
            da_out: F32sMut <- Source::Out(1),
            total: I32 <- Source::OutElements(0),
            num_heads: I32 <- Source::InWidth(0),
            time_step_min: F32 <- Source::Lit(Lit::F32(0.0)),
        ]),
    // Zamba's gated output RMSNorm: each norm GROUP of a row is scaled by
    // `silu(gate)`, normalised over the group, and multiplied by `weight`.
    //
    // **`GatedRms` states this launcher because this launcher is where the
    // rule came from.** `runtime::launch::gated_rms` cites it by name:
    //
    // ```text
    // constexpr int BLOCK = 256;
    // const int groups = hidden / group_size;
    // dim3 grid(N, groups);
    // device::zamba_rmsnorm_gated<device::bf16><<<grid, BLOCK, 0, stream>>>(...);
    // ```
    //
    // and the rule returns `grid [rows, kv_heads, 1]`, `block [256, 1, 1]`,
    // `smem 0`. Identical in all three, so this row is right by construction
    // rather than by an argument about coverage -- the one row in this family
    // that needs no such argument.
    //
    // The 256 is not negotiable in either direction, which is the rule's own
    // note and is worth repeating where the row is: the kernel declares
    // `__shared__ float buf[256]` STATICALLY and folds it with
    // `for (off = blockDim.x / 2; off > 0; off >>= 1)`. Wider indexes past a
    // static array, which the hardware does not report; narrower normalises
    // by a sum missing the terms the unlaunched lanes held. Finite, plausible,
    // wrong -- and it cannot be reached from here, because the rule's width
    // and the array's are the same literal. `l2norm_scale`'s trap was this
    // same shape with the two literals DIFFERENT, and its row exists now for
    // the same reason this one always did: `PerRowNarrow`'s 128 and the
    // `elem` argument's 128 are one number.
    //
    // **This row is what the arity finding bought this family.** The kernel
    // was a plain `__global__` and was reported blocked on that alone -- its
    // geometry having been the rule's model all along. It is
    // `template <class T>` now, `Elem<T>::to_f32`/`from_f32` at `T = bf16`
    // are the `bf16_to_float`/`float_to_bf16` it called before, and the
    // ahead-of-time launcher instantiates the same specialisation.
    //
    // `n` leaves the operand list -- the `__global__` takes
    // `(x, gate, weight, y, hidden, gate_stride, group_size, eps)` and the
    // launcher held `N` only to build `grid.x`, which is now the rule's.
    // `gate_stride` stays and is NOT `hidden`: the gate may be a window into
    // a wider fused projection, which is why the launcher defaults it rather
    // than assuming it.
    kernel!(zamba_rmsnorm_gated "ssm::zamba_rmsnorm_gated_bf16",
        file = Some("ssm/nemotron_h.cuh"),
        launch = LaunchRule::GatedRms,
        operands = operands![
            x: Buf <- Source::In(0),
            gate: Buf <- Source::In(1),
            weight: Buf <- Source::Weight(0),
            y: BufMut <- Source::Out(0),
            hidden: I32 <- Source::Width(&Source::In(0)),
            gate_stride: I32 <- Source::Width(&Source::In(1)),
            // The rule's `grid.y` is the GROUP count and this is the group
            // WIDTH, so the two are the same division read from its two
            // ends -- `hidden / n_groups` here, `hidden / group_size` in the
            // launcher. Stating it this way makes `grid.y * group_size ==
            // hidden` true by construction; a `Source::Gdn("group_size")`
            // would have made it a coincidence of two context values.
            group_size: I32 <- Source::Div(
                &Source::Width(&Source::In(0)),
                &Source::Gdn("n_groups"),
            ),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // `nemotron_h.cu:33-54`, the `gate != nullptr` arm at `:48-54`:
    //
    //     constexpr int BLOCK = 256;
    //     const int total = N * projection_dim;
    //     const int grid = (total + BLOCK - 1) / BLOCK;
    //     device::mamba_split<<<grid, BLOCK, 0, stream>>>(
    //         projected, gate, conv_in, dt,
    //         projection_dim, intermediate, conv_dim, num_heads, total);
    //
    // # This retires half of this file's "a grid no ported rule computes"
    //
    // That bullet named `mamba_split` and `mamba_split_conv_dt` together and
    // said *"`Elementwise` sizes on the OUTPUT width where the launcher used
    // the input's"*. That was a report on the vocabulary of the day and it
    // has gone stale for the first of the pair: [`kernels::LaunchRule::
    // ElementwiseIn`] sizes on `rows · in_width` — `runtime::launch::
    // elementwise_in` is `ceil(n / 256)` blocks of 256 — and `projected` IS
    // the input rectangle, `[N, projection_dim]`, which the kernel confirms
    // by indexing it `row = i / projection_dim` with no offset
    // (`nemotron_h.cuh:123-124`). So `rows · in_width` is `N ·
    // projection_dim` is `total`, to the digit and not approximately.
    //
    // **The second half of the bullet stands, and for a reason worth
    // stating**: `mamba_split_conv_dt` is `ceil(N · (conv_dim + num_heads) /
    // 256)`. It reads the same `[N, projection_dim]` input — the kernel
    // strides `projected` by `projection_dim` and skips `intermediate`
    // (`nemotron_h.cuh:153-154`) — so `ElementwiseIn` would over-launch it
    // by `N · intermediate` elements' worth of blocks. Those blocks return
    // on `i >= total` and the OUTPUT would be byte-identical, which is
    // precisely the near miss that is invisible at one shape. The extent it
    // needs is `rows · (width(conv_in) + width(dt))` — the sum of TWO
    // results' widths — and `runtime::Dims` carries one `width`. That is a
    // `Dims` field, named and not built.
    //
    // # `total` is an operand and the rule is the grid, and they must agree
    //
    // The kernel guards `if (i >= total) return;` with the same `total` the
    // launcher divided to size the grid. Here the rule computes the grid
    // from `Dims` and whatever fires this row supplies `total`, so the two
    // are two numbers that must be made equal rather than one number read
    // twice — the hazard `families/rope.rs` names against `heads_per_block`,
    // arriving through an element count. A caller that supplies a `total`
    // smaller than `rows · in_width` leaves the tail of the rectangle
    // unwritten; larger, and it reads past `projected`.
    //
    // # No `Source` on any operand
    //
    // `#split` is unspellable by any trace: `model-compiler` writes
    // `ssm::nemotron_mamba_split_bf16`, which is the two-armed launcher and
    // not this kernel. Only an `execution::Step::Fire` that names this
    // symbol, or a test that means to, reaches it, and both supply their own
    // operands. `abi.rs:737` skips a row with an unbound operand, so no
    // dispatch is generated — the correct answer for an arm the dispatcher
    // must not pick on its own.
    //
    // # `gate` is not `| null` here, and that is the whole arm
    //
    // The sibling arm exists *because* `gate` can be null. This row is the
    // branch where it is not, so a null in that slot is the other kernel's
    // job and not a spare cell — the same distinction `norm`'s two
    // `rmsnorm_vec8` rows draw on `y_fp16`.
    kernel!(nemotron_mamba_split "ssm::nemotron_mamba_split_bf16#split",
        file = Some("ssm/nemotron_h.cuh"),
        launch = LaunchRule::ElementwiseIn,
        operands = operands![
            projected: Buf,
            gate: BufMut,
            conv_in: BufMut,
            dt: BufMut,
            projection_dim: I32,
            intermediate: I32,
            conv_dim: I32,
            num_heads: I32,
            total: I32,
        ]),
];

/// The one template in `csrc/src/ssm/causal_conv1d.cuh` a row still states.
///
/// The header holds four, and this unit was down to three of them before
/// §28.4 took two more. What went is instructive, because the argument that
/// removed them is the argument this header had already made about a THIRD
/// kernel and then failed to apply to its own rows.
///
/// `causal_conv1d_prefill_noact_bf16` never had a row, on the grounds that a
/// row for it would be *"a contract naming a caller that does not exist and a
/// symbol sitting in `migration_status`' denominator forever"* — the conformer
/// loop fires `ssm::device::causal_conv1d_prefill<T, false>` directly, so the
/// host launcher is referenced by no C++ and named by no row, an EMPTY
/// consumer set. §28.9 then measured `ssm::causal_conv1d_prefill_bf16` and
/// `ssm::causal_conv1d_update_bf16` and found the same thing one step out:
/// their `dsl.rs` wrappers had zero call sites, no golden named either, and
/// the `_batched` twins carry every fire (6 goldens for the update). The
/// difference is that these two DID sit in the denominator, for the whole of
/// the migration. Both rows are gone; `causal_conv1d_prefill<T, true>` is no
/// longer instantiated by this unit and `PerChannel` — which landed ported
/// from that very launcher — is still the rule the `.cu` fires under.
///
/// `causal_conv1d_prefill_batched` is chosen against `_channel_tile` by the
/// HOST on `requests >= 8` — two grids behind one symbol, and the `.cuh`'s own
/// header records the decision that a row for either states half a contract.
static CAUSAL_CONV1D_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &CAUSAL_CONV1D_SIGS[0],
        template_path: "ssm::device::causal_conv1d_update_batched",
        elem: "device::bf16",
    },
];

/// The contracts, in [`CAUSAL_CONV1D_ROWS`]'s order.
#[rustfmt::skip]
static CAUSAL_CONV1D_SIGS: [KernelSig; 1] = [
    // The batched twin: R requests advancing R conv windows in one launch,
    // each window found through `slot_ids[r]` in a paged arena.
    //
    // **`SplitPacked` and not `ElementwiseRows`, and the difference is a
    // transpose.** The launcher is
    //
    // ```text
    // dim3 grid((C + BLOCK - 1) / BLOCK, R);   // BLOCK == 128
    // dim3 block(BLOCK);
    // ```
    //
    // and the kernel reads `r = blockIdx.y; c = blockIdx.x * blockDim.x +
    // threadIdx.x`. `SplitPacked` computes `grid [ceil(in_width / 256), rows,
    // 1]`, `block [256, 1, 1]` — the channel tile on `grid.x`, the row on
    // `grid.y`, the same axis assignment the kernel indexes with.
    // `ElementwiseRows` puts the row on `grid.x`; firing this kernel under it
    // would make `r` the channel tile and `c` the request, and with `R` and
    // `C` both nonzero the guard passes, `slot_ids[c]` reads past the request
    // table, and the fire writes a conv state at an address derived from
    // whatever was there. This row is the one place in `ssm` where two rules
    // of the same arithmetic differ only in which axis is which, and it was
    // decided by reading the kernel's two index lines rather than the
    // launcher's grid shape.
    //
    // The rule's 256 against the launcher's 128 is the coverage argument the
    // rows above make: `c` is bounded by `c >= C`, every channel is reached
    // by exactly one thread under either cut, and each thread touches only
    // its own column of `x`, `y` and `state` — the shift loop included, since
    // it moves `state[(k+1)*C + c]` into `state[k*C + c]` at fixed `c`. No
    // shared memory, no cross-lane fold, so a wider block changes no bit.
    //
    // `in_width` is `C` here because `x` and `y` are both `[R, C]`: this
    // kernel takes a packed buffer apart along K rather than along the width,
    // so the rule's input and output widths coincide and its "wider than the
    // launcher" licence is not even exercised.
    //
    // **`slot_stride_elems` is why this row could not exist before.** It is
    // `long long` deliberately — `K * C` elements into an arena of many
    // gigabytes, where `i32` overflows at 2^31 elements and produces a
    // negative offset that lands inside another request's state. `Args::bind`
    // now marshals `Ty::I64` through its own `ArgValue`, kept distinct from
    // `Usize` because a signed stride and an unsigned size are not the same
    // claim.
    //
    // `whole` for the reason the single-request row states, sharpened: the
    // permutation from row to slot is `slot_ids`, computed over the whole
    // fire, and a row window would advance the conv windows of requests it
    // was not given.
    kernel!(gdn_conv_update "ssm::causal_conv1d_update_batched_bf16",
        file = Some("ssm/causal_conv1d.cuh"),
        launch = LaunchRule::SplitPacked,
        whole = true,
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            // The checkpoint may ship none, which is a fact about the
            // checkpoint rather than drift -- so null, not a refusal.
            bias: Buf <- Source::WeightSuffix("_bias"),
            // The STATEMENT'S layer's conv window. Absent three ways (no
            // GDN context, no layer stated, no slab there) and all three
            // decline the branch.
            state_base: BufMut <- Source::GdnSlab("conv_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("conv_stride_elems"),
            y: BufMut <- Source::Out(0),
            r: I32 <- Source::Rows,
            c: I32 <- Source::Gdn("conv_dim"),
            k: I32 <- Source::Gdn("conv_k"),
        ]),
];

/// The two templates in `csrc/src/ssm/kda.cuh` a row can state.
///
/// The header's other two, `kda_recurrent_step_batched` and
/// `kda_prefill_batched`, are plain `__global__`s with no template parameter
/// at all — NVRTC answers `type name is not allowed` at the name-map pragma
/// for `path<bf16>` over a non-template, so a row for either fails at compile
/// rather than at fire. They are Kimi Delta Attention's actual recurrence and
/// they are the two this unit compiles and cannot reach.
static KDA_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &KDA_SIGS[0],
        template_path: "ssm::device::kda_gate_beta",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &KDA_SIGS[1],
        template_path: "ssm::device::kda_o_norm_gated",
        elem: "device::bf16",
    },
];

/// The contracts, in [`KDA_ROWS`]'s order.
#[rustfmt::skip]
static KDA_SIGS: [KernelSig; 2] = [
    // The per-head gate and beta: `softplus(a_log · f) · -1` and a sigmoid,
    // one thread per (token, head, channel).
    //
    // `PerHeadElementwise` — the launcher is
    //
    // ```text
    // dim3 grid(T, H);
    // const int threads = D < 256 ? D : 256;
    // device::kda_gate_beta<device::bf16><<<grid, threads, 0, stream>>>(...);
    // ```
    //
    // and the rule computes `grid [rows, q_heads, 1]`, `block
    // [clamp(head_dim, 32, 128), 1, 1]`. The grid is identical. The block is
    // identical at the head dimension Kimi K3 ships, 128, and NARROWER above
    // it — where the kernel's `for (d = threadIdx.x; d < D; d += blockDim.x)`
    // simply makes more passes. Below 32 the clamp makes it WIDER than the
    // launcher, and that direction was checked rather than assumed: the
    // surplus lanes have `threadIdx.x >= D`, enter no loop, and this kernel
    // declares no shared array for them to touch. A sub-warp block would in
    // any case be the launcher's bug and not the rule's — every
    // `__shfl_*_sync` in this file passes a full `0xffffffff` mask.
    //
    // The head axis is `grid.y` and the token axis is `grid.x`, matching the
    // kernel's `t = blockIdx.x; h = blockIdx.y` — the transpose of
    // `PerHead`, which the same vocabulary also offers and which would make
    // `t` the head. The guard `if (t >= T || h >= H) return;` would pass
    // under the transpose whenever `T` and `H` both exceed the other, so the
    // wrong rule here is silent.
    //
    // `t` STAYS an operand even though the rule recovers it, because the
    // `__global__` declares it and reads it in that guard — the same reading
    // `gdn_bf16_to_fp32` makes of its `n`. The list below is the AOT row's,
    // verbatim and in order, minus the stream.
    kernel!(kda_gate_beta "ssm::kda_gate_beta_bf16",
        file = Some("ssm/kda.cuh"),
        launch = LaunchRule::PerHeadElementwise,
        operands = operands![
            raw_g: Buf <- Source::In(0),
            raw_beta: Buf <- Source::In(1),
            // fp32 in a bf16 checkpoint, as GDN's `g_beta` records: HF stores
            // `A_log` that way and the fast path expects it. `dt_bias` is
            // fp32 too here, unlike GDN's -- Kimi's split widened both.
            a_log: F32s <- Source::Weight(0),
            dt_bias: F32s <- Source::Weight(1),
            gate_out: F32sMut <- Source::Out(0),
            beta_out: F32sMut <- Source::Out(1),
            t: I32 <- Source::Rows,
            h: I32 <- Source::OutWidth(1),
            d: I32 <- Source::Param(0),
            lower_bound: F32 <- Source::Lit(Lit::F32(0.0)),
        ]),
    // The output RMSNorm with its per-head gate, fused: the head's row is
    // normalised, scaled by `weight`, and multiplied by `silu(g)`.
    //
    // `PerHeadElementwise` on the same `dim3 grid(T, H)` at the same
    // `D < 256 ? D : 256`, and the same reading of the rule. This one carries
    // NO guard on `t` or `h` at all — it indexes `(t * H + h) * D` straight
    // off `blockIdx` — so a rule that over-launched either axis would write
    // past the tensor rather than return. `PerHeadElementwise` produces
    // exactly `[rows, q_heads]` and over-launches neither.
    //
    // **The narrower block is safe here even though this kernel REDUCES**,
    // which is the objection the coverage argument does not answer for
    // folds. It holds because the fold is already order-independent by
    // construction: the sum is a `__shfl_down_sync` within each warp
    // followed by one `atomicAdd(&ssum, acc)` per warp leader, so the order
    // warps arrive in is the runtime's choice at a FIXED block width too.
    // Changing 256 to 128 changes which nondeterministic sum is produced,
    // not whether one is — and the same statement launched twice at one
    // width already does that. A kernel whose fold were a tree over
    // `__shared__` would not pass this check, and `l2norm_scale` is exactly
    // that kernel.
    //
    // `t` is the one operand this row DROPS from its AOT twin, and the drop
    // is the kernel's rather than a judgement: the host function takes `T` to
    // build `dim3 grid(T, H)` and passes `(o, g, weight, out, H, D, eps)` —
    // no `T` reaches the device, because this kernel has no guard to spend it
    // on. `kda_gate_beta` next door keeps its `T` for exactly the opposite
    // reason. A row that carried `t` anyway would push every later cell one
    // slot along a `void**` nothing validates, and `eps` would arrive as `H`.
    kernel!(kda_o_norm_gated "ssm::kda_o_norm_gated_bf16",
        file = Some("ssm/kda.cuh"),
        launch = LaunchRule::PerHeadElementwise,
        operands = operands![
            o: F32s <- Source::In(0),
            g: Buf <- Source::In(1),
            weight: F32s <- Source::Weight(0),
            out: BufMut <- Source::Out(0),
            h: I32 <- Source::Param(0),
            d: I32 <- Source::Param(1),
            eps: F32 <- Source::Ctx("eps"),
        ]),
];

/// [`GATED_DELTA_NET`]'s instantiations — five of the header's fourteen.
///
/// Every one spells TWO template arguments, and both matter.
///
/// **Slot 1 is the state's element type**, and it is prefixed with
/// `::pie_cuda_driver::kernels::` before NVRTC sees it, so `float` and
/// `__nv_bfloat16` — the two types `gated_delta_net.cu` instantiates — cannot
/// be written as themselves: they live at global scope. `gated_delta_net.cuh`
/// declares `ssm::device::f32` and `ssm::device::state_bf16` for exactly this,
/// and the second alias names `__nv_bfloat16` rather than the prelude's
/// `device::bf16` because `state_load`/`state_store` are template-SPECIALISED
/// on `__nv_bfloat16`. A row spelling `device::bf16` would resolve, compile,
/// launch, and take the primary template's `static_cast` path on every state
/// element — a different rounding on the same two bytes, reported by nothing.
///
/// **Slot 2 is `KLast`, and it is `false` because the launcher's is.**
/// `gated_delta_net.cu` guards both arms with
/// `if (qwen_gdn_k_last_state_enabled())`, a file-scope
/// `constexpr bool ... { return false; }`, so exactly one instantiation is
/// emitted into the archive and it is the `false` one. This is the same kind
/// of number as `norm/dsv4_hc.cu`'s `constexpr int BLOCK = 256`, which
/// [`crate::families::norm`]'s rows spell for the same reason: a value the
/// kernel is COMPILED AGAINST, written where a reader changing it will see
/// what depends on it. `KLast` transposes the state slab's two axes; a row
/// that guessed it would address `[K_d, V_d]` as `[V_d, K_d]` and produce a
/// finite, wrong recurrence.
///
/// The GQA row's `KLast` is `false` for a second, sharper reason: the
/// launcher's fast paths for that shape are gated on
/// `!qwen_gdn_k_last_state_enabled()`, so `true` is not merely unshipped
/// there, it is unreachable.
static GATED_DELTA_NET_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GATED_DELTA_NET_SIGS[0],
        template_path: "ssm::device::recurrent_step_batched",
        elem: "ssm::device::f32, false",
    },
    DeviceKernel {
        sig: &GATED_DELTA_NET_SIGS[1],
        template_path: "ssm::device::recurrent_step_batched",
        elem: "ssm::device::state_bf16, false",
    },
    DeviceKernel {
        sig: &GATED_DELTA_NET_SIGS[2],
        template_path: "ssm::device::recurrent_step_batched_gqa",
        elem: "ssm::device::f32, false",
    },
    DeviceKernel {
        sig: &GATED_DELTA_NET_SIGS[3],
        template_path: "ssm::device::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
        elem: "ssm::device::f32, false",
    },
    DeviceKernel {
        sig: &GATED_DELTA_NET_SIGS[4],
        template_path: "ssm::device::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
        elem: "ssm::device::state_bf16, false",
    },
];

/// The contracts, in [`GATED_DELTA_NET_ROWS`]' order.
///
/// The first five are [`kernels::LaunchRule::RecurrentScan`] and all five
/// cite the same three lines of `ssm/gated_delta_net.cu`, repeated verbatim
/// in each of the five launchers:
///
/// ```text
/// constexpr int BLOCK = 128;
/// dim3 grid(R, V_h);
/// dim3 block(BLOCK);
/// const int shmem_bytes = 2 * K_d * sizeof(float);
/// ```
///
/// `runtime::launch::recurrent_scan` evaluates `grid [rows, kv_heads, 1]`,
/// `block [128, 1, 1]`, `smem 2 * head_dim * 4`. The three names map onto the
/// launcher's as `rows = R`, `kv_heads = V_h`, `head_dim = K_d`, and
/// [`crate::runtime::launch::Dims::head_dim`] says in its own words that the
/// KEY width is what this rule reads — *"every launcher whose shared memory
/// is `2 * K_d * sizeof(float)` reads the KEY width, which is what
/// `Rule::RecurrentScan` states this field to be"*.
///
/// **`V_d` is the extent that must NOT come from the rule.** It is the value
/// head's width, it is not `head_dim`, and it stays an operand on every row
/// below. A fire that let a rule supply it would walk the state slab's rows
/// at the key pitch: legal addresses, wrong cells, no fault. The rule reads
/// `K_d` because the SHARED ALLOCATION is `2 * K_d` floats — `sq` and `sk`,
/// the two key-width staging buffers — and the kernel reads `V_d` because the
/// state it strides is `[K_d, V_d]`.
///
/// **`V_h` is `Dims::kv_heads` and not `q_heads`**, on the GQA row above all:
/// the grid is opened over the VALUE heads and `K_h` is an operand the kernel
/// divides by. Handing the rule `q_heads` on a grouped fire opens the grid
/// over the key heads — a quarter of the blocks on Qwen3.5's 4:1 — and the
/// value heads past the first group are never stepped, so their recurrence
/// silently stops advancing.
///
/// **The last two rows are [`kernels::LaunchRule::WarpTiledScan`] and are
/// the file's first three-axis grid.** They share the two leading axes with
/// the five above and part company on everything else: `smem 0` rather than
/// `2 · K_d · sizeof(float)`, `ceil(V_d / 4)` on a third axis, and a block
/// of four warps rather than the scan's 128 threads read as one. `V_d`
/// reaches the rule as `Dims::width / Dims::kv_heads` and NOT as
/// `Dims::head_dim`, which `RecurrentScan` reads as the KEY width; the two
/// are equal in every Qwen3.5 GDN config measured and are not the same
/// number, which is why the rule asks for the quotient it can name rather
/// than the field it would have to trust.
#[rustfmt::skip]
static GATED_DELTA_NET_SIGS: [KernelSig; 5] = [
    // THE BATCHED STEP, and the row that `Ty::I64` unblocked.
    //
    // `slot_stride_elems` is a `long long` on purpose: it is an element count
    // into a multi-gigabyte state arena, where `i32` overflows at 2^31 and
    // yields a NEGATIVE offset that lands inside another request's state.
    // `gated_delta_net.cuh`'s header named `ArgError::Unsupported` on
    // `Ty::I64` as the third of its three refusals; the binder marshals it
    // now, through an `ArgValue` kept distinct from `Usize` because a signed
    // stride and an unsigned size are not the same claim.
    //
    // `r` is `Attn("num_requests")` and not `Rows` -- the twin's spelling,
    // kept. `Rule::RecurrentScan` opens `grid.x` over `Dims::rows`, so the
    // fire's rectangle and this operand must be the same number; they are the
    // same number because a decode step's rectangle IS its requests. The
    // operand stays because the kernel does not read `gridDim.x` -- it reads
    // `slot_ids[r]` at `r = blockIdx.x`, and `r` here bounds nothing the grid
    // does not already bound. It is the twin's, unchanged, and a row may not
    // drop a `__global__`'s parameter: `cuLaunchKernel` reads a missing
    // argument from whatever follows the array.
    kernel!(gdn_step "ssm::recurrent_gated_delta_step_batched",
        file = Some("ssm/gated_delta_net.cuh"),
        launch = LaunchRule::RecurrentScan,
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
        ]),
    // The batched step over a bf16 slab: production Qwen3.5's decode path.
    kernel!(gdn_step_state_bf16 "ssm::recurrent_gated_delta_step_batched_state_bf16",
        file = Some("ssm/gated_delta_net.cuh"),
        launch = LaunchRule::RecurrentScan,
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: BufMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
        ]),
    // THE GROUPED-QUERY STEP, whose fp32 launcher is the only one of the four
    // GQA entry points with a single arm.
    //
    // Its `<<<>>>` is the other four's, and the two fast paths in front of it
    // are both dead code: `qwen_gdn_fla_step_enabled()` is a file-scope
    // `constexpr bool ... { return false; }`, and the `fused` branch is
    // `qwen_gdn_fused_step_enabled() && K_d <= 256` with the first conjunct
    // another constexpr `false`. So `shmem_bytes` folds to `2 * K_d *
    // sizeof(float)` -- the rule's -- rather than to the `+ 1` float the
    // fused arm would have wanted.
    //
    // **Its bf16 twin is NOT a row yet, and two things are missing.**
    // `recurrent_gated_delta_step_batched_gqa_state_bf16` routes to
    // `recurrent_step_batched_gqa_smem<128>` -- a three-axis grid at a
    // staged state slab -- on `V_d == 128 && K_d == 128`.
    //
    // Until 2025-06 the predicate was `std::getenv`, and the refusal written
    // here was that an environment variable is not a value any fire holds:
    // `crate::device::Fact` is `Address | Int | Opaque` precisely so a
    // predicate over one is unspellable rather than merely discouraged. That
    // refusal was right and is now moot, because §30 did the measurement the
    // refusal skipped. The two arms are **byte-identical at eight shapes**,
    // on the state slab and on `out`, with `written > 0` on every one -- so
    // the variable was never choosing an operation, and the honest fix was
    // to delete it rather than to find it a home in this vocabulary. That is
    // the cheaper outcome and it is worth naming: a knob that survives
    // measurement needs a `Choose`; a knob that does not needs an editor.
    //
    // What is left is a real refusal and a smaller one. (a) The predicate is
    // an INTEGER COMPARISON on two operands, and `Term` has no `IntIs` --
    // §26.10(b). (b) The arm it selects has no row and no `LaunchRule`: it
    // opens `grid((V_d + 127) / 128, R, V_h)` on `K_d * 128 *
    // sizeof(__nv_bfloat16) + 2 * K_d * sizeof(float)`, which `RecurrentScan`
    // matches in neither axis count nor smem. So this is a `Choose` over TWO
    // rows once both exist -- not a `Specialisation`, because `agrees()`
    // refuses an arm whose `LaunchRule` differs from the base's, and these
    // two differ in exactly that. The launcher stays until then, and it is
    // the only one of the five decode steps that does.
    //
    // `K_h` stays an operand and `V_h` is the rule's. The kernel computes
    // `repeat = V_h / K_h` and `h_k = h / repeat` from `blockIdx.y`, so the
    // grid axis is the VALUE head count and the key head count is a divisor
    // it reads. Reversed, a 4:1 fire opens a quarter of the blocks it needs
    // and three of every four value heads stop advancing -- a recurrence that
    // silently freezes, which is the failure this row's operand order exists
    // to prevent.
    kernel!(gdn_step_gqa "ssm::recurrent_gated_delta_step_batched_gqa",
        file = Some("ssm/gated_delta_net.cuh"),
        launch = LaunchRule::RecurrentScan,
        operands = operands![
            q_norm_kh: F32s <- Source::In(0),
            k_norm_kh: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
        ]),
    // THE WARP-TILED GQA PREFILL, and the third grid axis this file said no
    // rule produced.
    //
    // `ssm/gated_delta_net.cu:792-829`, the shipped arm:
    //
    // ```text
    // constexpr int WARPS = 4;
    // constexpr int BLOCK = WARPS * 32;
    // dim3 grid(R, V_h, (V_d + WARPS - 1) / WARPS);
    // dim3 block(BLOCK);
    // device::chunk_gated_delta_prefill_batched_warp_tiled_gqa<float, false>
    //     <<<grid, block, 0, stream>>>(...);
    // ```
    //
    // `runtime::launch::warp_tiled_scan(rows, heads, value_width)` answers
    // `grid [rows, heads, ceil(value_width / 4)]`, `block [128, 1, 1]`,
    // `smem 0`. `grid.x` is the REQUEST count -- these launchers read
    // `qo_indptr` and each block walks one request's tokens -- so the
    // rectangle this row states is counted in requests, which is one of the
    // three things `Dims::rows` says it may be.
    //
    // `R` LEAVES the operand list where `gdn_step`'s stayed, and that is the
    // kernel's decision rather than a judgement of ours: the launcher takes
    // `int R` to build `dim3 grid(R, ...)`, and `gated_delta_net.cuh:411-424`
    // declares sixteen parameters without it, because the kernel reads
    // `const int r = blockIdx.x;` and has no guard to spend the operand on.
    // The batched STEP next door keeps its `r` for the opposite reason.
    //
    // **`smem 0` is the tell that this is a rule and not `RecurrentScan`
    // with a parameter.** The scan's block reads `2 * K_d` floats it must be
    // GIVEN; a warp-tiled block holds its slice of the state in REGISTERS and
    // stages nothing. A row that inherited `2 * K_d * sizeof(float)` here
    // would pay for an allocation the kernel never addresses -- harmless
    // until the day `K_d` is large enough that the extra request is refused
    // and the launch fails on a kernel that wanted no shared memory at all.
    //
    // **`V_d` is not `K_d`.** `warp_tiled_scan` takes the value width as
    // `Dims::width / Dims::kv_heads` because the output row of this launcher
    // is `V_h * V_d` wide; `RecurrentScan` reads `Dims::head_dim` as `K_d`
    // because its shared allocation is `2 * K_d` floats. `eval` checks
    // `width % kv_heads == 0` before dividing and refuses `Ungeometric::Empty`
    // otherwise. `v_d` still crosses as an operand -- the kernel guards
    // `v_idx >= V_d` with it, and the third grid axis is rounded UP to a
    // multiple of four, so the last tile's surplus warps have nothing but
    // that operand to stop them.
    //
    // **The two fast paths in front of this one are dead code, and the
    // second is why `KLast` is `false` rather than merely unshipped.**
    // `qwen_gdn_gqa_ilp2_enabled()` is a file-scope
    // `constexpr bool ... { return false; }` at `gated_delta_net.cu:62`, so
    // the `_ilp2` kernel and its `dim3 grid(R, V_h, ceil(V_d / 8))` are never
    // launched -- `runtime::launch::warp_tiled_scan`'s own doc names that
    // grid as the one it does NOT serve, and stating it would be a rule for
    // a launch no build makes. `qwen_gdn_k_last_state_enabled()` is the same
    // at `:50`, so `k_last` folds to `false` and exactly one instantiation
    // of this template reaches the archive.
    //
    // The launcher's `if (K_d > 256 || V_h % K_h != 0) throw` does not appear
    // here. It is a GQA precondition on values the host holds, not a
    // geometry: a fire that violates it has stated a shape the kernel does
    // not implement, which is a row's business rather than a grid's.
    kernel!(gdn_prefill_warp_tiled_gqa
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
        file = Some("ssm/gated_delta_net.cuh"),
        launch = LaunchRule::WarpTiledScan,
        operands = operands![
            q_norm_kh: F32s <- Source::In(0),
            k_norm_kh: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            // The prefill's two state-writing operands, and the twin's
            // spellings kept. `write_state` is a `Fact::Bool` the GDN context
            // holds; the mask is a per-request opt-out the driver does not
            // build yet, and a null is what the kernel reads as "every
            // request writes".
            write_state: Bool <- Source::Gdn("write_state"),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    // The same launch over a bf16 state slab: `gated_delta_net.cu:849-890`,
    // the same five lines over `__nv_bfloat16`.
    //
    // `elem`'s first argument is `ssm::device::state_bf16` and NOT
    // `device::bf16`, for the reason this table's header gives: the alias
    // names `__nv_bfloat16`, which is what `state_load`/`state_store` are
    // template-SPECIALISED on. A row spelling `device::bf16` would resolve,
    // compile, launch, and take the primary template's `static_cast` path on
    // every state element -- a different rounding on the same two bytes,
    // reported by nothing.
    kernel!(gdn_prefill_warp_tiled_gqa_state_bf16
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
        file = Some("ssm/gated_delta_net.cuh"),
        launch = LaunchRule::WarpTiledScan,
        operands = operands![
            q_norm_kh: F32s <- Source::In(0),
            k_norm_kh: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: BufMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            write_state: Bool <- Source::Gdn("write_state"),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
];
