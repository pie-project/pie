//! Launchers the DRIVER reaches for directly — and the one family in this
//! tree that is a `fn` and NOTHING else.
//!
//! The rows this module replaces stood in `table::driver_internal`, whose
//! header stated their defining property:
//!
//! > Launchers the DRIVER reaches for directly — no DSL statement, no place
//! > in the planner's vocabulary, and deliberately not rows of
//! > [`super::super::table::KERNELS`]: `model`'s `kernels_table` holds that
//! > table and `dsl::cuda` to the same set, and these have no statement a
//! > trace could record.
//!
//! # The fourth arrangement
//!
//! `x::SIGS`' doc names three ways a declaration drops its shim. This module
//! is a fourth, and it is worth saying out loud because it is the only one
//! that declares nothing at all:
//!
//! | arrangement | `unit!` | `contract!` | `bind!` | in `FAMILIES` | in `SIGS` |
//! |---|---|---|---|---|---|
//! | a ported kernel family — `rope`, `layout`, `sample` | yes | yes | yes | yes | yes |
//! | a row with no host program yet | yes | yes | `none:` | yes | yes |
//! | a driver op — `adapter`, and all of `gemm` | maybe | yes | **no** | **no** | yes |
//! | **this module** | **no** | **no** | **no** | **no** | **no** |
//!
//! The first three are `x::SIGS`' own "three registration shapes". The
//! fourth is this one, and the difference between it and the third is the
//! only one that matters: a driver op still has a *reading* consumer —
//! `model-compiler` reads its row and must not be able to tell what serves
//! it — and these six have none at all.
//!
//! `x/mod.rs` gets exactly one line for this family — `pub mod
//! driver_internal;` — and no `FAMILIES` element and no `SIGS` element,
//! because there is no `ENTRIES` and no `SIGS` to name. That is not an
//! omission. It is the placement rule applied at the granularity of a whole
//! family:
//!
//! > **Data only for what has a reading consumer. Everything that is only
//! > executed is code.**
//!
//! `contract!` exists so that `model-compiler` — which is GPU-free and must
//! not be able to tell a cuBLAS symbol from a JIT'd one — can read what a
//! trace may say. Nothing here is a thing a trace may say. A
//! `driver_internal` row was never in `table::TABLES`, so `table::sig` could
//! not see it, `dsl::cuda` could not generate a wrapper for it, and
//! `execution::RUST_SERVED` refused to admit it — the table's own header
//! recorded all three, and recorded that the only close available to such a
//! row was DELETION. Fn-world is the first arrangement in which that is not
//! a loss: the launcher survives as the function it always was.
//!
//! # What a fn without a `contract!` cannot do, stated plainly
//!
//! No `contract!` means no `Entry`, which means no `x::route` arm, which
//! means the generated dispatch cannot reach these six by symbol. **Every
//! caller here is a direct Rust call**, and after step 4 that needed
//! checking rather than asserting, because `Route::Unknown` refuses a model
//! at LOAD and a silent misclassification here would refuse every Qwen and
//! Gemma deployment. It checks out, and the reason is worth writing down.
//!
//! `model-compiler/src/lower.rs::semantic()` does name a kernel for each of
//! the ops these six serve — `AddBias` → `:1506`, `GdnPrep` → `:1518`,
//! `RmsnormGated` → `:1519`, `SplitQGate` → `:1520`, `SigmoidGateMul` →
//! `:1521`, `SplitQkv` → `:1545-1548` — but **the symbols it names are the
//! DEVICE rows, not these launchers**: `norm::add_bias_bf16`,
//! `ssm::qwen_gdn_post_conv_prep_bf16`, `norm::rmsnorm_gated_fp32_in_bf16`,
//! `layout::split_q_gate_bf16`, `mlp::sigmoid_gate_inplace_bf16`,
//! `attn::split_qkv_bf16`. Every one of those is a symbol the functions below
//! fire.
//!
//! **They are UNIT rows and not TABLE rows, and that distinction is the whole
//! of the answer.** `families/attn.rs`, `families/norm.rs`, `families/mlp.rs`
//! and their siblings declare the device text, so `unit::unit_of` answers
//! `Some` for all six and `x::fire::fire` — which resolves against the JIT's
//! rows — finds every one. That resolution is GLOBAL and does not consult
//! the module the call was written in, which is why the calls below are
//! typed rather than hand-built; see the note under *Cross-family launches*. `table::sig` answers `None` for all six, exactly
//! as it did before this port, because the launcher rows that carried them
//! were `table::driver_internal`'s and `table::driver_internal` was never in
//! `TABLES`. So `x::route` would answer `Route::Unknown` for any of the six
//! — and never does, because `resolve()` only ever sees `lowered.kernels`.
//!
//! `lowered.kernels` is the other half of the check, and it is narrower than
//! `semantic()`: it is written in one place, `lower.rs:1095-1096`, from the
//! launch-emitting path. A `driver_internal` symbol was never emitted there
//! — it was never in `table::TABLES`, so `check_plan` would have refused it
//! — which is the same fact the deleted table's header recorded from the
//! other side. So `resolve()` never sees one, and omitting `contract!` costs
//! these six nothing at load. It costs them the ability to BE named by a
//! trace, which is the whole point.
//!
//! # No `unit!` either, and that one is not a choice
//!
//! Five of these six fire device text that belongs to somebody else's root:
//! `attn/split_packed.cuh`, `norm/add_bias.cuh`, `layout/deinterleave.cuh`,
//! `mlp/swiglu.cuh`, `norm/rmsnorm.cuh`. A second `unit!` naming the same
//! text would be a second compilation of it under a second unit name, and
//! `unit_of` would then answer with whichever won. The rows stay where the
//! device text is — in `families::{attn,norm,layout,mlp}` today, and in
//! `x::{attn,norm,layout,mlp}` now that those families have landed — and
//! these functions fire them BY SYMBOL, which is the same resolution order
//! every other host program uses.
//!
//! # Cross-family launches
//!
//! **Every launch in this file is cross-family.** That is not incidental —
//! it is what a `driver_internal` fn IS: a host program for device text that
//! belongs to somebody else's `unit!`. So this file is the one most exposed
//! to the mechanism, and the mechanism is worth stating once here rather
//! than seven times below.
//!
//! A `raw::` stub **is not bound to the unit it was declared beside.** The
//! `unit!` expansion takes `symbol`, `launch`, the stub's typed parameters
//! and `stream`, and calls [`crate::x::fire::fire`], which resolves
//! `unit::unit_of(symbol)` GLOBALLY; `$UNIT` appears nowhere in a stub body.
//! The module path — `x::norm::add_bias::raw::` — is Rust namespacing and
//! only that. So each call below names another family's symbol through that
//! family's own stub, with the real [`crate::x::Abi`] `CPP` spellings and
//! full type checking, and declares nothing twice.
//!
//! An earlier draft of this file hand-built a `&[ArgValue]` per call and
//! went to `x::fire::fire` directly, reasoning from WHERE a stub is declared
//! to WHAT it can name — the inference `mod` blocks invite and Rust does not
//! make. Seven of them. Reach for `fire` by hand only for a symbol no
//! `unit!` declares, which after the sweep is nothing.
//!
//! The one real consequence, which no mechanism covers: a cross-family call
//! makes the callee's unit a dependency of this host program and **nothing
//! in the type system says so**, because `symbol` is a `&'static str`. A
//! missing unit panics at the fire naming the symbol — right behaviour,
//! wrong time. The remedy is a comment and not a mechanism: each call
//! carries `// fires: <symbol>` beside it, so the caller is greppable from
//! the callee. Same argument §6.1 makes about declared signatures.
//!
//! The sixth, `ssm::qwen_gdn_post_conv_prep_bf16`, is not a row anywhere and
//! never was: it is a WALK over two device rows, and
//! `fire::gated_delta_net`'s `no_launcher_is_a_row` test asserts that
//! `unit_of` does not answer for it.
//!
//! # Geometry
//!
//! Every launch below writes its `Launch` as a literal or as one of
//! [`Launch`]'s two conveniences, and every one cites the `<<<>>>` or the
//! `LaunchRule` it came from. Five of the six had **no host program at all**
//! before this file — they were rows, and the generated dispatch arm built
//! their grid from the rule. So the citation for those is the rule function
//! in `runtime/launch.rs` plus, where the header still has one, the
//! `<<<>>>` the rule was checked against. Nothing here is invented.

use core::ffi::c_void;

use crate::x::abi::bf16;
use crate::x::contract::{Fired, Refusal};
use crate::x::launch::Launch;

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`, the block every
/// pointwise rule in this tree uses and the block four of the six launches
/// below take.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:581` — `const MAX_BLOCK: u32 = 1024;`, the cap
/// `route_rows` clamps a row width to.
const MAX_BLOCK: u32 = 1024;

/// The QKV split — `attn::split_qkv_bf16`, `attn/split_packed.cuh:74`.
///
/// The generated bodies called this ~390 times: the loud case the attn
/// exhaustiveness test names, and the single most-fired `driver_internal`
/// row there was.
///
/// # The two widths come off what is WRITTEN
///
/// The deleted row sourced them `OutWidth(0)` and `OutWidth(1)` and said
/// why: *a `[N, q + 2*kv]` row cannot say where the cut falls, and both
/// results can.* As parameters they are the same fact with the reader
/// removed — the caller has both output rectangles in hand.
///
/// # Geometry
///
/// `split_packed.cuh:18` records the launcher both of this header's kernels
/// had:
///
/// ```text
/// <<<dim3(ceil(max(q_dim, kv_dim) / 256), n), 256>>>
/// ```
///
/// which is `LaunchRule::SplitPacked` — `runtime/launch.rs:1674`,
/// `grid: [in_width.div_ceil(BLOCK), rows, 1]` — with `in_width` being
/// `max(q_dim, kv_dim)` and NOT the packed width. That distinction is the
/// reason this is written as a literal rather than fetched from the rule: a
/// reader who assumed the grid spans the packed row would over-launch by
/// `2 * kv_dim / 256` blocks, every one of which returns on its bounds
/// check, and would never see a wrong byte to tell them so.
///
/// # Safety
///
/// `packed` is `[n_tokens, q_dim + 2 * kv_dim]` bf16; `q_out` is
/// `[n_tokens, q_dim]` and `k_out`/`v_out` are `[n_tokens, kv_dim]`, all
/// bf16 and all writable. All four live on `stream`, which must outlive the
/// launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn split_qkv_bf16(
    packed: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    v_out: *mut c_void,
    n_tokens: i32,
    q_dim: i32,
    kv_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n_tokens" });
    }
    if q_dim <= 0 && kv_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "q_dim and kv_dim" });
    }
    #[allow(clippy::cast_sign_loss)] // both guarded above
    let width = q_dim.max(kv_dim) as u32;
    #[allow(clippy::cast_sign_loss)] // guarded above
    let launch = Launch {
        grid: [width.div_ceil(BLOCK), n_tokens as u32, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // fires: attn::split_qkv_bf16
    //
    // SAFETY: the caller's assertion, forwarded. The device row states six
    // operands — `crate::x::attn::split_packed`'s, which drops `n_tokens`
    // into the grid and the stream out of the `void**`. It was
    // `families::attn`'s `SPLIT_PACKED_SIGS[0]` until `attn` took that root
    // into fn-world, which is the move this module's header anticipated:
    // *"the rows stay where the device text is"*.
    unsafe {
        crate::x::attn::split_packed::raw::split_qkv(
            "attn::split_qkv_bf16",
            launch,
            packed.cast::<bf16>(),
            q_out.cast::<bf16>(),
            k_out.cast::<bf16>(),
            v_out.cast::<bf16>(),
            q_dim,
            kv_dim,
            stream,
        );
    }
    Fired::Launched
}

/// The bias add — `norm::add_bias_bf16`, `norm/add_bias.cuh`.
///
/// In place over the value it biases — one operand, one result, the same
/// bytes — which is why the deleted row stated `in_place = &[(0, 0)]` and
/// bound `out` from `Out(0)`. The bias is the statement's named weight, like
/// the embedding's table.
///
/// # Geometry
///
/// `LaunchRule::RouteRows` — `runtime/launch.rs:1028`, `grid: [rows, 1, 1]`,
/// `block: [clamp(width rounded up to a warp, WARP, MAX_BLOCK), 1, 1]`,
/// `smem: 0`. `families::norm`'s row records what it replaced and why the
/// replacement is exact:
///
/// > The launcher was `<<<num_rows, 256>>>` with a stride loop over `dim`,
/// > so the rule's wider block reaches the same elements in fewer iterations
/// > and the arithmetic per element is unchanged.
///
/// [`crate::x::gemm::act_x_wt_bias_bf16`] fires this same row with this same
/// geometry as the second call of its two-call body; the two agree because
/// they compute it the same way, and if one is ever wrong both are.
///
/// # Safety
///
/// `out` is `[num_rows, dim]` bf16 and writable, `bias` is `[dim]` bf16.
/// Both live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn add_bias_bf16(
    out: *mut c_void,
    bias: *const c_void,
    num_rows: i32,
    dim: i32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "dim" });
    }
    #[allow(clippy::cast_sign_loss)] // both guarded above
    let block = (dim as u32).div_ceil(WARP).max(1) * WARP;
    #[allow(clippy::cast_sign_loss)] // guarded above
    let launch = Launch::per_row(num_rows as u32, block.min(MAX_BLOCK));
    // fires: norm::add_bias_bf16
    //
    // SAFETY: the caller's assertion, forwarded. Three operands — the row
    // dropped `num_rows` into the grid and the stream out of the `void**`.
    unsafe {
        crate::x::norm::add_bias::raw::add_bias(
            "norm::add_bias_bf16",
            launch,
            out.cast::<bf16>(),
            bias.cast::<bf16>(),
            dim,
            stream,
        );
    }
    Fired::Launched
}

/// Qwen3.5's post-convolution split — `ssm::qwen_gdn_post_conv_prep_bf16`,
/// `gated_delta_net.cu:139-168`.
///
/// The post-conv prep, fused: q/k split and L2-normalized, v widened to
/// fp32, and g/beta gated — the three launches that used to sit between the
/// conv and the recurrent step. Its five fp32 outputs are exactly the step's
/// first five inputs, which is the shape of it.
///
/// # Two kernels, unconditionally, in order
///
/// The q/k RMS norm over `K_h` heads, then the v/gate/beta split over `V_h`.
/// Not a switch — an `execution::Control::Supplies`, and what it supplies is
/// `q_scale = rsqrtf(K_d)`.
///
/// This is §2.3's `Composed` — two DIFFERENT kernels in one body — and like
/// [`crate::x::gemm::act_x_wt_bias_bf16`] it needs none of `Composed`'s
/// machinery: in fn-world a body that makes two calls makes two calls.
///
/// # Why the scale is computed here and not in the kernel
///
/// It is one reciprocal square root of an operand extent, and putting it in
/// the kernel would make every one of `N * K_h * 128` threads recompute it.
/// The archive computed it on the host for that reason and so does this;
/// `f32::sqrt().recip()` is `rsqrtf` to the last bit for the exact powers of
/// two `K_d` takes in production (128, 256), and IEEE-exact for any `K_d`
/// whose square root is exact.
///
/// # NO barrier between the two launches
///
/// The second reads `qkv_post` again rather than the first's output, so they
/// are independent and the stream's ordering is enough. A reader expecting
/// the second to consume the first's `q_norm_kh` would be wrong, and this is
/// where to say so.
///
/// # The refusal
///
/// `if (N <= 0 || K_h <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;` —
/// `gated_delta_net.cu:153`. All five, even though only three reach a grid:
/// `K_d` feeds `rsqrtf` and `V_d` a stride. Under the row world that was a
/// bare `return` inside a launcher and no caller could tell it from a
/// launch; here it is a [`Refusal`].
///
/// # Geometry
///
/// `gated_delta_net.cu:154` — `constexpr int BLOCK = 128;`, the prep's block
/// for both launches, which is why this file's [`BLOCK`] is not used here.
///
/// ```text
/// dim3 qk_grid(N, K_h);
/// device::qwen_gdn_qk_norm<device::bf16, BLOCK>
///     <<<qk_grid, BLOCK, 0, stream>>>(
///         qkv_post, q_norm_kh, k_norm_kh, K_h, K_d, conv_dim, q_scale);
/// ```
///
/// ```text
/// dim3 vg_grid(N, V_h);
/// device::qwen_gdn_v_g_beta<device::bf16, BLOCK>
///     <<<vg_grid, BLOCK, 0, stream>>>(
///         qkv_post, a, b, A_log, dt_bias,
///         v_fp32, g_log_out, beta_out, K_h, V_h, K_d, V_d, conv_dim);
/// ```
///
/// # Safety
///
/// `qkv_post` is `[N, conv_dim]` bf16; `a`, `b` and `dt_bias` are bf16 over
/// `[N, V_h]`, `[N, V_h]` and `[V_h]`; `a_log` is `[V_h]` fp32; the five
/// outputs are writable for `[N, K_h, K_d]`, `[N, K_h, K_d]`,
/// `[N, V_h, V_d]`, `[N, V_h]` and `[N, V_h]`. All live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn qwen_gdn_post_conv_prep_bf16(
    qkv_post: *const c_void,
    a: *const c_void,
    b: *const c_void,
    a_log: *const c_void,
    dt_bias: *const c_void,
    q_norm_kh: *mut f32,
    k_norm_kh: *mut f32,
    v_fp32: *mut f32,
    g_log_out: *mut f32,
    beta_out: *mut f32,
    n: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "N" });
    }
    if k_h <= 0 || v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "K_h or V_h" });
    }
    if k_d <= 0 || v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "K_d or V_d" });
    }
    /// `gated_delta_net.cu:154` — `constexpr int BLOCK = 128;`.
    const PREP_BLOCK: u32 = 128;
    // `const float q_scale = rsqrtf(static_cast<float>(K_d));` —
    // `gated_delta_net.cu:155`.
    #[allow(clippy::cast_precision_loss)] // `K_d` is a head width, <= 256
    let q_scale = (k_d as f32).sqrt().recip();
    #[allow(clippy::cast_sign_loss)] // guarded above
    let qk_launch = Launch {
        grid: [n as u32, k_h as u32, 1],
        block: [PREP_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // fires: ssm::qwen_gdn_post_conv_prep_bf16#qk_norm
    //
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        crate::x::ssm::gated_delta_net_prep::raw::qwen_gdn_qk_norm(
            "ssm::qwen_gdn_post_conv_prep_bf16#qk_norm",
            qk_launch,
            qkv_post,
            q_norm_kh,
            k_norm_kh,
            k_h,
            k_d,
            conv_dim,
            q_scale,
            stream,
        );
    }
    #[allow(clippy::cast_sign_loss)] // guarded above
    let vg_launch = Launch {
        grid: [n as u32, v_h as u32, 1],
        block: [PREP_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // fires: ssm::qwen_gdn_post_conv_prep_bf16#v_g_beta
    //
    // SAFETY: the caller's assertion, forwarded. No barrier between the two
    // — see the header paragraph that says why.
    unsafe {
        crate::x::ssm::gated_delta_net_prep::raw::qwen_gdn_v_g_beta(
            "ssm::qwen_gdn_post_conv_prep_bf16#v_g_beta",
            vg_launch,
            qkv_post,
            a,
            b,
            a_log.cast::<f32>(),
            dt_bias,
            v_fp32,
            g_log_out,
            beta_out,
            k_h,
            v_h,
            k_d,
            v_d,
            conv_dim,
            stream,
        );
    }
    Fired::Launched
}

/// The per-head query/gate split — `layout::split_q_gate_bf16`,
/// `layout/deinterleave.cuh`.
///
/// Full attention's q_proj packs the query and the per-token output gate PER
/// HEAD — `[N, heads, 2*head_dim]`, query first — so this is strided by
/// head, not a halves cut like `split_gate_up`. Three shape arguments rather
/// than one width, because the stride IS the layout.
///
/// # `n` and `num_heads` survive as operands where a rows count would not
///
/// The kernel guards `if (n >= N || h >= num_heads) return;` and, more to
/// the point, multiplies both back into every address it forms —
/// `(n * num_heads + h) * 2 * head_dim` — so they are addressing arithmetic
/// the grid happens to agree with rather than an extent the grid recovers.
///
/// # Geometry
///
/// `deinterleave.cu`'s `split_q_gate_bf16`:
///
/// ```text
/// dim3 grid(N, num_heads);
/// <<<grid, (head_dim < 128) ? 64 : 128, 0, stream>>>
/// ```
///
/// This is the LAUNCHER's block and not `LaunchRule::PerHeadElementwise`'s
/// `clamp(head_dim, 32, 128)`. `families::layout`'s row compared the two at
/// length and found them equivalent — *both cover the head, because the two
/// copy loops stride `i += blockDim.x` and stop at `i < head_dim`* — so
/// either is correct, and the literal is written here because §5.1 says a
/// convenience that does not fit is not reached for. The one shape where
/// they differ in a way worth knowing: under 32, the rule's clamp is WIDER
/// than the head, and the surplus lanes fail the `i < head_dim` test on
/// their first iteration.
///
/// # Safety
///
/// `packed` is `[n, num_heads, 2 * head_dim]` bf16; `q_out` and `gate_out`
/// are `[n, num_heads, head_dim]` bf16 and writable. All live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn split_q_gate_bf16(
    packed: *const c_void,
    q_out: *mut c_void,
    gate_out: *mut c_void,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    let block = if head_dim < 128 { 64 } else { 128 };
    #[allow(clippy::cast_sign_loss)] // all three guarded above
    let launch = Launch {
        grid: [n as u32, num_heads as u32, 1],
        block: [block, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // fires: layout::split_q_gate_bf16
    //
    // SAFETY: the caller's assertion, forwarded. Six operands — the stream
    // is `cuLaunchKernel`'s parameter and never was one.
    unsafe {
        crate::x::layout::deinterleave::raw::split_q_gate(
            "layout::split_q_gate_bf16",
            launch,
            packed.cast::<bf16>(),
            q_out.cast::<bf16>(),
            gate_out.cast::<bf16>(),
            n,
            num_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// That gate applied — `mlp::sigmoid_gate_inplace_bf16`, `mlp/swiglu.cuh:261`.
///
/// `a' = a * σ(g)`, IN PLACE on operand 0 — the header spells `x` *"bf16,
/// in-place"* in as many words, and the gate is read-only, which is what let
/// the deleted row state `in_place` on operand 0 alone.
///
/// `families::mlp` carries the twin of this row and its note survives the
/// move: *the gate is EMITTED by the model rather than stated by a trace.
/// Same kernel, same operands.*
///
/// # Geometry
///
/// `LaunchRule::Elementwise` — `runtime/launch.rs:828`,
/// `grid: [n.div_ceil(BLOCK), 1, 1]`, `block: [BLOCK, 1, 1]`, `smem: 0`,
/// which is [`Launch::flat`] at 256. `swiglu.cuh` declares no host launcher
/// for this kernel — it is device text only — so the rule IS the citation
/// and there is no `<<<>>>` to check it against.
///
/// # Safety
///
/// `x` and `gate` are both `num_elements` bf16 elements; `x` is writable and
/// is read and written by the same threads. Both live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn sigmoid_gate_inplace_bf16(
    x: *mut c_void,
    gate: *const c_void,
    num_elements: i32,
    stream: *mut c_void,
) -> Fired {
    if num_elements <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    #[allow(clippy::cast_sign_loss)] // guarded above
    let launch = Launch::flat(num_elements as u32, BLOCK);
    // fires: mlp::sigmoid_gate_inplace_bf16
    //
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        crate::x::mlp::swiglu::raw::sigmoid_gate_inplace(
            "mlp::sigmoid_gate_inplace_bf16",
            launch,
            x.cast::<bf16>(),
            gate.cast::<bf16>(),
            num_elements,
            stream,
        );
    }
    Fired::Launched
}

/// The gated norm with an FP32 `x` — `norm::rmsnorm_gated_fp32_in_bf16`,
/// `norm/rmsnorm.cuh:763`.
///
/// The GDN recurrent step lands in fp32, so this reads it there and the
/// separate conversion launch goes away. `x` and `weight` hold fp32 and the
/// header spells them `const void*`; the declaration is what has to agree,
/// not the contents.
///
/// # This function closes a binder gap that the row could not
///
/// `families::norm`'s row states the problem in full and it is the best
/// argument in this tree for fn-world:
///
/// > Its rectangle is `rows · v_h` rows of `v_d` and both numbers come off
/// > `GdnCtx` [...] `RowsPerHead` computes `rows · (width /
/// > stated_head_dim)`, which IS `rows · v_h` exactly when
/// > `stated_head_dim` carries `v_d` [...] and the binder is not there yet:
/// > `OpKind::RmsnormGated` never sets `spec.per_head_dim`, so a GDN fire
/// > reaches `jit_dims` with nothing to state and takes the ABSENT arm,
/// > which is `rows` blocks of `width` where `rows · v_h` blocks of `v_d`
/// > were meant.
///
/// The deleted `driver_internal` row said the same from the other side:
///
/// > UNSOURCED, and the two numbers say why. The GDN landing norm runs per
/// > (row, VALUE HEAD) over the trailing head width, so its rows are `rows *
/// > gdn.v_h` and its width is `gdn.v_d` — a PRODUCT of the fire's rows and
/// > a context field, which no `Source::` spells. A row that said `Rows` and
/// > `OutWidth(0)` would launch the right kernel over the wrong rectangle,
/// > which is worse than having no row: the hybrid's prefill found it
/// > immediately, and only because the walk asserts every launch ran.
///
/// **`num_rows` is a parameter here.** There is no `Source::` to be missing,
/// no `spec.per_head_dim` to be unset, and no arm to take by accident: the
/// caller states the rectangle it means, which is the third of the three
/// shim-dropping mechanisms doing exactly what it is for. The fix the row
/// asked `driver-cuda`'s binder for is not needed for THIS symbol any more.
/// It is still needed for any other row that reaches `jit_dims` under
/// `OpKind::RmsnormGated`.
///
/// # Geometry
///
/// `LaunchRule::RowsPerHead` — `runtime/launch.rs:815` — is
/// `grid: [blocks, 1, 1]`, `block: [BLOCK, 1, 1]`, `smem: 0`, with `blocks`
/// being `rows * (width / stated_head_dim)`. `num_rows` IS that product,
/// computed by the caller. `rmsnorm.cuh:14` records the block every kernel
/// in that header takes:
///
/// ```text
/// kernel<BLOCK><<<grid, 256, 0, stream>>>(..., hidden_size, eps);
/// ```
///
/// # Safety
///
/// `x` is `[num_rows, hidden]` fp32; `gate` is `[num_rows, hidden]` bf16;
/// `weight` is `[hidden]` fp32; `y` is `[num_rows, hidden]` bf16 and
/// writable. All live on `stream`.
#[cfg(feature = "_cuda")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rmsnorm_gated_fp32_in_bf16(
    x: *const c_void,
    gate: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    #[allow(clippy::cast_sign_loss)] // guarded above
    let launch = Launch::per_row(num_rows as u32, BLOCK);
    // fires: norm::rmsnorm_gated_fp32_in_bf16
    //
    // SAFETY: the caller's assertion, forwarded. Six operands — the row
    // dropped `num_rows` into the grid and the stream out of the `void**`.
    // The two fp32 parameters are `*const f32` in the stub and `*const
    // c_void` here for the reason this fn's header already gives: *"the
    // header spells them `const void*`; the declaration is what has to
    // agree, not the contents"* — so the cast is at the boundary, once.
    unsafe {
        crate::x::norm::rmsnorm::raw::rmsnorm_gated_f32_in(
            "norm::rmsnorm_gated_fp32_in_bf16",
            launch,
            x.cast::<f32>(),
            gate.cast::<bf16>(),
            weight.cast::<f32>(),
            y.cast::<bf16>(),
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}
