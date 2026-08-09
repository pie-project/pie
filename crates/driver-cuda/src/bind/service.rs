//! The rows the DRIVER executes itself — [`Execution::Service`], in Rust.
//!
//! [`Execution::Service`]: kernels_cuda_new::execution::Execution::Service
//!
//! # What this module is
//!
//! `kernels-cuda-new`'s `execution.rs` classifies every row as `Jit`,
//! `Composed` or `Service`, and says of the third: *a symbol whose body is
//! one library call and nothing else is not a kernel, and extracting it as
//! one is extracting nothing.* Fourteen rows are classified that way. Until
//! §45 that classification was **data with no consumer** — the calls were
//! still issued by `gemm/gemm.cpp`, a C++ translation unit, and a row served
//! by C++ is a row that cannot leave the archive.
//!
//! **`gemm/gemm.cpp` is now deleted.** §45 took the four pure-cuBLAS bodies
//! into this module; a later pass took the quantized router into
//! [`crate::bind::quant_gemm`]; and the last pass took the dense bf16
//! autotuner — the largest single thing in the file, and the reason it
//! outlived the rest — into [`crate::fire::gemm`]. Nothing in this tree
//! issues a cuBLAS call from C++ any more. The paragraph above is kept in the
//! past tense because it is the reason this module has the shape it has, not
//! because the condition still holds.
//!
//! This module is the consumer. It issues the same library calls from Rust,
//! through `cudarc`'s dynamically-loaded cuBLAS, and it exists so the C++
//! bodies could be deleted.
//!
//! # The constraint it is written under
//!
//! **The model compiler must not be able to tell whether a symbol is cuBLAS
//! or a JIT'd kernel.** Nothing above the dispatcher changes: [`KernelSig`]
//! is unchanged, the statement lowers the same way, and the arm that reaches
//! a function here is emitted by the same `abi::emit_dispatch` pass that
//! emits the JIT arms and the `pie_k_*` arms, from the same operand list.
//! The only difference is the callee's path, and that difference is decided
//! by one list — `execution::RUST_SERVED` — which no lowering reads.
//!
//! [`KernelSig`]: kernels::KernelSig
//!
//! # Why nothing new links
//!
//! Every entry below reaches cuBLAS through `cudarc::cublas::sys`, whose
//! `fallback-dynamic-loading` build resolves each symbol with `dlopen` on
//! first use. There is no `#[link]`, no `build.rs` flag and no header, so
//! `cargo check -p driver-cuda` with no CUDA toolkit on PATH still passes —
//! the hard gate that made `cudarc` the right seam and a C shim the wrong
//! one. The `cargo:rustc-link-lib=cublas` in `build.rs` is for the C++
//! ARCHIVE's remaining callers, not for this file, and it is why lifting it
//! out of the `bridge` block changes nothing about what this module needs.
//!
//! # A failure is a refusal, never a fallback
//!
//! Each C++ body this replaces ends in `check(status, ...)` or an explicit
//! `throw std::runtime_error(...)`, and the shim's `catch` turns that into an
//! abort with the cuBLAS status in the message. The ports below panic with
//! the same status number and the same shape identification. A non-success
//! status is **not** retried on another algorithm and **not** swallowed: the
//! one place the archive retried — `gemm_bf16_impl`'s
//! `CUBLAS_STATUS_NOT_SUPPORTED` second attempt — is in a body that stays in
//! C++ for an unrelated reason, and none of the four calls here ever had one.
//! `gemm_grouped_bf16_impl` says why in its own comment: *"a failed call
//! inside a graph capture invalidates the capture"*, so a speculative first
//! attempt is worse than no attempt.

use std::ffi::c_void;

use super::DispatchCtx;


// ── `attn/dsa_indexer.cu`'S THREE MOVED TO `kernels_cuda_new::x::attn` ─
//
// `attn_dsa_index_knorm_rope_bf16`, `attn_dsa_index_q_rope_bf16` and
// `attn_dsa_index_topk_mask` stood here over `fire::dsa_indexer`, and this
// file's three entry points were that module's ONLY consumer, so
// `fire/dsa_indexer.rs` crossed whole.
//
// None of the three needs a driver resource -- a grid, a block and a stream
// is the entire host side -- so all three are moves and not driver ops. Two
// of the three land as `none:` arms, and the reason is a STATEMENT rather
// than a query: see `x::attn`'s `DSA_INDEX_Q_ROPE`.
//
// `Indexer` / `IndexerDecline` did not cross. Both were one variant wide
// (`tokens <= 0`), and `Fired::Declined(Refusal::Empty { what: "tokens" })`
// says it in the floor's own vocabulary.

// ── MLA'S ABSORB PAIR MOVED TO `kernels_cuda_new::x::attn` ─────────────
//
// `gemm_mla_absorb_q_to_latent_bf16` and `gemm_mla_absorb_latent_to_v_bf16`
// were `execution::RUST_SERVED` rows resolved by the generated dispatch --
// eight operands each, all eight sourced, so both had live arms. They are
// `Service::DriverOp` now: two `contract!`s in `x::attn` with no `Entry`,
// two bodies taking `handle: *mut c_void`, and two hand arms in
// `bind::dispatch`.
//
// THE REASON IS NOT THAT THIS FILE IS `bridge`-GATED. Sec 78 measured that
// gate and found it marks where the archive WAS rather than where it is, so
// it is not a reason to move anything. The reason is the one `x::gemm`'s
// twelve are in `x::gemm`: a cuBLAS host program is a host program, it
// belongs beside the truth it is one of, and `handle` as a first parameter
// is how fn-world spells a resource it cannot own.
//
// The port gained one thing it could not have here. `tokens <= 0 ||
// heads <= 0` was a bare `return` because the signature was `()`; it is
// `Fired::Declined(Refusal::Empty { what })` there, naming WHICH extent was
// empty. The `write_kv_native` lesson a second time: a port that only
// preserves is a transcription.

// ── cuBLAS LEAVES THIS FILE — `COMPUTE`, `ALGO`, `check` AND `absorb` ──
//
// The three cuBLAS helpers and MLA's shared `absorb` stood here, and the
// last consumer of all four was the absorb pair. `x::attn` carries them now
// as `ABSORB_COMPUTE`, `ABSORB_ALGO`, `absorb_check` and `absorb`, verbatim
// down to the panic — including the reason for the panic, which is that the
// C++ threw and the shim aborted, so a `Result` here would be a fallback
// §45 forbids. The `cudarc::cublas::sys` import went with them; nothing
// left in this file speaks to cuBLAS.
//
// `x::gemm::dense` has its own `COMPUTE`, `ALGO_TENSOR_OP` and `check`, all
// three PRIVATE, so `x::attn`'s copies are a third instance the module
// boundary forces rather than one anybody chose. That is stated at the
// definitions there; it is not stated twice.

// ── `gemm_act_x_wt_bf16_out_fp32` MOVED TO `kernels_cuda_new::x::gemm::act_x_wt_bf16_out_fp32` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
// ///
// /// Ported from `gemm.cpp:1030-1058` (`gemm_bf16_out_fp32_impl`, reached
// /// through the one-line `act_x_wt_bf16_out_fp32` at `:2327`). Row-major
// /// `y[M, N] = act[M, K] @ W[N, K]^T`, written column-major as the transpose,
// /// which is where `OP_T/OP_N` and the `m=N, n=M` swap come from.
// ///
// /// # Safety
// ///
// /// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
// /// address `M*N` live floats, and all three must outlive the launch — which
// /// is asynchronous on the handle's stream, so "outlive" ends at the next
// /// synchronisation and not at this call's return.

// ── `gemm_grouped_act_x_wt_bf16` MOVED TO `kernels_cuda_new::x::gemm::grouped_act_x_wt_bf16` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
// ///
// /// Ported from `gemm.cpp:1242-1294` (`gemm_grouped_bf16_impl`, reached
// /// through `grouped_act_x_wt_bf16` at `:1632`). Every group shares `N`, `K`
// /// and the three leading dimensions; only `M` differs, which is why the
// /// arrays are filled from one scalar each and `n[]` from `M_array_host`.
// ///
// /// **This entry takes the handle rather than a [`DispatchCtx`]**, and it is
// /// the one that does. Its row states `Source::Unbound` for every operand — a
// /// group boundary is fire-global and no `Source` names one — so
// /// `emit_dispatch` writes no arm for it and its only consumer is
// /// `fire::lora`'s hand-written staged apply, which holds a `cublasHandle_t`
// /// and no context.
// ///
// /// # Safety
// ///
// /// The three pointer arrays must be HOST arrays of `group_count` device
// /// addresses (cuBLAS reads them on the host for the grouped form), and
// /// `m_array` a host array of `group_count` row counts.

// ── `gemm_act_x_wt_bf16` MOVED TO `kernels_cuda_new::x::gemm::act_x_wt_bf16` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::act_x_wt_bf16` — the dense bf16 GEMM. Body in
// /// [`crate::fire::gemm::act_x_wt_bf16`].
// ///
// /// `y[M, N] = act[M, K] @ W[N, K]^T + beta * y`, all bf16, fp32 accumulate.
// /// The hottest row in the tree: every linear layer of every model lands here.
// ///
// /// **This is not one cuBLAS call and that is why it took so long to arrive.**
// /// It is a runtime autotuner over three kernel families — the warp-per-row
// /// GEMV, `cublasGemmEx`, and each algorithm cuBLASLt's heuristic offers —
// /// with a per-device tactic memo, an on-disk tactic cache and a fallback
// /// ladder behind it. All of it host code, all of it now Rust; the module
// /// carries the measurements.
// ///
// /// The thing that held it in C++ for three arcs was that `gemm_bf16_impl`
// /// called `gemv_bf16`, whose `bool` meant *"I did not launch"*, and a row
// /// cannot decline. The resolution was not to make the row decline: a
// /// **driver-owned launch is not a row**, so [`crate::fire::gemv::gemv_bf16`]
// /// spells its refusal as a type and the tuner's GEMV candidate is a
// /// `matches!(.., Gemv::Launched)` in the same short-circuiting position the
// /// C++ put it in.
// ///
// /// # Why the handle is an operand and `ctx` is not enough
// ///
// /// The row states `handle: CublasHandle <- Source::Ctx("cublas")`, so the
// /// emitted arm passes both `ctx` and the bound handle — the same redundancy
// /// [`gemm_act_x_wt_bias_bf16`] documents, and for the same reason: the
// /// composition takes this row as its first step and `Composition::agrees`
// /// type-checks `Take::From(i)` against the operands as stated. They are the
// /// same pointer; `ctx.cublas` is the engine's handle, created once at boot by
// /// `device::cublas`.
// ///
// /// # Safety
// ///
// /// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
// /// and outlive the launch — asynchronous on the handle's stream, so "outlive"
// /// ends at the next synchronisation and not at this call's return.
// #[allow(clippy::too_many_arguments)]

// ── `gemm_act_x_wt_bias_bf16` MOVED TO `kernels_cuda_new::x::gemm::act_x_wt_bias_bf16` ──
//
// §5 step 5. Its body is verbatim where the device text and the tuner
// are; what stood here was the ~120 lines of argument assembly this
// crate no longer owns. **Its documentation is kept, unedited, as
// comments** — every measurement in it is a measurement about the body,
// and the body did not change when it changed crates:
//
// /// `gemm::act_x_wt_bias_bf16` — the COMPOSITION, not a service.
// ///
// /// `execution::COMPOSED` already stated this row, step for step, and cited
// /// `gemm.cpp:2395-2398` for it: a `gemm::act_x_wt_bf16` and then a
// /// `norm::add_bias_bf16` over the result. This is that statement, executed.
// /// It is in this module because the seam is the same one — a row the driver
// /// runs itself, with no entry in the C++ shim.
// ///
// /// # What is lost, exactly
// ///
// /// The archive had a second arm: at `M == 1` with a bias, it asked
// /// `dense_tactic_for` whether the tuner's chosen tactic could absorb the bias
// /// into its epilogue, and `run_dense_tactic` declines every tactic except the
// /// warp-per-row GEMV. So the fused arm fired **only** on the GEMV, and its
// /// kernels state what they compute: `out[n] = bf16(bf16(dot) + bias[n])`, the
// /// double rounding deliberate, *"bit-identical to running `add_bias_bf16`
// /// afterwards"*. (That was `gemv.hpp`'s wording; the header is deleted and
// /// the sentence is now at both epilogues of
// /// `kernels-cuda-new/csrc/src/gemm/gemv.cuh`, which is the text NVRTC
// /// compiles.) The composition therefore produces THE SAME BYTES and costs one
// /// extra launch per biased `M == 1` projection.
// ///
// /// That is the whole cost and it is stated rather than measured away: the
// /// fusion was worth 11.9% of gpt-oss-20b's decode time when it was added
// /// (`gemm.hpp`), and what buys it back is a bias epilogue on a JIT'd GEMV.
// /// **That kernel now exists** — the `gemm/gemv` unit's four rows all take
// /// `bias` and fold it, and `fire::gemv::gemv_bf16` passes it through — so what
// /// is missing is no longer a kernel but a Rust caller that reaches it at
// /// `M == 1` instead of reaching `pie_k_gemm_act_x_wt_bf16`, which means the
// /// dense tactic enumeration in Rust. **That enumeration now exists** —
// /// [`crate::fire::gemm`] — so the remaining work is a `fire::gemm` entry that
// /// takes a `bias` and, when the tuned tactic for the shape is
// /// `GemmKind::Gemv`, passes it down instead of adding it afterwards.
// /// [`crate::fire::gemm::dense_tactic_is_gemv`] is the side-effect-free peek
// /// that arm needs, ported and waiting.
// ///
// /// # Safety
// ///
// /// `act`, `w`, `bias` and `y` must address live device memory of the extents
// /// `M`, `N` and `K` describe, and `y` must be writable.
// ///
// /// # Why this one still takes a handle and a stream
// ///
// /// The other four dropped `handle: CublasHandle` from their rows, because a
// /// service carries its own. This row cannot: `execution::COMPOSED` states its
// /// first step as `gemm::act_x_wt_bf16`, whose row DOES take a handle, and
// /// `Composition::agrees` type-checks each `Take::From(i)` against the
// /// composed row's operands. Remove the handle here and the composition can no
// /// longer supply its own first step. So the row keeps the operands the
// /// composition needs, the arm binds them, and `ctx` arrives as well because
// /// every service arm is emitted the same way — the redundancy is the
// /// emitter's uniformity, and `ctx.cublas`/`ctx.stream` are what
// /// `Source::Ctx("cublas")`/`Source::Ctx("stream")` bind to anyway.
// #[allow(clippy::too_many_arguments)]

// ─────────────────────────────────────────────────────────────────────────
// The quantized rows — `gemm.cpp`'s three `WeightView` entry points
// ─────────────────────────────────────────────────────────────────────────
//
// Bodies in [`super::quant_gemm`]; the spellings are here because
// `every_rust_served_symbol_is_spelled_here` reads THIS file's text. Each is
// the `gemm.hpp` inline it replaces: build a `WeightView` from the row's
// operands, then call the one router. `execution::WALKED` states them as
// `Control::Switch { on: "w_dtype" }`, which is what the router is.

/// `gemm::act_x_wt_channel_scaled` — `gemm.hpp:160`.
///
/// `y[M, N] = act[M, K] x W[N, K]^T`, with `W` quantized per output channel:
/// one scale per row of `W`. Serves both FP8 E4M3 and INT8 weights, and the
/// two take completely different routes inside — FP8 per-channel always
/// dequants to bf16 (cuBLASLt has no per-channel FP8 scale mode this tree
/// targets), INT8 per-channel runs the native `CUBLAS_COMPUTE_32I` path.
///
/// `channel_axis` is accepted and NOT read, exactly as the archive's inline
/// accepted and did not read it: the row states it because a per-channel
/// scale has an axis, and every weight this driver materialises is `[N, K]`
/// row-major with the channel on axis 0. A non-zero value is not refused
/// here because the C++ did not refuse it either — recording that is worth
/// more than inventing a check the archive never made.
///
/// # Safety
///
/// Every pointer must be a device address on the current device, `w` must
/// hold at least `N * K` elements of `w_dtype` and `scale` at least `N`
/// values; `y` must be writable for `M * N` bf16. Checked as far as
/// `validate_quant_weight_view` can check it, which is the byte counts.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_channel_scaled(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    _zero_point: *const c_void,
    _channel_axis: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = super::quant_gemm::WeightView {
        data: w,
        dtype: w_dtype,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype,
        scale_numel,
        quant_kind: super::quant_gemm::quant_kind::PER_CHANNEL,
        group_size: 0,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        super::quant_gemm::act_x_w(
            handle,
            act,
            view,
            y,
            m,
            n,
            k,
            beta,
            super::quant_gemm::dtype::BF16,
            super::quant_gemm::dtype::BF16,
        );
    }
}

/// `gemm::act_x_wt_grouped_scaled` — `gemm.hpp:182`.
///
/// The same GEMM with `W` quantized per group along `K`. `group_size` is the
/// group extent, and for FP8 it is also the extent along `N`: DeepSeek's
/// `weight_block_size = [128, 128]` is a 2-D block scale, which is why
/// `validate_quant_weight_view` counts `ceil(N/gs) * ceil(K/gs)` scales for
/// FP8 and `N * ceil(K/gs)` for everything else.
///
/// **This is the row that reaches the block-scaled W8A8 path** — the one
/// arm here that does not dequant the weight, and the reason it exists is a
/// measurement: re-expanding a block-quantized FP8 weight to bf16 costs 5x
/// the weight bandwidth of the matmul and dominates decode.
///
/// # Safety
///
/// As [`gemm_act_x_wt_channel_scaled`], with the scale count above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_grouped_scaled(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    _zero_point: *const c_void,
    group_size: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = super::quant_gemm::WeightView {
        data: w,
        dtype: w_dtype,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype,
        scale_numel,
        quant_kind: super::quant_gemm::quant_kind::PER_GROUP,
        group_size,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        super::quant_gemm::act_x_w(
            handle,
            act,
            view,
            y,
            m,
            n,
            k,
            beta,
            super::quant_gemm::dtype::BF16,
            super::quant_gemm::dtype::BF16,
        );
    }
}

/// `gemm::act_x_wt_mxfp4_marlin` — `gemm.hpp:206`.
///
/// MXFP4: four-bit elements packed two per byte with one raw E8M0 exponent
/// byte per 32-element block. The scale dtype is UINT8 and the group size is
/// 32, and both are asserted rather than defaulted.
///
/// **"marlin" in the name is the checkpoint format's, not a kernel's.** The
/// vendored marlin tree went in §54; this row dequants to bf16 and runs the
/// classic GEMM, which is what the archive's arm did after the removal too.
///
/// # Safety
///
/// `w` must hold at least `ceil(N * K / 2)` bytes and `scale` at least
/// `N * ceil(K / 32)` bytes; `y` writable for `M * N` bf16.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_act_x_wt_mxfp4_marlin(
    _ctx: &DispatchCtx,
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    w_nbytes: usize,
    scale: *const c_void,
    scale_numel: usize,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    let view = super::quant_gemm::WeightView {
        data: w,
        dtype: super::quant_gemm::dtype::MXFP4_PACKED,
        nbytes: w_nbytes,
        scale_data: scale,
        scale_dtype: super::quant_gemm::dtype::UINT8,
        scale_numel,
        quant_kind: super::quant_gemm::quant_kind::PER_GROUP,
        group_size: 32,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        super::quant_gemm::act_x_w(
            handle,
            act,
            view,
            y,
            m,
            n,
            k,
            beta,
            super::quant_gemm::dtype::BF16,
            super::quant_gemm::dtype::BF16,
        );
    }
}

// `moe::moe_grouped_gemm_bf16` STOOD HERE, as `pub unsafe fn
// moe_moe_grouped_gemm_bf16`, and is DELETED. §5 step 5 took `moe` into
// fn-world: the host program is `x::moe::moe_grouped_gemm_bf16`, and the
// refusal this wrapper had to drop -- the generated arm returned `bool` and
// its `true` meant "a branch ran" -- is now the value the fire reports with
// the symbol named. The symbol left `execution::RUST_SERVED` with the family.
//
// A `bind!` arm fired it for one round and no longer does. It is a DRIVER OP
// now, body `fire::moe_grouped`, because only half its shapes have a kernel:
// `supported()` refuses `K = 2048` and the implementation that serves that
// half is a batched cuBLAS call over the pointer arrays `fire::moe_ptrs`
// builds. Both of those are the driver's, neither is `Cx`'s, and
// `bind/mod.rs`'s "a refusal is not a fallthrough" makes the bind's
// `Refusal::Wide` an answer rather than the first half of a choice.

// `moe::flashinfer_cutlass_moe_bf16` STOOD HERE, as
// `pub unsafe fn moe_flashinfer_cutlass_moe_bf16`, and is DELETED with the
// fused CUTLASS leg it was the seam to.
//
// It was the driver-op shape entire: a `contract!` in `x::moe` with no
// `Entry`, reached by name through this wrapper into
// `fire::flashinfer_moe::bf16`, because the body needed a workspace query, an
// allocation and an arch probe -- a device surface `Cx` must not grow. That
// shape is what made the retirement a deletion rather than an unpicking:
// nothing but this function reached it, so removing the leg removed the
// whole reachable set.
//
// The decision was the owner's, on a measurement: carrying CUTLASS so NVRTC
// could compile the GEMM is a 505-file, 13,891,303-byte `include_str!`
// closure, against the 429-file, 4,376,255-byte carry this tree already
// refused in writing for cub. Same mechanism, 3.2x a line already drawn.
//
// **What it leaves behind is one bind, and it is not optional.**
// `moe::build_moe_ptrs_aligned_bf16` declares `gu_stage`/`act_stage`/
// `out_stage` -- the destinations every op in the aligned leg writes into --
// and it has never had an arm in either world. Every condition that turned
// the fused leg off already returned the aligned one, so this deletion makes
// the aligned leg the ONLY leg, and it cannot start until that binds.
//
// One behaviour is recorded rather than carried, because it is gone with the
// only caller that could reach it: the C++ `to_cutlass_activation` ended in
// `case Relu2: default:`, so an enumerator this driver had not been taught
// became `Relu2` rather than an error. This wrapper reproduced that and did
// not widen it. Only `Swiglu` was ever reachable through the statement, so
// nothing observed the other two.

// `sample::lm_head_gemv_argmax_int8` — `sample/argmax.hpp:37` — STOOD HERE,
// as `pub unsafe fn sample_lm_head_gemv_argmax_int8`, and is DELETED.
//
// Its doc said, and every sentence is still true of the thing it now names:
//
// > Greedy decode straight off an int8 LM head: for each of `num_rows`
// > hidden vectors, the vocab index whose dequantized dot product is
// > largest, written as one i32. The vocab-wide logit row is never
// > materialised, which is why `table::sample` states this as its own symbol
// > rather than as an `lm_head` GEMM followed by an argmax over its output.
// >
// > # Why it is here rather than behind a `pie_k_` shim
// >
// > Two kernels, a device scratch between them that the row's operand list
// > does not mention, and a grid extent read off
// > `cudaDevAttrMultiProcessorCount`. `execution::WALKED` classifies it as a
// > `Walk` for exactly that — host control flow whose shape comes from the
// > input and from the machine. What reaches this function is one call with
// > eight operands, the same eight the C++ launcher took, and no model text
// > can tell that two `__global__`s run behind it.
//
// §5 step 5 took `sample` into fn-world. The whole program is
// `kernels_cuda_new::x::sample::lm_head_gemv_argmax_int8`, and this wrapper
// is gone because there is nothing left for it to do: it existed to turn a
// generated dispatch arm's argument list into a `fire::` call, and a bind
// body reads a `Cx` instead.
//
// **There is no bind either, and that is the honest end of the last
// paragraph above.** "What reaches this function is one call with eight
// operands" was never true — all eight of the row's operands were
// `Source::Unbound`, `emit_rust_dispatch` skipped the row whole, and nothing
// in `crates/model` states the symbol. `x::sample`'s contract carries a
// written refusal naming the one fact that is still missing: the int8 head
// and its per-row dequant scale are named weights, and no model text names
// them. The refusal is made at model load now instead of being an
// `UnknownKernel` at fire time.

// ── `norm/`: SIX WRAPPERS STOOD HERE AND ARE GONE ─────────────────────
//
// `norm` crossed into fn-world (`.wiki/kernel-x/northstar.md` §5 step 5).
// Its host programs are `kernels-cuda-new/src/x/norm.rs`, beside the six
// `csrc/src/norm/*.cuh` roots they fire, and `bind/mod.rs::dispatch` reaches
// them through `kernels_cuda_new::x::entry` — one lookup, no wrapper.
//
// The six were: `norm_rmsnorm_bf16_with_fp16`,
// `norm_rmsnorm_residual_add_scale_rmsnorm_bf16`,
// `norm_hc_pre_postprocess_bf16`, `norm_hc_post_bf16`,
// `norm_hc_head_postprocess_bf16` and `norm_hc_rmsnorm_to_f32`. See the
// `rope/` note below for why a ported family needs none of them: a family
// that states no `operands` is on neither side of the `RUST_SERVED` fork.
//
// Every measurement in their doc comments moved with the fns. Two claims
// they made are worth repeating because `x::norm` now proves them:
//
//   * *"the middle arm is still TWO launches with the bf16 result as the
//     intermediate, which is what `Composition`'s `Take` cannot spell"* —
//     §2.3's `Composed` shape, and `x::norm::rmsnorm_bf16_with_fp16` is the
//     first body in the tree to write it. Its second launch is
//     `quant::bf16_to_fp16`, another FAMILY's kernel, which is the one thing
//     §2.3 does not cover.
//   * *"the fused arm no longer silently degrades to a different reduction
//     order, which was the §21.14 failure the refusal was protecting."*
//
// `norm::add_bias_bf16` is NOT one of the six and is still fired from this
// file, by the gemm wrapper above: that call goes through `super::jit::fire`
// by symbol and keeps resolving, because `x::norm` declares the same symbol.

// ── `rope/`: NINE WRAPPERS STOOD HERE AND ARE GONE ──────────────────────
//
// `rope` crossed into fn-world (`.wiki/kernel-x/northstar.md` §5 step 3).
// Its host programs are `kernels-cuda-new/src/x/rope.rs`, beside the
// `rope.cuh` they fire, and `bind/mod.rs::dispatch` reaches them through
// `kernels_cuda_new::x::entry` — one lookup, no wrapper.
//
// # Why the wrappers existed, and why nothing needs them now
//
// A wrapper here was the price of `execution::RUST_SERVED`: that list is
// what decides whether `abi::emit_rust_dispatch` writes an arm calling
// `bind::service::<sym>` or `emit_c_shim` writes a `pie_k_*`, so a symbol on
// it had to be spelled here in EXACTLY the row's operand order, including
// the row's `Ty::Stream` position. Three signatures of one kernel —
// the row, the wrapper, and the `fire::` fn — had to agree, and only the
// numeric smoke could tell you when they stopped.
//
// A ported family states no `operands` at all, so it is on neither side of
// that fork: no shim, no generated arm, no wrapper. The host program's
// parameter list is the ONLY host-side spelling of the kernel's signature,
// and the typecheck TU checks it against the `__global__`.
//
// The nine were: `rope_rope_bf16`, `rope_rope_write_kv_bf16`,
// `rope_qk_rmsnorm_rope_bf16_devwin`, `rope_qk_rmsnorm_mrope_bf16`,
// `rope_qk_rmsnorm_rope_bf16_rounded`, `rope_rope_yarn_bf16`,
// `rope_rope_yarn_original_bf16`, `rope_rope_partial_bf16_position_delta`
// and `rope_rope_partial_last_bf16`. Every measurement in their doc
// comments — the `heads_per_block`/`cache_pairs` host conditionals most of
// all — moved with the fns to `x::rope`, which is where the launch that
// uses them now is.

// ── `ssm/`: ELEVEN WRAPPERS STOOD HERE AND ARE GONE ─────────────────────
//
// `ssm` crossed into fn-world (`.wiki/kernel-x/northstar.md` §5 step 5).
// Its host programs are `kernels-cuda-new/src/x/ssm.rs` — twenty-seven of
// them, in five inline `pub mod`s beside the five `.cuh` they fire — and
// `bind/mod.rs::dispatch` reaches them through `kernels_cuda_new::x::entry`,
// one lookup, no wrapper. `driver-cuda/src/fire/{causal_conv1d,
// gated_delta_net,kda,nemotron_h}.rs` are deleted with them.
//
// The eleven were: `ssm_causal_conv1d_prefill_batched_bf16`,
// `ssm_qwen_gdn_post_conv_prep_bf16`,
// `ssm_recurrent_gated_delta_step_batched_gqa_state_bf16`, the four
// `ssm_chunk_gated_delta_prefill_batched{,_state_bf16,_cached,
// _cached_state_bf16}`, `ssm_nemotron_mamba_split_bf16`,
// `ssm_nemotron_mamba_ssm_batched_bf16`, `ssm_kda_recurrent_step_batched`
// and `ssm_kda_prefill_batched`. Every measurement in their doc comments
// moved with the fns to `x::ssm` — the KDA prefill's block width most of
// all, `min(D, 32) * 32`, one warp per state `v` row, **2.2x at T=2048,
// 26.2 ms -> 12.0 ms per layer at K3's widths**, which is on
// `x::ssm::kda_prefill_batched` beside the `<<<>>>` that uses it.
//
// `ssm_qwen_gdn_post_conv_prep_bf16` is the one that did not go to `x::ssm`:
// it is `x::driver_internal::qwen_gdn_post_conv_prep_bf16`, a `fn` with two
// `fire` calls and no `contract!`, called by `bind/mod.rs`'s GDN path
// directly.
//
// # THE PARAGRAPH THIS BLOCK EXISTED TO WRITE DOWN, and it is now history
//
// *"The parameter lists are the TABLE ROW's, not the C++ launcher's, and
// where they differ the table wins. `abi::emit_rust_dispatch` writes the
// operands in row order including the `Ty::Stream` one, so
// `ssm_causal_conv1d_prefill_batched_bf16` takes its stream in the MIDDLE —
// after `k`, before `write_state` — because that is where `table::ssm` put
// it. A signature that 'tidied' the stream to the end would compile and
// would pass a `bool` where a `cudaStream_t` goes."*
//
// **That hazard is gone by construction and it is the sharpest single
// argument for the port this file can make.** There is no row, so there is
// no row order; the host program's parameter list is the ONLY host-side
// spelling of the kernel's signature and it is the one the `unit!` raw stub
// checks against the `__global__`. Three signatures had to agree here and
// only the numeric smoke could tell you when they stopped; one signature
// cannot disagree with itself.
//
// TWO OF THE ELEVEN WERE UNREACHABLE — the KDA pair state no `Source` on any
// operand, so `emit_rust_dispatch` skipped those rows whole and wrote no arm
// to them. They are `none:` arms in `x::ssm::kda` now, which is the same
// fact said out loud: `Route::Unbound` at MODEL LOAD with the sentence,
// rather than a wrapper nothing called and a comment explaining why.

// `moe::moe_gate_up_decode_gemv_bf16` STOOD HERE, as `pub unsafe fn
// moe_moe_gate_up_decode_gemv_bf16`, and is DELETED with `fire::moe_dispatch`
// -- `x::moe::moe_gate_up_decode_gemv_bf16` is the host program and its
// `bind!` arm is the fire.

// `moe::moe_down_decode_gemv_bf16` STOOD HERE, as `pub unsafe fn
// moe_moe_down_decode_gemv_bf16`, and is DELETED for its twin's reason.

// `moe::transpose_expert_scales_u8` STOOD HERE, as `pub unsafe fn
// moe_transpose_expert_scales_u8`, and is DELETED. `x::moe` keeps the host
// program and declares the symbol a `none:`: weight preparation is not a
// trace statement, which is what its five unsourced operands were saying.

// `moe::build_moe_ptrs_aligned_bf16` STOOD HERE, as `pub unsafe fn
// moe_build_moe_ptrs_aligned_bf16`, and is DELETED. `x::moe` keeps the host
// program -- twenty-one parameters, six of them pointer arrays -- and the
// symbol is a DRIVER OP: a `contract!` with no `Entry` at all,
// `Service::DriverOp` in `execution::SERVED`, a body in `fire::moe_ptrs` and
// an arm in `bind/mod.rs`'s driver-op table. It was a `none:` arm for one day
// in between, and the day is the interesting part -- see the gate the arm's
// comment carries.
//
// THE REASON IS THE SENTENCE THIS COMMENT ALREADY HAD and it did not change:
// **the aligned staging is the driver's arena and not the trace's.** What
// changed is what follows from it. Six pointer arrays with no stated
// consumer -- their only reader is the batched-cuBLAS arm INSIDE
// `moe_grouped_gemm_bf16`, a lowering and not a statement -- are six trace
// values `lower.rs:1911` frees at the first op past the build, so declaring
// them would hand that reader bytes the next allocation owns. A wrong
// answer, not a refusal.

// `moe::reorder_moe_aligned_output_bf16` STOOD HERE, as `pub unsafe fn
// moe_reorder_moe_aligned_output_bf16`, and is DELETED. `x::moe` keeps the
// host program, including the vectorisability fork that chooses between two
// symbols before a single launch, and it BINDS: `Cx::in_rows` landed in
// a41a1df0a and the arm reads `cx.in_rows(1)` for the sorted map's row
// count. This comment said `none: until Cx can be asked for an operand's
// row count` for as long as that was true and for a day after it was not,
// which is the failure mode a tombstone has: it is written beside the
// deletion and nothing re-derives it when the sentence it names comes true.
// `x::moe`'s three remaining `none:` arms are `add_moe_route_bias`,
// `transpose_expert_scales` and `moe_bucket_exact`; thirteen bind and three
// are driver ops. It said FIVE until `build_moe_ptrs_aligned` took
// `Service::DriverOp`, and FOUR until `scatter_add_weighted` was deleted as
// an orphan, which is this same paragraph's subject happening to this same
// paragraph.

// `ssm_build_nemotron_moe_ptrs_decode_batched_bf16` AND
// `ssm_build_nemotron_moe_ptrs_aligned_bf16` STOOD HERE, filed with `moe/`
// rather than with `ssm/` because that is what they feed, and both are GONE
// with the rest of `ssm` — §5 step 5, see the `ssm/` block above.
//
// They are `x::ssm::nemotron_h::build_nemotron_moe_ptrs_{decode_batched,
// aligned}_bf16` now, and they are the family's other two `none:` arms.
//
// # What they recorded, because it is a FINDING and not a status line
//
// *"The `table::ssm` row stays unbound and that is deliberate. §52.3's
// missing `Source::Scratch(name, extent)` still has no word for a slab this
// driver allocated, so no operand is sourced, `emit_rust_dispatch` writes no
// arm, and nothing in a model trace reaches this. What `RUST_SERVED` changed
// is only that the shim no longer emits an entry — which is what let
// `ssm/nemotron_h.cu` be deleted."*
//
// **Still true, and the port does not fix it** — it only moves where the
// sentence is said. A `none:` arm surfaces at MODEL LOAD as `Route::Unbound`
// carrying that reason, instead of a wrapper that compiles, is exported, and
// is called by nothing. The `attn::write_kv_to_pages` note below draws the
// contrast against these two and it still draws it.
//
// The two shapes, kept because they are the only prose on the pair's
// geometry outside `x::ssm`: the DECODE form is one thread per route with
// **`routes = n * top_k` and not `n`** as the bound — the trap
// `nemotron_h.cu:53-94` documents — filling six device-pointer arrays plus
// the router weight copied out as f32; the ALIGNED form is one thread per
// padded block of the sorted MoE layout, `nemotron_h.cu:96-137`, with **four
// guard terms rather than one**, because `block_size`, `hidden` and
// `intermediate` are multipliers inside the kernel's address arithmetic and
// a zero pitch aliases every pointer in the array onto row zero.

// THE FOUR `attn/kv_paged.hpp` SHIM ENTRIES ARE GONE — the three appenders
// and the dequantiser. They crossed into fn-world as `x::attn`'s
// `WRITE_KV_TO_PAGES`, `WRITE_KV_EXPLICIT`, `WRITE_KV_EXPLICIT_DEVWIN` (a
// `none:` arm) and `DEQUANT_KV_ACTIVE`, over `x::attn::kv_paged`'s twenty
// device rows and seven host programs.
//
// A service entry exists to be the target of a GENERATED dispatch arm, and
// `emit_rust_dispatch` writes an arm from a `table` row. Those four rows are
// deleted, so there is nothing left to write the call — which is why these
// go in the same change and not after it. The trace that reached
// `attn_dequant_kv_cache_layer_to_bf16_active` reaches `x::attn`'s `bind!`
// arm now; `dsl.rs:7750` and `lower.rs:1100` are unchanged and did not need
// to change, because a contract resolves the same symbol a row did.
//
// WHAT STAYED, and it is the thing to be careful of: the four FA2 preludes
// below still call
// `crate::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active`
// directly. Those are not traces and no `bind!` arm can serve them — an fp8
// cache must be bf16 before an FA2 row that carries one KV width can read
// it, and the widening is a step of the FA2 host program rather than an op a
// model states. That is why the moved body is a `pub fn`.

// `attn_mla_prepare_bf16` AND `attn_write_mla_to_pages` DELETED WITH THEIR
// CROSSING.
//
// Two thin resolutions over `fire::mla_paged`, which is itself deleted. The
// host programs are `x::attn::mla_prepare_bf16` and
// `x::attn::write_mla_to_pages`; the contracts are `x::attn`'s `MLA_PREPARE`
// and `WRITE_MLA_TO_PAGES`, both `none:` on `Cx::mla_layer`.
//
// These two carried something the other crossings did not: a `MlaCacheLayerView`
// PARAMETER, taken by value and forwarded. `bind::abi`'s type survives --
// `pools::mla_cache` produces it and `bind::abi` declares it -- but nothing
// in `bind::service` names it now, and the only consumer of the value was a
// host program that has moved to a crate which cannot name the type at all.
// Its `Cx` spelling, `x::MlaLayer`, is the same five fields.

// `layout::embed_bf16` — was `layout/embed.hpp`, and that file is DELETED
// with its `.cu` and the whole of `kernels-cuda/csrc/src/layout/` — STOOD
// HERE, as `pub unsafe fn layout_embed_bf16`, and is DELETED.
//
// Its doc read:
//
// > The first launch of every fire: one row of the vocabulary table per
// > token.
// >
// > Body: `crate::fire::embed::embed_bf16`. Classified `Execution::Walk`
// > with `Control::Switch` — `embed<true>` or `embed<false>`, chosen from a
// > 16-byte alignment test on `weight` and `y` plus `hidden % 8`, which also
// > sizes the grid.
//
// §5 step 5 took `layout` into fn-world. The program is
// `kernels_cuda_new::x::layout::embed_bf16`; the switch is
// `x::layout::vectorisable`, a `pub fn` a caller staging its own buffers can
// ask directly; and the bind that reads the operands off a `Cx` is
// `x::layout`'s `EMBED` arm. Nothing calls this wrapper any more: it existed
// to turn a generated dispatch arm's argument list into a `fire::` call.
//
// The one caller-visible fact worth keeping in reach: `weight` and `y` were
// `*const c_void`/`*mut c_void` here and are `*const bf16`/`*mut bf16` in
// the declaration. The opaque spelling was the shim's, because a `pie_k_`
// entry point is `extern "C"`; the typed one is the `.cuh`'s, and it is what
// the typecheck translation unit compares.

// `attn_split_qkv_bf16_devwin` DELETED WITH ITS CROSSING.
//
// A thin resolution over `fire::split_packed::split_qkv_bf16_devwin`, and
// that module is deleted too. The host program is
// `kernels_cuda_new::x::attn::split_qkv_bf16_devwin` and the contract is
// `x::attn`'s `SPLIT_QKV_DEVWIN`, WITH A REAL BIND.
//
// Its doc here carried the precondition that made the whole arrangement
// look forced -- *"The four buffer pointers must be BASE pointers -- the
// kernel windows them itself from `win_d`, so a pre-windowed pointer is
// windowed twice"* -- and the precondition is real. What was wrong is who
// was thought to be unable to meet it. `bind/mod.rs:3973` resolves every arg
// of a `_devwin` kernel at row 0 and says so in as many words, so a bind
// meets it by construction; the sentence belongs on the `fn`, for a caller
// that is not the binder, and that is where it now is.

// `attn_compact_page_csr`, `attn_mtp_shift_hidden_bf16` AND
// `attn_mtp_update_pending_hidden_bf16` DELETED WITH THEIR CROSSING.
//
// Three thin resolutions over `fire::page_compact` and
// `fire::attention_naive`, both of which are deleted. The host programs are
// `x::attn::compact_page_csr`, `::mtp_shift_hidden_bf16` and
// `::mtp_update_pending_hidden_bf16`; the contracts are `x::attn`'s
// `COMPACT_PAGE_CSR`, `MTP_SHIFT_HIDDEN` and `MTP_UPDATE_PENDING_HIDDEN`.
//
// These three needed their UNITS written -- `attn/page_compact.cuh` and
// `attn/attention_naive.cuh` had none -- so unlike the crossings above them
// this one declared a device half for the first time rather than binding a
// `fn` to one that already existed.

// `attn_combine_attn_outputs_bf16` STOOD HERE, and it is gone rather than
// moved: the symbol crossed into fn-world as
// `kernels_cuda_new::x::attn`'s `COMBINE_ATTN_OUTPUTS`, so its `table::attn`
// row is deleted, `emit_rust_dispatch` writes no arm that could call this,
// and a seam with nothing on either side of it is not a seam. Its
// `RUST_SERVED` entry and its `execution::WALKED` classification went in the
// same change.

// `attn_dsv4_boundary_meta_decode`, `attn_dsv4_boundary_meta_paged` AND
// `attn_attention_compressed_paged_bf16` DELETED WITH THEIR CROSSING.
//
// Three thin resolutions over `fire::dsv4_compress`, which is itself deleted:
// the host programs are `x::attn::dsv4_boundary_meta_decode`,
// `::dsv4_boundary_meta_paged` and `::attention_compressed_paged_bf16`, and
// the contracts are `x::attn`'s `DSV4_BOUNDARY_META_DECODE`,
// `DSV4_BOUNDARY_META_PAGED` and `DSV4_ATTENTION_COMPRESSED_PAGED`, all three
// `none:` on the compression RATIO, which no statement and no context carries.
//
// The `RUST_SERVED` entries went in the same index, which is what
// `every_rust_served_symbol_is_spelled_here` requires: it reads this file's
// text, so a symbol that outlives its entry point here goes red immediately.

// `attn_qkv_decode_qk_norm_rope_write_kv_bf16` DELETED WITH ITS CROSSING —
// AND IT WAS THE LAST ROW IN `ROW_TABLES`.
//
// The symbol is `kernels_cuda_new::x::attn`'s `QKV_DECODE_FUSED`, a
// `contract!` and a real `bind!` over the `attn/qkv_fused` unit's eleven
// device rows, with the host program in `x::attn::qkv_fused`.
//
// **THE ONE DISPATCH ON THIS LIST THAT WAS ALREADY LIVE.** Its table row was
// fully sourced — 23 of 23, `stream` included — so this was not a
// `RUST_SERVED` that only frees a `.cu` (§60.7): `abi.rs:810` kept the row,
// `emit_rust_dispatch` wrote an arm, and that arm called this function. Every
// other `attn` deletion above dropped a seam nothing reached; this one moved
// a real dispatch. If anything in the `attn` sweep shows up as a behaviour
// change rather than a link error, it is this symbol.
//
// `_ctx: &DispatchCtx` WAS UNREAD AND ALWAYS WAS, which is the measurement
// that made the crossing a MOVE rather than a driver op: name the resource,
// or it is a move, and there was no cuBLAS handle, communicator, pool,
// allocator or arena to name. `crate::fire::qkv_fused` — the cast-and-forward
// this called through — is deleted with it, having no other caller.
//
// The `RUST_SERVED` entry and the `execution::WALKED` classification went in
// the same index, which is what `every_rust_served_symbol_is_spelled_here`
// requires: it reads this file's text, so a symbol that outlives its entry
// point here goes red immediately.

/// `comm::all_reduce_bf16` — the custom P2P all-reduce.
///
/// Ported from `comm/custom_all_reduce.cu:603-621` by way of
/// [`crate::fire::all_reduce`], which holds the whole lifecycle. **That file
/// is deleted**, and with it `custom_all_reduce.hpp` and
/// `custom_all_reduce_stub.cpp`.
///
/// This is the first row in the tree that is on `execution::SERVED` and
/// `execution::RUST_SERVED` at once, and the pairing is the point: `SERVED`
/// says *the body is one library call*, `RUST_SERVED` says *Rust issues it*.
/// Every other `SERVED` row's library is cuBLAS; this one's is a header-only
/// P2P kernel in a CPM-fetched flashinfer tree, and until that text is
/// vendored the call **declines** — see
/// [`crate::fire::all_reduce::Decline::NoDeviceText`].
///
/// # A decline here is a panic, and that is faithful
///
/// The C++ threw `"custom_all_reduce: not initialised"` and the shim's
/// `catch` aborted. A decline that this arm swallowed would be a silent
/// wrong answer — the reduction would not have happened and every rank would
/// read stale activations. The panic names the refusal, which is the
/// specification for what would fix it.
///
/// # Safety
///
/// `car` is an opaque [`crate::fire::all_reduce::CustomAllReduce`] handle;
/// `input` and `output` address at least `count` bf16 elements on the device.
pub unsafe fn comm_all_reduce_bf16(
    _ctx: &DispatchCtx,
    car: *mut c_void,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let outcome =
        unsafe { crate::fire::all_reduce::all_reduce_bf16(car, input, output, count, stream) };
    if let crate::fire::all_reduce::AllReduce::Declined(why) = outcome {
        panic!("comm::all_reduce_bf16 declined: {why}");
    }
}

/// `comm::all_reduce_residual_rmsnorm_bf16` — all-reduce, residual add and
/// RMSNorm in one landing.
///
/// Ported from `comm/custom_all_reduce.cu:623-662`. The four runtime values
/// that select flashinfer's template point are computed in
/// [`crate::fire::all_reduce::CustomAllReduce::all_reduce_residual_rmsnorm_bf16`],
/// so a decline names the exact instantiation rather than the family.
///
/// # A decline here is a panic
///
/// As above, and more so: this row has no unfused spelling at the call site.
/// `custom_all_reduce.hpp` said it — *"the fused landing IS this kernel, and
/// there is no other way to spell it"* — which is why the header threw on a
/// null handle instead of returning `false`.
///
/// # Safety
///
/// `car` is an opaque handle; `input`, `residual_inout` and `norm_out`
/// address at least `tokens * hidden` bf16 elements, and `rms_gamma` at
/// least `hidden`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn comm_all_reduce_residual_rmsnorm_bf16(
    _ctx: &DispatchCtx,
    car: *mut c_void,
    input: *const c_void,
    residual_inout: *mut c_void,
    rms_gamma: *const c_void,
    norm_out: *mut c_void,
    tokens: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: forwarded unchanged; the caller's assertion is this function's.
    let outcome = unsafe {
        crate::fire::all_reduce::all_reduce_residual_rmsnorm_bf16(
            car,
            input,
            residual_inout,
            rms_gamma,
            norm_out,
            tokens,
            hidden,
            eps,
            stream,
        )
    };
    if let crate::fire::all_reduce::AllReduce::Declined(why) = outcome {
        panic!("comm::all_reduce_residual_rmsnorm_bf16 declined: {why}");
    }
}

// ───────────────────────────────────────────────────────────────────────────
// FLASHINFER'S SIX ENTRY POINTS STOOD HERE, AND THE SIX ARMS BESIDE THEM.
//
// `attn_dispatch_attention_flashinfer_decode`, `..._decode_capture`,
// `..._prefill_bf16`, `..._prefill_capture_bf16`, `..._prefill_custom` and
// `attn_attention_flashinfer_prefill` are now in
// `crate::fire::flashinfer_fa2_dispatch`; the `fa2_*` arms that call them
// are in `crate::bind` beside `window_of` and the driver-op `match`.
//
// **The gate is why, and it is worth stating rather than pointing at.** This
// module became `#[cfg(feature = "bridge")]` at `f38d199c2`, correctly: it
// is *"the consumer that makes the classification cost the C++ its body"*,
// which is `bridge`'s whole subject. These six are not that. They exist so
// their symbols keep firing AFTER `bridge` goes -- each was a
// `RUST_SERVED` entry reached through a generated dispatch arm, and both the
// list and the generator die with the feature. A body that must outlive a
// gate cannot live behind it.
//
// The split ran along the gate and not along the seam: the entry points need
// only `_cuda`-tier types once the unused `_ctx: &DispatchCtx` parameter
// goes, so they went to `fire`; the arms name `AttnCtx` and
// `DispatchRefusal`, which were `bridge`-gated, so they could not follow and
// went to `bind/mod.rs` instead -- which is where `attn_plan` wanted to be
// anyway, beside the `window_of` it calls.
//
// BOTH TYPES ARE `_cuda` AS OF NORTH STAR SEC 6's re-gate, so the reason the
// split ran where it did is retired. The split itself stands: `attn_plan`
// beside `window_of` is the seam, and that was true either way.
//
// What is left in this file is NOT what `bridge` is actually about -- see
// `bind/mod.rs`'s `pub mod service` heading. Nothing here reaches the
// archive; this file is `_cuda`.
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! What can be checked without a device: that the classification and
    //! this module agree about which symbols are here.
}
