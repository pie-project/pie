//! How a stated symbol is EXECUTED -- the driver's business, never the
//! compiler's and never the plan's.
//!
//! # The word "kernel" was doing three jobs
//!
//! [`crate::table`] holds 209 rows and calls every one of them a kernel. Three
//! different things are in there:
//!
//! * **a kernel** -- one symbol, one `<<<>>>`, one instantiation. This is what
//!   [`crate::device::DeviceKernel`] describes and what NVRTC compiles.
//! * **an op** -- one symbol, a HOST PROGRAM over several of our kernels: a
//!   quantize then a GEMM then a dequant, a `switch` over four cache schemes,
//!   a `while` loop issuing one launch per chunk. The kernels underneath are
//!   often migrated already; what a row cannot say is the COMPOSITION.
//! * **a service** -- one symbol served by a library the driver links
//!   (cuBLAS, NCCL, CUTLASS) or by the driver itself (a `cudaMemcpyAsync`
//!   pair). **Never a kernel, and never was.**
//!
//! Conflating them costs one measurement and one decision. The measurement is
//! the migration percentage: a service counted as an unmigrated kernel is a
//! unit of work that will never be done, sitting in a denominator forever, so
//! the number answers a question nobody asked. The decision is worse -- the
//! same conflation is what makes a reader open `gemm/gemm.cpp` looking for a
//! `__global__` to extract, twice.
//!
//! # What this is NOT allowed to change, and why that is the whole design
//!
//! **The distinction is invisible above the driver.** `model-compiler` must
//! not be able to tell whether a symbol is served by cuBLAS or by a JIT'd
//! kernel; a trace states a symbol and a plan states a shape, and both are
//! execution-agnostic on purpose.
//!
//! That was already true and is worth stating as a measurement rather than an
//! intention: **`model-compiler` does not read `KernelSig::launch` at all.**
//! The only two occurrences of `LaunchRule` under `crates/model-compiler/src/`
//! are in comments. What it reads is `name`, `whole`, `needs`, `sink`,
//! `in_place` and `depth_prefix_plan` -- the contract, not the execution.
//!
//! **So [`kernels::KernelSig`] does not change. Not one field.** The execution
//! fact lives HERE, beside the table, exactly as
//! [`crate::device::DeviceKernel`] already does: that struct is "how a JIT row
//! executes" and has never been part of the contract either. [`Execution`]
//! gives it two siblings.
//!
//! # Why a sibling module rather than more of `device.rs`
//!
//! [`crate::device`] is 1,600 lines making ONE argument: what a row must say
//! to name a `__global__` template, and how [`crate::device::Specialisation`]
//! lets one row choose an INSTANTIATION without choosing a geometry. Every
//! type in it is a piece of that argument.
//!
//! This module makes a different one -- a classification OVER symbols, whose
//! `Jit` arm merely points at that argument's output. Two theses in one header
//! is how a header stops being read. The practical half of the same point:
//! step two is `Step`'s vocabulary measured against the ten real compositions,
//! which is several hundred lines of its own, and it belongs beside
//! [`Execution`] rather than in the middle of `Specialisation`'s.
//!
//! # No mechanism here. This is DATA
//!
//! [`Service`] carries a **name only**. Not a handle, not `(transa, transb, m,
//! n, k, alpha, lda, …)`. Stating cuBLASLt's parameter list in
//! [`kernels::Source`] vocabulary would be inventing a binding language for
//! one library -- `new-horizon.md` §10.5's *no vocabulary growth for one
//! kernel*, at the largest scale it could be committed at. The driver already
//! knows how to call cuBLAS; it assembles the arguments from operands it
//! already binds. The row says only WHICH service serves the symbol.
//!
//! This is also why nothing here is behind `cuda-12`/`cuda-13`: there are no
//! cuBLAS types in this file, no `links` key, no dependency. Layers 1 and 2
//! stay compilable on a machine that has never seen a GPU, which is what lets
//! `model-compiler` depend on this crate unconditionally.
//!
//! And [`Service`] deliberately does not route through
//! [`crate::runtime::fire`]. `fire` takes `Dims`, and `Dims` is meaningless
//! for a cuBLAS call -- there is no grid. A shared signature carrying a
//! parameter that has no meaning for one arm is the "present and wrong" hazard
//! §21 measured twice. The uniform execution surface already exists one layer
//! up and it is `driver-cuda`'s `dispatch_generated`, which is a `match` on a
//! symbol and cares about nothing else.

use kernels::{KernelSig, Ty};

use crate::device::{DeviceKernel, Take};
use crate::{table, unit};

/// How a stated symbol is executed.
///
/// Three arms because there are three answers, and the third has never been a
/// kernel. See the module header for what each means and for what none of them
/// is allowed to do (be visible above the driver).
#[derive(Clone, Copy)]
pub enum Execution {
    /// A `__global__` this tree holds, compiled by NVRTC and fired by
    /// [`crate::runtime::fire`]. The row is [`DeviceKernel`] and it says
    /// which template, at which type, with which geometry.
    Jit(&'static DeviceKernel),
    /// A host program over several of our kernels.
    ///
    /// The slice's ORDER is the sequence -- there is no `Sequence` variant,
    /// because a slice already is one. See [`Step`] for the vocabulary the ten
    /// measured compositions demanded, which of it was built, and the two
    /// forms that were refused with the reason each.
    Composed(&'static [Step]),
    /// A library the driver links, or the driver itself.
    ///
    /// A name, never a call and never a parameter list. See [`Service`].
    Service(Service),
}

/// One step of a composition -- **one variant, because the ten measured
/// compositions demand exactly one that can be stated.**
///
/// # A step names a SYMBOL, never an execution
///
/// [`Step::Fire`]`{ symbol: "gemm::act_x_wt_bf16", .. }` names a ROW. That row
/// resolves its own [`Execution`], which may be [`Execution::Jit`],
/// [`Execution::Service`], another [`Execution::Composed`], or -- today, for
/// two of the four step symbols below -- nothing at all, because the symbol is
/// a kernel nobody has migrated. Three consequences, and they are the design:
///
/// 1. **A step never mentions cuBLAS**, because it never mentions an
///    execution. Step one's rule -- *a row states which service serves it, and
///    the driver assembles the arguments* -- is preserved exactly.
/// 2. **[`Kind`] is derived from the TOP-LEVEL execution only.** A symbol
///    whose execution is `Composed` is a [`Kind::Op`] whatever its steps
///    contain. A function that calls `printf` is still your function, not
///    libc's, so the partition stays total and needs no re-derivation.
/// 3. It is step one's information hiding **applied recursively**: a step does
///    not know whether the symbol it fires is JIT or cuBLAS, exactly as the
///    caller does not.
///
/// The counter-case settles it. If a step could not name a service, then *any
/// operation that touches cuBLAS anywhere is forever unmigrable* -- which
/// would make a bias-add unmigrable **because it happens after a GEMM**.
/// `gemm::act_x_wt_bias_bf16` is exactly that shape and is stated in
/// [`COMPOSED`] below.
///
/// # Why `take`, when step one said the driver assembles the arguments
///
/// Because for a composition it demonstrably cannot, and the measurement is
/// one operand wide. `attn::compact_page_csr` states `scratch_counts`;
/// `attn::count_kept` declares `counts`; they are the same buffer. Every other
/// operand of that op reaches its steps under its own name, so the ONLY
/// convention a driver could apply -- match by name -- is correct fourteen
/// times out of fifteen and **silently wrong on the one operand that carries
/// the dependency between the two launches**. A binding that is wrong exactly
/// where the composition lives is `new-horizon.md` §18.4's *well formed and
/// wrong* with the parameter list picked for it.
///
/// So a step says which of the OP's own operands fill the step's own operands,
/// in the step's order, and [`Take`] is reused from [`crate::device`]
/// unchanged -- the same type [`crate::device::Arm`] has used since §21.14, on
/// the same `From`/`Null` vocabulary, checked by the same shape of `agrees`.
/// **No new vocabulary was added for this**, which is the §10.5 test it had to
/// pass.
///
/// # §21.14, applied in writing, to this variant and to the two not here
///
/// The test is *does the new spelling make a wrong program well-formed?*
///
/// * **`Fire { symbol, take }` -- ADMITTED, with the hazard named and
///   answered.** A `Take::From(i)` naming a same-typed operand type-checks:
///   `compact_page_csr` states three `U32s` inputs in a row and transposing
///   two of them is well formed. That is answered twice and neither is
///   optional -- [`Composition::agrees`] requires every take to land on an
///   operand of the same NAME unless the composition writes the rename down
///   (three renames across twenty-six takes, counted by
///   `tests/layers.rs`), and `tests/composition.rs`
///   fires the whole sequence byte-identically at two shapes against a host
///   model. A wrong take is a failing test, not a plausible page list.
/// * **`Choose { when, then }` -- REFUSED FOR THE TEN, and the refusal is
///   narrower than it first reads.** It is refused because zero of the TEN
///   compositions can use it -- not because a choice is unstatable, and not
///   because the arms of a choice must share a geometry. That second point is
///   `Specialisation`'s limit and not composition's, and conflating the two
///   is a mistake this module made once in writing:
///
///   ```text
///   Specialisation   one row,  one rule   -> chooses an INSTANTIATION
///   Composition      N rows,   N rules    -> chooses a ROW
///   ```
///
///   `runtime::fire` evaluates geometry from the BASE row before consulting
///   an arm (`fire.rs:176-186`), which is why `tests/specialise.rs`
///   deliberately refuses an arm that changes a `LaunchRule`. A `Choose` over
///   two ROWS has no such constraint: each row carries its own `LaunchRule`
///   and its own evidence. See `new-horizon.md` §26.9 for the twelve rows
///   that turn on this, and §26.10 for the predicate shapes they need.
///
///   Of the ten, four branch. `attn::write_kv_to_pages` and
///   `attn::dequant_kv_cache_layer_to_bf16_active` switch on `layer.scheme`,
///   and `layer` is ONE operand of type [`Ty::KvCacheLayerView`] --
///   [`crate::device::Fact::Opaque`], with no `Term` that reads a struct
///   field and no operand to read. `gemm::act_x_wt_channel_scaled` branches
///   on `beta == 0.f`, and no `Fact` carries a float.
///   `norm::rmsnorm_bf16_with_fp16`'s three arms are all statable --
///   `Term::Present` on `y_fp16`, `Term::Multiple { of: 8 }` on `hidden`,
///   `Term::Aligned { bytes: 16 }` on three pointers, every one of them
///   already proven -- and the op is still refused, because the arm those
///   predicates SELECT (`rmsnorm_vec8<512, false, EMIT_FP16=true>`) is an
///   instantiation `families/norm.rs` does not carry. A `Choose` whose middle
///   branch has no row would send an aligned fire to the scalar fallback,
///   which is a different reduction order and therefore different bits: the
///   §21.14 failure exactly, bought with new vocabulary. So the predicate
///   vocabulary was measured to be SUFFICIENT and the row set to be
///   insufficient, and no enum variant fixes that.
/// * **`Repeat { times, advance }` -- REFUSED, and it is the clearest of the
///   three.** `ssm::chunk_gated_delta_prefill` was `for (t = 0; t < T; ++t)`
///   over `recurrent_gated_delta_step(q + t*V_h*K_d, k + t*V_h*K_d,
///   v + t*V_h*V_d, g + t*V_h, beta + t*V_h, state, out + t*V_h*V_d, ...)`.
///   Five of seven pointers advance, one is carried, and the strides are
///   PRODUCTS of three other operands. Spelling that needs a per-operand
///   pointer-advance vocabulary and an arithmetic-over-operands vocabulary --
///   which is [`kernels::Source`] re-invented inside `Step` -- for two rows,
///   both of which then need a per-step [`crate::runtime::Dims`] as well
///   because the loop body is `B = 1` and the op's rows are `T`. §10.5 says
///   no vocabulary growth for one kernel; this is three growths for two.
///   **Both rows have since gone** as §28.4 duplicates of the `_batched`
///   forms every golden actually names; the C++ loop is still in
///   `ssm/gated_delta_net.cu` and the refusal is still what a row for it
///   would have cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Fire a symbol this tree already states, over the op's own operands.
    Fire {
        /// The symbol. Whatever executes it is that symbol's business.
        symbol: &'static str,
        /// Where each of the step's operands comes from, in the STEP's order,
        /// indexing the OP's operand list.
        ///
        /// Checked against [`Composition::sig_of`]'s answer for `symbol`,
        /// which prefers the JIT row over the table row when both exist --
        /// because the JIT row is what a fire binds. The two genuinely differ:
        /// `norm::add_bias_bf16` is five operands in `table/driver_internal.rs`
        /// and three in `families/norm.rs`, the launcher's `num_rows` and
        /// `stream` having become the fire's geometry and the fire's stream.
        take: &'static [Take],
    },
}

impl Step {
    /// The symbol this step fires.
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Step::Fire { symbol, .. } => symbol,
        }
    }

    /// The step's argument map.
    #[must_use]
    pub fn take(&self) -> &'static [Take] {
        match self {
            Step::Fire { take, .. } => take,
        }
    }
}

/// One op: a symbol, the steps it is, and the launcher they were read off.
///
/// The order of [`Composition::steps`] IS the sequence, and that is the one
/// piece of vocabulary the ten compositions all needed and none of them had to
/// pay for. Five of the ten are sequences (`act_x_wt_bias_bf16`,
/// `act_x_wt_channel_scaled`, `act_x_wt_grouped_scaled`, `compact_page_csr`,
/// and `write_kv_to_pages`'s native arm); a `Sequence` variant would have
/// added a name for the thing a slice already is.
pub struct Composition {
    /// The stated symbol this composes -- a row of [`crate::table`].
    pub symbol: &'static str,
    /// The steps, in launch order, on one stream.
    pub steps: &'static [Step],
    /// Every operand whose name CHANGES between the op and a step, written
    /// down so that a same-typed transposition cannot hide.
    ///
    /// `(op's name, step's name)`. [`Composition::agrees`] requires every
    /// [`Take::From`] to land on an operand of the same name or on a stated
    /// rename, which is what turns "three `U32s` in a row" from a silent
    /// hazard into a compile-time-checkable fact. A pair licenses that
    /// renaming anywhere in this composition, which is a real looseness and a
    /// bounded one: the longest composition here is two steps.
    pub renames: &'static [(&'static str, &'static str)],
    /// The launcher, with lines, so the sequence can be checked against the
    /// C++ that is still running in the archive. [`crate::device::Arm`]'s
    /// `because` for the same reason: what a machine in this crate cannot
    /// check is whether the STEPS say what the launcher says.
    pub because: &'static str,
}

/// The ops this crate can state -- **two of the ten, and the eight refusals
/// are in `examples/migration_status.rs` with the line that decides each.**
///
/// # Why the numerator is two and not more
///
/// Every one of the ten was re-derived from its launcher before any vocabulary
/// was written, which is the order `new-horizon.md` §25.5 asked for. What the
/// ten need, measured:
///
/// | op | control flow, read off the launcher | statable |
/// |---|---|---|
/// | `attn::compact_page_csr` | two launches, one stream, second reads the first's buffer | **yes** |
/// | `gemm::act_x_wt_bias_bf16` | a GEMM then `norm::add_bias_bf16`, plus an `M == 1` fused-tactic arm | **yes**, minus the tuner |
/// | `gemm::act_x_wt_channel_scaled` | quantize, INT8 `cublasGemmEx`, dequant, and `residual_add` when `beta != 0` | no |
/// | `gemm::act_x_wt_grouped_scaled` | shape decline, quantize, `cublasLtMatmul`, and a latch on the heuristic | no |
/// | `gemm::act_x_wt_mxfp4_marlin` | `while (rest_m)`, 1 of 15 instantiations per iteration | no |
/// | `norm::rmsnorm_bf16_with_fp16` | three arms on `y_fp16 == nullptr` and `rmsnorm_vec8_ok` | no -- one arm has no row |
/// | `attn::write_kv_to_pages` | a throw, then a 5-way `switch (layer.scheme)`, one arm of which is itself two launches | no |
/// | `attn::dequant_kv_cache_layer_to_bf16_active` | a 5-way `switch (layer.scheme)`, one launch each | no |
///
/// # The open question step one left, answered by measurement rather than by
/// the decision
///
/// *May a `Step` name a service?* Yes -- see [`Step`]'s header for why the
/// alternative makes a bias-add unmigrable for happening after a GEMM. But the
/// measured finding is that **not one of the ten has a step that names a
/// stated service symbol**, and the reason is sharper than the question
/// assumed. `gemm::act_x_wt_channel_scaled`'s middle step is a raw
/// `cublasGemmEx` written inline in `gemm.cpp:2091-2107`; there is **no INT8
/// GEMM symbol anywhere in [`crate::table`]** for a step to name. So that op
/// is unspellable not because a step may not name a service, but because the
/// service it would name **has no row**. The decision is right and, over these
/// ten, unexercised: the closest case is `gemm::act_x_wt_bias_bf16`, whose
/// first step names a symbol that is neither JIT nor `Service` today -- it is
/// an unmigrated kernel behind `Wall::HostChoice` -- which demonstrates the
/// same information hiding one arm further out.
#[rustfmt::skip]
pub static COMPOSED: &[Composition] = &[
    // ── the one that fires end to end ────────────────────────────────────
    //
    // `attn/page_compact.cu:42-51`, and the header beside it already said
    // what a row could not: *"The two are also ORDERED: `scan_and_scatter`
    // reads the `counts` buffer `count_kept` fills, on the same stream. A row
    // per kernel states two geometries and no dependency, so whatever states
    // these will have to state that too."* This is that statement.
    //
    // Both steps are `LaunchRule::PerRow` over `num_requests`, so ONE `Dims`
    // serves the whole composition and no per-step geometry vocabulary is
    // needed. That is measured, not assumed, and it is why `Composed` carries
    // no `Dims`: a step that wanted a different one would be a refusal.
    Composition {
        symbol: "attn::compact_page_csr",
        steps: &[
            // `page_compact.cu:45` -- `count_kept<kBlock><<<num_requests,
            // kBlock, 0, stream>>>(page_indptr_in, keep, keep_stride,
            // num_requests, scratch_counts)`.
            Step::Fire {
                symbol: "attn::count_kept",
                take: &[Take::From(1), Take::From(3), Take::From(5), Take::From(6), Take::From(4)],
            },
            // `page_compact.cu:48` -- and note the last three: the launcher
            // passes `page_indptr_out, last_page_lens_out, page_indices_out`
            // in THAT order, which is not the order the op declares them in.
            // The op says `..._indices_out, ..._indptr_out, ..._lens_out`;
            // the kernel says `..._indptr_out, ..._lens_out, ..._indices_out`.
            // All three are `U32sMut`, so the transposition type-checks, and
            // the name check is what refuses it.
            Step::Fire {
                symbol: "attn::scan_and_scatter",
                take: &[
                    Take::From(0), Take::From(1), Take::From(2), Take::From(3), Take::From(4),
                    Take::From(5), Take::From(6), Take::From(8), Take::From(9), Take::From(7),
                ],
            },
        ],
        // The one operand whose name changes, and it is exactly the one that
        // carries the dependency between the two launches.
        renames: &[("scratch_counts", "counts")],
        because: "`attn/page_compact.cu:42-51`: `count_kept<kBlock><<<num_requests, kBlock, 0, stream>>>` \
                  then `scan_and_scatter<kBlock><<<num_requests, kBlock, 0, stream>>>`, the second reading \
                  the `counts` buffer the first fills. The launcher's `if (num_requests <= 0 || \
                  scratch_counts == nullptr) return;` is not a step: the first half is \
                  `Ungeometric::Empty` from `Dims::rows`, which every rule already answers",
    },

    // ── the demonstration case, stated and not fireable ──────────────────
    //
    // `gemm/gemm.cpp:2391-2394`. A bias-add that happens AFTER a GEMM, which
    // is the shape that would be unmigrable forever if a step could not name
    // a symbol whose execution is not this crate's.
    //
    // What is NOT stated here is the `M == 1` arm at `:2383-2390`: a fused
    // GEMV+bias epilogue whose tactic comes from `dense_tactic_for`, a
    // run-time tuning cache consulted under `cudaStreamIsCapturing`. A tuner
    // is not a predicate over a fire's operands and no `Fact` carries one. So
    // this composition states the GENERAL arm, and `migration_status.rs`
    // records the fused arm as the reason the row keeps its wall.
    Composition {
        symbol: "gemm::act_x_wt_bias_bf16",
        steps: &[
            // `:2395` -- `gemm_bf16_impl(handle, act, W, y, M, N, K, beta)`,
            // which is `gemm::act_x_wt_bf16`'s own body. `bias` (operand 3)
            // is the one the op carries and this step does not.
            Step::Fire {
                symbol: "gemm::act_x_wt_bf16",
                take: &[
                    Take::From(0), Take::From(1), Take::From(2), Take::From(4),
                    Take::From(5), Take::From(6), Take::From(7), Take::From(9),
                ],
            },
            // `:2397` -- `kernels::norm::add_bias_bf16(y, bias, M, N, stream)`,
            // a kernel THIS CRATE ALREADY FIRES. Three operands and not five:
            // the JIT row folded `num_rows` into `LaunchRule::RouteRows` and
            // `stream` into the fire, which is why `sig_of` must prefer it.
            Step::Fire {
                symbol: "norm::add_bias_bf16",
                take: &[Take::From(4), Take::From(3), Take::From(6)],
            },
        ],
        renames: &[("y", "out"), ("n", "dim")],
        because: "`gemm/gemm.cpp:2395-2398`: `gemm_bf16_impl(handle, act, W, y, M, N, K, beta)` and then \
                  `if (bias != nullptr) kernels::norm::add_bias_bf16(y, bias, M, N, stream)`. The `bias != \
                  nullptr` guard is `Term::Present` and statable; it is not stated because this row's \
                  `bias` is `Source::Weight(1)` and a fire that had no bias would state \
                  `gemm::act_x_wt_bf16`. The `M == 1` fused-tactic arm at `:2383-2390` is NOT here -- \
                  `dense_tactic_for` is a run-time tuning cache, not a predicate over operands",
    },
];


/// WHO serves a symbol. A name, and nothing else.
///
/// # The parameters are not here, and that is the point
///
/// The obvious next field is cuBLASLt's argument list. It must not be added.
/// A `Service` that carried `(handle, transa, transb, m, n, k, alpha, A, lda,
/// B, ldb, beta, C, ldc, compute_type, algo)` would be a binding language for
/// one library, spelled in [`kernels::Source`]'s vocabulary, growing every
/// time a second library was rowed -- and it would have to be READ by
/// something, which is a mechanism this task deliberately does not build.
/// The driver assembles the call from operands it already binds.
///
/// # Why five names rather than one, and why not six
///
/// Each variant below has at least one member with a citation, and
/// `tests/layers.rs` asserts that -- a variant with no member is vocabulary
/// admitted without evidence, which is the failure this whole classification
/// is arranged against.
///
/// There is no `CublasLt` variant for exactly that reason. cuBLASLt is
/// everywhere in `gemm/gemm.cpp`, but every entry point whose ONLY execution
/// is an Lt call turned out to be an entry point that also has a non-Lt arm
/// reaching one of our kernels -- so those rows are ops, not services, and a
/// `CublasLt` variant would have had no members. The five names below are the
/// ones the evidence supports today.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Service {
    /// Classic cuBLAS -- `cublasGemmEx` and its batched/grouped/strided
    /// siblings. `driver-cuda/build.rs` links `cublas`.
    Cublas,
    /// CUTLASS, through FlashInfer's grouped-GEMM MoE runner. The kernels are
    /// templates in headers this repo does not contain: CPM fetches them at
    /// configure time.
    Cutlass,
    /// NCCL. `driver-cuda/build.rs:422` links `nccl`; `kernels-cuda` neither
    /// includes `nccl.h` nor links it, and `csrc/src/dist/` does not exist.
    Nccl,
    /// The P2P/NVLink all-reduce plane -- vLLM's custom all-reduce and
    /// TensorRT-LLM's fused residual+RMSNorm landing, both reached through
    /// FlashInfer's `comm/` headers, which are fetched rather than vendored.
    ///
    /// Distinct from [`Service::Nccl`] because the two are ALTERNATIVES the
    /// text chooses between, which `comm/custom_all_reduce.hpp:163-166` says
    /// in as many words: *"`dist::all_reduce_bf16` -- NCCL -- is the other,
    /// and WHICH is a guard in the text rather than an `if` inside a driver
    /// method."* Collapsing them would erase a decision a model text makes.
    CustomAllReduce,
    /// The driver itself. A `cudaMemcpyAsync` pair, a staged LoRA apply built
    /// out of GEMM calls the driver already had -- an operation of the
    /// declared executor, with no C++ function anywhere to describe.
    /// `driver-cuda`'s `launch_abi.rs` calls these `Unstated::NotACppFunction`
    /// and says of them: *"never closes"*.
    DriverOp,
}

impl Service {
    /// Every variant, so a test can assert each has a member.
    pub const ALL: &'static [Service] =
        &[Service::Cublas, Service::Cutlass, Service::Nccl, Service::CustomAllReduce, Service::DriverOp];

    /// What to print. Not a `Display` impl, because this is a label in a
    /// report and not a rendering of a value anybody parses.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Service::Cublas => "cuBLAS",
            Service::Cutlass => "CUTLASS (flashinfer MoE)",
            Service::Nccl => "NCCL",
            Service::CustomAllReduce => "custom all-reduce (vLLM/TRT-LLM)",
            Service::DriverOp => "the driver itself",
        }
    }
}

/// What a symbol IS, as opposed to who runs it.
///
/// The three of the module header. Derived from [`Execution`] by [
/// `Execution::kind`], which is the `match` that makes a fourth arm of
/// [`Execution`] a compile error rather than a silent reclassification.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Kind {
    /// One `<<<>>>`, one instantiation. Migrable, whether or not migrated.
    Kernel,
    /// A host program over several of our kernels.
    Op,
    /// A library, or the driver. Never a kernel.
    Service,
}

impl Execution {
    /// Which of the three kinds this execution makes the symbol.
    ///
    /// The whole point of the enum in one function: a variant added tomorrow
    /// fails to compile HERE, and whoever adds it has to say which kind it
    /// makes a symbol before the report can count it.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Execution::Jit(_) => Kind::Kernel,
            Execution::Composed(_) => Kind::Op,
            Execution::Service(_) => Kind::Service,
        }
    }
}

impl core::fmt::Debug for Execution {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Execution::Jit(row) => write!(f, "Jit({})", row.sig.symbol),
            Execution::Composed(steps) => write!(f, "Composed({} step(s))", steps.len()),
            Execution::Service(service) => write!(f, "Service({service:?})"),
        }
    }
}

/// The stated symbols a SERVICE executes, with the evidence for each.
///
/// # The bar every entry had to clear
///
/// One of two facts, checked in the tree rather than read off a call site:
///
/// * **no `__global__` anywhere in the symbol's closure**, so there is nothing
///   to extract; or
/// * **a library whose source is not in this repository**, so what would be
///   extracted is not ours.
///
/// That bar is not decorative. `Wall::Library`'s stated meaning -- *"there is
/// nothing to extract"* -- was falsified FOUR times in one session by agents
/// who read a call site (`new-horizon.md` §23: Marlin twice, FlashInfer's
/// cascade merge, FlashInfer's Mamba SSU), and applying it here falsified it
/// six more times: every dense/quantized `gemm::` entry point that a previous
/// table called cuBLASLt turns out to reach `gemm/gemv.cu`'s warp-per-row
/// GEMV, or `quant::`'s quantize/dequant kernels, or `norm::add_bias_bf16` --
/// all of them ours, several of them already migrated. Those six are NOT
/// here; they are ops, and `examples/migration_status.rs` records each with
/// the line that decides it.
///
/// # Citations point at the archive, not at this crate
///
/// `crates/kernels-cuda/csrc/**` is C++ nobody is editing, so a reason that
/// cites it can be checked at any commit by anyone. A reason citing
/// `families/*.rs` would be measuring an edit in flight.
#[rustfmt::skip]
pub static SERVED: &[(&str, Service, &str)] = &[
    // ── cuBLAS: one library call, and nothing else in the body ───────────
    //
    // `gemm/gemm.cpp` holds 0 `__global__` -- it is host C++ compiled by
    // `g++` -- but that alone proves nothing, because it CALLS kernels of
    // ours from three places (`:544`, `:962`, `:2356` reach `gemv.cu`;
    // `:1814`, `:1855`, `:1912`, `:2085`, `:2122`, `:2263` reach `quant::`;
    // `:2130`, `:2224` reach `norm::residual_add_bf16`; `:2393` reaches
    // `norm::add_bias_bf16`). The four below are the entry points whose
    // bodies reach NONE of them.
    ("gemm::act_x_wt_bf16_out_fp32",       Service::Cublas,
     "one `cublasGemmEx`, bf16 in / fp32 out; `gemm.cpp:1030-1058` is the whole body"),
    // `gemm::batched_act_x_wt_bf16` WAS HERE. The service was real --
    // `gemm.cpp:1145-1241` is `cublasGemmGroupedBatchedEx` falling back to
    // `cublasGemmBatchedEx`, both arms the library's -- and the row is gone
    // anyway (`new-horizon.md` §38), because nothing asked for it. A true
    // statement about a launcher is not a statement that anything calls it.
    ("gemm::grouped_act_x_wt_bf16",        Service::Cublas,
     "one `cublasGemmGroupedBatchedEx`; `gemm.cpp:1242-1294`. Measured, not read: it is CLASSIC cuBLAS, not the cuBLASLt the previous entry claimed"),
    ("gemm::mla_absorb_q_to_latent_bf16",  Service::Cublas,
     "one `cublasGemmStridedBatchedEx` over the head axis; `gemm.cpp:2419-2442`. Its own comment names the per-head scalar kernels it REPLACED"),
    ("gemm::mla_absorb_latent_to_v_bf16",  Service::Cublas,
     "the second absorb, same single strided-batched call; `gemm.cpp:2444-2468`"),

    // ── CUTLASS ───────────────────────────────────────────────────────────
    ("moe::flashinfer_cutlass_moe_bf16",   Service::Cutlass,
     "THE EXEMPLAR. `csrc/third_party/flashinfer_moe/*.cu` holds 0 `__global__`; `src/moe/flashinfer_moe.cu` holds 0 and calls no kernel of ours; and `cutlass/` is in no source directory of this repo -- CPM fetches it into `target/**/_deps/flashinfer-src/3rdparty/cutlass` at configure time. The kernels are templates in headers we do not have. It returns `bool`, but a service that declines is still a service: the fallback is the CALLER's, not the row's"),

    // ── NCCL: not in this crate at all ────────────────────────────────────
    //
    // `csrc/src/dist/` does not exist, `kernels-cuda` neither includes
    // `nccl.h` nor links NCCL, and `driver-cuda/build.rs:422` is what emits
    // `cargo:rustc-link-lib=nccl`. These are methods on the DRIVER's
    // `NcclComm` -- `launch_abi.rs` calls them `SecondNamespaceRoot` and
    // asserts, for each, that `csrc/src/<family>/` is absent.
    ("dist::all_reduce_bf16",              Service::Nccl,
     "NCCL all-reduce, in place; no `csrc/src/dist/` exists, so there is no C++ of ours to extract"),
    ("dist::all_reduce_bf16_out",          Service::Nccl,
     "NCCL all-reduce, out of place; same absent directory, same `NcclComm` method"),
    ("dist::all_gather_bf16",              Service::Nccl,
     "NCCL all-gather; `driver-cuda/build.rs:422` links `nccl` and `kernels-cuda` does not include `nccl.h` at all"),

    // ── the P2P all-reduce plane ──────────────────────────────────────────
    //
    // `comm/` DOES exist and DOES hold one `__global__`
    // (`cross_device_reduce_residual_rmsnorm_1stage_exact_bf16`,
    // `custom_all_reduce.cu:118`) -- and it backs
    // `all_reduce_residual_rmsnorm_bf16_exact`, which HAS NO TABLE ROW. Both
    // rowed entry points take a `CustomAllReduce*` the driver owns as their
    // first argument and forward into headers this repo does not carry:
    // `csrc/vendor/flashinfer/` holds `attention/` only, and there is no
    // in-repo copy of `flashinfer/comm/vllm_custom_all_reduce.cuh` or
    // `flashinfer/comm/trtllm_allreduce_fusion.cuh`.
    ("comm::all_reduce_bf16",                   Service::CustomAllReduce,
     "`car->all_reduce_bf16` -> `impl_->allreduce<__nv_bfloat16>`, vLLM's one/two-shot NVLink kernel; `custom_all_reduce.cu:603-621`, header fetched not vendored. A null `car` is a REFUSAL, not a fallback (`custom_all_reduce.hpp:167-177`)"),
    ("comm::all_reduce_residual_rmsnorm_bf16",  Service::CustomAllReduce,
     "`flashinfer::trtllm_allreduce_fusion`'s `kARResidualRMSNorm` pattern; `custom_all_reduce.cu:623-662`. Throws when `can_fuse_residual_rmsnorm` is false -- the fused landing IS this kernel and there is no other way to spell it"),

    // ── the driver itself ─────────────────────────────────────────────────
    //
    // Pseudo-symbols: no C++ function exists, so `KernelSig::operands` is
    // EMPTY for all three and `launch_abi.rs`'s
    // `the_unstated_rows_are_exactly_the_ones_with_a_written_reason` pins
    // exactly this set as `Unstated::NotACppFunction`, of which it says
    // "never closes".
    ("qwen35_verify_stash_store", Service::DriverOp,
     "a `cudaMemcpyAsync` trio moving a layer's in-proj triple into the verify stash; `executor_bind.rs::AWAITING_THE_VERIFY_STASH_POOL`. No launcher, no grid"),
    ("qwen35_verify_stash_load",  Service::DriverOp,
     "the load half of the same trio, moving the stash back into the workspace"),
    ("pie_lora_qkv_correction",   Service::DriverOp,
     "the driver's own arm: `bind/mod.rs:1895` calls `(*state).apply(ctx.cublas, ...)`, a staged LoRA apply built out of grouped GEMM calls the driver already had. With no adapters staged it does NOTHING, which is an answer a `__global__` could not give"),
];

/// **The rows the DRIVER executes, in Rust — the consumer side of the
/// classification above.**
///
/// [`SERVED`] and [`COMPOSED`] are *findings*: they say what a body IS. This
/// list is an *instruction*: it says what the driver has taken over. Until
/// §45 the two were the same size as each other and zero as an effect,
/// because every one of those bodies was still called from `gemm/gemm.cpp`
/// — and a C++ translation unit calling a C++ launcher is a composition no
/// Rust dispatch can intercept.
///
/// # What being on this list does
///
/// Exactly what being in `JIT_DISPATCHED` does, one door over:
///
/// * [`crate::abi::emit_c_shim`] SKIPS the row, so no `pie_k_*` entry is
///   generated for it, so **the C++ body can be deleted**. That is the whole
///   mechanism; everything else follows from it.
/// * `emit_rust_bindings` skips it too, so `launch_bindings.rs` does not
///   declare an entry point the archive no longer defines —
///   `driver-cuda/build.rs`'s "a declaration with no definition is only
///   legitimate for a routed row" check would otherwise fire, correctly.
/// * [`crate::abi::emit_rust_dispatch`] writes the row's arm against
///   `driver-cuda`'s `bind::service` instead of against `bind::abi::ffi`,
///   from the SAME operand list, with the same guard and the same staging.
///
/// # What it must not do
///
/// **The model compiler must not be able to tell whether a symbol is cuBLAS
/// or a JIT'd kernel.** Nothing reads this list above the dispatcher: not a
/// lowering, not a `Ty`, not a `KernelSig`. A row moving onto it changes
/// which module the generated arm names and nothing else, which is why the
/// four cuBLAS rows below kept their operand lists verbatim when they moved
/// — minus the `handle`, because a handle is the SERVICE's, not the
/// statement's, and `Ty::CublasHandle` in a row is the vocabulary leaking
/// one backend's library into a table two backends share.
///
/// # Why it is a separate list and not `service(sym).is_some()`
///
/// Because most of [`SERVED`] must NOT be skipped. `comm::all_reduce_bf16`
/// and `comm::all_reduce_residual_rmsnorm_bf16` are `Service::CustomAllReduce`
/// and their shim entries are live; `moe::flashinfer_cutlass_moe_bf16` is
/// CUTLASS in a C++ file this crate compiles. Dropping their entries because
/// the classification is true of them would delete the only path they have.
/// A finding is not a plan.
///
/// # The invariant
///
/// Every symbol here is in [`SERVED`] or [`COMPOSED`] — you may not take over
/// a row whose execution has not been classified — and `tests::` below
/// asserts it. The other half of the invariant, that `driver-cuda` actually
/// spells a function for each, is asserted in `driver-cuda`'s own
/// `bind::service`, which is the crate that would fail to compile.
pub static RUST_SERVED: &[&str] = &[
    // The four pure-cuBLAS bodies. Each is one library call and argument
    // assembly; `bind/service.rs` is that assembly in Rust, ~130 lines of
    // C++ for ~120 of Rust and no third place for the transposes to live.
    "gemm::act_x_wt_bf16_out_fp32",
    "gemm::grouped_act_x_wt_bf16",
    "gemm::mla_absorb_q_to_latent_bf16",
    "gemm::mla_absorb_latent_to_v_bf16",
    // The composition [`COMPOSED`] already stated, executed. This one is
    // here for a reason the other four are not: `gemm.cpp:2393` called
    // `norm::add_bias_bf16`, a row of OURS that is already migrated, and
    // while that call existed the row could not be routed and
    // `norm/add_bias.cuh` could not be its only copy. Taking the composition
    // over is what frees it.
    "gemm::act_x_wt_bias_bf16",
];

/// The service that executes a symbol, if a service does.
///
/// A linear scan over fourteen entries, on a path nothing hot takes -- the
/// same trade [`crate::device::specialisation`] documents.
#[must_use]
pub fn service(symbol: &str) -> Option<Service> {
    SERVED.iter().find(|(s, _, _)| *s == symbol).map(|(_, service, _)| *service)
}

/// The composition that executes a symbol, if one does.
#[must_use]
pub fn composition(symbol: &str) -> Option<&'static Composition> {
    COMPOSED.iter().find(|c| c.symbol == symbol)
}

/// The operand list a fire of `symbol` would bind.
///
/// **The JIT row wins over the table row**, and the two genuinely differ: the
/// table states a LAUNCHER's C++ signature and a `DeviceKernel` states a
/// `__global__`'s. `norm::add_bias_bf16` is `(out, bias, num_rows, dim,
/// stream)` in `table/driver_internal.rs:134` and `(out, bias, dim)` in
/// `families/norm.rs:423`, because `num_rows` became
/// [`kernels::LaunchRule::RouteRows`] and `stream` became an argument of
/// [`crate::runtime::fire`] itself. A [`Step`] whose `take` was checked
/// against the wrong one of those would bind five cells for a three-parameter
/// kernel, and `cuLaunchKernel` reads the count out of the cubin and
/// **succeeds** -- which is §21.14's own hazard, one layer up.
///
/// `None` for a symbol that is neither, which is a symbol nothing states.
#[must_use]
pub fn sig_of(symbol: &str) -> Option<&'static KernelSig> {
    if let Some((_, unit)) = unit::unit_of(symbol) {
        if let Some(row) = unit.row(symbol) {
            return Some(row.sig);
        }
    }
    table::sig(symbol)
}

/// The `const` view of a pointer type, for the one narrowing a composition
/// needs and the one it must never get.
///
/// A step that READS what an earlier step WROTE is the definitional shape of a
/// composition -- `attn::compact_page_csr` states `scratch_counts: U32sMut`
/// and `attn::scan_and_scatter` declares `counts: U32s` -- so
/// [`Composition::agrees`] has to accept `T* -> const T*`. It accepts nothing
/// else, and the asymmetry is the whole content:
///
/// * `T* -> const T*` is a conversion C++ performs implicitly at every call
///   site in the tree. Admitting it admits no program the language does not.
/// * `const T* -> T*` is the direction that lets a step WRITE THROUGH an
///   operand the op declared read-only, which is a wrong program that runs.
///   It is refused, and `tests/layers.rs` asserts the refusal rather than
///   leaving it to this table's shape.
///
/// §21.14's test, applied: the spelling makes exactly the legal conversion
/// well-formed and leaves the illegal one a failing check.
fn read_only(ty: Ty) -> Option<Ty> {
    Some(match ty {
        Ty::BufMut => Ty::Buf,
        Ty::U32sMut => Ty::U32s,
        Ty::I32sMut => Ty::I32s,
        Ty::F32sMut => Ty::F32s,
        Ty::U8sMut => Ty::U8s,
        Ty::U16sMut => Ty::U16s,
        Ty::I8sMut => Ty::I8s,
        _ => return None,
    })
}

impl Composition {
    /// Everything about this composition a machine can check, checked.
    ///
    /// Callable with no device, no cubin and no driver, and called by
    /// `tests/layers.rs` on every entry of [`COMPOSED`] -- which is the
    /// feature-free target, so this runs on a machine that has never seen a
    /// GPU. [`crate::device::Specialisation::agrees`] is the model and most of
    /// the checks are the same ones; what is different is stated at each.
    ///
    /// # Errors
    ///
    /// A sentence naming the step and what does not line up.
    pub fn agrees(&self) -> Result<(), String> {
        let op = table::sig(self.symbol)
            .ok_or_else(|| format!("`{}` is composed and is not a row of `table::KERNELS`", self.symbol))?;
        if self.steps.len() < 2 {
            return Err(format!(
                "`{}` composes {} step(s) -- a SEQUENCE of one launch is a kernel and of none is \
                 nothing, and neither is an op. This check counts LAUNCHES, and that is only the \
                 right count while every `Step` is one: a choice-shaped step would name two rows \
                 in one step, and this rule would have to be re-derived as *fewer than two ROWS* \
                 rather than repaired. It is written this way deliberately -- an op that chooses \
                 between two kernels at one launch site is NOT excluded by anything here except \
                 the absence of a variant to spell it",
                self.symbol,
                self.steps.len()
            ));
        }
        if self.because.len() < 40 {
            return Err(format!("`{}` is composed on a citation too short to check", self.symbol));
        }
        for step in self.steps {
            let symbol = step.symbol();
            let at = format!("`{}` step `{symbol}`", self.symbol);
            if symbol == self.symbol {
                return Err(format!("{at} fires itself"));
            }
            let sig = sig_of(symbol)
                .ok_or_else(|| format!("{at} names a symbol no row and no unit states"))?;
            let take = step.take();
            if take.len() != sig.operands.len() {
                return Err(format!(
                    "{at} takes {} arguments and the row that would fire declares {}",
                    take.len(),
                    sig.operands.len()
                ));
            }
            for (slot, take) in take.iter().enumerate() {
                let wants = sig.operands[slot];
                match take {
                    Take::From(index) => {
                        let Some(source) = op.operands.get(*index) else {
                            return Err(format!(
                                "{at} fills `{}` from operand {index} of an op with {}",
                                wants.name,
                                op.operands.len()
                            ));
                        };
                        // The one relaxation `Specialisation::agrees` does not
                        // make, and the reason is in `read_only`'s header: a
                        // composition's steps pass buffers to each other.
                        if source.ty != wants.ty && read_only(source.ty) != Some(wants.ty) {
                            return Err(format!(
                                "{at} fills `{}` ({:?}) from `{}` ({:?})",
                                wants.name, wants.ty, source.name, source.ty
                            ));
                        }
                        // The check that answers §21.14 for `take` itself. An
                        // index is a number and every same-typed operand is a
                        // legal value for it; a NAME is not.
                        if source.name != wants.name
                            && !self.renames.contains(&(source.name, wants.name))
                        {
                            return Err(format!(
                                "{at} fills `{}` from `{}` and the composition does not state that \
                                 rename -- an unwritten rename is how a transposition of two \
                                 same-typed operands passes every type check there is",
                                wants.name, source.name
                            ));
                        }
                    }
                    Take::Null => {
                        if !wants.nullable {
                            return Err(format!(
                                "{at} nulls `{}`, which the row does not declare nullable",
                                wants.name
                            ));
                        }
                        if read_only(wants.ty).is_none() && !matches!(wants.ty, Ty::Buf | Ty::U32s | Ty::U8s | Ty::I32s | Ty::F32s | Ty::U16s | Ty::I8s | Ty::Bf16s | Ty::F16s | Ty::I64s) {
                            return Err(format!(
                                "{at} nulls `{}`, which is {:?} and not a pointer",
                                wants.name, wants.ty
                            ));
                        }
                    }
                }
            }
        }
        // A rename nobody uses is a rename that licenses a transposition
        // nobody meant to license -- the same reason `Service::ALL` has to
        // have a member per variant.
        for (from, to) in self.renames {
            let used = self.steps.iter().any(|step| {
                let Some(sig) = sig_of(step.symbol()) else { return false };
                step.take().iter().enumerate().any(|(slot, take)| {
                    matches!(take, Take::From(index)
                        if op.operands.get(*index).is_some_and(|o| o.name == *from)
                            && sig.operands.get(slot).is_some_and(|w| w.name == *to))
                })
            });
            if !used {
                return Err(format!(
                    "`{}` states the rename `{from}` -> `{to}` and no step performs it",
                    self.symbol
                ));
            }
        }
        Ok(())
    }

    /// Whether every step of this composition, transitively, is something this
    /// crate can fire today.
    ///
    /// The honest split between the two entries of [`COMPOSED`], and the
    /// reason it is a function rather than a field: it is derived from the
    /// unit tables, so it becomes true the day the last step migrates and
    /// nobody has to remember to flip it. `attn::compact_page_csr` is
    /// fireable; `gemm::act_x_wt_bias_bf16` is not, because
    /// `gemm::act_x_wt_bf16` is an unmigrated kernel behind
    /// `Wall::HostChoice`.
    #[must_use]
    pub fn fireable(&self) -> bool {
        self.steps.iter().all(|step| match execution(step.symbol()) {
            Some(Execution::Jit(_)) => true,
            Some(Execution::Composed(_)) => {
                composition(step.symbol()).is_some_and(Composition::fireable)
            }
            _ => false,
        })
    }
}

/// No symbol composes itself, through any number of steps.
///
/// # Why this exists at all
///
/// Because a [`Step`] names a SYMBOL and a symbol may be [`Composed`], a
/// symbol can now compose itself -- directly, or through a cycle of any
/// length. Nothing else in this crate could express one: a `DeviceKernel`
/// names a template and a `Service` names a library, and neither can point
/// back at a row. This is the first recursive edge in the table, so it is the
/// first place the table can loop.
///
/// A loop is not a compile error and not a wrong answer: it is
/// [`Composition::fireable`] recursing until the stack ends, and a driver
/// expanding a plan doing the same. So it is asserted, GPU-free, over the
/// whole table.
///
/// # Why it takes the table as an argument
///
/// So the assertion can be made to FAIL. A check that can only ever be run
/// against a table known to be acyclic proves nothing about the check --
/// `new-horizon.md` §21.9's *"a gate that filters must assert its own
/// denominator"* in its other form. `tests/layers.rs` hands this a
/// deliberately cyclic table and asserts the cycle is named, and hands it a
/// self-firing one-symbol table and asserts the same.
///
/// # Errors
///
/// A sentence naming the cycle, in the order it was walked.
pub fn acyclic(table: &[Composition]) -> Result<(), String> {
    // Three colours over an explicit stack. `Grey` is "on the current path",
    // which is what makes this find a cycle rather than merely a re-visit: a
    // diamond (two steps naming one composed symbol) is not a loop and must
    // not be reported as one.
    #[derive(Clone, Copy, PartialEq)]
    enum Colour {
        White,
        Grey,
        Black,
    }
    let mut colour = vec![Colour::White; table.len()];
    let index_of = |symbol: &str| table.iter().position(|c| c.symbol == symbol);

    for start in 0..table.len() {
        if colour[start] != Colour::White {
            continue;
        }
        // (node, how many of its steps have been walked), plus the path for
        // the diagnosis.
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        colour[start] = Colour::Grey;
        while let Some((node, cursor)) = stack.pop() {
            if cursor == table[node].steps.len() {
                colour[node] = Colour::Black;
                continue;
            }
            stack.push((node, cursor + 1));
            let Some(next) = index_of(table[node].steps[cursor].symbol()) else { continue };
            match colour[next] {
                Colour::Grey => {
                    let mut path: Vec<&str> =
                        stack.iter().map(|(n, _)| table[*n].symbol).collect();
                    path.push(table[next].symbol);
                    return Err(format!(
                        "a composition cycle: {} -- a step names a symbol whose steps reach it \
                         again, so expanding this op does not terminate",
                        path.join(" -> ")
                    ));
                }
                Colour::White => {
                    colour[next] = Colour::Grey;
                    stack.push((next, 0));
                }
                Colour::Black => {}
            }
        }
    }
    Ok(())
}

/// How a symbol executes -- the join over all three tables.
///
/// `None` means the symbol is a kernel this tree has not migrated: a row in
/// [`crate::table`] with no unit hosting it, no composition stating it and no
/// service serving it. That is a FOURTH answer and it is deliberately not an
/// [`Execution`] variant, because "nobody executes this yet" is a fact about
/// the migration and not about execution. `examples/migration_status.rs` is
/// what counts them.
///
/// The three tables may not overlap, and the ordering here is not what
/// enforces that: `tests/layers.rs` asserts the disjointness directly, so a
/// symbol that was both composed and hosted is a failing test rather than a
/// silent precedence.
#[must_use]
pub fn execution(symbol: &str) -> Option<Execution> {
    if let Some(service) = service(symbol) {
        return Some(Execution::Service(service));
    }
    if let Some(composition) = composition(symbol) {
        return Some(Execution::Composed(composition.steps));
    }
    let (_, unit) = unit::unit_of(symbol)?;
    unit.rows.iter().find(|row| row.sig.symbol == symbol).map(Execution::Jit)
}

#[cfg(test)]
mod tests {
    use super::{Execution, Kind, SERVED, Service, Step, execution, service};

    /// Every variant of [`Service`] has at least one member.
    ///
    /// A name with no row behind it is vocabulary admitted without evidence,
    /// which is the exact habit this module was built to stop. It is also the
    /// cheap half of `new-horizon.md` §21.9's rule -- a gate that filters must
    /// assert its own denominator -- applied to an enum.
    #[test]
    fn no_service_name_is_unevidenced() {
        for service in Service::ALL {
            assert!(
                SERVED.iter().any(|(_, s, _)| s == service),
                "`Service::{service:?}` has no member -- a library named on nobody's evidence"
            );
        }
    }

    /// Every entry carries a citation, and no symbol is served twice.
    #[test]
    fn every_served_row_is_cited_once() {
        let mut seen: Vec<&str> = Vec::new();
        for (symbol, _, why) in SERVED {
            assert!(!seen.contains(symbol), "`{symbol}` is served twice");
            assert!(why.len() > 20, "`{symbol}` is served on a citation too short to check: {why:?}");
            seen.push(symbol);
        }
    }

    /// The lookup is the table, and a symbol nobody serves gets `None`.
    #[test]
    fn the_lookup_is_the_table() {
        for (symbol, expected, _) in SERVED {
            assert_eq!(service(symbol), Some(*expected));
            assert!(matches!(execution(symbol), Some(Execution::Service(s)) if s == *expected));
        }
        assert_eq!(service("norm::residual_add_bf16"), None);
    }

    /// The kind of an execution is the kind of its arm, all three of them.
    ///
    /// [`Kind`] is derived from the TOP-LEVEL execution and from nothing
    /// deeper, which is what keeps the partition total now that a step can
    /// name a service: `gemm::act_x_wt_bias_bf16`'s first step reaches cuBLAS
    /// and the symbol is still an [`Kind::Op`].
    #[test]
    fn the_three_arms_are_the_three_kinds() {
        const STEPS: &[Step] = &[Step::Fire { symbol: "norm::residual_add_bf16", take: &[] }];
        assert_eq!(Execution::Composed(STEPS).kind(), Kind::Op);
        assert_eq!(Execution::Service(Service::Cublas).kind(), Kind::Service);

        let jit = crate::unit::rows().next().expect("some unit hosts a row");
        assert_eq!(Execution::Jit(jit).kind(), Kind::Kernel);

        // The real one: an op whose steps reach a service is still an op.
        let composed = super::composition("gemm::act_x_wt_bias_bf16").expect("the demonstration case");
        assert_eq!(Execution::Composed(composed.steps).kind(), Kind::Op);
    }

    /// A ROW MAY NOT BE TAKEN OVER BEFORE IT IS CLASSIFIED.
    ///
    /// [`RUST_SERVED`] is what drops a shim entry, and dropping one is what
    /// lets the C++ body be deleted. Doing that to a row whose execution
    /// nobody has written down is deleting a body on a hunch — precisely the
    /// move `SERVED`'s own header refuses ("a finding is not a plan", and
    /// neither is a plan a finding). Every symbol on the list must appear in
    /// [`SERVED`] or [`COMPOSED`] first, with its citation.
    #[test]
    fn every_taken_over_row_was_classified_first() {
        for symbol in super::RUST_SERVED {
            assert!(
                super::service(symbol).is_some() || super::composition(symbol).is_some(),
                "`RUST_SERVED` names `{symbol}`, which is in neither `SERVED` nor `COMPOSED`. \
                 The list drops the row's shim entry, so the C++ body goes -- state what the \
                 body IS, with the file and line, before taking it over."
            );
        }
    }

    /// A ROW TAKEN OVER MUST HAVE HAD SOMETHING TO TAKE.
    ///
    /// `emit_c_shim` only ever emitted entries for STATED rows — those with
    /// operands. Naming an unstated row here would drop nothing, generate
    /// nothing, and read like a migration that happened. It is a typo in a
    /// string literal, which is the failure mode a list of symbols has.
    #[test]
    fn every_taken_over_row_is_stated() {
        for symbol in super::RUST_SERVED {
            let sig = crate::table::sig(symbol).unwrap_or_else(|| {
                panic!("`RUST_SERVED` names `{symbol}`, which is in no family table")
            });
            assert!(
                !sig.operands.is_empty(),
                "`RUST_SERVED` names `{symbol}`, whose row states no operands. `emit_c_shim` \
                 never emitted an entry for it, so taking it over drops nothing and the C++ \
                 -- if any exists -- stays."
            );
        }
    }
}
