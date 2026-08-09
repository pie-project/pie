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
//! # A fourth ANSWER, and deliberately not a fourth KIND
//!
//! [`Execution`] has four arms and [`Kind`] still has three, and the gap
//! between those two numbers is [`Walk`].
//!
//! An **op** whose host program is a fixed sequence of our rows is
//! [`Execution::Composed`], and a `&'static [Step]` states it exactly. An op
//! whose host program's SHAPE comes from the input -- a loop whose trip count
//! is an operand, a `switch` whose arm is a run-time shape, a predicate over
//! two operands -- cannot be a slice, because a slice is a sequence fixed at
//! compile time. That is not a smaller fact than `Composed`'s; it is a
//! different one, and forcing it into `Step` would mean growing `Step` a loop
//! vocabulary, an arithmetic-over-operands vocabulary and a predicate
//! vocabulary, which is [`kernels::Source`] re-invented inside a table.
//! [`Step`]'s own header priced two of those three and refused both.
//!
//! **Three independent pieces of work hit this from three directions**, which
//! is why it is a category and not one launcher's excuse:
//!
//! * the three multimodal towers (`new-horizon.md` §42), whose walk is
//!   `for (int im = 0; im < num_images; ++im)` with host-side position-embed
//!   interpolation computed BETWEEN launches;
//! * our own `attention_mtp_paged_history_bf16` (§44.6), an `if` on
//!   `max_global_tokens + history_steps > 8192` selecting between two
//!   shared-memory budgets -- **written by us**, which is the evidence that
//!   the unstatable dispatch is not something FlashInfer did to us;
//! * FlashInfer's two capture dispatches (§44.1, §44.8), a `switch` over
//!   `src/kernels.def`'s head dims into a template instantiation that has no
//!   row and can never have one, because there are hundreds.
//!
//! So the arm is added, and [`Execution::kind`] answers [`Kind::Op`] for it.
//! A walk IS a host program -- the second of the three jobs above -- and the
//! difference between it and a composition is whether the program can be
//! ENUMERATED, which is a fact about statability rather than about what the
//! symbol is. Adding a fourth [`Kind`] would re-open the module's thesis for
//! a distinction the thesis was never about, and would move the migration
//! denominator for a reason that has nothing to do with migration.
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
//!
//! [`Walk`] carries the same discipline one step further, and it has to: a
//! walk's host program is exactly the thing that cannot be spelled as data, so
//! a `Walk` that carried a function pointer, a `Step` list or a geometry would
//! be claiming to state the unstatable. It carries WHAT SHAPE the control flow
//! is, WHAT it refuses, and WHERE the program is written, with lines. Nothing
//! executes off it. `driver-cuda`'s `dispatch_generated` reaches these symbols
//! through exactly the entry point it reached them through before this arm
//! existed, which is the strongest possible form of the owner's constraint:
//! **adding `Walk` generated no code and changed no call**, so `model-compiler`
//! cannot have observed it even in principle.

use kernels::{KernelSig, Ty};

use crate::device::{DeviceKernel, Take};
use crate::{table, unit};

/// How a stated symbol is executed.
///
/// Four arms because there are four answers, and the fourth has never been a
/// kernel either. See the module header for what each means, for why the
/// fourth is not a fourth [`Kind`], and for what none of them is allowed to do
/// (be visible above the driver).
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
    /// A host program whose SHAPE comes from the input, run by the driver.
    ///
    /// The arm [`Composed`] could not be. See [`Walk`].
    ///
    /// [`Composed`]: Execution::Composed
    Walk(&'static Walk),
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
/// [`Step::Fire`]`{ symbol: "attn::compact_page_csr", .. }` names a SYMBOL.
/// That symbol resolves its own [`Execution`], which may be
/// [`Execution::Jit`], [`Execution::Service`], another
/// [`Execution::Composed`], or nothing at all.
///
/// **`compact_page_csr` NO LONGER HAS A ROW and this paragraph still holds**,
/// which is the point of the heading: a step names a symbol, and what serves
/// that symbol was never the step's business. [`crate::x::route`] answers it
/// now, and answers `Entry` for this one. Three consequences, and they are
/// the design:
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
/// The table is the MEASUREMENT and not an index: every row still describes
/// the launcher it was read off. FIVE of the ten have since crossed into
/// fn-world, `attn::compact_page_csr` included -- its two launches are two
/// statements in `x::attn::compact_page_csr`, with both refusals hoisted
/// ahead of the first -- and the `switch` a "no" refused to state is an `if`
/// in a `fn` now. That is the answer the column was asking for, arrived at by removing
/// the question rather than by widening `Term`.
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

    // ── THE DEMONSTRATION CASE WAS HERE AND IT IS DELETED ───────────────
    //
    // `Composition { symbol: "gemm::act_x_wt_bias_bf16", .. }`, two
    // `Step::Fire`s — `gemm::act_x_wt_bf16` then `norm::add_bias_bf16`, with
    // `renames: &[("y", "out"), ("n", "dim")]` — from
    // `gemm/gemm.cpp:2395-2398`:
    //
    //     gemm_bf16_impl(handle, act, W, y, M, N, K, beta);
    //     if (bias != nullptr)
    //         kernels::norm::add_bias_bf16(y, bias, M, N, stream);
    //
    // It was *"stated and not fireable"*: §2.3's `Composed` — two DIFFERENT
    // kernels in one body — is written and was, at the time this entry was
    // added, unproven. This entry existed to say the shape was sayable.
    //
    // **§5 step 5 answered it by making it ordinary.**
    // `x::gemm::act_x_wt_bias_bf16` is a `fn` with two calls in its body: the
    // dense GEMM, then `x::driver_internal::add_bias_bf16` when `bias` is
    // non-null. The `renames` are gone because a `fn`'s parameters have one
    // name each; the `take: &[Take::From(..)]` index lists are gone because
    // the second call passes the first's `y`; and the `bias != nullptr`
    // guard, which this entry noted was `Term::Present` and statable but
    // deliberately unstated, is an `if` — which is what §5 step 5 says every
    // `Specialisation` becomes.
    //
    // THE ONE THING THAT DID NOT SURVIVE AS DATA, and it did not need to:
    // the note above about the `M == 1` arm at `gemm/gemm.cpp:2383-2390` — a
    // fused GEMV+bias epilogue whose tactic comes from `dense_tactic_for`, a
    // run-time tuning cache consulted under `cudaStreamIsCapturing`, which
    // *"is not a predicate over a fire's operands and no `Fact` carries
    // one"*. That is still true and is still the reason the fused arm is not
    // a composition. In fn-world it is also no longer a wall: the tuner IS
    // the host program now, it lives in `x::gemm::dense`, and a `fn` may
    // consult a cache where a `Composition` may not. `migration_status.rs`
    // recorded this row as walled for that reason and the wall is gone.
];

/// A host program whose SHAPE comes from the input -- the arm
/// [`Execution::Composed`] could not be.
///
/// # What a walk is, in one sentence
///
/// **Host control flow the driver owns, over device text that may belong to
/// anybody, whose shape depends on the input rather than on the model.**
///
/// Every clause of that is load-bearing and one of them is a correction of
/// [`Composed`]:
///
/// * *host control flow* -- a loop, a `switch`, a predicate. Not a launch.
/// * *the driver owns it* -- so the row is executed and not pending, which is
///   why a walk is a finished migration and not a wall.
/// * *device text that may belong to anybody* -- and this is where `Composed`
///   is too narrow. A [`Step`] names a SYMBOL, so `Composed` can only ever
///   describe a program built out of things this table rows. The towers launch
///   `.cuh` templates that never had rows; FlashInfer's capture launches a
///   template instantiation that has hundreds of siblings and can never have
///   one. **A walk states nothing about the device text on purpose**, because
///   the three members disagree about who owns it and any claim wide enough to
///   cover them would be true of everything.
/// * *shape depends on the input* -- `for im in 0..num_images`,
///   `switch (cache.head_dim)`. A `&'static [Step]` is a sequence fixed when
///   this crate compiles; these are not.
///
/// # Why `Walk`, and not the two other names on the table
///
/// **`Planned`** is `new-horizon.md` §44.8's own proposal -- *"text we
/// compile, with geometry that arrives from a host planner"* -- and it is
/// refused twice. It names the mechanism of ONE member: the towers have no
/// planner and `attention_mtp_paged_history_bf16` had none either, so a name
/// true of one of three is the same defect [`Service`]'s header refuses in the
/// other direction (a variant with no member). And the word is already spent
/// in this table: [`kernels::Prepare::DecodePlan`] and
/// [`kernels::Prepare::PrefillPlan`] are a live FIELD of [`KernelSig`], so an
/// `Execution::Planned` sitting beside `needs = Prepare::DecodePlan` would
/// read as a restatement of it. It is not one -- `attn::compact_page_csr`
/// needs no plan and `attn::dispatch_attention_flashinfer_decode` needs one
/// and is a walk, so the two facts are independent -- and a name that invites
/// the reading is a name that will get it.
///
/// **`Hosted`** was the third candidate and is worse than both: every arm here
/// except [`Execution::Jit`] is hosted, including `Composed`.
///
/// **`Walk`** says the shared thing and nothing more, and it is not a coinage:
/// this tree reached for the word independently three times before this arm
/// existed. `driver-cuda/build.rs`'s towers block -- *"what `kernels-cuda`
/// compiled for a tower was never device code: it was a HOST WALK over device
/// code that belongs to `kernels-cuda-new`"* -- `new-horizon.md` §42's title
/// *"A tower is a host walk"*, and `plan/mod.rs`'s framing of the scheduler as
/// arithmetic above the kernels. A word three independent readers picked for
/// the same shape is cheaper than a fourth one.
///
/// # A failure is a refusal, never a fallback
///
/// [`Walk::refuses`] is not documentation. A walk with an empty `refuses` is
/// rejected by [`Walk::agrees`], because the failure mode this category
/// carries is specific and measured: FlashInfer's dispatch macros end in
/// `throw std::invalid_argument` for an unsupported head dim or GQA group
/// size, an exception crossing the C ABI is undefined behaviour that in
/// practice unwinds through Rust frames to `SIGABRT` with no message, and the
/// tempting repair -- pick a different kernel -- is the one thing that must
/// never happen. A capture dispatch that silently ran the non-capturing kernel
/// would hand the eviction policy an all-zero score row, which reads as
/// *"nothing was attended to"* and evicts the whole prefix. That is a wrong
/// answer wearing a right answer's shape, and the refusals below are how the
/// table says so where a reader will find it.
///
/// # No mechanism, same as [`Service`]
///
/// A name, a shape, its refusals, and a citation. No function pointer, no
/// [`Step`] list, no geometry, no `Dims`. The driver already knows how to run
/// these -- for four of the five it is the same generated arm and the same
/// shim entry it used before this type existed -- and a `Walk` that carried
/// enough to be EXECUTED would be a plan, which `SERVED`'s header refuses in
/// as many words: *a finding is not a plan.*
pub struct Walk {
    /// The stated symbol this walks -- a row of [`crate::table`].
    pub symbol: &'static str,
    /// What the host control flow IS. See [`Control`].
    pub control: Control,
    /// Every way this walk REFUSES, in the launcher's own words.
    ///
    /// Never a fallback. [`Walk::agrees`] requires at least one, because a
    /// walk that cannot say no is a walk that will guess.
    pub refuses: &'static [&'static str],
    /// The program, with a file and lines, so the shape above can be checked
    /// against the code that runs. [`Composition::because`] for the same
    /// reason: what a machine in this crate cannot check is whether the words
    /// say what the program says.
    pub because: &'static str,
}

/// The shape of a walk's host control flow -- **two variants, because two
/// shapes have members.**
///
/// # The variants this enum has HAD, and the rule that governs both
///
/// A **predicate over two operands** was the third way a host program
/// escapes the vocabulary, and it was the best-documented of the three:
/// `attention_mtp_paged_history_bf16`'s `if (max_global_tokens +
/// history_steps > 8192)` chose between two kernels with different
/// shared-memory budgets, and `new-horizon.md` §44.6 records it as the
/// evidence that this whole problem is ours and not FlashInfer's -- *every
/// `Term` in the `LaunchRule` vocabulary is unary*, and that one is not.
///
/// **§44.6 also DELETED it**, and this header said so: *a variant with no
/// member is exactly what [`Service`]'s header calls vocabulary admitted
/// without evidence... whoever re-lands a two-operand predicate adds the
/// variant with its member in the same edit.*
///
/// [`Control::Supplies`] is that variant re-landed, and it came back with
/// **nine members**, every one a comparison or an arithmetic over two
/// operands that no unary `Term` can state:
///
/// ```text
///   half >= 256                                  rope::rope_bf16, rope_yarn_bf16,
///                                                rope_yarn_original_bf16
///   half <= 4096                                 rope::rope_bf16, rope_yarn_original_bf16
///   num_q_heads + num_kv_heads                   rope::qk_rmsnorm_* (three of them)
///   yarn_factor > 1 && orig_max_position > 0     rope::rope_partial_last_bf16
///   hc_mult > MAX_HC_MULT                        norm::hc_post_bf16
/// ```
///
/// **`Loop` LEFT BY THE SAME RULE, and this is that edit.** It said *a loop
/// whose TRIP COUNT is an operand*, and it had exactly three members: the
/// three multimodal towers, each a `for (int im = 0; im < num_images; ++im)`.
/// All three are Rust now (`driver-cuda/src/tower/`), so all three left
/// [`WALKED`], and the last one out took the variant with it. The shape is
/// real -- a tower still loops over its images, and the Rust `for` is the
/// same `for` -- but a driver's own control flow is not something this crate
/// classifies, and a variant kept "because the shape exists" is precisely the
/// vocabulary-without-evidence the paragraph above refuses. Whoever brings a
/// C++ host program with an operand-counted loop back under this crate's
/// classification adds `Loop` with its member in the same edit; the words to
/// restore are in this file's history and in `tower::qwen3_vl`'s header.
///
/// `tests::no_control_shape_is_unevidenced` is what keeps a shape and its
/// evidence together, in both directions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Control {
    /// A `switch`/`if` chain whose ARM is chosen from a run-time shape.
    ///
    /// `on` is what is switched on, spelled as the program spells it.
    Switch {
        /// The discriminant, in the program's own words (`cache.head_dim`).
        on: &'static str,
    },
    /// A VALUE the launch needs and no row can state, computed on the host.
    ///
    /// # How this differs from [`Control::Switch`], which is the only
    /// question worth asking about it
    ///
    /// **Where the predicate's answer GOES.** A `Switch`'s answer selects an
    /// arm: two kernels, or a kernel and a refusal, and the launch that runs
    /// is one of several written down. A `Supplies`'s answer is an OPERAND or
    /// an EXTENT of the one launch there is -- `heads_per_block = half >= 256
    /// ? 1 : 256 / half` is passed to the kernel AND divides the head axis of
    /// the grid; `cache_pairs = half <= 4096 ? half : 0` is passed to the
    /// kernel AND sizes the dynamic shared allocation. There is one `<<<>>>`
    /// and no arm to choose.
    ///
    /// That is why it is not a `Switch` wearing a different name, and why it
    /// is not a missing `LaunchRule` either: a rule computes a rectangle from
    /// a [`crate::runtime::Dims`] and cannot hand a value to the kernel, and a
    /// [`kernels::Source`] can hand a value to the kernel and has no
    /// comparison. The quantity needs both halves, so it needs a host.
    ///
    /// `families/rope.rs` states the same finding from the table's side:
    /// *"`heads_per_block = half>=256?1:256/half` and `cache_pairs =
    /// half<=4096?half:0` are host conditionals and the `Source` grammar has
    /// no comparison."* One `Source` that could state one comparison would
    /// retire four of the members below; until then this is where they live.
    Supplies {
        /// The value, in the program's own words
        /// (`heads_per_block = half >= 256 ? 1 : 256 / half`). Never empty.
        what: &'static str,
    },
}

impl Control {
    /// Every variant, so a test can assert each has a member.
    pub const ALL: &'static [&'static str] = &["Switch", "Supplies"];

    /// Which variant this is, as the name the member test compares.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Control::Switch { .. } => "Switch",
            Control::Supplies { .. } => "Supplies",
        }
    }

    /// The operand or field the shape reads. Never empty.
    #[must_use]
    pub fn reads(self) -> &'static str {
        match self {
            Control::Switch { on } => on,
            Control::Supplies { what } => what,
        }
    }
}

/// The rows a driver-owned host walk executes, with the evidence for each.
///
/// # The bar every entry had to clear
///
/// One fact, checked in the tree rather than read off a call site: **the
/// symbol's host program has control flow whose shape is not fixed when this
/// crate compiles.** That is the whole test, and it is narrow on purpose --
/// "the launcher is complicated" is not a walk, and neither is "the launcher
/// is C++". `attn::compact_page_csr` is two launches in a fixed order and is
/// [`Execution::Composed`]; `gemm::act_x_wt_bias_bf16` is a GEMM then a bias
/// add and is `Composed` too. Both have a `LaunchRule` per step and a
/// sequence anyone can write down.
///
/// # Citations point at the program, wherever it now lives
///
/// [`SERVED`]'s rule is *cite `crates/kernels-cuda/csrc/**`, which nobody is
/// editing*. It cannot be followed here, because a walk's defining property is
/// that its program had to LEAVE that archive: three of the five moved to
/// `driver-cuda/csrc/vision/` in §42 and two moved to `driver-cuda/csrc/attn/`
/// with this arm. So each entry cites its home today AND the archive path and
/// lines it came from, which is checkable with `git show` at any commit on
/// either side of the move.
#[rustfmt::skip]
pub static WALKED: &[Walk] = &[
    // ── the three multimodal towers (§42) — ALL THREE GONE ───────────────
    //
    // These moved out of `kernels-cuda` a whole migration before this arm
    // existed, and `driver-cuda/build.rs` wrote the refusal down at the time:
    // *"`Execution` has `Jit | Composed | Service`, and `Composed` is the near
    // miss: it carries a `&'static [Step]`, a list fixed at compile time, and
    // a tower is a data-dependent loop."* They are the reason this arm is a
    // category rather than an attention exception -- the towers reached it
    // from vision, alone, and proposed this name.
    //
    // `vision::gemma4_vision_encode`, `vision::gemma4_audio_encode` AND
    // `vision::qwen3vl_scatter` STOOD HERE, ahead of FlashInfer's two, and
    // their removal is the evidence this arm was right about what a walk IS.
    // The entries said the loops were `for (int im = 0; im < vin.num_images;
    // ++im)` at `gemma4_vision.cu:258`, `for (int clip = 0; clip <
    // ain.num_clips; ++clip)` at `gemma4_audio.cu:322` and the same
    // `num_images` loop at `qwen3_vl_tower.cu:486`, and cited ten refusals
    // between them. Every word of that is still true -- the loops still run,
    // the ten refusals still fire, `qwen3vl_vision` still declines a weights
    // table whose `hidden != heads*head_dim`, still declines a patch count
    // `not divisible by merge^2`, and `qwen3vl_vis_attn` still declines
    // rather than attending over a workspace it could not allocate -- but the
    // programs are Rust now (`driver-cuda/src/tower/gemma4_vision.rs`,
    // `.../gemma4_audio.rs` and `.../qwen3_vl.rs`), driving JIT'd units
    // through `runtime::fire`. `WALKED` is a table of symbols this crate
    // hands to a driver to RUN; a walk written in the driver is not one of
    // those, so neither is a row, and `execution("vision::qwen3vl_scatter")`
    // correctly answers `None` for a name that no longer exists.
    //
    // What did NOT change is the shape of the claim. `Control::Loop { over:
    // "vin.num_images" }` described a trip count that comes from the input,
    // and the Rust does exactly that. The arm's own argument -- "`Composed`
    // carries a `&'static [Step]` fixed at compile time, and a tower is a
    // data-dependent loop" -- is why the Rust could not be a `Composition`
    // either. They stay walks; they stopped being C++. `Control::Loop`
    // itself went with the last of them, under the rule its own header
    // states: a variant with no member is vocabulary without evidence.
    //
    // `qwen3vl_scatter` was the last of the three, and the last row in
    // `table::driver_internal` whose launcher lived in `driver-cuda/csrc/`.
    // Its entry said what its own retirement required: transcribe 772 lines
    // across three files that cannot be split, delete the
    // `table::driver_internal` row in the same commit, and rewrite
    // `driver-cuda/tests/bridge_smoke.rs`'s hand-written (that file is
    // deleted now, with `bind::abi::ffi`; this records what it did)
    // `ffi::pie_k_vision_qwen3vl_scatter` call — *"because that row's
    // launcher is `qwen3_vl_tower_c.cpp` and a shim entry forwarding to a
    // deleted launcher does not compile"*. All three happened together, and
    // `driver-cuda/csrc/vision/` is gone with them. The one thing its entry
    // predicted and did not get is the FlashInfer edge: the Rust tower still
    // reaches prefill through
    // `ffi::pie_k_attn_dispatch_attention_flashinfer_prefill_bf16`, which
    // existed already, so `vis_helpers.cpp:131` needed no forwarder — only a
    // caller, and the caller is Rust. The two entries below are what retires
    // that last edge.
    //
    // The category is not empty, and none of the three left because it was
    // wrong.

    // ── FlashInfer's two capture dispatches (§44.1, §44.8) ────────────────
    //
    // THE ENTRIES THIS ARM WAS ADDED FOR, and the ones §44 could not write.
    // Its report said so exactly: *"It cannot be `Jit`, because the kernel is
    // not ours and the geometry comes from a plan built at runtime, not from
    // `Dims`. It cannot be `Composed`, because a `Step` names a table row and
    // FlashInfer's dispatch resolves to a template instantiation that has no
    // row and can never have one -- there are hundreds. It is not `Service`,
    // because nothing is linked; we HAVE the source and we compile it."*
    //
    // All three of those are still true, and all three are now stated rather
    // than merely written down: the arm names the shape, the refusals and the
    // program, and claims nothing about whose device text it launches --
    // which is the one claim that would have been false for these two and for
    // the towers in opposite directions.
    //
    // ── WHAT THE FA2 JIT DOES AND DOES NOT CHANGE HERE ────────────────────
    //
    // `crate::families::fa2` now compiles FlashInfer's two paged
    // `__global__`s through NVRTC, and it is worth being exact about which of
    // the three refusals that touches, because it is tempting to read the
    // change as promoting these entries to `Jit`.
    //
    // It touches the middle one and only the middle one. *"FlashInfer's
    // dispatch resolves to a template instantiation that has no row and can
    // never have one -- there are hundreds"* is now FALSE: there are 460, and
    // every one of them has a row (20 decode units x 5 arms, 36 prefill units
    // x 10). The count was right; the conclusion drawn from it was that a
    // lattice cannot be a table, and a macro-generated family is the
    // counter-example.
    //
    // The other two are untouched, and they are the ones that decide the
    // classification:
    //
    //   * the geometry still comes from a PLAN built at run time, not from
    //     `Dims` -- `crate::plan::decode::estimate`'s `padded_batch_size` is a
    //     scheduler output and no `LaunchRule` can state it;
    //   * the symbol still names a HOST PROGRAM over several kernels, not one
    //     kernel: a `switch (cache.head_dim)`, six refusals, an optional
    //     dequant of the KV pages and a merge pass, of which the FA2 launch is
    //     one step.
    //
    // So these stay `Walk`, and §58's rule is why -- `Specialisation` vs
    // `Walk` is a category error, and so is `Jit` vs `Walk`. `Jit` says *"this
    // symbol IS a kernel we compile"*; the FA2 rows say that, under their own
    // names, in `crate::families::fa2`. `Walk` says *"this symbol is a program
    // that launches kernels"*, which is what these two are whether the kernels
    // arrive from an archive or from NVRTC. §60.6's `_dev` symbol split is the
    // same shape one layer down: the program keeps the launcher's name and the
    // kernel gets its own.
    //
    // What DOES change when north-star §5 step 8 finishes is the `program`
    // field: these walks become Rust, so the citation moves from
    // `attention_flashinfer.cu` to `driver-cuda/src/fire/flashinfer_fa2.rs`.
    // That is an edit to two strings, not a reclassification. **IT HAS NOW
    // HAPPENED, TWICE.** First the six went to [`RUST_SERVED`] and the
    // `because` strings were repointed at the Rust; then the rows went too
    // and the entries left this list entirely. The C++ line numbers are kept
    // inside the [`SERVED`] sentences that replaced them, because a citation
    // to a deleted file is still the only record of where a transcription
    // came from -- and note which half survived the second move: the line
    // numbers, which are about the KERNEL, and not the module paths, which
    // are about the tree.
    // ── AND ALL SIX ARE GONE, BECAUSE THEY ARE DRIVER OPS ───────────────
    //
    // Six `Walk`s stood here -- the two capture dispatches this arm was
    // added for, and the four plain launchers added beside them. They are
    // now [`SERVED`] entries with `Service::DriverOp`, and the reason is not
    // that the classification above was wrong. Every sentence of it still
    // holds: the geometry comes from a plan built at run time, the symbol
    // names a host program over several kernels, and none of that is `Jit`,
    // `Composed` or `Service` in the linked-library sense.
    //
    // What changed is the QUESTION this file is asked. `x::route` reads
    // [`SERVED`] as the driver-op oracle -- `Service::DriverOp` says *"the
    // driver already fires this"* -- and there is no route a `Walk` produces.
    // The six rows left `table::attn` in the same commit, so the alternative
    // was `Route::Rows` against a row that no longer exists. `x::gemm`'s
    // twelve made this move first and `moe::build_moe_ptrs_aligned_bf16`
    // made it from a `Walk` exactly as these do, under the test that refuses
    // both at once: **a walk and a service are two answers to one question.**
    //
    // The RESOURCE each names is `DecodePlanCache` / `PrefillPlanCache`,
    // boxed and mutable and living across fires -- §3.3's forbidden surface,
    // which is the driver-op test answered in one line. Five of the six
    // answer it that way. `attn::attention_flashinfer_prefill` does NOT, and
    // its entry below says so rather than borrowing its siblings' reason.
    //
    // AND THE `refuses` LISTS ARE NOT LOST PROSE. Each was a transcription
    // of a C++ `throw` string, and every one of them is now a variant of
    // `driver-cuda/src/fire/flashinfer_fa2_dispatch::Decline` --
    // `Unplanned`, `Sm90Unported`, `DecodeLatticePoint`,
    // `PrefillLatticePoint`, `CaptureSinkMissing`,
    // `CaptureVariantUnsupported`, `Geometry`. A list of strings beside the
    // code goes stale in silence; an enum the code returns cannot. That is
    // the whole argument for letting the entries go instead of repointing
    // them, and it is the one `moe`'s nine were repointed against.
    // ── the dense bf16 GEMM and its autotuner ─────────────────────────────
    //
    // THE HOTTEST ROW IN THE TREE, and the last thing `gemm.cpp` was holding.
    //
    // Why it is a walk and not a `Service::Cublas`. [`SERVED`]'s rule is *one
    // library call and nothing else in the body*, and four `gemm.cpp` entry
    // points cleared it. This one never could: the body chooses among THREE
    // kernel families at run time — the warp-per-row GEMV (ours, JIT'd), a
    // `cublasGemmEx`, and each algorithm cuBLASLt's heuristic offers for the
    // shape — from a MEASUREMENT taken on the shape the first time it is
    // seen. Which arm runs is not knowable when this crate compiles, is not
    // knowable when the driver compiles, and is not even knowable at the
    // first call: it depends on a timing loop, on a per-device memo and on a
    // file in `$XDG_CACHE_HOME/pie/dense_gemm.txt`.
    //
    // Why not a `Composition`. `Composed` carries a fixed `&[Step]`. Here
    // there is one launch and the question is WHICH, so there is no step list
    // to write; and the fallback ladder underneath (GEMV, then the cuBLASLt
    // ladder behind four shape gates, then `cublasGemmEx` with two documented
    // retries) is a sequence of ATTEMPTS, not of steps — each one runs only
    // because the one before declined.
    //
    // What it frees: `crates/kernels-cuda/csrc/src/gemm/gemm.cpp` entirely,
    // and with it the last `.cpp` of the `gemm/` family. The file held zero
    // `__global__` and zero `<<<>>>` from the day it was measured; what
    // condemned it was never that it held kernels but that it held a host
    // program, and the rule is that every piece of CPU-side code is Rust.
    // ── `gemm::act_x_wt_bf16`'s WALK WAS HERE AND IT IS DELETED ─────────
    //
    // `Control::Switch { on: "the tuned DenseTactic for (M, N, K, beta != 0)" }`.
    // The analysis above is kept because it is what the walk was FOR; what
    // is gone is the claim that the host program is still to be written.
    //
    // **It is written.** `x::gemm::dense` is that host program — the dense
    // autotuner, `GemmKind`, the cuBLASLt plan cache and the on-disk tactic
    // cache — moved verbatim from `driver-cuda/src/fire/gemm.rs`, with
    // `x::gemm::gemv` beside it as the fallback ladder's first rung. The
    // switch is an `if` over a `DenseTactic`, which is what §5 step 5 says a
    // `Specialisation` becomes, and the ladder of ATTEMPTS is a chain of
    // `Gemv::Declined` -> next rung, which is why that enum did not collapse
    // into `Fired`. `x::gemm`'s header carries the reconciliation.
    // ── the quantized GEMM router (§45's continuation) ────────────────────
    //
    // Three symbols, ONE program. `gemm.cpp:1999`'s `act_x_w` is a seven-arm
    // switch on `w.dtype`, and these three `gemm.hpp` inlines were its only
    // live callers: each built a `WeightView` and handed it over. The Rust
    // keeps that shape exactly — `bind::quant_gemm::act_x_w` is the switch,
    // `bind::service`'s three functions are the inlines — so `Control::Switch
    // { on: "w_dtype" }` is literally true of the program each of them runs.
    // What each entry point does is PIN the discriminant it can produce, and
    // that is stated in `because` rather than hidden.
    //
    // Why not a `Composition`. `Composed` carries a `&'static [Step]` with
    // `Take` bindings between them, and every arm here breaks that in a
    // different way: the FP8 arm allocates a bf16 weight expansion that no
    // operand of the row names, the INT8 arm allocates three (int8
    // activation, fp32 scale, int32 accumulator) and chooses between two
    // final steps on `beta`, and the MXFP4 arm's intermediate is sized from
    // `N * K` — a product of two operands, which `Take::From(i)` cannot
    // spell. `Composition::agrees` type-checks a step's operands against the
    // composed row's; an intermediate belongs to neither.
    //
    // Why not `Service`. Two of the three arms fire kernels of ours, and one
    // of them (`quant::dequant_mxfp4_to_bf16`) fires nothing else at all.
    // `SERVED`'s rule is *"one library call and nothing else"*, and a body
    // that dequantizes a weight before the library call is not that.
    // ── `gemm::act_x_wt_channel_scaled`'s WALK WAS HERE AND IT IS DELETED ─
    //
    // `Control::Switch { on: "w_dtype" }`. The two paragraphs above — why
    // not a `Composition`, why not a `Service` — are kept: they are the
    // reason this arm could never be data, and they are still true.
    //
    // The host program did NOT move into `x::gemm`, and that is deliberate.
    // It is `driver-cuda/src/bind/quant_gemm.rs`, and it reaches
    // `fire::quant_int8`, whose family belongs to `quant`. Moving the body
    // without its staging kernels would put half a walk in each crate. So
    // `x::gemm`'s `GEMM_XWT_CHANNEL_SCALED` contract is bound `none:` with
    // that sentence, and it moves when `quant` lands.
    //
    // ITS REFUSALS AND ITS `because:`, VERBATIM, because a refusal
    // sentence is a design record and this was its only copy:
    //
    //         control: Control::Switch { on: "w_dtype" },
    //         refuses: &[
    //             "`act_x_w[FP8_E4M3]: scale must be FP32` -- a per-channel scale arriving as \
    //              anything else is a checkpoint whose scale tensor was not converted, and the \
    //              dequant kernel would read it as float bits",
    //             "`act_x_w[INT8 W8A8]: only PerChannel weight scale supported (per-tensor / \
    //              per-group not yet wired)` -- this entry point states PerChannel, so the \
    //              refusal fires only for a caller that reached the INT8 arm another way",
    //             "`act_x_w[INT8 W8A8]: scale must be FP32`",
    //             "`quant weight buffer is smaller than GEMM shape requires` -- \
    //              `validate_quant_weight_view` counts `N*K*sizeof(dtype)` and refuses a short \
    //              buffer rather than letting the dequant kernel run off the end",
    //             "`quant scale tensor is smaller than GEMM shape requires` -- `N` values for a \
    //              per-channel recipe, checked for the same reason",
    //             "`act_x_w[INT4_PACKED]: GPTQ/AWQ W4A16 has no kernel here` -- reachable from \
    //              this entry point if a loader ever produces an INT4 per-channel weight, and it \
    //              PANICS rather than substituting bf16 arithmetic (§46)",
    //             "`ops::act_x_w: unsupported dtype combo` -- the switch's default",
    //         ],
    //         because: "`kernels-cuda/csrc/src/gemm/gemm.cpp:1999` -- `act_x_w`, seven arms on \
    //                   `w.dtype`, reached from the `gemm.hpp:160` inline that pins \
    //                   `QuantMeta::Kind::PerChannel`. The FP8 arm dequants the whole weight to \
    //                   bf16 into a growable scratch (or a cached expansion) and runs the classic \
    //                   GEMM; the INT8 arm quantizes the ACTIVATION per token, runs a \
    //                   `CUBLAS_COMPUTE_32I` `cublasGemmEx` into an int32 accumulator and \
    //                   dequants by the per-row x per-col scale product, choosing a residual-add \
    //                   second step when `beta != 0`. Neither shape is a fixed step list. Now \
    //                   `driver-cuda/src/bind/quant_gemm.rs`",
    // ── `gemm::act_x_wt_grouped_scaled`'s WALK WAS HERE AND IT IS DELETED ─
    //
    // Same close as `act_x_wt_channel_scaled` above and for the same reason:
    // the body is `driver-cuda/src/bind/quant_gemm.rs`'s FP8-blockwise arm,
    // which fires `quant`'s staging kernels. `x::gemm` states the contract
    // and binds it `none:`.
    //
    // ITS REFUSALS AND ITS `because:`, VERBATIM, because a refusal
    // sentence is a design record and this was its only copy:
    //
    //         control: Control::Switch { on: "w_dtype" },
    //         refuses: &[
    //             "`act_x_w[FP8_E4M3]: scale pointer is null -- weight_scale_inv must be attached \
    //              to the materialized WeightStore as an FP32 device tensor` -- the loader-side \
    //              failure this arm exists to name",
    //             "`act_x_w[FP8_E4M3]: scale must be FP32`",
    //             "`quant scale tensor is smaller than GEMM shape requires` -- and the count is \
    //              the subtle one: `ceil(N/gs) * ceil(K/gs)` for FP8, because DeepSeek's \
    //              `weight_block_size = [128,128]` is a 2-D block scale, against `N * ceil(K/gs)` \
    //              for everything else",
    //             "`quant weight buffer is smaller than GEMM shape requires`",
    //             "`group_size != 128` -- `blockwise_w8a8` declines and the dequant path runs; a \
    //              decline here means 'use the other path', not 'fail'",
    //             "`K % 128 != 0 || N % 16 != 0` -- block scales assume a whole number of \
    //              128-wide groups along K and the FP8 tensor-core path needs 16-byte-aligned \
    //              leading dimensions; same decline",
    //             "`ops::act_x_w: unsupported dtype combo` -- the switch's default",
    //         ],
    //         because: "`kernels-cuda/csrc/src/gemm/gemm.cpp:1999` reached from the `gemm.hpp:182` \
    //                   inline, which pins `QuantMeta::Kind::PerGroup`. This is the entry point \
    //                   that reaches `gemm_fp8_blockwise_w8a8_impl` (`:1748`) -- the one arm that \
    //                   does NOT expand the weight: it quantizes the activation to FP8 per \
    //                   128-element token group and hands cuBLASLt both scale tensors through \
    //                   `CUBLASLT_MATMUL_DESC_{A,B}_SCALE_MODE`. It returns `bool`, and `false` \
    //                   falls through to the dequant path -- a host decision, on a host value, \
    //                   with a LATCH behind it (`fp8_block_supported` turns off permanently on a \
    //                   zero heuristic count so later calls skip the round trip). Now \
    //                   `driver-cuda/src/bind/quant_gemm.rs`",
    // ── `gemm::act_x_wt_mxfp4_marlin`'s WALK WAS HERE AND IT IS DELETED ──
    //
    // Same close again. The MXFP4 arm's intermediate is sized from `N * K`,
    // which the paragraph above records as the specific thing `Take::From(i)`
    // cannot spell; in fn-world it is a local. The body is
    // `bind::quant_gemm`'s and moves with `quant`.
    //
    // ITS REFUSALS AND ITS `because:`, VERBATIM, because a refusal
    // sentence is a design record and this was its only copy:
    //
    //         control: Control::Switch { on: "w_dtype" },
    //         refuses: &[
    //             "`act_x_w[MXFP4]: scale must be raw E8M0 bytes` -- the MXFP4 block scale is an \
    //              exponent byte, not a float, and a converted scale tensor is a silently wrong \
    //              answer rather than a crash",
    //             "`act_x_w[MXFP4]: expected per-group scales with group_size=32` -- MXFP4's \
    //              block is 32 elements by definition of the format",
    //             "`quant weight buffer is smaller than GEMM shape requires` -- and the count is \
    //              `(N*K + 1) / 2` here, because MXFP4 packs two elements per byte",
    //             "`quant scale tensor is smaller than GEMM shape requires`",
    //             "`ops::act_x_w: unsupported dtype combo` -- the switch's default",
    //         ],
    //         because: "`kernels-cuda/csrc/src/gemm/gemm.cpp:2078` -- the MXFP4 arm of `act_x_w`, \
    //                   reached from the `gemm.hpp:206` inline. Two steps with an intermediate \
    //                   that neither operand list names: dequant the packed weight to bf16 into a \
    //                   growable device scratch sized `N * K * 2`, then run \
    //                   `gemm::act_x_wt_bf16` against it. `gemm.cpp:2102` was the LAST C++ caller \
    //                   of `quant::dequant_mxfp4_to_bf16`, which is why this row moving is what \
    //                   lets that one be routed. Now `driver-cuda/src/bind/quant_gemm.rs`",
    // ── moe: THE NARROWEST WALK IN THE TABLE, AND IT IS GONE TOO ─────────
    //
    // `moe::moe_grouped_gemm_bf16` stood here, and it was here because the
    // bar is about the SHAPE of a control flow and not its size: the host
    // decided, from numbers known only at fire time, whether this kernel ran
    // at all, and the alternative was a different implementation of the same
    // arithmetic. A `Switch` with two arms, one of which is "not this one" —
    // exactly the case `Walk::refuses` existed to stop anyone writing as a
    // silent fallback.
    //
    // It crossed with `moe`, and it is deleted on the argument the block
    // below makes for the other nine. The switch is `x::moe::supported(m, n,
    // k)` now — three comparisons the reader can read — and the "not this
    // one" arm is `Refusal::Wide`, a value the caller cannot drop where a
    // fallback a walk could only DESCRIBE was one an implementation could
    // silently take.
    //
    // AND THE SWITCH IS BACK, one level down, WHICH IS WHERE A WALK ALWAYS
    // SAID IT WAS. This symbol is a driver op now — see [`SERVED`] — and its
    // body, `driver-cuda/src/fire/moe_grouped.rs`, is the two-armed choice
    // this walk described: `supported()` yes fires the WMMA kernel, no calls
    // `x::gemm::dense::batched_act_x_wt_bf16` over the pointer arrays. What
    // changed between the walk and the body is the only thing that ever
    // mattered: the walk was a DESCRIPTION nothing checked, and the body is
    // the code, so the arm that used to be prose is now a `match` the
    // compiler reads. The bind that stood between the two served exactly one
    // side of the predicate.
    //
    // ITS OTHER ARM IS ALSO WHY `moe::build_moe_ptrs_aligned_bf16` IS A
    // DRIVER OP, and the cross-link is worth a sentence because the two
    // entries sat 200 lines apart and neither of them said it. The arm this
    // symbol takes when `supported()` refuses is BATCHED cuBLAS, and batched
    // cuBLAS is the only reader of the six device pointer arrays that build
    // produces. That reader is a LOWERING of this symbol and not a statement
    // of it: the grouped GEMM's operand list is `(a, weight_base, c,
    // expert_ids)` and names no pointer array anywhere. Six results with no
    // stated consumer are six values `lower.rs:1911`'s liveness frees at the
    // first op past the build, so declaring them would hand the fallback
    // bytes the next allocation owns — a wrong answer, not a refusal. The
    // arrays are the driver's arena instead, and [`SERVED`] carries both
    // symbols: the one that fills them, and this one, which reads them.

    // ── sample: an occupancy query, a growing scratch, and two kernels ────
    //
    // GONE, and not to a walk. `sample` crossed into fn-world (§5 step 5)
    // and its one `Walk` is deleted, for `rope/rope.cu`'s reason above: a
    // `Walk` says "this symbol has no row that can be dispatched, and a
    // hand-written host program launches it instead", and the whole of that
    // sentence is now false. The symbol has a contract (`x::sample::SIGS`)
    // and an `Entry` carrying `unbound` — a refusal made at MODEL LOAD with
    // the sentence its row carried, which is what §0 asks for and what a
    // `Walk` could never do.
    //
    // Everything the entry measured is carried, and this is where each part
    // of it went, because it was the clearest case in the table of the
    // owner's second clause — *where host code is needed to compose kernels,
    // because kernels produce intermediate results or because
    // device-specific tuning is involved, that host code is all Rust* — and
    // both halves of that sentence are true of one body at once.
    //
    // The `because`, verbatim, was:
    //
    // > `kernels-cuda/csrc/src/sample/argmax.cu:89-138` -- the one launcher
    // > left in that file, and five host facts in fifty lines:
    // > `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` at `:102`;
    // > `num_blocks_x = min(num_sms * 2, ceil(vocab / GEMV_WARPS))` at
    // > `:104-107`; a dynamic shared size of `hidden * sizeof(float)` at
    // > `:108-109`; a function-local `static device::u64* s_partial_pairs`
    // > with a `static usize s_pairs_cap` that `cudaFree`s and
    // > re-`cudaMalloc`s whenever `num_blocks_x * num_rows` outgrows it at
    // > `:111-119`; and then TWO launches --
    // > `lm_head_gemv_argmax_int8<bf16>` on `dim3(num_blocks_x, num_rows)`
    // > at `:123-132` and `select_lm_head_argmax_pairs` on
    // > `ceil(num_rows / 128)` at `:136-137` -- with the scratch as the
    // > intermediate that neither the row's operand list nor any
    // > `LaunchRule` names.
    //
    // All five host facts are `x::sample::lm_head_gemv_argmax_int8` now,
    // beside the two declarations that name the `__global__`s it fires, and
    // `driver-cuda/src/fire/lm_head_argmax.rs` is deleted. Its
    // `Control::Switch { on: "pairs_elems > s_pairs_cap" }` is the `if
    // pairs_elems > pairs.cap` in that `fn` — the control shape becomes an
    // `if`, which is §5 step 5's whole sentence about `Specialisation`s
    // applied to a `Walk`.
    //
    // Its four `refuses` are three `Fired::Declined(Refusal::Empty)` arms
    // and two `assert!`s with the same messages, and the distinction the
    // last of the four drew — *"a broken JIT is not a decline"*, `fire/gemv.rs`'
    // rule — is now the type system's: a `Fired` cannot carry a compile
    // failure, so a compile failure is a panic and nothing else can be.
    //
    // The entry's last sentence was *"It stays a row because the SYMBOL is
    // still `table::sample`'s only one: `RUST_SERVED` names it,
    // `emit_c_shim` drops the entry, and `emit_dispatch` writes the arm"* —
    // and that is the one claim that is retracted. There is no row, no
    // `RUST_SERVED` entry and no generated arm: the contract states no
    // `operands`, which is the THIRD of the three shim-dropping mechanisms
    // and needs no list kept in step.
    //
    // The entry above it still refers to this one — *"`sample::
    // lm_head_gemv_argmax_int8` two entries down splits the same way"* —
    // and the split it names (a walk may DRIVE JIT'd kernels, it may not BE
    // one) is unchanged and now visible in `x::sample` as the difference
    // between a host `fn` and the `raw` launchers it calls.

    // ══ `norm/rmsnorm.cu`, `norm/dsv4_hc.cu` and `rope/rope.cu` — ALL GONE ══
    //
    // Fifteen entries stood here, three files, and not one `__global__`
    // between them. All fifteen are deleted and none of them became a
    // library call: both families crossed into fn-world
    // (`.wiki/kernel-x/northstar.md` §5 steps 3 and 5).
    //
    // ── norm/rmsnorm.cu and norm/dsv4_hc.cu ──────────────────────
    //
    // Six `Walk`s: `norm::rmsnorm_bf16_with_fp16`,
    // `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`,
    // `norm::hc_pre_postprocess_bf16`, `norm::hc_post_bf16`,
    // `norm::hc_head_postprocess_bf16` and `norm::hc_rmsnorm_to_f32`. Their
    // host programs are `kernels-cuda-new/src/x/norm.rs`, and
    // `driver-cuda/src/fire/{rmsnorm,dsv4_hc}.rs` are deleted.
    //
    // What each `Walk` recorded is now written where the control flow is.
    // The two `Control::Switch` bodies are `if`s in `x::norm::strided_bf16`
    // and `x::norm::rmsnorm_residual_add_scale_rmsnorm_bf16` with the sweep
    // tables beside them; the four `Control::Supplies` entries are `none:`
    // arms on `x::norm`'s `bind!`, each naming what the context cannot
    // answer, and they surface at MODEL LOAD through `Route::Unbound`
    // instead of at fire time.
    //
    // Two of the four `Supplies` turned out to be supplying nothing a
    // statement does not already carry. `norm::hc_post_bf16` and
    // `norm::hc_rmsnorm_to_f32` are BOUND in `x::norm`: their DSL statements
    // (`dsl.rs:4552` and `:4486`) name every operand their kernels take, and
    // the rows were unsourced by proximity to `hc_pre`/`hc_head`, whose
    // `float` slabs genuinely have nowhere to come from. That was invisible
    // while the launcher's operand list was the only host-side spelling.
    //
    // One refusal CHANGED and it is worth finding here. `hc_post`'s
    // `hc_mult > MAX_HC_MULT` was a `panic!` in `fire/dsv4_hc.rs`; a `fn`
    // reached from a `bind!` body must not panic, so it is a `Refusal` now
    // — spelled `Narrow`, because `Refusal` has no variant meaning ABOVE a
    // compiled maximum. See `x::norm`'s `hc_mult_ok`.


    // ── rope/rope.cu — GONE, and not to a walk ──────────────────────────
    //
    // Nine `Walk`s stood here and all nine are deleted. `rope` crossed into
    // fn-world (`.wiki/kernel-x/northstar.md` §5 step 3): its host programs
    // are `kernels-cuda-new/src/x/rope.rs`, beside the declarations that
    // name the `__global__`s they fire, and `driver-cuda/src/fire/rope.rs`
    // is deleted.
    //
    // **A ported family is not a walk and must not be listed as one.** A
    // `Walk` says "this symbol has no row that can be dispatched, and a
    // hand-written host program launches it instead"; the whole of that
    // sentence is now false for `rope`. Its twelve symbols have contracts
    // (`x::rope::SIGS`), eight of them have binds, and the four that do not
    // carry `Entry::unbound` — a refusal made at MODEL LOAD with the
    // sentence its row carried, which is what §0 asks for and what a
    // `Walk` could never do.
    //
    // The three that were in `device::JIT_DISPATCHED` were never walks and
    // are gone from there too, for the same reason: nothing routes a row
    // any more, because there are no operands on a contract to route.
    // ── ssm/ — GONE, and not to a walk ──────────────────────────────────
    //
    // Ten `Walk`s stood here and all ten are deleted, for `rope`'s reason
    // above: §5 step 5 took `ssm`'s FIVE roots into fn-world. Its host
    // programs are `kernels-cuda-new/src/x/ssm.rs` — twenty-seven of them,
    // in five inline `pub mod`s beside the `unit!`s that name the
    // `__global__`s they fire — and `driver-cuda/src/fire/causal_conv1d.rs`,
    // `fire/gated_delta_net.rs`, `fire/kda.rs` and `fire/nemotron_h.rs` are
    // all four deleted.
    //
    // **A ported family is not a walk and must not be listed as one.** Every
    // one of these ten symbols has a `contract!` in `x::ssm::SIGS` and a
    // `bind!` in `x::ssm::ENTRIES`; the family's four unbindable members
    // carry `Entry::unbound` with their reason as a sentence, refused at
    // MODEL LOAD rather than at a fire. The prose these ten `because:`
    // strings carried is in `x/ssm.rs`, on the `fn` that does the launching,
    // where the geometry it explains is three lines above it.
    //
    // WHAT THE DELETED HEADER MEASURED, because it is a count of the archive
    // and not of this family's rows: `causal_conv1d.cu`, `gated_delta_net.cu`,
    // `kda.cu` and `nemotron_h.cu` held **35 of the archive's 109 `<<<>>>`**,
    // and **TEN of those thirty-five were UNREACHABLE** — eight behind
    // `constexpr false` selectors in `gated_delta_net.cu`'s anonymous
    // namespace, one behind an `if constexpr (false)` at `nemotron_h.cu:143`
    // and one after an unconditional `return` at `nemotron_h.cu:180`. None of
    // the ten is ported and none gets a row or a contract: a declaration for
    // an unreachable kernel is a contract with an empty consumer set. That
    // audit is re-stated in `x/ssm.rs`'s header against the `.cuh` line
    // numbers the rewrite left them at, which is where a reader can now
    // check it. `.wiki/driver/new-horizon.md` §52 is the specification the
    // eleven were written against and §57 records what did not land.
    //
    // The ELEVENTH of the eleven this header counted was
    // `ssm::qwen_gdn_post_conv_prep_bf16`, and it did not go to `x::ssm` —
    // its note is immediately below and it is the one member of `ssm/` that
    // stays out of the family.

    // ── moe/moe_dispatch.cu AND moe/dsv4_routing.cu, both DELETED ────────
    //
    // Five of `moe_dispatch.cu`'s eight came first; the last three followed
    // here, with `dsv4_routing.cu`'s one, and the block below deletes what
    // is left of all nine. **The block that stood here said the last four
    // could not move, and it was right about every mechanism and wrong about
    // the conclusion.** It read: each is already unit-hosted
    // (`LaunchRule::PerRow`, `Rms`, `RouterSort`, `RowsFlat`),
    // `a_walk_is_only_a_walk` refuses a walked symbol that a unit hosts, and
    // `JIT_DISPATCHED` is barred because `emit_rust_dispatch` skips a row
    // whole when an operand is unsourced. All of that still holds. What it
    // concluded — that a symbol split would buy *"a `Walk` that walks
    // nowhere"* — measured the wrong thing.
    //
    // A walk is not only a control SHAPE. `Control::Supplies` is *"a VALUE
    // the launch needs and no row can state, computed on the host"*, and each
    // of these four supplies one: an operand the row leaves unsourced ON
    // PURPOSE. That is the same fact that barred `JIT_DISPATCHED`, read from
    // the other side — the row is unsourced BECAUSE a host has to fill it,
    // which is what a host program is.
    //
    // So each is two symbols on the `moe_grouped_gemm_wmma_bf16` precedent
    // (§60.6), the device halves are `..._dev` in `families::moe`, and both
    // `.cu` files are deleted. `csrc/src/moe/` holds `flashinfer_moe.cu`
    // alone, which is no longer a host file at all: its 817 lines held 0
    // `__global__`, so the host program moved to
    // `driver-cuda/src/fire/flashinfer_moe.rs` and what stayed is a five-
    // function `extern "C"` seam that instantiates CUTLASS templates.
    //
    // NINE OF THE `because:` CLAUSES BELOW ENDED IN "Now
    // `driver-cuda/src/fire/moe_dispatch.rs`" (or `dsv4_routing.rs`), THEN
    // SAID `x::moe`, AND ARE NOW DELETED WITH THE ENTRIES THAT CARRIED THEM.
    // Both `fire/` modules existed for exactly one step: the port moved the
    // C++ launchers into them and §5 step 5 moved them again, into `x::moe`'s
    // twenty host `fn`s, deleting both. The clauses were written at the first
    // move and nothing re-derived them at the second — the same failure
    // `bind/service.rs`'s tombstones have, and worth naming rather than
    // quietly correcting: **a POINTER to where the code went is the part of a
    // note most likely to rot, because it is the only part that is about the
    // tree rather than about the kernel.**
    //
    // And the repoint was the wrong fix, which took one more edit to see. A
    // clause that needs correcting every time the code moves is a clause
    // about the tree, and the tree is the one thing a reader can already
    // look at. The block below deletes the entries instead.
    // ── AND ALL NINE ARE GONE, BECAUSE THE FAMILY CROSSED ───────────────
    //
    // Nine `Walk`s stood here: `moe::scatter_add_weighted_bf16`,
    // `moe::moe_bucket_exact`, `moe::add_moe_route_bias_bf16`,
    // `moe::hash_route_lookup`, `moe::moe_gate_up_decode_gemv_bf16`,
    // `moe::moe_down_decode_gemv_bf16`, `moe::transpose_expert_scales_u8`,
    // `moe::build_moe_ptrs_aligned_bf16` and
    // `moe::reorder_moe_aligned_output_bf16`. §5 step 5 moved all nine host
    // programs into `x::moe` and the entries did not follow — they were
    // REPOINTED, and repointing is what let them survive a crossing they
    // should not have.
    //
    // ONE CHECK GOES RED. THE OTHER EIGHT PASS EVERY CHECK THERE IS, AND
    // THAT is the finding — it is worth more than the deletion.
    //
    // The red one is `a_walk_is_only_a_walk`, which refuses a symbol that is
    // walked AND served. `moe::build_moe_ptrs_aligned_bf16` took
    // `Service::DriverOp` one table down, and the test is right to refuse
    // it: a walk and a service are two answers to one question.
    //
    // THE OTHER EIGHT WERE NEVER GOING TO BE CAUGHT, by two mechanisms that
    // each look like the opposite of a problem.
    //
    // * `Walk::agrees` refuses a walk whose symbol is not a row — *"is
    //   walked and is not a row of `table::KERNELS`"* — and `table/moe.rs`
    //   has been deleted for two edits. It passes anyway, because
    //   `table::sig` reads `KERNELS`, `KERNELS` is `TABLES` concatenated,
    //   and `TABLES` is `ROW_TABLES ++ x::SIGS`. The check is satisfied by
    //   the CONTRACT that replaced the row. That is the floor working
    //   exactly as designed, and it is why a crossing is invisible from
    //   here: the one check that reads a table cannot tell the two worlds
    //   apart, which is the whole point of `table::TABLES` and also the
    //   reason it can never date a walk.
    // * `a_walk_is_only_a_walk` ALSO refuses a walked symbol that a unit
    //   hosts, and all nine are unit-hosted now, in `x::moe`. It passes
    //   because the device halves carry `_dev`: `hash_route_lookup` walks
    //   and `hash_route_lookup_dev` compiles, `moe_bucket_exact` walks and
    //   `moe_bucket_exact_dev` compiles, and so on for all nine on the
    //   `moe_grouped_gemm_wmma_bf16` precedent §60.6 set. **The suffix split
    //   was invented to let a walk coexist with the JIT, and it is the same
    //   mechanism that hides the walk's obsolescence once the host program
    //   is Rust.** A name chosen to keep two things legal beside each other
    //   goes on keeping them legal after one of them stops being true.
    //
    // So nothing mechanical was ever going to ask for this deletion and no
    // test will notice it landing. It is a judgement against the placement
    // rule, and the argument is the next paragraph rather than a red test.
    //
    // THE ARGUMENT IS THE PLACEMENT RULE, read in the direction it is least
    // often read: *data only for what has a reading consumer, and everything
    // that is only executed is code.* A
    // `Walk` describes a control flow its reader CANNOT READ — `.cu` text
    // behind a shim, written down here so a Rust reader knows what the host
    // does. Once the host program is a Rust `fn` in this crate the reader
    // can read it, the description has no consumer left, and what remains is
    // a second spelling of executed code, which decays. The repointing above
    // IS that decay, caught twice and corrected twice before anyone noticed
    // the entries had no business being here at all.
    //
    // WHERE THE NINE FINDINGS WENT, because a deletion should say:
    //
    // * Seven `Control::Supplies` — the `num_routed` grid, the
    //   `(3 * num_experts + 1) * 4` dynamic shared allocation,
    //   `cols`/`out_stride`, `tid2eid`/`vocab_size`, the two
    //   `routes = num_tokens * top_k` products with their expert strides,
    //   and the 3-D `(k_groups/32, n/8, num_experts)` grid over a `(32, 8)`
    //   block — are arithmetic in `x::moe`'s host `fn`s, on the line above
    //   the `Launch` that uses each. None of the seven fits `Launch::flat`
    //   or `Launch::per_row`, so all seven write the literal, which is
    //   §5.1's rule and most of the reason it exists.
    // * `reorder_moe_aligned_output_bf16`'s `Control::Switch` — a
    //   vectorizability test AND a 16-byte alignment test on THREE
    //   allocations — is an `if` in that symbol's host `fn` with the
    //   alignment tests beside it. Two kernels, one symbol, chosen from
    //   three pointers' low bits.
    // * `build_moe_ptrs_aligned_bf16`'s `Control::Supplies` is the one that
    //   was never a grid: `moe_dispatch.cu:246-248`, *if EITHER
    //   shared-expert base is null, overwrite `routed_blocks` with
    //   `max_blocks`* — a host decision read off a POINTER's nullity, which
    //   no `Source` can state and which is why the row left the operand
    //   unsourced. It is code in that symbol's host `fn` with the
    //   `:246-248` citation beside it, and `fire::moe_ptrs::Banks`
    //   documents it for the caller and says in as many words not to
    //   reproduce it.
    // * Every `refuses:` string was the launcher's own words and is a
    //   `Refusal` the `fn` returns now, which is the one of the four that
    //   got STRONGER: a walk's refusal list is prose a reader checks, and a
    //   `#[must_use] Fired::Declined` is a value the caller cannot drop.
    //
    // WHAT THIS LEFT FOR THE LAST CROSSING, AND WHAT THE LAST CROSSING DID.
    // [`WALKED`] was ONE entry — `attn::qkv_decode_qk_norm_rope_write_kv_\
    // bf16` — and it is EMPTY now. `Walk::agrees` is never called on
    // anything, exactly as `the_driver_internal_rows_are_not_statable` was
    // before `table/mod.rs` retired it, and `no_control_shape_is_unevidenced`
    // would have gone RED in the same edit, because every member of every
    // shape is gone. That is correct behaviour and not a chore: a shape enum
    // whose variants have no evidence is vocabulary admitted on nobody's
    // word, and that test exists to say so out loud rather than to pass
    // vacuously.
    //
    // **"24 entries and all 24 are `attn::`" WAS NEVER TRUE OF THIS LIST.**
    // Re-derived two ways at `2dc9957b7` — a regex over `symbol:` inside the
    // bracket-matched static, and an `awk` counting `^    Walk {` between
    // `^pub static WALKED` and `^];` — and both answer ONE. §75, in the
    // sentence that was measuring the thing it stood next to.
    //
    // THE INSTRUCTION THIS BLOCK GAVE — *"the edit that empties this list
    // should delete `Control`, `Walk` and `Execution::Walk` with it"* — WAS
    // MEASURED AND DEFERRED, and the measurement is the useful half:
    //
    //   LIVE CODE           `tests/layers.rs` :326 (a match arm) and :365.
    //                       `:655`'s `matches!(first, Execution::Walk(_))`
    //                       WAS the third and is REPAIRED — see the note
    //                       there. It was red for several arcs and nothing
    //                       said so, which is the property that makes this
    //                       deletion worth doing rather than deferring
    //                       again: a walk is the one executor that can be
    //                       retracted without any list going empty, because
    //                       `WALKED` losing its last member is not a
    //                       compile error.
    //   INTRA-DOC LINKS     `x/gemm.rs` :293, :298, :306 link
    //                       [`crate::execution::Walk`], `Walk::refuses` and
    //                       [`crate::execution::Control`] BY PATH, so
    //                       deleting the types breaks four rustdoc links in
    //                       a file this pass does not own
    //   PROSE               ~30 further citations across
    //                       `x/{gemm,moe,ssm,driver_internal,attn}.rs`,
    //                       `device.rs`, `families/{attn,graph}.rs`,
    //                       `bind/service.rs`, `fire/mod.rs` and
    //                       `kernels-cuda/tests/sources.rs`
    //
    // So the deletion is a six-file edit across three ownership grants, and
    // `Walk`/`Control` are the ROW WORLD's vocabulary — a walk classifies a
    // row the row world still interprets — so they retire with the row
    // interpreters in step 6, as one deletion rather than three.
    // `no_control_shape_is_unevidenced` carries the tripwire meanwhile: it
    // returns early on an empty list and says why there.


    // ── ssm/nemotron_h.cu's last two — GONE, and not to a walk ──────────
    //
    // `ssm::build_nemotron_moe_ptrs_decode_batched_bf16` and
    // `ssm::build_nemotron_moe_ptrs_aligned_bf16` stood here and both are
    // deleted with the rest of `ssm/`, for the reason the block above gives.
    //
    // THE THING THEY RECORDED THAT NOTHING ELSE DOES, and it is a finding
    // rather than a measurement, so it is kept: §52.3, §56 and §57.5 all
    // called these two blocked, and all three were about the same thing —
    // `Ty::BufArrayOut` over a driver-owned slab that no `Source` names, so
    // `emit_rust_dispatch` writes no arm and no trace reaches either. **That
    // is still true and the port does not change it.** What was conflated
    // with it is the LAUNCHER, which needs a unit and a declaration and no
    // `Source` at all — the same distinction `ssm::kda_*` made two passes
    // earlier, and the distinction fn-world makes everywhere by construction.
    //
    // In `x::ssm::nemotron_h` they are two `pub fn`s with `contract!`s and
    // no `bind!`: the arrays of device pointers they fill have no `Source`
    // and the floor's `Cx` cannot reach a driver slab, so both carry
    // `none:` arms and surface as `Route::Unbound` at model load with that
    // sentence. That is the SAME refusal these entries encoded, moved from a
    // trace that never ran to a load that says why.
    //
    // The contract symbols end `_bf16` where the device rows end `_dev_bf16`,
    // which is deliberate and is in `x/ssm.rs`'s header: the `<<<>>>` at
    // `nemotron_h.cuh` names the `_dev_` form and the model text names the
    // other, and a rename would break one of the two.

    // ── THE THREE `attn/kv_paged.cu` WALKS ARE RETRACTED ─────────────────
    //
    // `attn::write_kv_to_pages`, `attn::write_kv_explicit_bf16` and
    // `attn::dequant_kv_cache_layer_to_bf16_active` have CROSSED INTO
    // FN-WORLD — `x::attn`'s `WRITE_KV_TO_PAGES`, `WRITE_KV_EXPLICIT` and
    // `DEQUANT_KV_ACTIVE`, over the `attn/kv_paged` unit, with their host
    // programs in `x::attn::kv_paged`. Their `table/attn.rs` rows are gone
    // and §52.11 requires the retraction: a walked symbol must satisfy
    // `unit_of(sym).is_none()`, and all three now host device text.
    //
    // Same shape as `layout::embed`'s retraction below: the classification
    // stopped existing rather than becoming false. A `Walk` says *this
    // symbol's host program is a branch no row can state*; a `fn` IS that
    // branch, written in the language the branch was always in, so there is
    // nothing left to classify.
    //
    // WHAT THE THREE ENTRIES ARGUED, kept because each is still true of the
    // program and would otherwise have to be rediscovered:
    //
    //   * WHY A WALK AND NEVER A COMPOSITION, argued three times here.
    //     `Composition::agrees` requires a fixed SEQUENCE — two or more
    //     steps, every step one launch, every operand filled from the
    //     composed row by index. `write_kv_to_pages` chooses among five
    //     cache schemes and one of the five arms is itself two launches;
    //     `write_kv_explicit_bf16` fires a second launch only when the layer
    //     carries envelopes, which is a run-time property of the cache and
    //     not a step; the dequant is five arms of which exactly one runs.
    //     **A composition that spelled any of them would be a sequence that
    //     is not always run, which is the one thing a sequence may not be.**
    //
    //   * WHY NOT A `Specialisation`: the discriminant is a FIELD of
    //     `Ty::KvCacheLayerView`, which is `device::Fact::Opaque`, and no
    //     `Term` reads a struct field. §26.9 names these symbols for exactly
    //     that. The five `Specialisation`s this family DID carry were over
    //     `hnd_layout` and turned out to be a second copy of a decision the
    //     driver had always made in Rust; `device::SPECIALISED` is empty.
    //
    //   * WHY THE DEQUANT COULD NOT LEAVE WITH ITS SIBLINGS, and the rule it
    //     was obeying: *do not transcribe a live switch into a second
    //     language until the first copy is dead or dies in the same change*
    //     — *"the copy that is NOT called is the one that drifts."* This
    //     switch had four live C++ callers in
    //     `driver-cuda/csrc/attn/attention_flashinfer.cu`, C++ calling C++
    //     by symbol with no shim between and therefore nothing a Rust
    //     dispatch could intercept. The FA2 host program moving to Rust
    //     emptied that set; the prohibition was discharged by the thing it
    //     was waiting for rather than overridden. Those four call sites are
    //     `bind::service`'s four FA2 entry points, and they now call
    //     `x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active` — which
    //     is why that body is a `pub fn` and not an arm's private one.
    //
    //   * WHAT DID NOT COME ACROSS, and deliberately: `:260`'s
    //     `CUDA_CHECK(cudaGetLastError())`. `hand::fire` reports a launch
    //     failure against the symbol it fired, which is strictly more than a
    //     sticky error attributing the fault to whichever of four kernels
    //     ran last, and a synchronous check would serialise a path that runs
    //     once per layer per step.
    //
    // The throws are still throws and the declines are still declines; the
    // declines gained information, because `Refusal` can carry two reasons
    // where `kv_paged.cu:197`'s `if (a || b) return;` could carry none.
    // ═══ `attn/mla_paged.cu` — BOTH LAUNCHERS, IN ONE CHANGE ═════════════
    //
    // Both are `Supplies`, and both supply values that come out of a
    // `MlaCacheLayerView` the dispatch passes as ONE argument and the kernel
    // takes as five fields. That is why neither could be a `Source`: a
    // `Source` can hand a value to the kernel and cannot take it apart.
    // ═══ `attn/dsv4_compress.cu` — THE FOUR THAT SURVIVED ═══════════════
    // ═══ `attn::combine_attn_outputs_bf16` — CROSSED, AND ITS WALK WITH IT ══
    //
    // Its `Walk` STOOD HERE and is retracted rather than moved. The entry
    // read `Control::Supplies { the BLOCK width, head_dim clamped into
    // [32, 256] -- and NOT into LaunchRule::PerHeadElementwise's [32, 128],
    // which is the whole reason no rule states this }`, refusing `N <= 0`
    // and a unit/row disagreement, and its `because` argued at length that
    // rowing it would agree at deepseek_v4's 128-wide heads and stop
    // agreeing at the first config that widened one, invisibly — a slower
    // kernel and never a wrong answer, so nothing fails and nothing reports.
    //
    // **The argument was right and it has stopped being about a
    // classification.** `Control::Supplies` says a host program computes
    // something no `Source` can produce; a `fn` computes its own geometry,
    // so there is nothing left for the classification to describe. The whole
    // content of this entry was one clamp, and the clamp is now four lines
    // from the launch in `x::attn::combine_attn`, with the divergence from
    // `SINK_BLOCK_MAX` stated where a reader of the launch will find it.
    //
    // This is `layout::embed`'s shape a second time, one screen down: a
    // `Walk` whose `Control` was the only thing keeping it out of a row, and
    // which fn-world dissolves rather than satisfies. `x::attn`'s
    // `COMBINE_ATTN_OUTPUTS` is the contract, `combine_attn_outputs_bf16`
    // the host program, and the `table::attn` row and the `RUST_SERVED`
    // entry went in the same change. `driver-cuda/src/fire/dsv4_compress.rs`
    // keeps the other three.
    // ═══ `attn::dsv4_boundary_meta_decode`, `attn::dsv4_boundary_meta_paged`
    // AND `attn::attention_compressed_paged_bf16` — RETRACTED. ═══
    //
    // Three `Walk`s stood here, and the sentence four lines up already said
    // what happened to them: *"`driver-cuda/src/fire/dsv4_compress.rs` keeps
    // the other three."* It does not; the file is DELETED. All three crossed
    // into fn-world as `x::attn`'s `DSV4_BOUNDARY_META_DECODE`,
    // `DSV4_BOUNDARY_META_PAGED` and `DSV4_ATTENTION_COMPRESSED_PAGED`, with
    // their host programs, and `a_walk_is_only_a_walk` would have caught the
    // three that remained.
    //
    // Each `Control::Supplies` was TRUE and none of it is lost: the decode
    // pair's grid is `Launch::flat(n, 128)`, spelled by the floor rather than
    // by a driver, and the compressed row's `(head_dim + 128) * sizeof(float)`
    // is four lines of `x::attn::attention_compressed_paged_bf16` beside the
    // launch it sizes. That is the `Walk` DISSOLVING rather than being
    // satisfied -- `combine_attn_outputs_bf16`'s shape, and `layout::embed`'s
    // before it -- because a `fn` supplies its own geometry and there is no
    // longer a classification to make.
    //
    // The refusals gained rather than lost. `Dsv4Decline`'s five variants
    // are `Refusal::Empty { what }` and `Refusal::Narrow { what, at }`, which
    // name the extent and carry the offending value; the enum's variant names
    // had to be read against the C++ to know which `<=` they came from.
    // ═══ `attn::write_kv_explicit_bf16_devwin` — RETRACTED, AND ITS `Walk`
    //     CARRIED THE CLEAREST FALSE FACT IN THIS FILE ══════════════════════
    //
    // CROSSED INTO FN-WORLD as `x::attn`'s `WRITE_KV_EXPLICIT_DEVWIN`, a
    // `contract!` with a `none:` arm: the host program
    // (`x::attn::kv_paged::write_kv_explicit_bf16_devwin`) is complete and
    // `win_d` has no producer in `AttnCtx` to bind it from.
    //
    // THE FALSE FACT, kept because §75 collects this class. This entry's
    // `because` used to end *"the row is fully sourced, so this moves a LIVE
    // dispatch off the shim."* `table/attn.rs`'s row stated `Source::Unbound`
    // on ALL NINE operands, and `abi.rs:810` skips a row with any `Unbound`
    // operand WHOLE — so `emit_rust_dispatch` never wrote a dispatch arm,
    // this `RUST_SERVED` entry was never reached, and the `bind::service`
    // shim never had a caller. **The claim was true of the SIBLING
    // `attn::write_kv_explicit_bf16` row, which was fully sourced, and was
    // written for this one.** Not stale — never true here. A fact derived
    // from one member of a set and stored next to a different member.
    //
    // §58 AND §60.6 WERE BOTH REASONING ABOUT A MECHANISM NOBODY CONSULTED.
    // §58 asked whether a specialised symbol may also be walked; §60.6
    // dissolved it by moving the device rows to `_dev` names. Both were
    // arguments about `device::SPECIALISED`, and `driver-cuda` had been
    // choosing `#hnd`/`#nhd` in Rust and firing by name the whole time.
    //
    // What the entry recorded and is still true of the program: two
    // `<<<n_max, 256, 0, stream>>>` on `layer.hnd_layout`; `n_max <= 0`
    // declines; a cache that is not native bf16 PANICS, and one carrying
    // envelopes PANICS, because envelope maintenance was never windowed and
    // merging a window's rows against a full-row envelope would widen it
    // with rows the window excluded.
    // ═══ `attn/qkv_fused.cu`'S WALK IS RETRACTED — AND IT WAS THE LAST ═══
    //
    // `attn::qkv_decode_qk_norm_rope_write_kv_bf16` crossed into fn-world as
    // `x::attn`'s `QKV_DECODE_FUSED`, a `contract!` and a real `bind!` over
    // the `attn/qkv_fused` unit. Its row was FULLY SOURCED and on
    // `RUST_SERVED`, so this crossing really did move a live generated
    // dispatch — unlike the unsourced retractions below it, which dropped
    // nothing. A walk classifies a row the row world still interprets, and
    // `ROW_TABLES` is now `&[]`, so there is nothing left to interpret and
    // `WALKED` is EMPTY.
    //
    // What the walk said, kept because it is still true of the kernels.
    // `qkv_fused.cu:31-159` was ONE launcher over FOUR kernels and EIGHT
    // instantiations, and the last `attn` file to fall *because it is the one
    // whose dispatch a `Specialisation` cannot state*. `families::attn` had
    // written its own refusal: the row states `RowsPackedHeadsNarrow` and the
    // warp triple states `WarpPackedHeads`, which is the whole refusal in two
    // lines — plus the leg that settles it, that lifting the
    // one-rule-per-row invariant would not land the row anyway, because
    // `head_dim == 64 | 128 | 256` is unspellable (`Term::Multiple { of: 64 }`
    // holds of 192). **A host program spells it, and that is the answer the
    // whole of §5 step 5 is.** Two choices compose: `head_dim` picks the form
    // (`:92`, `:96`, `:100`, fallthrough at `:105`) and `rope_table !=
    // nullptr` picks the `USE_ROPE_TABLE` arm (`:56`, `:107`). The warp grid
    // is `ceil(num_requests * (q + kv) / 8)` at block 256 — warpS per block,
    // not threads. `win` is `nullptr` on every path, the `_devwin` twin that
    // passed a real one having been deleted earlier, but the parameter stays
    // on `qkv_decode_fused_dispatch` because the kernels read it per row.
    //
    // The refusals it listed are all hoisted before the single launch in
    // `x::attn::qkv_fused::qkv_decode_fused_dispatch`, and it gained one the
    // walk could not have: a null `q_out` is `Refusal::Absent` rather than a
    // store through null, because the producer can write one and the device
    // text does not test it.
    //
    // ═══ `attn/dsa_indexer.cu`'s THREE WALKS ARE RETRACTED ══════════════
    //
    // All three crossed into fn-world together: `x::attn`'s
    // `DSA_INDEX_{KNORM_ROPE,Q_ROPE,TOPK_MASK}`, three contracts, one bind
    // and two `none:` arms. A walk classifies a row the row world still
    // interprets, and there is no row left to interpret -- so the entries go
    // rather than being restated, exactly as `attn::combine_attn_outputs_\
    // bf16`'s and `attn::write_kv_explicit_bf16_devwin`'s did.
    //
    // What the three walks said, kept because it is still true of the
    // kernels: one launch each, no host arithmetic beyond a grid and a block,
    // and `index_q_rope`'s block width -- `((n_heads + 31) / 32) * 32` with a
    // one-warp floor -- was the reason `dsa_indexer.cu`'s header called
    // itself *waiting on a launch rule*. The answer was that it wanted a
    // HOST, and it has one: `x::attn::dsa_index_q_rope_bf16`, with the
    // rounding in `dsa_indexer::q_rope_block`.
    //
    // `dsa_index_topk_mask`'s row WAS fully sourced, so its crossing
    // replaced a live generated arm with a bind rather than dropping an
    // entry nothing reached. The other two were unsourced and their `none:`
    // arms shadow nothing -- and what blocks them is a STATEMENT, not a
    // query: `rope_dim` is in no shape, no param and no context, and
    // `knorm_rope`'s statement names no weight bank for a kernel that reads
    // a LayerNorm weight and a bias.
    // ═══ `attn/attention_naive.cu` — THE MTP STATE PAIR ═════════════════
    //
    // Both are `Supplies`, and what they supply is the same thing twice: the
    // grid. Neither `total_tokens` nor `num_requests` reaches its own
    // kernel's argument list as the extent it is -- `total_tokens` never
    // reaches the kernel at all -- so the rectangle is a host quantity even
    // though a rule states the same one.
    // ═══ `attn::mtp_shift_hidden_bf16` AND
    // `attn::mtp_update_pending_hidden_bf16` — RETRACTED. ═══
    //
    // Both crossed into fn-world as `x::attn`'s `MTP_SHIFT_HIDDEN` and
    // `MTP_UPDATE_PENDING_HIDDEN`, with their host programs, and
    // `driver-cuda/src/fire/attention_naive.rs` is deleted. Their device
    // halves are `x::attn::attention_naive`'s unit, written for this
    // crossing: `attn/attention_naive.cuh` had none.
    //
    // Both `Control::Supplies` entries named a GRID the kernel does not read
    // — `total_tokens` for the shift, `num_requests` for the refresh — and
    // both are `Launch::per_row` now, three lines above the launch. The
    // refresh's is the interesting one and its measurement is preserved in
    // the unit: opening the grid over ROWS rather than requests would have a
    // fire of eight requests and ninety-three tokens write eighty-five slots
    // that are not its own.
    //
    // NEITHER BINDS, and the blocker is `slot_ids` for both. The only query
    // that reaches a request→slot map is `Cx::gdn`, whose `slot_ids_d` is
    // exactly this pointer and which answers `None` unless the fire has a
    // recurrent shape — and an MTP head on a dense transformer has none.
    // The refresh has a second, of a different kind: its `StateRef` names
    // `RecurrentState` for a buffer that is `RecurrentStateCache`'s THIRD
    // half, addressed by slot rather than by layer, and `Slab` has no
    // variant for it.
    // ═══ `attn::split_qkv_bf16_devwin` — RETRACTED. ═════════════════════
    //
    // It crossed into fn-world as `x::attn`'s `SPLIT_QKV_DEVWIN`, with a
    // REAL bind, and `driver-cuda/src/fire/split_packed.rs` is deleted.
    //
    // Its `Control::Supplies` was *"xblocks = ceil(max(q_dim, kv_dim) / 256)
    // and grid.y = n_max, the FIRE's lane count and not the statement's
    // rectangle"*, and both halves are supplied by `x::attn::split_qkv_bf16_
    // devwin`, four lines above the launch. Its two `refuses` are a
    // `Refusal::Empty` and the named panic, unchanged.
    //
    // **THE INTERESTING PART IS THE HALF THAT WAS FALSE.** The `because`
    // gave two reasons no rule and no bind could state this, and the second
    // read: *"the JIT binder would resolve `In`/`Out` through the
    // statement's window and double-window them"*. `bind/mod.rs:3973`:
    //
    // > `let row = if kernel.ends_with("_devwin") { 0 } else {
    // > launch.rows.start };` -- *"The `_devwin` forms are the stated
    // > exception. Their contract is BASE pointers."*
    //
    // The binder exempts this kernel BY NAME, three hundred lines before a
    // `Cx` exists, and `Fire::arg_in` returns `bound.args[i].ptr` unchanged.
    // The general fact was true of every other symbol and was stored beside
    // the one the exception is for.
    //
    // The first reason was true and is now answered rather than retracted:
    // `Cx::rows().total` is `DispatchCtx::rows_total`, and `bind/facts.rs:319`
    // names a `_devwin` launch in the field's own doc. So the whole of the
    // `because` survives as measurement; what went is the conclusion.
    //
    // The rest of it, because none of it is retracted: *"`kernels-cuda/csrc/
    // src/attn/split_packed.cu:35-52` -- one launch of `split_qkv_devwin
    // <bf16>` at `:46` over `dim3 grid(xblocks, n_max)` at `:45`. ...
    // `grid.x` covers the WIDER of the two outputs and not the packed width,
    // which `split_packed.cuh` licenses in one direction only."* All three
    // are in the `fn`, quoted rather than paraphrased.
    // ═══ `layout/embed.cu` — THE LAST LAUNCHER IN `csrc/src/layout/` ═════
    //
    // GONE, and not to a walk. `layout` crossed into fn-world (§5 step 5),
    // for `rope/rope.cu`'s reason above: the symbol has a contract
    // (`x::layout::SIGS`) and a bind, so it is not a symbol a host program
    // launches instead of a dispatch — it is a symbol whose dispatch IS the
    // host program.
    //
    // The whole of what this entry measured, because none of it is
    // retracted. The `because` read:
    //
    // > `kernels-cuda/csrc/src/layout/embed.cu:25-53` -- TWO launches of one
    // > template, `embed<true>` at `:41` and `embed<false>` at `:47`, and
    // > the arm is chosen from a host test no `Source` can produce:
    // > `layout/embed.cuh:18-25` says so in as many words and §10.5 refuses
    // > the invented `Source` that would fix it. The test also SIZES the
    // > grid -- `per_row = vec ? hidden/8 : hidden` and the extent is
    // > `num_tokens * per_row` widened to `long long` -- so the answer is an
    // > operand and a rectangle as well as an arm, which is why it is a
    // > `Switch` and not merely a `Supplies`. The row moved from
    // > `table::driver_internal` to `table::layout` in the same change,
    // > because `RUST_SERVED` is gated on `table::sig` resolving and
    // > `driver_internal` is outside `TABLES`; the move stands on its own,
    // > `lower.rs:1462` having named the symbol from a statement all along.
    //
    // Its `Control::Switch { on: "hidden % 8 == 0 && weight and y both
    // 16-byte aligned -- embed<true> or embed<false>" }` is
    // `x::layout::vectorisable`, a `pub fn` returning `bool`, and the two
    // arms are two `raw::embed` calls in `x::layout::embed_bf16` differing
    // in their symbol string. Its three `refuses` are two
    // `Refusal::Empty` arms and a panic.
    //
    // **The `Source` §10.5 refused is still refused, and fn-world is why it
    // never comes back**: `layout/embed.cuh:18-25`'s *"No `Source` in
    // `kernels/src/lib.rs` produces 'is this pointer 16-byte aligned'"* was
    // an argument about what a ROW can say, and a contract says nothing
    // about operands at all. The predicate is a host `fn` reading a pointer,
    // which is what it always was.
    //
    // The last clause — the row's move from `table::driver_internal` to
    // `table::layout` "to make this line legal" — is now history rather than
    // mechanism: `RUST_SERVED` is not what drops the shim entry any more.
    // An empty operand list is, and it is not gated on a table at all.
    // ═══ `attn::mla_prepare_bf16` AND `attn::write_mla_to_pages` —
    // RETRACTED. ═══
    //
    // Both crossed into fn-world as `x::attn`'s `MLA_PREPARE` and
    // `WRITE_MLA_TO_PAGES`, with their host programs, and
    // `driver-cuda/src/fire/mla_paged.rs` is deleted.
    //
    // `mla_prepare_bf16`'s `Control::Supplies` was this enum's own worked
    // example — `heads_per_block`, *"passed to the kernel AND divides the
    // head axis of the grid"* — and it is now `x::attn::mla_heads_per_block`,
    // four lines above the launch it divides, with the C++'s measurement
    // comment intact. `q_blocks`, the `kv_a_row_stride` default and the
    // YaRN ramp bounds travelled the same way; `ramp_bounds` is now one
    // crate away from its `.cuh` rather than two.
    //
    // `write_mla_to_pages` supplied `page_size`, `kv_lora_rank` and
    // `qk_rope_head_dim`, described here as *"three operands the kernel
    // declares and no `Source` can reach, because the view is one dispatch
    // argument and its fields are five."* THAT SENTENCE IS THE ARGUMENT FOR
    // `Cx`: `Cx::mla_layer` returns all five as `x::MlaLayer`, so the
    // unpacking is the host program's own first lines and there is no
    // classification left to make. The `Walk` dissolved.
    //
    // Neither BINDS. `Cx::mla_layer` refuses because `AttnCtx` carries no
    // MLA layer list, which is a producer no fire reaches rather than a
    // value that does not exist — see the `dsv4` retraction above for the
    // other kind.
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
/// `Execution::kind`], which is the `match` that makes a fifth arm of
/// [`Execution`] a compile error rather than a silent reclassification --
/// and which the fourth arm ([`Execution::Walk`]) had to answer before it
/// could exist. It answered [`Kind::Op`]; the module header argues why.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Kind {
    /// One `<<<>>>`, one instantiation. Migrable, whether or not migrated.
    Kernel,
    /// A host program over kernels.
    ///
    /// Ours, when the program is [`Execution::Composed`] and every step is a
    /// row of this table. **Or a library's device text that we compile**,
    /// when the program is an [`Execution::Walk`] and its shape comes from
    /// the input -- a tower's `for im in 0..num_images` launching `.cuh`
    /// templates that never had rows, FlashInfer's `switch (head_dim)`
    /// launching one of hundreds of instantiations. What makes both an `Op`
    /// is that the SYMBOL is a host program; who wrote the device text under
    /// it is the thing `Walk` deliberately declines to claim.
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
    ///
    /// Four arms and three kinds -- see [`Kind::Op`], which two of them share.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Execution::Jit(_) => Kind::Kernel,
            Execution::Composed(_) | Execution::Walk(_) => Kind::Op,
            Execution::Service(_) => Kind::Service,
        }
    }
}

impl core::fmt::Debug for Execution {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Execution::Jit(row) => write!(f, "Jit({})", row.sig.symbol),
            Execution::Composed(steps) => write!(f, "Composed({} step(s))", steps.len()),
            Execution::Walk(walk) => write!(f, "Walk({})", walk.control.label()),
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
    // `gemm::act_x_wt_bf16_out_fp32` AND `gemm::grouped_act_x_wt_bf16` WERE
    // HERE. Their services were real — *"one `cublasGemmEx`, bf16 in / fp32
    // out; `gemm.cpp:1030-1058` is the whole body"* and *"one
    // `cublasGemmGroupedBatchedEx`; `gemm.cpp:1242-1294`. Measured, not read:
    // it is CLASSIC cuBLAS, not the cuBLASLt the previous entry claimed"* —
    // and both sentences are kept here because a `Service` claim is a
    // measurement about a body.
    //
    // The rows are gone with `table::gemm` and the bodies are
    // `x::gemm::act_x_wt_bf16_out_fp32` and
    // `x::gemm::grouped_act_x_wt_bf16`, which `bind!` binds. A `Service`
    // entry says *"the generated dispatch should call Rust instead of the
    // shim"*; an `Entry` says the same thing without a generator.
    // `gemm::batched_act_x_wt_bf16` WAS HERE. The service was real --
    // `gemm.cpp:1145-1241` is `cublasGemmGroupedBatchedEx` falling back to
    // `cublasGemmBatchedEx`, both arms the library's -- and the row is gone
    // anyway (`new-horizon.md` §38), because nothing asked for it. A true
    // statement about a launcher is not a statement that anything calls it.
    // MLA'S ABSORB PAIR MOVED TO `Service::DriverOp`, one block down, when
    // `attn` crossed them. The `Service::Cublas` FINDING was true and is kept
    // verbatim in the sentences there -- one `cublasGemmStridedBatchedEx`
    // each, nothing else in the body, so extracting a kernel extracts
    // nothing. What changed is only the question `x::route` asks this list.

    // ── CUTLASS ───────────────────────────────────────────────────────────
    ("moe::flashinfer_cutlass_moe_bf16",   Service::Cutlass,
     "THE EXEMPLAR. `csrc/third_party/flashinfer_moe/*.cu` holds 0 `__global__`; `src/moe/flashinfer_moe.cu` holds 0 and calls no kernel of ours; and `cutlass/` is in no source directory of this repo -- CPM fetches it into `target/**/_deps/flashinfer-src/3rdparty/cutlass` at configure time. The kernels are templates in headers we do not have. It returns `bool`, but a service that declines is still a service: the fallback is the CALLER's, not the row's. The 0 `__global__` was read a second time and finished the argument: a file with no device text in it is not device text, so the 817-line HOST program (workspace query, arch probe, autotuner, per-device tactic memo, on-disk tactic cache, dispatch) is `driver-cuda/src/fire/flashinfer_moe.rs` and `src/moe/flashinfer_moe.cu` is a five-function `extern \"C\"` seam over `CutlassMoeFCRunner` with two standard headers. It is NOT on `RUST_SERVED`: since `moe` crossed into fn-world this symbol is `x::moe::MOE_FUSED_CUTLASS`, a `contract!` with no `Entry` -- the driver-op shape, the third row of `x/mod.rs`'s registration table -- and `bind::service::moe_flashinfer_cutlass_moe_bf16` is the seam through which it reaches `driver-cuda/src/fire/flashinfer_moe.rs`. The `Service::Cutlass` classification is unchanged and is the only claim this entry makes"),

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
    // `comm/` is GONE: `custom_all_reduce.cu` had zero `__global__` and zero
    // `<<<>>>` by the time it was measured -- a 664-line HOST PROGRAM wearing
    // a `.cu` extension for linkage -- and it, `custom_all_reduce.hpp` and
    // `custom_all_reduce_stub.cpp` are deleted. The whole lifecycle is
    // `driver-cuda/src/fire/all_reduce.rs`: peer-access enablement, the IPC
    // handle exchange, the `RankData` slab, the fusion plane's allocations
    // and Lamport init, and the NCCL crossover query.
    //
    // Both rowed entry points still take a `CustomAllReduce*` the driver owns
    // as their first argument, and both still forward into headers this repo
    // does not carry: `csrc/vendor/flashinfer/` holds `attention/` only, and
    // there is no in-repo copy of `flashinfer/comm/vllm_custom_all_reduce.cuh`
    // or `flashinfer/comm/trtllm_allreduce_fusion.cuh`. That is now a
    // *refusal with a name expression* rather than a link-time absence --
    // `fire::all_reduce::Decline::NoDeviceText`.
    //
    // These two are the first rows in the tree on BOTH this list and
    // `RUST_SERVED`, and the pairing is exact: `SERVED` says the body is one
    // library call, `RUST_SERVED` says Rust is the one issuing it.
    ("comm::all_reduce_bf16",                   Service::CustomAllReduce,
     "`impl_->allreduce<__nv_bfloat16>`, vLLM's one/two-shot NVLink kernel; was `custom_all_reduce.cu:603-621`, now `fire/all_reduce.rs::CustomAllReduce::all_reduce_bf16`, header fetched not vendored. A null `car` is a REFUSAL, not a fallback (`Decline::NoInstance`)"),
    ("comm::all_reduce_residual_rmsnorm_bf16",  Service::CustomAllReduce,
     "`flashinfer::trtllm_allreduce_fusion`'s `kARResidualRMSNorm` pattern -- 1 of the 240 template points `kernels.def`'s `PIE_AR_FUSION_PATTERN` axis existed to prune; was `custom_all_reduce.cu:623-662`, now `fire/all_reduce.rs::CustomAllReduce::all_reduce_residual_rmsnorm_bf16`. Declines when `can_fuse_residual_rmsnorm` refuses -- the fused landing IS this kernel and there is no other way to spell it"),

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

    // ── the aligned MoE pointer build, reclassified by re-reading it ──────
    //
    // The ONE entry on this list that is a `__global__` and still a driver
    // op, and the reason is not a device API: it is a LIFETIME. Its six
    // pointer arrays have no stated consumer -- the batched-cuBLAS fallback
    // that reads them is a lowering of `moe::moe_grouped_gemm_bf16`, not a
    // statement -- so declaring them as trace results would hand the plan six
    // values `lower.rs:1911`'s liveness frees at the first op past the build,
    // and the GEMM would dereference bytes the next allocation took. Not a
    // refusal; a wrong answer. So the arrays are the driver's arena, the
    // symbol takes the third registration shape, and `x::moe` declares a
    // `contract!` with NO `Entry` -- not even a `none:`, which would shadow
    // this arm and refuse a live model at load.
    ("moe::build_moe_ptrs_aligned_bf16", Service::DriverOp,
     "the aligned MoE leg's step 3 of 8: one launch of `build_moe_ptrs_aligned<bf16>` that bakes three staging bases into six device pointer arrays. It DECLARES `gu_stage`/`act_stage`/`out_stage`, the destinations every op below it writes into, so the aligned leg cannot start without it -- which makes this symbol the gate on retiring `moe::flashinfer_cutlass_moe_bf16`, the only leg qwen3.5 decode takes and the one every aligned-leg condition already falls back FROM. Body: `driver-cuda/src/fire/moe_ptrs.rs`, whose per-fire bump arena carves the six, called from `bind/mod.rs`'s driver-op table beside `pie_lora_qkv_correction`. It had a `Walk` until this entry existed and `a_walk_is_only_a_walk` is right to refuse both -- see the note where that walk stood for where its `Control::Supplies` went"),

    // ── and the ONE symbol that reads those six arrays ────────────────────
    //
    // Steps 4 and 5, the entry above's only consumer, and the only symbol in
    // the tree that came to this shape FROM A WORKING BIND. `x::moe` bound it
    // and fired the WMMA kernel for every shape `supported` accepts; what a
    // bind cannot do is the other side of that predicate, and qwen3.5 decode
    // fires one statement on each side:
    //
    //   gate_up  M=16 N=2*I=1024 K=H=2048  K > SHORT_K -> batched cuBLAS
    //   down     M=16 N=H=2048   K=I=512   supported   -> WMMA
    //
    // `supported`'s `K > 512` arm names its own replacement -- "above which
    // cuBLAS wins" -- and the replacement needs the cuBLAS handle (which §3.3
    // forbids `Cx` to hand over) and the six arrays (which are nobody's
    // operand). Nor could the caller finish the choice: `bind/mod.rs` states
    // *"a refusal is not a fallthrough"*, so a `Refusal::Wide` returned from a
    // bind is the answer the driver reports and not the first half of a
    // decision. One symbol, two implementations, one host program that picks.
    //
    // WHAT THIS ENTRY ACTUALLY CLOSED is a gap that had been open since the
    // aligned leg was written: `DispatchRefusal::ShapeDeclined` said "the C++
    // driver reads the same predicate and takes a batched-cuBLAS fallback;
    // until that fallback exists here, saying so is the only honest answer",
    // and the fallback DID exist -- `x::gemm::dense::batched_act_x_wt_bf16`,
    // ported under §45.2 with its row struck and no caller. The missing thing
    // was never the arithmetic; it was that nothing named the function.
    ("moe::moe_grouped_gemm_bf16", Service::DriverOp,
     "the aligned leg's steps 4 and 5, and the only consumer the pointer build's six arrays have. Two implementations behind one symbol, chosen by `x::moe::supported`: the WMMA kernel inside its rectangle (3.0x the library at both of qwen3.5's shapes, `x/moe.rs`'s decode census) and `x::gemm::dense::batched_act_x_wt_bf16` outside it, which is `gemm.cpp:1145-1241` -- grouped-batched falling back to plain batched, with the stream-capture latch. qwen3.5 needs both: gate_up is `K=2048` against a `SHORT_K` of 512 and down is `K=512`. Body: `driver-cuda/src/fire/moe_grouped.rs`, which reads `ctx.moe_ptrs` and picks its triple by the bank pointer the statement names. A `contract!` with NO `Entry` in `x::moe`, third registration shape -- the bind that stood there served half the symbol and its `Refusal::Wide` was final by `bind/mod.rs`'s \"a refusal is not a fallthrough\""),

    // ── the dense GEMMs, reclassified by the step-5 `gemm` port ───────────
    //
    // These four WERE `Service::Cublas` and `Service::Cutlass` — findings
    // about a body, and both findings were true and are kept in the
    // sentences below. What changed is not the body but the QUESTION this
    // list is now also asked: `x::route` reads it as the driver-op oracle,
    // and `Service::DriverOp` is the arm that says *"something the driver
    // owns already fires this."*
    //
    // For all four that something is the cuBLAS handle. `ctx.cublas` is
    // created once per shell, carries `cublasSetMathMode`, and has a stream
    // rebound per fire; it is §3.3's forbidden surface and a `Cx` will never
    // answer it. So `x::gemm` declares twelve contracts and NO `Entry` —
    // `x::SIGS`' third registration shape — and the driver keeps firing
    // them with the handle in hand. Writing `none:` arms instead would have
    // shadowed this arm and refused every dense matmul at load.
    //
    // The host programs are `x::gemm`'s now, not `bind::service`'s, and all
    // four take `handle: *mut c_void` as their first parameter.
    ("gemm::act_x_wt_bf16",             Service::DriverOp,
     "`bind::quant_gemm::act_x_w` is the router the lowering reaches (it is emitted as `gemm::act_x_w`, this row's `lowered_as`), and its bf16 arm is a direct Rust call to `x::gemm::dense::act_x_wt_bf16` — the autotuner, the cuBLASLt plan cache and the on-disk tactic cache. `beta` is `spec.beta_one`'s residual fold, which only a driver op can see"),
    ("gemm::act_x_wt_bf16_out_fp32",    Service::DriverOp,
     "one `cublasGemmEx`, bf16 in / fp32 out; `gemm.cpp:1030-1058` was the whole body and the body is now `x::gemm::act_x_wt_bf16_out_fp32`. The measurement that made it `Service::Cublas` still holds -- extracting a kernel from it extracts nothing -- and the handle it needs is the driver's"),
    ("gemm::act_x_wt_bias_bf16",        Service::DriverOp,
     "a `gemm::act_x_wt_bf16` and then a `norm::add_bias_bf16` over the result (`gemm.cpp:2395-2398`), which `COMPOSED` stated as two ops and fn-world spells as a two-call body: `x::gemm::act_x_wt_bias_bf16`. Its `beta` is a literal 0.0, so the one fact the dense sibling needs, this one never asks"),
    ("gemm::grouped_act_x_wt_bf16",     Service::DriverOp,
     "one `cublasGemmGroupedBatchedEx`; `gemm.cpp:1242-1294`. Measured, not read: it is CLASSIC cuBLAS, not the cuBLASLt the previous entry claimed. The group boundaries are fire-global and no `Source` names one, so its consumer is and always was `fire::lora`'s hand-written staged apply, calling `x::gemm::grouped_act_x_wt_bf16` directly"),

    // ── FLASHINFER'S SIX, RECLASSIFIED OUT OF [`WALKED`] ──────────────────
    //
    // See the tombstone in [`WALKED`] for why they moved and why their
    // `refuses` lists did not come with them. Here is the part that belongs
    // on this list: **what resource the driver owns that a `Cx` cannot
    // reach.**
    //
    // For five of the six, name it and stop: a `DecodePlanCache` or a
    // `PrefillPlanCache`, `Box`ed inside [`crate::prepare::Prepare`]'s
    // `DecodePlan` / `PrefillPlan`, mutated by the planner, read by every
    // dispatch of the fire and outliving all of them. That is §3.3's
    // forbidden surface named exactly, and it is the whole test.
    //
    // THE SIXTH IS THE FINDING AND IT IS RECORDED HERE RATHER THAN SMOOTHED
    // OVER. `attn::attention_flashinfer_prefill` names NO resource: it
    // builds a plan cache on its own stack, allocates nothing that survives
    // the call and asks one read-only capability query. It is a driver op by
    // the SECOND condition -- *can a `Cx` state what the body reads?* -- and
    // it cannot, because the body walks `qo_indptr_h` and `kv_page_indptr_h`
    // on the CPU. Those are HOST mirrors of a device CSR, no `Cx` query
    // answers a host pointer, and reading the device copy host-side is a
    // synchronise, which §0 forbids inside a fire. `split_qkv_bf16_devwin`
    // is the same shape and the pair is now the evidence that *name the
    // resource* is necessary and NOT sufficient.
    ("attn::dispatch_attention_flashinfer_decode", Service::DriverOp,
     "the plain paged decode. RESOURCE: `DecodePlanCache` (and `decode_plan_full`, the gemma-4 second kind -- `bind::attn_plan` picks between them by the layer's window, which is a DECISION and the reason it is module-level in `bind/mod.rs` rather than nested in the generated dispatcher that used to be its only caller). Arm: `bind::fa2_decode`, over `fire::flashinfer_fa2_dispatch::attn_dispatch_attention_flashinfer_decode` -> `::decode` -> `fire::flashinfer_fa2::fire_decode` -- a KV dequant of the active pages, the three-arm variant cascade (`decode_arm`, whose ORDER is load-bearing: a windowed layer with a soft cap takes the soft-cap arm), then one `fire_raw` with a `DecodeParams` by value. Transcribed from `driver-cuda/csrc/attn/attention_flashinfer.cu:490-522, 660-692` (deleted)"),
    ("attn::dispatch_attention_flashinfer_decode_capture", Service::DriverOp,
     "the same decode with the attention scores captured. RESOURCE: the same `DecodePlanCache`, plus the score sink itself -- `AttnCtx::score_out` is arena-STABLE across a fire because the capture predicate is FOLDED, so one exec serves a fire that wants scores and one that does not, and an address recorded at bind time has to still be right when the predicate goes true. A trace value could not promise that. Arm: `bind::fa2_decode_capture`, over `fire::flashinfer_fa2_dispatch::decode_capture`; the capture arm cascade is `families::fa2`'s `DecodeArm::{CaptureFull, CaptureWindow}` and the params mirror is `fa2::params::DecodeScoreParams`. THE POST-KERNEL IS STILL A ROW: `attn::attn_score_normalize` in `families::attn`'s `ATTN_SCORE_POST`, fired by `driver-cuda/src/fire/attn_score.rs`. Transcribed from `attention_flashinfer.cu:532-594` (deleted)"),
    ("attn::dispatch_attention_flashinfer_prefill_bf16", Service::DriverOp,
     "the plain paged prefill. RESOURCE: `PrefillPlanCache`. Arm: `bind::fa2_prefill`, over `fire::flashinfer_fa2_dispatch::prefill` -> `fire::flashinfer_fa2::fire_prefill`. The `DISPATCH_NUM_MMA_KV` switch (`utils.cuh:116-133`) does NOT survive as a switch: the archive instantiated all four points because the choice came from a device query, and `fa2::PrefillGeometry::derive` makes the query once on the host so the fire names ONE unit -- the largest single saving of this migration and invisible in the row count. An SM90 plan is `Decline::Sm90Unported` and NOT a forward to `kernels-cuda`'s hopper unit, per §44.7. Transcribed from `attention_flashinfer.cu:776-836` (deleted)"),
    ("attn::dispatch_attention_flashinfer_prefill_capture_bf16", Service::DriverOp,
     "the prefill sibling of the capturing decode. RESOURCE: `PrefillPlanCache` plus the arena-stable sinks, and it takes one MORE than the decode form -- `folded_out` beside `score_out`, since a prefill's raw scores and their per-request fold are different extents. Arm: `bind::fa2_prefill_capture`, over `fire::flashinfer_fa2_dispatch::prefill_capture`; `make_prefill_params` is the one Rust function all four prefill dispatches share. TWO post-kernels, both rows: `attn::attn_prefill_score_normalize` on `dim3(nr, nh, window) x 256` and `attn::attn_prefill_score_fold` on `dim3(nr, 32u) x 256`. The `32u` and `attn_score_fold_heads`' `64u` are DIFFERENT literal grid axes in one file, which is why neither is a `LaunchRule`. Transcribed from `attention_flashinfer.cu:831-933` (deleted)"),
    ("attn::dispatch_attention_flashinfer_prefill_custom", Service::DriverOp,
     "the custom-mask prefill. RESOURCE: `PrefillPlanCache` -- raised by `Prepare::CustomPlan` and not `PrefillPlan`, because the prepare that stages the mask and its CSR is a different one, plus `AttnCtx::mask_d` / `mask_indptr_d`, which ride the ctx for the score sink's reason -- the predicate is folded. Arm: `bind::fa2_prefill_custom`. Two arms, `PrefillArm::{Custom, CustomSoftcap}`, and NO causal axis: the mask IS the causality, so a custom dispatch that also set `CAUSAL` would mask twice. `window_left` is written `-1` for the same reason and NOT taken from the cache -- the one place the params filler's cache-sourced window is overwritten. Transcribed from `attention_flashinfer.cu:1115-1224` (deleted)"),
    ("attn::attention_flashinfer_prefill", Service::DriverOp,
     "the PLANLESS prefill: it plans and fires in one call, so the planner's own refusals are its refusals too. RESOURCE: **NONE, and that is the point** -- `fire::flashinfer_fa2_dispatch::attn_attention_flashinfer_prefill` builds a `PrefillPlanCache` on its own stack, drops it, and asks `plan_device()`, a read-only capability query and not a pool. It is a driver op by the second condition alone: it walks `AttnCtx::qo_indptr_h` and `kv_page_indptr_h`, the HOST mirrors of the CSR, to learn `num_pages_in_batch` -- and no `Cx` query answers a host pointer. Arm: `bind::fa2_prefill_planless`, which is the one of the six that also needs `rows`. `causal = true`, `full_attention_variant = false`, from the C++'s own call at `attention_flashinfer.cu:935-1075` (deleted)"),

    // ── MLA's absorb pair, reclassified by `attn`'s crossing ──────────────
    //
    // Both were `Service::Cublas`, both still ARE one `cublasGemmStrided\
    // BatchedEx` and nothing else. The reclassification is `x::gemm`'s: a
    // finding about the body, re-read as an answer to *"does something the
    // driver owns already fire this?"* -- and the resource is the same
    // `ctx.cublas` those twelve name, so the same third registration shape
    // follows. `x::attn` declares two contracts and NO `Entry`.
    ("gemm::mla_absorb_q_to_latent_bf16", Service::DriverOp,
     "one `cublasGemmStridedBatchedEx` over the head axis, `batchCount = heads`; `gemm.cpp:2419-2442`, whose own comment names the per-head scalar kernels it REPLACED. Body: `x::attn::mla_absorb_q_to_latent_bf16`, taking `handle: *mut c_void` as fn-world spells a resource it cannot own. Arm: `bind::mla_absorb_q`, reading the four widths off `LaunchSpec::params` because both absorbs take the WHOLE `kv_b_proj` bank and slice it themselves. The archive's `tokens <= 0 || heads <= 0` early return is a `Refusal::Empty` now: under `void` a caller could not tell it from a launch"),
    ("gemm::mla_absorb_latent_to_v_bf16", Service::DriverOp,
     "the second absorb, same single strided-batched call; `gemm.cpp:2444-2468`. It reads the SECOND half of each head's bank -- `kv_b_proj + qk_nope_dim * kv_lora_rank` bf16 elements -- which is the one pointer arithmetic step the port had to carry, and `OP_T` on the weight where the first absorb takes `OP_N`. Body: `x::attn::mla_absorb_latent_to_v_bf16`; arm: `bind::mla_absorb_v`"),
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
/// and their shim entries are live. Dropping their entries because the
/// classification is true of them would delete the only path they have.
/// A finding is not a plan.
///
/// `moe::flashinfer_cutlass_moe_bf16` used to be the second example here and
/// **is now on this list instead**, which is worth reading as a correction
/// rather than a deletion. The sentence was *"CUTLASS in a C++ file this
/// crate compiles"*, and the C++ file turned out to hold 817 lines with
/// **zero** `__global__`, `<<<>>>` and `__device__` — a workspace query, a
/// tuning cache, an autotuner and a dispatch. The CUTLASS is in
/// `${flashinfer_SOURCE_DIR}/csrc/fused_moe/cutlass_backend/
/// cutlass_fused_moe_kernels.cuh`, which CPM fetches and which is now the
/// LAST thing in this family still compiled ahead of time — a state, not a
/// settlement, because the principle has no exception any more (no nvcc,
/// under any circumstances). What that file needed was not a shim entry but
/// an `extern "C"` instantiation seam, five functions wide, with the host
/// program in `driver-cuda/src/fire/flashinfer_moe.rs`. The classification
/// was right; the inference *"therefore its entry must stay"* was the part
/// that did not follow.
///
/// # The invariant
///
/// Every symbol here is in [`SERVED`], [`COMPOSED`] or [`WALKED`] — you may
/// not take over a row whose execution has not been classified — and
/// `tests::` below asserts it. The other half of the invariant, that
/// `driver-cuda` actually spells a function for each, is asserted in
/// `driver-cuda`'s own `bind::service`, which is the crate that would fail
/// to compile.
/// # If you are auditing for orphaned shim entries, SUBTRACT THIS LIST
///
/// The failure worth preventing is real: `emit_c_shim` writes a body that
/// forwards to `::pie_cuda_driver::kernels::<symbol>`, and if the launcher
/// has been deleted the generated translation unit does not compile — before
/// `driver-cuda/build.rs`'s `every_call_resolves_in_the_shim` can run, since
/// that check reads the archive this TU is compiled INTO. So it is worth
/// scanning for from outside a build, and the scan has three terms, not two:
///
/// ```text
///   at_risk = stated(tables)          // operands non-empty
///           - device::JIT_DISPATCHED  // NVRTC compiles it
///           - execution::RUST_SERVED  // <-- THIS ONE, abi.rs:144
/// ```
///
/// An audit that drops the third term reports **every entry on this list**,
/// which is the whole point of the list, and it reports them as the thing
/// they were deliberately made to stop being. That has happened once at
/// exactly 36-for-36, and the three remedies such an audit proposes are all
/// wrong here: `JIT_DISPATCHED` is refused for a walk by
/// `tests::a_walk_is_only_a_walk`, deleting the row deletes a symbol model
/// text still states, and taking `operands` off drops the generated arm that
/// [`crate::abi::emit_rust_dispatch`] writes against `bind::service`. **A
/// row on this list is closed. Leave it.**
///
/// The rows a scan SHOULD find are the ones whose C++ launcher went without
/// any of the three lists learning about it — and the honest way to see one
/// is to resolve each surviving entry's `kernels::<ns>::<fn>` against a
/// declaration in the headers `emit_c_shim` is handed and a definition in
/// some `.cu`/`.cpp`. Both trees, because `table::driver_internal`'s rows are
/// backed by `driver-cuda/csrc`, not by the archive.
///
pub static RUST_SERVED: &[&str] = &[
    // The four pure-cuBLAS bodies. Each is one library call and argument
    // assembly; `bind/service.rs` is that assembly in Rust, ~130 lines of
    // C++ for ~120 of Rust and no third place for the transposes to live.
    // `gemm::act_x_wt_bf16_out_fp32` and `gemm::grouped_act_x_wt_bf16` WERE
    // HERE and are `x::gemm`'s two real `bind!` arms now.
    // MLA's absorb pair stood here. Both crossed: `x::attn`'s
    // `MLA_ABSORB_{Q_TO_LATENT,LATENT_TO_V}` contracts, bodies in
    // `x::attn::mla_absorb_*`, arms in `driver-cuda/src/bind`. A driver op
    // has no shim entry for this list to drop.
    // The composition [`COMPOSED`] already stated, executed. This one is
    // here for a reason the other four are not: `gemm.cpp:2393` called
    // `norm::add_bias_bf16`, a row of OURS that is already migrated, and
    // while that call existed the row could not be routed and
    // `norm/add_bias.cuh` could not be its only copy. Taking the composition
    // over is what frees it.
    //
    // IT IS FREED. The call went with this entry and the
    // `#include "norm/add_bias.hpp"` it left behind in `gemm.cpp` went after
    // it; no `.cu`, `.cpp` or `.hpp` in either archive calls the launcher
    // now. What has NOT happened is the consequence: the row is still absent
    // from `device::JIT_DISPATCHED`, so `abi::emit_c_shim` still emits
    // `pie_k_norm_add_bias_bf16` and `norm/add_bias.cu` still holds one
    // `<<<>>>` for it. That is one line in `JIT_DISPATCHED` and a file
    // deletion, in that order, and it is deliberately not bundled here —
    // this entry's claim is about the composition, and routing a row is a
    // change the shim's contents depend on.
    //
    // `"gemm::act_x_wt_bias_bf16"` WAS HERE. It is
    // `x::gemm::act_x_wt_bias_bf16`, a `fn` whose body is two calls, and the
    // consequence this comment said had NOT happened has now happened from
    // the other end: with `table::gemm` and `table::driver_internal` both
    // deleted, `emit_c_shim` is handed no row that names
    // `norm::add_bias_bf16`, so no `pie_k_norm_add_bias_bf16` is emitted
    // whether or not the symbol is in `JIT_DISPATCHED`.
    // ── the dense bf16 GEMM (the keystone) ───────────────────────────────
    //
    // The hottest row in the tree, and the one every other `gemm.cpp` entry
    // was waiting on. Its body is not a library call: it is a runtime
    // autotuner over three kernel families with a per-device memo and an
    // on-disk tactic cache, and one of the three families is a kernel of ours
    // fired through the JIT. [`WALKED`] states it; `fire/gemm.rs` is the
    // program; `bind::service::gemm_act_x_wt_bf16` is the spelling this
    // list's test looks for.
    //
    // WHAT IT FREES: `crates/kernels-cuda/csrc/src/gemm/gemm.cpp` and
    // `gemm.hpp`, and with them the last of the `gemm/` family. The C++
    // caller that outlived every previous attempt was
    // `bind::quant_gemm::gemm_bf16`, which reached this body through
    // `ffi::pie_k_gemm_act_x_wt_bf16` — Rust calling the shim calling the
    // archive — and the note there argued at length that doing so *"remains
    // correct"*. It was correct under the old standard and is void under the
    // one that replaced it: no `.cpp` anywhere, whether or not a Rust
    // dispatch could have intercepted it. That call is now
    // `crate::fire::gemm::act_x_wt_bf16`, as is
    // `bind::service::gemm_act_x_wt_bias_bf16`'s first step.
    //
    // What held it for three arcs, and how that ended: `gemm_bf16_impl`
    // called `gemv_bf16`, whose `bool` meant *"I did not launch"*, and a row
    // cannot decline (§45.5). The answer was never to let the row decline —
    // it was that a driver-owned launch is not a row. `fire::gemv::gemv_bf16`
    // returns `Gemv::Declined(_)` naming WHICH test refused, so the tuner's
    // GEMV candidate is a `matches!(.., Gemv::Launched)` in the same
    // short-circuiting position the C++ put it in, and the ambiguity is gone
    // by construction rather than by convention.
    //
    // `"gemm::act_x_wt_bf16"` WAS HERE. `fire/gemm.rs` is `x::gemm::dense`
    // and `fire::gemv` is `x::gemm::gemv`; the whole paragraph above still
    // describes them, one crate down. `Gemv::Declined(_)` did NOT become
    // `Fired::Declined(Refusal)` — see `x::gemm`'s header for why the two
    // spellings stay apart.
    // ── the quantized router (§45's continuation) ────────────────────────
    //
    // Three entry points, one body. `gemm.cpp:1999`'s `act_x_w` switched on
    // `w.dtype` across seven arms and these three inlines were its only live
    // callers; `bind::quant_gemm::act_x_w` is that switch, and the three
    // functions in `bind::service` are the three inlines. [`WALKED`] states
    // each as `Control::Switch { on: "w_dtype" }` — a walk rather than a
    // `Composition`, because the arms allocate intermediates that no
    // `Take` can name and two of them decline.
    //
    // WHAT THIS ENTRY FREES, and it is the reason it exists: `gemm.cpp` was
    // the LAST C++ caller of four migrated `quant::dequant_*` rows —
    // `dequant_fp8_e4m3_to_bf16`, `..._per_channel`, `..._per_group` and
    // `dequant_mxfp4_to_bf16`. C++ composing with C++ is a call no Rust
    // dispatch can intercept, so those four could not be routed while these
    // arms were in the archive. They are named in `device::JIT_DISPATCHED`
    // with this change. `norm::residual_add_bf16` was held by the same file
    // (the INT8 `beta != 0` arm) and is freed with them.
    //
    // ALL THREE WERE HERE and all three are gone with `table::gemm`.
    // `bind::quant_gemm` is unchanged and still holds the switch; what it
    // lost is a generated arm pointing at it. `x::gemm` states the three
    // contracts and binds each `none:` with the sentence above — the body
    // fires `quant`'s staging kernels, so it moves when `quant` does.
    // The first row on this list whose body is NOT a library call.
    //
    // Every entry above forwards into cuBLAS or into a composition ending in
    // cuBLAS. This one fires a `__global__` of ours, JIT'd by NVRTC out of
    // `moe/moe_grouped_gemm.cuh`, on a `Launch` `driver-cuda` builds by hand
    // because `LaunchRule` cannot state `dim3(N / kNTile, max_blocks)`.
    //
    // **That it can be on the same list is the whole claim.** The list is
    // what `emit_c_shim` reads to drop a shim entry and what `emit_dispatch`
    // reads to write `crate::bind::service::…` instead of
    // `crate::bind::abi::ffi::pie_k_…`; neither asks what is behind the
    // name, and no model text can tell. `table::moe`'s `moe_grouped_gemm`
    // row is byte-for-byte what it was.
    //
    // What it frees: `csrc/src/moe/moe_grouped_gemm.cu` and its `.hpp`,
    // which held nothing but this launcher and its predicate.
    //
    // The unit is rowed as `moe::moe_grouped_gemm_wmma_bf16`. A walked
    // symbol may not also be unit-hosted (`a_walk_is_only_a_walk`), so the
    // stated name and the fired name differ by a suffix, and `fire/moe.rs`
    // spells the fired one.
    //
    // THAT SUFFIX OUTLIVED ITS REASON AND THEN HID ONE — the walk block for
    // `moe` argues it, because eight stale walks passed every check on the
    // strength of exactly this split.
    //
    // **IT LEFT THIS LIST WITH THE FAMILY.** §5 step 5 took `moe` into
    // fn-world as `x::moe`, so `table::sig("moe::moe_grouped_gemm_bf16")`
    // now answers a `Contract::sig` — and a contract states NO operands,
    // which is the third mechanism by which a symbol loses its
    // ahead-of-time C entry and is exactly what `every_taken_over_row_is_stated`
    // tests for. Naming an unstated row here would drop nothing. The host
    // program is `x::moe::moe_grouped_gemm_bf16` and its `bind!` arm fires
    // it; `fire/moe.rs` and `bind::service::moe_moe_grouped_gemm_bf16` are
    // both deleted.
    // The second, and the one that fires TWO of our kernels with a device
    // buffer between them, STOOD HERE.
    //
    // `moe::moe_grouped_gemm_bf16` above is one JIT'd launch behind a host
    // predicate. This was the shape past that: an occupancy query decides the
    // grid AND an operand, a scratch buffer the row's operand list does not
    // mention carries the first kernel's output to the second, and the caller
    // sees one symbol. That description is still exactly right and is now
    // `x::sample::lm_head_gemv_argmax_int8`, firing
    // `sample::lm_head_gemv_argmax_int8_bf16` and
    // `sample::select_lm_head_argmax_pairs` — the same two rows, still both
    // `LaunchRule::Unstated`, because `unit!` cannot state a rule at all and
    // neither geometry ever was one.
    //
    // What it freed, and the files are still freed: `csrc/src/sample/argmax.cu`
    // and its `.hpp` — the last two under `csrc/src/sample/`, so the directory
    // goes with them.
    //
    // Why the line is gone rather than repointed: `sample` crossed into
    // fn-world and its one contract states no `operands`, which drops the
    // shim entry by the third mechanism. `RUST_SERVED` says "a Rust host
    // program serves this instead of the archive"; an empty operand list
    // says "there is no ahead-of-time launcher to describe", and the second
    // is the true sentence here.
    // ── `norm/rmsnorm.cu`, `norm/dsv4_hc.cu`, `rope/rope.cu` ─────────────
    //
    // Fifteen entries and three whole files, and the shape is the one
    // `moe::moe_grouped_gemm_bf16` opened above: not one of these is a
    // library call. Every kernel behind them is JIT'd by NVRTC out of
    // `kernels-cuda-new/csrc/src/{norm,rope}/*.cuh`; what was in the archive
    // was the host program that chose the instantiation, computed the
    // rectangle and supplied the values no `Source` can state. The programs
    // are `driver-cuda/src/fire/{rmsnorm,dsv4_hc,rope}.rs`, and [`WALKED`]
    // states each with the `.cu` line it came from.
    //
    // **WHAT THIS ENTRY FREES, and it is why the norm two are first.**
    // `new-horizon.md` §49.1 counts seven rows routable today -- unit
    // present, every operand sourced -- blocked only because C++ still calls
    // their launcher. THREE of the seven were held by `norm/rmsnorm.cu`
    // alone:
    //
    // ```text
    //   norm::rmsnorm_bf16           rmsnorm.cu:42 (the launcher), :59, :63
    //   norm::rmsnorm_strided_bf16   rmsnorm.cu:80 (the launcher), :42
    //   quant::bf16_to_fp16          rmsnorm.cu:64
    // ```
    //
    // §10.10's rule is that a launcher goes when its WHOLE consumer set has
    // gone and the shim is only one consumer; `rmsnorm.cu` composing with
    // its own siblings is C++ calling C++, which no Rust dispatch can
    // intercept. `fire/rmsnorm.rs` makes those three calls Rust, and all
    // three are named in `device::JIT_DISPATCHED` with this change.
    // `quant::bf16_to_fp16`'s launcher lives in `quant/dequant_wna16.cu`,
    // which is another agent's file and stays; routing it drops its shim
    // entry, and the `<<<>>>` goes when that file's own consumer set does.
    //
    // FOUR OF THE FIFTEEN TAKE OVER A LIVE GENERATED ARM, and it matters
    // which four, because for them the bits must not move:
    //
    // ```text
    //   norm::rmsnorm_residual_add_scale_rmsnorm_bf16   gemma-4, 4x per layer
    //   rope::rope_bf16                    gemma-2, nemotron-h, gemma-3n, llama
    //   rope::qk_rmsnorm_rope_bf16_rounded gemma-4
    //   rope::rope_yarn_original_bf16      gpt-oss
    // ```
    //
    // Their AOT rows source every operand, so `abi::emit_dispatch` was
    // already writing an arm that called `ffi::pie_k_*`; the arm now calls
    // `bind::service::*` and nothing above it can tell. The ports keep every
    // instantiation and every threshold the C++ chose between, including the
    // 2560 that picks a block width and the two ramps computed on the host.
    //
    // The other ELEVEN state no `Source` on some operand, so
    // `emit_dispatch` skipped them WHOLE -- and `bind/mod.rs`'s hand-written
    // `match` has one arm left that is none of them. They were unreachable
    // before this change and are unreachable after it, for exactly the same
    // reason, which is a `Source` this tree cannot yet write and not
    // anything this change did. `families/rope.rs` says what is missing for
    // its five and `WALKED` says it for all eleven.
    // `norm`'s six stood here. They are gone with the family, for `rope`'s
    // reason below: `norm` crossed into fn-world and its twenty-eight
    // contracts state no `operands`, which is the THIRD of the three
    // mechanisms by which a symbol loses its ahead-of-time C shim entry.
    //
    // The four in the ELEVEN above were `norm::hc_*`; two of those four are
    // BOUND now (`hc_post_bf16`, `hc_rmsnorm_to_f32`), because a `bind!`
    // body reads `Cx` directly and their statements carry every operand the
    // kernels take. The unsourced ROW was the obstacle, not the statement.
    // `rope`'s nine stood here. They are gone with the family: `rope`
    // crossed into fn-world and its twelve contracts state no `operands`,
    // which is the THIRD of the three mechanisms by which a symbol loses
    // its ahead-of-time C shim entry (`abi.rs`'s `stated()` drops a row
    // with an empty operand list, before `emit_c_shim` sees it).
    //
    // That mechanism is the right one and the other two were not. A
    // `RUST_SERVED` entry says "a Rust host program serves this instead of
    // the archive"; an empty operand list says "there is no ahead-of-time
    // launcher to describe". The second is the true sentence about a family
    // whose launcher is a `fn` in `src/x/` that holds its `.cuh` by
    // `include_str!`, and it needs no list to
    // be kept in step.
    // `ssm`'s TWELVE stood here — the deleted header said "eleven across
    // four files" and it was written before the two
    // `build_nemotron_moe_ptrs_*` were appended, so thirteen names were on
    // this list at one time or another and `qwen_gdn_post_conv_prep_bf16`
    // was the thirteenth. All of them are gone, for `rope`'s reason above:
    // §5 step 5 crossed `ssm` into fn-world and its twenty-seven contracts
    // state no `operands`, which is the THIRD of the three mechanisms by
    // which a symbol loses its ahead-of-time C shim entry.
    // `driver-cuda/src/fire/{causal_conv1d,gated_delta_net,kda,nemotron_h}.rs`
    // are all four deleted and the programs are
    // `kernels-cuda-new/src/x/ssm.rs`.
    //
    // WHAT THE DELETED HEADER RECORDED AND NOTHING ELSE DOES:
    //
    // * **THREE `.cu` FILES WENT WHOLE** with the original entry —
    //   `ssm/{causal_conv1d,gated_delta_net,kda}.cu` — and
    //   `ssm/nemotron_h.cu` lost nine of its eleven launches. That is **35
    //   of the archive's 109 `<<<>>>` accounted for: 25 ported, 10 found
    //   unreachable and deleted.** `csrc/src/ssm/` holds only `.cuh` now and
    //   `x::ssm`'s five units `include_str!` all five of them.
    //
    // * **FIVE of them sourced every operand**, so
    //   `abi::emit_rust_dispatch` wrote a live arm for each:
    //   `causal_conv1d_prefill_batched_bf16` (qwen3.5, kimi-k3),
    //   `qwen_gdn_post_conv_prep_bf16` (qwen3.5, and `driver_internal`),
    //   `recurrent_gated_delta_step_batched_gqa_state_bf16` (qwen3.5
    //   decode), `nemotron_mamba_split_bf16` and
    //   `nemotron_mamba_ssm_batched_bf16` (nemotron-h, zamba). **Those five
    //   arms are gone with the operand lists that generated them, and the
    //   `bind!`s in `x::ssm::ENTRIES` are what serve them now** — which is
    //   the whole substitution §5 step 5 is, made once for five symbols
    //   whose bits nobody was allowed to move before.
    //
    // * The four chunked-prefill symbols sourced every operand too; the
    //   **TWO KDA symbols did not**, the same `Source::Scratch` gap `rope`
    //   describes, and `emit_rust_dispatch` skipped them whole. They are
    //   `none:` arms in `x::ssm::kda` now — `Route::Unbound` at model load
    //   with the sentence, rather than a symbol no trace could reach and
    //   nothing saying so. (An `attn::write_kv_explicit_bf16_devwin` entry
    //   stood here too and was backed out; §56.1 says why.)
    //
    // * The two `build_nemotron_moe_ptrs_*` launchers were the reason
    //   `nemotron_h.cu` stayed alive: `Ty::BufArrayOut` over a driver-owned
    //   slab that no `Source` names. §57.5 records it beside the eight
    //   `kv_paged.cu` is blocked on, because it is the same missing word.
    //   They are the other two `none:` arms.
    //
    // `"ssm::qwen_gdn_post_conv_prep_bf16"` WAS THE ONE `driver_internal`
    // ROW ON THIS LIST (the table above marks it as such), and §5 step 5
    // took it to `x::driver_internal` as a `fn` with **no `contract!`** — so
    // there is no `Entry` and no arm of any kind for it, generated or bound.
    // Its caller is `bind/mod.rs`'s GDN path, which calls the function.
    // ── moe/moe_dispatch.cu AND moe/dsv4_routing.cu, ALL NINE ────────────
    //
    // THREE OF THESE NINE HAVE LIVE GENERATED ARMS and the other six do not,
    // which is the difference worth stating before anyone audits the list.
    // `moe::moe_gate_up_decode_gemv_bf16`, `moe::moe_down_decode_gemv_bf16`
    // and `moe::reorder_moe_aligned_output_bf16` source every operand in
    // `table::moe`, so `emit_rust_dispatch` writes a
    // `crate::bind::service::…` arm for each and a model trace reaches them
    // TODAY -- `crates/model/src/qwen_3_5/forward/mod.rs:222` states the
    // reorder. Those three arms change target with this list and must not
    // change behaviour, which is what `bind::service` is checked against.
    // `moe::transpose_expert_scales_u8` and
    // `moe::build_moe_ptrs_aligned_bf16` are unsourced and get no arm, like
    // the `kda` pair above.
    //
    // **The last four are new and they are why both `.cu` files are gone.**
    // Every one of them is unsourced in `table::moe`, so like the pair above
    // they get NO generated arm and `bind::service` gains nothing: listing
    // them here does one thing only, which is drop four `pie_k_moe_*` shim
    // entries whose definitions no longer exist. That is also the mechanism
    // -- not `JIT_DISPATCHED`, which `driver-cuda/build.rs`'s `armless` check
    // would refuse for exactly the same unsourcedness, and not an empty
    // `operands` list, which would cost a new `Unstated` variant in
    // `driver-cuda/tests/launch_abi.rs` and throw away four transcribed
    // signatures to buy what one line here buys. Their four device halves are
    // `..._dev` rows in `families::moe`; their `Walk`s are above.
    //
    // **ALL NINE LEFT THIS LIST WITH THE FAMILY**, for the reason
    // `moe::moe_grouped_gemm_bf16` above states: `x::moe` is fn-world now,
    // every one of these symbols resolves through a `Contract::sig`, and a
    // contract states no operands — so `emit_c_shim` emits nothing for them
    // with or without this list, and `every_taken_over_row_is_stated` would
    // be right to refuse them. Their host programs are `x::moe`'s twenty
    // `fn`s. FOUR of the nine bind (`moe_gate_up_decode_gemv_bf16`,
    // `moe_down_decode_gemv_bf16`, `reorder_moe_aligned_output_bf16`,
    // `hash_route_lookup`), FOUR are `none:` arms — which is what they
    // already were in effect, every one of their rows unsourced and getting
    // no arm — and the ninth, `build_moe_ptrs_aligned_bf16`, is a DRIVER OP
    // with no `Entry` at all. That last one is the correction worth keeping:
    // this paragraph read "two bind, the other seven are `none:` arms" for
    // two edits after both halves of it stopped being true, and neither half
    // was checkable from here. `x::moe`'s `bind!` block is the only place the
    // split is countable, and it is 13 binds and 4 refusals over 20
    // contracts with 3 driver ops taking no arm.
    // `fire/moe_dispatch.rs`, `fire/dsv4_routing.rs` and the six
    // `bind::service::moe_*` wrappers are deleted with them.
    // The two native KV appenders STOOD HERE. Naming them was what dropped
    // the two `pie_k_attn_write_kv_*` shim entries, and dropping those is
    // what let `attn/kv_paged.cu` lose its two envelope call sites -- which
    // is in turn what let `layout/envelope.cu` be deleted whole. The chain
    // ran the other way from how it read: the envelope port was the thing
    // these two were blocking on, and closing them is what unblocked it.
    //
    // **THEY LEFT THIS LIST WITH THE CROSSING.** `x::attn`'s
    // `WRITE_KV_TO_PAGES` and `WRITE_KV_EXPLICIT` are contracts, they state
    // no `operands`, and `emit_c_shim` emits nothing for a symbol that
    // resolves through a `Contract::sig` -- the THIRD shim-dropping
    // mechanism, and the one that is not gated on `table::sig` resolving.
    // Their host programs are `x::attn::kv_paged`'s. `layout::embed`'s
    // retraction below is the precedent for the sentence and for keeping it.
    // `attn/mla_paged.cu`, both launchers. Both rows are UNSOURCED, so neither
    // ever produced a dispatch arm; `RUST_SERVED` here takes the shim entry
    // and with it the last two `<<<>>>` in the file, which is then deleted.
    // §60.7 is the precedent (`ssm::build_nemotron_moe_ptrs_*` are the live
    // instances of it).
    // `attn::mla_prepare_bf16` and `attn::write_mla_to_pages` STOOD HERE and
    // left with the crossing -- two `x::attn` contracts with `none:` arms on
    // `Cx::mla_layer`. Neither dropped anything: both rows were unsourced on
    // every operand, `abi.rs:810` skipped each whole, and what the entries
    // bought was the shim entry and with it the `.cu`. The `.cuh` is
    // `x::attn::mla_paged`'s unit now, so the shim entry buys nothing.
    // `layout/embed.cu`, the last launcher under `csrc/src/layout/`, STOOD
    // HERE. It is gone with the family: `layout` crossed into fn-world and
    // its seven contracts state no `operands`, which is the THIRD of the
    // three shim-dropping mechanisms and is the true sentence about a
    // family whose launcher is a `fn` in `src/x/`. The note beside it read
    // *"The row moved out of `table::driver_internal` to make this line
    // legal"* — a move this list no longer needs, because the mechanism
    // that drops the entry is not gated on `table::sig` resolving.
    // `attn::split_qkv_bf16_devwin` STOOD HERE and left with the crossing --
    // `x::attn`'s `SPLIT_QKV_DEVWIN`, a contract with a real bind. Its `Walk`
    // went too, and see the retraction there for why: one of the two reasons
    // it could not bind was false, and `bind/mod.rs:3973` is where.
    // `attn/page_compact.cu`. Classified `Execution::Composed` since the
    // split -- the composition's "one that fires end to end" -- and the
    // classification was never the thing that was missing. `RUST_SERVED` is
    // what drops the shim entry, and `driver-cuda/src/fire/page_compact.rs`
    // is the host that fires the two steps in order.
    // `attn::compact_page_csr` STOOD HERE and left with the crossing --
    // `x::attn`'s `COMPACT_PAGE_CSR`, a contract with a `none:` arm. ITS
    // `COMPOSED` ENTRY STAYS: that entry is a finding about the BODY, and
    // the body is still two launches on one stream with the second reading
    // the first's buffer. What went is the row and this line.
    // `attn/attention_naive.cu`'s MTP state pair, both unsourced. The device
    // rows were renamed `_dev` in the same change so `unit_of` is `None` and
    // the `Walk` classification is legal.
    // `attn/dsv4_compress.cu`'s four survivors.
    //
    // **This comment said "All four rows unsourced" and one of them was
    // fully sourced.** `attn::combine_attn_outputs_bf16`'s row stated ten
    // `Source`s -- `In(0..3)`, `Out(0..1)`, `Rows`, `Param(0..1)` and
    // `Ctx("stream")` -- so it DID generate a dispatch arm and this entry was
    // moving a live dispatch rather than dropping an entry nothing reached.
    // The claim was written once for four symbols and re-derived for none,
    // which is this file's own hazard: a sentence that counts a set is a
    // count written twice. It is corrected rather than deleted because the
    // three that remain are unsourced and the sentence is true of them.
    //
    // `attn::combine_attn_outputs_bf16` LEFT THIS LIST when it crossed into
    // fn-world -- `x::attn`'s `COMBINE_ATTN_OUTPUTS`, a contract with a bind
    // -- and its `Walk` was retracted in the same change. Three remain.
    // `attn::dsv4_boundary_meta_decode`, `attn::dsv4_boundary_meta_paged` and
    // `attn::attention_compressed_paged_bf16` STOOD HERE and left with the
    // crossing -- three `x::attn` contracts with `none:` arms. NONE of the
    // three ever dropped anything: all three rows had `Source::Unbound` on
    // every operand, so `abi.rs:810` skipped each one whole, no arm was
    // generated, and the shim entries these named were never reached.
    //
    // "Three remain" above is now ZERO of the four, and the sentence it
    // corrects has finished being true of anything. `attn::dsv4_compress`'s
    // whole `.cu` is fn-world.
    // `attn::qkv_decode_qk_norm_rope_write_kv_bf16` STOOD HERE and left with
    // the crossing -- `x::attn`'s `QKV_DECODE_FUSED`, a contract and a real
    // bind. **This line DID drop something**: the row was fully sourced, so
    // `abi.rs:810` kept it, `emit_rust_dispatch` wrote an arm, and
    // `bind::service::attn_qkv_decode_qk_norm_rope_write_kv_bf16` had a live
    // caller until this commit. It is the last `attn` entry on this list and
    // the last row in `ROW_TABLES`; what remains here is `comm`'s two.
    // `attn::write_kv_explicit_bf16_devwin` STOOD HERE and left with the
    // crossing -- `x::attn`'s `WRITE_KV_EXPLICIT_DEVWIN`, a contract with a
    // `none:` arm. This line never dropped anything: the row stated
    // `Source::Unbound` on all nine operands, `abi.rs:810` skipped it whole,
    // no dispatch arm was ever generated and the shim entry it was meant to
    // take never had a caller. Its `Walk`'s "the row is fully sourced"
    // claim, corrected above, is why anyone looked.
    // `attn/dsa_indexer.cu`'s three STOOD HERE and left with the crossing.
    // `dsa_index_topk_mask`'s row was fully sourced, so this line really did
    // move a live dispatch; the other two were unsourced, so `abi.rs:810`
    // skipped them whole and neither line ever dropped anything. All three
    // are `x::attn` contracts now, and `driver-cuda/src/fire/dsa_indexer.rs`
    // went with them -- `bind::service`'s three entry points were that
    // module's only consumer.
    // `attn::mtp_shift_hidden_bf16` and `attn::mtp_update_pending_hidden_bf16`
    // STOOD HERE and left with the crossing -- two `x::attn` contracts with
    // `none:` arms on `slot_ids`. Both rows were unsourced on every operand,
    // so `abi.rs:810` skipped each whole and no arm was ever generated.
    // `moe/flashinfer_moe.cu`, the file's single dispatch — and the reversal
    // of the sentence this list's own header used to carry about it.
    //
    // It is `Service::Cutlass` in [`SERVED`] and stays so: the kernels really
    // are CUTLASS templates in `${flashinfer_SOURCE_DIR}/csrc/fused_moe/
    // cutlass_backend/cutlass_fused_moe_kernels.cuh`, 4,991 lines and 19
    // `__global__` of CPM-fetched upstream device text — the LAST thing in
    // this family still compiled ahead of time, which is a state and not a
    // settlement. What was measured wrongly is the OTHER file. With
    // comments stripped `src/moe/flashinfer_moe.cu` is 817 lines holding 0
    // `__global__`, 0 `<<<>>>`, 0 `__device__`, 2 `std::mutex` and 1
    // `std::unordered_map` — a workspace query, an arch probe, a
    // coordinate-descent autotuner, a per-device tactic memo, an on-disk
    // tactic cache and a dispatch. `std::mutex` is not a thing NVRTC could
    // compile; it is the evidence the file was never NVRTC's to compile.
    //
    // So this is the same shape as `attn::mla_prepare_bf16` above and not a
    // new one: all EIGHTEEN operands of the `table::moe` row are unsourced,
    // so the row never produced a dispatch arm and does not gain one — the
    // mechanism that carries it is the shim entry, which `abi::emit_c_shim`
    // now drops. `driver-cuda/src/bind/service.rs` spells it, the host
    // program is `driver-cuda/src/fire/flashinfer_moe.rs`, and what is left
    // in the archive is a five-function `extern "C"` seam over
    // `CutlassMoeFCRunner` that decides nothing.
    //
    // It is also the only `returns = "bool"` row on this list, and the `bool`
    // is a REFUSAL: `false` means the window declined and the caller must run
    // its own expert path. `fire::flashinfer_moe` answers `Fused::{Ran,
    // Declined}` so that cannot be spelled like a failure, and `bind::service`
    // flattens it at the ABI because `KernelSig` is unchanged.
    //
    // **IT LEFT THIS LIST TOO, and it is the only one that left without its
    // executor.** `x::moe` declares `MOE_FUSED_CUTLASS` — `contract!` yes,
    // `Entry` no, the third registration shape — because the body needs a
    // workspace allocation and a device API `Cx` must not have. A contract
    // states no operands, so `table::sig` answers an unstated row and
    // `every_taken_over_row_is_stated` would refuse it; nothing changes for
    // the shim, which never emitted an entry for an unstated row anyway.
    // `bind::service::moe_flashinfer_cutlass_moe_bf16` and
    // `fire/flashinfer_moe.rs` both STAY: they are the driver op.
    //
    // THE STALE CLAUSE IS TAKEN. The `SERVED` entry for this symbol ended
    // *"It is on `RUST_SERVED`"*, which the edit above made false; it was
    // left one round because the entry sits next to work belonging to the
    // agent holding `csrc/src/moe/flashinfer_moe.cu`. It now says what is
    // true -- the driver-op shape, and the `bind::service` seam -- and it
    // touches no part of the entry that agent's work is about: the
    // `Service::Cutlass` classification, the 0-`__global__` finding and the
    // CPM path are all unchanged. A sentence a reader will take as a fact is
    // worth one careful edit; leaving it was the worse half of the trade.

    // ── the P2P all-reduce plane, and the first `SERVED ∩ RUST_SERVED` ────
    //
    // `comm/custom_all_reduce.cu` was the fifth file this migration found
    // wearing a `.cu` extension for linkage rather than content: 664 lines,
    // **zero `__global__` and zero `<<<>>>`**. Its `__global__` was deleted
    // with `all_reduce_residual_rmsnorm_bf16_exact`, whose caller set was
    // empty rather than merely small. What was left was a host program —
    // peer-access enablement, an IPC handle exchange, a `RankData` slab, a
    // fusion plane of four allocations, a Lamport initialisation and an
    // NCCL crossover query — and CPU-side code that is not Rust is the thing
    // this migration is now against on its own terms.
    //
    // It is `driver-cuda/src/fire/all_reduce.rs`. `custom_all_reduce.cu`,
    // `custom_all_reduce.hpp` and `custom_all_reduce_stub.cpp` are deleted.
    //
    // **These are the first two rows on this list that are also on
    // [`SERVED`]**, and nothing about that is an accident to be tidied away.
    // The two lists answer different questions — `SERVED` says *the body is
    // one library call and extracting it as a kernel extracts nothing*,
    // `RUST_SERVED` says *the driver issues that call from Rust* — and every
    // other `SERVED` row's library is cuBLAS, which Rust reaches through
    // `cudarc`. This one's library is a header-only P2P kernel in a
    // CPM-fetched flashinfer tree that `csrc/vendor/` does not carry, so the
    // Rust issues everything EXCEPT the launch and answers
    // `Decline::NoDeviceText` for the launch, carrying the resolved template
    // point's name expression. `every_taken_over_row_was_classified_first`
    // asks only that a taken-over row appear in one of the three findings;
    // being in `SERVED` satisfies it, and the pairing is the honest answer.
    //
    // Both rows are fully unsourced, so neither produced a dispatch arm
    // before and neither gains one; the mechanism that carries each is the
    // shim entry, which `abi::emit_c_shim` drops for anything on this list.
    // `bind::service` spells both, as `comm_all_reduce_bf16` and
    // `comm_all_reduce_residual_rmsnorm_bf16`.
    //
    // Both were `throw` in C++ and the shim's `catch` aborted; both are a
    // `#[must_use] AllReduce::{Launched, Declined}` in Rust, flattened to a
    // panic naming the `Decline` at the ABI. "It declined" cannot be spelled
    // like "it ran", which is the whole of `fire::gemv`'s lesson applied
    // where the C++ had only an exception.
    "comm::all_reduce_bf16",
    "comm::all_reduce_residual_rmsnorm_bf16",

    // ── AND THE FA2 SIX ARE RETRACTED, HAVING SERVED THEIR ONE PURPOSE ───
    //
    // The six entries stood here and the block above them is kept below,
    // because what it recorded is still true and is the reason this list
    // existed at all: `driver-cuda/csrc/attn/attention_flashinfer.cu` (1,258
    // lines) and `attn/plan_lifecycle.cpp` (105) are deleted, and
    // `driver-cuda/build.rs`'s last `.cuda(true)` with them. **nvcc is no
    // longer invoked anywhere in this tree.** The params-layout hazard that
    // block names -- a `paged_kv_t` filled against the shim's
    // `__fast_div_modulo` and read against CCCL's, `num_heads` at +24 versus
    // +20 with `sizeof` reconverging at 96, a silent wrong answer and not a
    // crash -- is closed by there being one filler and one reader.
    //
    // WHAT RETRACTS THEM IS THE NEXT STEP OF THE SAME MOVE. This list takes
    // *the shim entry and nothing else*, so a symbol on it still had a ROW,
    // still routed `Route::Rows`, and still reached Rust through a generated
    // dispatch arm resolving 13-19 operands from `table::attn`. Those arms
    // are hand-written now, the six rows are deleted, and the symbols are
    // `Service::DriverOp` in [`SERVED`]. There is no shim entry left to take
    // over.
    //
    // AND THE TEST IS WHY THIS HAD TO BE ONE INDEX WITH THE MOVE.
    // `every_rust_served_symbol_is_spelled_here` reads
    // `include_str!("service.rs")` and asks that every symbol on this list
    // appear in that file. `bind::service` became `#[cfg(feature =
    // "bridge")]` at `f38d199c2` and these six must OUTLIVE `bridge`, so
    // their entry points moved to `fire::flashinfer_fa2_dispatch` -- the
    // same shape `dequant_kv_cache_layer_to_bf16_active` took. The instant
    // the first one left `service.rs` the test would have gone red with the
    // names still listed here. Removing them is not bookkeeping after the
    // fact; it is the other half of the move.

    // ── the FA2 host program, and with it the LAST TWO C++ FILES in
    //    `driver-cuda/csrc` ─────────────────────────────────────────────────
    //
    // `driver-cuda/csrc/attn/attention_flashinfer.cu` (1,258 lines) and
    // `attn/plan_lifecycle.cpp` (105) are deleted, and `driver-cuda/
    // build.rs`'s last `.cuda(true)` with them. **nvcc is no longer invoked
    // anywhere in this tree.**
    //
    // The measurement that made it safe to delete rather than merely
    // unnecessary: `attention_flashinfer.cu` held `__global__` 0, `__device__`
    // 0 and exactly one `<<<>>>`, `device::attn_score_fold_heads`, which is
    // ours and already rowed. Its four calls to
    // `dequant_kv_cache_layer_to_bf16_active` at `:648`, `:675`, `:1098` and
    // `:1244` were the entire remaining `<<<>>>` census reachable from this
    // crate's consumers, and `driver-cuda/src/fire/kv_paged.rs` has all four
    // arms. `plan_lifecycle.cpp` was seven `pie_x_*` forwarders whose own
    // header says the reason they exist is *"a `unique_ptr` with a custom
    // deleter"*; the caches are Rust structs and the deleter is `Drop`.
    //
    // THE HAZARD THIS CLOSES, which is the reason all six went in one pass:
    // `csrc/shim/cuda/cmath:245-280` records that this tree's
    // `__fast_div_modulo` is `{u32 @0, u64 @8}` align 8 while CCCL's is
    // `{u32,u32,u32,i32}` align 4, so `paged_kv_t::num_heads` sits at +24
    // under the shim and +20 under CCCL with `sizeof` reconverging at 96.
    // `kernels_cuda_new::fa2::params` is pinned to the shim's layout, which
    // is right for every JIT fire; `attention_flashinfer.cu` compiled against
    // real CCCL through `DEP_PIE_KERNELS_CUDA_CCCL` and filled the other one.
    // Both were correct and **a params block filled on one side and read on
    // the other is a silent wrong answer, not a crash**. With the `cc::Build`
    // gone there is one filler and one reader.
    //
    // FOUR OF THESE SIX HAVE LIVE GENERATED ARMS. `table::attn` sources every
    // operand of the two decode rows and the two plain prefill rows, and
    // `crates/model/src` states all four — `attention_flashinfer_decode` x5,
    // `_decode_capture` x2, `attention_flashinfer_prefill` x6 and
    // `_prefill_lse` x1 in the shared llama-like forward, the capture pair
    // behind `GuardPred::WantsAttnScore`. `_prefill_custom` is stated three
    // times and `_prefill_capture` once. So this block MOVES LIVE DISPATCHES
    // from the C shim to `bind::service` rather than dropping entries nothing
    // reached, and `bind::service`'s six `attn_*` spellings are what this
    // list's test looks for.
    //
    // The host program is `driver-cuda/src/fire/flashinfer_fa2.rs` (plan,
    // H2D, fire) and `flashinfer_fa2_dispatch.rs` (params, arms). The 460
    // kernel rows are `families::fa2` and carry NO operands, because each
    // `__global__` takes one `__grid_constant__` params struct — the third
    // shim mechanism, and the reason these six entries are about the
    // LAUNCHERS and not about the kernels.
    //
    // WHAT DID NOT COME ACROSS: the SM90 prefill route. `:783-798` forwarded
    // to `dispatch_attention_flashinfer_prefill_sm90_bf16`, which lives in
    // `kernels-cuda`'s hopper unit and not in the deleted file.
    // `PrefillPlanCache::sm90_plan` is still planned and the dispatch refuses
    // with `Decline::Sm90Unported`, under §44.7's rule that every sm_90 claim
    // in this migration is argued from the call graph and none from a run.

    // ── `attn/kv_paged.cu` AND THE CENSUS — THE LAST ENTRY THIS LIST EVER
    //    NEEDED, AND IT IS RETRACTED ────────────────────────────────────────
    //
    // `attn::dequant_kv_cache_layer_to_bf16_active` STOOD HERE, and the
    // history is worth keeping because the ORDERING was the finding:
    //
    // 284 lines holding ONE function and FOUR `<<<>>>`, and those four were
    // the entire remaining census in the whole tree. `attn/kv_paged.cu`,
    // `attn/kv_paged.hpp` and their `CMakeLists.txt` entry went with that
    // line, and **the `<<<>>>` census is ZERO, from 401.**
    //
    // IT COULD NOT GO ONE ENTRY EARLIER. Its consumer set was four C++ call
    // sites inside the file the block above deleted —
    // `attention_flashinfer.cu:648`, `:675`, `:1098`, `:1244` — so the two
    // deletions had to be ordered and could not be split across passes: the
    // FA2 host program had to be Rust before this switch had an empty C++
    // consumer set, and *do not transcribe a live switch into a second
    // language* is why the Rust arms were written in the same change that
    // emptied it rather than earlier.
    //
    // IT COULD NOT JUST BE DELETED EITHER, which is why the line existed at
    // all. The symbol was in NEITHER `device::JIT_DISPATCHED` nor this list,
    // and its `table::attn` row stated four fully-sourced operands — so all
    // three shim-dropping mechanisms were open, `abi::emit_c_shim` still
    // wrote `pie_k_attn_dequant_kv_cache_layer_to_bf16_active`, and deleting
    // the `.cu` would have left that entry with a declaration, a forwarder
    // and no definition. A link error, not a clean removal.
    //
    // **AND NOW THE ROW IS GONE**, which closes that mechanism the other
    // way: `x::attn`'s `DEQUANT_KV_ACTIVE` is a contract, it states no
    // operands, `table::sig` no longer resolves the symbol, and there is
    // nothing for `emit_c_shim` to write. `model-compiler/src/dsl.rs:7750`
    // still states the symbol and `lower.rs:1100` still names it — that
    // channel is what the `bind!` arm serves now, so the trace that reached
    // the shim reaches a `fn`.
];

/// The service that executes a symbol, if a service does.
///
/// A linear scan on a path nothing hot takes -- the same trade
/// [`crate::device::specialisation`] documents. The length is deliberately
/// not quoted, for [`walk`]'s reason: this doc said *"fourteen entries"* for
/// as long as it took the list to reach twenty, and `SERVED.len()` cannot go
/// stale the way a word can.
#[must_use]
pub fn service(symbol: &str) -> Option<Service> {
    SERVED.iter().find(|(s, _, _)| *s == symbol).map(|(_, service, _)| *service)
}

/// The composition that executes a symbol, if one does.
#[must_use]
pub fn composition(symbol: &str) -> Option<&'static Composition> {
    COMPOSED.iter().find(|c| c.symbol == symbol)
}

/// The host walk that executes a symbol, if one does.
///
/// The same linear scan as [`service`] and [`composition`], for the same
/// reason, over a list that is short and is checked at every call. The count
/// is deliberately not quoted here: [`WALKED`] has grown five times since
/// this sentence was first written and a number in a doc comment goes stale
/// silently, where `WALKED.len()` cannot.
#[must_use]
pub fn walk(symbol: &str) -> Option<&'static Walk> {
    WALKED.iter().find(|w| w.symbol == symbol)
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
    /// The honest split that [`COMPOSED`] had when it had two entries, and
    /// the reason it is a function rather than a field: it is derived from
    /// the unit tables, so it becomes true the day the last step migrates and
    /// nobody has to remember to flip it. `attn::compact_page_csr` is
    /// fireable; `gemm::act_x_wt_bias_bf16` was not, because
    /// `gemm::act_x_wt_bf16` was an unmigrated kernel behind
    /// `Wall::HostChoice` — and that entry is deleted, because §5 step 5 made
    /// it a `fn` (`x::gemm::act_x_wt_bias_bf16`) whose two calls need no
    /// step list. The split this documents is therefore currently
    /// one-sided, and the function stays because the next composition to
    /// arrive will need it.
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

impl Walk {
    /// Everything about this walk a machine can check, checked.
    ///
    /// [`Composition::agrees`] is the model, and the two lists of checks are
    /// almost disjoint -- which is the point of the arm. A composition can be
    /// checked against the ROWS it names: every step's operand count, every
    /// step's types, no self-fire, no cycle. A walk names no rows, so none of
    /// that exists to check, and what is left is exactly what a walk claims:
    /// it is a real symbol, its shape reads a real operand, it can say no, and
    /// somebody can find the program.
    ///
    /// The gap is deliberate and it is not a weakness of the arm. It is the
    /// arm's whole content: **a program whose shape comes from the input
    /// cannot be validated against a table, which is why it needed a variant
    /// instead of a longer [`Step`] enum.** `tests/layers.rs` is where the
    /// remaining fact -- that a walk is not ALSO hosted, composed or served --
    /// is asserted, because that one is about the join and not about the row.
    ///
    /// # Errors
    ///
    /// A sentence naming the walk and what does not line up.
    pub fn agrees(&self) -> Result<(), String> {
        // ONE TABLE NOW, AND THE SECOND ONE IS NOT MISSING — IT IS GONE.
        //
        // This read BOTH tables, and said why:
        //
        // > BOTH tables, unlike [`Composition::agrees`], which reads only
        // > [`table::sig`]. Three of the five walks are `driver_internal`
        // > rows -- launchers the driver fires with no DSL statement -- and
        // > `table::sig` is deliberately blind to those: `table/mod.rs`'s own
        // > `the_driver_internal_rows_are_not_statable` asserts it. A walk is
        // > a DRIVER-side program by construction, so "no model text can
        // > state it" is not evidence against one; it is half the population.
        //
        // **Three of five** is the measurement that sentence was written
        // from, and it is kept because it is what justified the second
        // disjunct: at that size, a one-table check would have refused 60% of
        // [`WALKED`]. Both halves of it have since stopped holding.
        //
        // §5 step 5 deleted `table::driver_internal`: its six launchers are
        // `fn`s in `x::driver_internal` with **no `contract!`**, hence no
        // `Entry`, hence nothing in `x::SIGS` and nothing in any list for the
        // second disjunct to scan. `table/mod.rs` retired
        // `the_driver_internal_rows_are_not_statable` on the same argument —
        // a test that a symbol is absent from every list, when the symbol is
        // no longer data in any list, is a test of the empty set.
        //
        // And [`WALKED`] grew from five to 45, and is ZERO now — §5 emptied
        // it a family at a time and `attn`'s last row took the last entry.
        // ("24 now" stood here and was wrong when it was written: the list
        // held ONE. Re-derived two ways.) The sentence that stood here said
        // *"every one of the 45 resolves
        // through [`table::sig`] today"*, and it is STILL TRUE, which is the
        // interesting part. `moe`'s nine outlived `table/moe.rs` by two
        // edits and went on resolving, because `table::sig` reads `KERNELS`
        // and `KERNELS` is `ROW_TABLES ++ x::SIGS`: `stated` quietly stopped
        // meaning "is a row" and started meaning "is a row OR a contract",
        // and no sentence anywhere had to change for that to happen. That is
        // the right answer for every other caller of `table::sig`, and it is
        // why this check can never date a walk — see the block where those
        // nine were deleted, which had to be a judgement instead of a red
        // test. The observation the sentence was making holds for the 24
        // that are left: the second disjunct decided nothing even before its
        // table was deleted — `||` short-circuited on `stated` for the whole
        // population. Repointing it at anything in `x::` would
        // have been worse than deleting it, because there is nothing in `x::`
        // for it to point AT (that is the fourth arrangement, and
        // `x/driver_internal.rs`'s header is the table of the four), so any
        // repoint would have been a second reading of `table::sig` wearing a
        // different name — a disjunct that cannot change the answer.
        //
        // What is LOST is real and is named rather than papered over: a walk
        // whose symbol is a driver-fired launcher with no row can no longer
        // be admitted here, so landing one means re-opening this check with
        // an oracle that exists. The check is now STRICTER than it was, which
        // is the safe direction for a `WALKED` entry to fail in.
        let stated = table::sig(self.symbol).is_some();
        if !stated {
            return Err(format!(
                "`{}` is walked and is not a row of `table::KERNELS`",
                self.symbol
            ));
        }
        if self.because.len() < 40 {
            return Err(format!("`{}` is walked on a citation too short to check", self.symbol));
        }
        if self.control.reads().is_empty() {
            return Err(format!(
                "`{}` walks a {} that reads nothing -- a shape whose discriminant is not named \
                 is not a shape, it is the word \"complicated\"",
                self.symbol,
                self.control.label()
            ));
        }
        if self.refuses.is_empty() {
            return Err(format!(
                "`{}` is walked and states no refusal. Every walk refuses something: the shape \
                 that makes it a walk -- an arm chosen from a run-time value -- is the same shape \
                 that has values with no arm. A walk that cannot say no will either fall through \
                 to a kernel the caller did not ask for, or let a `throw` cross the C ABI. \
                 Neither is a diagnosable failure, and the second is not even a failure: it is \
                 `SIGABRT` with no message",
                self.symbol
            ));
        }
        for refusal in self.refuses {
            if refusal.len() < 20 {
                return Err(format!(
                    "`{}` states a refusal too short to be the launcher's own words: `{refusal}`",
                    self.symbol
                ));
            }
        }
        Ok(())
    }
}

/// No symbol composes itself, through any number of steps.
///
/// # Why this exists at all///
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

/// How a symbol executes -- the join over all four tables.
///
/// `None` means the symbol is a kernel this tree has not migrated: a row in
/// [`crate::table`] with no unit hosting it, no composition stating it, no
/// walk running it and no service serving it. That is a FIFTH answer and it is
/// deliberately not an [`Execution`] variant, because "nobody executes this
/// yet" is a fact about the migration and not about execution.
/// `examples/migration_status.rs` is what counts them.
///
/// The four tables may not overlap, and the ordering here is not what
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
    if let Some(walk) = walk(symbol) {
        return Some(Execution::Walk(walk));
    }
    let (_, unit) = unit::unit_of(symbol)?;
    unit.rows.iter().find(|row| row.sig.symbol == symbol).map(Execution::Jit)
}

#[cfg(test)]
mod tests {
    use super::{
        COMPOSED, Control, Execution, Kind, SERVED, Service, Step, WALKED, Walk, execution,
        service, walk,
    };

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

    /// The kind of an execution is the kind of its arm -- four arms, three
    /// kinds.
    ///
    /// [`Kind`] is derived from the TOP-LEVEL execution and from nothing
    /// deeper, which is what keeps the partition total now that a step can
    /// name a service: `gemm::act_x_wt_bias_bf16`'s first step reaches cuBLAS
    /// and the symbol is still an [`Kind::Op`].
    ///
    /// [`Execution::Walk`] shares [`Kind::Op`] with [`Execution::Composed`],
    /// deliberately -- see the module header. This test is where that sharing
    /// is written down as an assertion rather than a paragraph, so that
    /// anyone tempted to give `Walk` a kind of its own trips over the
    /// migration denominator on the way.
    #[test]
    fn the_four_arms_are_the_three_kinds() {
        const STEPS: &[Step] = &[Step::Fire { symbol: "norm::residual_add_bf16", take: &[] }];
        // `WALKED` IS EMPTY, so the walk is built here rather than indexed.
        // `WALKED[0]` stood here until `attn::qkv_decode_qk_norm_rope_write_\
        // kv_bf16` — the last row in `ROW_TABLES` — crossed, and an index
        // into an empty list panics. The assertion is about
        // `Execution::Walk`'s `kind()` and never about which walk it is
        // handed, which is the same reason `STEPS` above is a literal: a
        // test of a variant should not depend on the variant having members.
        const WALK: Walk = Walk {
            symbol: "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
            control: Control::Switch { on: "head_dim -- the last walk there was" },
            refuses: &[],
            because: "the shape of a walk, kept after the last one was retracted",
        };
        assert_eq!(Execution::Composed(STEPS).kind(), Kind::Op);
        assert_eq!(Execution::Service(Service::Cublas).kind(), Kind::Service);
        assert_eq!(Execution::Walk(&WALK).kind(), Kind::Op);

        let jit = crate::unit::rows().next().expect("some unit hosts a row");
        assert_eq!(Execution::Jit(jit).kind(), Kind::Kernel);

        // The real one: an op whose steps reach a service is still an op.
        // It USED to be `composition("gemm::act_x_wt_bias_bf16")`, the
        // demonstration case, which §5 step 5 deleted by making it a `fn`
        // with two calls. `COMPOSED`'s first surviving member serves the
        // same purpose — the assertion is about `Execution::Composed`'s
        // `kind()`, not about which composition it is handed.
        let composed = COMPOSED.first().expect("COMPOSED still has a member");
        assert_eq!(Execution::Composed(composed.steps).kind(), Kind::Op);
    }

    /// Every variant of [`Control`] has at least one member.
    ///
    /// [`no_service_name_is_unevidenced`]'s rule, applied to the shape enum,
    /// and the reason neither a `Predicate` nor a `Loop` variant is there.
    /// `Predicate`'s only member, `attention_mtp_paged_history_bf16`'s `if
    /// (max_global_tokens + history_steps > 8192)`, was deleted by
    /// `new-horizon.md` §44.6. `Loop`'s three members were the three
    /// multimodal towers, and every one of them is Rust now. Both shapes are
    /// real, both sets of evidence are gone, and this test is what keeps the
    /// enum honest about the difference.
    ///
    /// [`no_service_name_is_unevidenced`]: tests::no_service_name_is_unevidenced
    #[test]
    fn no_control_shape_is_unevidenced() {
        // AND NOW THE LIST IS EMPTY, WHICH IS NOT THE SAME AS A VIOLATION.
        //
        // `attn::qkv_decode_qk_norm_rope_write_kv_bf16` was the last walk and
        // the last row in `ROW_TABLES`. This rule says *a shape named on
        // nobody's evidence should not be in the enum*; with no walks at all,
        // its SUBJECT is gone rather than its invariant broken, and both
        // `Switch` and `Supplies` are now in the same position `Predicate`
        // and `Loop` were when they were deleted.
        //
        // The difference is that `Walk`, `Control` and [`WALKED`] are the ROW
        // WORLD's vocabulary — a walk classifies a row the row world still
        // interprets — so they retire with the row interpreters in north star
        // step 6, as one deletion, and not here as three. This branch is the
        // tripwire that keeps that decision visible: if a walk is ever
        // re-landed, the loop below applies again unchanged, and if step 6
        // lands first, this test goes with the enum.
        if WALKED.is_empty() {
            return;
        }
        for shape in Control::ALL {
            assert!(
                WALKED.iter().any(|w| w.control.label() == *shape),
                "`Control::{shape}` has no member -- a control shape named on nobody's evidence. \
                 If you are re-landing the two-operand predicate, land its walk in the same edit."
            );
        }
    }

    /// Every walk agrees with the table, and no symbol is walked twice.
    ///
    /// [`super::Walk::agrees`] is the per-entry check -- a real row, a
    /// citation long enough to follow, a named discriminant, and at least one
    /// refusal in the launcher's own words. This is the loop over it, plus
    /// the one fact `agrees` cannot see from inside a single entry.
    #[test]
    fn every_walk_agrees_and_is_stated_once() {
        let mut seen: Vec<&str> = Vec::new();
        for entry in WALKED {
            entry.agrees().unwrap_or_else(|why| panic!("{why}"));
            assert!(!seen.contains(&entry.symbol), "`{}` is walked twice", entry.symbol);
            seen.push(entry.symbol);
        }
    }

    /// A walk is not also hosted, composed or served.
    ///
    /// The four tables partition; they do not layer. A symbol both walked and
    /// JIT-hosted would mean the driver runs a host program AND
    /// [`crate::runtime::fire`] runs a `__global__` under the same name, and
    /// [`execution`]'s ordering would pick one silently. `tests/layers.rs`
    /// asserts the same disjointness from the other side, over every row.
    #[test]
    fn a_walk_is_only_a_walk() {
        for entry in WALKED {
            assert!(service(entry.symbol).is_none(), "`{}` is walked and served", entry.symbol);
            assert!(
                super::composition(entry.symbol).is_none(),
                "`{}` is walked and composed -- if its steps really are a fixed sequence it is \
                 not a walk, and if they are not it is not a composition",
                entry.symbol
            );
            assert!(
                crate::unit::unit_of(entry.symbol).is_none(),
                "`{}` is walked and JIT-hosted",
                entry.symbol
            );
            assert!(matches!(execution(entry.symbol), Some(Execution::Walk(_))));
            assert_eq!(walk(entry.symbol).map(|w| w.symbol), Some(entry.symbol));
        }
        assert!(walk("norm::residual_add_bf16").is_none());
    }

    /// A ROW MAY NOT BE TAKEN OVER BEFORE IT IS CLASSIFIED.
    ///
    /// [`RUST_SERVED`] is what drops a shim entry, and dropping one is what
    /// lets the C++ body be deleted. Doing that to a row whose execution
    /// nobody has written down is deleting a body on a hunch — precisely the
    /// move `SERVED`'s own header refuses ("a finding is not a plan", and
    /// neither is a plan a finding). Every symbol on the list must appear in
    /// [`SERVED`], [`COMPOSED`] or [`WALKED`] first, with its citation.
    ///
    /// [`WALKED`] gained its first members on this list with §45's
    /// continuation. The four walks that predate it still keep their shim
    /// entries, because their host programs are still C++ — they moved
    /// archive, not language, and that is the difference between a walk and
    /// a takeover. The three `gemm::act_x_wt_*` quantized rows are the other
    /// case: their walk moved LANGUAGE, into
    /// `driver-cuda/src/bind/quant_gemm.rs`, so there is no C++ program left
    /// for a shim entry to reach and the entry has to go.
    #[test]
    fn every_taken_over_row_was_classified_first() {
        for symbol in super::RUST_SERVED {
            assert!(
                super::service(symbol).is_some()
                    || super::composition(symbol).is_some()
                    || super::walk(symbol).is_some(),
                "`RUST_SERVED` names `{symbol}`, which is in neither `SERVED`, `COMPOSED` nor \
                 `WALKED`. The list drops the row's shim entry, so the C++ body goes -- state \
                 what the body IS, with the file and line, before taking it over."
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
