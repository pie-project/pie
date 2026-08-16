//! CUDA kernels as a JIT: the rows, the host programs, and the compiler that
//! turns text into a cubin at run time.
//!
//! # The three layers `Cargo.toml`'s `default = []` gates
//!
//! `Cargo.toml:26` promises this section by name. The layers are not
//! directories and never were — they are what a build with no features
//! selected still has, and they nest:
//!
//! **Layer 1 — the rows.** [`sigs`] is every symbol a lowered model text may
//! state, as [`KernelSig`]s. `model-compiler` reads one on every trace and
//! links no CUDA to do it, which is the whole reason `cudarc` is optional:
//! a compiler dev loop must not pay a GPU dependency to look up whether a
//! symbol exists. **Every row is DERIVED** from the `fn`s of layer 2 — there
//! is no second, hand-written half any more, and [`sigs`] records what the
//! other half was and what took its place.
//!
//! **Layer 2 — the host programs.** One module per `kernels/` directory,
//! each a set of ordinary Rust `fn`s registered with [`routine!`]. A body
//! computes its own geometry, names the C++ instantiation it wants, and
//! launches through [`jit::Ctx`]. **This layer is feature-free too**, and that
//! is easy to miss: a body compiles, and its row is derived from it, on a
//! machine that has never seen a GPU. What a featureless build cannot do is
//! *run* one — [`jit::Ctx::launch`] answers `Refusal::Device` instead of
//! reaching a driver, as a value, saying so.
//!
//! **Layer 3 — the JIT.** NVRTC, the per-instantiation module cache, the
//! launch, the device scratch. This is the only `_cuda`-gated half, and it is
//! gated on the private `_cuda` rather than on `cuda-12`/`cuda-13` so that a
//! build selecting neither produces one actionable error instead of burying
//! it under every use of `cudarc`.
//!
//! The gate is therefore much smaller than the crate: layers 1 and 2 are
//! ~29,000 lines that build anywhere, and layer 3 is `jit/`'s `_cuda` half.
//!
//! # A kernel has exactly two truths
//!
//! The device text (a `.cuh` under `kernels/`) and the host program (a Rust
//! `fn`). Everything else is derived and nothing is written twice: the
//! argument list comes off the `fn`'s parameters, the trace namespace off the
//! module path ([`jit::Family`]), the C++ spelling off the [`jit::Abi`] impl
//! the type already has. What remains hand-stated is `whole`, `in_place` and
//! `depth_prefix_plan` — three facts no signature carries — and they are
//! written beside the `routine!` line that needs them.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

/// The words a row is written in, and the one a routine refuses with.
///
/// [`kernels::Refusal`] is re-exported here rather than only under
/// `kernels::` because it is what every host program in this crate returns:
/// `driver-cuda`'s arms match on it in about forty places and reached it
/// through `x::Refusal` while that module existed. One re-export keeps those
/// call sites naming the crate whose functions produce the value.
pub use kernels::{Cap, KernelSig, LaunchRule, Lit, Operand, Refusal, Source, Ty};

/// The per-symbol JIT, and the machinery a routine is written against.
pub mod jit;
/// The rows nothing can derive, because their host programs are `driver-cuda`'s.
pub mod source;

/// Host programs `driver-cuda` calls directly, which are not a family.
pub mod driver_internal;

/// The NCCL collectives: declared so a tensor-parallel model text resolves,
/// and refused because this build links no communicator.
pub mod dist;

/// The custom P2P all-reduce: the same algebra as [`dist`] over NVLink peer
/// mappings instead of a communicator.
///
/// TWO families and not one, because a trace names them with two namespaces
/// and the choice between them is a deployment's. What separates them here is
/// how far each one gets: `comm` has the whole host program — the cross
/// product, the parameter block, both predicates — and stops at a header this
/// tree does not vendor; `dist` has nothing at all.
pub mod comm;

/// The CuTile roots, which are not a family either: no routine in this crate
/// compiles a symbol out of one, and the module doc measures why.
pub mod tile;

// ── the families ────────────────────────────────────────────────────────
//
// One module per `kernels/` directory, at the crate root, because **the
// module path IS the trace namespace**:
//
//     kernels_cuda::rope::rope_bf16   <->   "rope::rope_bf16"
//
// They lived under `x` until the dissolution. `x` was a CONTRASTIVE name: it
// meant the world where a kernel is a plain `fn`, as against `table/`,
// `families/`, `unit.rs` and `execution.rs`. Those four are deleted, and a
// name that distinguishes A from B distinguishes nothing once B is gone. A
// `families/` wrapper would have been no better -- this crate IS its
// families, and a directory saying so is a second statement of one fact.
//
// `fa2` and `xqa` are under [`attn`] rather than beside it because that is
// where both their symbols and their device text already are: every FA2 and
// XQA row a trace names is `attn::`-namespaced, and the text is
// `kernels/{flashinfer,xqa}/`. Only the Rust used to disagree, and it
// disagreed by splitting one family across three levels. The scheduler
// (`attn::plan`) went with them: it is `flashinfer/attention/scheduler.cuh`
// as host Rust and nothing outside attention asks it anything.
pub mod attn;
pub mod cascade;
pub mod gemm;
pub mod graph;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;
pub mod vision;

/// Why a compile or a launch did not happen, at the top level because it is
/// what every launch path returns.
#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub use jit::Error;

pub use jit::ArgValue;

/// The families that have crossed to the routine shape.
///
/// Grew one entry per family as the kernel-x port landed
/// (`.wiki/kernel-x/refactor-plan.md` §10 step 4), and it is now the WHOLE
/// declared surface: a symbol not reachable from this list is one [`call`]
/// answers [`kernels::Refusal::Undeclared`] for, and there is no second table
/// to look in.
pub static FAMILIES: &[&jit::Family] = &[
    &rope::FAMILY,
    &sample::FAMILY,
    &mlp::FAMILY,
    &layout::FAMILY,
    &quant::FAMILY,
    &moe::FAMILY,
    &ssm::FAMILY,
    &norm::FAMILY,
    &attn::FAMILY,
    // XQA's one symbol is `attn`'s too. `Family::routine` strips the
    // namespace and then matches on the routine's own name, so two families
    // may share one namespace as long as no name repeats -- which
    // `no_symbol_is_declared_twice` below is what checks.
    //
    // Since the dissolution the shared namespace is no longer a coincidence
    // three lines have to keep true: `attn::xqa` and `attn::fa2` are MODULES
    // under `attn`, and a family's namespace is the first path segment after
    // the crate root, so all three answer "attn" by construction.
    &attn::xqa::FAMILY,
    // As XQA's, and for the same reason: the six FlashInfer FA2 dispatches
    // are `attn::` symbols whose bodies belong beside the lattice and the
    // params filling they fire.
    &attn::fa2::FAMILY,
    &gemm::FAMILY,
    // Three symbols and no implementation — see `dist`'s header for why that
    // is a declaration rather than an omission.
    &dist::FAMILY,
    // Two symbols with a host program and no device text, which is a third
    // state and not a shade of the second: every refusal `comm` returns names
    // the resolved template point and the header that would supply it.
    &comm::FAMILY,
];

/// The rows `model-compiler` reads, every one of them derived.
///
/// Four columns have a live reader: the symbol it looks up by, `whole` (the
/// Peel refusal and the row-window split), `depth_prefix_plan` (the
/// union-tail plan swap) and `in_place` (buffer aliasing). Everything else
/// `KernelSig` can carry had no live reader when the consumer audit went
/// looking, so nothing fills it.
///
/// `args` comes off the `fn`'s parameter list, so it costs nothing to carry
/// and cannot drift, and it is what a statability check will read. There is
/// no such check today — see the field's own doc for why the provenance it
/// carries is not yet a thing anything can act on.
///
/// # There was a second half, and what it cost is worth knowing
///
/// `not_yet_crossed.rs` held it: twenty-one `kernel!` rows, hand-stating the
/// four columns for symbols a live trace names. They were honest rows — the
/// file stated only the columns with a live reader, precisely so no consumer
/// could tell the two halves apart — and they existed for one reason. A
/// `routine!` derives its row from a `fn`, and it can only be written for a
/// body whose every argument a STATEMENT can supply, because [`call`]
/// recovers them from the `&[ArgValue]` a statement produced. A paged-KV
/// write takes the layer's page geometry; an all-reduce takes a
/// communicator; a quantised GEMM takes a weight representation. No trace
/// mentions any of those, so the extractor could not describe the `fn`, so
/// the row was typed out by hand instead.
///
/// The cost was not the typing. It was that eight of those twenty-one named
/// a `fn` sitting in this crate — the body was never what was missing — and
/// that a hand-typed column is a transcription, which is what
/// `tests/stated_columns.rs` exists to catch and did: four `whole` columns
/// had drifted before anyone compared.
///
/// [`kernels::driver_bound!`] is what emptied it. It derives the row from
/// the `fn` exactly as `routine!` does and leaves `args` empty — which the
/// field's own doc already calls UNSTATED, and which is true — with a body
/// that refuses a dispatch by STRING, because the driver calls those symbols
/// by path where the compiler checks the call. The two lists became one, and
/// which symbols a statement can bind became a per-symbol fact stated where
/// the symbol is, rather than a property of which file a line was written
/// in.
///
/// Built once and leaked: the rows outlive the process and a `&'static` is
/// what the compiler's lookup takes.
#[must_use]
pub fn sigs() -> &'static [KernelSig] {
    static ROWS: std::sync::OnceLock<&'static [KernelSig]> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| {
        let mut rows: Vec<KernelSig> = Vec::new();
        for family in FAMILIES {
            for r in family.routines {
                let symbol: &'static str = String::leak(family.symbol(r));
                rows.push(KernelSig {
                    name: symbol,
                    symbol,
                    args: r.args,
                    whole: r.whole,
                    depth_prefix_plan: r.depth_prefix_plan,
                    in_place: r.in_place,
                    ..SIG_BASE
                });
            }
        }
        Vec::leak(rows)
    })
}

// `copy_sig` STOOD HERE. `KernelSig` is not `Copy` -- deriving it on a public
// contract type is a promise `kernels` should not make -- so a stated row was
// carried across field by field, every field, so that a row said what it was
// written to say. Its one caller was the `extend` that appended the stated
// half, and there is no stated half.

/// A `KernelSig` that claims nothing, to update from.
const SIG_BASE: KernelSig = KernelSig {
    name: "",
    symbol: "",
    file: None,
    launch: LaunchRule::Unstated,
    whole: false,
    lacks: &[],
    sink: None,
    in_place: &[],
    depth_prefix_plan: false,
    args: &[],
    operands: &[],
    axes: &[],
    grid_param: None,
    head_param: None,
    heads_param: None,
    rows_param: None,
};

/// The routine one trace symbol names, or `None` if no crossed family
/// declares it.
///
/// Feature-free, so a reader with no GPU can ask what a symbol takes, whether
/// it consumes its whole operand, and which of its operands alias.
///
/// It answers for every row of [`sigs`], which it did not always: a symbol
/// stated in the hand-written half had a row and no routine. There is no such
/// half now, so `None` means UNDECLARED rather than "declared elsewhere" —
/// and a `driver_bound!` symbol answers `Some` with a body that refuses a
/// string dispatch by name, which is a different and more useful answer than
/// silence.
#[must_use]
pub fn routine(symbol: &str) -> Option<&'static jit::Routine> {
    FAMILIES.iter().find_map(|family| family.routine(symbol))
}

/// Fire the routine `symbol` names.
///
/// The one dynamic entry point. Feature-free rather than gated on `_cuda`:
/// everything below it is, and a gate here would remove a symbol without
/// adding a check — a build with no CUDA runtime refuses at
/// [`jit::Ctx::launch`], as a value, saying so.
///
/// # Errors
///
/// [`Refusal::Undeclared`] if no family declares `symbol`; otherwise whatever
/// the routine refuses — including [`Refusal::Arity`] or [`Refusal::Kind`] if
/// `args` does not fit the signature.
///
/// # Safety
///
/// `stream` must name a live CUDA stream for the duration of the launch, and
/// every [`ArgValue::Ptr`] in `args` must address device memory that is live
/// and large enough for the argument the routine binds it to. The launch is
/// asynchronous, so "for the duration" outlives this call and ends when the
/// stream is synchronised. **Nothing here checks either fact and nothing can**:
/// this is the same obligation every `<<<>>>` carried.
pub unsafe fn call(
    symbol: &str,
    args: &[ArgValue],
    stream: *mut core::ffi::c_void,
) -> Result<(), kernels::Refusal> {
    let Some(routine) = routine(symbol) else {
        return Err(kernels::Refusal::Undeclared);
    };
    // SAFETY: the caller's obligation on `stream`, forwarded.
    let ctx = unsafe { jit::Ctx::on(stream) };
    (routine.body)(&ctx, args)
}

#[cfg(test)]
mod surface {
    use super::{FAMILIES, call, routine, sigs};
    use kernels::{Refusal, Ty};

    /// A row carries the signature it was derived from.
    ///
    /// `sigs()` builds `KernelSig`s field by field, so a derived column
    /// reaches a reader only if this function names it — and a column that
    /// silently fell back to the base's `&[]` would look exactly like a table
    /// whose rows had not been filled in yet.
    #[test]
    fn a_row_carries_its_arguments() {
        let symbol = "rope::rope_bf16";
        let row = kernels::sig_in(sigs(), symbol).expect("rope has crossed");
        let routine = routine(symbol).expect("and resolves to its `fn`");
        assert_eq!(row.args, routine.args);
        assert!(!row.args.is_empty(), "`rope_bf16` takes arguments and the row says so");
    }

    /// A symbol resolves to the `fn` that runs it, through its family's
    /// namespace and nothing else.
    #[test]
    fn a_symbol_resolves_to_its_routine() {
        let rope = routine("rope::rope_bf16").expect("rope has crossed");
        assert_eq!(rope.name, "rope_bf16");
        assert_eq!(rope.args[0].0, Ty::Bf16sMut);
        // One per crossed family, so a family dropped out of `FAMILIES` is a
        // failure here rather than a symbol that silently stops resolving.
        for symbol in [
            "sample::argmax_bf16",
            "mlp::swiglu_bf16",
            "layout::gather_bf16_rows",
            "quant::cast_fp32_to_bf16",
            "moe::topk_sigmoid_bf16",
            "ssm::causal_conv1d_update_batched_bf16",
            "norm::rmsnorm_strided_bf16",
            "attn::compact_page_csr",
            "gemm::act_x_wt_bf16",
        ] {
            assert!(routine(symbol).is_some(), "{symbol} is registered and does not resolve");
        }
    }

    /// A symbol no crossed family declares is refused as a value, not a panic,
    /// and without touching a device.
    #[test]
    fn an_undeclared_symbol_is_refused() {
        assert!(routine("rope::a_kernel_nobody_wrote").is_none());
        // SAFETY: nothing launches -- the symbol is refused before any
        // argument is read and before the stream is used.
        let refused = unsafe { call("rope::a_kernel_nobody_wrote", &[], core::ptr::null_mut()) };
        assert_eq!(refused, Err(Refusal::Undeclared));
    }

    /// No two crossed families claim the same symbol, which is what makes
    /// [`routine`]'s first-match answer a function rather than a race.
    #[test]
    fn no_symbol_is_declared_twice() {
        let mut seen: Vec<String> = Vec::new();
        for family in FAMILIES {
            for r in family.routines {
                let symbol = family.symbol(r);
                assert!(!seen.contains(&symbol), "{symbol} is declared by two families");
                seen.push(symbol);
            }
        }
        assert!(seen.len() > 10, "the walk found {} symbols, so it stopped early", seen.len());
    }

    // `no_symbol_is_both_derived_and_stated` STOOD HERE, and it went with its
    // subject rather than with a decision.
    //
    // It walked `NOT_YET_CROSSED` and asserted `routine(row.symbol).is_none()`
    // for each, because a symbol in both halves of `sigs()` is one symbol
    // under two contracts and `sig_in` would answer with whichever it reached
    // first. The failure it was watching for was specific: §6.3 crosses a
    // symbol and the hand-written row is left behind.
    //
    // There is one half now, built by a loop over `FAMILIES`, so a symbol
    // cannot be in it twice under two contracts -- it can only be declared
    // twice, which is a different fault and one `no_symbol_is_declared_twice`
    // below already refuses. A test whose subject is gone is not a test that
    // passes; it is one that reports nothing, and this file has spent enough
    // of this refactor deleting those.
}
