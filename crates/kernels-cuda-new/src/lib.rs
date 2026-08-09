#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda-new's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda-new: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

/// The words a row is written in.
pub use kernels::{Cap, KernelSig, LaunchRule, Lit, Operand, Prepare, Source, Ty};

pub mod fa2;
/// The per-symbol JIT, and the machinery a routine is written against.
pub mod jit;
/// The rows nothing can derive, because their host programs are `driver-cuda`'s.
pub mod not_yet_crossed;
/// The attention scheduler: `flashinfer/attention/scheduler.cuh` as host Rust.
pub mod plan;
pub mod source;
/// **kernel-x** — the floor a kernel stands on when it is written as a
pub mod x;

#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub mod runtime;

/// The launch path, at the top level because it is the one thing every
#[cfg(feature = "_cuda")]
pub use runtime::{Error, Stream};

pub use jit::ArgValue;

/// The families that have crossed to the routine shape.
///
/// Grows one entry per family as the kernel-x port lands
/// (`.wiki/kernel-x/refactor-plan.md` §10 step 4). A family that is not here
/// yet is still fired through `x::route`/`x::Entry`, and [`call`] answers
/// [`Refusal::Undeclared`] for its symbols.
pub static FAMILIES: &[&jit::Family] = &[
    &x::rope::FAMILY,
    &x::sample::FAMILY,
    &x::mlp::FAMILY,
    &x::layout::FAMILY,
    &x::quant::FAMILY,
    &x::moe::FAMILY,
    &x::ssm::FAMILY,
    &x::norm::FAMILY,
    &x::attn::FAMILY,
    &x::gemm::FAMILY,
];

/// The rows `model-compiler` reads: the derived half, then the stated one.
///
/// Four columns have a live reader and both halves build those four: the
/// symbol it looks up by, `whole` (the Peel refusal and the row-window
/// split), `depth_prefix_plan` (the union-tail plan swap) and `in_place`
/// (buffer aliasing). Everything else `KernelSig` can carry had no live reader
/// when the consumer audit went looking, so nothing fills it.
///
/// `args` is the exception that is filled anyway, and only a derived row can:
/// it comes off the `fn`'s parameter list, so it costs nothing to carry and
/// cannot drift, and it is what a statability check will read. There is no
/// such check today — see the field's own doc for why the provenance it
/// carries is not yet a thing anything can act on.
///
/// [`not_yet_crossed::NOT_YET_CROSSED`] is the second half, and its header
/// says what it is and when it goes: symbols a live trace names whose host
/// programs are `driver-cuda`'s, so no signature here derives them. The two
/// halves may not both claim a symbol — a test below refuses that.
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
        rows.extend(not_yet_crossed::NOT_YET_CROSSED.iter().map(copy_sig));
        Vec::leak(rows)
    })
}

/// `KernelSig` is not `Copy` — deriving it on a public contract type is a
/// promise `kernels` should not make — so a stated row is carried across field
/// by field. Every field, so a row says what it was written to say.
const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name,
        symbol: k.symbol,
        file: k.file,
        launch: k.launch,
        whole: k.whole,
        needs: k.needs,
        lacks: k.lacks,
        sink: k.sink,
        in_place: k.in_place,
        depth_prefix_plan: k.depth_prefix_plan,
        args: k.args,
        operands: k.operands,
        axes: k.axes,
        grid_param: k.grid_param,
        head_param: k.head_param,
        heads_param: k.heads_param,
        rows_param: k.rows_param,
    }
}

/// A `KernelSig` that claims nothing, to update from.
const SIG_BASE: KernelSig = KernelSig {
    name: "",
    symbol: "",
    file: None,
    launch: LaunchRule::Unstated,
    whole: false,
    needs: Prepare::None,
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
/// it consumes its whole operand, and which of its operands alias. It answers
/// for the derived half of [`sigs`] only: a symbol in
/// [`not_yet_crossed::NOT_YET_CROSSED`] has a row and no routine, which is
/// what having no `fn` here means.
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
    use super::{FAMILIES, call, not_yet_crossed::NOT_YET_CROSSED, routine, sigs};
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

    /// A symbol is derived or stated, never both.
    ///
    /// This is the failure that lands when §6.3 crosses one of
    /// [`NOT_YET_CROSSED`]'s symbols and the row is left behind: the two
    /// halves of [`sigs`] would then hold one symbol under two contracts, and
    /// `sig_in` would answer with whichever it reached first. Crossing a
    /// symbol means deleting its row here, in the same change.
    #[test]
    fn no_symbol_is_both_derived_and_stated() {
        for row in NOT_YET_CROSSED {
            assert!(
                routine(row.symbol).is_none(),
                "{} has a routine now -- delete its `not_yet_crossed` row",
                row.symbol
            );
        }
        // And the stated half does not repeat itself either.
        for (i, row) in NOT_YET_CROSSED.iter().enumerate() {
            for other in &NOT_YET_CROSSED[i + 1..] {
                assert_ne!(row.symbol, other.symbol, "{} is stated twice", row.symbol);
            }
        }
    }
}
