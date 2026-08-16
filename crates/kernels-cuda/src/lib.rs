//! CUDA kernels as a JIT: the rows, the host programs, and the compiler that
//! turns text into a cubin at run time.
//!
//! # The three layers `Cargo.toml`'s `default = []` gates
//!
//! They are not directories -- they are what a build with no features selected
//! still has, and they nest:
//!
//! **Layer 1 -- the rows.** [`sigs`] is every symbol a lowered model text may
//! state, as [`KernelSig`]s, every one DERIVED from the `fn`s of layer 2.
//! `model-compiler` reads one on every trace and links no CUDA to do it, which
//! is why `cudarc` is optional.
//!
//! **Layer 2 -- the host programs.** One module per `kernels/` directory, each
//! a set of ordinary Rust `fn`s registered with [`routine!`]. Feature-free too:
//! a body compiles, and its row is derived from it, on a machine that has never
//! seen a GPU. What a featureless build cannot do is *run* one --
//! [`jit::Ctx::launch`] answers `Refusal::Device` as a value.
//!
//! **Layer 3 -- the JIT.** NVRTC, the per-instantiation module cache, the
//! launch, the device scratch. Gated on the private `_cuda` rather than on
//! `cuda-12`/`cuda-13` so a build selecting neither produces one actionable
//! error instead of burying it under every use of `cudarc`.
//!
//! # A kernel has exactly two truths
//!
//! The device text (a `.cuh` under `kernels/`) and the host program (a Rust
//! `fn`). Everything else is derived: the argument list off the `fn`'s
//! parameters, the trace namespace off the module path ([`jit::Family`]), the
//! C++ spelling off the [`jit::Abi`] impl. What remains hand-stated is `whole`,
//! `in_place` and `depth_prefix_plan` -- three facts no signature carries --
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
/// [`kernels::Refusal`] is re-exported here because it is what every host
/// program in this crate returns, and `driver-cuda`'s arms match on it in about
/// forty places.
pub use kernels::{Cap, KernelSig, LaunchRule, Lit, Refusal, Source, Ty};
// The four position wrappers, re-exported for callers that hold no `Fire`.
// `model-loader`'s CUDA executor calls these launchers directly and should not
// have to depend on `kernels` to spell their signatures.
pub use kernels::{In, Out};

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
/// TWO families and not one, because a trace names them with two namespaces and
/// the choice between them is a deployment's. `comm` has the whole host program
/// and stops at a header this tree does not vendor; `dist` has nothing at all.
pub mod comm;

/// The CuTile roots, which are not a family either: no routine in this crate
/// compiles a symbol out of one, and the module doc measures why.
pub mod tile;

// ── the families ────────────────────────────────────────────────────────
//
// One module per `kernels/` directory, at the crate root, because **the module
// path IS the trace namespace**:
//
//     kernels_cuda::rope::rope_bf16   <->   "rope::rope_bf16"
//
// `fa2` and `xqa` are under [`attn`] because that is where both their symbols
// and their device text already are: every FA2 and XQA row a trace names is
// `attn::`-namespaced, and the text is `kernels/{flashinfer,xqa}/`. The
// scheduler (`attn::plan`) went with them.
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
/// The multimodal towers' host walks. `_cuda`-gated for [`gemm::dense`]'s
/// reason: a walk is a device program end to end — it allocates, uploads,
/// launches and reads back — so there is no featureless half to keep.
#[cfg(feature = "_cuda")]
pub mod tower;
pub mod vision;

/// Why a compile or a launch did not happen, at the top level because it is
/// what every launch path returns.
#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub use jit::Error;

pub use jit::ArgValue;

/// The families that have crossed to the routine shape.
///
/// The WHOLE declared surface: a symbol not reachable from this list is one
/// [`call`] answers [`kernels::Refusal::Undeclared`] for, and there is no
/// second table to look in.
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
    // XQA's one symbol is `attn`'s too. `Family::routine` strips the namespace
    // and then matches on the routine's own name, so two families may share one
    // namespace as long as no name repeats -- which `no_symbol_is_declared_twice`
    // below is what checks.
    &attn::xqa::FAMILY,
    // As XQA's: the six FlashInfer FA2 dispatches are `attn::` symbols whose
    // bodies belong beside the lattice and the params filling they fire.
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
/// Peel refusal and the row-window split), `depth_prefix_plan` (the union-tail
/// plan swap) and `in_place` (buffer aliasing). Everything else `KernelSig` can
/// carry had no live reader when the consumer audit went looking, so nothing
/// fills it. `args` comes off the `fn`'s parameter list, so it costs nothing to
/// carry and cannot drift.
///
/// [`kernels::driver_bound!`] emptied the hand-written half: it derives the row
/// from the `fn` exactly as `routine!` does and leaves `args` empty, with a body
/// that refuses a dispatch by STRING, because the driver calls those symbols by
/// path where the compiler checks the call. Which symbols a statement can bind
/// is now a per-symbol fact stated where the symbol is, rather than a property
/// of which file a line was written in.
///
/// Built once and leaked: the rows outlive the process and a `&'static` is what
/// the compiler's lookup takes.
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
                    sides: r.sides,
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

/// A `KernelSig` that claims nothing, to update from.
const SIG_BASE: KernelSig = KernelSig {
    name: "",
    symbol: "",
    whole: false,
    in_place: &[],
    depth_prefix_plan: false,
    args: &[],
    sides: &[],
    axes: &[],
};

/// The routine one trace symbol names, or `None` if no crossed family declares
/// it.
///
/// Feature-free, so a reader with no GPU can ask what a symbol takes, whether it
/// consumes its whole operand, and which of its operands alias.
///
/// It answers for every row of [`sigs`], so `None` means UNDECLARED rather than
/// "declared elsewhere" -- and a `driver_bound!` symbol answers `Some` with a
/// body that refuses a string dispatch by name.
#[must_use]
pub fn routine(symbol: &str) -> Option<&'static jit::Routine> {
    FAMILIES.iter().find_map(|family| family.routine(symbol))
}

/// Fire the routine `symbol` names.
///
/// The one dynamic entry point. Feature-free rather than gated on `_cuda`: a
/// gate here would remove a symbol without adding a check, since a build with no
/// CUDA runtime already refuses at [`jit::Ctx::launch`] as a value.
///
/// # Errors
///
/// [`Refusal::Undeclared`] if no family declares `symbol`; otherwise whatever
/// the routine refuses -- including [`Refusal::Arity`] or [`Refusal::Kind`] if
/// `args` does not fit the signature.
///
/// # Safety
///
/// `stream` must name a live CUDA stream for the duration of the launch, and
/// every [`ArgValue::Ptr`] in `args` must address device memory that is live and
/// large enough for the argument the routine binds it to. The launch is
/// asynchronous, so "for the duration" ends when the stream is synchronised.
/// **Nothing here checks either fact and nothing can**: this is the same
/// obligation every `<<<>>>` carried.
pub unsafe fn call(
    symbol: &str,
    args: &[ArgValue],
    stream: *mut core::ffi::c_void,
) -> Result<(), kernels::Refusal> {
    // SAFETY: the caller's obligation on `stream`, forwarded. A null handle is
    // the context `Ctx::on` builds, and `Ctx::cublas()` turns it into a refusal.
    unsafe { call_with_cublas(symbol, args, stream, core::ptr::null_mut()) }
}

/// [`call`], with the engine's cuBLAS handle attached to the context.
///
/// A handle is not a fact about a statement: it never touches
/// [`kernels::Facts`], which stays `Copy` and statement-scoped, and rides with
/// the CONTEXT instead ([`jit::Ctx::with_cublas`]) -- null IS the answer when
/// no handle exists, so `cublas()` is a pass-through and not an `Option`.
///
/// # Errors
///
/// [`Refusal::Undeclared`] if no family declares `symbol`; otherwise whatever
/// the routine refuses.
///
/// # Safety
///
/// [`call`]'s obligations on `stream`, plus [`jit::Ctx::with_cublas`]': if
/// `cublas` is non-null it must be a live `cublasHandle_t` for the duration of
/// the launch, and it must be the handle paired with this stream. Null is always
/// sound and means the bodies that need one refuse.
pub unsafe fn call_with_cublas(
    symbol: &str,
    args: &[ArgValue],
    stream: *mut core::ffi::c_void,
    cublas: *mut core::ffi::c_void,
) -> Result<(), kernels::Refusal> {
    let Some(routine) = routine(symbol) else {
        return Err(kernels::Refusal::Undeclared);
    };
    // SAFETY: the caller's obligation on both pointers, forwarded.
    let ctx = unsafe { jit::Ctx::on(stream).with_cublas(cublas) };
    (routine.body)(&ctx, args)
}

#[cfg(test)]
mod surface {
    use super::{FAMILIES, call, routine, sigs};
    use kernels::{Refusal, Ty};

    /// A row carries the signature it was derived from.
    ///
    /// `sigs()` builds `KernelSig`s field by field, so a column that silently
    /// fell back to the base's `&[]` would look like a table not yet filled in.
    #[test]
    fn a_row_carries_its_arguments() {
        let symbol = "rope::rope_bf16";
        let row = kernels::sig_in(sigs(), symbol).expect("rope has crossed");
        let routine = routine(symbol).expect("and resolves to its `fn`");
        assert_eq!(row.args, routine.args);
        assert!(
            !row.args.is_empty(),
            "`rope_bf16` takes arguments and the row says so"
        );
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
            assert!(
                routine(symbol).is_some(),
                "{symbol} is registered and does not resolve"
            );
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
                assert!(
                    !seen.contains(&symbol),
                    "{symbol} is declared by two families"
                );
                seen.push(symbol);
            }
        }
        assert!(
            seen.len() > 10,
            "the walk found {} symbols, so it stopped early",
            seen.len()
        );
    }

}

/// Slots a signature WROTE DOWN, and slots a counter reached, over every
/// declared routine.
///
/// The `#[cfg(test)]` census in `driver-cuda`'s `bind/table.rs` asks the same
/// question of `sigs()` and has been stale since Kilimanjaro III, because a
/// `#[test]` does not run under a `cargo check`-only regime. This walks
/// `FAMILIES` instead — every hop is `&'static` — so the compiler recomputes
/// it on each build and it cannot go stale.
///
/// `driver_bound!` rows carry `derived: &[]` and are invisible here, which is
/// not a hole: `operands` refuses such a row before it reads an index.
#[must_use]
pub const fn slot_census() -> (usize, usize) {
    let (mut stated, mut counted) = (0usize, 0usize);
    let mut f = 0;
    while f < FAMILIES.len() {
        let routines = FAMILIES[f].routines;
        let mut r = 0;
        while r < routines.len() {
            let d = routines[r].derived;
            let mut i = 0;
            while i < d.len() {
                if matches!(
                    d[i].source,
                    Some(kernels::Source::Slot(kernels::Kind::In | kernels::Kind::Out, _))
                ) {
                    if d[i].stated {
                        stated += 1;
                    } else {
                        counted += 1;
                    }
                }
                i += 1;
            }
            r += 1;
        }
        f += 1;
    }
    (stated, counted)
}

// COUNTED reached zero and `alias()` was deleted on the strength of it; a
// non-zero reading means a column can be mis-bound with nothing left to
// correct it. STATED rides beside it so a family DELETING launchers instead
// of converting them cannot look like progress.
/// Unstated rows whose source is neither a slot nor `None`.
///
/// Zero, and it is a ONE-WAY DOOR: `kernels-macros` sends `Shape::Env` to
/// `None`, so the parameter-name table that used to read `"eps"` as
/// [`kernels::keys::RmsEps`] cannot produce a hit. A non-zero reading means
/// somebody reintroduced a coupling between a parameter's SPELLING and a
/// launch's behaviour, with nothing written down at either end.
#[must_use]
pub const fn name_table_hits() -> usize {
    let mut hits = 0usize;
    let mut f = 0;
    while f < FAMILIES.len() {
        let routines = FAMILIES[f].routines;
        let mut r = 0;
        while r < routines.len() {
            let d = routines[r].derived;
            let mut i = 0;
            while i < d.len() {
                if !d[i].stated
                    && d[i].source.is_some()
                    && !matches!(
                        d[i].source,
                        Some(kernels::Source::Slot(kernels::Kind::In | kernels::Kind::Out, _))
                    )
                {
                    hits += 1;
                }
                i += 1;
            }
            r += 1;
        }
        f += 1;
    }
    hits
}

/// Rows whose `Named` source no launcher on this backend answers.
///
/// The CUDA binder answers every `keys::` fact a CUDA routine names, with
/// five exceptions the shader planes construct and this one does not:
/// `AttentionMask`, `AttentionMaskEnabled`, `AttentionMaskStride`,
/// `KvHeadStride` and `KvSeqStride`. Zero rows here NAME one, which is what
/// makes the exception list a statement about the other backends rather than
/// a gap in this one.
///
/// The hazard this catches: declaring a key and forgetting to answer it in
/// `operand()` is a SILENT demotion to `keys.rs` §2 — the row compiles, and
/// refuses at bind on a machine nobody is watching.
#[must_use]
pub const fn unanswerable_named_rows() -> usize {
    const UNANSWERED: [&str; 5] = [
        <kernels::keys::AttentionMask as kernels::keys::Fact>::KEY,
        <kernels::keys::AttentionMaskEnabled as kernels::keys::Fact>::KEY,
        <kernels::keys::AttentionMaskStride as kernels::keys::Fact>::KEY,
        <kernels::keys::KvHeadStride as kernels::keys::Fact>::KEY,
        <kernels::keys::KvSeqStride as kernels::keys::Fact>::KEY,
    ];
    let mut hits = 0usize;
    let mut f = 0;
    while f < FAMILIES.len() {
        let routines = FAMILIES[f].routines;
        let mut r = 0;
        while r < routines.len() {
            let d = routines[r].derived;
            let mut i = 0;
            while i < d.len() {
                if let Some(kernels::Source::Named(k)) = d[i].source {
                    let mut u = 0;
                    while u < UNANSWERED.len() {
                        if kernels::source_is_named(&d[i].source, UNANSWERED[u]) {
                            hits += 1;
                        }
                        u += 1;
                    }
                    let _ = k;
                }
                i += 1;
            }
            r += 1;
        }
        f += 1;
    }
    hits
}

const _: () = {
    assert!(
        unanswerable_named_rows() == 0,
        "a CUDA routine names a fact only the shader planes answer"
    );
    let (stated, counted) = slot_census();
    assert!(name_table_hits() == 0, "a parameter reached a fact through its NAME");
    assert!(counted == 0, "a signature still reaches a slot by counting");
    assert!(stated == 523, "the stated-slot count moved");
};
