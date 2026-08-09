//! Layer 1: the rows, and nothing that needs a GPU to read one.
//!
//! One row per launcher symbol the ahead-of-time archive defines, written in
//! `kernels`' vocabulary — [`KernelSig`], `whole`, `needs`, `lacks`, `sink` —
//! which is also where the reasons for each of those words are.
//!
//! # These rows were `kernels-cuda`'s until the seam closed
//!
//! They were authored beside the `.cu` files they describe, so that adding a
//! kernel was one source file and one table row in the same directory and the
//! same diff hunk. While the JIT crate was new, this module was a `pub use`
//! of them and said why at length: *while the ahead-of-time path and the JIT
//! path must BOTH run, a symbol has to have exactly one contract. Two copies
//! of a table are two contracts, and the way they fail is not that one is
//! wrong — it is that each is right for whichever half of the tree its own
//! tests exercise, so nothing goes red until a model text picks the other
//! one.*
//!
//! That argument is why the rows moved here rather than being copied, and it
//! is the only thing about them that changed. The direction of the dependency
//! did: `kernels-cuda` now takes the table FROM this crate and re-exports it
//! under the paths it used to own, so `kernels_cuda::attn::KERNELS` and
//! `kernels_cuda::KERNELS` still resolve and not one consumer was edited.
//!
//! The reason the rows are here and not there is the smaller half of the
//! crate's job: `kernels-cuda` builds an archive and a shim, which is a
//! CONSUMER of the table, and a build that needs CMake, nvcc and a Linux
//! target must not be the thing a compiler dev loop depends on to read a
//! symbol's operand list. 109 of the 198 rows below still have no JIT twin,
//! so the archive is not going away — but it reads the table rather than
//! owning it.
//!
//! # Two tables, one crate, and they are not the same contract
//!
//! A row here describes a `pie_k_*` entry point: a C++ host function that
//! holds a `<<<>>>`, takes a stream as an argument, and was compiled by nvcc
//! months ago. A row in [`crate::device`] describes a `__global__` template
//! and the type to instantiate it at, and states its geometry as a
//! `LaunchRule` — the thing the launcher used to hold. A symbol with rows in
//! both has one contract per PATH, and [`crate::device::JIT_DISPATCHED`] is
//! what says which path a fire takes.
//!
//! # What "no features" buys
//!
//! Nothing below needs `cudarc`, a toolkit or a driver. That is what lets
//! `model-compiler` depend on this crate unconditionally: it reads a row on
//! every trace, and a compiler dev loop must not pay a C++ build to look up a
//! symbol's operand list. The table is kept honest from the other end —
//! `model-compiler`'s `kernels::check_plan` refuses any `OpKind::Launch`
//! symbol no row declares, so a kernel cannot be stated by a model text
//! without its contract.

use kernels::{KernelSig, Prepare};

pub mod adapter;
pub mod attn;
pub mod driver_internal;
pub mod gemm;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod sample;
pub mod ssm;

/// Every kernel a lowered declaration may state.
///
/// The concatenation of the per-family tables, in the order [`TABLES`] lists
/// them. Order is not semantic — [`sig`] scans linearly and callers look rows
/// up by symbol — but it is stable, so a diff that adds a kernel touches one
/// module and one line.
pub static KERNELS: &[KernelSig] = &concat_tables();

/// The per-family tables, in the concatenation's order.
///
/// Public, and that is what stopped `kernels-cuda/build.rs` from having to
/// keep its own copy of this list. The shim generator needs every table the
/// archive defines an entry point for, and a hand-written second list is the
/// exact shape that goes stale silently: a family added here and forgotten
/// there emits no `extern "C"` and fails at link time in whichever binary
/// happened to state one of its symbols first.
///
/// [`driver_internal`] is deliberately absent, for the reason its own module
/// doc gives — its rows are launchers the driver fires with no DSL statement,
/// so they must not be `check_plan`-visible. The shim generator adds them
/// back explicitly, because an entry point is what they need and a statement
/// is not.
///
/// # `rope` is not a module here, and that is the migration
///
/// `crate::x::rope::SIGS` is the same list from the other world: twelve
/// contracts, derived by `Contract::sig` rather than written. A row here
/// carried a launcher's binding instructions; a contract carries only what a
/// trace may say, so **the derived rows state no `operands`** — which is one
/// of the three mechanisms by which a symbol loses its ahead-of-time C shim
/// entry, and is the one every ported family uses. `check_plan` still
/// refuses a symbol nothing declares, which is the whole reason this list
/// keeps them.
pub static TABLES: &[&[KernelSig]] = &[
    attn::KERNELS, crate::x::rope::SIGS, norm::KERNELS, mlp::KERNELS, gemm::KERNELS,
    moe::KERNELS, ssm::KERNELS, quant::KERNELS, layout::KERNELS,
    sample::KERNELS, adapter::KERNELS,
];

/// The row a symbol names, or nothing.
///
/// A linear scan, deliberately. The table is a few hundred rows and a lookup
/// happens once per statement at COMPILE time — `model-compiler`'s
/// `check_plan` — rather than per launch; the launch path resolves through
/// [`crate::unit::unit_of`], which scans a much shorter list. A map here would
/// buy nothing and would need a `OnceLock` in a module whose whole claim is
/// that it holds no state.
#[must_use]
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    KERNELS.iter().find(|row| row.symbol == symbol)
}

/// `[&[T]] -> [T]` at compile time, because [`KERNELS`] must stay a
/// `&'static [KernelSig]` for every consumer that already reads it, and
/// neither `concat` nor iterator chaining is const.
const fn concat_tables() -> [KernelSig; TOTAL] {
    // `KernelSig` is not `Copy` — deriving it on a public contract type is a
    // promise this crate should not make — so the array is filled by index.
    let mut out = [EMPTY; TOTAL];
    let mut w = 0;
    let mut t = 0;
    while t < TABLES.len() {
        let table = TABLES[t];
        let mut i = 0;
        while i < table.len() {
            out[w] = copy_sig(&table[i]);
            w += 1;
            i += 1;
        }
        t += 1;
    }
    out
}

const TOTAL: usize = total();

const fn total() -> usize {
    let mut n = 0;
    let mut t = 0;
    while t < TABLES.len() {
        n += TABLES[t].len();
        t += 1;
    }
    n
}

const EMPTY: KernelSig = KernelSig {
    name: "", symbol: "", file: None, launch: kernels::LaunchRule::Unstated,
    whole: false, needs: Prepare::None,
    lacks: &[], sink: None, in_place: &[], depth_prefix_plan: false, publishes_aux: &[],
    operands: &[],
    returns: "", axes: &[], grid_param: None,
    head_param: None, heads_param: None, lowered_as: None,
};

const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name, symbol: k.symbol, file: k.file, launch: k.launch,
        whole: k.whole, needs: k.needs,
        lacks: k.lacks, sink: k.sink, in_place: k.in_place,
        depth_prefix_plan: k.depth_prefix_plan, publishes_aux: k.publishes_aux,
        operands: k.operands, returns: k.returns, axes: k.axes,
        grid_param: k.grid_param,
        head_param: k.head_param, heads_param: k.heads_param,
        lowered_as: k.lowered_as,
    }
}

#[cfg(test)]
mod tests {
    use super::{KERNELS, TABLES, sig};

    /// A symbol names at most one row. Two rows sharing one symbol is the
    /// defect this whole module is arranged to prevent, so it is checked
    /// rather than assumed.
    #[test]
    fn a_symbol_names_one_row() {
        let mut seen: Vec<&str> = Vec::with_capacity(KERNELS.len());
        for row in KERNELS {
            assert!(!seen.contains(&row.symbol), "{} is stated twice", row.symbol);
            seen.push(row.symbol);
        }
    }

    /// The lookup finds what the table holds, and refuses what it does not.
    #[test]
    fn the_lookup_is_the_table() {
        for row in KERNELS {
            assert_eq!(sig(row.symbol).map(|r| r.symbol), Some(row.symbol));
        }
        assert!(sig("norm::a_kernel_nobody_wrote").is_none());
    }

    /// The concatenation spans every family, so a table listed in [`TABLES`]
    /// and empty — or a family declared and never listed — is visible here
    /// rather than at a `check_plan` refusal.
    #[test]
    fn the_concatenation_is_the_tables() {
        let counted: usize = TABLES.iter().map(|t| t.len()).sum();
        assert_eq!(KERNELS.len(), counted);
        assert!(KERNELS.len() > 100, "{} rows is not the CUDA table", KERNELS.len());
        for table in TABLES {
            assert!(!table.is_empty(), "a family listed in TABLES declares no rows");
        }
    }

    /// `driver_internal`'s rows are reachable and are NOT in [`KERNELS`].
    ///
    /// The check the seam's closing made cheap: both tables are in one crate
    /// now, so the rule that kept them apart — a driver-fired launcher has an
    /// entry point and no statement — is testable in one place instead of
    /// being restated in two crates' prose.
    #[test]
    fn the_driver_internal_rows_are_not_statable() {
        for row in super::driver_internal::DRIVER_KERNELS {
            assert!(
                sig(row.symbol).is_none(),
                "{} is fired by the driver and also statable by a model text",
                row.symbol
            );
        }
    }
}
