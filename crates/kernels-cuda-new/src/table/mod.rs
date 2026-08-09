//! Layer 1: the rows, and nothing that needs a GPU to read one.
//!
//! **THERE ARE NO ROWS LEFT.** [`ROW_TABLES`] is `&[]` as of the `attn`
//! crossing, so [`KERNELS`] is the flattening of [`crate::x::SIGS`] and
//! nothing else, and this module is no longer a table. What it still is, and
//! why it survives its own contents:
//!
//! * **the flattener.** `x::SIGS` is `&[&[KernelSig]]` — a list per family,
//!   each derived by `Contract::sig`. Every consumer needs a flat
//!   `&[KernelSig]`: `model-compiler`'s `check_plan`, `kernels-cuda`'s
//!   `build.rs`, and six test files. [`concat_tables`] is the only const
//!   `[&[T]] -> [T]` in the tree.
//! * **the lookup.** [`sig`] is what answers *"does anything declare this
//!   symbol"*, which is the question a model load asks per statement.
//!
//! So `table/` does NOT die with the rows, and the claim [`ROW_TABLES`]
//! carried for six ports — *"when it is, `KERNELS` is `x::SIGS` and this file
//! is deletable"* — was one step too strong. Deleting this file means moving
//! `KERNELS`, `TABLES`, `sig` and `total` into `x/` and editing every consumer
//! that spells `table::`. That is a rename of live API with no behaviour
//! change: step 6's cost, paid for a name. The honest end state is that this
//! module is renamed, not removed — and until it is, the one thing worth
//! keeping straight is that **`table::` now means `x::SIGS`, flattened.**
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
//! **Both numbers are the census taken before §5 step 5 began**, when every
//! family was still a row module; they are kept as written because they are
//! what justified the split, and re-deriving them per port would make the
//! argument look like it depended on the current total. It does not: the
//! archive stays until the last of the 109 has a JIT twin, and a step-5 port
//! moves a row from this file to `x::SIGS` without touching that count.
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

// `adapter`, `layout` and `sample` ARE GONE — §5 step 5's "boring first"
// batch took all three into fn-world.
//
// `layout` is `x::layout`: seven contracts, five binds, thirteen host `fn`s
// and five `unit!`s over five roots. `sample` is `x::sample`: one contract,
// one `none:` arm and four host `fn`s over one root, one of which is this
// tree's first two-kernel body.
//
// `adapter` is the ODD ONE, in the other direction from `driver_internal`
// below: it is `x::adapter` and holds a `contract!` and NOTHING ELSE — no
// `unit!` (the LoRA seam is cuBLAS batched GEMMs, so there is no
// `__global__`) and no `bind!` (an `Entry` would make `x::route` answer
// `Bound` for a symbol `bind/mod.rs` must keep firing by hand). So it is in
// `x::SIGS` and not in `x::FAMILIES`, which is the placement rule read from
// the DECLARATION side: `model-compiler` must not be able to tell a cuBLAS
// symbol from a JIT'd one, so the contract must reach it, and nothing else
// about the symbol may.
pub mod attn;
// `attn::KERNELS` IS EMPTY AND IS IN NO LIST — §5 step 5 finished here.
//
// The module is declared and read by path from
// `driver-cuda/tests/launch_abi.rs` (`:540`, `:809`), and it holds the
// derivations of all forty-one crossings. It is NOT in [`ROW_TABLES`], which
// is `&[]`, so it contributes nothing to [`KERNELS`] and nothing to
// [`TABLES`]. That is deliberate and not an oversight: an empty list in no
// list is inert, while an empty list still IN `ROW_TABLES` would leave the
// list non-empty and the north star's step-5 gate unmet on a technicality.
// `driver_internal` and `gemm` ARE GONE — §5 step 5 took both into fn-world.
//
// `gemm` is `x::gemm`: twelve contracts, two binds, ten `none:` arms and a
// `unit!` for the two GEMV rows, with the dense autotuner and the cuBLASLt
// plan cache beside them as `x::gemm::dense`.
//
// `driver_internal` is `x::driver_internal` and is the ODD ONE: six plain
// `fn`s with **no `contract!` at all**, so it is in neither `x::FAMILIES` nor
// `x::SIGS`. That is the placement rule applied to a whole family — a
// `driver_internal` row was never in [`TABLES`], `table::sig` could not see
// it, `dsl::cuda` could not wrap it and `execution::RUST_SERVED` refused it,
// so it had no reading consumer and is therefore not data. See that module's
// header for the four-way table.
//
// `mlp` and `quant` ARE GONE — §5 step 5 took both into fn-world. Neither
// module here is replaced by a table: `x::mlp` states twelve contracts and
// `x::quant` eleven, `Contract::sig` derives their rows, and `x::SIGS` is
// concatenated below by [`concat_lists`]. **Both state no `operands`** — the
// third shim-dropping mechanism, and the reason `pie_k_mlp_*`/`pie_k_quant_*`
// were never generated in the first place: the operand list that used to be
// the binding instruction is now the host `fn`'s parameter list.
//
// `ssm` IS GONE for the same reason — §5 step 5 took its five roots into
// `x::ssm`, where twenty-seven contracts and twenty-seven host programs
// replace twenty-seven rows. Four of the twenty-seven are `none:` arms:
// their rows stated EVERY operand unsourced, so the sentence a load-time
// refusal prints is the prose those rows were already carrying.

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
/// `driver_internal` was deliberately absent, for the reason its own module
/// doc gave — its rows were launchers the driver fires with no DSL statement,
/// so they must not be `check_plan`-visible. **That module is gone**: §5 step
/// 5 took its six rows to `x::driver_internal` as plain `fn`s with no
/// `contract!`, which is the same claim made by construction instead of by
/// omission from a list. The shim generator no longer adds them back, because
/// a direct Rust call needs no entry point.
///
/// # A ported family is registered in ONE place, and it is not here
///
/// The row modules below are the ones still written as rows. Every family
/// that has crossed into fn-world arrives through [`crate::x::SIGS`], which
/// this list concatenates — so **porting a family means appending to
/// `x::SIGS` and deleting a module here, and never adding a line to two
/// lists.**
///
/// It was two lists for exactly one commit, and the hazard is worth naming
/// because the §5 step-5 sweep runs several agents in parallel: a family
/// whose contracts exist and whose `SIGS` is not in `TABLES` compiles, links,
/// and is refused by `check_plan` at model load with *"no kernel! signature
/// declares it"* — a message that sends the reader to the table for a symbol
/// whose declaration is fine. One list cannot fail that way.
///
/// `crate::x::rope::SIGS` is the same list from the other world: twelve
/// contracts, derived by `Contract::sig` rather than written. A row here
/// carried a launcher's binding instructions; a contract carries only what a
/// trace may say, so **the derived rows state no `operands`** — which is one
/// of the three mechanisms by which a symbol loses its ahead-of-time C shim
/// entry, and is the one every ported family uses. `check_plan` still
/// refuses a symbol nothing declares, which is the whole reason this list
/// keeps them.
pub static TABLES: &[&[KernelSig]] = &concat_lists();

/// The families still written as rows. **EMPTY.**
///
/// Shrank by one module per §5 step-5 port and is empty now the sweep is
/// done. [`KERNELS`] is [`crate::x::SIGS`] — which is the clearest available
/// statement of what step 5 finishing means.
///
/// `static` rather than `const`, and deliberately: the members were
/// `static`s, and [`TABLES`] has read `static`s from a `const` context here
/// since [`total`] was written. Matching that shape exactly meant this line
/// added no construct the tree had not already compiled.
///
/// **`moe::KERNELS` left on the commit that crossed `quant`'s four routed
/// decode GEMVs**, and the shape of that crossing is worth keeping: those
/// four rows lived in `table/moe.rs` because `moe` DISPATCHED them, while
/// their host programs live in `x/quant.rs` because `quant` OWNS them. So
/// `table/moe.rs` outlived `moe` as `quant`'s tenant, and the four lists that
/// walk everything had been carrying one family's rows under another
/// family's name for as long as both entries were there. Nothing was wrong
/// until one had to go.
///
/// **`attn::KERNELS` was the last**, and its forty-one rows were the whole
/// remaining structural distance in the CUDA lane. They are gone, so
/// [`KERNELS`] is `x::SIGS`, `driver-cuda`'s `bridge` feature is deletable,
/// and with it `kernels-cuda/native`, which is the only switch over every
/// nvcc and `.cpp` compile in the workspace. **`table::attn` itself is still
/// declared** — an empty list and eight hundred lines of tombstones, read by
/// path from `driver-cuda/tests/launch_abi.rs`; its module doc says why, and
/// removing it is one line here plus one `git rm`.
static ROW_TABLES: &[&[KernelSig]] = &[];

const N_LISTS: usize = ROW_TABLES.len() + crate::x::SIGS.len();

/// The row lists and the fn-world lists as one list of lists.
///
/// Const, because [`TABLES`] is `&'static` and read by
/// `kernels-cuda/build.rs` before anything runs.
const fn concat_lists() -> [&'static [KernelSig]; N_LISTS] {
    let mut out = [EMPTY_LIST; N_LISTS];
    let mut w = 0;
    let mut i = 0;
    while i < ROW_TABLES.len() {
        out[w] = ROW_TABLES[i];
        w += 1;
        i += 1;
    }
    let mut j = 0;
    while j < crate::x::SIGS.len() {
        out[w] = crate::x::SIGS[j];
        w += 1;
        j += 1;
    }
    out
}

const EMPTY_LIST: &[KernelSig] = &[];

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
    head_param: None, heads_param: None, rows_param: None, lowered_as: None,
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
        rows_param: k.rows_param,
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

    /// `driver_internal`'s rows were reachable and NOT in [`KERNELS`].
    ///
    /// THE TEST IS DELETED WITH ITS SUBJECT. It read
    /// `super::driver_internal::DRIVER_KERNELS` and asserted `sig(symbol)`
    /// answered `None` for every row — the rule that kept the two tables
    /// apart, *a driver-fired launcher has an entry point and no statement*.
    ///
    /// §5 step 5 made the rule unstatable by making it true by construction:
    /// there is no second table to compare against, because the six launchers
    /// are `fn`s in `x::driver_internal` with no `contract!`, hence no
    /// `Entry`, hence nothing in `x::SIGS` and nothing for `sig` to find. A
    /// test that a symbol is absent from every list, when the symbol no
    /// longer exists as data in any list, is a test of the empty set.
    ///
    /// What replaced it is not a test here but the four-way table in
    /// `x::driver_internal`'s header, which says which of the four
    /// arrangements each family took and why this one took the fourth.
    const _THE_DRIVER_INTERNAL_ROWS_ARE_NOT_STATABLE: () = ();
}
