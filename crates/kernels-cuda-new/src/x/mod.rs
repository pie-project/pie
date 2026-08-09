//! **kernel-x** — the floor a kernel stands on when it is written as a
//! program rather than as a row.
//!
//! `.wiki/kernel-x/northstar.md` is the design; this module is its §5 steps 1
//! and 2, and [`rope`] is its step 3.
//!
//! > A kernel has exactly two truths — the device text (`.cuh`) and the host
//! > program (a Rust `fn`). One small declaration serves the readers that
//! > cannot call. Everything else is derived, and nothing is written twice.
//!
//! # What is here
//!
//! | thing | §  | what it is |
//! |---|---|---|
//! | [`launch`] conveniences | 3.1 | `flat`/`per_row` over the existing [`crate::runtime::Launch`] |
//! | [`Abi`] | 3.2 | an OPEN set of impls, one per crossing type |
//! | [`Cx`] | 3.3 | the query-only fire context a bind body reads |
//! | [`Contract`] / [`Entry`] / [`Refusal`] | 3.4 | the declaration the readers that cannot call read |
//! | `unit!` / `contract!` / `bind!` | 1, 2.1 | the three declarations |
//!
//! # The placement rule, applied
//!
//! > Data only for what has a reading consumer. Everything that is only
//! > executed is code.
//!
//! A grid is executed, so it is an expression in a `fn`. An operand's C++
//! type has a reading consumer — the typecheck translation unit — so it is
//! data, reached through [`Abi::CPP`] rather than written out. A symbol's
//! trace-facing shape has a reading consumer — `model-compiler`, which is
//! GPU-free and must not be able to tell a cuBLAS symbol from a JIT'd one —
//! so it is data, and [`Contract`] is that data.
//!
//! # Where this diverges from §5, and why
//!
//! §5 was written before the fn-world existed and the first family was
//! expected to settle idioms it could not foresee. Three departures, each
//! also recorded in `northstar.md` §5.1:
//!
//! 1. **The host program lives in `src/x/`, not beside the `.cuh`.**
//!    §1 asks for `rope.rs` next to `rope.cuh` in `csrc/`. It cannot be:
//!    `crates/kernels-cuda-new/carried.rs`'s `walk()` carries EVERY file
//!    under `csrc/` into the binary as device text for the NVRTC virtual
//!    filesystem, filtered only by a five-name `NOT_SOURCE` list, and it
//!    sets `cargo:rerun-if-changed=csrc`. A `.rs` there would be handed to
//!    NVRTC as a header and would rebuild the crate on every Rust edit.
//!    The two truths stay adjacent by `include_str!`, which is how the unit
//!    already reaches its text.
//! 2. **[`Cx`] is a facade over a driver-implemented trait ([`Facts`]).**
//!    §3.3 names `DispatchCtx`, `AttnCtx`, `BoundLaunch` and `Frame` as
//!    `Cx`'s contents. Those live in `driver-cuda`, which DEPENDS on this
//!    crate; naming them here is a cycle. The trait keeps §3.3's safety
//!    argument exactly — the vocabulary is query-only, so there is no device
//!    API, no allocator and no stream to misbehave on — and moves the
//!    unsafe assembly to the one place that already holds it.
//! 3. **[`Abi`] carries `Abi::TY` as well as §3.2's `CPP`.**
//!    `kernels::Ty` is the runtime's marshalling tag: `Args::bind` checks it
//!    per operand and `ArgValue` is chosen from it. It survives until §5
//!    step 9 retires the dynamic path. Putting both spellings of a type on
//!    one impl is the only way they cannot drift, which is the same argument
//!    §6.1 makes for the typecheck TU.

pub mod abi;
pub mod contract;
pub mod cx;
pub mod launch;
#[macro_use]
pub mod macros;
pub mod rope;

#[cfg(feature = "_cuda")]
pub mod fire;

pub use abi::Abi;
pub use contract::{Contract, Entry, Fired, Refusal};
pub use cx::{Cx, Facts, KvLayer, Plan, Rows, Slab, Yarn};

/// Every family that has crossed into fn-world.
///
/// One entry per family, appended as §5 step 5 walks the census. `rope` is
/// the pilot and for now the whole list.
///
/// The linear scan is load-time work — §5 step 4 interns these into
/// `lowered.kernels` once per model — and twelve rows is not a data
/// structure.
///
/// **Gated on `_cuda`, where [`SIGS`] is not.** An [`Entry`] holds a bind,
/// and a bind launches; a [`Contract`] holds what a trace may say, and
/// `model-compiler` reads it with no GPU anywhere. That the two lists are
/// gated differently IS the "must not be able to tell cuBLAS from a JIT'd
/// kernel" rule, expressed where the compiler can enforce it.
#[cfg(feature = "_cuda")]
pub static FAMILIES: &[&[Entry]] = &[rope::ENTRIES];

/// The [`Entry`] for one symbol, or `None` if no family declares it.
///
/// The single lookup every consumer uses: the load-time intern §5 step 4
/// builds, the bridge probe `driver-cuda`'s `dispatch()` makes while the
/// row-world path is still live, and the tests.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn entry(symbol: &str) -> Option<&'static Entry> {
    FAMILIES
        .iter()
        .flat_map(|family| family.iter())
        .find(|entry| entry.contract.symbol == symbol)
}

/// Every contract in fn-world, as the `KernelSig` rows `model-compiler`
/// reads.
///
/// **These rows state no `operands`.** That is not an omission — it is the
/// third of the three mechanisms by which a row loses its ahead-of-time C
/// shim entry (`abi.rs`'s `stated()` drops a row with an empty operand
/// list), and it is the mechanism every ported row is carried by. A symbol
/// in fn-world has no ahead-of-time launcher to name, and the operand list
/// that used to be its binding instruction is now the `fn`'s parameter list.
pub static SIGS: &[&[kernels::KernelSig]] = &[rope::SIGS];
