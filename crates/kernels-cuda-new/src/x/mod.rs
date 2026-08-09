//! **kernel-x** — the floor a kernel stands on when it is written as a
//! program rather than as a row.
//!
//! `.wiki/kernel-x/northstar.md` is the design; this module is its §5 steps 1
//! and 2, [`rope`] is its step 3, and [`route`] is its step 4.
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
//! | [`Route`] / [`route`] | 5 (4) | what will fire a symbol, decided at model load |
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
//! 4. **The flip resolves to a [`Route`], not to an `Option<&Entry>`.**
//!    §5 step 4 writes `lowered.kernels: Vec<&'static Entry>`. Two answers
//!    are not enough for a question with four: `None` was carrying "not
//!    ported yet", "the driver's own op" and "nothing declares this" at
//!    once, and **"unknown symbols refuse at load" cannot be written against
//!    a value that cannot say "unknown".** Naming the four is the whole of
//!    what made step 4's second half landable. (§5.1 ④.)
//!
//! # Registering a ported family — two lists, and they are gated apart
//!
//! Appending to [`FAMILIES`] gives a symbol a bind; appending to [`SIGS`]
//! gives it a row `model-compiler` can read and `table::TABLES`
//! concatenates. Do both, in that order, and delete the family's module from
//! `table::ROW_TABLES` in the same change. Nothing else is a registration —
//! see each list's doc for what forgetting it looks like.
//!
//! **Both lists is the usual shape, not the only one.** A family whose
//! symbols the driver itself serves — `adapter` — contributes to `SIGS`
//! alone, because an `Entry` for a driver op would shadow [`route`]'s
//! `DriverOp` arm and refuse a live model at load. [`SIGS`] names the three
//! shapes and [`route`] gives the mechanism.

pub mod abi;
pub mod contract;
pub mod cx;
pub mod launch;
#[macro_use]
pub mod macros;
pub mod adapter;
pub mod attn;
pub mod driver_internal;
pub mod gemm;
pub mod layout;
pub mod mlp;
pub mod moe;
// `pub mod moe_glue;` WAS HERE, for six `__global__`s extracted out of
// `cutlass_fused_moe_kernels.cuh`, and it lived for about four hours.
//
// It went with the fused CUTLASS MoE leg, on the owner's decision and this
// measurement: carrying CUTLASS so the JIT could compile the GEMM is a
// **505-file, 13,891,303-byte `include_str!` closure**, against the
// **429-file, 4,376,255-byte** carry this tree already refused in writing for
// cub (`csrc/src/moe/expert_offsets.cuh:48`). Same mechanism — the carried
// set is `include_str!`, so it is binary size — and 3.2 times a line already
// drawn. The nine shim names that made the glue compile without CUTLASS went
// with it; they existed for these six kernels and nothing else.
//
// **The glue was the fused leg's machinery, not the general path's.** The
// aligned leg runs on `x::moe`'s twenty contracts, and what it is missing is
// one bind, not these six kernels — see `BUILD_MOE_PTRS_ALIGNED`'s `none:`
// arm, which is the gate on this retirement being complete rather than
// merely done.
pub mod norm;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;
pub mod xqa;

#[cfg(feature = "_cuda")]
pub mod fire;

pub use abi::{Abi, ByValue, Layout, fp8_kind};
pub use contract::{Contract, Entry, Fired, Refusal};
#[cfg(feature = "_cuda")]
pub use contract::Route;
pub use cx::{
    AttnWorkspace, Cx, Facts, Gdn, KvDType, KvLayer, KvScheme, MlaLayer, MlaPlan, Plan, Rows, Slab,
    Yarn,
};

/// Every family that has crossed into fn-world.
///
/// One entry per family, appended as §5 step 5 walks the census. `rope` is
/// the pilot; `layout` and `sample` are §5's "boring first", landed to
/// settle the idioms before `norm`, `ssm` and `attn`.
///
/// **`adapter` is deliberately absent**, and it is the third registration
/// shape: it has a [`Contract`] in [`SIGS`] and no [`Entry`] here, because
/// an `Entry` would make [`route`] answer `Bound` for a symbol that needs a
/// cuBLAS handle a [`Cx`] cannot lend. See `x/adapter.rs`'s header, which is
/// the worked example.
///
///
/// The linear scan is load-time work and only load-time work: [`route`]
/// resolves every symbol in a lowering once, at model load, and no fire
/// compares a symbol string. Twelve rows scanned once per distinct symbol is
/// not a data structure.
///
/// **Gated on `_cuda`, where [`SIGS`] is not.** An [`Entry`] holds a bind,
/// and a bind launches; a [`Contract`] holds what a trace may say, and
/// `model-compiler` reads it with no GPU anywhere. That the two lists are
/// gated differently IS the "must not be able to tell cuBLAS from a JIT'd
/// kernel" rule, expressed where the compiler can enforce it. [`Route`] is
/// gated for the same reason and is the sharper case: telling them apart is
/// the whole of what it does.
///
/// It is also why the step-4 intern lives on `driver-cuda`'s op join and
/// not on `Lowered` as §5 draws it: a lowering carrying an [`Entry`] would
/// hand that same distinction to the GPU-free crate. See §5.1.
#[cfg(feature = "_cuda")]
pub static FAMILIES: &[&[Entry]] = &[
    rope::ENTRIES,
    layout::ENTRIES,
    sample::ENTRIES,
    quant::ENTRIES,
    mlp::ENTRIES,
    norm::ENTRIES,
    ssm::ENTRIES,
    // `moe` crossed whole: nine `.cuh` roots, six units, twenty contracts.
    // Eight of the twenty bind, ELEVEN are `none:` and one — the fused
    // CUTLASS block — is the third shape, a driver op with a contract and
    // no entry. The eleven are not shy: four routers and a table lookup
    // want two deployment constants `Cx` cannot ask for and two members of
    // the aligned path want an operand's row count it cannot ask for
    // either, and `x/moe.rs`'s header states the six-line patch that turns
    // all six into binds. The other five never fired.
    moe::ENTRIES,
    // `attn` IS HALF A FAMILY AND THAT IS DELIBERATE. Six of its
    // forty-one rows have crossed; the other thirty-five are still
    // `table::attn`'s and still reach [`Route::Rows`], which is what that
    // variant's fallthrough is for. A symbol appears in exactly one of the
    // two worlds, so the scan below stays disjoint — `x/attn.rs`'s header
    // says which six and why the rest waited.
    attn::ENTRIES,
    // `xqa` was FLOOR until this line: 197 lines of `KvCacheList` mirror and
    // `by_value!` support, no `unit!` and no `contract!`, and
    // `no_hollow_family.rs` exempted it on exactly that basis. It is a family
    // now — five `Unit`s over `attn/attention_xqa_mha.cuh` and one contract
    // over all five, because the host program picks the member and fires
    // once. The archive spelled that choice as six translation units and a
    // C++ dispatcher; **all seven files are deleted** and the choice lives in
    // `driver-cuda/src/fire/xqa.rs::XqaMember::pick`.
    //
    // The `bind!` is a `none:` arm and its own comment argues three `Cx`
    // facts rather than one about XQA — the host program is complete and 959
    // lines, and what it cannot become is a `bind!` body.
    xqa::ENTRIES,
];

/// The [`Entry`] for one symbol, or `None` if no family declares it.
///
/// The narrow lookup: "does fn-world hold this". [`route`] is the one every
/// load path should call, because `None` here is three different answers and
/// `route` tells them apart.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn entry(symbol: &str) -> Option<&'static Entry> {
    FAMILIES
        .iter()
        .flat_map(|family| family.iter())
        .find(|entry| entry.contract.symbol == symbol)
}

/// What will fire one symbol — §5 step 4's resolution, in the one crate that
/// can see both registries.
///
/// # Why here and not in the driver
///
/// Both oracles are this crate's. [`FAMILIES`] holds fn-world; `table::sig`
/// holds the rows; `execution::service` holds the driver ops. A driver-side
/// `resolve` would have had to import all three and re-state the precedence,
/// and the precedence is a property of the registries rather than of the
/// driver. `driver-cuda` maps this over `lowered.kernels` and no more.
///
/// # The order, which is not a preference
///
/// The order below is the cheapest way to ask, and — with one exception
/// named next — the four sets are disjoint by construction:
///
/// 1. **fn-world first**, because a ported family's symbol is deleted from
///    every row table in the same change that ports it, so a symbol in both
///    is a bug in the port and not a case this function resolves. Asking
///    fn-world first means that bug surfaces as the *right* answer rather
///    than as a row-world dispatch of a ported kernel.
/// 2. **Driver ops before rows**, because `execution::SERVICE` and
///    `table::KERNELS` both hold `pie_lora_qkv_correction` — the row is what
///    lets a trace state it, the service row is what says the driver
///    implements it. The service answer is the more specific one.
/// 3. **Rows last**, and [`Route::Rows`] carries the condition that removes
///    it.
///
/// # THE ONE OVERLAP, and the rule it forces on a port
///
/// A driver op is the exception: nothing stops a family module from putting
/// an [`Entry`] in [`FAMILIES`] for a symbol `execution::SERVICE` already
/// answers `DriverOp` for. Step 1 shadows step 2 and the symbol never
/// reaches the `Driver` arm. If that `Entry` is a `none:` — which is the
/// natural thing to write for a symbol whose host program is not a kernel —
/// the answer becomes [`Route::Unbound`] and **`build_lowered_fire` refuses
/// the model at load for a symbol that fires correctly today.**
/// `pie_lora_qkv_correction` is on the first launch of every LoRA fire, so
/// the blast radius of that one line is every adapter deployment.
///
/// So the rule, and it is short: **a driver op contributes to [`SIGS`] and
/// not to [`FAMILIES`].** The contract still exists — `model-compiler` must
/// read a row, and must not be able to tell what serves it — but fn-world
/// declares no `Entry`, `entry()` answers `None`, and the DriverOp arm is
/// reached. `x/adapter.rs` is the worked example; see [`SIGS`]'s "three
/// registration shapes".
///
/// # WHAT MAKES A SYMBOL A DRIVER OP, and it is not difficulty
///
/// **A driver op is a symbol whose body needs a driver RESOURCE** — a cuBLAS
/// handle, an NCCL communicator, a memory pool, an allocator, an arena. Not
/// a symbol whose host program is long, and not one whose bind would be
/// awkward.
///
/// The distinction cost a port a wrong answer before it was written down.
/// `kv_paged`'s four walks were classified as driver ops on the ground that
/// *"their host programs live in `driver-cuda` and `kernels-cuda-new` cannot
/// call `driver-cuda`."* **True, and not the reason** — the dependency runs
/// the other way, and `fire/kv_paged.rs` already calls
/// `x::layout::envelope_merge_written` from the middle of two of those very
/// bodies. Every fact those four read is a field of one
/// `KvCacheLayerView`, so they are a **move**, not a driver op, and the
/// eleven fields [`Cx::kv_layer`] grew are what the move needed.
///
/// `x::gemm`'s twelve are driver ops because `cublasLtMatmul` is across a
/// seam no [`Cx`] can cross. `moe::flashinfer_cutlass_moe_bf16` was one
/// because it needed a workspace query, an allocation and an arch probe.
/// `moe::build_moe_ptrs_aligned_bf16` is one because the six pointer arrays
/// it carves live in the driver's arena. **In each case the test is the same
/// and it is answerable in one line: name the resource.** If you cannot,
/// it is a move.
///
/// This is the only case where a `none:` arm is wrong. Everywhere else a
/// `none:` is exactly right and is the whole point of [`Route::Unbound`]:
/// it converts a row the old grammar could not source into a load-time
/// sentence. The distinction is whether something ELSE already fires the
/// symbol.
///
/// # What this deliberately does NOT decide
///
/// Whether the row world has an ARM for a [`Route::Rows`] symbol. That is
/// `emit_rust_dispatch`'s rule — every operand carrying a `Source` — and
/// deciding it here would be writing the emitter's rule a second place, in a
/// crate that cannot see the emitter's output. A row symbol with no arm
/// still refuses at the fire, as it does today. The gap closes when
/// `Route::Rows` does.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn route(symbol: &str) -> Route {
    if let Some(entry) = entry(symbol) {
        return match (entry.bind, entry.unbound) {
            (Some(_), _) => Route::Bound(entry),
            (None, Some(why)) => Route::Unbound(entry, why),
            // A declaration with neither is the towers case — a `fn` that
            // exists, is public, and is never trace-fired. A trace naming
            // one is asking for something the declaration says it will not
            // do, and the contract's own symbol is the only sentence there
            // is to print.
            (None, None) => Route::Unbound(entry, "this symbol is not trace-fired"),
        };
    }
    if matches!(
        crate::execution::service(symbol),
        Some(crate::execution::Service::DriverOp)
    ) {
        return Route::Driver;
    }
    if crate::table::sig(symbol).is_some() {
        return Route::Rows;
    }
    Route::Unknown
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
///
/// # This is the ONE place a ported family is registered
///
/// `table::TABLES` concatenates this list, so appending here puts a family
/// in `table::KERNELS` and therefore past `check_plan`. A §5 step-5 port is
/// three edits: append to [`FAMILIES`], append here, delete the row module
/// from `table::ROW_TABLES`. **Two lists, not three**, and they are gated
/// differently on purpose — see [`FAMILIES`].
///
/// Appending here and forgetting [`FAMILIES`] is the failure worth knowing:
/// the symbol passes `check_plan`, resolves to [`Route::Rows`] because a row
/// exists, and dispatches through a row table the port just emptied. It
/// fails at the fire, not at the load. Appending to `FAMILIES` and
/// forgetting here fails at the load with a clear sentence, which is the
/// better half of the two — so when in doubt, do `FAMILIES` first.
///
/// # The three registration shapes
///
/// | shape | `FAMILIES` | here | what it is |
/// |---|---|---|---|
/// | a ported kernel family | yes | yes | `rope`, `layout`, `sample` — the sweep's normal case |
/// | a family with unbindable members | yes | yes | the `none:` arms; [`Route::Unbound`] refuses them at load with the reason |
/// | a DRIVER OP | **no** | yes | `adapter` — the driver already fires it; see [`route`]'s "the one overlap" |
///
/// The third shape was found by the `layout`/`sample`/`adapter` port and is
/// worth stating rather than deriving, because the natural thing to write
/// for a driver op is a `none:` arm — a symbol whose host program is not a
/// kernel *looks* exactly like a symbol nothing can fire. It is the opposite:
/// something already fires it, and an `Entry` shadows the arm that says so.
/// The contract still belongs here, because `model-compiler` must read a row
/// and must not be able to tell what serves it — which is the same sentence
/// [`FAMILIES`]'s gating enforces, arriving from the other side.
pub static SIGS: &[&[kernels::KernelSig]] = &[
    rope::SIGS,
    layout::SIGS,
    sample::SIGS,
    adapter::SIGS,
    quant::SIGS,
    mlp::SIGS,
    norm::SIGS,
    ssm::SIGS,
    // Every `moe` contract, including the one with no [`Entry`]: the fused
    // CUTLASS block is `fire::flashinfer_moe`'s, and `model-compiler` must
    // still be able to answer "is this a symbol?" once `table::moe`'s row
    // is gone. Being HERE is what answers it — and it is also what took the
    // symbol off `execution::RUST_SERVED`, since a `Contract::sig` states no
    // operands and that list exists to drop shim entries a stated row would
    // otherwise generate.
    moe::SIGS,
    // `gemm` is the third shape entire: twelve contracts, zero entries. Its
    // host programs need a cuBLAS handle, which is a device API with a
    // settable stream, math mode and workspace — precisely the surface §3.3
    // keeps out of `Cx` — so every one of them is a driver op and none of
    // them is a bind. It is absent from `FAMILIES` for that reason and not
    // by omission.
    gemm::SIGS,
    // The five `attn` contracts that crossed. `table::attn` keeps the other
    // thirty-six, and `table::TABLES` concatenates both — so
    // `model-compiler` reads one vocabulary and cannot tell which world
    // serves a symbol, which is the property a partial port is allowed to
    // have.
    attn::SIGS,
    // The other half of `xqa`'s registration, and the half that decides
    // whether the symbol resolves at all: `attn::attention_xqa_decode_bf16_
    // prepared` left `table::attn` in the same change that added the
    // `contract!`, so between the two edits it was `Route::Unknown`. That is
    // the `gemm` shape and the reason registration is ATOMIC rather than
    // ordered — three edits, one commit, or a symbol refuses at model load
    // with nothing in the diff to say why.
    xqa::SIGS,
];
