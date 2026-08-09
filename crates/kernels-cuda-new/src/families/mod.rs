//! One module per kernel family, each owning the units it compiles.
//!
//! # Why the units are split by family rather than listed in one place
//!
//! A unit is one NVRTC compile: one `.cuh` of `__global__` templates, one
//! header set, many name expressions. There are as many of them as there are
//! device headers in `kernels-cuda/csrc/src`, and a single list would be a file
//! every migration touches — the shape that makes parallel work collide and a
//! diff unreadable.
//!
//! So a family owns its own units, its own rows, and the `include_str!` that
//! carries its source. Adding a family is one module and one line in [`ALL`];
//! adding a unit to a family touches one file.
//!
//! # What a family module holds
//!
//! * `UNITS` — one entry per `.cuh` the family compiles.
//! * The [`crate::device::DeviceKernel`] rows those units instantiate, and the
//!   `KernelSig`s behind them.
//!
//! The sigs are written here rather than reused from [`crate::table`] because
//! they are not the same contract. A table row describes a `pie_k_*` entry
//! point: a host function holding a `<<<>>>`, taking a stream. A row here
//! describes a template instantiation and states its geometry as a
//! `LaunchRule` — the thing the launcher used to hold. `norm_device.rs`
//! records the measurement: the same six kernels went from thirty-one operands
//! to twenty-one, and the ten that vanished were six streams and four extents
//! the rules recover.
//!
//! # The order is stable, not semantic
//!
//! [`crate::unit::UNITS`] concatenates these in [`ALL`]'s order, and a unit's
//! position there is its slot in the module cache. Nothing depends on which
//! slot a unit gets, and a reordering invalidates nothing: the cache is
//! per-process and the cubin cache keys on the unit's NAME.

use crate::unit::Unit;

// `families::adapter` IS GONE — §5 step 5 took it into fn-world as
// `x::adapter`. It never held a unit and never will: the LoRA seam is a
// sequence of cuBLAS batched GEMMs and there is no `__global__` in it. Its
// header's argument for why a family with no device text is still a family
// moved with the file, where it is now also the worked example of the third
// registration shape — a `Contract` in `x::SIGS` with no `Entry` in
// `x::FAMILIES`.
pub mod attn;
/// FlashInfer's cascade merge — the split-KV path's other half.
///
/// A family of its own and not a corner of [`attn`] or of [`fa2`], for a
/// reason each of those two gives from its own side: it is upstream's device
/// text rather than ours, so it does not belong beside [`attn`]'s launchers;
/// and it is not part of the FA2 instantiation lattice — its axes are a head
/// dim and a choice of three kernels, not [`fa2`]'s four — so a unit of it
/// inside [`fa2`] would be a fifty-seventh unit that none of that family's
/// derivations reach.
pub mod cascade;
/// The FlashInfer FA2 lattice — 56 units over four axes, the last thing in
/// the tree that needed nvcc.
///
/// A family of its own and not a corner of [`attn`], for the reason [`graph`]
/// is one: it is a whole instantiation lattice with its own axes, its own
/// derivation ([`crate::fa2`]) and its own vendored header closure, and
/// [`attn`] is already 4,000 lines of unrelated launchers. It is also the only
/// family whose units are macro-generated, which reads badly interleaved with
/// hand-written rows.
pub mod fa2;
// `families::gemm` IS GONE — §5 step 5 took it into fn-world as `x::gemm`,
// which is where its `unit!` and its two GEMV rows are now. The 240-launch
// census, the three retired findings and the `gemv_bf16` porting notes moved
// with the file and are its module header.
/// The supergraph's two arming kernels — the one family named after a SHELL
/// object rather than a kind of value. See its header for why that is a
/// directory of its own and not a corner of `layout`.
pub mod graph;
// `families::layout` IS GONE — §5 step 5 took it into fn-world as
// `x::layout`, which is where its five units and sixteen device rows are now.
// The five roots became five inline `pub mod`s, one per `unit!`, because a
// `unit!` generates `UNITS`/`ROWS`/`PARAMS`/`raw` at its own scope and two
// invocations in one module collide; `x::mlp` found that first.
pub mod marlin;
// `families::mlp` IS GONE — §5 step 5 took it into fn-world as `x::mlp`,
// which is where its two units and sixteen device rows are now. Its
// `Composed` pair — the one statement that fired two different kernels —
// became a two-call `fn` body, which is §2.3's shape and needed no
// combinator.
// `families::moe` IS GONE — §5 step 5 took it into fn-world as `x::moe`,
// nine `csrc/src/moe/` roots arranged as SIX units — one inline `pub mod`
// per root, `x::layout`'s idiom — over twenty-five device rows and twenty
// host programs. Three of the nine roots are text only and stay text: the
// two CuTile `_tile.cuh` variants and `moe_fused_tile.cuh` need a compiler
// this tree does not have, and `x::moe`'s header carries their measurements
// so the ask survives the port. `csrc/src/moe/flashinfer_moe.cu` is NOT in
// that count: it is the last ahead-of-time compile in the tree, it belongs
// to the CUTLASS runner behind it, and its symbol is a driver op.
// `families::norm` IS GONE — §5 step 5 took it into fn-world as `x::norm`,
// six units over `csrc/src/norm/`'s eight roots, thirty-five device rows and
// thirty-two host programs. It is §5.1's named proof of `Composed`/`Walk`:
// `norm::rmsnorm_bf16_with_fp16`'s three arms are a two-call `fn` body whose
// second launch is another FAMILY's kernel, which is the one shape §2.3 does
// not cover. Its `SPECIALISATIONS` went with it — `RMSNORM_STRIDED_VEC8`'s
// six-term predicate is an `if` in `x::norm::strided_bf16`.
// `families::quant` IS GONE — §5 step 5 took it into fn-world as `x::quant`,
// seven roots, thirty-eight device rows and fifteen host programs. Two of the
// seven join a `unit!`'s rows to hand-written ones: `table::moe`'s four routed
// MoE decode GEMVs live in `quant`'s headers but are fired rule-driven, so
// `x::quant` keeps their `KernelSig`s verbatim and concatenates.
// `families::sample` IS GONE — §5 step 5 took it into fn-world as
// `x::sample`. Its "five things no row says" table moved with it and is now
// a table of where each one went, all five into one host `fn`.
// `families::ssm` IS GONE — §5 step 5 took it into fn-world as `x::ssm`,
// five roots (`causal_conv1d`, `gated_delta_net`, `gated_delta_net_prep`,
// `kda`, `nemotron_h`), thirty-eight device rows and twenty-seven host
// programs. Fifteen of those host programs are new: their rows were
// `device::JIT_DISPATCHED` and fired through a `LaunchRule`, so there was no
// `.cu` launcher to move and the rule's expression is transcribed beside each
// one. `x::driver_internal` keeps the two `qwen_gdn_post_conv_prep_bf16`
// symbols, which are an ordered pair a driver op fires and not a bind.
pub mod vision;

/// Every family's units, in a stable order.
///
/// Listed by name rather than discovered, because a family that appeared
/// silently would be compilable and unreachable — `unit_of` scans this, so a
/// unit not in it hosts no symbol and every fire of its rows is refused as
/// unknown.
///
/// # `rope` is not a module here, and that is the migration
///
/// [`crate::x::rope`] declares its own unit with `unit!`, which generates the
/// same [`Unit`] and the same `DeviceKernel` rows this module's families
/// write by hand — so `unit_of`, `cache::module`, `Args::bind` and
/// `tests/units.rs` reach it unchanged. What is gone is the hand-written
/// `KernelSig` beside each row: the operand list comes from the declared
/// parameter types through `x::Abi`, and the geometry is a `fn`.
///
/// # `adapter::UNITS` is empty, and is listed anyway
///
/// The LoRA seam compiles no device text. It stays in this list because the
/// list is a claim about the census — every family, in a stable order — and
/// an omitted name reads as an oversight where an empty slice reads as a
/// fact. Concatenating it costs nothing.
pub static ALL: &[&[Unit]] = &[
    crate::x::adapter::UNITS,
    // BOTH HALVES OF `attn`, and the pair is the shape of a family mid-port:
    // three roots have crossed and twenty have not, and a root is in exactly
    // one of the two lists — a second `unit!` naming the same text would
    // compile it twice and `unit_of` would answer with whichever won.
    crate::x::attn::UNITS,
    attn::UNITS,
    cascade::UNITS,
    fa2::UNITS,
    crate::x::gemm::UNITS,
    graph::UNITS,
    crate::x::layout::UNITS,
    marlin::UNITS,
    crate::x::mlp::UNITS,
    crate::x::moe::UNITS,
    crate::x::norm::UNITS,
    crate::x::quant::UNITS,
    crate::x::rope::UNITS,
    crate::x::sample::UNITS,
    crate::x::ssm::UNITS,
    vision::UNITS,
];
