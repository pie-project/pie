//! The six `Dispatch*` impls: every arm is destructure → resolve → call
//! (decision #13), one arm per variant, matches exhaustive.
//!
//! No arm selects a kernel — dtype, lattice point, gemv-vs-dense, and smem
//! arm all live inside the `kernels-cuda` entries — and no arm syncs
//! (#15): a returned `Ok` means the launch is on the stream, nothing more,
//! so the same arms run identically inside a graph capture. The one routing
//! an arm does perform is *resolution*: following a plan output's declared
//! `StructKind`, or a plan slot's held kind — choices the trace already
//! wrote down, not choices made here.
//!
//! Alias outputs (`#[out(alias = x)]`) bind as `_`: the compiler folded
//! them onto their input's slot, so the input name is the one the in-place
//! kernel reads. An entry that refuses (`attention.prefill_sm90`) still
//! gets a real resolve → call arm, so the typed refusal carries the
//! entry's own name.
//!
//! The plan-building arms are the prepare phase's whole population (#16):
//! each one runs a pure builder over the host twins in [`FireBindings`],
//! `stage`s the schedule's upload immediately — eagerly, on the stream,
//! before any capture begins — and seats the payload for the consuming
//! arms.
//!
//! The impls live one per family in the submodules below — [`attn`],
//! [`linear`], [`elemwise`], [`layout`], [`collective`], [`custom`] — and
//! within each impl the arms keep their old family grouping, marked by
//! section comments: the merged `Attention` walks attention, mla, ssm,
//! index, pool; `Linear` walks gemm, mlp, moe; `Elementwise` walks norm,
//! rope, gate, hc.

mod attn;
mod collective;
pub(crate) mod copy;
mod custom;
mod elemwise;
mod layout;
mod linear;
