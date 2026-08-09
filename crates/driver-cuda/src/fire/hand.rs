//! A LAUNCH THIS DRIVER STATES, and the single path that fires it.
//!
//! `bind::jit::fire` is the other one, and the difference is where the
//! geometry comes from. There, a [`kernels::LaunchRule`] on the row produces
//! the rectangle from a [`kernels_cuda_new::Dims`]; here the caller builds a
//! [`Launch`] itself, because the launcher it is porting stated a geometry no
//! rule states — a `dim3(token, head)` grid, a dynamic shared allocation
//! sized off `head_dim`, a block width chosen from an alignment test.
//!
//! `fire/attn_score.rs` and `fire/gemv.rs` each carry a private copy of this
//! function; this is the same body, factored, so that the ported families
//! (`fire::rmsnorm`, `fire::dsv4_hc`, and `rope` until it crossed to
//! `kernels_cuda_new::x::rope`) do not add three more. **The resolution
//! order is the contract** — `unit_of`, then `unit.row`, then
//! `cache::module`, then `Args::bind`, then `fire` — and a copy per call
//! site is one place per copy for that order to drift.
//!
//! A FOURTH COPY NOW EXISTS AND IS DELIBERATE: `kernels_cuda_new::x::fire`
//! reproduces this resolution order for the families that live beside their
//! `.cuh`. It cannot call this one — `driver-cuda` depends on
//! `kernels-cuda-new` and not the reverse — and the north star's §5 ledger
//! has this module dying once every family has crossed. Until then the two
//! must agree, and `x::fire`'s doc says so from its side.
//!
//! # Why every failure is a panic
//!
//! Every one of them is drift between this driver and its kernel table, or a
//! unit that will not compile. A caller that reached here has already decided
//! it will launch: `emit_c_shim` emitted no entry for the row (that is what
//! [`kernels_cuda_new::execution::RUST_SERVED`] and
//! [`kernels_cuda_new::device::JIT_DISPATCHED`] each do), so there is no
//! ahead-of-time launcher left to fall back to and no second answer to give.
//! A `false` here would report a broken table as an unknown kernel, which is
//! the one diagnosis that sends a reader to the wrong file.
//!
//! **A refusal is never a fallback.** Where a ported launcher declines — an
//! empty extent, a `hc_mult` past the kernel's register array — the port
//! declines in the launcher's own words, before it reaches this function.

use kernels_cuda_new::runtime::{ArgValue, Args, Launch, Stream, cache};

/// Resolve one row through the JIT table, bind the operands, launch.
///
/// `symbol` names a [`kernels_cuda_new::device::DeviceKernel`] — the
/// `__global__`'s contract, not the launcher's — so `values` is that row's
/// operand list and carries neither the stream nor any extent a grid already
/// states. `Args::bind` checks the list against the signature, so a drift
/// between a call site here and the family table is a refusal at the bind
/// rather than a shifted argument at the kernel.
///
/// # Panics
///
/// If `symbol` is in no unit, is not one of its unit's rows, its unit will
/// not compile or load, the operand list disagrees with the row, or the
/// driver refuses the launch. See the module header for why none of these may
/// be answered with a `bool`.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
pub fn fire(
    symbol: &'static str,
    launch: Launch,
    values: &[ArgValue],
    stream: *mut std::ffi::c_void,
) {
    let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
    };
    let module = match cache::module(index, unit) {
        Ok(module) => module,
        Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
    };
    let mut args = match Args::bind(sig, values) {
        Ok(args) => args,
        Err(why) => panic!("{symbol}: {why}"),
    };
    // SAFETY: the caller holds the fire's stream live across the launch — the
    // same assertion it made when it handed the stream to a C++ launcher that
    // put it in a `<<<>>>`.
    let stream = unsafe { Stream::from_runtime(stream) };
    if let Err(why) = module.fire(sig, launch, &mut args, stream) {
        panic!("{symbol}: {why}");
    }
}

/// `rmsnorm.cu:30`, `gemv.cu:299`, `rope.cu` — `(uintptr_t)p & 15u) == 0`.
///
/// A HOST test made before any launch, over an ADDRESS. No `LaunchRule` and
/// no `Source` can see one — `Term::Aligned` exists for a
/// [`kernels_cuda_new::device::Specialisation`] and chooses an
/// instantiation, not a geometry — which is why every launcher that made this
/// test had to be ported rather than routed.
#[must_use]
pub fn aligned16(p: *const std::ffi::c_void) -> bool {
    p.addr() & 15 == 0
}
