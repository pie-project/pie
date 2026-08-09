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
//! (`fire::rmsnorm` and `fire::dsv4_hc` until §5 step 5 took `norm` to
//! `kernels_cuda_new::x::norm`, and `rope` until it crossed to
//! `kernels_cuda_new::x::rope`) do not add three more. All three of those
//! callers are gone now, which is the point of the paragraph below rather
//! than a contradiction of this one: they did not stop needing the body,
//! they moved to the copy that lives beside the `.cuh`. **The resolution
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

use kernels_cuda_new::ArgValue;
use kernels_cuda_new::jit::{Ctx, Launch, Root};

/// Launch one instantiation of `root`, with the operand list already built.
///
/// The five call sites left here are the driver's own kernels — the KV cell
/// copy, the two page-view builders and the MLA naive pair. They have no
/// `bind!` arm and no trace names them, so they never became routines; what
/// they need is the launch and nothing else.
///
/// # Panics
///
/// If the compile, the load or the launch refuses. See the module header for
/// why none of these may be answered with a `bool`.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
pub fn fire(
    root: &'static Root,
    instantiation: &str,
    launch: Launch,
    values: &[ArgValue],
    stream: *mut std::ffi::c_void,
) {
    // SAFETY: the caller built `values` from its own live allocations and
    // holds `stream` across the launch -- the obligation every `<<<>>>` made.
    let fired = unsafe { Ctx::on(stream).launch(root, instantiation, launch, values) };
    if let Err(why) = fired {
        panic!("{instantiation}: {why}");
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
