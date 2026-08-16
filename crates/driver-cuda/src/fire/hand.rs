//! A LAUNCH THIS DRIVER STATES, and the single path that fires it.
//!
//! The other path is a ROUTINE: a crossed family's host `fn`, which builds
//! its own rectangle inside `kernels-cuda` beside the `.cuh` it fires.
//! The difference is not who computes the geometry — both do it by hand now —
//! but WHO OWNS THE KERNEL. A routine's caller names a symbol; this
//! function's caller names a `Root` and an instantiation, because the thing
//! it is firing is the driver's own and no trace mentions it.
//!
//! `fire/attn_score.rs` and `fire/gemv.rs` each carried a private copy of
//! this body, as did the ported families (`fire::rmsnorm`, `fire::dsv4_hc`
//! and `rope`) before §5 step 5 took them to `kernels_cuda::{norm,rope}`.
//! Those callers are gone, and the point of the paragraph is that they did
//! not stop needing the body — they moved to the copy beside the `.cuh`.
//!
//! **THE RESOLUTION ORDER THIS HEADER USED TO STATE IS GONE, AND WITH IT THE
//! REASON THIS MODULE WAS SHARED.** It was `unit_of`, then `unit.row`, then
//! `cache::module`, then `Args::bind`, then `fire` — five steps, in an order
//! a second copy could get wrong, which is what made one factored body worth
//! having. There are no units and no rows: `Ctx::launch` takes the root and
//! the instantiation and `jit::cache::resolve` compiles that one template-id.
//! One step cannot be put in the wrong order, so the copies below the seam
//! are no longer a drift risk and this module is no longer load-bearing for
//! the reason it was written. It survives on its five call sites alone.
//!
//! `kernels_cuda::jit`'s own path is that one step and cannot call this
//! one — `driver-cuda` depends on `kernels-cuda` and not the reverse.
//! The north star's §5 ledger has this module dying once every family has
//! crossed, and the five below say which crossings are left.
//!
//! # Why every failure is a panic
//!
//! Every one is drift between this driver and the device text, or text that
//! will not compile. A caller that reached here has already decided it will
//! launch, and there is no ahead-of-time launcher to fall back to — there is
//! no ahead-of-time anything, which is what the per-symbol JIT means. A
//! `false` here would report a broken root as an unknown kernel, which is the
//! one diagnosis that sends a reader to the wrong file.
//!
//! (The sentence that stood here named `emit_c_shim`, `execution::RUST_SERVED`
//! and `pie::JIT_DISPATCHED` as the things that decided a row had no shim
//! entry. All three are deleted, and the argument outlived them intact
//! because it never depended on there being a table — only on there being
//! nothing else to try.)
//!
//! **A refusal is never a fallback.** Where a ported launcher declines — an
//! empty extent, a `hc_mult` past the kernel's register array — the port
//! declines in the launcher's own words, before it reaches this function.

use kernels_cuda::ArgValue;
use kernels_cuda::jit::{Ctx, Launch};

/// Launch one instantiation out of the carried file `file` names, with the
/// operand list already built.
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
    file: &'static str,
    instantiation: &str,
    launch: Launch,
    values: &[ArgValue],
    stream: *mut std::ffi::c_void,
) {
    // SAFETY: the caller built `values` from its own live allocations and
    // holds `stream` across the launch -- the obligation every `<<<>>>` made.
    let fired = unsafe { Ctx::on(stream).launch(file, instantiation, launch, values) };
    if let Err(why) = fired {
        panic!("{instantiation}: {why}");
    }
}

// `aligned16` STOOD HERE and was the FIFTH copy of one predicate.
//
// A HOST test over an ADDRESS, made before any launch: `(uintptr_t)p & 15u)
// == 0`, from `rmsnorm.cu:30`, `gemv.cu:299` and `rope.cu`. No `LaunchRule`
// and no `Source` could see one, which is why every launcher that made this
// test had to be ported rather than routed — and that sentence is the whole
// reason it was worth stating here.
//
// It has no caller in this crate and has not had one since the families that
// made the test crossed. `kernels_cuda::jit::aligned16` is where it
// lives now, and that function's doc records what happened in between: four
// families each grew a private copy, producing three spellings of one
// predicate, each citing a different address that no longer resolves. A
// fifth copy here — unused, un-warned because it is `pub` — is exactly the
// shape that produced those three.
