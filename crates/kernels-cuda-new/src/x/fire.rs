//! The single path from a host program to the device.
//!
//! `driver-cuda`'s `fire::hand::fire` is this function one crate up, and its
//! header explains why there is exactly one: **the resolution order is the
//! contract** — `unit_of`, then `unit.row`, then `cache::module`, then
//! `Args::bind`, then `fire` — and a copy per call site is one place per
//! copy for that order to drift. `fire::attn_score` and `fire::gemv` each
//! carried a private copy; factoring them into `fire::hand` was the first
//! step and this is the second, because a host program that lives beside its
//! device text should not have to reach up into the driver to launch.
//!
//! # Why every failure is a panic
//!
//! Every one of them is drift between a declaration and its device text, or
//! a unit that will not compile. A caller that reached here has already
//! decided it will launch — its contract states no `operands`, so
//! `emit_c_shim` emitted no ahead-of-time entry and there is no second
//! answer to give. A `false` here would report a broken table as an unknown
//! kernel, which is the one diagnosis that sends a reader to the wrong file.
//!
//! **A refusal is never a fallback.** Where a host program declines — an
//! empty extent, a head narrower than one pair — it declines in its own
//! words, as a `Fired::Declined`, before it reaches this function.

use core::ffi::c_void;

use crate::runtime::{ArgValue, Args, Stream, cache};
use crate::x::launch::Launch;

/// Resolve one symbol through the JIT unit table, bind its operands, launch.
///
/// `symbol` names a `DeviceKernel` — the `__global__`'s contract, not the
/// host program's — so `values` is that row's parameter list in order and
/// carries neither the stream nor any extent the grid already states.
///
/// # Panics
///
/// When the symbol is in no unit, is not one of its unit's rows, its unit
/// will not compile, the values do not match the declared signature, or the
/// launch itself fails. Each is drift, and each names the symbol.
///
/// # Safety
///
/// Every pointer in `values` must address live device memory of the size the
/// kernel will read or write, and `stream` must be live across the launch.
/// That is the same assertion the caller made when it handed a stream to a
/// C++ launcher that put it in a `<<<>>>`.
pub unsafe fn fire(symbol: &'static str, launch: Launch, values: &[ArgValue], stream: *mut c_void) {
    // SAFETY: the caller's contract, forwarded unchanged.
    unsafe { fire_ex(symbol, launch, false, values, stream) }
}

/// [`fire`], for a kernel that must be launched COOPERATIVELY.
///
/// # Why this exists rather than a copy at the call site
///
/// A `unit!`'s `raw::` stub calls [`fire`], which calls
/// `KernelModule::fire` — so a `unit!` launcher could not be cooperative at
/// all, whatever `runtime::module::fire_ex` offered. The alternative was a
/// local copy of this body in the one family that needs it, and this module's
/// header refuses that: **a copy per call site is one place per copy for the
/// order to drift**, and the order here is six steps long with two panics
/// between them.
///
/// # What `cooperative` buys and what it costs
///
/// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` makes `cooperative_groups::this_grid()`
/// meaningful — `csrc/shim/cooperative_groups.h` provides `grid_group` only
/// because this path exists, and says so at length.
///
/// **The residency precondition is the caller's and nothing here can check
/// it.** A grid barrier is safe only when every block is resident at once,
/// which depends on the compiled function's register and shared-memory
/// footprint against the device — `cuOccupancyMaxActiveBlocksPerMultiprocessor`
/// against the live `CUfunction`, which this function has but the caller's
/// grid arithmetic already committed to. `x::attn::mla_fa2` sizes its grid at
/// `num_sm` blocks, resident by construction. **A caller that does not is
/// asking for a hang rather than an error**, and neither the shim nor the
/// driver will say so.
///
/// # Safety
///
/// [`fire`]'s, and additionally: `cooperative` may be `true` only for a grid
/// every block of which is resident.
pub unsafe fn fire_ex(
    symbol: &'static str,
    launch: Launch,
    cooperative: bool,
    values: &[ArgValue],
    stream: *mut c_void,
) {
    debug_assert!(
        !launch.disagrees(),
        "{symbol}: smem_opt_in disagrees with smem — the mismatch `LaunchRule::Rope` had"
    );
    let Some((index, unit)) = crate::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this declaration and its unit disagree");
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
    // SAFETY: the caller holds the fire's stream live across the launch.
    let stream = unsafe { Stream::from_runtime(stream) };
    // The ordinary path stays `fire`, not `fire_ex` with three falses: it is
    // the one every family takes, and routing it through the attribute array
    // would put a `cuLaunchKernelEx` where a `cuLaunchKernel` has always been
    // for the sake of a flag that is off.
    let fired = if cooperative {
        // SAFETY: as above, plus the residency precondition this function's
        // doc puts on the caller.
        unsafe {
            module.fire_ex(symbol, launch.into(), None, false, true, args.slots_mut(), stream)
        }
    } else {
        module.fire(sig, launch.into(), &mut args, stream)
    };
    if let Err(why) = fired {
        panic!("{symbol}: {why}");
    }
}

/// `rmsnorm.cu:30`, `gemv.cu:299`, `rope.cu` — `((uintptr_t)p & 15u) == 0`.
///
/// A HOST test made before any launch, over an ADDRESS. No `LaunchRule` and
/// no `Source` could see one — `Term::Aligned` existed for a
/// `Specialisation` and chose an instantiation, not a geometry — which is
/// why every launcher that made this test had to be ported rather than
/// routed. In fn-world it is an `if`.
#[must_use]
pub fn aligned16(p: *const c_void) -> bool {
    p.addr() & 15 == 0
}
