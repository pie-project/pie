use core::ffi::c_void;

use crate::runtime::{ArgValue, Args, Stream, cache};
use crate::x::launch::Launch;

/// Resolve one symbol through the JIT unit table, bind its operands, launch.
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
    let fired = if cooperative {
        // SAFETY: as above, plus the residency precondition this function's
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
#[must_use]
pub fn aligned16(p: *const c_void) -> bool {
    p.addr() & 15 == 0
}
