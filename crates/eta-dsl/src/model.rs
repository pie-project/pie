//! Trace-known model constants supplied by the SDK from runtime host calls.
//! Layer count and intrinsic gates never enter this guest-side state; the
//! runtime-owned `ModelProfile` is authoritative at bind.

use core::cell::Cell;

#[derive(Clone, Copy, Debug)]
struct TraceConstants {
    vocab: u32,
    page_size: u32,
}

impl Default for TraceConstants {
    fn default() -> Self {
        TraceConstants {
            vocab: 32_000,
            page_size: 16,
        }
    }
}

thread_local! {
    static MODEL: Cell<TraceConstants> = Cell::new(TraceConstants::default());
}

pub(crate) fn with_constants<R>(vocab: u32, page_size: u32, f: impl FnOnce() -> R) -> R {
    MODEL.with(|model| {
        let previous = model.replace(TraceConstants { vocab, page_size });
        let result = f();
        model.set(previous);
        result
    })
}

pub(crate) fn vocab() -> u32 {
    MODEL.with(|m| m.get().vocab)
}
pub(crate) fn page_size() -> u32 {
    MODEL.with(|m| m.get().page_size)
}
/// Standalone unit goldens may install a complete profile, but production
/// author code has no model-configuration surface.
#[cfg(test)]
pub fn with_test_profile<R>(
    profile: &eta_ir::registry::ModelProfile,
    f: impl FnOnce() -> R,
) -> R {
    with_constants(profile.vocab, profile.page_size, f)
}

