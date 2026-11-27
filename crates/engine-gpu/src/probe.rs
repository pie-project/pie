#![allow(unsafe_code)]
use std::any::TypeId;
use std::cell::Cell;

/// The frame this thread records into, and its type.
#[derive(Clone, Copy)]
pub struct Recording {
    frame: *const (),
    kind: Option<TypeId>,
}

impl Recording {
    const NONE: Recording = Recording {
        frame: std::ptr::null(),
        kind: None,
    };
}

thread_local! {
    static FRAME: Cell<Recording> = const { Cell::new(Recording::NONE) };
}

/// Points this thread's probes at `frame` and returns what they pointed at
/// before. A kernel sink sets it for as long as it borrows the frame and
/// puts the previous one back when dropped ([`restore`]), so the pointer is
/// live or null.
pub fn set_frame<F: 'static>(frame: *const F) -> Recording {
    FRAME.with(|f| {
        f.replace(Recording {
            frame: frame.cast(),
            kind: Some(TypeId::of::<F>()),
        })
    })
}

/// Puts back what [`set_frame`] returned.
pub fn restore(outer: Recording) {
    FRAME.with(|f| f.set(outer));
}

/// Runs `f` on this thread's recording frame, if it is an `F`.
pub fn with_frame<F: 'static, R>(f: impl FnOnce(&F) -> R) -> Option<R> {
    let at = FRAME.with(Cell::get);
    if at.frame.is_null() || at.kind != Some(TypeId::of::<F>()) {
        return None;
    }

    // SAFETY: a non-null `FRAME` is the frame a live sink on this thread
    // borrows (see `set_frame`), so it outlives this call, and its type is
    // `F` (checked above); only a shared reference is handed out, as the
    // sink itself holds.
    Some(f(unsafe { &*at.frame.cast::<F>() }))
}
