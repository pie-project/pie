//! C ABI surface — in-process vtable for cuda / portable / dummy.
//!
//! The vtable passes `Pie*Desc` POD pointers directly (no rkyv encode/
//! decode on the in-process path). The shmem path (Python downstream)
//! lives in [`crate::ipc`] and exchanges rkyv-archived bytes.

use std::ffi::c_void;

use crate::schema::{PieFrameDesc, PieResponseFrameDesc};

// ===========================================================================
// Vtable — the C ABI handoff
// ===========================================================================

/// Return value for [`InProcVTable::recv`]: zero on success, non-zero on
/// error (the caller may inspect a side-channel for details).
pub type RecvResult = i32;

/// Vtable signature mirrored in `include/pie_bridge.h`. The C++ side
/// populates this struct and hands its address to Rust.
///
/// # Safety
///
/// - `recv` must, on success, leave `*out_request` pointing to a
///   `PieFrameDesc` that remains valid until the matching
///   `send_response` is invoked for the same `*out_req_id`. The Rust
///   side will not modify the descriptor's bytes.
/// - The `PieFrameDesc` and every nested slice pointer it carries must
///   remain valid for the same lifetime — typically owned by the
///   caller's request queue, freed in `send_response`.
/// - `send_response` must, before returning, copy any data it needs
///   from the supplied `*const PieResponseFrameDesc`. After return,
///   the caller may reuse / free the descriptor memory.
/// - `ctx` is opaque to Rust; the C++ side may store any pointer there.
#[repr(C)]
pub struct InProcVTable {
    /// Block until a request is available; write the request's
    /// `PieFrameDesc` pointer and id to the out-parameters. Return 0
    /// on success, non-zero on shutdown / error.
    pub recv: unsafe extern "C" fn(
        ctx: *mut c_void,
        out_request: *mut *const PieFrameDesc,
        out_req_id: *mut u32,
    ) -> RecvResult,

    /// Post a `PieResponseFrameDesc` for `req_id`. The callee must
    /// consume the descriptor synchronously (the pointer and all the
    /// slice pointers inside it are invalid after this call returns).
    pub send_response:
        unsafe extern "C" fn(ctx: *mut c_void, req_id: u32, response: *const PieResponseFrameDesc),

    /// Opaque context pointer threaded through both calls.
    pub ctx: *mut c_void,
}

// SAFETY: callers must ensure ctx + the function pointers are safe to
// use across threads. By contract, drivers implement this for Send + Sync.
unsafe impl Send for InProcVTable {}
unsafe impl Sync for InProcVTable {}

// ===========================================================================
// Rust-side wrappers over the vtable
// ===========================================================================

/// Sends responses back through the vtable. Wraps the raw fn pointer
/// with a safe API.
pub struct FfiResponseSource<'v> {
    vt: &'v InProcVTable,
}

impl<'v> FfiResponseSource<'v> {
    pub fn new(vt: &'v InProcVTable) -> Self {
        Self { vt }
    }

    /// Post a response descriptor for `req_id`. The vtable contract
    /// requires the callee (Rust side) to consume the descriptor and
    /// every slice pointer it carries synchronously.
    pub fn send(&self, req_id: u32, response: &PieResponseFrameDesc) {
        // SAFETY: caller provides a valid vtable; `send_response`
        // contract requires synchronous consumption.
        unsafe {
            (self.vt.send_response)(self.vt.ctx, req_id, response as *const _);
        }
    }
}

/// Receives requests from the vtable. Each call to [`recv`] blocks
/// until a request is available (or the vtable returns a non-zero
/// shutdown code).
pub struct FfiRequestSink<'v> {
    vt: &'v InProcVTable,
}

/// One in-flight FFI request — a borrowed `PieFrameDesc` (lifetime
/// tied to the caller's request queue, valid until the matching
/// response is sent) plus the request id.
pub struct FfiRequest<'a> {
    pub req_id: u32,
    pub request: &'a PieFrameDesc,
}

impl<'v> FfiRequestSink<'v> {
    pub fn new(vt: &'v InProcVTable) -> Self {
        Self { vt }
    }

    /// Block until a request lands or the vtable returns a non-zero code.
    /// On shutdown / error, returns [`None`].
    ///
    /// # Safety
    ///
    /// The returned reference borrows from the caller's queue. The
    /// borrow is valid only until the matching response is sent via
    /// the paired [`FfiResponseSource::send`]. Callers must not retain
    /// the reference (or any of its inner slice pointers) past that
    /// point.
    pub unsafe fn recv(&self) -> Option<FfiRequest<'v>> {
        let mut request_ptr: *const PieFrameDesc = ::core::ptr::null();
        let mut req_id: u32 = 0;
        // SAFETY: out-pointers are valid for the duration of this call.
        let rc = unsafe { (self.vt.recv)(self.vt.ctx, &mut request_ptr, &mut req_id) };
        if rc != 0 || request_ptr.is_null() {
            return None;
        }
        // SAFETY: contract requires the caller (Rust queue) to keep
        // the descriptor valid until `send_response` is called.
        Some(FfiRequest {
            req_id,
            request: unsafe { &*request_ptr },
        })
    }
}
