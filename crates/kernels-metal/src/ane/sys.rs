//! The C face of `native/ane.m`: every call into the private framework goes
//! through one of these. Nothing outside this module touches them.

use std::ffi::{CStr, c_char, c_int, c_void};

#[repr(C)]
pub(super) struct RawSurface {
    pub(super) surface: *mut c_void,
    pub(super) base: *mut c_void,
    pub(super) bytes: u64,
    pub(super) stride: u32,
}

pub(super) type Report = extern "C" fn(*mut c_void, c_int);

unsafe extern "C" {
    pub(super) fn pie_ane_available(err: *mut c_char, cap: c_int) -> c_int;
    pub(super) fn pie_ane_surface(
        rows: u32,
        width: u32,
        int8: c_int,
        out: *mut RawSurface,
        err: *mut c_char,
        cap: c_int,
    ) -> c_int;
    pub(super) fn pie_ane_surface_free(surface: *mut c_void);
    pub(super) fn pie_ane_program(
        directory: *const c_char,
        key: *const c_char,
        err: *mut c_char,
        cap: c_int,
    ) -> *mut c_void;
    pub(super) fn pie_ane_program_free(program: *mut c_void);
    pub(super) fn pie_ane_procedure(program: *mut c_void, function: *const c_char) -> c_int;
    pub(super) fn pie_ane_input_count(program: *mut c_void, procedure: c_int) -> c_int;
    pub(super) fn pie_ane_input_name(
        program: *mut c_void,
        procedure: c_int,
        input: c_int,
    ) -> *const c_char;
    pub(super) fn pie_ane_bind(
        program: *mut c_void,
        procedure: c_int,
        inputs: *const *mut c_void,
        count: c_int,
        output: *mut c_void,
        err: *mut c_char,
        cap: c_int,
    ) -> *mut c_void;
    pub(super) fn pie_ane_binding_free(binding: *mut c_void);
    pub(super) fn pie_ane_enqueue(
        program: *mut c_void,
        binding: *mut c_void,
        event: *mut c_void,
        wait: u64,
        signal: u64,
        report: Report,
        context: *mut c_void,
        err: *mut c_char,
        cap: c_int,
    ) -> c_int;
}

/// The bridge writes its failure messages into a buffer this long.
pub(super) const CAP: usize = 1024;

pub(super) type Message = [c_char; CAP];

pub(super) fn message() -> Message {
    [0 as c_char; CAP]
}

pub(super) fn failure(err: &Message) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}
