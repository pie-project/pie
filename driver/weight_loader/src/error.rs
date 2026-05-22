use std::ffi::CString;

use thiserror::Error;

use crate::ffi_types::{PieLoaderError, PieLoaderStatus};

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("null argument: {0}")]
    NullArgument(&'static str),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl CompileError {
    pub fn status(&self) -> PieLoaderStatus {
        match self {
            Self::NullArgument(_) => PieLoaderStatus::NullArgument,
            Self::InvalidInput(_) => PieLoaderStatus::InvalidInput,
            Self::Internal(_) => PieLoaderStatus::InternalError,
        }
    }
}

fn sanitize_cstring_message(message: &str) -> CString {
    let sanitized = message.replace('\0', "\\0");
    CString::new(sanitized).expect("sanitized message contains no NUL")
}

/// Clear and free an FFI error object previously written by this crate.
///
/// # Safety
///
/// `error` may be null. When non-null, it must point to a valid
/// `PieLoaderError` whose `message` field is either null or was allocated by
/// `write_error`/`CString::into_raw` from this crate.
pub unsafe fn clear_error(error: *mut PieLoaderError) {
    if error.is_null() {
        return;
    }
    let error_ref = unsafe { &mut *error };
    if !error_ref.message.is_null() {
        let _ = unsafe { CString::from_raw(error_ref.message) };
    }
    error_ref.code = PieLoaderStatus::Ok;
    error_ref.message = std::ptr::null_mut();
}

/// Write a structured FFI error object.
///
/// # Safety
///
/// `error` may be null. When non-null, it must point to a valid mutable
/// `PieLoaderError`; any existing `message` field must satisfy the same
/// ownership requirements as `clear_error`.
pub unsafe fn write_error(error: *mut PieLoaderError, err: &CompileError) {
    if error.is_null() {
        return;
    }
    unsafe { clear_error(error) };
    let error_ref = unsafe { &mut *error };
    error_ref.code = err.status();
    error_ref.message = sanitize_cstring_message(&err.to_string()).into_raw();
}
