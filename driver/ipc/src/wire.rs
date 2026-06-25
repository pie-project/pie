//! Thin wire-format helpers around rkyv. The canonical schema is in
//! [`crate::schema`]; this module just wraps rkyv's encode/access/
//! deserialize entry points with the project's error idiom.

use rkyv::rancor::Error as RancorError;

use crate::schema::{
    ArchivedFrame, ArchivedResponseFrame, Frame, ResponseFrame, ResponsePayload, StatusResponse,
};

#[derive(thiserror::Error, Debug)]
pub enum WireError {
    #[error("rkyv access/verify failed: {0}")]
    Verify(String),

    /// Server-side handler dropped the lease without commit; the response
    /// frame's `aborted` bit is set.
    #[error("handler aborted")]
    HandlerAborted,
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a request `Frame` to its wire bytes.
pub fn encode_request(frame: &Frame) -> Result<Vec<u8>, WireError> {
    rkyv::to_bytes::<RancorError>(frame)
        .map(|aligned| aligned.to_vec())
        .map_err(|e| WireError::Verify(format!("encode_request: {e}")))
}

/// Encode a response `ResponseFrame` to its wire bytes.
pub fn encode_response(frame: &ResponseFrame) -> Result<Vec<u8>, WireError> {
    rkyv::to_bytes::<RancorError>(frame)
        .map(|aligned| aligned.to_vec())
        .map_err(|e| WireError::Verify(format!("encode_response: {e}")))
}

// ---------------------------------------------------------------------------
// Verified zero-copy access (shmem path — untrusted producer)
// ---------------------------------------------------------------------------

/// Validate the buffer and return a zero-copy archived view. Use on the
/// shmem read path where the producer is in another process.
pub fn parse_request(buf: &[u8]) -> Result<&ArchivedFrame, WireError> {
    rkyv::access::<ArchivedFrame, RancorError>(buf)
        .map_err(|e| WireError::Verify(format!("parse_request: {e}")))
}

/// Validate + check the `aborted` bit. Returns [`WireError::HandlerAborted`]
/// if the server-side handler dropped the lease without commit.
pub fn parse_response(buf: &[u8]) -> Result<&ArchivedResponseFrame, WireError> {
    let archived = rkyv::access::<ArchivedResponseFrame, RancorError>(buf)
        .map_err(|e| WireError::Verify(format!("parse_response: {e}")))?;
    if archived.aborted {
        return Err(WireError::HandlerAborted);
    }
    Ok(archived)
}

// ---------------------------------------------------------------------------
// Trusted access (in-process FFI — same-process producer)
// ---------------------------------------------------------------------------

/// Zero-copy view without verification. Use ONLY when the producer is in
/// the same process and linked against the same schema.
///
/// # Safety
/// Caller asserts `buf` is a valid rkyv `Frame` archive. UB on violation.
pub unsafe fn parse_request_trusted(buf: &[u8]) -> &ArchivedFrame {
    unsafe { rkyv::access_unchecked::<ArchivedFrame>(buf) }
}

/// Trusted response access. Mirrors [`parse_request_trusted`].
///
/// # Safety
/// Caller asserts `buf` is a valid rkyv `ResponseFrame` archive.
pub unsafe fn parse_response_trusted(buf: &[u8]) -> &ArchivedResponseFrame {
    unsafe { rkyv::access_unchecked::<ArchivedResponseFrame>(buf) }
}

// ---------------------------------------------------------------------------
// Owned deserialize (when you need a fully owned copy)
// ---------------------------------------------------------------------------

/// Convert an archived view back to its owned form. Allocates per-field.
pub fn deserialize_response(archived: &ArchivedResponseFrame) -> Result<ResponseFrame, WireError> {
    rkyv::deserialize::<ResponseFrame, RancorError>(archived)
        .map_err(|e| WireError::Verify(format!("deserialize: {e}")))
}

// ---------------------------------------------------------------------------
// Convenience: build the canonical abort frame
// ---------------------------------------------------------------------------

/// Build the abort frame written by `Lease::Drop` when a lease drops
/// without commit. A well-formed `ResponseFrame` with `aborted: true`
/// and a `StatusResponse { -1 }` payload.
pub fn build_abort_frame(driver_id: u32) -> Result<Vec<u8>, WireError> {
    encode_response(&ResponseFrame {
        driver_id,
        aborted: true,
        payload: ResponsePayload::Status(StatusResponse { status: -1 }),
    })
}
