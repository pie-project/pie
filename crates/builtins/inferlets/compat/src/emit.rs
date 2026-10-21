//! The streaming side of the gateway contract.
//!
//! Every `session.send` from a built-in inferlet is one SSE frame:
//!
//! ```json
//! {"event": "content_block_delta", "data": {…}}
//! {"data": "[DONE]"}
//! ```
//!
//! `event` is optional and becomes the frame's `event:` line; `data` is
//! serialized as JSON unless it is a string, which is written verbatim (that
//! is how OpenAI's `[DONE]` terminator goes out). The gateway knows nothing
//! else: which frames exist, and whether there is a terminator, is the API's
//! business and therefore the adapter's.

use serde_json::{Value, json};

/// A frame, carrying the tokens it covers, so the gateway can continue the
/// turn elsewhere ([`crate::continuation`]).
fn frame(frame: Value) {
    crate::continuation::send(frame);
}

/// A JSON frame with no event name.
pub fn send(data: &Value) {
    frame(json!({"data": data}));
}

/// A JSON frame under an SSE event name.
pub fn send_event(event: &str, data: &Value) {
    frame(json!({"event": event, "data": data}));
}

/// A frame whose data is written as-is.
pub fn send_raw(data: &str) {
    frame(json!({"data": data}));
}
