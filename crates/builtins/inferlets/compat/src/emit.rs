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

/// A frame, naming for the gateway the tokens it covers since the frame
/// before and the response's id, so the turn can move ([`crate::resume`]).
fn frame(mut frame: Value) {
    let tokens = crate::resume::take();
    if !tokens.is_empty() {
        frame["plex"] = json!({"tokens": tokens, "id": crate::short_id()});
    }
    inferlet::session::send(&frame.to_string());
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
