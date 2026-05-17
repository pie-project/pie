//! Caller inferlet for the `runtime::launch` E2E test.
//!
//! Demonstrates the v2 launch API: `inferlet::launch(...)?` returns a `Child`
//! handle that can be `.await`ed directly via its `IntoFuture` impl.

use inferlet::{Result, launch};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    callee: String,
    prompt: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    println!("[caller] launching {} with prompt: {}", input.callee, input.prompt);
    let child = launch(&input.callee, &input.prompt)?;
    println!("[caller] child pid: {}", child.pid());
    let reply = child.await?;
    println!("[caller] child returned: {}", reply);
    Ok(reply)
}
