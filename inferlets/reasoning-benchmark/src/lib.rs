//! Compatibility inferlet for comparing reasoning workflows with a pattern switch.
//!
//! New benchmark runs can target the method-specific `reasoning-*` inferlets.
//! This wrapper remains as a prototype/reference surface.

use inferlet::Result;
use reasoning_core::{Input, Output};

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    reasoning_core::run(input).await
}
