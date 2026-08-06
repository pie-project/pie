//! Method-isolated Tree-of-Thought inferlet for reasoning benchmarks.

use inferlet::Result;
use reasoning_core::{Input, Output};

#[inferlet::main]
async fn main(mut input: Input) -> Result<Output> {
    input.force_pattern("tree_of_thought");
    reasoning_core::run(input).await
}
