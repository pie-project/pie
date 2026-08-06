//! Method-isolated Best-of-N inferlet for reasoning benchmarks.

use inferlet::Result;
use reasoning_core::{Input, Output};

#[inferlet::main]
async fn main(mut input: Input) -> Result<Output> {
    input.force_pattern("best_of_n");
    reasoning_core::run(input).await
}
