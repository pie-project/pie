//! Echo test inferlet — returns input as output.

use inferlet::Result;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    Ok(input)
}
