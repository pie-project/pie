//! Error test inferlet — always returns an error.

use inferlet::Result;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    Err("intentional test error".to_string())
}
