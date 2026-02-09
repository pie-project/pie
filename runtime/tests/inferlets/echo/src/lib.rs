//! Echo test inferlet — returns args joined as output.

use inferlet::Result;

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    Ok(args.join(" "))
}
