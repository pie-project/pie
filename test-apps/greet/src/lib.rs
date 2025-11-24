//! Test application that greets a user by name.
//!
//! This program imports the `greet-lib` component and calls its `greet` function
//! to generate a greeting message. It accepts a required `--name` or `-n` argument.
//!
//! Used for testing component linking with standalone libraries that don't import
//! any Pie runtime APIs.

use inferlet::{Args, Result};

// Generate WIT bindings for importing greet-lib
wit_bindgen::generate!({
    path: "wit",
    world: "greet",
    generate_all,
});

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let name: String = args.value_from_str(["-n", "--name"])?;
    Ok(greet::lib::greet::greet(&name))
}
