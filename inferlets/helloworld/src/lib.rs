//! A simple hello world example demonstrating typed JSON input/output.
//!
//! Shows the `#[inferlet::main]` macro with `Deserialize` input
//! and `Serialize` output structs.

use inferlet::{Result, runtime};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    help: bool,
}

#[derive(Serialize)]
struct Output {
    message: String,
    instance_id: String,
    version: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.help {
        println!("Usage: helloworld\n\nA simple hello world program for the Pie runtime.");
        return Ok(Output {
            message: String::new(),
            instance_id: String::new(),
            version: String::new(),
        });
    }

    let inst_id = runtime::instance_id();
    let version = runtime::version();

    println!("Hello World!!");
    println!(
        "I am an instance (id: {}) running in the Pie runtime (version: {})!",
        inst_id, version
    );

    Ok(Output {
        message: "Hello World!!".into(),
        instance_id: inst_id,
        version,
    })
}
