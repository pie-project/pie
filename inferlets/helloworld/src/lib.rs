//! A simple hello world example demonstrating the inferlet runtime.
//!
//! This example prints a greeting and displays runtime information including
//! the instance ID and runtime version.

use inferlet::{Result, runtime};

const HELP: &str = "\
Usage: helloworld [OPTIONS]

A simple hello world program for the Pie runtime.

Options:
  -h, --help  Prints this help message";

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    println!("Hello World!!");

    let inst_id = runtime::instance_id();
    let version = runtime::version();
    println!(
        "I am an instance (id: {}) running in the Pie runtime (version: {})!",
        inst_id, version
    );

    Ok(String::new())
}
