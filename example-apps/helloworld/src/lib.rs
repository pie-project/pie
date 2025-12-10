//! A simple hello world example demonstrating the inferlet runtime.
//!
//! This example prints a greeting and displays runtime information including
//! the instance ID and runtime version.

use inferlet::{Args, Result};

const HELP: &str = "\
Usage: helloworld [OPTIONS]

A simple hello world program for the Pie runtime.

Options:
  -h, --help  Prints this help message";

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    println!("Hello World!!");

    let inst_id = inferlet::get_instance_id();
    let version = inferlet::get_version();
    println!(
        "I am an instance (id: {}) running in the Pie runtime (version: {})!",
        inst_id, version
    );

    Ok(())
}
