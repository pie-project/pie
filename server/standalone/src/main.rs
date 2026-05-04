//! Pie — standalone (no-Python) server entry point.
//!
//! Subcommand layout (mirrors `pie_cli`):
//!
//! ```text
//! pie serve   [--config --host --port --no-auth --verbose --no-snapshot --monitor]
//! pie run     <inferlet> [...]
//! pie new     <name> [--ts -o <dir>]      # forwards to `python3 -m bakery create`
//! pie build   <path> -o <output>          # forwards to `python3 -m bakery build`
//! pie config  init|show|set
//! pie auth    add|remove|list
//! pie model   list|download|remove
//! pie doctor
//! pie check   <toml>
//! pie smoke   [--rpc]
//! ```

mod aux_ipc;
mod bootstrap_translate;
mod cli;
mod config;
mod driver_ffi;
mod embedded_driver;
mod hf;
mod paths;
mod py_runtime;
mod rpc_loop;
mod serve;

fn main() {
    if let Err(e) = cli::dispatch() {
        eprintln!("pie: {e:#}");
        std::process::exit(1);
    }
}
