//! `pie` CLI binary — thin wrapper over [`pie_standalone::cli::dispatch`].
//!
//! Subcommand layout (mirrors the legacy `pie_cli`):
//!
//! ```text
//! pie serve   [--config --host --port --no-auth --verbose --no-snapshot --monitor]
//! pie run     <inferlet> [...]
//! pie new     <name> [--ts -o <dir>]      # forwards to `python3 -m bakery create`
//! pie build   <path> -o <output>          # forwards to `python3 -m bakery build`
//! pie config  init|show|set
//! pie auth    add|remove|list
//! pie model   list|download|remove
//! pie driver  list | <type> {install,doctor,set,unset,show,exec}
//! pie doctor
//! pie check   <toml>
//! pie smoke   [--rpc]
//! ```
//!
//! All of the work happens in `pie_standalone::cli::dispatch`.

fn main() {
    if let Err(e) = pie_standalone::cli::dispatch() {
        eprintln!("pie: {e:#}");
        std::process::exit(1);
    }
}
