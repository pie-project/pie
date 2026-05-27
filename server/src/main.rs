//! `pie` CLI binary — thin wrapper over [`pie_server::cli::dispatch`].
//!
//! Subcommand layout (mirrors the legacy `pie_cli`):
//!
//! ```text
//! pie serve   [--config --host --port --no-auth --debug --no-snapshot --monitor]
//! pie run     <inferlet> [...]
//! pie new     <name> [--ts -o <dir>]      # forwards to `python3 -m bakery create`
//! pie build   <path> -o <output>          # forwards to `python3 -m bakery build`
//! pie config  init|show|set
//! pie auth    add|remove|list
//! pie model   list|download|remove
//! pie driver  list | <type> {install,doctor,set,unset,show,exec}
//! pie doctor
//! pie check   <toml> [--debug]
//! pie smoke   [--rpc]
//! ```
//!
//! All of the work happens in `pie_server::cli::dispatch`.

// jemalloc as the global allocator: thread-cached, reduces malloc
// contention on the burst-allocation pattern the scheduler +
// chain-extender pool produce. Significant on AMD EPYC multi-socket
// systems where glibc malloc's locking-arena fallback bottlenecks
// under N>30 parallel allocators.
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(windows)]
fn main() {
    let handle = std::thread::Builder::new()
        .name("pie-main".to_string())
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .expect("spawn pie main thread");

    match handle.join() {
        Ok(()) => {}
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[cfg(not(windows))]
fn main() {
    run();
}

fn run() {
    if let Err(e) = pie_server::cli::dispatch() {
        eprintln!("pie: {e:#}");
        std::process::exit(1);
    }
}
