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

// mimalloc as the global allocator: thread-cached, low contention,
// good performance for the burst-allocation pattern the scheduler +
// chain-extender pool produce.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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

// Graph-build regression test for the portable driver's uniform-top-K
// sampling reshape. The five per-arch graph builders share
// `build_sampling_outputs`, whose gather reshape must key on the slot count
// (`n_slots`) rather than the request count (`n_req`): pass-level
// speculation samples more than one slot per request (draft verification),
// so `n_slots >= n_req`. A test-only C entry in the portable driver builds
// the sampling subgraph and reports the slot dimension of `top_k_probs`.
//
// This lives in the `pie` bin because CI runs `cargo test -p pie-server
// --bin pie`, and the driver/portable ctest targets never run in CI.
#[cfg(all(test, feature = "driver-portable"))]
mod portable_graph_sampling_tests {
    use std::os::raw::c_int;

    unsafe extern "C" {
        fn pie_portable_test_uniform_top_slot_dim(
            n_req: c_int,
            n_slots: c_int,
            k: c_int,
            vocab: c_int,
        ) -> c_int;
    }

    fn slot_dim(n_req: i32, n_slots: i32, k: i32, vocab: i32) -> i32 {
        unsafe { pie_portable_test_uniform_top_slot_dim(n_req, n_slots, k, vocab) }
    }

    // The speculation case: more sampling slots than requests. If a future
    // change keys the gather reshape on n_req again, GGML's nelements
    // assert aborts this process instead of returning n_slots.
    #[test]
    fn uniform_top_reshape_keys_on_n_slots_under_speculation() {
        assert_eq!(slot_dim(2, 5, 8, 64), 5);
    }

    // Plain decode: one slot per request. The invariant must still hold.
    #[test]
    fn uniform_top_reshape_handles_n_slots_eq_n_req() {
        assert_eq!(slot_dim(3, 3, 4, 32), 3);
    }
}
