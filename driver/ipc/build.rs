//! Build script — publishes the ipc C header directory to downstream
//! crates via `links = "pie_ipc"`.
//!
//! `pie-worker`'s build.rs reads `DEP_PIE_IPC_INCLUDE` to locate this
//! directory and passes it to CMake via `-DPIE_IPC_INCLUDE_DIR=...` so
//! the C++ driver backends (cuda/metal) can `#include <pie_ipc.h>`
//! (the in-proc vtable mechanism) and `<pie_ipc/inproc_server.hpp>`.
//! The schema descriptor types those headers reference come from
//! `pie-driver-abi`'s parallel `DEP_PIE_DRIVER_ABI_INCLUDE` handoff.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = crate_dir.join("include");

    println!(
        "cargo:rerun-if-changed={}",
        include.join("pie_ipc.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        include.join("pie_ipc/inproc_server.hpp").display()
    );
    println!("cargo:include={}", include.display());
}
