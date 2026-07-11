//! DIAGNOSTIC (echo): isolate whether the real-driver inproc *forward* path
//! hangs in general, or only for sampling-program fires. Runs the plain
//! `generate` inferlet (legacy `Sampler::TopK`, NO sampling program) through the
//! same boot + submit path as the capability harness. If THIS hangs too, the
//! issue is the general real-driver forward/response cycle (not the IR path);
//! if it completes, the hang is sampling-program-specific.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_plain_gen -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "diagnostic: needs the 4090 + cuda + qwen-3-0.6b"]
async fn plain_generate_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[diag] booted, listen_addr={}", pie.listen_addr);

    // Build the plain `generate` inferlet (legacy TopK sampler, no IR program).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "generate"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "generate wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/generate.wasm");
    let manifest = ws.join("generate/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client
        .authenticate("test-user", &None)
        .await
        .context("auth")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;
    eprintln!("[diag] program installed, launching plain generate…");

    let mut proc = client
        .launch_process("generate@0.1.0".to_string(), "{}".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[diag] plain generate returned: {json}");

    pie.shutdown().await;
    Ok(())
}
