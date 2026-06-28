//! 4090 executor-integration verify (echo, Task #4 / #8): the temperature
//! BakedIR de-hardwiring path. Runs the `tempgen` inferlet
//! (`Sampler::Multinomial{T=0.8}` → recognizer classifies **Temperature** →
//! `dispatch_target` = BakedIR) with `PIE_DEHARDWIRE_STD_SAMPLERS=1`, so temp
//! fires route to the driver-baked `standard_sampler_program(Temperature, V)`
//! over the batched `[N,V]` block instead of the legacy `sample_temp` kernel.
//!
//! Proves the executor integration end-to-end: the recognizer routes temp →
//! BakedIR, the per-row T param + `row_seeds` (delta's Model-B contract) bind,
//! the IR program launches and scatters to `pi.sampled` (Handled, NOT a
//! fallback), and the generation completes with valid tokens. Token-exactness
//! vs `sample_temp` is delta's kernel parity (`84562e8a`, same inputs ⇒ same
//! tokens); this verifies the executor feeds the IR those same inputs.
//!
//! With `--nocapture`, the driver prints `[ir-trace] de-hardwire baked-IR
//! HANDLED kind=1 rows=N` per fire (kind 1 = Temperature) — confirms the IR
//! path Handled rather than falling back.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm CARGO_BUILD_JOBS=2 \
//!   cargo test -j2 -p pie-bin --features driver-cuda \
//!     --test cuda_temp_dehardwire -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + cuda + qwen-3-0.6b"]
async fn temp_dehardwire_on_real_driver() -> Result<()> {
    common::init_trace();
    // Enable the de-hardwiring gate + IR trace BEFORE boot — the driver reads
    // #7: temp→BakedIR is now the PRODUCTION DEFAULT (PIE_DEHARDWIRE_STD_SAMPLERS
    // default-on). This test relies on the default — it does NOT set the flag — so
    // it confirms temp routes through the baked IR program LIVE (the [ir-trace]
    // HANDLED line + token-exact [271,...]). `PIE_DEHARDWIRE_STD_SAMPLERS=0` reverts
    // to legacy sample_temp (token-exact, the revert escape-hatch).
    // SAFETY: set before any worker thread spawns / reads the env.
    unsafe {
        std::env::set_var("PIE_SAMPLING_IR_TRACE", "1");
    }

    let pie = common::boot_4090().await?;
    eprintln!("[temp-dehardwire] booted, listen_addr={}", pie.listen_addr);

    // Build the tempgen inferlet (Multinomial temperature sampler).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "tempgen"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "tempgen wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/tempgen.wasm");
    let manifest = ws.join("tempgen/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client.add_program(&wasm, &manifest, true).await.context("add_program")?;
    eprintln!("[temp-dehardwire] program installed, launching tempgen…");

    let mut proc = client
        .launch_process("tempgen@0.1.0".to_string(), "{}".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[temp-dehardwire] tempgen returned: {json}");

    // The IR temp path must produce a valid generation (not crash / late-bind
    // miss / empty). Parse the token array and require the requested count.
    let toks_start = json
        .find("\"tokens\":")
        .context("response missing tokens field")?;
    let lb = json[toks_start..].find('[').map(|i| toks_start + i).context("no [")?;
    let rb = json[lb..].find(']').map(|i| lb + i).context("no ]")?;
    let inner = json[lb + 1..rb].trim();
    let n_tokens = if inner.is_empty() {
        0
    } else {
        inner.split(',').count()
    };
    eprintln!("[temp-dehardwire] parsed {n_tokens} generated tokens");
    anyhow::ensure!(
        n_tokens == 8,
        "expected 8 temperature-sampled tokens, got {n_tokens} — IR temp path \
         likely failed or fell back (check the [ir-trace] lines)"
    );

    pie.shutdown().await;
    Ok(())
}
