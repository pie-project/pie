//! Design B COMPACTION device e2e — real driver (4090). First real end-to-end
//! exercise of the `pipeline.copy_into` KV cell-move primitive (K3/K6 resolved:
//! the move rides the SAME scheduler FIFO / CUDA stream as the forward fires, so
//! the B3 ordering invariant sequences it — no QUIESCE/drain barrier).
//!
//! The `beam-designb-compact` inferlet runs the exact `beam-designb` steady-state
//! mask-out beam decode over a single materialised page, but in `move` mode it
//! issues one `pipeline.copy_into` at a mid-run step that physically relocates
//! the shared BOS KV cell (all layers, post-RoPE) from flat pool position 0 to a
//! materialised-but-never-written cell (flat 15), and remaps the per-beam
//! `AttnMask` in-graph to attend the moved cell at its new home:
//!
//!   guest mask-out beam program + copy_into(dst,src)
//!     → runtime device-geometry submit (run-ahead FIFO)
//!     → PendingOp::Move enqueued on the same FIFO right behind the fire
//!     → driver `is_kv_move()` branch → `launch_copy_kv_cells_bf16` per layer
//!       (two-pointer disjoint move, no scratch) on the fire stream
//!     → the next fire's masked attention reads BOS from its NEW physical slot.
//!
//! **Correctness oracle — identical tokens.** Because KV is stored POST-RoPE, a
//! physical slot is pure storage; moving BOS's K/V and pointing the mask at the
//! new column is the same query attending the same stored K/V, so a faithful
//! `copy_into` must leave the emitted token stream UNCHANGED. This test runs the
//! SAME wasm twice on the same booted driver — once in `no-move` mode, once in
//! `move` mode (identical code, geometry and klen; the only difference is the
//! physical KV relocation) — and asserts the two token vectors are byte-identical.
//!
//!   PIE_PTIR_TRACE=1 cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_beam_designb_compact_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Extract the `tokens=[...]` list from an inferlet return summary.
fn extract_tokens(out: &str) -> Option<String> {
    let start = out.find("tokens=[")?;
    let rest = &out[start + "tokens=".len()..];
    let end = rest.find(']')?;
    Some(rest[..=end].to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Design B compaction device e2e: needs the 4090 + cuda + qwen-3-0.6b + the ptir feature"]
async fn beam_designb_compact_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[beam-compact-e2e] booted, listen_addr={}", pie.listen_addr);

    // Build the compaction inferlet to wasm (member of the runtime test-inferlets
    // ws). The crate name normalizes to `beam_designb_compact.wasm`.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "beam-designb-compact",
        ])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "beam-designb-compact wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/beam_designb_compact.wasm");
    let manifest = ws.join("beam-designb-compact/Pie.toml");
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());

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

    // Same wasm, two runs. The ONLY difference is the mid-run physical KV move.
    let nomove_out = {
        let mut proc = client
            .launch_process(
                "beam-designb-compact@0.1.0".to_string(),
                "{}".to_string(),
                true,
            )
            .await
            .context("launch no-move")?;
        proc.wait_for_return().await.context("wait no-move")?
    };
    eprintln!("[beam-compact-e2e] no-move returned: {nomove_out}");

    let move_out = {
        let mut proc = client
            .launch_process(
                "beam-designb-compact@0.1.0".to_string(),
                "{\"move\":1}".to_string(),
                true,
            )
            .await
            .context("launch move")?;
        proc.wait_for_return().await.context("wait move")?
    };
    eprintln!("[beam-compact-e2e] move    returned: {move_out}");

    pie.shutdown().await;

    // The move run must have completed the whole loop (no crash on the move).
    anyhow::ensure!(
        move_out.contains("BEAM_DESIGNB_COMPACT"),
        "compaction e2e: unexpected return (a fire/move-path crash returns the \
         error string): {move_out:?}"
    );
    let move_tokens = extract_tokens(&move_out)
        .with_context(|| format!("no tokens in move return: {move_out:?}"))?;
    let nomove_tokens = extract_tokens(&nomove_out)
        .with_context(|| format!("no tokens in no-move return: {nomove_out:?}"))?;

    // Non-degeneracy: the move run harvested tokens through the fire+move path.
    anyhow::ensure!(
        move_tokens != "[]",
        "compaction e2e: no tokens harvested — the submit/move/take loop produced \
         an empty hypothesis: {move_out:?}"
    );

    // Correctness oracle: a faithful post-RoPE cell move leaves the token stream
    // unchanged. Any divergence means copy_into moved the wrong bytes, landed on
    // the wrong slot, or raced the fires' KV writes/reads (a real ordering fork).
    anyhow::ensure!(
        nomove_tokens == move_tokens,
        "compaction e2e: token streams DIVERGED after the copy_into move — \
         no-move={nomove_tokens} move={move_tokens}. A faithful all-layers \
         post-RoPE cell move must be token-neutral; divergence is a real move/ordering bug."
    );

    eprintln!(
        "[beam-compact-e2e] GREEN — pipeline.copy_into moved BOS's KV on device \
         (same-FIFO, no drain) and the token stream is IDENTICAL to the \
         no-move run: tokens={move_tokens}"
    );
    Ok(())
}
