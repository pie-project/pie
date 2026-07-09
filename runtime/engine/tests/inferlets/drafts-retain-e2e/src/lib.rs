//! **Drafts-channel retain plumbing gate** (charlie) — the tight, unconfounded
//! proof that the `[k+1]` `[seed, drafts]` device window is byte-identical through
//! the `pipeline_source_kind=1` retain→inject path, ISOLATED from MTP acceptance
//! semantics (same philosophy as the carrier FLEET=1 byte-identity gate).
//!
//! A SINGLE retain→inject cycle (independent of the multi-step n_acc position-
//! advance seam, which is pipe-audit's device-resident swap):
//!   1. PRODUCER fire: `mtp_specdecode(vocab, k)` (out[0]=commit[k+1],
//!      out[1]=drafts[k], out[2]=seed[1]) + `carrier::next_inputs_drafts(pass, k)`
//!      → the driver retains `[k+1] = [seed→row0, drafts→rows1..k]` off the
//!      `mtp_drafts` buffer (bravo's retain). Read out[1]/out[2] host-side = the
//!      EXPECTED window; emit `DRAFTS_RETAIN_EXPECTED`.
//!   2. CONSUMER fire: `[k+1]` placeholders on the same context → the carrier
//!      injects the retained window at `src_rows=[0..=k]` into `pi.tokens`.
//!
//! The harness (`cuda_drafts_retain.rs`) runs with `PIE_DRAFTS_VERIFY=1` and
//! asserts the driver's INJECT value-dump == this inferlet's EXPECTED window
//! (byte-identity) + `src_rows==[0..=k]`.
//!
//! `#[ignore]`-gated in the harness (needs 4090 + cuda + Qwen3.5-0.8B MTP head).

use inferlet::program::{encode_f32, encode_i32, resolve_bindings};
use inferlet::sampling::program::mtp_specdecode;
use inferlet::working_set::{KvWorkingSet, RsWorkingSet};
use inferlet::{model, tensor, Result};

const PROMPT: &str = "The quick brown fox jumps over";

fn read_i32(t: &tensor::Tensor) -> Result<Vec<i32>> {
    let bytes = t.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    let vocab = model::output_vocab_size();
    let (built, keys) = mtp_specdecode(vocab, k).map_err(|e| format!("mtp_specdecode: {e:?}"))?;
    let program = inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;

    let mut prompt = model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Constant lane vectors (v4 has no vector-const/iota).
    let lanes_k: Vec<i32> = (0..k as i32).collect();
    let lanes_k1: Vec<f32> = (0..=k).map(|i| i as f32).collect();
    // Bootstrap drafts: k placeholders that fully reject (-1) — the first window
    // commits the target's bonus and out[1] yields the first REAL drafts/seed.
    let draft: Vec<i32> = vec![-1; k as usize];

    let kv = KvWorkingSet::new();
    let rs = RsWorkingSet::new();
    let page = kv.page_size();

    // ---- PRODUCER fire: [prompt + k placeholder drafts], activate the retain ----
    let window: Vec<u32> = prompt
        .iter()
        .copied()
        .chain(draft.iter().map(|&t| t.max(0) as u32))
        .collect();
    let n = window.len() as u32;
    let seq_len: u32 = 0;
    let first_write_page = seq_len / page;
    let total_pages = (seq_len + n).div_ceil(page);
    let have = kv.size();
    if total_pages > have {
        kv.alloc(total_pages - have).map_err(|e| format!("alloc: {e}"))?;
    }

    let pass = inferlet::inference::ForwardPass::new();
    pass.kv_working_set(
        &kv,
        0,
        first_write_page,
        first_write_page * page,
        first_write_page,
        total_pages - first_write_page,
        seq_len % page,
    );
    if rs.state_size() > 0 {
        pass.rs_working_set(&rs, 0, n);
    }
    let positions: Vec<u32> = (seq_len..seq_len + n).collect();
    pass.input_tokens(&window, &positions);

    let base = (seq_len + n).saturating_sub(k + 1);
    let logits_positions: Vec<u32> = (0..k + 1).map(|i| base + i).collect();
    let submit = vec![
        (keys.draft, encode_i32(&draft)),
        (keys.lanes_k, encode_i32(&lanes_k)),
        (keys.lanes_k1, encode_f32(&lanes_k1)),
    ];
    let bindings = resolve_bindings(&built.bindings, &built.host_inputs, &logits_positions, &submit)?;
    pass.sampler(&program, bindings);

    // ★ Activate the drafts-channel retain: window [0..=k] + pipeline_source_kind=1.
    inferlet::carrier::next_inputs_drafts(&pass, k);
    pass.execute();

    // out[0]=commit[k+1], out[1]=drafts[k], out[2]=seed[1] — the retained window
    // is [seed, drafts] (out[2]→row0, out[1]→rows1..k). Read the EXPECTED window.
    let tensors = pass.outputs().await.map_err(|e| format!("outputs: {e}"))?;
    if tensors.len() < 3 {
        return Err(format!("expected 3 outputs (commit, drafts, seed), got {}", tensors.len()));
    }
    let drafts = read_i32(&tensors[1])?;
    let seed = read_i32(&tensors[2])?;
    let expected: Vec<i32> = seed.iter().copied().chain(drafts.iter().copied()).collect();

    let result = format!(
        "DRAFTS_RETAIN_EXPECTED k={k} seed={seed:?} drafts={drafts:?} window={expected:?}"
    );
    eprintln!("[drafts-retain-e2e] {result}");
    Ok(result)
}
