//! **MTP Stage-2 — DEVICE-RESIDENT spec-decode** (pipe-audit): the drafts-channel
//! swap of `mtp-native-verify`. Where `mtp-native-verify` round-trips the drafts
//! through host (`read out[1] → submit as next `draft``), this reads the CURRENT
//! window's drafts DEVICE-RESIDENT via the `Binding::MtpDrafts` intrinsic — the
//! driver source-selects bravo's retained `mtp_drafts` `[k]` buffer onto the
//! verify's `draft` operand (`mtp_specdecode_device`, `hit = target_argmax ==
//! mtp_drafts`). Zero host round-trip on the `[k]` drafts.
//!
//! Loop (PARTIAL-residency, seam-lock (b)): fire `mtp_specdecode_device` over the
//! device-injected `[k+1]` window → `carrier::next_inputs_drafts(k)` declares the
//! retain+inject of the NEXT window (`[seed, drafts]` = out[2]→row0, out[1]→rows
//! 1..k, `pipeline_source_kind=1`) → read ONLY out[0] (the `[k+1]` commit tail)
//! host-side for the committed count (`n_acc+1`, the scalar advance) + the
//! committed text. out[1]/out[2] stay on-GPU (the retain injects them). The next
//! fire's window rows are host placeholders that the carrier overwrites from the
//! retained buffer.
//!
//! ⚠️ GPU-only (like `mtp-native-verify`): the `MtpDrafts`/`MtpLogits` intrinsics
//! are disabled in the host/mock profile → RUNS on the 4090 with charlie's CUDA
//! source-select (the drafts-channel value-verify A/B vs `mtp-native-verify`,
//! accepted-tok/s). Host-COMPILES now (the full binding plumbing type-checks).
//! Two seams resolved on-device with bravo+charlie: the n_acc KV position-advance
//! (guest advances the host cursor by out[0]'s count; the retain lands the window
//! at that position) + the first-fire bootstrap (the MtpDrafts buffer is empty
//! pre-fire-0, so fire 0's commit is discarded — its out[1] seeds fire 1).

use inferlet::inference::ForwardPass;
use inferlet::program::{encode_f32, encode_i32, resolve_bindings, HostInputDecl};
use inferlet::sampling::program::{
    mtp_specdecode_bootstrap, mtp_specdecode_device, MtpSpecdecodeDeviceKeys,
};
use inferlet::working_set::{KvWorkingSet, RsWorkingSet};
use inferlet::{carrier, model, tensor, Result};

const PROMPT: &str = "The quick brown fox jumps over";
const MAX_TOKENS: u32 = 16;

/// The lowered `mtp_specdecode_device` program + its (lanes-only) host-input keys.
/// Outputs: `[0]` = `[k+1]` committed tail; `[1]` = `[k]` next drafts (retained,
/// NOT host-read); `[2]` = `[1]` seed (retained, NOT host-read).
struct Composed {
    program: tensor::Program,
    bindings: Vec<inferlet::sampling::ir::Binding>,
    host_inputs: Vec<HostInputDecl>,
    keys: MtpSpecdecodeDeviceKeys,
    k: u32,
}

fn build(vocab: u32, k: u32) -> Result<Composed> {
    let (built, keys) =
        mtp_specdecode_device(vocab, k).map_err(|e| format!("compose build: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;
    Ok(Composed {
        program,
        bindings: built.bindings,
        host_inputs: built.host_inputs,
        keys,
        k,
    })
}

/// Read a `[k+1]`/`[k]` i32 tensor's little-endian bytes into token ids.
fn read_i32(t: &tensor::Tensor) -> Result<Vec<i32>> {
    let bytes = t.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Committed length = tokens before the first `-1` (accepted prefix + bonus), ≥1.
fn committed_len(tail: &[i32]) -> usize {
    tail.iter().take_while(|&&t| t >= 0).count()
}

/// Fire ONE device-resident window: verify vs the device `MtpDrafts` drafts,
/// declare the next window's device carryover, read ONLY out[0] (the commit tail).
/// The `[k]` drafts + `[1]` seed (out[1]/out[2]) stay on-GPU (the retain injects
/// them into the next fire). Only host inputs = the two constant lane vectors.
async fn fire_window(
    prog: &Composed,
    kv: &KvWorkingSet,
    rs: &RsWorkingSet,
    window: &[u32],
    seq_len: u32,
    lanes_k: &[i32],
    lanes_k1: &[f32],
) -> Result<Vec<i32>> {
    let n = window.len() as u32;
    let k = prog.k;
    let page = kv.page_size();
    let first_write_page = seq_len / page;
    let total_pages = (seq_len + n).div_ceil(page);
    let have = kv.size();
    if total_pages > have {
        kv.alloc(total_pages - have).map_err(|e| format!("alloc: {e}"))?;
    }

    let pass = ForwardPass::new();
    pass.kv_working_set(
        kv,
        0,
        first_write_page,
        first_write_page * page,
        first_write_page,
        total_pages - first_write_page,
        seq_len % page,
    );
    if rs.state_size() > 0 {
        pass.rs_working_set(rs, 0, n);
    }
    let positions: Vec<u32> = (seq_len..seq_len + n).collect();
    pass.input_tokens(window, &positions);

    // Verify+bonus positions = the last k+1 rows; the driver serves target `logits`
    // there, `mtp_logits` at the MTP draft rows, and `mtp_drafts` (the retained
    // window's drafts) as the verify operand — all device-side.
    let base = (seq_len + n).saturating_sub(k + 1);
    let logits_positions: Vec<u32> = (0..k + 1).map(|i| base + i).collect();
    // Host inputs = ONLY the two constant lane vectors (NO `draft` submit — the
    // drafts are the device-resident MtpDrafts intrinsic).
    let submit = vec![
        (prog.keys.lanes_k, encode_i32(lanes_k)),
        (prog.keys.lanes_k1, encode_f32(lanes_k1)),
    ];
    let bindings = resolve_bindings(&prog.bindings, &prog.host_inputs, &logits_positions, &submit)?;
    pass.sampler(&prog.program, bindings);

    // Device-resident window carryover: retain THIS fire's [seed, drafts]
    // (out[2]→row0, out[1]→rows1..k) into bravo's mtp_drafts buffer + inject into
    // the NEXT fire's input rows 0..=k (pipeline_source_kind=1). The next fire's
    // MtpDrafts intrinsic reads these drafts device-side.
    carrier::next_inputs_drafts(&pass, k);
    pass.execute();

    // PARTIAL-residency: read ONLY out[0] (the [k+1] commit tail) host-side — the
    // scalar n_acc (committed length) + the committed tokens. out[1]/out[2] are
    // NOT read (device-injected via the retain).
    let tensors = pass.outputs().await.map_err(|e| format!("outputs: {e}"))?;
    if tensors.is_empty() {
        return Err("expected the commit output".into());
    }
    read_i32(&tensors[0])
}

/// Fire-0 BOOTSTRAP: materialize `prompt + (k-1) fillers` — the mtp-native-verify
/// verify-window SHAPE — so the last k positions carry the k logit rows the
/// `[k,vocab]` MtpLogits matrix AND the k-row target matrix map their K rows onto.
/// (`mtp_specdecode_bootstrap`, no verify) → the first `[seed, drafts]`;
/// `carrier::next_inputs_drafts(k)` retains them for fire 1's device-resident
/// verify. Returns the seed (out[0] = row-0 target argmax = the first committed
/// token). A bare M=1 single-position fire made both intrinsics collapse onto the
/// anchor row (out[1]=anchor not draft; out[2]/seed=0) — the k-position fire gives
/// each matrix intrinsic its K real draft rows, matching the working verify path.
async fn bootstrap_fire(
    program: &tensor::Program,
    bindings: &[inferlet::sampling::ir::Binding],
    host_inputs: &[HostInputDecl],
    kv: &KvWorkingSet,
    rs: &RsWorkingSet,
    prompt: &[u32],
    k: u32,
) -> Result<i32> {
    // Verify-window shape: prompt + (k-1) fillers ⇒ the last k positions carry the
    // k draft rows. logits_positions = the last k positions (prompt-last..+k-1).
    let l = prompt.len() as u32;
    let mut input = prompt.to_vec();
    input.extend(core::iter::repeat(0u32).take((k - 1) as usize));
    let n = input.len() as u32;
    let page = kv.page_size();
    let total_pages = n.div_ceil(page);
    let have = kv.size();
    if total_pages > have {
        kv.alloc(total_pages - have).map_err(|e| format!("alloc: {e}"))?;
    }
    let pass = ForwardPass::new();
    pass.kv_working_set(kv, 0, 0, 0, 0, total_pages, 0);
    if rs.state_size() > 0 {
        pass.rs_working_set(rs, 0, n);
    }
    let positions: Vec<u32> = (0..n).collect();
    pass.input_tokens(&input, &positions);
    // k logit rows at the last k positions (prompt-last + k-1 fillers) — matches
    // mtp-native-verify. Row 0 (prompt-last) ⇒ the seed; the MTP matrix reads the k
    // MTP head rows. Both intrinsics get K real rows (no anchor-row collapse).
    let logits_positions: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();
    let resolved = resolve_bindings(bindings, host_inputs, &logits_positions, &[])?;
    pass.sampler(program, resolved);
    // Retain the first [seed, drafts] window for fire 1 (device-resident).
    carrier::next_inputs_drafts(&pass, k);
    pass.execute();
    let tensors = pass.outputs().await.map_err(|e| format!("bootstrap outputs: {e}"))?;
    let commit = read_i32(tensors.first().ok_or("bootstrap: no commit")?)?;
    commit.first().copied().ok_or_else(|| "bootstrap: empty seed".to_string())
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    let vocab = model::output_vocab_size();
    let prog = build(vocab, k)?;

    // Constant lane vectors (v4 has no iota — submit once per attach).
    let lanes_k: Vec<i32> = (0..k as i32).collect();
    let lanes_k1: Vec<f32> = (0..=k).map(|i| i as f32).collect();

    let mut prompt = model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    // The fire-0 bootstrap program (no verify — anchors the MTP head/target greedy
    // at the prompt's REAL last position; produces the first [seed, drafts]).
    let boot_built =
        mtp_specdecode_bootstrap(vocab, k).map_err(|e| format!("bootstrap build: {e:?}"))?;
    let boot_program =
        inferlet::emit::emit_program(&boot_built.program).map_err(|e| format!("emit: {e}"))?;

    let kv = KvWorkingSet::new();
    let rs = RsWorkingSet::new();
    let mut committed: Vec<u32> = prompt.clone();
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut generated: u32 = 0;

    // FIRE 0 (bootstrap): materialize the prompt, sample the first seed at its REAL
    // last position, and retain the first REAL drafts (MTP head @ prompt last) for
    // fire 1 — the fix for the zero-cascade charlie's A/B caught.
    let seed0 = bootstrap_fire(
        &boot_program,
        &boot_built.bindings,
        &boot_built.host_inputs,
        &kv,
        &rs,
        &prompt,
        k,
    )
    .await?;
    let mut seq_len: u32 = prompt.len() as u32;
    committed.push(seed0 as u32);
    generated += 1;

    // FIRE 1+ (device-resident): the [seed, drafts] window is DEVICE-INJECTED by the
    // carrier (retained from the previous fire). Host rows 0..=k are placeholders the
    // carrier overwrites; the MtpDrafts intrinsic serves the drafts to the verify.
    let mut window: Vec<u32> = core::iter::once(seed0 as u32)
        .chain(core::iter::repeat(0u32).take(k as usize))
        .collect();

    while generated < MAX_TOKENS {
        let commit = fire_window(&prog, &kv, &rs, &window, seq_len, &lanes_k, &lanes_k1).await?;

        let clen = committed_len(&commit); // n_acc accepted + 1 bonus (≥1)
        accepted_lengths.push(clen.saturating_sub(1));
        let commit_toks: Vec<u32> = commit.iter().take(clen).map(|&t| t as u32).collect();
        committed.extend(&commit_toks);
        generated += clen.max(1) as u32;

        // n_acc position-advance (partial-residency, host scalar): the fire wrote
        // n rows, but only `clen` are committed — advance the cursor by `clen`; the
        // rejected suffix's KV is reused by the next fire (the driver-side retain
        // lands the next window at seq_len + clen — bravo/charlie's commit-advance).
        seq_len += clen as u32;

        // The NEXT window is DEVICE-INJECTED by the carrier (`[seed, drafts]` from
        // the retained buffer). The guest submits host placeholders for rows 0..=k;
        // the carrier overwrites them pre-forward from the retain. Row 0 seeds from
        // the last committed token (for the geometry / non-carrier fallback).
        let seed = *committed.last().unwrap_or(&0);
        window = core::iter::once(seed)
            .chain(core::iter::repeat(0u32).take(k as usize))
            .collect();
    }

    let total_acc: usize = accepted_lengths.iter().sum();
    let steps = accepted_lengths.len();
    let mean_acc = if steps > 0 { total_acc as f64 / steps as f64 } else { 0.0 };
    let result = format!(
        "mtp-specdecode(device): k={k} steps={steps} accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} (device-resident: drafts via MtpDrafts intrinsic, \
         window via carrier::next_inputs_drafts, out[0]-only host read = partial-residency n_acc)",
        committed.len()
    );
    eprintln!("{result}");
    // Committed token dump for the (e) Metal↔CUDA cross-check (mac): the exact
    // device-resident spec-decode output sequence.
    eprintln!("[mtp-specdecode] committed[{}]={committed:?}", committed.len());
    Ok(result)
}
