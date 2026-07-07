//! **MTP Stage 2 — PTIR-native explicit draft→verify→accept** (bravo), on DIRECT
//! WIT bindings. The north-star: the speculative decode is a single traced
//! sampling-IR program — the target VERIFY (match + bonus tail) AND the next
//! window's DRAFTING (native MTP argmax) in ONE lowered graph, not a
//! driver-hardwired black box.
//!
//! Program = `sampling::program::mtp_native_verify(vocab, k)` (echo's canonical
//! §6.1 golden `mtp_verify_tail` shape, pure v4 ops — lanes are host consts, no
//! iota/cast wire change):
//!   `picked = logits[k+1, vocab].argmax()`         // [k+1] target greedy (k verify + bonus)
//!   `head   = gather(picked, lanes_k)`             // picked[0..k] (host-const lane idx)
//!   `hit    = head.eq(draft)`                      // [k] bool — verify vs the EMBEDDED drafts
//!   `n_acc  = reduce_sum(cumprod(hit))`            // scalar accepted count 0..k
//!   `keep   = broadcast(n_acc).ge(lanes_k1_f32)`   // [k+1] i <= n_acc (f32 cmp, no cast)
//!   out[0]  = `select(keep, picked, -1)`           // [k+1] accepted prefix + BONUS@n_acc, then -1
//!   out[1]  = `argmax(mtp_logits[k])`              // [k] FRESH drafts — NEXT window's proposals
//!
//! Dataflow (echo's semantic correction): verify compares `picked` against the
//! drafts EMBEDDED in this window (`draft` host input — the PREVIOUS step's MTP
//! proposals, which the loop fed to `input-tokens`), NOT this step's mtp argmax.
//! This step's mtp argmax is out[1] = the NEXT window's drafts, fed back as
//! `draft`. Loop: submit(draft_t, lanes) → read commit + drafts_{t+1} → advance
//! by #non-sentinel (accepted + bonus, ≥1) → embed [seed, drafts_{t+1}] as the
//! next window. The bonus at lane `n_acc` rides the SAME `[k+1]` output, so a
//! full-reject step still advances by 1.
//!
//! Host inputs: `{draft [k] i32, lanes_k [k] i32 = [0..k-1], lanes_k1 [k+1] f32 =
//! [0..k]}` (lanes are per-attach constants — v4 has no vector-const/iota). Both
//! logits are INTRINSICS (`logits [k+1]` target + `mtp_logits [k]` proposals);
//! the driver computes both (target lm-head + the MTP head's K draft rows via
//! `ctx.mtp_draft_row`, charlie's Stage-2 driver half).
//!
//! Verified host-side against the golden committed-tail shape by
//! `sampling-edsl/tests/mtp.rs::mtp_native_verify_matches_the_golden_tail_contract`
//! (`[3,5,2,-1]` = 2 accepted + bonus + sentinel).
//!
//! ⚠️ RUNS after charlie's Stage-2 driver `mtp_logits` wiring (`ctx.mtp_draft_row`
//! + MTP head rows into `ws.logits`), post-§6.2. Host-COMPILES now (echo's
//! `intrinsic_mtp_logits_matrix_dyn` + the `MtpLogits` binding are in the base).
//! JSON/plain input: optional draft window `k` (default 4).

use inferlet::inference::ForwardPass;
use inferlet::program::{encode_f32, encode_i32, resolve_bindings, HostInputDecl};
use inferlet::sampling::program::{mtp_native_verify, MtpNativeVerifyKeys};
use inferlet::working_set::{KvWorkingSet, RsWorkingSet};
use inferlet::{Result, model, tensor};

const PROMPT: &str = "The quick brown fox jumps over";
const MAX_TOKENS: u32 = 16;

/// The lowered `mtp_native_verify` program + its host-input keys. Outputs:
/// `[0]` = `[k+1]`-Token committed tail (accepted + bonus@n_acc + -1);
/// `[1]` = `[k]`-Token fresh drafts (next window's proposals).
struct Composed {
    program: tensor::Program,
    bindings: Vec<inferlet::sampling::ir::Binding>,
    host_inputs: Vec<HostInputDecl>,
    keys: MtpNativeVerifyKeys,
    k: u32,
}

/// Build echo's canonical `mtp_native_verify` program (`[k+1]` golden tail +
/// next-drafts output) and lower it to the reusable WIT `tensor::Program`.
fn build(vocab: u32, k: u32) -> Result<Composed> {
    let (built, keys) =
        mtp_native_verify(vocab, k).map_err(|e| format!("compose build: {e:?}"))?;
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

/// Fire ONE verify window over `window` and read echo's two outputs: the
/// sentinel-coded `[k+1]` committed tail (accepted + bonus + `-1`s) and the `[k]`
/// FRESH drafts (next window's proposals). Submits the embedded `draft` + the two
/// constant lane vectors; the driver serves `logits` (k+1 rows) + `mtp_logits`.
async fn verify_window(
    prog: &Composed,
    kv: &KvWorkingSet,
    rs: &RsWorkingSet,
    window: &[u32],
    seq_len: u32,
    draft: &[i32],
    lanes_k: &[i32],
    lanes_k1: &[f32],
) -> Result<(Vec<i32>, Vec<i32>)> {
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
    // Qwen3.5 is a GDN/hybrid model: its linear-attention layers need a bound
    // RsWorkingSet (the recurrent-state slots) or the driver rejects the fire
    // ("rs_cache forward missing runtime-assigned slot ids"). Mirror generate-gdn:
    // bind the RS working set as the in-forward-write signal when the model has
    // recurrent state. (charlie: added to unblock the mtp_logits GPU value-verify.)
    if rs.state_size() > 0 {
        pass.rs_working_set(rs, 0, n);
    }
    let positions: Vec<u32> = (seq_len..seq_len + n).collect();
    pass.input_tokens(window, &positions);

    // The k+1 verify+bonus positions = the last k+1 rows of this window. The driver
    // serves `logits` there and `mtp_logits` at the MTP draft rows (Stage-2 driver
    // half). `saturating_sub` keeps the skeleton total-safe for short windows.
    let base = (seq_len + n).saturating_sub(k + 1);
    let logits_positions: Vec<u32> = (0..k + 1).map(|i| base + i).collect();
    // Host inputs: the embedded drafts + the two constant lane vectors.
    let submit = vec![
        (prog.keys.draft, encode_i32(draft)),
        (prog.keys.lanes_k, encode_i32(lanes_k)),
        (prog.keys.lanes_k1, encode_f32(lanes_k1)),
    ];
    let bindings = resolve_bindings(&prog.bindings, &prog.host_inputs, &logits_positions, &submit)?;
    pass.sampler(&prog.program, bindings);
    pass.execute();

    // Two declared outputs (attach then output-slot order): [0]=commit, [1]=drafts.
    let tensors = pass.outputs().await.map_err(|e| format!("outputs: {e}"))?;
    if tensors.len() < 2 {
        return Err(format!("expected 2 outputs (commit, drafts), got {}", tensors.len()));
    }
    Ok((read_i32(&tensors[0])?, read_i32(&tensors[1])?))
}

/// Decode a `[k]`/`[k+1]` i32 tensor's little-endian bytes into token ids.
fn read_i32(t: &tensor::Tensor) -> Result<Vec<i32>> {
    let bytes = t.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Committed length of a sentinel `[k+1]` tail = the count of tokens before the
/// first `-1` (accepted prefix + the bonus at lane `n_acc`), always ≥ 1.
fn committed_len(tail: &[i32]) -> usize {
    tail.iter().take_while(|&&t| t >= 0).count()
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    let vocab = model::output_vocab_size();
    let prog = build(vocab, k)?;

    let mut prompt = model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Constant lane vectors (v4 has no vector-const/iota — submit once per attach).
    let lanes_k: Vec<i32> = (0..k as i32).collect(); // [0, 1, .., k-1]
    let lanes_k1: Vec<f32> = (0..=k).map(|i| i as f32).collect(); // [0.0, .., k.0]

    let kv = KvWorkingSet::new();
    let rs = RsWorkingSet::new();
    let mut committed: Vec<u32> = prompt.clone();
    let mut seq_len: u32 = 0;
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut generated: u32 = 0;

    // Bootstrap: no prior MTP proposals yet → a placeholder draft that fully
    // rejects (—1 never matches a token id ≥ 0), so the first window commits just
    // the target's bonus token and out[1] yields the first REAL drafts.
    let mut draft: Vec<i32> = vec![-1; k as usize];
    let mut pending: Vec<u32> = {
        let mut w = prompt.clone();
        w.extend(draft.iter().map(|&t| t.max(0) as u32)); // embed k placeholder drafts
        w
    };

    // North-star spec-decode loop: verify the embedded drafts (draft_t) against the
    // target → commit the [k+1] tail (accepted + bonus) → take out[1] as the NEXT
    // window's drafts (drafts_{t+1}) → embed [seed, drafts_{t+1}] → repeat.
    while generated < MAX_TOKENS {
        let n = pending.len() as u32;
        let (commit, drf) =
            verify_window(&prog, &kv, &rs, &pending, seq_len, &draft, &lanes_k, &lanes_k1).await?;
        seq_len += n;

        let clen = committed_len(&commit); // n_acc accepted + 1 bonus (≥ 1)
        let n_acc = clen.saturating_sub(1); // accepted-DRAFT count for the metric
        accepted_lengths.push(n_acc);
        // Commit the whole [k+1] tail before the first -1: accepted prefix AND the
        // bonus at lane n_acc (echo's §6.1 contract — a full-reject step still
        // advances by 1, never stalls).
        let commit_toks: Vec<u32> = commit.iter().take(clen).map(|&t| t as u32).collect();
        committed.extend(&commit_toks);
        generated += clen.max(1) as u32;

        // Feed THIS step's fresh drafts (out[1]) forward as the next verify's
        // embedded drafts, seeded from the last committed token.
        draft = drf;
        let seed = *committed.last().unwrap_or(&0);
        pending = core::iter::once(seed)
            .chain(draft.iter().map(|&t| t.max(0) as u32))
            .collect();
    }

    let total_acc: usize = accepted_lengths.iter().sum();
    let steps = accepted_lengths.len();
    let mean_acc = if steps > 0 { total_acc as f64 / steps as f64 } else { 0.0 };
    let result = format!(
        "mtp-native-verify: k={k} steps={steps} accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} (PTIR-native verify+draft: verify vs embedded \
         drafts, next-drafts from mtp_logits argmax, [k+1] bonus tail, all traced)",
        committed.len()
    );
    inferlet_eprintln(&result);
    Ok(result)
}

fn inferlet_eprintln(s: &str) {
    eprintln!("{s}");
}
