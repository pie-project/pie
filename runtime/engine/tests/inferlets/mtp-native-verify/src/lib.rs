//! **MTP Stage 2 — PTIR-native explicit draft→verify→accept** (bravo), on the
//! `inferlet::ptir` bridge. The speculative decode is a single traced eDSL
//! epilogue per window — the target VERIFY (match + bonus tail) AND the next
//! window's DRAFTING (native MTP argmax) in ONE lowered graph:
//!
//!   `picked = logits[k+1, vocab].argmax()`         // [k+1] target greedy (k verify + bonus)
//!   `head   = gather(picked, lanes_k)`              // picked[0..k]
//!   `hit    = head.eq(draft)`                       // [k] bool — verify vs the EMBEDDED drafts
//!   `n_acc  = reduce_sum(cumprod(hit))`              // scalar accepted count 0..k
//!   `keep   = broadcast(n_acc).ge(lanes_k1)`          // [k+1] i <= n_acc
//!   commit  = select(keep, picked, -1)                // accepted prefix + BONUS@n_acc, then -1
//!   drafts' = argmax(mtp_logits(k))                   // [k] FRESH drafts — NEXT window's proposals
//!
//! `draft` is read DEVICE-ALIAS off the SAME embedded window tokens (`toks.read()`,
//! non-consuming peek — rows `1..=k`, the previous step's MTP proposals fed as
//! this window's input), not a separately submitted draft channel; each fresh
//! pass seeds its `toks` channel from the host's running commit/draft bookkeeping
//! (the ptir-bridge equivalent of the deleted
//! `sampling::program::mtp_native_verify` + `resolve_bindings` blob-submit
//! surface — there is no lower-level "device-resident retain" primitive on the
//! current `inferlet::ptir` surface, so drafts round-trip through host state
//! each iteration, exactly mirroring this inferlet's ORIGINAL host-round-trip
//! dataflow, contrasted with `mtp-specdecode`'s attempted device residency).
//!
//! Bootstrap fire: `prompt + (k-1)` fillers ⇒ the last k positions carry k
//! logit rows ⇒ both `logits` (target) and `mtp_logits` (drafts) get REAL k-row
//! data (no anchor-row collapse) — yields the seed (row-0 target argmax) + the
//! first REAL drafts (mtp argmax) for window 1.
//!
//! JSON/plain input: optional draft window `k` (default 4).

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const PROMPT: &str = "The quick brown fox jumps over";
const MAX_TOKENS: u32 = 16;
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Decode a `[k]`/`[k+1]` i32 host vector.
async fn get_i32(t: inferlet::ptir::Taken) -> Result<Vec<i32>> {
    t.get::<i32>()
        .await
        .map_err(|e| format!("tensor take: {e}"))
}

/// Committed length of a sentinel `[k+1]` tail = the count before the first
/// `-1` (accepted prefix + the bonus at lane `n_acc`), always ≥ 1.
fn committed_len(tail: &[i32]) -> usize {
    tail.iter().take_while(|&&t| t >= 0).count()
}

/// Bootstrap fire over `prompt + (k-1)` fillers: yields the seed (row-0 target
/// argmax at the prompt's REAL last position) + the first REAL `[k]` drafts
/// (native MTP argmax) for window 1. No verify (nothing to verify yet).
async fn bootstrap(
    ws: &'static WorkingSet,
    rs: &'static RsWorkingSet,
    prompt: &[u32],
    k: u32,
) -> Result<(i32, Vec<i32>)> {
    let l = prompt.len() as u32;
    let mut window: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    window.extend(std::iter::repeat(0i32).take((k - 1) as usize));
    let n = l + k - 1;

    let toks = bx(Channel::from(window).named("b_toks"));
    let klen = bx(Channel::from(vec![n]).named("b_klen"));
    let seed_out = bx(Channel::new([1], dtype::i32).named("b_seed"));
    let drafts_out = bx(Channel::new([k], dtype::i32).named("b_drafts"));

    // k read-out rows (the last k positions) ⇒ intrinsics::logits() AND
    // intrinsics::mtp_logits(k) both declare [k, vocab] — real k-row data.
    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(toks, Tensor::constant(vec![0u32, n]));
    fwd.attn_working_set(ws, klen);
    fwd.rs_working_set(rs);
    fwd.readout(&Tensor::constant(readout));
    fwd.epilogue(move || {
        let picked = reduce_argmax(intrinsics::logits()); // [k] target argmax
        let seed = gather(&picked, Tensor::constant(vec![0u32])); // [1] row-0
        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let drafts = reduce_argmax(mtp); // [k] fresh drafts
        seed_out.put(&seed);
        drafts_out.put(&drafts);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline)
        .map_err(|e| format!("bootstrap submit: {e}"))?;
    let seed = get_i32(seed_out.take())
        .await?
        .first()
        .copied()
        .ok_or_else(|| "bootstrap: empty seed".to_string())?;
    let drafts = get_i32(drafts_out.take()).await?;
    pipeline.close();
    Ok((seed, drafts))
}

/// One `[k+1]`-wide verify window: embed `[seed, draft]` (a freshly seeded
/// channel each iteration) at the explicit ABSOLUTE positions
/// `[seq_len .. seq_len+k+1)` (a fresh `pos` channel bound via
/// `ForwardPass::positions` — every fresh `verify_window` fire is its own new
/// `ForwardPass`, so a repeated implicit `0..k+1` default would misalign RoPE
/// against `klen`'s growing attended length), verify `draft` (device-alias
/// peeked off the SAME embedded tokens) against the target's per-row argmax,
/// and draft the NEXT window natively off `mtp_logits`. Returns `(commit
/// [k+1], next_drafts [k])`.
async fn verify_window(
    ws: &'static WorkingSet,
    rs: &'static RsWorkingSet,
    k: u32,
    seed: i32,
    draft: &[i32],
    seq_len: u32,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let kp1 = k + 1;
    let mut window: Vec<i32> = vec![seed];
    window.extend_from_slice(draft);

    let toks = bx(Channel::from(window).named("v_toks"));
    let pos = bx(Channel::from((seq_len..seq_len + kp1).collect::<Vec<u32>>()).named("v_pos"));
    let klen = bx(Channel::from(vec![seq_len + kp1]).named("v_klen"));
    let commit_out = bx(Channel::new([kp1], dtype::i32).named("v_commit"));
    let drafts_out = bx(Channel::new([k], dtype::i32).named("v_drafts"));

    let lanes: Vec<u32> = (0..=kp1).collect(); // one token per window row
    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(toks, Tensor::constant(lanes));
    fwd.positions(pos);
    fwd.attn_working_set(ws, klen);
    fwd.rs_working_set(rs);
    fwd.epilogue(move || {
        // Device-alias read: peek the embedded window (NOT a resubmitted draft
        // channel) and gather rows 1..=k as the verify operand.
        let win = toks.read().tensor(); // [k+1] i32
        let draft_v = gather(&win, Tensor::constant((1..=k).collect::<Vec<u32>>())); // [k]
        let picked = reduce_argmax(intrinsics::logits()); // [k+1] target (k verify + bonus)
        let head = gather(&picked, iota(k)); // [k] picked[0..k]
        let hit = eq(&head, &draft_v); // [k] bool
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        let run = cumprod(select(&hit, &ones, &zeros)); // [k]
        let n_acc = cast(reduce_sum(run), DType::U32); // accepted-prefix length
        let keep = ge(broadcast(&n_acc, [kp1]), iota(kp1)); // [k+1] i <= n_acc
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1); // accepted prefix + bonus + -1s

        let mtp = intrinsics::mtp_logits(k); // [k, vocab]
        let next_drafts = reduce_argmax(mtp); // [k] fresh drafts — NEXT window

        commit_out.put(&commit);
        drafts_out.put(&next_drafts);
    });

    let pipeline = Pipeline::new();
    fwd.submit(&pipeline)
        .map_err(|e| format!("verify submit: {e}"))?;
    let commit = get_i32(commit_out.take()).await?;
    let drafts = get_i32(drafts_out.take()).await?;
    pipeline.close();
    Ok((commit, drafts))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let k: u32 = input.trim().parse().unwrap_or(4).max(2);
    let vocab = wit_model::output_vocab_size();
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let rs: &'static RsWorkingSet = bx(RsWorkingSet::new());
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    model::configure_gates(
        /* has_mtp_logits */ true, /* has_value_head */ false,
    );

    let mut prompt = wit_model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Bootstrap: real seed + real first drafts off the prompt's REAL last position.
    let (seed0, draft0) = bootstrap(ws, rs, &prompt, k).await?;
    let mut seq_len: u32 = prompt.len() as u32 + k - 1;

    let mut committed: Vec<u32> = prompt.clone();
    committed.push(seed0 as u32);
    let mut seed = seed0;
    let mut draft = draft0;
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut generated: u32 = 1;

    // North-star spec-decode loop: verify the embedded drafts against the
    // target → commit the [k+1] tail (accepted + bonus) → take the fresh
    // native-MTP drafts as the NEXT window's proposals → repeat.
    while generated < MAX_TOKENS {
        let (commit, drafts) = verify_window(ws, rs, k, seed, &draft, seq_len).await?;
        seq_len += k + 1;

        let clen = committed_len(&commit); // n_acc accepted + 1 bonus (≥ 1)
        let n_acc = clen.saturating_sub(1);
        accepted_lengths.push(n_acc);
        let commit_toks: Vec<u32> = commit.iter().take(clen).map(|&t| t as u32).collect();
        committed.extend(&commit_toks);
        generated += clen.max(1) as u32;

        draft = drafts;
        seed = *committed.last().unwrap_or(&0) as i32;
    }

    let total_acc: usize = accepted_lengths.iter().sum();
    let steps = accepted_lengths.len();
    let mean_acc = if steps > 0 {
        total_acc as f64 / steps as f64
    } else {
        0.0
    };
    let result = format!(
        "mtp-native-verify: k={k} steps={steps} accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} (PTIR-native verify+draft: verify vs embedded \
         drafts, next-drafts from mtp_logits argmax, [k+1] bonus tail, all traced)",
        committed.len()
    );
    eprintln!("{result}");
    Ok(result)
}
