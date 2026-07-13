//! **MTP Stage-2 — spec-decode, drafts-channel swap of `mtp-native-verify`**
//! (`inferlet::ptir` bridge rewrite). The ORIGINAL classic-WIT design read this
//! window's drafts DEVICE-RESIDENT via a `Binding::MtpDrafts` intrinsic + a
//! driver-side `carrier::next_inputs_drafts` retain/inject command (zero host
//! round-trip on the `[k]` drafts). Both that intrinsic's author-facing eDSL
//! wrapper and the retain/inject WIT command have been REMOVED in the ptir
//! refactor (there is no `intrinsics::mtp_drafts()` in `ptir-dsl`, and no
//! `pipeline_source_kind`/retain surface anywhere in the current `forward`/
//! `pipeline` WIT interfaces) — this is a genuine capability gap, not a stale
//! rename, so it cannot be surgically restored without extending the WIT/driver
//! surface (out of this migration's scope).
//!
//! This inferlet is therefore, for now, IMPLEMENTATION-IDENTICAL to
//! `mtp-native-verify`'s host-round-trip baseline (see that inferlet's module
//! docs for the full dataflow): the `[k+1]` window is a host-writer channel
//! re-`put` each iteration from the host's running commit/draft bookkeeping,
//! and the draft is device-alias read (`toks.read()`, non-consuming peek) off
//! the SAME embedded window tokens. `cuda_mtp_specdecode_ab.rs`'s hard gate is
//! only that both A and B return a finite `mean_accept` and their own name in
//! the result string — both hold here. The PERF/VALUE A-vs-B commentary in that
//! harness (device-resident should avoid B's host round-trip and be ≥ as fast)
//! is no longer a meaningful distinction until `MtpDrafts`/retain functionality
//! is reintroduced on the `inferlet::ptir` surface; that harness is `#[ignore]`d
//! and only soft-prints the delta, so this does not regress any hard assertion.
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
/// (native MTP argmax) for window 1.
async fn bootstrap(ws: &'static WorkingSet, prompt: &[u32], k: u32) -> Result<(i32, Vec<i32>)> {
    let l = prompt.len() as u32;
    let mut window: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    window.extend(std::iter::repeat(0i32).take((k - 1) as usize));
    let n = l + k - 1;

    let toks = bx(Channel::from(window).named("b_toks"));
    let klen = bx(Channel::from(vec![n]).named("b_klen"));
    let seed_out = bx(Channel::new([1], dtype::i32).named("b_seed"));
    let drafts_out = bx(Channel::new([k], dtype::i32).named("b_drafts"));

    let readout: Vec<u32> = (0..k).map(|i| l - 1 + i).collect();

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(toks, Tensor::constant(vec![0u32, n]));
    fwd.attn_working_set(ws, klen);
    fwd.readout(&Tensor::constant(readout));
    fwd.epilogue(move || {
        let picked = reduce_argmax(intrinsics::logits());
        let seed = gather(&picked, Tensor::constant(vec![0u32]));
        let mtp = intrinsics::mtp_logits(k);
        let drafts = reduce_argmax(mtp);
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

/// One `[k+1]`-wide verify window: embed `[seed, draft]` at the explicit
/// ABSOLUTE positions `[seq_len .. seq_len+k+1)` (a fresh `pos` channel bound
/// via `ForwardPass::positions` — every fresh `verify_window` fire is its own
/// new `ForwardPass`, so a repeated implicit `0..k+1` default would misalign
/// RoPE against `klen`'s growing attended length), verify `draft`
/// (device-alias peeked off the SAME embedded tokens) against the target's
/// per-row argmax, and draft the NEXT window natively off `mtp_logits`.
/// Returns `(commit [k+1], next_drafts [k])`.
async fn verify_window(
    ws: &'static WorkingSet,
    k: u32,
    seed: i32,
    draft: &[i32],
    seq_len: u32,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let kp1 = k + 1;
    let mut window: Vec<i32> = vec![seed];
    window.extend_from_slice(draft);

    let toks = bx(Channel::new([kp1], dtype::i32).named("v_toks"));
    let pos = bx(Channel::new([kp1], dtype::u32).named("v_pos"));
    let klen = bx(Channel::new([1], dtype::u32).named("v_klen"));
    let commit_out = bx(Channel::new([kp1], dtype::i32).named("v_commit"));
    let drafts_out = bx(Channel::new([k], dtype::i32).named("v_drafts"));

    let lanes: Vec<u32> = (0..=kp1).collect();
    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(toks, Tensor::constant(lanes));
    fwd.positions(pos);
    fwd.attn_working_set(ws, klen);
    fwd.epilogue(move || {
        let win = toks.read().tensor(); // [k+1] i32 device-alias peek
        let draft_v = gather(&win, Tensor::constant((1..=k).collect::<Vec<u32>>()));
        let picked = reduce_argmax(intrinsics::logits()); // [k+1]
        let head = gather(&picked, iota(k));
        let hit = eq(&head, &draft_v);
        let ones = broadcast(Tensor::constant(1.0f32), [k]);
        let zeros = broadcast(Tensor::constant(0.0f32), [k]);
        let run = cumprod(select(&hit, &ones, &zeros));
        let n_acc = cast(reduce_sum(run), DType::U32);
        let keep = ge(broadcast(&n_acc, [kp1]), iota(kp1));
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1);

        let mtp = intrinsics::mtp_logits(k);
        let next_drafts = reduce_argmax(mtp);

        commit_out.put(&commit);
        drafts_out.put(&next_drafts);
    });

    toks.put(window);
    pos.put((seq_len..seq_len + kp1).collect::<Vec<u32>>());
    klen.put(vec![seq_len + kp1]);
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
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    model::configure_gates(
        /* has_mtp_logits */ true, /* has_value_head */ false,
    );

    let mut prompt = wit_model::encode(PROMPT);
    if prompt.is_empty() {
        prompt.push(0);
    }

    let (seed0, draft0) = bootstrap(ws, &prompt, k).await?;
    let mut seq_len: u32 = prompt.len() as u32 + k - 1;

    let mut committed: Vec<u32> = prompt.clone();
    committed.push(seed0 as u32);
    let mut seed = seed0;
    let mut draft = draft0;
    let mut accepted_lengths: Vec<usize> = Vec::new();
    let mut generated: u32 = 1;

    while generated < MAX_TOKENS {
        let (commit, drafts) = verify_window(ws, k, seed, &draft, seq_len).await?;
        seq_len += k + 1;

        let clen = committed_len(&commit);
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
        "mtp-specdecode: k={k} steps={steps} accepted_lengths={accepted_lengths:?} \
         mean_accept={mean_acc:.2} committed={} (host round-trip drafts — the device-resident \
         MtpDrafts/carrier-retain path is unavailable on the current inferlet::ptir surface; \
         see the module docs)",
        committed.len()
    );
    eprintln!("{result}");
    eprintln!(
        "[mtp-specdecode] committed[{}]={committed:?}",
        committed.len()
    );
    Ok(result)
}
