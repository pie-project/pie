//! **PTIR multi-token PROMPT PREFILL — device REPRODUCER of the driver gap.**
//!
//! The classic `forward-pass` removal hinges on one capability that no ptir
//! inferlet has exercised: a **variable-length prompt prefill** (what the classic
//! `carrier::submit_pass` does — `input_tokens(prompt)` over N>1 tokens in one
//! pass). Every existing ptir inferlet (ptir-greedy-e2e, the migrated masks,
//! beam-designb) seeds a SINGLE token then decodes.
//!
//! FINDING (device-verified on the 4090, 2026-07-09): a naive N-wide prefill
//! fire on the ptir device-geometry path does NOT attend the prompt — the
//! continuation is incoherent/degenerate (the model behaves as if the prompt KV
//! is absent). Root cause is in the driver, NOT this inferlet:
//!   - The device-geometry AttnMask dense-pack (executor.cpp ~L2788) computes
//!     `lanes = qo_indptr.size() - 1` = number of SEQUENCES and packs ONE mask
//!     row per lane (`launch_pack_dense_mask(.., lanes, stride, ..)`). It is
//!     DECODE-SHAPED: it cannot express an `[N_query, KV]` prefill mask (a single
//!     sequence with N query rows collapses to lanes=1 + a garbled stride).
//!   - The explicit-KV-write descriptor (executor.cpp ~L2838) is "the single
//!     new-token K/V write" — ONE cell per lane, not the N cells a prefill writes.
//!   - Routing instead through the standard (no WSlot/WOff, pure-causal) path also
//!     fails: the KvLen→kv_last_page_lens read/write-range semantics for a fresh
//!     multi-query prefill are ambiguous on the ptir path (no classic
//!     inp_len/output_len/valid split), so the queries don't attend the written
//!     cells either.
//!
//! => Variable-length prompt prefill on the ptir path is UNBUILT at the driver
//! level. Closing it is a load-bearing contract call (how ptir expresses a
//! prefill: a genuine multi-query custom-mask pack + multi-cell write, or a
//! classic-style page-derived causal prefill with explicit read/write ranges) —
//! FLAGGED to the manager, not invented here. This inferlet + `cuda_ptir_prefill_e2e`
//! are the reproducer and the go-green target once that lands.
//!
//! Structure (the intended end-state it will prove): (1) ONE N-wide `ForwardPass`
//! embeds the whole prompt (`embed_indptr=[0,N]`, N-cell KV write, causal
//! `[N,POOL]` mask), read-out row N-1 = g0; (2) a windowed-style 1-token/pass
//! decode loop continues over the SAME pool pages.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const PAGE_T: u32 = 16; // tokens per pool page
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B
const POOL_PAGES: u32 = 24; // shared physical pool (prompt + decode headroom)
const POOL: u32 = POOL_PAGES * PAGE_T; // flat pool token positions (384)
const DECODE_STEPS: usize = 24;

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    // A REAL multi-token prompt — the whole point is N > 1 prefill. Raw factual
    // completion so a WORKING prefill has an unambiguous continuation ("Paris").
    let prompt: Vec<u32> = wit_model::encode("The capital of France is the city of");
    let n = prompt.len() as u32;
    if n < 2 {
        return Err(format!(
            "prompt must be multi-token to prove prefill (n={n})"
        ));
    }
    if n + DECODE_STEPS as u32 >= POOL {
        return Err(format!("prompt ({n}) + decode exceeds pool ({POOL})"));
    }
    println!("--- PTIR PREFILL e2e: prefilling n={n} prompt tokens in ONE fire ---");

    // Shared physical page pool (the KV store both pipelines bind).
    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let pool = ws
        .reserve(POOL_PAGES)
        .map_err(|e| format!("ws.reserve: {e}"))?;
    let pool_ids: &'static Vec<u32> = bx(pool.ids().to_vec()); // [POOL_PAGES] physical

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    // token_ids = the N prompt tokens (SEEDED → device-resident, exactly how a
    // seeded 1-token channel embeds on fire-0; seeding, not host `.put`, is the
    // path the runtime resolves for EmbedTokens). qo_indptr = [0, N] (one lane,
    // N query rows). positions default to [0..N]; read-out defaults to row N-1.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = bx(Channel::from(prompt_i32).named("toks_p")); // [N] i32
    let embed_indptr_p = Tensor::constant(vec![0u32, n]); // [2] qo_indptr

    // Explicit-write descriptor for the N prefill cells: cell c → physical page
    // pool_ids[c / PAGE_T] at offset c % PAGE_T.
    let w_slot_v: Vec<u32> = (0..n).map(|c| pool_ids[(c / PAGE_T) as usize]).collect();
    let w_off_v: Vec<u32> = (0..n).map(|c| c % PAGE_T).collect();
    let w_slot_p = bx(Channel::from(w_slot_v).named("w_slot_p")); // [N]
    let w_off_p = bx(Channel::from(w_off_v).named("w_off_p")); // [N]
    let klen_p = bx(Channel::from(vec![n; 1]).named("klen_p")); // [1] one seq, N kv len
    let pages_p = bx(Channel::from(pool_ids.clone()).named("pages_p")); // [POOL_PAGES]
    let page_indptr_p = bx(Channel::from_shaped([2], vec![0u32, POOL_PAGES]).named("pidx_p"));

    // Causal prefill mask [N, POOL]: query row i attends KV cols j with j <= i.
    let mask_pv: Vec<bool> = (0..n)
        .flat_map(|i| (0..POOL).map(move |j| j <= i))
        .collect();
    let mask_p = bx(Channel::from_shaped([n, POOL], mask_pv).named("mask_p"));
    let g0_ch = bx(Channel::new([1], dtype::i32).named("g0")); // host-read first gen token

    let fwd_p: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd_p.embed(toks_p, embed_indptr_p);
    fwd_p.attn_working_set(ws, klen_p);
    fwd_p.port_channel(Port::Pages, pages_p);
    fwd_p.port_channel(Port::PageIndptr, page_indptr_p);
    fwd_p.port_channel(Port::WSlot, w_slot_p);
    fwd_p.port_channel(Port::WOff, w_off_p);
    fwd_p.attn_mask(mask_p);
    fwd_p.epilogue(move || {
        // Read-out row (N-1) logits [1, V] → greedy next token.
        let tok = reduce_argmax(intrinsics::logits()); // [1] i32
        g0_ch.put(&tok);
    });

    let prefill = Pipeline::new();
    fwd_p
        .submit(&prefill)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];
    prefill.close();
    println!("[prefill] N-wide fire committed; first generated token g0={g0}");

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    // Seeded with g0 at position N, continuing over the SAME pool pages (so the
    // attention reads the prefill's KV cells 0..N). Full causal mask (attend all
    // filled positions). Device loop-carried exactly like windowed-attention.
    let phys_n = pool_ids[(n / PAGE_T) as usize];
    let tok_in = bx(Channel::from(vec![g0; 1]).named("tok_in"));
    let pos = bx(Channel::from(vec![n; 1]).named("pos"));
    let fill = bx(Channel::from(vec![n + 1; 1]).named("fill")); // next free after this fire writes cell N
    let klen = bx(Channel::from(vec![n + 1; 1]).named("klen")); // cells 0..=N present
    let w_slot = bx(Channel::from(vec![phys_n; 1]).named("w_slot"));
    let w_off = bx(Channel::from(vec![n % PAGE_T; 1]).named("w_off"));
    // Seed mask: the first decode query at position N attends 0..=N.
    let seed_mask: Vec<bool> = (0..POOL).map(|j| j <= n).collect();
    let mask = bx(Channel::from_shaped([1, POOL], seed_mask).named("mask"));
    let pages = bx(Channel::from(pool_ids.clone()).named("pages"));
    let page_indptr = bx(Channel::from_shaped([2], vec![0u32, POOL_PAGES]).named("page_indptr"));
    let pool_ids_ch = bx(Channel::new([POOL_PAGES], dtype::u32).named("pool_ids"));
    let out = bx(Channel::new([1], dtype::i32).named("out"));
    let lane1 = Tensor::constant(vec![0u32, 1u32]); // embed indptr [0,1]

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(tok_in, lane1);
    fwd.positions(pos);
    fwd.attn_working_set(ws, klen);
    fwd.port_channel(Port::Pages, pages);
    fwd.port_channel(Port::PageIndptr, page_indptr);
    fwd.port_channel(Port::WSlot, w_slot);
    fwd.port_channel(Port::WOff, w_off);
    fwd.attn_mask(mask);
    fwd.epilogue(move || {
        // Takes + compute first, PUTS last (value-id discipline).
        let base = fill.take().tensor(); // [1] u32 — position this next fire writes
        let pids = pool_ids_ch.take();

        let tok = reduce_argmax(intrinsics::logits()); // [1] i32

        // Full causal mask for the query at `base`: attend all j <= base.
        let col = iota(POOL);
        let base_b = broadcast(reshape(&base, [1]), [POOL]);
        let new_mask = reshape(le(&col, &base_b), [1, POOL]);

        let logical_slot = div(&base, PAGE_T);
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = rem(&base, PAGE_T);
        let klen_v = add(&base, 1u32);
        let next_free = add(&base, 1u32);
        let pages_v = reshape(&pids, [POOL_PAGES]);
        let pidx_v = mul(&iota(2), POOL_PAGES);

        tok_in.put(&tok);
        out.put(&tok);
        mask.put(&new_mask);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
    });

    let decode = Pipeline::new();
    let mut generated: Vec<u32> = Vec::new();
    generated.push(g0 as u32);
    for step in 0..DECODE_STEPS {
        pool_ids_ch.put(pool_ids.clone());
        fwd.submit(&decode)
            .map_err(|e| format!("decode submit @{step}: {e}"))?;
        let t = out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("out.take @{step}: {e}"))?;
        if let Some(&t0) = t.first() {
            generated.push(t0 as u32);
        }
    }
    decode.close();

    let distinct: std::collections::BTreeSet<u32> = generated.iter().copied().collect();
    let text = wit_model::decode(&generated).unwrap_or_default();
    let result = format!(
        "PTIR_PREFILL_E2E n={n} g0={g0} distinct={} tokens={generated:?} text={text:?}",
        distinct.len()
    );
    println!("{result}");
    if distinct.len() < 2 {
        return Err(format!(
            "degenerate continuation (all same token): {generated:?}"
        ));
    }
    Ok(result)
}
