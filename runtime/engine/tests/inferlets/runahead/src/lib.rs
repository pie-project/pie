//! 1a run-ahead carryover verify — **PTIR rewrite** (bravo). Reimplements the
//! deleted classic-surface probe (`inference::ForwardPass` + `next-inputs`
//! device carrier + Sampling-IR greedy program) on `inferlet::ptir`: the
//! run-ahead carrier IS the ptir wire form — a loop-carried `tok_in` channel
//! (each fire's epilogue puts its greedy sample, the next fire's `embed` takes
//! it, device-side; no host round-trip for the token) and a `Pipeline` whose
//! `submit` never blocks, so the host submits fire `t+k` before taking fire
//! `t`'s output. Depth-`k` run-ahead = `k` submits ahead of the FIFO drain,
//! with the host-read `out` ring widened to `k` cells.
//!
//! Wire form per generate: (1) ONE N-wide PREFILL fire at the context cursor
//! (explicit positions + N-cell write descriptor + causal `[N, POOL]` mask,
//! greedy read-out row = token #1); (2) a loop-carried 1-wide DECODE pass
//! (the `ptir-prefill-e2e` / windowed-attention idiom: geometry + mask evolve
//! in-graph) submitted `depth` ahead. The prefill→decode handoff is a host
//! take/seed (`g0`), so unlike the classic carrier there is NO dangling carry
//! to clear on `fresh-generate` — the #26 probe's verdict (`CLEAR_OK`) now
//! proves clean multi-generate reuse of one context/pool.
//!
//!  1. **Scenario A** (`MATCH` + `ANCHOR_OK`): the depth-2 run-ahead stream
//!     equals the synchronous stream AND positively equals the verified
//!     milestone tokens.
//!  2. **Scenario B** (`CLEAR_OK`): gen-2 on a context whose gen-1 stopped on
//!     its first token equals the sync-gen-1 reference context's gen-2.
//!  3. **Scenario C** (`DEEP_MATCH` / `DEEP4_MATCH`): the depth-`k`
//!     submit-ahead chain (k=2 asserted, k=4 observed — run with
//!     `PIE_SCHED_MAX_IN_FLIGHT=4` to exercise true 4-in-flight) is
//!     byte-identical to the synchronous stream.
//!  4. **Scenario D** (`DEEP_STOP_MATCH`): the depth-`k` EOS-rollback —
//!     over-shoot a mid-stream stop by ≤`depth`−1 fires, DRAIN (finalize, not
//!     drop) the over-shot, roll the cursor back — byte-identical to the
//!     synchronous stop stream.
//!  5. **Scenario E** (`DEEP_STOP_CLEAR`): after a deep-stop rollback the SAME
//!     context runs a clean gen-2 (the over-shot KV cells are overwritten by
//!     the next generate's window) equal to the sync-gen-1 reference.
//!
//! Mock note: the harness gate is `#[ignore]`d — token identity across
//! separate program instances is a REAL-DRIVER property (the eval-mock seeds
//! logits per instance/epoch, so distinct pipelines legitimately diverge).
//!
//! JSON/plain input: an optional token budget (defaults to 8), e.g. `"16"`.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const PROMPT: &str = "hello world";
/// Fixed 2nd-turn continuation for the clear probes (Scenarios B/E).
const CONT: &str = " Tell me more.";
const PAGE_T: u32 = 16; // tokens per pool page
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B

/// The known-good greedy decode of `PROMPT` on qwen3-0.6b — the verified #6/#21
/// milestone stream (argmax == greedy at temperature 0).
const MILESTONE: [u32; 8] = [198, 9707, 1879, 374, 264, 4285, 2025, 429];

/// Ring capacity for the host-facing decode channels (`out`, `pool_ids`):
/// must cover the deepest submit-ahead window.
const RING: u32 = 8;

fn anchor_ok(tokens: &[u32]) -> bool {
    match tokens.len() {
        0 => false,
        n if n <= MILESTONE.len() => tokens == &MILESTONE[..n],
        _ => tokens[..MILESTONE.len()] == MILESTONE && tokens.iter().any(|&t| t != 0),
    }
}

/// One decode context: its own working-set pool + host cursor. The cursor is
/// the host MIRROR of the device-carried `fill` position (advanced on SUBMIT,
/// exactly like the classic probe advanced `seq_len` on submit) — it prices
/// the pool, places continuation prefills, and rolls back on a deep-stop.
struct Decoder {
    ws: WorkingSet,
    pool_ids: Vec<u32>,
    pool_pages: u32,
    pool: u32, // pool token positions
    seq: u32,  // committed cursor (host mirror)
}

impl Decoder {
    fn new(capacity_tokens: u32) -> Result<Decoder> {
        let pool_pages = capacity_tokens.div_ceil(PAGE_T).max(1);
        let ws = WorkingSet::new();
        let grant = ws
            .reserve(pool_pages)
            .map_err(|e| format!("ws.reserve: {e}"))?;
        let pool_ids = grant.ids().to_vec();
        Ok(Decoder {
            ws,
            pool_ids,
            pool_pages,
            pool: pool_pages * PAGE_T,
            seq: 0,
        })
    }

    /// ONE N-wide prefill fire at the cursor: embeds `tokens` at positions
    /// `seq..seq+n`, writes their N KV cells, greedy-argmaxes the read-out row
    /// (row N-1). Advances the cursor by N and returns the generated token.
    async fn prefill(&mut self, tokens: &[u32], pipeline: &Pipeline) -> Result<u32> {
        let n = tokens.len() as u32;
        let base = self.seq;
        if base + n >= self.pool {
            return Err(format!(
                "prefill overflows pool ({} + {n} >= {})",
                base, self.pool
            ));
        }

        let toks_v: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let toks = Channel::from(toks_v).named("toks_p"); // [N] i32 (seeded)
        let pos_v: Vec<u32> = (base..base + n).collect();
        let pos = Channel::from(pos_v).named("pos_p");
        // Explicit N-cell write descriptor: cell c → pool_ids[c/PAGE_T] @ c%PAGE_T.
        let w_slot_v: Vec<u32> = (base..base + n)
            .map(|c| self.pool_ids[(c / PAGE_T) as usize])
            .collect();
        let w_off_v: Vec<u32> = (base..base + n).map(|c| c % PAGE_T).collect();
        let w_slot = Channel::from(w_slot_v).named("w_slot_p");
        let w_off = Channel::from(w_off_v).named("w_off_p");
        let klen = Channel::from(vec![base + n; 1]).named("klen_p");
        let pages = Channel::from(self.pool_ids.clone()).named("pages_p");
        let page_indptr = Channel::from_shaped([2], vec![0u32, self.pool_pages]).named("pidx_p");
        // Causal mask [N, POOL]: query row i (abs pos base+i) attends j <= base+i.
        let mask_v: Vec<bool> = (0..n)
            .flat_map(|i| (0..self.pool).map(move |j| j <= base + i))
            .collect();
        let mask = Channel::from_shaped([n, self.pool], mask_v).named("mask_p");
        let g_ch = Channel::new([1], dtype::i32).named("g0");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, Tensor::constant(vec![0u32, n]));
        fwd.positions(&pos);
        fwd.port_channel(Port::KvLen, &klen);
        fwd.attn_working_set(&self.ws, .., (base / self.ws.page_size())..)?;
        fwd.derive_dense_geometry();
        fwd.port_channel(Port::Pages, &pages);
        fwd.port_channel(Port::PageIndptr, &page_indptr);
        fwd.port_channel(Port::WSlot, &w_slot);
        fwd.port_channel(Port::WOff, &w_off);
        fwd.attn_mask(&mask);
        fwd.epilogue(move || {
            let tok = reshape(reduce_argmax(intrinsics::logits()), [1]); // [1] i32
            g_ch.put(&tok);
        });

        fwd.submit(&pipeline)
            .map_err(|e| format!("prefill submit: {e}"))?;
        self.seq += n;
        let g0 = g_ch
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("g0 take: {e}"))?[0];
        Ok(g0 as u32)
    }

    /// Build the loop-carried 1-wide decode pass seeded with `g0` at the
    /// cursor. Geometry + mask evolve in-graph (the device carrier); the host
    /// only feeds `pool_ids` per fire and drains `out` — so submits can run
    /// `depth` ahead of the drain.
    fn decode_pass(&self, g0: u32, pipeline: Pipeline) -> Result<DecodeLoop> {
        let n = self.seq;
        let pool = self.pool;
        let pool_pages = self.pool_pages;
        let phys_n = self.pool_ids[(n / PAGE_T) as usize];

        let tok_in = Channel::from(vec![g0 as i32; 1]).named("tok_in");
        let pos = Channel::from(vec![n; 1]).named("pos");
        let fill = Channel::from(vec![n + 1; 1]).named("fill");
        let klen = Channel::from(vec![n + 1; 1]).named("klen");
        let w_slot = Channel::from(vec![phys_n; 1]).named("w_slot");
        let w_off = Channel::from(vec![n % PAGE_T; 1]).named("w_off");
        let seed_mask: Vec<bool> = (0..pool).map(|j| j <= n).collect();
        let mask = Channel::from_shaped([1, pool], seed_mask).named("mask");
        let pages = Channel::from(self.pool_ids.clone()).named("pages");
        let page_indptr = Channel::from_shaped([2], vec![0u32, pool_pages]).named("page_indptr");
        let pool_ids_ch = Channel::new([pool_pages], dtype::u32)
            .capacity(RING)
            .named("pool_ids");
        let out = Channel::new([1], dtype::i32).capacity(RING).named("out");
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, lane1);
        fwd.positions(&pos);
        fwd.port_channel(Port::KvLen, &klen);
        fwd.attn_working_set(&self.ws, .., (n / self.ws.page_size())..)?;
        fwd.derive_dense_geometry();
        fwd.port_channel(Port::Pages, &pages);
        fwd.port_channel(Port::PageIndptr, &page_indptr);
        fwd.port_channel(Port::WSlot, &w_slot);
        fwd.port_channel(Port::WOff, &w_off);
        fwd.attn_mask(&mask);
        fwd.epilogue(move || {
            // Takes + compute first, puts last (value-id discipline).
            let base = fill.take().tensor(); // [1] u32 — position this next fire writes
            let pids = pool_ids_ch.take();

            let tok = reshape(reduce_argmax(intrinsics::logits()), [1]); // [1] i32

            // Full causal mask for the query at `base`: attend all j <= base.
            let col = iota(pool);
            let base_b = broadcast(reshape(&base, [1]), [pool]);
            let new_mask = reshape(le(&col, &base_b), [1, pool]);

            let logical_slot = div(&base, PAGE_T);
            let w_slot_v = gather(&pids, &logical_slot);
            let w_off_v = rem(&base, PAGE_T);
            let klen_v = add(&base, 1u32);
            let next_free = add(&base, 1u32);
            let pages_v = reshape(&pids, [pool_pages]);
            let pidx_v = mul(&iota(2), pool_pages);

            tok_in.put(&tok);
            out.put(&tok);
            mask.take();
            mask.put(&new_mask);
            w_slot.put(&w_slot_v);
            w_off.put(&w_off_v);
            klen.take();
            klen.put(&klen_v);
            pos.put(&base);
            fill.put(&next_free);
            pages.take();
            pages.put(&pages_v);
            page_indptr.take();
            page_indptr.put(&pidx_v);
        });

        Ok(DecodeLoop {
            fwd,
            out,
            pool_ids_ch,
            pipeline,
        })
    }
}

/// A live decode chain: submit-ahead fires + FIFO drain over the `out` ring.
struct DecodeLoop {
    fwd: ForwardPass,
    out: Channel,
    pool_ids_ch: Channel,
    pipeline: Pipeline,
}

impl DecodeLoop {
    /// Submit one fire run-ahead (never blocks on the fire's result). Advances
    /// the decoder cursor on SUBMIT, like the classic probe.
    fn submit(&self, d: &mut Decoder) -> Result<()> {
        d.pool_ids_ch_put(&self.pool_ids_ch);
        self.fwd
            .submit(&self.pipeline)
            .map_err(|e| format!("decode submit: {e}"))?;
        d.seq += 1;
        Ok(())
    }

    /// Drain the oldest in-flight fire's token (blocks until committed).
    async fn take(&self) -> Result<u32> {
        let t = self
            .out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("out.take: {e}"))?;
        Ok(*t.first().unwrap_or(&0) as u32)
    }

    fn close(self) {
        self.pipeline.close();
    }
}

impl Decoder {
    fn pool_ids_ch_put(&self, ch: &Channel) {
        ch.put(self.pool_ids.clone());
    }
}

/// Greedy decode with a submit-ahead window of `depth` fires.
///
///  - `depth == 1` — the synchronous reference (build → submit → drain per
///    token, the classic `decode_sync`).
///  - `depth >= 2`, no stop — the run-ahead chain (`decode_pipelined` /
///    `decode_pipelined_deep`): fill the window upfront, FIFO drain + refill.
///  - stop set + `!overshoot` — decline to speculate past a possible stop
///    (the classic shallow bounded-speculation path): effective depth 1.
///  - stop set + `overshoot` — the depth-`k` EOS-rollback
///    (`decode_pipelined_deep_stop`): keep the window full THROUGH the stop
///    boundary; on the stop, DRAIN the ≤`depth`−1 over-shot fires (finalize,
///    not drop — the #17 arm-2 discipline) and roll the cursor back to the
///    committed prefix. The over-shot KV cells are overwritten by the next
///    generate's window.
async fn generate(
    d: &mut Decoder,
    prompt: &[u32],
    max_tokens: usize,
    stop: &[u32],
    depth: usize,
    overshoot: bool,
) -> Result<Vec<u32>> {
    if max_tokens == 0 {
        return Ok(Vec::new());
    }
    let fallback = [0u32];
    let prompt = if prompt.is_empty() {
        &fallback[..]
    } else {
        prompt
    };

    // ONE pipeline per generate (R4-4): fire #1 — the N-wide prompt prefill
    // (token #1 of the budget) — and every decode fire submit here; the
    // DecodeLoop takes it over and closes it after the final drain.
    let pipeline = Pipeline::new();
    let g0 = d.prefill(prompt, &pipeline).await?;
    let mut out = Vec::with_capacity(max_tokens);
    if stop.contains(&g0) {
        return Ok(out); // stopped on the first token; nothing decoded
    }
    out.push(g0);
    if max_tokens == 1 {
        return Ok(out);
    }

    let depth = if !stop.is_empty() && !overshoot {
        1
    } else {
        depth.max(1)
    };
    let budget = max_tokens - 1; // decode fires (each yields one token)

    let dl = d.decode_pass(g0, pipeline)?;
    let mut submitted = 0usize;
    let mut inflight = 0usize;

    // Prime + fill: launch up to `depth` chain-linked fires upfront, NONE
    // awaited (with `overshoot` this deliberately runs past a possible stop).
    while inflight < depth && submitted < budget {
        dl.submit(d)?;
        submitted += 1;
        inflight += 1;
    }

    // FIFO drain; refill one fire per drained token to sustain `depth` in flight.
    let mut hit_stop = false;
    while inflight > 0 {
        let token = dl.take().await?;
        inflight -= 1;
        if stop.contains(&token) {
            hit_stop = true; // the stop fire committed; `inflight` are over-shot
            break;
        }
        out.push(token);
        if submitted < budget {
            dl.submit(d)?;
            submitted += 1;
            inflight += 1;
        }
    }

    if hit_stop {
        // Depth-k rollback: DRAIN each over-shot fire (finalize via `take`,
        // ignoring the token) rather than drop it mid-flight, then roll the
        // cursor back to the committed prefix. The next generate's prefill
        // reuses/overwrites the over-shot cells.
        let overshot = inflight as u32;
        while inflight > 0 {
            let _ = dl.take().await; // finalize-drain; a discard never fails the decode
            inflight -= 1;
        }
        d.seq = d.seq.saturating_sub(overshot);
    }
    dl.close();
    Ok(out)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(8);
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    let prompt = wit_model::encode(PROMPT);
    let prompt = if prompt.is_empty() {
        vec![0u32]
    } else {
        prompt
    };
    let cont = wit_model::encode(CONT);

    // Deep-scenario budget bound (#17/#18): the deep scenarios test the chain
    // MECHANISM, which needs only the stop region — and byte-identity to the
    // sync reference is only reliable within a short window at high co-batch
    // density (concurrent-decode numeric non-invariance diverges long streams).
    let deep_budget = max_tokens.min(16);

    // One pool sizing covers every scenario context (gen-1 + gen-2 + margin).
    let cap =
        (prompt.len() + cont.len()) as u32 + 2 * (max_tokens.max(deep_budget) as u32) + PAGE_T;

    // ── Scenario A — run-ahead token-exactness: depth-2 == synchronous ──
    let mut d_p = Decoder::new(cap)?;
    let tokens_p = generate(&mut d_p, &prompt, max_tokens, &[], 2, false).await?;
    let mut d_s = Decoder::new(cap)?;
    let tokens_s = generate(&mut d_s, &prompt, max_tokens, &[], 1, false).await?;
    let matched = tokens_p == tokens_s;

    // ── Scenario C — DEEP chain byte-identity: depth-k submit-ahead == sync ──
    //  - depth=2 (`DEEP_MATCH`): asserted by the harness.
    //  - depth=4 (`DEEP4_MATCH`, observed, not asserted): run with
    //    `PIE_SCHED_MAX_IN_FLIGHT=4` to exercise true 4-in-flight.
    let mut d_d2 = Decoder::new(cap)?;
    let tokens_d2 = generate(&mut d_d2, &prompt, deep_budget, &[], 2, false).await?;
    let deep_matched = tokens_d2.as_slice() == &tokens_s[..deep_budget.min(tokens_s.len())];
    let mut d_d4 = Decoder::new(cap)?;
    let tokens_d4 = generate(&mut d_d4, &prompt, deep_budget, &[], 4, false).await?;
    let deep4_matched = tokens_d4.as_slice() == &tokens_s[..deep_budget.min(tokens_s.len())];

    // ── Scenario D — DEEP STOP rollback byte-identity: depth-k over-shoot == sync ──
    // A MID-stream stop so the depth-4 chain over-shoots (≤3 fires past the
    // stop) then rolls back; the committed prefix must equal the sync stop run.
    let deep_stop_matched = if let Some(&mid_stop) = tokens_s.get(3) {
        let mut d_ss = Decoder::new(cap)?;
        let tokens_ss = generate(&mut d_ss, &prompt, deep_budget, &[mid_stop], 4, true).await?;
        let mut d_sy = Decoder::new(cap)?;
        let tokens_sy = generate(&mut d_sy, &prompt, deep_budget, &[mid_stop], 1, false).await?;
        !tokens_sy.is_empty() && tokens_ss == tokens_sy
    } else {
        // Too few tokens to place a mid-stream stop — nothing to over-shoot.
        true
    };

    // ── Scenario B — multi-generate CLEAR (the #26 probe's verdict) ──
    // ctx_a: run-ahead gen-1 stopping on its first token, then gen-2 on the
    // SAME context; ctx_b: synchronous gen-1, then gen-2 run-ahead. The gen-2
    // streams must match (and be non-degenerate).
    let clear_ok = if let Some(&stop_tok) = tokens_s.first() {
        let mut ctx_a = Decoder::new(cap)?;
        let _g1a = generate(&mut ctx_a, &prompt, max_tokens, &[stop_tok], 2, false).await?;
        let g2a = generate(&mut ctx_a, &cont, max_tokens, &[], 2, false).await?;

        let mut ctx_b = Decoder::new(cap)?;
        let _g1b = generate(&mut ctx_b, &prompt, max_tokens, &[stop_tok], 1, false).await?;
        let g2b = generate(&mut ctx_b, &cont, max_tokens, &[], 2, false).await?;

        !g2a.is_empty() && g2a.iter().any(|&t| t != 0) && g2a == g2b
    } else {
        false
    };

    // ── Scenario E — DEEP-STOP CLEAR: gen-2 after a rollback is clean ──
    // ctx_a: deep-stop gen-1 (over-shoot + rollback), deep gen-2 same context;
    // ctx_b: sync-stop gen-1 (no over-shoot), deep gen-2 — the clean reference.
    let deep_stop_clear_ok = if let Some(&mid_stop) = tokens_s.get(3) {
        let mut ctx_a = Decoder::new(cap)?;
        let _g1a = generate(&mut ctx_a, &prompt, deep_budget, &[mid_stop], 4, true).await?;
        let g2a = generate(&mut ctx_a, &cont, deep_budget, &[], 4, false).await?;

        let mut ctx_b = Decoder::new(cap)?;
        let _g1b = generate(&mut ctx_b, &prompt, deep_budget, &[mid_stop], 1, false).await?;
        let g2b = generate(&mut ctx_b, &cont, deep_budget, &[], 4, false).await?;

        !g2a.is_empty() && g2a.iter().any(|&t| t != 0) && g2a == g2b
    } else {
        true
    };

    let anchor = anchor_ok(&tokens_p);
    let result = format!(
        "MATCH={matched} DEEP_MATCH={deep_matched} DEEP4_MATCH={deep4_matched} \
         DEEP_STOP_MATCH={deep_stop_matched} DEEP_STOP_CLEAR={deep_stop_clear_ok} \
         ANCHOR_OK={anchor} CLEAR_OK={clear_ok} \
         pipelined={tokens_p:?} deep2={tokens_d2:?} deep4={tokens_d4:?} sync={tokens_s:?}"
    );
    eprintln!("[RUNAHEAD] {result}");
    Ok(result)
}
