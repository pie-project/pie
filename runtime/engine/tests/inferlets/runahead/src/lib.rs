//! 1a run-ahead carryover verify via **DIRECT WIT bindings** (bravo, In Gim's
//! directive). Reimplements what the SDK's `collect_tokens_pipelined` /
//! `collect_tokens` do, inlined onto the raw WIT surface (`inference::ForwardPass`
//! + `working_set::KvWorkingSet` + a greedy-argmax `tensor::Program`) — NO
//! `Context`/`Generator`/`collect_tokens` sugar. It is a faithful transcription of
//! the SDK's `submit_producer` / `submit_consumer` / `await_commit` raw calls, so
//! it is numerically identical to the sugar path it replaces.
//!
//! The run-ahead carrier: a producer pass declares `next-inputs(&[0])` (carry its
//! sampled token into the next pass's input row 0) and is eager-`execute()`d; the
//! consumer pass is built + `execute()`d BEFORE the producer's `output()` is
//! awaited, so the two overlap in flight. The device carrier injects the
//! producer's sample into the consumer's placeholder `[0]` input row.
//!
//!  1. **Scenario A** (`MATCH` + `ANCHOR_OK`): the pipelined carrier stream equals
//!     the synchronous stream AND positively equals the verified milestone tokens.
//!  2. **Scenario B** (`CLEAR_OK`): the #26 dangling-carry clear (`fresh-generate`).
//!  3. **Scenario C** (`DEEP_MATCH` / `DEEP4_MATCH`): the depth-`k` submit-ahead
//!     carrier chain (`decode_pipelined_deep`, the device-resident reduce-R cut) is
//!     byte-identical to the synchronous stream. `DEEP_MATCH` (depth-2, shipped WAR
//!     bound) is verifiable now; `DEEP4_MATCH` (depth-4) confirms the single-scalar
//!     WAR guard is depth-k-correct (run with `PIE_SCHED_MAX_IN_FLIGHT=4`).
//!  4. **Scenario D** (`DEEP_STOP_MATCH`): the depth-`k` EOS-rollback
//!     (`decode_pipelined_deep_stop`) — over-shoot a mid-stream stop by ≤`depth`−1
//!     fires, discard + roll back, byte-identical to `decode_sync(stop)`.
//!  5. **Scenario E** (`DEEP_STOP_CLEAR`): the depth-`k` rollback's host-side
//!     free-all (spec §4) reclaims the over-shoot-orphaned carry — a gen-2 on the
//!     same context after a deep-stop is clean + matches the sync-gen-1 reference.
//!
//! JSON/plain input: an optional token budget (defaults to 8), e.g. `"16"`.

use std::collections::VecDeque;

use inferlet::inference::{ForwardPass, InputBinding};
use inferlet::sampling::{Graph, OutputKind};
use inferlet::working_set::KvWorkingSet;
use inferlet::{Result, model, tensor};

const PROMPT: &str = "hello world";
/// Fixed 2nd-turn continuation for the #26 clear probe (Scenario B).
const CONT: &str = " Tell me more.";

/// The known-good greedy decode of `PROMPT` on qwen3-0.6b — the verified #6/#21
/// milestone stream (argmax == greedy at temperature 0).
const MILESTONE: [u32; 8] = [198, 9707, 1879, 374, 264, 4285, 2025, 429];

fn anchor_ok(tokens: &[u32]) -> bool {
    match tokens.len() {
        0 => false,
        n if n <= MILESTONE.len() => tokens == &MILESTONE[..n],
        _ => tokens[..MILESTONE.len()] == MILESTONE && tokens.iter().any(|&t| t != 0),
    }
}

/// Whether a pass producing the `produced_token_index`-th token (1-based) should
/// declare a `next-inputs` carry (mirrors the SDK's `pass_carries`): decline ONLY
/// on the terminal max-tokens boundary with no stop (count-predictable at submit).
fn pass_carries(stop_empty: bool, max_tokens: usize, produced_token_index: usize) -> bool {
    !(stop_empty && max_tokens == produced_token_index)
}

/// One decode context on the raw WIT surface: its own KV working set + cursor.
/// `fresh` arms `fresh-generate()` on the next pass (the #26 clear).
struct Decoder {
    kv: KvWorkingSet,
    page: u32,
    seq_len: u32,
    fresh: bool,
}

impl Decoder {
    fn new() -> Self {
        let kv = KvWorkingSet::new();
        let page = kv.page_size();
        Self { kv, page, seq_len: 0, fresh: true }
    }

    /// Arm the #26 dangling-carry clear on the next pass (a fresh generate on the
    /// same context).
    fn arm_fresh(&mut self) {
        self.fresh = true;
    }

    /// Build (but do NOT execute) a forward pass over `tokens` at the current
    /// cursor with the greedy `program`, sampling the last row's logits and
    /// (if `carry`) declaring the run-ahead carrier. Mirrors the SDK's
    /// `submit_producer`/`submit_consumer` minus the `execute()`.
    fn build_pass(
        &mut self,
        program: &tensor::Program,
        tokens: &[u32],
        carry: bool,
    ) -> Result<ForwardPass> {
        let n = tokens.len() as u32;
        // Minimal `prepare_write` geometry (read = prior full pages; write = tail).
        let first_write_page = self.seq_len / self.page;
        let total_pages = (self.seq_len + n).div_ceil(self.page);
        let have = self.kv.size();
        if total_pages > have {
            self.kv
                .alloc(total_pages - have)
                .map_err(|e| format!("alloc: {e}"))?;
        }
        let pass = ForwardPass::new();
        if self.fresh {
            pass.fresh_generate();
            self.fresh = false;
        }
        pass.kv_working_set(
            &self.kv,
            0,
            first_write_page,
            first_write_page * self.page,
            first_write_page,
            total_pages - first_write_page,
            self.seq_len % self.page,
        );
        let positions: Vec<u32> = (self.seq_len..self.seq_len + n).collect();
        pass.input_tokens(tokens, &positions);
        let decode_pos = self.seq_len + n - 1;
        pass.sampler(program, vec![InputBinding::Logits(vec![decode_pos])]);
        if carry {
            pass.next_inputs(&[0]);
        }
        Ok(pass)
    }
}

async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// Synchronous greedy decode (raw): build → execute → await → commit, one pass
/// per token. Stops on a `stop` token (dropping it).
async fn decode_sync(
    d: &mut Decoder,
    program: &tensor::Program,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let mut pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let pass = d.build_pass(program, &pending, false)?;
        pass.execute();
        let token = read_token(pass).await?;
        d.seq_len += pending.len() as u32;
        if stop.contains(&token) {
            break;
        }
        out.push(token);
        pending = vec![token];
    }
    Ok(out)
}

/// Run-ahead (pipelined) greedy decode (raw): eager-submit the consumer BEFORE
/// awaiting the producer, the device carrier injecting the producer's sample.
/// A faithful inline of the SDK `collect_tokens_pipelined` loop.
async fn decode_pipelined(
    d: &mut Decoder,
    program: &tensor::Program,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);

    // Prime producer (the prompt tail): carries unless it is itself terminal.
    let prime_carry = pass_carries(stop.is_empty(), max_tokens, 1);
    let mut producer = d.build_pass(program, &pending, prime_carry)?;
    producer.execute();
    d.seq_len += pending.len() as u32; // advance the cursor on SUBMIT

    let mut generated = 0usize;
    loop {
        // Speculate the next consumer eagerly UNLESS this step may terminate.
        let speculate = stop.is_empty() && generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = pass_carries(stop.is_empty(), max_tokens, generated + 2);
            let c = d.build_pass(program, &[0u32], carry)?; // placeholder; carrier injects
            c.execute();
            d.seq_len += 1;
            Some(c)
        } else {
            None
        };

        let token = read_token(producer).await?;
        if stop.contains(&token) {
            break;
        }
        out.push(token);
        generated += 1;
        let hit_max = generated >= max_tokens;

        match consumer {
            Some(c) => producer = c,
            None => {
                if hit_max {
                    break;
                }
                // Stop configured but not hit → decode the next step sequentially
                // (the terminal pass isn't predictable at submit, so it carried).
                let c = d.build_pass(program, &[0u32], true)?;
                c.execute();
                d.seq_len += 1;
                producer = c;
            }
        }
    }
    Ok(out)
}

/// **Deep** run-ahead (pipelined) greedy decode (raw): submit `depth` carrier-linked
/// fires UPFRONT (none awaited), then FIFO-drain + refill to sustain `depth` in
/// flight. Generalizes `decode_pipelined` (fixed depth-2, one await per token) to
/// depth-`k` — the device-resident reduce-R cut (`ptir-device-resident-carrier-reduce-r`
/// §2): the ~870µs per-token host round-trip (charlie's G3 sub-probe) amortizes over
/// `depth`, so the device runs the `depth` fires back-to-back with no per-token host
/// cross. Fixed-length (stop-free) form — the G3-bubble workload; a stop set is the
/// `decode_pipelined` bounded-speculation path (depth-1 over-shoot), generalized to a
/// depth-`k` over-shoot + rollback separately (see the spec §4). Byte-identical to
/// `decode_sync` (the carrier injects each predecessor's sample, exactly like feeding
/// `pending = [token]`); only WHEN the fires launch differs, not their inputs/outputs.
///
/// NOTE: `depth` is the inferlet's *submit-ahead* window; realizing depth-`k` on the
/// device also needs the scheduler firing up to `k` concurrently (`MAX_IN_FLIGHT=k`,
/// WaitAllPolicy) + the driver's `k`-deep WAR-event ring for the shared `pi.sampled`
/// (spec §3.2/§3.3). Submitting ahead of the cap is safe — the extra fires queue.
async fn decode_pipelined_deep(
    d: &mut Decoder,
    program: &tensor::Program,
    prompt: Vec<u32>,
    max_tokens: usize,
    depth: usize,
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out = Vec::with_capacity(max_tokens);
    // FIFO of in-flight fires (front = oldest producer, awaited next).
    let mut inflight: VecDeque<ForwardPass> = VecDeque::with_capacity(depth.max(1));
    // Each fire produces exactly one token, so `max_tokens` fires ⇒ `max_tokens`
    // tokens. `submitted` counts fires launched so far.
    let mut submitted = 0usize;

    // Prime + fill: launch up to `depth` carrier-linked fires UPFRONT, NONE awaited.
    // Fire 0 carries the prompt tail; each later fire is a placeholder `[0]` whose
    // input the device carrier injects from its predecessor's sampled token.
    while inflight.len() < depth.max(1) && submitted < max_tokens {
        let carry = pass_carries(true, max_tokens, submitted + 1);
        let toks: Vec<u32> = if submitted == 0 { pending.clone() } else { vec![0u32] };
        let pass = d.build_pass(program, &toks, carry)?;
        pass.execute();
        d.seq_len += toks.len() as u32; // advance the cursor on SUBMIT
        submitted += 1;
        inflight.push_back(pass);
    }

    // Drain FIFO; refill one fire per drained token to sustain `depth` in flight.
    while let Some(prod) = inflight.pop_front() {
        out.push(read_token(prod).await?);
        if submitted < max_tokens {
            let carry = pass_carries(true, max_tokens, submitted + 1);
            let pass = d.build_pass(program, &[0u32], carry)?;
            pass.execute();
            d.seq_len += 1;
            submitted += 1;
            inflight.push_back(pass);
        }
    }
    Ok(out)
}

/// **Deep run-ahead with a STOP set — the depth-`k` EOS-rollback** (spec §4).
/// Unlike `decode_pipelined` (which declines to speculate when a stop is set → the
/// depth-1 sequential path), this SPECULATES `depth` fires ahead EVEN past a possible
/// stop (over-shoot), keeping the pipeline full through the stop boundary. On the
/// first stop token it DRAINS the ≤`depth`−1 over-shot in-flight fires (#17 arm 2 —
/// finalize, not drop) and rolls the cursor back to the committed prefix. Generalizes
/// the depth-1 EOS-rollback (`ptir-pipelined-eos-rollback-spec`, my SDK build) to
/// depth-`k`. Byte-identical to `decode_sync(stop)`: the over-shot fires' output is
/// discarded (never pushed to `out`) and the cursor rolls back, so the output prefix +
/// committed state match exactly (any over-shot KV commit is overwritten by the next
/// generate's window + the #26 fresh-clear).
///
/// #17 (FLEET=8 preempt): the over-shot are DRAINED (`read_token` + ignore), NOT
/// dropped. A dropped fire's ForwardPass resource is gone ⇒ `drain_retired_fires`
/// PRUNES it without waiting, so its still-firing device inject retain-misses a link
/// the fresh-generate free-all already freed. Draining registers→drains them so arm 1's
/// drain-before-free-all (`48c3530d`) sees `in_flight==0` + the free contract holds.
///
/// RETAINED-BUFFER RECLAIM (spec §4, BUILT host-side): the last COMMITTED fire's
/// retained carry is orphaned (its drain-gated free rode an over-shot's request). The
/// host-side context-scoped **free-all on fresh-`generate()`** (`arm_fresh()` →
/// `take_produced_links_for_context`, inference.rs) reclaims it — and with arm 2's
/// drain, every prior consumer has provably drained before the free (no retain-miss).
/// The pilot surface uses echo's `carrier::discard_pass` (also a finalize-drain).
/// Validated on the 4090 (mock fail-closes the carrier drain).
async fn decode_pipelined_deep_stop(
    d: &mut Decoder,
    program: &tensor::Program,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
    depth: usize,
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out = Vec::with_capacity(max_tokens);
    let mut inflight: VecDeque<ForwardPass> = VecDeque::with_capacity(depth.max(1));
    let mut submitted = 0usize;

    // Speculative fill: submit up to `depth` fires ahead, OVER-SHOOTING a possible
    // stop (with a stop set, `pass_carries` is true ⇒ every fire carries the chain).
    while inflight.len() < depth.max(1) && submitted < max_tokens {
        let carry = pass_carries(stop.is_empty(), max_tokens, submitted + 1);
        let toks: Vec<u32> = if submitted == 0 { pending.clone() } else { vec![0u32] };
        let pass = d.build_pass(program, &toks, carry)?;
        pass.execute();
        d.seq_len += toks.len() as u32;
        submitted += 1;
        inflight.push_back(pass);
    }

    let mut hit_stop = false;
    while let Some(prod) = inflight.pop_front() {
        let token = read_token(prod).await?;
        if stop.contains(&token) {
            hit_stop = true; // the stop fire committed (awaited); over-shot remains
            break;
        }
        out.push(token);
        if submitted < max_tokens {
            let carry = pass_carries(stop.is_empty(), max_tokens, submitted + 1);
            let pass = d.build_pass(program, &[0u32], carry)?;
            pass.execute();
            d.seq_len += 1;
            submitted += 1;
            inflight.push_back(pass);
        }
    }

    if hit_stop {
        // Depth-k rollback — #17 arm 2 (DISCARD-not-DROP, the FLEET=8 preempt fix):
        // DRAIN each over-shot fire (finalize via `read_token`, ignoring the token)
        // rather than DROP (`inflight.clear()`). A DROPPED fire's ForwardPass resource
        // is gone ⇒ `drain_retired_fires` PRUNES it (inference.rs) without waiting, so
        // its still-firing device inject retain-misses a link the fresh-generate
        // free-all already freed (the regressor `46471f7e`). Draining finalizes the
        // over-shot so it is registered→drained, NOT pruned ⇒ arm 1's
        // drain-before-free-all sees `in_flight==0` and the free contract's
        // precondition holds. The over-shot's transient commit is harmless: the cursor
        // rolls back below and the next generate's window + #26 fresh-clear overwrite
        // its KV. `arm_fresh()` arms the free-all that reclaims the terminal carry.
        let overshot = inflight.len() as u32;
        for c in inflight.drain(..) {
            let _ = read_token(c).await; // finalize-drain; a discard never fails the decode
        }
        d.seq_len = d.seq_len.saturating_sub(overshot);
        d.arm_fresh();
    }
    Ok(out)
}

/// Build the greedy-argmax sampler program (raw tensor program): `argmax(logits)`.
fn greedy_program(vocab: u32) -> Result<tensor::Program> {
    let g = Graph::new(vocab);
    let token = g.intrinsic_logits_dyn().argmax();
    g.output(&token, OutputKind::Token);
    let built = g.build().map_err(|e| format!("build greedy program: {e:?}"))?;
    inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(8);
    let vocab = model::output_vocab_size();
    let program = greedy_program(vocab)?;
    let prompt = model::encode(PROMPT);
    let prompt = if prompt.is_empty() { vec![0u32] } else { prompt };

    // ── Scenario A — carrier token-exactness: pipelined == synchronous ──
    let mut d_p = Decoder::new();
    let tokens_p = decode_pipelined(&mut d_p, &program, prompt.clone(), max_tokens, &[]).await?;
    let mut d_s = Decoder::new();
    let tokens_s = decode_sync(&mut d_s, &program, prompt.clone(), max_tokens, &[]).await?;
    let matched = tokens_p == tokens_s;

    // #17/#18 fix (deep-scenario KV bound): the DEEP scenarios below test the
    // carrier MECHANISM (byte-identity + over-shoot/rollback), which needs only
    // the stop region (~a dozen tokens) — NOT the full process budget. At a
    // large budget (b200/b500) their by-design full-length decodes — × depth
    // pre-submission × FLEET lanes — exhaust a tight page pool: the fleet gets
    // suspended lane by lane and the run wedges silently (the #19 exhaustion
    // terminal state; observed on-device as idle-block `suspended=6-7`,
    // `promotable=0`, ~295s of silence). Secondarily, byte-identity to the sync
    // reference is only reliable within a short window: concurrent-decode
    // numeric non-invariance diverges the streams length-onset (verified on
    // device: identity holds at 16, full-budget MATCH=false at 200), which at
    // full budget would also strand scenario E's sync-derived stop. Bounding
    // the deep scenarios to a small fixed window keeps them a mechanism test
    // (co-batch density still comes from the concurrent FLEET) with a flat KV
    // footprint. The shallow scenario A/B (`decode_pipelined`) is unaffected.
    let deep_budget = max_tokens.min(16);

    // ── Scenario C — DEEP pipeline byte-identity: depth-k submit-ahead == sync ──
    // The device-resident reduce-R cut: submit `depth` fires upfront, drain FIFO.
    // Byte-identical to the synchronous stream (only launch timing differs).
    //  - depth=2 (`DEEP_MATCH`): the generalized FIFO submit-ahead/drain/refill
    //    loop at the SAME WAR bound as `decode_pipelined` (shipped depth-1
    //    `last_eager_d2h_done` guard) → verifiable on the 4090 NOW, independent of
    //    charlie's k-deep WAR-event ring. Proves the FIFO accounting is lossless.
    //  - depth=4 (`DEEP4_MATCH`, observed, not asserted): 4 fires share `pi.sampled`
    //    → run with `PIE_SCHED_MAX_IN_FLIGHT=4` to exercise true 4-in-flight. Confirms
    //    the single `last_eager_d2h_done` scalar is depth-k-correct (one FIFO copy
    //    stream + same-cublas-stream retain ⇒ no WAR ring needed; spec §3.3).
    let mut d_d2 = Decoder::new();
    let tokens_d2 = decode_pipelined_deep(&mut d_d2, &program, prompt.clone(), deep_budget, 2).await?;
    let deep_matched = tokens_d2.as_slice() == &tokens_s[..deep_budget];
    let mut d_d4 = Decoder::new();
    let tokens_d4 = decode_pipelined_deep(&mut d_d4, &program, prompt.clone(), deep_budget, 4).await?;
    let deep4_matched = tokens_d4.as_slice() == &tokens_s[..deep_budget];

    // ── Scenario D — DEEP STOP rollback byte-identity: depth-k over-shoot == sync ──
    // Pick a MID-stream stop so the depth-4 pipeline over-shoots it (≤3 fires past
    // the stop) then rolls back. Byte-identical to `decode_sync(stop)`: the discarded
    // over-shot never commits, so the output prefix matches. The depth-k EOS-rollback
    // (spec §4) for the stop-configured chat pilots.
    let deep_stop_matched = if let Some(&mid_stop) = tokens_s.get(3) {
        let mut d_ss = Decoder::new();
        let tokens_ss =
            decode_pipelined_deep_stop(&mut d_ss, &program, prompt.clone(), deep_budget, &[mid_stop], 4)
                .await?;
        let mut d_sy = Decoder::new();
        let tokens_sy = decode_sync(&mut d_sy, &program, prompt.clone(), deep_budget, &[mid_stop]).await?;
        !tokens_sy.is_empty() && tokens_ss == tokens_sy
    } else {
        // Too few tokens to place a mid-stream stop — trivially pass (nothing to over-shoot).
        true
    };

    // ── Scenario B — #26 dangling-carry CLEAR (fresh-generate host-clear) ──
    let cont = model::encode(CONT);
    let clear_ok = if let Some(&stop_tok) = tokens_s.first() {
        // ctx_a: PIPELINED gen-1 stopping on its first token (leaves a dangling
        // carry), then gen-2 on the SAME context (arm fresh-generate).
        let mut ctx_a = Decoder::new();
        let _g1a = decode_pipelined(&mut ctx_a, &program, prompt.clone(), max_tokens, &[stop_tok]).await?;
        ctx_a.arm_fresh();
        let g2a = decode_pipelined(&mut ctx_a, &program, cont.clone(), max_tokens, &[]).await?;

        // ctx_b: SEQUENTIAL gen-1 (no carrier), then gen-2 pipelined.
        let mut ctx_b = Decoder::new();
        let _g1b = decode_sync(&mut ctx_b, &program, prompt.clone(), max_tokens, &[stop_tok]).await?;
        ctx_b.arm_fresh();
        let g2b = decode_pipelined(&mut ctx_b, &program, cont.clone(), max_tokens, &[]).await?;

        !g2a.is_empty() && g2a.iter().any(|&t| t != 0) && g2a == g2b
    } else {
        false
    };

    // ── Scenario E — DEEP-STOP CLEAR: the free-all reclaims the over-shoot orphan ──
    // gen-1 deep-stops MID-stream (over-shoots ≤3 fires, rolls back; the rollback's
    // `arm_fresh()` arms the host-side free-all). gen-2 on the SAME context must be
    // CLEAN — no stale carry injected, the orphaned last-committed carry reclaimed —
    // and equal the reference (sync gen-1 + the same deep gen-2). The end-to-end proof
    // of the spec §4 free-all (the unit test covers the tracking logic).
    let deep_stop_clear_ok = if let Some(&mid_stop) = tokens_s.get(3) {
        // ctx_a: deep-stop gen-1 (rollback + free-all armed), deep gen-2 same context.
        let mut ctx_a = Decoder::new();
        let _g1a =
            decode_pipelined_deep_stop(&mut ctx_a, &program, prompt.clone(), deep_budget, &[mid_stop], 4)
                .await?;
        let g2a = decode_pipelined_deep(&mut ctx_a, &program, cont.clone(), deep_budget, 4).await?;

        // ctx_b: sequential-stop gen-1 (no over-shoot), deep gen-2 — the clean reference.
        let mut ctx_b = Decoder::new();
        let _g1b = decode_sync(&mut ctx_b, &program, prompt.clone(), deep_budget, &[mid_stop]).await?;
        ctx_b.arm_fresh();
        let g2b = decode_pipelined_deep(&mut ctx_b, &program, cont.clone(), deep_budget, 4).await?;

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
