//! **rs_cache T0 — E1 discriminator: SINGLE-token (N=1) prefill.** A byte-for-byte
//! copy of `generate-gdn` (same greedy program, same KV geometry, same in-forward
//! RS write) with EXACTLY ONE change: the prefill fire carries a SINGLE token
//! (N=1), not the 2-token "hello world" prompt (N=2). Its sole purpose is to
//! isolate the rs_cache T0 boundary bug:
//!
//!   generate-gdn (N=2 prefill) GLITCHES ~4 decode steps at the prefill->decode
//!   boundary, then recovers to HF-exact. Run THIS (N=1 prefill) on the same 4090
//!   + a matching HF golden:
//!     * glitch VANISHES ⇒ the bug is the MULTI-TOKEN prefill state-commit
//!       (the N=2 in-forward fold doesn't commit both tokens' state — charlie's
//!       driver in-forward commit / `commit_len` boundary).
//!     * glitch PERSISTS ⇒ the bug is the GENERAL prefill->decode slab handoff
//!       (CoW-move timing / folded_block toggle — the runtime RS wiring), which
//!       is independent of the prefill token count.
//!
//! The single prefill token = `encode("hello world")[0]` (the FIRST token of
//! `generate-gdn`'s repro prompt), so charlie's HF golden for E1 is just
//! `HF-generate([tok0])` — reproducible from the existing tokenization, no new
//! prompt. Everything downstream (decode loop, RS binding, sampler) is identical
//! to `generate-gdn` so the ONLY independent variable is prefill N. Does NOT
//! touch `generate-gdn` (charlie's `ea24c720` repro stays intact).
//!
//! Input: an optional token budget (default 5), e.g. `"24"` — same as generate-gdn.

use inferlet::inference::{ForwardPass, InputBinding};
use inferlet::sampling::{Graph, OutputKind};
use inferlet::working_set::{KvWorkingSet, RsWorkingSet};
use inferlet::{Result, model};

const DEFAULT_MAX_TOKENS: usize = 5;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);

    let vocab = model::output_vocab_size();

    // Greedy-argmax program: `argmax(logits) -> Token` (identical to generate-gdn).
    let g = Graph::new(vocab);
    let token_v = g.intrinsic_logits_dyn().argmax();
    g.output(&token_v, OutputKind::Token);
    let built = g.build().map_err(|e| format!("build greedy program: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;

    // Attention KV working set (identical geometry to generate-gdn).
    let kv = KvWorkingSet::new();
    let page = kv.page_size();

    // Recurrent-state working set — same in-forward-write binding as generate-gdn.
    let rs = RsWorkingSet::new();
    let has_rs = rs.state_size() > 0;

    // E1 DISCRIMINATOR: force a SINGLE-token (N=1) prefill. Take only the FIRST
    // token of the repro prompt so there is NO multi-token (N=2) prefill fold —
    // the one and only difference from generate-gdn.
    let full_prompt = model::encode("hello world");
    let first_tok = full_prompt.first().copied().unwrap_or(0);
    let mut pending: Vec<u32> = vec![first_tok];
    let mut seq_len: u32 = 0;
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    for step in 0..max_tokens {
        let n = pending.len() as u32;

        // KV geometry (read = prior full pages; write = tail pages).
        let first_write_page = seq_len / page;
        let total_pages = (seq_len + n).div_ceil(page);
        let have = kv.size();
        if total_pages > have {
            kv.alloc(total_pages - have)
                .map_err(|e| format!("kv.alloc @{step}: {e}"))?;
        }

        let pass = ForwardPass::new();
        pass.kv_working_set(
            &kv,
            0,
            first_write_page,
            first_write_page * page,
            first_write_page,
            total_pages - first_write_page,
            seq_len % page,
        );

        // RS in-forward write: bind the RS working set as the "this forward writes
        // recurrent state" signal (identical to generate-gdn). execute_impl's
        // prepare_write_in_txn allocates the folded slot (+ RS_FLAG_RESET on the
        // fresh fire) and the GDN forward writes the recurrent state in-forward.
        if has_rs {
            pass.rs_working_set(&rs, 0, n);
        }

        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&pending, &positions);

        let decode_pos = seq_len + n - 1;
        pass.sampler(&program, vec![InputBinding::Logits(vec![decode_pos])]);

        pass.execute();
        let out = pass
            .output()
            .await
            .map_err(|e| format!("output @{step}: {e}"))?;
        let bytes = out.read().map_err(|e| format!("tensor read @{step}: {e:?}"))?;
        let token = if bytes.len() >= 4 {
            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
        } else {
            return Err(format!("output @{step}: short tensor ({} bytes)", bytes.len()));
        };

        generated.push(token);
        seq_len += n;
        pending = vec![token];
    }

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE_GDN_N1] {result}");
    Ok(result)
}
