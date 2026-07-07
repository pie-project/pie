//! **Greedy decode on a GDN/hybrid model (KV + RECURRENT-STATE working sets)**
//! via DIRECT WIT bindings (bravo) — the MTP Stage-1 harness's model driver for
//! Qwen3.5-0.8B. It is the `generate` inferlet extended for models with
//! linear-attention layers: those layers need a RUNTIME-assigned recurrent-state
//! (rs_cache) slot per request, exactly as the attention layers need KV page
//! slots. The dense `generate` binds only `KvWorkingSet`; on a GDN model the
//! driver's forward requires `rs_slot_ids` (executor.cpp:2671 "rs_cache forward
//! missing runtime-assigned slot ids") — marshaled from a bound `RsWorkingSet`.
//!
//! **In-forward write (NOT the parked Ph7 buffered fold).** The basic MTP forward
//! writes the recurrent state IN-FORWARD to the folded slot `rs_slot_ids[r]` (the
//! `commit_len` GDN primitive, executor.cpp:2786) — reset-then-write on the fresh
//! fire. So this inferlet just BINDS an `RsWorkingSet` (`pass.rs_working_set`) as
//! the "this forward writes recurrent state" signal; execute_impl's `prepare_write`
//! allocates the folded slot + sets `RS_FLAG_RESET` on the first fire, and the GDN
//! forward writes it in-forward. NO `alloc_buffer`/`rs_fold_buffered` — the buffered
//! fold-from-slabs (`rs_buffer_slot_ids`) path is the parked Ph7 real-driver work.
//!
//! Same greedy-argmax program + KV geometry as `generate` (read = prior FULL
//! pages, write = the tail pages, `offset` = the mid-page cursor). Input: an
//! optional token budget (default 5), e.g. `"24"`.

use inferlet::inference::{ForwardPass, InputBinding};
use inferlet::sampling::{Graph, OutputKind};
use inferlet::working_set::{KvWorkingSet, RsWorkingSet};
use inferlet::{Result, model};

const DEFAULT_MAX_TOKENS: usize = 5;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);

    let vocab = model::output_vocab_size();

    // Greedy-argmax program: `argmax(logits) -> Token`.
    let g = Graph::new(vocab);
    let token_v = g.intrinsic_logits_dyn().argmax();
    g.output(&token_v, OutputKind::Token);
    let built = g.build().map_err(|e| format!("build greedy program: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;

    // Attention KV working set (identical geometry to `generate`).
    let kv = KvWorkingSet::new();
    let page = kv.page_size();

    // Recurrent-state working set for the model's linear-attention layers. Its
    // buffered pages hold the un-folded RS tokens; `rs-fold-buffered` advances
    // the folded state. `state_size() == 0` ⇒ the model has no recurrent state
    // (pure attention) → skip the RS path entirely (this inferlet then behaves
    // like `generate`).
    let rs = RsWorkingSet::new();
    let has_rs = rs.state_size() > 0;

    let prompt = model::encode("hello world");
    let mut pending: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
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
        // recurrent state" signal. execute_impl's `prepare_write` allocates the
        // folded slot (+ RS_FLAG_RESET on the fresh fire) and the GDN forward
        // writes the recurrent state IN-FORWARD to it (the `commit_len` primitive,
        // executor.cpp:2786). NO `alloc_buffer`/`rs_fold_buffered` — the Ph7
        // buffered fold-from-slabs path is parked; the basic MTP decode is
        // in-forward-write. `start-token`/`len-tokens` are unused by the write path.
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
    eprintln!("[GENERATE_GDN] {result}");
    Ok(result)
}
