//! **Greedy decode via DIRECT WIT bindings** (bravo). Per In Gim's directive —
//! no `inferlet/` crate decode helpers (`Context`, `Generator`, `collect_tokens`,
//! the `Forward`/`Output` wrappers). This drives the whole decode loop through the
//! raw WIT surface:
//!   - `working_set::KvWorkingSet` — `new` / `page-size` / `size` / `alloc`;
//!   - `inference::ForwardPass` — `kv-working-set` / `input-tokens` / `sampler` /
//!     `execute` / `output`;
//!   - a greedy-argmax `tensor::Program` (logits intrinsic → argmax → Token).
//!
//! It reimplements the minimal KV geometry the `Context` facade wraps (read =
//! prior full pages `[0, first_write_page)`; write = the tail pages; `offset` =
//! the mid-page cursor). This is the exact WIT path the M3 inferlets use, and the
//! decode inferlet the harnesses drive — so the harness
//! exercises real WIT bindings, not the SDK sugar.
//!
//! Input: an optional token budget (default 5), e.g. `"16"` or `{"lane":N}`
//! (ignored → default). On the mock's `EchoBehavior(42)` a clean run returns
//! `generated 5 tokens: [42, 42, 42, 42, 42]`.

use inferlet::inference::{ForwardPass, InputBinding};
use inferlet::sampling::{Graph, OutputKind};
use inferlet::working_set::KvWorkingSet;
use inferlet::{Result, model};

const DEFAULT_MAX_TOKENS: usize = 5;

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // guru's discriminator: NAME the launch-input encoding (rules out stale-wasm /
    // quote-wrap — a bare int here ⇒ the budget reached the inferlet live).
    eprintln!("[generate] input={input:?}");
    // Bare-integer input → budget; anything else (e.g. `{"lane":N}`) → default.
    let max_tokens: usize = input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS);

    let vocab = model::output_vocab_size();

    // Greedy-argmax program: `argmax(logits) -> Token`. Program-authoring (the WIT
    // tensor-program path), not a decode-loop helper.
    let g = Graph::new(vocab);
    let token_v = g.intrinsic_logits_dyn().argmax();
    g.output(&token_v, OutputKind::Token);
    let built = g.build().map_err(|e| format!("build greedy program: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("emit: {e}"))?;

    // Raw KV working set (binds the single served model implicitly).
    let kv = KvWorkingSet::new();
    let page = kv.page_size();

    let prompt = model::encode("hello world");
    let mut pending: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let mut seq_len: u32 = 0;
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    for step in 0..max_tokens {
        let n = pending.len() as u32;

        // Minimal `prepare_write` geometry: read = prior FULL pages
        // `[0, first_write_page)`; write = the tail pages; grow the slot array to
        // cover `seq_len + n` tokens.
        let first_write_page = seq_len / page;
        let total_pages = (seq_len + n).div_ceil(page);
        let have = kv.size();
        if total_pages > have {
            kv.alloc(total_pages - have)
                .map_err(|e| format!("alloc @{step}: {e}"))?;
        }

        let pass = ForwardPass::new();
        pass.kv_working_set(
            &kv,
            0,                                // inp-start
            first_write_page,                 // inp-len   (prior full pages = read context)
            first_write_page * page,          // valid-tokens
            first_write_page,                 // output-start
            total_pages - first_write_page,   // output-len (write tail pages)
            seq_len % page,                   // offset     (mid-page cursor)
        );

        let positions: Vec<u32> = (seq_len..seq_len + n).collect();
        pass.input_tokens(&pending, &positions);

        // Sample the last row's logits (the decode position).
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
        seq_len += n; // committed n tokens (prompt on step 0, then 1 per step)
        pending = vec![token]; // the sampled token is the next step's input
    }

    let result = format!("generated {} tokens: {:?}", generated.len(), generated);
    eprintln!("[GENERATE] {result}");
    Ok(result)
}
