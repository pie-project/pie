//! CodeACT agent on top of Pie. Each step the model emits a
//! `{thought, kind, code, answer}` JSON object under a JSON-Schema
//! constraint; the host either runs `code` (Boa-evaluated JavaScript) and
//! feeds the value back as the next observation, or accepts `answer` as
//! the terminal output.
//!
//! **Low-level ① rewrite (grammar, SEQUENTIAL).** Off the
//! `Context`/`Generator`/`Sampler`/`constrain_with` facade onto the keep-core
//! (`ptir-grammar-tranche-conversion-spec`):
//!   - `chat::` template fillers + an in-inferlet `Ctx` (KV working set + cursor
//!     + prompt buffer) replace the deleted `Context`;
//!   - `JsonSchema(schema).build_constraint()` → a kept `constraint::
//!     GrammarConstraint` (host `Matcher`): `advance` + `mask` + `is_terminated`;
//!   - `sampler::grammar_program_sampled` — the masked SAMPLED grammar sampler
//!     `argmax(mask_apply(logits, mask) + gumbel)`. The shipped facade
//!     computed-but-DROPPED the grammar mask (Stage-1 incomplete), so this
//!     conversion now ENFORCES the JSON schema while keeping the `Multinomial`
//!     diversity (Gumbel) that breaks the greedy-Argmax agent-loop lock.
//!
//! **Grammar decode is SEQUENTIAL** — the mask for token N+1 depends on token N
//! (the matcher advances on it), so the run-ahead carrier does NOT apply; each
//! step is a per-fire `grammar_fire` with a fresh host-computed mask.

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, sampler, Constrain, JsonSchema, Result, Schema};
use serde::Deserialize;
use serde_json::Value;

mod js;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_task")]
    task: String,
}

fn default_max_steps() -> u32 { 5 }
fn default_max_tokens() -> usize { 384 }
fn default_task() -> String {
    "Compute 25 squared, then add 17 to it.".to_string()
}

const SYSTEM_PROMPT: &str = "\
You are CodeACT, a problem-solver that thinks one short JavaScript step at \
a time. Each turn output one JSON object {thought, kind, code, answer}:

  - kind=\"code\":  put a JS snippet in `code` and leave `answer` empty. \
                  The host evaluates the snippet; the LAST expression's \
                  value comes back as the next Observation. Each \
                  evaluation is stateless — re-define helpers each turn.
  - kind=\"final\": you are done. Put the final answer in `answer` (the \
                  value from the latest Observation) and leave `code` empty.

Worked example:

  User: Task: Compute (12 * 7) + 5.
  Assistant: {\"thought\":\"Multiply.\",\"kind\":\"code\",\"code\":\"12 * 7\",\"answer\":\"\"}
  User: Observation: 84
  Assistant: {\"thought\":\"Add 5.\",\"kind\":\"code\",\"code\":\"84 + 5\",\"answer\":\"\"}
  User: Observation: 89
  Assistant: {\"thought\":\"Done.\",\"kind\":\"final\",\"code\":\"\",\"answer\":\"89\"}

Each `code` block is a single self-contained JS expression or short \
statement list — do not fence with backticks, just put the source in \
the `code` field.";

const STEP_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought": { "type": "string", "minLength": 1 },
        "kind":    { "type": "string", "enum": ["code", "final"] },
        "code":    { "type": "string" },
        "answer":  { "type": "string" }
    },
    "required": ["thought", "kind", "code", "answer"],
    "additionalProperties": false
}"#;

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let vocab = model::output_vocab_size();
    let g = sampler::grammar_program_sampled(vocab)?;
    let stop = chat::stop_tokens();

    let mut ctx = Ctx::new();
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!("Task: {t}\nWhat is the next step?", t = input.task));
    ctx.cue();

    let mut final_answer: Option<String> = None;

    for step in 1..=input.max_steps {
        let raw = ctx
            .generate_step(&g, STEP_SCHEMA, input.max_tokens, &stop)
            .await
            .map_err(|e| format!("step {step}: generate: {e}"))?;

        let v = serde_json::from_str::<Value>(&raw)
            .map_err(|e| format!("step {step}: JSON parse: {e} (raw: {raw})"))?;
        let thought = v.get("thought").and_then(Value::as_str).unwrap_or("");
        let kind = v.get("kind").and_then(Value::as_str).unwrap_or("");
        let code = v.get("code").and_then(Value::as_str).unwrap_or("");
        let answer = v.get("answer").and_then(Value::as_str).unwrap_or("");

        println!("\n[step {step}] thought: {thought}");

        match kind {
            "code" => {
                println!("[step {step}] code: {code}");
                let observation = js::eval(code);
                println!("[step {step}] observation: {observation}");
                ctx.user(&format!(
                    "Observation: {observation}\nTask (reminder): {t}\nWhat is \
                     the next step?",
                    t = input.task
                ));
                ctx.cue();
            }
            "final" => {
                let answer = answer.trim().to_string();
                println!("[step {step}] final: {answer}");
                final_answer = Some(answer);
                break;
            }
            _ => unreachable!("schema enforces kind ∈ {{code, final}}"),
        }
    }

    match final_answer {
        Some(a) => println!("\nFinal answer: {a}"),
        None => println!("\nNo final answer found within the iteration limit."),
    }
    Ok(String::new())
}

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// In-inferlet decode context (raw-WIT keep-core, no `Context` facade): its own
/// KV working set + sequence cursor + a `buffer` of chat-template tokens not yet
/// prefilled. Chat templating is the kept thin `chat::` bindings.
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
    buffer: Vec<u32>,
    pending_system: Option<String>,
}

impl Ctx {
    fn new() -> Self {
        Self {
            kv: KvWorkingSet::new(),
            seq_len: 0,
            fresh: true,
            buffer: Vec::new(),
            pending_system: None,
        }
    }

    fn flush_pending_system(&mut self) {
        if let Some(system) = self.pending_system.take() {
            self.buffer.extend(chat::system(&system));
        }
    }

    fn is_first_chat_fill(&self) -> bool {
        self.seq_len == 0 && self.buffer.is_empty()
    }

    fn system(&mut self, message: &str) {
        self.flush_pending_system();
        self.pending_system = Some(message.to_string());
    }

    fn user(&mut self, message: &str) {
        let tokens = match self.pending_system.take() {
            Some(system) => chat::system_user(&system, message),
            None if self.is_first_chat_fill() => chat::first_user(message),
            None => chat::user(message),
        };
        self.buffer.extend(tokens);
    }

    fn cue(&mut self) {
        self.flush_pending_system();
        self.buffer.extend(chat::cue());
    }

    /// One SEQUENTIAL grammar fire: geometry + input + the masked SAMPLED grammar
    /// sampler + execute, advancing the cursor. No run-ahead carrier (grammar
    /// can't speculate: the next mask depends on this token). `packed_mask` = the
    /// matcher's per-step allowed bitmask. Fires `fresh_generate` once per
    /// session (first fire).
    fn grammar_fire(
        &mut self,
        g: &sampler::LoweredGrammar,
        tokens: &[u32],
        packed_mask: &[u32],
    ) -> Result<ForwardPass> {
        let n = tokens.len() as u32;
        let pass = ForwardPass::new();
        if self.fresh {
            pass.fresh_generate();
            self.fresh = false;
        }
        let geom = geometry::ensure_pages(
            &self.kv,
            geometry::kv_write_geometry(self.seq_len, n, self.kv.page_size()),
        )?;
        geometry::attach_kv_write(&pass, &self.kv, &geom);
        let positions: Vec<u32> = (self.seq_len..self.seq_len + n).collect();
        pass.input_tokens(tokens, &positions);
        let decode_pos = self.seq_len + n - 1;
        pass.sampler(&g.program, g.bindings(decode_pos, packed_mask)?);
        pass.execute();
        self.seq_len += n;
        Ok(pass)
    }

    /// Generate one grammar-constrained step: prefill the buffered prompt +
    /// sequentially decode under a fresh `JsonSchema` matcher until it TERMINATES
    /// (JSON object complete), a stop token fires, or `max_tokens` is hit. Returns
    /// the decoded text (the JSON object). The final content token is parked as a
    /// residual in `buffer` (not yet in KV) so the next turn's prefill materializes
    /// it, matching the facade's auto-buffer of the last token.
    async fn generate_step(
        &mut self,
        g: &sampler::LoweredGrammar,
        schema: &str,
        max_tokens: usize,
        stop: &[u32],
    ) -> Result<String> {
        let mut matcher = JsonSchema(schema).build_constraint()?;
        let mut decoder = chat::Decoder::new();
        let mut text = String::new();

        self.flush_pending_system();
        let mut pending = std::mem::take(&mut self.buffer);
        if pending.is_empty() {
            pending = vec![0u32];
        }
        let mut generated = 0usize;

        loop {
            // Allowed set for THIS step (matcher state = tokens accepted so far);
            // an empty mask means "no restriction" → an all-ones transparent mask.
            let m = matcher.mask();
            let packed: Vec<u32> = if m.is_empty() {
                vec![u32::MAX; g.mask_words]
            } else {
                m
            };

            let pass = self.grammar_fire(g, &pending, &packed)?;
            let token = read_token(pass).await?;

            // Stop token → drop it (never fed, never materialized), matching the
            // facade's stop-truncation. The last FED token is already in KV.
            if stop.contains(&token) {
                return Ok(text);
            }

            generated += 1;
            match decoder.feed(&[token])? {
                chat::Event::Delta(sd) => text.push_str(&sd),
                chat::Event::Done(sd) => {
                    text = sd;
                    self.buffer.push(token);
                    return Ok(text);
                }
                _ => {}
            }
            matcher.advance(&[token]);
            pending = vec![token];

            if matcher.is_terminated() || generated >= max_tokens {
                // `token` sampled but NOT yet fed to a fire → park as the residual
                // so the next turn's prefill materializes it into KV.
                self.buffer.push(token);
                return Ok(text);
            }
        }
    }
}
