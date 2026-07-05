//! Modular-cache inferlet.
//!
//! This is a standalone example of "modular caching":
//! instead of caching one giant prompt prefix, we split the prompt into
//! reusable semantic modules and cache every stable module-prefix.
//!
//! Example module chain:
//!   system/base -> style/policy -> project/context -> task/current
//!
//! If only task/current changes, system/style/context KV pages can be reused.
//! If project/context changes, task/current cache keys change too (because the
//! cache key for each prefix folds in every earlier module's content).
//!
//! Behavior:
//!   * first run with a given module chain  -> `cache_miss`
//!   * identical re-run                     -> `cache_hit_modules=N` (full reuse)
//!   * only the final task changed          -> reuse the stable earlier prefix
//!   * `use_cache=false`                    -> never open a saved snapshot
//!   * `save_cache=false`                   -> never save new snapshots

//! Low-level ① rewrite (chat-EOS, pipelined + SNAPSHOT): off the `Context`/
//! `Generator`/`Sampler` facade onto the keep-core. A small `Ctx` tracks the
//! materialized token log in lockstep with `seq_len` (what the facade's `history`
//! did); `snapshot::save`/`open` persist that manifest; `open` REPLAYS the log via
//! `prefill::tokens` (the replay lives in the inferlet, not the SDK). The final
//! answer decodes greedily on the run-ahead carrier. `Ctx`/`to_snapshot`/
//! `from_snapshot` mirror echo's `demo-persistent-kv` template.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, snapshot, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

/// A minimal chat context on the keep-core: KV + cursor + the materialized token
/// log (`tokens`, = the facade's `history`) + the unflushed tail (`buffer`), so a
/// snapshot is a serializable manifest and `open` replays the log via `prefill`.
struct Ctx {
    kv: KvWorkingSet,
    page_size: u32,
    seq_len: u32,
    tokens: Vec<u32>,
    buffer: Vec<u32>,
    pending_system: Option<String>,
    fresh: bool,
}

impl Ctx {
    fn new() -> Result<Self> {
        let kv = KvWorkingSet::new();
        let page_size = kv.page_size();
        Ok(Self { kv, page_size, seq_len: 0, tokens: Vec::new(), buffer: Vec::new(), pending_system: None, fresh: true })
    }

    /// Rebuild from a snapshot manifest by REPLAYING its token log (a prefill).
    fn from_snapshot(snap: snapshot::SnapshotData) -> Result<Self> {
        let mut ctx = Self::new()?;
        if !snap.tokens.is_empty() {
            prefill::tokens(&ctx.kv, &mut ctx.seq_len, &snap.tokens)?;
            ctx.tokens = snap.tokens;
            ctx.fresh = false;
        }
        ctx.buffer = snap.buffer;
        ctx.pending_system = snap.pending_system;
        Ok(ctx)
    }

    fn to_snapshot(&self) -> snapshot::SnapshotData {
        snapshot::SnapshotData {
            version: snapshot::SNAPSHOT_VERSION,
            page_size: self.page_size,
            seq_len: self.seq_len,
            tokens: self.tokens.clone(),
            buffer: self.buffer.clone(),
            pending_system: self.pending_system.clone(),
            cas_hashes: Vec::new(),
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

    /// Materialize the buffered tail into KV (the keep-core prefill) + record it
    /// into the log. Mirrors `Context::flush`.
    fn flush(&mut self) -> Result<()> {
        self.flush_pending_system();
        if self.buffer.is_empty() {
            return Ok(());
        }
        let tokens = std::mem::take(&mut self.buffer);
        prefill::tokens(&self.kv, &mut self.seq_len, &tokens)?;
        self.tokens.extend(tokens);
        Ok(())
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

fn pass_carries(stop_empty: bool, max_tokens: usize, produced_token_index: usize) -> bool {
    !(stop_empty && max_tokens == produced_token_index)
}

/// Run-ahead (pipelined) chat-EOS decode with depth-1 EOS rollback, continuing
/// from `*seq_len` (the prefilled module prefix). `prompt` is the buffered tail
/// (the cue).
async fn decode_pipelined(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &sampler::LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(out);
    }
    let prime_carry = pass_carries(stop.is_empty(), max_tokens, 1);
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, s, &pending, prime_carry)?;
    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = pass_carries(stop.is_empty(), max_tokens, generated + 2);
            Some(carrier::submit_pass(kv, seq_len, fresh, s, &[0u32], carry)?)
        } else {
            None
        };
        let token = read_token(producer).await?;
        if stop.contains(&token) {
            if let Some(c) = consumer {
                carrier::discard_pass(c, seq_len).await;
            }
            break;
        }
        out.push(token);
        generated += 1;
        match consumer {
            Some(c) => producer = c,
            None => break,
        }
    }
    Ok(out)
}

fn decode_text(tokens: &[u32]) -> Result<String> {
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    match dec.feed(tokens)? {
        chat::Event::Delta(s) | chat::Event::Done(s) => text.push_str(&s),
        _ => {}
    }
    Ok(text)
}

/// Bump this when the on-disk snapshot layout / key meaning changes so old
/// snapshots can never be confused with new ones.
const CACHE_SCHEMA: &str = "modular-cache-v1";

/// Snapshot-name namespace. Keeps Rust snapshots from colliding with the
/// Python / JS ports, which use their own namespaces.
const CACHE_NS: &str = "modular-cache-rust";

#[derive(Debug, Clone, Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_use_cache")]
    use_cache: bool,
    #[serde(default = "default_save_cache")]
    save_cache: bool,
    #[serde(default)]
    modules: Vec<ModuleInput>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModuleInput {
    id: String,
    #[serde(default = "default_role")]
    role: String,
    text: String,
    #[serde(default)]
    deps: Vec<String>,
}

#[derive(Debug, Clone)]
struct Module {
    id: String,
    role: Role,
    text: String,
    deps: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
enum Role {
    System,
    User,
}

fn default_prompt() -> String {
    "Explain modular KV caching for LLM serving in simple terms.".to_string()
}
fn default_max_tokens() -> usize { 256 }
fn default_use_cache() -> bool { true }
fn default_save_cache() -> bool { true }
fn default_role() -> String { "user".to_string() }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    // If the caller did not provide a custom module graph, build a useful default.
    let modules = if input.modules.is_empty() {
        default_modules(&input.prompt)
    } else {
        parse_modules(input.modules)?
    };

    // Dependency-order modules so every dependency is appended before its user.
    // This also rejects duplicate ids, missing deps, and cycles.
    let ordered = topo_sort(modules)?;

    println!("--- modular-cache-rust ---");
    println!("modules={}", ordered.len());
    println!("order={}", ordered.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(" -> "));
    println!("use_cache={} save_cache={}", input.use_cache, input.save_cache);

    let vocab = model::output_vocab_size();
    let s = sampler::sampler_program(SamplerSpec::Argmax, vocab)?;
    let stop = chat::stop_tokens();

    // Resume from the longest saved prefix. `snapshot::open` reads the manifest;
    // `Ctx::from_snapshot` REPLAYS its token log (a prefill) — the snapshot blob
    // stays on disk (an implicit fork).
    let mut resume_index = 0usize;
    let mut ctx = if input.use_cache {
        match open_longest_prefix(&ordered) {
            Some((cached, len)) => {
                println!("cache_hit_modules={}", len);
                resume_index = len;
                cached
            }
            None => {
                println!("cache_miss");
                Ctx::new()?
            }
        }
    } else {
        println!("cache_miss (use_cache=false)");
        Ctx::new()?
    };

    // Append only the modules that were not already cached.
    for i in resume_index..ordered.len() {
        let module = &ordered[i];

        // System modules become system messages; everything else is user context.
        match module.role {
            Role::System => { ctx.system(&module.text); }
            Role::User => { ctx.user(&module.text); }
        }

        // flush() materializes this module's KV pages (keep-core prefill).
        ctx.flush()?;

        // Snapshot every prefix so a later run can resume from any stable one.
        // Best-effort: save() errors if an earlier run already saved this exact
        // prefix (names are content-addressed) — a miss shouldn't kill the run.
        if input.save_cache {
            let name = prefix_key(&ordered[..=i]);
            match snapshot::save(&name, &ctx.to_snapshot()) {
                Ok(()) => println!("saved={}", name),
                Err(e) => println!("save_skipped={} ({})", name, e),
            }
        }
    }

    // Mark the assistant turn (the decode prime tail), then decode greedily on
    // the run-ahead carrier.
    ctx.cue();
    let prime = std::mem::take(&mut ctx.buffer);
    let toks = decode_pipelined(
        &ctx.kv, &mut ctx.seq_len, &mut ctx.fresh, &s, prime, input.max_tokens, &stop,
    )
    .await?;

    Ok(decode_text(&toks)?)
}

fn default_modules(prompt: &str) -> Vec<Module> {
    vec![
        Module {
            id: "system/base".into(),
            role: Role::System,
            deps: vec![],
            text: "You are a concise technical assistant.".into(),
        },
        Module {
            id: "style/simple".into(),
            role: Role::User,
            deps: vec!["system/base".into()],
            text: "Explain simply first, then give implementation details.".into(),
        },
        Module {
            id: "context/pie".into(),
            role: Role::User,
            deps: vec!["style/simple".into()],
            text: "Pie inferlets can control forward passes, KV cache snapshots, and generation loops.".into(),
        },
        Module {
            id: "task/current".into(),
            role: Role::User,
            deps: vec!["context/pie".into()],
            text: prompt.into(),
        },
    ]
}

fn parse_modules(inputs: Vec<ModuleInput>) -> Result<Vec<Module>> {
    let mut out = Vec::with_capacity(inputs.len());
    for m in inputs {
        if m.id.trim().is_empty() {
            return Err("module id must not be empty".to_string());
        }
        let role = match m.role.to_lowercase().as_str() {
            "system" => Role::System,
            "user" => Role::User,
            other => return Err(format!("unsupported role '{}' on module {}", other, m.id)),
        };
        out.push(Module { id: m.id, role, text: m.text, deps: m.deps });
    }
    Ok(out)
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
    }
}

/// FNV-1a 64-bit. A fixed hash keeps keys stable across runs and matches the
/// Python/JS ports; `DefaultHasher` (SipHash) isn't stable across Rust versions.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Deterministic snapshot name for a module prefix. Folds in the schema
/// version plus every module's id, role, text, and dependency list so any
/// change anywhere in the prefix produces a different key (invalidation).
fn prefix_key(modules: &[Module]) -> String {
    let mut parts: Vec<&str> = vec![CACHE_SCHEMA];
    for m in modules {
        parts.push(&m.id);
        parts.push(role_str(m.role));
        parts.push(&m.text);
        for d in &m.deps {
            parts.push(d);
        }
    }
    format!("{}/{:016x}", CACHE_NS, fnv1a(&parts.join("\u{1f}")))
}

/// Open the longest saved prefix snapshot, replay it, and return the rebuilt
/// context + the number of modules it covers. `None` if nothing is cached.
fn open_longest_prefix(modules: &[Module]) -> Option<(Ctx, usize)> {
    // Longest first; snapshot::open is Err when that prefix isn't cached, so fall through.
    for len in (1..=modules.len()).rev() {
        let name = prefix_key(&modules[..len]);
        if let Ok(snap) = snapshot::open(&name) {
            if let Ok(ctx) = Ctx::from_snapshot(snap) {
                return Some((ctx, len));
            }
        }
    }
    None
}

fn topo_sort(modules: Vec<Module>) -> Result<Vec<Module>> {
    // Index by id, rejecting duplicates (a silent overwrite would drop a module).
    let mut by_id: HashMap<String, Module> = HashMap::with_capacity(modules.len());
    for m in modules {
        if by_id.contains_key(&m.id) {
            return Err(format!("duplicate module id: {}", m.id));
        }
        by_id.insert(m.id.clone(), m);
    }

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    let mut ordered = Vec::new();

    // Sort start ids for a deterministic order independent of input order.
    let mut ids: Vec<_> = by_id.keys().cloned().collect();
    ids.sort();

    for id in ids {
        visit(&id, &by_id, &mut visiting, &mut visited, &mut ordered)?;
    }

    Ok(ordered)
}

fn visit(
    id: &str,
    by_id: &HashMap<String, Module>,
    visiting: &mut HashSet<String>,
    visited: &mut HashSet<String>,
    ordered: &mut Vec<Module>,
) -> Result<()> {
    if visited.contains(id) { return Ok(()); }
    if visiting.contains(id) { return Err(format!("dependency cycle at module {}", id)); }

    let m = by_id.get(id).ok_or_else(|| format!("missing module {}", id))?;
    visiting.insert(id.to_string());

    for dep in &m.deps {
        if !by_id.contains_key(dep) {
            return Err(format!("module {} depends on missing module {}", id, dep));
        }
        visit(dep, by_id, visiting, visited, ordered)?;
    }

    visiting.remove(id);
    visited.insert(id.to_string());
    ordered.push(m.clone());
    Ok(())
}
