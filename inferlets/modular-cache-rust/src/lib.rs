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

use inferlet::{Context, Result, sample::Sampler};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

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

    // Resume from the longest saved prefix. open() already forks (the snapshot
    // stays immutable), so the context it hands back is ours to append to.
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
                Context::new()?
            }
        }
    } else {
        println!("cache_miss (use_cache=false)");
        Context::new()?
    };

    // Append only the modules that were not already cached.
    for i in resume_index..ordered.len() {
        let module = &ordered[i];

        // System modules become system messages; everything else is user context.
        match module.role {
            Role::System => { ctx.system(&module.text); }
            Role::User => { ctx.user(&module.text); }
        }

        // flush() materializes this module's KV pages.
        ctx.flush().await?;

        // Snapshot every prefix so a later run can resume from any stable one.
        // Best-effort: save() errors if an earlier run already saved this exact
        // prefix (names are content-addressed) — a miss shouldn't kill the run.
        if input.save_cache {
            let name = prefix_key(&ordered[..=i]);
            match ctx.save(&name) {
                Ok(()) => println!("saved={}", name),
                Err(e) => println!("save_skipped={} ({})", name, e),
            }
        }
    }

    // Mark assistant turn start for the chat template, then generate.
    ctx.cue();

    let text = ctx.generate(Sampler::Argmax)
        .max_tokens(input.max_tokens)
        .collect_text()
        .await?;

    Ok(text)
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

/// Open the longest saved prefix snapshot, returning the forked context and
/// the number of modules it covers. Returns `None` if nothing is cached.
fn open_longest_prefix(modules: &[Module]) -> Option<(Context, usize)> {
    // Longest first; open() is Err when that prefix isn't cached, so fall through.
    for len in (1..=modules.len()).rev() {
        let name = prefix_key(&modules[..len]);
        if let Ok(ctx) = Context::open(&name) {
            return Some((ctx, len));
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
