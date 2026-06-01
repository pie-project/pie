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
//! If project/context changes, task/current cache keys change too.

use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

#[derive(Debug, Clone)]
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
    let model = Model::load(runtime::models().first().ok_or("No models available")?)?;

    // If the caller did not provide a custom module graph, create a useful default.
    let modules = if input.modules.is_empty() {
        default_modules(&input.prompt)
    } else {
        parse_modules(input.modules)?
    };

    // Dependency-order modules so every dependency is appended before its user.
    let ordered = topo_sort(modules)?;

    println!("--- modular-cache-rust ---");
    println!("modules={}", ordered.len());
    println!("use_cache={} save_cache={}", input.use_cache, input.save_cache);

    // Resume from the longest saved prefix if possible.
    let mut resume_index = 0usize;
    let mut ctx = if input.use_cache {
        match open_longest_prefix(&model, &ordered) {
            Ok(Some((cached, len))) => {
                println!("cache_hit_modules={}", len);
                resume_index = len;
                cached.fork()?
            }
            _ => {
                println!("cache_miss");
                Context::new(&model)?
            }
        }
    } else {
        Context::new(&model)?
    };

    // Append only the missing modules.
    for i in resume_index..ordered.len() {
        let module = &ordered[i];

        // System modules become system messages; everything else becomes user context.
        match module.role {
            Role::System => { ctx.system(&module.text); }
            Role::User => { ctx.user(&module.text); }
        }

        // Flush materializes KV pages for this module.
        ctx.flush().await?;

        // Save prefix snapshot after every module.
        // That means a later run can resume from any stable prefix.
        if input.save_cache {
            let name = prefix_key(&ordered[..=i]);
            ctx.save(&name)?;
            println!("saved={}", name);
        }
    }

    // Cue tells the chat template that assistant generation starts here.
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
    let mut out = Vec::new();
    for m in inputs {
        let role = match m.role.to_lowercase().as_str() {
            "system" => Role::System,
            "user" => Role::User,
            other => return Err(format!("unsupported role: {}", other)),
        };
        out.push(Module { id: m.id, role, text: m.text, deps: m.deps });
    }
    Ok(out)
}

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
    }
}

fn prefix_key(modules: &[Module]) -> String {
    let mut h = DefaultHasher::new();
    "modular-cache-v1".hash(&mut h);
    for m in modules {
        m.id.hash(&mut h);
        role_str(&m.role).hash(&mut h);
        m.text.hash(&mut h);
        for d in &m.deps { d.hash(&mut h); }
    }
    format!("modular-cache/{:016x}", h.finish())
}

fn open_longest_prefix(model: &Model, modules: &[Module]) -> Result<Option<(Context, usize)>> {
    for len in (1..=modules.len()).rev() {
        let name = prefix_key(&modules[..len]);
        if let Ok(ctx) = Context::open(model, &name) {
            return Ok(Some((ctx, len)));
        }
    }
    Ok(None)
}

fn topo_sort(modules: Vec<Module>) -> Result<Vec<Module>> {
    let by_id: HashMap<String, Module> =
        modules.into_iter().map(|m| (m.id.clone(), m)).collect();

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    let mut ordered = Vec::new();

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
    if visiting.contains(id) { return Err(format!("cycle at module {}", id)); }

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
