//! What does a lazily filled artifact cost on a real request?
//!
//! An eager artifact groups every lexer state up front, and that is where the
//! memory goes. A mask is a pure function of the state, though, and a document
//! reaches only some of the states its grammar can, so the states can be
//! grouped when they are first reached and kept afterwards.
//!
//! This replays each corpus document through a cache and reports what was
//! actually filled: how many states, how much memory, and how the misses are
//! distributed over the document — because a miss is only affordable if they
//! stop happening.

use std::fs;
use std::sync::Arc;

use anyhow::{Result, bail};
use gpugrammar_ir::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use gpugrammar_lex::lexicon::{extract, terminal_automata_within};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{build_lexer_within, group_vocabulary};
use gpugrammar_lr::cfg::flatten_within;
use gpugrammar_lr::tables::build;
use gpugrammar_run::Matcher;
use gpugrammar_run::cache::Cache;
use gpugrammar_tables::{emit, emit_ungrouped};
use serde::Deserialize;

#[derive(Deserialize)]
struct Corpus {
    instances: Vec<Instance>,
}

#[derive(Deserialize)]
struct Instance {
    schema: String,
    text: String,
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 2 {
        bail!("usage: gpugrammar-cache <instances.json> [limit]");
    }
    let corpus: Corpus = serde_json::from_str(&fs::read_to_string(&arguments[1])?)?;
    let limit: usize = arguments
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(24);
    let state_limit: usize = std::env::var("GPUGRAMMAR_MAX_LEXER_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4_000);

    // A byte vocabulary keeps the comparison about policy rather than about a
    // particular tokenizer; the ratio between eager and lazy does not depend on
    // the vocabulary, since both scale with it identically.
    let vocabulary: Arc<Vec<Vec<u8>>> = Arc::new((0u8..=255).map(|byte| vec![byte]).collect());

    println!(
        "{:>4} {:>7} {:>8} {:>7} {:>10} {:>10} {:>7} {:>12}",
        "idx", "states", "filled", "share", "eager KB", "lazy KB", "saving", "last miss"
    );

    let mut eager_total = 0usize;
    let mut lazy_total = 0usize;
    let mut reported = 0usize;

    for (index, instance) in corpus.instances.iter().enumerate() {
        let Ok(grammar) = json_schema_to_grammar(&instance.schema, &JsonSchemaOptions::default())
        else {
            continue;
        };
        let lexicon = extract(&grammar, &analyze(&grammar));
        let Some(automata) = terminal_automata_within(&grammar, &lexicon, state_limit as u64)
        else {
            continue;
        };
        let Some(lexer) = build_lexer_within(automata, state_limit) else {
            continue;
        };
        let cfg = match flatten_within(&lexicon, gpugrammar_lr::cfg::DEFAULT_PRODUCTION_BUDGET) {
            Some(cfg) => cfg,
            None => continue,
        };
        let Ok(tables) = build(&cfg) else { continue };

        let eager = emit(
            &lexicon,
            &lexer,
            &group_vocabulary(&lexer, &vocabulary),
            &cfg,
            &tables,
            vocabulary.len(),
        )?;
        let eager_bytes = eager.resident_bytes();

        let lexer = Arc::new(lexer);
        let mut cache = Cache::new(
            emit_ungrouped(&lexicon, &lexer, &cfg, &tables, vocabulary.len())?,
            lexer.clone(),
            vocabulary.clone(),
        );

        // Replay the document, filling each state the first time it is reached.
        // The matcher is rebuilt from the growing artifact each step, which is
        // wasteful but keeps this a measurement of the policy rather than of an
        // implementation.
        let mut state = 0u32;
        let mut stack: Option<Vec<u32>> = None;
        let mut last_miss = 0usize;
        let mut steps = 0usize;
        let mut refused = false;
        for (offset, byte) in instance.text.as_bytes().iter().enumerate() {
            let before = cache.misses();
            cache.ensure(state);
            if cache.misses() > before {
                last_miss = offset;
            }
            let artifact = Arc::new(cache.artifact().clone());
            let mut matcher = Matcher::new(artifact, 0);
            if let Some(saved) = &stack {
                matcher.restore(state, saved.clone());
            }
            if matcher.accept_token(*byte as u32).is_err() {
                refused = true;
                break;
            }
            state = matcher.lexer_state();
            stack = Some(matcher.stack().to_vec());
            steps += 1;
        }
        if refused || steps < 8 {
            continue;
        }

        let lazy_bytes = cache.resident_bytes();
        eager_total += eager_bytes;
        lazy_total += lazy_bytes;
        println!(
            "{index:>4} {:>7} {:>8} {:>6.1}% {:>9.1}KB {:>9.1}KB {:>6.1}x {:>8}/{}",
            lexer.num_states(),
            cache.filled_states(),
            100.0 * cache.filled_states() as f64 / lexer.num_states() as f64,
            eager_bytes as f64 / 1024.0,
            lazy_bytes as f64 / 1024.0,
            eager_bytes as f64 / lazy_bytes.max(1) as f64,
            last_miss,
            steps,
        );
        reported += 1;
        if reported >= limit {
            break;
        }
    }

    println!(
        "\nover {reported} documents: eager {:.1} MB, lazy {:.1} MB, {:.1}x",
        eager_total as f64 / 1048576.0,
        lazy_total as f64 / 1048576.0,
        eager_total as f64 / lazy_total.max(1) as f64
    );
    Ok(())
}
