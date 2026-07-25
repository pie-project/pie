//! Does the compiled grammar accept documents a model actually produced?
//!
//! The corpus pairs each JSON Schema with an instance generated under
//! XGrammar's constraint, so every instance is valid for that schema. Compiling
//! against a byte vocabulary and feeding the instance one byte at a time turns
//! that into an end-to-end check of lowering, terminal extraction, the lexer
//! and the parser at once: a rejection is a bug in one of them, and the byte
//! offset says where to look.

use std::fs;
use std::sync::Arc;

use anyhow::{Result, bail};
use gpugrammar_ir::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use gpugrammar_lex::lexicon::{extract, terminal_automata_within};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{build_lexer_within, group_vocabulary};
use gpugrammar_lr::cfg::flatten;
use gpugrammar_lr::tables::build;
use gpugrammar_run::Matcher;
use gpugrammar_tables::emit;
use serde::Deserialize;

#[derive(Deserialize)]
struct Corpus {
    instances: Vec<Instance>,
}

#[derive(Deserialize)]
struct Instance {
    config: String,
    schema: String,
    text: String,
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 2 {
        bail!("usage: gpugrammar-validate <instances.json> [max-reported]");
    }
    let corpus: Corpus = serde_json::from_str(&fs::read_to_string(&arguments[1])?)?;
    let report_limit: usize = arguments
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);

    let bytes: Vec<Vec<u8>> = (0u8..=255).map(|byte| vec![byte]).collect();

    let state_limit: usize = std::env::var("GPUGRAMMAR_MAX_LEXER_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20_000);
    let mut lexer_states: Vec<usize> = Vec::new();
    let mut oversized = 0usize;
    let mut compiled = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut reported = 0usize;

    for instance in corpus.instances.iter() {
        let Ok(grammar) = json_schema_to_grammar(&instance.schema, &JsonSchemaOptions::default())
        else {
            continue;
        };
        let lexicon = extract(&grammar, &analyze(&grammar));
        // A length bound is unrolled into the automaton, so refuse the schema
        // from its declared bounds rather than after building.
        let Some(automata) = terminal_automata_within(&grammar, &lexicon, state_limit as u64)
        else {
            oversized += 1;
            continue;
        };
        let Some(lexer) = build_lexer_within(automata, state_limit) else {
            oversized += 1;
            continue;
        };
        lexer_states.push(lexer.num_states());
        let groups = group_vocabulary(&lexer, &bytes);
        let cfg = flatten(&lexicon);
        let Ok(tables) = build(&cfg) else { continue };
        let artifact = Arc::new(emit(&lexicon, &lexer, &groups, &cfg, &tables, bytes.len())?);
        compiled += 1;

        let mut matcher = Matcher::new(artifact, 0);
        let mut failure = None;
        for (offset, byte) in instance.text.as_bytes().iter().enumerate() {
            if matcher.accept_token(*byte as u32).is_err() {
                failure = Some(offset);
                break;
            }
        }
        if failure.is_none() && !matcher.can_terminate() {
            failure = Some(instance.text.len());
        }

        match failure {
            None => accepted += 1,
            Some(offset) => {
                rejected += 1;
                if reported < report_limit {
                    let text = instance.text.as_bytes();
                    let from = offset.saturating_sub(40);
                    let to = (offset + 20).min(text.len());
                    eprintln!(
                        "--- rejected ({}) at byte {offset}/{} ---\n  ...{}<HERE>{}",
                        instance.config,
                        text.len(),
                        String::from_utf8_lossy(&text[from..offset]).replace('\n', "\\n"),
                        String::from_utf8_lossy(&text[offset..to]).replace('\n', "\\n"),
                    );
                    reported += 1;
                }
            }
        }
    }

    lexer_states.sort_unstable();
    if !lexer_states.is_empty() {
        println!(
            "lexer states: median {} p90 {} max {}",
            lexer_states[lexer_states.len() / 2],
            lexer_states[lexer_states.len() * 9 / 10],
            lexer_states[lexer_states.len() - 1]
        );
    }
    println!("oversized : {oversized} (over {state_limit} lexer states)");
    println!("compiled  : {compiled}");
    println!("accepted  : {accepted}");
    println!("rejected  : {rejected}");
    if compiled > 0 {
        println!(
            "acceptance: {:.1}%",
            100.0 * accepted as f64 / compiled as f64
        );
    }
    if rejected > 0 {
        std::process::exit(1);
    }
    Ok(())
}
