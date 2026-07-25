//! Feed one corpus instance through the matcher and print where it stops.
//!
//! Prints, per byte, the lexer state, the terminals the scan emitted and the
//! terminals a pending lexeme could still become. A rejection is then readable
//! as either "the lexer had no transition" or "the parser refused a terminal",
//! which are different bugs.

use std::fs;
use std::sync::Arc;

use anyhow::{Result, bail};
use gpugrammar_ir::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use gpugrammar_lex::lexicon::{extract, terminal_automata};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{LexState, build_lexer, group_vocabulary};
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
    schema: String,
    text: String,
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 3 {
        bail!("usage: gpugrammar-trace <instances.json> <index>");
    }
    let corpus: Corpus = serde_json::from_str(&fs::read_to_string(&arguments[1])?)?;
    let instance = &corpus.instances[arguments[2].parse::<usize>()?];

    let bytes: Vec<Vec<u8>> = (0u8..=255).map(|byte| vec![byte]).collect();
    let grammar = json_schema_to_grammar(&instance.schema, &JsonSchemaOptions::default())?;
    let lexicon = extract(&grammar, &analyze(&grammar));
    let lexer = build_lexer(terminal_automata(&grammar, &lexicon));
    let groups = group_vocabulary(&lexer, &bytes);
    let cfg = flatten(&lexicon);
    let tables = build(&cfg)?;
    let artifact = Arc::new(emit(&lexicon, &lexer, &groups, &cfg, &tables, bytes.len())?);

    let name = |terminal: u32| lexicon.terminals[terminal as usize].name.clone();
    let mut matcher = Matcher::new(artifact, 0);

    for (offset, byte) in instance.text.as_bytes().iter().enumerate() {
        let before = matcher.lexer_state();
        let scanned = lexer.scan(&[*byte], LexState(before));
        let pending: Vec<String> = scanned
            .as_ref()
            .map(|scan| {
                lexer
                    .reachable_terminals(scan.next_state)
                    .into_iter()
                    .map(|terminal| name(terminal.0))
                    .collect()
            })
            .unwrap_or_default();
        let emitted: Vec<String> = scanned
            .as_ref()
            .map(|scan| {
                scan.choices
                    .iter()
                    .map(|choice| {
                        choice
                            .iter()
                            .map(|terminal| name(terminal.0))
                            .collect::<Vec<_>>()
                            .join("+")
                    })
                    .collect()
            })
            .unwrap_or_default();

        let result = matcher.accept_token(*byte as u32);
        println!(
            "{offset:4} {:?} lex {before} parser {} | emit {:?} pending {:?} -> {}",
            *byte as char,
            matcher.parser_state(),
            emitted,
            &pending[..pending.len().min(6)],
            match &result {
                Ok(()) => "ok".to_string(),
                Err(error) => format!("{error:?}"),
            }
        );
        if result.is_err() {
            println!("\nlexer scan was: {scanned:?}");
            let state = matcher.parser_state() as usize;
            let admissible: Vec<String> = tables
                .admissible(state)
                .map(|terminal| {
                    if terminal == tables.eof {
                        "<eof>".to_string()
                    } else {
                        name(terminal)
                    }
                })
                .collect();
            println!("parser state {state} accepts {admissible:?}");
            return Ok(());
        }
    }
    println!("\nend: can_terminate = {}", matcher.can_terminate());
    Ok(())
}
