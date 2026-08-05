//! How far does the pipeline get on real JSON Schemas?
//!
//! Reads a JSON array of schema strings and reports, per stage, how many
//! survive: schema lowering, lexicon extraction, flattening and LALR(1)
//! construction. Conflicts are printed with their grammar so they can be
//! reduced to a test case.

use std::collections::BTreeMap;
use std::fs;

use anyhow::{Result, bail};
use engrain_ir::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use engrain_lex::lexicon::extract_within;
use engrain_lex::regular::analyze;
use engrain_lr::cfg::flatten_within;
use engrain_lr::tables::build;

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 2 {
        bail!("usage: engrain-coverage <schemas.json> [max-reported]");
    }
    let schemas: Vec<String> = serde_json::from_str(&fs::read_to_string(&arguments[1])?)?;
    let report_limit: usize = arguments
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    let mut lowered = 0usize;
    let mut extracted = 0usize;
    let mut compiled = 0usize;
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut reported = 0usize;
    let mut states = Vec::new();
    let mut terminals = Vec::new();

    for schema in &schemas {
        let grammar = match json_schema_to_grammar(schema, &JsonSchemaOptions::default()) {
            Ok(grammar) => grammar,
            Err(error) => {
                *reasons
                    .entry(format!("lowering: {}", first_line(&error.to_string())))
                    .or_default() += 1;
                continue;
            }
        };
        lowered += 1;

        let budget: u64 = std::env::var("ENGRAIN_TERMINAL_BUDGET")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(engrain_lex::lexicon::DEFAULT_TERMINAL_BUDGET);
        let clock = std::time::Instant::now();
        let lexicon = extract_within(&grammar, &analyze(&grammar), budget);
        extracted += 1;
        if lexicon.skeleton.len() == 1 {
            *reasons
                .entry("(note) purely regular, no stack needed".into())
                .or_default() += 1;
        }
        terminals.push(lexicon.terminals.len());

        let Some(cfg) = flatten_within(&lexicon, engrain_lr::cfg::DEFAULT_PRODUCTION_BUDGET) else {
            *reasons
                .entry("parser: grammar exceeds the production budget".into())
                .or_default() += 1;
            continue;
        };
        if std::env::var("ENGRAIN_REPORT").is_ok() {
            eprintln!(
                "cfg: {} terminals, {} productions, {} nonterminals",
                lexicon.terminals.len(),
                cfg.productions.len(),
                cfg.num_nonterminals()
            );
        }
        if clock.elapsed().as_millis() > 500 {
            eprintln!(
                "slow: {} terminals, {} productions, {} nonterminals, {:?}",
                lexicon.terminals.len(),
                cfg.productions.len(),
                cfg.num_nonterminals(),
                clock.elapsed()
            );
        }
        match build(&cfg) {
            Ok(tables) => {
                compiled += 1;
                states.push(tables.num_states());
            }
            Err(error) => {
                let message = first_line(&error.to_string());
                if reported < report_limit {
                    eprintln!(
                        "--- conflict ---\n{}\n{}",
                        message,
                        &schema[..schema.len().min(300)]
                    );
                    reported += 1;
                }
                *reasons.entry(format!("lalr: {message}")).or_default() += 1;
            }
        }
    }

    println!("schemas            : {}", schemas.len());
    println!("lowered to grammar : {lowered}");
    println!("split into lexicon : {extracted}");
    println!("LALR(1) tables     : {compiled}");
    if !states.is_empty() {
        states.sort_unstable();
        terminals.sort_unstable();
        println!(
            "parser states      : median {} max {}",
            states[states.len() / 2],
            states[states.len() - 1]
        );
        println!(
            "terminals          : median {} max {}",
            terminals[terminals.len() / 2],
            terminals[terminals.len() - 1]
        );
    }
    println!("\nfailures by reason:");
    let mut ordered: Vec<_> = reasons.into_iter().collect();
    ordered.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (reason, count) in ordered.iter().take(12) {
        println!("  {count:5}  {reason}");
    }
    Ok(())
}

fn first_line(message: &str) -> String {
    message
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(110)
        .collect()
}
