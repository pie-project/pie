//! Are the corpus's LALR conflicts real, or artefacts of state merging?
//!
//! Every conflict this corpus produces is reduce/reduce and none is
//! shift/reduce. That is the signature of LALR's one weakness: it merges LR(1)
//! states sharing an LR(0) core, and merging their lookahead sets can invent a
//! reduce/reduce conflict the grammar does not have. Canonical LR(1) never
//! merges, so building both says which conflicts are which - and an artefact is
//! what IELR(1) removes while keeping LALR's size.
//!
//! Reports the split, and the state counts, because canonical state counts are
//! the reason LALR is the target in the first place.

use std::sync::atomic::{AtomicUsize, Ordering};

use engrain_ir::json_schema::{JsonSchemaOptions, Precision, json_schema_to_grammar};
use engrain_lex::lexicon::{DEFAULT_TERMINAL_BUDGET, extract_within};
use engrain_lex::regular::analyze;
use engrain_lr::cfg::{DEFAULT_PRODUCTION_BUDGET, flatten_within};
use engrain_lr::tables::{build, build_canonical};
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Corpus {
    instances: Vec<Instance>,
}

#[derive(Deserialize)]
struct Instance {
    schema: String,
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results/jsonschemabench-instances.json".to_string());
    let budget: usize = std::env::var("ENGRAIN_CANONICAL_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(400_000);
    let corpus: Corpus = serde_json::from_str(&std::fs::read_to_string(path)?)?;

    let artefact = AtomicUsize::new(0);
    let genuine = AtomicUsize::new(0);
    let over = AtomicUsize::new(0);
    let lalr_states = AtomicUsize::new(0);
    let canonical_states = AtomicUsize::new(0);
    let control_ok = AtomicUsize::new(0);
    let control_bad = AtomicUsize::new(0);
    let control_big = AtomicUsize::new(0);

    corpus
        .instances
        .par_iter()
        .enumerate()
        .for_each(|(index, instance)| {
            // Only the schemas that fail at every precision level, since a
            // schema that compiles at a coarser one is not a coverage loss.
            let mut conflicted = None;
            for precision in Precision::LEVELS {
                let options = JsonSchemaOptions {
                    precision,
                    ..Default::default()
                };
                let Ok(grammar) = json_schema_to_grammar(&instance.schema, &options) else {
                    continue;
                };
                let lexicon =
                    extract_within(&grammar, &analyze(&grammar), DEFAULT_TERMINAL_BUDGET);
                let Some(cfg) = flatten_within(&lexicon, DEFAULT_PRODUCTION_BUDGET) else {
                    continue;
                };
                match build(&cfg) {
                    Ok(_) => {
                        conflicted = None;
                        break;
                    }
                    Err(_) => conflicted = Some(cfg),
                }
            }
            let Some(cfg) = conflicted else {
                // A control: this schema has an LALR parser, so canonical LR(1)
                // must have one too - it never merges, and merging is the only
                // thing LALR does that can lose a parse. If this ever fails,
                // the canonical builder is broken and the headline answer would
                // be "every conflict is genuine" for the wrong reason.
                if index % 17 == 0
                    && let Ok(grammar) = json_schema_to_grammar(
                        &instance.schema,
                        &JsonSchemaOptions::default(),
                    )
                {
                    let lexicon =
                        extract_within(&grammar, &analyze(&grammar), DEFAULT_TERMINAL_BUDGET);
                    if let Some(cfg) = flatten_within(&lexicon, DEFAULT_PRODUCTION_BUDGET)
                        && build(&cfg).is_ok()
                    {
                        match build_canonical(&cfg, budget) {
                            Ok(_) => {
                                control_ok.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(error) if error.to_string().contains("exceeds") => {
                                control_big.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(_) => {
                                control_bad.fetch_add(1, Ordering::Relaxed);
                                println!(
                                    "schema {index:>3}: CONTROL FAILED - LALR builds \
                                     but canonical LR(1) does not"
                                );
                            }
                        }
                    }
                }
                return;
            };

            match build_canonical(&cfg, budget) {
                Ok(tables) => {
                    artefact.fetch_add(1, Ordering::Relaxed);
                    canonical_states.fetch_add(tables.action.len(), Ordering::Relaxed);
                    if let Ok(merged) = build(&cfg) {
                        lalr_states.fetch_add(merged.action.len(), Ordering::Relaxed);
                    }
                    println!(
                        "schema {index:>3}: LALR conflicts, canonical LR(1) does not \
                         ({} states) - an artefact of merging",
                        tables.action.len()
                    );
                }
                Err(error) if error.to_string().contains("exceeds") => {
                    over.fetch_add(1, Ordering::Relaxed);
                    println!("schema {index:>3}: canonical LR(1) over {budget} states");
                }
                Err(_) => {
                    genuine.fetch_add(1, Ordering::Relaxed);
                    println!("schema {index:>3}: conflicts under canonical LR(1) too");
                }
            }
        });

    println!(
        "\ncontrol: of the schemas LALR accepts, canonical LR(1) accepts {} \
         and refuses {} ({} over the state budget)",
        control_ok.into_inner(),
        control_bad.into_inner(),
        control_big.into_inner()
    );
    let artefact = artefact.into_inner();
    let genuine = genuine.into_inner();
    let over = over.into_inner();
    println!("\nschemas refused for conflicts: {}", artefact + genuine + over);
    println!("  {artefact} are LALR artefacts - IELR(1) would take them");
    println!("  {genuine} conflict under canonical LR(1) too - genuinely ambiguous");
    println!("  {over} could not be decided within {budget} canonical states");
    if artefact > 0 {
        println!(
            "  canonical states over those {artefact}: {} against LALR's {}",
            canonical_states.into_inner(),
            lalr_states.into_inner()
        );
    }
    Ok(())
}
