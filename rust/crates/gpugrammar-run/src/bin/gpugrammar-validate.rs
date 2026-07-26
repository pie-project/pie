//! Does the compiled grammar accept documents a model actually produced?
//!
//! The corpus pairs each JSON Schema with an instance generated under
//! XGrammar's constraint, so every instance is valid for that schema. Compiling
//! against a byte vocabulary and feeding the instance one byte at a time turns
//! that into an end-to-end check of lowering, terminal extraction, the lexer
//! and the parser at once: a rejection is a bug in one of them, and the byte
//! offset says where to look.
//!
//! The corpus was generated with a token budget, so 88 of the 533 instances
//! stop in the middle of a document. Those are still evidence - every byte the
//! model wrote had to be legal at the point it wrote it - but asking whether
//! the parser can *finish* there is asking the wrong question, because the
//! document does not finish either. A truncated instance therefore counts as
//! accepted when the parser consumed all of it, and the two kinds of failure
//! are reported apart: a byte the parser refused is always a bug, while a
//! refusal to terminate is only a bug on a document that was complete.

use std::fs;
use std::sync::Arc;

use anyhow::{Result, bail};
use gpugrammar_run::Matcher;
use gpugrammar_tables::pipeline::{Failure, Limits, compile_json_schema};
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
    let limits = Limits {
        lexer_states: state_limit,
        ..Default::default()
    };
    let report = std::env::var("GPUGRAMMAR_REPORT").is_ok();

    let mut compiled = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut reported = 0usize;
    let mut truncated = 0usize;
    let mut refused_byte = 0usize;
    let mut refused_end = 0usize;
    let mut levels: std::collections::BTreeMap<String, usize> = Default::default();
    let mut failures: Vec<Failure> = Vec::new();

    for (index, instance) in corpus.instances.iter().enumerate() {
        let artifact = match compile_json_schema(&instance.schema, &bytes, limits) {
            Ok(result) => {
                *levels.entry(format!("{:?}", result.precision)).or_default() += 1;
                Arc::new(result.artifact)
            }
            Err(failure) => {
                failures.push(failure);
                if report {
                    println!("{failure:?} {index}");
                }
                continue;
            }
        };
        compiled += 1;

        let complete = serde_json::from_str::<serde_json::Value>(&instance.text).is_ok();
        if !complete {
            truncated += 1;
        }

        let mut matcher = Matcher::new(artifact, 0);
        let mut failure = None;
        for (offset, byte) in instance.text.as_bytes().iter().enumerate() {
            if matcher.accept_token(*byte as u32).is_err() {
                failure = Some(offset);
                refused_byte += 1;
                break;
            }
        }
        if failure.is_none() && complete && !matcher.can_terminate() {
            failure = Some(instance.text.len());
            refused_end += 1;
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
                        "--- rejected [{index}] ({}) at byte {offset}/{} ---\n  ...{}<HERE>{}",
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

    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for failure in &failures {
        *counts.entry(format!("{failure:?}")).or_default() += 1;
    }
    println!("schemas   : {}", corpus.instances.len());
    println!("compiled  : {compiled}");
    for (level, count) in &levels {
        println!("  lowered : {count} at {level}");
    }
    for (reason, count) in counts {
        println!("  refused : {count} {reason}");
    }
    println!("accepted  : {accepted}");
    println!("rejected  : {rejected} (byte {refused_byte}, end {refused_end})");
    println!("truncated : {truncated} (checked as prefixes)");
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
