//! Compile a grammar and a vocabulary into the device artifact.
//!
//! Usage:
//!   gpugrammar-compile <grammar.ebnf> <root-rule> <vocab.json> <out-prefix>
//!
//! The vocabulary is a JSON array of strings, one per token id. Output is
//! `<prefix>.json` describing the tables and `<prefix>.bin` holding the arrays
//! back to back, so the runtime can map them without a parser.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use gpugrammar_ir::grammar::Grammar;
use gpugrammar_lex::lexicon::{extract, terminal_automata};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{build_lexer, group_vocabulary};
use gpugrammar_lr::cfg::flatten;
use gpugrammar_lr::tables::build;
use gpugrammar_tables::emit;

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() != 5 {
        bail!("usage: gpugrammar-compile <grammar.ebnf> <root> <vocab.json> <out-prefix>");
    }
    let source =
        fs::read_to_string(&arguments[1]).with_context(|| format!("reading {}", arguments[1]))?;
    let vocabulary: Vec<String> = serde_json::from_str(
        &fs::read_to_string(&arguments[3]).with_context(|| format!("reading {}", arguments[3]))?,
    )?;
    let vocabulary: Vec<Vec<u8>> = vocabulary
        .into_iter()
        .map(|token| token.into_bytes())
        .collect();

    let grammar =
        Grammar::from_ebnf(&source, &arguments[2]).map_err(|error| anyhow::anyhow!("{error}"))?;
    let lexicon = extract(&grammar, &analyze(&grammar));
    let lexer = build_lexer(terminal_automata(&grammar, &lexicon));
    let groups = group_vocabulary(&lexer, &vocabulary);
    let cfg = flatten(&lexicon);
    let tables = build(&cfg)?;
    let artifact = emit(&lexicon, &lexer, &groups, &cfg, &tables, vocabulary.len())?;

    eprintln!(
        "terminals {}  lexer states {}  parser states {}  groups {}",
        artifact.num_terminals,
        artifact.num_lexer_states,
        artifact.num_parser_states,
        artifact.groups.len()
    );
    eprintln!(
        "resident {:.1} KiB, against {:.1} KiB for per-state token rows",
        artifact.resident_bytes() as f64 / 1024.0,
        artifact.rows_equivalent_bytes() as f64 / 1024.0
    );

    let prefix = PathBuf::from(&arguments[4]);
    let mut blob: Vec<u8> = Vec::new();
    let mut layout = serde_json::Map::new();
    macro_rules! section {
        ($name:expr, $array:expr, $kind:expr) => {{
            let start = blob.len();
            for value in &$array {
                blob.extend_from_slice(&value.to_le_bytes());
            }
            layout.insert(
                $name.to_string(),
                serde_json::json!({
                    "offset": start,
                    "count": $array.len(),
                    "dtype": $kind,
                }),
            );
        }};
    }
    section!("set_payload", artifact.set_payload, "uint32");
    section!("group_offsets", artifact.group_offsets, "uint32");
    section!("action_offsets", artifact.action_offsets, "uint32");
    section!("action_terminals", artifact.action_terminals, "uint32");
    section!("action_values", artifact.action_values, "int32");
    section!("goto_offsets", artifact.goto_offsets, "uint32");
    section!("goto_nonterminals", artifact.goto_nonterminals, "uint32");
    section!("goto_targets", artifact.goto_targets, "uint32");
    section!("production_lhs", artifact.production_lhs, "uint32");
    section!("production_arity", artifact.production_arity, "uint32");

    let manifest = serde_json::json!({
        "vocab_size": artifact.vocab_size,
        "bitset_words": artifact.bitset_words,
        "num_lexer_states": artifact.num_lexer_states,
        "num_terminals": artifact.num_terminals,
        "num_nonterminals": artifact.num_nonterminals,
        "num_parser_states": artifact.num_parser_states,
        "eof_terminal": artifact.eof_terminal,
        "start_parser_state": artifact.start_parser_state,
        "resident_bytes": artifact.resident_bytes(),
        "rows_equivalent_bytes": artifact.rows_equivalent_bytes(),
        "groups": artifact.groups,
        "arrays": layout,
    });

    fs::write(prefix.with_extension("bin"), &blob)?;
    fs::write(
        prefix.with_extension("json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    eprintln!(
        "wrote {} and {}",
        prefix.with_extension("json").display(),
        prefix.with_extension("bin").display()
    );
    Ok(())
}
