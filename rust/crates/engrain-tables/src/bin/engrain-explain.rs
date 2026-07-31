//! Print the intermediate forms for one schema, so a conflict can be read.
//!
//! Without an index, lists the schemas that fail LALR(1) construction. With
//! one, dumps that schema's terminals and flattened productions, which is
//! what a reduce/reduce conflict has to be explained from.

use std::fs;

use anyhow::{Result, bail};
use engrain_ir::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use engrain_lex::lexicon::extract;
use engrain_lex::regular::analyze;
use engrain_lr::cfg::{Cfg, Symbol};
use engrain_lr::tables::build;

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 2 {
        bail!("usage: engrain-explain <schemas.json> [index]");
    }
    let schemas: Vec<String> = serde_json::from_str(&fs::read_to_string(&arguments[1])?)?;

    let Some(index) = arguments
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
    else {
        for (index, schema) in schemas.iter().enumerate() {
            let Ok(grammar) = json_schema_to_grammar(schema, &JsonSchemaOptions::default()) else {
                continue;
            };
            let lexicon = extract(&grammar, &analyze(&grammar));
            let cfg = engrain_lr::cfg::flatten(&lexicon);
            if let Err(error) = build(&cfg) {
                println!(
                    "{index:4}  {}  |  {}",
                    error.to_string().lines().next().unwrap_or(""),
                    schema[..schema.len().min(150)].replace('\n', " ")
                );
            }
        }
        return Ok(());
    };

    let schema = &schemas[index];
    println!("=== schema ===\n{schema}\n");

    let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default())?;
    let regularity = analyze(&grammar);
    println!("=== rules ({}) ===", grammar.rules().len());
    for (index, rule) in grammar.rules().iter().enumerate() {
        let id = engrain_ir::grammar::RuleId(index as u32);
        println!(
            "  {:<32} regular={} refs={:?}",
            rule.name,
            regularity.is_regular(id),
            engrain_lex::regular::references(&grammar, rule.body)
                .iter()
                .map(|r| grammar.get_rule(*r).name.as_str())
                .collect::<Vec<_>>()
        );
    }
    println!("  root = {}", grammar.get_rule(grammar.root_rule()).name);
    let lexicon = extract(&grammar, &regularity);
    println!("=== terminals ({}) ===", lexicon.terminals.len());
    for (id, terminal) in lexicon.terminals.iter().enumerate() {
        println!("  t{id:<3} {}", terminal.name);
    }

    let cfg = engrain_lr::cfg::flatten(&lexicon);
    println!("\n=== productions ({}) ===", cfg.productions.len());
    for (id, production) in cfg.productions.iter().enumerate() {
        println!(
            "  p{id:<3} {} -> {}",
            cfg.nonterminal_names[production.lhs as usize],
            render(&cfg, &lexicon, &production.rhs)
        );
    }
    println!(
        "\nstart: {} ({} nonterminals, {} terminals)",
        cfg.nonterminal_names[cfg.start as usize],
        cfg.num_nonterminals(),
        cfg.num_terminals
    );

    match build(&cfg) {
        Ok(tables) => println!("\nLALR(1): {} states", tables.num_states()),
        Err(error) => println!("\n{error}"),
    }
    Ok(())
}

fn render(cfg: &Cfg, lexicon: &engrain_lex::lexicon::Lexicon, rhs: &[Symbol]) -> String {
    if rhs.is_empty() {
        return "ε".into();
    }
    rhs.iter()
        .map(|symbol| match symbol {
            Symbol::Terminal(terminal) => {
                format!("t{}:{}", terminal.0, lexicon.terminal_name(*terminal))
            }
            Symbol::Nonterminal(nonterminal) => {
                cfg.nonterminal_names[*nonterminal as usize].clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
