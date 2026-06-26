//! Demonstrates EBNF grammar-constrained decoding.
//!
//! `Schema::Ebnf` compiles the grammar into a host matcher; every generated
//! token is masked to keep the output a valid sentence in the grammar.

use inferlet::{
    Context, Result, Schema, sample::Sampler,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_num_tokens")]
    num_tokens: usize,
}

fn default_num_tokens() -> usize { 512 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = input.num_tokens;

    let grammar = r#"
root ::= "[" person ("," person)* "]"
person ::= "{" "\"name\"" ":" string "," "\"age\"" ":" number "}"
string ::= "\"" [^"]+ "\""
number ::= [0-9]+
"#;

    let mut ctx = Context::new()?;
    ctx.system("You are a helpful assistant that outputs structured data in JSON format.");
    ctx.user(
        "List three famous scientists with their approximate birth years. \
         Format the output as a JSON array of objects with 'name' and 'age' fields. \
         For 'age', use their approximate birth year.",
    );
    ctx.cue();

    let start = std::time::Instant::now();

    let text = ctx
        .generate(Sampler::Argmax)
        .max_tokens(max_tokens)
        .constrain_with(inferlet::Ebnf(grammar))?
        .collect_text()
        .await?;

    println!("Generated (constrained):\n{}", text);
    println!("\nElapsed: {:?}", start.elapsed());

    Ok(String::new())
}
