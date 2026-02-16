//! Demonstrates grammar-constrained decoding using the Grammar and Matcher APIs.
//!
//! This example constrains model output to follow a specific grammar format,
//! using the built-in Grammar/Matcher infrastructure in the SDK.

use inferlet::{
    context::Context, model::Model, runtime,
    ContextExt, InstructExt, Result,
    inference::{Grammar, Sampler},
    Constrain,
};

const HELP: &str = "\
Usage: constrained-decoding [OPTIONS]

A program to generate text constrained by a grammar (EBNF, regex, or JSON schema).

Options:
  -n, --num-tokens <TOKENS>  Maximum number of tokens to generate [default: 512]
  -h, --help                 Prints this help message";

/// A Grammar-based constraint for TokenStream.
///
/// This implements the `Constrain` trait from the SDK, using the built-in
/// Grammar/Matcher infrastructure to produce BRLE logit masks.
struct GrammarConstraint {
    matcher: inferlet::inference::Matcher,
}

impl GrammarConstraint {
    fn new(grammar: &Grammar, model: &Model) -> Self {
        let tokenizer = model.tokenizer();
        let matcher = inferlet::inference::Matcher::new(grammar, &tokenizer);
        Self { matcher }
    }
}

impl Constrain for GrammarConstraint {
    fn mask(&self) -> Vec<u32> {
        self.matcher.next_token_logit_mask()
    }

    fn accept(&mut self, tokens: &[u32]) {
        let _ = self.matcher.accept_tokens(tokens);
    }

    fn reset(&mut self) {
        self.matcher.reset();
    }

    fn rollback(&mut self, _num_tokens: usize) {
        // Grammar matcher doesn't support partial rollback; reset instead
        self.matcher.reset();
    }
}

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let max_tokens: usize = args.value_from_str(["-n", "--num-tokens"]).unwrap_or(512);

    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    // Define the grammar for the output (GBNF / standard EBNF format)
    let grammar_str = r#"
root ::= "[" person ("," person)* "]"
person ::= "{" "\"name\"" ":" string "," "\"age\"" ":" number "}"
string ::= "\"" [^"]+ "\""
number ::= [0-9]+
"#;

    let grammar = Grammar::from_ebnf(grammar_str)?;
    let constraint = GrammarConstraint::new(&grammar, &model);

    let ctx = Context::create(&model, "constrained", None)?;

    ctx.system(
        "You are a helpful assistant that outputs structured data in JSON format."
    );
    ctx.user(
        "List three famous scientists with their approximate birth years. \
        Format the output as a JSON array of objects with 'name' and 'age' fields. \
        For 'age', use their approximate birth year."
    );
    ctx.cue();

    let start = std::time::Instant::now();

    let text = ctx
        .generate(Sampler::TopP((0.0, 1.0)))
        .with_max_tokens(max_tokens)
        .with_constraint(constraint)
        .collect_text()
        .await?;

    println!("Generated (constrained):\n{}", text);
    println!("\nElapsed: {:?}", start.elapsed());

    Ok(String::new())
}
