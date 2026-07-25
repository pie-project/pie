//! Regex grammar frontend.

mod parser;

use anyhow::Result;

use crate::frontend::FrontendExpr;
use crate::frontend::FrontendGrammar;
use crate::grammar::Grammar;

/// Convert a regex pattern directly to a grammar.
pub fn regex_to_grammar(pattern: &str) -> Result<Grammar> {
    FrontendGrammar::single_root(parser::parse(pattern)?).to_grammar()
}

/// Convert a regex pattern to a compatible EBNF representation.
pub fn regex_to_ebnf(pattern: &str) -> Result<String> {
    Ok(FrontendGrammar::single_root(parser::parse(pattern)?).to_ebnf())
}

pub(crate) fn regex_to_expr(pattern: &str) -> Result<FrontendExpr> {
    parser::parse(pattern)
}
