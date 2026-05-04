//! Integration tests for the `structured` module (grammar-guided generation).

#[path = "structured/ebnf_matcher.rs"]
mod ebnf_matcher;
#[path = "structured/ebnf_parser.rs"]
mod ebnf_parser;
#[path = "structured/json_schema.rs"]
mod json_schema;
#[path = "structured/matcher_basic.rs"]
mod matcher_basic;
#[path = "structured/real_world.rs"]
mod real_world;
#[path = "structured/regex_converter.rs"]
mod regex_converter;
#[path = "structured/regex_matcher.rs"]
mod regex_matcher;
