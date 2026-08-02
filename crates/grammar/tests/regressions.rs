//! Focused regression tests for bugs found during the Rust rewrite.

mod common;

#[path = "regressions/compiler.rs"]
mod compiler;
#[path = "regressions/json_schema.rs"]
mod json_schema;
#[path = "regressions/matcher.rs"]
mod matcher;
#[path = "regressions/regex.rs"]
mod regex;
