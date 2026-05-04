//! Expr — JSON DSL for workflow composition.

use serde::Deserialize;
use serde_json::Value;

/// Recursive expression tree representing a workflow composition.
///
/// Deserialized from JSON with `{"type": "pipe", "stages": [...]}` etc.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum Expr {
    /// A literal data value. Evaluates to `value` without computation.
    #[serde(rename = "literal")]
    Literal { value: Value },

    /// A single process invocation. Receives its input from the pipeline.
    /// Optional `config` is merged into the input as static data.
    #[serde(rename = "process")]
    Process {
        program_name: String,
        #[serde(default)]
        config: Option<Value>,
    },

    /// Sequential pipeline: output of stage N feeds into stage N+1.
    #[serde(rename = "pipe")]
    Pipe { stages: Vec<Expr> },

    /// Map: apply `function` to each element of `over`.
    #[serde(rename = "map")]
    Map { function: Box<Expr>, over: Box<Expr> },

    /// Fold: reduce `over` with `function`, starting from `init`.
    #[serde(rename = "fold")]
    Fold { function: Box<Expr>, init: Box<Expr>, over: Box<Expr> },

    /// Fork: run N branches concurrently, collect all results.
    #[serde(rename = "fork")]
    Fork { branches: Vec<Expr> },

    /// Conditional: evaluate `predicate`, then run `then` or `otherwise`.
    /// Uses lazy branching — only the chosen branch is executed.
    #[serde(rename = "cond")]
    Cond {
        predicate: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },

    /// Iterate: run `body` repeatedly until `until` returns truthy.
    #[serde(rename = "iterate")]
    Iterate { body: Box<Expr>, until: Box<Expr> },

    /// Take: keep only the first `k` results (for map/fork fan-out).
    #[serde(rename = "take")]
    Take { k: usize },
}
